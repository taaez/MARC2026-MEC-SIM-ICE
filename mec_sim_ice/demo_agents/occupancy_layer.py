"""Probabilistic occupancy grid (log-odds) for the MPPI planner -- replaces lidar discs.

## Why
`lidar_obstacles.LidarObstacleLayer` marks a cell blocked once two points ever landed in it and
never lowers that count: a beam passing straight through the cell ("it is empty here") is not
used. Cells are only released by leaving the 15 m window or by the depth camera seeing floor
there. Momentary returns (a ghost from a turn, a passing pedestrian) therefore pile up --
live rt04 / rt04b: up to 1800 lidar discs next to the target, the robot boxed in with
V(robot) = inf (rt04_r8_mppi) -- and the planner grew rules on top (stale-disc stages, disc-set
stabilization, freezing) that fight each other.

## Method (standard occupancy grid mapping, log-odds form)
Per cell a log-odds value L (0 = unknown, > 0 occupied, < 0 free):
- cells hit by an obstacle-band return (same ground separation as the disc layer,
  `lidar_obstacles.estimate_ground`):   L += L_OCC   (once per sweep per cell)
- cells a beam passed through before its return:   L += L_FREE   (once per sweep per cell)
  * obstacle-band return at range r: free up to r - RES along that azimuth;
  * ground return at range r: free only while the beam is still above the obstacle band,
    i.e. up to r * (1 - BAND_MIN / h) with h = sensor height above the ground there (a beam
    hitting the ground passes *under* whatever it later meets -- it says nothing below the band);
  * azimuths with no return give no evidence (conservative).
  Rays are taken per azimuth bin (AZ_BINS), not per point.
- floor seen by the depth camera (`depth_ground_veto`): L += L_FREE on those cells.
- L is clamped to [L_MIN, L_MAX] so a long-lived cell is released after a few free looks and a
  long-free cell becomes occupied again after a few hits.
Outputs: L sampled at world points (`sample`), cells above a threshold (`occupied_xy`), and the
reactive-brake test (`danger_ahead`, same semantics as the disc layer).

The grid is a dense float32 array of GRID_M x GRID_M around the first pose, re-centred when the
robot gets within RECENTER_M of its edge.
"""

from __future__ import annotations

import math

import numpy as np

from lidar_obstacles import (GROUND_BAND_MAX_M, GROUND_BAND_MIN_M, MAX_RANGE_M,
                             estimate_ground, lidar_xyz)

RES_M = 0.1
GRID_M = 120.0
RECENTER_M = 15.0
L_OCC = 0.85                 # one sweep's hit: p 0.5 -> 0.70
L_FREE = -0.4                # one sweep's pass-through: weaker than a hit (low obstacles are
                             # seen intermittently; a few passes must not erase a real one)
L_MIN, L_MAX = -2.0, 3.5     # clamp: from L_MAX, ~6 free sweeps release a cell
L_OCCUPIED = 1.2             # > one hit, < two hits: the reactive brake / "occupied" view
L_HARD = 2.5                 # >= 3 hits net: lethal in the planner (see nav_field soft costs)
AZ_BINS = 720                # 0.5 deg rays
RAY_START_M = 0.3            # the chassis itself


class OccupancyLayer:
    def __init__(self, res=RES_M, size_m=GRID_M):
        self.res = float(res)
        self.n = int(round(size_m / self.res))
        self.L = np.zeros((self.n, self.n), np.float32)   # [row = y, col = x]
        self.ox = self.oy = None                          # world xy of cell (0, 0) corner
        self.scans = 0

    # -- grid bookkeeping --
    def _ensure(self, x, y):
        if self.ox is None:
            self.ox = x - 0.5 * self.n * self.res
            self.oy = y - 0.5 * self.n * self.res
            return
        m = RECENTER_M
        if (x - self.ox < m or self.ox + self.n * self.res - x < m
                or y - self.oy < m or self.oy + self.n * self.res - y < m):
            nox = x - 0.5 * self.n * self.res
            noy = y - 0.5 * self.n * self.res
            dj = int(round((nox - self.ox) / self.res))
            di = int(round((noy - self.oy) / self.res))
            new = np.zeros_like(self.L)
            si0, si1 = max(0, di), min(self.n, self.n + di)
            sj0, sj1 = max(0, dj), min(self.n, self.n + dj)
            if si1 > si0 and sj1 > sj0:
                new[si0 - di:si1 - di, sj0 - dj:sj1 - dj] = self.L[si0:si1, sj0:sj1]
            self.L = new
            self.ox += dj * self.res
            self.oy += di * self.res

    def _cells(self, wx, wy):
        """Unique (i, j) inside the grid for world points."""
        j = np.floor((np.asarray(wx) - self.ox) / self.res).astype(np.int64)
        i = np.floor((np.asarray(wy) - self.oy) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        if not np.any(ok):
            return np.zeros(0, np.int64)
        return np.unique(i[ok] * self.n + j[ok])

    def _add(self, flat_idx, delta):
        if len(flat_idx):
            v = self.L.ravel()
            v[flat_idx] = np.clip(v[flat_idx] + delta, L_MIN, L_MAX)

    # -- updates --
    def update(self, msg, pose, mount_offset=(0.0, 0.0)):
        """One sweep. Returns (#hit cells, #free cells) updated."""
        pts = lidar_xyz(msg)
        if pts is None or pts.shape[0] == 0 or pose is None:
            return 0, 0
        x, y, yaw = pose
        self._ensure(x, y)
        self.scans += 1
        ground, _ = estimate_ground(pts)
        z_rel = pts[:, 2] - ground
        rng = np.hypot(pts[:, 0], pts[:, 1])
        az = np.arctan2(pts[:, 1], pts[:, 0])
        b = np.clip(((az + math.pi) / (2.0 * math.pi) * AZ_BINS).astype(np.int64), 0, AZ_BINS - 1)
        obst = (z_rel > GROUND_BAND_MIN_M) & (z_rel < GROUND_BAND_MAX_M) & (rng <= MAX_RANGE_M)
        gnd = (z_rel <= GROUND_BAND_MIN_M) & (rng <= MAX_RANGE_M)
        # per-bin free length
        obs_min = np.full(AZ_BINS, np.inf)
        if np.any(obst):
            np.minimum.at(obs_min, b[obst], rng[obst])
        g_free = np.zeros(AZ_BINS)
        if np.any(gnd):
            h = -ground[gnd]                                   # sensor height above ground
            d_free = np.where(h > GROUND_BAND_MIN_M, rng[gnd] * (1.0 - GROUND_BAND_MIN_M / np.maximum(h, 1e-3)), 0.0)
            np.maximum.at(g_free, b[gnd], d_free)
        free_len = np.where(np.isfinite(obs_min), obs_min - self.res, g_free)
        free_len = np.minimum(free_len, MAX_RANGE_M)
        c, s = math.cos(yaw), math.sin(yaw)
        mx, my = float(mount_offset[0]), float(mount_offset[1])

        def to_world(sx, sy):
            sx, sy = sx + mx, sy + my
            return x + sx * c - sy * s, y + sx * s + sy * c

        free_idx = np.zeros(0, np.int64)
        bins = np.flatnonzero(free_len > RAY_START_M)
        if len(bins):
            steps = np.arange(RAY_START_M, MAX_RANGE_M, 0.5 * self.res)
            ang = -math.pi + (bins + 0.5) * (2.0 * math.pi / AZ_BINS)
            dd = steps[None, :]
            ok = dd <= free_len[bins][:, None]
            sx = (np.cos(ang)[:, None] * dd)[ok]
            sy = (np.sin(ang)[:, None] * dd)[ok]
            free_idx = self._cells(*to_world(sx, sy))
        hit_idx = np.zeros(0, np.int64)
        if np.any(obst):
            hit_idx = self._cells(*to_world(pts[obst, 0], pts[obst, 1]))
        if len(hit_idx) and len(free_idx):
            free_idx = np.setdiff1d(free_idx, hit_idx, assume_unique=True)
        self._add(free_idx, L_FREE)
        self._add(hit_idx, L_OCC)
        return len(hit_idx), len(free_idx)

    def free_points(self, points_world, keep=None):
        """Floor seen by the depth camera -> free evidence (cells holding a `keep` point excluded)."""
        if self.ox is None or points_world is None or len(points_world) == 0:
            return 0
        p = np.asarray(points_world, dtype=np.float64)
        idx = self._cells(p[:, 0], p[:, 1])
        if keep is not None and len(keep):
            k = np.asarray(keep, dtype=np.float64)
            idx = np.setdiff1d(idx, self._cells(k[:, 0], k[:, 1]), assume_unique=True)
        # [r9b] the camera looking straight at floor is decisive: drop to L_MIN at once. With the
        # ordinary L_FREE (-0.4/frame) the lidar's systematic slope ghosts (+0.85 every sweep)
        # out-voted it -- rt04_r9_mppi: the camera saw floor in 13,406 lidar-blocked cells while
        # hard cells grew to 1,400 and V(robot) = inf. (The disc layer's veto cleared them outright.)
        if len(idx):
            self.L.ravel()[idx] = L_MIN
        return len(idx)

    # -- queries --
    def sample(self, xs, ys):
        """L at world points (0 = unknown outside the grid)."""
        xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
        if self.ox is None:
            return np.zeros(xs.shape, np.float32)
        j = np.floor((xs - self.ox) / self.res).astype(np.int64)
        i = np.floor((ys - self.oy) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        out = np.zeros(xs.shape, np.float32)
        out[ok] = self.L[i[ok], j[ok]]
        return out

    def sample_max(self, xs, ys, cell_m):
        """Max L over a cell_m x cell_m square centred at each world point (5 sub-samples)."""
        q = 0.25 * cell_m
        v = self.sample(xs, ys)
        for dx, dy in ((-q, -q), (-q, q), (q, -q), (q, q)):
            v = np.maximum(v, self.sample(np.asarray(xs) + dx, np.asarray(ys) + dy))
        return v

    def occupied_xy(self, min_l=L_OCCUPIED, near_xy=None, within_m=None):
        if self.ox is None:
            return np.zeros((0, 2))
        if near_xy is not None and within_m is not None:
            k = int(math.ceil(within_m / self.res))
            ci = int((near_xy[1] - self.oy) / self.res)
            cj = int((near_xy[0] - self.ox) / self.res)
            i0, i1 = max(ci - k, 0), min(ci + k + 1, self.n)
            j0, j1 = max(cj - k, 0), min(cj + k + 1, self.n)
            ii, jj = np.nonzero(self.L[i0:i1, j0:j1] >= min_l)
            ii, jj = ii + i0, jj + j0
        else:
            ii, jj = np.nonzero(self.L >= min_l)
        return np.stack([self.ox + (jj + 0.5) * self.res, self.oy + (ii + 0.5) * self.res], axis=1)

    def danger_ahead(self, pose, max_dist, cone_deg=70.0, goal=None, goal_exclude_m=0.0):
        """Any occupied cell within max_dist inside the forward cone (LidarObstacleLayer semantics)."""
        x, y, yaw = pose
        pts = self.occupied_xy(L_OCCUPIED, near_xy=(x, y), within_m=max_dist + self.res)
        if not len(pts):
            return False
        if goal is not None and goal_exclude_m > 0.0:
            pts = pts[np.hypot(pts[:, 0] - goal[0], pts[:, 1] - goal[1]) > goal_exclude_m]
            if not len(pts):
                return False
        d = pts - np.array([x, y])
        dist = np.hypot(d[:, 0], d[:, 1])
        close = dist - self.res <= max_dist
        if not np.any(close):
            return False
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        cosang = (d[close] @ fwd) / np.maximum(dist[close], 1e-6)
        return bool(np.any(cosang >= math.cos(math.radians(cone_deg))))

    def stats(self):
        return {"scans": self.scans, "occupied": int((self.L >= L_OCCUPIED).sum()),
                "hard": int((self.L >= L_HARD).sum()), "free": int((self.L < 0).sum())}
