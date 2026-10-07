"""Kerb / step layer -- height discontinuities in the ground, from lidar and depth.

## 왜 필요한가
연석(도로 경계의 턱)과 꺼지는 단차는 장애물 레이어(lidar_obstacles)가 보지 않는다 — 지면 위
0.15 m 아래는 "지면·연석·노면 잡음"으로 버리기 때문이다. 점유지도는 도로(0)/도로 밖(−1)만 있고
높이가 없다. 그래서 지금까지는 "도로 밖 = lethal"로 연석을 피했는데, 이러면
  - 지도 경계가 실제로는 평평한 곳(포장이 이어지는 곳, 정원 진입로)도 막히고
  - 도로 안의 단차(rtC2 런: 지도상 도로에서 pitch −38.8° 로 꺼짐)는 모른다.
이 레이어는 센서가 실제로 본 지면에서 높이 계단을 찾아, 단차 칸(lethal 후보)과 평평하게 관측된 칸
(도로 밖이어도 통과 가능)을 낸다.

## 기준값 (씬 지면 메시의 정답 분석, 2026-10-06 — 분석용으로만 썼고 코드는 그 값을 참조하지 않는다)
- 지도 경계 4388칸: 평평(<3 cm) 53 %, 올라가는 턱 36 % (중앙값 9.1 cm, p10 4.9, p90 18.1),
  꺼지는 단차 11 % (중앙값 −20.6 cm). 차도→보도 7.8 cm, 차도→잔디 5.3 cm, 연석석 14.6 cm.
- 센서 측정 vs 정답: 라이다 칸별 MAE 6.6 cm 이지만 평평/단차 분류 일치 93.6 %, depth 88.7 %.
  → 칸 하나의 높이는 거칠고 분류는 믿을 만하다. 여러 프레임의 중앙값으로 쓴다.

## 방법 (프레임 단위)
1. 지면 근처 점(이 프레임 하위 percentile 기준 −0.6 … +0.35 m)을 0.2 m 격자에 넣고 칸별 지면 높이 =
   하위 30 percentile.
2. 각 칸을 0.6 m 이내 이웃 칸과 비교: dz = z_이웃 − z_칸. 완만한 경사는 허용(|dz| − SLOPE_TOL·거리).
   남는 가장 큰 오르막 / 내리막을 그 칸의 이번 프레임 단차로.
3. 높이는 프레임 안에서만 비교한다(센서 기준 z) — 프레임 사이 자세·경사 차이는 상쇄된다.
4. 월드 칸별로 최근 N 프레임 단차의 중앙값을 유지. 관측 수가 MIN_OBS 이상이면 분류:
   step(오르막 ≥ STEP_UP_M 또는 내리막 ≤ −STEP_DROP_M), flat(양쪽 모두 FLAT_M 미만).
"""

from __future__ import annotations

import math
import numpy as np

RES_M = 0.2
NEIGH_CELLS = 3                 # compare with cells up to 0.6 m away (lidar rings are 0.3-0.6 m apart)
SLOPE_TOL = 0.04                # m per m of gentle slope tolerated before it counts as a step
FLAT_M = 0.03
STEP_UP_M = 0.05
STEP_DROP_M = 0.04
# Step LOCATION: a height jump between two cells is recorded at the cell halfway between them
# (the kerb is between them). Recording it on both end cells made a ~1.2 m band that covered a
# robot standing 0.8 m from a kerb (survey_rtC2_p1 spawn: enclosed, V(spawn)=inf).
STEP_MID_M = 0.045
# hysteresis: a cell confirmed as a step stays one until its median falls below STEP_CLEAR_M.
# Live rtC2/rt04b_curb_mppi: medians hovering around STEP_MID_M made the step set change every
# rebuild, the navfield topology flipped (V(robot) 38 -> 13 in 6 s) and MPPI danced.
STEP_CLEAR_M = 0.025
# hazard_xy(): kerbs (lines near the map road edge) from CURB_M, ledges (lines elsewhere) from
# LEDGE_M; blobs never (objects). Lines: >= LINE_MIN_LEN_M long, elongation >= LINE_MIN_ELONG.
CURB_M = 0.06
LEDGE_M = 0.10
LINE_MIN_LEN_M = 1.0
LINE_MIN_ELONG = 3.0
MIN_OBS = 3
HIST = 9                        # per-cell history length (median)
BAND_LO, BAND_HI = -0.6, 0.35   # around the frame's ground reference
MAX_CELLS = 200000


class CurbLayer:
    def __init__(self, res=RES_M):
        self.res = float(res)
        self._idx = {}                                   # (i, j) -> row
        self._keys = np.zeros((1024, 2), np.int64)
        self._up = np.full((1024, HIST), np.nan)         # per-frame max rise history
        self._dn = np.full((1024, HIST), np.nan)         # per-frame max drop history
        self._ptr = np.zeros(1024, np.int64)
        self._nobs = np.zeros(1024, np.int64)
        self._up_med = np.zeros(1024)
        self._dn_med = np.zeros(1024)
        self._mid = np.full((1024, HIST), np.nan)       # step magnitude seen AT this cell (midpoint)
        self._mptr = np.zeros(1024, np.int64)
        self._mobs = np.zeros(1024, np.int64)
        self._mid_med = np.zeros(1024)
        self._is_step = np.zeros(1024, bool)
        self._n = 0
        self.frames = 0

    def _grow(self):
        m = len(self._keys)
        self._keys = np.vstack([self._keys, np.zeros((m, 2), np.int64)])
        self._up = np.vstack([self._up, np.full((m, HIST), np.nan)])
        self._dn = np.vstack([self._dn, np.full((m, HIST), np.nan)])
        self._mid = np.vstack([self._mid, np.full((m, HIST), np.nan)])
        for name in ("_ptr", "_nobs", "_mptr", "_mobs"):
            setattr(self, name, np.concatenate([getattr(self, name), np.zeros(m, np.int64)]))
        for name in ("_up_med", "_dn_med", "_mid_med"):
            setattr(self, name, np.concatenate([getattr(self, name), np.zeros(m)]))
        self._is_step = np.concatenate([self._is_step, np.zeros(m, bool)])

    # ---------------------------------------------------------------- update --
    def update(self, wx, wy, z, src_xy=None, max_range=None):
        """One frame of ground-ish points in world XY with z in ONE consistent frame (sensor or
        base_link). src_xy/max_range: drop points farther than max_range from the sensor.
        Returns the number of cells updated."""
        wx, wy, z = np.asarray(wx, float), np.asarray(wy, float), np.asarray(z, float)
        if src_xy is not None and max_range:
            m = np.hypot(wx - src_xy[0], wy - src_xy[1]) <= max_range
            wx, wy, z = wx[m], wy[m], z[m]
        if len(z) < 20:
            return 0
        ref = float(np.percentile(z, 10))
        m = (z > ref + BAND_LO) & (z < ref + BAND_HI)
        wx, wy, z = wx[m], wy[m], z[m]
        if len(z) < 20:
            return 0
        ci = np.floor(wy / self.res).astype(np.int64)
        cj = np.floor(wx / self.res).astype(np.int64)
        i0, j0 = ci.min(), cj.min()
        h, w = ci.max() - i0 + 1, cj.max() - j0 + 1
        if h * w > 4_000_000:
            return 0
        # per-cell ground = 30th percentile; via sort by (cell, z)
        key = (ci - i0) * w + (cj - j0)
        order = np.lexsort((z, key))
        key, zs = key[order], z[order]
        uk, start, cnt = np.unique(key, return_index=True, return_counts=True)
        ok = cnt >= 2
        uk, start, cnt = uk[ok], start[ok], cnt[ok]
        zc = zs[start + (0.3 * (cnt - 1)).astype(np.int64)]
        G = np.full((h, w), np.nan)
        G[uk // w, uk % w] = zc
        up = np.full((h, w), -np.inf)
        dn = np.full((h, w), np.inf)
        mid = np.full((h, w), -np.inf)          # step magnitude at the midpoint cell
        for di in range(-NEIGH_CELLS, NEIGH_CELLS + 1):
            for dj in range(-NEIGH_CELLS, NEIGH_CELLS + 1):
                if di == 0 and dj == 0:
                    continue
                dist = math.hypot(di, dj) * self.res
                if dist > NEIGH_CELLS * self.res + 1e-6:
                    continue
                a0, a1 = max(0, -di), h - max(0, di)
                b0, b1 = max(0, -dj), w - max(0, dj)
                if a1 <= a0 or b1 <= b0:
                    continue                # frame narrower than this offset
                S = np.full((h, w), np.nan)
                S[a0:a1, b0:b1] = G[a0 + di:a1 + di, b0 + dj:b1 + dj]
                d = S - G
                tol = SLOPE_TOL * dist
                up = np.maximum(up, np.where(np.isfinite(d), d - tol, -np.inf))
                dn = np.minimum(dn, np.where(np.isfinite(d), d + tol, np.inf))
                pi, pj = np.nonzero(np.isfinite(d))
                if len(pi):
                    mi, mj = (2 * pi + di) // 2, (2 * pj + dj) // 2
                    np.maximum.at(mid, (mi, mj), np.maximum(np.abs(d[pi, pj]) - tol, 0.0))
        # midpoint step evidence (any cell a valid pair straddles, ground points or not)
        mi_, mj_ = np.nonzero(np.isfinite(mid))
        if len(mi_):
            mrows = self._rows(mi_ + i0, mj_ + j0)
            ok_ = mrows >= 0
            mrows, mval = mrows[ok_], mid[mi_, mj_][ok_]
            q = self._mptr[mrows]
            self._mid[mrows, q] = mval
            self._mptr[mrows] = (q + 1) % HIST
            self._mobs[mrows] += 1
            med = np.nanmedian(self._mid[mrows], axis=1)
            self._mid_med[mrows] = med
            st = self._is_step[mrows]
            on = (self._mobs[mrows] >= MIN_OBS) & (med >= STEP_MID_M)
            self._is_step[mrows] = np.where(st, med >= STEP_CLEAR_M, on)
        ii, jj = np.nonzero(np.isfinite(G) & (np.isfinite(up) | np.isfinite(dn)))
        if not len(ii):
            self.frames += 1
            return 0
        u = np.clip(up[ii, jj], 0.0, None)
        v = np.clip(dn[ii, jj], None, 0.0)
        rows = self._rows(ii + i0, jj + j0)
        keep = rows >= 0
        rows, u, v = rows[keep], u[keep], v[keep]
        p = self._ptr[rows]
        self._up[rows, p] = u
        self._dn[rows, p] = v
        self._ptr[rows] = (p + 1) % HIST
        self._nobs[rows] += 1
        self._up_med[rows] = np.nanmedian(self._up[rows], axis=1)
        self._dn_med[rows] = np.nanmedian(self._dn[rows], axis=1)
        self.frames += 1
        return len(rows)

    def _rows(self, gi, gj):
        rows = np.empty(len(gi), np.int64)
        for n_, k in enumerate(zip(gi.tolist(), gj.tolist())):
            r = self._idx.get(k)
            if r is None:
                if self._n >= MAX_CELLS:
                    r = -1
                else:
                    if self._n >= len(self._keys):
                        self._grow()
                    r = self._n
                    self._idx[k] = r
                    self._keys[r] = k
                    self._n += 1
            rows[n_] = r
        return rows

    # ---------------------------------------------------------------- queries --
    def _stats(self):
        n = self._n
        m = self._nobs[:n] >= MIN_OBS
        return self._keys[:n][m], self._up_med[:n][m], self._dn_med[:n][m]

    def cells_xy(self, keys):
        return np.stack([(keys[:, 1] + 0.5) * self.res, (keys[:, 0] + 0.5) * self.res], 1)

    def step_discs(self, mag_m=None, near_xy=None, within_m=None):
        """[(x, y, r)] of cells where a step of median magnitude >= mag_m was located (midpoint
        evidence, MIN_OBS frames)."""
        n = self._n
        if mag_m is None:
            m = self._is_step[:n]
        else:
            m = (self._mobs[:n] >= MIN_OBS) & (self._mid_med[:n] >= mag_m)
        xy = self.cells_xy(self._keys[:n][m]) if m.any() else np.zeros((0, 2))
        if near_xy is not None and within_m and len(xy):
            xy = xy[np.hypot(xy[:, 0] - near_xy[0], xy[:, 1] - near_xy[1]) <= within_m]
        r = 0.5 * self.res
        return [(float(x), float(y), r) for x, y in xy]

    def hazard_xy(self, near_edge=None, curb_m=None, ledge_m=None, min_len_m=None,
                  min_elong=None):
        """(N, 2) step cells that are LINES (kerbs, ledges), not blobs (objects on the ground).

        Step cells are grouped into 8-connected components; a component is kept only if its
        extent along its principal axis is >= min_len_m and it is elongated (sqrt of the
        eigenvalue ratio >= min_elong). A lying umbrella, a tumbler, a scooter's base are blobs
        (live rt04b_curb_mppi: the pick target itself read as a step and blocked the goal).
        near_edge(xy) -> bool array: True where a cell is near the map's road edge. Lines near it
        are kerbs (kept from curb_m), lines elsewhere are ledges inside the road (kept only from
        ledge_m -- the rtC2 drop was on map road, 20 cm).
        """
        from scipy import ndimage
        curb_m = CURB_M if curb_m is None else curb_m
        ledge_m = LEDGE_M if ledge_m is None else ledge_m
        min_len_m = LINE_MIN_LEN_M if min_len_m is None else min_len_m
        min_elong = LINE_MIN_ELONG if min_elong is None else min_elong
        n = self._n
        m = self._is_step[:n] & (self._mid_med[:n] >= min(curb_m, ledge_m))
        if not m.any():
            return np.zeros((0, 2))
        keys, mag = self._keys[:n][m], self._mid_med[:n][m]
        i0, j0 = keys[:, 0].min(), keys[:, 1].min()
        G = np.zeros((keys[:, 0].max() - i0 + 1, keys[:, 1].max() - j0 + 1), bool)
        G[keys[:, 0] - i0, keys[:, 1] - j0] = True
        lab, nl = ndimage.label(G, structure=np.ones((3, 3), bool))
        cl_ = lab[keys[:, 0] - i0, keys[:, 1] - j0]
        xy = self.cells_xy(keys)
        keep = np.zeros(len(keys), bool)
        for L in range(1, nl + 1):
            idx = np.nonzero(cl_ == L)[0]
            if len(idx) < 3:
                continue
            p = xy[idx] - xy[idx].mean(0)
            ev, evec = np.linalg.eigh(np.cov(p.T))
            proj = p @ evec[:, 1]
            length = proj.max() - proj.min() + self.res
            elong = math.sqrt(ev[1] / max(ev[0], 1e-6))
            if length >= min_len_m and elong >= min_elong:
                keep[idx] = True
        if near_edge is not None:
            edge = np.asarray(near_edge(xy), bool)
            keep &= np.where(edge, mag >= curb_m, mag >= ledge_m)
        else:
            keep &= mag >= curb_m
        return xy[keep]

    def flat_xy(self, flat_m=None):
        """(N, 2) centres of cells observed flat (no rise / drop >= flat_m around them)."""
        flat_m = FLAT_M if flat_m is None else flat_m
        keys, up, dn = self._stats()
        m = (up < flat_m) & (dn > -flat_m)
        return self.cells_xy(keys[m]) if m.any() else np.zeros((0, 2))

    def summary(self):
        keys, up, dn = self._stats()
        return {"cells": int(self._n), "confirmed": int(len(keys)),
                "step": int(self._is_step[:self._n].sum()),
                "flat": int(((up < FLAT_M) & (dn > -FLAT_M)).sum()), "frames": self.frames}


def lidar_world(pts, pose, mount_xy=(0.0, 0.0), rmin=1.0, rmax=5.5):
    """Sensor-frame lidar points -> (wx, wy, z_sensor) of the ground-seeing ring band."""
    r = np.hypot(pts[:, 0], pts[:, 1])
    m = (r >= rmin) & (r <= rmax)
    p = pts[m]
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    px, py = p[:, 0] + mount_xy[0], p[:, 1] + mount_xy[1]
    return x + px * c - py * s, y + px * s + py * c, p[:, 2]
