"""Navigation function (cost-to-go field) for the Stage 2 MPPI local planner.

## 무엇인가
목표까지의 "남은 비용" 장 V(x, y) 를 격자 전체에 대해 Dijkstra 로 한 번에 계산한다
(wavefront / navfn 계열). 격자 위에서 V 는 목표 외에 **지역 최소점이 없다** -- 모든 칸이
더 작은 V 를 가진 이웃을 따라 목표까지 내려갈 수 있다. 인공 포텐셜장(APF)이 인력/척력의
합으로 만드는 장과 쓰임새는 같지만 그 성질이 보장된다는 점이 다르다.

MPPI 는 이 V 를 궤적 끝점의 비용으로 쓴다 (mppi.py). 그래서 원호 우회(waypoint_avoid.
relocate)나 접선 그래프 같은 별도 우회 경로 생성이 필요 없다 -- 장애물을 돌아가는 최단
방향이 이미 V 에 들어 있다.

## 격자
- 시작/목표 bounding box + MARGIN_M 영역만, CELL_M 해상도 (점유지도 0.05 m 를 max-pool).
- lethal(로봇 중심이 들어갈 수 없음): 지도 점유(100) + 라이다/정체기억 원판, 로봇 반경
  INSCRIBED_M 만큼 팽창 (거리변환).
- 미지 칸(-1): 지도상 도로 밖(잔디, 보도). 도로 경계 = 연석인데, 연석은 라이다 밴드
  (지면 + 0.15 m 부터) 아래라 센서로는 안 보인다. 그래서 **기본은 lethal** (레거시 A* 와
  같이 경계가 벽이 되고, 팽창 비용으로 경계에서 떨어져 달린다). 예외로 `offroad_ok`
  원 안(목표 근처, hint_radius -- 도로에 닿을 때까지 넓힘)만 **통과 가능하되 비싸다**
  (UNKNOWN_COST) -- rtC2 처럼 물체가 잔디 위에 있거나 owner zone 이 보도 깊숙이 있어도
  계획이 실패하지 않게. 로봇 자신이 도로 밖에 서 있으면
  가장 가까운 도로 칸까지(최소 OFFROAD_ESCAPE_M) 원을 열어 도로로 돌아올 길을 둔다.
  (이전 버전은 미지 칸 전체를 "비싸지만 통과"로 봤고, rt01 에서 도로 밖 지형으로 올라가
  전복/tilt guard 가 났다.)
- 팽창 비용: lethal 경계에서 INFLATION_M 까지 지수 감쇠 -- 장애물에 붙은 경로를 싫어하게.
- 목표 칸이 lethal 이면(정차 지점이 차/연석과 겹침, rt04/rt04b) SNAP_M 안의 가장 가까운
  통과 가능 칸으로 목표를 옮기고 `goal_snapped` 로 알린다.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

CELL_M = 0.2
MARGIN_M = 10.0
INSCRIBED_M = 0.30        # chassis half-width + slack: centre closer than this = contact
INFLATION_M = 1.0         # inflation cost decays to ~0 by this distance from lethal
INFLATION_DECAY = 3.0     # 1/m
UNKNOWN_COST = 1.5        # extra cost per metre on an unknown (off-road) cell
INFLATION_WEIGHT = 2.0    # extra cost per metre at the lethal edge
SNAP_M = 1.2
OFFROAD_ESCAPE_M = 1.5    # robot standing off-road: unknown cells this close (at least as
                          # far as the nearest road cell + margin) stay passable
LETHAL = np.inf


class NavField:
    def __init__(self, cell_m=CELL_M, inscribed_m=INSCRIBED_M):
        self.cell = float(cell_m)
        self.inscribed = float(inscribed_m)
        self.ready = False
        self.goal = None            # effective goal (after snapping)
        self.goal_snapped = False

    # ------------------------------------------------------------------ build --
    def build(self, grid, origin, res, start_xy, goal_xy, discs=(), goal_clear_m=0.0,
              keep_discs=(), free_xy=None, offroad_ok=(), step_discs=(), flat_xy=None,
              edge_band_m=0.0, hazard_xy=None):
        """grid: int8 (H, W) occupancy, row 0 = bottom (OccupancyGridPlanner._grid);
        origin: (ox, oy); res: map m/cell; discs: [(x, y, r)] extra obstacles (lidar cells,
        stall memory); goal_clear_m: obstacle discs within this of the goal are dropped
        (the pick target itself shows up in the lidar); keep_discs: discs never dropped --
        the pick target itself, which sits 0.58 m from the standoff goal and is too low for
        the lidar band (live rt01: MPPI circled the standoff straight over the tumbler);
        free_xy: the robot's own position -- a disc reaching it is cut to its part in front
        of the robot, so the robot is never lethal (it is standing there, so it is free).
        Live rt01_mppi_r4: the stall mark 0.85 m ahead (r=0.6) + inscribed 0.3 covered the
        robot, then grew to r=1.4 -- V(robot)=inf, every rollout "collided", 140 s pinned.
        offroad_ok: [(x, y, r)] circles where unknown (off-road) cells are passable at
        UNKNOWN_COST; everywhere else they are obstacles (see the module docstring). A circle
        is widened to reach the nearest road cell (+ inscribed + 2 cells), so it always
        connects to the road; the radii used are in `offroad_ok_eff`.
        None = all off-road passable at UNKNOWN_COST (fallback when the goal is cut off).
        step_discs: [(x, y, r)] kerb / step cells from curb_layer -- obstacles in every case
        (never dropped near the goal, also with offroad_ok=None).
        flat_xy: (N, 2) cell centres curb_layer observed flat -- off-road there is passable
        (the map's road edge is not a kerb where the sensors see no step).
        hazard_xy: (N, 2) kerb / ledge cells (curb_layer.hazard_xy) that only the ROLLOUT
        collision check sees (lookup()'s lethal) -- not the cost-to-go, not the inflation. The
        path stays the map-based one; MPPI just never picks a trajectory over a kerb or ledge.
        edge_band_m: with offroad_ok=None (all off-road open), off-road cells within this of the
        road stay obstacles unless observed flat -- the robot may only leave the road where the
        sensors saw no kerb, while the off-road interior beyond is open (live rtC2_curb_mppi:
        with the whole edge open it kept heading for unseen edge right next to it, found a kerb,
        replanned, for minutes; the flush entry it had driven past was 25 m back)."""
        ox, oy = origin
        x0 = min(start_xy[0], goal_xy[0]) - MARGIN_M
        y0 = min(start_xy[1], goal_xy[1]) - MARGIN_M
        x1 = max(start_xy[0], goal_xy[0]) + MARGIN_M
        y1 = max(start_xy[1], goal_xy[1]) + MARGIN_M
        f = max(1, int(round(self.cell / res)))
        c0 = max(0, int((x0 - ox) / res) // f * f)
        r0 = max(0, int((y0 - oy) / res) // f * f)
        c1 = min(grid.shape[1], int(math.ceil((x1 - ox) / res / f)) * f)
        r1 = min(grid.shape[0], int(math.ceil((y1 - oy) / res / f)) * f)
        sub = grid[r0:r1, c0:c1]
        h, w = sub.shape[0] // f, sub.shape[1] // f
        sub = sub[:h * f, :w * f].reshape(h, f, w, f)
        occ = (sub == 100).any(axis=(1, 3))
        unknown = (sub < 0).mean(axis=(1, 3)) > 0.5
        self.ox = ox + c0 * res
        self.oy = oy + r0 * res
        self.h, self.w = h, w
        yy, xx = np.mgrid[0:h, 0:w]
        cx = self.ox + (xx + 0.5) * self.cell
        cy = self.oy + (yy + 0.5) * self.cell

        # off-road (unknown) cells: obstacles except inside the offroad_ok circles
        if offroad_ok is not None:
            offroad = unknown.copy()
        elif edge_band_m > 0:
            road_ = ~unknown & ~occ
            k_ = max(1, int(round(edge_band_m / self.cell)))
            offroad = unknown & ndimage.binary_dilation(road_, iterations=k_)
        else:
            offroad = np.zeros_like(unknown)
        road = ~unknown & ~occ
        self.offroad_ok_eff = []
        for (ax, ay, ar) in offroad_ok or ():
            # widen the circle until it reaches the road, so the goal is always connected.
            # Live rt01_ab_mppi 2026-10-05: the owner zone (leg 2 goal, shared by every
            # problem) sits deeper off-road than the 1.0 m circle -> "goal unreachable" on
            # nearly every rebuild -> fallback opened ALL off-road -> the robot took a
            # detour over the kerb and hung on it (pitch 13.5 deg).
            d2 = ((cx - ax) ** 2 + (cy - ay) ** 2)[road]
            if d2.size:
                ar = max(ar, math.sqrt(float(d2.min())) + self.inscribed + 2 * self.cell)
            self.offroad_ok_eff.append((ax, ay, ar))
            offroad &= (cx - ax) ** 2 + (cy - ay) ** 2 > ar * ar
        if free_xy is not None and offroad.any():
            ri, rj = self._ij(*free_xy)
            ri, rj = min(max(ri, 0), h - 1), min(max(rj, 0), w - 1)
            # off-road already: open a way back. On the road next to the edge: keep the
            # robot itself out of lethal (as for discs below).
            hole = self.inscribed + self.cell
            if unknown[ri, rj]:
                d2 = ((cx - free_xy[0]) ** 2 + (cy - free_xy[1]) ** 2)[road]
                to_road = math.sqrt(float(d2.min())) if d2.size else 0.0
                hole = max(OFFROAD_ESCAPE_M, to_road + self.inscribed + 2 * self.cell)
            offroad &= (cx - free_xy[0]) ** 2 + (cy - free_xy[1]) ** 2 > hole * hole
        if flat_xy is not None and len(flat_xy) and offroad.any():
            # sensed flat ground: the map's road edge is not a kerb there (curb_layer)
            fi = np.floor((np.asarray(flat_xy)[:, 1] - self.oy) / self.cell).astype(np.int64)
            fj = np.floor((np.asarray(flat_xy)[:, 0] - self.ox) / self.cell).astype(np.int64)
            ok = (fi >= 0) & (fi < h) & (fj >= 0) & (fj < w)
            self.flat_opened = int(offroad[fi[ok], fj[ok]].sum())
            offroad[fi[ok], fj[ok]] = False
        else:
            self.flat_opened = 0
        occ |= offroad
        unknown &= ~offroad                 # what is left is the passable-but-costly part
        self.offroad = offroad

        # extra obstacles as discs
        keep_discs = list(keep_discs) + list(step_discs)
        if discs or keep_discs:
            gx, gy = goal_xy
            keep = [(float(a), float(b), float(c), True) for (a, b, c) in keep_discs]
            for (dx, dy, r, *kept) in [(a, b, c) for (a, b, c) in discs] + keep:
                if not kept and goal_clear_m > 0 and math.hypot(dx - gx, dy - gy) <= goal_clear_m:
                    continue
                j0 = int((dx - r - self.ox) / self.cell) - 1
                j1 = int((dx + r - self.ox) / self.cell) + 2
                i0 = int((dy - r - self.oy) / self.cell) - 1
                i1 = int((dy + r - self.oy) / self.cell) + 2
                if j1 <= 0 or i1 <= 0 or j0 >= w or i0 >= h:
                    continue
                j0, i0, j1, i1 = max(j0, 0), max(i0, 0), min(j1, w), min(i1, h)
                px, py = cx[i0:i1, j0:j1], cy[i0:i1, j0:j1]
                m = (px - dx) ** 2 + (py - dy) ** 2 <= r * r
                if free_xy is not None:
                    # A disc reaching the robot keeps only its part beyond a line in front
                    # of it: full width (a grown stall mark still forces a wide detour), but
                    # the robot is outside it and the way back stays open.
                    hole = self.inscribed + self.cell
                    ux, uy = dx - free_xy[0], dy - free_xy[1]
                    d = math.hypot(ux, uy)
                    if d < r + hole:
                        if d > 1e-3:
                            m &= (px - free_xy[0]) * ux / d + (py - free_xy[1]) * uy / d >= hole
                        else:
                            m &= (px - free_xy[0]) ** 2 + (py - free_xy[1]) ** 2 > hole * hole
                occ[i0:i1, j0:j1] |= m

        # distance to the nearest obstacle cell (m)
        dist = ndimage.distance_transform_edt(~occ) * self.cell if occ.any() else \
            np.full((h, w), 1e3)
        self.dist = dist
        self.lethal = dist < self.inscribed
        infl = np.where(self.lethal, 0.0,
                        np.exp(-INFLATION_DECAY * np.maximum(dist - self.inscribed, 0.0)))
        infl[dist > INFLATION_M + self.inscribed] = 0.0
        self.infl = infl
        # per-metre traversal cost of each cell
        cost = 1.0 + INFLATION_WEIGHT * infl + UNKNOWN_COST * unknown
        cost[self.lethal] = LETHAL
        self.cell_cost = cost

        # effective goal
        gi, gj = self._ij(*goal_xy)
        gi = min(max(gi, 0), h - 1)
        gj = min(max(gj, 0), w - 1)
        self.goal_snapped = False
        if self.lethal[gi, gj]:
            k = int(math.ceil(SNAP_M / self.cell))
            i0, i1, j0, j1 = max(gi - k, 0), min(gi + k + 1, h), max(gj - k, 0), min(gj + k + 1, w)
            free = ~self.lethal[i0:i1, j0:j1]
            if free.any():
                ii, jj = np.nonzero(free)
                d2 = (ii + i0 - gi) ** 2 + (jj + j0 - gj) ** 2
                n = int(np.argmin(d2))
                if math.sqrt(d2[n]) * self.cell <= SNAP_M:
                    gi, gj = int(ii[n] + i0), int(jj[n] + j0)
                    self.goal_snapped = True
        self.goal = self._xy(gi, gj) if self.goal_snapped else (float(goal_xy[0]), float(goal_xy[1]))
        self.field = self._dijkstra(gi, gj)
        self.ready = np.isfinite(self.field).any()
        self.hazard = np.zeros((h, w), bool)
        if hazard_xy is not None and len(hazard_xy):
            hz = np.zeros((h, w), bool)
            hi_ = np.floor((np.asarray(hazard_xy)[:, 1] - self.oy) / self.cell).astype(np.int64)
            hj_ = np.floor((np.asarray(hazard_xy)[:, 0] - self.ox) / self.cell).astype(np.int64)
            ok_ = (hi_ >= 0) & (hi_ < h) & (hj_ >= 0) & (hj_ < w)
            hz[hi_[ok_], hj_[ok_]] = True
            if hz.any():
                self.hazard = ndimage.distance_transform_edt(~hz) * self.cell < self.inscribed
        # descent direction (world angle of -grad V): the field has no notion of heading, so
        # MPPI also scores how well the chassis faces downhill -- otherwise "spin to face the
        # goal" and "stand still" cost the same (measured offline: leg2 starting 145 deg off
        # crept at ~0.1 m/s for 120 s).
        f = self.field.copy()
        fin = np.isfinite(f)
        f[~fin] = (f[fin].max() + 10.0) if fin.any() else 0.0
        gy, gx = np.gradient(f, self.cell)
        self.descent = np.arctan2(-gy, -gx)
        self.descent_ok = fin & (np.hypot(gx, gy) > 1e-3)
        return self

    def _dijkstra(self, gi, gj):
        h, w = self.h, self.w
        n = h * w
        c = self.cell_cost.ravel()
        idx = np.arange(n).reshape(h, w)
        rows, cols, wts = [], [], []
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = idx[max(0, -di):h - max(0, di), max(0, -dj):w - max(0, dj)]
            b = idx[max(0, di):h + min(0, di) or None, max(0, dj):w + min(0, dj) or None]
            a, b = a.ravel(), b.ravel()
            step = self.cell * (math.sqrt(2.0) if di and dj else 1.0)
            wt = 0.5 * (c[a] + c[b]) * step
            ok = np.isfinite(wt)
            rows += [a[ok], b[ok]]
            cols += [b[ok], a[ok]]
            wts += [wt[ok], wt[ok]]
        g = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n))
        d = dijkstra(g, directed=True, indices=int(idx[gi, gj]))
        return d.reshape(h, w)

    # ---------------------------------------------------------------- queries --
    def _ij(self, x, y):
        return int((y - self.oy) / self.cell), int((x - self.ox) / self.cell)

    def _xy(self, i, j):
        return (self.ox + (j + 0.5) * self.cell, self.oy + (i + 0.5) * self.cell)

    def lookup(self, xs, ys):
        """Vectorised (field, inflation, lethal) at world points. Out of the window counts
        as lethal (the window already spans start/goal + MARGIN_M)."""
        j = np.floor((np.asarray(xs) - self.ox) / self.cell).astype(np.int64)
        i = np.floor((np.asarray(ys) - self.oy) / self.cell).astype(np.int64)
        inside = (i >= 0) & (i < self.h) & (j >= 0) & (j < self.w)
        ic, jc = np.clip(i, 0, self.h - 1), np.clip(j, 0, self.w - 1)
        field = np.where(inside, self.field[ic, jc], np.inf)
        lethal = ~inside | self.lethal[ic, jc] | ~np.isfinite(field) | self.hazard[ic, jc]
        return field, np.where(inside, self.infl[ic, jc], 1.0), lethal

    def descent_at(self, xs, ys):
        """(angle, valid) of the downhill direction at world points."""
        j = np.clip(np.floor((np.asarray(xs) - self.ox) / self.cell).astype(np.int64), 0, self.w - 1)
        i = np.clip(np.floor((np.asarray(ys) - self.oy) / self.cell).astype(np.int64), 0, self.h - 1)
        return self.descent[i, j], self.descent_ok[i, j]

    def value(self, x, y):
        f, _, _ = self.lookup(np.array([x]), np.array([y]))
        return float(f[0])
