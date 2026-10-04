"""Mock navigation agent -- stands in for the participant's Stage 2 navigation.

This module bundles the demo's navigation pieces (participants replace/extend them):
  - GoalSeekController / PathFollower -- P-control that returns ``(linear_x, angular_z)``
    scalars to publish with ``client.send_cmd_vel(...)``.
  - OccupancyGridPlanner -- A* path planning over the runtime occupancy grid
    (``MARCClient.get_occupancy_map()``).
  - load_obstacles / obstacles_excluding_goal -- extra 3D obstructions the occupancy map
    omits (vehicles/bicycles/people/objects), read from ``mock_demo_data.yaml``
    (pre-extracted by ``tools/mock_data_builder/gen_mock_demo_data.py``).
"""

# ============================================================================
# [파일 출처 안내]
# 이 파일은 MARC 2026 starter kit이 제공한 demo/mock_agent_navigation.py가 원본이다.
# GoalSeekController / PathFollower / OccupancyGridPlanner(A*)와 load_obstacles 계열
# 함수는 전부 starter kit 원본 그대로이며(로직 변경 없음), 우리 팀이 실제로 추가한 건
# 아래 3곳뿐이다 (각 위치에 [팀 추가] 표시):
#   1) 모듈 레벨 STEERING_TRACK 상수 + cap_angular() 함수
#      -- 원래는 GoalSeekController.compute() 안에 똑같은 로직이 인라인으로 박혀 있었는데,
#         participant_app.py의 _approach_straight()에서도 같은 계산이 필요해져서 공용
#         함수로 뽑아냄.
#   2) OccupancyGridPlanner.is_free() 메서드 -- waypoint_avoid.relocate()가 우회 방향이
#      도로 밖(잔디/벽)인지 확인할 때 씀.
#   3) nearest_only_exclude() 함수 -- waypoint_avoid.relocate()용 장애물 원 목록을 만들 때,
#      목표에 가장 가까운 원 하나만 제외하기 위함(obstacles_excluding_goal은 반경 안 전부를
#      지워서 라이다 클러스터 회피용으로는 너무 거칠다).
# ============================================================================

import math


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# [팀 추가] 아래 STEERING_TRACK 상수 + cap_angular() 함수는 starter kit에 없던 것.
# 4-wheel-steer track (m). Same value as the platform's chassis steering track.
# The chassis STOPS (applies 0) on a command where 2*v <= w*T -- a turn too tight for the
# speed. It reports no error and keeps the last command, so the robot simply stops moving.
# (원래 GoalSeekController.compute() 안에 인라인으로만 있던 계산을, 다른 파일
#  participant_app._approach_straight()에서도 재사용하려고 모듈 함수로 승격시켰다.)
STEERING_TRACK = 0.38


def cap_angular(linear: float, angular: float, track: float = STEERING_TRACK) -> float:
    """Clamp angular_z so that (linear, angular) stays a *valid* chassis command.

    Every producer of cmd_vel must pass through here. A command with 2*v <= w*T makes the
    chassis stall silently, which is indistinguishable from "goal unreachable" in the logs
    -- see the straight-approach stall that cost 20s/attempt before this was shared.
    Pure rotation (linear ~ 0) is exempt: the chassis can spin in place (X-pattern steering).
    """
    if linear <= 1e-3:
        return angular
    w_max = 1.8 * linear / track
    return _clamp(angular, -w_max, w_max)


class GoalSeekController:
    """Simple proportional (P) goal-seeking controller.

    ``compute(pose, goal, cruise)`` returns one step of ``(linear_x, angular_z, reached)``.
    When the heading error is large it rotates in place; when it is small it moves forward
    with heading correction, decelerating near the goal (unless ``cruise``).
    """

    # Reached test -- XY distance (m)
    GOAL_TOLERANCE = 0.7
    # Above this heading error, rotate in place (linear=0 -> pure wz -> chassis X-pattern
    # in-place spin); below it, move forward with heading correction. The chassis can spin in
    # place (X-pattern steering + opposite left/right wheels), so large turns (e.g. a goal
    # behind the robot) spin in place first, then drive forward -- avoids the Ackermann-only
    # "turn around and stall" problem.
    HEADING_TOLERANCE = math.pi / 4   # ~45 deg: beyond this, rotate in place then advance

    # Speed / turn caps. Large turns are handled in place (see HEADING_TOLERANCE), so forward
    # motion rarely needs a hard swerve; the Ackermann w-limit below guarantees a tight arc.
    # [팀 수정] 0.8 -> 1.1. SDK 상한은 1.5 지만 상단 무거운 차체라 여유를 남긴다.
    MAX_LINEAR = 1.1           # m/s
    MAX_ANGULAR = 1.2          # rad/s
    K_ANGULAR = 0.8
    # [팀 수정] 감속은 K_LINEAR*dist(비례) 대신 사다리꼴 프로파일 v = sqrt(2*a*d).
    # 비례 감속은 1m 남은 지점에서 이미 0.5 m/s 로 떨어져 꼬리가 길었다. 여기서 d 는
    # "정지 경계(GOAL_TOLERANCE)까지 남은 거리"라서 경계에 V_MIN 으로 들어선다.
    # V_MIN 이 없으면 경계 직전에 속도가 0 으로 수렴해 영원히 도달하지 못한다.
    DECEL = 0.5                # m/s^2 -- 계획 감속도 (1.1 m/s 에서 정지까지 ~1.2m)
    V_MIN = 0.12               # m/s   -- 정지 경계 진입 속도
    # [팀 추가] 횡가속도(v*w) 상한 -- 전복 방지. 1.0 은 예전 0.8 m/s x 1.2 rad/s 의
    # 최악값(0.96)과 같은 수준이라, 선회 중 하중은 기존과 같고 직선만 빨라진다.
    MAX_LAT_ACCEL = 1.0        # m/s^2
    # 4-wheel-steer track (m) -- see the module-level STEERING_TRACK / cap_angular().
    STEERING_TRACK = STEERING_TRACK

    def compute(self, pose: tuple, goal: tuple, cruise: bool = False,
                remaining: float = None) -> tuple:
        """Compute one step of commands from the current pose and goal.

        Args:
            pose: (x, y, yaw_radians) -- world coordinates (client.get_world_pose()).
            goal: (gx, gy)           -- target XY coordinates (m).
            cruise: if True, skip the near-goal deceleration and cruise at MAX_LINEAR --
                for pass-through goals that are not a stop point.
            remaining: distance (m) left to the actual stop point, when it is further than
                ``goal`` (PathFollower passes the remaining path length, so the profile
                starts braking for the final stop and not for every intermediate
                waypoint). None -> straight-line distance to ``goal``.

        Returns:
            (linear_x, angular_z, reached: bool)
        """
        x, y, yaw = pose
        gx, gy = goal
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        if dist < self.GOAL_TOLERANCE:
            return 0.0, 0.0, True

        target_yaw = math.atan2(dy, dx)
        heading_err = _wrap_pi(target_yaw - yaw)

        angular = _clamp(self.K_ANGULAR * heading_err,
                         -self.MAX_ANGULAR, self.MAX_ANGULAR)
        if abs(heading_err) > self.HEADING_TOLERANCE:
            linear = 0.0                       # rotate in place
        elif cruise:
            linear = self.MAX_LINEAR           # pass-through waypoint: no deceleration
        else:
            to_stop = (dist if remaining is None else remaining) - self.GOAL_TOLERANCE
            linear = _clamp(math.sqrt(2.0 * self.DECEL * max(0.0, to_stop)),
                            self.V_MIN, self.MAX_LINEAR)
        if linear > 0.0 and abs(angular) > 1e-6:
            linear = min(linear, self.MAX_LAT_ACCEL / abs(angular))

        # Ackermann validity: a turn too tight for the speed (2v <= w*T) makes the chassis
        # stop. Cap w within 2v/T so even a goal behind the robot keeps moving in the
        # tightest valid arc (this chassis must move forward to steer).
        angular = cap_angular(linear, angular, self.STEERING_TRACK)

        return linear, angular, False


class PathFollower:
    """Controller that follows a waypoint path in sequence.

    Takes the world waypoint list produced by ``OccupancyGridPlanner.plan()`` and,
    every step, computes ``GoalSeekController`` commands from the current pose toward
    the active waypoint. When it gets close enough to the active waypoint it advances
    to the next, and it is done once it reaches the last waypoint (= goal).
    """

    # Distance to consider an intermediate waypoint passed (m). Looser than the goal
    # tolerance to smooth out corners.
    WAYPOINT_TOLERANCE = 1.0

    def __init__(self, path, controller: GoalSeekController = None,
                 passthrough: bool = False):
        self.path = [(float(x), float(y)) for (x, y) in path]
        self.idx = 0
        self.ctrl = controller or GoalSeekController()
        # passthrough=True: the final point is also a pass-through (no deceleration) --
        # used when this goal is itself a waypoint, not a stop point.
        self._passthrough = bool(passthrough)

    def compute(self, pose: tuple):
        """Compute one step of commands from the current pose.

        Returns:
            (linear_x, angular_z, done: bool, info: dict)
            info = {"idx", "total", "waypoint": (x, y)}
        """
        x, y, _ = pose
        # Skip intermediate waypoints that have been reached (keep the last one).
        while self.idx < len(self.path) - 1:
            wx, wy = self.path[self.idx]
            if math.hypot(wx - x, wy - y) < self.WAYPOINT_TOLERANCE:
                self.idx += 1
            else:
                break

        wx, wy = self.path[self.idx]
        final = self.idx >= len(self.path) - 1
        # [팀 수정] 중간 웨이포인트도 무조건 cruise 하지 않고, "남은 경로 길이" 기준
        # 감속 프로파일을 따른다. 마지막 웨이포인트가 가까운 꺾인 경로에서 최고속으로
        # 마지막 점에 들어서면 그 짧은 거리 안에 못 멈추기 때문이다. 멀리 있을 때는
        # 프로파일이 MAX_LINEAR 로 포화되므로 예전 cruise 와 같다.
        remaining = math.hypot(wx - x, wy - y)
        for (ax, ay), (bx, by) in zip(self.path[self.idx:], self.path[self.idx + 1:]):
            remaining += math.hypot(bx - ax, by - ay)
        linear, angular, reached = self.ctrl.compute(
            pose, (wx, wy), cruise=self._passthrough, remaining=remaining)
        done = final and reached
        info = {"idx": self.idx, "total": len(self.path), "waypoint": (wx, wy)}
        return linear, angular, done, info


# ---------------------------------------------------------------------------
# Occupancy-grid A* path planning.
# ---------------------------------------------------------------------------
import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)


class OccupancyGridPlanner:
    """A* planner over an occupancy grid.

    Update the grid with ``update_from_msg()`` and get a list of world-coordinate
    waypoints with ``plan()``. If there is no grid, ``ready == False``.
    """

    def __init__(self):
        self._grid = None        # int8 (H, W), row0 = bottom
        self._res = None         # m/cell
        self._ox = None          # origin x (world)
        self._oy = None          # origin y (world)
        self._h = 0
        self._w = 0

    # -- grid update --

    def update_from_msg(self, msg) -> bool:
        """Update the grid from a nav_msgs/OccupancyGrid message."""
        info = msg.info
        w = int(info.width)
        h = int(info.height)
        if w == 0 or h == 0:
            return False
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        self._grid = grid
        self._res = float(info.resolution)
        self._ox = float(info.origin.position.x)
        self._oy = float(info.origin.position.y)
        self._h = h
        self._w = w
        return True

    @property
    def ready(self) -> bool:
        return self._grid is not None

    @property
    def resolution(self):
        return self._res

    # -- coordinate conversion --

    def _w2c(self, wx, wy):
        """world (x, y) -> cell (row, col)."""
        col = int((wx - self._ox) / self._res)
        row = int((wy - self._oy) / self._res)
        return row, col

    def _c2w(self, row, col):
        """cell (row, col) -> world (x, y) (cell center)."""
        wx = self._ox + (col + 0.5) * self._res
        wy = self._oy + (row + 0.5) * self._res
        return wx, wy

    # -- path planning --

    def plan(self, start_xy, goal_xy, inflation_m=0.25, margin_m=12.0,
             snap_m=1.5, extra_obstacles=None):
        """Return a detour path from start_xy -> goal_xy as a list of world waypoints.

        Args:
            start_xy: (x, y) current robot position (world).
            goal_xy:  (x, y) target position (world).
            inflation_m: erode the free area by the robot radius to build a path that
                         stays away from the road boundary (avoids step stalls).
            margin_m: search margin outside the start/goal bounding box (crop for planning speed).
            snap_m:  maximum radius to correct a start/goal cell to the nearest free
                     cell when it is not traversable.
            extra_obstacles: [(x, y, radius_m), ...] static obstacles not in the
                     occupancy map (vehicles etc.). Their disc is carved out as not traversable.

        Returns:
            [(x, y), ...] path waypoints. If the goal cell is traversable the last one is
            exactly goal_xy; if it is blocked (a vehicle etc.), up to the free cell nearest
            the goal. None on failure.
        """
        if not self.ready:
            return None

        sr, sc = self._w2c(*start_xy)
        gr, gc = self._w2c(*goal_xy)
        if not (self._in_bounds(sr, sc) and self._in_bounds(gr, gc)):
            return None

        # 1) Crop only around start/goal -- avoid A*/erosion over the full grid (tens of millions of cells).
        margin_c = int(math.ceil(margin_m / self._res))
        rmin = max(0, min(sr, gr) - margin_c)
        rmax = min(self._h, max(sr, gr) + margin_c + 1)
        cmin = max(0, min(sc, gc) - margin_c)
        cmax = min(self._w, max(sc, gc) + margin_c + 1)
        sub = self._grid[rmin:rmax, cmin:cmax]

        # 2) traversable (free) mask. First carve out obstacles not in the occupancy
        #    map (vehicles etc.), then erode (inflate) by the robot radius.
        trav = sub == 0
        if extra_obstacles:
            self._block_obstacles(trav, extra_obstacles, rmin, cmin)
        infl_c = int(round(inflation_m / self._res))
        if infl_c > 0:
            trav = self._erode(trav, infl_c)

        # crop-local coordinates. Record whether the goal cell is traversable (after
        # carve/erosion) -- if it is blocked, do not overwrite the last waypoint with the
        # exact goal (so the robot does not charge back into the vehicle).
        ls, lg = (sr - rmin, sc - cmin), (gr - rmin, gc - cmin)
        H, W = trav.shape
        goal_free = (0 <= lg[0] < H and 0 <= lg[1] < W and bool(trav[lg[0], lg[1]]))
        ls = self._snap(trav, ls, snap_m)
        lg = self._snap(trav, lg, snap_m)
        if ls is None or lg is None:
            return None

        # 3) A*
        cells = self._astar(trav, ls, lg)
        if not cells:
            return None

        # 4) line-of-sight simplification -> convert to world coordinates
        cells = self._simplify(trav, cells)
        path = [self._c2w(r + rmin, c + cmin) for (r, c) in cells]

        # Replace the last waypoint with the exact goal only when the goal cell is traversable.
        # Remove the first waypoint (the current-position cell).
        if path:
            if goal_free:
                path[-1] = (float(goal_xy[0]), float(goal_xy[1]))
            if len(path) > 1:
                path = path[1:]
        return path

    def _block_obstacles(self, trav, obstacles, rmin, cmin):
        """Mark each (x, y, radius) disc from extra_obstacles as not traversable in trav."""
        H, W = trav.shape
        for (wx, wy, rad) in obstacles:
            r, c = self._w2c(wx, wy)
            lr, lc = r - rmin, c - cmin
            rad_c = int(round(float(rad) / self._res))
            if rad_c <= 0:
                continue
            if lr + rad_c < 0 or lr - rad_c >= H or lc + rad_c < 0 or lc - rad_c >= W:
                continue  # outside the crop
            r0, r1 = max(0, lr - rad_c), min(H, lr + rad_c + 1)
            c0, c1 = max(0, lc - rad_c), min(W, lc + rad_c + 1)
            yy, xx = np.ogrid[r0:r1, c0:c1]
            disc = (yy - lr) ** 2 + (xx - lc) ** 2 <= rad_c ** 2
            trav[r0:r1, c0:c1][disc] = False

    # -- internal helpers --

    # [팀 추가] is_free()는 starter kit에 없던 메서드. waypoint_avoid.relocate()가
    # 우회 경로 후보(원 둘레를 따라 도는 호)가 실제 도로 위인지, 잔디/벽 밖으로
    # 새는지 확인하는 용도로만 쓴다.
    def is_free(self, x, y):
        """월드 점이 점유지도 상 자유공간(원시 셀값 0)인가.

        `extra_obstacles`/inflation 은 모른다 -- 계획용이 아니라, waypoint_avoid.relocate()
        가 원 둘레 우회의 어느 쪽이 도로 밖(잔디·벽)으로 나가는지 고를 때 쓴다. 범위
        밖/미탐사 셀은 자유공간이 아닌 것으로 본다 -- 안 보이는 셀을 안전한 우회로
        취급하면 안 되므로 보수적으로 막힌 쪽으로 판정한다.
        """
        if not self.ready:
            return True   # 격자가 아직 없으면 판단 근거가 없다 -- 막지 않는다
        r, c = self._w2c(x, y)
        if not self._in_bounds(r, c):
            return False
        return bool(self._grid[r, c] == 0)

    def _in_bounds(self, r, c):
        return 0 <= r < self._h and 0 <= c < self._w

    @staticmethod
    def _erode(mask, iters):
        """8-neighbor erosion, iters times. Border cells are treated as blocked (protected by the crop margin)."""
        g = mask
        for _ in range(iters):
            e = np.zeros_like(g)
            e[1:-1, 1:-1] = (
                g[1:-1, 1:-1] & g[:-2, 1:-1] & g[2:, 1:-1]
                & g[1:-1, :-2] & g[1:-1, 2:]
                & g[:-2, :-2] & g[:-2, 2:] & g[2:, :-2] & g[2:, 2:]
            )
            g = e
        return g

    @staticmethod
    def _snap(trav, cell, snap_m, res_cells=None):
        """If the cell is not traversable, return the nearest free cell via a spiral search."""
        r, c = cell
        H, W = trav.shape
        if not (0 <= r < H and 0 <= c < W):
            return None
        if trav[r, c]:
            return cell
        # snap_m is in meters, but here it is approximated as a cell-grid radius
        max_rad = int(max(1, round(snap_m / 0.05)))
        for rad in range(1, max_rad + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and trav[nr, nc]:
                        return (nr, nc)
        return None

    @staticmethod
    def _astar(grid, start, goal):
        """8-connected A*. A cell list (start..goal) or None."""
        H, W = grid.shape

        def h(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        openh = [(h(start, goal), 0.0, start)]
        gscore = {start: 0.0}
        came = {}
        while openh:
            _, gc, cur = heapq.heappop(openh)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            if gc > gscore.get(cur, 1e18):
                continue
            r, c = cur
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W) or not grid[nr, nc]:
                        continue
                    # Diagonal moves only when both orthogonal cells are free (prevents corner cutting)
                    if dr != 0 and dc != 0 and not (grid[r + dr, c] and grid[r, c + dc]):
                        continue
                    step = SQRT2 if (dr and dc) else 1.0
                    ng = gc + step
                    nb = (nr, nc)
                    if ng < gscore.get(nb, 1e18):
                        gscore[nb] = ng
                        came[nb] = cur
                        heapq.heappush(openh, (ng + h(nb, goal), ng, nb))
        return None

    @classmethod
    def _simplify(cls, grid, path):
        """Merge straight segments via line-of-sight to reduce the number of waypoints."""
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        n = len(path)
        while i < n - 1:
            j = n - 1
            while j > i + 1 and not cls._los(grid, path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out

    @staticmethod
    def _los(grid, a, b):
        """Check whether the straight line a->b is entirely free, using Bresenham."""
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            if not grid[r, c]:
                return False
            if (r, c) == (r1, c1):
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc


# ---------------------------------------------------------------------------
# Demo obstacles + Stage 2 answer, read from mock_demo_data.yaml
# (data pre-extracted by tools/mock_data_builder instead of the scenario).
# ---------------------------------------------------------------------------
import os
import logging

import yaml

log = logging.getLogger("participant_app")


def _default_data_path() -> str:
    """demo/mock_demo_data.yaml next to this module."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "mock_demo_data.yaml")


def load_mock_data(path: str | None = None) -> dict:
    """Load the pre-extracted demo data (obstacles + Stage 2 answer)."""
    path = path or os.environ.get("MOCKUP_DEMO_DATA") or _default_data_path()
    if not os.path.exists(path):
        log.warning("[mock-data] not found: %s -- run tools/mock_data_builder/gen_mock_demo_data.py", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_obstacles(path: str | None = None) -> list:
    """Return a list of obstacles [(x, y, radius_m), ...] from mock_demo_data.yaml.

    The runtime occupancy map omits 3D obstructions (vehicles/bicycles/people/objects),
    so the mock navigation injects these to route around them (demo only).
    """
    data = load_mock_data(path)
    obstacles = [(float(o["x"]), float(o["y"]), float(o["radius"]))
                 for o in (data.get("obstacles") or [])]
    log.info("[mock-data] loaded %d obstacles", len(obstacles))
    return obstacles


def obstacles_excluding_goal(obstacles, goal_xy, exclude_m=0.6):
    """Return the list with obstacles within exclude_m of the goal removed.

    Keeps the goal itself (target object / owner zone) reachable, while still routing
    around nearby obstructions.
    """
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    ex2 = float(exclude_m) ** 2
    out = []
    for (x, y, r) in obstacles:
        if (x - gx) ** 2 + (y - gy) ** 2 <= ex2:
            continue
        out.append((x, y, r))
    return out


# [팀 추가] 이 함수 전체가 starter kit에 없던 것. waypoint_avoid.relocate()에 넘길
# "장애물 원" 목록을 만들 때, obstacles_excluding_goal()(반경 안 전부 제거)이 너무
# 거칠어서 만든 정밀 버전이다.
def nearest_only_exclude(obstacles, goal_xy, search_m):
    """목표 근방에서 가장 가까운 장애물 **하나만** 제외한다 (전부가 아니라).

    `obstacles_excluding_goal()` 은 `search_m` 안의 것을 전부 지운다 -- 목표 지점
    자체의 크기만큼 좁은 반경이면 맞는 동작이지만, `search_m` 을 위치 불확실성
    (reveal 의 hint_radius) 만큼 넓혀야 하면 그 반경 안의 다른 모든 것 -- 대상 물체가
    기대어 있는 기둥·벤치 같은 랜드마크까지 -- 같이 사라진다. 장애물이 이미
    개체 단위 원(클러스터)으로 들어와 있다면, 목표와 안 붙어 있는 것들은 애초에
    별개의 원이므로 목표에 가장 가까운 원 하나만 "이게 목표 자체"로 보고 빼면 되고
    나머지는 그대로 회피 대상으로 남겨도 목표 도달 가능성이 깨지지 않는다.

    목표 물체와 랜드마크가 맞닿아 하나의 원으로 합쳐진 경우(우산이 기둥에 기댄 등)는
    이 함수로 못 푼다 -- 그 원의 중심이 랜드마크 쪽으로 끌려가 있으면 이 원마저
    회피 대상으로 남아 진입로가 막힐 수 있다. 알려진 한계이며, 라이브에서 관측되면
    그때 클러스터 내부의 목표 인근 셀만 떼어내는 부분 제외로 확장한다.
    """
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    search2 = float(search_m) ** 2
    best_i, best_d2 = None, search2
    for i, (x, y, _r) in enumerate(obstacles):
        d2 = (x - gx) ** 2 + (y - gy) ** 2
        if d2 <= best_d2:
            best_i, best_d2 = i, d2
    return [o for i, o in enumerate(obstacles) if i != best_i]
