"""LiDAR 로컬 장애물 레이어 -- 점유지도에 없는 장애물을 센서로 찾아 플래너에 넣는다.

## 왜 필요한가

`get_occupancy_map()` 은 **정적 도로만** 담는다. 주차 차량·자전거·사람은 없다(데스크탑
실측: 로봇이 경로상의 차에 그대로 머리를 박는다). 개발자 가이드가 참가자 구현으로 지목한
부분이 정확히 여기다 -- "센서로 장애물을 감지하는 대신, 미리 뽑아 둔 장애물 데이터로 이동
경로를 계획하는 모의 구현을 라이다·카메라 센서로 감지하도록 교체".

`participant_app._remember_blockage()` 의 정체 기억은 **박고 나서** 배우므로 한 번에
`STAGE2_STUCK_TIME_S` 씩 대가를 치른다(오프라인 실측: rt04b 가 600초 예산 중 344초 소진).
이 레이어는 **박기 전에** 보는 쪽이고, 정체 기억은 이 레이어가 놓친 것(연석, 낮은 턱,
바퀴 빠짐)에 대한 최후 안전망으로 그대로 남는다.

## 이건 SLAM 이 아니다

위치추정은 `get_world_pose()` 가, 전역 지도는 `get_occupancy_map()` 이 이미 준다. 그래서
여기서 만드는 건 nav2 로 치면 `obstacle_layer` 하나뿐이다 -- scan matching·loop closure·
pose graph 같은 건 필요 없다.

## 좌표계

`get_lidar()` 의 점은 **Base_LiDAR 프레임**(차체 정렬, 원점은 장착 위치)이다. 월드로
옮길 때 mount 오프셋은 `update(mount_offset=...)` 로 받는다 -- `Base_LiDAR -> base` 는
정적 변환이라 참가자 앱이 `/tf_static` 에서 **한 번만** 조회해 넘기면 되고, 못 얻으면
(0, 0) 로 둔다(센서가 차체 중심 위에 있다는 근사).

## 지면 분리

높이 임계를 **절대 z 로 잡으면 안 된다.** chungmu 만 해도 지형 z 가 15.53~16.71 로
1.2m 차이난다 -- 임계 하나로 자르면 낮은 지역은 지면이 통째로 장애물이 되고 높은 지역은
차가 지면이 된다. 씬이 바뀌면 더 나빠지는 종류의 하드코딩이다.

여기서는 **스캔 자체에서 지면 높이를 추정한다**(하위 백분위수). 장착 높이 가정도, tf 의
z 성분도 필요 없고, 지형이 통째로 오르내려도 따라간다.

**한계 -- 경사면**: 지면 추정이 스캔 전체에 대해 하나뿐이라, 앞쪽이 오르막이면 그 지면이
밴드에 들어와 유령 벽이 된다. 링 간 수직각으로 판정하는 방식(LeGO-LOAM 계열)이나 sector
별 평면 적합(Patchwork 계열)이 정답이지만, **오프라인 하네스의 지면이 평면 z=0 이라
경사 강건성을 검증할 방법이 지금 없다.** 검증할 수 없는 정교함을 미리 넣는 대신, 라이브에서
경사 오탐이 관측되면 그때 올린다. 오탐 감시용으로 `stats()` 가 차단 셀 수를 돌려준다.
"""

from __future__ import annotations

import math

import numpy as np

# -- 지면 분리 (센서 스케일) --
# 지면 높이 추정에 쓰는 하위 백분위수. 0 이 아니라 5 인 이유: 최저점 하나가 이상치면
# 지면 전체가 밀린다.
GROUND_PERCENTILE = 5.0
# 추정 지면에서 이만큼 위부터 장애물로 본다. 이하는 지면·연석·노면 잡음.
GROUND_BAND_MIN_M = 0.15
# 이 위는 무시한다 -- 나뭇가지·차양처럼 밑으로 지나갈 수 있는 것.
GROUND_BAND_MAX_M = 1.8

# -- 격자 --
# 점유지도 해상도를 모를 때의 기본값. 알면 set_resolution() 으로 맞춘다.
DEFAULT_RESOLUTION_M = 0.25
# 셀 하나를 막힌 것으로 인정하는 최소 누적 점 수. 스퓨리어스 리턴 하나로 경로를 막지
# 않기 위한 것이다.
#
# **주의**: 이 값을 오프라인 하네스에서 튜닝하면 라이브를 과소평가한다. 하네스의 방위
# 분해능은 1deg 인데 실물 VLP-16 은 ~0.2deg 라 같은 물체에 점이 5배 붙는다. 반대로 점밀도는
# 거리에 따라 급감하므로(하네스 실측: 같은 물체가 3m 526점 -> 20m 10점) 이 값을 올릴수록
# 먼 장애물을 통째로 놓친다.
MIN_POINTS_PER_CELL = 2
# 차단 셀 상한. 지면 오분류가 폭주해도 플래너를 마비시키지 않게 하는 안전장치다.
#
# **주의 -- 이건 개수지 면적이 아니다.** 의도는 면적 예산인데 셀 개수로 박혀 있어서
# 해상도에 따라 뜻이 달라진다: 설계 기준 0.25m 에선 500m2 지만, 라이브 점유지도 해상도
# 0.05m 에 동기화되면 같은 8000 셀이 **20m2** (4.5m x 4.5m) 밖에 안 된다. 아래 원거리
# 컷과 롤링 윈도우가 들어오기 전에는 라이브에서 몇 초 만에 여기 닿았고, 닿는 순간
# update() 가 새 셀을 통째로 버려서 (a) 눈앞의 진짜 장애물이 레이어에 못 들어오고
# (b) new_blocked 가 항상 0 이라 participant_app 의 `if new_cells` 게이트 때문에
# blocks_path() 재계획 트리거까지 영구히 죽었다 (라이브 실측: blocked_cells 가 정확히
# 8000 에 고정된 채 로봇이 차 옆구리에 끼여 계속 비빔).
MAX_BLOCKED_CELLS = 8000

# 이 거리(m, 센서 프레임 수평거리) 밖의 리턴은 아예 안 본다.
#
# 지역 회피에 20m 밖 점은 필요 없는데, 단일 percentile 지면추정은 원거리에서 특히 크게
# 틀린다 -- 지면 추정이 스윕 전체에 하나뿐이라 앞쪽이 오르막이면 그 지면이 밴드로 들어와
# 유령 벽이 된다(모듈 docstring 의 알려진 한계). 라이브 실측(chungmu, 정지 상태 1스윕):
# 장애물 밴드 4487점 중 15-25m 링이 3230점(72%)이고 그 z_rel 이 0.23 -> 1.80 으로 거리에
# 따라 매끈하게 상승 -- 전형적인 경사 유령이다. 반면 0-6m 에는 사실상 아무것도 없었다.
# 여기서 자르는 게 경사 강건 지면추정(Patchwork 류)을 제대로 구현하는 것보다 훨씬 싸고,
# 무엇보다 **오프라인 하네스가 검증할 수 없는 정교함을 미리 넣지 않는다**는 원칙과 맞는다.
MAX_RANGE_M = 8.0

# 로봇에서 이 거리(m) 밖의 차단 셀은 버린다 -- 롤링 로컬 윈도우.
#
# 이 모듈 이름은 "**로컬** 장애물 레이어"인데 원래 구현은 삭제가 없는 전역 누적기였다.
# 정적인 씬이라 지울 필요가 없다는 게 근거였지만, 지면 오분류가 섞이는 순간 그 가정은
# "오분류도 영원히 남는다"가 된다. nav2 의 local costmap 과 같은 롤링 윈도우로 바꾸면
# 상한(MAX_BLOCKED_CELLS)에 닿는 일 자체가 없어지고, 레이어 내용이 실제로 "지금 근처에
# 있는 것"이 된다. 원거리 컷(MAX_RANGE_M)보다 넉넉하게 잡아, 잠깐 시야에서 벗어난
# 장애물이 곧바로 지워지지는 않게 한다.
LOCAL_WINDOW_M = 15.0

# -- 경로 간섭 판정 --
# 남은 경로에서 이 거리 안에 차단 셀이 있으면 재계획한다. 차체 반폭 + 여유.
PATH_CORRIDOR_M = 0.45


def lidar_xyz(msg):
    """`get_lidar()` 결과 -> (N, 3) float32, **센서 프레임**. None 이면 쓸 점이 없다.

    ROS 의존을 이 함수 하나에 가둔다 -- 그래야 지면분리·격자투영 같은 실제 로직 전부가
    ROS 없이 오프라인 하네스에서 검증된다. 세 경로를 순서대로 시도한다:

    1. 하네스의 `.xyz` 지름길 (`validation/fake_client.FakePointCloud2`)
    2. `sensor_msgs_py.point_cloud2` (ROS 2 Humble 표준)
    3. `fields`/`point_step` 를 보고 직접 파싱 (xyz 가 float32 인 경우)
    """
    if msg is None:
        return None

    arr = getattr(msg, "xyz", None)
    if arr is not None:
        pts = np.asarray(arr, dtype=np.float32)
        return pts.reshape(-1, 3) if pts.size else None

    try:
        from sensor_msgs_py import point_cloud2
    except Exception:
        return _parse_pointcloud2(msg)

    try:
        raw = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # Humble 은 structured ndarray 를, 예전 버전은 제너레이터를 준다.
        if isinstance(raw, np.ndarray):
            if raw.size == 0:
                return None
            pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=-1)
        else:
            pts = np.asarray(list(raw), dtype=np.float32)
        return pts.reshape(-1, 3).astype(np.float32) if pts.size else None
    except Exception:
        return _parse_pointcloud2(msg)


def _parse_pointcloud2(msg):
    """`fields`/`point_step` 기반 직접 파싱 폴백. xyz 가 float32 라고 가정한다."""
    try:
        offsets = {f.name: int(f.offset) for f in msg.fields if f.name in ("x", "y", "z")}
        if len(offsets) != 3:
            return None
        step = int(msg.point_step)
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        n = raw.size // step
        if n == 0:
            return None
        raw = raw[:n * step].reshape(n, step)
        out = np.empty((n, 3), dtype=np.float32)
        for i, key in enumerate(("x", "y", "z")):
            o = offsets[key]
            out[:, i] = raw[:, o:o + 4].copy().view(np.float32).ravel()
        good = np.isfinite(out).all(axis=1)
        return out[good] if good.any() else None
    except Exception:
        return None


class LidarObstacleLayer:
    """누적 2D 장애물 레이어. 월드 좌표 원판 목록으로 플래너에 넘긴다.

    씬이 정적이므로(사람도 안 움직인다) 관측을 라운드 내내 **누적**하고 free-space
    raycasting 으로 지우지 않는다. 지우기가 필요해지는 건 장애물이 움직일 때인데 그렇지
    않기 때문이다. 대신 오분류가 쌓이는 걸 막으려고 셀당 최소 점 수와 총 셀 상한을 둔다.
    """

    def __init__(self, resolution=DEFAULT_RESOLUTION_M):
        self._res = float(resolution)
        self._counts = {}                  # (row, col) -> 누적 점 수
        self._blocked = {}                 # (row, col) -> (world_x, world_y)
        self.scans = 0
        self.last_ground_z = None
        self.capped = False

    # -- 설정 --
    def set_resolution(self, resolution):
        """점유지도 해상도에 맞춘다. 값이 바뀌면 격자가 무의미해지므로 누적을 버린다."""
        resolution = float(resolution)
        if resolution <= 0.0 or abs(resolution - self._res) < 1e-9:
            return
        self._res = resolution
        self._counts.clear()
        self._blocked.clear()

    @property
    def resolution(self):
        return self._res

    # -- 갱신 --
    def update(self, msg, pose, mount_offset=(0.0, 0.0)):
        """스윕 하나를 누적한다. 새로 차단된 셀 수를 돌려준다.

        Args:
            msg: `client.get_lidar()` 결과.
            pose: `(x, y, yaw)` -- `client.get_world_pose()`.
            mount_offset: 차체 프레임에서 센서까지의 (dx, dy). `/tf_static` 의
                `Base_LiDAR` 정적 변환에서 한 번만 얻어 넘기면 된다.
        """
        pts = lidar_xyz(msg)
        if pts is None or pts.shape[0] == 0 or pose is None:
            return 0
        self.scans += 1

        # 1) 지면 추정 -- 절대 높이가 아니라 이 스캔의 하위 백분위수
        gz = float(np.percentile(pts[:, 2], GROUND_PERCENTILE))
        self.last_ground_z = gz
        z_rel = pts[:, 2] - gz
        keep = (z_rel > GROUND_BAND_MIN_M) & (z_rel < GROUND_BAND_MAX_M)
        if not np.any(keep):
            return 0
        p = pts[keep]

        # 1b) 원거리 컷 -- MAX_RANGE_M 참고. 지면추정 오차가 가장 크게 나타나는 구간을
        # 통째로 빼는 것이라, 지면분리 자체를 정교하게 만들지 않고도 유령의 대부분을 없앤다.
        rng = np.hypot(p[:, 0], p[:, 1])
        near = rng <= MAX_RANGE_M
        if not np.any(near):
            return 0
        p = p[near]

        # 2) 센서 -> 월드 (yaw 회전 + 로봇 위치). roll/pitch 는 다루지 않는다.
        x, y, yaw = pose
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        px = p[:, 0] + float(mount_offset[0])
        py = p[:, 1] + float(mount_offset[1])
        wx = x + px * cos_y - py * sin_y
        wy = y + px * sin_y + py * cos_y

        # 3) 격자 누적. 점 단위로 파이썬 루프를 돌면 안 된다 -- VLP-16 은 초당 30만 점이다.
        # 셀 단위로 먼저 접어서(np.unique) 루프 길이를 점 수가 아니라 **셀 수**로 줄인다.
        cols = np.floor(wx / self._res).astype(np.int64)
        rows = np.floor(wy / self._res).astype(np.int64)
        uniq, counts = np.unique(np.stack([rows, cols], axis=1), axis=0, return_counts=True)

        new_blocked = 0
        for (r, c), hits in zip(map(tuple, uniq.tolist()), counts.tolist()):
            key = (r, c)
            n = self._counts.get(key, 0) + hits
            self._counts[key] = n
            if n >= MIN_POINTS_PER_CELL and key not in self._blocked:
                if len(self._blocked) >= MAX_BLOCKED_CELLS:
                    self.capped = True
                    continue
                self._blocked[key] = ((c + 0.5) * self._res, (r + 0.5) * self._res)
                new_blocked += 1

        # 4) 롤링 로컬 윈도우 -- LOCAL_WINDOW_M 참고. 누적 전용이던 것을 "지금 근처"만
        # 남기도록 바꾼다. _counts 도 같이 버린다: 남겨두면 셀이 윈도우에 재진입할 때
        # 낡은 카운트만으로 즉시 다시 막혀서, 결국 지우지 않은 것과 같아진다.
        self._evict_far((x, y))
        return new_blocked

    def _evict_far(self, xy):
        """로봇에서 LOCAL_WINDOW_M 밖 셀을 버린다."""
        if not self._blocked:
            return
        keys = np.fromiter((k for kv in self._blocked.keys() for k in kv),
                           dtype=np.int64, count=2 * len(self._blocked)).reshape(-1, 2)
        cx = (keys[:, 1] + 0.5) * self._res - xy[0]
        cy = (keys[:, 0] + 0.5) * self._res - xy[1]
        far = (cx * cx + cy * cy) > (LOCAL_WINDOW_M * LOCAL_WINDOW_M)
        if not np.any(far):
            return
        for r, c in map(tuple, keys[far].tolist()):
            self._blocked.pop((r, c), None)
            self._counts.pop((r, c), None)
        # 윈도우가 다시 여유를 만들었으므로 상한 플래그도 현재 상태에 맞춘다.
        self.capped = len(self._blocked) >= MAX_BLOCKED_CELLS

    # -- 출력 --
    def discs(self):
        """플래너용 `[(x, y, r), ...]`.

        셀 하나당 반경 = 해상도인 작은 원판을 낸다. `OccupancyGridPlanner._block_obstacles`
        가 원판을 격자에 칠하므로, 이렇게 하면 결과가 격자 오버레이와 같아지면서 **플래너를
        한 줄도 고치지 않아도 된다** -- 자동차처럼 길쭉한 장애물을 원 하나로 감싸 과차단하는
        문제도 함께 피한다.
        """
        r = self._res
        return [(wx, wy, r) for (wx, wy) in self._blocked.values()]

    def blocks_path(self, path, corridor_m=PATH_CORRIDOR_M, start_xy=None):
        """남은 경로의 회랑 안에 차단 셀이 있는가 -- 재계획 트리거.

        주기적으로 다시 계획하면 경로가 매번 미세하게 바뀌어 추종이 떨린다. 실제로 막혔을
        때만 재계획하기 위한 판정이다.
        """
        if not path or not self._blocked:
            return False
        pts = [start_xy] + list(path) if start_xy is not None else list(path)
        if len(pts) < 2:
            return False
        seg = np.asarray(pts, dtype=np.float64)
        a, b = seg[:-1], seg[1:]
        cells = np.asarray(list(self._blocked.values()), dtype=np.float64)

        d = b - a                                   # (S, 2)
        L2 = (d * d).sum(axis=1)
        L2 = np.where(L2 < 1e-12, 1e-12, L2)
        # (C, S) 로 브로드캐스트해 점-선분 거리를 한 번에
        ap = cells[:, None, :] - a[None, :, :]
        t = np.clip((ap * d[None, :, :]).sum(axis=2) / L2[None, :], 0.0, 1.0)
        proj = a[None, :, :] + t[:, :, None] * d[None, :, :]
        dist = np.linalg.norm(cells[:, None, :] - proj, axis=2)
        return bool((dist.min() if dist.size else np.inf) < corridor_m)

    def danger_ahead(self, pose, max_dist, cone_deg=70.0, goal=None, goal_exclude_m=0.0):
        """차단 셀 중 로봇 전방 `cone_deg` 안, `max_dist` 이내에 있는 게 하나라도 있는가.

        `blocks_path()`는 "남은 경로가 막혔나"를 보고 A* 재계획을 트리거한다 -- 재계획
        사이클(격자 갱신 + A* + follower 교체) 한 바퀴만큼 항상 늦다. 이건 그 대신 매 틱
        직접 속도를 깎는 반응형 제동용이라 계획을 전혀 참조하지 않는다 -- "지금 정면
        가까이에 뭐가 있나"만 본다. 일단 멈추면 정체판정(STAGE2_STUCK_TIME_S)이 이어받아
        우회를 만든다.

        `goal`/`goal_exclude_m`: `_plan_obstacles`가 A* 에 넘기기 전 목표 근처 장애물을
        빼는 것과 같은 이유로 여기서도 빼야 한다 -- 픽업 대상 자체가 라이다에는 그냥
        장애물이라, 이 제외가 없으면 standoff 로 다가가는 정상 접근이 "장애물 코앞"으로
        보여 영원히 제동만 걸리고 못 붙는다 (실측: rt02 가 이 누락으로 1800s 타임아웃).
        """
        if not self._blocked:
            return False
        x, y, yaw = pose
        pts = np.asarray(list(self._blocked.values()), dtype=np.float64)   # (N, 2)
        if goal is not None and goal_exclude_m > 0.0:
            near_goal = np.linalg.norm(pts - np.array(goal, dtype=np.float64), axis=1) <= goal_exclude_m
            pts = pts[~near_goal]
            if pts.size == 0:
                return False
        d = pts - np.array([x, y])
        dist = np.linalg.norm(d, axis=1) - self._res
        close = dist <= max_dist
        if not np.any(close):
            return False
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        dn = d[close]
        distn = np.maximum(np.linalg.norm(dn, axis=1), 1e-6)
        cosang = (dn @ fwd) / distn
        return bool(np.any(cosang >= math.cos(math.radians(cone_deg))))

    def stats(self):
        """로깅용 -- 오분류 폭주를 눈으로 잡기 위한 것."""
        return {
            "scans": self.scans,
            "blocked_cells": len(self._blocked),
            "blocked_area_m2": len(self._blocked) * self._res * self._res,
            "ground_z": self.last_ground_z,
            "capped": self.capped,
        }
