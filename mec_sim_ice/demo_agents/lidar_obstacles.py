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
MAX_BLOCKED_CELLS = 8000

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
        return new_blocked

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

    def stats(self):
        """로깅용 -- 오분류 폭주를 눈으로 잡기 위한 것."""
        return {
            "scans": self.scans,
            "blocked_cells": len(self._blocked),
            "blocked_area_m2": len(self._blocked) * self._res * self._res,
            "ground_z": self.last_ground_z,
            "capped": self.capped,
        }
