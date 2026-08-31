"""전방 depth 카메라로 라이다 회전 유령벽을 지운다.

## 왜 필요한가

`LidarObstacleLayer` 의 지면분리는 스윕 **전체**에 대해 지면 높이를 percentile
하나로 잡는다(모듈 자체 문서의 "경사면" 한계 참고). 로봇이 회전하면 방위각마다
다른 지형을 새로 훑는데 지면 추정은 하나뿐이라, 방위각별 오차가 그대로 유령
장애물로 쌓인다 -- 회전 자체가 유령 생성기가 된다(라이브 실측: 정지+회전 90초
동안 blocked_area 17.6m2 -> 69.3m2, 4배).

전방 depth 카메라는 이 문제에서 원리적으로 자유롭다: 프레임 **하나**로 완결된
평면적합이라(지면 높이도 그 프레임 안에서만 median 으로 계산) 여러 방위각을
누적하는 스윕과 달리 방위각별 오차가 쌓일 자리가 없다. 이미
`participant_app._locate_object_by_depth()` 가 같은 카메라로 "바닥 위로 솟은
것"(파지 대상)을 찾고 있는데, 여기서는 정반대로 **바닥 자체**를 찾아서 -- 카메라가
바닥이라고 확인한 자리는 라이다가 뭐라고 판단했든 차단을 지운다.

## 한계

카메라는 전방 좁은 화각·짧은 유효거리(MAX_VETO_RANGE_M)만 본다 -- 라이다의
360도 커버리지를 대체하지 않는다. 순수 보정(veto)이라, 카메라가 못 보는 방향의
유령은 여전히 1층 대책(원거리 컷, 롤링 윈도우, `LidarObstacleLayer` 자체)에
의존한다. 시간적 기억도 없다 -- 매 프레임 다시 확인해서 지울 뿐이라, 로봇이
고개를 돌리면(=회전하면) 그 순간부터는 다시 라이다 단독 판정으로 돌아간다.
"""

from __future__ import annotations

import math

import numpy as np

# 카메라 프레임 자체 지면추정(이 함수 안의 floor_z) 기준, 이 위까지만 "바닥"으로
# 본다. lidar_obstacles.GROUND_BAND_MIN_M 과 성격은 같지만(추정 지면 위 몇 cm까지
# 잡음으로 볼지) 센서가 다르므로 별도 상수 -- 카메라 depth 잡음이 라이다보다 훨씬
# 작아 더 좁게 잡아도 된다.
FLOOR_BAND_M = 0.05
# 카메라 앞 이 거리(m) 안쪽만 veto 후보로 본다 -- 라이다의 원거리 컷(MAX_RANGE_M)과
# 같은 이유로, 원거리일수록 프레임 안 평면적합 오차도 커진다.
MAX_VETO_RANGE_M = 4.0
# 로봇 자신의 범퍼/후드가 프레임 하단에 걸리는 아주 가까운 픽셀은 뺀다.
MIN_FWD_M = 0.15
# 뎁스 이미지를 이 배수만큼 서브샘플링한다 -- 라이다 격자(기본 0.25m)보다 훨씬
# 촘촘한 픽셀 단위 정밀도가 veto 에는 필요 없고, 매 틱 전체 픽셀을 도는 비용을 던다.
PIXEL_STRIDE = 4


def quat_to_rotmat(x, y, z, w):
    """단위 쿼터니언 -> 3x3 회전행렬 (participant_app._locate_object_by_depth 와 동일 공식)."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def confirmed_floor_world_xy(depth, K, mount_R, mount_t, pose):
    """이번 프레임에서 카메라가 바닥으로 확인한 지점들의 월드 (x, y).

    Args:
        depth: (H, W) float32, 미터 단위. NaN/0/음수는 무효 픽셀.
        K: 3x3 카메라 내부파라미터 (camera_info.k.reshape(3,3)).
        mount_R, mount_t: `base_link` 프레임에서 본 카메라 장착 회전/평행이동 --
            `base_link -> Base_Camera_Center` tf 로 한 번만 얻으면 된다(정적
            변환. `lidar_obstacles` 의 라이다 마운트 오프셋과 같은 성격이지만,
            카메라는 기울어져 장착돼 있어 dx,dy 만으로는 부족해 회전까지 필요).
        pose: `(x, y, yaw)` -- `client.get_world_pose()`.

    Returns:
        `(N, 2)` ndarray 월드 (x, y). 바닥 확인 점이 하나도 없으면 `(0, 2)` 빈 배열.
    """
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    vs, us = np.mgrid[0:h:PIXEL_STRIDE, 0:w:PIXEL_STRIDE]
    d = depth[vs, us]
    valid = np.isfinite(d) & (d > 0.05) & (d < MAX_VETO_RANGE_M)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    # 표준 핀홀 역투영 (카메라 광학 프레임: x=우, y=하, z=깊이) -- 실제 축이
    # base_link 의 어디를 가리키는지는 mount_R 이 정한다(장착 회전 전부를 여기서
    # 흡수하므로, 카메라가 얼마나 기울어져 있든 이 식 자체는 안 바뀐다).
    x_cam = (us - cx) / fx * d
    y_cam = (vs - cy) / fy * d
    pts_cam = np.stack([x_cam, y_cam, d], axis=-1)          # (h', w', 3)
    pts_base = pts_cam @ mount_R.T + mount_t                # base_link 프레임

    fwd, lat, up = pts_base[..., 0], pts_base[..., 1], pts_base[..., 2]
    valid = valid & (fwd > MIN_FWD_M)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    # 이 프레임 하나 안에서 지면 높이 추정 -- 스윕을 누적하는 라이다와 달리, 여기서
    # 계산한 floor_z 는 이 프레임의 시야 안에서만 유효하면 되므로 방위각별 오차가
    # 쌓일 여지가 없다. (대부분의 화소가 바닥이라는 가정은 grasp 용
    # _locate_object_by_depth 와 동일 -- 전방을 보는 카메라라 합리적.)
    floor_z = float(np.median(up[valid]))
    floor = valid & (up <= floor_z + FLOOR_BAND_M)
    if not np.any(floor):
        return np.empty((0, 2), dtype=np.float64)

    x, y, yaw = pose
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    f, l = fwd[floor], lat[floor]
    wx = x + f * cos_y - l * sin_y
    wy = y + f * sin_y + l * cos_y
    return np.stack([wx, wy], axis=1)
