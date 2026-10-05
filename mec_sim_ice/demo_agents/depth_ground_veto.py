"""전방 depth 카메라로 라이다 회전 유령벽을 지운다.

## 왜 필요한가

`LidarObstacleLayer` 의 지면분리는 스윕 **전체**에 대해 지면 높이를 percentile
하나로 잡는다(모듈 자체 문서의 "경사면" 한계 참고). 로봇이 회전하면 방위각마다
다른 지형을 새로 훑는데 지면 추정은 하나뿐이라, 방위각별 오차가 그대로 유령
장애물로 쌓인다 -- 회전 자체가 유령 생성기가 된다(라이브 실측: 정지+회전 90초
동안 blocked_area 17.6m2 -> 69.3m2, 4배).

전방 depth 카메라는 이 문제에서 원리적으로 자유롭다: 프레임 **하나**로 완결된
평면적합이라(1차 평면 z=a*fwd+b*lat+c 를 그 프레임 안에서 적합 -- 경사도 명시적으로
흡수) 여러 방위각을 누적하는 스윕과 달리 방위각별 오차가 쌓일 자리가 없다. 이미
`participant_app._locate_object_by_depth()` 가 같은 카메라로 "바닥 위로 솟은
것"(파지 대상)을 찾고 있는데, 여기서는 정반대로 **바닥 자체**를 찾아서 -- 카메라가
바닥이라고 확인한 자리는 라이다가 뭐라고 판단했든 차단을 지운다.

## 한계

카메라는 전방 좁은 화각·짧은 유효거리(MAX_VETO_RANGE_M)만 본다 -- 라이다의
360도 커버리지를 대체하지 않는다. 순수 보정(veto)이라, 카메라가 못 보는 방향의
유령은 여전히 1층 대책(원거리 컷, 롤링 윈도우, `LidarObstacleLayer` 자체)에
의존한다. 시간적 기억도 없다 -- 매 프레임 다시 확인해서 지울 뿐이라, 로봇이
고개를 돌리면(=회전하면) 그 순간부터는 다시 라이다 단독 판정으로 돌아간다.

## [파일 출처 안내]
이 파일도 lidar_obstacles.py와 마찬가지로 MARC 2026 starter kit에는 없던 파일이다
-- 전체가 우리 팀이 새로 작성했다. `participant_app._locate_object_by_depth()`가
같은 base 카메라의 depth로 "바닥 위로 솟은 것"(파지 대상)을 찾는 것과 정반대로,
여기서는 "바닥 자체"를 찾아서 라이다의 오판을 지우는 데 쓴다 -- 좌표 변환 공식
(quat_to_rotmat, 핀홀 역투영)은 그쪽 코드와 동일한 방식을 재사용한 것이다.
"""

from __future__ import annotations

import math

import numpy as np

# 적합된 바닥 평면 기준, 이 위까지만 "바닥"으로 본다. lidar_obstacles.GROUND_BAND_MIN_M
# 과 성격은 같지만(추정 지면 위 몇 cm까지 잡음으로 볼지) 센서가 다르므로 별도 상수 --
# 카메라 depth 잡음이 라이다보다 훨씬 작아 더 좁게 잡아도 된다.
FLOOR_BAND_M = 0.05
# 평면 적합 관련. 평평한 median 밴드는 경사 지형에서 양방향으로 틀린다 -- 4m 창에서
# 2deg 경사만 있어도 14cm 가 변해, 오르막 쪽 바닥은 밴드 밖(veto 무력화)이고 내리막
# 쪽 실제 장애물이 밴드 안(진짜 장애물을 지움)으로 들어온다. 스폰 지점이 주변보다
# 높은 씬에서 실제로 겪는 조건이라, median 대신 1차 평면 z = a*fwd + b*lat + c 를
# 적합해 잔차로 판정한다.
PLANE_ROUGH_BAND_M = 0.20      # 적합 전 물체 픽셀을 걷어내는 러프 마스크 (median +/- 이 값)
PLANE_MAX_SLOPE = 0.27         # |a|,|b| 상한 (~tan 15deg) -- 넘으면 벽에 적합된 것, median 폴백
PLANE_MIN_PTS = 200            # 적합에 필요한 최소 픽셀 수 (미달 시 median 폴백)
# 카메라 앞 이 거리(m) 안쪽만 veto 후보로 본다 -- 라이다의 원거리 컷(MAX_RANGE_M)과
# 같은 이유로, 원거리일수록 프레임 안 평면적합 오차도 커진다.
MAX_VETO_RANGE_M = 4.0
# 로봇 자신의 범퍼/후드가 프레임 하단에 걸리는 아주 가까운 픽셀은 뺀다.
MIN_FWD_M = 0.15
# 뎁스 이미지를 이 배수만큼 서브샘플링한다 -- 라이다 격자(기본 0.25m)보다 훨씬
# 촘촘한 픽셀 단위 정밀도가 veto 에는 필요 없고, 매 틱 전체 픽셀을 도는 비용을 던다.
PIXEL_STRIDE = 4
# [팀 수정 2026-10-05] 라이브 rt01_mppi_r4: SUV 옆면에 붙어 3분 동안 veto 가 991칸을
# 지웠고(로봇은 v=1.1 로 80초 밀어붙임), 라이다는 그 거리에서 전방 점이 0 이라 다시
# 채우지 못했다. 원인 두 가지와 대책:
#  1) 바닥 기준이 median -- 차가 화면을 채우면 median 이 차체 높이가 되고 차체 아래쪽이
#     "바닥"이 된다. -> 하위 FLOOR_REF_PERCENTILE 을 기준으로 (라이다 지면 추정과 같은
#     방식), 바닥 픽셀이 MIN_FLOOR_FRAC 미만인 프레임은 veto 하지 않는다.
#  2) 차 밑(지상고) 바닥이 보이면 그 칸을 지웠다 -- 바닥은 맞지만 그 위가 막혀 있다.
#     -> 같은 방위에서 더 가까운 장애물 픽셀이 있으면 그 너머 바닥은 버리고(SHADOW_*),
#     장애물 점을 함께 돌려줘 그 점이 든 칸은 지우지 않게 한다 (veto_free(keep=)).
FLOOR_REF_PERCENTILE = 5.0
MIN_FLOOR_FRAC = 0.3
OBSTACLE_MIN_M = 0.15          # 바닥 평면 위 이만큼부터 장애물 (lidar GROUND_BAND_MIN_M 와 같음)
OBSTACLE_MAX_M = 1.8           # 이 위는 무시 (lidar GROUND_BAND_MAX_M 와 같음)
SHADOW_BIN_RAD = math.radians(2.0)
SHADOW_MARGIN_M = 0.1          # 장애물 거리 - 이 값보다 먼 바닥은 그림자로 본다
#  3) 차체에 밀착하면(카메라가 차체에서 수 cm) 차체 픽셀은 depth 가 너무 짧거나 무효라
#     장애물로 안 잡히고 차 밑 바닥만 남는다 (합성 검증). -> 바닥을 향해야 할 아래쪽
#     픽셀(광선 방향 z < BELOW_DIR_Z, 카메라 자세로 계산) 중 무효이거나 "그 광선이 바닥에
#     닿을 거리의 NEAR_RATIO 배보다 짧은" 픽셀이 MAX_BLOCKED_FRAC 를 넘으면 veto 하지
#     않고, 그런 근접 픽셀은 장애물 점으로도 쓴다. 절대 거리(예: 0.3 m)로 자르면 안 된다:
#     실측(2026-10-05 depth_probe) 카메라는 바닥 위 ~0.27 m, 20deg 하향이라 화면 아래쪽
#     바닥 자체가 depth 0.28-0.39 m 로 찍힌다 -- 절대 0.3 m 컷이 바닥을 장애물로 보고
#     veto 를 거의 매 프레임 꺼 버렸다(라이브 rt01_ab_legacy: veto 4회 vs 이전 ~300-440회,
#     유령벽이 117 m2 까지 쌓여 A* 가 막힘).
#     카메라 높이는 프레임에서 추정한다 (아래쪽 픽셀의 d * -dir_z 의 상위 percentile =
#     바닥까지 높이), CAM_HEIGHT_M 의 [0.8, 1.25] 배로 클램프 -- 차가 화면을 채워도
#     추정이 장애물 높이로 끌려 내려가지 않게.
CAM_HEIGHT_M = 0.27
CAM_HEIGHT_PERCENTILE = 80.0
NEAR_RATIO = 0.8
NEAR_ABS_M = 0.3               # 바닥을 향하지 않는 픽셀(수평 근처)은 절대 거리로
BELOW_DIR_Z = -0.15
MAX_BLOCKED_FRAC = 0.3


def quat_to_rotmat(x, y, z, w):
    """단위 쿼터니언 -> 3x3 회전행렬 (participant_app._locate_object_by_depth 와 동일 공식)."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def confirmed_floor_world_xy(depth, K, mount_R, mount_t, pose):
    """`floor_and_obstacle_world_xy` 의 바닥 점만 (이전 호출부 호환)."""
    return floor_and_obstacle_world_xy(depth, K, mount_R, mount_t, pose)[0]


def floor_and_obstacle_world_xy(depth, K, mount_R, mount_t, pose):
    """이번 프레임에서 카메라가 바닥으로 확인한 지점들과 장애물 지점들의 월드 (x, y).

    Args:
        depth: (H, W) float32, 미터 단위. NaN/0/음수는 무효 픽셀.
        K: 3x3 카메라 내부파라미터 (camera_info.k.reshape(3,3)).
        mount_R, mount_t: `base_link` 프레임에서 본 카메라 장착 회전/평행이동 --
            `base_link -> Base_Camera_Center` tf 로 한 번만 얻으면 된다(정적
            변환. `lidar_obstacles` 의 라이다 마운트 오프셋과 같은 성격이지만,
            카메라는 기울어져 장착돼 있어 dx,dy 만으로는 부족해 회전까지 필요).
        pose: `(x, y, yaw)` -- `client.get_world_pose()`.

    Returns:
        `(floor, obstacle)` -- 각각 `(N, 2)` ndarray 월드 (x, y), 없으면 `(0, 2)`.
        floor 는 장애물 그림자(같은 방위에서 장애물보다 먼 바닥)를 뺀 것.
    """
    empty = np.empty((0, 2), dtype=np.float64)
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    vs, us = np.mgrid[0:h:PIXEL_STRIDE, 0:w:PIXEL_STRIDE]
    d = depth[vs, us]
    ok = np.isfinite(d) & (d > 0)
    valid = ok & (d > 0.05) & (d < MAX_VETO_RANGE_M)
    dir_z = ((us - cx) / fx * mount_R[2, 0] + (vs - cy) / fy * mount_R[2, 1] + mount_R[2, 2])
    below = dir_z < BELOW_DIR_Z
    hb = ok & below
    cam_h = (float(np.percentile(d[hb] * -dir_z[hb], CAM_HEIGHT_PERCENTILE)) if np.any(hb)
             else CAM_HEIGHT_M)
    cam_h = min(max(cam_h, 0.8 * CAM_HEIGHT_M), 1.25 * CAM_HEIGHT_M)
    floor_d = np.where(below, cam_h / np.maximum(-dir_z, 1e-6), np.inf)
    near = ok & np.where(below, d < NEAR_RATIO * floor_d, d < NEAR_ABS_M)
    blocked_frac = (float(np.sum(below & (~ok | near))) / max(int(np.sum(below)), 1))
    if not np.any(valid | near):
        return empty, empty
    # 무효 픽셀(NaN/inf/0 -- 뎁스 카메라가 흔히 내보낸다)을 아래 역투영에 넣기 전에
    # 0 으로 눌러둔다. 결과는 어차피 valid 마스크가 걸러내므로 값 자체는 무관하지만,
    # NaN 을 그대로 흘리면 matmul 이 매 프레임 RuntimeWarning 을 찍어 라이브 로그를
    # 덮는다 (실측: 2026-09-01 제출본 라이브 런에서 veto 프레임마다 발생).
    d = np.where(ok, d, 0.0)

    # 표준 핀홀 역투영 (카메라 광학 프레임: x=우, y=하, z=깊이) -- 실제 축이
    # base_link 의 어디를 가리키는지는 mount_R 이 정한다(장착 회전 전부를 여기서
    # 흡수하므로, 카메라가 얼마나 기울어져 있든 이 식 자체는 안 바뀐다).
    x_cam = (us - cx) / fx * d
    y_cam = (vs - cy) / fy * d
    pts_cam = np.stack([x_cam, y_cam, d], axis=-1)          # (h', w', 3)
    pts_base = pts_cam @ mount_R.T + mount_t                # base_link 프레임

    fwd, lat, up = pts_base[..., 0], pts_base[..., 1], pts_base[..., 2]
    valid = valid & (fwd > MIN_FWD_M)
    near = near & (fwd > MIN_FWD_M)
    x, y, yaw = pose
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    def world(m):
        if not np.any(m):
            return empty
        f, l = fwd[m], lat[m]
        return np.stack([x + f * cos_y - l * sin_y, y + f * sin_y + l * cos_y], axis=1)

    if not np.any(valid):
        return empty, world(near)

    # 이 프레임 하나 안에서 지면을 추정 -- 스윕을 누적하는 라이다와 달리 프레임 안에서
    # 완결되므로 방위각별 오차가 쌓일 여지가 없다. (대부분의 화소가 바닥이라는 가정은
    # grasp 용 _locate_object_by_depth 와 동일 -- 전방을 보는 카메라라 합리적.)
    # 평평한 기준 하나가 아니라 1차 평면을 적합한다 -- PLANE_* 상수 주석 참고
    # (경사/스폰 고도차에서 평평한 밴드는 veto 무력화와 실제 장애물 오소거 양쪽으로
    # 틀린다). 적합이 미덥지 않으면(픽셀 부족, 벽에 적합) 기준 밴드로 폴백한다.
    # 기준은 median 이 아니라 하위 percentile -- FLOOR_REF_PERCENTILE 주석 참고.
    med = float(np.percentile(up[valid], FLOOR_REF_PERCENTILE))
    rough = valid & (up >= med - FLOOR_BAND_M) & (up <= med + PLANE_ROUGH_BAND_M)
    plane = None
    if int(np.sum(rough)) >= PLANE_MIN_PTS:
        A = np.stack([fwd[rough], lat[rough], np.ones(int(np.sum(rough)))], axis=1)
        coef, *_ = np.linalg.lstsq(A, up[rough], rcond=None)
        if abs(float(coef[0])) <= PLANE_MAX_SLOPE and abs(float(coef[1])) <= PLANE_MAX_SLOPE:
            plane = coef[0] * fwd + coef[1] * lat + coef[2]
    if plane is None:
        plane = med
    floor = valid & (up <= plane + FLOOR_BAND_M)
    obst = (valid & (up >= plane + OBSTACLE_MIN_M) & (up <= plane + OBSTACLE_MAX_M)) | near
    # mostly-not-floor frame (a car filling the view), or the downward view is mostly
    # blocked/blank at close range (pressed against a car body): no veto at all
    if (int(np.sum(floor)) < MIN_FLOOR_FRAC * int(np.sum(valid))
            or blocked_frac > MAX_BLOCKED_FRAC):
        floor = np.zeros_like(floor)
    elif np.any(obst):
        # shadow: floor seen past (or under) a nearer obstacle in the same bearing
        nb = int(math.ceil(math.pi / SHADOW_BIN_RAD))
        # bearing/range from the camera itself: occlusion follows the camera's rays (from
        # base_link the image edges put a car body and the floor under it ~3 deg apart)
        cf, cl = fwd - mount_t[0], lat - mount_t[1]
        b = np.clip(((np.arctan2(cl, cf) + math.pi / 2) / SHADOW_BIN_RAD).astype(np.int64),
                    0, nb - 1)
        rng = np.hypot(cf, cl)
        nearest = np.full(nb, np.inf)
        np.minimum.at(nearest, b[obst], rng[obst])
        floor &= rng < nearest[b] - SHADOW_MARGIN_M

    return world(floor), world(obst)
