"""
stage1/geometry.py
------------------
CCTV 픽셀 → world 좌표 역투영.

좌표 규약 (MARC API 레퍼런스 "TF / 좌표계" 절 기준):
  * world      : REP-103, X=앞 Y=왼쪽 Z=위, 오른손.  제출 좌표는 모두 이 프레임.
  * camera_id  : ROS optical 규약. +Z=전방, +X=이미지 오른쪽, +Y=이미지 아래.
                 CameraInfo.K 와 같은 규약이므로 별도 축 뒤집기 없음.
  * /tf_static : parent=world, child=camera_id.  tf2 의 transform 은
                 p_world = R(q) @ p_cam + t  를 뜻하므로
                 t = 카메라의 world 위치(C), R(q) = cam→world 회전.

의존성: numpy 만. (ROS 메시지 객체는 받되 import 하지 않음 → Isaac Sim 없는 노트북에서도 테스트 가능)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# 기본 자료형
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraModel:
    """한 CCTV 의 내부/외부 파라미터 묶음."""
    camera_id: str
    K: np.ndarray               # (3,3) intrinsic
    R_cw: np.ndarray            # (3,3) cam → world 회전
    C_w: np.ndarray             # (3,)  카메라 world 위치
    ground_z: float             # 이 카메라가 내려다보는 지면 world z (m)
    width: int = 1280
    height: int = 720

    @property
    def forward_w(self) -> np.ndarray:
        """카메라 광축(+Z_cam) 의 world 방향 벡터."""
        return self.R_cw[:, 2]

    @property
    def forward_ground(self) -> np.ndarray:
        """광축을 지면(XY)에 투영해 정규화. 좌/우/앞/뒤 판정의 기준축."""
        f = self.forward_w.copy()
        f[2] = 0.0
        n = np.linalg.norm(f)
        if n < 1e-9:
            # 거의 수직으로 내려다보는 카메라: 이미지의 '위' 방향(-Y_cam)을 앞으로 쓴다.
            f = -self.R_cw[:, 1].copy()
            f[2] = 0.0
            n = np.linalg.norm(f)
        return f / n


# ----------------------------------------------------------------------------
# 변환 유틸
# ----------------------------------------------------------------------------
def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """geometry_msgs/Quaternion (x,y,z,w) → 3x3 회전행렬. 정규화 포함."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def camera_from_ros(camera_id: str, camera_info, tf_stamped, ground_z: float) -> CameraModel:
    """
    sensor_msgs/CameraInfo + geometry_msgs/TransformStamped(world→camera_id) + ground_height
    → CameraModel.

    camera_info.k 는 row-major 9개 float (ROS 2 필드명은 소문자 k).
    """
    K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)
    tr = tf_stamped.transform
    R_cw = quat_to_rotmat(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
    C_w = np.array([tr.translation.x, tr.translation.y, tr.translation.z], dtype=np.float64)
    return CameraModel(
        camera_id=camera_id, K=K, R_cw=R_cw, C_w=C_w, ground_z=float(ground_z),
        width=int(camera_info.width), height=int(camera_info.height),
    )


# ----------------------------------------------------------------------------
# 핵심: 픽셀 → world
# ----------------------------------------------------------------------------
def pixel_to_ray_world(cam: CameraModel, u: float, v: float) -> np.ndarray:
    """픽셀 (u,v) → world 프레임 단위 광선 방향."""
    d_cam = np.linalg.inv(cam.K) @ np.array([u, v, 1.0])
    d_w = cam.R_cw @ d_cam
    return d_w / np.linalg.norm(d_w)


def pixel_to_world(
    cam: CameraModel, u: float, v: float, plane_z: Optional[float] = None
) -> Optional[np.ndarray]:
    """
    픽셀 (u,v) 를 평면 z = plane_z (기본: cam.ground_z) 와 교차시켜 world 점 반환.
    교차 불가(광선이 위로 향하거나 평면과 평행) 시 None.

    공식:
        d_w = R_cw @ K^-1 [u, v, 1]^T
        t   = (plane_z - C.z) / d_w.z
        P   = C + t * d_w
    """
    z0 = cam.ground_z if plane_z is None else float(plane_z)
    d_w = pixel_to_ray_world(cam, u, v)
    if abs(d_w[2]) < 1e-9:
        return None
    t = (z0 - cam.C_w[2]) / d_w[2]
    if t <= 0:          # 교점이 카메라 뒤에 있음
        return None
    return cam.C_w + t * d_w


def world_to_pixel(cam: CameraModel, P_w: Sequence[float]) -> Optional[Tuple[float, float]]:
    """world 점 → 픽셀. 카메라 뒤(Z_cam<=0) 이면 None. 검증/디버그 용도."""
    P = np.asarray(P_w, dtype=np.float64)
    p_cam = cam.R_cw.T @ (P - cam.C_w)
    if p_cam[2] <= 1e-9:
        return None
    uv = cam.K @ (p_cam / p_cam[2])
    return float(uv[0]), float(uv[1])


# ----------------------------------------------------------------------------
# 바운딩박스/마스크 → 지면 접촉점
# ----------------------------------------------------------------------------
def bbox_ground_pixel(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """
    bbox 의 '바닥 중앙' 픽셀. 지면 평면 역투영에는 bbox 중심이 아니라
    지면과 닿는 점을 써야 한다 (중심을 쓰면 물체 높이의 절반만큼 카메라 쪽으로 당겨진 위치가 나온다).
    """
    return (x1 + x2) / 2.0, y2


def mask_ground_pixel(mask: np.ndarray, bottom_frac: float = 0.05) -> Optional[Tuple[float, float]]:
    """
    세그멘테이션 마스크(H,W bool) 의 하단 bottom_frac 행에 속하는 픽셀들의 평균 → 지면 접촉점.
    bbox 바닥 중앙보다 기울어진 물체/사람에 강건하다.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y_max = ys.max()
    y_min_band = y_max - max(1, int(bottom_frac * (y_max - ys.min() + 1)))
    sel = ys >= y_min_band
    return float(xs[sel].mean()), float(ys[sel].mean())


def detection_to_world(
    cam: CameraModel,
    bbox_xyxy: Sequence[float],
    mask: Optional[np.ndarray] = None,
    object_height_m: float = 0.0,
    stands_on_z: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    탐지 결과 → world 좌표.

    * bbox_xyxy          : [x1,y1,x2,y2] 픽셀
    * mask               : 있으면 마스크 하단으로 접촉점을 잡는다
    * object_height_m    : 0 이면 지면 접촉점 자체를 반환.
                           >0 이면 z 를 접촉점 + height/2 로 올린다 (GT 가 물체 중심이라고 판명될 때 사용).
                           → GT 정의(바닥점 vs 중심)는 데이터셋 생성기 GT 로 실측 필요. 기본 0.
    * stands_on_z        : 물체가 지면이 아니라 다른 물체 위에 있을 때(relation == 'on')
                           그 물체의 상면 z. None 이면 cam.ground_z.
    """
    if mask is not None:
        gp = mask_ground_pixel(mask)
        if gp is None:
            gp = bbox_ground_pixel(*bbox_xyxy)
    else:
        gp = bbox_ground_pixel(*bbox_xyxy)
    P = pixel_to_world(cam, gp[0], gp[1], plane_z=stands_on_z)
    if P is None:
        return None
    P = P.copy()
    P[2] += object_height_m / 2.0
    return P


# ----------------------------------------------------------------------------
# 공간 관계 (world 좌표 기반)
# ----------------------------------------------------------------------------
RELATIONS = ("near", "beside", "behind", "front", "left", "right", "on", "above")


def relation_scores(
    cam: CameraModel,
    anchor_w: np.ndarray,
    target_w: np.ndarray,
    near_radius: float = 2.0,
    beside_radius: float = 1.2,
) -> dict:
    """
    anchor→target 상대 벡터를 '카메라 시점' 기준축으로 분해해 각 relation 의 점수를 낸다.

    기준축 정의 (중요 — 이 정의가 채점 정의와 같다는 보장은 없다. 실측 항목):
      forward = 카메라 광축의 지면 투영 (카메라에서 멀어지는 방향)
      left    = z_up × forward   (REP-103 오른손: 앞을 보고 섰을 때 왼쪽)
      'front' = target 이 anchor 보다 카메라에 가까움 (forward 성분 < 0)
      'behind'= target 이 anchor 보다 카메라에서 멂  (forward 성분 > 0)

    반환: {relation: score(0~1)}. 최댓값을 택하거나, VLM 이 준 relation 과 일치 여부 검증에 쓴다.
    """
    rel = np.asarray(target_w) - np.asarray(anchor_w)
    fwd = cam.forward_ground
    left = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    dxy = np.array([rel @ fwd, rel @ left])       # (앞, 왼쪽) 성분
    dist_xy = np.linalg.norm(dxy)
    dz = rel[2]

    s = {r: 0.0 for r in RELATIONS}
    if dist_xy < 1e-6 and abs(dz) < 1e-6:
        return s
    # beside: 접촉 수준의 근접 (0 에서 1, beside_radius 에서 0)
    s["beside"] = float(np.clip(1.0 - dist_xy / beside_radius, 0, 1))
    # near: beside 보다 멀지만 near_radius 안 (띠 구간에서 1, 밖/안에서 선형 감쇠)
    if dist_xy < beside_radius:
        s["near"] = float(dist_xy / beside_radius)
    else:
        s["near"] = float(np.clip(1.0 - (dist_xy - beside_radius) / (near_radius - beside_radius), 0, 1))
    if dist_xy > 1e-6:
        ang = np.arctan2(dxy[1], dxy[0])          # 0 = 앞(behind), ±pi = 뒤(front), +pi/2 = 왼쪽
        # 아주 가까우면 방향보다 'beside' 가 우선 → 방향 점수에 거리 가중
        w = float(np.clip(dist_xy / beside_radius, 0, 1))
        s["behind"] = w * float(max(0.0, np.cos(ang)))
        s["front"] = w * float(max(0.0, -np.cos(ang)))
        s["left"] = w * float(max(0.0, np.sin(ang)))
        s["right"] = w * float(max(0.0, -np.sin(ang)))
    if dz > 0.05:
        close_xy = dist_xy < beside_radius
        s["on"] = 1.0 if close_xy else 0.3
        s["above"] = 0.8 if close_xy else 0.2
        if close_xy:                              # 위에 얹힌 경우 수평 관계는 후순위
            s["beside"] *= 0.5
            s["near"] *= 0.5
    return s


def best_relation(cam: CameraModel, anchor_w, target_w) -> str:
    s = relation_scores(cam, anchor_w, target_w)
    return max(s, key=s.get)
