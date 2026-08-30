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
# XY 평면 OBB 표면거리 (near/beside 판정 척도)
#
# near/beside 는 지면 위의 평면 관계이고, 랜드마크 크기가 제각각이라(자판기 2.0m
# vs 쓰레기통 0.7m) **중심거리로는 관계 순서가 뒤집힌다**. 실측 근거와 임계값
# 유도는 리포 루트 MARC2026_Chungmu_near_beside_OBB_analysis.md (2.5/8/11 절):
#   vending_machine-umbrella  원점거리 1.85m / OBB 표면거리 0.470m  정답 beside
#   trash_can-umbrella        원점거리 1.15m / OBB 표면거리 0.697m  정답 near
# 원점거리는 순서가 반대인데 OBB 표면거리는 바로 선다.
#
# (같은 판정을 taejeong/grounding.py 도 한다 — 그쪽이 대회 제출 경로다. 두 모듈은
#  의도적으로 서로 독립이라 헬퍼를 공유하지 않고 같은 규칙을 각자 구현한다.)
# ----------------------------------------------------------------------------
# XY OBB 표면거리 임계값. 문서 11 절의 유도값은 0.58m 지만 그건 **공식 OBB 척도**의
# 값이고, 여기서 재는 것은 bbox 로부터의 단안 추정치라 같은 쌍이 더 크게 나온다
# (자판기-우산: 공식 0.470m vs 단안 0.61m). 우리 척도의 경계로 재보정한 값이다
# (taejeong/grounding.py 의 GroundingConfig.beside_gap_m 과 같은 근거·같은 값 --
#  근거 실측표는 그쪽 주석에 있다. 값을 바꿀 때는 반드시 두 곳을 같이 고칠 것).
#
# 2026-08-30 2차 재보정: 0.63 -> 0.80. 현재 검출기 조합(소형모델 라우팅 + 새 B
# 가중치)의 실측 gap 은 beside 최대 0.606 / near 최소 0.987 로 분리되는데,
# 0.63 은 beside 쪽 여유가 0.024m 뿐이라 라이브에서 뒤집혔다. 구간 중앙으로 옮겨
# 양쪽 여유를 0.19m 로 맞춘다.
BESIDE_GAP_M = 0.80


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-18 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _separated(A: np.ndarray, B: np.ndarray) -> bool:
    """두 볼록다각형이 분리축을 갖는가(=겹치지 않는가). SAT — 볼록에서 정확."""
    for P in (A, B):
        n = len(P)
        for i in range(n):
            e = P[(i + 1) % n] - P[i]
            axis = np.array([-e[1], e[0]])
            L = float(np.linalg.norm(axis))
            if L < 1e-12:
                continue
            axis = axis / L
            a, b = A @ axis, B @ axis
            if a.max() < b.min() - 1e-12 or b.max() < a.min() - 1e-12:
                return True
    return False


def polygon_gap_xy(A: np.ndarray, B: np.ndarray) -> float:
    """XY 볼록다각형 표면 사이 최소거리. 겹치거나 닿으면 0.

    떨어진 두 볼록다각형의 최소거리는 항상 '한쪽 꼭짓점 ↔ 다른 쪽 변'에서
    실현되므로 그 조합만 훑으면 된다. 점(1x2)도 다각형으로 받아들인다.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    if len(A) > 2 and len(B) > 2 and not _separated(A, B):
        return 0.0
    best = float("inf")
    for P, Q in ((A, B), (B, A)):
        for q in Q:
            for i in range(len(P)):
                d = _point_seg_dist(q, P[i], P[(i + 1) % len(P)])
                if d < best:
                    best = d
    return best


def footprint_obb_xy(cam: CameraModel, bbox_xyxy: Sequence[float],
                     plane_z: Optional[float] = None) -> Optional[np.ndarray]:
    """탐지 bbox → 지면(XY) footprint 를 직사각형(OBB)으로 본 꼭짓점 (4,2).

    단안 CCTV 에서 관측 가능한 축은 둘뿐이다:
      가로 : 시선에 수직. bbox 좌/우 아랫점을 지면에 투영한 폭의 절반이 반길이.
      깊이 : 시선 방향. bbox 로는 볼 수 없어 **정사각형 footprint 로 가정**하고
             같은 반길이를 쓴다. 상자의 카메라 쪽 면이 관측된 접촉선에 놓이도록
             중심을 그 반길이만큼 카메라 반대쪽으로 민다.
    """
    x1, _y1, x2, y2 = bbox_xyxy
    near = pixel_to_world(cam, (x1 + x2) / 2.0, y2, plane_z=plane_z)
    left = pixel_to_world(cam, x1, y2, plane_z=plane_z)
    right = pixel_to_world(cam, x2, y2, plane_z=plane_z)
    if near is None or left is None or right is None:
        return None
    h = float(np.linalg.norm((right - left)[:2]) / 2.0)
    away = near[:2] - cam.C_w[:2]
    n = float(np.linalg.norm(away))
    away = away / n if n > 1e-9 else np.array([1.0, 0.0])
    c = near[:2] + h * away
    perp = np.array([-away[1], away[0]])
    return np.array([c + h * away + h * perp, c + h * away - h * perp,
                     c - h * away - h * perp, c - h * away + h * perp])


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
    anchor_fp: Optional[np.ndarray] = None,
    target_fp: Optional[np.ndarray] = None,
) -> dict:
    """
    anchor→target 상대 벡터를 '카메라 시점' 기준축으로 분해해 각 relation 의 점수를 낸다.

    기준축 정의 (중요 — 이 정의가 채점 정의와 같다는 보장은 없다. 실측 항목):
      forward = 카메라 광축의 지면 투영 (카메라에서 멀어지는 방향)
      left    = z_up × forward   (REP-103 오른손: 앞을 보고 섰을 때 왼쪽)
      'front' = target 이 anchor 보다 카메라에 가까움 (forward 성분 < 0)
      'behind'= target 이 anchor 보다 카메라에서 멂  (forward 성분 > 0)

    near/beside 는 **XY 평면 OBB 표면거리**로 가른다 (BESIDE_GAP_M 주석 참조).
    ``anchor_fp``/``target_fp`` 는 ``footprint_obb_xy`` 가 만든 (4,2) 꼭짓점이며,
    둘 중 하나라도 없으면 점끼리의 거리로 퇴화한다(=물체 크기를 못 쓰는 옛 동작).
    두 점수의 크기가 뒤집히는 지점이 정확히 ``BESIDE_GAP_M`` 이 되도록 램프
    파라미터를 2*임계값으로 두었다.

    반환: {relation: score(0~1)}. 최댓값을 택하거나, VLM 이 준 relation 과 일치 여부 검증에 쓴다.
    """
    rel = np.asarray(target_w) - np.asarray(anchor_w)
    fwd = cam.forward_ground
    left = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    dxy = np.array([rel @ fwd, rel @ left])       # (앞, 왼쪽) 성분
    dist_xy = np.linalg.norm(dxy)
    dz = rel[2]

    # near/beside 판정거리: 가능하면 XY OBB 표면거리, 없으면 중심 XY 거리
    if anchor_fp is not None and target_fp is not None:
        gap = polygon_gap_xy(anchor_fp, target_fp)
    else:
        gap = dist_xy
    ramp = 2.0 * BESIDE_GAP_M          # 두 점수의 교차점이 곧 BESIDE_GAP_M

    s = {r: 0.0 for r in RELATIONS}
    if dist_xy < 1e-6 and abs(dz) < 1e-6:
        return s
    # beside: 표면이 붙어 있는 수준 (gap 0 에서 1, ramp 에서 0)
    s["beside"] = float(np.clip(1.0 - gap / ramp, 0, 1))
    # near: beside 보다 멀지만 near_radius 안 (띠 구간에서 1, 밖/안에서 선형 감쇠)
    if gap < ramp:
        s["near"] = float(gap / ramp)
    else:
        s["near"] = float(np.clip(1.0 - (gap - ramp) / max(near_radius - ramp, 1e-9), 0, 1))
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
