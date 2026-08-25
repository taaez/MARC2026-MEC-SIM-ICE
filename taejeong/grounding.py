"""General geometric grounding for MARC 2026 Stage 1/2.

parse_command() 결과 + (임의 탐지기의) 카메라별 탐지 결과 + 카메라 파라미터
  → 카메라 선택, 랜드마크 인스턴스 확정, relation 기하 판정, 좌표 계산
  → GroundingResult payload

GENERAL by design — 대회 요구사항("문제·배경이 바뀐다, 씬 하드코딩 금지")에 맞춘
원칙:
  * 특정 시나리오/문항/좌표/씬 배치를 참조하지 않는다.
  * 클래스별 높이·크기 상수를 두지 않는다 — 물체/랜드마크의 높이는 그 탐지
    bbox 와 카메라 기하에서 그 자리에서 추정한다 (estimate_top_z).
  * 어휘(클래스 목록)만 공식 로스터에 닫혀 있고, 판정 로직은 전부 기하다.
  * 임계값은 GroundingConfig 로 외부화 — 연습 채점 역산으로 보정하되 코드는 불변.

좌표 규약 (marc_sdk / API 레퍼런스):
  world: REP-103 (X앞 Y왼쪽 Z위), 제출 좌표는 world.
  camera: ROS optical (+Z 전방, +X 오른쪽, +Y 아래) — CameraInfo.K 와 동일 규약.
  /tf_static (world→camera_id): p_world = R @ p_cam + t.

제출 좌표 정의(팀 결정): target_coord/anchor_coord 는 **물체 중심점** —
바닥 접촉점 (x, y) 에 z 를 (base+top)/2 로 올린 값.

의존성: numpy 만 (ROS 무의존 → 어느 머신에서든 합성 씬으로 테스트 가능).
탐지기 무관: Detection(label, bbox, score) 만 맞추면 YOLO/DINO 어느 쪽이든 연결.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from command_parser import LANDMARK_SYNONYMS, ParsedCommand
from situation_resolver import POSE_CLASS as PERSON_POSE_CLASSES
from situation_resolver import resolve_situation

RELATIONS = ("near", "beside", "behind", "front", "left", "right", "on", "above")

# 제너릭 "car" 힌트가 허용하는 차량 클래스 (공식 landmark_assets 의 차량 변형)
CAR_CLASSES = {"police_car", "normal_car_1", "normal_car_2",
               "sports_car", "sports_car_2", "suv", "taxi", "car"}

# pose 코스 버킷(command_parser) → 공식 pose 클래스 후보 집합.
# 언어 표면형이 pose 클래스와 1:1 이 아니므로(실측: "lying face-down" 이
# person_bending 인 예 존재) 버킷은 후보를 좁힐 뿐 단정하지 않는다.
POSE_BUCKET_CANDIDATES = {
    "facedown": {"person_prone", "person_bending", "person_lying_down"},
    "bent":     {"person_bending", "person_crouching"},
    "down":     {"person_prone", "person_bending", "person_lying_down",
                 "person_crouching"},
    "lying":    {"person_lying_down", "person_bending", "person_reclining"},
    "sitting":  {"person_sitting", "person_reclining"},
    "standing": {"person_walking", "person_walking_away", "person_reaching"},
}


# ────────────────────────────────────────────────────────────────────────────
# 카메라 모델과 기본 기하
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CameraModel:
    camera_id: str
    K: np.ndarray               # (3,3) intrinsic
    R_cw: np.ndarray            # (3,3) cam→world 회전
    C_w: np.ndarray             # (3,)  카메라 world 위치
    ground_z: float             # 이 카메라가 보는 지면 world z
    width: int = 1280
    height: int = 720

    @property
    def forward_ground(self) -> np.ndarray:
        """광축(+Z_cam)의 지면(XY) 투영 단위벡터 — 방향 relation 의 기준축."""
        f = self.R_cw[:, 2].copy()
        f[2] = 0.0
        n = np.linalg.norm(f)
        if n < 1e-9:                       # 수직 하방 카메라: 이미지 '위'를 앞으로
            f = -self.R_cw[:, 1].copy()
            f[2] = 0.0
            n = np.linalg.norm(f)
        return f / n


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def camera_from_ros(camera_id: str, camera_info, tf_stamped, ground_z: float) -> CameraModel:
    """sensor_msgs/CameraInfo + TransformStamped(world→camera) → CameraModel."""
    K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)
    tr = tf_stamped.transform
    return CameraModel(
        camera_id=camera_id, K=K,
        R_cw=quat_to_rotmat(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w),
        C_w=np.array([tr.translation.x, tr.translation.y, tr.translation.z]),
        ground_z=float(ground_z),
        width=int(camera_info.width), height=int(camera_info.height),
    )


def pixel_ray(cam: CameraModel, u: float, v: float) -> np.ndarray:
    d = cam.R_cw @ (np.linalg.inv(cam.K) @ np.array([u, v, 1.0]))
    return d / np.linalg.norm(d)


def pixel_to_plane(cam: CameraModel, u: float, v: float,
                   plane_z: Optional[float] = None) -> Optional[np.ndarray]:
    """픽셀 → 평면 z=plane_z(기본 ground_z) 교점. 교차 불가면 None."""
    z0 = cam.ground_z if plane_z is None else float(plane_z)
    d = pixel_ray(cam, u, v)
    if abs(d[2]) < 1e-9:
        return None
    t = (z0 - cam.C_w[2]) / d[2]
    if t <= 0:
        return None
    return cam.C_w + t * d


def world_to_pixel(cam: CameraModel, P_w: Sequence[float]) -> Optional[tuple]:
    p_cam = cam.R_cw.T @ (np.asarray(P_w, dtype=np.float64) - cam.C_w)
    if p_cam[2] <= 1e-9:
        return None
    uv = cam.K @ (p_cam / p_cam[2])
    return float(uv[0]), float(uv[1])


def estimate_top_z(cam: CameraModel, bbox: Sequence[float],
                   ref_xy: np.ndarray, fallback_z: float) -> float:
    """bbox 상단이 나타내는 물체 꼭대기의 world z 를 기하로 추정.

    클래스별 높이 상수 없이: 상단 중앙 픽셀의 광선이 ref_xy 수직선에 가장
    가까워지는 지점의 z. CCTV(물체 꼭대기보다 높은 카메라)에서 bbox 상단은
    '먼 쪽 윗모서리'의 투영이므로 ref_xy 는 footprint 의 먼 쪽 가장자리를
    넘겨야 한다. (직립 물체 가정 — 기울었으면 근사치)
    """
    x1, y1, x2, _ = bbox
    d = pixel_ray(cam, (x1 + x2) / 2.0, y1)
    d_xy = d[:2]
    denom = float(d_xy @ d_xy)
    if denom < 1e-12:                      # 광선이 수직: 높이 추정 불가
        return fallback_z
    t = float(d_xy @ (np.asarray(ref_xy) - cam.C_w[:2])) / denom
    if t <= 0:
        return fallback_z
    return float(cam.C_w[2] + t * d[2])


# ────────────────────────────────────────────────────────────────────────────
# 탐지 → world 좌표 (중심점)
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Detection:
    label: str                       # 클래스명 (공식 로스터 / person_* pose 클래스)
    bbox: tuple                      # (x1, y1, x2, y2) 픽셀
    score: float = 1.0


@dataclass
class Located:
    """한 탐지의 world 기하 요약."""
    det: Detection
    base: np.ndarray                 # 바닥(또는 지지면) 접촉점 world (x,y,z)
    top_z: float                     # 꼭대기 z (기하 추정)
    radius: float                    # 수평 footprint 반경 추정 (m)

    @property
    def center(self) -> list:
        """제출 좌표 정의(팀 결정): 중심점 = 접촉점 (x,y) + z 는 (base+top)/2."""
        return [float(self.base[0]), float(self.base[1]),
                float((self.base[2] + self.top_z) / 2.0)]


def locate(cam: CameraModel, det: Detection,
           base_z: Optional[float] = None) -> Optional[Located]:
    """탐지 bbox → Located. base_z 지정 시 그 평면(예: 랜드마크 상면) 기준.

    bbox 하단 중앙의 역투영은 물체의 '카메라 쪽 가장자리' 접촉점이므로,
    footprint 반경만큼 카메라 반대 방향으로 밀어 footprint **중심**을 얻는다
    (넓은 물체에서 중요 — 벤치·테이블은 반경이 수십 cm). 상단 z 추정은 먼 쪽
    가장자리에서 매칭한다 (estimate_top_z 참조).
    """
    x1, y1, x2, y2 = det.bbox
    near = pixel_to_plane(cam, (x1 + x2) / 2.0, y2, plane_z=base_z)
    if near is None:
        return None
    # footprint 반경: bbox 좌/우 하단을 같은 평면에 투영한 폭의 절반
    left = pixel_to_plane(cam, x1, y2, plane_z=base_z)
    right = pixel_to_plane(cam, x2, y2, plane_z=base_z)
    radius = 0.0
    if left is not None and right is not None:
        radius = float(np.linalg.norm((right - left)[:2]) / 2.0)
    away = near[:2] - cam.C_w[:2]
    n = float(np.linalg.norm(away))
    away = away / n if n > 1e-9 else np.zeros(2)
    center_xy = near[:2] + radius * away
    base = np.array([center_xy[0], center_xy[1], float(near[2])])
    far_xy = near[:2] + 2.0 * radius * away
    top_z = max(estimate_top_z(cam, det.bbox, far_xy, float(base[2])),
                float(base[2]))
    return Located(det=det, base=base, top_z=top_z, radius=radius)


# ────────────────────────────────────────────────────────────────────────────
# relation 기하 판정
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class GroundingConfig:
    """임계값 외부화 — 연습 채점 역산으로 보정하는 대상 (코드 로직은 불변)."""
    beside_gap_m: float = 0.6        # footprint 간 간격이 이내면 beside
    near_gap_m: float = 3.0          # 이내면 near
    on_xy_slack: float = 0.6         # 'on' 판정: 수평 중심이 anchor 반경*(1+slack) 안
    on_z_tol_m: float = 0.45         # 'on' 판정: target 바닥과 anchor 상면 z 차 허용
    above_clearance_m: float = 0.15  # anchor 상면보다 이만큼 높이 떠 있으면 above
    directional_min_gap_m: float = 0.15  # 방향 relation 을 고려할 최소 수평 거리


def classify_relation(cam: CameraModel, anchor: Located, target: Located,
                      cfg: GroundingConfig = GroundingConfig(),
                      hint: Optional[str] = None) -> str:
    """anchor(랜드마크) 대비 target 의 공간 관계를 기하로 판정.

    기본은 근접 계열(on/above/beside/near)로만 판정한다. 방향(behind/front/
    left/right)은 텍스트가 방향 relation 을 힌트로 줄 때만 기하로 확인해
    반환한다 — 방향은 명령이 말해줄 때만 의미가 있고, 기하만으로 near 대신
    방향을 고르는 것은 근거가 없기 때문.

    방향의 기준 시점은 카메라 광축의 지면 투영으로 가정한다(채점 정의
    미공개 — 실측 검증 항목). 'front' = target 이 anchor 보다 카메라에 가까움.
    """
    rel_xy = target.base[:2] - anchor.base[:2]
    dist = float(np.linalg.norm(rel_xy))
    gap = dist - anchor.radius - target.radius        # footprint 간 순간격

    # on / above: 수평으로 anchor footprint 위 + 수직 배치로 구분
    if dist <= anchor.radius * (1.0 + cfg.on_xy_slack) + target.radius:
        dz_base = float(target.base[2]) - anchor.top_z
        if abs(dz_base) <= cfg.on_z_tol_m and target.base[2] > anchor.base[2] + 1e-6:
            return "on"
        if dz_base > cfg.above_clearance_m:
            return "above"

    if hint in ("behind", "front", "left", "right") and \
            dist >= cfg.directional_min_gap_m:
        fwd = cam.forward_ground
        left = np.cross(np.array([0.0, 0.0, 1.0]), fwd)[:2]
        a_fwd, a_left = float(rel_xy @ fwd[:2]), float(rel_xy @ left)
        if abs(a_fwd) >= abs(a_left):
            return "behind" if a_fwd > 0 else "front"
        return "left" if a_left > 0 else "right"

    if gap <= cfg.beside_gap_m:
        return "beside"
    return "near"    # near_gap 밖이어도 근접 계열에선 near 가 가장 보수적


def _looks_on(anchor: Located, target_det: Detection) -> bool:
    """이미지 공간 사전판정: target bbox 가 anchor bbox 위쪽에 겹쳐 있는가.

    'on' 후보일 때만 지지면(anchor 상면) 평면으로 재투영하기 위한 저비용 체크.
    """
    ax1, ay1, ax2, ay2 = anchor.det.bbox
    tx1, ty1, tx2, ty2 = target_det.bbox
    overlap = min(ax2, tx2) - max(ax1, tx1)
    if overlap <= 0.3 * (tx2 - tx1):
        return False
    return ay1 - (ty2 - ty1) <= ty2 <= ay2      # target 바닥이 anchor 수직범위 부근


# ────────────────────────────────────────────────────────────────────────────
# 전체 파이프라인: (파싱, 탐지, 카메라) → GroundingResult 재료
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class GroundingDecision:
    camera_id: str
    target_type: str
    landmark: str                    # 랜드마크 클래스 ("" = 미확정)
    relation: Optional[str]          # 사람 문제는 None
    situation: Optional[str]         # 물체 문제는 None
    target_coord: list
    anchor_coord: list
    pose_type: Optional[str] = None  # 사람 문제: 선택된 pose 클래스
    debug: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "interpretation": {
                "target_type": self.target_type,
                "landmark": self.landmark,
                "relation": self.relation,
                "situation": self.situation,
            },
            "grounding": {
                "anchor_coord": self.anchor_coord,
                "target_coord": self.target_coord,
            },
        }


def _landmark_ok(label: str, hint: Optional[str]) -> bool:
    if hint is None:
        return label in LANDMARK_SYNONYMS or label in CAR_CLASSES
    if hint == "car":
        return label in CAR_CLASSES
    return label == hint


def _target_ok(label: str, parsed: ParsedCommand) -> bool:
    if parsed.is_person:
        if not (label in PERSON_POSE_CLASSES or label == "person"):
            return False
        cand = POSE_BUCKET_CANDIDATES.get(parsed.pose_hint or "")
        return cand is None or label == "person" or label in cand
    return label == parsed.target_type


def resolve(parsed: ParsedCommand,
            detections: dict,               # {camera_id: [Detection, ...]}
            cameras: dict,                  # {camera_id: CameraModel}
            cfg: GroundingConfig = GroundingConfig(),
            ground_surface_token: Optional[str] = None,
            ) -> Optional[GroundingDecision]:
    """모든 카메라의 탐지에서 (카메라, target, anchor) 최적 조합을 고르고
    relation/situation/좌표를 계산한다. target 미탐지면 None (상위 폴백 몫).

    ground_surface_token: 사람이 랜드마크 위가 아닐 때 situation 규칙에 넣을
    지면 토큰(grass/road/...). 판정 수단이 없으면 None → 공식 fallback 경로.
    """
    best = None      # (score, cam_id, target Located, anchor Located|None)
    for cam_id, dets in detections.items():
        cam = cameras.get(cam_id)
        if cam is None:
            continue
        targets = [d for d in dets if _target_ok(d.label, parsed)]
        anchors = [d for d in dets if _landmark_ok(d.label, parsed.landmark_type)]
        for tdet in targets:
            t_loc = locate(cam, tdet)
            if t_loc is None:
                continue
            # 랜드마크 후보: 명령에 타입이 있으면 그 타입만, 없으면 전 랜드마크
            # 중 3D 최근접 (팀 결정). 'on' 후보면 지지면 평면으로 target 재투영.
            a_best, a_dist = None, float("inf")
            for adet in anchors:
                if adet is tdet:
                    continue
                a_loc = locate(cam, adet)
                if a_loc is None:
                    continue
                t_here = t_loc
                if _looks_on(a_loc, tdet):
                    # 지지면(anchor 상면) 평면으로 재투영. 채택 조건: 재투영이
                    # target 을 anchor 중심으로 '끌어당기고'(진짜 on 이면 중심
                    # 부근에 떨어짐) footprint 안일 때만. 이미지상 겹칠 뿐
                    # 앞/뒤 지면에 서 있는 물체는 재투영하면 오히려 anchor 에서
                    # 멀어지므로 여기서 걸러진다.
                    re_t = locate(cam, tdet, base_z=a_loc.top_z)
                    if re_t is not None:
                        d_ground = float(np.linalg.norm(
                            t_loc.base[:2] - a_loc.base[:2]))
                        d_top = float(np.linalg.norm(
                            re_t.base[:2] - a_loc.base[:2]))
                        if d_top < d_ground and \
                                d_top <= a_loc.radius * (1.0 + cfg.on_xy_slack):
                            t_here = re_t
                d = float(np.linalg.norm(t_here.base[:2] - a_loc.base[:2]))
                if d < a_dist:
                    a_best, a_dist, t_loc_adj = a_loc, d, t_here
            score = tdet.score
            if a_best is not None:
                score += 0.5                       # 랜드마크 동반 카메라 우대
                if parsed.landmark_type is not None:
                    score += 0.5                   # 텍스트 타입 일치 우대
                score += 1.0 / (1.0 + a_dist)      # 가까울수록 우대
            cand = (score, cam_id,
                    t_loc if a_best is None else t_loc_adj, a_best)
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _, cam_id, t_loc, a_loc = best
    cam = cameras[cam_id]

    landmark = ""
    relation: Optional[str] = None
    situation: Optional[str] = None
    anchor_coord = [0.0, 0.0, 0.0]
    geo_rel = None
    if a_loc is not None:
        landmark = a_loc.det.label
        anchor_coord = a_loc.center
        geo_rel = classify_relation(cam, a_loc, t_loc, cfg,
                                    hint=parsed.relation_hint)

    if parsed.is_person:
        pose = t_loc.det.label if t_loc.det.label.startswith("person_") else None
        support = landmark if (geo_rel in ("on", "above") and landmark) \
            else ground_surface_token
        situation = resolve_situation(pose or "", support).situation
        target_type = "person"
        relation = None
    else:
        target_type = parsed.target_type
        pose = None
        # 텍스트가 확정적이면 텍스트, 아니면 기하 판정 (기하 불가 시 텍스트 힌트)
        if parsed.relation_confident and parsed.relation_hint:
            relation = parsed.relation_hint
        else:
            relation = geo_rel or parsed.relation_hint or "near"

    return GroundingDecision(
        camera_id=cam_id, target_type=target_type, landmark=landmark,
        relation=relation, situation=situation,
        target_coord=t_loc.center, anchor_coord=anchor_coord,
        pose_type=pose,
        debug={"geo_relation": geo_rel,
               "text_relation": parsed.relation_hint,
               "text_confident": parsed.relation_confident,
               "target_base": [float(v) for v in t_loc.base],
               "target_top_z": t_loc.top_z},
    )
