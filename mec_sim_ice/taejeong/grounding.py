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

try:                              # package import (e.g. `from taejeong.grounding import ...`)
    from .command_parser import LANDMARK_SYNONYMS, ParsedCommand
    from .situation_resolver import POSE_CLASS as PERSON_POSE_CLASSES
    from .situation_resolver import resolve_situation
except ImportError:               # run directly as a script from within taejeong/
    from command_parser import LANDMARK_SYNONYMS, ParsedCommand
    from situation_resolver import POSE_CLASS as PERSON_POSE_CLASSES
    from situation_resolver import resolve_situation

RELATIONS = ("near", "beside", "behind", "front", "left", "right", "on", "above")

# 제너릭 "car" 힌트가 허용하는 차량 클래스 (공식 landmark_assets 의 차량 변형)
CAR_CLASSES = {"police_car", "normal_car_1", "normal_car_2",
               "sports_car", "sports_car_2", "suv", "taxi", "car"}

# 구역(command_parser.area_hint) → 그 구역을 정의하는 지표 클래스.
#
# 명령이 랜드마크를 말하지 않고 구역만 말하는 경우가 있다 ("Find the umbrella
# dropped in the empty parking space."). 구역 자체는 탐지 대상이 아니지만, 그
# 구역을 정의하는 지표 클래스는 탐지된다 — "주차 구역" 은 곧 "차량이 있는 곳"
# 이다. 이 매핑은 씬 배치가 아니라 **단어의 일반적 의미**에서 나온 것이라
# 배경이 바뀌어도 성립한다 (특정 시나리오/좌표를 참조하지 않는다).
#
# 안전성: 지표가 한 카메라에서도 안 잡히면 게이트가 그냥 풀려서 이 계층이 없을
# 때와 똑같이 동작한다 (순수 폴백 — 더 맞힐 수는 있어도 나빠지지는 않는다).
# "road" 는 연습 코퍼스에 근거 사례가 없어 일부러 비워 뒀다: 근거 없이 매핑을
# 만들면 멀쩡한 카메라를 걸러낼 위험만 생긴다.
AREA_INDICATOR_CLASSES = {
    "parking": CAR_CLASSES,
    "park": {"picnic_table", "bench"},
}

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


NEARISH_RELATIONS = ("on", "above", "beside", "near")


def _support_adjusted(cam: CameraModel, tdet: Detection, t_loc: Located,
                      a_loc: Located, cfg: GroundingConfig) -> Located:
    """'on' 후보면 지지면(anchor 상면) 평면으로 target 을 재투영해 돌려준다.

    채택 조건: 재투영이 target 을 anchor 중심으로 '끌어당기고'(진짜 on 이면
    중심 부근에 떨어짐) footprint 안일 때만. 이미지상 겹칠 뿐 앞/뒤 지면에 서
    있는 물체는 재투영하면 오히려 anchor 에서 멀어지므로 여기서 걸러진다.
    """
    if not _looks_on(a_loc, tdet):
        return t_loc
    re_t = locate(cam, tdet, base_z=a_loc.top_z)
    if re_t is None:
        return t_loc
    d_ground = float(np.linalg.norm(t_loc.base[:2] - a_loc.base[:2]))
    d_top = float(np.linalg.norm(re_t.base[:2] - a_loc.base[:2]))
    if d_top < d_ground and d_top <= a_loc.radius * (1.0 + cfg.on_xy_slack):
        return re_t
    return t_loc


def _relation_fit(cam: CameraModel, a_loc: Located, t_loc: Located,
                  cfg: GroundingConfig, hint: Optional[str]) -> tuple:
    """이 (anchor, target) 쌍이 명령이 말한 관계를 얼마나 잘 실현하는가.

    (주 점수, 부 점수) 튜플을 그대로 크기 비교해 순위를 매긴다. 두 값을 하나로
    합치지 않는 이유: 근접 계열에서 '조금 더 먼데 버킷만 맞는' 후보가 더 가까운
    후보를 이겨버리는 역전이 생기기 때문이다.

    카메라 선택의 주 근거다. CCTV 들은 서로 다른 공간을 보므로(시야가 겹치지
    않는다) 탐지 confidence 를 카메라 간에 비교하는 것은 근거가 없다 — 서로
    다른 장면에서 나온 값이기 때문이다. 대신 '명령대로 배치된 쌍이 있는
    카메라'를 고른다.
    """
    gap = (float(np.linalg.norm(t_loc.base[:2] - a_loc.base[:2]))
           - a_loc.radius - t_loc.radius)
    prox = max(0.0, 1.0 - max(0.0, gap) / max(cfg.near_gap_m, 1e-9))
    if hint is None:
        return (prox, 0.0)
    geo = classify_relation(cam, a_loc, t_loc, cfg, hint=hint)
    if hint in NEARISH_RELATIONS:
        # on/above/beside/near 는 전부 '가깝다'의 변주다 (거리 임계로만 갈린다).
        # 그래서 근접도를 주 신호로 두고 버킷 일치는 동점일 때만 쓴다.
        return (prox, 1.0 if geo == hint else 0.0)
    # 방향 관계(behind/front/left/right)는 범주적이다 — 맞는 쪽에 있는지가 핵심
    # 이고 거리는 부차적이다.
    return (1.0 if geo == hint else 0.0, prox)


def _build_decision(parsed: ParsedCommand, cam: CameraModel, cam_id: str,
                    t_loc: Optional[Located], a_loc: Optional[Located],
                    cfg: GroundingConfig, ground_surface_token: Optional[str],
                    extra_debug: dict) -> GroundingDecision:
    """(카메라, target, anchor) → 제출 payload 재료. t_loc=None 이면 랜드마크만."""
    landmark = ""
    anchor_coord = [0.0, 0.0, 0.0]
    geo_rel = None
    if a_loc is not None:
        landmark = a_loc.det.label
        anchor_coord = a_loc.center
        if t_loc is not None:
            geo_rel = classify_relation(cam, a_loc, t_loc, cfg,
                                        hint=parsed.relation_hint)

    # target 미탐지 시 anchor 좌표를 target 좌표로 대신 낸다: 허용 반경(5m)에
    # 비해 실측 anchor↔target 거리가 훨씬 짧아 부분점수를 크게 건진다.
    target_coord = t_loc.center if t_loc is not None else list(anchor_coord)

    relation: Optional[str] = None
    situation: Optional[str] = None
    if parsed.is_person:
        pose = (t_loc.det.label if t_loc is not None
                and t_loc.det.label.startswith("person_") else None)
        support = landmark if (geo_rel in ("on", "above") and landmark) \
            else ground_surface_token
        # pose 미상이면 공식 fallback 경로로 기본 situation 이 나온다. 오답에
        # 감점이 없으므로 None 을 내는 것보다 항상 낫다.
        situation = resolve_situation(pose or "", support).situation
        target_type = "person"
    else:
        pose = None
        target_type = parsed.target_type
        # 텍스트가 확정적이면 텍스트, 아니면 기하 판정 (기하 불가 시 텍스트 힌트)
        if parsed.relation_confident and parsed.relation_hint:
            relation = parsed.relation_hint
        else:
            relation = geo_rel or parsed.relation_hint or "near"

    debug = {"geo_relation": geo_rel,
             "text_relation": parsed.relation_hint,
             "text_confident": parsed.relation_confident,
             "target_base": ([float(v) for v in t_loc.base]
                             if t_loc is not None else None),
             "target_top_z": t_loc.top_z if t_loc is not None else None}
    debug.update(extra_debug)
    return GroundingDecision(
        camera_id=cam_id, target_type=target_type, landmark=landmark,
        relation=relation, situation=situation,
        target_coord=target_coord, anchor_coord=anchor_coord,
        pose_type=pose, debug=debug,
    )


def resolve(parsed: ParsedCommand,
            detections: dict,               # {camera_id: [Detection, ...]}
            cameras: dict,                  # {camera_id: CameraModel}
            cfg: GroundingConfig = GroundingConfig(),
            ground_surface_token: Optional[str] = None,
            ) -> Optional[GroundingDecision]:
    """카메라 선택 → (target, anchor) 확정 → relation/situation/좌표 계산.

    CCTV 들은 **서로 다른 공간**을 본다 (시야가 겹치지 않는다). 그래서 카메라
    선택은 '여러 후보 중 점수 높은 것 고르기'가 아니라 '명령이 가리킨 물체
    쌍이 실제로 있는 곳 찾기'다. 순서:

      1. 카메라 게이트 — 명령이 랜드마크를 지목했으면 그것이 실제로 검출된
         카메라만 후보로 남긴다. 랜드마크를 말하지 않고 구역만 말했으면
         (area_hint) 그 구역의 지표 클래스가 보이는 카메라로 좁힌다. 둘 다
         못 좁히면 게이트를 푼다 — 즉 게이트는 순수 폴백 계층이라 실패해도
         이전 동작으로 되돌아갈 뿐 나빠지지 않는다.
         (랜드마크를 여러 카메라가 볼 수 있다고 가정한다 — 유일성 가정 없음.)
      2. 남은 후보에서 relation 기하 적합도가 가장 높은 (target, anchor) 쌍을
         고른다. 탐지 confidence 는 동점 처리에만 쓴다 (1번 주석 참조).
      3. target 을 못 찾으면 pose 조건을 풀어 재탐색하고, 그래도 없으면
         랜드마크만으로라도 제출한다 — 빈 제출은 0점이 확정이지만 랜드마크만
         맞아도 camera/landmark/anchor 배점이 살아있다.

    ground_surface_token: 사람이 랜드마크 위가 아닐 때 situation 규칙에 넣을
    지면 토큰(grass/road/...). 판정 수단이 없으면 None → 공식 fallback 경로.
    """
    per_cam = {}
    for cam_id, dets in detections.items():
        cam = cameras.get(cam_id)
        if cam is None:
            continue
        per_cam[cam_id] = (
            cam, dets,
            [d for d in dets if _landmark_ok(d.label, parsed.landmark_type)],
        )
    if not per_cam:
        return None

    # -- 1) 카메라 게이트: 명시 랜드마크 → area_hint → 해제 --
    # 사람 문제는 landmark 배점이 0 점이지만, 카메라를 좁혀 주는 단서는 사실상
    # 랜드마크뿐이다 (그 0 점짜리 랜드마크가 camera/anchor/target 배점을 연다).
    # 그래서 배점과 무관하게 동일하게 건다.
    gated = ([cid for cid, (_, _, adets) in per_cam.items() if adets]
             if parsed.landmark_type is not None else [])
    gate_kind = "landmark" if gated else None

    if not gated:
        # 명령이 랜드마크 없이 구역만 말한 경우 — 그 구역의 지표 클래스가
        # 보이는 카메라로 좁힌다 (AREA_INDICATOR_CLASSES 주석 참조).
        indicators = AREA_INDICATOR_CLASSES.get(parsed.area_hint or "")
        if indicators:
            gated = [cid for cid, (_, dets, _) in per_cam.items()
                     if any(d.label in indicators for d in dets)]
            if gated:
                gate_kind = "area"

    gate_applied = bool(gated)
    if not gated:
        gated = list(per_cam)          # 단서 미지목/미검출 → 게이트 해제

    # -- 2) relation 적합도로 (카메라, target, anchor) 선택 --
    def search(target_ok):
        # 후보 = (관계적합도_주, 관계적합도_부, 탐지conf, cam_id, t_loc, a_loc).
        # 앞 3개까지만 비교한다 — conf 는 관계 적합도가 동점일 때의 최후 수단.
        best = None
        for cam_id in gated:
            cam, dets, adets = per_cam[cam_id]
            for tdet in (d for d in dets if target_ok(d)):
                t0 = locate(cam, tdet)
                if t0 is None:
                    continue
                paired = False
                for adet in adets:
                    if adet is tdet:
                        continue
                    a_loc = locate(cam, adet)
                    if a_loc is None:
                        continue
                    paired = True
                    t_loc = _support_adjusted(cam, tdet, t0, a_loc, cfg)
                    fit = _relation_fit(cam, a_loc, t_loc, cfg,
                                        parsed.relation_hint)
                    cand = (fit[0], fit[1], tdet.score, cam_id, t_loc, a_loc)
                    if best is None or cand[:3] > best[:3]:
                        best = cand
                if not paired:
                    # 랜드마크가 없는 카메라 (게이트가 풀렸을 때만 여기 온다).
                    # 비교 근거가 confidence 뿐이라 적합도 0 으로 최하위에 둔다.
                    cand = (0.0, 0.0, tdet.score, cam_id, t0, None)
                    if best is None or cand[:3] > best[:3]:
                        best = cand
        return best

    best = search(lambda d: _target_ok(d.label, parsed))
    relaxed = False
    if best is None and parsed.is_person and parsed.pose_hint:
        # 완화 재탐색: pose 버킷 제약을 풀고 사람 클래스 전체를 후보로 본다.
        best = search(lambda d: d.label in PERSON_POSE_CLASSES
                      or d.label == "person")
        relaxed = best is not None

    gate_info = {"gate_applied": gate_applied, "gate_kind": gate_kind,
                 "gate_cameras": sorted(gated)}
    if best is not None:
        fit_a, fit_b, _, cam_id, t_loc, a_loc = best
        return _build_decision(
            parsed, per_cam[cam_id][0], cam_id, t_loc, a_loc, cfg,
            ground_surface_token,
            dict(gate_info, selection="pair", relation_fit=(fit_a, fit_b),
                 pose_relaxed=relaxed))

    # -- 3) target 미탐지 → 랜드마크만으로 제출 --
    anchor_best = None                 # (det_score, cam_id, a_loc)
    for cam_id in gated:
        cam, _, adets = per_cam[cam_id]
        for adet in adets:
            a_loc = locate(cam, adet)
            if a_loc is None:
                continue
            if anchor_best is None or adet.score > anchor_best[0]:
                anchor_best = (adet.score, cam_id, a_loc)
    if anchor_best is None:
        return None                    # 타겟도 랜드마크도 없음 → 상위 폴백
    _, cam_id, a_loc = anchor_best
    return _build_decision(
        parsed, per_cam[cam_id][0], cam_id, None, a_loc, cfg,
        ground_surface_token,
        dict(gate_info, selection="anchor_only",
             # 게이트가 카메라를 1대로 좁혔다 = 이 씬에서 그 랜드마크 클래스가
             # 한 카메라에만 보인다 = 변별력이 높다 → 카메라를 신뢰할 수 있다.
             # 여러 대로 남았으면 추측이므로 낮은 신뢰로 표시만 해 둔다.
             landmark_discriminative=gate_applied and len(gated) == 1))
