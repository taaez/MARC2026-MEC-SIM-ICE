"""General geometric grounding for MARC 2026 Stage 1/2.

parse_command() 결과 + (임의 탐지기의) 카메라별 탐지 결과 + 카메라 파라미터
  → 카메라 선택, 랜드마크 인스턴스 확정, relation 기하 판정, 좌표 계산
  → GroundingResult payload

GENERAL by design — 대회 요구사항("문제·배경이 바뀐다, 씬 하드코딩 금지")에 맞춘
원칙:
  * 특정 시나리오/문항/좌표/씬 배치를 참조하지 않는다.
  * 클래스별 높이·크기 상수를 두지 않는다 — 물체/랜드마크의 높이는 그 탐지
    bbox 와 카메라 기하에서 그 자리에서 추정한다 (estimate_top_z).
  * near/beside 는 중심거리가 아니라 **XY 평면 OBB 표면거리**로 가른다
    (obb_gap_xy). 크기가 제각각인 랜드마크에서 중심거리는 관계 순서를 뒤집기
    때문 — 근거와 임계값(0.58m) 유도는 리포 루트의
    MARC2026_Chungmu_near_beside_OBB_analysis.md.
  * 어휘(클래스 목록)만 공식 로스터에 닫혀 있고, 판정 로직은 전부 기하다.
  * 임계값은 GroundingConfig 로 외부화 — 연습 채점 역산으로 보정하되 코드는 불변.

좌표 규약 (marc_sdk / API 레퍼런스):
  world: REP-103 (X앞 Y왼쪽 Z위), 제출 좌표는 world.
  camera: ROS optical (+Z 전방, +X 오른쪽, +Y 아래) — CameraInfo.K 와 동일 규약.
  /tf_static (world→camera_id): p_world = R @ p_cam + t.

제출 좌표 정의: target_coord/anchor_coord 둘 다 **접촉면 위의 점** — 바닥
접촉점 (x, y) 에 z 는 그 물체가 놓인 면의 높이(지면, 또는 'on' 이면 랜드마크
상면). 시나리오 yaml 실측 근거는 Located.ground_point 의 docstring 참조.

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
# 구역을 정의하는 지표 클래스는 탐지된다 — "주차 구역" 은 곧 "차량이 밀집한
# 곳" 이다. 이 매핑은 씬 배치가 아니라 **단어의 일반적 의미**에서 나온 것이라
# 배경이 바뀌어도 성립한다 (특정 시나리오/좌표를 참조하지 않는다).
#
# "있다" 가 아니라 "가장 많다" 로 판정하는 것이 중요하다: 실측에서 차량은 6대
# 중 4대 카메라에 흩어져 잡혔고(길가에 세워진 차 한 대는 주차장이 아니다),
# 단순 존재로 게이트를 걸면 후보가 안 좁혀져 오히려 오답을 골랐다. 지표 최다
# 카메라로 좁히면 연습 8문항 전부 정답 카메라가 유일하게 남는다.
#
# 안전성: 지표가 한 카메라에서도 안 잡히면 게이트가 그냥 풀려서 이 계층이 없을
# 때와 똑같이 동작한다 (순수 폴백 — 더 맞힐 수는 있어도 나빠지지는 않는다).
# "road" 는 연습 코퍼스에 근거 사례가 없어 일부러 비워 뒀다: 근거 없이 매핑을
# 만들면 멀쩡한 카메라를 걸러낼 위험만 생긴다.
AREA_INDICATOR_CLASSES = {
    "parking": CAR_CLASSES,
    "park": {"picnic_table", "bench"},
}

# "그 구역"으로 인정할 지표 밀집도 — 1등 카메라의 지표 수가 2등의 이 배 이상.
# 길가에 세워진 차 한 대는 주차장이 아니고, 4대 대 3대로 갈리는 두 곳 중 어느
# 쪽도 주차장이라 단정할 수 없다. 애매하면 게이트를 걸지 않는다(폴백).
AREA_DOMINANCE_RATIO = 2.0

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
    # 사람 탐지에 한해 pose 모델이 찍어 준 '발목 가로 픽셀' (pose_keypoints.py).
    # 플랫폼의 정답이 사람의 중심이 아니라 발이라, 제출 target 좌표의 가로
    # 위치를 이걸로 옮긴다 (_foot_adjusted_point). 선택 기하(locate)는 일부러
    # 건드리지 않는다 -- 이유는 그 함수의 docstring. None 이면 기존 동작 그대로.
    foot_u: Optional[float] = None


@dataclass
class Located:
    """한 탐지의 world 기하 요약."""
    det: Detection
    base: np.ndarray                 # 바닥(또는 지지면) 접촉점 world (x,y,z)
    top_z: float                     # 꼭대기 z (기하 추정)
    radius: float                    # 수평 footprint 반경 추정 (m) = OBB 가로 반길이
    view_axis: Optional[np.ndarray] = None   # XY OBB 의 '깊이' 축 (카메라→물체 시선의 지면 투영, 단위벡터)
    half_depth: float = 0.0          # 그 깊이축 방향 반길이 (m). 0 이면 radius 로 대체

    @property
    def footprint_xy(self) -> np.ndarray:
        """지면(XY) footprint 를 **직사각형(OBB)** 으로 본 꼭짓점 4개, (4,2).

        near/beside 는 지면 위의 평면 관계라, 3D OBB 를 XY 로 투영한 사각형
        사이의 거리로 판정한다 (MARC2026_Chungmu_near_beside_OBB_analysis.md
        6.3 절). 축은 다음과 같이 잡는다 -- 단안 CCTV 에서 실제로 관측 가능한
        두 방향이 그것뿐이다:

          가로(v) : 시선에 수직. bbox 좌/우 아랫점을 지면에 투영한 폭의 절반이
                    그대로 반길이가 된다 (``radius``) -- 직접 측정값이다.
          깊이(u) : 시선 방향. 단안 bbox 로는 깊이를 볼 수 없어 ``locate`` 가
                    이미 쓰고 있는 '정사각형 footprint 가정'을 그대로 따른다
                    (반길이 = ``push * radius``). 이 값은 접촉선(bbox 아랫변
                    역투영)에서 중심까지의 거리와 **같게** 잡혀 있어서, OBB 의
                    카메라 쪽 면이 관측된 접촉선에 정확히 놓인다.

        원(반경 radius) 대신 사각형을 쓰는 이유: 원은 어느 방향으로든 반경이
        radius 라 모서리 쪽으로 놓인 이웃과의 간격을 최대 41%(√2-1) 과대평가
        한다. 공식 OBB 로 잰 chungmu 값(자판기-우산 0.47m = beside)이
        우리 원 근사에서는 0.6m 를 넘어 near 로 뒤집혔던 원인이 이것이다.
        """
        c = np.asarray(self.base[:2], dtype=np.float64)
        u = self.view_axis
        if u is None or float(np.linalg.norm(np.asarray(u))) < 1e-9:
            u = np.array([1.0, 0.0])          # 축 미상: 축 정렬 정사각형으로 퇴화
        else:
            u = np.asarray(u, dtype=np.float64)
            u = u / float(np.linalg.norm(u))
        v = np.array([-u[1], u[0]])
        hu = float(self.half_depth) if self.half_depth > 1e-9 else float(self.radius)
        hv = float(self.radius)
        return np.array([c + hu * u + hv * v, c + hu * u - hv * v,
                         c - hu * u - hv * v, c - hu * u + hv * v])

    @property
    def center(self) -> list:
        """중심점 = 접촉점 (x,y) + z 는 (base+top)/2.

        **제출 좌표에는 쓰지 않는다** (ground_point 의 docstring 참조 -- GT 는
        중심이 아니라 접촉면이다). 높이를 쓰는 판정용으로만 남겨 둔다.
        """
        return [float(self.base[0]), float(self.base[1]),
                float((self.base[2] + self.top_z) / 2.0)]

    @property
    def ground_point(self) -> list:
        """접촉점 (x, y, base_z). anchor/target 제출 좌표 **양쪽 모두**에 쓴다.

        GT 는 물체가 놓인 **면의 높이**다 (중심점이 아니다). 근거는 시나리오
        yaml(`content-image-2026.R03/marc2026_chungmu.yaml`) 실측 -- 채점
        정답지가 그 파일의 `resolved.position` 을 그대로 쓰므로(validation/
        chungmu_gt.py) 이것이 GT 정의 그 자체다. 지면 대비 prim z 중앙값:

            person_prone   +0.006   person_lying_down +0.025
            person_bending +0.097   person_walking    -0.125
            mug            +0.848   (= 피크닉 테이블 상판)
            person_sitting +1.537   (= 자판기 위)

        1.7m 인체의 GT 가 지면 ±0.1m 인 것이 결정적이다 -- 중심점 규약이면
        +0.85 가 나와야 한다. 'on' 인 경우도 상판 높이지 상판+물체높이/2 가
        아니다(mug +0.848).

        이력: 랜드마크는 큰 물체라 center 의 z 오차가 바로 드러나 먼저 고쳤고
        (clean 12라운드 스윕, 평균 오차 1.00m -> 0.64m), 그때 "target(작은
        물체)은 center 가 맞다"고 판단해 남겨 뒀다. 그 스윕이 **캔·티슈 같은
        소형 물체만** 봤기 때문에 생긴 오판이다 -- 높이 0.1m 물체에서는 두 규약
        차이가 5cm 라 구분되지 않는다. 사람 문항(1.7m)에서만 드러났다: target
        오차 중앙값이 물체 0.47m 대 사람 1.10m 로 갈렸고, sqrt(0.47^2+0.85^2)
        = 0.97m 로 '중심점 규약이 z 에 +0.85 를 더한다'는 설명과 정합한다.
        """
        return [float(self.base[0]), float(self.base[1]), float(self.base[2])]


def locate(cam: CameraModel, det: Detection,
           base_z: Optional[float] = None, push: float = 1.0) -> Optional[Located]:
    """탐지 bbox → Located. base_z 지정 시 그 평면(예: 랜드마크 상면) 기준.

    bbox 하단 중앙의 역투영은 물체의 '카메라 쪽 가장자리' 접촉점이므로,
    footprint 반경만큼 카메라 반대 방향으로 밀어 footprint **중심**을 얻는다
    (넓은 물체에서 중요 — 벤치·테이블은 반경이 수십 cm). 상단 z 추정은 먼 쪽
    가장자리에서 매칭한다 (estimate_top_z 참조).

    ``push`` 는 그 보정량의 배수다 (``GroundingConfig.footprint_push``). 반경을
    bbox '가로 폭'에서 얻어 '깊이 방향'으로 미는 것은 footprint 가 정사각형이라는
    가정이라, 실제 물체 형상에 따라 과/소보정이 난다 — 그 계통 편향을 흡수하는
    보정 계수다. 1.0 = 정사각형 가정 그대로.
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
    center_xy = near[:2] + push * radius * away
    base = np.array([center_xy[0], center_xy[1], float(near[2])])
    far_xy = near[:2] + 2.0 * push * radius * away
    top_z = max(estimate_top_z(cam, det.bbox, far_xy, float(base[2])),
                float(base[2]))
    # XY OBB: 깊이축 = 시선(away), 깊이 반길이 = 중심을 민 그 거리와 동일하게
    # 두어 상자의 카메라 쪽 면이 관측된 접촉선에 놓이게 한다 (footprint_xy 주석).
    return Located(det=det, base=base, top_z=top_z, radius=radius,
                   view_axis=away, half_depth=float(push * radius))


# ────────────────────────────────────────────────────────────────────────────
# XY 평면 OBB 표면거리
#
# near/beside 는 지면 위의 평면 관계이므로, 두 물체의 footprint 를 XY 평면의
# 볼록다각형(여기서는 OBB 를 XY 로 투영한 직사각형)으로 보고 **표면 사이의 최소
# 거리**를 잰다. 중심 거리가 아니라 표면 거리인 것이 핵심이다 -- 랜드마크 크기가
# 제각각이라(자판기 2.0m vs 쓰레기통 0.7m) 중심거리는 관계 순서를 뒤집는다.
# 실측(MARC2026_Chungmu_near_beside_OBB_analysis.md 2.5 절): 정답이 beside 인
# 자판기-우산의 원점거리(1.85m)가 정답이 near 인 쓰레기통-우산(1.15m)보다 오히려
# 멀었고, OBB 표면거리로 바꾸면 0.47m < 0.70m 로 순서가 바로잡힌다.
# ────────────────────────────────────────────────────────────────────────────
def _separated(A: np.ndarray, B: np.ndarray) -> bool:
    """두 볼록다각형이 분리축을 갖는가(=겹치지 않는가). SAT — 볼록에서 정확하다."""
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


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-18 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_gap_xy(A: np.ndarray, B: np.ndarray) -> float:
    """XY 평면 볼록다각형 A, B 표면 사이의 최소 거리. 겹치거나 닿으면 0.

        d(A,B) = min_{a in A, b in B} ||a - b||

    떨어져 있는 두 볼록다각형에서 최소거리는 항상 '한쪽 꼭짓점 ↔ 다른 쪽 변'
    에서 실현되므로 그 조합만 훑으면 충분하다 (변-변 조합을 다 볼 필요가 없다).
    """
    if not _separated(A, B):
        return 0.0
    best = float("inf")
    for P, Q in ((A, B), (B, A)):
        for q in Q:
            for i in range(len(P)):
                d = _point_seg_dist(q, P[i], P[(i + 1) % len(P)])
                if d < best:
                    best = d
    return best


def obb_gap_xy(a: Located, b: Located) -> float:
    """두 탐지의 XY OBB footprint 표면거리 (m). 겹치면 0."""
    return polygon_gap_xy(a.footprint_xy, b.footprint_xy)


# ────────────────────────────────────────────────────────────────────────────
# relation 기하 판정
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class GroundingConfig:
    """임계값 외부화 — 연습 채점 역산으로 보정하는 대상 (코드 로직은 불변)."""
    # XY OBB 표면거리가 이 값 이내면 beside, 넘으면 near.
    # 0.58m 는 공식 R03 에셋으로 재구성한 chungmu 경계 두 쌍의 중간값이다
    # (MARC2026_Chungmu_near_beside_OBB_analysis.md 11 절):
    #   vending_machine-umbrella  0.470m  정답 beside   ← beside 최대
    #   trash_can-umbrella        0.697m  정답 near     ← (의미 예외 제외) near 최소
    #   (0.470 + 0.697) / 2 = 0.583 → 0.58
    # 회전 쿼터니언 해석을 바꿔 계산해도(0.489 / 0.656, 중간값 0.573) 0.58 은
    # 같은 경계 판정을 낸다. 주최측 공표값이 아니라 chungmu 전용 경험값이다.
    #
    # 다만 **우리가 실제로 재는 거리는 공식 OBB 가 아니라 단안 추정치**라,
    # 같은 쌍이 더 크게 나온다 (자판기-우산: 공식 0.470m vs 우리 0.61m).
    # 그래서 문서 값 0.58 을 그대로 쓰면 우리 척도에서는 beside 쌍이 near 로
    # 넘어간다. 임계값은 문서의 절대값이 아니라 **우리 척도의 경계**여야 하므로
    # 0.63 으로 올려 둔다 (2026-08-30 재보정).
    beside_gap_m: float = 0.63       # XY OBB 표면거리가 이내면 beside
    # 임계값 주변의 불확정 밴드 (문서 12.3 절). 이 안에서는 기하가 결론을 낼
    # 만큼 정확하지 않다고 보고 텍스트 힌트에 맡긴다.
    # **0 으로 둔다.** 밴드를 도입한 이유는 임계값(문서의 0.58)이 우리 척도와
    # 다른 척도의 값이라 경계가 미더웠기 때문인데, 위에서 임계값을 우리 척도로
    # 재보정하면서 그 이유가 사라졌다. 0.05 를 유지하면 밴드가 [0.58, 0.68] 이
    # 되어 우리가 맞히려던 자판기-우산(0.61m)을 그대로 삼키고, 판정을 다시
    # 텍스트 힌트에 넘긴다 -- 그 힌트("next to")는 문서 10.1 절이 near/beside
    # 어느 쪽으로도 매핑하지 말라고 못 박은 표현이라 근거가 없다.
    beside_band_m: float = 0.0
    near_gap_m: float = 3.0          # 이내면 near
    on_xy_slack: float = 0.6         # 'on' 판정: 수평 중심이 anchor 반경*(1+slack) 안
    on_z_tol_m: float = 0.45         # 'on' 판정: target 바닥과 anchor 상면 z 차 허용
    above_clearance_m: float = 0.15  # anchor 상면보다 이만큼 높이 떠 있으면 above
    directional_min_gap_m: float = 0.15  # 방향 relation 을 고려할 최소 수평 거리
    footprint_push: float = 1.0      # locate(): 접촉점 → 중심 보정 계수 (radius 배수)


def classify_relation(cam: CameraModel, anchor: Located, target: Located,
                      cfg: GroundingConfig = GroundingConfig(),
                      hint: Optional[str] = None) -> str:
    """anchor(랜드마크) 대비 target 의 공간 관계를 기하로 판정.

    beside/near 는 **XY 평면 OBB 표면거리**로 가른다 (``obb_gap_xy``,
    임계값 ``cfg.beside_gap_m``). 중심거리도, 원 근사 간격도 아니다 -- 근거는
    ``polygon_gap_xy`` 위의 주석과 ``GroundingConfig.beside_gap_m`` 참조.

    기본은 근접 계열(on/above/beside/near)로만 판정한다. 방향(behind/front/
    left/right)은 텍스트가 방향 relation 을 힌트로 줄 때만 기하로 확인해
    반환한다 — 방향은 명령이 말해줄 때만 의미가 있고, 기하만으로 near 대신
    방향을 고르는 것은 근거가 없기 때문.

    방향의 기준 시점은 카메라 광축의 지면 투영으로 가정한다(채점 정의
    미공개 — 실측 검증 항목). 'front' = target 이 anchor 보다 카메라에 가까움.
    """
    rel_xy = target.base[:2] - anchor.base[:2]
    dist = float(np.linalg.norm(rel_xy))
    gap = obb_gap_xy(anchor, target)                  # XY OBB 표면거리

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

    if gap <= cfg.beside_gap_m - cfg.beside_band_m:
        return "beside"
    if gap >= cfg.beside_gap_m + cfg.beside_band_m:
        return "near"    # near_gap 밖이어도 근접 계열에선 near 가 가장 보수적
    # 경계 밴드 안: 기하만으로는 못 가른다. 텍스트가 near/beside 계열을 말했다면
    # (확정적이지 않더라도) 그 힌트를 쓴다 -- 확정적이면 애초에 여기까지 오지도
    # 않는다(_build_decision 이 텍스트를 그대로 낸다). 힌트가 없으면 임계값으로.
    if hint in ("near", "beside"):
        return hint
    return "beside" if gap <= cfg.beside_gap_m else "near"


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
    re_t = locate(cam, tdet, base_z=a_loc.top_z, push=cfg.footprint_push)
    if re_t is None:
        return t_loc
    d_ground = float(np.linalg.norm(t_loc.base[:2] - a_loc.base[:2]))
    d_top = float(np.linalg.norm(re_t.base[:2] - a_loc.base[:2]))
    if d_top < d_ground and d_top <= a_loc.radius * (1.0 + cfg.on_xy_slack):
        return re_t
    return t_loc


def _open_space_pick(gated, per_cam, a_locs, target_ok,
                     cfg: GroundingConfig) -> Optional[tuple]:
    """"빈 자리/열린 바닥" 명령 전용 선택. 적용 못 하면 None (→ 일반 경로 폴백).

    target 선택과 landmark 선택을 **분리**한다:
      - target 은 '여유(clearance) = 최근접 랜드마크까지 거리' 가 가장 큰 것.
        "비어 있다"는 말 자체가 '무언가 옆에 붙어 있지 않다'는 뜻이라, 평소의
        근접 선호를 그대로 쓰면 명령과 정반대인 후보(차 옆에 붙은 디코이)를
        고르게 된다.
      - landmark 는 그렇게 고른 target 의 **최근접** 객체. 여기서까지 '먼 것'을
        고르면 앵커 좌표가 엉뚱해진다 (실측으로 확인: 쌍으로 묶어 고르면
        anchor/landmark 점수가 같이 떨어졌다).

    **절대 거리 임계값을 쓰지 않는다.** "비어 있다"가 몇 미터를 뜻하는지는
    공간 규모에 따라 다르고(넓은 광장 vs 좁은 통로), 연습 씬에서 잰 값을
    기준으로 삼으면 배경이 바뀔 때 그대로 깨진다. 여기서 하는 판정은 후보들
    **사이의 상대 비교**뿐이다 -- "어느 쪽이 더 트여 있는가". 그래서 규모가
    달라져도 성립한다.

    적용 조건: 후보가 2개 이상일 때만. 하나뿐이면 고를 것이 없어 일반 경로와
    결과가 같으므로, 굳이 다른 경로를 타지 않는다.
    """
    cands = []
    for cam_id in gated:
        cam, dets, _adets = per_cam[cam_id]
        for tdet in (d for d in dets if target_ok(d)):
            t0 = locate(cam, tdet, push=cfg.footprint_push)
            if t0 is None:
                continue
            nearest = None
            for adet, a_loc in a_locs[cam_id]:
                if adet is tdet:
                    continue
                # 여기(후보 랭킹)는 일부러 등방(원) 근사를 유지한다 -- 이유는
                # _relation_fit 의 주석 참조. near/beside 판정만 OBB 로 간다.
                gap = (float(np.linalg.norm(t0.base[:2] - a_loc.base[:2]))
                       - a_loc.radius - t0.radius)
                if nearest is None or gap < nearest[0]:
                    nearest = (gap, a_loc)
            if nearest is None:
                continue                      # 랜드마크가 없는 카메라
            # 겹침(음수)은 0 으로만 자른다 — 상한을 두면 그 값이 곧 절대 임계가 된다.
            clearance = max(nearest[0], 0.0)
            cands.append((clearance, tdet.score, cam_id, t0, nearest[1]))

    if len(cands) < 2:
        return None
    cands.sort(key=lambda c: (-c[0], -c[1]))
    clearance, score, cam_id, t_loc, a_loc = cands[0]
    return (clearance, 0.0, score, cam_id, t_loc, a_loc)


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
    # **여기는 OBB 가 아니라 등방(원) 근사를 쓴다.** near/beside 판정
    # (classify_relation)과 달리 이 함수는 '어느 후보 쌍이 명령대로 놓였나'를
    # 재는 상대 랭킹이고, 그 대상에는 사람도 들어간다. 누운 사람의 footprint 는
    # 정사각형이 아닌데 OBB 는 bbox 가로폭을 깊이에도 그대로 쓰므로(단안에서
    # 깊이는 관측 불가) 사람에서 상자가 시선 방향으로 과하게 부푼다. 실측:
    # 이 함수까지 OBB 로 바꿨더니 sar08 이 엉뚱한 카메라를 골라 93.90 -> 0.00
    # 이 됐다. 판정 척도(문서가 지정한 OBB)와 랭킹 척도를 분리해 둔다.
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


def _foot_adjusted_point(cam: CameraModel, t_loc: Located,
                         cfg: GroundingConfig) -> list:
    """사람 target 의 **제출 좌표만** 발목 가로 위치로 옮긴다.

    플랫폼의 정답은 사람의 중심이 아니라 발이라, bbox 가로 중앙을 쓰면 누운
    사람에서 몸길이의 절반만큼 어긋난다 (근거·실측은 pose_keypoints.py
    docstring). 발목 keypoint 가 그 가로 위치를 준다.

    **선택 기하(``Located.base``)는 일부러 건드리지 않는다.** base 는 제출
    좌표이자 relation 판정·anchor 선택의 입력이기도 해서, locate() 안에서
    같이 옮기면 (target, anchor) 쌍 선택이 뒤집힌다 -- 실측에서 sar07/sarH6
    의 anchor 가 0.95 -> 0.04 로 무너져 target 이득(+0.07)을 넘는 손해가
    났다. 그래서 보정은 payload 를 만드는 이 지점에서만 적용한다.

    세로는 접지선(bbox 아랫변)을 그대로 쓴다. 얕게 내려다보는 카메라에서
    세로 픽셀 오차는 world 로 크게 증폭되므로(rig_7_a 실측 1px = 0.075m)
    발목의 세로까지 쓰면 오히려 나빠진다.
    """
    u = t_loc.det.foot_u
    if u is None:
        return t_loc.ground_point
    _, _, _, y2 = t_loc.det.bbox
    z = float(t_loc.base[2])
    near = pixel_to_plane(cam, float(u), y2, plane_z=z)
    if near is None:
        return t_loc.ground_point
    away = near[:2] - cam.C_w[:2]
    n = float(np.linalg.norm(away))
    away = away / n if n > 1e-9 else np.zeros(2)
    xy = near[:2] + cfg.footprint_push * t_loc.radius * away
    return [float(xy[0]), float(xy[1]), z]


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
        anchor_coord = a_loc.ground_point
        if t_loc is not None:
            geo_rel = classify_relation(cam, a_loc, t_loc, cfg,
                                        hint=parsed.relation_hint)

    # target 미탐지 시 anchor 좌표를 target 좌표로 대신 낸다: 허용 반경(5m)에
    # 비해 실측 anchor↔target 거리가 훨씬 짧아 부분점수를 크게 건진다.
    # anchor 와 같은 ground_point 규약을 쓴다 (Located.ground_point docstring).
    target_coord = (_foot_adjusted_point(cam, t_loc, cfg) if t_loc is not None
                    else list(anchor_coord))

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
        # 우선순위 (MARC2026_Chungmu_near_beside_OBB_analysis.md 12.1 절):
        #   1. 텍스트가 관계어를 확정적으로 말했으면 그대로
        #   2. 명령이 '빈 자리/열린 바닥'이라는 **영역**을 가리키면 near
        #   3. 그 외에는 기하 (XY OBB 표면거리)
        if parsed.relation_confident and parsed.relation_hint:
            relation = parsed.relation_hint
        elif parsed.open_space and geo_rel not in ("on", "above"):
            # 영역 지시문("in the empty parking space")은 물체가 랜드마크 옆면에
            # 붙어 있다고 말하는 것이 아니라, 랜드마크를 기준점 삼아 그 **주변
            # 공간**을 가리킨다. 거리만으로는 못 가르는 의미 예외다 -- 실측에서
            # police_car-umbrella 는 공식 OBB 표면거리가 0m 인데도 정답이 near
            # 라, 어떤 양의 임계값을 써도 거리 규칙만으로는 beside 로 뒤집힌다
            # (같은 문서 8.1/10.2 절). open_space 는 이미 파서가 뽑아 두는
            # 일반 신호이고(command_parser.OPEN_SPACE_PAT), 씬을 참조하지 않는다.
            # 'on'/'above' 는 수직 관계라 이 예외의 대상이 아니다.
            relation = "near"
        else:
            relation = geo_rel or parsed.relation_hint or "near"

    debug = {"geo_relation": geo_rel,
             "obb_gap_xy": (obb_gap_xy(a_loc, t_loc)
                            if (a_loc is not None and t_loc is not None) else None),
             "open_space": parsed.open_space,
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
    지면 토큰(grass/road/...). 미지정이면 명령에서 파싱한 area_hint 를 쓴다.
    """
    # 명령이 지목한 구역을 지지면 토큰으로 쓴다 -- "parking lot 바닥에 쓰러진"
    # 처럼 사람이 랜드마크 위가 아닌 문항에서, 지금까지 support 가 None 이라
    # 공식 규칙표(situation_rules.yaml)를 못 타고 자세 기본값으로 떨어졌다.
    # 랜드마크 위(on/above)인 경우는 _build_decision 이 랜드마크를 우선하므로
    # 이 값이 끼어들지 않는다 -- 즉 순수 추가 계층이라 나빠지는 경로가 없다.
    # (실측 39문항: sar01/sarH4/sar06 의 situation 이 맞아지고 회귀 0건,
    #  Stage1 평균 68.54 -> 70.25.)
    if ground_surface_token is None:
        ground_surface_token = parsed.area_hint

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
    landmark_gated = ([cid for cid, (_, _, adets) in per_cam.items() if adets]
                      if parsed.landmark_type is not None else [])

    # 구역 게이트 — 그 구역의 지표 클래스가 **밀집한** 카메라
    # (AREA_INDICATOR_CLASSES 주석 참조: "있다" 가 아니라 "밀집해 있다" 가
    # 구역의 정의다).
    #
    # 밀집의 판정은 AREA_DOMINANCE_RATIO 로 명시한다: 1등이 2등의 그 배 이상일
    # 때만 "그 구역"으로 인정한다. 근거는 씬이 아니라 **말의 의미**다 -- 차가
    # 4대와 3대로 갈리는 두 곳 중 어느 쪽도 "주차장"이라 단정할 수 없고, 그런
    # 애매한 신호로 카메라를 걸러내면 새 배경에서 정답을 배제할 수 있다.
    # 애매하면 게이트를 걸지 않아 이 계층이 없을 때와 같아진다.
    area_gated = []
    indicators = AREA_INDICATOR_CLASSES.get(parsed.area_hint or "")
    if indicators:
        counts = {cid: sum(1 for d in dets if d.label in indicators)
                  for cid, (_, dets, _) in per_cam.items()}
        top = max(counts.values(), default=0)
        if top > 0:
            leaders = [cid for cid, n in counts.items() if n == top]
            runner_up = max((n for cid, n in counts.items() if cid not in leaders),
                            default=0)
            if top >= AREA_DOMINANCE_RATIO * runner_up:
                area_gated = leaders

    # 두 단서는 서로 독립이므로 **둘 다 있으면 교집합**을 쓴다. 랜드마크가
    # 여러 카메라에 보이는 경우("trash can" 은 흔하다) 구역이 그중 하나로
    # 좁혀 준다 -- 실측에서 카메라 오선택의 절반이 이 조합으로 풀렸다.
    # 교집합이 비면 랜드마크 게이트로 되돌아가므로 나빠지는 경로가 없다.
    if landmark_gated and area_gated:
        both = [cid for cid in landmark_gated if cid in area_gated]
        gated, gate_kind = ((both, "landmark+area") if both
                            else (landmark_gated, "landmark"))
    elif landmark_gated:
        gated, gate_kind = landmark_gated, "landmark"
    elif area_gated:
        gated, gate_kind = area_gated, "area"
    else:
        gated, gate_kind = [], None

    gate_applied = bool(gated)
    if not gated:
        gated = list(per_cam)          # 단서 미지목/미검출 → 게이트 해제

    # 랜드마크의 Located 는 target 과 무관하므로 카메라당 한 번만 계산한다
    # (아래 이중 루프 안에서 매 target 마다 다시 구하면 같은 값을 T배 반복한다).
    a_locs = {cid: [(adet, loc) for adet, loc in
                    ((a, locate(per_cam[cid][0], a, push=cfg.footprint_push))
                     for a in per_cam[cid][2])
                    if loc is not None]
              for cid in gated}

    # 랜드마크를 지목하지 않은 채 "빈 자리/열린 바닥"이라고 명시한 명령인가.
    # (랜드마크가 지목됐다면 관계가 이미 확정돼 있어 이 해석이 끼어들 자리가 없다.)
    open_space = parsed.landmark_type is None and parsed.open_space

    # -- 2) relation 적합도로 (카메라, target, anchor) 선택 --
    def search(target_ok):
        # 후보 = (관계적합도_주, 관계적합도_부, 탐지conf, cam_id, t_loc, a_loc).
        # 앞 3개까지만 비교한다 — conf 는 관계 적합도가 동점일 때의 최후 수단.
        if open_space:
            picked = _open_space_pick(gated, per_cam, a_locs, target_ok, cfg)
            if picked is not None:
                return picked
            # 후보가 하나거나 여유 차이가 애매하면 아래 일반 경로로 되돌아간다.

        best = None
        for cam_id in gated:
            cam, dets, _adets = per_cam[cam_id]
            for tdet in (d for d in dets if target_ok(d)):
                t0 = locate(cam, tdet, push=cfg.footprint_push)
                if t0 is None:
                    continue
                paired = False
                for adet, a_loc in a_locs[cam_id]:
                    if adet is tdet:
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
        for adet, a_loc in a_locs[cam_id]:
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
