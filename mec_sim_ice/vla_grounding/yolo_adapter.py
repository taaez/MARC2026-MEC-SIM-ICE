"""Ultralytics YOLO 출력 → grounding.Detection 어댑터.

경계 계약:
  * YOLO bbox: results[i].boxes.xyxy = 픽셀 단위 (x1,y1,x2,y2) — 우리
    Detection.bbox 와 동일 규약이라 변환 없이 통과.
  * 라벨: model.names[int(cls_id)] — 학습 data.yaml 의 names 순서
    (= test/det_classes.py 의 DET_CLASSES 순서) 가 그대로 문자열로 나온다.
    우리 파이프라인은 라벨 '문자열'만 보므로 id 순서엔 무관하지만,
    verify_model_names() 로 가중치가 정말 그 클래스로 학습됐는지 확인할 것.
  * 이미지: ultralytics 는 numpy 배열을 BGR 로 해석한다(cv2 규약).
    RGB 배열(PIL 등)을 넘길 땐 rgb[..., ::-1] 로 뒤집어서 줄 것.
"""

from __future__ import annotations

import logging
import os
import sys

try:                              # package import (e.g. `from vla_grounding.yolo_adapter import ...`)
    from .grounding import Detection
except ImportError:               # run directly as a script from within vla_grounding/
    from grounding import Detection

_log = logging.getLogger(__name__)


# 정상 문턱에서 사라진 ``on`` target 을 같은 detector 로 한 번 더 확인할 때의
# probe 문턱. 이 값은 곧바로 일반 탐지 문턱으로 쓰지 않는다. 명령이 ``on`` 을
# 말했고, 해당 landmark 가 이미 검출된 카메라에서, bbox 가 landmark 상면과
# 이미지상 겹치는 같은 클래스 후보에만 적용한다. 따라서 낮은 raw confidence
# 하나만으로 카메라/target 이 선택되는 경로는 없다.
ON_TARGET_PROBE_CONF = 0.001


# 물체가 작아 whole-frame confidence 가 구조적으로 낮게 나오는 클래스는
# 전역 문턱(0.25)보다 낮은 값을 쓴다. 실측(2026-08-29,
# validation/diag_small_object_conf.py, conf=0.001 raw 분포 39문항 전수):
# umbrella 는 GT 카메라 confidence 가 0.25 문턱을 사이에 두고 정확히
# 갈렸다 -- lf08(0.249, 문턱 미달 -> 25점) vs hf08(0.254)/rt02(0.258,
# 문턱 통과 -> 94점대). 같은 클래스, 같은 랜드마크(police_car), 거의 같은
# 장면인데 0.25라는 정확한 값 하나 차이로 점수가 25 대 94로 갈리는 건
# "문턱이 우연히 그 값 근처"라는 신호지 실제 신뢰도 차이가 아니다.
# tumbler/pencilcase 도 같은 부류(작은 물체)라 함께 낮춘다 -- 이 두 클래스는
# 실측에서 GT 카메라 conf 가 0.25 근처가 아니라 아예 0(총체적 미탐, lf02/
# lf03/lf07/hf02/rtC2) 이거나 이미 0.25를 넉넉히 넘는 경우(lf09/lf10/hf07/
# rt01/rt04, 0.77~0.82)뿐이라 0.20 으로 낮춘다고 새로 걸리는 오탐도, 새로
# 구제되는 문항도 없다 -- 위험 없이 일관성만 맞추는 값이다.
# 0.20 을 고른 이유: lf08 의 0.249 를 넉넉히 통과시키면서, 관측된 잡음
# (무관 카메라의 decoy, 대부분 <0.10, taxi/police_car 인접 카메라만 예외
# 적으로 0.2~0.3 대) 쪽으로 더 다가가진 않는 최소 하향폭.
SMALL_TARGET_MIN_CONF = {"tumbler": 0.20, "pencilcase": 0.20, "umbrella": 0.20}


def _yolo_conf_floor(min_conf: float, min_conf_by_class: dict) -> float:
    """YOLO 내부 컷으로 넘길 값 = 기본 문턱과 클래스별 문턱 중 가장 낮은 값.

    ultralytics 는 conf 를 안 넘기면 기본 0.25 로 먼저 잘라서, 그보다 낮은
    클래스별 문턱(SMALL_TARGET_MIN_CONF)까지 박스가 도달하지 못한다. 컷을
    내려도 클래스별 NMS 는 높은 점수 박스를 먼저 남기므로 기존 문턱 이상의
    박스 집합은 그대로이고, 늘어난 박스는 뒤에서 클래스별 문턱으로 다시 거른다.
    """
    return min([min_conf, *(min_conf_by_class or {}).values()])


def detections_from_result(result, min_conf: float = 0.25,
                           min_conf_by_class: dict = SMALL_TARGET_MIN_CONF) -> list:
    """ultralytics Results(단일 이미지) → [Detection, ...].

    min_conf_by_class 에 라벨이 있으면 그 값을, 없으면 min_conf 를 쓴다.
    """
    out = []
    names = result.names
    if result.boxes is None:
        return out
    for b in result.boxes:
        conf = float(b.conf[0])
        label = names[int(b.cls[0])]
        thr = (min_conf_by_class or {}).get(label, min_conf)
        if conf < thr:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        out.append(Detection(label=label, bbox=(x1, y1, x2, y2), score=conf))
    return out


def infer_cameras(model, images_bgr: dict, min_conf: float = 0.25,
                  imgsz: int = 1280,
                  min_conf_by_class: dict = SMALL_TARGET_MIN_CONF) -> dict:
    """{cam_id: BGR ndarray} → {cam_id: [Detection, ...]}.

    ``model`` 은 plain ultralytics YOLO 또는 load_detector() 가 돌려준
    RoutedDetector. 두 경로 모두 B 담당 클래스에 min_conf/min_conf_by_class
    를 적용한다. 라우팅 경로의 소형 3클래스는 서브모델 자기 문턱(small_conf)
    을 이미 통과했으므로 다시 걸지 않는다 (confidence 스케일이 B 와 달라
    문턱을 다시 걸면 소형 검출을 부당하게 잘라낸다 — predict.py 도입부
    'WHY ROUTING' 참조). 라우팅 경로의 imgsz 는 로드 시점에 고정돼 있다.
    """
    out = {}
    routed = hasattr(model, "small_to_det")     # RoutedDetector 덕타이핑
    for cam_id, img in images_bgr.items():
        if routed:
            out[cam_id] = detections_from_routed(model, img, min_conf=min_conf,
                                                 min_conf_by_class=min_conf_by_class)
        else:
            results = model(img, imgsz=imgsz,
                            conf=_yolo_conf_floor(min_conf, min_conf_by_class),
                            verbose=False)
            out[cam_id] = detections_from_result(results[0], min_conf=min_conf,
                                                 min_conf_by_class=min_conf_by_class)
    return out


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _looks_on_bbox(anchor: Detection, target: Detection) -> bool:
    """grounding._looks_on 과 같은 저비용 이미지 공간 사전판정."""
    ax1, ay1, ax2, ay2 = anchor.bbox
    tx1, ty1, tx2, ty2 = target.bbox
    overlap = min(ax2, tx2) - max(ax1, tx1)
    if overlap <= 0.3 * (tx2 - tx1):
        return False
    return ay1 - (ty2 - ty1) <= ty2 <= ay2


def on_target_probe_augment(parsed, model, images_bgr: dict, detections: dict,
                            imgsz: int = 1280,
                            probe_conf: float = ON_TARGET_PROBE_CONF) -> None:
    """정상 문턱에서 빠진 ``on`` target 을 문맥으로 제한해 보강한다.

    작은 물체는 whole-frame YOLO 에서 올바른 bbox 를 내고도 confidence 만 매우
    낮을 수 있다. 그렇다고 클래스별 전역 문턱을 0.001 로 낮추면 무관 카메라의
    오탐까지 모두 resolver 후보가 된다. 이 함수는 아래 조건을 전부 만족할 때만
    raw 후보를 ``detections`` 에 추가한다.

      * 물체 문항이고 텍스트 relation 이 ``on``
      * 명시 landmark 가 정상 문턱으로 이미 검출된 카메라
      * 그 카메라에는 정상 문턱 target 이 아직 없음
      * raw target bbox 의 아랫변이 landmark bbox 상면 영역과 겹침
      * 높은 confidence 의 다른 클래스 bbox 와 사실상 같은 박스가 아님

    routed detector 에서는 3종 소형 서브모델을 다시 돌리지 않고 ``general``
    YOLO 만 probe 한다. 이 함수가 다루는 target 클래스는 general 모델 소유이며,
    불필요한 타일 추론과 confidence scale 혼합을 피할 수 있다.
    """
    if (parsed.is_person or parsed.relation_hint != "on"
            or not parsed.target_type or not parsed.landmark_type):
        return

    # package/flat 양쪽 import 경로를 지원한다. 모듈 상단에서 가져오면
    # yolo_adapter 를 가볍게 import 하는 도구까지 command_parser 를 끌어온다.
    try:
        from .grounding import _landmark_ok, _target_ok
    except ImportError:
        from grounding import _landmark_ok, _target_ok

    general = model.general if hasattr(model, "general") else model
    general_imgsz = getattr(model, "imgsz", imgsz)
    general_iou = getattr(model, "iou", 0.7)
    general_device = getattr(model, "device", None)
    if hasattr(model, "owned"):
        names = getattr(general, "names", {})
        target_ids = {int(i) for i, name in names.items()
                      if name == parsed.target_type} if isinstance(names, dict) else {
                          i for i, name in enumerate(names) if name == parsed.target_type}
        if target_ids & set(model.owned):
            return                    # 이 클래스는 타일 소형모델 소유

    for cam_id, img in images_bgr.items():
        cam_dets = detections.setdefault(cam_id, [])
        anchors = [d for d in cam_dets
                   if _landmark_ok(d.label, parsed.landmark_type)]
        if not anchors or any(_target_ok(d.label, parsed) for d in cam_dets):
            continue

        result = general.predict(img, imgsz=general_imgsz, conf=probe_conf,
                                 iou=general_iou, device=general_device,
                                 verbose=False)[0]
        if result.boxes is None:
            continue
        rescued = []
        for b in result.boxes:
            label = result.names[int(b.cls[0])]
            if label != parsed.target_type:
                continue
            score = float(b.conf[0])
            if score < probe_conf:
                continue
            bbox = tuple(float(v) for v in b.xyxy[0])
            cand = Detection(label=label, bbox=bbox, score=score)
            if not any(_looks_on_bbox(anchor, cand) for anchor in anchors):
                continue
            # DINO fallback 의 decoy 방어와 같은 원칙: 일반 detector 가 이미
            # 다른 이름으로 확신하는 바로 그 소품이면 낮은 target 오탐을 버린다.
            if any(d.label != label and d.score >= 0.5
                   and _bbox_iou(bbox, d.bbox) >= 0.5 for d in cam_dets):
                continue
            rescued.append(cand)

        if rescued:
            cam_dets.extend(rescued)
            _log.info("[yolo-probe] rescued %d low-conf target %r on %s "
                      "(best=%.4f)", len(rescued), parsed.target_type, cam_id,
                      max(d.score for d in rescued))


# ── 소형객체 서브모델 라우팅 (training/small_object_detection/predict.py) ──
#
# 3클래스(sunblock/tissue/disposable_cup)는 원거리에서 8px 미만으로 찍혀
# B 단독 recall 이 0.42~0.50 에 그친다. 타일 확대 서브모델을 붙이면
# recall 0.58~0.71 (<8px 구간 최대 2.2배), 나머지 31클래스는 B 출력을
# 그대로 쓰므로 회귀가 구조적으로 불가능하다 (2026-08-30 dataset_v2 val
# 500장 실측 — evaluate.py 로 재확인 가능).
#
# 환경변수 MARC_DISABLE_SMALL_DET=1 이면 기본 경로가 있어도 B 단독으로
# 돌아간다 (pose_keypoints 의 MARC_DISABLE_POSE_KEYPOINTS 와 같은 관례).

_MEC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SMALL_DIR = os.path.join(_MEC_ROOT, "training", "small_object_detection")
DEFAULT_SMALL_WEIGHTS = os.path.join(
    _SMALL_DIR, "runs", "detect", "small_t640_r2", "weights", "best.pt")
DEFAULT_TILING = os.path.join(_MEC_ROOT, "datasets", "dataset_small_t640")


def small_det_disabled() -> bool:
    return os.environ.get("MARC_DISABLE_SMALL_DET", "").lower() in ("1", "true", "yes")


def load_detector(weights: str, conf: float = 0.25, imgsz: int = 1280,
                  small_weights: str = None, tiling: str = None,
                  small_conf: float = 0.25):
    """B 가중치 → plain YOLO 또는 (서브모델 산출물이 있으면) RoutedDetector.

    small_weights/tiling 을 지정하지 않으면 기본 경로가 둘 다 존재할 때만
    라우팅을 켠다. 없거나 MARC_DISABLE_SMALL_DET=1 이면 조용히 B 단독으로
    떨어진다 — 나빠지는 경로가 없다.
    """
    from ultralytics import YOLO

    small_weights = small_weights or DEFAULT_SMALL_WEIGHTS
    tiling = tiling or DEFAULT_TILING
    if (small_det_disabled() or not os.path.exists(small_weights)
            or not os.path.exists(os.path.join(tiling, "tiling.json"))):
        _log.info("[yolo] plain B (small-det %s)",
                  "disabled by env" if small_det_disabled() else "artifacts not found")
        return YOLO(weights)

    if _SMALL_DIR not in sys.path:            # predict.py 는 자기 폴더 기준 import
        sys.path.insert(0, _SMALL_DIR)
    from predict import RoutedDetector, TiledPredictor  # noqa: E402

    # batch=4: 8GB GPU 에서 DINO/pose 모델과 동시 상주하므로 기본 8 대신
    # 절반으로. 타일 6장/프레임이라 호출당 2배치 — 속도 손해는 미미하다.
    small = TiledPredictor(small_weights, tiling, conf=small_conf, batch=4)
    # B 의 내부 컷은 클래스별 문턱까지 내려 두고, 실제 판정은 detections_from_routed
    # (min_conf/min_conf_by_class) 와 predict_result(conf) 가 한다.
    det = RoutedDetector(weights, small,
                         conf=_yolo_conf_floor(conf, SMALL_TARGET_MIN_CONF),
                         imgsz=imgsz)
    _log.info("[yolo] routed B+small (small_conf=%.2f, %s)",
              small_conf, os.path.basename(os.path.dirname(
                  os.path.dirname(small_weights))))
    return det


class _ShimBox:
    """ultralytics Box 의 우리가 쓰는 면만 흉내낸다 (box.xyxy[0]/conf[0]/cls[0])."""
    __slots__ = ("xyxy", "conf", "cls")

    def __init__(self, box, score, cls_id):
        import numpy as np
        self.xyxy = np.asarray([box], dtype=float)
        self.conf = np.asarray([score], dtype=float)
        self.cls = np.asarray([cls_id], dtype=float)


class RoutedResult:
    """RoutedDetector 출력 → ultralytics Results 호환 셸 (.boxes / .names)."""

    def __init__(self, boxes, scores, classes, names: dict):
        self.names = names
        self.boxes = [_ShimBox(b, s, c) for b, s, c in zip(boxes, scores, classes)]


def _routed_names(model) -> dict:
    # RoutedDetector 의 클래스 id 는 DET_CLASSES 인덱스다 (predict.py 계약).
    if _SMALL_DIR not in sys.path:
        sys.path.insert(0, _SMALL_DIR)
    tr_dir = os.path.join(_MEC_ROOT, "training")
    if tr_dir not in sys.path:
        sys.path.insert(0, tr_dir)
    from det_classes import DET_CLASSES  # noqa: E402
    return dict(enumerate(DET_CLASSES))


def predict_result(model, img, imgsz: int = 1280, conf: float = 0.35):
    """단일 프레임 추론 → .boxes/.names 를 가진 결과 (plain/routed 공용).

    record_agent_vla 의 _best_box() 류가 그대로 동작하도록, 라우팅 경로에서는
    RoutedResult 셸을 돌려준다. 라우팅 경로에서도 B 담당 클래스는 conf 로
    자른다 (load_detector 가 B 내부 컷을 클래스별 문턱까지 내려 두었으므로).
    클래스별 문턱은 여기선 적용하지 않는다 -- 기록용 경로의 기존 동작 유지.
    """
    if hasattr(model, "small_to_det"):
        boxes, scores, classes = model.predict(img)
        keep = [i for i, (s, c) in enumerate(zip(scores, classes))
                if int(c) in model.owned or float(s) >= conf]
        return RoutedResult(boxes[keep], scores[keep], classes[keep],
                            _routed_names(model))
    return model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]


def detections_from_routed(model, img, min_conf: float = 0.25,
                           min_conf_by_class: dict = SMALL_TARGET_MIN_CONF) -> list:
    """RoutedDetector 한 프레임 → [Detection, ...] (infer_cameras 라우팅 경로).

    B 담당 클래스에는 detections_from_result 와 같은 문턱을 건다. 소형 모델
    담당 클래스(model.owned)는 서브모델 문턱을 이미 통과했으므로 그대로 둔다.
    """
    names = _routed_names(model)
    boxes, scores, classes = model.predict(img)
    out = []
    for b, s, c in zip(boxes, scores, classes):
        cls_id, score = int(c), float(s)
        label = names[cls_id]
        if cls_id not in model.owned:
            thr = (min_conf_by_class or {}).get(label, min_conf)
            if score < thr:
                continue
        out.append(Detection(label=label,
                             bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                             score=score))
    return out


def verify_model_names(model, expected_classes) -> list:
    """가중치의 클래스 목록이 기대 목록과 같은지 검사. 불일치 목록 반환(빈=OK)."""
    names = [model.names[i] for i in sorted(model.names)]
    problems = []
    if list(names) != list(expected_classes):
        missing = [c for c in expected_classes if c not in names]
        extra = [c for c in names if c not in expected_classes]
        if missing:
            problems.append(f"weights missing classes: {missing}")
        if extra:
            problems.append(f"weights extra classes: {extra}")
        if not missing and not extra:
            problems.append("same classes but DIFFERENT ORDER (ok for us — "
                            "labels are strings — but retrain scripts must match)")
    return problems
