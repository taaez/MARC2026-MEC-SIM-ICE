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

try:                              # package import (e.g. `from taejeong.yolo_adapter import ...`)
    from .grounding import Detection
except ImportError:               # run directly as a script from within taejeong/
    from grounding import Detection

_log = logging.getLogger(__name__)


def detections_from_result(result, min_conf: float = 0.25) -> list:
    """ultralytics Results(단일 이미지) → [Detection, ...]."""
    out = []
    names = result.names
    if result.boxes is None:
        return out
    for b in result.boxes:
        conf = float(b.conf[0])
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        out.append(Detection(label=names[int(b.cls[0])],
                             bbox=(x1, y1, x2, y2), score=conf))
    return out


def infer_cameras(model, images_bgr: dict, min_conf: float = 0.25,
                  imgsz: int = 1280) -> dict:
    """{cam_id: BGR ndarray} → {cam_id: [Detection, ...]}.

    ``model`` 은 plain ultralytics YOLO 또는 load_detector() 가 돌려준
    RoutedDetector. 라우팅 경로에서는 conf/imgsz 가 로드 시점에 이미
    고정돼 있으므로 min_conf/imgsz 인자는 쓰지 않는다 (소형 3클래스의
    confidence 는 B 와 스케일이 달라 min_conf 를 다시 걸면 소형 검출을
    부당하게 잘라낸다 — predict.py 도입부 'WHY ROUTING' 참조).
    """
    out = {}
    routed = hasattr(model, "small_to_det")     # RoutedDetector 덕타이핑
    for cam_id, img in images_bgr.items():
        if routed:
            out[cam_id] = detections_from_routed(model, img)
        else:
            results = model(img, imgsz=imgsz, verbose=False)
            out[cam_id] = detections_from_result(results[0], min_conf=min_conf)
    return out


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
    det = RoutedDetector(weights, small, conf=conf, imgsz=imgsz)
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

    real_agent_vla 의 _best_box() 류가 그대로 동작하도록, 라우팅 경로에서는
    RoutedResult 셸을 돌려준다. 라우팅의 conf 는 로드 시점 고정이라 여기
    conf 인자는 plain 경로에서만 쓰인다.
    """
    if hasattr(model, "small_to_det"):
        boxes, scores, classes = model.predict(img)
        return RoutedResult(boxes, scores, classes, _routed_names(model))
    return model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]


def detections_from_routed(model, img) -> list:
    """RoutedDetector 한 프레임 → [Detection, ...] (infer_cameras 라우팅 경로)."""
    names = _routed_names(model)
    boxes, scores, classes = model.predict(img)
    return [Detection(label=names[int(c)],
                      bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                      score=float(s))
            for b, s, c in zip(boxes, scores, classes)]


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
