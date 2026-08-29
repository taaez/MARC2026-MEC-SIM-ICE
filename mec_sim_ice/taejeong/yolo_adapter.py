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

try:                              # package import (e.g. `from taejeong.yolo_adapter import ...`)
    from .grounding import Detection
except ImportError:               # run directly as a script from within taejeong/
    from grounding import Detection


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
    """{cam_id: BGR ndarray} → {cam_id: [Detection, ...]}."""
    out = {}
    for cam_id, img in images_bgr.items():
        results = model(img, imgsz=imgsz, verbose=False)
        out[cam_id] = detections_from_result(results[0], min_conf=min_conf,
                                             min_conf_by_class=min_conf_by_class)
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
