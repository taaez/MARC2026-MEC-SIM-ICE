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

from grounding import Detection


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
    """{cam_id: BGR ndarray} → {cam_id: [Detection, ...]}."""
    out = {}
    for cam_id, img in images_bgr.items():
        results = model(img, imgsz=imgsz, verbose=False)
        out[cam_id] = detections_from_result(results[0], min_conf=min_conf)
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
