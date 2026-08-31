"""사람 target 의 '발 위치'를 keypoint 로 찍어 제출 좌표를 바로잡는다.

## 왜 필요한가 — 플랫폼의 정답이 사람의 '중심'이 아니라 '발'이다

시나리오 yaml 의 사람 인스턴스를 보면 누운·엎드린·웅크린 자세가 전부 X축
±90° 로 눕혀 놓은 것이다 (person_prone rot=[-94.5, ...], person_lying_down
rot=[93.3, ...], person_bending rot=[-90.6, ...]; 서 있는 person_walking 만
rot[0]≈0). 회전의 pivot 이 prim 원점이라, 눕히면 원점은 제자리에 남고 몸이
거기서 1.7m 뻗어 나간다. 그리고 그 원점이 캐릭터의 발이다.

실측으로도 확인된다 — 정답 좌표를 이미지로 되던져 bbox 안 상대위치(0=왼쪽
끝, 1=오른쪽 끝)를 재면 14문항 어느 것도 중앙(0.5) 근처가 아니다:

    person_prone      -0.08     person_lying_down  +1.03 / +0.18
    person_bending    -0.04 / -0.19 / -0.56        person_sitting +1.04

전부 한쪽 끝이고, 왼쪽이냐 오른쪽이냐는 그 사람이 어느 방향으로 누웠느냐에
달렸다. 우리 ``locate()`` 는 bbox 가로 중앙을 쓰므로 구조적으로 몸 중앙을
찍고, 누운 사람에서 몸길이의 절반(~0.85m)만큼 어긋난다.

## 어떻게 푸는가 — COCO 사전학습 pose 모델

발이 어느 쪽 끝인지는 bbox 만으로 알 수 없다. 대신 COCO 로 사전학습된 사람
keypoint 모델이 발목(15/16번)을 바로 준다. 우리가 라벨을 만들 필요가 없고
(데이터셋 생성기는 2D bbox 만 준다), 규정도 허용한다 — FAQ: "빌드 시점에는
인터넷으로 모델 가중치·의존성을 이미지에 baking 할 수 있습니다."

## 가로만 쓰고 세로는 건드리지 않는다 (중요)

발목 픽셀을 그대로 지면에 역투영하면 **오히려 나빠진다**. 얕게 내려다보는
카메라에서 세로 픽셀 오차가 world 로 크게 증폭되기 때문이다 (rig_7_a 실측:
세로 1px = 0.075m). 발목 관절은 실제 접지점보다 이미지에서 위에 있어서 그
몇십 px 이 2~3m 로 불어난다. 가로 방향에는 이 증폭이 없다.

그래서 이 모듈은 **발목의 가로 좌표만** 내보내고, 세로(접지선 = bbox 아랫변)
와 footprint push 는 기존 그대로 둔다. 사람 14문항 실측:

    발목 픽셀 그대로 역투영          385 점   (현재보다 나쁨)
    현재 (bbox 가로중앙)             419 점
    발목 가로 + 접지선 + push        434 점   <- 이 모듈이 하는 것

## 일반성

발이 몸의 한쪽 끝이라는 것도, 얕은 각도에서 세로 오차가 증폭된다는 것도
이 씬의 성질이 아니라 기하학이다. 연습 씬 레이아웃에 종속된 상수를 박지
않으므로 본선 배경이 바뀌어도 그대로 성립한다.

환경변수:
  MARC_DISABLE_POSE_KEYPOINTS  "1"이면 비활성화 (디버그/타임아웃 대비)
  MARC_POSE_MODEL              가중치 경로. 기본 demo_agents/weights/yolo11m-pose.pt
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger("participant_app")

# COCO 17-keypoint 규약에서 좌/우 발목.
_ANKLE_IDX = (15, 16)

# keypoint 자체의 신뢰도 하한. 가려진 발은 좌표가 엉뚱하게 찍히므로 버린다.
_KP_CONF_MIN = 0.20

# pose 모델이 찾은 사람과 우리 person_* 탐지가 같은 사람인지 판정하는 IoU 하한.
# 같은 사람을 두 모델이 잡으면 보통 0.7 이상 나온다 -- 0.3 은 자세 차이로
# bbox 가 크게 달라지는 경우까지 받아주되 옆 사람과 섞이지는 않는 선이다.
_IOU_MIN = 0.30

# 모델을 한 번에 태울 입력 해상도. YOLO11 검출과 같은 값을 쓴다.
_IMGSZ = 1280

# yolo11n-pose 는 먼 사람을 놓치고(실측 11/14) yolo11x-pose 는 m 과 결과가
# 같으면서 3배 느리다. m 이 14/14 로 최적점.
_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "demo_agents", "weights", "yolo11m-pose.pt")


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 1e-9 else 0.0


class AnkleKeypointEstimator:
    """카메라 프레임 → {사람 bbox: 발목 가로좌표}. 생성자에서 1회 로드."""

    def __init__(self, weights: Optional[str] = None, conf: float = 0.25):
        from ultralytics import YOLO

        self.weights = weights or os.getenv("MARC_POSE_MODEL", _DEFAULT_WEIGHTS)
        self.conf = conf
        self.model = YOLO(self.weights)
        log.info("[pose-kp] loaded %s", self.weights)

    def foot_u(self, image_bgr, boxes: list) -> dict:
        """한 프레임의 사람 bbox 목록 → {bbox 인덱스: (발목 가로, 발목 세로) 픽셀}.

        찾지 못한 bbox 는 결과에 넣지 않는다 (호출부가 기존 동작 유지).
        세로(v)는 prostrate 자세(bending/lying/prone)에서만 쓰인다 -- 그 자세는
        bbox 아랫변이 발이 아니라 카메라에 가장 가까운 몸 부위라서. 다른
        자세에서는 호출부가 v 를 무시하고 기존대로 접지선(bbox 아랫변)을 쓴다.
        """
        if not boxes:
            return {}
        try:
            res = self.model.predict(image_bgr, imgsz=_IMGSZ, conf=self.conf,
                                     verbose=False)[0]
        except Exception as e:                                    # noqa: BLE001
            log.warning("[pose-kp] inference failed: %s", e)
            return {}
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return {}

        pboxes = res.boxes.xyxy.cpu().numpy()
        kps = res.keypoints.data.cpu().numpy()          # (N, 17, 3) = x, y, conf
        out = {}
        for i, box in enumerate(boxes):
            j, best = None, _IOU_MIN
            for k in range(len(pboxes)):
                v = _iou(box, pboxes[k])
                if v > best:
                    j, best = k, v
            if j is None:
                continue
            ank = [kps[j][a] for a in _ANKLE_IDX if kps[j][a][2] > _KP_CONF_MIN]
            if not ank:
                continue
            out[i] = (float(np.mean([a[0] for a in ank])),
                      float(np.mean([a[1] for a in ank])))
        return out


def is_disabled() -> bool:
    return os.getenv("MARC_DISABLE_POSE_KEYPOINTS", "").lower() in ("1", "true", "yes")
