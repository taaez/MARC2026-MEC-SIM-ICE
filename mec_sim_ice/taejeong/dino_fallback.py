"""Grounding DINO 개방어휘 탐지 폴백 -- YOLO11이 실패했을 때만 호출된다.

왜 필요한가 (real_agent_vla.py._ground() 의 실패 경로 2가지):
  (a) 텍스트 파싱이 학습 어휘 밖 단어를 만나 target_type 이 빈 값이 됨
  (b) target_type 은 맞게 뽑혔는데 YOLO11 이 그 장면에서 그 클래스를 못 찾음
      (파인튜닝 부족/각도/가려짐 -- 우리 34클래스 안에 있는 물체라도 발생 가능)

이 모듈은 (b)를 구제한다: YOLO 가 다 실패한 뒤, 같은 이미지를 Grounding DINO
(제로샷 개방어휘 탐지기, IDEA-Research/grounding-dino-base)에 텍스트 프롬프트로
한 번 더 태운다. (a)(완전히 새 단어, 우리 동의어 사전에도 없는 경우)는 이 모듈
혼자서는 못 푼다 -- 검색할 phrase 자체가 없으므로. 그건 vlm_fallback.py가
"이 문장에서 뭘 찾아야 하는지" phrase 를 만들어 real_agent_vla.py를 거쳐 여기로
넘겨준다(vlm_search_phrase).

real_agent_vla.py 는 ultralytics Boxes 객체(.xyxy, .conf, .cls)를 그대로 쓰는
관례라, 여기서도 같은 최소 인터페이스(_DinoBox)로 반환해 호출부(_best_box 계열)
코드를 새로 만들지 않고 그대로 재사용할 수 있게 한다.

환경변수:
  MARC_DISABLE_DINO_FALLBACK   "1"이면 폴백 자체를 비활성화 (디버그/타임아웃 대비)
  MARC_DET_MODEL               기본 IDEA-Research/grounding-dino-base
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("participant_app")


class _DinoBox:
    """ultralytics box와 같은 최소 인터페이스(.xyxy, .conf) -- _box_center() 등
    기존 호출부를 고치지 않고 그대로 통과시키기 위한 shim."""

    def __init__(self, xyxy: tuple, conf: float):
        self.xyxy = [list(xyxy)]
        self.conf = [conf]


class DinoFallbackDetector:
    """YOLO11 실패 시 2차로 태우는 제로샷 탐지기. 생성자에서 1회 로드(§7 규칙)."""

    def __init__(self, model_name: Optional[str] = None,
                 box_thr: float = 0.3, text_thr: float = 0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_name = model_name or os.getenv("MARC_DET_MODEL", "IDEA-Research/grounding-dino-base")
        # NVIDIA(CUDA, 채점 서버/랩 데스크탑) -> Apple Silicon(MPS, 로컬 Mac 테스트) -> CPU.
        self.device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name).to(self.device).eval()
        self.box_thr, self.text_thr = box_thr, text_thr
        log.info("[dino-fallback] loaded %s on %s", self.model_name, self.device)

    def best_box(self, img_bgr: np.ndarray, phrases: Sequence[str]) -> Optional[_DinoBox]:
        """phrases 중 아무거나 매치되는 것 중 가장 점수 높은 박스 하나. 없으면 None.

        phrase 를 하나씩 따로 호출한다(합쳐서 한 프롬프트로 보내지 않음).
        실측(2026-08-28, Mac MPS, 실제 CCTV 기록): "pencil case" 단독 호출은
        score=0.11로 잡히는데, 동의어 4개를 ". "로 합친 한 프롬프트로 보내면
        동일 임계값에서도 전혀 안 잡혔다 -- 후보 문구가 여러 개 섞이면 서로
        간섭해 신호가 죽는 것으로 보인다. 호출 수는 늘지만(phrase 개수만큼),
        이 함수 자체가 YOLO 실패 후에만 도는 최후 폴백이라 그 비용은 감내한다.

        img_bgr: real_agent_vla.py 관례(ultralytics/cv2, BGR). DINO/PIL 은 RGB
        를 기대하므로 여기서 뒤집는다 (호출부는 신경 쓸 필요 없음).
        """
        if not phrases:
            return None
        img_rgb = img_bgr[:, :, ::-1]
        best_box, best_score = None, -1.0
        for phrase in phrases:
            box, score = self._detect_one(img_rgb, phrase)
            if box is not None and score > best_score:
                best_box, best_score = box, score
        if best_box is None:
            return None
        return _DinoBox(best_box, best_score)

    def _detect_one(self, img_rgb: np.ndarray, phrase: str):
        """phrase 하나에 대한 (bbox, score) 중 최고점. 없으면 (None, -1.0)."""
        import torch
        from PIL import Image

        pil = Image.fromarray(img_rgb)
        text = phrase.strip().rstrip(".") + "."
        inputs = self.processor(images=pil, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[pil.size[::-1]])[0]

        box, score = None, -1.0
        for b, s in zip(res["boxes"], res["scores"]):
            s = float(s)
            if s > score:
                box, score = tuple(float(v) for v in b.tolist()), s
        return box, score


def is_disabled() -> bool:
    return os.getenv("MARC_DISABLE_DINO_FALLBACK", "").lower() in ("1", "true", "yes")
