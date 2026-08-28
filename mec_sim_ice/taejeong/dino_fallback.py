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


def _from_pretrained_local_first(cls, model_name: str, **kwargs):
    """캐시에 이미 있으면 로컬에서만 로드 -- 없으면 평범하게 온라인 폴백.

    실측(2026-08-28): HF_HUB_OFFLINE 을 안 켠 로컬 개발 환경에서는 이미 캐시된
    모델이라도 from_pretrained() 가 매번 config.json/preprocessor_config.json
    등 여러 파일에 대해 HEAD 요청을 huggingface.co 에 보내(갱신 확인용) 로드
    시작이 눈에 띄게 늦어진다. local_files_only=True 를 먼저 시도하면 캐시가
    있을 때 이 네트워크 왕복을 통째로 건너뛴다. 캐시가 없으면(최초 실행) 여기서
    예외가 나므로 평범한 온라인 로드로 자연스럽게 폴백한다 -- 즉 첫 실행은
    그대로 동작하고, 그 이후부터는 네트워크를 안 탄다.
    """
    try:
        return cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        log.info("[hf-cache] %s 로컬 캐시 없음 -- 온라인 다운로드", model_name)
        return cls.from_pretrained(model_name, **kwargs)


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
        # torch_dtype 을 안 정해주면 fp32 로 로드된다 -- vlm_fallback.py 는 이미
        # bf16 인데 DINO 만 그대로였다. 실측(2026-08-29, 윈도우 RTX 4070): 타일링
        # 폴백(_dino_best_box_tiled, 8타일 배치)이 걸린 pencilcase 한 카메라가
        # 2분 넘게 걸렸다 -- fp32 는 텐서 코어(fp16/bf16 전용 가속)를 못 써서
        # CUDA 에서 수 배 느리다. CUDA만 bf16 로 낮춘다 -- MPS 는 이 모델의
        # 일부 연산(행렬곱 커널)이 bf16 조합에서 바로 assert 로 죽는 걸 실측으로
        # 확인해서(2026-08-29, MPSNDArrayMatrixMultiplication dtype mismatch)
        # 제외, CPU 도 가속기가 없어 낮출 이득이 없다.
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = _from_pretrained_local_first(AutoProcessor, self.model_name)
        self.model = _from_pretrained_local_first(
            AutoModelForZeroShotObjectDetection, self.model_name,
            torch_dtype=self.dtype).to(self.device).eval()
        self.box_thr, self.text_thr = box_thr, text_thr
        log.info("[dino-fallback] loaded %s on %s (dtype=%s)",
                 self.model_name, self.device, self.dtype)

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

    def best_box_multi_images(self, imgs_bgr: Sequence[np.ndarray],
                              phrases: Sequence[str]) -> list:
        """이미지 여러 장(예: 타일 8장)에 대해 각각 best_box() 를 부르는 것과
        같은 결과를, 이미지 개수만큼 순차 호출하는 대신 phrase 당 배치 호출
        1번으로 묶어서 낸다.

        실측(2026-08-28): 8분할 타일링을 도입한 뒤 카메라 1대당 (1 whole-frame
        + 8타일) x phrase 개수(동의어 3~4개) = 최대 36번의 개별 forward 를
        순차로 돌게 되면서 카메라 1대에 10분 넘게 걸리는 문제가 생겼다. 개별
        호출마다의 파이썬/전처리 오버헤드가 누적된 게 원인이라, 이미지를 배치로
        묶으면(phrase 당 1번, 이미지 N장을 한 forward 에) 호출 수가 phrase
        개수(보통 3~4번)로 줄어든다.

        반환: imgs_bgr 순서대로 Optional[_DinoBox] 리스트.
        """
        if not phrases or not imgs_bgr:
            return [None] * len(imgs_bgr)
        imgs_rgb = [img[:, :, ::-1] for img in imgs_bgr]
        best = [(None, -1.0)] * len(imgs_bgr)
        for phrase in phrases:
            for i, (box, score) in enumerate(self._detect_batch(imgs_rgb, phrase)):
                if box is not None and score > best[i][1]:
                    best[i] = (box, score)
        return [_DinoBox(box, score) if box is not None else None
                for box, score in best]

    def _detect_batch(self, imgs_rgb: Sequence[np.ndarray], phrase: str):
        """phrase 하나를 이미지 여러 장에 배치로 적용 -- 이미지 순서대로
        (bbox, score) 리스트(없으면 (None, -1.0))."""
        import torch
        from PIL import Image

        pils = [Image.fromarray(img) for img in imgs_rgb]
        text = phrase.strip().rstrip(".") + "."
        inputs = self.processor(images=pils, text=[text] * len(pils),
                                return_tensors="pt", padding=True).to(self.device)
        # processor 는 pixel_values 를 항상 fp32 로 낸다 -- 모델을 bf16 으로
        # 낮춰도 여기서 안 맞춰주면 forward 에서 dtype 불일치 에러가 난다.
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            out = self.model(**inputs)
        target_sizes = [pil.size[::-1] for pil in pils]
        results = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=target_sizes)
        out_list = []
        for res in results:
            box, score = None, -1.0
            for b, s in zip(res["boxes"], res["scores"]):
                s = float(s)
                if s > score:
                    box, score = tuple(float(v) for v in b.tolist()), s
            out_list.append((box, score))
        return out_list

    def _detect_one(self, img_rgb: np.ndarray, phrase: str):
        """phrase 하나에 대한 (bbox, score) 중 최고점. 없으면 (None, -1.0)."""
        import torch
        from PIL import Image

        pil = Image.fromarray(img_rgb)
        text = phrase.strip().rstrip(".") + "."
        inputs = self.processor(images=pil, text=text, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
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
