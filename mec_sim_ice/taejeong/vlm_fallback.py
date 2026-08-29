"""VLM(Qwen2.5-VL) 텍스트 파싱 폴백 -- command_parser.py가 target을 못 뽑았을 때만 호출.

역할을 의도적으로 아주 좁게 잡는다: "이 문장에서 뭘 찾아야 하는지"만 다시 물어본다.
relation/좌표/situation 추론에는 절대 쓰지 않는다 -- 그건 이미 기하(geometry)와
공식 규칙표(situation_resolver.py)가 결정론적으로 처리하고 있고, 그게 VLM보다
검증 가능하고 정확하다는 게 이 프로젝트의 기존 설계 결정이다 (DESIGN.md 2.2).

호출 조건 (real_agent_vla.py._ground()):
  parsed.target_type is None and not parsed.is_person
  -- command_parser의 닫힌 동의어 사전에 없는 완전히 새로운 단어가 나온 경우만.
  이 경우가 아니면(정상적으로 클래스가 뽑혔는데 YOLO가 그 장면에서 놓친 경우)는
  dino_fallback.py가 처리하므로 여기까지 오지 않는다.

이 모듈이 하는 일은 둘 중 하나:
  (a) 실은 우리가 아는 클래스인데 동의어 사전에 없는 표현이었다 -> 그 클래스명을 반환
      (예: "grabbed my notes tumbler" 같은 사전 미등재 표현)
  (b) 정말 새로운 물체다 -> DINO에 넘길 자연어 검색어(1~4단어)를 반환

스펙(GPU/VRAM/인터넷) 근거는 MARC2026_STAGE1_BRIEF.md §1, §7 참고:
  - 제출 기본값 = 3B fp16(2026-08-28 변경, VRAM 예산 12GB 이내 -- 7B fp16은
    DINO(~1GB)와 합쳐 16GB를 넘겨 초과했다. crop 검증(이미지) 은 폐기돼서
    (fallback.py 참고) 지금 VLM 이 하는 일은 텍스트 분류 두 건뿐이라 3B로
    충분함을 이번 세션 실측으로 확인). docker/Dockerfile, docker-compose.yml
    이 실제 빌드값.
  - 품질을 올리고 싶으면 7B + MARC_VLM_4BIT=1(~5~6GB, CUDA 전용 -- 채점
    서버/랩 데스크탑만 해당, 로컬 Mac(MPS)에선 검증 불가)
  - 가중치는 빌드 시점에 굽는다(HF_HUB_OFFLINE=1 런타임 고정) -- 런타임 다운로드 금지

기본값 (2026-08-29부터 OFF):
  chungmu 39라운드 실측(report_fb_fix.json) -- 이 폴백이 켜진 조건에서 39/39
  전부 command_parser 가 닫힌 동의어 사전 안에서 target 을 뽑아, 호출 조건
  (target_type is None)이 단 한 번도 성립하지 않았다. 즉 이 모듈은 39라운드
  내내 0회 호출되면서 3B bf16(~7GB)을 상주시키기만 했다. 데스크탑 GPU는
  8GB 를 Isaac Sim 과 나눠 쓰므로(MARC2026_STAGE1_BRIEF.md §1), 켜 둬 봐야
  chungmu 에서는 얻는 것 없이 OOM 위험만 진다 -- 그래서 기본을 OFF 로 뒤집는다.
  DINO 는 같은 조건에서 실제로 호출됐고 rig_3_b 3문항을 구제했으므로(+107.3점)
  그쪽 기본은 그대로 ON 이다 -- 이 둘은 독립적으로 켜고 끈다.
  본선에서 닫힌 어휘를 벗어난 문항이 나올 경우의 보험으로 다시 켜려면
  MARC_ENABLE_VLM=1.

환경변수:
  MARC_ENABLE_VLM    "1"이면 켠다 (기본은 OFF -- 위 참고)
  MARC_DISABLE_VLM   "1"이면 확실히 끈다 (MARC_ENABLE_VLM 보다 우선)
  MARC_VLM_MODEL     기본 Qwen/Qwen2.5-VL-3B-Instruct (7B로 올리려면 명시 지정 + 4bit 권장)
  MARC_VLM_4BIT      "1"이면 bitsandbytes 4bit 로드
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

log = logging.getLogger("participant_app")


def _from_pretrained_local_first(cls, model_name: str, **kwargs):
    """캐시에 이미 있으면 로컬에서만 로드 -- 없으면 온라인 폴백.

    dino_fallback.py 의 동일 함수와 같은 근거: HF_HUB_OFFLINE 없이 로컬
    개발하면 이미 캐시된 Qwen 가중치도 매번 huggingface.co 에 HEAD 요청을
    보내 로드가 느려진다(실측, 2026-08-28). local_files_only=True 를 먼저
    시도해 캐시가 있으면 네트워크를 아예 안 타게 하고, 캐시가 없으면(최초
    실행) 예외를 잡아 평범한 온라인 다운로드로 폴백한다.
    """
    try:
        return cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        log.info("[hf-cache] %s 로컬 캐시 없음 -- 온라인 다운로드", model_name)
        return cls.from_pretrained(model_name, **kwargs)

_PROMPT = """You are parsing a lost-item report for a campus CCTV search system.
The system's object detector only recognizes these exact classes: {classes}.

Read the request below.
- If the lost item is one of those classes (even if worded differently, e.g. a
  synonym or brand name), answer with that EXACT class string.
- If it is NOT one of those classes (a genuinely different object), set
  "target_type" to null and instead give a short plain-English search phrase
  (1-4 words, the object's common name) suitable as a query for an open-vocabulary
  object detector.

Respond with JSON only, no prose, exactly these two keys:
{{"target_type": <one of {classes} or null>, "search_phrase": <string or null>}}

Request: "{command}" """

# "behind" 중의성 중재 (command_parser 가 spatial+구동사 동시 감지로 confident 를
# 낮춘 문장에서만 호출된다 -- 규칙이 확신하는 문장엔 VLM 을 쓰지 않는다).
_BEHIND_PROMPT = """In the sentence below, decide how the word "behind" is used:
- SPATIAL: it describes a location at the back side of an object
  (e.g. "the box is behind the bench").
- PHRASAL: it is part of the verb "to leave something behind", meaning the
  item was forgotten / left at a place (e.g. "I left my bag behind").

Respond with JSON only, exactly one key:
{{"spatial": true or false}}

Sentence: "{command}" """

# DINO 탐지 crop 재검증. DINO 는 재현율은 좋지만 낮은 임계값에선 오탐도 많다 --
# 특히 "탐지가 여러 카메라에서 동시에 나오는" 경우 진짜는 한 장면뿐이므로
# 나머지는 오탐이 확정적이다(CCTV 시야는 겹치지 않는다). crop 을 VLM 에 보여
# 물체 종류가 맞는지 이진 판정을 받는다. 씬 하드코딩 아님: 판정 대상은 명령이
# 말한 물체 이름과 그 픽셀 영역뿐이다.
_VERIFY_PROMPT = """Look at the object at the center of this image crop from an
outdoor CCTV frame. It may be small and it is a stylized 3D asset, not a photo.

Is the object a {phrase}? If it is clearly some other kind of object, answer
false. If it plausibly is a {phrase}, answer true.

Respond with JSON only, exactly one key:
{{"match": true or false}}"""


class QwenVLBackend:
    """stage1/pipeline.py의 QwenVLBackend 계승 -- 텍스트 파싱 + crop 검증(이미지) 겸용."""

    def __init__(self, model_name: Optional[str] = None, use_4bit: Optional[bool] = None):
        import torch
        try:
            # 제출 이미지의 transformers==4.53.2 경로 (docker/requirements.txt)
            from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
        except ImportError:
            # transformers 5.x (일부 개발 머신): 전용 클래스 대신 auto 클래스 사용
            from transformers import AutoModelForImageTextToText as _ModelCls
        from transformers import AutoProcessor

        self.model_name = model_name or os.getenv("MARC_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
        use_4bit = (os.getenv("MARC_VLM_4BIT", "0") == "1") if use_4bit is None else use_4bit
        # NVIDIA(CUDA) -> Apple Silicon(MPS) -> CPU.
        device = ("cuda" if torch.cuda.is_available()
                 else "mps" if torch.backends.mps.is_available() else "cpu")
        if use_4bit and device != "cuda":
            # bitsandbytes 4bit 양자화는 CUDA 전용 -- MPS/CPU 에선 로드 자체가 실패한다.
            # Mac(M-시리즈)은 통합 메모리가 넉넉한 편이라 4bit 없이 bf16 그대로도
            # 3B(~6GB)는 대개 무리 없다. 여기서 자동으로 꺼서 조용히 실패하지 않게 한다.
            log.warning("[vlm-fallback] 4bit 요청했지만 device=%s (CUDA 아님) -- "
                       "bitsandbytes 미지원이라 무시하고 bf16 그대로 로드합니다", device)
            use_4bit = False
        kw = dict(torch_dtype=torch.bfloat16, device_map=device)
        if use_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        t0 = time.time()
        self.model = _from_pretrained_local_first(_ModelCls, self.model_name, **kw).eval()
        self.processor = _from_pretrained_local_first(AutoProcessor, self.model_name)
        log.info("[vlm-fallback] %s loaded in %.1fs (4bit=%s)", self.model_name, time.time() - t0, use_4bit)

    def generate(self, prompt: str, image=None, max_new_tokens: int = 96) -> str:
        """image: HxWx3 uint8 RGB ndarray (선택). DINO 탐지 crop 검증에 쓴다."""
        import numpy as np
        import torch

        content = []
        pil = None
        if image is not None:
            from PIL import Image
            pil = Image.fromarray(np.ascontiguousarray(image))
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil] if pil is not None else None,
                                return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]


class VlmTargetResolver:
    """command_parser가 target_type=None을 낸 문장을 VLM으로 한 번 더 해석."""

    def __init__(self, backend: Optional[QwenVLBackend] = None):
        self.backend = backend or QwenVLBackend()

    def resolve(self, command: str, known_classes) -> tuple:
        """-> (matched_class_or_None, search_phrase_or_None).

        matched_class가 있으면 search_phrase는 신경 안 써도 됨(정상 YOLO 경로로
        복귀). matched_class가 None이면 search_phrase를 DINO 검색어로 쓴다.
        둘 다 None이면(VLM도 실패) 호출부는 최종 (0,0,0) 제출로 넘어간다.
        """
        classes = list(known_classes)
        prompt = _PROMPT.format(classes=json.dumps(classes), command=command.replace('"', "'"))
        txt = self.backend.generate(prompt)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            target = js.get("target_type")
            phrase = js.get("search_phrase")
            if target in classes:
                return target, None
            return None, (str(phrase).strip() if phrase else None)
        except Exception as e:  # noqa: BLE001
            log.warning("[vlm-fallback] parse failed: %s | raw=%r", e, txt[:200])
            return None, None

    def behind_is_spatial(self, command: str):
        """중의적 "behind"(구동사 leave-behind vs 공간관계)를 VLM 으로 중재.

        -> True(공간) / False(구동사) / None(판정 실패 -- 호출부는 기존 힌트 유지).
        command_parser 가 confident=False 로 넘긴 문장에서만 부를 것.
        """
        prompt = _BEHIND_PROMPT.format(command=command.replace('"', "'"))
        txt = self.backend.generate(prompt, max_new_tokens=24)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            v = js.get("spatial")
            return v if isinstance(v, bool) else None
        except Exception as e:  # noqa: BLE001
            log.warning("[vlm-fallback] behind arbitration failed: %s | raw=%r", e, txt[:120])
            return None

    def verify_crop(self, img_bgr, bbox, phrase: str, pad: float = 1.5,
                    min_ctx_px: int = 96):
        """DINO 탐지 crop 이 정말 phrase 물체인지 VLM 이진 판정.

        -> True(맞음) / False(다른 물체) / None(판정 실패 -- 호출부는 유지 권장).
        bbox 는 (x1,y1,x2,y2) 픽셀. 소형 물체는 bbox 만 자르면 문맥이 없어
        판정이 어렵다 -- 사방 pad 배 + 최소 min_ctx_px 만큼 주변을 포함한다.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        H, W = img_bgr.shape[:2]
        ex = max(pad * w, (min_ctx_px - w) / 2.0, 0.0)
        ey = max(pad * h, (min_ctx_px - h) / 2.0, 0.0)
        cx1, cy1 = max(0, int(x1 - ex)), max(0, int(y1 - ey))
        cx2, cy2 = min(W, int(x2 + ex)), min(H, int(y2 + ey))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None
        crop_rgb = img_bgr[cy1:cy2, cx1:cx2, ::-1]
        txt = self.backend.generate(_VERIFY_PROMPT.format(phrase=phrase),
                                    image=crop_rgb, max_new_tokens=16)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            v = js.get("match")
            return v if isinstance(v, bool) else None
        except Exception as e:  # noqa: BLE001
            log.warning("[vlm-fallback] crop verify failed: %s | raw=%r", e, txt[:120])
            return None


def is_disabled() -> bool:
    """기본 OFF (모듈 docstring 참고 -- chungmu 39/39 에서 호출 0회, 상주 비용 ~7GB).

    MARC_ENABLE_VLM=1 로 켜고, MARC_DISABLE_VLM=1 은 그래도 확실히 끈다
    (명시적 비활성화가 명시적 활성화보다 우선 -- 둘 다 세팅된 애매한 상태에서
    안전한 쪽으로 넘어간다).
    """
    if os.getenv("MARC_DISABLE_VLM", "").lower() in ("1", "true", "yes"):
        return True
    return os.getenv("MARC_ENABLE_VLM", "").lower() not in ("1", "true", "yes")
