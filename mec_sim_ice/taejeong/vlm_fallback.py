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
  - 로컬 8GB 개발: MARC_VLM_4BIT=1 + 3B 모델 (Isaac Sim과 동시 실행 시 더 타이트해짐 --
    필요하면 MARC_DISABLE_VLM=1로 끄고 eval_offline.py/records 재실행으로 따로 검증)
  - 채점 서버 48GB: 7B bf16 그대로
  - 가중치는 빌드 시점에 굽는다(HF_HUB_OFFLINE=1 런타임 고정) -- 런타임 다운로드 금지

환경변수:
  MARC_DISABLE_VLM   "1"이면 폴백 자체를 비활성화
  MARC_VLM_MODEL     기본 Qwen/Qwen2.5-VL-7B-Instruct (로컬 8GB: Qwen/Qwen2.5-VL-3B-Instruct)
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


class QwenVLBackend:
    """stage1/pipeline.py의 QwenVLBackend와 동일 -- 텍스트 전용 폴백이라 이미지 입력은 안 쓴다."""

    def __init__(self, model_name: Optional[str] = None, use_4bit: Optional[bool] = None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_name = model_name or os.getenv("MARC_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
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
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        log.info("[vlm-fallback] %s loaded in %.1fs (4bit=%s)", self.model_name, time.time() - t0, use_4bit)

    def generate(self, prompt: str, max_new_tokens: int = 96) -> str:
        import torch

        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=None, return_tensors="pt").to(self.model.device)
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


def is_disabled() -> bool:
    return os.getenv("MARC_DISABLE_VLM", "").lower() in ("1", "true", "yes")
