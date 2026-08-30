"""소형 텍스트 LM 폴백 -- command_parser.py가 target을 못 뽑았을 때만 호출.

역할을 의도적으로 아주 좁게 잡는다: "이 문장에서 뭘 찾아야 하는지"만 다시 물어본다.
relation/좌표/situation 추론에는 절대 쓰지 않는다 -- 그건 이미 기하(geometry)와
공식 규칙표(situation_resolver.py)가 결정론적으로 처리하고 있고, 그게 이 모델보다
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

2026-08-29: VL(비전-언어) 모델 경로를 완전히 제거했다. 하는 일이 텍스트 분류
(target 재해석 / "behind" 중의성 두 건뿐이라(위 참고) 애초에 이미지 입력이
필요 없었다(이미지 crop 검증은 이미 그 전에 폐기됨, generate() 옛 docstring
참고). VL 모델(3B/7B)은 이미지 인코더까지 얹은 만큼만 VRAM 을 더 쓰는데,
쓰지도 않는 기능에 6~7GB 를 상주시키는 건 낭비다 -- 8GB 데스크탑 GPU 를
Isaac Sim 과 나눠 쓰는 제약(MARC2026_STAGE1_BRIEF.md §1)에서 이 낭비가 VLM
을 기본 OFF 로 둘 수밖에 없었던 직접적인 이유였다. 소형 텍스트 전용 LM
(0.5~1.5B, ~1~3GB)만 남기면 그 제약 안에서도 켜둘 수 있어, 이제 VL/텍스트
자동판별 분기 자체를 지운다 -- 코드가 그 자체로 안 켜질 선택지를 갖고
있을 이유가 없다.

**TODO(테스트 필요)**: 아래를 아직 아무도 확인하지 않았다.
  (1) 소형 모델이 _PROMPT / _BEHIND_PROMPT 에 대해 JSON 을 제대로 뱉는가
      (작은 모델일수록 형식 이탈이 잦다 -- resolve()/behind_is_spatial()
      은 파싱 실패 시 None 을 돌려주므로 조용히 무력화될 수 있다).
  (2) 판정 품질 (특히 OOV 물체명 -> 클래스 복구. "thermos"->tumbler,
      "parasol"->umbrella 같은 것).
  (3) 실제 VRAM 점유와 문항당 지연 (라이브 예산 30초).
검증 전에는 제출 기본값을 바꾸지 말 것.

기본값 (2026-08-29부터 OFF):
  chungmu 39라운드 실측(report_fb_fix.json) -- 이 폴백이 켜진 조건에서 39/39
  전부 command_parser 가 닫힌 동의어 사전 안에서 target 을 뽑아, 호출 조건
  (target_type is None)이 단 한 번도 성립하지 않았다. 즉 이 모듈은 39라운드
  내내 0회 호출되면서 모델을 상주시키기만 했다. 켜 둬 봐야 chungmu 에서는
  얻는 것 없이 VRAM 만 먹으므로 기본은 OFF 다. DINO 는 같은 조건에서 실제로
  호출됐고 rig_3_b 3문항을 구제했으므로(+107.3점) 그쪽 기본은 그대로 ON 이다
  -- 이 둘은 독립적으로 켜고 끈다.
  본선에서 닫힌 어휘를 벗어난 문항이 나올 경우의 보험으로 다시 켜려면
  MARC_ENABLE_VLM=1.

가중치는 빌드 시점에 굽는다(HF_HUB_OFFLINE=1 런타임 고정) -- 런타임 다운로드 금지.

환경변수:
  MARC_ENABLE_VLM    "1"이면 켠다 (기본은 OFF -- 위 참고)
  MARC_DISABLE_VLM   "1"이면 확실히 끈다 (MARC_ENABLE_VLM 보다 우선)
  MARC_VLM_MODEL     기본 Qwen/Qwen2.5-1.5B-Instruct (텍스트 전용 소형 LM만
                     지정할 것 -- VL 모델은 더 이상 지원 안 함)
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
    개발하면 이미 캐시된 가중치도 매번 huggingface.co 에 HEAD 요청을
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
# 낮춘 문장에서만 호출된다 -- 규칙이 확신하는 문장엔 이 모델을 쓰지 않는다).
_BEHIND_PROMPT = """In the sentence below, decide how the word "behind" is used:
- SPATIAL: it describes a location at the back side of an object
  (e.g. "the box is behind the bench").
- PHRASAL: it is part of the verb "to leave something behind", meaning the
  item was forgotten / left at a place (e.g. "I left my bag behind").

Respond with JSON only, exactly one key:
{{"spatial": true or false}}

Sentence: "{command}" """

# 2026-08-29 추가 3건 -- 전부 command_parser 의 닫힌 규칙(사전/정규식)이
# 막다른 곳(None)에 도달했을 때만 호출된다(호출부는 fallback.py.vlm_assist
# 참고). 새 서사형 문장(hf/sarH 계열, 대회 본선에서 새 표현이 나올 수 있는
# 부분)에 대한 일반화 보험이다 -- 전부 "이 문장이 뭘 말하는지"뿐이고 좌표/
# relation/situation 추론에는 안 쓴다(모듈 docstring 원칙 유지).
_LANDMARK_PROMPT = """You are parsing a lost-item report for a campus CCTV search system.
The system's landmark detector only recognizes these exact reference-object classes: {classes}.

Find the reference object/landmark the request mentions -- the thing the lost
item is described as being on, near, or beside. This is NOT the lost item
itself.
- If that reference object is one of the classes above (even worded
  differently, e.g. a synonym), answer with that EXACT class string.
- If it names a different reference object not in that list, set
  "landmark_type" to null and give a short plain-English search phrase
  (1-4 words) for it instead.
- If the request does not clearly name any reference object at all, set both
  fields to null.

Respond with JSON only, no prose, exactly these two keys:
{{"landmark_type": <one of {classes} or null>, "search_phrase": <string or null>}}

Request: "{command}" """

_PERSON_PROMPT = """You are parsing a request for a campus CCTV search system.
Decide whether the request asks to find a PERSON (a human being, described by
role, clothing, or their situation/posture) or asks to find an OBJECT (an
item like a mug, bag, bike, etc. that someone left behind).

Respond with JSON only, exactly one key:
{{"is_person": true or false}}

Request: "{command}" """

_POSE_PROMPT = """You are parsing a request to find a person on campus CCTV.
The system only recognizes these coarse pose categories: {buckets}.

Decide which single category best matches how the person's posture/situation
is described. If none clearly applies, use null -- do not guess.

Respond with JSON only, exactly one key:
{{"pose": <one of {buckets} or null>}}

Request: "{command}" """

class SmallLmBackend:
    """텍스트 전용 소형 인과 LM(causal LM) 래퍼.

    2026-08-29 이전엔 VL(비전-언어) 모델과 자동판별하는 분기가 있었으나
    통째로 제거했다(위 모듈 docstring 참고) -- 이 모듈은 이미지를 절대
    받지 않으므로 VL 모델을 지원할 이유가 없다. 항상 AutoModelForCausalLM
    + AutoTokenizer 로만 로드한다.
    """

    def __init__(self, model_name: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name or os.getenv("MARC_VLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        # NVIDIA(CUDA) -> Apple Silicon(MPS) -> CPU.
        device = ("cuda" if torch.cuda.is_available()
                 else "mps" if torch.backends.mps.is_available() else "cpu")
        t0 = time.time()
        self.model = _from_pretrained_local_first(
            AutoModelForCausalLM, self.model_name,
            torch_dtype=torch.bfloat16, device_map=device).eval()
        self.tokenizer = _from_pretrained_local_first(AutoTokenizer, self.model_name)
        log.info("[vlm-fallback] %s loaded in %.1fs (device=%s)",
                 self.model_name, time.time() - t0, device)

    def generate(self, prompt: str, max_new_tokens: int = 96) -> str:
        """텍스트 전용. 이미지는 애초에 받지 않는다(클래스 docstring 참고).

        좌표/relation/situation 은 기하와 공식 규칙표가 결정론적으로 처리하고
        그게 이 모델보다 검증 가능하다는 게 이 프로젝트의 설계 결정이다
        (DESIGN.md 2.2). 이 모듈이 하는 일은 **텍스트 이해뿐**이다.
        """
        import torch

        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


class VlmTargetResolver:
    """command_parser가 target_type=None을 낸 문장을 소형 LM으로 한 번 더 해석.

    이름은 VLM 시절 그대로 남겨둔다(offline_score.py/taejeong_agent_vla.py
    가 이 이름으로 import) -- 실제로 vision 은 안 쓰지만 호출부 시그니처를
    바꿀 이유가 없다.
    """

    def __init__(self, backend: Optional[SmallLmBackend] = None):
        self.backend = backend or SmallLmBackend()

    def resolve(self, command: str, known_classes) -> tuple:
        """-> (matched_class_or_None, search_phrase_or_None).

        matched_class가 있으면 search_phrase는 신경 안 써도 됨(정상 YOLO 경로로
        복귀). matched_class가 None이면 search_phrase를 DINO 검색어로 쓴다.
        둘 다 None이면(모델도 실패) 호출부는 최종 (0,0,0) 제출로 넘어간다.
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
        """중의적 "behind"(구동사 leave-behind vs 공간관계)를 소형 LM으로 중재.

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

    def resolve_landmark(self, command: str, known_classes) -> tuple:
        """resolve()와 동일 패턴, 대상만 target 이 아니라 랜드마크.

        -> (matched_class_or_None, search_phrase_or_None). matched_class 가
        있으면 dino_augment() 의 기존 랜드마크 재탐지 경로가 그대로 처리
        (LANDMARK_SYNONYMS 에서 동의어를 찾음). search_phrase 만 있으면
        그 문자열 자체가 landmark_type 이 되어 DINO 검색어로 그대로 쓰인다
        (target 의 OOV 경로와 동일한 재사용).
        """
        classes = list(known_classes)
        prompt = _LANDMARK_PROMPT.format(classes=json.dumps(classes), command=command.replace('"', "'"))
        txt = self.backend.generate(prompt)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            lm = js.get("landmark_type")
            phrase = js.get("search_phrase")
            if lm in classes:
                return lm, None
            return None, (str(phrase).strip() if phrase else None)
        except Exception as e:  # noqa: BLE001
            log.warning("[vlm-fallback] landmark parse failed: %s | raw=%r", e, txt[:200])
            return None, None

    def classify_person(self, command: str):
        """이 요청이 사람을 찾는 건지 물체를 찾는 건지 판정.

        -> True(사람) / False(물체) / None(판정 실패 -- 호출부는 기존 상태 유지).
        command_parser 가 target_type/is_person 둘 다 못 찾은(진짜 중의)
        문장에서만 부를 것.
        """
        prompt = _PERSON_PROMPT.format(command=command.replace('"', "'"))
        txt = self.backend.generate(prompt, max_new_tokens=24)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            v = js.get("is_person")
            return v if isinstance(v, bool) else None
        except Exception as e:  # noqa: BLE001
            log.warning("[vlm-fallback] person classify failed: %s | raw=%r", e, txt[:120])
            return None

    def classify_pose(self, command: str, buckets) -> Optional[str]:
        """자유 서술 자세 표현을 공식 pose 코스 버킷(command_parser.POSE_BUCKETS
        이름) 중 하나로 매핑. situation 자체는 여전히 situation_resolver.py
        가 결정한다 -- 이건 여러 사람 후보 중 고르는 데 쓰는 힌트일 뿐이다.

        -> 버킷 이름 또는 None(판정 실패/불명확 -- 호출부는 pose_hint=None 유지).
        """
        bucket_list = list(buckets)
        prompt = _POSE_PROMPT.format(buckets=json.dumps(bucket_list), command=command.replace('"', "'"))
        txt = self.backend.generate(prompt, max_new_tokens=24)
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            pose = js.get("pose")
            return pose if pose in bucket_list else None
        except Exception as e:  # noqa: BLE001
            log.warning("[vlm-fallback] pose classify failed: %s | raw=%r", e, txt[:120])
            return None


def is_disabled() -> bool:
    """기본 OFF (모듈 docstring 참고 -- chungmu 39/39 에서 호출 0회).

    MARC_ENABLE_VLM=1 로 켜고, MARC_DISABLE_VLM=1 은 그래도 확실히 끈다
    (명시적 비활성화가 명시적 활성화보다 우선 -- 둘 다 세팅된 애매한 상태에서
    안전한 쪽으로 넘어간다).
    """
    if os.getenv("MARC_DISABLE_VLM", "").lower() in ("1", "true", "yes"):
        return True
    return os.getenv("MARC_ENABLE_VLM", "").lower() not in ("1", "true", "yes")
