"""
stage1/pipeline.py
------------------
Stage 1 인지 파이프라인. ROS 에 의존하지 않는다 — 입력은 numpy 이미지 + CameraModel,
출력은 payload dict (GroundingResult.from_payload 로 바로 변환 가능).

구성:
  CommandParser  : 자연어 명령 → {target_type, landmark, relation, situation, is_person}
                   1) 규칙 기반(빠름, 결정적)  2) VLM(Qwen2.5-VL) 폴백/교차검증
  Detector       : 텍스트 프롬프트 → bbox.  기본 Grounding DINO(HF transformers),
                   선택적으로 파인튜닝 YOLO(ultralytics) 를 앞에 둔다.
  SituationClassifier : 사람 문제의 situation 을 VLM 으로 분류 (crop 입력)
  Stage1Pipeline : 위를 조합. 카메라 선택 + 관계 일관성 + 역투영.

환경변수:
  MARC_VLM_MODEL     기본 Qwen/Qwen2.5-VL-7B-Instruct   (로컬 8GB: Qwen/Qwen2.5-VL-3B-Instruct)
  MARC_VLM_4BIT      "1" 이면 bitsandbytes 4bit 로드 (로컬 8GB 개발용)
  MARC_DET_MODEL     기본 IDEA-Research/grounding-dino-base
  MARC_YOLO_WEIGHTS  지정 시 ultralytics 모델 우선 사용 (파인튜닝 결과)
  MARC_DISABLE_VLM   "1" 이면 VLM 미사용(규칙 파서만) — 디버그/타임아웃 대비
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import CameraModel, detection_to_world, relation_scores, RELATIONS

log = logging.getLogger("stage1.pipeline")

# ----------------------------------------------------------------------------
# 어휘 (연습 씬 기준. 실제 대회 배경이 바뀔 수 있으므로 좌표는 절대 넣지 않고 '이름' 만 둔다)
# ----------------------------------------------------------------------------
OBJECT_VOCAB = {
    # 실측 근거: demo/mock_demo_data.yaml 의 stage1_answers target_type 분포 (33문항)
    "mug": ["mug", "coffee cup", "cup", "머그", "머그컵", "컵"],
    "tumbler": ["tumbler", "thermos", "water bottle", "bottle", "텀블러", "물병"],
    "sunblock": ["sunblock", "sunscreen", "sun cream", "선크림", "썬크림"],
    "umbrella": ["umbrella", "우산"],
    "pencilcase": ["pencil case", "pencilcase", "pencil-case", "pencil pouch", "필통"],
    "cola_can": ["cola can", "coke can", "soda can", "can", "콜라캔", "캔"],
    "person": ["person", "people", "someone", "man", "woman", "student", "one is", "who ",
               "사람", "학생", "남자", "여자", "누군가"],
}
LANDMARK_VOCAB = {
    # 실측 근거: 같은 파일의 landmark 분포. 키 표기(언더스코어)가 채점 기준이므로 절대 바꾸지 말 것.
    "picnic_table": ["picnic table", "outdoor table", "table", "피크닉 테이블", "테이블", "탁자"],
    "bench": ["bench", "벤치"],
    "hydrant": ["fire hydrant", "hydrant", "소화전"],
    "vending_machine": ["vending machine", "the machine", "machine", "자판기"],
    "trash_can": ["trash can", "trashcan", "garbage can", "the bin", "bin", "쓰레기통"],
    "police_car": ["police car", "patrol car", "경찰차"],
    "taxi": ["taxi", "cab", "택시"],
    "bicycle": ["bicycle", "bike", "자전거"],
}
RELATION_VOCAB = {
    # 실측: on / near / beside 만 등장 (33문항). 나머지는 API 규격상 허용되므로 유지.
    "beside": ["beside", "leaning against", "propped against", "against", "by the", "옆"],
    "near": ["near", "next to", "close to", "around", "nearby", "근처", "주변", "가까이"],
    "behind": ["behind the", "뒤에"],
    "front": ["in front of", "front", "앞"],
    "left": ["left of", "to the left", "왼쪽"],
    "right": ["right of", "to the right", "오른쪽"],
    "on": ["on top of", "on the", "on a", "위에"],
    "above": ["above", "over the", "위쪽"],
}
SITUATION_VOCAB = {
    # 실측 명령 표현에서 추출. 라벨 경계가 모호하므로(아래 DESIGN.md) 최종 판단은 VLM+규칙 앙상블.
    "accident": ["accident", "hit by", "collision", "crash", "bike accident", "사고"],
    "emergency": ["collapsed", "motionless", "immobile", "unconscious", "not moving", "bleeding",
                  "injured", "need help", "needs help", "fallen on the ground", "face-down",
                  "쓰러", "의식", "움직이지", "다쳤", "응급"],
    "abnormal": ["climbed", "on top of", "up where", "should not be", "suspicious", "strange",
                 "fighting", "이상", "수상", "올라가"],
}

# 텍스트 프롬프트로 탐지기에 넣을 때 쓰는 표현 (Grounding DINO 는 '.' 으로 구분)
DET_PHRASE = {
    "mug": "mug", "tumbler": "tumbler bottle", "sunblock": "sunscreen bottle", "umbrella": "umbrella",
    "pencilcase": "pencil case", "cola_can": "soda can", "person": "person",
    "picnic_table": "picnic table", "bench": "bench", "hydrant": "fire hydrant",
    "vending_machine": "vending machine", "trash_can": "trash can", "police_car": "police car",
    "taxi": "taxi", "bicycle": "bicycle",
}

# 대략적 높이(m). GT 가 '중심' 으로 판명될 때만 사용. 기본 파이프라인은 바닥점(높이 0) 사용.
# 랜드마크 '상면' 높이(m, 지면 기준). relation == on 일 때 target z 를 이 평면으로 올린다.
# 실측 근거(mock_demo_data): picnic_table 위 머그 +1.09, bench 위 텀블러 +0.37, vending_machine 위 사람 +1.74,
# trash_can 위 사람 +0.25(앉음). 값은 GT z - anchor z 로 역산한 것이며 자산이 바뀌면 재실측.
LANDMARK_TOP_M = {"picnic_table": 1.09, "bench": 0.37, "vending_machine": 1.74, "trash_can": 0.9,
                  "police_car": 1.4, "taxi": 1.4, "hydrant": 0.7, "bicycle": 0.9}
OBJECT_HEIGHT_M = {"person": 1.7}


# ----------------------------------------------------------------------------
# 명령 해석
# ----------------------------------------------------------------------------
@dataclass
class Interpretation:
    target_type: str = ""
    landmark: str = ""
    relation: Optional[str] = None
    situation: Optional[str] = None
    confidence: float = 0.0
    source: str = "rule"

    @property
    def is_person(self) -> bool:
        return self.target_type == "person"


def _match_vocab(text: str, vocab: Dict[str, List[str]]) -> List[Tuple[str, int]]:
    """text 안에서 vocab 항목을 찾아 (key, 등장위치) 목록 반환. 긴 표현 우선."""
    hits = []
    t = text.lower()
    for key, phrases in vocab.items():
        for p in sorted(phrases, key=len, reverse=True):
            m = re.search(r"(?<![a-z])" + re.escape(p.lower()) + r"(?![a-z])", t) if re.match(r"^[a-z]", p) \
                else re.search(re.escape(p), t)
            if m:
                hits.append((key, m.start()))
                break
    return sorted(hits, key=lambda x: x[1])


class RuleParser:
    """결정적 규칙 파서. VLM 이 없어도 흔한 문형은 처리한다."""

    def parse(self, command: str) -> Interpretation:
        it = Interpretation(source="rule")
        objs = _match_vocab(command, OBJECT_VOCAB)
        lms = _match_vocab(command, LANDMARK_VOCAB)
        rels = _match_vocab(command, RELATION_VOCAB)
        sits = _match_vocab(command, SITUATION_VOCAB)

        # 대상: 사람 표현이 있으면 사람, 아니면 첫 물건
        target_pos = None
        if any(k == "person" for k, _ in objs):
            it.target_type = "person"
            target_pos = next(p for k, p in objs if k == "person")
        elif objs:
            it.target_type, target_pos = objs[0]

        lm_hits = [(k, p) for k, p in lms if k != it.target_type]

        # 관계어-랜드마크 결합: relation 은 스펙상 target↔landmark 관계이므로, 관계어 바로
        # 뒤(<25자)에 랜드마크가 따라오는 쌍만 관계로 인정한다. "on the road/ground" 처럼
        # 랜드마크가 붙지 않은 관계어를 첫 등장 순으로 잡으면 'on' 오인 → z 보정 오류가 난다.
        bound = []
        for rk, rp in rels:
            for lk, lp in lm_hits:
                if 0 < lp - rp < 25:
                    bound.append((rk, rp, lk, lp))
        if bound:
            # 대상 언급 뒤에 오는 랜드마크 우선 ("Near the taxi, ... person on the bench" → bench).
            # 같은 랜드마크에 관계어가 여럿 결합되면 랜드마크에 가장 가까운 관계어를 채택
            # ("on the road next to the taxi" → on 이 아니라 next to).
            after = [b for b in bound if target_pos is None or b[3] > target_pos]
            rk, _, lk, _ = min(after or bound, key=lambda b: (b[3], b[3] - b[1]))
            it.relation, it.landmark = rk, lk
        elif lm_hits:
            it.landmark = lm_hits[0][0]
            # 결합된 쌍이 없으면 종전대로 첫 관계어, 그것도 없으면 near 로 가정
            it.relation = rels[0][0] if rels else "near"

        if it.is_person:
            it.relation = None
            keys = [k for k, _ in sits]
            # 우선순위: 'accident'(사고 명시) > 'emergency'(쓰러짐/무반응) > 'abnormal'(올라감 등)
            it.situation = next((k for k in ("accident", "emergency", "abnormal") if k in keys), "normal")
        it.confidence = 0.9 if it.target_type else 0.0
        return it


_VLM_PROMPT = """You are parsing a natural-language request for a campus CCTV search system.
Extract a JSON object with exactly these keys:
 "target_type": one of {objects} (use "person" for any human),
 "landmark": one of {landmarks} or "" if none,
 "relation": one of {relations} or null,
 "situation": one of ["accident","emergency","abnormal","normal"] if target_type is "person", else null.
Respond with JSON only, no prose.
Request: "{command}" """


class VLMParser:
    """
    Qwen2.5-VL 텍스트 모드로 명령을 JSON 으로 파싱. 이미지 없이 텍스트만 넣는다.
    (같은 모델 인스턴스를 SituationClassifier 가 이미지와 함께 재사용한다 → VRAM 1회 로드)
    """

    def __init__(self, backend: "QwenVLBackend"):
        self.backend = backend

    def parse(self, command: str) -> Interpretation:
        prompt = _VLM_PROMPT.format(
            objects=json.dumps(list(OBJECT_VOCAB)), landmarks=json.dumps(list(LANDMARK_VOCAB)),
            relations=json.dumps(list(RELATIONS)), command=command.replace('"', "'"),
        )
        txt = self.backend.generate(prompt, image=None, max_new_tokens=96)
        it = Interpretation(source="vlm")
        try:
            js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            it.target_type = str(js.get("target_type", "") or "")
            it.landmark = str(js.get("landmark", "") or "")
            rel = js.get("relation")
            it.relation = rel if rel in RELATIONS else None
            sit = js.get("situation")
            it.situation = sit if sit in ("accident", "emergency", "abnormal", "normal") else None
            it.confidence = 0.8
        except Exception as e:                       # noqa: BLE001
            log.warning("VLM parse failed: %s | raw=%r", e, txt[:200])
        return it


class CommandParser:
    """규칙 → VLM 순으로 시도하고 둘을 합친다. 규칙이 확실하면 VLM 을 생략해 시간을 아낀다."""

    def __init__(self, vlm: Optional[VLMParser]):
        self.rule = RuleParser()
        self.vlm = vlm

    def parse(self, command: str, time_budget_s: float = 10.0) -> Interpretation:
        r = self.rule.parse(command)
        # 서사형(hf) 명령은 규칙만으로 부족하다("the machine", "outdoor table") → VLM 을 주력으로, 규칙은 검증/폴백.
        explicit = command.strip().lower().startswith(("find ", "locate ", "where is", "찾아"))
        if self.vlm is None or time_budget_s < 3.0 or (explicit and r.target_type and r.landmark):
            return r
        v = self.vlm.parse(command)
        if not v.target_type:
            return r
        # 병합: VLM 값을 기본으로, 규칙이 '명시적 어휘' 로 잡은 칸은 규칙 우선
        r.target_type = r.target_type if (r.target_type and explicit) else (v.target_type or r.target_type)
        r.landmark = r.landmark if (r.landmark and explicit) else (v.landmark or r.landmark)
        r.relation = None if r.is_person else (r.relation or v.relation or ("near" if r.landmark else None))
        if r.is_person:
            r.situation = v.situation if v.situation and v.situation != "normal" else (r.situation or v.situation or "normal")
        r.source = "vlm+rule"
        return r


# ----------------------------------------------------------------------------
# VLM 백엔드 (Qwen2.5-VL, transformers)
# ----------------------------------------------------------------------------
class QwenVLBackend:
    def __init__(self, model_name: Optional[str] = None, use_4bit: Optional[bool] = None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_name = model_name or os.getenv("MARC_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        use_4bit = (os.getenv("MARC_VLM_4BIT", "0") == "1") if use_4bit is None else use_4bit
        kw = dict(torch_dtype=torch.bfloat16, device_map="cuda")
        if use_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        t0 = time.time()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        log.info("VLM %s loaded in %.1fs (4bit=%s)", self.model_name, time.time() - t0, use_4bit)

    def generate(self, prompt: str, image: Optional[np.ndarray], max_new_tokens: int = 128) -> str:
        import torch
        from PIL import Image

        content = []
        pil = None
        if image is not None:
            pil = Image.fromarray(image)
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


class SituationClassifier:
    """사람 crop(+주변 여백) 을 VLM 에 보여주고 situation 을 고른다."""
    PROMPT = ("Look at the person in this CCTV crop. Classify the situation as exactly one word from: "
              "accident (fallen, lying on the ground, hit), emergency (injured, needs urgent help, "
              "waving for help), abnormal (suspicious/odd behaviour, fighting, loitering), normal. "
              "Answer with one word only.")

    def __init__(self, backend: QwenVLBackend):
        self.backend = backend

    def classify(self, img: np.ndarray, bbox_xyxy: Sequence[float], pad: float = 0.6) -> str:
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2 - x1, y2 - y1
        H, W = img.shape[:2]
        cx1, cy1 = max(0, int(x1 - pad * w)), max(0, int(y1 - pad * h))
        cx2, cy2 = min(W, int(x2 + pad * w)), min(H, int(y2 + pad * h))
        crop = img[cy1:cy2, cx1:cx2]
        txt = self.backend.generate(self.PROMPT, crop, max_new_tokens=8).strip().lower()
        for k in ("accident", "emergency", "abnormal", "normal"):
            if k in txt:
                return k
        return "normal"


# ----------------------------------------------------------------------------
# 탐지기
# ----------------------------------------------------------------------------
@dataclass
class Detection:
    label: str                 # 우리 어휘 키 (cola_can, bench, ...)
    score: float
    bbox: Tuple[float, float, float, float]   # xyxy 픽셀
    mask: Optional[np.ndarray] = None


class GroundingDinoDetector:
    def __init__(self, model_name: Optional[str] = None, box_thr: float = 0.3, text_thr: float = 0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_name = model_name or os.getenv("MARC_DET_MODEL", "IDEA-Research/grounding-dino-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device).eval()
        self.box_thr, self.text_thr = box_thr, text_thr

    def detect(self, img: np.ndarray, labels: Sequence[str]) -> List[Detection]:
        import torch
        from PIL import Image

        phrases = [DET_PHRASE.get(l, l.replace("_", " ")) for l in labels]
        text = ". ".join(phrases) + "."
        pil = Image.fromarray(img)
        inputs = self.processor(images=pil, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[pil.size[::-1]])[0]
        dets: List[Detection] = []
        # transformers 버전에 따라 키가 'labels'(str) 또는 'text_labels' 이다 → 둘 다 처리
        txt_labels = res.get("text_labels", res.get("labels"))
        for box, score, tl in zip(res["boxes"], res["scores"], txt_labels):
            tl = str(tl).lower()
            key = None
            for l, p in zip(labels, phrases):
                if p.split()[0] in tl or tl in p:
                    key = l
                    break
            if key is None:
                continue
            dets.append(Detection(key, float(score), tuple(float(v) for v in box.tolist())))
        return dets


class YoloDetector:
    """데이터셋 생성기로 파인튜닝한 ultralytics 모델. 클래스명은 우리 어휘 키와 같아야 한다."""

    def __init__(self, weights: str, conf: float = 0.35):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, img: np.ndarray, labels: Sequence[str]) -> List[Detection]:
        r = self.model.predict(img, conf=self.conf, verbose=False)[0]
        names = r.names
        want = set(labels)
        dets = []
        for b in r.boxes:
            lab = names[int(b.cls)]
            if lab in want:
                dets.append(Detection(lab, float(b.conf), tuple(float(v) for v in b.xyxy[0].tolist())))
        return dets


class HybridDetector:
    """YOLO(있으면) 먼저, 못 찾은 라벨만 Grounding DINO 로."""

    def __init__(self, gdino: GroundingDinoDetector, yolo: Optional[YoloDetector]):
        self.gdino, self.yolo = gdino, yolo

    def detect(self, img: np.ndarray, labels: Sequence[str]) -> List[Detection]:
        dets: List[Detection] = []
        remaining = list(labels)
        if self.yolo is not None:
            dets = self.yolo.detect(img, labels)
            found = {d.label for d in dets}
            remaining = [l for l in labels if l not in found]
        if remaining:
            dets += self.gdino.detect(img, remaining)
        return dets


# ----------------------------------------------------------------------------
# 파이프라인
# ----------------------------------------------------------------------------
@dataclass
class CameraObservation:
    cam: CameraModel
    image: np.ndarray          # HxWx3 rgb8


@dataclass
class Stage1Config:
    time_margin_s: float = 3.0         # 제한시간에서 이만큼 남기고 제출
    max_cameras_vlm: int = 1           # situation 분류는 최종 카메라 1대만
    relation_weight: float = 0.35      # 카메라 선택 시 관계 일관성 가중치


class Stage1Pipeline:
    def __init__(self, detector: HybridDetector, parser: CommandParser,
                 situation: Optional[SituationClassifier], cfg: Stage1Config = Stage1Config(),
                 now=None):
        self.det, self.parser, self.sit, self.cfg = detector, parser, situation, cfg
        # 시각 소스. 대회 런타임에서는 client.now_s (시뮬레이션 시간 /clock) 를 넘긴다.
        # RTF<1 일 때 벽시계로 데드라인을 재면 실제보다 훨씬 일찍 끊긴다.
        self.now = now or time.time

    # ---- 카메라 1대에 대한 후보 평가 ----
    def _evaluate_camera(self, obs: CameraObservation, it: Interpretation) -> Optional[dict]:
        labels = [it.target_type] + ([it.landmark] if it.landmark else list(LANDMARK_VOCAB))
        dets = self.det.detect(obs.image, labels)
        targets = [d for d in dets if d.label == it.target_type]
        if not targets:
            return None
        cam = obs.cam
        if it.landmark:
            anchors = [d for d in dets if d.label == it.landmark]
        else:
            # 명령에 랜드마크가 없다(실측: "collapsed on the parking lot floor" 의 GT landmark 는 hydrant).
            # 채점은 landmark/anchor 를 요구하므로, 대상에서 가장 가까운 랜드마크를 anchor 로 채택한다.
            lms = [d for d in dets if d.label in LANDMARK_VOCAB]
            t0 = max(targets, key=lambda d: d.score)
            T0 = detection_to_world(cam, t0.bbox, t0.mask)
            anchors = []
            if lms and T0 is not None:
                def _dist(d):
                    A0 = detection_to_world(cam, d.bbox, d.mask)
                    return np.linalg.norm((A0 - T0)[:2]) if A0 is not None else 1e9
                nearest = min(lms, key=_dist)
                if _dist(nearest) < 6.0:
                    anchors = [nearest]
                    it.landmark = nearest.label
                    log.info("landmark inferred from image: %s (%.1fm)", nearest.label, _dist(nearest))

        best = None
        if it.landmark and anchors:
            for a in anchors:
                A = detection_to_world(cam, a.bbox, a.mask)
                if A is None:
                    continue
                for t in targets:
                    T = detection_to_world(cam, t.bbox, t.mask)
                    if T is None:
                        continue
                    rs = relation_scores(cam, A, T)
                    rel_ok = rs.get(it.relation, 0.0) if it.relation else max(rs.values())
                    # 후보 점수: 탐지 신뢰도 + 관계 일관성
                    s = (1 - self.cfg.relation_weight) * (0.5 * a.score + 0.5 * t.score) \
                        + self.cfg.relation_weight * rel_ok
                    if best is None or s > best["score"]:
                        best = dict(score=s, target=t, anchor=a, A=A, T=T, rel_scores=rs)
        else:
            # 랜드마크 없음(사람 문제 등) 또는 랜드마크 미탐지 → 대상 최고 신뢰도
            t = max(targets, key=lambda d: d.score)
            T = detection_to_world(cam, t.bbox, t.mask)
            if T is None:
                return None
            penalty = 0.6 if it.landmark else 1.0       # 랜드마크를 못 찾은 카메라는 감점
            best = dict(score=t.score * penalty, target=t, anchor=None, A=None, T=T, rel_scores={})
        best["cam"] = cam
        best["image"] = obs.image
        return best

    # ---- 전체 ----
    def run(self, command: str, observations: List[CameraObservation], time_limit_s: float) -> dict:
        t_start = self.now()
        deadline = t_start + max(1.0, time_limit_s - self.cfg.time_margin_s)

        it = self.parser.parse(command, time_budget_s=deadline - self.now())
        log.info("interpretation: %s", it)
        if not it.target_type:
            it.target_type = "person" if re.search(r"person|people|사람|학생", command, re.I) else "cola_can"
            log.warning("target_type fallback → %s", it.target_type)

        candidates = []
        for obs in observations:
            if self.now() > deadline:
                log.warning("deadline reached while scanning cameras; stopping at %d/%d",
                            len(candidates), len(observations))
                break
            try:
                c = self._evaluate_camera(obs, it)
            except Exception as e:                  # noqa: BLE001
                log.exception("camera %s failed: %s", obs.cam.camera_id, e)
                c = None
            if c:
                candidates.append(c)
                log.info("cam %s score %.3f target %.2f anchor %s",
                         obs.cam.camera_id, c["score"], c["target"].score,
                         f"{c['anchor'].score:.2f}" if c["anchor"] else "-")

        if not candidates:
            return self._fallback_payload(it, observations)

        best = max(candidates, key=lambda c: c["score"])
        cam = best["cam"]

        # relation 이 파서에서 안 나왔으면 기하로 채움
        relation = it.relation
        if it.landmark and best["anchor"] is not None and not relation:
            relation = max(best["rel_scores"], key=best["rel_scores"].get)

        # situation (사람 문제만, 시간 남을 때만)
        situation = it.situation
        if it.is_person:
            if self.sit is not None and self.now() < deadline - 2.0:
                try:
                    situation = self.sit.classify(best["image"], best["target"].bbox)
                except Exception as e:              # noqa: BLE001
                    log.warning("situation VLM failed: %s", e)
            situation = situation or "normal"

        # 'on' 관계면 대상 z 를 랜드마크 높이로 보정 (bbox 바닥점이 지면이 아니라 랜드마크 상면이므로)
        T = best["T"]
        on_top = relation in ("on", "above") or (it.is_person and re.search(
            r"on top of|climbed|lying on (top of )?the (bench|table|picnic)|sitting on", command, re.I))
        if on_top and best["A"] is not None and it.landmark in LANDMARK_TOP_M:
            T = detection_to_world(cam, best["target"].bbox, best["target"].mask,
                                   stands_on_z=cam.ground_z + LANDMARK_TOP_M[it.landmark]) or T

        A = best["A"] if best["A"] is not None else T      # 랜드마크 없는 문제: anchor = target 로 채움
        payload = {
            "camera_id": cam.camera_id,
            "interpretation": {
                "target_type": it.target_type,
                "landmark": it.landmark,
                "relation": (relation if (it.landmark and not it.is_person) else None),
                "situation": situation if it.is_person else None,
            },
            "grounding": {
                "anchor_coord": [round(float(v), 3) for v in A],
                "target_coord": [round(float(v), 3) for v in T],
            },
        }
        log.info("payload (%.1fs sim): %s", self.now() - t_start, json.dumps(payload))
        return payload

    def _fallback_payload(self, it: Interpretation, observations: List[CameraObservation]) -> dict:
        """
        아무 카메라에서도 대상을 못 찾음. 무제출보다 부분점수 기대값이 높으므로
        첫 카메라의 화면 중앙 지면점을 낸다. (카메라 선택·해석 항목은 맞을 수 있음)
        """
        from .geometry import pixel_to_world
        cam = observations[0].cam
        P = pixel_to_world(cam, cam.width / 2, cam.height * 0.75)
        P = [round(float(v), 3) for v in (P if P is not None else cam.C_w)]
        log.error("no detection in any camera → fallback payload")
        return {
            "camera_id": cam.camera_id,
            "interpretation": {"target_type": it.target_type, "landmark": it.landmark,
                               "relation": (it.relation if (it.landmark and not it.is_person) else None),
                               "situation": (it.situation or "normal") if it.is_person else None},
            "grounding": {"anchor_coord": P, "target_coord": P},
        }


# ----------------------------------------------------------------------------
# 팩토리
# ----------------------------------------------------------------------------
def build_pipeline(now=None) -> Stage1Pipeline:
    """환경변수에 따라 구성. 모델 로드는 여기서 1회 (connect() 전에 호출해 워밍업)."""
    gdino = GroundingDinoDetector()
    yolo_w = os.getenv("MARC_YOLO_WEIGHTS")
    yolo = YoloDetector(yolo_w) if yolo_w and os.path.exists(yolo_w) else None
    if yolo_w and yolo is None:
        log.warning("MARC_YOLO_WEIGHTS=%s not found; Grounding DINO only", yolo_w)
    detector = HybridDetector(gdino, yolo)

    vlm_parser, sit = None, None
    if os.getenv("MARC_DISABLE_VLM", "0") != "1":
        backend = QwenVLBackend()
        vlm_parser, sit = VLMParser(backend), SituationClassifier(backend)
    return Stage1Pipeline(detector, CommandParser(vlm_parser), sit, now=now)
