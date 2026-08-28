"""Text parsing (closed vocabulary) for MARC 2026 Stage 1/2 voice commands.

GENERAL by design: the vocabulary is the OFFICIAL asset roster (config.yaml
mission_object_assets / landmark_assets from the content image), not any one
practice scenario. Nothing here references specific problem ids, scene
layouts, instances, or coordinates — chungmu is used only as a test corpus.

Extracts what TEXT can reliably decide and leaves the rest explicitly
undecided for the vision/geometry layer (grounding.py):

  target_type       - object class or "person" (None = out-of-vocab -> caller
                      falls back to open-vocab detection)
  landmark_type     - type-level hint only ("machine" -> vending_machine);
                      None = command names no landmark -> resolve visually
                      (nearest landmark to the found target in 3D)
  relation_hint     - relation_confident=True: wording is decisive ("on the
                      bench") and can be submitted as-is. False: ambiguous
                      ("next to" maps to near OR beside in real data) ->
                      geometry decides.
  is_person / pose_hint - SAR: coarse pose bucket to pick the right person
                      among several (situation itself comes from
                      situation_resolver, never from text)
  stage2            - "Pick up ... and bring it to the owner" phrasing

Stdlib-only (re, dataclasses).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── 어휘: 공식 클래스 → 표면형(동의어/환유). config.yaml 로스터 기준 ───────
# (long_umbrella 는 주최측이 삭제 확정 → 제외)
OBJECT_SYNONYMS = {
    "mug":            ["mug", "coffee mug", "coffee cup"],
    "tumbler":        ["tumbler", "thermos", "water bottle", "flask"],
    "sunblock":       ["sunblock", "sunscreen", "sun cream", "suncream"],
    "umbrella":       ["umbrella", "parasol", "brolly"],
    "pencilcase":     ["pencil case", "pencilcase", "pencil pouch", "pencil box"],
    "cola_can":       ["cola can", "cola", "coke can", "coke", "soda can", "drink can"],
    "disposable_cup": ["disposable cup", "paper cup", "plastic cup", "takeaway cup"],
    "juice":          ["juice box", "juice bottle", "juice pack", "juice"],
    "tissue":         ["tissue", "tissues", "tissue pack", "napkin"],
    "cracker_box":    ["cracker box", "box of crackers", "snack box", "crackers"],
    "master_chef_can": ["food can", "tin can", "canned food"],
    "wood_block":     ["wood block", "wooden block", "block of wood"],
}

# "car" 는 제너릭: 비전 계층이 차량 변형(police_car/taxi/suv/...) 중 target 에
# 실제 인접한 인스턴스로 좁힌다.
LANDMARK_SYNONYMS = {
    "picnic_table":    ["picnic table", "outdoor table", "table"],
    "bench":           ["bench"],
    "hydrant":         ["fire hydrant", "hydrant"],
    "vending_machine": ["vending machine", "drink machine", "drinks machine",
                        "beverage machine", "machine"],
    "trash_can":       ["trash can", "trashcan", "trash bin", "garbage can",
                        "garbage bin", "rubbish bin", "bin"],
    "bicycle":         ["bicycle", "bike"],
    "kick_scooter":    ["kick scooter", "kickboard", "scooter"],
    "postbox":         ["postbox", "post box", "mailbox"],
    "taxi":            ["taxi", "cab"],
    "police_car":      ["police car", "patrol car"],
    "car":             ["car", "vehicle", "suv", "sports car"],
}

PERSON_WORDS = ("person", "people", "someone", "somebody", "anyone",
                "man", "woman", "student", "professor")

# pose 코스 버킷 — 비전 계층이 공식 9종 pose 클래스 후보 집합으로 매핑한다.
# (실데이터에서 "lying face-down" 이 person_bending 으로 라벨링되는 등 표면형과
#  pose 클래스가 1:1 이 아니므로 버킷은 단정하지 않는다.)
POSE_BUCKETS = [
    ("facedown", r"face[- ]?down|facedown|prone"),
    ("bent",     r"bent over|hunched|doubled over|bending"),
    ("down",     r"collapsed|fallen|fell|is down\b|on the ground|on the floor"),
    ("lying",    r"lying|laying|lie(s)? down|reclin"),
    ("sitting",  r"sitting|seated|sat |climbed"),
    ("standing", r"standing|walking"),
]

AREA_HINTS = [
    ("parking", r"parking lot|parking space|parking area|car park"),
    ("park",    r"\bpark\b|rest area|on the grass|lawn"),
    ("road",    r"\broad\b|\bstreet\b|sidewalk|walkway"),
]


@dataclass
class ParsedCommand:
    target_type: str | None          # 오브젝트 클래스 또는 "person"
    is_person: bool
    landmark_type: str | None        # 클래스 힌트("car"=제너릭) / None=비전이 확정
    relation_hint: str | None        # 8종 중 텍스트로 읽힌 것 (사람 문제는 무시됨)
    relation_confident: bool         # True: 그대로 제출 가능 / False: 기하가 확정
    pose_hint: str | None            # 사람 고르기용 코스 버킷 (situation 판정 아님)
    stage2: bool
    area_hint: str | None            # 카메라 우선순위 힌트
    cues: dict = field(default_factory=dict)   # 디버깅: 무엇이 어디서 매칭됐나


def _phrase_pattern(phrase: str) -> str:
    return r"\b" + r"\s+".join(map(re.escape, phrase.split())) + r"\b"


def _find_mentions(text: str, table: dict) -> list[tuple[int, str, str]]:
    """[(match_start, class_key, surface), ...] — 긴 표면형 우선, 겹침 제거."""
    hits = []
    for key, phrases in table.items():
        for p in sorted(phrases, key=len, reverse=True):
            for m in re.finditer(_phrase_pattern(p), text):
                hits.append((m.start(), m.end(), key, p))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out, covered = [], []
    for s, e, key, p in hits:
        if any(s < ce and e > cs for cs, ce in covered):
            continue                      # 겹침("trash can" 안의 "can" 등) 제거
        covered.append((s, e))
        out.append((s, key, p))
    return out


def parse_command(command: str) -> ParsedCommand:
    text = re.sub(r"\s+", " ", command.lower()).strip()
    cues: dict = {}

    stage2 = bool(re.search(r"\bpick up\b|\bbring it to\b|\bbring them to\b", text))

    person_word_hit = any(re.search(r"\b" + w + r"\b", text) for w in PERSON_WORDS)
    objs = _find_mentions(text, OBJECT_SYNONYMS)
    cues["object_mentions"] = [(k, p) for _, k, p in objs]
    pose_cue = any(re.search(pat, text) for _, pat in POSE_BUCKETS)
    # "someone/a person" 같은 단어가 있다고 무조건 사람 문제로 확정하지 않는다.
    # 서사형 분실물 문제도 3인칭 화자로 쓰일 수 있다("Someone dropped their
    # tumbler near the bike rack." -- 타겟은 tumbler지 사람이 아님). 자세
    # 단서(sar/sarH의 특징, 예: "collapsed", "climbed")가 있거나 아는 물건이
    # 아예 문장에 없을 때만 사람 문제로 본다 -- 그 외엔 물건 쪽이 진짜 타겟이다.
    is_person = person_word_hit and (pose_cue or not objs)

    # ── target ──
    if is_person:
        target = "person"
    elif not objs:
        target = None                 # 어휘 밖 → 상위에서 DINO/VLM 폴백
    elif len(objs) == 1:
        target = objs[0][1]
    else:
        # 다중 물체 언급: find/pick up/left/dropped + my/the 에 붙은 쪽 우선
        def score(hit):
            s, _, p = hit
            pre = text[max(0, s - 28):s]
            sc = 0
            if re.search(r"(find|pick up|left|dropped|lost)\b[^.;]*?(my|the)\s*$", pre):
                sc += 2
            if re.search(r"\bmy\s*$", pre):
                sc += 1
            return sc
        target = max(objs, key=score)[1]

    # ── landmark (타입 힌트) ──
    lms = _find_mentions(text, LANDMARK_SYNONYMS)
    cues["landmark_mentions"] = [(k, p) for _, k, p in lms]
    target_pos = None
    if target and target != "person":
        for s, k, _ in _find_mentions(text, {target: OBJECT_SYNONYMS[target]}):
            target_pos = s
            break
    elif is_person:
        m = re.search(r"\b(" + "|".join(PERSON_WORDS) + r")\b", text)
        target_pos = m.start() if m else None

    landmark = None
    if lms:
        def lm_score(hit):
            s, _, p = hit
            pre = text[max(0, s - 16):s]
            sc = 0
            if re.search(r"\bon (top of )?(the |a )?$", pre):
                sc += 4                   # "on (top of) the X" 가 최우선 부착
            if re.search(r"\b(near|next to|beside|against|by|from)\s+(the |a |its? )?$", pre):
                sc += 2
            if target_pos is not None and s > target_pos:
                sc += 1                   # 대상 언급 뒤에 오는 랜드마크 우선
            return sc
        best = max(lms, key=lm_score)
        if lm_score(best) > 0 or len(lms) == 1:
            landmark = best[1]

    # ── relation ──
    relation, confident = None, False
    if re.search(r"\bon top of\b", text) or \
       (landmark and re.search(r"\bon (the |a )?" + _lm_pat(landmark), text)):
        relation, confident = "on", True
    elif re.search(r"\bbehind\b(?=\s+(the|a|an|it|him|her|them|that)\b)", text):
        relation, confident = "behind", True    # "left it behind"(문말)는 공간관계 아님
    elif re.search(r"\bin front of\b", text):
        relation, confident = "front", True
    elif re.search(r"\b(to the |on the )?left of\b", text):
        relation, confident = "left", True
    elif re.search(r"\b(to the |on the )?right of\b", text):
        relation, confident = "right", True
    elif re.search(r"\babove\b|\bhanging over\b", text):
        relation, confident = "above", True
    elif re.search(r"\bbeside\b", text):
        relation, confident = "beside", True
    elif re.search(r"\bnear\b|\bclose to\b", text):
        relation, confident = "near", True
    elif re.search(r"\bnext to\b|\bby the\b", text):
        relation, confident = "near", False      # 실데이터에서 near/beside 혼재 → 기하 확정
    elif re.search(r"\bpropped against\b|\bleaning against\b|\bagainst\b", text):
        relation, confident = "beside", False    # 접촉 함의 → beside 쪽 힌트
    elif re.search(r"\bdropped\b|\bon the ground\b|\bon the floor\b", text):
        relation, confident = "near", False
    elif landmark and re.search(r"\bat (an? |the )?" + _lm_pat(landmark), text) and \
            re.search(r"\bleft\b.*\b(there|behind)\b|\bleft (my|the)\b", text):
        relation, confident = "on", False        # "at the table ... left it there"
    cues["relation_source"] = relation

    # ── pose / area 힌트 ──
    pose_hint = None
    if is_person:
        for bucket, pat in POSE_BUCKETS:
            if re.search(pat, text):
                pose_hint = bucket
                break
    area_hint = None
    for area, pat in AREA_HINTS:
        if re.search(pat, text):
            area_hint = area
            break

    return ParsedCommand(
        target_type=target, is_person=is_person, landmark_type=landmark,
        relation_hint=relation, relation_confident=confident,
        pose_hint=pose_hint, stage2=stage2, area_hint=area_hint, cues=cues,
    )


def _lm_pat(landmark: str) -> str:
    return "(" + "|".join(_phrase_pattern(p)
                          for p in LANDMARK_SYNONYMS[landmark]) + ")"


if __name__ == "__main__":
    import sys
    cmd = " ".join(sys.argv[1:]) or \
        "After working out I grabbed a drink from the machine and left my umbrella standing next to it."
    p = parse_command(cmd)
    print(f"target={p.target_type} person={p.is_person} landmark={p.landmark_type} "
          f"relation={p.relation_hint}(conf={p.relation_confident}) pose={p.pose_hint} "
          f"stage2={p.stage2} area={p.area_hint}")
