"""시나리오 yaml(`marc2026_chungmu.yaml`)에서 정답지를 직접 만들어낸다.

과거 라이브 기록(`runs/chungmu_full39_2026-08-25.jsonl`)은 그날 실제로 출제된 문항의
정답만 담고 있다. 이 모듈은 같은 정답을 **시나리오 파일에서 유도**하므로, 기록에 없는
문항(문항을 새로 켜거나 시나리오가 갱신된 경우)도 오프라인 채점이 가능해진다.
`offline_score.py` 는 jsonl 정답지를 우선 쓰고, 거기 없는 voice_command 만 여기서 채운다.

## yaml → 정답 유도 규칙 (39문항 전수 대조로 검증)

`challenge_problems[].expected` 에는 좌표가 없다. `missions[]` 에서 조인해야 한다:

* **target_coord** — `target_object_id` 의 끝 숫자를
  `missions[camera_id].objects[target_type].resolved.instances` 의 **인덱스**로 쓴다.
  prim 이름으로 매칭하면 안 된다: rt01 의 `target_object_id` 는 `rig_7_a/tumbler_2` 인데
  실제 prim 은 `/world/trash/rig_7_a_tumbler_4` 다(이름의 숫자와 순서가 어긋난다).
  인덱스 규칙으로는 39/39 가 일치한다(오차 ≤ 0.081m, 대부분 0.000m).
* **anchor_coord** — `missions[camera_id].landmarks[landmark_name].resolved.position`.
  39문항 중 33문항이 0.000m 로 정확히 일치한다. 예외는 아래 ANCHOR_OVERRIDE 참고.
* **landmark_prefix** — `landmark_name` 에서 뒤쪽 인스턴스 번호를 뗀다
  (`police_car_01` → `police_car`). 채점은 이 prefix 로 한다.
* **owner_position** — `owner_zone.resolved.position` (Stage2 배달 목적지).

카테고리 필드(camera_id / target_kind / target_type / relation / situation)는 yaml 값을
그대로 쓰면 되고, 39문항 전부 플랫폼 정답과 불일치 0건이었다.
"""

from __future__ import annotations

import os
import re

import yaml

_VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
_VLA_GROUNDING_DIR = os.path.dirname(_VALIDATION_DIR)
DEFAULT_SCENARIO = os.path.join(_VLA_GROUNDING_DIR, "content-image-2026.R03",
                                "marc2026_chungmu.yaml")

# 랜드마크 기준점 보정 -- yaml 의 `resolved.position` 과 플랫폼이 정답으로 돌려준
# anchor_coord 가 어긋나는 예외를 실측값으로 덮어쓴다.
#
# police_car_01 (rig_7_a) 하나만 xy 로 정확히 2.2934m 어긋난다:
#   yaml     [-42.65816, 118.29972, 16.69391]
#   플랫폼   [-40.98046, 116.73601, 16.69391]   (차이 [+1.6777, -1.56371, 0.0])
# z 는 소수점까지 같고, 이 랜드마크를 쓰는 6문항(lf08/lf09/hf07/hf08/rt01/rt02)의 정답
# anchor 가 전부 같은 값이라 문항별 노이즈가 아니라 **랜드마크 하나에 붙은 상수 오프셋**이다.
# 같은 차량 계열인 taxi_01 은 0.0000m 로 정확히 맞으므로 "차량 애셋 전반"의 문제도 아니다.
# 애셋의 prim 원점과 플랫폼이 쓰는 기준점(바운딩박스 중심 등)이 이 애셋에서만 다른 것으로
# 보이나 데이터만으로는 확정할 수 없어, 라이브 정답지에서 얻은 실측값을 그대로 박아둔다.
# 보정하지 않으면 저 6문항의 anchor 점수 상한이 1-2.2934/5 = 54% 로 깎인다.
ANCHOR_OVERRIDE = {
    ("rig_7_a", "police_car_01"): [-40.98046, 116.73601, 16.69391],
}

# Stage1/Stage2 grounding 공통 배점. 라이브 기록에서 39문항 전부 이 값이었다
# (Stage2 문항 rt01~rtC2 포함). 사람 문제는 플랫폼이 landmark/relation 배점을 0 으로
# 돌려주므로 target_kind 가 person 이면 그렇게 맞춘다.
OBJECT_WEIGHTS = {"camera": 0.1, "object_type": 0.15, "landmark": 0.1,
                  "relation": 0.1, "anchor": 0.15, "target": 0.4}
PERSON_WEIGHTS = {"camera": 0.1, "object_type": 0.15, "landmark": 0.0,
                  "relation": 0.0, "anchor": 0.15, "target": 0.4}


def strip_instance_index(name: str) -> str:
    """`police_car_01` → `police_car`. 채점에 쓰이는 landmark prefix."""
    return re.sub(r"_\d+$", "", name or "")


def load_gt(scenario_path: str = DEFAULT_SCENARIO) -> dict:
    """voice_command -> (problem_id, expected, weights, kind, owner_position).

    ``kind`` 는 시나리오의 문항 종류다: ``grounding``(Stage1) / ``retrieval``(Stage2).
    ``owner_position`` 은 시나리오 전체 공통(배달 목적지)이라 문항마다 같은 값이 들어간다.
    """
    with open(scenario_path, encoding="utf-8") as f:
        ch = yaml.safe_load(f)

    landmarks, objects = {}, {}
    for m in ch.get("missions", []):
        mid = m["id"]
        for lm in m.get("landmarks", []):
            landmarks[(mid, lm["name"])] = lm["resolved"]["position"]
        for ob in m.get("objects", []):
            # 같은 type 이 여러 블록으로 쪼개져 있을 수 있어 이어붙인다 (예: rig_1_a 의
            # person_sitting 이 두 번 등장). 인덱스는 이 합친 순서를 기준으로 센다.
            objects.setdefault((mid, ob["type"]), []).extend(ob["resolved"]["instances"])

    owner = (ch.get("owner_zone") or {}).get("resolved", {}).get("position")

    key = {}
    for p in ch.get("challenge_problems", []):
        e = p["expected"]
        mid = e["camera_id"]
        lname = e.get("landmark_name")

        tail = (e.get("target_object_id") or "").rsplit("_", 1)[-1]
        instances = objects.get((mid, e["target_type"]), [])
        idx = int(tail) if tail.isdigit() else None
        target = (instances[idx]["position"]
                  if idx is not None and idx < len(instances) else None)

        anchor = ANCHOR_OVERRIDE.get((mid, lname)) or landmarks.get((mid, lname))

        is_person = e.get("target_kind") == "person"
        expected = {
            "camera_id": mid,
            "landmark_name": lname,
            "landmark_prefix": strip_instance_index(lname),
            "target_kind": e.get("target_kind"),
            "target_type": e.get("target_type"),
            "relation": e.get("relation"),
            "situation": e.get("situation"),
            "anchor_coord": anchor,
            "target_coord": target,
        }
        key[p["voice_command"]] = (
            p["id"], expected,
            PERSON_WEIGHTS if is_person else OBJECT_WEIGHTS,
            p.get("kind", "grounding"), owner,
        )
    return key


def problem_kinds(scenario_path: str = DEFAULT_SCENARIO) -> dict:
    """problem_id -> kind. jsonl 정답지에는 kind 가 없어서 별도로 조인해야 한다."""
    with open(scenario_path, encoding="utf-8") as f:
        ch = yaml.safe_load(f)
    return {p["id"]: p.get("kind", "grounding")
            for p in ch.get("challenge_problems", [])}
