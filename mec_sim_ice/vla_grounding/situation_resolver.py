"""Official MARC 2026 SAR situation resolver — situation = f(pose x support).

Faithful implementation of the organizer's PUBLIC deterministic scoring rule
(stage1_scorer / Configurations.resolve_situation), transcribed 1:1 from the
content image ghcr.io/marc-challenge/marc-platform-content:2026.R03:

  scenarios/situation_rules.yaml            (version 2026.1)  — rule tables
  scenarios/people_poses_game_engine.yaml   — per-pose situation defaults

Local copies live in vla_grounding/content-image-2026.R03/ and the tables below
are verified against them by test_situation_resolver.py --verify-yaml.
GENERAL by design: these are the organizer's scoring rules themselves, not a
practice-scenario fit — they apply to the hidden competition scenario too.

Official resolution order (from situation_rules.yaml header):
  1) incapacitation: strongly incapacitated poses override any location.
  2) rules[pose_class][support_class]: deterministic when the support surface
     is known.
  3) fallback: the pose's default situation from people_poses (then
     default_situation).

Stdlib-only: safe to import inside the runtime container.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SITUATIONS = ("normal", "abnormal", "emergency", "accident")

# ── situation_rules.yaml: 자세(type) → pose_class ──────────────────────────
POSE_CLASS = {
    "person_walking":      "upright",
    "person_walking_away": "upright",
    "person_reaching":     "upright",
    "person_sitting":      "seated",
    "person_reclining":    "seated",
    "person_crouching":    "low",
    "person_bending":      "prostrate",
    "person_lying_down":   "prostrate",
    "person_prone":        "prostrate",
}

# ── situation_rules.yaml: incapacitation — 장소 무관 우선 적용 ─────────────
INCAPACITATION = {
    "person_prone": "accident",
}

# ── situation_rules.yaml: 지지면 토큰 → support_class ──────────────────────
SUPPORT_CLASS = {
    # seating
    "bench": "seating", "picnic_table": "seating", "chair": "seating",
    # rest_ground
    "grass": "rest_ground", "lawn": "rest_ground",
    # transit
    "walkway": "transit", "sidewalk": "transit", "road": "transit",
    "parking": "transit", "parking_space": "transit",
    # non_mount
    "trash_can": "non_mount", "vending_machine": "non_mount",
    "postbox": "non_mount", "hydrant": "non_mount", "car": "non_mount",
    # vehicle_fallen
    "bicycle": "vehicle_fallen", "kick_scooter": "vehicle_fallen",
}

# ── situation_rules.yaml: rules[pose_class][support_class] → situation ─────
RULES = {
    "upright": {
        "seating": "abnormal", "rest_ground": "normal", "transit": "normal",
        "non_mount": "abnormal", "vehicle_fallen": "abnormal",
    },
    "seated": {
        "seating": "normal", "rest_ground": "normal", "transit": "abnormal",
        "non_mount": "abnormal", "vehicle_fallen": "abnormal",
    },
    "low": {
        "seating": "abnormal", "rest_ground": "abnormal", "transit": "abnormal",
        "non_mount": "abnormal", "vehicle_fallen": "abnormal",
    },
    "prostrate": {
        "seating": "abnormal", "rest_ground": "normal", "transit": "emergency",
        "non_mount": "emergency", "vehicle_fallen": "emergency",
    },
}

DEFAULT_SITUATION = "normal"   # situation_rules.yaml: default_situation

# ── people_poses_game_engine.yaml: 자세별 situation 기본값 (fallback 소스) ─
POSE_DEFAULT_SITUATION = {
    "person_walking":      "normal",
    "person_reaching":     "normal",
    "person_sitting":      "normal",
    "person_walking_away": "normal",
    "person_reclining":    "normal",
    "person_crouching":    "abnormal",
    "person_lying_down":   "abnormal",
    "person_bending":      "emergency",
    "person_prone":        "accident",
}

# ── 우리 쪽 정규화 (공식 표 밖의 보조 계층) ────────────────────────────────
# config.yaml: "시나리오의 landmark name 접두사로 에셋 매칭 (예: bench_01 → bench)"
# → 인스턴스 접미사(_01 등)를 벗긴다. 차량 변형 자산은 support_class 에 개별 키가
#   없고 "car" 토큰만 있으므로 car 로 정규화한다.
_CAR_TYPES = {"police_car", "normal_car_1", "normal_car_2",
              "sports_car", "sports_car_2", "suv", "taxi"}


def normalize_support_token(token: str | None) -> str | None:
    """Landmark instance name / surface token -> official support token."""
    if not token:
        return None
    t = re.sub(r"_\d+$", "", token.strip().lower())   # bench_01 -> bench
    if t in _CAR_TYPES or re.sub(r"_\d+$", "", t) in _CAR_TYPES:
        return "car"
    return t


@dataclass
class SituationDecision:
    situation: str
    path: str        # "incapacitation" | "rules" | "pose-default" | "default"
    pose_type: str
    pose_class: str | None
    support_token: str | None    # normalized token actually used (None if unused)
    support_class: str | None


def resolve_situation(pose_type: str, support_token: str | None = None) -> SituationDecision:
    """Official resolution order: incapacitation -> pose x support -> fallback.

    ``pose_type`` must be one of the 9 official person_* pose classes; an
    unknown value goes to the fallback path (default_situation) rather than
    raising, so a detector mislabel degrades gracefully instead of crashing a
    round. ``support_token`` may be a raw landmark instance name (bench_01) or
    a ground token; it is normalized here.
    """
    pose = pose_type.strip().lower() if pose_type else ""
    norm = normalize_support_token(support_token)
    pose_class = POSE_CLASS.get(pose)
    support_class = SUPPORT_CLASS.get(norm) if norm else None

    # 1) incapacitation — 장소 무관
    if pose in INCAPACITATION:
        return SituationDecision(INCAPACITATION[pose], "incapacitation",
                                 pose, pose_class, None, None)

    # 2) 결정론적 pose x support 매핑 (지지면이 표의 토큰으로 확정된 경우)
    if pose_class and support_class:
        return SituationDecision(RULES[pose_class][support_class], "rules",
                                 pose, pose_class, norm, support_class)

    # 3) fallback — people_poses 의 자세별 기본값, 그것도 없으면 default
    if pose in POSE_DEFAULT_SITUATION:
        return SituationDecision(POSE_DEFAULT_SITUATION[pose], "pose-default",
                                 pose, pose_class, norm, None)
    return SituationDecision(DEFAULT_SITUATION, "default",
                             pose, pose_class, norm, None)


def verify_against_yaml(scenario_dir: str) -> list[str]:
    """Compare the embedded tables with the extracted official YAML files.

    Returns a list of mismatch descriptions (empty = in sync). Requires PyYAML;
    meant for tests/CI on the dev machine, never called at runtime.
    """
    import os
    import yaml

    problems: list[str] = []
    with open(os.path.join(scenario_dir, "situation_rules.yaml")) as f:
        official = yaml.safe_load(f)
    for name, ours in (("pose_class", POSE_CLASS),
                       ("incapacitation", INCAPACITATION),
                       ("support_class", SUPPORT_CLASS),
                       ("rules", RULES)):
        if official.get(name) != ours:
            problems.append(f"{name} differs from situation_rules.yaml")
    if official.get("default_situation") != DEFAULT_SITUATION:
        problems.append("default_situation differs from situation_rules.yaml")

    with open(os.path.join(scenario_dir, "people_poses_game_engine.yaml")) as f:
        poses = yaml.safe_load(f)
    official_defaults = {k: v["situation"] for k, v in poses["poses"].items()}
    if official_defaults != POSE_DEFAULT_SITUATION:
        problems.append("pose defaults differ from people_poses_game_engine.yaml")
    return problems


if __name__ == "__main__":
    import sys
    pose = sys.argv[1] if len(sys.argv) > 1 else "person_lying_down"
    support = sys.argv[2] if len(sys.argv) > 2 else None
    d = resolve_situation(pose, support)
    print(f"{d.pose_type} on {d.support_token or '(unknown)'} -> {d.situation}"
          f"  [path={d.path}, pose_class={d.pose_class}, support_class={d.support_class}]")
