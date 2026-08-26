"""Tests for situation_resolver.

1. Golden cases straight from the official comments in situation_rules.yaml.
2. Exhaustive property checks (every pose x every support token resolves to a
   valid label; prone is accident everywhere; fallback = pose defaults).
3. --verify-yaml: comparison of the embedded tables with the extracted
   official YAMLs in content-image-2026.R03/ (needs PyYAML).

Runs with plain python3.
"""

from __future__ import annotations

import sys

from situation_resolver import (
    POSE_CLASS, POSE_DEFAULT_SITUATION, SITUATIONS, SUPPORT_CLASS,
    normalize_support_token, resolve_situation,
)

# (pose_type, support_token, expected situation, expected path)
GOLDEN = [
    ("person_prone", None,            "accident", "incapacitation"),
    ("person_prone", "grass",         "accident", "incapacitation"),
    ("person_prone", "bench",         "accident", "incapacitation"),
    ("person_prone", "parking",       "accident", "incapacitation"),
    ("person_sitting", "bench",        "normal",    "rules"),
    ("person_sitting", "grass",        "normal",    "rules"),
    ("person_sitting", "road",         "abnormal",  "rules"),
    ("person_sitting", "trash_can",    "abnormal",  "rules"),
    ("person_lying_down", "grass",     "normal",    "rules"),
    ("person_lying_down", "walkway",   "emergency", "rules"),
    ("person_lying_down", "bench",     "abnormal",  "rules"),
    ("person_lying_down", "bicycle",   "emergency", "rules"),
    ("person_walking", "sidewalk",     "normal",    "rules"),
    ("person_walking", "picnic_table", "abnormal",  "rules"),
    ("person_reaching", "vending_machine", "abnormal", "rules"),
    ("person_crouching", "grass",      "abnormal",  "rules"),
    ("person_bending", "parking",      "emergency", "rules"),
    ("person_bending", "grass",        "normal",    "rules"),
    ("person_sitting", "bench_01",     "normal",    "rules"),
    ("person_sitting", "taxi",         "abnormal",  "rules"),
    ("person_walking", "police_car",   "abnormal",  "rules"),
    ("person_sitting", "suv",          "abnormal",  "rules"),
    ("person_lying_down", None,        "abnormal",  "pose-default"),
    ("person_bending", None,           "emergency", "pose-default"),
    ("person_crouching", None,         "abnormal",  "pose-default"),
    ("person_sitting", None,           "normal",    "pose-default"),
    ("person_walking", "fountain",     "normal",    "pose-default"),
    ("person_handstand", "road",       "normal",    "default"),
]


def main() -> int:
    fails = 0
    for pose, support, want_sit, want_path in GOLDEN:
        d = resolve_situation(pose, support)
        if (d.situation, d.path) != (want_sit, want_path):
            print(f"FAIL: {pose} on {support} -> ({d.situation}, {d.path}), "
                  f"expected ({want_sit}, {want_path})")
            fails += 1
    print(f"golden: {len(GOLDEN) - fails}/{len(GOLDEN)}")

    supports = list(SUPPORT_CLASS) + [None, "unknown_thing"]
    checked = 0
    for pose in POSE_CLASS:
        for s in supports:
            d = resolve_situation(pose, s)
            checked += 1
            if d.situation not in SITUATIONS:
                print(f"FAIL: invalid label {d.situation} for {pose} on {s}")
                fails += 1
            if pose == "person_prone" and d.situation != "accident":
                print(f"FAIL: prone on {s} must be accident, got {d.situation}")
                fails += 1
            if s is None and pose != "person_prone" and \
                    d.situation != POSE_DEFAULT_SITUATION[pose]:
                print(f"FAIL: fallback for {pose} != pose default")
                fails += 1
    print(f"exhaustive: {checked} combinations checked")

    assert normalize_support_token("bench_01") == "bench"
    assert normalize_support_token("sports_car_2") == "car"
    assert normalize_support_token(None) is None

    if "--verify-yaml" in sys.argv:
        from situation_resolver import verify_against_yaml
        import os
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "content-image-2026.R03")
        problems = verify_against_yaml(d)
        for p in problems:
            print("YAML MISMATCH:", p)
            fails += 1
        if not problems:
            print("official-yaml verification OK (tables in sync)")

    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
