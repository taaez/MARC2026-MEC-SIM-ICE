"""실행 결과가 진짜 marc2026_chungmu 문제셋인지 검증한다.

플랫폼이 낸 results/scores.jsonl 의 expected 필드를, content 이미지에서 추출해둔
공식 marc2026_chungmu.yaml 과 문항 단위로 대조한다. 시나리오가 섞였거나
problems.yaml 선택이 잘못됐거나 문항 정의가 바뀐 경우를 잡아낸다.

사용 (cd mec_sim_ice/vla_grounding/validation 후):
    python3 verify_run.py ../../../results/scores.jsonl
    python3 verify_run.py <scores.jsonl> --scenario <chungmu.yaml 경로>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENARIO = os.path.join(_HERE, "..", "content-image-2026.R03",
                                "marc2026_chungmu.yaml")


def load_rounds(path: str):
    rounds, final = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("stage") == "final":
                final = d["scores"]
            else:
                rounds[d["scores"]["problem_id"]] = d
    return rounds, final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", help="results/scores.jsonl 경로")
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    args = ap.parse_args()

    import yaml
    with open(args.scenario, encoding="utf-8") as f:
        official = yaml.safe_load(f)
    off = {p["id"]: p for p in official["challenge_problems"]}
    rounds, final = load_rounds(args.scores)

    print(f"공식 시나리오 문항 수 : {len(off)}")
    print(f"실행된 라운드 수      : {len(rounds)}")
    if final:
        print(f"final.stage1_rounds   : {final.get('stage1_rounds')}")

    issues = []
    for pid, prob in off.items():
        if pid not in rounds:
            continue          # 문항 선택으로 일부만 돌린 경우는 정상
        exp_run = rounds[pid]["scores"]["expected"]
        exp_off = prob["expected"]
        lm_off = exp_off.get("landmark_name")
        pairs = [
            ("voice_command", prob["voice_command"], rounds[pid]["voice_command"]),
            ("target_type", exp_off.get("target_type"), exp_run.get("target_type")),
            ("landmark_prefix",
             lm_off.rsplit("_", 1)[0] if lm_off else None, exp_run.get("landmark_prefix")),
            ("relation", exp_off.get("relation"), exp_run.get("relation")),
            ("camera_id", exp_off.get("camera_id"), exp_run.get("camera_id")),
        ]
        for field, a, b in pairs:
            if a != b:
                issues.append(f"{pid}.{field}: 공식={a!r} 실행={b!r}")

    unknown = set(rounds) - set(off)
    if unknown:
        issues.append(f"공식 chungmu 에 없는 problem_id: {sorted(unknown)}")

    checked = len(set(rounds) & set(off))
    print(f"대조한 문항 수        : {checked}")
    if issues:
        print(f"\n불일치 {len(issues)}건:")
        for i in issues:
            print("  -", i)
        print("\nFAIL — 이 결과는 공식 chungmu 정의와 다릅니다.")
        return 1
    print("\nPASS — 실행 결과가 공식 chungmu 문제셋과 일치합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
