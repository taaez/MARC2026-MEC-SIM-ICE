"""records/s1_* 라운드를 taejeong 파이프라인(지금 코드)으로 재실행하고, 그 자리에서
바로 채점까지 한다 — Isaac Sim/ROS/플랫폼 전부 불필요.

정답(anchor/target 좌표 포함)은 어디서 오나:
과거 실제 라이브 실행(``runs/chungmu_full39_2026-08-25.jsonl``)이 남긴 각 라운드의
"expected"/"weights" 는, 그날 우리가 제출에 성공했는지와 무관하게 **플랫폼이 계산해
돌려준 공식 정답·가중치**다 (camera_id, landmark_prefix, target_type, relation,
situation, anchor_coord, target_coord 전부 포함). voice_command 로 조인해서 그대로
정답지로 재사용하면, 이후 코드를 아무리 고쳐도 Isaac Sim 없이 즉시 재채점할 수 있다.
새 문제(정답지에 없는 voice_command)가 섞여 있으면 그 라운드는 "정답 없음"으로
건너뛰고 목록에 표시한다.

채점식은 score_report.py 와 동일한 카테고리(0/1) + 좌표(``max(0, 1 - 3D오차/5m)``) 구조를
쓰되, 정확도가 필드마다 다르다는 걸 39개 과거 기록 전수 역산으로 확인했다:
  * camera / object_type / landmark / relation (0/1) 과 **anchor 좌표 점수는 위 공식과
    100% 일치** — 39건 전부 소수점까지 정확히 재현됨. 그대로 믿어도 된다.
  * **target 좌표 점수는 이 공식만으로 재현되지 않는 경우가 있다** (예: 오차 0.88m가
    만점인데 오차 0.93m가 0점인 사례 존재 — 아마 물체 실제 크기/영역 기반 판정이라
    중심점 거리만으로는 알 수 없는 정보가 관여함). 아래 target 점수는 **참고용
    근사치**이며, 실제 라이브 채점과 다를 수 있다 — 코드 변화의 방향(개선/악화)을
    가늠하는 용도로만 쓰고, 정확한 target 점수/총점은 라이브 실행으로 확인할 것.
person 문제의 "object_type" 칸은 target_type 문자열이 아니라 situation 일치 여부로
채점된다 (이것도 39문항 전수 대조로 확인 — object_type_score == 1.0 은 항상 situation
일치와 정확히 일치했다).

target 미탐지(``grounding.resolve`` 가 None)일 때의 fallback 값은 실제 라이브 코드
(``demo_agents/taejeong_agent_vla.py._ground``)와 동일하게 재현한다: camera_id="",
좌표=[0,0,0], 나머지 필드는 파싱된 값을 그대로 채워 제출한다.

사용 (cd mec_sim_ice/taejeong/validation 후):
  python3 offline_score.py --weights ../../demo_agents/weights/best.pt

옵션(기본값은 이 폴더 기준 상대경로):
  --records <dir>       기본 ../../records
  --weights <best.pt>   기본 ../../demo_agents/weights/best.pt
  --answer-key <jsonl>  기본 runs/chungmu_full39_2026-08-25.jsonl
  --conf 0.25 --imgsz 1280
  --json report.json    리포트를 JSON 으로도 저장 (score_report.py 와 동일 스키마)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

_VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
_TAEJEONG_DIR = os.path.dirname(_VALIDATION_DIR)
sys.path.insert(0, _TAEJEONG_DIR)   # command_parser / grounding / yolo_adapter / run_offline

from command_parser import parse_command          # noqa: E402
from grounding import resolve                      # noqa: E402
from run_offline import load_round                 # noqa: E402
from yolo_adapter import infer_cameras              # noqa: E402

from score_report import report                     # noqa: E402  (same dir)

COORD_TOLERANCE_M = 5.0


def load_answer_key(path: str) -> dict:
    """voice_command -> (problem_id, expected, weights)."""
    key = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("stage") == "final":
                continue
            s = d["scores"]
            key[d["voice_command"]] = (s["problem_id"], s["expected"], s["weights"])
    return key


def fallback_submission(parsed) -> dict:
    """target 미탐지 시 제출값 -- taejeong_agent_vla.py._ground 의 fallback과 동일."""
    fallback_type = parsed.target_type or ("person" if parsed.is_person else "")
    return {
        "camera_id": "", "target_type": fallback_type,
        "relation": None if parsed.is_person else parsed.relation_hint,
        "situation": None, "landmark": parsed.landmark_type or "",
        "anchor_coord": [0.0, 0.0, 0.0], "target_coord": [0.0, 0.0, 0.0],
    }


def flatten_payload(payload: dict) -> dict:
    interp = payload.get("interpretation", {})
    ground = payload.get("grounding", {})
    return {
        "camera_id": payload.get("camera_id", ""),
        "target_type": interp.get("target_type"),
        "relation": interp.get("relation"),
        "situation": interp.get("situation"),
        "landmark": interp.get("landmark"),
        "anchor_coord": ground.get("anchor_coord", [0.0, 0.0, 0.0]),
        "target_coord": ground.get("target_coord", [0.0, 0.0, 0.0]),
    }


def score_round(problem_id: str, weights: dict, expected: dict, submission: dict) -> dict:
    is_person = expected.get("target_kind") == "person"

    def eq(a, b) -> float:
        return 1.0 if a == b else 0.0

    def coord(key):
        a, b = expected.get(key), submission.get(key)
        if not a or not b:
            return 0.0, None
        d = math.dist(a, b)
        return max(0.0, 1.0 - d / COORD_TOLERANCE_M), d

    camera = eq(submission.get("camera_id"), expected.get("camera_id"))
    object_type = (eq(submission.get("situation"), expected.get("situation")) if is_person
                   else eq(submission.get("target_type"), expected.get("target_type")))
    landmark = eq(submission.get("landmark"), expected.get("landmark_prefix"))
    relation = eq(submission.get("relation"), expected.get("relation"))
    anchor, anchor_err = coord("anchor_coord")
    target, target_err = coord("target_coord")

    fields = {"camera": camera, "object_type": object_type, "landmark": landmark,
              "relation": relation, "anchor": anchor, "target": target}
    total = 100.0 * sum(weights.get(k, 0.0) * v for k, v in fields.items())

    return {
        "problem_id": problem_id, **fields, "total": total, "weights": weights,
        "expected": expected, "submission": submission,
        "anchor_err_m": anchor_err, "target_err_m": target_err,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="../../records")
    ap.add_argument("--weights", default="../../demo_agents/weights/best.pt")
    ap.add_argument("--answer-key", default="runs/chungmu_full39_2026-08-25.jsonl")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    answer_key = load_answer_key(args.answer_key)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    round_dirs = sorted(glob.glob(os.path.join(args.records, "s1_*")))
    scored_rounds = []
    unmatched = []
    for d in round_dirs:
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta, cams, images = load_round(d)
        cmd = meta["command"]
        hit = answer_key.get(cmd)
        if hit is None:
            unmatched.append((os.path.basename(d), cmd))
            continue
        problem_id, expected, weights = hit

        parsed = parse_command(cmd)
        dets = infer_cameras(model, images, min_conf=args.conf, imgsz=args.imgsz)
        decision = resolve(parsed, dets, cams)
        submission = (flatten_payload(decision.to_payload()) if decision is not None
                      else fallback_submission(parsed))

        s = score_round(problem_id, weights, expected, submission)
        scored_rounds.append({"stage": 1, "voice_command": cmd, "scores": s})

    if unmatched:
        print(f"[정답 없음, 건너뜀] {len(unmatched)}건 (records 에는 있지만 --answer-key 에 없는 voice_command)")
        for name, cmd in unmatched:
            print(f"  - {name}: {cmd!r}")
        print()

    if not scored_rounds:
        print("채점할 라운드가 없습니다 (records 와 --answer-key 를 확인하세요).")
        return 1

    scored_rounds.sort(key=lambda r: r["scores"]["problem_id"])
    final = {
        "stage1_average": sum(r["scores"]["total"] for r in scored_rounds) / len(scored_rounds),
        "stage1_rounds": len(scored_rounds),
    }
    print("!" * 78)
    print("주의: camera/object_type/landmark/relation/anchor 는 과거 39건 전수 대조로")
    print("검증된 정확한 채점식이지만, target(좌표) 점수는 근사치다 -- 실제 라이브 채점은")
    print("물체 크기/영역까지 보는 것으로 보여 중심점 거리(1-오차/5)만으로는 벗어나는")
    print("사례가 있었다. 총점도 그 영향을 받으니 참고용으로만 볼 것 (정확한 총점은")
    print("Isaac Sim 라이브 실행에서만 확인 가능).")
    print("!" * 78)
    print()
    return report(scored_rounds, final, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
