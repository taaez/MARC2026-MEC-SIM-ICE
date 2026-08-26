"""자가 채점 리포트: results/scores.jsonl → 항목별 정확도 + 실패 원인 분류.

플랫폼이 낸 라운드별 세부 점수를 읽어, 어느 단계에서 점수를 잃었는지
(카메라 선택 / 대상 탐지 / 랜드마크 / relation / situation / 좌표)를 분류한다.
반복 실행하며 개선 효과를 비교하는 것이 목적이다.

채점식(실측 역산): 좌표 점수 = max(0, 1 - 3D오차 / COORD_TOLERANCE_M)

사용 (cd mec_sim_ice/taejeong/validation 후):
    python3 score_report.py ../../../results/scores.jsonl
    python3 score_report.py <scores.jsonl> --json report.json   # 기계가 읽을 형태로도 저장
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

COORD_TOLERANCE_M = 5.0     # 실측 역산값 (score = 1 - err/5)
FIELDS = ("camera", "object_type", "landmark", "relation", "anchor", "target")


def load(path: str):
    rounds, final = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("stage") == "final":
                final = d["scores"]
            else:
                rounds.append(d)
    return rounds, final


def classify(d: dict) -> str:
    """이 라운드가 점수를 잃은 1차 원인을 한 가지로 분류."""
    s = d["scores"]
    sub, exp = s["submission"], s["expected"]
    if not sub.get("camera_id") and sub.get("anchor_coord") == [0.0, 0.0, 0.0]:
        return "대상_미탐지(빈제출)"
    if s.get("camera", 0) == 0:
        return "카메라_오선택"
    if s.get("object_type", 0) == 0:
        return "대상종류/상황_오답"
    if s.get("landmark", 0) == 0:
        return "랜드마크_오답"
    if s.get("relation", 0) == 0:
        return "relation_오답"
    if s.get("target", 0) < 0.9 or s.get("anchor", 0) < 0.9:
        return "좌표_오차"
    return "정상"


def report(rounds: list, final: dict | None = None, json_path: str | None = None) -> int:
    """rounds: [{"scores": {...}}, ...] (scores.jsonl 한 줄과 같은 스키마).
    라이브 실행이 아니라 메모리에서 바로 만든 rounds 로도 호출할 수 있다
    (offline_score.py 가 이렇게 재사용한다)."""
    if not rounds:
        print("라운드 기록이 없습니다.")
        return 1

    print("=" * 78)
    print(f"자가 채점 리포트 — {len(rounds)} 라운드")
    print("=" * 78)
    if final:
        if final.get("stage1_average") is not None:
            print(f"Stage1 평균 : {final.get('stage1_average')}")
        if final.get("stage2") is not None:
            print(f"Stage2      : {final.get('stage2')}")
        if final.get("total") is not None:
            print(f"총점        : {final.get('total')} / {final.get('max_total')}")
        print()

    # -- 항목별 평균 (실제 배점 반영: camera 10 / object_type 15 / landmark 10 /
    #    relation 10 / anchor 15 / target 40 -- person 문제는 landmark/relation 배점이
    #    0 이라 그 필드는 집계에서 빠진다) --
    print("[항목별 평균 점수]  (배점 기준, 만점 = 그 필드의 배점)")
    for f in FIELDS:
        applicable = [d for d in rounds if d["scores"].get("weights", {}).get(f, 0) > 0]
        if not applicable:
            continue
        max_pts = applicable[0]["scores"]["weights"][f] * 100.0
        earned = [d["scores"].get(f, 0) * max_pts for d in applicable]
        perfect = sum(1 for e in earned if e >= max_pts - 1e-6)
        print(f"  {f:12s} 평균 {sum(earned)/len(earned):5.2f} / {max_pts:4.1f}점   "
              f"만점 {perfect}/{len(applicable)}건  (해당 {len(applicable)}/{len(rounds)}문항)")
    print()

    # -- 실패 원인 분류 --
    buckets = defaultdict(list)
    for d in rounds:
        buckets[classify(d)].append(d)
    print("[실패 원인 분류]")
    for cause, ds in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        avg = sum(x["scores"]["total"] for x in ds) / len(ds)
        print(f"  {cause:22s} {len(ds):2d}건  평균 {avg:6.2f}점")
        for d in sorted(ds, key=lambda x: x["scores"]["total"])[:4]:
            s = d["scores"]
            print(f"       - {s['problem_id']:24s} {s['total']:6.2f}점")
    print()

    # -- 좌표 오차 --
    print("[좌표 오차]  (제출한 라운드만)")
    errs = []
    for d in rounds:
        s = d["scores"]
        sub, exp = s["submission"], s["expected"]
        if not sub.get("camera_id"):
            continue
        for kind in ("anchor", "target"):
            a, b = exp.get(f"{kind}_coord"), sub.get(f"{kind}_coord")
            if a and b and b != [0.0, 0.0, 0.0]:
                errs.append((kind, math.dist(a, b), s["problem_id"]))
    for kind in ("anchor", "target"):
        v = sorted(e[1] for e in errs if e[0] == kind)
        if v:
            print(f"  {kind:7s} n={len(v):2d}  중앙값 {v[len(v)//2]:.2f}m  "
                  f"최소 {v[0]:.2f}m  최대 {v[-1]:.2f}m  "
                  f"(허용 {COORD_TOLERANCE_M:.0f}m)")
    print()

    # -- 카메라 혼동 --
    conf = Counter()
    for d in rounds:
        s = d["scores"]
        sub, exp = s["submission"], s["expected"]
        if sub.get("camera_id") and s.get("camera", 0) == 0:
            conf[(exp["camera_id"], sub["camera_id"])] += 1
    if conf:
        print("[카메라 오선택 패턴]  정답 → 제출")
        for (want, got), n in conf.most_common():
            print(f"  {want} → {got}   {n}회")
        print()

    # -- 문항별 표 (문항 번호순. 각 칸 = 획득점수/배점, "–" = 이 문제 유형엔 배점 없음) --
    print("[문항별 상세]  (문항 번호순 -- 각 칸: 획득점수/배점)")
    print(f"  {'problem_id':26s} {'총점':>6s}  {'camera':>7s} {'type':>7s} "
          f"{'landmark':>9s} {'relation':>9s} {'anchor':>10s} {'target':>10s}")

    def cell(s: dict, f: str, decimals: int) -> str:
        mx = s.get("weights", {}).get(f, 0) * 100.0
        if mx <= 0:
            return "–"
        earned = s.get(f, 0) * mx
        return f"{earned:.{decimals}f}/{mx:.0f}"

    for d in sorted(rounds, key=lambda x: x["scores"]["problem_id"]):
        s = d["scores"]
        print(f"  {s['problem_id']:26s} {s['total']:6.2f}  "
              f"{cell(s, 'camera', 0):>7s} {cell(s, 'object_type', 0):>7s} "
              f"{cell(s, 'landmark', 0):>9s} {cell(s, 'relation', 0):>9s} "
              f"{cell(s, 'anchor', 1):>10s} {cell(s, 'target', 1):>10s}")

    if json_path:
        out = {
            "final": final,
            "field_averages": {f: (sum(d["scores"].get(f, 0) for d in rounds) / len(rounds))
                               for f in FIELDS},
            "causes": {c: [d["scores"]["problem_id"] for d in ds] for c, ds in buckets.items()},
            "rounds": [{"problem_id": d["scores"]["problem_id"],
                        "total": d["scores"]["total"],
                        "cause": classify(d),
                        "fields": {f: d["scores"].get(f) for f in FIELDS}}
                       for d in rounds],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 리포트 저장: {json_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--json", default=None, help="리포트를 JSON 으로도 저장")
    args = ap.parse_args()

    rounds, final = load(args.scores)
    return report(rounds, final, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
