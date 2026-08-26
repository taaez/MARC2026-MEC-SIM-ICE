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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--json", default=None, help="리포트를 JSON 으로도 저장")
    args = ap.parse_args()

    rounds, final = load(args.scores)
    if not rounds:
        print("라운드 기록이 없습니다.")
        return 1

    print("=" * 78)
    print(f"자가 채점 리포트 — {len(rounds)} 라운드")
    print("=" * 78)
    if final:
        print(f"Stage1 평균 : {final.get('stage1_average')}")
        print(f"Stage2      : {final.get('stage2')}")
        print(f"총점        : {final.get('total')} / {final.get('max_total')}")
        print()

    # -- 항목별 평균 --
    print("[항목별 평균 점수]  (1.0 = 만점)")
    for f in FIELDS:
        vals = [d["scores"].get(f, 0) for d in rounds if f in d["scores"]]
        if not vals:
            continue
        perfect = sum(1 for v in vals if v >= 0.999)
        print(f"  {f:12s} {sum(vals)/len(vals):.3f}   만점 {perfect}/{len(vals)}")
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

    # -- 문항별 표 --
    print("[문항별 상세]  (점수 낮은 순)")
    print(f"  {'problem_id':26s} {'점수':>6s}  cam obj lmk rel  anchor target")
    for d in sorted(rounds, key=lambda x: x["scores"]["total"]):
        s = d["scores"]
        print(f"  {s['problem_id']:26s} {s['total']:6.2f}  "
              f"{s.get('camera',0):3.0f} {s.get('object_type',0):3.0f} "
              f"{s.get('landmark',0):3.0f} {s.get('relation',0):3.0f}  "
              f"{s.get('anchor',0):6.3f} {s.get('target',0):6.3f}")

    if args.json:
        report = {
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
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 리포트 저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
