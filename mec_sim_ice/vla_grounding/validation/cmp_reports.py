"""두 채점 리포트(JSON)를 문항 단위로 비교 — 개선/악화가 어디서 났는지 본다.

offline_score.py / score_report.py 가 --json 으로 남긴 리포트 두 개를 받아
문항별 총점 차이와, 달라진 항목(camera/landmark/anchor/...)을 함께 찍는다.
코드를 고칠 때마다 "평균은 올랐는데 무엇이 대신 나빠졌나"를 확인하는 용도다.

    python3 mec_sim_ice/score.py --json before.json
    # ... 코드 수정 ...
    python3 mec_sim_ice/score.py --json after.json
    python3 vla_grounding/validation/cmp_reports.py before.json after.json
"""

import json
import sys

a, b = sys.argv[1], sys.argv[2]
base = {r["problem_id"]: r for r in json.load(open(a, encoding="utf-8"))["rounds"]}
new = {r["problem_id"]: r for r in json.load(open(b, encoding="utf-8"))["rounds"]}
rows = sorted(((new[k]["total"] - base[k]["total"], k) for k in base), key=lambda x: x[0])

print("=== worse ===")
for d, k in rows:
    if d < -0.01:
        print(f"  {k:26s} {base[k]['total']:6.2f} -> {new[k]['total']:6.2f} ({d:+.2f})")
        bf, nf = base[k]["fields"], new[k]["fields"]
        print("      " + "  ".join(f"{f}:{bf[f]:.2f}->{nf[f]:.2f}"
                                   for f in bf if abs((bf[f] or 0) - (nf[f] or 0)) > 1e-9))
print("=== better ===")
for d, k in rows:
    if d > 0.01:
        print(f"  {k:26s} {base[k]['total']:6.2f} -> {new[k]['total']:6.2f} ({d:+.2f})")
w = sum(1 for d, _ in rows if d < -0.01)
g = sum(1 for d, _ in rows if d > 0.01)
print(f"\nbetter {g} / worse {w} / same {len(rows)-g-w}")
