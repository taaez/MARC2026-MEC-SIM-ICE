"""라이브 Stage2 `reveal`(msg 411)의 hint 오차를 실측한다.

`demo/participant_app.py` 가 매 Stage2 문항마다 `MARC_REVEAL_LOG`
(기본 `./results/stage2_reveals.jsonl`)에 남긴 줄을 시나리오 정답과 대조해서,
플랫폼이 주는 `hint_center` 가 실제 물체에서 얼마나 떨어져 있는지 계산한다.

## 왜 이 값이 필요한가

Stage2 주행은 grounding 결과가 아니라 `hint_center` 를 목표로 삼는다. 그런데 앱은
그 좌표를 **그대로 팔의 목표점으로** 넘긴다 (`_object_arm_xy` -> `run_pick_sequence`).
즉 재탐지가 없는 현재 구조에서는

    파지 오차 = hint 오차

가 1:1로 성립한다. `stage2_sim.py --hint-noise 0.3` 이 6/6 -> 0/6 으로 뒤집히는 게
그 증거다(주행은 전부 정상, 파지만 실패). 따라서 **hint 오차가 그리퍼 포획 범위보다
큰지 작은지**가 "근접 재탐지가 정밀도 옵션인가, 필수인가"를 가른다.

이 스크립트는 그 하나를 답하려고 있다. 지금까지 오프라인 하네스는 hint 오차를 0 으로
가정해 왔고(`stage2_sim.py` 헤더의 "hint 오차 0.0m"), 실측된 적이 없다.

## 쓰는 법

    # 데스크탑에서 Stage2 라운드를 한 번이라도 돌린 뒤
    python3 mec_sim_ice/vla_grounding/validation/hint_error.py

    python3 ... --log results/stage2_reveals.jsonl --tol 0.06

`--tol` 은 판정에 쓰는 그리퍼 포획 반경(m)이다. 기본값은 `fake_client.py` 가 쓰는
**가정치**이지 실측이 아니다 (접촉 솔버가 없어 원리적으로 모른다). 절대 판정이 아니라
자릿수 감각을 얻는 용도로 본다.

정답이 없는 문항(다른 시나리오 등)은 hint_radius 만 요약한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chungmu_gt import DEFAULT_SCENARIO, load_gt   # noqa: E402

DEFAULT_LOG = "./results/stage2_reveals.jsonl"
# fake_client.py 의 포획 반경 가정과 같은 값 (실측 아님).
DEFAULT_TOL_M = 0.06


def read_log(path: str) -> list:
    """reveal JSONL 을 읽는다. 깨진 줄은 건너뛰고 셈만 한다."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  (깨진 줄 {bad}개 건너뜀)")
    return rows


def dist_xy(a, b) -> float | None:
    """두 좌표의 수평거리. 파지는 xy 정렬 문제라 z 는 빼고 본다."""
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=DEFAULT_LOG, help=f"reveal JSONL (기본 {DEFAULT_LOG})")
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO, help="정답 시나리오 yaml")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_M,
                    help=f"판정용 그리퍼 포획 반경 m (기본 {DEFAULT_TOL_M}, 가정치)")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"reveal 기록이 없다: {args.log}")
        print("데스크탑에서 Stage2 문항을 한 번이라도 돌려야 생긴다 "
              "(problems.yaml 의 rt* 중 하나를 켜고 라이브 실행).")
        print("오프라인 시뮬은 가짜 hint 를 쏘므로 여기에 쓰지 않는다 "
              "(runs/stage2_reveals_SIM.jsonl 로 분리돼 있다).")
        return 1

    rows = read_log(args.log)
    if not rows:
        print(f"{args.log} 에 읽을 줄이 없다.")
        return 1

    try:
        gt = load_gt(args.scenario)
    except Exception as e:                       # 정답지가 없어도 radius 요약은 한다
        print(f"  (정답지를 못 읽었다: {e} -- hint_radius 만 요약한다)")
        gt = {}

    print("=" * 78)
    print(f"Stage2 hint 오차 실측 — {args.log}  ({len(rows)}건)")
    print("=" * 78)
    print(f"{'문항':24} {'hint오차':>9} {'radius':>8} {'grounding오차':>13} {'판정':>10}")
    print("-" * 78)

    hint_errs, radii, ground_errs, unmatched = [], [], [], 0
    for r in rows:
        cmd = r.get("task_description") or ""
        entry = gt.get(cmd)
        pid = entry[0] if entry else "(정답없음)"
        truth = entry[1].get("target_coord") if entry else None

        he = dist_xy(r.get("hint_center"), truth)
        ge = dist_xy(r.get("submitted_target_coord"), truth)
        rad = r.get("hint_radius")

        if rad is not None:
            radii.append(float(rad))
        if he is None:
            unmatched += 1
            verdict = "-"
        else:
            hint_errs.append(he)
            verdict = "재탐지필수" if he > args.tol else "hint로충분"
        if ge is not None:
            ground_errs.append(ge)

        print(f"{pid:24} {('%.3f' % he) if he is not None else '-':>9} "
              f"{('%.2f' % float(rad)) if rad is not None else '-':>8} "
              f"{('%.3f' % ge) if ge is not None else '-':>13} {verdict:>10}")

    print("-" * 78)
    if radii:
        print(f"  hint_radius     평균 {sum(radii)/len(radii):.2f}m  "
              f"최소 {min(radii):.2f}  최대 {max(radii):.2f}")
    if not hint_errs:
        print("  정답과 매칭된 문항이 없어 hint 오차는 계산하지 못했다.")
        print(f"  (매칭 실패 {unmatched}건 — 다른 시나리오의 기록일 수 있다)")
        return 0

    mean = sum(hint_errs) / len(hint_errs)
    over = sum(1 for e in hint_errs if e > args.tol)
    print(f"  hint 오차        평균 {mean:.3f}m  최소 {min(hint_errs):.3f}  "
          f"최대 {max(hint_errs):.3f}   (n={len(hint_errs)})")
    if ground_errs:
        print(f"  grounding 오차   평균 {sum(ground_errs)/len(ground_errs):.3f}m   "
              f"(참고: 우리 인식이 낸 좌표)")
    if unmatched:
        print(f"  매칭 실패        {unmatched}건")

    print()
    print(f"  포획 반경 {args.tol:.3f}m 기준 초과 {over}/{len(hint_errs)}건")
    if over:
        print("  => 재탐지 없이는 파지가 hint 정확도에 그대로 묶인다. 근접 재탐지"
              " (base/gripper 카메라 + depth) 가 필수다.")
    else:
        print("  => hint 만으로 포획 범위 안이다. 재탐지는 정밀도 개선 옵션에 가깝다"
              " (표본이 적으면 더 모아서 재확인할 것).")
    print("  주의: 포획 반경은 가정치다. 라이브 파지 성패와 함께 봐야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
