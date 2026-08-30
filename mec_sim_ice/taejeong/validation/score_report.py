"""자가 채점 리포트: results/scores.jsonl → 항목별 정확도 + 실패 원인 분류.

플랫폼이 낸 라운드별 세부 점수를 읽어, 어느 단계에서 점수를 잃었는지
(카메라 선택 / 대상 탐지 / 랜드마크 / relation / situation / 좌표)를 분류한다.

person 문항(sar*/sarH*)은 플랫폼이 object_type 점수를 target_type 이 아니라 situation
일치로 매기므로, situation 정답/제출값과 혼동 패턴을 "[situation 판정]" 절에 따로 찍고
문항별 표에도 ``sit`` 칸(O/X/–)을 둔다. JSON 리포트에는 각 라운드의
``situation: {expected, submitted, ok}`` 와 섹션별 ``situation`` 요약으로 들어간다.
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


def situation_check(s: dict):
    """person 문항의 situation 판정 -> (정답, 제출, 일치여부). object 문항이면 None.

    person 문항에서 플랫폼이 돌려주는 ``object_type`` 점수는 target_type 이 아니라
    **situation 일치**를 뜻한다 -- 2026-08-25 라이브 14문항 전수 대조 결과다.
    제출 target_type 은 규격상 항상 "person" 이라 14문항 전부 같은 값인데
    ``object_type`` 은 9건 1.0 / 5건 0.0 으로 갈렸고, 그 5건이 situation 이 틀린
    5건과 정확히 일치했다. 그래서 'type' 칸만 보면 상황 판정을 맞혔는지 알 수 없다.
    """
    exp = s.get("expected") or {}
    sub = s.get("submission") or {}
    if exp.get("target_kind") != "person":
        return None
    want, got = exp.get("situation"), sub.get("situation")
    return want, got, want == got


def _situation_json(s: dict) -> dict | None:
    """JSON 리포트용 situation 결과. object 문항이면 None.

    ``rounds[i]["fields"]`` 에는 넣지 않는다 -- cmp_reports.py 가 그 안의 값을 전부
    숫자로 보고 ``:.2f`` 로 찍기 때문에 문자열이 섞이면 깨진다.
    """
    c = situation_check(s)
    if not c:
        return None
    return {"expected": c[0], "submitted": c[1], "ok": c[2]}


def _sit_text(v) -> str:
    return "(미제출)" if v is None else str(v)


def classify(d: dict) -> str:
    """이 라운드가 점수를 잃은 1차 원인을 한 가지로 분류.

    배점이 0 인 항목은 원인에서 뺀다. person 문항은 landmark/relation 배점이 0 이라
    (플랫폼이 점수는 돌려주지만 총점에 안 들어간다) 그걸 원인으로 세면 엉뚱한 진단이
    나온다 -- 실제로 sar09/sarH1 은 relation 이 0.0 으로 와서 "relation_오답" 으로
    분류됐지만, 그 문항이 실제로 잃은 점수는 전부 target 좌표(50점)였다.
    """
    s = d["scores"]
    sub, exp = s["submission"], s["expected"]
    w = s.get("weights") or {}

    def counts(f) -> bool:
        return (not w) or w.get(f, 0) > 0

    if not sub.get("camera_id") and sub.get("anchor_coord") == [0.0, 0.0, 0.0]:
        return "대상_미탐지(빈제출)"
    if counts("camera") and s.get("camera", 0) == 0:
        return "카메라_오선택"
    if counts("object_type") and s.get("object_type", 0) == 0:
        return "situation_오답" if situation_check(s) else "대상종류_오답"
    if counts("landmark") and s.get("landmark", 0) == 0:
        return "랜드마크_오답"
    if counts("relation") and s.get("relation", 0) == 0:
        return "relation_오답"
    if ((counts("target") and s.get("target", 0) < 0.9)
            or (counts("anchor") and s.get("anchor", 0) < 0.9)):
        return "좌표_오차"
    return "정상"


def report(rounds: list, final: dict | None = None, json_path: str | None = None) -> int:
    """rounds: [{"scores": {...}}, ...] (scores.jsonl 한 줄과 같은 스키마).
    라이브 실행이 아니라 메모리에서 바로 만든 rounds 로도 호출할 수 있다
    (offline_score.py 가 이렇게 재사용한다).

    ``stage`` 가 2 인 라운드(Stage2 = retrieval 문항)는 별도 섹션으로 갈라서 집계한다 --
    Stage1 평균에 섞이면 대회의 Stage1 점수와 다른 숫자가 나오기 때문이다.
    """
    if not rounds:
        print("라운드 기록이 없습니다.")
        return 1

    s1 = [d for d in rounds if d.get("stage") != 2]
    s2 = [d for d in rounds if d.get("stage") == 2]

    print("=" * 78)
    print(f"자가 채점 리포트 — {len(rounds)} 라운드"
          + (f"  (Stage1 {len(s1)} / Stage2 {len(s2)})" if s2 else ""))
    print("=" * 78)
    # 평균은 항상 이 리포트가 실제로 채점한 라운드에서 다시 계산한다. final 에 담긴
    # 플랫폼 값은 집계 범위가 다를 수 있어(예: Stage2 문항까지 포함한 39라운드 평균)
    # 그대로 쓰면 섹션 평균과 어긋난다 -- 아래 "라이브 실측" 로 따로 보여준다.
    def _avg(rs):
        return sum(d["scores"]["total"] for d in rs) / len(rs)

    def _borrow_note(rs):
        """대여 프레임으로 채점한 문항이 섞였으면 실측만의 평균을 함께 보여준다."""
        measured = [d for d in rs if not d.get("borrowed")]
        if len(measured) == len(rs):
            return ""
        if not measured:
            return f"  [전부 대여 프레임 {len(rs)}문항]"
        return (f"   [실측 레코드 {len(measured)}문항 {_avg(measured):.2f} "
                f"+ 프레임 대여 {len(rs) - len(measured)}문항]")

    if s1:
        print(f"Stage1 평균          : {_avg(s1):6.2f}"
              f"   ({len(s1)}문항){_borrow_note(s1)}")
    if s2:
        print(f"Stage2 grounding 평균 : {_avg(s2):6.2f}"
              f"   ({len(s2)}문항){_borrow_note(s2)}")
        print("  └ Stage2 의 주행/집기/배달 실행 점수는 물리 시뮬이 필요해 오프라인에서")
        print("    채점할 수 없다. 위 숫자는 Stage2 의 **인식(grounding) 부분만**이다.")
    # 문항당 예산이 30초다 (시나리오 time_limit). 점수만 보고 폴백을 켜면 라이브에서
    # 시간 초과로 0점이 될 수 있으니, 예산 대비 어디쯤인지 같이 보여 준다.
    secs = [d["sec"] for d in rounds if d.get("sec")]
    if secs:
        tot = sorted(s["total"] for s in secs)
        over = sum(1 for s in tot if s > 30.0)
        parts = "  ".join(f"{k} {sum(s.get(k, 0) for s in secs) / len(secs):.1f}s"
                          for k in ("yolo", "dino", "vlm", "resolve"))
        print(f"\n[라운드 소요]  평균 {sum(tot) / len(tot):.1f}s   중앙값 {tot[len(tot) // 2]:.1f}s"
              f"   최대 {tot[-1]:.1f}s   30초 초과 {over}건")
        print(f"  └ 단계별 평균: {parts}")
    # 폴백이 꺼진 점수는 라이브(항상 켬)보다 낮게 나온다. 그걸 모르고 다른 머신의
    # 리포트와 나란히 놓으면 없는 회귀를 쫓게 되므로 평균 바로 밑에 붙여 둔다.
    fb = (final or {}).get("fallback")
    if fb is not None:
        print(f"  └ DINO/VLM 폴백: {'사용' if fb.get('used') else '건너뜀'}"
              f" ({fb.get('why')})"
              f" [dino={'O' if fb.get('dino') else 'X'} vlm={'O' if fb.get('vlm') else 'X'}]")
        if not fb.get("used"):
            print("    라이브 에이전트는 폴백을 항상 켠다 -- 이 점수는 하한이다.")
    # offline_score.py 가 자기가 계산한 값을 넣어 보낸 경우는 "실측"이 아니다.
    if final and final.get("source") != "offline":
        live = []
        if final.get("stage1_average") is not None:
            live.append(f"Stage1 평균 {final['stage1_average']:.2f}"
                        f"({final.get('stage1_rounds', '?')}라운드)")
        if final.get("stage2") is not None:
            live.append(f"Stage2 {final['stage2']}")
        if final.get("total") is not None:
            live.append(f"총점 {final['total']} / {final.get('max_total')}")
        if live:
            print(f"\n[플랫폼 실측]  {'  |  '.join(live)}")
    print()

    if s2:
        _section("Stage 1 (grounding 문항)", s1)
        _section("Stage 2 (retrieval 문항 -- grounding 부분만)", s2)
    else:
        _section(None, s1)

    if json_path:
        out = {
            "final": final,
            "stage1": _summary(s1),
            "stage2_grounding": _summary(s2) if s2 else None,
            # 하위 호환: 기존 스키마를 읽는 도구가 있어 전체 합산도 남겨둔다.
            "field_averages": {f: (sum(d["scores"].get(f, 0) for d in rounds) / len(rounds))
                               for f in FIELDS},
            "rounds": [{"problem_id": d["scores"]["problem_id"],
                        "stage": d.get("stage", 1),
                        "record": d.get("record"),
                        "sec": d.get("sec"),
                        "borrowed": bool(d.get("borrowed")),
                        "total": d["scores"]["total"],
                        "cause": classify(d),
                        "situation": _situation_json(d["scores"]),
                        "fields": {f: d["scores"].get(f) for f in FIELDS}}
                       for d in rounds],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 리포트 저장: {json_path}")
    return 0


def _summary(rounds: list) -> dict | None:
    """섹션 하나의 기계 판독용 요약."""
    if not rounds:
        return None
    buckets = defaultdict(list)
    for d in rounds:
        buckets[classify(d)].append(d)
    measured = [d for d in rounds if not d.get("borrowed")]
    return {
        "rounds": len(rounds),
        "borrowed_rounds": len(rounds) - len(measured),
        "average": sum(d["scores"]["total"] for d in rounds) / len(rounds),
        "average_measured_only": (sum(d["scores"]["total"] for d in measured)
                                  / len(measured)) if measured else None,
        "field_averages": {f: (sum(d["scores"].get(f, 0) for d in rounds) / len(rounds))
                           for f in FIELDS},
        "causes": {c: [d["scores"]["problem_id"] for d in ds] for c, ds in buckets.items()},
        "situation": _situation_summary(rounds),
    }


def max_points(s: dict, f: str) -> float:
    """이 라운드에서 항목 f 의 **실효 배점** (100점 만점 기준).

    플랫폼은 해당 없는 항목에 배점 0 을 주고 (person 문항의 landmark/relation), 남은
    가중치 합(0.8)을 100점으로 정규화한다 -- 39문항 전수 재계산으로 확인. 그래서
    person 문항의 실효 배점은 명목값과 다르다:
        분실물: camera 10 / object_type 15 / landmark 10 / relation 10 / anchor 15 / target 40
        사람  : camera 12.5 / situation 18.75 / anchor 18.75 / target 50  (landmark·relation 0)
    이 값을 쓰면 문항별 표의 각 칸을 더한 값이 그 문항 총점과 정확히 맞는다.
    """
    w = s.get("weights", {})
    wsum = sum(w.get(k, 0.0) for k in FIELDS)
    return 100.0 * w.get(f, 0.0) / (wsum or 1.0)


def _pad(s: str, width: int) -> str:
    """한글은 터미널에서 두 칸을 먹는다 -- 표시 폭 기준으로 오른쪽을 채운다."""
    shown = sum(2 if ord(c) > 0x1100 else 1 for c in s)
    return s + " " * max(0, width - shown)


def _print_weight_profiles(rounds: list) -> None:
    """이 구간에 섞여 있는 배점 프로필을 문제 유형별로 한 줄씩 찍는다.

    사람 문항과 분실물 문항은 배점이 다른데(아래), 그걸 안 보여주면 표의 숫자가
    왜 문항마다 다른 만점을 갖는지 알 수 없다. 값은 그때그때 채점 결과에 들어 있는
    weights 에서 계산하므로, 배점이 바뀌면 이 줄도 저절로 따라간다.
    """
    profiles = {}
    for d in rounds:
        s = d["scores"]
        label = "사람" if situation_check(s) else "분실물"
        key = tuple(round(max_points(s, f), 4) for f in FIELDS)
        profiles.setdefault((label, key), []).append(s["problem_id"])
    if not profiles:
        return
    print("[배점]  (플랫폼이 라운드마다 돌려준 weights 에서 계산 -- 100점 만점 기준)")
    for (label, key), pids in sorted(profiles.items()):
        parts, zero = [], []
        for f, pts in zip(FIELDS, key):
            name = "situation" if (f == "object_type" and label == "사람") else f
            (parts if pts > 0 else zero).append(
                f"{name} {pts:g}" if pts > 0 else name)
        note = f"   ({'·'.join(zero)} 배점 0)" if zero else ""
        print(f"  {_pad(label, 7)}{len(pids):2d}문항: {' / '.join(parts)}{note}")
    if any(label == "사람" for label, _ in profiles):
        print("  사람 문항의 situation 점수는 플랫폼이 'object_type' 칸에 담아 돌려준다"
              " -- 아래 표의 'type' 칸이 그것이다.")
    print()


def _score_items(rounds: list, person: list, other: list) -> list:
    """항목별 집계에 쓸 (표시이름, 대상 라운드, 점수 키) 목록.

    사람 문항과 분실물 문항은 같은 키라도 뜻이나 배점이 달라서, 섞인 구간에서는
    한 줄로 합치면 안 된다:
      * object_type -- 분실물은 물건 종류, 사람은 situation 판정이다 (뜻이 다름).
      * 나머지 -- 사람 문항은 landmark/relation 배점 0 을 뺀 나머지를 정규화하므로
        camera/anchor/target 의 실효 배점이 분실물보다 크다 (10 vs 12.5 등).
    두 유형이 같이 있고 배점이 다르면 "(물건)/(사람)" 으로 갈라 찍는다.
    """
    def name(f, suffix=""):
        return ("situation" if f == "object_type" and suffix == "(사람)" else f) + suffix

    items = []
    for f in FIELDS:
        if not person:
            items.append((f, rounds, f))
            continue
        if not other:
            items.append((name(f, "(사람)") if f == "object_type" else f, rounds, f))
            continue
        pm = {round(max_points(d["scores"], f), 3) for d in person}
        om = {round(max_points(d["scores"], f), 3) for d in other}
        if f == "object_type" or pm != om:
            if om != {0.0}:
                items.append((name(f, "(물건)"), other, f))
            if pm != {0.0}:
                items.append((name(f, "(사람)"), person, f))
        else:
            items.append((f, rounds, f))
    return items


def _situation_summary(rounds: list) -> dict | None:
    """person 문항 situation 정확도 요약 (기계 판독용). person 문항이 없으면 None."""
    checks = [(d["scores"]["problem_id"], situation_check(d["scores"])) for d in rounds]
    checks = [(pid, c) for pid, c in checks if c]
    if not checks:
        return None
    correct = sum(1 for _, c in checks if c[2])
    return {
        "problems": len(checks),
        "correct": correct,
        "accuracy": correct / len(checks),
        "per_problem": [{"problem_id": pid, "expected": c[0], "submitted": c[1],
                         "ok": c[2]} for pid, c in checks],
        # person 문항 landmark 는 플랫폼이 점수를 돌려주지만 배점이 0 이라 총점에
        # 안 들어간다. 그래도 파싱이 맞았는지 보려면 필요해서 같이 남긴다.
        "landmark_correct": sum(1 for d in rounds
                                if situation_check(d["scores"])
                                and d["scores"].get("landmark", 0) >= 1.0),
        "landmark_weighted": any(d["scores"].get("weights", {}).get("landmark", 0) > 0
                                 for d in rounds if situation_check(d["scores"])),
        "wrong": [{"problem_id": pid, "expected": c[0], "submitted": c[1]}
                  for pid, c in checks if not c[2]],
    }


def _situation_section(rounds: list) -> None:
    """person 문항(sar*/sarH*)의 situation 정답 여부를 따로 찍는다.

    표의 'type' 칸은 person 문항에서 situation 점수지만, 무엇을 정답으로 봤고
    무엇을 냈는지가 안 보여서 어디를 고쳐야 할지 알 수 없었다. 여기서 정답/제출
    문자열과 혼동 패턴(emergency 를 abnormal 로 냈다 등)을 같이 보여준다.
    """
    rows = [(d["scores"], situation_check(d["scores"])) for d in rounds]
    rows = [(s, c) for s, c in rows if c]
    if not rows:
        return
    ok = sum(1 for _, c in rows if c[2])
    lm_ok = sum(1 for s, _ in rows if s.get("landmark", 0) >= 1.0)
    lm_weighted = any(s.get("weights", {}).get("landmark", 0) > 0 for s, _ in rows)
    sit_pts = max_points(rows[0][0], "object_type")
    print(f"[situation 판정]  person {len(rows)}문항 -- situation 정답 {ok}/{len(rows)} "
          f"({100.0 * ok / len(rows):.1f}%)")
    print(f"  분실물의 object_type(물건 종류, 15점) 자리를 사람 문항에서는 situation 이"
          f" 쓴다 ({sit_pts:.2f}점).")
    print("  플랫폼이 situation 점수를 별도 키가 아니라 'object_type' 칸에 담아 돌려주기")
    print("  때문에, 표의 항목 이름만 봐서는 상황 판정을 맞혔는지 알 수 없다.")
    print(f"  landmark 일치 {lm_ok}/{len(rows)}"
          + ("" if lm_weighted else "   (person 문항 landmark 는 배점 0 -- 맞아도 점수 없음)"))
    print(f"  {'problem_id':26s} {'정답':>11s} {'제출':>11s}   situation  landmark")
    for s, (want, got, hit) in sorted(rows, key=lambda x: x[0]["problem_id"]):
        lm = "O" if s.get("landmark", 0) >= 1.0 else "X"
        print(f"  {s['problem_id']:26s} {_sit_text(want):>11s} "
              f"{_sit_text(got):>11s}   {'O' if hit else 'X':^9s}  {lm}")
    conf = Counter((want, got) for _, (want, got, hit) in rows if not hit)
    if conf:
        print("  [혼동 패턴]  정답 → 제출")
        for (want, got), n in conf.most_common():
            print(f"    {_sit_text(want)} → {_sit_text(got)}   {n}회")
    print()


def _section(title: str | None, rounds: list) -> None:
    """한 단계(Stage1/Stage2)에 대한 상세 표를 찍는다."""
    if not rounds:
        return
    if title:
        print("-" * 78)
        print(f"### {title} — {len(rounds)}문항  "
              f"평균 {sum(d['scores']['total'] for d in rounds) / len(rounds):.2f}점")
        print("-" * 78)

    _print_weight_profiles(rounds)

    # -- 항목별 평균 --
    # 배점은 문제 유형마다 다르다 (아래 max_points 주석). person 문항의 object_type
    # 칸은 물건 종류가 아니라 situation 판정이라, 두 유형을 한 줄에 합치면 무슨 항목의
    # 평균인지 알 수 없다 -- "object_type(물건)" 과 "situation(사람)" 으로 갈라 찍는다.
    person = [d for d in rounds if situation_check(d["scores"])]
    other = [d for d in rounds if not situation_check(d["scores"])]
    print("[항목별 평균 점수]  (만점 = 그 항목의 실효 배점)")
    for label, subset, f in _score_items(rounds, person, other):
        applicable = [d for d in subset if max_points(d["scores"], f) > 0]
        if not applicable:
            continue
        maxes = [max_points(d["scores"], f) for d in applicable]
        earned = [d["scores"].get(f, 0) * m for d, m in zip(applicable, maxes)]
        perfect = sum(1 for e, m in zip(earned, maxes) if e >= m - 1e-6)
        same = len({round(m, 3) for m in maxes}) == 1
        mx = f"{maxes[0]:4.1f}점" if same else f"{sum(maxes)/len(maxes):4.1f}점(평균)"
        print(f"  {label:16s} 평균 {sum(earned)/len(earned):5.2f} / {mx}   "
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

    # -- situation (person 문항) --
    _situation_section(rounds)

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
    print("[문항별 상세]  (문항 번호순 -- 각 칸: 획득점수/실효배점, 칸을 다 더하면 총점)")
    print("  person 문항은 'type' 칸이 곧 situation 점수다 (sit 칸에 O/X 로 같이 표시).")
    print("  (O)/(X) = 맞았는지는 채점되지만 배점 0 인 칸,  – = 해당 없음")
    print(f"  {'problem_id':26s} {'총점':>6s}  {'camera':>9s} {'type':>9s} {'sit':>4s} "
          f"{'landmark':>9s} {'relation':>9s} {'anchor':>10s} {'target':>10s}")

    def cell(s: dict, f: str, decimals: int) -> str:
        mx = max_points(s, f)
        # 사람 문항의 실효 배점은 12.5 / 18.75 처럼 정수가 아니다. 정수로 반올림해
        # 찍으면 칸을 다 더해도 총점과 안 맞아 보여서, 그때는 소수 한 자리로 찍는다.
        if abs(mx - round(mx)) > 1e-6:
            decimals = max(decimals, 1)
        if mx <= 0:
            # person 문항의 landmark 는 플랫폼이 맞았는지 계산해서 돌려주기는 하는데
            # 배점이 0 이다 (아래 [배점] 주석 참고). 그냥 "–" 로 숨기면 "사람 문제는
            # landmark 를 안 본다" 로 읽히므로, 배점 없는 참고값이라고 괄호로 표시한다.
            if f == "landmark" and situation_check(s) and s.get(f) is not None:
                return "(O)" if s.get(f, 0) >= 1.0 else "(X)"
            return "–"
        earned = s.get(f, 0) * mx
        return f"{earned:.{decimals}f}/{mx:.{decimals}f}"

    for d in sorted(rounds, key=lambda x: x["scores"]["problem_id"]):
        s = d["scores"]
        pid = s["problem_id"] + (" †" if d.get("borrowed") else "")
        chk = situation_check(s)
        sit = "–" if not chk else ("O" if chk[2] else "X")
        print(f"  {pid:26s} {s['total']:6.2f}  "
              f"{cell(s, 'camera', 0):>9s} {cell(s, 'object_type', 0):>9s} {sit:>4s} "
              f"{cell(s, 'landmark', 0):>9s} {cell(s, 'relation', 0):>9s} "
              f"{cell(s, 'anchor', 1):>10s} {cell(s, 'target', 1):>10s}")
    borrowed = [d for d in rounds if d.get("borrowed")]
    if borrowed:
        print(f"  † {len(borrowed)}문항 -- records/ 에 그 라운드가 없어 다른 라운드의 "
              f"프레임을 빌려 채점 (정적 물체 문항만)")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--json", default=None, help="리포트를 JSON 으로도 저장")
    ap.add_argument("--scenario", default=None,
                    help="시나리오 yaml. 주면 문항 종류(kind: retrieval)로 Stage1/Stage2 를 "
                         "갈라서 따로 집계한다. 플랫폼이 Stage2 문항을 Stage1 라운드로 "
                         "출제한 기록도 이걸로 제대로 분리된다")
    args = ap.parse_args()

    rounds, final = load(args.scores)

    if args.scenario:
        # 라이브 기록의 `stage` 는 "몇 단계에서 출제됐나"라서, Stage2 문항이 Stage1
        # 라운드로 나온 기록에서는 전부 1 이다. 문항 종류로 다시 매긴다.
        from chungmu_gt import problem_kinds
        kinds = problem_kinds(args.scenario)
        for d in rounds:
            pid = d.get("scores", {}).get("problem_id")
            if kinds.get(pid) == "retrieval":
                d["stage"] = 2

    return report(rounds, final, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
