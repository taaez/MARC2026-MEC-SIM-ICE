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
사람 문항(sar*/sarH*)의 배점은 분실물과 다르다 — 39문항 전수 대조로 확인했다:
  * "object_type" 칸은 target_type 이 아니라 **situation 일치 여부**다. 제출
    target_type 은 규격상 항상 "person" 이라 14문항이 전부 같은 값인데 object_type 은
    9/5 로 갈렸고, 그 5건이 situation 오답 5건과 정확히 일치했다.
  * **landmark 와 relation 은 배점 0** 이다 (점수 자체는 플랫폼이 계산해 돌려준다).
    남은 가중치 합 0.8 을 100점으로 정규화하므로 실효 배점은
    camera 12.5 / situation 18.75 / anchor 18.75 / target 50 이 된다.
    landmark 에 배점을 주면 14문항 전부 총점 재현에 실패한다 (sar01 실제 18.75 vs 27.78).
그래서 리포트는 situation 을 따로 보여준다 — 섹션마다 "[배점]" 줄, "[situation 판정]"
절(정답/제출/혼동 패턴/landmark 참고), 문항별 표의 ``sit`` 칸(O/X/–), --json 의 라운드별
``situation`` 항목. 실점 원인 분류도 배점 0 인 항목은 원인에서 제외한다.

target 미탐지(``grounding.resolve`` 가 None)일 때의 fallback 값은 실제 라이브 코드
(``demo_agents/taejeong_agent_vla.py._ground``)와 동일하게 재현한다: camera_id="",
좌표=[0,0,0], 나머지 필드는 파싱된 값을 그대로 채워 제출한다.

사용 — 인자 없이 그냥 실행하면 된다. 기본 경로가 CWD 가 아니라 이 스크립트
위치 기준이라 **어느 디렉토리에서 실행하든** 동작한다:

  python3 mec_sim_ice/score.py            # 리포 루트에서 (가장 짧은 형태)
  python3 offline_score.py                # 이 폴더에서

Stage1 / Stage2 는 따로 집계한다:
  * 단계 구분은 기록 폴더 이름(s1_*/s2_*)이 아니라 **시나리오의 문항 종류**로 한다
    (``kind: grounding`` → Stage1, ``kind: retrieval`` → Stage2). 위 39문항 기록 세션은
    Stage2 문항(rt01~rtC2)까지 전부 Stage1 라운드로 출제해서 s1_034~s1_039 에 들어가
    있기 때문에, 폴더 이름으로 가르면 Stage1 평균에 Stage2 문항이 섞인다.
  * **Stage2 는 grounding(인식) 부분만 채점된다.** 주행/집기/배달 실행 점수는 물리 시뮬
    판정이 필요해 records 만으로는 재현할 수 없다 (LiDAR/점유지도/is_grasping/로봇 pose 가
    기록에 없다). 그래서 Stage2 점수는 0~100 독립 점수로만 내고 총점 환산은 하지 않는다.
  * 같은 문항이 여러 폴더에 잡히면(rt01 은 s1_034/s2_004/s2_040) 전부 채점해 보여주되
    평균에는 **가장 최근 기록만** 반영한다 (`[중복 레코드]` 로 표시).

정답지에 없는 문항은 시나리오 yaml 에서 유도한다 (``chungmu_gt.load_gt``) — 문항을 새로
켜도 바로 채점된다. 겹치는 문항은 플랫폼이 돌려준 jsonl 값을 우선한다.

옵션(전부 선택):
  --records <dir>       기본 mec_sim_ice/records
  --weights <best.pt>   기본 mec_sim_ice/demo_agents/weights/best.pt
  --answer-key <jsonl>  기본 validation/runs/chungmu_full39_2026-08-25.jsonl
  --scenario <yaml>     기본 taejeong/content-image-2026.R03/marc2026_chungmu.yaml
  --conf 0.25 --imgsz 1280
  --json report.json    리포트를 JSON 으로도 저장 (score_report.py 와 동일 스키마)
  --fallback / --no-fallback   DINO/VLM 폴백 강제 on/off. 기본은 **자동**이고, 자동은
                        '라이브와 같게 맞추되 못 돌릴 환경이면 알리고 빠진다' 규칙이다
                        (transformers + 가속기(CUDA/MPS) 있으면 켬, 없으면 끄고 리포트에 명시).
                        라이브 에이전트는 폴백을 항상 켜므로, 폴백이 꺼진 점수는
                        데스크탑 실측보다 낮게 나온다 -- 그대로 비교하면 안 된다.

주의: macOS 는 "--" 를 em-dash("—") 로 자동 치환하는 경우가 있다. 옵션이
"unrecognized arguments" 로 튕기면 대시가 한 글자로 바뀐 것이니 하이픈 두 개로
다시 입력할 것 (main() 이 이 경우를 감지해 안내한다).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict

_VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
_TAEJEONG_DIR = os.path.dirname(_VALIDATION_DIR)
_MEC_ROOT = os.path.dirname(_TAEJEONG_DIR)          # mec_sim_ice/
sys.path.insert(0, _VALIDATION_DIR)  # score_report
sys.path.insert(0, _TAEJEONG_DIR)   # command_parser / grounding / yolo_adapter / run_offline

from command_parser import parse_command          # noqa: E402
from dino_fallback import DinoFallbackDetector      # noqa: E402
from dino_fallback import is_disabled as dino_disabled  # noqa: E402
from fallback import (dino_augment, pose_augment, vlm_assist,  # noqa: E402
                      yolo_tile_augment)
from pose_keypoints import AnkleKeypointEstimator   # noqa: E402
from pose_keypoints import is_disabled as pose_disabled       # noqa: E402
from grounding import resolve                      # noqa: E402
from run_offline import load_round                 # noqa: E402
from vlm_fallback import VlmTargetResolver          # noqa: E402
from vlm_fallback import is_disabled as vlm_disabled  # noqa: E402
from yolo_adapter import infer_cameras              # noqa: E402

from chungmu_gt import DEFAULT_SCENARIO, load_gt    # noqa: E402  (same dir)
from score_report import report                     # noqa: E402  (same dir)

COORD_TOLERANCE_M = 5.0

# 폴백(DINO/VLM)이 켜진 채로 프레임 대여 채점을 하면 (대여문항 x 라운드) 만큼
# 모델이 돌아 아주 느려진다. 그때만 최신 라운드 이 개수로 표본을 줄인다.
BORROW_SAMPLES_FALLBACK = 8

# 기본 경로는 CWD 가 아니라 **이 스크립트 위치** 기준으로 잡는다 — 어느
# 디렉토리에서 실행하든 인자 없이 그냥 돌아가게 하려는 것.
DEFAULT_RECORDS = os.path.join(_MEC_ROOT, "records")
DEFAULT_WEIGHTS = os.path.join(_MEC_ROOT, "demo_agents", "weights", "best.pt")
DEFAULT_ANSWER_KEY = os.path.join(_VALIDATION_DIR, "runs",
                                  "chungmu_full39_2026-08-25.jsonl")


def load_answer_key(path: str, scenario: str | None = None) -> dict:
    """voice_command -> (problem_id, expected, weights, kind).

    두 군데서 정답을 모아 합친다:
      1. 시나리오 yaml (`chungmu_gt.load_gt`) -- 모든 문항을 커버하고, Stage1/Stage2 구분
         (``kind``: grounding / retrieval)이 여기에만 있다.
      2. 라이브 기록 jsonl -- 플랫폼이 직접 계산해 돌려준 값이라 더 정확하다
         (yaml 유도는 police_car_01 anchor 하나가 보정 테이블에 의존한다). 겹치는
         문항은 이쪽으로 덮어쓰고, ``kind`` 만 yaml 쪽 값을 유지한다.
    """
    key = {}
    if scenario and os.path.exists(scenario):
        for cmd, (pid, expected, weights, kind, _owner) in load_gt(scenario).items():
            key[cmd] = (pid, expected, weights, kind)

    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("stage") == "final":
                    continue
                s = d["scores"]
                cmd = d["voice_command"]
                prev = key.get(cmd)
                key[cmd] = (s["problem_id"], s["expected"], s["weights"],
                            prev[3] if prev else "grounding")
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
    # 해당 없는 필드는 플랫폼이 가중치를 0 으로 준다 (person 문항의 landmark/relation).
    # 그러면 가중치 합이 1.0 이 아니라 0.8 이 되는데, 플랫폼은 거기에 맞춰 정규화해
    # 100점 만점을 유지한다 -- 안 나누면 person 14문항이 통째로 25% 저평가된다.
    # (runs/chungmu_full39_2026-08-25.jsonl 39건 전수 재계산: 안 나누면 9건 불일치,
    #  나누면 0건 불일치.)
    wsum = sum(weights.get(k, 0.0) for k in fields)
    total = 100.0 * sum(weights.get(k, 0.0) * v for k, v in fields.items()) / (wsum or 1.0)

    return {
        "problem_id": problem_id, **fields, "total": total, "weights": weights,
        "expected": expected, "submission": submission,
        "anchor_err_m": anchor_err, "target_err_m": target_err,
    }


def stage2_execution(scenario: str, occupancy=None):
    """Stage2 실행(주행·집기·배달)을 오프라인 시뮬로 돌린다. 실패하면 None.

    `stage2_sim.run_execution` 이 `demo/participant_app.py` 를 원본 그대로 실행한다
    (가짜 MARCClient). 무엇이 진짜 검증되고 무엇이 가정인지는 `fake_client.py` 참고.
    """
    try:
        from stage2_sim import run_execution
        rows, info = run_execution(scenario=scenario, occupancy=occupancy,
                                   progress=False, header=False)
    except Exception as e:
        print(f"[Stage2 실행] 건너뜀 -- {type(e).__name__}: {e}")
        return None
    return {"rows": rows, "info": info}


def summarize_stage2_execution(exec_result):
    """Stage2 실행 시뮬 결과를 요약 dict 로 -- **출력은 하지 않는다** (순수 계산).

    `main()` 이 이 값을 `report()` 호출 **전에** `final["stage2_execution"]` 에 채워
    넣어야, `report()` 가 한 번에 쓰는 `--json` 파일에 이 요약이 함께 들어간다.
    (예전에는 `report()` 가 JSON 을 쓴 뒤 이 함수가 따로 한 번 더 덮어썼는데, 그
    두 번째 쓰기가 `report()` 의 `rounds[i]["fields"]` 스키마를 깨서 `cmp_reports.py`
    가 못 읽는 파일이 됐다 -- report() 를 유일한 JSON 저장 지점으로 되돌린다.)
    """
    rows = exec_result["rows"]
    n = len(rows)
    return {"problems": n,
            "nav_ok": sum(1 for r in rows if r["legs"] and r["legs"][0]["reached"]),
            "pick_ok": sum(1 for r in rows if r["capture"]),
            "delivery_ok": sum(1 for r in rows
                               if r["in_basket"] and r["delivery_err_m"] < 0.7),
            "per_problem": [{"problem_id": r["pid"],
                             "nav": bool(r["legs"] and r["legs"][0]["reached"]),
                             "pick": bool(r["capture"]),
                             "delivery": bool(r["in_basket"] and r["delivery_err_m"] < 0.7),
                             "sim_s": r["sim_s"]} for r in rows]}


def print_stage2_execution(exec_result, summary, grounding_avg=None):
    """Stage2 를 공식 3요소로 나눠 출력한다 (출력 전용, JSON 은 안 쓴다).

    개발자 가이드가 Stage2 채점을 세 가지로 명시한다 -- VLA grounding / 물품 집기 /
    지정된 장소로 이동. 다만 같은 문서가 "정확한 임계값·가중치는 공개하지 않습니다"
    라고 못박고 있어, 세 값을 하나의 Stage2 총점으로 합칠 근거가 없다.
    리포 안의 관측치로도 분해가 안 된다: 라이브 최종 `stage2 = 87.0` (만점 100) 인데,
    그때 실제 제출한 Stage2 grounding(`records/s2_040`)은 25.00점이라 관측된 로그
    문자열 "Stage 2 recovery: 0/80" 과 어떤 조합으로도 87 이 나오지 않는다.

    각 열(주행/집기/배달)이 실제로 확인하는 것과 안 하는 것이 다르므로, 화면에
    그 경계를 매번 명시한다 -- "OK" 라는 글자만 보고 등급이 같다고 오해하면 안 된다.
    """
    rows = exec_result["rows"]
    n = summary["problems"]
    pick_ok, dlv_ok, nav_ok = (summary["pick_ok"], summary["delivery_ok"],
                               summary["nav_ok"])

    print()
    print("-" * 78)
    print(f"### Stage 2 실행 (주행·집기·배달) — {n}문항   [오프라인 시뮬]")
    print("-" * 78)
    print("현재 상태: 팀의 실제 파지 로직은 아직 미도착 -- 지금 도는 건 스타터킷")
    print("기본값(manipulation/arm_pick.py + 기본 pnp_params.json)뿐이다. 그러므로")
    print("아래 숫자는 '우리 코드가 문제를 잘 푸는가' 가 아니라 'Stage2 파이프라인이")
    print("(가짜 클라이언트로) 끝까지 도는가' 를 확인한 것이다 -- 시험지와 채점기가")
    print("완성됐다는 뜻이지, 아직 답을 잘 쓴다는 뜻이 아니다.")
    print()
    print("공식 Stage2 채점 3요소 (개발자 가이드):")
    if grounding_avg is not None:
        print(f"  (1) VLA grounding      평균 {grounding_avg:5.2f} / 100점   <- 위 표")
    else:
        print("  (1) VLA grounding      (해당 기록 없음)")
    print(f"  (2) 물품 집기          {pick_ok}/{n} 문항 성공   <- 아래 시뮬 (기하 기준, 가정)")
    print(f"  (3) 지정 장소로 이동   {dlv_ok}/{n} 문항 성공")
    print()
    print("  ※ 공식 문서가 '정확한 임계값·가중치는 공개하지 않습니다' 라고 명시해,")
    print("    이 셋을 Stage2 총점으로 합치지 않는다. 라이브 관측치로도 분해가 안 된다")
    print("    (stage2=87.0 인데 그때 제출한 grounding 은 25.00점 -- 어떤 조합으로도 안 맞음).")
    print()
    print("  이 열이 실제로 보장하는 것 / 안 하는 것:")
    print()
    print("    주행 OK  확정.  차체 운동학(가속제한 포함) + A* + 제어루프가 진짜로 돌아")
    print("             표준 도착판정(0.7m) 안에 들어왔다는 뜻. 접촉 물리와 무관해")
    print("             신뢰도가 가장 높다. '실패' 도 확정 실패다.")
    print()
    print("    집기 OK  가정.  placo 로 계산한 실제 EE 위치가 물체와 포획반경(0.06m)")
    print("             안까지 '명령대로 왔다' 는 뜻일 뿐이다. 다음은 전혀 모른다:")
    print("               - 손가락이 실제로 닫혀 물체를 감쌌는지 (접촉 솔버가 없음)")
    print("               - 쥐는 도중 물체가 밀리거나 넘어지는지")
    print("               - 그리퍼 폐쇄력이 물체 무게를 버티는지")
    print("               - 물체의 실제 z높이 (ground_z 는 카메라별 고정 평면일 뿐")
    print("                 지형고가 아니다 -- stage2_offline.py [3](c) 참고)")
    print("             '실패' 로 뜬 문항(팔이 도달범위 밖)은 확정 실패다. 'OK' 는")
    print("             '닿을 가능성 있음' 정도로만 읽을 것 -- 파지 성공의 증거가 아니다.")
    print()
    print("    배달 OK  조건부.  '집기 OK' 였던 문항에서만 의미가 있다. 목적지 도착")
    print("             자체는 주행과 같은 이유로 신뢰도가 높지만, 그 전제인 '물체를")
    print("             정말 들고 있었나' 가 위 '집기' 의 가정을 그대로 물려받는다.")
    print()
    print("  물리 엔진(접촉·마찰·질량) 없이는 원리적으로 확인 불가능:")
    print("    파지 성공 여부, 이동 중 낙하, 차체 전복, 그리퍼 폐쇄력, 라이다 지역회피,")
    print("    플랫폼 Stage2 배점 실측치.")
    print()
    print(f"  {'문항':24} {'주행':6} {'집기':6} {'배달':6} {'시뮬시간':>9}  비고")
    print("  " + "-" * 74)
    for r in rows:
        nav = "OK" if (r["legs"] and r["legs"][0]["reached"]) else "실패"
        pk = "OK" if r["capture"] else "실패"
        dl = "OK" if (r["in_basket"] and r["delivery_err_m"] < 0.7) else "실패"
        note = ""
        if not r["capture"]:
            aa = r["approach_arm_xy"]
            if aa:
                note = f"팔이 겨눌 거리 {math.hypot(aa[0], aa[1]):.2f}m (도달 ~0.75m)"
        if r["error"]:
            note = r["error"]
        print(f"  {r['pid']:24} {nav:6} {pk:6} {dl:6} {r['sim_s']:8.1f}s  {note}")
    print(f"\n  주행 {nav_ok}/{n}(확정)  집기 {pick_ok}/{n}(가정 -- 겨냥만 확인)  "
          f"배달 {dlv_ok}/{n}(집기 가정을 물려받음)")
    print("  차체/팔 모델 상수는 taejeong/validation/fake_client.py 상단 참고.")
    print("  manipulation/ 에 팀 파지 로직이 들어오면 같은 명령을 다시 실행해서 비교할 것.")


def resolve_fallback_mode(flag) -> tuple:
    """DINO/VLM 폴백을 쓸지 결정한다. 반환 (사용여부, 사람이 읽을 이유).

    flag 는 --fallback 이면 True, --no-fallback 이면 False, 아무것도 안 주면 None(자동).
    자동일 때의 규칙은 '라이브와 같게 맞추되, 못 돌릴 환경이면 조용히 빠지지 말고 알린다':

      * 이 함수가 켜는 건 DINO 뿐이다. VLM 은 2026-08-29부터 기본이 꺼져 있고
        (vlm_fallback.is_disabled -- chungmu 39/39 에서 호출 0회, 상주 비용만
        ~7GB) MARC_ENABLE_VLM=1 로 따로 켜야 한다 -- 라이브 에이전트
        (taejeong_agent_vla.py)도 같은 함수를 보므로 둘이 갈라지지 않는다.
      * 라이브와 같은 env 스위치(MARC_DISABLE_DINO_FALLBACK)가 켜져 있으면 끈다.
      * transformers 가 없으면 끈다 (노트북 기본 환경).
      * 가속기(CUDA 또는 Apple Silicon MPS)가 없으면 끈다. Qwen2.5-VL / Grounding DINO
        를 CPU 로 올리면 몇십 분씩 걸리거나 메모리가 터진다. MPS 는 dino_fallback.py /
        vlm_fallback.py 가 직접 지원하므로(362fa1c) 로컬 Mac 에서도 켠다 -- 다만 4bit
        는 bitsandbytes 가 CUDA 전용이라 무시되고 bf16 으로 올라간다.
      * 그 외에는 켠다 (Isaac Sim 데스크탑 = CUDA 있음 = 라이브와 동일 조건).
    """
    if flag is False:
        return False, "--no-fallback 지정"
    forced = flag is True
    if dino_disabled() and vlm_disabled():
        # VLM 은 기본이 이미 꺼져 있어(위 docstring) 이 조건은 사실상
        # "DINO 도 꺼졌으면"이다 -- 그러면 폴백 전체가 할 일이 없으니
        # transformers import 조차 시도하지 않고 여기서 바로 끝낸다.
        if not forced:
            return False, ("DINO 도 VLM 도 꺼짐 (--no-dino / VLM 기본 OFF 또는 "
                           "MARC_DISABLE_DINO_FALLBACK) -- 폴백 계층이 할 일이 없다")
    try:
        import transformers  # noqa: F401
    except ImportError:
        if forced:
            return True, "--fallback 강제 (transformers 없음 -- 로드에 실패할 것)"
        return False, "transformers 미설치 (pip install transformers 필요)"
    # 폴백 모듈이 실제로 고르는 디바이스와 같은 순서로 판정한다
    # (dino_fallback.py / vlm_fallback.py: cuda -> mps -> cpu).
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    except Exception:  # noqa: BLE001
        pass
    if forced:
        return True, ("--fallback 강제 (device=cpu -- 매우 느릴 수 있음)" if device == "cpu"
                      else f"--fallback 강제 (device={device})")
    if device == "cpu":
        return False, "가속기 없음 (CUDA/MPS 둘 다 -- 자동 우회, 강제하려면 --fallback)"
    if device == "mps":
        return True, "Apple Silicon MPS 사용 가능 (4bit 는 무시되고 bf16 으로 로드)"
    return True, "CUDA 사용 가능 -- 라이브 에이전트와 동일 조건"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=DEFAULT_RECORDS)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--answer-key", default=DEFAULT_ANSWER_KEY)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO,
                    help="시나리오 yaml -- 정답지에 없는 문항의 정답을 여기서 유도하고, "
                         "Stage1/Stage2(retrieval) 구분도 여기서 읽는다")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-stage2-exec", action="store_true",
                    help="Stage2 실행(주행/집기/배달) 시뮬을 건너뛴다. "
                         "기본은 실행 -- stage2_sim.py 로 participant_app 을 그대로 굴린다")
    ap.add_argument("--stage2-occupancy", default=None,
                    help="Stage2 실행 시뮬이 쓸 실제 점유지도 npz (없으면 합성 지도)")
    ap.add_argument("--no-borrowed-frames", action="store_true",
                    help="records/ 에 라운드가 없는 문항을 다른 라운드의 프레임을 빌려서 "
                         "채점하지 않는다. 기본은 대여 채점 -- 레코드 폴더는 6개 카메라 "
                         "프레임을 전부 담고 있고 물체는 라운드가 달라도 같은 자리에 있어서 "
                         "정적 물체 문항은 그대로 채점된다. 이 옵션을 주면 그 문항들은 "
                         "채점되지 않고 목록만 출력된다.")
    ap.add_argument("--fallback", dest="fallback", action="store_true", default=None,
                    help="DINO/VLM 폴백을 강제로 켠다 (GPU 없어도 시도 -- CPU 로 7B VLM 을 "
                         "올리려 들어 아주 느리거나 메모리가 터질 수 있다).")
    ap.add_argument("--no-fallback", dest="fallback", action="store_false",
                    help="DINO/VLM 폴백을 강제로 끈다. 라이브 에이전트는 폴백을 항상 켜므로 "
                         "이걸 쓰면 라이브보다 약한 파이프라인을 채점하게 된다.")
    # 기본값은 '자동': 라이브(demo_agents/taejeong_agent_vla.py)는 폴백을 항상 켜므로
    # 오프라인도 켜는 게 맞다. 다만 노트북에는 보통 GPU 가 없고 Qwen2.5-VL / Grounding DINO
    # 는 GPU 전제라, CUDA 가 안 보이면 자동으로 건너뛰고 그 사실을 리포트에 명시한다
    # (조용히 빠지면 '폴백 없는 점수'를 라이브 점수로 착각하게 된다).
    # 모델별 on/off. --fallback/--no-fallback 은 "폴백 계층 전체"를 켜고 끄는
    # 마스터 스위치이고, 아래 둘은 그 안에서 어느 모델을 올릴지 고른다 --
    # 8GB 한 장으로 검증할 때 "DINO 만", "VLM 만" 을 나눠 돌려야 하기 때문이다
    # (둘을 같이 올리면 DINO fp32 타일링 피크 + Qwen bf16 상주가 겹쳐 터진다;
    #  근거는 dino_fallback.py._detect_batch 주석의 6.37GiB 실측).
    #
    # 플래그는 **환경변수로 번역해서** 적용한다. 판정의 단일 출처는 여전히
    # dino_fallback.is_disabled() / vlm_fallback.is_disabled() 이고 라이브
    # 에이전트도 같은 함수를 보므로, 여기서 별도 분기를 만들면 두 경로가
    # 갈라진다 -- 그래서 스위치만 바꾸고 판정 로직은 건드리지 않는다.
    ap.add_argument("--dino", dest="dino", action="store_true", default=None,
                    help="Grounding DINO 폴백을 켠다 (기본: 켜짐).")
    ap.add_argument("--no-dino", dest="dino", action="store_false",
                    help="Grounding DINO 폴백만 끈다 -- VLM 만 단독으로 재볼 때. "
                         "MARC_DISABLE_DINO_FALLBACK=1 과 같다.")
    ap.add_argument("--vlm", dest="vlm", action="store_true", default=None,
                    help="소형 LM(Qwen) 폴백을 켠다 (기본: 꺼짐 -- chungmu 39/39 "
                         "에서 호출 0회라 상주 비용만 든다. 본선 OOV 대비 검증용). "
                         "MARC_ENABLE_VLM=1 과 같다.")
    ap.add_argument("--no-vlm", dest="vlm", action="store_false",
                    help="소형 LM 폴백을 확실히 끈다 (MARC_DISABLE_VLM=1 과 같다).")
    ap.add_argument("--dino-box-thr", type=float, default=None,
                    help="DINO box confidence 임계값 (기본 DinoFallbackDetector 기본값 0.3). "
                         "실측(2026-08-28)상 소형물체(pencilcase) score가 0.11까지 나와 "
                         "기본값에선 폴백이 사실상 발동 안 함 -- 낮춰서 스윕할 때 씀.")
    ap.add_argument("--dino-text-thr", type=float, default=None,
                    help="DINO text 임계값 (기본 0.25).")

    # macOS 는 입력 중 "--" 를 em-dash("—") 로 자동 치환하는 경우가 있다.
    # 그대로 두면 argparse 가 "unrecognized arguments" 만 뱉어 원인을 알기 어렵다.
    for a in sys.argv[1:]:
        if a.startswith("—") or a.startswith("–"):
            print(f"인자 {a!r} 가 하이픈 두 개(--)가 아니라 대시 한 글자입니다 "
                  f"(macOS 자동 치환). '--{a.lstrip(chr(0x2013) + chr(0x2014))}' 로 "
                  f"다시 입력하세요.", file=sys.stderr)
            return 2

    args = ap.parse_args()

    # 모델별 플래그 -> 환경변수 (라이브와 같은 스위치를 그대로 쓴다, 위 주석 참고).
    # 플래그를 안 주면 아무것도 안 건드린다 -- 셸에서 미리 export 한 값이 그대로 산다.
    if args.dino is True:
        os.environ.pop("MARC_DISABLE_DINO_FALLBACK", None)
    elif args.dino is False:
        os.environ["MARC_DISABLE_DINO_FALLBACK"] = "1"
    if args.vlm is True:
        os.environ["MARC_ENABLE_VLM"] = "1"
        os.environ.pop("MARC_DISABLE_VLM", None)
    elif args.vlm is False:
        os.environ["MARC_DISABLE_VLM"] = "1"

    # 기본 root logger 레벨은 WARNING이라 fallback.py/grounding.py 의 log.info(...)
    # (DINO 구제, pair 후보 경쟁 등 진행상황 로그)가 조용히 버려진다 -- 39문항처럼
    # 오래 걸리는 실행에서 화면에 아무 것도 안 찍혀 멈춘 것처럼 보이는 원인이었다
    # (2026-08-29, 실제 진행 중이었는데 GPU 사용률로만 확인 가능했던 사례).
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)

    if not os.path.exists(args.weights):
        print(f"가중치를 찾을 수 없습니다: {args.weights}\n"
              f"  학습 담당 팀원에게 받은 best.pt 를 그 경로에 두거나 "
              f"--weights 로 경로를 지정하세요 (git 에 포함되지 않는 파일입니다).",
              file=sys.stderr)
        return 2
    if not os.path.isdir(args.records):
        print(f"records 폴더를 찾을 수 없습니다: {args.records}", file=sys.stderr)
        return 2

    answer_key = load_answer_key(args.answer_key, args.scenario)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    use_fallback, why = resolve_fallback_mode(args.fallback)
    print(f"[fallback] {'사용' if use_fallback else '건너뜀'} -- {why}")
    if not use_fallback:
        print("[fallback] 주의: 라이브 에이전트는 폴백을 항상 켠다. 아래 점수는 폴백이 "
              "구제했을 문항을 실패로 계산하므로 데스크탑 실측보다 낮게 나올 수 있다.")

    # 8GB 한 장으로 검증할 때 "지금 뭐가 올라가 있고 얼마나 남았는지"를 모르면
    # OOM 이 나야 알게 된다 -- 모델을 하나 올릴 때마다 찍는다 (CUDA 일 때만;
    # MPS/CPU 는 통합 메모리라 같은 의미의 숫자가 안 나온다).
    def _vram(tag):
        try:
            import torch
            if not torch.cuda.is_available():
                return
            free, total = torch.cuda.mem_get_info()
            print(f"[vram] {tag}: 프로세스 할당 "
                  f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB / 카드 여유 "
                  f"{free / 2**30:.2f} GiB (총 {total / 2**30:.2f} GiB)", flush=True)
        except Exception:  # noqa: BLE001
            pass

    _vram("YOLO 로드 후")
    dino = vlm = None
    if use_fallback and not dino_disabled():
        try:
            dino_kw = {}
            if args.dino_box_thr is not None:
                dino_kw["box_thr"] = args.dino_box_thr
            if args.dino_text_thr is not None:
                dino_kw["text_thr"] = args.dino_text_thr
            dino = DinoFallbackDetector(**dino_kw)
            _vram("DINO 로드 후")
        except Exception as e:  # noqa: BLE001
            print(f"[fallback] DINO 로드 실패, target/landmark 재탐색 없이 진행: {e}",
                  file=sys.stderr)
    if use_fallback and not vlm_disabled():
        try:
            vlm = VlmTargetResolver()
            _vram("VLM 로드 후")
        except Exception as e:  # noqa: BLE001
            print(f"[fallback] VLM 로드 실패, behind 중재/OOV 재해석 없이 진행: {e}",
                  file=sys.stderr)
    # 사람 target 의 발목 keypoint. DINO/VLM 폴백과 달리 --fallback 여부와
    # 무관하게 켠다 -- 라이브 에이전트가 항상 켜므로 여기서도 항상 켜야 두
    # 경로의 동작이 갈라지지 않는다 (fallback.py 모듈 docstring 의 원칙).
    pose = None
    if not pose_disabled():
        try:
            pose = AnkleKeypointEstimator()
            _vram("pose 로드 후")
        except Exception as e:  # noqa: BLE001
            print(f"[pose-kp] 로드 실패, 사람 좌표는 bbox 중앙 기준으로 진행: {e}",
                  file=sys.stderr)
    print(f"[모델] YOLO=on  DINO={'on' if dino else 'off'}  "
          f"VLM={'on' if vlm else 'off'}  pose={'on' if pose else 'off'}")
    if dino is not None and vlm is not None:
        print("[모델] 주의: 8GB 한 장이면 DINO 타일링 피크(실측 8타일 fp32 = 6.37GiB)와 "
              "Qwen 상주(~3.1GiB)가 겹쳐 OOM 이 난다. MARC_DINO_BATCH=2 를 주거나 "
              "--no-vlm / --no-dino 로 나눠 돌릴 것.")
    args.fallback = use_fallback

    # s1_* 는 Stage1 단계에서, s2_* 는 Stage2 단계에서 뜬 기록이다. 어느 단계에서 떴는지와
    # 무관하게 **문항 종류**(시나리오의 kind)로 Stage1/Stage2 를 가른다 -- 2026-08-25 기록
    # 세션은 39문항을 전부 Stage1 라운드로 출제해서 Stage2 문항(rt*)까지 s1_034~s1_039 에
    # 들어가 있기 때문이다.
    round_dirs = sorted(glob.glob(os.path.join(args.records, "s[12]_*")))
    round_dirs = [d for d in round_dirs
                  if os.path.exists(os.path.join(d, "meta.json"))]
    scored_rounds = []
    unmatched = []

    def run_round(cmd, hit, cams, images, dets, record_name, borrowed=False):
        """명령 + (카메라, 프레임, 검출) → 채점된 라운드 dict. 파이프라인은 라이브와 동일.

        ``dets`` 는 프레임에만 의존하므로 (명령과 무관) 한 프레임을 여러 문항에 재사용할
        수 있다. 다만 augment 들은 명령별 폴백 탐지를 dets 에 **in-place 로 넣기**
        때문에, 재사용할 때는 카메라별 리스트를 복사해서 넘긴다 -- 안 그러면 앞 문항이
        구제한 탐지가 뒤 문항에 새어 들어간다.

        단계별 초를 함께 잰다: 시나리오의 time_limit 이 문항당 30초라, 폴백이 그
        예산을 넘기면 점수가 올라도 라이브에서는 못 쓴다. 어느 단계가 먹는지
        갈라 둬야 그 판단을 할 수 있다 (리포트의 ``sec`` 항목).
        """
        problem_id, expected, weights, kind = hit
        # augment 들이 dets 를 제자리 수정하므로 항상 복사한다 (위 docstring).
        dets = {cid: list(v) for cid, v in dets.items()}

        t = {}
        t0 = time.perf_counter()
        parsed = parse_command(cmd)
        target_phrases = vlm_assist(cmd, parsed, vlm) if args.fallback else None
        t["vlm"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if args.fallback:
            # DINO 보다 먼저 -- YOLO 타일링으로 구제되면 dino_augment 의
            # 카메라별 스킵이 그대로 적용돼 그만큼 DINO 호출이 줄어든다.
            yolo_tile_augment(parsed, images, dets, model)
        t["yolo_tile"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if args.fallback:
            dino_augment(parsed, images, dets, target_phrases, dino, vlm)
        t["dino"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pose_augment(parsed, images, dets, pose)
        t["pose"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        decision = resolve(parsed, dets, cams)
        t["resolve"] = time.perf_counter() - t0
        # YOLO 추론(infer_cameras)은 프레임당 1회라 이 함수 밖에서 돈다 -- 대여
        # 채점이 같은 dets 를 여러 문항에 재사용하기 때문. 호출부가 재 둔 값을
        # 여기에 합쳐야 total 이 라이브 한 문항의 실제 비용과 같아진다.
        t["yolo"] = _yolo_sec.get(record_name, 0.0)
        t["total"] = sum(t.values())

        submission = (flatten_payload(decision.to_payload()) if decision is not None
                      else fallback_submission(parsed))
        if t["total"] >= 10.0:      # 30초 예산의 1/3을 넘긴 라운드는 그 자리에서 알린다
            print(f"      느림: {t['total']:.1f}s (yolo {t['yolo']:.1f} / "
                  f"dino {t['dino']:.1f} / vlm {t['vlm']:.1f} / pose {t['pose']:.1f})",
                  flush=True)
        return {"stage": 2 if kind == "retrieval" else 1,
                "voice_command": cmd, "record": record_name, "sec": t,
                "borrowed": borrowed,
                "scores": score_round(problem_id, weights, expected, submission)}

    # -- 정답지엔 있는데 records/ 에 라운드가 없는 문항 --
    # 예전에는 이 문항들이 리포트에서 **아무 말 없이 사라졌다** (lf02~lf08 7문항).
    # "Stage1 평균 26문항" 이 문제집 전체인 것처럼 보여서, 채점기가 문제집의 1/5 를
    # 건너뛰고 있다는 사실조차 드러나지 않았다. 이제 항상 드러내고, 빌릴 수 있으면
    # 다른 라운드의 프레임을 빌려서라도 채점한다.
    #
    # 프레임 대여가 성립하는 이유: 레코드 한 폴더에는 **6개 카메라 프레임이 전부** 들어
    # 있고(라운드가 달라도 시야는 같다), 대여 대상은 정적 물체 문항이다 -- 물체는
    # 라운드가 바뀌어도 같은 자리에 있다. 사람은 라운드마다 자세·위치가 바뀌므로
    # (레코드별 person_* 검출이 실제로 다르다) 사람 문항은 빌리지 않는다.
    #
    # 한 라운드만 빌리면 안 된다: 작은 물체(tumbler/umbrella)의 YOLO 검출이 프레임마다
    # 불안정해서, 그 프레임에서 우연히 안 잡히면 문항이 통째로 0 점이 된다 (실측 대조:
    # rt02 는 자기 레코드에서 94.50 인데 s2_032 프레임으로는 35.00). 그래서 **가능한 모든
    # 라운드 프레임으로 채점한 뒤 중앙값 라운드를 채택**하고, 산포를 함께 보고한다 --
    # 프레임 운에 좌우되는 문항이면 그 사실 자체가 드러나야 한다.
    recorded_cmds = set()
    for d in round_dirs:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            recorded_cmds.add(json.load(f)["command"])
    missing = [(cmd, hit) for cmd, hit in answer_key.items()
               if cmd not in recorded_cmds]
    borrow_targets, skipped_person = [], []
    for cmd, hit in sorted(missing, key=lambda x: x[1][0]):
        if (hit[1] or {}).get("target_kind") == "person":
            skipped_person.append(hit[0])
        else:
            borrow_targets.append((cmd, hit))
    if args.no_borrowed_frames:
        borrow_targets = []

    # 폴백이 켜져 있으면 대여 채점 한 건마다 DINO/VLM 이 돌아 비용이 (문항 x 라운드)로
    # 커진다. 그때는 최신 라운드 몇 개로만 표본을 잡는다 -- 중앙값의 취지는 유지된다.
    borrow_dirs = round_dirs if not args.fallback else round_dirs[-BORROW_SAMPLES_FALLBACK:]
    borrow_pool = defaultdict(list)
    # YOLO 추론은 프레임당 1회라 run_round 밖에서 돈다(대여 채점이 같은 dets 를
    # 여러 문항에 재사용하기 때문). 레코드별로 재 두고 run_round 가 자기 total 에
    # 합쳐 쓴다 -- 그래야 sec.total 이 라이브 한 문항의 실제 비용과 같아진다.
    _yolo_sec = {}

    for i, d in enumerate(round_dirs, 1):
        # 라운드당 DINO/VLM 폴백까지 겹치면 수 초~수십 초가 걸릴 수 있어, 진행상황이
        # 하나도 안 찍히면 (특히 --fallback 켰을 때) 멈춘 것처럼 보인다 -- 최소한
        # "몇 번째 도는 중"인지는 알 수 있게 매 라운드 시작마다 찍는다.
        print(f"[{i}/{len(round_dirs)}] {os.path.basename(d)}", flush=True)
        meta, cams, images = load_round(d)
        cmd = meta["command"]
        name = os.path.basename(d)
        t0 = time.perf_counter()
        dets = infer_cameras(model, images, min_conf=args.conf, imgsz=args.imgsz)
        _yolo_sec[name] = time.perf_counter() - t0
        hit = answer_key.get(cmd)
        if hit is None:
            unmatched.append((name, cmd))
        else:
            scored_rounds.append(run_round(cmd, hit, cams, images, dets, name))
        if d in borrow_dirs:
            for bcmd, bhit in borrow_targets:
                borrow_pool[bhit[0]].append(
                    run_round(bcmd, bhit, cams, images, dets, name, borrowed=True))

        # 8GB급 GPU(랩 데스크탑)에서 41라운드를 도는 동안 PyTorch 캐싱 allocator 가
        # 안 쓰는 블록을 계속 쥐고 있다가 결국 OOM 이 났다 (2026-08-29 실측, CUDA,
        # "GPU 0 has a total capacity of 7.53 GiB" -- 자기 프로세스가 6.62GiB 를 이미
        # 물고 있는 상태에서 타일링 배치 하나(1.91GiB)를 더 요청해 터짐). dtype 을
        # bf16 으로 낮추는 건 이미 grounding-dino 의 dtype 불일치로 크래시한 이력이
        # 있어(dino_fallback.py 상단 주석) 재시도하지 않는다 -- 대신 라운드마다 안 쓰는
        # 캐시를 돌려준다. 실제 사용 중인 텐서는 안 건드리므로 정확도에는 영향이 없고,
        # 코퍼스가 끝까지 도는가만 바뀐다. 대여 채점이 붙으면 라운드당 모델 호출이
        # (문항 x 프레임)으로 늘어 이 반납이 더 중요해진다.
        if args.fallback:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if borrow_pool:
        spread = []
        for pid in sorted(borrow_pool):
            cand = sorted(borrow_pool[pid], key=lambda r: r["scores"]["total"])
            chosen = cand[len(cand) // 2]           # 중앙값 라운드
            scored_rounds.append(chosen)
            spread.append((pid, cand[0]["scores"]["total"],
                           chosen["scores"]["total"], cand[-1]["scores"]["total"],
                           chosen["record"]))
        n_frames = len(borrow_dirs)
        print(f"[프레임 대여 채점] {len(spread)}문항 -- records/ 에 그 라운드가 없어 "
              f"다른 라운드 {n_frames}개의 프레임으로 채점했다.")
        print("  레코드 폴더는 6개 카메라 프레임을 전부 담고 있고 대여 문항은 전부 정적")
        print("  물체 문항이라(사람 자세 무관) 씬은 같다. 프레임마다 작은 물체 검출이")
        print("  흔들리므로 **중앙값 라운드**를 채택하고 산포를 함께 적는다 -- 최소와")
        print("  최대가 벌어진 문항은 점수가 프레임 운에 좌우된다는 뜻이다.")
        print(f"  {'문항':26} {'최소':>7} {'중앙값':>7} {'최대':>7}  채택 레코드")
        for pid, lo, mid, hi, rec in spread:
            flag = "  <- 프레임에 따라 요동" if hi - lo > 10 else ""
            print(f"  {pid:26} {lo:7.2f} {mid:7.2f} {hi:7.2f}  {rec}{flag}")
        if skipped_person:
            print(f"  [대여 제외] 사람 문항 {len(skipped_person)}건은 라운드마다 자세가 "
                  f"달라 빌리지 않았다: {', '.join(skipped_person)}")
        print()
    elif missing:
        print(f"[레코드 없음] {len(missing)}문항이 채점에서 빠졌다 "
              f"(정답지엔 있으나 records/ 에 그 라운드가 없음):")
        for cmd, hit in sorted(missing, key=lambda x: x[1][0]):
            print(f"  - {hit[0]:26s} {cmd!r}")
        print()

    if unmatched:
        print(f"[정답 없음, 건너뜀] {len(unmatched)}건 (records 에는 있지만 정답지/시나리오에 없는 voice_command)")
        for name, cmd in unmatched:
            print(f"  - {name}: {cmd!r}")
        print()

    if not scored_rounds:
        print("채점할 라운드가 없습니다 (records 와 --answer-key 를 확인하세요).")
        return 1

    # 같은 문항이 여러 레코드로 잡히는 경우가 있다 (예: rt01 이 s1_034 / s2_004 / s2_040).
    # 전부 채점해서 보여주되 평균은 문항당 한 번만 반영한다. 채택 기준은 **가장 최근에 뜬
    # 기록** -- record 는 수집 시점의 씬 스냅샷이라 시나리오/에셋이 갱신되면 옛 기록은
    # 낡은 장면을 채점하게 되고, 한 문항만 다른 세션 기록을 쓰면 나머지와 비교가 깨진다.
    # 같은 세션이면 Stage2 단계에서 뜬 s2_* 를 앞세운다(실제 Stage2 라운드의 장면).
    def _rank(r):
        name = r["record"]
        tail = name.rsplit("_", 1)[-1]
        ts = int(tail) if tail.isdigit() else 0
        # 대여 라운드(borrowed)는 그 문항의 실측 레코드가 없을 때만 생기지만, 혹시
        # 둘 다 있으면 실측을 항상 앞세운다 -- 대여는 어디까지나 보조 수단이다.
        return (1 if r.get("borrowed") else 0, -ts,
                0 if name.startswith("s2_") else 1, name)

    scored_rounds.sort(key=lambda r: (r["scores"]["problem_id"], _rank(r)))
    primary, dupes, seen = [], [], set()
    for r in scored_rounds:
        pid = r["scores"]["problem_id"]
        (dupes if pid in seen else primary).append(r)
        seen.add(pid)

    if dupes:
        print(f"[중복 레코드] {len(dupes)}건 -- 같은 문항의 다른 기록. 최신 기록만 평균에 반영한다.")
        for r in dupes:
            pid = r["scores"]["problem_id"]
            chosen = next(p for p in primary if p["scores"]["problem_id"] == pid)
            print(f"  - {pid:24s} {r['record']} {r['scores']['total']:6.2f}점  "
                  f"(평균 반영: {chosen['record']} {chosen['scores']['total']:.2f}점)")
        print()

    s1 = [r for r in primary if r["stage"] == 1]
    s2 = [r for r in primary if r["stage"] == 2]
    final = {"source": "offline"}   # 플랫폼 실측이 아니라 이 스크립트가 계산한 값이라는 표시
    # 폴백 on/off 는 점수를 크게 바꾸는데(라이브 에이전트는 항상 켠다) 터미널에만
    # 찍혀서, 나중에 report.json 만 보면 어느 쪽 점수인지 알 수 없었다. 리포트에
    # 남겨야 서로 다른 머신에서 나온 리포트를 비교할 때 사과-오렌지를 피한다.
    final["fallback"] = {
        "used": bool(use_fallback), "why": why,
        "dino": dino is not None, "vlm": vlm is not None,
        "pose": pose is not None,
    }

    def _avg(rs):
        return sum(r["scores"]["total"] for r in rs) / len(rs) if rs else None

    if s1:
        final["stage1_average"] = _avg(s1)
        final["stage1_rounds"] = len(s1)
        measured = [r for r in s1 if not r.get("borrowed")]
        if len(measured) != len(s1):
            final["stage1_average_measured_only"] = _avg(measured)
            final["stage1_rounds_measured_only"] = len(measured)
            final["stage1_borrowed_rounds"] = len(s1) - len(measured)
    if s2:
        final["stage2_grounding_average"] = _avg(s2)
        final["stage2_rounds"] = len(s2)
        measured2 = [r for r in s2 if not r.get("borrowed")]
        if len(measured2) != len(s2):
            final["stage2_grounding_average_measured_only"] = _avg(measured2)
            final["stage2_borrowed_rounds"] = len(s2) - len(measured2)
    scored_rounds = primary
    print("!" * 78)
    print("주의: camera/object_type(사람은 situation)/landmark/relation/anchor 와 유형별")
    print("배점(사람 문항은 landmark·relation 이 0)은 과거 39건 전수 대조로 검증된")
    print("채점식이지만, target(좌표) 점수는 근사치다 -- 실제 라이브 채점은")
    print("물체 크기/영역까지 보는 것으로 보여 중심점 거리(1-오차/5)만으로는 벗어나는")
    print("사례가 있었다. 총점도 그 영향을 받으니 참고용으로만 볼 것 (정확한 총점은")
    print("Isaac Sim 라이브 실행에서만 확인 가능).")
    print("Stage2 는 공식적으로 grounding/집기/이동 3요소로 채점되는데 가중치가 비공개다.")
    print("여기서는 셋을 따로 내고 총점 환산은 하지 않는다 (아래 'Stage 2 실행' 절 참고).")
    print("!" * 78)
    print()

    s2_exec, s2_summary = None, None
    if s2 and not args.no_stage2_exec:
        s2_exec = stage2_execution(args.scenario, args.stage2_occupancy)
        if s2_exec:
            s2_summary = summarize_stage2_execution(s2_exec)
            # report() 호출 *전에* 채워야 --json 이 report() 안에서 한 번에 정확히
            # 쓰인다 (report() 가 유일한 JSON 저장 지점 -- cmp_reports.py 가 기대하는
            # rounds[i]["fields"] 스키마를 다른 곳에서 덮어쓰지 않는다).
            final["stage2_execution"] = s2_summary

    rc = report(scored_rounds, final, args.json)
    if s2_exec:
        print_stage2_execution(s2_exec, s2_summary,
                               final.get("stage2_grounding_average"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
