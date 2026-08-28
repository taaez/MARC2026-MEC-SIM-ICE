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

_VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
_TAEJEONG_DIR = os.path.dirname(_VALIDATION_DIR)
_MEC_ROOT = os.path.dirname(_TAEJEONG_DIR)          # mec_sim_ice/
sys.path.insert(0, _VALIDATION_DIR)  # score_report
sys.path.insert(0, _TAEJEONG_DIR)   # command_parser / grounding / yolo_adapter / run_offline

from command_parser import parse_command          # noqa: E402
from dino_fallback import DinoFallbackDetector      # noqa: E402
from dino_fallback import is_disabled as dino_disabled  # noqa: E402
from fallback import dino_augment, vlm_assist       # noqa: E402
from grounding import resolve                      # noqa: E402
from run_offline import load_round                 # noqa: E402
from vlm_fallback import VlmTargetResolver          # noqa: E402
from vlm_fallback import is_disabled as vlm_disabled  # noqa: E402
from yolo_adapter import infer_cameras              # noqa: E402

from chungmu_gt import DEFAULT_SCENARIO, load_gt    # noqa: E402  (same dir)
from score_report import report                     # noqa: E402  (same dir)

COORD_TOLERANCE_M = 5.0

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

      * 라이브와 같은 env 스위치(MARC_DISABLE_DINO_FALLBACK / MARC_DISABLE_VLM)가 켜져
        있으면 끈다 -- 두 경로가 같은 스위치를 보게 해서 동작이 갈라지지 않게 한다.
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
        if not forced:
            return False, "MARC_DISABLE_DINO_FALLBACK / MARC_DISABLE_VLM 이 켜져 있음"
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

    dino = vlm = None
    if use_fallback and not dino_disabled():
        try:
            dino_kw = {}
            if args.dino_box_thr is not None:
                dino_kw["box_thr"] = args.dino_box_thr
            if args.dino_text_thr is not None:
                dino_kw["text_thr"] = args.dino_text_thr
            dino = DinoFallbackDetector(**dino_kw)
        except Exception as e:  # noqa: BLE001
            print(f"[fallback] DINO 로드 실패, target/landmark 재탐색 없이 진행: {e}",
                  file=sys.stderr)
    if use_fallback and not vlm_disabled():
        try:
            vlm = VlmTargetResolver()
        except Exception as e:  # noqa: BLE001
            print(f"[fallback] VLM 로드 실패, behind 중재/OOV 재해석 없이 진행: {e}",
                  file=sys.stderr)
    args.fallback = use_fallback

    # s1_* 는 Stage1 단계에서, s2_* 는 Stage2 단계에서 뜬 기록이다. 어느 단계에서 떴는지와
    # 무관하게 **문항 종류**(시나리오의 kind)로 Stage1/Stage2 를 가른다 -- 2026-08-25 기록
    # 세션은 39문항을 전부 Stage1 라운드로 출제해서 Stage2 문항(rt*)까지 s1_034~s1_039 에
    # 들어가 있기 때문이다.
    round_dirs = sorted(glob.glob(os.path.join(args.records, "s[12]_*")))
    scored_rounds = []
    unmatched = []
    for i, d in enumerate(round_dirs, 1):
        # 라운드당 DINO/VLM 폴백까지 겹치면 수 초~수십 초가 걸릴 수 있어, 진행상황이
        # 하나도 안 찍히면 (특히 --fallback 켰을 때) 멈춘 것처럼 보인다 -- 최소한
        # "몇 번째 도는 중"인지는 알 수 있게 매 라운드 시작마다 찍는다.
        print(f"[{i}/{len(round_dirs)}] {os.path.basename(d)}", flush=True)
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta, cams, images = load_round(d)
        cmd = meta["command"]
        hit = answer_key.get(cmd)
        if hit is None:
            unmatched.append((os.path.basename(d), cmd))
            continue
        problem_id, expected, weights, kind = hit

        parsed = parse_command(cmd)
        target_phrases = vlm_assist(cmd, parsed, vlm) if args.fallback else None
        dets = infer_cameras(model, images, min_conf=args.conf, imgsz=args.imgsz)
        if args.fallback:
            dino_augment(parsed, images, dets, target_phrases, dino, vlm)
        decision = resolve(parsed, dets, cams)
        submission = (flatten_payload(decision.to_payload()) if decision is not None
                      else fallback_submission(parsed))

        s = score_round(problem_id, weights, expected, submission)
        scored_rounds.append({"stage": 2 if kind == "retrieval" else 1,
                              "voice_command": cmd,
                              "record": os.path.basename(d), "scores": s})

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
        return (-ts, 0 if name.startswith("s2_") else 1, name)

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
    if s1:
        final["stage1_average"] = sum(r["scores"]["total"] for r in s1) / len(s1)
        final["stage1_rounds"] = len(s1)
    if s2:
        final["stage2_grounding_average"] = sum(r["scores"]["total"] for r in s2) / len(s2)
        final["stage2_rounds"] = len(s2)
    scored_rounds = primary
    print("!" * 78)
    print("주의: camera/object_type/landmark/relation/anchor 는 과거 39건 전수 대조로")
    print("검증된 정확한 채점식이지만, target(좌표) 점수는 근사치다 -- 실제 라이브 채점은")
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
