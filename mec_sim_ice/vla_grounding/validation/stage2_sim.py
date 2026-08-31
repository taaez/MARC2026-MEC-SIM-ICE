"""Stage2 **실행** 오프라인 하네스 -- `demo/participant_app.py` 를 원본 그대로 굴려본다.

`stage2_offline.py` 는 계산을 **검사**한다(standoff 가 맞나, A* 가 경로를 뱉나).
이 스크립트는 그 코드를 **실행**한다: 가짜 MARCClient 를 끼워 넣고 participant_app 의
`_drive()` 를 통째로 돌려, 주행 제어루프와 placo 팔 서보가 실제로 물체까지 갔다가
owner 까지 배달하는지를 시뮬 시간으로 확인한다. ROS·Isaac Sim 불필요.

    stage2_offline.py   계산이 맞나          (정적 검사)
    stage2_sim.py       코드가 굴러가나      (동적 실행)   <- 이 파일

무엇이 진짜 검증되고 무엇이 가정인지는 `fake_client.py` 의 docstring 을 반드시 읽을 것.
요약하면 차체 운동학·A*·PathFollower·placo FK/IK/DLS 서보는 진짜로 돌고,
**파지 성공 판정만 가정**이다(접촉 물리가 없어 원리적으로 불가).

사용:
    python3 mec_sim_ice/vla_grounding/validation/stage2_sim.py
    python3 ... --problem rt01_parking_tumbler --verbose
    python3 ... --occupancy mec_sim_ice/records/occupancy.npz   # 실제 지도가 있으면
    python3 ... --hint-noise 0.5      # reveal 이 주는 위치가 GT 에서 어긋날 때의 강건성
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import types

_VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
_VLA_GROUNDING_DIR = os.path.dirname(_VALIDATION_DIR)
_MEC_ROOT = os.path.dirname(_VLA_GROUNDING_DIR)
_REPO_ROOT = os.path.dirname(_MEC_ROOT)
sys.path.insert(0, _VALIDATION_DIR)
sys.path.insert(0, _VLA_GROUNDING_DIR)

import yaml                                                        # noqa: E402

import fake_client                                                 # noqa: E402
from chungmu_gt import DEFAULT_SCENARIO, load_gt                    # noqa: E402
from stage2_offline import (                                        # noqa: E402
    build_synthetic_grid, load_occupancy, scenario_obstacles,
)


def build_gt(scenario_path):
    """retrieval 문항 + 로봇 시작자세 + owner 목적지."""
    raw = load_gt(scenario_path)
    with open(scenario_path, encoding="utf-8") as f:
        ch = yaml.safe_load(f)
    problems = sorted((pid, e) for (pid, e, _w, kind, _o) in raw.values()
                      if kind == "retrieval")
    return {
        "problems": problems,
        "robot_start": ch["robot"]["start_point"],
        "robot_yaw": math.radians(float(ch["robot"].get("yaw", 0.0))),
        "owner_position": ch["owner_zone"]["resolved"]["position"],
        "time_limit": {p["id"]: p.get("time_limit", 600.0)
                       for p in ch["challenge_problems"]},
        "voice": {p["id"]: p.get("voice_command", "")
                  for p in ch["challenge_problems"]},
    }


def make_occupancy(scenario_path, gt, occupancy_path, hide_obstacles=False):
    """합성 점유지도. `hide_obstacles` 면 장애물을 **지도에 안 넣는다**.

    라이브 점유지도에는 주차 차량이 없다는 게 데스크탑에서 실측으로 확인됐다(로봇이
    경로상의 차에 그대로 머리를 박는다). 기본 합성 지도는 장애물을 격자에 구워버려서
    그 상황이 재현되지 않으므로, 이 모드에서는 지도는 비우고 대신 같은 원판을 차체
    **충돌체**로 넘겨(run_one) 라이브와 같은 조건을 만든다.
    """
    if occupancy_path and os.path.exists(occupancy_path):
        return load_occupancy(occupancy_path), f"실제 점유지도 {occupancy_path}"
    obstacles = scenario_obstacles(scenario_path)
    xs = ([gt["robot_start"][0], gt["owner_position"][0]]
          + [e["target_coord"][0] for _, e in gt["problems"]] + [o[0] for o in obstacles])
    ys = ([gt["robot_start"][1], gt["owner_position"][1]]
          + [e["target_coord"][1] for _, e in gt["problems"]] + [o[1] for o in obstacles])
    bounds = (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)
    if hide_obstacles:
        return (build_synthetic_grid([], bounds),
                f"합성 지도 -- **장애물 없음**(라이브 점유지도 재현), "
                f"장애물 {len(obstacles)}개는 차체 충돌체로만 존재")
    return (build_synthetic_grid(obstacles, bounds),
            f"합성 지도 -- 시나리오 장애물 {len(obstacles)}개, 도로 경계 없음")


def colliders_for(gt, obj, args):
    """이 문항에서 차체가 실제로 못 뚫는 원판 목록.

    출발점/대상/owner 를 품는 원판은 뺀다 -- 그건 애초에 풀 수 없는 문항이라
    "정체 기억이 우회를 만드는가" 라는 질문과 무관한 실패가 되기 때문이다.
    """
    if not getattr(args, "hidden_obstacles", False):
        return []
    keep, start = [], gt["robot_start"]
    anchors = [(obj[0], obj[1]), (gt["owner_position"][0], gt["owner_position"][1]),
               (start[0], start[1])]
    for (ox, oy, orad) in gt.get("colliders", []):
        if any(math.hypot(ax - ox, ay - oy) <= float(orad) + 1.0 for ax, ay in anchors):
            continue
        keep.append((ox, oy, orad))
    return keep


def run_one(pid, entry, gt, occ, mods, args):
    """문항 하나를 참가자 앱 그대로 실행하고 계측값을 돌려준다."""
    participant_app, arm_pick = mods["app"], mods["arm_pick"]
    GroundingResult = mods["types"].GroundingResult
    Stage2Mission, Stage2Reveal = mods["types"].Stage2Mission, mods["types"].Stage2Reveal

    obj = [float(v) for v in entry["target_coord"]]
    owner = gt["owner_position"]
    start = (gt["robot_start"][0], gt["robot_start"][1], gt["robot_yaw"])

    client = fake_client.FakeMARCClient(
        start_pose=start, occupancy=occ, object_xyz=obj,
        home_q=arm_pick.HOME, kin=arm_pick._kin(),
        capture_radius=args.capture_radius,
        arm_base_fwd=participant_app.ARM_BASE_FWD,
        sim_time_cap=args.sim_cap, stall_rule=not args.no_stall_rule,
        colliders=colliders_for(gt, obj, args),
    )
    fake_client.set_current_client(client)
    fake_client.patch_sim_clock(client)

    # grounding 답은 offline_score.py 소관이므로 여기서는 정답을 그대로 넣는다.
    fake_client._StubVLA.stage2_answer = GroundingResult(
        camera_id=entry.get("camera_id", ""), target_type=entry.get("target_type", ""),
        anchor_coord=entry.get("anchor_coord", [0, 0, 0]), target_coord=obj,
        landmark=entry.get("landmark_prefix", ""), relation=entry.get("relation"),
    )

    app = participant_app.DemoParticipant()

    # -- 계측 훅: 원본을 고치지 않고 반환값만 가로챈다 --
    legs = []
    _nav = app._navigate_to

    def nav(goal, deadline, *a, **kw):
        t0, p0 = client.now_s(), client.get_world_pose()
        reached = _nav(goal, deadline, *a, **kw)
        x, y, _ = client.get_world_pose()
        legs.append({"goal": goal, "reached": reached,
                     "sim_s": client.now_s() - t0,
                     "err_m": math.hypot(goal[0] - x, goal[1] - y),
                     "from": p0})
        return reached
    app._navigate_to = nav

    approach_end = {}
    _straight = app._approach_straight

    def straight(obj_xy, deadline, *a, **kw):
        t0, s0 = client.now_s(), client.chassis.stall_ticks
        p0 = client.get_world_pose()
        approach_end["arm_xy_before"] = app._object_arm_xy(obj_xy)
        send = client.send_cmd_vel
        if args.fix_approach_cap:
            # 후보 수정안 A/B -- participant_app 은 손대지 않고, 직진접근 구간에서
            # 나가는 cmd_vel 에만 GoalSeekController 와 같은 Ackermann 상한을 씌운다.
            def capped(linear_x=0.0, linear_y=0.0, angular_z=0.0, twist=None):
                if abs(linear_x) > 1e-3:
                    wm = 1.8 * abs(linear_x) / fake_client.CHASSIS_STEERING_TRACK
                    angular_z = max(-wm, min(wm, angular_z))
                return send(linear_x, linear_y, angular_z, twist)
            client.send_cmd_vel = capped
        try:
            r = _straight(obj_xy, deadline, *a, **kw)
        finally:
            client.send_cmd_vel = send
        x, y, _ = client.get_world_pose()
        approach_end["arm_xy"] = app._object_arm_xy(obj_xy)
        approach_end["sim_s"] = client.now_s() - t0
        approach_end["stall"] = client.chassis.stall_ticks - s0
        approach_end["moved_m"] = math.hypot(x - p0[0], y - p0[1])
        return r
    app._approach_straight = straight

    pick_ret = {}
    _pick = participant_app.run_pick_sequence

    def pick(c, log, grasp_xy=None):
        pick_ret["grasp_xy"] = grasp_xy
        t0 = client.now_s()
        held = _pick(c, log, grasp_xy)
        pick_ret["held"] = held
        pick_ret["sim_s"] = client.now_s() - t0
        return held
    participant_app.run_pick_sequence = pick

    mission = Stage2Mission(task_description=gt["voice"].get(pid, ""),
                            time_limit=gt["time_limit"].get(pid, 600.0),
                            owner_position=list(owner))
    client.fire_stage2_mission(mission)

    hint = list(obj)
    if args.hint_noise:
        hint[0] += args.hint_noise           # 결정론 유지를 위해 +x 방향 고정 오프셋
    client.fire_stage2_reveal(Stage2Reveal(
        grounding_score=None, target_type=entry.get("target_type", ""),
        hint_center=hint, hint_radius=1.0))

    wall0 = time.time()
    err = None
    try:
        client.run_stage2()
    except Exception as e:                   # 앱이 터지는 것도 결과다
        err = f"{type(e).__name__}: {e}"
    wall = time.time() - wall0
    participant_app.run_pick_sequence = _pick   # 다음 문항을 위해 원복

    fx, fy, _ = client.get_world_pose()
    return {
        "pid": pid, "error": err,
        "sim_s": client.now_s(), "wall_s": wall,
        "limit_s": gt["time_limit"].get(pid, 600.0),
        "legs": legs,
        "approach_arm_xy": approach_end.get("arm_xy"),
        "grasp_xy": pick_ret.get("grasp_xy"),
        "held": pick_ret.get("held"),
        "pick_sim_s": pick_ret.get("sim_s"),
        "grasp_err_m": client.arm.grasp_attempt_err,
        "grasp_ee": client.arm.grasp_attempt_ee,
        "capture": client.arm.captured,
        "in_basket": client.arm.in_basket,
        "approach": dict(approach_end),
        "delivery_err_m": math.hypot(owner[0] - fx, owner[1] - fy),
        "task_complete": client.task_completed,
        "hit_cap": client.hit_time_cap,
        "stall_ticks": client.chassis.stall_ticks,
        "blocked_ticks": client.chassis.blocked_ticks,
        "clamped": client.chassis.clamped,
        "travelled_m": client.chassis.dist_travelled,
        "max_joint_lag": client.arm.max_joint_lag,
    }


def run_execution(scenario=DEFAULT_SCENARIO, occupancy=None, problem=None,
                  hint_noise=0.0, capture_radius=None, sim_cap=1800.0,
                  stall_rule=True, fix_approach_cap=False, hidden_obstacles=False,
                  progress=True, header=False):
    """rt 문항들을 participant_app 그대로 실행하고 `(rows, info)` 를 돌려준다.

    `offline_score.py` 도 이 함수를 불러 Stage2 실행 결과를 채점표에 붙인다.
    """
    args = types.SimpleNamespace(
        capture_radius=(fake_client.GRASP_CAPTURE_RADIUS_M if capture_radius is None
                        else float(capture_radius)),
        sim_cap=float(sim_cap), no_stall_rule=not stall_rule,
        fix_approach_cap=bool(fix_approach_cap), hint_noise=float(hint_noise),
        hidden_obstacles=bool(hidden_obstacles))

    gt = build_gt(scenario)
    if problem:
        gt["problems"] = [(p, e) for p, e in gt["problems"] if p == problem]
        if not gt["problems"]:
            raise LookupError(f"그런 retrieval 문항이 없다: {problem}")
    occ, occ_label = make_occupancy(scenario, gt, occupancy, hide_obstacles=hidden_obstacles)
    gt["colliders"] = scenario_obstacles(scenario) if hidden_obstacles else []

    # 이 하네스가 쏘는 reveal 은 가짜다(hint_center = GT, hint_radius 고정). 앱의
    # reveal 기록(MARC_REVEAL_LOG)은 **라이브 hint 오차 실측용** 파일이므로, 여기서
    # 다른 경로로 돌려 실측치가 시뮬 값으로 오염되지 않게 한다.
    os.environ["MARC_REVEAL_LOG"] = "./results/stage2_reveals_SIM.jsonl"

    # shim 을 먼저 깔고 나서야 participant_app 을 import 할 수 있다.
    marc_types, _nav = fake_client.install_shims()
    import arm_pick                                            # noqa: E402
    import participant_app                                     # noqa: E402
    mods = {"app": participant_app, "arm_pick": arm_pick, "types": marc_types}

    if header:
        print("=" * 78)
        print(f"Stage2 실행 시뮬 — retrieval {len(gt['problems'])}문항  "
              f"(Isaac Sim/ROS 불필요)")
        print("=" * 78)
        print(f"참가자 앱   {participant_app.__file__}")
        print(f"점유지도    {occ_label}")
        print(f"로봇 시작   {[round(v, 2) for v in gt['robot_start'][:2]]} "
              f"yaw {math.degrees(gt['robot_yaw']):.1f}deg   "
              f"owner {[round(v, 2) for v in gt['owner_position'][:2]]}")
        print(f"가정        포획반경 {args.capture_radius}m, "
              f"관절속도 {fake_client.ARM_MAX_JOINT_RATE}rad/s, "
              f"가속 {fake_client.CHASSIS_ACCEL}m/s^2, hint 오차 {args.hint_noise}m"
              + ("   [무효명령 정지규칙 OFF -- 대조군]" if args.no_stall_rule else "")
              + ("   [직진접근 Ackermann 상한 적용 -- 수정안 A/B]"
                 if args.fix_approach_cap else ""))

    rows = []
    for pid, entry in gt["problems"]:
        if progress:
            print(f"\n... {pid} 실행 중", flush=True)
        rows.append(run_one(pid, entry, gt, occ, mods, args))
    return rows, {"occ_label": occ_label, "app_file": participant_app.__file__}


def _fmt_leg(leg):
    return ("도달" if leg["reached"] else "미도달") + f" {leg['sim_s']:5.1f}s 오차 {leg['err_m']:.2f}m"


def report(rows):
    print("\n" + "=" * 78)
    print("문항별 결과")
    print("=" * 78)
    for r in rows:
        print(f"\n  {r['pid']}")
        if r["error"]:
            print(f"    예외 발생: {r['error']}")
        for i, leg in enumerate(r["legs"], 1):
            name = "leg1 접근" if i == 1 else ("leg2 배달" if i == 2 else f"leg{i}")
            print(f"    {name:10} {_fmt_leg(leg)}  -> {tuple(round(v, 1) for v in leg['goal'])}")
        ap_ = r["approach"]
        bb, aa = ap_.get("arm_xy_before"), r["approach_arm_xy"]
        if aa:
            print(f"    직진접근     fwd {bb[0]:+.3f} -> {aa[0]:+.3f} (목표 0.40), "
                  f"lat {bb[1]:+.3f} -> {aa[1]:+.3f} (목표 0)")
            print(f"                 {ap_.get('sim_s', 0):.1f}s 동안 {ap_.get('moved_m', 0):.3f}m "
                  f"전진, 차체정지 {ap_.get('stall', 0)}틱")
            reach = math.hypot(aa[0], aa[1])
            print(f"                 => 팔이 겨눠야 할 거리 {reach:.3f}m "
                  f"(도달범위 상한 0.75 기준 {'초과' if reach > 0.75 else '이내'})")
        if r["grasp_err_m"] is not None:
            print(f"    파지시도     EE-물체 수평오차 {r['grasp_err_m']:.3f}m "
                  f"(포획반경 기준) -> {'포획' if r['capture'] else '빗나감'}")
        else:
            print("    파지시도     그리퍼를 닫는 지점까지 가지 못했다")
        print(f"    앱 판정      picked={r['held']}  task_complete={r['task_complete']}")
        print(f"    배달         owner 까지 {r['delivery_err_m']:.2f}m "
              f"(도착판정 0.7m), 바구니 적재={r['in_basket']}")
        print(f"    시간         시뮬 {r['sim_s']:.1f}s / 제한 {r['limit_s']:.0f}s"
              + ("  <-- 초과" if r["sim_s"] > r["limit_s"] else "")
              + f"   (벽시계 {r['wall_s']:.1f}s)")
        extra = []
        if r["stall_ticks"]:
            extra.append(f"차체정지 {r['stall_ticks']}틱(무효 조향명령)")
        if r["clamped"]:
            extra.append(f"속도상한 초과명령 {r['clamped']}회")
        if r["hit_cap"]:
            extra.append("하네스 시간상한 도달")
        if extra:
            print("    경고         " + ", ".join(extra))

    print("\n" + "=" * 78)
    print("요약")
    print("=" * 78)
    hdr = f"  {'문항':24} {'leg1':6} {'파지':6} {'leg2':6} {'배달':8} {'시뮬시간':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    n_ok = 0
    for r in rows:
        l1 = "OK" if (r["legs"] and r["legs"][0]["reached"]) else "실패"
        l2 = "OK" if (len(r["legs"]) > 1 and r["legs"][1]["reached"]) else "실패"
        gr = "OK" if r["capture"] else "빗나감"
        dl = "OK" if r["delivery_err_m"] < 0.7 and r["in_basket"] else "실패"
        if l1 == "OK" and l2 == "OK" and gr == "OK":
            n_ok += 1
        print(f"  {r['pid']:24} {l1:6} {gr:6} {l2:6} {dl:8} {r['sim_s']:8.1f}s")
    print(f"\n  주행+파지 기하 통과 {n_ok}/{len(rows)} 문항")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--occupancy", default=None,
                    help="데스크탑에서 떠온 실제 점유지도 npz. 없으면 합성 지도")
    ap.add_argument("--problem", default=None, help="문항 id 하나만 실행")
    ap.add_argument("--verbose", action="store_true", help="participant_app 로그를 그대로 출력")
    ap.add_argument("--hint-noise", type=float, default=0.0,
                    help="reveal 의 hint_center 를 GT 에서 +x 로 이만큼 어긋나게 (m)")
    ap.add_argument("--capture-radius", type=float,
                    default=fake_client.GRASP_CAPTURE_RADIUS_M,
                    help="파지 포획 반경 가정 (m)")
    ap.add_argument("--sim-cap", type=float, default=1800.0,
                    help="문항당 시뮬 시간 상한 (s). 폭주 방지")
    ap.add_argument("--fix-approach-cap", action="store_true",
                    help="후보 수정안 A/B -- _approach_straight 가 내보내는 cmd_vel 에 "
                         "GoalSeekController 와 같은 Ackermann 상한(w<=1.8v/T)을 씌워본다. "
                         "participant_app.py 는 수정하지 않는다")
    ap.add_argument("--hidden-obstacles", action="store_true",
                    help="라이브 재현 -- 장애물을 점유지도에서 빼고 차체 충돌체로만 둔다. "
                         "점유지도에 주차 차량이 없다는 실측을 반영한다(로봇이 차에 "
                         "머리를 박는 그 상황)")
    ap.add_argument("--no-stall-rule", action="store_true",
                    help="대조군 -- Ackermann 무효명령(2v<=wT) 정지 규칙을 끈다. "
                         "이 규칙이 결론을 만들어낸 게 아닌지 확인할 때")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="      | %(name)s: %(message)s")

    try:
        rows, info = run_execution(
            scenario=args.scenario, occupancy=args.occupancy, problem=args.problem,
            hint_noise=args.hint_noise, capture_radius=args.capture_radius,
            sim_cap=args.sim_cap, stall_rule=not args.no_stall_rule,
            fix_approach_cap=args.fix_approach_cap,
            hidden_obstacles=args.hidden_obstacles, progress=True, header=True)
    except LookupError as e:
        print(e)
        return 1
    report(rows)
    print("""
읽는 법 — 무엇이 진짜 검증됐나
  진짜  차체 운동학 / A* / PathFollower 제어루프 / placo FK·IK·DLS 팔 서보 /
        시간 예산. 여기서 실패하면 라이브에서도 십중팔구 실패한다.
  가정  파지 포획 판정([C]) -- 접촉 솔버가 없어 '손가락이 진짜 물었나' 는 모른다.
        여기 '포획' 은 **명령 궤적이 물체 위치까지 EE 를 데려갔다** 는 뜻이다.
  불가  물체가 밀림/넘어짐, 차체 전복, 라이다 지역회피, 합성 지도의 실제 통행성,
        플랫폼 Stage2 배점.
모델 상수는 fake_client.py 상단에 모아뒀다. 실측이 아니라 추정이므로 절대 판정이
아니라 **회귀 검출과 실패 원인 분리**에 쓴다.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
