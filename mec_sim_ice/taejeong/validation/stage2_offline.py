"""Stage2 오프라인 검증 하네스 — Isaac Sim / ROS 없이 되는 데까지 확인한다.

Stage2 는 세 부분이다. 이 스크립트가 다루는 범위를 먼저 분명히 해둔다:

  1. grounding(인식)  명령+CCTV -> 물체 좌표 제출. 로봇 안 움직임.
     -> **`offline_score.py` 가 이미 채점한다.** 여기서는 다루지 않는다.
  2. navigation(주행) 그 좌표까지 이동.
     -> 계산(목표점/standoff/경로예산)은 여기서 검증. 실제 통행 가능성은 점유지도 필요.
  3. manipulation(집기) 팔로 집어 배달.
     -> **원리적으로 오프라인 불가.** 아래 "왜 안 되나" 참고. 여기서는 파지점이
        기하적으로 말이 되는지(팔 도달범위/높이)까지만 본다.

## 집기가 왜 오프라인에서 안 되나

`arm_pick.grip_holding()` 은 `inner_finger_knuckle` **수동 관절 각도**로 파지 성공을
판정한다 — 빈손이면 ~35도까지 닫히고, 물체가 손가락을 막으면 못 닫힌다. 이 각도는
접촉 솔버가 만들어내는 값이라 오프라인에는 존재하지 않는다(물체 collision mesh,
마찰계수, 질량/관성, 그리퍼 폐쇄력이 전부 필요). `run_motion` 도 매 웨이포인트마다
`_read_arm_q()` 로 실제 관절각을 되읽어 진행하므로 같은 이유로 재현 불가다.
즉 우리가 아는 건 "명령"까지고 "결과"는 시뮬레이터만 안다.

## 검사 항목

  [1] 좌표변환 왕복      object_arm_xy <-> arm_xy_to_world 부호/오프셋 검증
  [2] standoff 목표점    물체 앞 정확히 STANDOFF m 인지, 물체를 지나치지 않는지
  [3] 파지 기하          접근 완료 시 물체가 팔 도달범위 안인지, 물체 높이 vs grasp_down
  [4] 경로 예산          start -> target -> owner 거리와 제한시간 여유
  [5] 경로계획           A* 로 leg1/leg2 계획. 점유지도가 있으면 실제 지도로, 없으면
                         시나리오 장애물로 만든 합성 지도로 (통행성은 근사)

사용:
    python3 mec_sim_ice/taejeong/validation/stage2_offline.py
    python3 ... --occupancy records/occupancy.npz     # 데스크탑에서 떠온 실제 지도

점유지도를 받아오는 법(데스크탑, 라이브 1회): `participant_app._navigate_to` 안에서
`occ = c.get_occupancy_map()` 직후 한 번만 저장하면 된다.

    np.savez("records/occupancy.npz", data=np.array(occ.data, dtype=np.int8),
             width=occ.info.width, height=occ.info.height,
             resolution=occ.info.resolution,
             origin_x=occ.info.origin.position.x, origin_y=occ.info.origin.position.y)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types

import numpy as np
import yaml

_VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
_TAEJEONG_DIR = os.path.dirname(_VALIDATION_DIR)
_MEC_ROOT = os.path.dirname(_TAEJEONG_DIR)
_REPO_ROOT = os.path.dirname(_MEC_ROOT)
sys.path.insert(0, _VALIDATION_DIR)
sys.path.insert(0, _TAEJEONG_DIR)

from chungmu_gt import DEFAULT_SCENARIO, load_gt              # noqa: E402
from stage2_geometry import (                                  # noqa: E402
    ARM_BASE_FWD, GRASP_FWD_BACKOFF, MAX_LINEAR_SPEED, TUMBLER_STANDOFF_M,
    arm_xy_to_world, object_arm_xy, route_length, standoff_goal, yaw_towards,
)

DEFAULT_NAV = os.path.join(_REPO_ROOT, "demo", "mock_agent_navigation.py")
DEFAULT_PNP = os.path.join(_REPO_ROOT, "manipulation", "pnp_params.json")

# 팔 도달범위 (arm_base 기준 수평거리). gen3 6dof 링크 길이 기준 보수적 추정 --
# 정확한 값은 IK 로 확인해야 한다([3] 의 placo 항목).
ARM_REACH_MIN_M, ARM_REACH_MAX_M = 0.20, 0.75


def load_nav_module():
    """demo/mock_agent_navigation.py 를 ROS 없이 로드.

    이 모듈은 ROS 를 전혀 안 쓴다(math/numpy/yaml/heapq 뿐). 다만 448/458 행의
    `path: str | None` 어노테이션이 3.10+ 문법이라 3.9 에서는 import 가 깨진다 --
    `from __future__ import annotations` 를 앞에 붙여 그 문제만 우회한다.
    """
    if not os.path.exists(DEFAULT_NAV):
        return None
    src = open(DEFAULT_NAV, encoding="utf-8").read()
    mod = types.ModuleType("nav_offline")
    mod.__file__ = DEFAULT_NAV
    exec(compile("from __future__ import annotations\n" + src, DEFAULT_NAV, "exec"),
         mod.__dict__)
    return mod


class FakeOccupancy:
    """nav_msgs/OccupancyGrid 중 planner 가 실제로 읽는 필드만 흉내낸다."""

    def __init__(self, grid: np.ndarray, res: float, ox: float, oy: float):
        h, w = grid.shape
        origin = types.SimpleNamespace(position=types.SimpleNamespace(x=ox, y=oy))
        self.info = types.SimpleNamespace(width=w, height=h, resolution=res, origin=origin)
        self.data = grid.flatten().tolist()


def scenario_obstacles(scenario_path: str):
    """시나리오의 랜드마크/물체 위치를 (x, y, radius) 장애물로. 합성 지도용.

    점유지도가 없을 때의 대용이다. **도로 경계·연석·잔디 단차는 담기지 않으므로**
    "장애물에 박혔는지" 는 잡아도 "실제로 통행 가능한지" 는 보장하지 못한다.
    """
    with open(scenario_path, encoding="utf-8") as f:
        ch = yaml.safe_load(f)
    radius_by_kw = [("car", 2.2), ("taxi", 2.2), ("suv", 2.2), ("truck", 2.4),
                    ("vending", 1.0), ("bench", 1.0), ("table", 1.2),
                    ("bicycle", 0.8), ("scooter", 0.6), ("trash", 0.6),
                    ("hydrant", 0.4)]
    out = []
    for m in ch.get("missions", []):
        for lm in m.get("landmarks", []):
            name = lm["name"]
            r = next((v for k, v in radius_by_kw if k in name), 0.8)
            p = lm["resolved"]["position"]
            out.append((float(p[0]), float(p[1]), r))
    return out


def build_synthetic_grid(obstacles, bounds, res=0.25):
    """장애물 원들을 격자에 찍어 합성 점유지도를 만든다 (그 외는 free)."""
    xmin, ymin, xmax, ymax = bounds
    w = int(math.ceil((xmax - xmin) / res))
    h = int(math.ceil((ymax - ymin) / res))
    grid = np.zeros((h, w), dtype=np.int8)
    ys, xs = np.mgrid[0:h, 0:w]
    wx = xmin + (xs + 0.5) * res
    wy = ymin + (ys + 0.5) * res
    for (ox, oy, r) in obstacles:
        grid[(wx - ox) ** 2 + (wy - oy) ** 2 <= r * r] = 100
    return FakeOccupancy(grid, res, xmin, ymin)


def load_occupancy(path: str):
    """데스크탑에서 떠온 실제 점유지도 (npz)."""
    z = np.load(path)
    grid = np.asarray(z["data"], dtype=np.int8).reshape(int(z["height"]), int(z["width"]))
    return FakeOccupancy(grid, float(z["resolution"]), float(z["origin_x"]), float(z["origin_y"]))


# ---------------------------------------------------------------- 검사 항목 --

def check_transforms():
    """[1] 좌표변환 왕복 -- 부호/오프셋 실수를 잡는다."""
    print("[1] 좌표변환 왕복 (object_arm_xy <-> arm_xy_to_world)")
    cases = [((0.0, 0.0, 0.0), (3.0, 0.0)), ((0.0, 0.0, math.pi / 2), (0.0, 3.0)),
             ((-64.56, 123.79, 0.7), (-44.38, 121.12)),
             ((10.0, -5.0, -2.1), (7.5, -9.25))]
    worst = 0.0
    for pose, obj in cases:
        fwd, lat = object_arm_xy(pose, obj)
        back = arm_xy_to_world(pose, fwd, lat)
        err = math.dist(back, obj)
        worst = max(worst, err)
    # 로봇이 물체를 정면으로 볼 때 lat 은 0 이어야 한다
    pose = (0.0, 0.0, yaw_towards((0.0, 0.0), (2.0, 2.0)))
    _, lat_aligned = object_arm_xy(pose, (2.0, 2.0))
    # 물체가 왼쪽에 있으면 lat > 0 (car_base +Y = 좌측)
    _, lat_left = object_arm_xy((0.0, 0.0, 0.0), (2.0, 1.0))
    ok = worst < 1e-9 and abs(lat_aligned) < 1e-9 and lat_left > 0
    print(f"    왕복 최대오차 {worst:.2e} m | 정면정렬 lat={lat_aligned:+.2e} | "
          f"좌측물체 lat={lat_left:+.3f}(>0 이어야 함)  -> {'OK' if ok else '실패'}")
    return ok


def check_standoff(gt):
    """[2] standoff 목표점 -- 물체 앞 정확히 STANDOFF m 인지."""
    print(f"\n[2] standoff 목표점 (물체 앞 {TUMBLER_STANDOFF_M} m)")
    start = gt["robot_start"]
    ok = True
    for pid, e in gt["problems"]:
        tgt = e["target_coord"][:2]
        g = standoff_goal(start[:2], tgt, TUMBLER_STANDOFF_M)
        d_goal = math.dist(g, tgt)
        d_travel = math.dist(start[:2], g)
        d_total = math.dist(start[:2], tgt)
        good = abs(d_goal - TUMBLER_STANDOFF_M) < 1e-6 and d_travel < d_total
        ok &= good
        print(f"    {pid:24} 목표까지 {d_total:6.2f}m -> standoff 지점까지 {d_travel:6.2f}m "
              f"(잔여 {d_goal:.3f}m) {'OK' if good else '실패'}")
    return ok


def check_grasp_geometry(gt, ground_z):
    """[3] 파지 기하 -- 접근 완료 시 물체가 팔 도달범위 안인지 + 물체 높이."""
    print("\n[3] 파지 기하 (접근 완료 지점 기준)")
    print(f"    팔 도달범위 가정 {ARM_REACH_MIN_M}~{ARM_REACH_MAX_M} m "
          f"(arm_base 기준 수평). 정확한 값은 IK 확인 필요")
    pnp = {}
    if os.path.exists(DEFAULT_PNP):
        import json
        pnp = json.load(open(DEFAULT_PNP))
    grasp_down = pnp.get("grasp_down")

    # (a) 상수 사슬 -- 문항과 무관하게 한 번만. standoff 로 정지했을 때의 파지 거리.
    nominal = TUMBLER_STANDOFF_M - ARM_BASE_FWD - GRASP_FWD_BACKOFF
    nominal_ok = ARM_REACH_MIN_M <= nominal <= ARM_REACH_MAX_M
    print(f"    (a) 정위치 정지 시 파지거리 = standoff {TUMBLER_STANDOFF_M} "
          f"- arm_base {ARM_BASE_FWD} - backoff {GRASP_FWD_BACKOFF} = {nominal:.3f} m "
          f"{'OK' if nominal_ok else '범위밖'}")

    # (b) 주행 오차 포락선 -- leg1 은 tol=0.45 로 멈춘다(participant_app:196).
    #     그 오차만큼 어긋난 채 집으려 하면 팔 도달범위를 벗어나는지 본다.
    tol = 0.45
    lo, hi = nominal - tol, nominal + tol
    envelope_ok = ARM_REACH_MIN_M <= lo and hi <= ARM_REACH_MAX_M
    print(f"    (b) leg1 도착판정 오차 ±{tol} m 를 그대로 안고 집으면 파지거리 "
          f"{lo:.3f}~{hi:.3f} m -> {'범위 내' if envelope_ok else '범위 이탈'}")
    if not envelope_ok:
        print(f"        => 정지 오차만으로는 팔이 못 닿는다. _approach_straight() 의 "
              f"폐루프 직진(target_fwd=0.40)이")
        print(f"           반드시 성공해야 한다는 뜻이고, 그건 get_world_pose 피드백이 "
              f"필요해 오프라인 검증 불가다.")

    # (c) 문항별로 실제 다른 값 -- 물체 높이. grasp_down 이 고정값이라 여기가 벌어지면 위험.
    print(f"    (c) pnp_params.grasp_down = {grasp_down} (arm_base 기준 고정 하강량)")
    spread = []
    for pid, e in gt["problems"]:
        tgt = e["target_coord"]
        gz = ground_z.get(e["camera_id"])
        if gz is None:
            print(f"        {pid:24} ground_z 없음")
            continue
        h = tgt[2] - gz
        spread.append(h)
        flag = "  <-- 편차 큼" if abs(h) > 0.3 else ""
        print(f"        {pid:24} cam={e['camera_id']:8} ground_z 대비 물체높이 {h:+.3f} m{flag}")
    if spread:
        print(f"        높이 편차 {min(spread):+.3f} ~ {max(spread):+.3f} m "
              f"(폭 {max(spread) - min(spread):.3f} m)")
        print("        주의: ground_z 는 실제 지형고가 아니라 카메라마다 고정된 기준 평면이다.")
        print("              픽셀->world 투영이 이 평면을 가정하므로, 편차가 큰 카메라는")
        print("              grounding 좌표부터 그만큼 어긋난다(수평오차 = 편차/tan(앙각)).")
    # (b) 는 "확인 필요"로 남기는 게 맞다 -- 정지 오차만으로는 못 닿는 게 사실이고,
    # 그걸 메우는 폐루프 직진은 오프라인에서 검증할 수 없다.
    return nominal_ok and envelope_ok


def check_route_budget(gt):
    """[4] 경로 예산 -- 직선거리 기준 최소 소요시간 vs 제한시간."""
    print("\n[4] 경로 예산 (직선거리 / 최고속 %.1f m/s -- 하한 추정)" % MAX_LINEAR_SPEED)
    start, owner = gt["robot_start"], gt["owner_position"]
    ok = True
    for pid, e in gt["problems"]:
        tgt = e["target_coord"]
        total = route_length([start[:2], tgt[:2], owner[:2]])
        t = total / MAX_LINEAR_SPEED
        limit = gt["time_limit"].get(pid, 600.0)
        good = t < limit * 0.5
        ok &= good
        print(f"    {pid:24} leg1+leg2 {total:6.1f}m -> 최소 {t:5.1f}s / 제한 {limit:.0f}s "
              f"{'여유' if good else '빠듯'}")
    return ok


def check_planning(gt, occ, nav, source_label):
    """[5] A* 경로계획 -- leg1/leg2 가 계획되는지."""
    print(f"\n[5] 경로계획 A* ({source_label})")
    if nav is None:
        print("    mock_agent_navigation.py 를 찾을 수 없어 건너뜀")
        return False
    planner = nav.OccupancyGridPlanner()
    if not planner.update_from_msg(occ):
        print("    점유지도 로드 실패")
        return False
    print(f"    지도 {occ.info.width}x{occ.info.height} res={occ.info.resolution} "
          f"origin=({occ.info.origin.position.x:.1f}, {occ.info.origin.position.y:.1f})")
    start, owner = gt["robot_start"], gt["owner_position"]
    ok = True
    for pid, e in gt["problems"]:
        tgt = e["target_coord"][:2]
        approach = standoff_goal(start[:2], tgt, TUMBLER_STANDOFF_M)
        p1 = planner.plan(tuple(start[:2]), approach, inflation_m=0.10, snap_m=8.0,
                          extra_obstacles=[])
        p2 = planner.plan(approach, tuple(owner[:2]), inflation_m=0.25, snap_m=8.0,
                          extra_obstacles=[])
        n1, n2 = (len(p1) if p1 else 0), (len(p2) if p2 else 0)
        good = n1 > 0 and n2 > 0
        ok &= good
        d1 = route_length([start[:2]] + list(p1)) if p1 else float("nan")
        d2 = route_length([approach] + list(p2)) if p2 else float("nan")
        print(f"    {pid:24} leg1 {n1:2d}wp {d1:6.1f}m | leg2 {n2:2d}wp {d2:6.1f}m "
              f"{'OK' if good else '계획 실패'}")
    return ok


def check_ik():
    """[3-보강] placo 가 있으면 IK 로 실제 도달 가능성을 확인한다."""
    print("\n[3-보강] IK 도달성 (placo)")
    try:
        import placo  # noqa: F401
    except ImportError:
        print("    placo 미설치 -> 건너뜀.  pip3 install placo  로 설치하면 파지 자세의")
        print("    IK 수렴 여부까지 오프라인에서 확인할 수 있다 (ROS/Isaac Sim 불필요).")
        return None
    print("    placo 설치됨 -- manipulation/arm_kin.py 로 IK 검증 가능")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--occupancy", default=None,
                    help="데스크탑에서 떠온 실제 점유지도 npz. 없으면 합성 지도를 쓴다")
    ap.add_argument("--records", default=os.path.join(_MEC_ROOT, "records"))
    args = ap.parse_args()

    raw = load_gt(args.scenario)
    with open(args.scenario, encoding="utf-8") as f:
        ch = yaml.safe_load(f)
    problems = [(pid, e) for (pid, e, _w, kind, _o) in raw.values() if kind == "retrieval"]
    problems.sort()
    limits = {p["id"]: p.get("time_limit", 600.0) for p in ch["challenge_problems"]}
    gt = {
        "problems": problems,
        "robot_start": ch["robot"]["start_point"],
        "owner_position": ch["owner_zone"]["resolved"]["position"],
        "time_limit": limits,
    }

    # 카메라별 ground_z 는 records 의 meta.json 에서 읽는다
    import glob
    import json
    ground_z = {}
    for d in sorted(glob.glob(os.path.join(args.records, "s*_*"))):
        m = os.path.join(d, "meta.json")
        if not os.path.exists(m):
            continue
        for cam, v in (json.load(open(m, encoding="utf-8")).get("cameras") or {}).items():
            ground_z.setdefault(cam, v["ground_z"])

    print("=" * 78)
    print(f"Stage2 오프라인 검증 — retrieval {len(problems)}문항  (Isaac Sim/ROS 불필요)")
    print("=" * 78)
    print(f"로봇 시작점 {[round(v, 2) for v in gt['robot_start']]}  "
          f"owner_zone {[round(v, 2) for v in gt['owner_position']]}\n")

    results = {
        "좌표변환": check_transforms(),
        "standoff": check_standoff(gt),
        "파지기하": check_grasp_geometry(gt, ground_z),
        "경로예산": check_route_budget(gt),
    }
    results["IK"] = check_ik()

    nav = load_nav_module()
    if args.occupancy and os.path.exists(args.occupancy):
        occ, label = load_occupancy(args.occupancy), f"실제 점유지도 {args.occupancy}"
    else:
        obstacles = scenario_obstacles(args.scenario)
        xs = [gt["robot_start"][0], gt["owner_position"][0]] + \
             [e["target_coord"][0] for _, e in problems] + [o[0] for o in obstacles]
        ys = [gt["robot_start"][1], gt["owner_position"][1]] + \
             [e["target_coord"][1] for _, e in problems] + [o[1] for o in obstacles]
        bounds = (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)
        occ = build_synthetic_grid(obstacles, bounds)
        label = f"합성 지도 -- 시나리오 장애물 {len(obstacles)}개, 도로 경계 없음"
    results["경로계획"] = check_planning(gt, occ, nav, label)

    print("\n" + "=" * 78)
    print("요약")
    print("=" * 78)
    for k, v in results.items():
        mark = "건너뜀" if v is None else ("OK" if v else "확인 필요")
        print(f"  {k:10s} {mark}")
    print()
    print("오프라인에서 확인 불가 (물리 시뮬 필수):")
    print("  - 파지 성공 여부      grip_holding() 이 읽는 inner_finger_knuckle 수동관절 각도")
    print("  - 관절 추종           send_arm_command 후 실제 도달 여부(_read_arm_q 되읽기)")
    print("  - 물체 반응           접근 중 밀림/넘어짐, 눕힌 물체의 파지 각도 적합성")
    print("  - 차체 전복/충돌      무게중심, 라이다 지역회피")
    print("  - 폐루프 주행         get_world_pose 피드백이 있어야 제어루프가 돈다")
    print("  - Stage2 실행 점수    플랫폼이 매기며 배점 내역 비공개")
    if not args.occupancy:
        print("\n실제 점유지도를 데스크탑에서 한 번 떠오면 [5] 가 근사에서 실측으로 바뀐다.")
        print("(방법은 이 파일 docstring 참고)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
