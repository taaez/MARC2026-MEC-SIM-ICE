"""Synthetic-scene tests for grounding.py — environment-independent by design.

No scenario data is used: virtual CCTV cameras are constructed mathematically,
world objects are PROJECTED into pixel bboxes, and the pipeline must recover
camera choice, landmark instance, relation, and CENTER-point coordinates.
This proves the code is general (parse -> geometry -> coords), not fitted to
any practice scenario.

Run:  <python-with-numpy> test_grounding.py
"""

from __future__ import annotations

import numpy as np

from command_parser import parse_command
from grounding import (CameraModel, Detection, GroundingConfig, locate,
                       pixel_to_plane, resolve, world_to_pixel)

FX = 900.0
W, H = 1280, 720


def make_camera(cam_id: str, pos, look_at, ground_z: float = 0.0) -> CameraModel:
    """world 위치 pos 에서 look_at 을 바라보는 카메라 (optical 규약)."""
    pos = np.asarray(pos, dtype=np.float64)
    fwd = np.asarray(look_at, dtype=np.float64) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_cw = np.column_stack([right, down, fwd])   # x=오른쪽, y=아래, z=전방
    K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])
    return CameraModel(cam_id, K, R_cw, pos, ground_z, W, H)


def synth_det(cam: CameraModel, label: str, xy, height: float, radius: float,
              base_z: float = None, score: float = 0.9) -> Detection:
    """world 상의 직립 물체(원기둥 근사)를 카메라에 투영해 bbox 탐지를 합성."""
    bz = cam.ground_z if base_z is None else base_z
    cx, cy = xy
    us, vs = [], []
    for dx in (-radius, radius):
        for dy in (-radius, radius):
            for dz in (0.0, height):
                uv = world_to_pixel(cam, [cx + dx, cy + dy, bz + dz])
                assert uv is not None, f"{label} behind camera"
                us.append(uv[0]); vs.append(uv[1])
    return Detection(label, (min(us), min(vs), max(us), max(vs)), score)


def close(a, b, tol):
    return abs(a - b) <= tol


def main() -> int:
    fails = 0

    def check(name, cond, info=""):
        nonlocal fails
        if not cond:
            print(f"FAIL {name}: {info}")
            fails += 1

    cam = make_camera("cam_a", pos=[0, 0, 4.0], look_at=[6, 0, 0])
    cfg = GroundingConfig()

    # T1. 왕복 정합: world 지면점 → 픽셀 → 지면 재투영
    for P in ([5, 1, 0], [8, -2, 0], [3, 0.5, 0]):
        uv = world_to_pixel(cam, P)
        back = pixel_to_plane(cam, *uv)
        check("T1-roundtrip", np.allclose(back, P, atol=1e-6), f"{P} -> {back}")

    # T2. 중심점 복원: 높이 0.4m 물체 at (5, 1) → center ≈ (5, 1, 0.2)
    det = synth_det(cam, "umbrella", (5, 1), height=0.4, radius=0.05)
    loc = locate(cam, det)
    c = loc.center
    check("T2-center-xy", close(c[0], 5, 0.15) and close(c[1], 1, 0.15), c)
    check("T2-center-z", close(c[2], 0.2, 0.08), c)

    # T3. 최근접 랜드마크 인스턴스 선택 (동일 클래스 2개)
    parsed = parse_command("Find the umbrella next to the bench.")
    dets = [synth_det(cam, "umbrella", (5, 1), 0.9, 0.05),
            synth_det(cam, "bench", (5.8, 1.4), 0.8, 0.5),
            synth_det(cam, "bench", (9, -2), 0.8, 0.5)]
    d = resolve(parsed, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T3-nearest-instance", close(d.anchor_coord[0], 5.8, 0.3)
          and close(d.anchor_coord[1], 1.4, 0.3), d.anchor_coord)
    check("T3-landmark", d.landmark == "bench", d.landmark)

    # T4. 텍스트 타입 우선: trash_can 이 더 가까워도 명령이 bench 면 bench
    dets = [synth_det(cam, "umbrella", (5, 1), 0.9, 0.05),
            synth_det(cam, "trash_can", (5.4, 1), 1.0, 0.3),
            synth_det(cam, "bench", (6.5, 1), 0.8, 0.5)]
    d = resolve(parsed, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T4-text-type-priority", d.landmark == "bench", d.landmark)

    # T5. 랜드마크 미언급 → 아무 랜드마크 중 3D 최근접 (팀 결정)
    parsed5 = parse_command("Find the umbrella dropped in the empty spot.")
    check("T5-no-text-landmark", parsed5.landmark_type is None, parsed5.landmark_type)
    d = resolve(parsed5, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T5-nearest-any", d.landmark == "trash_can", d.landmark)

    # T6. 'on' 판정 + 지지면 재투영: 테이블(높이 1.0) 위 머그
    parsed6 = parse_command("I was reading at an outdoor table and left my mug there.")
    dets = [synth_det(cam, "picnic_table", (6, 0), 1.0, 0.8),
            synth_det(cam, "mug", (6, 0), 0.15, 0.05, base_z=1.0)]
    d = resolve(parsed6, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T6-relation-on", d.relation == "on", d.relation)
    check("T6-mug-xy", close(d.target_coord[0], 6, 0.35)
          and close(d.target_coord[1], 0, 0.35), d.target_coord)
    check("T6-mug-z", 0.85 <= d.target_coord[2] <= 1.25, d.target_coord)
    # anchor 는 중심점이 아니라 지면 접촉점 (Located.ground_point 주석의 실측 근거)
    check("T6-table-z", abs(d.anchor_coord[2]) <= 0.15, d.anchor_coord)

    # T6b. 대조: 'on' 재투영 없이 지면 평면에 꽂으면 xy 가 크게 밀린다
    naive = locate(cam, dets[1]).base
    check("T6b-naive-biased", abs(naive[0] - 6) > 0.6, naive)

    # T7. beside vs near (기하 확정 — "next to" 는 비확정 힌트)
    parsed7 = parse_command("Find the umbrella next to the trash can.")
    dets = [synth_det(cam, "trash_can", (5, 2), 1.0, 0.3),
            synth_det(cam, "umbrella", (5.6, 2), 0.9, 0.05)]
    d = resolve(parsed7, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T7-beside", d.relation == "beside", d.relation)
    dets = [synth_det(cam, "trash_can", (5, 2), 1.0, 0.3),
            synth_det(cam, "umbrella", (7.5, 2), 0.9, 0.05)]
    d = resolve(parsed7, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T7-near", d.relation == "near", d.relation)

    # T7b. 확정적 텍스트 relation 은 기하보다 우선
    parsed7b = parse_command("Find the umbrella beside the trash can.")
    d = resolve(parsed7b, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T7b-text-wins", d.relation == "beside", d.relation)

    # T8. 방향 relation 은 텍스트 힌트가 있을 때만 기하로 확인
    parsed8 = parse_command("Find the umbrella behind the trash can.")
    check("T8-parse", (parsed8.relation_hint, parsed8.relation_confident) ==
          ("behind", True), parsed8.relation_hint)
    d = resolve(parsed8, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T8-directional-text", d.relation == "behind", d.relation)

    # T9. 사람 문제: pose 후보 필터 + situation 공식 규칙 연동
    parsed9 = parse_command("Find the person sitting on top of the trash can.")
    dets = [synth_det(cam, "trash_can", (5, 1), 1.0, 0.3),
            synth_det(cam, "person_walking", (8, -1), 1.7, 0.25),
            synth_det(cam, "person_sitting", (5, 1), 0.9, 0.25, base_z=1.0)]
    d = resolve(parsed9, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T9-pose-picked", d.pose_type == "person_sitting", d.pose_type)
    check("T9-situation", d.situation == "abnormal", d.situation)     # seated x non_mount
    check("T9-relation-none", d.relation is None, d.relation)
    check("T9-target-type", d.target_type == "person", d.target_type)

    # T9b. prone 은 어디서든 accident (incapacitation)
    parsed9b = parse_command("Find the person lying face-down and motionless.")
    dets = [synth_det(cam, "person_prone", (6, 1), 0.35, 0.6),
            synth_det(cam, "person_walking", (8, -1), 1.7, 0.25),
            synth_det(cam, "bicycle", (6.6, 1), 0.9, 0.4)]
    d = resolve(parsed9b, {"cam_a": dets}, {"cam_a": cam}, cfg)
    check("T9b-prone-picked", d.pose_type == "person_prone", d.pose_type)
    check("T9b-accident", d.situation == "accident", d.situation)

    # T10. 카메라 선택: target 이 보이는 카메라만 선택 가능
    cam_b = make_camera("cam_b", pos=[12, 8, 4.0], look_at=[6, 0, 0])
    dets_a = [synth_det(cam, "bench", (6.5, 1), 0.8, 0.5)]          # target 없음
    dets_b = [synth_det(cam_b, "umbrella", (5, 1), 0.9, 0.05),
              synth_det(cam_b, "bench", (5.8, 1.2), 0.8, 0.5)]
    d = resolve(parsed, {"cam_a": dets_a, "cam_b": dets_b},
                {"cam_a": cam, "cam_b": cam_b}, cfg)
    check("T10-camera", d.camera_id == "cam_b", d.camera_id)

    # T11. payload 형식 (GroundingResult 필드 구조)
    p = d.to_payload()
    check("T11-payload", set(p) == {"camera_id", "interpretation", "grounding"}
          and set(p["interpretation"]) == {"target_type", "landmark", "relation", "situation"}
          and set(p["grounding"]) == {"anchor_coord", "target_coord"}, p.keys())

    # T12. target 미탐지 + 랜드마크는 있음 → 빈 제출 대신 랜드마크만으로 제출.
    #      (camera/landmark/anchor 배점이 살아있고, 오답에 감점이 없으므로
    #       포기하는 것보다 항상 낫다. target 좌표는 anchor 좌표로 대신 낸다.)
    d = resolve(parsed, {"cam_a": [synth_det(cam, "bench", (6, 1), 0.8, 0.5)]},
                {"cam_a": cam}, cfg)
    check("T12-anchor-only-camera", d is not None and d.camera_id == "cam_a",
          d)
    check("T12-anchor-only-landmark", d is not None and d.landmark == "bench",
          d.landmark if d else None)
    check("T12-anchor-only-coord",
          d is not None and d.target_coord == d.anchor_coord
          and d.anchor_coord != [0.0, 0.0, 0.0], d.anchor_coord if d else None)
    check("T12-anchor-only-flag",
          d is not None and d.debug.get("selection") == "anchor_only",
          d.debug.get("selection") if d else None)

    # T13. target 도 랜드마크도 없음 → 그때만 None (상위 폴백 몫)
    d = resolve(parsed, {"cam_a": []}, {"cam_a": cam}, cfg)
    check("T13-nothing-at-all", d is None, d)

    # T14. 같은 랜드마크 클래스가 여러 카메라에 있을 때 (chungmu 의 trash_can 은
    #      3대, picnic_table/bench/bicycle 은 2대에 있다) 탐지 confidence 가
    #      카메라 선택을 뒤집으면 안 된다. CCTV 들은 서로 다른 공간을 보므로
    #      confidence 는 카메라 간 비교 근거가 못 되고, 명령이 말한 배치를
    #      실제로 실현하는 쪽(= 더 가까운 쌍)이 정답이다.
    #      cam_a: 자신감 높은 umbrella 가 trash_can 에서 2.5m
    #      cam_b: 자신감 낮은 umbrella 가 trash_can 에서 0.3m  ← 이쪽이 정답
    parsed14 = parse_command("Find the umbrella beside the trash can.")
    dets_a = [synth_det(cam, "umbrella", (5, 0), 0.9, 0.05, score=0.95),
              synth_det(cam, "trash_can", (5, 2.5), 1.0, 0.3, score=0.9)]
    #      (옛 점수식 conf+1.0+1/(1+dist) 로는 cam_a 2.212 > cam_b 2.163 이라
    #       오답을 골랐다 — 즉 이 판단이 conf 노이즈에 좌우됐다.)
    dets_b = [synth_det(cam_b, "umbrella", (6, 0), 0.9, 0.05, score=0.40),
              synth_det(cam_b, "trash_can", (6.3, 0), 1.0, 0.3, score=0.9)]
    d = resolve(parsed14, {"cam_a": dets_a, "cam_b": dets_b},
                {"cam_a": cam, "cam_b": cam_b}, cfg)
    check("T14-conf-does-not-outrank-relation", d.camera_id == "cam_b",
          f"{d.camera_id} fit={d.debug.get('relation_fit')}")

    # T15. 명령이 랜드마크 없이 구역만 말하는 경우 (chungmu 39문항 중 8개가
    #      이렇다: "Find the umbrella dropped in the empty parking space.").
    #      구역의 지표 클래스(주차장=차량)가 보이는 카메라로 좁혀야 한다.
    #      cam_a: 자신감 높은 umbrella, 차량 없음  ← 예전엔 conf 로 여기가 뽑혔다
    #      cam_b: 자신감 낮은 umbrella + police_car ← 정답
    parsed15 = parse_command("Find the umbrella dropped in the empty parking space.")
    check("T15-parses-area", parsed15.area_hint == "parking"
          and parsed15.landmark_type is None,
          f"area={parsed15.area_hint} lm={parsed15.landmark_type}")
    dets_a = [synth_det(cam, "umbrella", (5, 0), 0.9, 0.05, score=0.95),
              synth_det(cam, "bench", (5, 2.0), 0.5, 0.5, score=0.9)]
    dets_b = [synth_det(cam_b, "umbrella", (6, 0), 0.9, 0.05, score=0.35),
              synth_det(cam_b, "police_car", (6.5, 1.0), 1.5, 1.0, score=0.9)]
    d = resolve(parsed15, {"cam_a": dets_a, "cam_b": dets_b},
                {"cam_a": cam, "cam_b": cam_b}, cfg)
    check("T15-area-gate-camera", d.camera_id == "cam_b",
          f"{d.camera_id} gate={d.debug.get('gate_kind')}")
    check("T15-area-gate-kind", d.debug.get("gate_kind") == "area",
          d.debug.get("gate_kind"))

    # T16. 구역 지표가 어느 카메라에도 없으면 게이트가 풀려야 한다 (순수 폴백:
    #      새 배경에 주차장이 없으면 이 계층은 개입하지 않고 이전 동작 그대로).
    dets_b_nocar = [synth_det(cam_b, "umbrella", (6, 0), 0.9, 0.05, score=0.35),
                    synth_det(cam_b, "bench", (6.5, 1.0), 0.5, 0.5, score=0.9)]
    d = resolve(parsed15, {"cam_a": dets_a, "cam_b": dets_b_nocar},
                {"cam_a": cam, "cam_b": cam_b}, cfg)
    check("T16-no-indicator-releases-gate",
          d is not None and d.debug.get("gate_applied") is False,
          f"gate_applied={d.debug.get('gate_applied') if d else None}")

    # T17. 명시 랜드마크가 area 보다 우선한다 (더 구체적인 단서이므로).
    #      "trash can" 을 말했으므로 차량이 있는 cam_b 가 아니라 cam_a 여야 한다.
    parsed17 = parse_command("Find the umbrella beside the trash can in the parking lot.")
    dets_a17 = [synth_det(cam, "umbrella", (5, 0), 0.9, 0.05, score=0.4),
                synth_det(cam, "trash_can", (5.3, 0), 1.0, 0.3, score=0.9)]
    dets_b17 = [synth_det(cam_b, "umbrella", (6, 0), 0.9, 0.05, score=0.95),
                synth_det(cam_b, "police_car", (6.5, 1.0), 1.5, 1.0, score=0.9)]
    d = resolve(parsed17, {"cam_a": dets_a17, "cam_b": dets_b17},
                {"cam_a": cam, "cam_b": cam_b}, cfg)
    check("T17-landmark-beats-area",
          d.camera_id == "cam_a" and d.debug.get("gate_kind") == "landmark",
          f"{d.camera_id} gate={d.debug.get('gate_kind')}")

    # T18. 두 단서가 모두 있고 **양립하면** 교집합으로 좁힌다. 랜드마크가 여러
    #      카메라에 보이는 흔한 경우("trash can")를 구역이 갈라 준다.
    #      cam_a: trash_can 은 있지만 구역 지표(차량) 없음 + 높은 conf
    #      cam_b: trash_can + 차량 밀집 ← 정답
    dets_a18 = [synth_det(cam, "umbrella", (5, 0), 0.9, 0.05, score=0.95),
                synth_det(cam, "trash_can", (5.3, 0), 1.0, 0.3, score=0.95)]
    dets_b18 = [synth_det(cam_b, "umbrella", (6, 0), 0.9, 0.05, score=0.35),
                synth_det(cam_b, "trash_can", (6.3, 0), 1.0, 0.3, score=0.35),
                synth_det(cam_b, "police_car", (7.5, 1.5), 1.5, 1.0, score=0.9),
                synth_det(cam_b, "suv", (8.5, -1.5), 1.6, 1.0, score=0.9)]
    d = resolve(parsed17, {"cam_a": dets_a18, "cam_b": dets_b18},
                {"cam_a": cam, "cam_b": cam_b}, cfg)
    check("T18-landmark-and-area-intersect",
          d.camera_id == "cam_b" and d.debug.get("gate_kind") == "landmark+area",
          f"{d.camera_id} gate={d.debug.get('gate_kind')}")

    # T19. 구역 지표가 **애매하면**(밀집이라 할 수 없으면) 구역 게이트가 풀려
    #      랜드마크 게이트만 남는다 — 새 배경 안전장치(AREA_DOMINANCE_RATIO).
    #      cam_a 차량 2대 vs cam_b 3대: 어느 쪽도 "그 주차장"이라 단정 못 한다.
    dets_a19 = dets_a18 + [
        synth_det(cam, "taxi", (7.5, 1.5), 1.5, 1.0, score=0.9),
        synth_det(cam, "suv", (8.5, -1.5), 1.6, 1.0, score=0.9)]
    dets_b19 = dets_b18 + [synth_det(cam_b, "taxi", (9.5, 0.5), 1.5, 1.0, score=0.9)]
    d = resolve(parsed17, {"cam_a": dets_a19, "cam_b": dets_b19},
                {"cam_a": cam, "cam_b": cam_b}, cfg)
    check("T19-ambiguous-area-releases-gate",
          d.debug.get("gate_kind") == "landmark",
          f"gate={d.debug.get('gate_kind')} cams={d.debug.get('gate_cameras')}")

    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
