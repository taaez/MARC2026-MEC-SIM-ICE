"""
python -m pytest tests/  또는  python tests/test_geometry.py
ROS 없이 동작. 합성 CCTV 를 만들어 world→pixel→world 왕복 오차와 relation 판정을 검사한다.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stage1.geometry import (
    CameraModel, quat_to_rotmat, pixel_to_world, world_to_pixel,
    detection_to_world, best_relation,
)


def make_cam(yaw_deg=30.0, pitch_down_deg=35.0, height=6.0, ground_z=16.4):
    """
    optical 프레임 카메라: 먼저 광축(+Z_cam) 을 world +X 로 놓는 기본 정렬
    (X_cam→-Y_w, Y_cam→-Z_w, Z_cam→+X_w), 그 다음 pitch(내려다보기) 와 yaw 를 적용.
    """
    R_base = np.array([[0, 0, 1],
                       [-1, 0, 0],
                       [0, -1, 0]], dtype=float)          # cam→world, 수평 전방 주시
    p = np.deg2rad(pitch_down_deg)
    R_pitch = np.array([[np.cos(p), 0, np.sin(p)],
                        [0, 1, 0],
                        [-np.sin(p), 0, np.cos(p)]])      # world Y 축 기준 회전(내려다보기)
    y = np.deg2rad(yaw_deg)
    R_yaw = np.array([[np.cos(y), -np.sin(y), 0],
                      [np.sin(y), np.cos(y), 0],
                      [0, 0, 1]])
    R_cw = R_yaw @ R_pitch @ R_base
    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1]])
    return CameraModel("rig_test", K, R_cw, np.array([-60.0, 150.0, ground_z + height]), ground_z)


def test_roundtrip():
    cam = make_cam()
    rng = np.random.default_rng(0)
    worst = 0.0
    n = 0
    for _ in range(500):
        # 카메라 앞쪽 지면 위 임의 점
        fwd = cam.forward_ground
        left = np.cross([0, 0, 1], fwd)
        P = cam.C_w.copy(); P[2] = cam.ground_z
        P += fwd * rng.uniform(3, 25) + left * rng.uniform(-8, 8)
        uv = world_to_pixel(cam, P)
        if uv is None or not (0 <= uv[0] < 1280 and 0 <= uv[1] < 720):
            continue
        Q = pixel_to_world(cam, *uv)
        assert Q is not None
        worst = max(worst, np.linalg.norm(Q - P)); n += 1
    assert n > 100, "시야 안 샘플이 너무 적음"
    assert worst < 1e-6, f"왕복 오차 {worst}"
    print(f"[roundtrip] {n} samples, worst err {worst:.2e} m")


def test_quat_identity():
    R = quat_to_rotmat(0, 0, 0, 1)
    assert np.allclose(R, np.eye(3))


def test_bbox_bottom_vs_center():
    """높이 1.7m 사람: bbox 중심으로 역투영하면 카메라 쪽으로 당겨진다 → 바닥점 사용 근거."""
    cam = make_cam()
    P_feet = cam.C_w.copy(); P_feet[2] = cam.ground_z
    P_feet += cam.forward_ground * 10.0
    P_head = P_feet + np.array([0, 0, 1.7])
    u1, v1 = world_to_pixel(cam, P_head); u2, v2 = world_to_pixel(cam, P_feet)
    bbox = [min(u1, u2) - 20, v1, max(u1, u2) + 20, v2]
    Q_bottom = detection_to_world(cam, bbox)
    Q_center = pixel_to_world(cam, (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    e_bottom = np.linalg.norm(Q_bottom - P_feet)
    e_center = np.linalg.norm(Q_center - P_feet)
    print(f"[bbox] bottom err {e_bottom:.3f} m, center err {e_center:.3f} m")
    assert e_bottom < 0.01 and e_center > 0.5


def test_relations():
    cam = make_cam()
    A = cam.C_w.copy(); A[2] = cam.ground_z; A += cam.forward_ground * 10
    fwd, left = cam.forward_ground, np.cross([0, 0, 1], cam.forward_ground)
    assert best_relation(cam, A, A + fwd * 3) == "behind"
    assert best_relation(cam, A, A - fwd * 3) == "front"
    assert best_relation(cam, A, A + left * 3) == "left"
    assert best_relation(cam, A, A - left * 3) == "right"
    assert best_relation(cam, A, A + left * 0.3) == "beside"
    assert best_relation(cam, A, A + np.array([0, 0, 0.5])) == "on"
    print("[relations] ok")


if __name__ == "__main__":
    test_quat_identity(); test_roundtrip(); test_bbox_bottom_vs_center(); test_relations()
    print("ALL OK")
