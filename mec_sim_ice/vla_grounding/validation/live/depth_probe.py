"""Grab live base-camera depth frames + camera_info + the arm_base_link -> Base_Camera_Center
transform, for offline tuning of depth_ground_veto. Subscribe-only (no registration).

Saves LIVE_OUT/depth_probe.npz: depth (N, H, W), K (3, 3), mount_R, mount_t (as
participant_app._camera_mount_transform builds them), pose (N, 3) world pose per frame.
"""

import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

sys.path.insert(0, "/agent/mec_sim_ice/demo_agents")
import depth_ground_veto  # noqa: E402

TEAM = os.environ.get("MARC_TEAM_ID", "mec_sim_ice")
OUT = os.environ.get("LIVE_OUT", "/agent/live_out")
N_FRAMES = int(os.environ.get("N_FRAMES", "5"))
ARM_BASE_FWD = 0.18          # participant_app.ARM_BASE_FWD


def main():
    rclpy.init()
    node = rclpy.create_node("depth_probe")
    be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
    static = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL)
    frames, info, tfs = [], {}, {}
    base = f"/marc/{TEAM}/robot/base_camera/depth"
    node.create_subscription(Image, base + "/image", lambda m: frames.append(m), be)
    node.create_subscription(CameraInfo, base + "/info", lambda m: info.setdefault("k", m), be)

    def on_tf(m):
        for t in m.transforms:
            tfs[(t.header.frame_id, t.child_frame_id)] = t
    node.create_subscription(TFMessage, "/tf_static", on_tf, static)
    node.create_subscription(TFMessage, "/tf", on_tf, QoSProfile(depth=100))
    t0 = time.time()
    while (len(frames) < N_FRAMES or "k" not in info) and time.time() - t0 < 90:
        rclpy.spin_once(node, timeout_sec=0.2)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    print("frames", len(frames), "info", "k" in info)
    print("tf pairs with camera:", [k for k in tfs if "Camera" in k[1] or "Camera" in k[0]])
    tr = tfs.get(("arm_base_link", "Base_Camera_Center"))
    if tr is None:
        print("no arm_base_link->Base_Camera_Center"); return
    q = tr.transform.rotation
    R = depth_ground_veto.quat_to_rotmat(q.x, q.y, q.z, q.w)
    t = tr.transform.translation
    mt = np.array([t.x + ARM_BASE_FWD, t.y, t.z])
    m0 = frames[0]
    depth = np.stack([np.frombuffer(f.data, np.float32).reshape(f.height, f.width) for f in frames])
    K = np.array(info["k"].k, float).reshape(3, 3)
    np.savez(os.path.join(OUT, "depth_probe.npz"), depth=depth, K=K, mount_R=R, mount_t=mt,
             encoding=m0.encoding)
    d = depth[0]
    print("size", d.shape, "enc", m0.encoding, "K", K.round(1).tolist())
    print("mount_R", R.round(3).tolist(), "mount_t", mt.round(3).tolist())
    fin = np.isfinite(d) & (d > 0)
    print("finite>0 frac %.2f  min %.3f  p5 %.3f  p50 %.3f  max %.3f" % (
        fin.mean(), d[fin].min(), np.percentile(d[fin], 5), np.median(d[fin]), d[fin].max()))
    for r0 in range(0, d.shape[0], d.shape[0] // 8):
        row = d[r0]
        print(f"row {r0:4d}: finite {np.isfinite(row).mean():.2f}  median {np.nanmedian(np.where(row > 0, row, np.nan)):.2f}")


if __name__ == "__main__":
    main()
