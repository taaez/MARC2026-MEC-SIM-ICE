"""Measure the live lidar's geometry: rings, mount height above ground, ground blind radius,
and the lowest obstacle height the obstacle layer can see at each distance.

Runs against a live platform with the robot deployed (no registration needed -- it only
subscribes). Saves raw frames to LIVE_OUT for offline inspection.
"""

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage

sys.path.insert(0, "/agent/mec_sim_ice/demo_agents")
from lidar_obstacles import lidar_xyz, GROUND_BAND_MIN_M, GROUND_BAND_MAX_M, MAX_RANGE_M  # noqa

TEAM = os.environ.get("MARC_TEAM_ID", "mec_sim_ice")
OUT = os.environ.get("LIVE_OUT", "/agent/live_out")
N_FRAMES = 10


def main():
    rclpy.init()
    node = rclpy.create_node("lidar_probe")
    frames, statics = [], {}
    qos_be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)
    qos_static = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(PointCloud2, f"/marc/{TEAM}/robot/lidar/points",
                             lambda m: frames.append(m), qos_be)

    def on_static(m):
        for t in m.transforms:
            statics[(t.header.frame_id, t.child_frame_id)] = t
    node.create_subscription(TFMessage, "/tf_static", on_static, qos_static)
    t0 = time.time()
    while len(frames) < N_FRAMES and time.time() - t0 < 60:
        rclpy.spin_once(node, timeout_sec=0.2)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    if not frames:
        print("no lidar frames"); return

    pts = np.concatenate([lidar_xyz(f) for f in frames])
    np.save(os.path.join(OUT, "lidar_frames.npy"), pts)
    r_xy = np.hypot(pts[:, 0], pts[:, 1])
    elev = np.degrees(np.arctan2(pts[:, 2], r_xy))
    rings = np.unique(np.round(elev[r_xy > 1.0]))
    # ground: lowest ring hits close to the robot; mount height from near-field ground z
    near = (r_xy > 0.5) & (r_xy < 6.0)
    ground_z = float(np.percentile(pts[near, 2], 5))
    h = -ground_z
    lowest = float(elev[r_xy > 0.3].min())
    blind_r = h / math.tan(math.radians(-lowest)) if lowest < 0 else float("inf")
    min_r = float(r_xy.min())
    lidar_tf = {f"{k[0]}->{k[1]}": [t.transform.translation.x, t.transform.translation.y,
                                    t.transform.translation.z]
                for k, t in statics.items() if "lidar" in k[1].lower() or "lidar" in k[0].lower()}
    # Lowest obstacle top the layer can see at distance d: the lowest beam passes at
    # height h + d*tan(lowest) above ground, and the layer needs points >= GROUND_BAND_MIN_M.
    table = []
    for d in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0):
        beam = h + d * math.tan(math.radians(lowest))
        table.append({"d_m": d, "lowest_beam_height_m": round(beam, 3),
                      "min_visible_top_m": round(max(beam, GROUND_BAND_MIN_M), 3)})
    res = {"frames": len(frames), "points_per_frame": int(len(pts) / len(frames)),
           "frame_id": frames[0].header.frame_id,
           "ring_elevations_deg": rings.tolist(), "lowest_ring_deg": round(lowest, 2),
           "highest_ring_deg": round(float(elev[r_xy > 0.3].max()), 2),
           "mount_height_above_ground_m": round(h, 3), "ground_blind_radius_m": round(blind_r, 2),
           "min_return_range_m": round(min_r, 3), "layer_band_m": [GROUND_BAND_MIN_M, GROUND_BAND_MAX_M],
           "layer_max_range_m": MAX_RANGE_M, "static_tf_lidar": lidar_tf,
           "visibility_by_distance": table}
    with open(os.path.join(OUT, "lidar_probe.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
