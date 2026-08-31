"""Depth-only grasp localization test (v1) -- manip-trainer only, no YOLO/CV model needed.

Back-projects the WHOLE depth image to 3D points in the arm_base_link frame (using the
camera's own TF, which is fine -- this is camera->arm_base_link, not the broken
world->arm_base_link link), estimates the floor height as the median Z over the image
(robust: most pixels are floor, the object only covers a small fraction), then treats
points clearly above that as the object and grasps at their centroid.

No learned model at all -> immune to the CCTV-vs-robot-camera domain mismatch that killed
the earlier YOLO-based attempt (camera_pick_test.py, kept only on the GPU machine's
~/jw_test, never committed). Limitation: this finds "something sticking up off the floor",
not "the correct object" -- fine when only one object is in view (which is the case in the
trainer), not a distractor-rejection solution by itself.

Usage (from a checkout of this repo, e.g. ~/jw_test on the GPU machine):
    docker run --rm -it --network host \
      -e MARC_TEAM_ID=jw_test -e MARC_TOKEN=jw_test_token \
      -e TARGET_LABEL=sunblock \
      -v ~/jw_test/mec_sim_ice/manip_tools/floor_grasp_test.py:/agent/demo/floor_grasp_test.py \
      -v ~/jw_test/demo/pnp_params.json:/agent/demo/pnp_params.json \
      -v ~/jw_test/demo/arm_pick.py:/agent/demo/arm_pick.py \
      marc-agent:local python3 /agent/demo/floor_grasp_test.py

TARGET_LABEL only selects the pnp_params.json per-object grip threshold (see
arm_pick.load_params) -- it does not affect how the object is found here.
"""

import os
import time
import logging

os.environ.setdefault("MARC_TEAM_ID", "jw_test")
os.environ.setdefault("MARC_TOKEN", "jw_test_token")
TARGET_LABEL = os.environ.get("TARGET_LABEL") or None
CAMERA_FRAME = os.environ.get("CAMERA_FRAME", "Base_Camera_Center")
# How far above the estimated floor (m) counts as "object, not floor noise".
HEIGHT_MARGIN_M = float(os.environ.get("HEIGHT_MARGIN_M", "0.03"))
# Ignore points beyond this forward reach -- background clutter, not the pick target.
MAX_FWD_M = float(os.environ.get("MAX_FWD_M", "1.2"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("floor_grasp_test")

import numpy as np
from sensor_msgs.msg import CameraInfo

from marc_sdk import MARCClient
from arm_pick import run_pick_sequence


def _depth_array(depth_msg):
    """sensor_msgs/Image (32FC1) -> HxW float32 ndarray (meters), NaN/0 where invalid."""
    h, w = depth_msg.height, depth_msg.width
    return np.frombuffer(depth_msg.data, dtype=np.float32).reshape(h, w).copy()


def quat_to_rotmat(x, y, z, w):
    """Unit quaternion -> 3x3 rotation matrix (same convention as taejeong/grounding.py)."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def locate_object_by_depth(client, tf_buffer, log):
    """Find the floor-outlier centroid in arm_base_link frame. Returns (fwd, lat) or None."""
    import rclpy

    t_end = time.time() + 5.0
    img_msg = client.get_robot_image("base_left")
    depth_msg = client.get_robot_depth("base")
    while (img_msg is None or depth_msg is None) and time.time() < t_end:
        time.sleep(0.1)
        img_msg = client.get_robot_image("base_left")
        depth_msg = client.get_robot_depth("base")
    if depth_msg is None:
        log.warning("[floor] no depth image after 5s wait")
        return None

    info_holder = {}
    def _on_info(msg):
        info_holder["msg"] = msg
    client.subscribe(client.topics.robot("base_camera/depth/info"), CameraInfo, _on_info)
    t_end = time.time() + 2.0
    while "msg" not in info_holder and time.time() < t_end:
        time.sleep(0.05)
    if "msg" not in info_holder:
        log.warning("[floor] no depth camera_info within 2s")
        return None
    K = np.array(info_holder["msg"].k, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    info_w, info_h = info_holder["msg"].width, info_holder["msg"].height
    if (info_w, info_h) != (depth_msg.width, depth_msg.height):
        # Wrong intrinsics silently produce wrong-but-plausible-looking 3D points (no crash,
        # no error) -- this is the one failure mode here that would not announce itself.
        log.warning("[floor] camera_info size (%dx%d) != depth image size (%dx%d) -- "
                    "K matrix likely does not match this depth image, aborting",
                    info_w, info_h, depth_msg.width, depth_msg.height)
        return None

    try:
        tf = tf_buffer.lookup_transform("arm_base_link", CAMERA_FRAME, rclpy.time.Time(),
                                         timeout=rclpy.duration.Duration(seconds=1.0))
    except Exception as e:
        log.warning("[floor] tf lookup %s -> arm_base_link failed: %s", CAMERA_FRAME, e)
        return None
    tr, rot = tf.transform.translation, tf.transform.rotation
    R = quat_to_rotmat(rot.x, rot.y, rot.z, rot.w)
    C = np.array([tr.x, tr.y, tr.z])

    depth = _depth_array(depth_msg)
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    if not np.any(valid):
        log.warning("[floor] no valid depth pixels")
        return None

    # Kept as full (h, w) grids (not flattened) so candidate pixels can be clustered by
    # image-space connectivity below -- a flat list of points has no notion of "which
    # pixels are touching", which is what separates one blob from unrelated noise elsewhere
    # in frame.
    us_grid, vs_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_cam = (us_grid - cx) / fx * depth
    y_cam = (vs_grid - cy) / fy * depth
    z_cam = depth
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)          # (h, w, 3), camera frame
    pts_arm = pts_cam @ R.T + C                                 # (h, w, 3), arm_base_link frame

    fwd, lat, up = pts_arm[..., 0], pts_arm[..., 1], pts_arm[..., 2]
    floor_z = float(np.median(up[valid]))
    log.info("[floor] estimated floor height (arm_base z) = %.3fm from %d points",
             floor_z, int(np.sum(valid)))

    candidate = valid & (up > floor_z + HEIGHT_MARGIN_M) & (fwd > 0.0) & (fwd < MAX_FWD_M)
    n_cand = int(np.sum(candidate))
    if n_cand < 20:
        log.warning("[floor] only %d candidate pixels above floor+%.2fm -- not enough, giving up",
                    n_cand, HEIGHT_MARGIN_M)
        return None

    # Cluster into connected blobs and keep only the largest -- averaging every candidate
    # pixel unconditionally would blend a second object/decoy or scattered noise elsewhere
    # in frame into one meaningless point between them.
    try:
        from scipy import ndimage
        labeled, n_blobs = ndimage.label(candidate)
        sizes = ndimage.sum(candidate, labeled, index=range(1, n_blobs + 1))
        best_label = int(np.argmax(sizes)) + 1
        blob_mask = labeled == best_label
        log.info("[floor] %d blob(s) found -- using the largest (%d of %d candidate px)",
                 n_blobs, int(sizes[best_label - 1]), n_cand)
    except ImportError:
        log.warning("[floor] scipy not available -- treating all candidate pixels as one blob "
                    "(no separation from unrelated noise elsewhere in frame)")
        blob_mask = candidate

    obj_fwd = float(np.mean(fwd[blob_mask]))
    obj_lat = float(np.mean(lat[blob_mask]))
    obj_up = float(np.mean(up[blob_mask]))
    log.info("[floor] object blob: %d px, centroid fwd=%.3f lat=%.3f height_above_floor=%.3fm",
             int(np.sum(blob_mask)), obj_fwd, obj_lat, obj_up - floor_z)
    return (obj_fwd, obj_lat)


def main():
    from tf2_ros import Buffer, TransformListener

    client = MARCClient.from_env()
    ok = client.connect(timeout=3.0)
    log.info("connect() -> %s (normal to be False in the trainer)", ok)

    tf_buffer = Buffer()
    TransformListener(tf_buffer, client.node)
    time.sleep(1.0)

    log.info("=== locating object via depth floor-segmentation (no model) ===")
    grasp_xy = locate_object_by_depth(client, tf_buffer, log)

    if grasp_xy is None:
        log.warning("depth localization failed -- falling back to fixed pnp_params.json offset (grasp_xy=None)")
    else:
        log.info("using depth-refined grasp_xy=%s (target_label=%s for grip threshold only)",
                 grasp_xy, TARGET_LABEL)

    held = run_pick_sequence(client, log, grasp_xy, object_type=TARGET_LABEL)
    basket = client.is_basket_occupied()
    log.info("result: held=%s, basket_occupied=%s", held, basket)


if __name__ == "__main__":
    main()
