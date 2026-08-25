"""Real VLA grounding: YOLO11 detection over the refined DET_CLASSES taxonomy.

The detector is the stock YOLO11 architecture fine-tuned on the class list in
test/det_classes.py (10 targets + 15 landmarks + 9 person_* pose subclasses --
the file moves to demo/ once verified, see test/README.md; both locations are
searched at import time). Person pose is classified DIRECTLY by the detector,
so the earlier YOLO11-pose keypoint pass is gone; ``situation`` now comes from
the organizer's official pose x support rules (taejeong/situation_resolver.py),
which is what the stage1 scorer itself uses.

Command parsing reuses taejeong/command_parser.py: real voice_command values
are English sentences (see demo/mock_demo_data.yaml keys, e.g. "Find the
person sitting on top of the trash can in the parking lot.") -- the Korean
keyword tables this file used to carry never matched them and are removed.
Note person problems DO carry a landmark in the answer key (mock data), so the
old "SAR problems have no landmark" assumption is also dropped.

Camera extrinsics come from tf2 (world <- camera_id, static). Frame names
("world", camera_id as-is) are assumed from the summary doc's wording --
confirm with ``ros2 run tf2_tools view_frames`` against a running platform and
adjust ``_pixel_to_world`` if the actual frame ids differ.
"""

import logging
import os
import sys

import numpy as np
import rclpy
from ultralytics import YOLO

from marc_sdk import GroundingResult

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # taejeong/ package lives next to this file

# det_classes.py stages in test/ until the training pipeline is verified, then
# moves to demo/ (test/README.md) -- look in both so this file survives the move.
for _sub in ("test", "demo"):
    if os.path.exists(os.path.join(_HERE, _sub, "det_classes.py")):
        sys.path.insert(0, os.path.join(_HERE, _sub))
        break
from det_classes import LANDMARK_CLASSES, POSE_CLASSES  # noqa: E402

from taejeong.command_parser import parse_command  # noqa: E402
from taejeong.situation_resolver import (  # noqa: E402
    normalize_support_token,
    resolve_situation,
)

log = logging.getLogger("participant_app")

# The parser may return the generic landmark hint "car" -- accept any of the
# car-variant detector classes for it. Derived via the official support-token
# normalization so the set stays in lock-step with situation_resolver.
CAR_CLASSES = frozenset(
    c for c in LANDMARK_CLASSES if normalize_support_token(c) == "car")

# parse_command pose_hint bucket -> candidate detector pose classes, most
# likely first. Buckets are deliberately loose: surface wording and labeled
# pose class are not 1:1 in real data (command_parser.py POSE_BUCKETS note),
# so these only PRIORITIZE person boxes -- any pose class is still accepted
# when no preferred one is detected.
POSE_BUCKET_CLASSES = {
    "facedown": ("person_prone", "person_lying_down", "person_bending"),
    "bent":     ("person_bending", "person_crouching"),
    "down":     ("person_lying_down", "person_prone", "person_reclining"),
    "lying":    ("person_lying_down", "person_reclining", "person_prone"),
    "sitting":  ("person_sitting", "person_reclining", "person_crouching"),
    "standing": ("person_walking", "person_walking_away", "person_reaching"),
}

# Team convention (taejeong/RUNBOOK.md): the fine-tuned detector lives at
# demo/weights/best.pt (received from the training teammate, not in git).
DET_WEIGHTS = os.environ.get("VLA_DET_WEIGHTS",
                             os.path.join(_HERE, "demo", "weights", "best.pt"))
DET_IMGSZ = 1280   # CCTV native resolution (1280x720)
DET_CONF = 0.35


class RealVLAGrounding:
    """Stage 1 / Stage 2 grounding via YOLO11 detection on DET_CLASSES.

    Needs the live ``MARCClient`` (not just its sensor getters) because camera
    extrinsics are read from tf2, which requires the client's spinning rclpy node.
    """

    def __init__(self, client, det_weights: str = DET_WEIGHTS):
        self._client = client
        self._det = YOLO(det_weights)
        # tf2 needs client.node, which only exists after client.connect() -- init lazily
        # on first use rather than here (RealVLAGrounding is constructed before connect()).
        self._tf_buffer = None
        self._tf_listener = None

    def _tf(self):
        if self._tf_buffer is None:
            import tf2_ros
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._client.node)
        return self._tf_buffer

    # -- Stage 1 / Stage 2 entry points (signature must match MockVLAGrounding) --

    def process(self, voice_command: str, camera_ids: list = None) -> GroundingResult:
        return self._ground(voice_command, camera_ids or self._client.list_cctv())

    def process_stage2(self, task_description: str) -> GroundingResult:
        return self._ground(task_description, self._client.list_cctv())

    def _ground(self, command: str, camera_ids: list) -> GroundingResult:
        parsed = parse_command(command)
        target_type = parsed.target_type or ("person" if parsed.is_person else "")
        landmark = parsed.landmark_type or ""

        for cam_id in camera_ids:
            img = self._to_bgr(self._client.get_cctv_image(cam_id))
            if img is None:
                continue
            det = self._det.predict(img, imgsz=DET_IMGSZ, conf=DET_CONF, verbose=False)[0]

            if parsed.is_person:
                target_box, pose_class = self._best_person_box(det, parsed.pose_hint)
            else:
                target_box, pose_class = self._best_box(det, target_type), None
            if target_box is None:
                continue

            anchor_box = None
            if landmark:
                anchor_box = self._best_box(
                    det, CAR_CLASSES if landmark == "car" else landmark)
                if anchor_box is None:
                    continue  # landmark not visible here -- wrong camera

            relation, situation = parsed.relation_hint, None
            if parsed.is_person:
                # situation = f(detected pose x support surface), the official rule.
                # Best support guess: the named landmark, else the area wording
                # ("parking lot" -> parking); resolve_situation normalizes both.
                situation = resolve_situation(
                    pose_class, landmark or parsed.area_hint).situation
                relation = None  # person answer keys carry no relation (mock data)

            target_coord = self._pixel_to_world(cam_id, self._box_center(target_box))
            anchor_coord = (self._pixel_to_world(cam_id, self._box_center(anchor_box))
                             if anchor_box is not None else target_coord)
            return GroundingResult(
                camera_id=cam_id, target_type=target_type,
                anchor_coord=anchor_coord, target_coord=target_coord,
                landmark=landmark, relation=relation, situation=situation,
            )

        log.warning("[vla] no camera matched target=%r landmark=%r pose_hint=%r",
                    target_type, landmark, parsed.pose_hint)
        situation = None
        if parsed.is_person:
            # Last-ditch guess so person submissions still carry a situation:
            # assume the most likely pose for the hinted bucket (walking if none).
            guess_pose = POSE_BUCKET_CLASSES.get(parsed.pose_hint, ("person_walking",))[0]
            situation = resolve_situation(guess_pose, landmark or parsed.area_hint).situation
        return GroundingResult(
            camera_id="", target_type=target_type, anchor_coord=[0.0, 0.0, 0.0],
            target_coord=[0.0, 0.0, 0.0], landmark=landmark,
            relation=None if parsed.is_person else parsed.relation_hint,
            situation=situation,
        )

    # -- detection helpers --

    @staticmethod
    def _best_box(det, class_names):
        """Highest-confidence box whose class is class_names (a name or a set of names)."""
        if not class_names:
            return None
        if isinstance(class_names, str):
            class_names = {class_names}
        best, best_conf = None, -1.0
        for box in det.boxes:
            if det.names.get(int(box.cls[0])) not in class_names:
                continue
            conf = float(box.conf[0])
            if conf > best_conf:
                best, best_conf = box, conf
        return best

    @staticmethod
    def _best_person_box(det, pose_hint):
        """(box, pose_class) for the best person_* detection.

        A box whose pose class matches the command's pose bucket wins over a
        higher-confidence box in a different pose (several people may be in
        frame; the wording tells us which one the problem means). Falls back
        to the best person of ANY pose when nothing matches the bucket.
        """
        preferred = POSE_BUCKET_CLASSES.get(pose_hint, ())
        best = pref = None
        best_conf = pref_conf = -1.0
        best_cls = pref_cls = None
        for box in det.boxes:
            cls = det.names.get(int(box.cls[0]))
            if cls not in POSE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf > best_conf:
                best, best_conf, best_cls = box, conf, cls
            if cls in preferred and conf > pref_conf:
                pref, pref_conf, pref_cls = box, conf, cls
        if pref is not None:
            return pref, pref_cls
        return best, best_cls

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    # -- pixel -> world (ground-plane back-projection) --

    def _pixel_to_world(self, cam_id: str, pixel):
        info = self._client.get_cctv_info(cam_id)
        ground_z = self._client.get_cctv_ground_height(cam_id)
        if info is None or ground_z is None:
            return [0.0, 0.0, 0.0]

        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        u, v = pixel
        # optical convention: +Z forward, +Y down (see summary doc #4 coord section)
        ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

        try:
            tf = self._tf().lookup_transform(
                "world", cam_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:  # noqa: BLE001
            log.warning("[vla] tf lookup world<-%s failed: %s", cam_id, e)
            return [0.0, 0.0, float(ground_z)]

        t = tf.transform.translation
        q = tf.transform.rotation
        R = self._quat_to_R(q.w, q.x, q.y, q.z)
        cam_origin = np.array([t.x, t.y, t.z])
        ray_world = R @ ray_cam

        if abs(ray_world[2]) < 1e-6:
            return [float(cam_origin[0]), float(cam_origin[1]), float(ground_z)]
        s = (ground_z - cam_origin[2]) / ray_world[2]
        p = cam_origin + s * ray_world
        return [float(p[0]), float(p[1]), float(ground_z)]

    @staticmethod
    def _quat_to_R(w, x, y, z):
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def _to_bgr(img_msg):
        """sensor_msgs/Image -> HxWx3 uint8 BGR ndarray (what ultralytics/opencv expect)."""
        if img_msg is None:
            return None
        h, w, enc = img_msg.height, img_msg.width, img_msg.encoding
        arr = np.frombuffer(img_msg.data, dtype=np.uint8)
        if enc in ("bgr8", "rgb8"):
            arr = arr.reshape(h, w, 3)
            return arr[:, :, ::-1] if enc == "rgb8" else arr
        if enc in ("bgra8", "rgba8"):
            arr = arr.reshape(h, w, 4)
            arr = arr[:, :, :3]
            return arr[:, :, ::-1] if enc == "rgba8" else arr
        log.warning("[vla] unexpected CCTV encoding %r -- add a case in _to_bgr", enc)
        return None
