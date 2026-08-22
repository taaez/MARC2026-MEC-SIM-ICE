"""Real VLA grounding: YOLO11 detection (target/landmark/person) + YOLO11-pose (situation).

Both models use the stock YOLO11 architecture at its default kernel/channel widths (the
"n" scale in ultralytics/cfg/models/11/yolo11.yaml) -- no custom backbone. Only the
detection head is fine-tuned on our classes (``DET_WEIGHTS``); the pose model keeps its
COCO-pretrained weights unchanged and is only loaded lazily (person/situation rounds).

Class vocab (TARGET_CLASSES / LANDMARK_CLASSES) is inferred from the problem ids in
problems.yaml (e.g. lf01_mug_table -> target "mug", landmark "table"). Verify against the
actual ``detections[].class`` field in trainer_output/*.json once dataset-gen has run --
if the sim uses different internal names, update these lists (they double as the
data.yaml class order for training, see train_yolo11.py).

Korean keyword lists for command parsing (RELATION_WORDS / SITUATION_WORDS) are guesses --
the developer guide does not publish example voice_command sentences (see
MARC2026_요약_v2.md #12). Run the platform once, log real Mission.voice_command values,
and correct these tables before relying on them for scoring.

Camera extrinsics come from tf2 (world <- camera_id, static). Frame names ("world",
camera_id as-is) are assumed from the summary doc's wording -- confirm with
``ros2 run tf2_tools view_frames`` against a running platform and adjust
``_pixel_to_world`` if the actual frame ids differ.
"""

import logging
import os

import numpy as np
import rclpy
from ultralytics import YOLO

from marc_sdk import GroundingResult

log = logging.getLogger("participant_app")

TARGET_CLASSES = ["mug", "tumbler", "sunblock", "umbrella", "pencilcase"]
LANDMARK_CLASSES = ["table", "bench", "hydrant", "vending", "fallenbike",
                     "parking", "taxi", "bikerack", "park", "trash"]
# Training class order (data.yaml must list them in exactly this order).
DET_CLASSES = TARGET_CLASSES + LANDMARK_CLASSES + ["person"]

# Korean keyword -> canonical value. First match wins; longer/more specific keys should
# come first when they overlap (e.g. "위에" before "위").
RELATION_WORDS = [
    ("근처", "near"), ("주변", "near"),
    ("옆", "beside"),
    ("뒤", "behind"),
    ("앞", "front"),
    ("왼쪽", "left"),
    ("오른쪽", "right"),
    ("위에 붙", "on"), ("위에 놓", "on"),
    ("공중", "above"), ("위", "above"),
]

SITUATION_WORDS = [
    ("사고", "accident"),
    ("위급", "emergency"), ("긴급", "emergency"),
    ("쓰러", "abnormal"), ("이상", "abnormal"), ("누워", "abnormal"),
    ("정상", "normal"),
]

PERSON_WORDS = ("사람", "사람이", "쓰러", "앉아", "누워", "구조")

_HERE = os.path.dirname(os.path.abspath(__file__))
DET_WEIGHTS = os.environ.get("VLA_DET_WEIGHTS", os.path.join(_HERE, "weights", "best.pt"))
# Left at the stock filename -- ultralytics downloads/caches it at build time (baked into
# the image); it is COCO-pretrained and not fine-tuned on our scenes.
POSE_WEIGHTS = os.environ.get("VLA_POSE_WEIGHTS", "yolo11n-pose.pt")
DET_IMGSZ = 1280   # CCTV native resolution (1280x720)
DET_CONF = 0.35

# COCO keypoint indices used for the sit/lie/stand heuristic.
KP_SHOULDER = (5, 6)
KP_HIP = (11, 12)
KP_ANKLE = (15, 16)


class RealVLAGrounding:
    """Stage 1 / Stage 2 grounding via YOLO11 detection + (situation rounds only) YOLO11-pose.

    Needs the live ``MARCClient`` (not just its sensor getters) because camera
    extrinsics are read from tf2, which requires the client's spinning rclpy node.
    """

    def __init__(self, client, det_weights: str = DET_WEIGHTS, pose_weights: str = POSE_WEIGHTS):
        self._client = client
        self._det = YOLO(det_weights)
        self._pose_weights = pose_weights
        self._pose = None  # lazy -- only person/situation rounds pay for loading it
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

    def _pose_model(self):
        if self._pose is None:
            self._pose = YOLO(self._pose_weights)
        return self._pose

    # -- Stage 1 / Stage 2 entry points (signature must match MockVLAGrounding) --

    def process(self, voice_command: str, camera_ids: list = None) -> GroundingResult:
        return self._ground(voice_command, camera_ids or self._client.list_cctv())

    def process_stage2(self, task_description: str) -> GroundingResult:
        return self._ground(task_description, self._client.list_cctv())

    def _ground(self, command: str, camera_ids: list) -> GroundingResult:
        target_type, landmark, relation, situation = self._parse_command(command)
        for cam_id in camera_ids:
            img = self._to_bgr(self._client.get_cctv_image(cam_id))
            if img is None:
                continue
            det = self._det.predict(img, imgsz=DET_IMGSZ, conf=DET_CONF, verbose=False)[0]

            target_box = self._best_box(det, target_type)
            if target_box is None:
                continue
            if target_type == "person" and situation:
                if self._classify_situation(img, target_box) != situation:
                    continue  # right camera may show a *different* person -- keep looking

            anchor_box = self._best_box(det, landmark) if landmark else None
            if landmark and anchor_box is None:
                continue  # landmark not visible here -- wrong camera

            target_coord = self._pixel_to_world(cam_id, self._box_center(target_box))
            anchor_coord = (self._pixel_to_world(cam_id, self._box_center(anchor_box))
                             if anchor_box is not None else target_coord)
            return GroundingResult(
                camera_id=cam_id, target_type=target_type,
                anchor_coord=anchor_coord, target_coord=target_coord,
                landmark=landmark or "", relation=relation, situation=situation,
            )

        log.warning("[vla] no camera matched target=%r landmark=%r situation=%r",
                    target_type, landmark, situation)
        return GroundingResult(
            camera_id="", target_type=target_type, anchor_coord=[0.0, 0.0, 0.0],
            target_coord=[0.0, 0.0, 0.0], landmark=landmark or "",
            relation=relation, situation=situation,
        )

    # -- command parsing (rule-based; replace with real voice_command examples, see module docstring) --

    def _parse_command(self, text: str):
        text = text or ""
        target_type = next((c for c in TARGET_CLASSES if c in text), None)
        if target_type is None and any(w in text for w in PERSON_WORDS):
            target_type = "person"
        landmark = next((c for c in LANDMARK_CLASSES if c in text), None)
        relation = next((v for k, v in RELATION_WORDS if k in text), None)
        situation = None
        if target_type == "person":
            situation = next((v for k, v in SITUATION_WORDS if k in text), "normal")
            landmark = None  # SAR problems have no landmark (see summary doc #4)
            relation = None
        return target_type or "", landmark or "", relation, situation

    # -- detection helpers --

    @staticmethod
    def _best_box(det, class_name: str):
        if not class_name:
            return None
        best, best_conf = None, -1.0
        for box in det.boxes:
            if det.names.get(int(box.cls[0])) != class_name:
                continue
            conf = float(box.conf[0])
            if conf > best_conf:
                best, best_conf = box, conf
        return best

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _classify_situation(self, img, person_box) -> str:
        x1, y1, x2, y2 = [int(v) for v in person_box.xyxy[0]]
        pad = 12
        crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if crop.size == 0:
            return "normal"
        result = self._pose_model().predict(crop, imgsz=max(crop.shape[:2]), verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return "normal"
        kpts = result.keypoints.xy[0].cpu().numpy()
        return self._situation_from_keypoints(kpts, crop.shape[0])

    @staticmethod
    def _situation_from_keypoints(kpts, crop_h: int) -> str:
        """Sit/lie/collapsed vs. standing from COCO-17 keypoint y-coordinates.

        Standing: shoulders well above hips well above ankles, spanning most of the
        crop height. Lying/collapsed: shoulders and hips at nearly the same height
        (torso flat). This is a coarse heuristic -- tune thresholds against real
        footage once the platform is running (SAR situations beyond "normal" vs.
        "abnormal" -- accident/emergency -- likely need scene context, not just pose).
        """
        shoulder_y = float(np.mean([kpts[i][1] for i in KP_SHOULDER]))
        hip_y = float(np.mean([kpts[i][1] for i in KP_HIP]))
        ankle_y = float(np.mean([kpts[i][1] for i in KP_ANKLE]))
        torso_span = abs(ankle_y - shoulder_y) or 1.0
        if torso_span > 0.55 * crop_h and hip_y > shoulder_y:
            return "normal"
        return "abnormal"

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
