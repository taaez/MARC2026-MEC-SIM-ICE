"""ROS-facing adapter around taejeong/ (text parsing + geometry, no scenario hardcoding).

Wires taejeong's pure-Python pipeline (command_parser -> yolo_adapter -> grounding)
to the live MARCClient: builds CameraModel from tf2 + CameraInfo per camera,
runs the teammate's YOLO detector, and converts the result to GroundingResult.

Failure-path fallbacks (taejeong/dino_fallback.py, taejeong/vlm_fallback.py):
the VLM re-reads commands whose target is out of the closed vocabulary and
arbitrates ambiguous "behind" (phrasal vs spatial); Grounding DINO re-detects
a target/landmark that YOLO missed in every camera. Both only ADD Detection
entries to the per-camera dict, so resolve()'s camera gating and relation
geometry apply to fallback detections unchanged -- there is no separate
submission path, and the normal YOLO route costs nothing extra.

This is the team's Stage 1 implementation: participant_app.py (the default
entry point) wires it. The Grounding DINO + Qwen2.5-VL experiment stays
available side-by-side via MARC_ENTRYPOINT=participant_app_agent_vla.py.

Weights: demo/weights/best.pt -- the team's YOLO detector, classes per
test/det_classes.py. Override with TAEJEONG_DET_WEIGHTS if needed.
"""

import logging
import os

from ultralytics import YOLO

from marc_sdk import GroundingResult

from taejeong.command_parser import parse_command
from taejeong.dino_fallback import DinoFallbackDetector
from taejeong.dino_fallback import is_disabled as dino_disabled
from taejeong.fallback import dino_augment, pose_augment, vlm_assist
from taejeong.grounding import camera_from_ros, resolve
from taejeong.pose_keypoints import AnkleKeypointEstimator
from taejeong.pose_keypoints import is_disabled as pose_disabled
from taejeong.vlm_fallback import VlmTargetResolver
from taejeong.vlm_fallback import is_disabled as vlm_disabled
from taejeong.yolo_adapter import infer_cameras

log = logging.getLogger("participant_app")

_HERE = os.path.dirname(os.path.abspath(__file__))
DET_WEIGHTS = os.environ.get("TAEJEONG_DET_WEIGHTS",
                             os.path.join(_HERE, "weights", "best.pt"))
DET_IMGSZ = 1280   # CCTV native resolution (1280x720)
# 0.35 -> 0.25: 오프라인 실측에서 0.35 는 53.48, 0.25 이하는 57.80 이었다
# (39문항 동일 records). 탐지 recall 이 이 파이프라인의 병목이라 임계값이
# 그대로 점수다. 0.10/0.15/0.25 는 결과가 완전히 동일해서, 오검출 여지가
# 가장 적은 0.25 를 고른다 (채점 스크립트 기본값과도 일치).
DET_CONF = 0.25


class TaejeongVLAGrounding:
    """Stage 1 / Stage 2 grounding via general text parsing + pure geometry.

    Needs the live MARCClient (not just its sensor getters) because camera
    extrinsics are read from tf2, which requires the client's spinning rclpy
    node (tf2 needs a spinning node).
    """

    def __init__(self, client, det_weights: str = DET_WEIGHTS,
                conf: float = DET_CONF, imgsz: int = DET_IMGSZ):
        self._client = client
        self._det = YOLO(det_weights)
        self._conf = conf
        self._imgsz = imgsz
        self._tf_buffer = None
        self._tf_listener = None
        # DINO/VLM 폴백 (real_agent_vla 와 동일 정책). §7 규칙대로 생성자에서
        # 로드하고, 로드 실패는 경고만 -- 기본 YOLO+기하 경로는 절대 안 깨진다.
        # MARC_DISABLE_DINO_FALLBACK / MARC_DISABLE_VLM 으로 개별 차단 가능.
        self._dino = None
        if not dino_disabled():
            try:
                self._dino = DinoFallbackDetector()
            except Exception as e:  # noqa: BLE001
                log.warning("[taejeong] DINO fallback unavailable (%s) -- YOLO miss = no rescue", e)
        self._vlm = None
        if not vlm_disabled():
            try:
                self._vlm = VlmTargetResolver()
            except Exception as e:  # noqa: BLE001
                log.warning("[taejeong] VLM fallback unavailable (%s) -- OOV target = no rescue", e)
        # 사람 target 의 발 위치용 keypoint 모델. 정답이 사람의 '중심' 이 아니라
        # '발' 이라 bbox 가로 중앙으로는 몸길이의 절반만큼 어긋난다
        # (pose_keypoints.py docstring). 없으면 기존 동작 그대로다.
        self._pose = None
        if not pose_disabled():
            try:
                self._pose = AnkleKeypointEstimator()
            except Exception as e:  # noqa: BLE001
                log.warning("[taejeong] pose keypoints unavailable (%s) -- "
                            "person coords fall back to bbox center", e)

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
        target_phrases = vlm_assist(command, parsed, self._vlm)

        images = {}
        for cam_id in camera_ids:
            img = self._to_bgr(self._client.get_cctv_image(cam_id))
            if img is not None:
                images[cam_id] = img

        detections = infer_cameras(self._det, images, min_conf=self._conf,
                                   imgsz=self._imgsz)
        dino_augment(parsed, images, detections, target_phrases, self._dino)
        pose_augment(parsed, images, detections, self._pose)

        cameras = {}
        for cam_id in images:
            cam = self._camera_model(cam_id)
            if cam is not None:
                cameras[cam_id] = cam

        decision = resolve(parsed, detections, cameras)
        if decision is None:
            log.warning("[taejeong] no camera matched target=%r landmark=%r "
                       "(person=%s pose=%s)", parsed.target_type,
                       parsed.landmark_type, parsed.is_person, parsed.pose_hint)
            fallback_type = parsed.target_type or ("person" if parsed.is_person else "")
            return GroundingResult(
                camera_id="", target_type=fallback_type,
                anchor_coord=[0.0, 0.0, 0.0], target_coord=[0.0, 0.0, 0.0],
                landmark=parsed.landmark_type or "",
                relation=None if parsed.is_person else parsed.relation_hint,
                situation=None,
            )
        return GroundingResult.from_payload(decision.to_payload())

    # -- camera model from live sensors --

    def _camera_model(self, cam_id: str):
        info = self._client.get_cctv_info(cam_id)
        ground_z = self._client.get_cctv_ground_height(cam_id)
        if info is None or ground_z is None:
            return None
        try:
            tf = self._tf().lookup_transform(
                "world", cam_id, __import__("rclpy").time.Time(),
                timeout=__import__("rclpy").duration.Duration(seconds=0.5))
        except Exception as e:  # noqa: BLE001
            log.warning("[taejeong] tf lookup world<-%s failed: %s", cam_id, e)
            return None
        return camera_from_ros(cam_id, info, tf, float(ground_z))

    # -- sensor_msgs/Image -> BGR ndarray (what ultralytics/opencv expect) --

    @staticmethod
    def _to_bgr(img_msg):
        import numpy as np
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
        log.warning("[taejeong] unexpected CCTV encoding %r -- add a case in _to_bgr", enc)
        return None
