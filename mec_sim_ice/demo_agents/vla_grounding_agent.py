"""ROS-facing adapter around vla_grounding/ (text parsing + geometry, no scenario hardcoding).

Wires vla_grounding's pure-Python pipeline (command_parser -> yolo_adapter -> grounding)
to the live MARCClient: builds CameraModel from tf2 + CameraInfo per camera,
runs the teammate's YOLO detector, and converts the result to GroundingResult.

Failure-path fallbacks (vla_grounding/dino_fallback.py, vla_grounding/vlm_fallback.py):
the VLM re-reads commands whose target is out of the closed vocabulary and
arbitrates ambiguous "behind" (phrasal vs spatial); Grounding DINO re-detects
a target/landmark that YOLO missed in every camera. Both only ADD Detection
entries to the per-camera dict, so resolve()'s camera gating and relation
geometry apply to fallback detections unchanged -- there is no separate
submission path, and the normal YOLO route costs nothing extra.

This is the team's Stage 1 implementation: participant_app.py (the default
entry point) wires it. A separate record-only pipeline (record_agent_vla.py)
stays available side-by-side via MARC_ENTRYPOINT=participant_app_record.py.

Weights: demo_agents/weights/best.pt (this file's own directory) -- the team's
YOLO detector, classes per training/det_classes.py. Override with
VLA_DET_WEIGHTS if needed (record_agent_vla.py reads the same variable).
"""

import logging
import os

from marc_sdk import GroundingResult

from vla_grounding.command_parser import parse_command
from vla_grounding.dino_fallback import DinoFallbackDetector
from vla_grounding.dino_fallback import is_disabled as dino_disabled
from vla_grounding.fallback import dino_augment, pose_augment, vlm_assist
from vla_grounding.grounding import camera_from_ros, resolve
from vla_grounding.pose_keypoints import AnkleKeypointEstimator
from vla_grounding.pose_keypoints import is_disabled as pose_disabled
from vla_grounding.vlm_fallback import VlmTargetResolver
from vla_grounding.vlm_fallback import is_disabled as vlm_disabled
from vla_grounding.yolo_adapter import (infer_cameras, load_detector,
                                         on_target_probe_augment)

log = logging.getLogger("participant_app")

_HERE = os.path.dirname(os.path.abspath(__file__))
DET_WEIGHTS = os.environ.get("VLA_DET_WEIGHTS",
                             os.path.join(_HERE, "weights", "best.pt"))
DET_IMGSZ = 1280   # CCTV native resolution (1280x720)
# 0.35 -> 0.25: 오프라인 실측에서 0.35 는 53.48, 0.25 이하는 57.80 이었다
# (39문항 동일 records). 탐지 recall 이 이 파이프라인의 병목이라 임계값이
# 그대로 점수다. 0.10/0.15/0.25 는 결과가 완전히 동일해서, 오검출 여지가
# 가장 적은 0.25 를 고른다 (채점 스크립트 기본값과도 일치).
DET_CONF = 0.25


class VLAGroundingAgent:
    """Stage 1 / Stage 2 grounding via general text parsing + pure geometry.

    Needs the live MARCClient (not just its sensor getters) because camera
    extrinsics are read from tf2, which requires the client's spinning rclpy
    node (tf2 needs a spinning node).
    """

    def __init__(self, client, det_weights: str = DET_WEIGHTS,
                conf: float = DET_CONF, imgsz: int = DET_IMGSZ):
        self._client = client
        # 소형객체 서브모델(sunblock/tissue/disposable_cup) 산출물이 있으면
        # 라우팅 검출기로, 없으면 기존과 똑같은 B 단독 YOLO 로 로드된다
        # (vla_grounding/yolo_adapter.load_detector). MARC_DISABLE_SMALL_DET=1 로 차단.
        self._det = load_detector(det_weights, conf=conf, imgsz=imgsz)
        self._conf = conf
        self._imgsz = imgsz
        self._tf_buffer = None
        self._tf_listener = None
        # DINO/VLM 폴백. §7 규칙대로 생성자에서 로드하고, 로드 실패는 경고만
        # -- 기본 YOLO+기하 경로는 절대 안 깨진다.
        # DINO 는 기본 ON (chungmu 39/39 실측으로 검증: rig_3_b 3문항 구제,
        # +107.3점 -- MARC_DISABLE_DINO_FALLBACK=1 로 끌 수 있음).
        # VLM 은 기본 OFF (2026-08-29부터, vlm_fallback.py 참고 -- 같은 실측에서
        # 39/39 전부 호출 조건이 안 걸려 0회 호출되면서 ~7GB 만 상주했다. 이
        # 데스크탑은 그 8GB 를 Isaac Sim 과 나눠 쓰므로 얻는 것 없이 OOM
        # 위험만 진다. 본선 대비 OOV 보험으로 켜려면 MARC_ENABLE_VLM=1).
        self._dino = None
        if not dino_disabled():
            try:
                self._dino = DinoFallbackDetector()
            except Exception as e:  # noqa: BLE001
                log.warning("[vla_grounding] DINO fallback unavailable (%s) -- YOLO miss = no rescue", e)
        self._vlm = None
        if not vlm_disabled():
            try:
                self._vlm = VlmTargetResolver()
            except Exception as e:  # noqa: BLE001
                log.warning("[vla_grounding] VLM fallback unavailable (%s) -- OOV target = no rescue", e)
        # 사람 target 의 발목 keypoint. 정답이 사람의 '중심' 이 아니라 '발' 이라
        # bbox 가로 중앙으로는 몸길이의 절반만큼 어긋난다 (pose_keypoints.py
        # docstring). DINO/VLM 과 달리 폴백 스위치와 무관하게 켠다 --
        # offline_score.py 가 "라이브는 항상 켠다"를 전제로 같은 자리에서 항상
        # 호출하므로, 여기서 끄면 두 경로의 점수가 갈라진다. 없으면 기존 동작 그대로.
        self._pose = None
        if not pose_disabled():
            try:
                self._pose = AnkleKeypointEstimator()
            except Exception as e:  # noqa: BLE001
                log.warning("[vla_grounding] pose keypoints unavailable (%s) -- "
                            "사람 좌표는 bbox 중앙 기준", e)

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
        # ``on`` 물체가 정상 문턱에서 빠졌을 때만, landmark 상면과 겹치는
        # 동일 YOLO raw 후보를 보강한다. 전역 confidence 문턱은 건드리지 않는다.
        on_target_probe_augment(parsed, self._det, images, detections,
                                imgsz=self._imgsz)
        dino_augment(parsed, images, detections, target_phrases, self._dino, self._vlm)
        # 마지막 -- 위 단계가 추가한 person 탐지까지 발목 보정을 받아야 한다.
        pose_augment(parsed, images, detections, self._pose)

        cameras = {}
        for cam_id in images:
            cam = self._camera_model(cam_id)
            if cam is not None:
                cameras[cam_id] = cam

        decision = resolve(parsed, detections, cameras)
        if decision is None:
            log.warning("[vla_grounding] no camera matched target=%r landmark=%r "
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
            log.warning("[vla_grounding] tf lookup world<-%s failed: %s", cam_id, e)
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
        log.warning("[vla_grounding] unexpected CCTV encoding %r -- add a case in _to_bgr", enc)
        return None
