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

from taejeong.command_parser import (
    LANDMARK_SYNONYMS,
    OBJECT_SYNONYMS,
    parse_command,
)
from taejeong.dino_fallback import DinoFallbackDetector
from taejeong.dino_fallback import is_disabled as dino_disabled
from taejeong.grounding import (
    Detection,
    _landmark_ok,
    _target_ok,
    camera_from_ros,
    resolve,
)
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
        target_phrases = self._vlm_assist(command, parsed)

        images = {}
        for cam_id in camera_ids:
            img = self._to_bgr(self._client.get_cctv_image(cam_id))
            if img is not None:
                images[cam_id] = img

        detections = infer_cameras(self._det, images, min_conf=self._conf,
                                   imgsz=self._imgsz)
        self._dino_augment(parsed, images, detections, target_phrases)

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

    # -- VLM/DINO 폴백 (규칙/YOLO 가 실패한 지점만 보충; 정상 경로 비용 0) --

    def _vlm_assist(self, command: str, parsed):
        """VLM 보조 2건: behind 중의 중재 + OOV target 재해석 (parsed 제자리 수정).

        반환: DINO 검색용 target phrase 목록. 아는 클래스면 동의어 사전, 진짜
        신규 물체면 VLM 이 만든 검색어 1개, 해석 불능/사람 문제면 None.
        """
        if (self._vlm is not None and not parsed.is_person
                and parsed.relation_hint == "behind" and not parsed.relation_confident):
            # 구동사 leave-behind vs 공간관계 -- command_parser 가 confident 를
            # 낮춘 중의 문장만 온다. phrasal 이면 물건은 그 자리 근처에 남은 것.
            spatial = self._vlm.behind_is_spatial(command)
            if spatial is False:
                parsed.relation_hint = "near"
                log.info("[taejeong] VLM arbitration: 'behind' is phrasal -> near")
            elif spatial is True:
                log.info("[taejeong] VLM arbitration: 'behind' is spatial -> keep hint")
        if parsed.is_person:
            return None
        if parsed.target_type is None and self._vlm is not None:
            matched, phrase = self._vlm.resolve(command, OBJECT_SYNONYMS.keys())
            if matched:
                parsed.target_type = matched     # 사전 미등재 표현 -> 아는 클래스로 복귀
                log.info("[taejeong] VLM fallback resolved OOV wording -> %r", matched)
            elif phrase:
                # 진짜 신규 물체: phrase 를 클래스명처럼 쓴다. type 점수는 어차피
                # 닫힌 어휘 밖이라 못 얻지만, resolve() 의 _target_ok 가 label
                # 일치로 DINO 탐지를 통과시키려면 target_type 과 label 이 같아야
                # 하고, 그래야 좌표 점수는 살릴 수 있다.
                parsed.target_type = phrase
                log.info("[taejeong] VLM fallback: unknown object, DINO phrase=%r", phrase)
                return [phrase]
        if parsed.target_type:
            return OBJECT_SYNONYMS.get(parsed.target_type,
                                       [parsed.target_type.replace("_", " ")])
        return None

    def _dino_augment(self, parsed, images: dict, detections: dict, target_phrases):
        """YOLO 가 어디서도 못 찾은 target/landmark 를 DINO 로 보충.

        detections dict 에 Detection 을 추가할 뿐 별도 제출 경로가 없다 --
        resolve() 의 카메라 게이트/기하 판정/relation 적합도가 폴백 탐지에도
        똑같이 적용된다. 사람 문제는 pose 분류가 필요해 대상 밖.
        """
        if self._dino is None or parsed.is_person:
            return
        if target_phrases and not any(_target_ok(d.label, parsed)
                                      for dets in detections.values() for d in dets):
            for cam_id, img in images.items():
                box = self._dino.best_box(img, target_phrases)
                if box is not None:
                    detections.setdefault(cam_id, []).append(Detection(
                        label=parsed.target_type, bbox=tuple(box.xyxy[0]),
                        score=float(box.conf[0])))
                    log.info("[taejeong] DINO fallback rescued target %r on %s",
                             parsed.target_type, cam_id)
        lm = parsed.landmark_type
        if lm and not any(_landmark_ok(d.label, lm)
                          for dets in detections.values() for d in dets):
            # 제너릭 "car" 힌트도 그대로 label 로 쓴다 -- grounding.CAR_CLASSES 에
            # "car" 가 포함돼 _landmark_ok 를 통과하고, situation 규칙의 support
            # 토큰 정규화도 "car" 를 안다.
            phrases = LANDMARK_SYNONYMS.get(lm, [lm.replace("_", " ")])
            for cam_id, img in images.items():
                box = self._dino.best_box(img, phrases)
                if box is not None:
                    detections.setdefault(cam_id, []).append(Detection(
                        label=lm, bbox=tuple(box.xyxy[0]), score=float(box.conf[0])))
                    log.info("[taejeong] DINO fallback rescued landmark %r on %s",
                             lm, cam_id)

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
