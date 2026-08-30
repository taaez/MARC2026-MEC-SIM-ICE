"""Record-only VLA grounding: YOLO11 detection over the refined DET_CLASSES taxonomy.

Kept as a separate pipeline from vla_grounding/ (mec_sim_ice/vla_grounding/) on
purpose -- routing recording through the exact same code path as the live
submission would slow down data collection. This module reuses vla_grounding's
command-parsing/geometry/fallback building blocks but keeps its own detector
wiring and MARC_RECORD_DIR dump logic.

The detector is the stock YOLO11 architecture fine-tuned on the class list in
test/det_classes.py (10 targets + 15 landmarks + 9 person_* pose subclasses --
the file moves to demo/ once verified, see test/README.md; both locations are
searched at import time). Person pose is classified DIRECTLY by the detector,
so the earlier YOLO11-pose keypoint pass is gone; ``situation`` now comes from
the organizer's official pose x support rules (vla_grounding/situation_resolver.py),
which is what the stage1 scorer itself uses.

Command parsing reuses vla_grounding/command_parser.py: real voice_command values
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

import json
import logging
import os
import sys
import time

import numpy as np
from ultralytics import YOLO

# marc_sdk pulls in rclpy at import time; tolerate its absence so the class
# constants/static helpers stay importable on a ROS-less PC
# (tools/run_offline_real_vla.py replays MARC_RECORD_DIR dumps there).
# rclpy itself is imported lazily inside the tf-touching methods for the same
# reason -- live runs always have it, offline replay never calls those paths.
try:
    from marc_sdk import GroundingResult
except ImportError:
    GroundingResult = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # vla_grounding/ package lives next to this file

# det_classes.py stages in test/ until the training pipeline is verified, then
# moves to demo/ (test/README.md) -- look in both so this file survives the move.
# (repo cleanup: this file + test/ now live under mec_sim_ice/, while demo/ stays at
# the repo root -- so the demo/ candidate is one level up from _HERE.)
for _candidate in (os.path.join(_HERE, "test"), os.path.join(_HERE, "demo"),
                   os.path.join(os.path.dirname(_HERE), "demo"),
                   os.path.join(_HERE, "training")):  # 폴더 정리 후 현 위치
    if os.path.exists(os.path.join(_candidate, "det_classes.py")):
        sys.path.insert(0, _candidate)
        break
from det_classes import LANDMARK_CLASSES, POSE_CLASSES  # noqa: E402

from vla_grounding.command_parser import (  # noqa: E402
    LANDMARK_SYNONYMS,
    OBJECT_SYNONYMS,
    parse_command,
)
from vla_grounding.dino_fallback import DinoFallbackDetector
from vla_grounding.dino_fallback import is_disabled as dino_disabled  # noqa: E402
from vla_grounding.situation_resolver import (  # noqa: E402
    normalize_support_token,
    resolve_situation,
)
from vla_grounding.vlm_fallback import VlmTargetResolver
from vla_grounding.vlm_fallback import is_disabled as vlm_disabled  # noqa: E402
from vla_grounding.yolo_adapter import load_detector, predict_result  # noqa: E402

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

# Team convention (vla_grounding/RUNBOOK.md): the fine-tuned detector lives at
# demo/weights/best.pt (received from the training teammate, not in git).
DET_WEIGHTS = os.environ.get("VLA_DET_WEIGHTS",
                             os.path.join(_HERE, "demo", "weights", "best.pt"))
DET_IMGSZ = 1280   # CCTV native resolution (1280x720)
DET_CONF = 0.35

# CCTV 프레임을 기다리는 시간. 세션 시작 직후에는 아직 아무 프레임도 안 와 있다:
#   * CCTV image/camera_info 토픽의 QoS 는 BEST_EFFORT + VOLATILE(비래칭, protocol.py
#     QOS_IMAGE) 이라, 구독을 붙여도 **다음 발행이 올 때까지** get_cctv_image() 는 None 이다.
#     (ground_height 만 TRANSIENT_LOCAL 이라 즉시 읽힌다 -- 그래서 카메라가 '발견'됐다고
#     이미지가 있는 게 아니다.)
#   * client.list_cctv() 는 '구독됨'만 뜻하고 데이터 준비 신호가 아니다. 카메라 발견은
#     client.py 의 2초 주기 _discover_cctv 타이머가 한다.
# 그래서 한 번만 긁고 포기하면 세션 앞쪽 라운드가 통째로 빈 제출이 된다 -- 실측으로
# 2026-08-25 수집 세션의 1~4라운드(lf01~lf04)가 이미지 0장으로 기록됐고 35.00 고정이었다
# (프레임은 1라운드 기준 6~8초 뒤에 도착했다). 첫 라운드만 길게 기다리고, 그 뒤에는
# 짧게 재시도한다. 라운드 제한시간이 보통 60s 라 워밍업 30s 는 충분히 안전하다.
CAM_WARMUP_S = float(os.environ.get("MARC_CAM_WARMUP_S", "30"))  # 첫 프레임을 받을 때까지
CAM_RETRY_S = float(os.environ.get("MARC_CAM_RETRY_S", "3"))     # 그 이후 라운드


class RecordOnlyVLAGrounding:
    """Stage 1 / Stage 2 grounding via YOLO11 detection on DET_CLASSES.

    Needs the live ``MARCClient`` (not just its sensor getters) because camera
    extrinsics are read from tf2, which requires the client's spinning rclpy node.
    """

    def __init__(self, client, det_weights: str = DET_WEIGHTS):
        self._client = client
        # MARC_RECORD_ONLY=1: 데이터 수집 전용 모드. YOLO 를 로드/추론하지 않고
        # (weights 파일도 불필요) 라운드마다 CCTV 이미지+카메라 파라미터만
        # MARC_RECORD_DIR 에 덤프한 뒤 빈 payload 를 즉시 제출해 다음 라운드로
        # 넘어간다. 수집한 기록은 tools/run_offline_real_vla.py 로 재실행한다.
        self._record_only = os.getenv("MARC_RECORD_ONLY", "").lower() in ("1", "true", "yes")
        # 소형객체 서브모델 산출물이 있으면 라우팅(B 31클래스 + 서브 3클래스)으로,
        # 없으면 기존 그대로 B 단독으로 로드된다 (vla_grounding/yolo_adapter.load_detector).
        self._det = None if self._record_only else load_detector(
            det_weights, conf=DET_CONF, imgsz=DET_IMGSZ)
        # DINO 2차 탐지 폴백: YOLO가 실패했을 때만 쓴다(§7 규칙대로 생성자에서 로드).
        # torch/transformers 미설치 개발 환경(예: GPU 없는 노트북)에서도 본 파이프라인
        # 자체는 죽지 않도록, 로드 실패는 경고만 남기고 폴백 없이 진행한다.
        self._dino = None
        if not self._record_only and not dino_disabled():
            try:
                self._dino = DinoFallbackDetector()
            except Exception as e:  # noqa: BLE001
                log.warning("[vla] DINO fallback unavailable (%s) -- YOLO miss = no rescue", e)
        # VLM 텍스트 폴백: command_parser가 target을 아예 못 뽑았을 때만 쓴다(§7 규칙대로
        # 생성자에서 로드). 8GB 로컬+Isaac Sim 동시 구동 시 VRAM이 타이트할 수 있으니
        # MARC_DISABLE_VLM=1로 끄고 tools/run_offline_real_vla.py 로 records/ 재실행해서 따로 검증할 것.
        self._vlm = None
        if not self._record_only and not vlm_disabled():
            try:
                self._vlm = VlmTargetResolver()
            except Exception as e:  # noqa: BLE001
                log.warning("[vla] VLM fallback unavailable (%s) -- OOV target = no rescue", e)
        # tf2 needs client.node, which only exists after client.connect() -- init lazily
        # on first use rather than here (RecordOnlyVLAGrounding is constructed before connect()).
        self._tf_buffer = None
        self._tf_listener = None
        # MARC_RECORD_DIR: dump each round's inputs (per-camera PNG + camera
        # params + command) and the submitted payload, in the meta.json schema
        # defined by _record() below -- replayable without Isaac Sim/ROS via
        # tools/run_offline_real_vla.py.
        self._record_dir = os.getenv("MARC_RECORD_DIR")
        self._round = 0
        # 한 번이라도 프레임을 받았는지. 라운드 번호가 아니라 이 플래그로 워밍업을
        # 판단한다 -- 1라운드가 빨리 성공하면 안 기다리고, 앞 라운드가 계속 실패하면
        # 성공할 때까지 계속 길게 기다린다.
        self._cams_warm = False

    def _tf(self):
        if self._tf_buffer is None:
            import tf2_ros
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._client.node)
        return self._tf_buffer

    # -- Stage 1 / Stage 2 entry points (signature must match MockVLAGrounding) --

    def process(self, voice_command: str, camera_ids: list = None) -> GroundingResult:
        self._round += 1
        return self._ground(voice_command, camera_ids or self._client.list_cctv(), "s1")

    def process_stage2(self, task_description: str) -> GroundingResult:
        self._round += 1
        return self._ground(task_description, self._client.list_cctv(), "s2")

    def _observe_images(self, camera_ids: list) -> dict:
        """CCTV 프레임이 다 들어올 때까지 기다렸다가 긁는다. {camera_id: BGR ndarray}.

        왜 기다려야 하는지는 CAM_WARMUP_S 주석 참고 (요약: 이미지 토픽이 비래칭이라
        구독 직후엔 None 이고, list_cctv() 는 데이터 준비 신호가 아니다).

        종료 조건 두 가지:
          * 알려진 카메라 전부의 프레임을 받았고, 그 개수가 CAM_SETTLE_S 동안 더
            안 늘었다 -- 카메라 발견 자체가 2초 타이머라 순차적으로 늘어날 수 있어,
            '지금 아는 카메라가 다 찼다'만으로 끊으면 6대 중 3대만 받고 나갈 수 있다.
          * 제한시간(첫 프레임 전 CAM_WARMUP_S, 그 뒤 CAM_RETRY_S) 초과.

        client.sleep() 은 sim 시간 기반이고, 조기 반환은 Stage2 주행 중에만 걸리므로
        (client.py: ``self._stage2_active and not self.is_running``) 여기서는 안전하다.
        """
        budget = CAM_RETRY_S if self._cams_warm else CAM_WARMUP_S
        # 아직 카메라 구성을 모르는 첫 워밍업에서만 '개수가 안 늘 때까지' 더 본다.
        # 한 번 데워진 뒤에는 카메라 집합이 확정이라, 다 차는 즉시 나가면 된다
        # (안 그러면 매 라운드 CAM_RETRY_S 를 통째로 낭비한다).
        settle_ticks = 0 if self._cams_warm else 25   # 2.5s > _discover_cctv 주기 2.0s
        images, stable = {}, 0
        for _ in range(max(1, int(budget / 0.1))):
            # camera_ids 가 비어 있을 수 있다: participant_app 은 list_cctv() 를 넘기는데
            # 세션 시작 직후엔 아직 아무 카메라도 발견 안 됐을 수 있다. 매 틱 다시 묻는다.
            ids = list(camera_ids) or self._client.list_cctv()
            before = len(images)
            for cam_id in ids:
                if cam_id in images:
                    continue
                img = self._to_bgr(self._client.get_cctv_image(cam_id))
                if img is not None:
                    images[cam_id] = img
            filled = bool(ids) and len(images) == len(ids)
            stable = stable + 1 if (filled and len(images) == before) else 0
            if filled and stable >= settle_ticks:
                break
            self._client.sleep(0.1)

        if images:
            if not self._cams_warm:
                log.info("[vla] CCTV warm-up done: %d camera(s) ready", len(images))
            self._cams_warm = True
        else:
            log.error("[vla] round %d: no CCTV frame after %.0fs -- submitting empty "
                      "grounding. (discovered=%s)", self._round, budget,
                      self._client.list_cctv())
        return images

    def _ground(self, command: str, camera_ids: list, stage: str = "s1") -> GroundingResult:
        parsed = parse_command(command)
        target_type = parsed.target_type or ("person" if parsed.is_person else "")
        landmark = parsed.landmark_type or ""

        # target_type 파싱 실패(닫힌 어휘 밖 단어) -> VLM에게 한 번 더 묻는다.
        # 성공하면 알던 클래스로 복귀(정상 YOLO 경로 그대로 탄다), 실패하면
        # DINO에 넘길 검색어(vlm_search_phrase)만 얻고 target_type은 빈 채로 둔다.
        # "behind" 중의(구동사 leave-behind vs 공간관계) -- command_parser 가
        # confident 를 낮춘 문장에서만 VLM 이 중재한다. 구동사로 판정되면 물건은
        # 그 자리 근처에 남아있는 것이므로 near 로 바꾼다. 공간이면 behind 유지
        # (확신 승격은 안 한다 -- VLM 도 틀릴 수 있어 기하 재검증 여지를 남김).
        if (self._vlm is not None and not parsed.is_person
                and parsed.relation_hint == "behind" and not parsed.relation_confident):
            spatial = self._vlm.behind_is_spatial(command)
            if spatial is False:
                parsed.relation_hint = "near"
                log.info("[vla] VLM arbitration: 'behind' is phrasal -> relation near")
            elif spatial is True:
                log.info("[vla] VLM arbitration: 'behind' is spatial -> keep hint")

        vlm_search_phrase = None
        if not target_type and not parsed.is_person and self._vlm is not None:
            matched, phrase = self._vlm.resolve(command, OBJECT_SYNONYMS.keys())
            if matched:
                target_type = matched
                log.info("[vla] VLM fallback resolved OOV wording -> %r", matched)
            elif phrase:
                vlm_search_phrase = phrase
                log.info("[vla] VLM fallback: unknown object, DINO search phrase=%r", phrase)

        # Grab every camera frame up front (not lazily inside the match loop) so a
        # MARC_RECORD_DIR dump always contains the full round regardless of which
        # camera matched first.
        images = self._observe_images(camera_ids)

        if self._record_only:
            # 수집 전용: 추론 없이 기록만 남기고 빈 제출로 라운드를 넘긴다.
            result = GroundingResult(
                camera_id="", target_type=target_type,
                anchor_coord=[0.0, 0.0, 0.0], target_coord=[0.0, 0.0, 0.0],
                landmark=landmark,
                relation=None if parsed.is_person else parsed.relation_hint,
                situation=None,
            )
            self._record(stage, command, images, result.to_payload())
            log.info("[vla] record-only round done (%d cameras)", len(images))
            return result

        # camera_ids 가 아니라 images 를 돈다: 세션 시작 직후엔 participant_app 이 넘긴
        # camera_ids 가 빈 리스트일 수 있고(_observe_images 가 그 사이 발견된 카메라까지
        # 채워 온다), images 의 키가 곧 '프레임이 있는 카메라'라 조건도 하나 준다.
        for cam_id, img in images.items():
            det = predict_result(self._det, img, imgsz=DET_IMGSZ, conf=DET_CONF)

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
            result = GroundingResult(
                camera_id=cam_id, target_type=target_type,
                anchor_coord=anchor_coord, target_coord=target_coord,
                landmark=landmark, relation=relation, situation=situation,
            )
            self._record(stage, command, images, result.to_payload())
            return result

        # -- YOLO가 전 카메라에서 실패 -> DINO로 한 번 더 (사물 문제만; 사람/situation
        #    문제는 pose 분류가 필요해 개방어휘 탐지기로 구제 불가하므로 대상 밖) --
        if self._dino is not None and not parsed.is_person and (target_type or vlm_search_phrase):
            # target_type이 있으면 그 클래스의 동의어들로 검색(정밀), 없으면(VLM이 준
            # 자유 검색어만 있는 완전 신규 물체) 그 phrase 하나로 검색(관대).
            target_phrases = (self._dino_phrases(target_type, OBJECT_SYNONYMS) if target_type
                              else [vlm_search_phrase])
            for cam_id, img in images.items():   # YOLO 루프와 같은 이유로 images 기준
                target_box = self._dino.best_box(img, target_phrases)
                if target_box is None:
                    continue

                anchor_box = None
                if landmark:
                    anchor_box = self._dino.best_box(img, self._dino_phrases(landmark, LANDMARK_SYNONYMS))
                    if anchor_box is None:
                        continue  # landmark not visible here -- wrong camera

                target_coord = self._pixel_to_world(cam_id, self._box_center(target_box))
                anchor_coord = (self._pixel_to_world(cam_id, self._box_center(anchor_box))
                                 if anchor_box is not None else target_coord)
                result = GroundingResult(
                    camera_id=cam_id, target_type=target_type,
                    anchor_coord=anchor_coord, target_coord=target_coord,
                    landmark=landmark, relation=parsed.relation_hint, situation=None,
                )
                log.info("[vla] DINO fallback rescued %s on %s (YOLO missed it)", target_type, cam_id)
                self._record(stage, command, images, result.to_payload())
                return result

        log.warning("[vla] no camera matched target=%r landmark=%r pose_hint=%r",
                    target_type, landmark, parsed.pose_hint)
        situation = None
        if parsed.is_person:
            # Last-ditch guess so person submissions still carry a situation:
            # assume the most likely pose for the hinted bucket (walking if none).
            guess_pose = POSE_BUCKET_CLASSES.get(parsed.pose_hint, ("person_walking",))[0]
            situation = resolve_situation(guess_pose, landmark or parsed.area_hint).situation
        result = GroundingResult(
            camera_id="", target_type=target_type, anchor_coord=[0.0, 0.0, 0.0],
            target_coord=[0.0, 0.0, 0.0], landmark=landmark,
            relation=None if parsed.is_person else parsed.relation_hint,
            situation=situation,
        )
        self._record(stage, command, images, result.to_payload())
        return result

    # -- MARC_RECORD_DIR round dump (offline replay input) --

    def _record(self, stage: str, command: str, images: dict, payload: dict):
        """Dump one round: <dir>/<stage>_<round>_<ts>/{<cam_id>.png..., meta.json}.

        Read back by tools/run_offline_real_vla.py for offline replay. Never
        allowed to break the live round -- failures are logged and swallowed.
        """
        if not self._record_dir:
            return
        try:
            from PIL import Image
            d = os.path.join(self._record_dir,
                             f"{stage}_{self._round:03d}_{int(time.time())}")
            os.makedirs(d, exist_ok=True)
            cams = {}
            for cam_id, bgr in images.items():
                meta = self._camera_meta(cam_id)
                if meta is None:
                    continue
                # saved as RGB; offline loaders convert back to BGR for YOLO
                Image.fromarray(bgr[:, :, ::-1]).save(os.path.join(d, f"{cam_id}.png"))
                cams[cam_id] = meta
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"command": command, "cameras": cams, "payload": payload},
                          f, indent=1)
            log.info("[vla] recorded -> %s", d)
        except Exception as e:  # noqa: BLE001
            log.warning("[vla] record failed: %s", e)

    def _camera_meta(self, cam_id: str):
        """{K, R_cw, C_w, ground_z, width, height} for meta.json, or None."""
        import rclpy
        info = self._client.get_cctv_info(cam_id)
        ground_z = self._client.get_cctv_ground_height(cam_id)
        if info is None or ground_z is None:
            return None
        try:
            tf = self._tf().lookup_transform(
                "world", cam_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:  # noqa: BLE001
            log.warning("[vla] record: tf lookup world<-%s failed: %s", cam_id, e)
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        R = self._quat_to_R(q.w, q.x, q.y, q.z)
        return {"K": np.array(info.k, dtype=float).reshape(3, 3).tolist(),
                "R_cw": R.tolist(), "C_w": [t.x, t.y, t.z],
                "ground_z": float(ground_z),
                "width": int(info.width), "height": int(info.height)}

    # -- detection helpers --

    @staticmethod
    def _dino_phrases(class_name: str, synonym_table: dict) -> list:
        """클래스명 -> DINO 검색용 자연어 phrase 목록.

        command_parser.py의 동의어 사전을 그대로 재사용(어휘 이원화 방지).
        빈 문자열(파싱 자체가 실패한 OOV 케이스)이면 검색할 phrase가 없다 --
        그 경우 호출부는 이 함수 대신 VLM 폴백이 만든 vlm_search_phrase를 쓴다.
        """
        if not class_name:
            return []
        return synonym_table.get(class_name, [class_name.replace("_", " ")])

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
        """Ground-contact pixel: bottom-center ((x1+x2)/2, y2), NOT the bbox center.

        ``_pixel_to_world`` intersects the pixel ray with the ground plane, so the
        pixel handed to it must be the one that actually lies on the ground -- the
        bottom edge of the box. Back-projecting the bbox CENTER instead makes the
        ray hit the ground beyond the object, and the overshoot grows as the camera
        gets lower: roughly ``distance * (H/2) / (h - H/2)`` for an object of height
        H under a camera h above the ground. For a 1.7 m person 10 m away that is
        +4.6 m on rig_3_b (2.68 m mount) and +2.0 m on rig_2_b (5.04 m) -- at or past
        the 5 m scoring radius. DESIGN.md 2.3 measured the same thing directly:
        1.73 m error from the center pixel, 0 from the floor point.

        Name kept as-is so the call sites read unchanged; it returns the contact
        point, which is what every caller here actually wants.
        """
        x1, _y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        return (x1 + x2) / 2.0, y2

    # -- pixel -> world (ground-plane back-projection) --

    def _pixel_to_world(self, cam_id: str, pixel):
        import rclpy
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
