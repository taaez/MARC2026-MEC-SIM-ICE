"""
demo/agent_vla.py  --  mock_agent_vla.MockVLAGrounding 의 실제 구현 (drop-in 교체).

participant_app.py 에서 바꿀 것은 두 줄뿐이다:
    from agent_vla import VLAGrounding            # (mock_agent_vla 대신)
    self.vla = VLAGrounding(self.client)          # (MockVLAGrounding() 대신; client 가 필요)

인터페이스는 mock 과 동일:
    process(voice_command, camera_ids=None) -> GroundingResult          # Stage 1 (msg 301)
    process_stage2(task_description)        -> GroundingResult          # Stage 2 (msg 311)

mock 과의 차이:
  * CCTV 이미지/K/tf_static/ground_height 를 실제로 읽어 stage1.pipeline 으로 추론한다.
  * 모델은 생성자에서 1회 로드한다 (participant_app 은 connect() 전에 생성하므로 첫 라운드 지연 없음).
  * MARC_RECORD_DIR 이 설정돼 있으면 라운드마다 입력(이미지·카메라·명령)과 출력(payload)을 덤프한다
    → Isaac Sim 없는 PC 에서 tools/eval_offline.py 로 재현·회귀 테스트.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from marc_sdk import GroundingResult

from stage1.geometry import CameraModel, camera_from_ros
from stage1.pipeline import CameraObservation, Stage1Pipeline, build_pipeline

log = logging.getLogger("participant_app")


def image_msg_to_rgb(msg) -> np.ndarray:
    """sensor_msgs/Image(rgb8|bgr8) -> HxWx3 uint8 RGB. cv_bridge 없이 처리."""
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unexpected encoding {msg.encoding}")
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.step != msg.width * 3:
        arr = arr.reshape(msg.height, msg.step)[:, : msg.width * 3]
    img = arr.reshape(msg.height, msg.width, 3)
    return np.ascontiguousarray(img[..., ::-1] if msg.encoding == "bgr8" else img)


class VLAGrounding:
    def __init__(self, client, pipeline: Optional[Stage1Pipeline] = None,
                 default_time_limit: float = 60.0):
        self.client = client
        self.pipe = pipeline or build_pipeline(now=client.now_s)
        self.default_time_limit = default_time_limit
        self._cams: Dict[str, CameraModel] = {}
        self._tf: Dict[str, object] = {}
        self._tf_lock = threading.Lock()
        self._tf_sub = None
        self._record_dir = os.getenv("MARC_RECORD_DIR")
        self._round = 0
        # 현재 라운드 제한시간. participant_app 의 _on_mission 이 set_time_limit() 으로 넘겨주면 사용.
        self._time_limit = default_time_limit

    # ---- 외부에서 라운드 메타를 넘길 때 (선택) ----
    def set_time_limit(self, seconds: float):
        if seconds and seconds > 0:
            self._time_limit = float(seconds)

    # ---- /tf_static (SDK 가 감싸지 않음 → escape hatch subscribe) ----
    def _ensure_tf(self):
        if self._tf_sub is not None:
            return
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from tf2_msgs.msg import TFMessage

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=50)

        def cb(msg):
            with self._tf_lock:
                for t in msg.transforms:
                    if t.header.frame_id == "world":
                        self._tf[t.child_frame_id] = t

        self._tf_sub = self.client.subscribe("/tf_static", TFMessage, cb, qos=qos)

    def _camera(self, cid: str) -> Optional[CameraModel]:
        if cid in self._cams:
            return self._cams[cid]
        info = self.client.get_cctv_info(cid)
        gz = self.client.get_cctv_ground_height(cid)
        with self._tf_lock:
            tfm = self._tf.get(cid)
        if info is None or gz is None or tfm is None:
            return None
        cm = camera_from_ros(cid, info, tfm, gz)
        self._cams[cid] = cm
        log.info("[vla] camera %s ready C=%s ground_z=%.3f", cid, np.round(cm.C_w, 2), cm.ground_z)
        return cm

    def _observe(self, camera_ids: Optional[List[str]], wait_s: float = 3.0) -> List[CameraObservation]:
        self._ensure_tf()
        ids = list(camera_ids or self.client.list_cctv())
        obs: List[CameraObservation] = []
        # 반복 횟수로 예산을 잰다(시간값 비교 아님): client.now_s()는 /clock을 아직 못 받았을 때
        # time.time()(벽시계)으로 폴백하는데, 이 메서드의 첫 호출이 하필 그 순간과 겹치면
        # 이후 /clock 수신 후의 (훨씬 작은) sim 시간과 비교돼 종료 조건이 영원히 거짓이 된다
        # (실측: 라운드1에서 이 경합으로 무한 루프 재현됨). 매 반복은 client.sleep()으로 페이싱하므로
        # 여전히 sim 시간 기반이고 벽시계 sleep은 쓰지 않는다.
        missing: List[str] = []
        for _ in range(max(1, int(wait_s / 0.1))):
            missing = []
            for cid in ids:
                if any(o.cam.camera_id == cid for o in obs):
                    continue
                cm, img = self._camera(cid), self.client.get_cctv_image(cid)
                if cm is None or img is None:
                    missing.append(cid)
                else:
                    obs.append(CameraObservation(cm, image_msg_to_rgb(img)))
            if not missing:
                break
            self.client.sleep(0.1)
        if missing:
            log.warning("[vla] cameras without data: %s", missing)
        return obs

    # ---- 기록 (오프라인 재현용) ----
    def _record(self, command: str, obs: List[CameraObservation], payload: dict, stage: str):
        if not self._record_dir:
            return
        try:
            from PIL import Image
            # 벽시계는 폴더명 유일성 확보용(세션 재시작 시 sim 시간이 리셋돼 충돌) — 타이밍 용도 아님.
            d = os.path.join(self._record_dir, f"{stage}_{self._round:03d}_{int(time.time())}")
            os.makedirs(d, exist_ok=True)
            cams = {}
            for o in obs:
                Image.fromarray(o.image).save(os.path.join(d, f"{o.cam.camera_id}.png"))
                cams[o.cam.camera_id] = {"K": o.cam.K.tolist(), "R_cw": o.cam.R_cw.tolist(),
                                         "C_w": o.cam.C_w.tolist(), "ground_z": o.cam.ground_z,
                                         "width": o.cam.width, "height": o.cam.height}
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"command": command, "cameras": cams, "payload": payload}, f, indent=1)
            log.info("[vla] recorded -> %s", d)
        except Exception as e:                      # noqa: BLE001
            log.warning("[vla] record failed: %s", e)

    # ---- 인터페이스 ----
    def process(self, voice_command: str, camera_ids: list = None) -> GroundingResult:
        self._round += 1
        t0 = self.client.now_s()          # 시뮬레이션 시간 (/clock). 벽시계 아님.
        obs = self._observe(camera_ids)
        if not obs:
            log.error("[vla] no CCTV observation -- submitting empty grounding")
            return GroundingResult.from_payload({"camera_id": "", "interpretation": {}, "grounding": {}})
        payload = self.pipe.run(voice_command, obs, time_limit_s=self._time_limit - (self.client.now_s() - t0))
        self._record(voice_command, obs, payload, "s1")
        return GroundingResult.from_payload(payload)

    def process_stage2(self, task_description: str) -> GroundingResult:
        # Stage 2 해석(msg 311)은 Stage 1 과 같은 payload 규격이다. 동일 파이프라인을 재사용한다.
        # (reveal 이 대략 위치를 주므로 정확도보다 빠른 제출이 우선 — 제한시간의 절반만 쓴다)
        self._round += 1
        obs = self._observe(None)
        if not obs:
            return GroundingResult.from_payload({"camera_id": "", "interpretation": {}, "grounding": {}})
        payload = self.pipe.run(task_description, obs, time_limit_s=self._time_limit * 0.5)
        self._record(task_description, obs, payload, "s2")
        return GroundingResult.from_payload(payload)
