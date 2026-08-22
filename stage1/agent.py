"""
stage1/agent.py  —  참가자 애플리케이션 진입점 (mock_agent_vla.py 대체)

실행:  python -m stage1.agent
환경변수: MARC_TEAM_ID, MARC_TOKEN (compose 의 기본값으로 하드코딩), 모델 관련은 pipeline.py 참조.

흐름:
  1) 모델 로드(워밍업) → 2) client.connect() → 3) /tf_static 구독으로 CCTV extrinsic 캐시
  4) on_mission: 모든 CCTV 의 최신 프레임 + K + ground_height 수집 → 파이프라인 → submit_grounding
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from marc_sdk import GroundingResult, MARCClient

from .geometry import CameraModel, camera_from_ros
from .pipeline import CameraObservation, Stage1Pipeline, build_pipeline

logging.basicConfig(level=os.getenv("MARC_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stage1.agent")


# ----------------------------------------------------------------------------
# ROS 메시지 → numpy (cv_bridge 불필요: rgb8 고정이므로 직접 변환)
# ----------------------------------------------------------------------------
def image_msg_to_rgb(msg) -> np.ndarray:
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unexpected encoding {msg.encoding}")
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.step != msg.width * 3:                      # row padding 대응
        arr = arr.reshape(msg.height, msg.step)[:, : msg.width * 3]
    img = arr.reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        img = img[..., ::-1]
    return np.ascontiguousarray(img)


# ----------------------------------------------------------------------------
# /tf_static 캐시 — SDK 는 TF 를 감싸지 않으므로 직접 구독 (api-reference "SDK 와 TF")
# ----------------------------------------------------------------------------
class StaticTFCache:
    def __init__(self, client: MARCClient):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from tf2_msgs.msg import TFMessage

        self._lock = threading.Lock()
        self._tf: Dict[str, object] = {}      # child_frame_id → TransformStamped (parent == world)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,     # latched 수신
                         history=HistoryPolicy.KEEP_LAST, depth=50)
        client.subscribe("/tf_static", TFMessage, self._cb, qos=qos)

    def _cb(self, msg):
        with self._lock:
            for t in msg.transforms:
                if t.header.frame_id == "world":
                    self._tf[t.child_frame_id] = t

    def get(self, frame: str):
        with self._lock:
            return self._tf.get(frame)

    def frames(self) -> List[str]:
        with self._lock:
            return list(self._tf)


# ----------------------------------------------------------------------------
# 에이전트
# ----------------------------------------------------------------------------
class Stage1Agent:
    def __init__(self, client: MARCClient, pipeline: Stage1Pipeline):
        self.client, self.pipe = client, pipeline
        self.tf: Optional[StaticTFCache] = None
        self._cam_cache: Dict[str, CameraModel] = {}
        self._busy = threading.Lock()

    # -- 센서 수집 --
    def _camera_model(self, cam_id: str) -> Optional[CameraModel]:
        if cam_id in self._cam_cache:
            return self._cam_cache[cam_id]
        info = self.client.get_cctv_info(cam_id)
        tfm = self.tf.get(cam_id) if self.tf else None
        gz = self.client.get_cctv_ground_height(cam_id)
        if info is None or tfm is None or gz is None:
            log.warning("cam %s incomplete: info=%s tf=%s ground=%s",
                        cam_id, info is not None, tfm is not None, gz)
            return None
        cm = camera_from_ros(cam_id, info, tfm, gz)
        self._cam_cache[cam_id] = cm
        log.info("cam %s ready: C=%s ground_z=%.3f", cam_id, np.round(cm.C_w, 2), cm.ground_z)
        return cm

    def collect_observations(self, wait_s: float = 2.0) -> List[CameraObservation]:
        obs: List[CameraObservation] = []
        deadline = time.time() + wait_s
        cam_ids = self.client.list_cctv()
        while True:
            missing = []
            for cid in cam_ids:
                if any(o.cam.camera_id == cid for o in obs):
                    continue
                cm = self._camera_model(cid)
                img = self.client.get_cctv_image(cid)
                if cm is None or img is None:
                    missing.append(cid)
                    continue
                obs.append(CameraObservation(cm, image_msg_to_rgb(img)))
            if not missing or time.time() > deadline:
                if missing:
                    log.warning("cameras without data: %s", missing)
                return obs
            time.sleep(0.1)

    # -- 콜백 --
    def on_mission(self, mission):
        if not self._busy.acquire(blocking=False):
            log.warning("mission arrived while busy; skipping round %s", mission.round)
            return
        try:
            t0 = time.time()
            log.info("=== round %s/%s  limit %.0fs  cmd=%r", mission.round, mission.total_rounds,
                     mission.time_limit, mission.voice_command)
            obs = self.collect_observations()
            if not obs:
                log.error("no camera observations; cannot submit")
                return
            payload = self.pipe.run(mission.voice_command, obs,
                                    time_limit_s=mission.time_limit - (time.time() - t0))
            seq = self.client.submit_grounding(GroundingResult.from_payload(payload))
            log.info("submitted seq=%s in %.1fs", seq, time.time() - t0)
        except Exception as e:                      # noqa: BLE001
            log.exception("round failed: %s", e)
        finally:
            self._busy.release()

    def on_score(self, score):
        log.info("score round=%s total=%.2f final=%s detail=%s",
                 score.round, score.total, score.is_final, score.scores)

    def on_warning(self, typ, msg):
        log.warning("platform warning [%s]: %s", typ, msg)

    def on_state_change(self, old, new):
        log.info("state %s → %s", old, new)


def main() -> int:
    # 1) 모델 먼저 로드 — connect 이후 첫 미션까지의 지연을 피한다
    pipeline = build_pipeline()
    # 2) SDK 연결
    client = MARCClient.from_env()
    agent = Stage1Agent(client, pipeline)
    client.on_mission(agent.on_mission)
    client.on_score(agent.on_score)
    client.on_warning(agent.on_warning)
    client.on_state_change(agent.on_state_change)
    if not client.connect(timeout=float(os.getenv("MARC_CONNECT_TIMEOUT", "120"))):
        log.error("registration failed (check MARC_TEAM_ID / MARC_TOKEN / ROS_DOMAIN_ID)")
        return 1
    # 3) TF 캐시 + 카메라 사전 준비 (latched 라 곧 도착)
    agent.tf = StaticTFCache(client)
    for _ in range(50):
        ids = client.list_cctv()
        if ids and all(agent.tf.get(i) is not None for i in ids):
            break
        time.sleep(0.2)
    log.info("cctv=%s  tf_frames=%s", client.list_cctv(), agent.tf.frames())
    for cid in client.list_cctv():
        agent._camera_model(cid)
    # 4) 대기
    try:
        client.run()
    finally:
        client.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
