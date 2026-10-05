"""LiDAR slice accumulator -- every PointCloud2 slice since the last update, merged.

## 왜 필요한가 (라이브 실측 2026-10-05)
Isaac Sim 의 RTX 라이다는 한 바퀴를 메시지 하나로 내지 않는다. sim 시간 ~60 Hz 로 **방위각
~70도짜리 조각**을 따로 발행한다 (az [-40, 32], [-100, -28], [-158, -88], [88, 142] ...).
조각 스탬프는 수신 시점 /clock 보다 0.10-0.15s(sim) 뒤처진다(센서 파이프라인 지연, 점별
시간 필드는 없음 -- 조각 하나가 한 렌더 시점이라 조각 내부 skew 는 없다).
`client.get_lidar()` 는 마지막 메시지 하나만 돌려주므로, 2 틱(5 Hz)마다 그걸 읽던
`LidarObstacleLayer` 는 매번 임의의 조각 하나만 봤다 -- 전방 +-40도 원뿔이 담긴 스캔은
539개 중 166개(31%)였고, 조각 위상이 RTF 와 맞물려 천천히 미끄러지면서 rt01 leg1 에서는
**31.7초 동안 전방을 한 번도 못 봤다** (그 사이 1.08 m/s 로 주차 차량 옆 턱에 박음).

여기서는 토픽을 직접 구독해 조각을 전부 큐에 쌓고, 업데이트 때 꺼내서 하나의 스윕으로
합친다. 각 조각은 **자기 스탬프 시점의 로봇 자세**(틱마다 기록한 자세 이력을 보간)로 월드에
올린 뒤 **현재 센서 프레임**으로 내린다 -- 0.2s 동안 1.1 m/s 면 0.22 m 를 움직이므로
보정 없이 합치면 장애물이 그만큼 번진다. roll/pitch 는 레이어와 같이 무시한다.

## 기울기 보정 (IMU)
차체가 턱에 걸려 pitch -8~-10 도가 되면 센서 프레임의 지면이 그만큼 기울어 보이고, 레이어의
지면 추정(섹터별 직선, 반경 방향 기울기만 흡수)이 틀어진다. 그래서 각 조각을 자기 스탬프
시점의 IMU roll/pitch 로 수평화한다: p_level = Ry(pitch) Rx(roll) p_sensor (yaw 는 이어지는
월드 변환이 처리). 부호는 라이브 스캔으로 확정했다 (rt01 2026-10-05: pitch -10 도에서 지면
기울기 -8.7 -> 1.4 도, roll -19 도에서 18.9 -> -0.4 도; 반대 부호는 -18.7 도로 두 배가 된다).
평지에서 남는 ~1 도는 실제 노면 경사이고 레이어의 섹터 직선 적합이 흡수한다.

반환값은 `.xyz` 를 가진 객체라 `lidar_obstacles.lidar_xyz()` 의 지름길로 그대로 들어간다
(오프라인 하네스의 FakePointCloud2 와 같은 경로). 구독을 못 하는 클라이언트(오프라인
fake_client)에서는 `ok` 가 False 이고, 호출부는 예전처럼 `get_lidar()` 를 쓴다.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections import deque

import numpy as np

from lidar_obstacles import lidar_xyz

POSE_HISTORY = 200          # ticks of (sim_t, pose) kept for slice deskew (~20 s at 10 Hz)
MAX_QUEUED_SLICES = 120     # ~2 s of slices at 60 Hz -- older ones are dropped, not merged


def imu_roll_pitch(imu):
    """sensor_msgs/Imu orientation -> (roll, pitch) rad, or None."""
    if imu is None:
        return None
    try:
        q = imu.orientation
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    except Exception:
        return None
    if not all(map(math.isfinite, (x, y, z, w))) or (x * x + y * y + z * z + w * w) < 1e-6:
        return None
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def level_matrix(roll, pitch):
    """Ry(pitch) @ Rx(roll): sensor frame -> yaw-aligned horizontal frame."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return (ry @ rx).astype(np.float32)


class _Merged:
    """Duck-typed stand-in for PointCloud2 (lidar_xyz() reads `.xyz` first)."""

    def __init__(self, xyz):
        self.xyz = xyz


class LidarSliceAccumulator:
    def __init__(self, client, topic, msg_type, qos=None):
        self._lock = threading.Lock()
        self._slices = deque(maxlen=MAX_QUEUED_SLICES)
        self._recent_keys = deque(maxlen=8)
        self._pose_t = deque(maxlen=POSE_HISTORY)
        self._pose = deque(maxlen=POSE_HISTORY)
        self._rp = deque(maxlen=POSE_HISTORY)     # (roll, pitch) per tick, None = unknown
        self.level = True                          # IMU leveling on/off (A/B switch)
        self.ok = False
        self.slices_merged = 0      # cumulative, for logging
        try:
            client.subscribe(topic, msg_type, self._on_msg, qos)
            self.ok = True
        except Exception:
            self.ok = False

    def _on_msg(self, msg):
        try:
            st = msg.header.stamp
            stamp = float(st.sec) + float(st.nanosec) * 1e-9
        except Exception:
            stamp = None
        # The bridge republishes some slices with a NEW stamp but identical points (measured:
        # consecutive 20418 / 762-point slices 1/60 s apart). Merging both would count every
        # point twice and let a single stray return pass MIN_POINTS_PER_CELL, so dedupe on
        # content (size + head/tail bytes), not on the stamp.
        try:
            data = msg.data
            key = (int(msg.width), bytes(data[:96]), bytes(data[-96:]))
        except Exception:
            key = None
        with self._lock:
            if key is not None and key in self._recent_keys:
                return
            if key is not None:
                self._recent_keys.append(key)
            self._slices.append((stamp, msg))

    def record_pose(self, t, pose, imu=None):
        """Call once per nav tick with sim time + world pose (x, y, yaw) + the latest
        sensor_msgs/Imu (or None)."""
        if pose is None or t is None:
            return
        if self._pose_t and t <= self._pose_t[-1]:
            return
        self._pose_t.append(float(t))
        self._pose.append(tuple(float(v) for v in pose))
        self._rp.append(imu_roll_pitch(imu))

    def _rp_at(self, t):
        """(roll, pitch) of the tick nearest to t, or None."""
        if not self._pose_t:
            return None
        i = min(max(bisect.bisect_left(self._pose_t, t if t is not None else self._pose_t[-1]), 0),
                len(self._pose_t) - 1)
        return self._rp[i]

    def _pose_at(self, t, fallback):
        if t is None or not self._pose_t:
            return fallback
        ts = self._pose_t
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            return self._pose[0]
        if i >= len(ts):
            return self._pose[-1]
        t0, t1 = ts[i - 1], ts[i]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        p0, p1 = self._pose[i - 1], self._pose[i]
        dyaw = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
        return (p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]), p0[2] + a * dyaw)

    def take_merged(self, pose_now, mount_xy=(0.0, 0.0)):
        """All slices queued since the last call, deskewed into the current sensor frame.
        None when nothing arrived (caller falls back to get_lidar())."""
        with self._lock:
            items = list(self._slices)
            self._slices.clear()
        if not items or pose_now is None:
            return None
        mx, my = float(mount_xy[0]), float(mount_xy[1])
        xn, yn, yawn = pose_now
        cn, sn = math.cos(yawn), math.sin(yawn)
        out = []
        for stamp, msg in items:
            pts = lidar_xyz(msg)
            if pts is None or pts.shape[0] == 0:
                continue
            if self.level:
                rp = self._rp_at(stamp)
                if rp is not None:
                    pts = pts @ level_matrix(*rp).T
            x, y, yaw = self._pose_at(stamp, pose_now)
            c, s = math.cos(yaw), math.sin(yaw)
            # sensor(stamp) -> world
            px, py = pts[:, 0] + mx, pts[:, 1] + my
            wx = x + px * c - py * s
            wy = y + px * s + py * c
            # world -> sensor(now)
            dx, dy = wx - xn, wy - yn
            qx = dx * cn + dy * sn - mx
            qy = -dx * sn + dy * cn - my
            out.append(np.stack([qx, qy, pts[:, 2]], axis=1).astype(np.float32))
        if not out:
            return None
        self.slices_merged += len(out)
        return _Merged(np.concatenate(out))
