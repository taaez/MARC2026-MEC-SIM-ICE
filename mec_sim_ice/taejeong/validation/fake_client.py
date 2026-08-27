"""Isaac Sim / ROS 없이 `demo/participant_app.py` 의 Stage2 를 **원본 그대로** 돌리는 가짜 클라이언트.

## 왜 되나

`participant_app._drive()` 는 ROS 를 직접 부르지 않는다. 바깥과의 대화가 전부
`MARCClient` 메서드를 통하고, Stage2 가 실제로 쓰는 건 여덟 개뿐이다:

    now_s / sleep / is_running / get_world_pose / send_cmd_vel
    get_occupancy_map / send_arm_command / get_arm_state / task_complete

이걸 구현한 객체를 `MARCClient.from_env()` 자리에 끼워 넣으면 participant_app 은 자기가
가짜와 얘기하는지 모른다. `mock_agent_navigation`(A*·PathFollower)과
`arm_pick`/`arm_kin`(placo DLS 서보)은 **한 줄도 안 고치고** 그대로 돈다 --
FK/IK 는 진짜 placo + 진짜 gen3 URDF 계산이다.

## 무엇이 진짜고 무엇이 가짜인가

[A] 차체   cmd_vel -> 유니사이클 적분 + 가속제한 + Ackermann 무효명령 정지 규칙.
           => 제어 루프가 **실제로 돈다**. 도달 실패/진동/stall-replan/시간초과가 드러난다.
[B] 팔     관절 속도제한 추종 모델. FK/IK 는 진짜.
           => DLS 수렴, WP_TIMEOUT, 특이점, CART_LEASH 가 진짜로 검증된다.
[C] 파지   inner_finger_knuckle 을 "닫는 순간 EE 가 물체 반경 안이면 막힘" 으로 **가정**.
           => 여기가 가짜다. 접촉 솔버가 없으니 "손가락이 진짜 물체를 물었나" 는 모른다.
              대신 "명령 궤적이 물체 위치까지 EE 를 데려갔나" 라는 **기하 질문**에는 답한다
              (= stage2_offline.py [3] 이 열어둔 바로 그 질문).

## 여전히 확인 불가

접촉 물리(진짜 파지 성공), 접근 중 물체가 밀림/넘어짐, 차체 전복, 라이다 지역회피,
합성 점유지도의 실제 통행성, 플랫폼 Stage2 배점.

## 모델 상수는 전부 가정이다

아래 CHASSIS_* / ARM_* 값은 실측이 아니라 **문서값과 보수적 추정**이다. 실제 플랫폼
값과 다를 수 있으므로, 이 하네스의 결과는 "이 가정 아래에서 코드가 어떻게 굴러가나" 로
읽어야 한다. 절대 판정이 아니라 **회귀 검출과 실패 원인 분리**가 목적이다.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types

import numpy as np

_VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
_TAEJEONG_DIR = os.path.dirname(_VALIDATION_DIR)
_MEC_ROOT = os.path.dirname(_TAEJEONG_DIR)
_REPO_ROOT = os.path.dirname(_MEC_ROOT)
_DEMO_DIR = os.path.join(_REPO_ROOT, "demo")

# ---------------------------------------------------------------- 모델 상수 --
# 차체 -- marc_sdk.client.send_cmd_vel docstring 의 상한값 + 보수적 가속 추정.
CHASSIS_MAX_LINEAR = 1.5      # m/s   (SDK 문서값)
CHASSIS_MAX_ANGULAR = 1.5     # rad/s (SDK 문서값)
CHASSIS_ACCEL = 2.0           # m/s^2   <- 추정. 명령 급변 시 잔여속도(오버슛)를 만든다
CHASSIS_ANG_ACCEL = 3.0       # rad/s^2 <- 추정
# 4륜조향 트랙폭. mock_agent_navigation.GoalSeekController.STEERING_TRACK 과 같은 값.
# 2v <= w*T 인 명령은 이 차체에서 "너무 급한 선회" 라 정지한다(주석 근거: 같은 파일 51행).
CHASSIS_STEERING_TRACK = 0.38

# 팔 -- Isaac 의 position drive 추종 속도. 실측 불가라 보수적 추정.
ARM_MAX_JOINT_RATE = 2.0      # rad/s
# 그리퍼: outer_knuckle 은 힘구동이라 명령각에 그대로 물린다(arm_pick.py:206 주석).
# inner_finger_knuckle 은 수동관절 -- 빈손이면 grip_close(0.58) 에서 35deg 까지 닫힌다.
INNER_FK_FREE_AT_CLOSE_RAD = math.radians(35.0)   # arm_pick.INNER_FK_CLOSED_DEG
GRIP_CLOSE_REF = 0.58                             # 그 35deg 가 관측된 grip 명령값
# 물체가 손가락을 막았을 때 도달하는 각도. arm_pick 의 판정 임계는 34deg 이므로 그보다 낮게.
INNER_FK_BLOCKED_RAD = math.radians(24.0)
# [C] 파지 포획 반경 -- 닫는 순간 EE 와 물체의 arm_base 수평거리가 이 안이면 "물었다".
# robotiq 140 개폐폭(0.14m) 의 절반에서 물체 반경을 뺀 값 근처. **가정치**.
GRASP_CAPTURE_RADIUS_M = 0.06


# ------------------------------------------------------------------ shim --

class JointState:
    """sensor_msgs/JointState 중 arm_pick / arm_kin 이 실제로 읽는 필드만."""

    def __init__(self):
        self.name = []
        self.position = []


def _install_module(name: str, mod: types.ModuleType):
    sys.modules[name] = mod
    return mod


def _shim_sensor_msgs():
    """`from sensor_msgs.msg import JointState` 만 통과시키는 최소 shim."""
    if "sensor_msgs.msg" in sys.modules:
        return
    pkg = types.ModuleType("sensor_msgs")
    msg = types.ModuleType("sensor_msgs.msg")
    msg.JointState = JointState
    pkg.msg = msg
    _install_module("sensor_msgs", pkg)
    _install_module("sensor_msgs.msg", msg)


def _load_py39_safe(name: str, path: str):
    """`str | None` 어노테이션(3.10 문법) 때문에 3.9 에서 import 가 깨지는 모듈을 로드.

    `from __future__ import annotations` 를 앞에 붙여 어노테이션 평가를 미룬다.
    stage2_offline.load_nav_module() 과 같은 트릭이고, 원본 파일은 건드리지 않는다.
    """
    src = open(path, encoding="utf-8").read()
    mod = types.ModuleType(name)
    mod.__file__ = path
    exec(compile("from __future__ import annotations\n" + src, path, "exec"), mod.__dict__)
    return _install_module(name, mod)


def _load_marc_types():
    """marc_sdk/types.py 만 파일 경로로 직접 로드 (패키지 __init__ 은 rclpy 를 끈다)."""
    path = os.path.join(_REPO_ROOT, "marc_sdk", "types.py")
    spec = importlib.util.spec_from_file_location("marc_sdk_types_offline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubVLA:
    """TaejeongVLAGrounding 대역. grounding 은 offline_score.py 가 이미 채점하므로,
    여기서는 Stage2 주행/집기만 보려고 정답(또는 지정값)을 그대로 돌려준다."""

    stage2_answer = None      # 하네스가 GroundingResult 를 넣어준다

    def __init__(self, client=None):
        self.client = client

    def process(self, command, cameras=None):
        return self.stage2_answer

    def process_stage2(self, task_description):
        return self.stage2_answer


_CURRENT_CLIENT = None


def install_shims():
    """participant_app 을 import 하기 **전에** 부른다. 반환: (types 모듈, nav 모듈).

    - sensor_msgs        : JointState shim
    - marc_sdk           : MARCClient -> 가짜. from_env() 는 현재 가짜 인스턴스를 준다
    - mock_agent_navigation : 3.9 안전 로드 (participant_app 이 from-import 하기 전에 선점)
    - taejeong_agent_vla : 스텁
    """
    _shim_sensor_msgs()
    for p in (_DEMO_DIR, os.path.join(_REPO_ROOT, "manipulation")):
        if p not in sys.path:
            sys.path.insert(0, p)

    marc_types = _load_marc_types()

    if "marc_sdk" not in sys.modules:
        pkg = types.ModuleType("marc_sdk")

        class _MARCClientProxy:
            @classmethod
            def from_env(cls, **kwargs):
                if _CURRENT_CLIENT is None:
                    raise RuntimeError("set_current_client() 를 먼저 불러야 한다")
                return _CURRENT_CLIENT

        pkg.MARCClient = _MARCClientProxy
        pkg.types = marc_types
        for n in ("GroundingResult", "Mission", "Stage2Mission", "Stage2Reveal", "Score"):
            setattr(pkg, n, getattr(marc_types, n))
        _install_module("marc_sdk", pkg)
        _install_module("marc_sdk.types", marc_types)

    nav = sys.modules.get("mock_agent_navigation")
    if nav is None:
        nav = _load_py39_safe("mock_agent_navigation",
                              os.path.join(_DEMO_DIR, "mock_agent_navigation.py"))

    if "taejeong_agent_vla" not in sys.modules:
        m = types.ModuleType("taejeong_agent_vla")
        m.TaejeongVLAGrounding = _StubVLA
        _install_module("taejeong_agent_vla", m)

    return marc_types, nav


def set_current_client(client):
    """`MARCClient.from_env()` 가 돌려줄 인스턴스를 지정한다."""
    global _CURRENT_CLIENT
    _CURRENT_CLIENT = client


def patch_sim_clock(client):
    """arm_kin / arm_pick 이 보는 `time` 을 시뮬 시계로 갈아끼운다.

    arm_kin.tick() 은 dwell 만료와 웨이포인트 타임아웃(WP_TIMEOUT=15s)을
    `time.monotonic()` 으로 판정한다. 시뮬 시계로 도는 하네스에서는 벽시계가 거의 안
    흐르므로 그대로 두면 첫 dwell 에서 영원히 안 빠져나온다. 원본을 고치는 대신
    모듈이 참조하는 `time` 만 바꾼다.
    """
    import time as _real_time

    class _SimTime:
        def monotonic(self):
            return client.now_s()

        def time(self):
            return client.now_s()

        def sleep(self, s):
            client.sleep(s)

        def __getattr__(self, k):
            return getattr(_real_time, k)

    st = _SimTime()
    for name in ("arm_kin", "arm_pick"):
        mod = sys.modules.get(name)
        if mod is not None:
            mod.time = st
    return st


# ------------------------------------------------------------------ 시뮬 --

class Chassis:
    """[A] cmd_vel -> world pose. 유니사이클 + 가속제한 + Ackermann 무효명령 정지.

    무효명령 규칙: `2v <= w*T` 인 명령은 이 4륜조향 차체에서 "속도에 비해 너무 급한
    선회" 라 실제로 멈춘다(mock_agent_navigation.py:51 주석). 컨트롤러가 그런 명령을
    내면 여기서도 안 움직여야 버그가 드러난다. 단 v~0 인 순수 회전(X자 조향 제자리
    선회)은 허용된다 -- 같은 주석의 근거.
    """

    def __init__(self, x, y, yaw, stall_rule=True):
        # stall_rule=False 로 두면 무효명령 규칙을 끈다 -- 이 규칙이 결론을 만들어낸 게
        # 아닌지 확인하는 대조군용(--no-stall-rule).
        self.stall_rule = bool(stall_rule)
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.v = self.w = 0.0            # 실제 속도 (가속제한으로 명령을 따라간다)
        self.cmd_v = self.cmd_w = 0.0
        self.stall_ticks = 0             # 무효명령으로 정지한 스텝 수
        self.dist_travelled = 0.0
        self.clamped = 0                 # 상한을 넘겨 잘린 명령 수

    def command(self, v, w):
        cv = max(-CHASSIS_MAX_LINEAR, min(CHASSIS_MAX_LINEAR, float(v)))
        cw = max(-CHASSIS_MAX_ANGULAR, min(CHASSIS_MAX_ANGULAR, float(w)))
        if abs(cv - float(v)) > 1e-9 or abs(cw - float(w)) > 1e-9:
            self.clamped += 1
        self.cmd_v, self.cmd_w = cv, cw

    def step(self, dt):
        # 가속 제한 -- 명령이 0 으로 떨어져도 속도가 즉시 0 이 되지 않는다(잔여속도).
        dv = self.cmd_v - self.v
        self.v += max(-CHASSIS_ACCEL * dt, min(CHASSIS_ACCEL * dt, dv))
        dw = self.cmd_w - self.w
        self.w += max(-CHASSIS_ANG_ACCEL * dt, min(CHASSIS_ANG_ACCEL * dt, dw))

        if abs(self.v) < 1e-3:
            self.yaw += self.w * dt      # 제자리 선회 (X자 조향)
            return
        if self.stall_rule and 2.0 * abs(self.v) <= abs(self.w) * CHASSIS_STEERING_TRACK:
            self.stall_ticks += 1        # 무효 명령 -> 차체 정지
            return
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt
        self.dist_travelled += abs(self.v) * dt

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)


class ArmSim:
    """[B] 관절 속도제한 추종 + [C] 가정 기반 파지 판정.

    joint_1..6 은 명령각을 ARM_MAX_JOINT_RATE 로 따라간다. 이 지연이 arm_kin 의
    CART_LEASH(되먹임 페이스 조절)를 진짜로 작동시킨다.

    파지 판정([C], 가정): grip 명령이 닫힘으로 넘어가는 순간 EE(placo FK) 와 물체의
    arm_base **수평거리**가 GRASP_CAPTURE_RADIUS_M 안이면 "물었다" 로 latch 하고
    inner_finger_knuckle 을 임계 아래로 고정한다. 수직(z) 정렬은 오프라인에서
    검증 불가라 판정에 넣지 않는다 -- ground_z 가 실제 지형고가 아니어서 물체의
    arm_base z 를 유도할 수 없기 때문이다(stage2_offline.py [3](c) 참고).
    """

    ARM_NAMES = ["joint_%d" % i for i in range(1, 7)]
    GRIP_NAMES = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
    FK_NAMES = ["left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"]

    def __init__(self, home_q, kin=None, capture_radius=GRASP_CAPTURE_RADIUS_M):
        self.q = np.array(home_q, dtype=np.float64)
        self.cmd_q = self.q.copy()
        self.grip_cmd = 0.0
        self.kin = kin                    # placo FK (arm_base 프레임 EE 위치)
        self.capture_radius = float(capture_radius)
        self.holding = False              # 지금 그리퍼가 물체를 물고 있나
        self.captured = False             # 한 번이라도 물었나 (latch -- 놓아도 안 풀린다)
        self.in_basket = False            # 물고 있다가 열었다 = 바구니에 놓았다
        self.object_arm_xy = None         # 물체의 현재 arm_base (fwd, lat). 하네스가 갱신
        self.closed_before = False
        # 진단용 기록
        self.grasp_attempt_err = None     # 닫는 순간 EE-물체 수평거리 (m)
        self.grasp_attempt_ee = None      # 그때 EE 의 arm_base 좌표
        self.max_joint_lag = 0.0          # 명령-실제 최대 괴리 (rad)

    def command(self, names, positions):
        by = dict(zip(names, positions))
        for i, n in enumerate(self.ARM_NAMES):
            if n in by:
                self.cmd_q[i] = float(by[n])
        for n in self.GRIP_NAMES:
            if n in by:
                self.grip_cmd = float(by[n])
                break

    def step(self, dt):
        d = self.cmd_q - self.q
        lim = ARM_MAX_JOINT_RATE * dt
        self.q = self.q + np.clip(d, -lim, lim)
        self.max_joint_lag = max(self.max_joint_lag,
                                 float(np.max(np.abs(self.cmd_q - self.q))))

        closing = self.grip_cmd > 0.3 * GRIP_CLOSE_REF
        if closing and not self.closed_before:
            self.closed_before = True
            self._try_capture()
        elif not closing:
            if self.holding:
                # 물고 있다가 열었다 -> 배치 완료. 이 시퀀스에서 열리는 지점은
                # 'release'(바구니 위) 하나뿐이므로 바구니에 넣은 것으로 본다.
                self.in_basket = True
            self.closed_before = False
            self.holding = False

    def _try_capture(self):
        """[C] 가정 -- 닫는 순간의 EE-물체 수평거리로 파지 성패를 정한다."""
        if self.kin is None or self.object_arm_xy is None:
            return
        ee = self.kin.fk(self.q)[:3, 3]
        err = math.hypot(float(ee[0]) - self.object_arm_xy[0],
                         float(ee[1]) - self.object_arm_xy[1])
        self.grasp_attempt_err = err
        self.grasp_attempt_ee = (float(ee[0]), float(ee[1]), float(ee[2]))
        self.holding = err <= self.capture_radius
        self.captured = self.captured or self.holding

    def joint_state(self):
        js = JointState()
        # inner_finger_knuckle(수동관절): 빈손이면 grip 에 비례해 35deg 까지, 막히면 고정.
        free = self.grip_cmd / GRIP_CLOSE_REF * INNER_FK_FREE_AT_CLOSE_RAD
        fk = min(free, INNER_FK_BLOCKED_RAD) if self.holding else free
        js.name = list(self.ARM_NAMES) + list(self.GRIP_NAMES) + list(self.FK_NAMES)
        js.position = ([float(v) for v in self.q]
                       + [self.grip_cmd] * 2      # 힘구동 -> 명령각에 그대로 물린다
                       + [float(fk)] * 2)
        return js


class FakeMARCClient:
    """participant_app 이 보는 MARCClient 의 Stage2 부분 전체.

    시뮬레이션은 `sleep()` 안에서만 전진한다 -- 앱이 시계를 요구할 때만 시간이 흐르므로
    단일 스레드로 결정론적이고, 벽시계보다 훨씬 빠르다.
    """

    SUBSTEP = 0.02      # 내부 적분 스텝 (s). sleep 요청을 이 단위로 쪼갠다

    def __init__(self, start_pose, occupancy, object_xyz, home_q, kin=None,
                 capture_radius=GRASP_CAPTURE_RADIUS_M, arm_base_fwd=0.18,
                 sim_time_cap=1800.0, stall_rule=True):
        self.chassis = Chassis(*start_pose, stall_rule=stall_rule)
        self.arm = ArmSim(home_q, kin=kin, capture_radius=capture_radius)
        self._occ = occupancy
        self.object_xyz = [float(v) for v in object_xyz]
        self.arm_base_fwd = float(arm_base_fwd)
        self._t = 0.0
        # 폭주 방지 -- 시뮬 시간이 이만큼 흐르면 is_running 을 내려 모든 루프를 빠져나오게
        # 한다. 앱 자신의 STAGE2_NAV_TIMEOUT_S 와 별개인 하네스 쪽 안전장치다.
        self._sim_time_cap = float(sim_time_cap)
        self.hit_time_cap = False
        self._running = True
        self.task_completed = False
        # 핸들러 (participant_app._register_handlers 가 채운다)
        self._h = {}
        self.cmd_vel_calls = 0
        self.arm_cmd_calls = 0
        self.log_lines = []

    # -- 핸들러 등록 (participant_app 이 부른다) --
    def on_mission(self, fn): self._h["mission"] = fn
    def on_stage2_mission(self, fn): self._h["s2_mission"] = fn
    def on_stage2_reveal(self, fn): self._h["s2_reveal"] = fn
    def on_stage2_run(self, fn): self._h["s2_run"] = fn
    def on_score(self, fn): self._h["score"] = fn
    def on_state_change(self, fn): self._h["state"] = fn
    def on_time_expired(self, fn): self._h["expired"] = fn
    def on_time_remaining(self, fn): self._h["remaining"] = fn
    def on_transition(self, fn): self._h["transition"] = fn
    def on_warning(self, fn): self._h["warning"] = fn

    # -- 시계 --
    def now_s(self):
        return self._t

    def sleep(self, seconds):
        """시뮬 시간을 `seconds` 만큼 전진시키며 그동안 물리를 적분한다."""
        if seconds <= 0:
            return
        remaining = float(seconds)
        while remaining > 1e-9 and self._running:
            dt = min(self.SUBSTEP, remaining)
            self.chassis.step(dt)
            self.arm.step(dt)
            self.arm.object_arm_xy = self._object_arm_xy()
            self._t += dt
            remaining -= dt
            if self._t >= self._sim_time_cap:
                self.hit_time_cap = True
                self._running = False

    # -- 상태 --
    @property
    def is_running(self):
        return self._running

    @property
    def state(self):
        return "STAGE2_RUN" if self._running else "FINISHED"

    @property
    def registered(self):
        return True

    # -- 로봇 제어 --
    def send_cmd_vel(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, twist=None):
        self.cmd_vel_calls += 1
        self.chassis.command(linear_x, angular_z)

    def stop(self):
        self.chassis.command(0.0, 0.0)

    def send_arm_command(self, joint_state):
        self.arm_cmd_calls += 1
        self.arm.command(list(joint_state.name), list(joint_state.position))

    # -- 센서 --
    def get_world_pose(self):
        return self.chassis.pose

    def get_occupancy_map(self):
        return self._occ

    def get_arm_state(self):
        return self.arm.joint_state()

    def is_grasping(self):
        return bool(self.arm.holding)

    def is_basket_occupied(self):
        return bool(self.arm.in_basket)

    def get_odom(self): return None
    def get_imu(self): return None
    def get_lidar(self): return None
    def get_robot_image(self, which="base_left"): return None
    def get_robot_depth(self, which="base"): return None
    def list_cctv(self): return []
    def get_cctv_image(self, camera_id): return None
    def get_cctv_info(self, camera_id): return None
    def get_cctv_ground_height(self, camera_id): return None

    # -- 제출 --
    def submit_grounding(self, result): return 0
    def submit_stage2_grounding(self, result): return 0

    def task_complete(self):
        self.task_completed = True
        self._running = False
        return 0

    def connect(self, timeout=30.0, register_period=2.0):
        return True

    def run(self):
        pass

    def shutdown(self):
        self._running = False

    # -- 하네스가 쓰는 부분 --
    def _object_arm_xy(self):
        """물체 world XY -> arm_base (fwd, lat). participant_app._object_arm_xy 와 같은 식.

        ArmSim 의 파지 판정이 쓸 "물체의 현재 arm_base 좌표" 다. 여기서는 GRASP_FWD_BACKOFF
        를 빼지 않는다 -- 그건 앱이 겨누는 **목표점** 보정이고, 우리가 필요한 건 물체의
        **실제** 위치이기 때문이다.
        """
        x, y, yaw = self.chassis.pose
        dx, dy = self.object_xyz[0] - x, self.object_xyz[1] - y
        cc, ss = math.cos(yaw), math.sin(yaw)
        return (dx * cc + dy * ss - self.arm_base_fwd, -dx * ss + dy * cc)

    def fire_stage2_mission(self, mission):
        fn = self._h.get("s2_mission")
        if fn:
            fn(mission)

    def fire_stage2_reveal(self, reveal):
        fn = self._h.get("s2_reveal")
        if fn:
            fn(reveal)

    def run_stage2(self):
        fn = self._h.get("s2_run")
        if fn is None:
            raise RuntimeError("on_stage2_run 핸들러가 등록되지 않았다")
        fn()
