"""Stage2 주행/집기에 쓰이는 **순수 기하 계산**. ROS·Isaac Sim·MARCClient 전부 불필요.

`demo/participant_app.py` 의 `_standoff_goal`(:248) / `_object_arm_xy`(:265) 는 계산 자체는
순수한데 내부에서 `self.client.get_world_pose()` 를 부르기 때문에 라이브가 아니면 단위
테스트를 할 수 없다. 여기서는 **pose 를 인자로 받도록** 같은 계산을 떼어놓아, 오프라인
하네스(`validation/stage2_offline.py`)가 정답 좌표로 검증할 수 있게 한다.

주의 -- 지금은 `participant_app.py` 와 **계산이 두 벌 존재한다.** 라이브 코드를 건드리지
않은 이유는 그쪽이 Isaac Sim 없이는 실행 검증이 불가능하기 때문이다(고쳐도 로컬에서
확인할 방법이 없다). 한쪽만 바꾸면 갈라지므로, 다음 데스크탑 세션에서 라이브 코드를
아래처럼 이 모듈로 돌리고 중복을 없애는 것을 권한다:

    from taejeong.stage2_geometry import standoff_goal, object_arm_xy
    ...
    def _standoff_goal(self, dest, standoff):
        pose = self.client.get_world_pose()
        return dest if pose is None else standoff_goal(pose[:2], dest, standoff)

상수는 `participant_app.py:66~74` 값을 그대로 옮긴 것이다.
"""

from __future__ import annotations

import math

# -- participant_app.py 와 같은 값 (거기 바뀌면 여기도 바꿔야 한다) --
TUMBLER_STANDOFF_M = 0.58    # 물체 앞 몇 m 에 차체를 세우나
ARM_BASE_FWD = 0.18          # car_base(world_pose) -> arm_base 전방 오프셋
GRASP_FWD_BACKOFF = -0.01    # 물체 중심 대비 파지점 오프셋(+fwd). 음수 = 살짝 더 들어감
GOAL_TOLERANCE = 0.7         # 도착 판정 거리 (m)
MAX_LINEAR_SPEED = 1.5       # cmd_vel 상한 (marc_sdk client.send_cmd_vel 문서값)


def standoff_goal(robot_xy, dest_xy, standoff: float = TUMBLER_STANDOFF_M):
    """robot->dest 직선 위에서 dest 앞 ``standoff`` m 지점.

    물체 자체가 아니라 그 앞을 목표로 삼아야 차체가 물체를 타고 넘지 않는다.
    이미 standoff 안쪽에 있으면 제자리를 반환한다(전진 금지).
    """
    x, y = float(robot_xy[0]), float(robot_xy[1])
    dx, dy = float(dest_xy[0]) - x, float(dest_xy[1]) - y
    d = math.hypot(dx, dy)
    if d <= standoff:
        return (x, y)
    f = (d - standoff) / d
    return (x + dx * f, y + dy * f)


def object_arm_xy(pose, obj_xy,
                  arm_base_fwd: float = ARM_BASE_FWD,
                  grasp_fwd_backoff: float = GRASP_FWD_BACKOFF):
    """물체의 world XY -> arm_base 프레임 (fwd, lat).

    world 오프셋을 car_base 프레임으로 회전시킨 뒤, car_base->arm_base 전방 오프셋만큼
    민다. 파지 목표를 이 좌표로 주면 주행 정렬이 조금 틀어져도 실제 물체를 겨눈다.

    pose: (x, y, yaw[rad]) -- 라이브에서는 client.get_world_pose() 가 주는 값.
    """
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    dx, dy = float(obj_xy[0]) - x, float(obj_xy[1]) - y
    cc, ss = math.cos(yaw), math.sin(yaw)
    fwd = dx * cc + dy * ss          # car_base 전방 (+X)
    lat = -dx * ss + dy * cc         # car_base 좌측 (+Y)
    return (fwd - arm_base_fwd - grasp_fwd_backoff, lat)


def arm_xy_to_world(pose, fwd: float, lat: float,
                    arm_base_fwd: float = ARM_BASE_FWD,
                    grasp_fwd_backoff: float = GRASP_FWD_BACKOFF):
    """``object_arm_xy`` 의 역변환. 왕복 검증용 (부호 실수를 잡는 게 목적)."""
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    f = fwd + arm_base_fwd + grasp_fwd_backoff
    cc, ss = math.cos(yaw), math.sin(yaw)
    return (x + f * cc - lat * ss, y + f * ss + lat * cc)


def yaw_towards(from_xy, to_xy) -> float:
    """from -> to 를 바라보는 yaw (rad). 정렬된 접근 자세를 만들 때 쓴다."""
    return math.atan2(float(to_xy[1]) - float(from_xy[1]),
                      float(to_xy[0]) - float(from_xy[0]))


def route_length(waypoints) -> float:
    """XY 경유점 리스트의 총 길이 (m)."""
    pts = [(float(p[0]), float(p[1])) for p in waypoints]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
