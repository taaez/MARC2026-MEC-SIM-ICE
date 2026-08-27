"""
world_to_arm_base.py

Perception(CCTV/로봇 카메라)이 계산한 물체의 world 좌표를,
매니퓰레이션 파트(arm_pick.pick_and_place)가 요구하는
arm_base 프레임 기준 (fwd, lat) 좌표로 변환한다.

*** 중요 ***
gen3_6dof_vision_2f140.urdf(팔 자체 URDF)를 직접 확인해본 결과, 이 파일에는
"arm_base"라는 링크가 아예 없다 - base_link는 이 파일 안에서만 쓰는 팔의 로컬
원점이고, 로봇 몸체(모바일 베이스)에 팔이 실제로 어떤 위치·각도로 마운트됐는지는
이 파일에 없다(별도의 전체 로봇 조립 URDF에 있을 내용, 참가자에게 미공개).

매니퓰레이션 킷 README에 "로봇 전체(모바일 베이스+팔) USD는 플랫폼 안에만 있고
참가자에게 별도 URDF로 제공되지 않는다"고 명시되어 있어, 애초에 정적 파일로는
이 오프셋을 알아낼 방법이 없다 - 확인은 오직 런타임 tf2 조회로만 가능하다.

API 레퍼런스의 "TF / 좌표계" 절에 실제 TF 프레임으로 `arm_base_link`가
언급되어 있으므로(Base_LiDAR와 같은 예시로), 플랫폼이 켜져 있으면 아래처럼
직접 조회해서 실제 오프셋·회전을 눈으로 확인할 수 있다:

    ros2 run tf2_ros tf2_echo world arm_base_link

그래서 이 파일은 두 가지 버전을 함께 둔다.
  1) transform_via_tf2()  - 실제 배포에 쓸, 정확한 버전. tf2가 world -> arm_base_link
     변환(이동 + 회전)을 그대로 적용해주므로 팔이 몸체에 어떤 각도로 붙어있든 상관없다.
  2) world_to_arm_base()  - tf2 없이 순수 계산만으로 돌려보는 근사 버전. 로봇 yaw만
     반영하고 "arm_base가 몸체와 같은 방향으로, ARM_BASE_OFFSET만큼만 떨어져 붙어있다"고
     가정한다. 이 가정이 맞는지는 검증되지 않았으므로, 시뮬레이션에 접근하기 전까지
     로직만 미리 짜보는 용도로만 쓰고 실제 제출 코드는 1)번으로 바꿔야 한다.
"""

import numpy as np

# --- 아래 값은 검증되지 않은 가정이다. tf2_echo로 확인 후 반드시 갱신할 것. ---
ARM_BASE_OFFSET = (0.0, 0.0)  # (dx_fwd, dy_lat), base_link 기준 arm_base 오프셋 가정값


def world_to_arm_base(robot_pose, object_world_xy, arm_base_offset=ARM_BASE_OFFSET):
    """
    Args:
        robot_pose: (x, y, yaw_rad) - client.get_world_pose() 결과
        object_world_xy: (x, y) - 인식된 물체의 world 좌표
            (z는 보통 불필요 - arm_pick의 grasp_down 파라미터가 하강 깊이를 처리함)
        arm_base_offset: (dx_fwd, dy_lat) - base_link -> arm_base 오프셋

    Returns:
        (fwd, lat): arm_pick.pick_and_place(..., grasp_xy=(fwd, lat)) 에 그대로 넘길 값
    """
    rx, ry, ryaw = robot_pose
    ox, oy = object_world_xy

    # 1) world 프레임에서 로봇 -> 물체 벡터
    dx = ox - rx
    dy = oy - ry

    # 2) world -> robot(base_link) 프레임으로 회전
    c, s = np.cos(ryaw), np.sin(ryaw)
    fwd = c * dx + s * dy
    lat = -s * dx + c * dy

    # 3) base_link -> arm_base 오프셋 보정
    fwd -= arm_base_offset[0]
    lat -= arm_base_offset[1]

    return fwd, lat


def transform_via_tf2(tf_buffer, object_world_xyz, target_frame="arm_base_link",
                       source_frame="world", timeout_sec=1.0):
    """실제 배포용 - tf2가 world -> arm_base_link 변환(이동+회전)을 그대로 적용한다.

    ARM_BASE_OFFSET 같은 가정이 전혀 필요 없다 - 팔이 몸체에 어떤 각도로
    붙어있든 tf 트리에 발행된 실제 값을 그대로 쓰기 때문이다.

    Args:
        tf_buffer: client.node(공개 프로퍼티로 확인됨, marc_sdk/client.py 참조)에
                   tf2_ros.TransformListener를 붙여서 만든 tf2_ros.Buffer.
                   client.py에는 tf 관련 코드가 전혀 없으므로(구독/버퍼 둘 다 없음)
                   참가자가 직접 만들어야 한다 - 아래 사용 예시 참고.
        object_world_xyz: (x, y, z) - 물체의 world 좌표
        target_frame: 변환 대상 프레임 (기본 "arm_base_link")
        source_frame: 원본 좌표계 (기본 "world")

    Returns:
        (fwd, lat): arm_pick.pick_and_place(..., grasp_xy=(fwd, lat)) 에 넘길 값
    """
    import rclpy
    from geometry_msgs.msg import PointStamped
    import tf2_geometry_msgs  # noqa: F401  (PointStamped 변환 등록용, import만 하면 됨)

    pt = PointStamped()
    pt.header.frame_id = source_frame
    pt.point.x, pt.point.y, pt.point.z = [float(v) for v in object_world_xyz]

    transformed = tf_buffer.transform(
        pt, target_frame, timeout=rclpy.duration.Duration(seconds=timeout_sec)
    )
    return transformed.point.x, transformed.point.y


# ----- 실제 SDK와 연결하는 방법 (on_stage2_run 콜백 안 예시) -----
#
# --- 권장: tf2 버전 ---
#   from tf2_ros import Buffer, TransformListener
#   import arm_pick
#   from world_to_arm_base import transform_via_tf2
#
#   tf_buffer = Buffer()
#   tf_listener = TransformListener(tf_buffer, client.node)  # client.node는 공개 프로퍼티(확인됨)
#   fwd, lat = transform_via_tf2(tf_buffer, object_world_xyz)
#   arm_pick.pick_and_place(client, log, grasp_xy=(fwd, lat))
#
# --- 임시(오프라인 확인용): 수식 버전 ---
#   robot_pose = client.get_world_pose()          # (x, y, yaw)
#   fwd, lat = world_to_arm_base(robot_pose, object_world_xy)
#   arm_pick.pick_and_place(client, log, grasp_xy=(fwd, lat))


if __name__ == "__main__":
    # 시뮬레이션 없이 바로 돌려볼 수 있는 단위 테스트.

    # 케이스 1: 로봇이 원점에서 정면(yaw=0)을 보고 있고,
    #           물체가 로봇 정면 0.5m, 왼쪽 0.1m에 있는 경우.
    # world와 robot 프레임이 일치하므로 결과가 입력과 같아야 함.
    print(world_to_arm_base((0.0, 0.0, 0.0), (0.5, 0.1)))
    # -> (0.5, 0.1)

    # 케이스 2: 로봇이 90도(왼쪽으로) 회전해 있는 경우 (yaw = pi/2).
    # 같은 물체가 이제는 로봇 기준 정면 0.1m, 오른쪽 0.5m 위치가 되어야 함.
    print(world_to_arm_base((0.0, 0.0, np.pi / 2), (0.5, 0.1)))
    # -> (0.1, -0.5)