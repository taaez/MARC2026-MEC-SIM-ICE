# MARC2025(MetaSejong) vs MARC2026 — 지도(Map)/라이다(LiDAR) 제공 여부 비교

작성일: 2026-09-01

## 배경

"지금 대회(MARC2026)는 정적 전역 지도가 안 주어진다"는 전제로 라이다 기반 장애물 인식/회피
(`mec_sim_ice/demo_agents/lidar_obstacles.py`, `[[LidarObstacleLayer]]`)를 구현해왔는데, 작년
대회(MARC2025, 대회명은 동일하게 "MetaSejong AI Robot Challenge")는 실제로 어땠는지, 그리고
올해 공식 문서 기준으로 정말 지도가 안 주어지는 게 맞는지 확인한 결과를 정리한다.

**참고용으로 작년 참가팀 저장소를 클론해서 실제 구현을 분석했다**:
`https://github.com/DguAiCps/metasejong-airobotics.git` (로컬: `~/Desktop/2025-MARC/metasejong-airobotics`,
이 프로젝트 리포와는 별개, 순수 참고/비교 목적).

## 결론 먼저

**"지도가 없다"는 전제는 정정이 필요하다.** 2025년에도 2026년에도 정적 지도는 제공된다.
다만 그 지도가 **"주행 가능한 도로만" 담고, 그 위에 놓인 장애물(사람·차량·자전거·박스 등)은
포함하지 않는다** — 이게 라이다가 필요한 진짜 이유다. "지도 없음"이 아니라 "불완전한 지도 +
그 공백을 메우는 라이다"가 정확한 그림이다.

## MARC2025 (MetaSejong) — 공식 문서 + 실제 참가팀 코드 확인

출처: [MetaSejong AI Robotics Challenge API Reference](https://metasejong-competition.readthedocs.io/api-reference.html)

| 센서 | 제공 여부 | 토픽 / 타입 |
|---|---|---|
| 정적 지도 | ✅ 제공 | `/metasejong2025/map` (`nav_msgs/OccupancyGrid`) — "Map data for autonomous navigation provided via the ROS 2 Nav2 Map Server" |
| 라이다 | ✅ 제공 | `/metasejong2025/robot/scan` (`sensor_msgs/LaserScan`, **2D**) — "Sensor data collected from the LiDAR sensor mounted on the robot" |

**흥미로운 발견**: `DguAiCps/metasejong-airobotics` 팀의 실제 제출 코드(`robot_node.py`)를 전수
검색한 결과, `lidar`/`laser`/`scan`/`pointcloud` 관련 코드가 **단 한 줄도 없었다.** 즉 이 팀은
라이다가 없어서 못 쓴 게 아니라, 플랫폼이 제공한 라이다를 **아예 구독하지 않고** 정적 지도만
썼다. 그 결과 이 팀의 실시간 주행 로직(MPC, `mpc_controller.py`)은 "다른 감지된 물체"만
장애물로 인지하고, **지도에 없는 임시 장애물이나 벽을 실시간으로는 전혀 인지하지 못하는**
구조로 구현돼 있었다 (별도 분석 문서 참고 가치는 있으나 이 문서의 범위 밖).

## MARC2026 (이번 대회) — 공식 문서 확인

출처: [MARC 2026 Developer Guide API Reference](https://marc-challenge.github.io/marc-dev-guide/en/api-reference.html)

| 센서 | 제공 여부 | 토픽 / 타입 |
|---|---|---|
| 정적 지도 | ✅ 제공 | `get_occupancy_map()` / `/marc/env/map/occupancy` (`OccupancyGrid`) — **"the occupancy map contains drivable road only"** (사람·랜드마크 미포함, 문서에 명문화됨) |
| 라이다 | ✅ 제공 | `get_lidar()` / `/marc/{team_id}/robot/lidar/points` (`PointCloud2`, **3D**, Velodyne VLP-16) — 용도로 **"close-range obstacle avoidance"** 라고 명시 |

공식 문서의 "drivable road only" 표현은, 우리 팀이 `lidar_obstacles.py` 모듈 docstring에 스스로
적어둔 추정 — *"`get_occupancy_map()`은 정적 도로만 담는다. 주차 차량·자전거·사람은 없다
(데스크탑 실측: 로봇이 경로상의 차에 그대로 머리를 박는다)"* — 과 **정확히 일치**한다. 실측
기반 판단이 공식 스펙과 맞았다는 뜻이다.

## 연도 비교 요약

| 항목 | MARC2025 (MetaSejong) | MARC2026 |
|---|---|---|
| 정적 지도 제공 | ✅ (Nav2 Map Server 경유) | ✅ (`get_occupancy_map()`) |
| 지도 범위 | 명시적 설명 없음 | **도로만** (사람/랜드마크 제외, 문서 명시) |
| 라이다 제공 | ✅ 2D LaserScan | ✅ **3D** PointCloud2 (Velodyne VLP-16) |
| 라이다 용도 명시 | 없음 | **"close-range obstacle avoidance"** 명시 |
| 작년 대비 공식 변경사항(changelog) | 두 문서 어디에도 연도 간 직접 비교/changelog 없음 (아래 두 문서를 나란히 놓고 대조한 결과) | 〃 |

## 우리 프로젝트에 대한 함의

- 지도(A*, `OccupancyGridPlanner`)와 라이다(`LidarObstacleLayer`)를 **둘 다 쓰는 지금 구조가
  공식 스펙과 정확히 맞다.** "지도 없음"을 전제로 라이다만으로 새로 설계할 필요는 없다.
- 라이다가 채워야 할 정확한 공백은 **"도로 지도에는 없는, 그 위에 놓인 장애물"** 뿐이다 —
  이건 지금 구현(정체 기억 + 라이다 로컬 레이어가 A* 장애물 목록을 보강하는 구조)의 설계
  방향과 일치한다.
- 작년 참가팀(metasejong-airobotics)의 사례는 "라이다가 있어도 실시간 주행 로직에 연결을
  안 하면 정적 지도만으로 회피하다가 지도 밖 장애물에 그대로 부딪힌다"는 **반면교사**로
  참고할 만하다.

## 출처

- [MetaSejong AI Robotics Challenge — API Reference](https://metasejong-competition.readthedocs.io/api-reference.html)
- [MARC 2026 Developer Guide — API Reference](https://marc-challenge.github.io/marc-dev-guide/en/api-reference.html)
- [DguAiCps/metasejong-airobotics](https://github.com/DguAiCps/metasejong-airobotics) (2025년 참가팀 제출 코드, 참고용 클론 분석)
