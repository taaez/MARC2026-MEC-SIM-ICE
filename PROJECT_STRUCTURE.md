# marc-starter-kit 파일 구조

**MARC 2026 (MetaSejong AI Robot Challenge)** 참가자용 스타터 킷의 전체 구조 정리.
시뮬레이션 플랫폼(Docker), 참가자 SDK(`marc_sdk`), 실행 가능한 데모 에이전트, 매니퓰레이션(로봇 팔) 레퍼런스 코드로 구성됨.

- 개발자 가이드(정본 문서): https://marc-challenge.github.io/marc-dev-guide/
- 대회 홈페이지: https://marc-challenge.github.io/

## 최상위 구조

```
marc-starter-kit/
├── README.md                  # 전체 킷 개요 + Quick start
├── NOTICES.md                 # 라이선스/저작권/실행환경(인터넷 정책) 고지
├── problems.yaml               # 문제(미션) 정의 데이터
├── simulation-platform/        # 시뮬레이터(Isaac Sim) Docker 레시피 + 실행 스크립트
├── marc_sdk/                   # 참가자 SDK (ROS 2 통신 래퍼)
├── demo/                       # 실행 가능한 레퍼런스(모의) 에이전트
└── manipulation/                # 로봇 팔 기구학/pick-and-place 레퍼런스
```

## 1. `simulation-platform/` — 시뮬레이션 플랫폼

Isaac Sim 기반 대회 플랫폼을 **참가자가 직접 빌드**하는 Docker 레시피 모음. (Isaac Sim 바이너리 재배포 금지 → NGC 계정으로 직접 pull 후 로컬 빌드)

| 파일/폴더 | 설명 |
|---|---|
| `marc.sh` | 통합 실행 래퍼. `setup`(베이스 이미지 빌드) / `platform`(시뮬레이션 플랫폼 실행) / `manip-trainer`(매니퓰레이션 연습 환경) 등 서브커맨드 제공 |
| `Makefile` | 빌드/실행 단축 타깃 |
| `Dockerfile.base` | 공통 베이스 이미지 (Isaac Sim + 의존성) |
| `Dockerfile.practice` | 연습용(practice) 이미지 |
| `docker-compose.yml` / `docker-compose.gui.yml` | 플랫폼 실행 compose (GUI 유무) |
| `.env.example` | 환경 변수 템플릿 |
| `problems.yaml.example` | 문제 정의 템플릿 |
| `patches/` | Isaac Sim 부팅/오브젝트 배치 관련 런타임 패치 스크립트 (`boot.py`, lidar frame-id, object rotation/xformOps 패치) |

**하나의 통합 이미지**가 실행 시점에 선택되는 3가지 툴(시뮬레이션 플랫폼 / 데이터셋 생성기 / 매니퓰레이션 트레이너)을 모두 서빙함.

## 2. `marc_sdk/` — 참가자 SDK

플랫폼과의 ROS 2 통신(등록 핸드셰이크, 메시지 타입 구분, 세션/시퀀스 관리, QoS, CCTV 자동 탐색)을 대신 처리해주는 라이브러리. 참가자는 데코레이터로 이벤트 핸들러만 등록하면 됨. **pip 패키지가 아니라 소스로 배포**되며 `PYTHONPATH`에 추가해 사용.

| 파일 | 설명 |
|---|---|
| `__init__.py` | 패키지 진입점, `MARCClient` 등 공개 API export |
| `client.py` (860줄) | 핵심 클래스 `MARCClient` — 연결/등록, `@on_mission`/`@on_stage2_mission`/`@on_stage2_run` 등 콜백 데코레이터, `submit_grounding()`/`task_complete()` 등 제출 메서드, `send_cmd_vel()`/`send_arm_command()` 등 제어, `get_cctv_image()`/`get_odom()`/`get_lidar()`/`get_occupancy_map()`/`get_world_pose()` 등 센서 getter |
| `protocol.py` | 메시지 프로토콜 정의 (msg 201/211/301/302/311 등 타입 코드 및 스키마) |
| `types.py` | 데이터 클래스/타입 정의 (`GroundingResult` 등) |
| `auth.py` | 인증/토큰 처리 |

라이선스: MIT

## 3. `demo/` — 레퍼런스(모의) 에이전트

`marc_sdk` 위에 만든 **실행 가능한 베이스라인 에이전트**. 실제로 문제를 "푸는" 것이 아니라 사전 추출된 정답을 조회해 전체 흐름(등록→인식→내비게이션→매니퓰레이션→제출)을 시연함. **제출 전 mock 부분을 실제 인식/판단 로직으로 교체해야 함.**

| 파일 | 설명 |
|---|---|
| `participant_app.py` (443줄) | 엔트리 포인트. `MARCClient` 콜백을 등록하고 Stage 1/Stage 2 전체 흐름을 연결 |
| `mock_agent_vla.py` | 모의 인식(VLA). CCTV를 분석하는 대신 `mock_demo_data.yaml`에서 음성 명령에 대한 정답을 조회해 `GroundingResult` 반환 |
| `mock_agent_navigation.py` (484줄) | Stage 2 내비게이션: 목표 추적 제어 + occupancy-grid A* 경로계획 + 장애물 회피 |
| `mock_agent_manipulation.py` | 로봇 팔 pick 동작 (joint-angle keyframe 시퀀스) |
| `mock_demo_data.yaml` | 데모 전용 데이터 (문제별 정답 좌표 + 장애물). 실제 참가자는 갖지 못하는 데이터 |
| `arm_kin.py` / `arm_pick.py` | `manipulation/`과 동일한 기구학/pick 로직 사본 (데모에서 바로 실행되도록 포함) |
| `pnp_params.json` | pick 파라미터 튜닝값 |
| `urdf/gen3_6dof_vision_2f140.urdf` | 로봇 팔(Kinova Gen3 6DoF + Robotiq 2F-140) URDF 모델 |
| `Dockerfile` / `docker-compose.yml` | 표준 제출 형태(컨테이너 실행). 팀 ID/토큰을 compose 파일에 입력 |
| `launch.sh` | 호스트에서 직접 실행하는 개발용 스크립트 (ROS 2 Humble 환경 + PYTHONPATH 설정) |

실행 방법:
- **A. Docker (제출 표준)**: `docker-compose.yml`에 `MARC_TEAM_ID`/`MARC_TOKEN` 입력 후 `docker compose up`
- **B. 호스트 실행 (개발용)**: `MARC_TEAM_ID`/`MARC_TOKEN` 환경변수 설정 후 `./launch.sh u1`

## 4. `manipulation/` — 매니퓰레이션(로봇 팔) 레퍼런스

Joint-angle 제어만 허용되는 6축 로봇 팔로 pick-and-place를 풀기 위한 기구학 레퍼런스 (IK/모션 설계는 참가자 몫).

| 파일 | 설명 |
|---|---|
| `urdf/gen3_6dof_vision_2f140.urdf` | 로봇 기구학 모델 (6축 팔 + Robotiq 2F-140 그리퍼). 외부 메시 불필요, self-contained |
| `arm_kin.py` (388줄) | 기구학 엔진 (**placo**/Pinocchio 기반). URDF 로드 + FK/Jacobian + damped-least-squares(DLS) closed-loop IK 서보 |
| `arm_pick.py` (419줄) | 베이스라인 pick-and-place. waypoint 테이블(`pre→descend→grasp→lift→over→place→release→back`)을 `arm_kin`으로 서보. `grasp_xy`(물체 위치)를 입력받고 `pnp_params.json`으로 파라미터 튜닝 |
| `pnp_params.json` | grasp depth, carry/basket 높이, gripper close, wrist roll(`grasp_yaw`), 속도 등 튜닝값 |

두 가지 활용 방식:
- **최대 활용**: `arm_pick.py`를 그대로 쓰고 `grasp_xy`만 계산해서 전달 (부분 점수 획득 가능)
- **최소 활용**: `arm_kin`(FK/Jacobian/DLS)과 URDF만 재료로 가져가 직접 팔 제어 로직 설계

의존성: `placo` (pip install placo, Pinocchio 기반). 빌드 시점에 이미지에 baked-in 필요 (채점 런타임은 인터넷 미연결).

## 기타

- **`problems.yaml`**: 대회 문제(미션) 정의 파일.
- **`NOTICES.md`**: 실행 환경 정책(런타임 인터넷 차단, 빌드 시점만 허용), Isaac Sim 라이선스/배포 정책, USD/3D 에셋 출처 및 라이선스(CC0/CC BY 등) 고지.

## 전체 흐름 요약

1. `simulation-platform/`으로 채점 플랫폼을 별도 머신에서 실행 (Isaac Sim 시뮬레이터 + 미션 로직)
2. 참가자는 별도 머신에서 `marc_sdk`를 사용해 자신의 에이전트(`demo/`가 그 예시)를 개발
3. 에이전트는 ROS 2(DDS, 같은 `ROS_DOMAIN_ID`)로 플랫폼과 통신: CCTV/센서 수신 → 인식(VLA) → 내비게이션 → `manipulation/` 로직으로 팔 제어 → 정답/완료 제출
4. Docker 이미지로 패키징해 제출 (`docker compose up`으로 채점)
