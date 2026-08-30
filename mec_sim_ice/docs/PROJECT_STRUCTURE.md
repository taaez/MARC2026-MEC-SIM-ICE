# marc-starter-kit 파일 구조

**MARC 2026 (MetaSejong AI Robot Challenge)** 참가자용 스타터 킷의 전체 구조 정리.
시뮬레이션 플랫폼(Docker), 참가자 SDK(`marc_sdk`), 실행 가능한 데모 에이전트, 매니퓰레이션(로봇 팔) 레퍼런스 코드로 구성됨.

- 개발자 가이드(정본 문서): https://marc-challenge.github.io/marc-dev-guide/
- 대회 홈페이지: https://marc-challenge.github.io/

> **리포 정리 (2026-08-26):** 대회측이 제공한 파일/폴더(아래 1~4절)는 원래 위치 그대로
> 두고, **우리 팀이 만든 코드/문서/기록은 전부 `mec_sim_ice/`로 모았다** (이 문서 자체도
> 그 안에 있다). 단 두 종류는 팀 것이지만 빌드·제출 구조상 루트 고정이 필요해 그대로
> 뒀다: 제출 진입점 `docker-compose.yml`/`docker/` (대회측이 clone 직후 그대로
> `docker compose up` 실행) 와 `problems.yaml` (organizer `simulation-platform/`가
> `../problems.yaml` 고정 경로로 마운트). `demo/`는 organizer 원본이지만
> `participant_app.py`/`Dockerfile`/`launch.sh`는 그 자리에서 직접 수정해 쓰는 파일이라
> 함께 남아 있다 — 팀이 새로 만든 대체 진입점(`vla_grounding_agent.py`,
> `participant_app_record.py` 등)만 `mec_sim_ice/demo_agents/`로 옮겼다.
> 자세한 팀 구조는 [`mec_sim_ice/` 절](#5-mec_sim_ice--팀-제작-코드문서기록) 참고.

## 최상위 구조

```
marc-starter-kit/
├── README.md                  # 전체 킷 개요 + Quick start        [대회측 원본]
├── NOTICES.md                 # 라이선스/저작권/실행환경(인터넷 정책) 고지  [대회측 원본]
├── problems.yaml               # 문제(미션) 정의 데이터           [팀 설정, 루트 고정 필요]
├── docker-compose.yml          # 팀 제출 진입점 (docker compose up)  [팀 제작, 루트 고정 필요]
├── docker/                     # 팀 제출 Dockerfile/entrypoint    [팀 제작, 루트 고정 필요]
├── simulation-platform/        # 시뮬레이터(Isaac Sim) Docker 레시피 + 실행 스크립트  [대회측 원본]
├── marc_sdk/                   # 참가자 SDK (ROS 2 통신 래퍼)          [대회측 원본]
├── demo/                       # 실행 가능한 레퍼런스(모의) 에이전트    [대회측 원본 + 팀이 그 자리에서 수정한 파일]
├── manipulation/                # 로봇 팔 기구학/pick-and-place 레퍼런스  [대회측 원본]
└── mec_sim_ice/                 # 팀이 만든 모든 것 (코드/문서/기록) — 아래 5절
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

## 5. `mec_sim_ice/` — 팀 제작 코드/문서/기록

대회측 원본(1~4절)과 구분해서 **우리 팀이 만든 것만** 모아 둔 폴더.

이 안에는 서로 다른 용도의 3갈래 진입점이 있다 (이전에 `taejeong`/`stage1`/`real_agent_vla`
3가지 파이프라인이 병렬로 얽혀 있던 걸 2026-08-31에 정리):

1. **메인 제출** — `vla_grounding/` 패키지 + `demo_agents/vla_grounding_agent.py`.
   `demo/participant_app.py`(대회측 원본, 팀이 그 자리에서 수정)가 기본으로 wiring하는
   경로이고, 루트 `docker-compose.yml`/`docker/`가 빌드하는 게 바로 이 경로다.
2. **record 전용** — `record_agent_vla.py` + `demo_agents/participant_app_record.py`
   (`RUNBOOK_record.md`). 메인과 같은 코드를 태우면 느려서 별도 파이프라인으로 유지한다
   (`MARC_ENTRYPOINT=participant_app_record.py`로 선택, `MARC_RECORD_DIR`로 세션 덤프).
   `records/`에 있는 기록은 이 경로로 수집됐다.
3. **로컬 오프라인 채점** — `score.py`(얇은 래퍼) → `vla_grounding/validation/offline_score.py`.
   `records/`를 재실행해 Stage1/2 점수를 로컬에서 재현한다 (인터넷·Isaac Sim 불필요).
   Stage2 채점은 `demo/participant_app.py`의 실제 주행/픽업 로직을 가짜 `MARCClient`
   (`vla_grounding/validation/fake_client.py`)로 그대로 실행한다.

DINO+Qwen2.5-VL 실험 파이프라인(`stage1/`)은 제출 경로에서 전혀 쓰이지 않아 삭제했다
(git 히스토리에는 남아 있음).

```
mec_sim_ice/
├── vla_grounding/       # ① 메인 제출 파이프라인 (텍스트 파싱 + 순수 기하, RUNBOOK.md 포함)
│   └── validation/      # ③ 오프라인 채점 로직 (offline_score.py, fake_client.py, runs/*.jsonl)
├── demo_agents/         # organizer demo/ 의 대체 진입점
│   ├── vla_grounding_agent.py    # ① 제출 진입점이 쓰는 에이전트
│   ├── participant_app_record.py # ② record 전용 진입점
│   └── weights/                  # best.pt, yolo11m-pose.pt
├── record_agent_vla.py    # ② record 전용 파이프라인 (RUNBOOK_record.md)
├── score.py                # ③ 오프라인 채점 실행기 (mec_sim_ice/score.py)
├── records/                  # ③ 입력 데이터 -- Isaac Sim 세션에서 수집한 라운드별 기록
├── training/                   # YOLO11 학습 코드 (det_classes.py, train_yolo11.py, small_object_detection/)
├── datasets/                    # 학습/소형객체 서브모델 데이터셋
├── tests/                         # 단위 테스트 (pytest, ROS 불필요)
├── tools/                          # 오프라인 재실행 스크립트 (run_offline_real_vla.py 등)
├── scripts/                         # 환경 점검/링크 진단/오프라인 검증 셸 스크립트
├── models/                           # (현재 비어 있음, .gitkeep)
├── problems.yaml.bak                  # problems.yaml 백업
└── *.md                                 # 설계/브리프/런북/세션 리포트 문서
```

각 서브디렉토리의 내부 구조는 대회측 원본과 별개로 팀이 자유롭게 정한 것이라 필요 시
계속 바뀔 수 있다. 대회측 원본(1~4절)을 수정해야 할 일이 생기면, 원본을 직접 고치지
말고 `mec_sim_ice/`에 별도 파일로 얹거나 이 문서에 그 사실을 기록해 둘 것.

## 기타

- **`problems.yaml`**: 대회 문제(미션) 정의 파일.
- **`NOTICES.md`**: 실행 환경 정책(런타임 인터넷 차단, 빌드 시점만 허용), Isaac Sim 라이선스/배포 정책, USD/3D 에셋 출처 및 라이선스(CC0/CC BY 등) 고지.

## 전체 흐름 요약

1. `simulation-platform/`으로 채점 플랫폼을 별도 머신에서 실행 (Isaac Sim 시뮬레이터 + 미션 로직)
2. 참가자는 별도 머신에서 `marc_sdk`를 사용해 자신의 에이전트(`demo/`가 그 예시)를 개발
3. 에이전트는 ROS 2(DDS, 같은 `ROS_DOMAIN_ID`)로 플랫폼과 통신: CCTV/센서 수신 → 인식(VLA) → 내비게이션 → `manipulation/` 로직으로 팔 제어 → 정답/완료 제출
4. Docker 이미지로 패키징해 제출 (`docker compose up`으로 채점)
