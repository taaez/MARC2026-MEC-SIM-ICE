# Stage 2 기록 세션 진행 보고 (2026-08-22)

`claude_code_prompt.md` 2단계("연구실 데스크탑에서 첫 기록 세션") 작업 기록. 목표는 `agent_vla.py`
(Grounding DINO + Qwen2.5-VL, VLM은 끔) 파이프라인으로 CCTV 기록 세션 한 번을 완주하는 것이었으나,
**아직 완주하지 못했다.** 아래에 진행 상황과 막힌 지점을 정리한다.

## 1. 목표

- `problems.yaml`에서 `lf01_mug_table`, `lf02_tumbler_bench`, `sar01_trash_sit` 3문항만 켜고
- `MARC_DISABLE_VLM=1`로 (VLM 없이 탐지기만) 실제 접속해서 3라운드를 돌리고
- `./records/`에 라운드별 이미지+메타데이터가 쌓이는 것을 확인하는 것

## 2. 사전 점검 (환경 검증)

| 항목 | 결과 |
|---|---|
| GPU | RTX 5060 Ti, VRAM 8151MiB. 세션 시작 시 팀원의 `dataset-gen` 작업(GT 데이터셋 생성) 중이라 대기 → 팀원 작업 종료 확인 후 진행 |
| 0/1단계 코드 | `demo/participant_app.py`에 `self.vla`, `set_time_limit`, `_on_score` 등 이미 반영 확인 |
| `problems.yaml` | lf01/lf02/sar01만 `enabled: true`로 수정, 원본은 `problems.yaml.bak`으로 백업 |
| 호스트 ROS2 | 원래 미설치 상태였음 → `ros-humble-ros-base`, `ros-humble-tf2-ros`를 사용자가 sudo apt로 설치 |
| 호스트 Python ML 스택 | `torch`(2.13+cu130), `torchvision`, `transformers==4.53.2`, `accelerate`, `huggingface_hub`, `qwen-vl-utils`, `opencv-python-headless`, `placo`를 시스템 파이썬에 `pip3 install --user`로 설치 |
| Grounding DINO 로딩 단독 테스트 | 성공 (`IDEA-Research/grounding-dino-base`, 최초 다운로드 포함 182초, `device=cuda`) |

## 3. 발견한 이슈와 대응

### 3-1. `scripts/diag_link.sh`가 ROS2 `setup.bash`와 `set -u` 충돌로 즉시 종료
- 원인: ROS2 `setup.bash`/`local_setup.sh`가 `set -u`에서 걸리는 미정의 변수를 다수 참조함(잘 알려진 ROS2 이슈).
- 조치: `source` 구간만 `set +u` / `set -u`로 감싸서 우회. 진단 로직 자체는 손대지 않음.
- 결과: 4가지 필수 항목(`/marc/ops/announce`, CCTV 이미지, `/tf_static`, `/clock`) 전부 정상 통과 확인 (Isaac Sim GUI에서 **Play**를 눌러야 이 토픽들이 살아난다는 것도 확인함 — OmniGraph 카메라/시계/TF 노드가 타임라인 재생에 물려있는 것으로 보임).

### 3-2. `demo/launch.sh`가 우리가 통합한 파이프라인을 실행하지 않음
- `launch.sh`는 `participant_app.py`(팀원의 `real_agent_vla.py`/YOLO11 배선)를 하드코딩 실행.
- 이번 세션에서 만든 `agent_vla.py`(Grounding DINO+Qwen, `stage1/` 사용)는 별도 진입점
  `demo/participant_app_agent_vla.py`에 연결돼 있음 — `demo/participant_app.patch` 주석에
  "두 Stage1 구현을 나란히 실험하기 위한 의도적 분리"라고 명시돼 있어 실수가 아님.
- 조치: `launch.sh` 마지막 줄을 `exec "${RUN_PY}" "${MARC_ENTRYPOINT:-participant_app.py}"`로
  변경. 기본값은 그대로 `participant_app.py`(팀원 워크플로 영향 없음), `MARC_ENTRYPOINT=participant_app_agent_vla.py`로
  우리 파이프라인을 선택 실행 가능하게 함.

### 3-3. Round 1 타임아웃 (핵심 미해결 문제)
**1차 시도**: 접속 후 Round 1(`lf01_mug_table`, 30초 제한)에서 클라이언트가 멈춤 → 서버가
Round 1/2/3 전부 타임아웃(0점) 처리 → Stage2(`rt01_parking_tumbler`)로 자동 진행.

- 스레드 상태 확인(`/proc/<pid>/task/*/wchan`): 메인 스레드 `do_select`, 3개 스레드
  `__skb_wait_for_more_packets`(소켓에서 데이터 대기) — 네트워크 I/O에 멈춰 있는 패턴.
- 가설: `GroundingDinoDetector`가 `from_pretrained()` 호출 시 캐시가 있어도 HuggingFace Hub에
  온라인 버전 확인을 시도하다 느린 네트워크에 멈춤. → `HF_HUB_OFFLINE=1` 추가하기로 함.
- 멈춘 프로세스를 `kill -9`로 강제 종료 → **서버 쪽에 정상 종료 신호가 전달되지 않아 세션 상태가
  꼬임** (같은 team_id로 재등록해도 Auto-plan이 처음부터 다시 시작되지 않고, 죽기 전 멈춰있던
  Stage2 지점에서 계속 "Stage 2 recovery: 0/80"만 반복).
- 사용자 승인 하에 **1회 한정으로 `simulation-platform`을 재기동**(원래는 절대 금지 규칙이나
  이번 건 예외 승인받음). 재기동 중 `marc.sh down`이 platform뿐 아니라 dataset-gen 프로파일까지
  같이 내려서, 그 사이 팀원이 다시 띄워둔 dataset-gen 컨테이너도 함께 종료됨 — 사용자 확인 결과
  인지하고 있던 상황이라 문제 없음으로 확인됨.
- 재기동 후 Play 재클릭 → `diag_link.sh` 재검증 통과 → **`HF_HUB_OFFLINE=1` 추가해서 2차 시도**.

**2차 시도 (`HF_HUB_OFFLINE=1` 적용 후)**: 환경변수가 프로세스에 정상 전달된 것을 `/proc/<pid>/environ`으로
확인했음에도 **동일하게 Round 1에서 멈춤** → Round 1/2/3 다시 전부 타임아웃(0점) → Stage2 재진입.
즉 `HF_HUB_OFFLINE` 가설은 **틀렸거나 부분적 원인**이었던 것으로 보인다.

- `demo/agent_vla.py` 코드 확인 결과, 모델 로드(`build_pipeline`)는 `VLAGrounding.__init__`
  (즉 `connect()` 이전, 라운드 시작 전)에서 이미 끝나는 구조 — 라운드 안에서 모델을 처음 로드하는
  게 아님을 코드로 확인함.
- `_observe()` 메서드는 sim-시간 기준 최대 3초로 명시적으로 시간 제한이 걸려 있어, 여기서
  무한정 멈추는 구조가 아님(코드 확인 완료).
- 따라서 실제 hang 지점은 **`marc_sdk`(수정 금지 보호 폴더) 내부의 `client.sleep()`/`now_s()`**
  이거나 **`stage1/pipeline.py`(수정 시 승인 필요) 안의 실제 추론 루프** 둘 중 하나로 좁혀짐 —
  코드만 읽어서는 확정할 수 없는 지점까지 옴.
- 정확한 스택을 보려고 `py-spy dump`를 시도했으나, 이 머신은 `ptrace_scope=1`이라 같은 사용자라도
  다른 프로세스를 ptrace하려면 **sudo가 필요** — Claude Code는 sudo 대화형 입력을 할 수 없어
  사용자에게 직접 실행을 요청한 상태에서 세션이 종료됨. **아직 결과를 받지 못함.**

## 4. 근본 원인 확정 및 수정 (`py-spy`로 확정)

이후 세션에서 `sudo py-spy dump --pid <pid>`를 두 번(시간차를 두고) 떠서 완전히 동일한 스택을
확인했다 — `demo/agent_vla.py`의 `_observe()` 안 `self.client.sleep(0.1)` 호출에서 매번 정확히
같은 줄에 멈춰 있었다. `/clock` 토픽 자체는 독립적으로 `ros2 topic hz /clock`으로 확인한 결과
~16Hz로 정상 발행 중이었다.

**확정된 원인**: `marc_sdk/client.py`의 `now_s()`는 "`/clock`을 아직 못 받았으면 `time.time()`
(벽시계, ~17억대 큰 값)으로 폴백하고, 받았으면 시뮬레이션 시간(작은 값)을 반환"한다. `_observe()`는
시작 시 `t_end = self.client.now_s() + wait_s`를 **딱 한 번** 계산하는데, 하필 그 첫 호출이
아직 `/clock`을 못 받은 순간과 겹치면 `t_end`가 거대한 벽시계 기준 값으로 고정된다. 직후
`/clock`을 받기 시작하면 이후의 `now_s()`는 훨씬 작은 시뮬레이션 시간을 반환하므로
`now_s() > t_end` 비교가 **영원히 거짓**이 되어 루프를 못 빠져나간다. 개별 `sleep(0.1)` 호출은
`marc_sdk` 자체의 안전장치(`wall_cap`) 덕에 최대 1.1초씩만 걸리지만, 반복 자체가 무한히
계속된다. (`HF_HUB_OFFLINE=1`로 등록→라운드1 시작 시간을 앞당긴 게 오히려 `/clock` 수신 전에
`_observe()`가 먼저 호출될 레이스 확률을 높여 버그를 더 잘 재현시켰던 것으로 보인다.)

**수정**: `marc_sdk/`와 `stage1/pipeline.py`는 건드리지 않고, `demo/agent_vla.py`의 `_observe()`만
시간값 비교 대신 **최대 반복 횟수(`wait_s/0.1` = 30회)** 기준으로 바꿔 이 시간 기준 불일치를
근본적으로 우회했다(사용자 승인 받고 적용). `time.time()`/`time.sleep()`은 여전히 안 씀 — 각 반복은
그대로 `client.sleep()`으로 페이싱한다.

**검증**: 플랫폼을 2차 재기동(사용자 승인, 매번 예외 승인받음) 후 같은 3라운드로 재접속한 결과,
**더 이상 멈추지 않고 3라운드 전부 제한시간 안에 제출됨**을 확인했다(Round1=0.0, Round2=45.0,
Round3=18.75). 무한루프 버그는 해결된 것으로 결론.

## 5. 수정 후 새로 드러난 문제 두 가지 (별개 원인)

수정 후에도 라운드별로 "카메라 데이터"와 관련된 문제가 남아있었는데, **두 가지는 서로 다른 원인**이다:

### 5-1. Round 1: 카메라 데이터 자체가 3초 안에 준비 안 됨
로그: `[vla] no CCTV observation -- submitting empty grounding` → 점수 0.0, `records/`에 해당
라운드 폴더가 생기지 않음(기록할 이미지가 없었으므로). 등록 직후 `/tf_static` 구독을 새로
만드는 데 걸리는 초기 지연(DDS 디스커버리)이 `_observe()`의 3초 예산을 넘긴 것으로 추정 —
**이미지가 영영 안 오는 게 아니라, 딱 그 라운드의 3초 창 안에만 늦게 도착한 것**으로 보인다
(Round2/3에서는 카메라 데이터 자체는 정상 수신됨). 실측 필요: `_camera()`가 실제로 몇 초 만에
준비되는지 로그로 재보지 않았음.

### 5-2. Round 2, 3: 이미지는 도착했지만 GPU 메모리 부족으로 AI 추론 실패
로그: `stage1.pipeline ERROR: camera rig_X failed: CUDA out of memory. ... GPU 0 has a total
capacity of 7.53 GiB of which ~200-300MiB is free. Process(Isaac Sim) has ~5GB in use` — 6개
카메라 전부에서 동일하게 발생, 이어서 `no detection in any camera → fallback payload`로
넘어감. 즉 **이미지 수신은 정상**이었고, Grounding DINO 추론 한 번(328MiB 필요)을 할 여유조차
GPU에 없었던 것. `stage1/pipeline.py`의 규칙 기반 fallback 덕분에 0점은 면해 45.0/18.75점을
받았지만, 실제 비전 인식은 매 카메라마다 실패한 상태다. 이건 BRIEF.md가 애초에 경고한
"8GB를 Isaac Sim과 나눠 써야 한다"는 리소스 제약이 그대로 드러난 것 — VLM을 이미 꺼뒀는데도
탐지기 하나조차 여유가 빠듯하다는 뜻이라 다음 세션에서 우선 다뤄야 할 이슈.

### 5-3. `results/scores.jsonl` 권한 오류 (해결됨)
`results/`, `results/marc2026_chungmu/`가 이전에 Docker 컨테이너(root)가 만들어서 `root:root`
소유였음 → 참가자 프로세스(`parktaejeong`)가 못 씀 → 매 라운드마다
`[Score] JSONL logging failed: Permission denied` 발생. 사용자가
`sudo chown -R parktaejeong:parktaejeong results/` 실행해서 해결.

## 6. 이번 세션 최종 산출물 확인

`./records/`에 라운드별 폴더 생성 확인 (카메라 6대 PNG + `meta.json`, 총 ~24MB):
- `records/s1_002_...`, `records/s1_003_...` (Stage1 Round 2, 3 — Round1은 위 5-1 사유로 폴더 없음)
- `records/s2_004_...` (Stage2 진입 시점 기록)

Stage2(`rt01_parking_tumbler`, 최대 600초)는 사용자 지시에 따라 자연 종료될 때까지 그대로 두는
중 — 별도로 결과를 더 분석하지 않음(원래 프롬프트 범위 밖).

## 7. (다음 프롬프트 범위) Isaac Sim을 끄고 VLM까지 켜서 오프라인 재검증하는 법

`MARC_RECORD_DIR`에 남긴 `records/s1_*`는 ROS bag과 비슷한 역할을 한다 — 다만 rosbag2 형식이
아니라 라운드별 폴더에 카메라 6대 PNG + `meta.json`(명령, 카메라 intrinsics/extrinsics/ground_z,
그 순간의 payload)으로 직접 저장한다. **ROS/Isaac Sim/marc_sdk에 대한 의존이 전혀 없는 순수
데이터**라서, Isaac Sim이 완전히 꺼진 상태에서도 그대로 재사용 가능하다.

재생 도구는 `tools/eval_offline.py` — `meta.json`에서 `CameraModel`을 복원하고 PNG를 다시 읽어서
`stage1/pipeline.py`의 `build_pipeline().run(...)`(라이브 세션과 동일한 파이프라인 함수)를
ROS 없이 순수 로컬 Python으로 다시 돌린다.

**절차**:
1. `bash simulation-platform/marc.sh down` — Isaac Sim을 완전히 내려서 GPU 8GB를 통째로 비운다
   (라이브 세션 중엔 ~5GB를 Isaac Sim이 점유하고 있었음).
2. `pip3 install --user bitsandbytes` — 4bit 양자화용, 이번 세션엔 설치 안 함
   (`docker/requirements.txt`엔 있으나 라이브 감지기 테스트엔 불필요해서 생략했었음).
3. VLM을 켠 채로 오프라인 재실행:
   ```bash
   MARC_VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct MARC_VLM_4BIT=1 \
   python3 tools/eval_offline.py --records ./records
   ```
   - `MARC_DISABLE_VLM`을 안 주면(또는 `0`) `build_pipeline()`이 Qwen2.5-VL도 로드한다.
   - `stage1/pipeline.py` 주석: "로컬 8GB엔 7B 대신 3B, 4bit 권장" — 그대로 적용.
   - `--gt demo/mock_demo_data.yaml`은 **쓰면 안 됨** — 그 파일은 `marc2026_demo` 시나리오용
     가짜 GT라 우리가 기록한 `marc2026_chungmu`의 lf01/lf02 명령과 매칭되지 않는다
     (`_generated_from: marc2026_demo.yaml`로 확인됨). GT 비교 없이 새 payload만 눈으로
     확인하는 용도로만 쓴다.
   - `tools/eval_offline.py`는 `records/s1_*` 폴더만 읽는다 — 지금 있는 `s1_002`, `s1_003`
     2개 라운드만 재생 대상이고, Round1(폴더 없음, 위 5-1 사유)과 Stage2 `s2_004`는 대상 아님.
4. 실행 전 `nvidia-smi`로 Isaac Sim 종료 후 실제 여유 VRAM을 반드시 재확인.

이 항목은 `claude_code_prompt.md`가 명시적으로 "다음 프롬프트"로 분리해둔 범위라 이번 세션에서는
**설명·문서화만 하고 실행하지 않았다.**

## 8. 다음 세션에서 먼저 할 일

1. **GPU 메모리 경합(5-2) 해결 방안 검토** — 예: 라운드에 실제로 필요한 카메라 1~2대만 추론하도록
   범위를 좁히거나(현재는 6대 전부 순회), Isaac Sim 해상도/설정을 더 낮추거나, 탐지기를 더 가벼운
   모델로 바꾸는 방안 등. `stage1/pipeline.py` 로직 변경이 필요하면 절대 규칙에 따라 먼저 설명하고
   승인받을 것.
2. **Round1 콜드스타트 지연(5-1) 실측** — `_camera()`가 실제로 몇 초 만에 준비되는지 로그를 남겨서,
   `_observe()`의 기본 `wait_s=3.0`을 늘려야 하는지 판단.
3. `problems.yaml.bak` 복구 여부는 여전히 사용자와 상의 필요(원본 프롬프트 6번 항목, 아직 미결).
4. GPU 여유는 매번 `nvidia-smi`로 재확인 후 진행.

## 9. 이번 세션에서 변경된 파일 목록

- `problems.yaml` (수정, 백업: `problems.yaml.bak`)
- `scripts/diag_link.sh` (set -u / ROS setup.bash 충돌 우회, 진단 로직 자체는 변경 없음)
- `demo/launch.sh` (`MARC_ENTRYPOINT` 환경변수로 실행 진입점 선택 가능하도록 마지막 줄 + 사용법 주석 추가, 기본 동작은 기존과 동일)
- `demo/agent_vla.py` (`_observe()`의 종료 조건을 시간값 비교 → 최대 반복 횟수로 변경, 무한루프 버그 수정)
- 호스트 시스템 변경(코드 아님): `ros-humble-ros-base`, `ros-humble-tf2-ros`, `python3-pip` (apt),
  `torch`/`torchvision`/`transformers`/`accelerate`/`huggingface_hub`/`qwen-vl-utils`/
  `opencv-python-headless`/`placo`/`py-spy` (pip --user), `results/` 소유권을 `parktaejeong`으로 변경(sudo chown)
- `simulation-platform`을 사용자 승인 하에 2회 재기동(원래는 금지 규칙이나 이번 세션 한정 예외)
