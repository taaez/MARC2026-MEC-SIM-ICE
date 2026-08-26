# RealVLAGrounding 자가 채점 실행 절차 (Isaac Sim 데스크탑)

`mec_sim_ice/real_agent_vla.py`(YOLO11 + det_classes 34종 + 공식 situation 규칙)를
`marc2026_chungmu` 문제셋으로 자가 채점하는 절차. 전체 흐름은
`mec_sim_ice/taejeong/RUNBOOK.md` 와 동일하고, **3단계의 진입점만 다르다**
(`MARC_ENTRYPOINT=participant_app_real_vla.py`).

(리포 정리로 팀 제작 코드/문서는 `mec_sim_ice/` 아래로 옮겼다. organizer 원본 `demo/`
는 루트에 그대로 있고, 팀의 대체 진입점만 `mec_sim_ice/demo_agents/`에 있다 —
`launch.sh`가 두 위치를 모두 찾아보므로 아래 3단계 명령은 그대로 쓴다.)

## 구성 요소

| 구성 | 파일 |
|---|---|
| Stage 1 grounding | `mec_sim_ice/real_agent_vla.py` — `RealVLAGrounding` |
| 탐지 클래스 정의 (34종) | `mec_sim_ice/test/det_classes.py` (import 시 `test/` → `demo/` 순 탐색) |
| 명령 파싱 / situation 규칙 | `mec_sim_ice/taejeong/command_parser.py`, `mec_sim_ice/taejeong/situation_resolver.py` (재사용) |
| 진입점 | `mec_sim_ice/demo_agents/participant_app_real_vla.py` — `participant_app.py` 의 Stage 2 주행/픽업을 그대로 상속, grounding 구현만 교체 |
| 학습된 가중치 | `mec_sim_ice/demo_agents/weights/best.pt` — **팀원에게 받아 직접 배치** (git 미포함) |
| 문제셋 정의 | content 이미지의 `marc2026_chungmu` (`marc.sh select` 로 선택) |

## 0. 사전 준비 (최초 1회)

```bash
git pull                                  # ptj/stage1-vla-integration 브랜치
mkdir -p mec_sim_ice/demo_agents/weights
cp <팀원에게 받은 best.pt> mec_sim_ice/demo_agents/weights/best.pt
```

**가중치는 반드시 새 34-클래스(`test/det_classes.py` DET_CLASSES)로 학습된 것**이어야
한다. 옛 16-클래스 모델이면 `person_*` pose 클래스가 출력되지 않아 person 문제가 전부
빈 제출이 된다. 정합성 검사:

```bash
cd mec_sim_ice/taejeong
python3 check_interfaces.py \
    --det-classes ../test/det_classes.py \
    --weights ../demo_agents/weights/best.pt \
    --sdk ../..
cd ../..
```

`PASS` 가 아니면 실행하지 말고 원인부터 해결한다.

## 1. 문제 선택

```bash
bash simulation-platform/marc.sh select marc2026_chungmu
```

TUI 키: `a`(전부 켜기, 39문항) → `s`(저장). Stage2 는 라디오 방식이라 하나만 켜진다.

## 2. 플랫폼 실행

```bash
bash simulation-platform/marc.sh platform
```

Isaac Sim GUI 에서 **Play** 를 눌러야 CCTV/clock/TF 토픽이 살아난다.
`"Auto-plan: waiting for a participant to register..."` 로그가 뜨면 준비 완료.

## 3. RealVLA 파이프라인으로 접속

```bash
cd demo
MARC_ENTRYPOINT=participant_app_real_vla.py ./launch.sh <team_id>
```

**라운드 기록을 남기려면** `MARC_RECORD_DIR` 를 함께 준다 (한 번만 기록해 두면
이후에는 Isaac Sim 없이 오프라인으로 반복 실험 가능 — 아래 5절):

```bash
MARC_RECORD_DIR=../mec_sim_ice/records MARC_ENTRYPOINT=participant_app_real_vla.py ./launch.sh <team_id>
```

라운드마다 `records/s1_<라운드>_<ts>/` 에 카메라별 PNG + `meta.json`(명령,
카메라 K/R_cw/C_w/ground_z, 제출 payload)이 쌓인다.

**수집 전용 모드** — 점수는 필요 없고 이미지/카메라 파라미터만 모으려면
`MARC_RECORD_ONLY=1` 을 추가한다. YOLO 를 로드/추론하지 않으므로(weights 불필요)
라운드당 수 초에 끝나고, 빈 payload 를 즉시 제출해 다음 라운드로 넘어간다:

```bash
MARC_RECORD_ONLY=1 MARC_RECORD_DIR=../mec_sim_ice/records \
MARC_ENTRYPOINT=participant_app_real_vla.py ./launch.sh <team_id>
```

이때 1단계 `marc.sh select` 에서 **Stage2 문제는 꺼 둘 것** — Stage2 가 켜져 있으면
제출 후 주행/픽업 시퀀스(최대 30분)가 돌아 수집 시간이 크게 늘어난다.

시작 로그에 `[entry] Stage 1/2 grounding = RealVLAGrounding` 이 찍히는지 확인.
라운드별 점수는 동일하게 `results/scores.jsonl` 에 쌓인다:

```bash
tail -f results/scores.jsonl        # 별도 터미널
```

(팀 기본 taejeong 파이프라인과 비교하려면 진입점 없이 `./launch.sh <team_id>` 로
한 번 더 돌리면 된다 — 단, scores.jsonl 이 이어 붙으므로 실행 전에 파일을 비우거나
`MARC_SCORE_LOG` 로 경로를 분리할 것.)

## 4. 자가 채점

taejeong 파이프라인과 같은 도구를 그대로 쓴다:

```bash
cd mec_sim_ice/taejeong/validation
python3 verify_run.py ../../../results/scores.jsonl        # chungmu 문제셋 맞는지 검증
python3 score_report.py ../../../results/scores.jsonl --json report.json
cp ../../../results/scores.jsonl runs/chungmu_realvla_$(date +%Y-%m-%d_%H%M).jsonl
```

비교 기준선(taejeong 파이프라인, 2026-08-25, 39문항): Stage1 평균 44.03 / 총점
131.03 — `mec_sim_ice/taejeong/validation/runs/chungmu_full39_2026-08-25.jsonl`.

## 5. 오프라인 재실행 (Isaac Sim/ROS 불필요)

3단계에서 `MARC_RECORD_DIR` 로 기록해 뒀다면, 이후에는 아무 PC 에서나
(ultralytics + PyYAML + Pillow 만 설치) 파이프라인 수정 → 재채점을 반복할 수 있다:

```bash
# (리포 루트에서) 파이프라인 재실행 + 기록 당시 제출과 비교
python3 mec_sim_ice/tools/run_offline_real_vla.py --records ./mec_sim_ice/records --weights mec_sim_ice/demo_agents/weights/best.pt

# 좌표 포함 GT 와 비교 (필드 일치 + 좌표 점수 = 1 - 3D오차/5m)
python3 mec_sim_ice/tools/run_offline_real_vla.py --records ./mec_sim_ice/records --weights mec_sim_ice/demo_agents/weights/best.pt \
    --gt demo/mock_demo_data.yaml

# 공식 시나리오 expected 와 해석 필드(camera/target/landmark/relation) 비교
python3 mec_sim_ice/tools/run_offline_real_vla.py --records ./mec_sim_ice/records --weights mec_sim_ice/demo_agents/weights/best.pt \
    --gt mec_sim_ice/taejeong/content-image-2026.R03/marc2026_chungmu.yaml
```

한계: 오프라인 점수는 근사치다(공식 항목별 가중치는 플랫폼만 안다). 정확한 총점은
라이브 실행의 `results/scores.jsonl` 로 확인하고, 오프라인은 **개선 전후의 상대 비교**
(탐지/파싱/좌표 어디서 갈렸는지)에 쓴다. 참고로 `mec_sim_ice/records/` 에는 예전 세션이 남긴
샘플 라운드(s1_002, s1_003)가 이미 들어 있어 파이프라인 동작 확인용으로 쓸 수 있다.

## 주의사항

- **Docker 제출 경로에는 이 파이프라인이 없다.** `demo/Dockerfile` 은
  `real_agent_vla.py`(`mec_sim_ice/`) 와 `test/` 를 COPY 하지 않으므로 컨테이너 실행
  (`docker compose up`) 으로는 돌지 않는다. 자가 채점은 위처럼 호스트에서
  `launch.sh` 로 돌린다. 제출 파이프라인 결정은 팀 합의 사항
  (`mec_sim_ice/taejeong/RUNBOOK.md` "제출 경로" 절 참고).
- situation 은 검출된 pose × support 공식 규칙으로 판정하지만, support 는 명령문의
  landmark/지역 표현으로 근사한다 — 잔디/도로 위 사람 문제에서 어긋날 수 있다
  (개선 후보).
- `master_chef_can`, `wood_block` 은 파서 어휘에는 있으나 DET_CLASSES 에 없어
  검출 불가 → 해당 문제는 빈 제출이 된다.
- tf2 프레임 이름("world", camera_id)은 가정이다 — 이상하면
  `ros2 run tf2_tools view_frames` 로 확인.
