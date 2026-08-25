# chungmu 자가 채점 실행 절차 (Isaac Sim 데스크탑)

우리 파이프라인(`taejeong/` 텍스트 파싱 + 기하)과 팀원 YOLO 모델을 합쳐
`marc2026_chungmu` 문제를 풀고, 라운드별 점수로 자가 채점하는 절차.

## 구성 요소와 출처

| 구성 | 파일 | 출처 |
|---|---|---|
| 탐지 클래스 정의 (34종) | `test/det_classes.py` | 팀원 `feature/init` 브랜치 |
| 학습 스크립트 | `test/train_yolo11.py` | 팀원 `feature/init` 브랜치 |
| 학습된 가중치 | `demo/weights/best.pt` | **팀원에게 파일로 받아 직접 배치** (git 미포함) |
| 명령 파싱 · 기하 · situation | `taejeong/*.py` | 우리 구현 |
| ROS 배선 | `demo/taejeong_agent_vla.py` | 우리 구현 |
| 진입점 | `demo/participant_app_taejeong.py` | 우리 구현 |
| 공식 정답 정의 | `taejeong/content-image-2026.R03/marc2026_chungmu.yaml` | content 이미지에서 추출 |

`demo/mock_agent_vla.py` 는 건드리지 않는다. 대회 가이드가 요구하는 것은
`process()` / `process_stage2()` 시그니처 유지이고, `TaejeongVLAGrounding` 이 이를 지킨다
(팀의 기존 `real_agent_vla.py`, `agent_vla.py` 도 동일한 방식의 별도 파일이다).

## 0. 사전 준비 (최초 1회)

```bash
git pull                                  # ptj/stage1-vla-integration 브랜치에서
mkdir -p demo/weights
cp <팀원에게 받은 best.pt> demo/weights/best.pt
```

인터페이스 정합성 확인 — 가중치 클래스가 `det_classes.py` 와 맞는지 검사한다:

```bash
cd taejeong
python3 check_interfaces.py \
    --det-classes ../test/det_classes.py \
    --weights ../demo/weights/best.pt \
    --sdk ..
```

`PASS` 가 아니면 실행하지 말고 원인부터 해결한다.

## 1. 문제 선택

```bash
bash simulation-platform/marc.sh select marc2026_chungmu
```

TUI 키: `n`(전부 끄기) → `↑↓` 이동 → `space`(개별 토글) → `s`(저장). Stage2 는
라디오 방식이라 하나만 켜진다. 전체 39문항을 돌리려면 `a`(전부 켜기).

## 2. 플랫폼 실행

```bash
bash simulation-platform/marc.sh platform
```

Isaac Sim GUI 에서 **Play** 를 눌러야 CCTV/clock/TF 토픽이 살아난다.
`"Auto-plan: waiting for a participant to register..."` 로그가 뜨면 준비 완료.

## 3. 우리 파이프라인으로 접속

```bash
cd demo
MARC_ENTRYPOINT=participant_app_taejeong.py ./launch.sh <team_id>
```

라운드별 점수가 `results/scores.jsonl` 에 쌓인다. 별도 터미널에서 실시간 확인:

```bash
tail -f results/scores.jsonl
```

권한 오류(`Permission denied`)가 나면 `results/` 가 root 소유인 경우다:
`sudo chown -R $(whoami):$(whoami) results/`

## 4. 자가 채점

**(a) 결과가 진짜 chungmu 문제셋인지 먼저 검증** — 시나리오 혼선·문항 정의 변경을 잡는다:

```bash
cd taejeong/validation
python3 verify_run.py ../../results/scores.jsonl
```

**(b) 항목별 정확도와 실패 원인 분류:**

```bash
python3 score_report.py ../../results/scores.jsonl --json report.json
```

출력 내용:
- 항목별 평균 (camera / object_type / landmark / relation / anchor / target)
- 실패 원인 분류 (대상 미탐지 / 카메라 오선택 / 랜드마크 오답 / relation 오답 / 좌표 오차)
- 좌표 오차 분포 — **채점식은 실측 역산 결과 `점수 = 1 - 3D오차/5m`** (허용 반경 5m)
- 카메라 오선택 패턴 (정답 → 제출)
- 문항별 상세 표

**(c) 보존할 실행은 runs/ 로 복사** — 개선 전후 비교용:

```bash
cp ../../results/scores.jsonl runs/chungmu_$(date +%Y-%m-%d_%H%M).jsonl
```

## 기준선 (2026-08-25, 39문항 전체)

| 항목 | 값 |
|---|---|
| Stage1 평균 | 44.03 |
| Stage2 | 87.0 |
| 총점 | 131.03 / 200 |
| camera | 0.385 (15/39 정답) |
| object_type | 0.872 |
| landmark | 0.692 |
| relation | 0.513 |
| anchor / target | 0.245 / 0.279 |

주요 실점: **카메라 오선택 15건(평균 16.5점)**, **대상 미탐지로 인한 빈 제출 9건(평균 31.7점)**.
원본: `runs/chungmu_full39_2026-08-25.jsonl`

## 알려진 개선 후보

1. **대상 미탐지 시 landmark-only 폴백** — 현재는 `[0,0,0]` 을 내고 카메라·앵커·타겟 점수를
   전부 포기한다. 랜드마크만 찾아도 그 카메라와 앵커 좌표를 제출해야 한다 (기대 +5점, 위험 없음).
2. **target 미탐지 시 `target_coord = anchor_coord` 대체 제출** — 허용 반경 5m 이고
   GT 앵커↔타겟 거리 중앙값이 1.85m 라, target 가중치(0.4)에서 상당한 부분점수를 얻는다.
3. **랜드마크 판별력 가중** — `vending_machine`/`taxi`/`hydrant`/`police_car` 는 카메라 1대에서만
   보여 판별력이 높고, `bench` 는 6대 전부에서 보여 판별력이 없다. 이를 카메라 선택 점수에 반영.
4. **`area_hint` 반영** — `command_parser` 가 파싱은 하지만 `grounding.resolve()` 가 쓰지 않는다.

## 주의: 제출 경로는 아직 우리 코드가 아니다

루트 `docker-compose.yml` → `docker/Dockerfile` 은 `participant_app_agent_vla.py`
(Grounding DINO + Qwen) 를 실행하며, `COPY` 목록에 `taejeong/` 가 없다.
현재는 세 구현을 나란히 실험하는 단계라 정상이지만, **제출 전에 팀 합의로
`docker/Dockerfile` 의 `COPY` 와 `CMD`, 그리고 가중치 baking 을 정리해야 한다.**
