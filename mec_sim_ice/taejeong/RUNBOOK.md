# chungmu 자가 채점 실행 절차 (Isaac Sim 데스크탑)

우리 파이프라인(`taejeong/` 텍스트 파싱 + 기하)과 팀원 YOLO 모델을 합쳐
`marc2026_chungmu` 문제를 풀고, 라운드별 점수로 자가 채점하는 절차.

## 구성 요소와 출처

| 구성 | 파일 | 출처 |
|---|---|---|
| 탐지 클래스 정의 (34종) | `mec_sim_ice/test/det_classes.py` | 팀원 `feature/init` 브랜치 |
| 학습 스크립트 | `mec_sim_ice/test/train_yolo11.py` | 팀원 `feature/init` 브랜치 |
| 학습된 가중치 | `mec_sim_ice/demo_agents/weights/best.pt` | **팀원에게 파일로 받아 직접 배치** (git 미포함) |
| 명령 파싱 · 기하 · situation | `mec_sim_ice/taejeong/*.py` | 우리 구현 |
| ROS 배선 | `mec_sim_ice/demo_agents/taejeong_agent_vla.py` | 우리 구현 |
| 진입점 | `demo/participant_app.py` (**기본**) | 우리 파이프라인으로 배선됨 |
| 공식 정답 정의 | `mec_sim_ice/taejeong/content-image-2026.R03/marc2026_chungmu.yaml` | content 이미지에서 추출 |

(리포 정리로 대회측 원본 `demo/`/`marc_sdk/`/`simulation-platform/`/`manipulation/` 은 그대로 두고,
팀 제작 코드/문서/기록은 전부 `mec_sim_ice/` 아래로 옮겼다. 단, `demo/participant_app.py` 는
organizer 템플릿을 그 자리에서 직접 수정해 쓰는 파일이라 `demo/`에 남아 있다.)

`demo/mock_agent_vla.py` 는 건드리지 않는다. 대회 가이드가 요구하는 것은
`process()` / `process_stage2()` 시그니처 유지이고, `TaejeongVLAGrounding` 이 이를 지킨다.

`demo/real_agent_vla.py`(YOLO + 인라인 휴리스틱)는 **팀 합의로 삭제**했다. 함께 삭제한 것:
`demo/train_yolo11.py`(그 파일에서 `DET_CLASSES` 를 import 하던 구버전 — 최신본은
`test/train_yolo11.py`)와 `demo/YOLO11_학습_실행_테스트_가이드.md`(두 파일 전용 문서 —
대체본은 `test/YOLO11_학습_가이드.md`). 셋 다 팀원의 `feature/init` 브랜치에서도 이미
삭제된 것들이라 방향이 일치한다. 복구가 필요하면 git 이력에 남아 있다.

Grounding DINO + Qwen 실험은 `MARC_ENTRYPOINT=participant_app_agent_vla.py` 로 계속 쓸 수 있다.

**Isaac Sim 없이 바로 재채점하고 싶으면** (코드만 고쳐가며 반복할 때) `validation/offline_score.py`
를 쓴다 — `records/s1_*` 를 지금 코드로 재실행하고, 과거 라이브 실행이 남긴 정답(좌표 포함)으로
그 자리에서 채점해 문항별 표를 터미널에 바로 찍는다. 자세한 건 `validation/offline_score.py`
의 모듈 docstring과 `../OFFLINE_검증_가이드.md` 참고.

## 0. 사전 준비 (최초 1회)

```bash
git pull                                  # ptj/stage1-vla-integration 브랜치에서
mkdir -p mec_sim_ice/demo_agents/weights
cp <팀원에게 받은 best.pt> mec_sim_ice/demo_agents/weights/best.pt
```

인터페이스 정합성 확인 — 가중치 클래스가 `det_classes.py` 와 맞는지 검사한다:

```bash
cd mec_sim_ice/taejeong
python3 check_interfaces.py \
    --det-classes ../test/det_classes.py \
    --weights ../demo_agents/weights/best.pt \
    --sdk ../..
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
./launch.sh <team_id>
```

`participant_app.py` 가 기본 진입점이고 이미 우리 파이프라인으로 배선돼 있으므로
`MARC_ENTRYPOINT` 를 줄 필요가 없다.

라운드별 점수가 `results/scores.jsonl` 에 쌓인다. 별도 터미널에서 실시간 확인:

```bash
tail -f results/scores.jsonl
```

권한 오류(`Permission denied`)가 나면 `results/` 가 root 소유인 경우다:
`sudo chown -R $(whoami):$(whoami) results/`

## 4. 자가 채점

**(a) 결과가 진짜 chungmu 문제셋인지 먼저 검증** — 시나리오 혼선·문항 정의 변경을 잡는다:

```bash
cd mec_sim_ice/taejeong/validation
python3 verify_run.py ../../../results/scores.jsonl
```

**(b) 항목별 정확도와 실패 원인 분류:**

```bash
python3 score_report.py ../../../results/scores.jsonl --json report.json
```

출력 내용:
- 항목별 평균 (camera / object_type / landmark / relation / anchor / target)
- 실패 원인 분류 (대상 미탐지 / 카메라 오선택 / 랜드마크 오답 / relation 오답 / 좌표 오차)
- 좌표 오차 분포 — **채점식은 실측 역산 결과 `점수 = 1 - 3D오차/5m`** (허용 반경 5m)
- 카메라 오선택 패턴 (정답 → 제출)
- 문항별 상세 표

**(c) 보존할 실행은 runs/ 로 복사** — 개선 전후 비교용:

```bash
cp ../../../results/scores.jsonl runs/chungmu_$(date +%Y-%m-%d_%H%M).jsonl
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

## 카메라 선택에 대해 알아낸 사실 (2026-08-26, records 39라운드 GT 투영 분석)

기록된 카메라 파라미터로 GT 좌표를 6대에 투영해 확인한 것:

- **CCTV 6대는 서로 다른 공간을 본다 (시야가 겹치지 않는다).** 39문제 중 38개에서
  정답 target 이 보이는 카메라가 **정확히 1대뿐**이다. 즉 카메라 선택은 순위 매기기가
  아니라 "명령이 가리킨 물체 쌍이 실제로 있는 곳 찾기"다.
- **target 클래스만으로는 카메라가 절대 안 정해진다.** 씬에 `umbrella` 4개가 4대에,
  `tumbler` 4개가 3대에, `trash_can` 3개가 3대에 흩어져 있다.
- **랜드마크가 결정적이다.** 39문제 중 18개는 명령이 지목한 랜드마크가 카메라 1대에만
  존재해 그것만으로 확정된다. 나머지 21개 중 20개는 "랜드마크와 target 이 가장 가까운
  카메라" 로 해결된다 → 이론상 상한 38/39 (실제 실행은 15/39 였다).
- **사람 문제의 랜드마크는 0점이지만 65점을 여는 열쇠다.** landmark 배점이 0 이어도
  카메라를 좁혀 주는 단서는 사실상 랜드마크뿐이고, 그게 camera(10)+anchor(15)+target(40)
  을 결정한다. 그래서 배점과 무관하게 선택에 쓴다.

## 알려진 개선 후보

1. ~~대상 미탐지 시 landmark-only 폴백~~ — **적용됨**. `resolve()` 가 target 을 못 찾아도
   랜드마크만으로 제출한다 (`selection="anchor_only"`). 빈 제출은 0점 확정이지만
   랜드마크만 맞아도 camera/landmark/anchor 배점이 살아있다.
2. ~~target 미탐지 시 `target_coord = anchor_coord` 대체 제출~~ — **적용됨** (위와 함께).
   허용 반경 5m 대비 GT 앵커↔타겟 거리 중앙값이 1.85m 라 부분점수를 크게 건진다.
3. ~~랜드마크 판별력 가중~~ — **적용됨(폴백 판단 한정)**. 게이트가 카메라를 1대로 좁혔으면
   그 랜드마크는 변별력이 높다는 뜻이라 카메라를 신뢰한다(`landmark_discriminative`).
   순위 점수식에는 넣지 않았다 — relation 적합도만으로 이미 20/21 이라 상승 여지가 작고
   스케일 튜닝 위험만 커진다.
4. ~~`area_hint` 반영~~ — **적용됨**. 랜드마크 파싱이 실패하는 8문항은 애초에 랜드마크를
   말하지 않고 **구역**만 말한다 ("in the parking lot", "at the park"). 이제
   `AREA_INDICATOR_CLASSES` 로 구역 → 지표 클래스를 매핑해 게이트를 건다
   (`parking` → `CAR_CLASSES`, `park` → `{picnic_table, bench}`). 근거: "주차 구역"
   이라는 말 자체가 "차량이 있는 곳" 이라는 뜻이라 **씬이 아니라 단어의 의미에서
   나온 매핑**이고, 실제로 GT 상 police_car 는 rig_7_a 에만 있어 이 8문항의 정답
   카메라(7건 rig_7_a, 1건 rig_1_a)를 그대로 짚는다. 8문항 전부 게이트가 걸리며
   그 중 7건이 기존 카메라 오선택이었다.

   **새 배경에서의 안전성**: 게이트는 순수 폴백 계층이다. `area_hint` 는 명령문에
   해당 문구가 있어야만 잡히고(씬을 스캔하지 않는다), 지표 클래스가 한 카메라에서도
   안 잡히면 게이트가 그냥 풀려서 이 계층이 없을 때와 동일하게 동작한다. 즉 더
   맞힐 수는 있어도 이전보다 나빠지는 경로가 없다 (`test_grounding.py` T16 이 이
   성질을 고정한다). `road` 는 연습 코퍼스에 근거 사례가 없어 일부러 매핑을 비웠다.

## 남은 개선 후보

- **명시 랜드마크 파싱 실패 보완**: 위 8문항은 area 게이트로 카메라는 잡지만,
  landmark 필드 자체는 "target 에 가장 가까운 랜드마크" 로 추정한다. object 문제
  4건(lf08/lf09/rt01/rt02)은 landmark 배점 10점이 걸려 있어 여기서 더 딸 여지가 있다.
- **`sar04_table_lie`**: GT target 이 6대 어디에도 투영되지 않는다 (데이터 이상).
  다른 라운드를 다시 수집하기 전엔 이 문항은 상한이 막혀 있다.

## 주의: 제출 경로는 아직 우리 코드가 아니다

루트 `docker-compose.yml` → `docker/Dockerfile` 은 `participant_app_agent_vla.py`
(Grounding DINO + Qwen) 를 실행하며, `COPY` 목록에 `taejeong/` 가 없다.
현재는 세 구현을 나란히 실험하는 단계라 정상이지만, **제출 전에 팀 합의로
`docker/Dockerfile` 의 `COPY` 와 `CMD`, 그리고 가중치 baking 을 정리해야 한다.**
