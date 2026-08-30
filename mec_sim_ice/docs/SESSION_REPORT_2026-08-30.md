# SESSION REPORT 2026-08-30 — 소형객체 서브모델 통합 + trash_can 실점 원인 확정 + 라이브 진입점 연결

작성: 2026-08-30. 소형객체 서브모델 학습~평가~파이프라인 통합, rig_3_b
trash_can "탐지 실패" 의혹의 실측 반증, trash_can 관련 6문항 채점 분석,
그리고 §4/§5 개선이 실제 라이브 진입점엔 비어 있던 문제(§7)까지.

---

## 1. 소형객체 서브모델 (small_t640_r2) — 학습·평가 완료

sunblock / tissue / disposable_cup 3클래스 전용 타일 확대 서브모델.
`training/small_object_detection/소형객체_서브모델_가이드.md` 의 절차 그대로:
selftest 33항목 통과 → 학습(200 epoch 설정, **epoch 95 best**, patience 100
early stop, batch 4) → 원본 1280×720 val 500장 실측 평가(evaluate.py).

### 도입 전(B 단독) vs 도입 후(B+서브 라우팅) — conf 0.35 / small_conf 0.25

| 클래스 | recall | AP50 | TP | FP |
|---|---|---|---|---|
| sunblock | 0.416 → **0.575** | 0.443 → 0.582 | 91 → 126 | 32 → 41 |
| tissue | 0.443 → **0.652** | 0.461 → 0.634 | 102 → 150 | 29 → 48 |
| disposable_cup | 0.498 → **0.709** | 0.531 → 0.694 | 123 → 175 | 35 → 51 |
| 전체 34클래스 | R 0.819 → 0.829 | F1 0.851 → 0.856 | | |

크기 버킷별 recall (승부처는 `<8px` — val GT 의 49~75% 가 이 구간):

| 구간 | sunblock | tissue | disposable_cup |
|---|---|---|---|
| <8px | 0.185 → **0.407** | 0.382 → **0.584** | 0.480 → **0.667** |
| 8–16px | 0.750 → 0.803 | 0.878 → **0.980** | 0.846 → 0.923 |
| 16px+ | 0.97~1.00 | 1.00 | 0.91 |

- 나머지 31클래스는 B 출력 그대로 → **회귀 구조적으로 불가**.
- `small_conf=0.25` 확정 (스윕상 F1 정점 0.25~0.30).
- B 박스와의 공간 겹침 2.5%(TP4/FP11)뿐 → veto 없이 **concat 유지**.
- 산출물: `training/small_object_detection/runs/detect/small_t640_r2/weights/best.pt`,
  평가 전문 `~/MARC2026-MEC-SIM-ICE/evaluate_small_live.log`.

주의할 해석: 전체 recall 0.58~0.71 이 낮아 보이는 건 원거리 `<8px` 인스턴스가
분모의 절반 이상이라서다. "그 물체가 제대로 보이는 카메라" 구간(8px+)은
0.80~1.00. 현 충무 문제셋에서 소형객체 타깃은 sunblock 2문항(lf04/hf04)뿐이고
이미 98.1점이라, 서브모델은 지금 점수용이 아니라 **문제셋 변경 대비 보험**이다.
`<8px` 를 더 밀려면 tile 320 재학습(배율 4배, 비용 2.5배)이 유일한 구조적 수단.

## 2. trash_can "탐지 실패" 의혹 — 실측 반증

### 의혹: "rig_3_b 에서 쓰레기통이 커지면 못 잡는다" / "hf06 앵커 0점 = 탐지 실패"

### 실측 결과: 탐지는 전 카메라 정상. 실패는 인스턴스 선택이다.

1. **충무 프레임 재추론**: records 의 rig_3_b 프레임에 B 를 직접 추론 →
   trash_can **conf 0.92** 검출. 정답 앵커 좌표를 카메라 투영하면 정확히 그
   박스에 떨어짐(= 정답 인스턴스 trash_can_01 맞음). 6캠 전부 0.83~0.96 검출.
   학습셋은 전부 엎어진 자세인데 서 있는 것도 잡는다.
2. **제출 좌표의 정체**: lf06/hf06 이 제출한 (-86.0, 108.2) 는 rig_3_c 인도
   위에 실재하는 **다른 trash_can 인스턴스**의 정확한 위치(역투영으로 박스
   정중앙 확인). 탐지 못 한 시스템은 이렇게 못 찍는다.
3. **왜 rig_3_b 를 안 골랐나**: 타깃 우산의 정답 좌표를 rig_3_b 로 투영하면
   픽셀 (1041,268) — 피크닉 테이블 위 **누운 사람 몸통 위**. 우산이 완전히
   가려져 rig_3_b 에 umbrella 검출이 0개다. grounding.resolve() 는 landmark
   게이트(클래스 문자열 매칭이라 6캠 전부 통과 = 변별력 0) 뒤 **(target,
   landmark) 쌍의 관계 적합도**로 카메라를 고르므로, 쌍을 못 만드는 rig_3_b
   대신 trash_can 0.87 + umbrella 0.56 쌍이 깔끔한 rig_3_c 가 이겼다.
   → camera/anchor/target 동반 0점 (35점 고정).
4. **val recall 편차의 원인도 크기가 아니라 가림**: dataset_v2 val 실측
   rig_3_b recall 0.825 (타 rig 0.88~0.97). 놓친 7장 중 6장이 occlusion
   0.83~0.94, 1장은 프레임 잘림. 엎어진 자세는 전 rig 공통이라 원인 아님.

시각 자료: `~/bongbak-cuda-test/detection_preview/{demo_0825,chungmu_0828}/`
(6캠 박스 오버레이 — 빨강 trash_can, 파랑 umbrella).

### trash_can 관련 6문항 채점 현황 (2026-08-29 20:52 채점 기준)

| 문항 | 카메라 | anchor 오차 | target 오차 | 총점 | 판정 |
|---|---|---|---|---|---|
| sar01 / sarH4 | rig_7_a ✓ | 0.46m | 0.62m | 98.3 | 3838ca4 로 해결됨 |
| sar09 / sarH1 | rig_1_a ✓ | 0.71m | **0.98m ✗** | 47.4 | target 합격선(0.62~0.98 사이) 초과 |
| lf06 / hf06 | rig_3_c ✗ | 15.3m | 13.0m | 35.0 | 위 인스턴스 선택 실패 |

우선순위: **P1** sar09/sarH1 사람 좌표 30cm 개선(+50점×2 — ptj 브랜치
pose-keypoints 커밋이 이 방향, §3) / **P2** lf06/hf06 landmark 단독 카메라
유지 로직(+25~30점×2) / P3 rig_7_a 차 지붕 trash_can 오탐(0.73) 감시.

## 3. ptj 브랜치 코드 최신화 (받기만, 푸시 없음)

`origin/ptj/stage1-vla-integration` 의 3커밋에서 **코드 파일만** 워킹트리로
가져옴(`git show origin/...:파일 > 파일` — 브랜치 포인터·인덱스·폴더 구조
불변):

- `6c92088` pose-keypoints: 사람 target 을 발목 가로 위치로 보정
  (fallback.py, grounding.py, pose_keypoints.py 신규, test_grounding.py,
  offline_score.py) — P1 방향의 팀 커밋. 실측 419→434점 근거 포함.
- `9544e8d` score_report: situation 정답 여부·실배점 표시 (score_report.py)
- 문서 커밋(869991b)은 코드 아님이라 제외.

부수 수리: pose 기본 가중치 `demo_agents/weights/yolo11m-pose.pt` 다운로드,
real_agent_vla.py 의 det_classes 탐색 후보에 `training/` 추가(폴더 정리 후
import 깨져 있던 것).

## 4. 소형모델 파이프라인 통합

원칙: 산출물이 있으면 라우팅, 없으면 기존 그대로 — **나빠지는 경로 없음**.
`MARC_DISABLE_SMALL_DET=1` 로 강제 비활성(관례: MARC_DISABLE_POSE_KEYPOINTS).

- `taejeong/yolo_adapter.py`: `load_detector()`(B 단독 ↔ RoutedDetector 자동),
  `infer_cameras()` 라우팅 분기, `predict_result()`/`RoutedResult`
  (real_agent 의 `_best_box` 호환 셸). 소형 3클래스 conf 는 B 와 스케일이
  달라 min_conf 재적용 금지 (predict.py 'WHY ROUTING' 참조). TiledPredictor
  batch=4 (8GB 에서 DINO/pose 동시 상주 대비).
- `taejeong/validation/offline_score.py`: `--small-weights/--tiling/
  --small-conf/--no-small` 추가, 기본은 자동 감지. `--no-small` 이 도입 전
  비교 기준.
- `real_agent_vla.py`: `YOLO(det_weights)` → `load_detector(...)`,
  추론 호출부 `predict_result(...)` 로 교체.

스모크: 라우팅 로드 → 실제 프레임에서 B 검출 16개 + 소형 sunblock 0.38 추가
확인. 로컬 테스트 3종(test_command_parser 10, test_situation_resolver 171,
test_grounding) 전부 PASS, real_agent_vla import OK.

## 5. 회귀 채점 (데모 → 충무) — 완료

두 세트 모두 `--no-fallback`(DINO/VLM 완전 배제) + 라우팅(B+소형) 조건.
DINO/VLM 없이 순수 (YOLO 라우팅 + pose-keypoints + grounding) 스택만으로
어디까지 나오는지 보는 하한선 측정이라, 라이브(폴백 항상 ON) 대비 낮게
나올 수 있음을 감안하고 읽는다(로그 자체가 명시).

- **데모** 41라운드(8/25 records, `~/bongbak-cuda-test/mec_sim_ice/records`
  + full39 정답지): Stage1 33문항 평균 **73.16점**, Stage2 grounding 6문항
  평균 76.62점. 로그: `~/MARC2026-MEC-SIM-ICE/offline_demo_routed.log`.
- **충무** 33문항(8/28 재생성 records, `~/record_260829/.../records`):
  Stage1 평균 **77.62점**(실측 레코드 26문항 80.32 + 프레임 대여 7문항).
  로그: `offline_chungmu_routed.log`.

### P1 (pose-keypoints, ptj 브랜치 통합분) 효과 — 두 세트에서 일관되게 확인

| 문항 | 도입 전(§2 표) | 데모 재측정 | 충무 재측정 |
|---|---|---|---|
| sar09_park_bending | 47.4 | **89.86** | **89.75** |
| sarH1_park_needhelp | 47.3 | **89.83** | **89.79** |

발목 가로좌표 보정이 target 커트라인(0.62~0.98m 사이)을 확실히 아래로
밀어냈다 — anchor·target 모두 만점권으로 이동, 라운드당 +42점 안팎.
sar01/sarH4/sar04/sar06~sarH6 등 나머지 사람 문항도 89~93점대로 고르게
나와, 이번 통합이 sar09/sarH1 두 건에 국한된 우연이 아님을 보여준다.

### P2 (카메라 선택 로직) 미착수 — 예상대로 불변

| 문항 | 도입 전 | 데모 재측정 | 충무 재측정 |
|---|---|---|---|
| lf06_umbrella_fallenbike | 35.0 | 35.0 (불변) | 35.0 (불변, † 프레임 대여) |
| hf06_fallenbike_umbrella | 35.0 | 35.0 (불변) | 35.0 (불변) |

이번 세션은 §4 통합(소형모델 라우팅) + ptj 브랜치 pull(P1) 만 다뤘고
grounding.resolve() 의 카메라 선택 자체는 손대지 않았으니 당연한 결과.
§2 의 진단(landmark 게이트가 클래스 문자열만 보고, target 쌍이 있는
카메라가 무조건 이긴다)이 그대로 남아 있다는 재확인이기도 하다.

### 소형모델 라우팅이 다른 문항을 깨뜨리지 않았는지

두 세트 모두 object_type 15.00/15.0(만점), landmark·relation 8~10/10대
유지 — B 31클래스 출력이 그대로 통과한다는 §4 설계대로. 회귀 없음.

## 7. 라이브 진입점 연결 — §4/§5 의 개선이 실제 라이브에 비어 있던 문제 수정

**증상**: 사용자가 `demo/launch.sh`(옵션 없이, 즉 기본 진입점)로 Isaac Sim
라이브를 직접 돌렸는데 성능 차이가 미미했다. §5 offline_score.py 재측정으로
본 sar09/sarH1 의 47→89.8점 도약이 라이브엔 안 보였다.

**원인**: `launch.sh` 기본 진입점 `participant_app.py` 는 `real_agent_vla.py`
(RealVLAGrounding, §4 에서 고친 파일) 가 아니라 별도 파일
**`demo_agents/taejeong_agent_vla.py`**(TaejeongVLAGrounding) 를 쓴다.
여기가 §4/§5 개선을 둘 다 안 타고 있었다:

- `self._det = YOLO(det_weights)` 로 B 단독을 직접 로드 → 소형모델
  라우팅이 이 파일에는 애초에 연결된 적이 없었다(§4 는 real_agent_vla.py 만
  고쳤다).
- `pose_augment()` 호출부가 `offline_score.py` 에만 있고 `fallback.py`/
  `grounding.py` 자체(ptj 브랜치 pull, §3)에는 정의만 있어 어디서도 안
  불렀다 → `Detection.foot_u` 가 항상 `None` → `_foot_adjusted_point()`
  가 no-op. offline_score.py 쪽 주석은 "라이브가 항상 켜므로 여기서도
  켠다"고 전제했지만, 그 전제 자체가 아직 사실이 아니었다.
- 반면 `grounding.py` 의 `ground_surface_token = parsed.area_hint` 폴백
  (ptj pull, §3) 은 `resolve()` 안의 순수 로직이라 게이트 없이 즉시
  적용됐다 — 사용자가 본 "미미한 차이"의 정체는 이것 하나였다.

**조치**: `taejeong_agent_vla.py` 한 곳만 수정 (+19/-5줄).
`YOLO(det_weights)` → `load_detector(det_weights, ...)`,
생성자에서 `AnkleKeypointEstimator` 로드, `dino_augment` 다음 줄에
`pose_augment(parsed, images, detections, self._pose)` 추가 —
offline_score.py 의 `run_round()` 와 같은 자리·같은 순서로 맞췄다.
두 기능 모두 산출물/모델이 없으면 조용히 기존 동작(B 단독, bbox 중앙)으로
떨어져 나빠지는 경로가 없다.

**검증**(record_260829 실프레임, 정답지 GT, `run_offline.load_round` 로
카메라/이미지 구성 → `infer_cameras` → `pose_augment` ON/OFF → `resolve`
전체 파이프라인 재현):

| 문항 | pose OFF | pose ON | 비고 |
|---|---|---|---|
| sar04_table_lie | target 0.89m | **0.51m** | |
| sar07_parking_prone | target 1.08m | **0.75m** | |
| sar09_park_bending | target 0.98m | **0.76m** | 합격선(0.62~0.98) 통과 |

anchor 는 세 문항 모두 불변 — 보정이 제출 좌표에만 적용된다는 설계 그대로.
검출기 로드 로그로 라우팅 활성(`[yolo] routed B+small`) 확인, pose 모델
로드 확인, `MARC_DISABLE_SMALL_DET=1` 로 라우팅이 실제로 꺼지는 것도 확인.
회귀 3종(test_command_parser/test_situation_resolver/test_grounding) PASS.

커밋: `feature/init` 브랜치 `f9e76ff` (기존에 스테이징돼 있던 records
229개 파일과는 분리해서 이 파일 1개만 커밋·push).

## 8. 남은 일

1. **lf06/hf06 카메라 선택 로직 (P2)** — 유일하게 남은 큰 구멍. target
   부재 시 landmark 고신뢰 카메라를 쌍 있는 카메라가 무조건 이기지 않도록
   — relation 오프셋으로 target 추정하는 경로 검토. +25~30점×2 기대.
2. §7 수정 후 **라이브(Isaac Sim) 재측정** — 이번 절의 검증은 오프라인
   파이프라인 재현이라, 실제 launch.sh 라이브 실행으로 한 번 더 확인할 것.
3. (보류) tile 320 재학습 — 본선 문제셋에서 원거리 소형 타깃이 나오면.
4. §5 회귀 채점은 `--no-fallback` 하한선이라, DINO/VLM 켠 상태(라이브
   조건) 재측정도 한 번 더 돌려서 최종 비교선을 만들어 둘 것.
