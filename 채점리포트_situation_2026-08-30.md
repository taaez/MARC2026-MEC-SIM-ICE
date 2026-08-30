# 자가 채점 결과창에 situation 을 드러냈다 (2026-08-30)

`python3 mec_sim_ice/score.py` 를 돌려도 **sar/sarH(사람·상황) 문항의 situation 을
맞혔는지 알 수 있는 칸이 없었다.** 그 칸을 만들면서, 사람 문항의 배점 구조가
분실물과 어떻게 다른지 플랫폼 원본 응답으로 다시 확인한 기록이다.

점수 계산식 자체는 건드리지 않았다 — 39문항 리플레이 `better 0 / worse 0 / same 39`.

---

## 1. 왜 안 보였나

플랫폼이 라운드마다 돌려주는 점수 키는 **여섯 개뿐**이다:

```
camera / object_type / landmark / relation / anchor / target
```

`situation` 이라는 키는 없다. 사람 문항의 situation 판정은 **`object_type` 칸에
담겨서** 온다. 그래서 리포트의 `type` 칸(15점)만 보고는 "물건 종류를 맞혔나 보다"
로 읽히고, 정작 상황 분류가 맞았는지는 화면 어디에도 없었다.

## 2. 근거 — 플랫폼이 보낸 원본 그대로

데이터 출처: `runs/chungmu_full39_2026-08-25.jsonl`. 이 파일의 `scores` 는
`marc_sdk/types.py:87` 의 `Score.from_payload` 가 **401 SCORE_RESULT 페이로드를
그대로** 담은 것이다 — 우리 코드가 계산한 값이 아니다.

### (a) situation 은 `object_type` 칸에 온다

제출 `target_type` 은 규격상 사람 문항에서 항상 `"person"` 이다(14문항 전부 동일).
그런데 `object_type` 은 9건 1.0 / 5건 0.0 으로 갈렸고, **그 5건이 situation 오답
5건과 정확히 일치**한다:

| problem_id | 정답 situation | 제출 situation | object_type |
|---|---|---|---|
| sar04_table_lie | abnormal | normal | 0.0 |
| sar06_parking_collapsed | emergency | abnormal | 0.0 |
| sar07_parking_prone | accident | abnormal | 0.0 |
| sarH2_parking_bikeacc | accident | emergency | 0.0 |
| sarH6_most_critical | accident | abnormal | 0.0 |
| (나머지 9건) | = 제출 | | 1.0 |

### (b) `relation` 칸은 situation 이 아니다

"분실물의 relation 자리가 사람 문항에서 situation 으로 바뀐다"는 해석은 데이터와
맞지 않는다. `relation` 은 situation 을 **맞힌** 문항에서도 14문항 전부 0.0 이고
배점도 0 이다 (예: `sar01` situation 맞음 → `relation=0.0`, `object_type=1.0`).

### (c) landmark 는 배점 0

플랫폼이 사람 문항 14개 전부에 돌려준 `weights`:

```json
{"camera": 0.1, "object_type": 0.15, "landmark": 0.0, "relation": 0.0, "anchor": 0.15, "target": 0.4}
```

landmark 점수 자체는 계산돼서 온다(맞으면 1.0). **배점만 0**이다. 총점으로 교차검증:

| 가설 | 총점 불일치 |
|---|---|
| A: 플랫폼 weights 그대로 (landmark·relation 0, 합 0.8 을 100 으로 정규화) | **0 / 14** |
| B: relation 자리 = situation + landmark 에 배점 있음 | 14 / 14 (sar01 실제 18.75 vs 35.00) |

## 3. 확정된 배점 (100점 만점 환산)

| 항목 | 분실물 (lf/hf/rt) | 사람 (sar/sarH) |
|---|---:|---:|
| camera | 10 | **12.5** |
| object_type / **situation** | 15 (물건 종류) | **18.75** (situation) |
| landmark | 10 | **0** |
| relation | 10 | **0** |
| anchor | 15 | **18.75** |
| target | 40 | **50** |

사람 문항은 해당 없는 두 항목(각 0.1)이 빠져 가중치 합이 0.8 이 되고, 플랫폼이
거기에 맞춰 100점으로 정규화한다.

> ⚠️ 이건 2026-08-25 **연습 플랫폼 실측**이다. 공식 문서는 "정확한 임계값·가중치는
> 공개하지 않습니다" 라고 못박고 있으므로, 본선에서 사람 문항 landmark 배점이
> 살아 있을 수 있다. 그래서 리포트는 **배점이 0이어도 맞았는지는 항상 표시**한다.
> 배점이 바뀐 게 확인되면 `validation/chungmu_gt.py` 의 `PERSON_WEIGHTS` 한 줄만
> 고치면 표·평균·총점·설명 문구가 전부 따라간다 (전부 weights 에서 계산한다).

## 4. 결과창에서 바뀐 것

### `[배점]` 절 신설 — 섹션마다 유형별 배점을 먼저 찍는다

```
[배점]  (플랫폼이 라운드마다 돌려준 weights 에서 계산 -- 100점 만점 기준)
  분실물 19문항: camera 10 / object_type 15 / landmark 10 / relation 10 / anchor 15 / target 40
  사람   14문항: camera 12.5 / situation 18.75 / anchor 18.75 / target 50   (landmark·relation 배점 0)
  사람 문항의 situation 점수는 플랫폼이 'object_type' 칸에 담아 돌려준다 -- 아래 표의 'type' 칸이 그것이다.
```

### 항목별 평균을 유형별로 분리

같은 키라도 뜻(object_type ↔ situation)이나 배점이 다르면 한 줄로 합치지 않는다.

```
  object_type(물건)  평균 15.00 / 15.0점   만점 19/19건
  situation(사람)    평균 18.75 / 18.8점   만점 14/14건
  target(물건)       평균 15.61 / 40.0점 │ target(사람) 평균 38.99 / 50.0점
```

### `[situation 판정]` 절 신설

```
[situation 판정]  person 14문항 -- situation 정답 9/14 (64.3%)
  landmark 일치 11/14   (person 문항 landmark 는 배점 0 -- 맞아도 점수 없음)
  problem_id                          정답          제출   situation  landmark
  sar04_table_lie               abnormal      normal       X      O
  sar06_parking_collapsed      emergency    abnormal       X      X
  ...
  [혼동 패턴]  정답 → 제출
    accident → abnormal   2회
    abnormal → normal   1회
```

혼동 패턴까지 찍는 이유: 무엇을 무엇으로 착각하는지가 곧 다음에 고칠 규칙이다
(위 기록에서는 `accident` 를 못 내는 것이 최대 손실이었다).

### 문항별 표를 실효 배점으로

전에는 사람 문항이 명목값(15/15, 10/10)이라 **칸을 다 더해도 총점과 안 맞았다.**
이제 맞는다. `sit` 칸 = situation O/X, `(O)/(X)` = 채점은 되지만 배점 0인 칸.

```
  problem_id                     총점     camera      type  sit  landmark  relation     anchor     target
  sar02_vending_sit           95.07  12.5/12.5 18.8/18.8    O       (O)         –  13.8/18.8  50.0/50.0
```

### 실점 원인 분류 버그 수정

배점 0인 항목을 실점 원인에서 제외한다. 사람 문항은 `relation` 이 항상 0.0 으로
오는데 예전 로직이 그걸 원인으로 세서, 라이브 기록에서 `relation_오답` 이 9건으로
나왔다 — **그 중 7건이 사람 문항 오진**이었고 실제 원인은 target 좌표였다.

```
before:  relation_오답 9건 (sarH1 47.55, sar09 47.59, ...)   좌표_오차 4건
after :  relation_오답 2건 (hf01, hf05)                       좌표_오차 10건
```

### `--json` 리포트

- 라운드마다 `situation: {expected, submitted, ok}` (분실물은 `null`)
- 섹션 요약에 `situation: {problems, correct, accuracy, per_problem, wrong, landmark_correct, landmark_weighted}`
- `fields` 는 숫자만 유지했다 — `cmp_reports.py` 가 그 값을 `:.2f` 로 찍기 때문에
  문자열이 섞이면 깨진다.

## 5. 고친 파일

| 파일 | 내용 |
|---|---|
| `taejeong/validation/score_report.py` | `situation_check` / `max_points` / `_print_weight_profiles` / `_situation_section` / `_situation_summary` 추가, `classify` 가 배점 0 항목을 제외, 표·평균을 실효 배점으로 |
| `taejeong/validation/offline_score.py` | 모듈 독스트링과 상단 경고 문구를 확정된 배점 구조로 갱신 |

## 6. 검증

- 라이브 기록 리플레이(`score_report.py runs/chungmu_full39_2026-08-25.jsonl`):
  situation 9/14, 혼동 패턴이 `object_type` 0점 문항 5건과 정확히 일치
- 현재 코드 오프라인 채점(`score.py --no-fallback`): situation 14/14, Stage1 평균 70.91
- `cmp_reports.py` 로 변경 전후 JSON 비교: `better 0 / worse 0 / same 39` (점수 불변)
- `test_command_parser` / `test_grounding` / `test_situation_resolver` 전부 PASS

## 7. 같이 커밋되는 직전 작업 (pose keypoints)

이 커밋 세트에는 커밋되지 않고 남아 있던 직전 작업도 함께 들어간다:

- **`pose_keypoints.py` 신규 + `fallback.pose_augment`** — 플랫폼 정답이 사람의
  중심이 아니라 **발**이라(눕힌 prim 의 회전 pivot 이 발), bbox 가로 중앙을 쓰면
  누운 사람에서 몸길이 절반(~0.85m)만큼 어긋난다. COCO 사전학습 pose 모델의 발목
  keypoint 로 **가로만** 보정한다 — 세로까지 쓰면 얕은 각도에서 픽셀 오차가 world 로
  증폭돼 오히려 나빠진다(rig_7_a 실측 세로 1px = 0.075m). 사람 14문항: 발목 그대로
  역투영 385점 < 현재 419점 < 발목 가로+접지선 434점.
- **`grounding._foot_adjusted_point`** — 보정을 payload 만드는 지점에만 적용한다.
  `Located.base` 를 옮기면 relation 판정·anchor 선택 입력까지 흔들려서 sar07/sarH6
  의 anchor 가 0.95 → 0.04 로 무너졌다.
- **`grounding.resolve` 의 `ground_surface_token` 기본값** — 명령에서 파싱한
  `area_hint` 를 쓴다. "parking lot 바닥에 쓰러진" 류에서 support 가 None 이라
  공식 규칙표를 못 타고 자세 기본값으로 떨어지던 것을 고친다
  (실측 sar01/sarH4/sar06 의 situation 이 맞아지고 회귀 0건, 68.54 → 70.25).
- **`MARC2026_Chungmu_Stage1_33문항_정답_CCTV씬.md`** — Stage1 33문항의 정답·CCTV 씬
  기준표. 이번 배점 검증에서 "SAR 제출 `target_type` 은 항상 `person`" 근거로 썼다.

## 8. 남은 것

- **target 좌표 점수는 여전히 근사치다.** 라이브 채점이 물체 크기/영역까지 보는
  것으로 보여 중심점 거리(`1 - 오차/5`)만으로 재현되지 않는 사례가 있다. 방향
  (개선/악화) 판단에만 쓸 것.
- 위 3절의 경고대로, 본선 배점이 연습 플랫폼과 같다는 보장은 없다.
