# MARC 2026 Chungmu 문제·정답·CCTV 씬 기준표

> 검증 결론: **Stage 1은 33문항**, 시나리오 전체 정의는 **39문항**이다. 이전 39문항 정리에는 Stage 2 회수형(RT) 6개가 합쳐져 있었다. 이 문서는 Stage 1 33개를 본표로 두고 RT 6개는 부록으로 분리한다.

- 기준 시나리오: `marc2026_chungmu.yaml`, version `2026.2`
- 원본 SHA-256: `8b824346cbf05b1c8d5626bbee6ae7f82d7fd55c1c55d66f80ff02516a2574a4`
- 검증일: 2026-08-29 (UTC)
- 좌표 단위: m, world frame `[x, y, z]`

## 1. 33개인가, 39개인가

| 범위 | 구성 | 개수 | 판정 |
|---|---:|---:|---|
| Stage 1 grounding | LF 11 + HF 8 + SAR 직접형 8 + SAR 서사형 6 | **33** | 사용자가 말한 분실물/사람·상황 네 유형의 문제 수 |
| Stage 2 retrieval | RT 6 | **6** | 회수·운반 명령이며 Stage 1 유형에 포함되지 않음 |
| YAML `challenge_problems` 전체 | Stage 1 + Stage 2 | **39** | 시나리오 파일에 정의된 총 엔트리 수 |

공식 Starter Kit의 `problems.yaml`도 `stage1:` 아래 33개 ID와 `stage2:` 아래 6개 ID를 분리한다. 따라서 **“Chungmu Stage 1 문제”라고 하면 33개가 정확하고**, **“Chungmu 시나리오의 전체 문제 정의”라고 하면 39개가 정확하다.**

별도의 “사람” 카테고리가 하나 더 있는 구조는 아니다. 사람 문제는 모두 `target_kind: person`인 SAR 직접형 또는 SAR 서사형이다.

## 2. 표 읽는 법

- `camera / scene`은 YAML의 정답 `camera_id`와 그 영상에 보이는 주요 구성을 함께 적은 것이다. 씬 설명은 공식 필드명이 아니라 화면 식별용 요약이다.
- `type`은 실제 제출값이다. SAR은 YAML의 pose가 `person_sitting` 등이어도 API 제출 `target_type`은 반드시 `person`이다.
- `landmark`의 첫 줄은 제출 문자열, 둘째 줄은 YAML 원본 `landmark_name`이다. 예: `picnic_table` / `picnic_table_01`.
- 분실물은 `relation`, SAR은 `situation`을 제출한다. SAR의 `support`는 상황 판정 근거를 확인하기 위한 YAML GT이며 별도 제출 필드는 아니다.
- `anchor`는 해당 landmark의 GT world 좌표, `target`은 대상의 GT world 좌표다.
- `target_object_id` 끝 숫자는 USD prim 이름의 끝 숫자와 항상 같지 않다. 같은 타입의 `resolved.instances[index]`를 가리키는 인덱스로 해석해야 한다.

## 3. CCTV 씬 아틀라스

| camera | 화면 식별용 씬 요약 | Stage 1 33개 선택 횟수 | 전체 39개 선택 횟수 |
|---|---|---:|---:|
| `rig_1_a` | 공원 휴게공간 · 피크닉 테이블/벤치 | 7 | 7 |
| `rig_2_b` | 벤치·쓰레기통 구역 · 미선택 디코이 | 0 | 0 |
| `rig_2_c` | 자판기 · 자전거 거치대 | 5 | 6 |
| `rig_3_b` | 넘어진 자전거 · 쓰레기통/피크닉 테이블 | 6 | 7 |
| `rig_3_c` | 택시 도로 · 벤치 | 4 | 6 |
| `rig_7_a` | 주차장 · 경찰차/소화전 | 11 | 13 |

`rig_2_b`는 CCTV로 존재하지만 Stage 1 33개와 RT 6개의 GT camera로 한 번도 선택되지 않는다.

## 4. 분실물 직접형 (LF) — 11개

| # | ID | 문제 원문 (`voice_command`) | 정답 camera / 씬 | 제출 type | 제출 landmark / YAML GT | relation 또는 situation | anchor GT | target GT / `target_object_id` |
|---:|---|---|---|---|---|---|---|---|
| 1 | `lf01_mug_table` | Find the mug on the picnic table. | `rig_1_a`<br><sub>공원 휴게공간 · 피크닉 테이블/벤치</sub> | `mug` | `picnic_table`<br><sub>YAML: `picnic_table_01`</sub> | `on` | `[-58.92264, 119.89141, 16.35574]` | `[-58.62067, 119.96513, 17.44791]`<br><sub>`rig_1_a/mug_0`</sub> |
| 2 | `lf02_tumbler_bench` | Find the tumbler on the bench. | `rig_1_a`<br><sub>공원 휴게공간 · 피크닉 테이블/벤치</sub> | `tumbler` | `bench`<br><sub>YAML: `bench_01`</sub> | `on` | `[-52.42612, 113.37388, 16.60000]` | `[-52.53231, 113.34800, 16.96660]`<br><sub>`rig_1_a/tumbler_0`</sub> |
| 3 | `lf03_tumbler_hydrant` | Find the tumbler near the fire hydrant in the parking lot. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `tumbler` | `hydrant`<br><sub>YAML: `hydrant_01`</sub> | `near` | `[-33.80530, 130.52874, 16.85787]` | `[-36.07784, 128.70259, 16.84848]`<br><sub>`rig_7_a/tumbler_1`</sub> |
| 4 | `lf04_sunblock_table` | Find the sunblock beside the picnic table. | `rig_1_a`<br><sub>공원 휴게공간 · 피크닉 테이블/벤치</sub> | `sunblock` | `picnic_table`<br><sub>YAML: `picnic_table_01`</sub> | `beside` | `[-58.92264, 119.89141, 16.35574]` | `[-57.85454, 120.84175, 16.40427]`<br><sub>`rig_1_a/sunblock_0`</sub> |
| 5 | `lf05_umbrella_vending` | Find the umbrella beside the vending machine. | `rig_2_c`<br><sub>자판기 · 자전거 거치대</sub> | `umbrella` | `vending_machine`<br><sub>YAML: `vending_machine_01`</sub> | `beside` | `[-62.80626, 135.53058, 16.20000]` | `[-61.04500, 136.08806, 16.27506]`<br><sub>`rig_2_c/umbrella_0`</sub> |
| 6 | `lf06_umbrella_fallenbike` | Find the umbrella next to the trash can. | `rig_3_b`<br><sub>넘어진 자전거 · 쓰레기통/피크닉 테이블</sub> | `umbrella` | `trash_can`<br><sub>YAML: `trash_can_01`</sub> | `near` | `[-76.58481, 96.19682, 15.70586]` | `[-77.66590, 96.58378, 15.80732]`<br><sub>`rig_3_b/umbrella_0`</sub> |
| 7 | `lf07_pencilcase_table` | Find the pencil case near the picnic table. | `rig_3_b`<br><sub>넘어진 자전거 · 쓰레기통/피크닉 테이블</sub> | `pencilcase` | `picnic_table`<br><sub>YAML: `picnic_table_02`</sub> | `near` | `[-80.72785, 97.24997, 15.65287]` | `[-77.25906, 97.11535, 15.86412]`<br><sub>`rig_3_b/pencilcase_0`</sub> |
| 8 | `lf08_umbrella_parking` | Find the umbrella dropped in the empty parking space. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `umbrella` | `police_car`<br><sub>YAML: `police_car_01`</sub> | `near` | `[-42.65816, 118.29972, 16.69391]` | `[-42.84795, 120.28879, 16.70770]`<br><sub>`rig_7_a/umbrella_0`</sub> |
| 9 | `lf09_tumbler_parking` | Find the tumbler dropped in the empty parking space. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `tumbler` | `police_car`<br><sub>YAML: `police_car_01`</sub> | `near` | `[-42.65816, 118.29972, 16.69391]` | `[-44.37683, 121.12430, 16.67988]`<br><sub>`rig_7_a/tumbler_2`</sub> |
| 10 | `lf10_tumbler_taxi` | Find the tumbler on the road next to the taxi. | `rig_3_c`<br><sub>택시 도로 · 벤치</sub> | `tumbler` | `taxi`<br><sub>YAML: `taxi_01`</sub> | `near` | `[-83.61583, 103.44018, 15.45533]` | `[-85.89217, 108.42254, 15.52783]`<br><sub>`rig_3_c/tumbler_0`</sub> |
| 11 | `lf11_umbrella_taxi` | Find the umbrella on the road next to the taxi. | `rig_3_c`<br><sub>택시 도로 · 벤치</sub> | `umbrella` | `taxi`<br><sub>YAML: `taxi_01`</sub> | `near` | `[-83.61583, 103.44018, 15.45533]` | `[-85.60271, 108.66516, 15.53514]`<br><sub>`rig_3_c/umbrella_0`</sub> |

## 5. 분실물 서사형 (HF) — 8개

| # | ID | 문제 원문 (`voice_command`) | 정답 camera / 씬 | 제출 type | 제출 landmark / YAML GT | relation 또는 situation | anchor GT | target GT / `target_object_id` |
|---:|---|---|---|---|---|---|---|---|
| 12 | `hf01_vending_umbrella` | After working out I grabbed a drink from the machine and left my umbrella standing next to it. | `rig_2_c`<br><sub>자판기 · 자전거 거치대</sub> | `umbrella` | `vending_machine`<br><sub>YAML: `vending_machine_01`</sub> | `beside` | `[-62.80626, 135.53058, 16.20000]` | `[-61.04500, 136.08806, 16.27506]`<br><sub>`rig_2_c/umbrella_0`</sub> |
| 13 | `hf02_table_pencilcase` | I was taking notes at an outdoor table and left my pencil case behind. | `rig_3_b`<br><sub>넘어진 자전거 · 쓰레기통/피크닉 테이블</sub> | `pencilcase` | `picnic_table`<br><sub>YAML: `picnic_table_02`</sub> | `near` | `[-80.72785, 97.24997, 15.65287]` | `[-77.25906, 97.11535, 15.86412]`<br><sub>`rig_3_b/pencilcase_0`</sub> |
| 14 | `hf03_table_mug` | I was reading with a coffee at an outdoor table and left my mug there. | `rig_1_a`<br><sub>공원 휴게공간 · 피크닉 테이블/벤치</sub> | `mug` | `picnic_table`<br><sub>YAML: `picnic_table_01`</sub> | `on` | `[-58.92264, 119.89141, 16.35574]` | `[-58.62067, 119.96513, 17.44791]`<br><sub>`rig_1_a/mug_0`</sub> |
| 15 | `hf04_park_sunblock` | I was resting in the sun at the park and left the sunblock I had been using. | `rig_1_a`<br><sub>공원 휴게공간 · 피크닉 테이블/벤치</sub> | `sunblock` | `picnic_table`<br><sub>YAML: `picnic_table_01`</sub> | `beside` | `[-58.92264, 119.89141, 16.35574]` | `[-57.85454, 120.84175, 16.40427]`<br><sub>`rig_1_a/sunblock_0`</sub> |
| 16 | `hf05_bikerack_umbrella` | After grabbing a drink from the vending machine I left my umbrella propped against its side. | `rig_2_c`<br><sub>자판기 · 자전거 거치대</sub> | `umbrella` | `vending_machine`<br><sub>YAML: `vending_machine_01`</sub> | `beside` | `[-62.80626, 135.53058, 16.20000]` | `[-61.04500, 136.08806, 16.27506]`<br><sub>`rig_2_c/umbrella_0`</sub> |
| 17 | `hf06_fallenbike_umbrella` | I stopped to throw something in the bin and left my umbrella leaning against the trash can. | `rig_3_b`<br><sub>넘어진 자전거 · 쓰레기통/피크닉 테이블</sub> | `umbrella` | `trash_can`<br><sub>YAML: `trash_can_01`</sub> | `near` | `[-76.58481, 96.19682, 15.70586]` | `[-77.66590, 96.58378, 15.80732]`<br><sub>`rig_3_b/umbrella_0`</sub> |
| 18 | `hf07_parking_tumbler` | I think I dropped my tumbler on the ground as I got out of my car in the parking lot. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `tumbler` | `police_car`<br><sub>YAML: `police_car_01`</sub> | `near` | `[-42.65816, 118.29972, 16.69391]` | `[-44.37683, 121.12430, 16.67988]`<br><sub>`rig_7_a/tumbler_2`</sub> |
| 19 | `hf08_parking_umbrella` | Getting out of my car I dropped my umbrella in the empty space next to it; find it in the parking lot. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `umbrella` | `police_car`<br><sub>YAML: `police_car_01`</sub> | `near` | `[-42.65816, 118.29972, 16.69391]` | `[-42.84795, 120.28879, 16.70770]`<br><sub>`rig_7_a/umbrella_0`</sub> |

## 6. 사람·상황 직접형 (SAR) — 8개

| # | ID | 문제 원문 (`voice_command`) | 정답 camera / 씬 | 제출 type | 제출 landmark / YAML GT | relation 또는 situation | anchor GT | target GT / `target_object_id` |
|---:|---|---|---|---|---|---|---|---|
| 20 | `sar01_trash_sit` | Find the person sitting on top of the trash can in the parking lot. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `person`<br><sub>YAML pose: `person_sitting`</sub> | `trash_can`<br><sub>YAML: `trash_can_01`</sub> | situation=`abnormal`<br><sub>support=`trash_can`</sub> | `[-38.18341, 134.45073, 16.80000]` | `[-37.93124, 134.61274, 17.05000]`<br><sub>`rig_7_a/person_sitting_0`</sub> |
| 21 | `sar02_vending_sit` | Find the person sitting on top of the vending machine. | `rig_2_c`<br><sub>자판기 · 자전거 거치대</sub> | `person`<br><sub>YAML pose: `person_sitting`</sub> | `vending_machine`<br><sub>YAML: `vending_machine_01`</sub> | situation=`abnormal`<br><sub>support=`vending_machine`</sub> | `[-62.80626, 135.53058, 16.20000]` | `[-62.77176, 135.78356, 17.93731]`<br><sub>`rig_2_c/person_sitting_0`</sub> |
| 22 | `sar03_bench_lie` | Find the person lying face-down on the bench near the taxi. | `rig_3_c`<br><sub>택시 도로 · 벤치</sub> | `person`<br><sub>YAML pose: `person_bending`</sub> | `bench`<br><sub>YAML: `bench_02`</sub> | situation=`abnormal`<br><sub>support=`bench`</sub> | `[-84.10907, 112.51235, 15.47582]` | `[-84.92714, 111.87660, 15.99720]`<br><sub>`rig_3_c/person_bending_0`</sub> |
| 23 | `sar04_table_lie` | Find the person lying on top of the picnic table. | `rig_3_b`<br><sub>넘어진 자전거 · 쓰레기통/피크닉 테이블</sub> | `person`<br><sub>YAML pose: `person_lying_down`</sub> | `picnic_table`<br><sub>YAML: `picnic_table_02`</sub> | situation=`abnormal`<br><sub>support=`picnic_table`</sub> | `[-80.72785, 97.24997, 15.65287]` | `[-80.93486, 96.48996, 16.82847]`<br><sub>`rig_3_b/person_lying_down_0`</sub> |
| 24 | `sar06_parking_collapsed` | Find the person collapsed on the parking lot floor. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `person`<br><sub>YAML pose: `person_lying_down`</sub> | `hydrant`<br><sub>YAML: `hydrant_01`</sub> | situation=`emergency`<br><sub>support=`parking`</sub> | `[-33.80530, 130.52874, 16.85787]` | `[-35.31731, 129.45959, 16.82526]`<br><sub>`rig_7_a/person_lying_down_0`</sub> |
| 25 | `sar07_parking_prone` | Find the person lying face-down and motionless in the parking lot. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `person`<br><sub>YAML pose: `person_prone`</sub> | `bicycle`<br><sub>YAML: `bicycle_01`</sub> | situation=`accident`<br><sub>support=`parking`</sub> | `[-48.56376, 128.79317, 17.02055]` | `[-49.27203, 125.65040, 16.73964]`<br><sub>`rig_7_a/person_prone_0`</sub> |
| 26 | `sar08_fallenbike_down` | Find the person fallen on the ground next to the toppled bicycle. | `rig_3_b`<br><sub>넘어진 자전거 · 쓰레기통/피크닉 테이블</sub> | `person`<br><sub>YAML pose: `person_bending`</sub> | `bicycle`<br><sub>YAML: `bicycle_01`</sub> | situation=`emergency`<br><sub>support=`bicycle`</sub> | `[-78.86305, 101.85303, 16.11596]` | `[-76.88852, 102.46404, 16.05381]`<br><sub>`rig_3_b/person_bending_0`</sub> |
| 27 | `sar09_park_bending` | Find the person bent over and motionless near the trash can in the park rest area. | `rig_1_a`<br><sub>공원 휴게공간 · 피크닉 테이블/벤치</sub> | `person`<br><sub>YAML pose: `person_bending`</sub> | `trash_can`<br><sub>YAML: `trash_can_01`</sub> | situation=`emergency`<br><sub>support=`trash_can`</sub> | `[-53.99992, 111.80938, 16.85219]` | `[-55.23807, 111.34714, 16.77229]`<br><sub>`rig_1_a/person_bending_0`</sub> |

## 7. 사람·상황 서사형 (SAR-H) — 6개

| # | ID | 문제 원문 (`voice_command`) | 정답 camera / 씬 | 제출 type | 제출 landmark / YAML GT | relation 또는 situation | anchor GT | target GT / `target_object_id` |
|---:|---|---|---|---|---|---|---|---|
| 28 | `sarH1_park_needhelp` | People are relaxing at the park and one is even lying on the grass; find the one near the trash can who is bent over and not moving and actually needs help. | `rig_1_a`<br><sub>공원 휴게공간 · 피크닉 테이블/벤치</sub> | `person`<br><sub>YAML pose: `person_bending`</sub> | `trash_can`<br><sub>YAML: `trash_can_01`</sub> | situation=`emergency`<br><sub>support=`trash_can`</sub> | `[-53.99992, 111.80938, 16.85219]` | `[-55.23807, 111.34714, 16.77229]`<br><sub>`rig_1_a/person_bending_0`</sub> |
| 29 | `sarH2_parking_bikeacc` | In the parking lot a person is down next to a bicycle and looks like they had an accident; find them. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `person`<br><sub>YAML pose: `person_prone`</sub> | `bicycle`<br><sub>YAML: `bicycle_01`</sub> | situation=`accident`<br><sub>support=`parking`</sub> | `[-48.56376, 128.79317, 17.02055]` | `[-49.27203, 125.65040, 16.73964]`<br><sub>`rig_7_a/person_prone_0`</sub> |
| 30 | `sarH3_climbed_machine` | Someone is up where people should not be; find the person who climbed on top of the drink machine. | `rig_2_c`<br><sub>자판기 · 자전거 거치대</sub> | `person`<br><sub>YAML pose: `person_sitting`</sub> | `vending_machine`<br><sub>YAML: `vending_machine_01`</sub> | situation=`abnormal`<br><sub>support=`vending_machine`</sub> | `[-62.80626, 135.53058, 16.20000]` | `[-62.77176, 135.78356, 17.93731]`<br><sub>`rig_2_c/person_sitting_0`</sub> |
| 31 | `sarH4_parking_trashsit` | In the parking lot, find the person who climbed up and is sitting on top of the trash can. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `person`<br><sub>YAML pose: `person_sitting`</sub> | `trash_can`<br><sub>YAML: `trash_can_01`</sub> | situation=`abnormal`<br><sub>support=`trash_can`</sub> | `[-38.18341, 134.45073, 16.80000]` | `[-37.93124, 134.61274, 17.05000]`<br><sub>`rig_7_a/person_sitting_0`</sub> |
| 32 | `sarH5_taxi_benchlie` | Near the taxi, find the person lying down on top of the bench. | `rig_3_c`<br><sub>택시 도로 · 벤치</sub> | `person`<br><sub>YAML pose: `person_bending`</sub> | `bench`<br><sub>YAML: `bench_02`</sub> | situation=`abnormal`<br><sub>support=`bench`</sub> | `[-84.10907, 112.51235, 15.47582]` | `[-84.92714, 111.87660, 15.99720]`<br><sub>`rig_3_c/person_bending_0`</sub> |
| 33 | `sarH6_most_critical` | Several people are in the parking lot; one is completely face-down and immobile; find the most critical one. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `person`<br><sub>YAML pose: `person_prone`</sub> | `bicycle`<br><sub>YAML: `bicycle_01`</sub> | situation=`accident`<br><sub>support=`parking`</sub> | `[-48.56376, 128.79317, 17.02055]` | `[-49.27203, 125.65040, 16.73964]`<br><sub>`rig_7_a/person_prone_0`</sub> |

## 8. Stage 2 회수형 (RT) — 6개 부록

아래 6개는 YAML 전체 39개에는 포함되지만 Stage 1의 33문항에는 포함되지 않는다. Starter Kit 기본 `problems.yaml`에서는 `rt01`만 enabled이고, Stage 2는 첫 번째 enabled 엔트리를 사용한다.

| # | ID | 문제 원문 (`voice_command`) | 정답 camera / 씬 | 제출 type | 제출 landmark / YAML GT | relation 또는 situation | anchor GT | target GT / `target_object_id` |
|---:|---|---|---|---|---|---|---|---|
| 34 | `rt01_parking_tumbler` | Pick up the tumbler dropped on the parking lot floor and bring it to the owner. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `tumbler` | `police_car`<br><sub>YAML: `police_car_01`</sub> | `near` | `[-42.65816, 118.29972, 16.69391]` | `[-44.37683, 121.12430, 16.67988]`<br><sub>`rig_7_a/tumbler_2`</sub> |
| 35 | `rt02_parking_umbrella` | Pick up the umbrella in the empty parking space and bring it to the owner. | `rig_7_a`<br><sub>주차장 · 경찰차/소화전</sub> | `umbrella` | `police_car`<br><sub>YAML: `police_car_01`</sub> | `near` | `[-42.65816, 118.29972, 16.69391]` | `[-42.84795, 120.28879, 16.70770]`<br><sub>`rig_7_a/umbrella_0`</sub> |
| 36 | `rt03_vending_umbrella` | Pick up the umbrella beside the vending machine and bring it to the owner. | `rig_2_c`<br><sub>자판기 · 자전거 거치대</sub> | `umbrella` | `vending_machine`<br><sub>YAML: `vending_machine_01`</sub> | `beside` | `[-62.80626, 135.53058, 16.20000]` | `[-61.04500, 136.08806, 16.27506]`<br><sub>`rig_2_c/umbrella_0`</sub> |
| 37 | `rt04_taxi_tumbler` | Pick up the tumbler on the road next to the taxi and bring it to the owner. | `rig_3_c`<br><sub>택시 도로 · 벤치</sub> | `tumbler` | `taxi`<br><sub>YAML: `taxi_01`</sub> | `near` | `[-83.61583, 103.44018, 15.45533]` | `[-85.89217, 108.42254, 15.52783]`<br><sub>`rig_3_c/tumbler_0`</sub> |
| 38 | `rt04b_taxi_umbrella` | Pick up the umbrella on the road next to the taxi and bring it to the owner. | `rig_3_c`<br><sub>택시 도로 · 벤치</sub> | `umbrella` | `taxi`<br><sub>YAML: `taxi_01`</sub> | `near` | `[-83.61583, 103.44018, 15.45533]` | `[-85.60271, 108.66516, 15.53514]`<br><sub>`rig_3_c/umbrella_0`</sub> |
| 39 | `rtC2_fallenbike_pencil` | Pick up the pencil case near the picnic table and bring it to the owner. | `rig_3_b`<br><sub>넘어진 자전거 · 쓰레기통/피크닉 테이블</sub> | `pencilcase` | `picnic_table`<br><sub>YAML: `picnic_table_02`</sub> | `near` | `[-80.72785, 97.24997, 15.65287]` | `[-77.25906, 97.11535, 15.86412]`<br><sub>`rig_3_b/pencilcase_0`</sub> |

## 9. 동일 GT를 공유하는 문제 묶음

같은 시각 대상을 공유하는 직접형·서사형을 함께 돌리면, 카메라/검출 문제인지 언어 해석 문제인지 분리하기 쉽다.

| 공유 GT | 문제 ID |
|---|---|
| `rig_1_a` · `rig_1_a/mug_0` | `lf01_mug_table`, `hf03_table_mug` |
| `rig_1_a` · `rig_1_a/sunblock_0` | `lf04_sunblock_table`, `hf04_park_sunblock` |
| `rig_2_c` · `rig_2_c/umbrella_0` | `lf05_umbrella_vending`, `hf01_vending_umbrella`, `hf05_bikerack_umbrella` |
| `rig_3_b` · `rig_3_b/umbrella_0` | `lf06_umbrella_fallenbike`, `hf06_fallenbike_umbrella` |
| `rig_3_b` · `rig_3_b/pencilcase_0` | `lf07_pencilcase_table`, `hf02_table_pencilcase` |
| `rig_7_a` · `rig_7_a/umbrella_0` | `lf08_umbrella_parking`, `hf08_parking_umbrella` |
| `rig_7_a` · `rig_7_a/tumbler_2` | `lf09_tumbler_parking`, `hf07_parking_tumbler` |
| `rig_7_a` · `rig_7_a/person_sitting_0` | `sar01_trash_sit`, `sarH4_parking_trashsit` |
| `rig_2_c` · `rig_2_c/person_sitting_0` | `sar02_vending_sit`, `sarH3_climbed_machine` |
| `rig_3_c` · `rig_3_c/person_bending_0` | `sar03_bench_lie`, `sarH5_taxi_benchlie` |
| `rig_7_a` · `rig_7_a/person_prone_0` | `sar07_parking_prone`, `sarH2_parking_bikeacc`, `sarH6_most_critical` |
| `rig_1_a` · `rig_1_a/person_bending_0` | `sar09_park_bending`, `sarH1_park_needhelp` |

## 10. 문제별 성능 점검 권장 순서

1. 명령만 보고 `type/person`, `landmark`, `relation/situation`을 파싱한다.
2. 표의 GT camera와 모델이 선택한 camera를 비교한다.
3. 올바른 camera 안에서 anchor 검출 오차와 target 검출 오차를 따로 기록한다.
4. 직접형이 맞고 동일 GT 서사형만 틀리면 언어 파서/VLM 문제, 둘 다 틀리면 카메라 라우팅·검출·좌표화 문제를 먼저 의심한다.
5. Chungmu 좌표나 ID를 제출 코드에 하드코딩하지 말고 회귀 테스트 정답으로만 사용한다. 실제 평가 배경은 달라질 수 있다.

## 11. 검증 출처

- [Official Starter Kit `problems.yaml`](https://github.com/marc-challenge/marc-starter-kit/blob/main/problems.yaml) — Stage 1 33개와 Stage 2 6개의 ID 분리
- [Official Technical Guide](https://marc-challenge.github.io/marc-dev-guide/en/technical-guide.html) — Chungmu의 expected 포함 자가채점 용도, 문제 선택, Stage 1/2 평가 축
- [Official API Reference](https://marc-challenge.github.io/marc-dev-guide/en/api-reference.html) — `GroundingResult`의 `camera_id`, `target_type`, `landmark`, `relation/situation`, `anchor_coord`, `target_coord` 규격
- [Official Challenge page](https://marc-challenge.github.io/en/challenge/) — Stage 1 grounding과 Stage 2 retrieval의 범위
- 로컬 원본 `marc2026_chungmu.yaml` version `2026.2` — 39개 expected와 mission별 landmark/object 좌표를 전부 재계산

> 범위 제한: 카메라 ID와 좌표는 공개 연습 배경 Chungmu 전용 GT다. 실제 예선/본선 배경에도 그대로 적용된다고 가정하면 안 된다.
