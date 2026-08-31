# 사람 target 발목 세로 보정 — prostrate 자세 (2026-08-31)

Stage-1 사람 문항의 `target_coord` 를, 눕/엎/굽힘 자세에서 발목 keypoint 의
**세로 픽셀도** 써서 실제 발 위치로 옮긴다. 라이브 실측 **+4.5점** (33라운드),
회귀 0. 씬 하드코딩 없음.

브랜치: `ptj/stage1-vla-integration`. 대상 파일: `mec_sim_ice/vla_grounding/`
(`grounding.py`, `pose_keypoints.py`, `fallback.py`).

---

## 1. 문제

- **플랫폼 GT = 사람 prim 원점 = 발.** 시나리오가 눕/엎/굽힘 자세를 X축
  ±90° 회전으로 만드는데 pivot 이 원점(발)이라, 몸이 발에서 1.7m 뻗어 나간다.
- 기존 `_foot_adjusted_point` 는 발목 keypoint 의 **가로(u)만** 쓰고 **세로는
  bbox 아랫변(y2)** 을 썼다. 근거: "얕은 카메라에서 발목 세로 역투영은 오차
  증폭" (서 있는 사람 기준 맞는 판단).
- 그런데 **prostrate 자세**(몸이 수평)에서는 bbox 아랫변 = **카메라에 가장
  가까운 몸 부위**(엎어지면 머리, 굽히면 손·머리)이고 발은 반대쪽 끝이다.
  → 제출 좌표가 발에서 세로로 0.5~0.8m 어긋난다.

### 왜 이게 target=0 을 만드나 (라이브 채점)

`target` 축(배점 40%) 은 **귀속 게이트**: 제출좌표가 GT-타겟보다 **다른 물체**
(같은 미션의 모든 객체, 랜드마크 제외)에 더 가까우면 → 거리 무관 0점.

- **sar06** (person_lying): 옆에 cracker_box 0.6m. 제출이 발 대신 몸 쪽으로
  밀려 box 에 0.36m < 사람 0.57m → 0점.
- **sar03 / sarH5** (person_bending, rig_3_c): 동일 패턴.

---

## 2. 수정

`POSE_CLASS`(situation_resolver) 가 `"prostrate"` 인 자세
(`person_bending` / `person_lying_down` / `person_prone`) 에서만:

1. **발목 keypoint 의 세로(v)도 사용** — bbox 아랫변 대신.
   근거: prostrate 자세는 발목 관절이 지면 높이에 있어서 세로 픽셀 증폭이
   없다 (서 있는 사람과 다름).
2. **footprint push 생략** — 발목이 곧 제출점이라 카메라 반대로 밀 이유가 없다.
3. **keypoint 오류 가드** — 발목-세로 결과가 접지선(y2) 결과에서 **2.5m** 넘게
   벗어나면 접지선으로 폴백 (사람 몸길이 1.7m 기준, 명백한 오검출 컷).

upright / seated / low 자세는 무변경 (발이 원래 bbox 아래에 있어 기존 로직이 맞음).

### 파일별

| 파일 | 변경 |
|---|---|
| `pose_keypoints.py` | `AnkleKeypointEstimator.foot_u()` 반환값 `float` → `(u, v)` 튜플 |
| `grounding.py` | `Detection` 에 `foot_v: Optional[float]=None` 필드; `_foot_adjusted_point` 에 prostrate 분기 (+가드) |
| `fallback.py` | `pose_augment` 가 `(u,v)` 언팩 (bare-float 하위호환 유지) |

---

## 3. 라이브 실측 (feature/init 등가 코드, 33라운드)

| | 평균 |
|---|---|
| baseline | 70.31 |
| **발목 세로 보정** | **74.81** (+4.50) |

| 라운드 | base → fix | |
|---|---|---|
| sar03_bench_lie | 40.3 → **89.6** | person_bending, target 0→1 |
| sar06_parking_collapsed | 48.7 → **98.8** | person_lying, target 0→1 |
| sarH5_taxi_benchlie | 40.3 → **89.4** | person_bending, target 0→1 |
| sar09 / sarH1 | 47.4 → 47.4 | person_bending(rig_1_a) — cola_can 이 사람에서 0.09m, 사실상 불가 |
| 나머지 28라운드 | byte-identical | 회귀 0 |

---

## 4. 하드코딩 / 안전성

- **씬 종속 상수 0** — 좌표·yaml·chungmu 값 없음. 자세 판정은 기존
  `POSE_CLASS` 재사용. `2.5m` 가드는 사람 몸길이(1.7m) 기준의 일반 상한.
- **순수 추가 계층** — `foot_v` 없거나(pose 모델 미탐지/비활성) prostrate 아니면
  기존 동작 그대로. `_foot_adjusted_point` 진입 조건도 기존과 동일.
- `foot_u()` 반환 타입 변경의 유일 소비자는 `pose_augment` 하나, 하위호환 분기 포함.
- 테스트: `test_grounding` / `test_command_parser` / `test_situation_resolver` 전부 PASS.
- **게이트 없음 — `docker compose up --build` 로 바로 적용.** (env 변수 불필요)

## 5. 미해결

- `sar09` / `sarH1`: 사람과 cola_can 이 0.09m → 어떤 좌표 방법으로도 귀속 게이트
  통과 불가. target 탐지·인스턴스 판별 쪽 문제.
