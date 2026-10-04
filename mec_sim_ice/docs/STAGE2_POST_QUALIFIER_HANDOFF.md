# Stage2 성능 개선 인계 문서 (qualifier 이후, 2026-10-04)

> **이 문서를 읽는 Claude Code 세션에게**: Isaac Sim이 있는 환경에서 이어서 작업하기 위한
> 맥락이다. 아래 변경은 맥북(Isaac Sim 없음)에서 오프라인 하네스로만 검증됐다. 이 세션의
> 1차 목표는 **Isaac Sim에서 실측해 오프라인 결과를 확인하고, 엑셀의 `Isaac_*` 시트를 채우는 것**이다.
> 범위: Stage2 주행(로컬 회피, 주행 시간)과 파지(대상 파악, 정밀 파지). Stage1과 공유하는
> grounding 점수 개선은 범위 밖이다.

---

## 0. 저장소 상태

| 항목 | 값 |
|---|---|
| 기준 | `qualifier` 태그 = `ed416bb` (= `origin/main`, `origin/master`) |
| 이 작업 | qualifier 위에 커밋 (이 문서와 같은 커밋) |
| 폐기한 작업 | 원격 브랜치 `post-qualifier`(`c71a243`, "leg1 거리 적응형 inflation" + depth 덩어리 분리/높이 0.6m 필터)는 **적용하지 않기로 결정하고 원격에서 삭제했다** (2026-10-04). 필요하면 `git branch recover c71a243`으로 복구할 수 있다 (로컬 객체가 남아 있는 동안). 이 문서의 주행/파지 변경은 그 커밋 없이 qualifier 위에서 독립적으로 만든 것이다. |
| 원격의 다른 브랜치 | `origin/feature/init` `6df1249` "confidence socre error fix" (Stage1 쪽) |
| 주석만 추가된 파일 | `depth_ground_veto.py`, `lidar_obstacles.py`, `waypoint_avoid.py` 그리고 `participant_app.py`/`mock_agent_navigation.py`의 `[파일 출처 안내]`·`[팀 추가]` 블록은 이 세션 이전부터 있던 **주석 전용** 변경이다. 주석과 docstring을 제외한 AST 비교로 qualifier와 코드가 같다는 것을 확인했다. |

---

## 1. qualifier 시점의 Stage2 동작 (요약)

`participant_app._drive()`의 흐름은 다음과 같다.

1. reveal 대기 → 팔 HOME
2. **leg1**: 물체 앞 0.58m standoff까지 `_navigate_to` (inflation 0.10, tol 0.45)
3. `_align_to_object` → `_approach_straight` (arm_base fwd 0.40까지 직진)
4. 파지 최대 3회: tf로 계산한 추정 위치 → `_locate_object_by_depth`로 보정 → `run_pick_sequence`
5. **leg2**: owner까지 `_navigate_to` → `task_complete`

`_navigate_to`의 구성:
- 경로 계획: 점유지도 A* + 라이다 장애물 원(disc) + 정체 기억(stall memory)
- 추종: `PathFollower`/`GoalSeekController` P제어
- 회피 단계: `relocate`(웨이포인트 원 우회) → 반응형 브레이크 → 정체 감지 시 재계획 → 후진 탈출

---

## 2. 이번에 바꾼 것

### 2-1. 주행 (`demo/mock_agent_navigation.py`, `demo/participant_app.py`)

| 요소 (ablation id) | 내용 | 위치 |
|---|---|---|
| `speed` | `MAX_LINEAR` 0.8 → **1.1 m/s** (SDK 상한 1.5) | `mock_agent_navigation.py:88` |
| `profile` | 감속을 `0.5·dist`에서 **`v = sqrt(2·0.5·(남은거리 − tol))`, 하한 `V_MIN=0.12`**로 변경. `PathFollower`가 **남은 경로 전체 길이**를 넘겨주므로, 중간 웨이포인트에서도 마지막 정지를 대비해 감속한다 | `:95`, `:197` |
| `lat_acc` | 횡가속 상한 `|v·ω| ≤ 1.0 m/s²` (예전 0.8 m/s × 1.2 rad/s의 최악값 0.96과 같은 수준) | `:99` |
| `long_acc` | 종방향 가속 0.6 / 감속 1.0 m/s² 제한. 비상 브레이크에는 적용하지 않는다. 제한이 걸린 뒤 angular를 `cap_angular`와 횡가속 상한으로 다시 자른다 | `participant_app.py:195`, `:1348`, `:1643` |
| `brake` | 반응형 브레이크 거리 = `max(0.6, 0.25 + 0.1 + v·0.3 + v²/(2·1.0))`. v는 직전 명령 속도 | `:163`, `:1340`, `:1606` |
| `approach` | `_approach_straight` 속도를 `clamp(0.8·(fwd−0.40), 0.06, 0.30)`로 변경 (기존 0.12 고정). lat ≥ 0.15이면 0.06 | `:232` |
| `halt` | 고정 정지 대기(5×0.05s, 그리고 접근 후 5×0.1s 중복)를 `_halt()`로 교체. pose로 실제 정지를 확인하면 바로 진행하고, 최대 0.5s | `:1303` |

**시도했다가 되돌린 것**: follower 컨트롤러의 도착 허용치를 leg tol(0.45)로 맞추는 변경.
목표 셀이 막혀 경로가 목표보다 짧게 끝나면, follower의 done 판정이 켜졌다 꺼졌다를 반복했다.
그 결과 fallback 컨트롤러와 follower가 서로 반대 방향을 가리켜 제자리에서 진동했다 (rt01 leg1 30s → 45s).
기존 0.7m 허용치가 히스테리시스 역할을 하고 있었으므로 그대로 둔다. `participant_app.py:1388`에 주석으로 남겼다.

### 2-2. 파지 (`demo/grasp_shape.py` 신규, `participant_app.py:912`, `arm_pick.py`)

`_locate_object_by_depth`는 0.5m 윈도우 안에서 바닥보다 솟은 점들을 모은다. 기존에는 이 점들의 **평균**을 파지점으로 썼고, 이제 `grasp_shape.estimate_grasp()`를 쓴다. 각 단계는 모듈 상단의 스위치로 켜고 끌 수 있다.

| 스위치 | 내용 |
|---|---|
| `USE_CLUSTER` | 위에서 내려다본 2cm 격자로 연결 성분을 나누고, nav 추정 위치에 중심이 가장 가까운 덩어리만 남긴다. 라이브 호스트에 scipy가 없어서 numpy만 사용한다 |
| `USE_TOP_BAND_CENTER` | 서 있는 물체는 윗면 2cm 띠의 외접 사각형 중심을 쓴다. 카메라 쪽 앞면만 보여서 생기는 쏠림을 제거한다 |
| `USE_LYING_AXIS` | 누운 물체는 능선(높이 60% 이상인 점들)의 PCA로 축을 구하고, 보이는 구간의 중앙을 축 높이(h/2)에서 잡는다. 팔이 안 닿으면 축을 따라 로봇 쪽으로 당긴다 |
| `USE_YAW` | 긴 축을 가로질러 닫히도록 그리퍼 yaw를 정한다. 상자는 윗면이 길쭉할 때만 적용하고, 둥근 물체는 기본 yaw 0.52를 유지한다 |

- **yaw 관계식** (URDF placo FK로 실측): 손가락 닫힘축 각도 = 90° + `grasp_yaw` (mod 180°). 긴 축 각도가 θ이면 `grasp_yaw = θ`.
- `arm_pick.pick_and_place`는 `grasp_xy`의 4번째 값을 `grasp_yaw`로 쓴다. 단, `pnp_params.json`에 그 물체 타입의 `grasp_yaw` 키가 있으면 튜닝된 상수가 우선한다 (`grasp_down`과 같은 규칙).
- **안전장치**:
  - 보정된 위치가 평균에서 12cm 이상 벗어나면 평균으로 폴백한다 (누운 물체는 축에 수직인 성분만 검사).
  - 결과가 유한하지 않으면 None을 반환하고, 기존 경로가 처리한다.
- **팔 실행 가능성** (오프라인 ArmSim): yaw −89°~+89° × 위치 4곳 = 28개 조합 모두 시퀀스를 끝까지 완료했다. IK 방향 오차는 최대 0.36°.

---

## 3. 오프라인 결과 (논문 ablation 표: `mec_sim_ice/docs/ablation/stage2_ablation.xlsx`)

### 주행 (결정론적, 6문항 시뮬 시간 합계)
| 설정 | plain (s) | 장애물 숨김 (s) | 비고 |
|---|---|---|---|
| Baseline (qualifier) | 634.9 | 659.8 | 수정 전 qualifier 코드 실측치와 정확히 일치 (패치가 충실하다는 확인) |
| **Full** | **531.0 (−16.4%)** | **588.7 (−10.8%)** | 6/6 성공, 접촉 0 |
| w/o speed | +107.3 | +80.9 | 시간 기여가 가장 큼 |
| w/o halt | +4.2 | +2.8 | 접근/파지 구간에서 3.6s 절약 |
| w/o profile / approach | −1.3 / −0.8 | −4.8 / −13.4 | **오프라인에서는 시간 이득 없음**. 장애물 숨김 시나리오의 차이는 대부분 leg2 경로가 달라져서 생긴 것 |
| w/o long_acc | −6.6 | −38.7 | 끄면 빨라지지만 명령 가속도가 11 m/s²(vs 0.6), 브레이크가 159회(vs 46). 안전과 시간의 트레이드오프 |
| w/o lat_acc | 0 | 0 | 이 경로들에서는 상한이 한 번도 걸리지 않음 (안전 가드 역할만) |
| w/o brake | 0 | −9.7 | 접촉은 둘 다 0. 안전 이득은 오프라인 모델(섀시 감속 2 m/s²)로는 측정할 수 없음 |

### 파지 (합성 depth 6,000프레임, 짝지은 비교, 포획 모델은 가정)
| 설정 | 포획률 | 방해물 있음 |
|---|---|---|
| Baseline (평균 + yaw 0.52) | 36.6% | 9.1% |
| **Full** | **99.4%** | **98.9%** |
| w/o USE_CLUSTER | 56.5% | 13.2% |
| w/o USE_YAW | 63.6% | (누운 물체 yaw 오차 중앙값 44°) |
| w/o USE_TOP_BAND_CENTER | 90.8% | (서 있는 상자 56%) |
| w/o USE_LYING_AXIS | 93.7% | (누운 우산 70%) |

**한계**:
- 대상이 큰 물체에 5cm 이내로 붙어 있으면 하나의 덩어리로 합쳐져 분리되지 않는다 (누운 우산 + 통: 약 18%).
- 연필은 높이가 약 1cm라 "바닥보다 3cm 이상 솟은 점" 조건에 걸리지 않는다. 이번 개선과 무관하다.

---

## 4. Isaac Sim에서 할 일 (우선순위 순)

1. **회귀 확인**: rt01~rtC2를 기존 방식으로 실행해서 6/6 완주하는지 본다. 로그에서 확인할 것:
   - `[Stage2] lidar: obstacle within X m ahead (v=..) -> braking`: 브레이크 거리가 속도에 비례하는지
   - `follower done ...`, `stall detected`, `brake held`: 이 횟수가 qualifier 때보다 늘었는지
2. **전복과 흔들림**: 1.1 m/s 직진, 코너, 급정지(브레이크)에서 차체 pitch/roll을 확인한다. IMU `get_imu()`를 로깅하면 좋다.
   - 문제가 보이면 조정 순서: `STAGE2_ACCEL_MPS2` → `MAX_LAT_ACCEL` → `MAX_LINEAR`
   - `REACT_BRAKE_DECEL=1.0`은 추정치다. 1.1 m/s에서 브레이크를 건 뒤 실제 정지거리를 재서 보정한다.
3. **카메라 장착 tf 기록**: `arm_base_link → Base_Camera_Center`의 translation/rotation과 depth `camera_info`(해상도, K)를 출력해 둔다. 그 값을 `grasp_shape_eval.py`의 `MOUNTS`/`HFOV`에 넣고 다시 돌리면 오프라인 결과가 더 정확해진다.
4. **파지 실측**: 로그 `[Stage2] depth grasp shape=... yaw=...`, `[grasp-yaw] measured ...`를 본다.
   - 특히 누운 우산(rt02, rt03, rt04b)에서 yaw 적용 후 파지 성공률과 **바구니 투입**(긴 물체를 회전된 상태로 놓을 때)을 확인한다.
   - 성공한 물체·자세 조합의 `grasp_down`/`grasp_yaw`는 `pnp_params.json`에 상수로 고정하는 것을 검토한다.
5. **ablation 실측**: 엑셀의 `Isaac_Drive`/`Isaac_Grasp` 시트(노란 칸, 설정당 3회)를 채운다.
   - 전 설정을 다 돌리기 어려우면 Baseline / Full / 기여도가 큰 요소(speed, halt, USE_CLUSTER, USE_YAW)부터 한다.
   - 요소 끄는 방법:
     - 주행: `stage2_drive_ablation.py`의 `apply()`와 같은 값을 임시로 적용
     - 파지: `grasp_shape.USE_*`를 False로

## 5. 오프라인 도구 재실행

```bash
python3 mec_sim_ice/vla_grounding/validation/stage2_sim.py [--hidden-obstacles] [--problem rt01_parking_tumbler --verbose]
python3 mec_sim_ice/vla_grounding/validation/stage2_drive_ablation.py          # -> runs/stage2_drive_ablation.json (~3분)
python3 mec_sim_ice/vla_grounding/validation/grasp_shape_eval.py [--off USE_YAW]
python3 mec_sim_ice/vla_grounding/validation/grasp_shape_ablation.py --trials 200   # -> runs/stage2_grasp_ablation.json (~2분)
python3 mec_sim_ice/vla_grounding/validation/make_stage2_ablation_xlsx.py      # -> docs/ablation/stage2_ablation.xlsx (openpyxl 필요)
```
`--hint-noise 0.5` 시나리오의 "빗나감"은 qualifier 때도 같았다. 오프라인에서는 depth 보정이 없기 때문이다.

## 6. 다음 단계 (사용자가 정한 순서)

1. (현재) 위 변경을 Isaac Sim에서 검증하고, 좋아지는 부분만 채택한다.
2. 로봇 base 카메라에 **Stage1 YOLO를 파인튜닝**해 target_type을 식별한다. 방해물 제거, 그리고 접근 중에 hint를 보정하는 데 쓴다.
3. **지역 플래너**(DWA/MPPI 계열)로 이벤트 기반 회피(브레이크 → 재계획)를 대체하거나 보강한다.

아직 하지 않은 후보 (처음 제안한 10가지 중):
- 그리퍼 카메라로 `pre` 자세에서 다시 측정하는 폐루프 보정 (⑨)
- 실패 원인별 재시도 전략과 물체별 파라미터 자동 튜닝 (⑩)
- leg1 도착 자세(헤딩)를 계획에 포함 (⑤)
- 4WS 크랩 주행은 사용자가 제외함
