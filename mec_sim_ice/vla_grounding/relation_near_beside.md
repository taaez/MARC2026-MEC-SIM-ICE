# relation `near` / `beside` 판정 근거 조사 (2026-08-29)

Stage1 `relation` 오답이 **전부 `beside` 쌍**에 몰려 있어 원인을 파헤쳤다.
결론부터: **공개된 데이터만으로는 `near`와 `beside`를 가르는 기하 규칙이 존재하지 않는다.**
중심거리 · 원 표면간극 · 방향 있는 박스(OBB) 면-면 거리 **세 가지 방식 전부 반증**됐다.

3D 라벨/에셋 파이프라인 쪽에 확인이 필요해 조사 내용을 정리한다.
**물어보고 싶은 것은 §7에 모아 뒀다.**

---

## 0. 왜 조사했나

최신 오프라인 채점(`validation/runs/offline_2026-08-28_after-zfix.json`, Stage1 33라운드):

| 필드 | 점수 | 가중치 |
|---|---:|---:|
| camera | 0.758 | 0.10 |
| object_type | 0.879 | 0.15 |
| landmark | 0.939 | 0.10 |
| **relation** | **0.879** | 0.10 |
| anchor | 0.524 | 0.15 |
| target | 0.523 | 0.40 |

`relation` 오답 3건이 **전부 GT가 `beside`인데 우리가 `near`로 낸 것**이다:

- `hf01_vending_umbrella` — "standing next to it"
- `hf04_park_sunblock` — 관계어 없음
- `hf05_bikerack_umbrella` — "propped against its side"

3건 모두 camera / landmark / object_type 은 맞았다. relation 만 틀렸다.

---

## 1. 데이터 소스 — 공개된 관계 GT는 충무 25문항이 전부

| 소스 | 관계 문항 | 판정 |
|---|---:|---|
| `content-image-2026.R03/marc2026_chungmu.yaml` | 25 | ✅ 유일한 깨끗한 GT |
| `demo/mock_demo_data.yaml` | 19 | ⚠️ 같은 씬 + 의도적 오답 주입 |
| `records/*/meta.json` | — | 우리 캡처(카메라 파라미터), GT 아님 |
| `problems.yaml` | — | 문항 on/off 스위치 |
| `PUBLIC_SCENARIOS.txt` | — | 공개 시나리오는 chungmu + demo **둘뿐** |

### 39 → 33 → 25 → 11

```
39  challenge_problems
├─ 33  Stage1 (kind: grounding)
│   ├─ 19  target_kind=object  → relation 있음   (lf 11 + hf 8)
│   └─ 14  target_kind=person  → relation = None (sar 14, situation 으로 대체)
└─  6  Stage2 (kind: retrieval) → relation 있음   (rt 6)

relation 보유 문항 = 19 + 6 = 25
고유 (landmark, target) 물리 쌍 = 11
```

**25문항이 실제로는 11쌍이다.** lf / hf / rt 가 같은 물리 배치를 다르게 물어본다.

### mock_demo_data.yaml 은 새 정보가 없다

`_generated_from: marc2026_demo.yaml` 이지만 좌표가 충무와 완전히 동일하다(같은 씬).
relation 라벨이 충무와 16/19 일치하고, 어긋나는 3건이 문서에 명시된 의도적 오답이다:

```
pencilcase@picnic_table   chungmu: near  → mock: beside   ✗
hf02 pencil case behind   chungmu: near  → mock: beside   ✗
lf08 umbrella parking     chungmu: near  → mock: beside   ✗ (target_type 도 cola_can 으로 오염)
```

landmark 필드도 오염돼 있다(sunblock@bench, umbrella@bench 등).

---

## 2. GT 성질 ① — relation 은 문항이 아니라 **물리 쌍**에 붙어 있다

같은 (landmark 인스턴스, target 인스턴스) 쌍은 문항이 달라도 **항상 같은 relation** 이다.
11쌍 전수 검증, 예외 0.

```
umbrella@vending_machine_01   4문항  lf05, hf01, hf05, rt03   → beside
tumbler@police_car_01         3문항  lf09, hf07, rt01         → near
umbrella@police_car_01        3문항  lf08, hf08, rt02         → near
pencilcase@picnic_table_02    3문항  lf07, hf02, rtC2         → near
mug@picnic_table_01           2문항  lf01, hf03               → on
sunblock@picnic_table_01      2문항  lf04, hf04               → beside
umbrella@trash_can_01         2문항  lf06, hf06               → near
tumbler@taxi_01               2문항  lf10, rt04               → near
umbrella@taxi_01              2문항  lf11, rt04b              → near
tumbler@bench_01              1문항  lf02                     → on
tumbler@hydrant_01            1문항  lf03                     → near
```

즉 near / beside 판단의 **독립 관측치는 9개**(on 2쌍 제외)이고, 그중 **beside 는 단 2개**다.

---

## 3. GT 성질 ② — 명령 문구로는 안 갈린다

| 문구 | → near | → beside |
|---|---:|---:|
| `beside` | 0 | **3** |
| `near` / `close to` | 3 | 0 |
| `next to` | 6 | 1 |
| `against` (propped / leaning) | 1 | 1 |
| 관계어 없음 | 4 | 1 |

같은 쌍을 두고도 문구가 갈린다:

```
umbrella@vending_machine :  "beside" / "next to it" / "propped against its side"  → 전부 beside
umbrella@trash_can       :  "next to" / "leaning against the trash can"           → 전부 near
```

`hf05` "propped against its **side**" = beside 인데 `hf06` "leaning against" = near 다.
**의미가 사실상 같은데 라벨이 반대**다.

신뢰할 수 있는 건 `beside` 라는 단어뿐이다(3/3). 현재 파서는 이것만 confident 처리한다
(`command_parser.py:232`).

---

## 4. 기하 검증 ① — 중심-중심 거리

`resolved.position` 두 개의 xy 노름.

| GT | 중심거리 | 쌍 |
|---|---:|---|
| on | 0.11 | tumbler@bench_01 |
| on | 0.31 | mug@picnic_table_01 |
| **beside** | **1.43** | sunblock@picnic_table_01 |
| **beside** | **1.85** | umbrella@vending_machine_01 |
| near | 1.15 | umbrella@trash_can_01 |
| near | 2.00 | umbrella@police_car_01 |
| near | 2.92 | tumbler@hydrant_01 |
| near | 3.31 | tumbler@police_car_01 |
| near | 3.47 | pencilcase@picnic_table_02 |
| near | 5.48 | tumbler@taxi_01 |
| near | 5.59 | umbrella@taxi_01 |

**반증**: `near` 최소(1.15) < `beside` 최소(1.43). 어떤 임계값으로도 분리 불가.

> 참고: `lf03`, `lf09` / `hf07` / `rt01` 은 GT의 `target_object_id` 가 stale 하다.
> `rig_7_a` 에 실재하는 tumbler 인스턴스는 `tumbler_0/3/4` 인데 GT는 `tumbler_1` / `tumbler_2`
> 를 가리킨다. 위 표는 같은 타입 중 해당 랜드마크 최근접 인스턴스로 대체해 잰 값이다.
> **→ 이것도 확인 부탁드립니다 (§7-⑤)**

---

## 5. 기하 검증 ② — 원 표면 간극 (공식 충돌반경 사용)

`mock_demo_data.yaml` 의 `obstacles` 리스트에 **모든 객체의 충돌 반경**이 들어 있다.
좌표로 역매핑하니 95개 중 91개가 충무 객체와 0.15m 이내로 일치했다.

| 클래스 | 반경 |
|---|---:|
| 차량 (police_car, taxi, suv, normal_car, sports_car) | 2.00 |
| picnic_table, vending_machine | 1.00 |
| bench, bicycle, kick_scooter | 0.60 |
| person | 0.50 |
| trash_can, hydrant | 0.40 |
| 소형 물체 (tumbler, umbrella, mug, sunblock, pencilcase …) | 0.25 |

이 값으로 `gap = 중심거리 − r_anchor − r_target` (현재 `grounding.py:313` 과 같은 방식):

| GT | 원 표면 gap | 쌍 |
|---|---:|---|
| **beside** | **0.18** | sunblock@picnic_table_01 |
| **beside** | **0.60** | umbrella@vending_machine_01 |
| near | −0.25 | umbrella@police_car_01 |
| near | 0.50 | umbrella@trash_can_01 |
| near | 1.06 | tumbler@police_car_01 |
| near | 2.22 | pencilcase@picnic_table_02 |
| near | 2.27 | tumbler@hydrant_01 |
| near | 3.23 | tumbler@taxi_01 |
| near | 3.34 | umbrella@taxi_01 |

**반증**: `beside` 구간 [0.18, 0.60] 안에 `near` 0.50 이 들어오고, police_car 는 −0.25(겹침)인데 near 다.

---

## 6. 기하 검증 ③ — 방향 있는 박스 면-면 거리 (OBB)

위 두 방식은 **등방 원 근사**라 물체 방향을 무시한다. 피크닉 테이블은 긴 변 / 짧은 변의
실제 여유가 완전히 다른데 원으로 뭉개면 사라진다.

GT에 **yaw 가 전부 들어 있으므로**(`resolved.rotation` = [Roll, Pitch, Yaw], `config.yaml` 에서 확인)
방향은 실측값을 쓰고, **치수만 실물 기준으로 가정**해 OBB-OBB 최단거리를 계산했다.

### 가정한 치수 (⚠️ 이 부분이 확인 대상)

```
picnic_table     1.4~2.4 × 1.2~2.0        umbrella (눕힘)  0.9~1.3 × 0.12~0.2
vending_machine  0.9~1.5 × 0.7~1.2        umbrella (세움)  정사각 0.12~0.2
trash_can        0.4~0.8 × 0.4~0.8        tumbler          0.08~0.12
hydrant          0.3~0.5 × 0.3~0.5        sunblock         0.12~0.2 × 0.05~0.09
police_car/taxi  4.0~5.2 × 1.7~2.1        pencilcase       0.2~0.3 × 0.08~0.14
bench            1.4~1.9 × 0.5~0.8        mug              0.09~0.13
```

`rotation` 의 roll / pitch 로 물체가 누웠는지 섰는지 판정해 footprint 를 다르게 잡았다.

### 결과 (중간값)

| GT | 중심거리 | **박스 면-면** | 변화 | 쌍 |
|---|---:|---:|---:|---|
| on | 0.11 | 0.00 | 접촉 | tumbler@bench |
| on | 0.31 | 0.00 | 접촉 | mug@picnic_table_01 |
| **beside** | 1.43 | **0.14** | −1.29 | sunblock@picnic_table_01 |
| **beside** | 1.85 | **1.06** | −0.79 | umbrella@vending_machine |
| near | 1.15 | 0.35 | −0.80 | umbrella@trash_can |
| near | 2.00 | **0.00** | −2.00 | umbrella@police_car |
| near | 3.31 | 0.78 | −2.53 | tumbler@police_car |
| near | 3.47 | 2.40 | −1.07 | pencilcase@picnic_table_02 |
| near | 2.92 | 2.56 | −0.36 | tumbler@hydrant |
| near | 5.59 | 2.61 | −2.98 | umbrella@taxi |
| near | 5.48 | 2.94 | −2.54 | tumbler@taxi |

순서가 크게 뒤집힌다(중심거리로는 hydrant 2.92 < table 3.47 인데, 면-면으로는 2.56 > 2.40).
**중심거리가 물체 크기를 못 담고 있었다는 것은 확인됐다.**

### 그런데 분리는 오히려 더 나빠진다

```
중심거리 :  beside [1.43, 1.85]   near 최소 1.15              → 위반 1건
면-면    :  beside [0.14, 1.06]   near [0.00, 0.35, 0.78, …]  → 위반 3건
```

분리에 최대한 유리한 극단(beside 앵커는 최대 치수 / near 앵커는 최소 치수)으로 밀어도:

```
beside 최대 = 0.87    near 최소 = 0.14    → 분리 실패
```

### 걸리는 두 지점

1. **`umbrella@police_car`** — 중심거리 2.00 m 인데 차가 4.5×1.9 라 면-면 **0.00 m**,
   즉 우산이 차에 닿아 있다. 그런데 GT는 `near` 다.
2. **`umbrella@vending_machine`** — GT `beside` 이고 명령이 "propped against its **side**"
   인데 면-면 **1.06 m**. 기대 밖으로 멀다. 자판기 한 변이 **2.1 m 이상**이어야 0에 근접하는데
   실물 자판기(폭 1.1 / 깊이 0.8)로는 불가능하다.

계산 스크립트: `mec_sim_ice/vla_grounding/validation/relation_obb_probe.py`
(레포 루트에서 실행. 치수 테이블 `LAND` / `TGT` 만 실측값으로 갈아끼우면 그대로 재검증된다)

---

## 7. 3D 라벨 / 에셋 쪽에 확인 부탁드리는 것

### ① 랜드마크 · 오브젝트 USD 의 실제 footprint 치수

`config.yaml` 의 `landmark_assets` / `mission_object_assets` 에 정의된 각 USD 의
**scale 적용 후 bbox (L × W × H)** 가 필요합니다. 특히:

```
landmark/street/vending_machine/vending_machine.usd
landmark/furniture/picnic_table/picnic_table.usd
landmark/furniture/bench/bench.usd            (scale 0.9)
landmark/street/trash_can/trash_can.usd
landmark/car/police_car/police_car.usd
landmark/car/taxi/taxi.usd
object/etc/umbrella/umbrella.usd  +  object/etc/closedlongumbrella/closedlongumbrella.usd
```

USD 가 컨테이너(`ghcr.io/marc-challenge/marc-platform-content:2026.R03`) 안에만 있어
저희 쪽에서 못 재고 있습니다. `UsdGeom.BBoxCache` 로 뽑은 값이면 충분합니다.

### ② prim 원점 규약

`resolved.position` 이 물체의 **어느 점**인지 확인 부탁드립니다.

- xy 가 bbox 가로 중심이 맞습니까?
- z 는 접촉면(바닥)인가요? (팀 내부 검토에서 접촉면으로 결론냈습니다)
- **특히 vending_machine** — §6 결과가 이상해서, 원점이 한쪽으로 치우쳐 있을 가능성을 의심합니다.
- `config.yaml:31` 이 로봇 USD 에 대해 "prim 원점 기준 바닥(z_min)이 −0.3315m" 라고
  명시한 걸 보면 원점 규약이 에셋마다 다를 수 있다고 이해했습니다.

### ③ `relation` 라벨을 어떻게 붙였습니까

- 배치 스크립트의 파라미터(예: "이 오브젝트를 beside 로 배치")에서 나온 값인가요,
  아니면 배치 후 사람이 보고 붙인 값인가요?
- 기하 규칙에서 유도했다면 그 규칙(임계값)을 알 수 있을까요?

`situation` 은 `situation_rules.yaml` 로 **결정론적 공개 규칙표**가 제공되는데
(*"결정론적·공개 규칙(참가자 동일 기준·GT 분쟁 방어)"*), `relation` 에는 대응하는 파일이
없습니다. 같은 성격의 규칙표가 존재하는지 궁금합니다.

### ④ 아래 세 쌍의 라벨이 의도한 것이 맞습니까

| 쌍 | GT | 저희가 재본 결과 | 의문 |
|---|---|---|---|
| `umbrella@police_car_01` (rig_7_a) | `near` | 면-면 0.00 m (차에 접촉) | 이렇게 붙어 있는데 near? |
| `umbrella@vending_machine_01` (rig_2_c) | `beside` | 면-면 1.06 m | "propped against its side" 인데 1 m? |
| `umbrella@trash_can_01` (rig_3_b) | `near` | 면-면 0.35 m | 위 vending 보다 훨씬 가까운데 near? |

### ⑤ `target_object_id` stale 문제

`marc2026_chungmu.yaml` 의 아래 문항들이 **존재하지 않는 인스턴스**를 가리킵니다.
`rig_7_a` 에 실재하는 tumbler prim 은 `tumbler_0` / `tumbler_3` / `tumbler_4` 입니다.

```
lf03_tumbler_hydrant   → rig_7_a/tumbler_1  (없음)
lf09_tumbler_parking   → rig_7_a/tumbler_2  (없음)
hf07_parking_tumbler   → rig_7_a/tumbler_2  (없음)
rt01_parking_tumbler   → rig_7_a/tumbler_2  (없음)
```

채점은 `target_coord` 로 하므로 실채점에는 영향이 없어 보이지만,
객체 재생성 과정에서 인덱스가 밀린 흔적 같습니다. 확인 부탁드립니다.

---

## 8. 현재 결론

1. **명령 문구로는 판정 불가.** `beside` 라는 단어만 신뢰 가능(3/3). `next to` 6:1, `against` 1:1.
2. **기하로도 판정 불가** — 중심거리 / 원 표면간극 / 박스 면-면 셋 다 반증됨.
   단, ③은 **가정한 치수** 하에서의 결론이므로 §7-①② 가 확인되면 재검증이 필요합니다.
3. **`relation` 은 물리 쌍에 고정**돼 있다(11/11). 이것만은 확실합니다.
4. 데이터를 100% 설명하는 유일한 규칙은
   `beside = (picnic_table | vending_machine 같은 큰 수직 가구) AND (거리 ≲ 2 m)` 인데,
   **beside 관측이 고유 쌍 2개뿐**이라 일반화 근거가 없습니다. 대회 배경이 바뀌면
   그대로 깨지므로 코드에 넣지 않았습니다.

### 덤 — 좌표 점수 쪽에 더 큰 건이 있습니다

`grounding.py:262-270` 은 bbox 밑변 폭을 역투영해 반경을 얻고, **"가로 폭 = 세로 깊이"**
라고 가정해 접촉점을 중심으로 밀어 보정합니다. 길쭉한 물체에서 이게 크게 틀립니다:

```
택시(4.5 × 1.8) 를 옆에서 볼 때   코드가 미는 양 2.25 m / 실제 필요 0.9 m   → 1.35 m 과대
택시를 앞에서 볼 때               코드가 미는 양 0.9 m  / 실제 필요 2.25 m  → 1.35 m 과소
```

anchor 좌표 점수가 `1 − 오차/5` 이고 가중치 0.15 이므로 문항당 약 4점입니다.
실제로 좌표 오답 목록이 **택시 · 경찰차 · 자판기 · 벤치 · 테이블** 로만 채워져 있습니다
(`lf10_tumbler_taxi`, `lf11_umbrella_taxi`, `hf08_parking_umbrella`, `sar02_vending_sit`,
`sar03_bench_lie`, `sar04_table_lie` …).

**§7-① 의 실측 치수는 relation 규명보다 이쪽(anchor 0.524 / target 0.523, 가중치 합 0.55)에
훨씬 크게 쓰입니다.** 치수만 확보되면 yaw 는 이미 GT 에 있으므로 바로 적용 가능합니다.
