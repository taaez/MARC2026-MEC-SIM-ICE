# 오프라인 검증 가이드 — Isaac Sim 없이 record 자료로 파이프라인 검증하기

Isaac Sim 데스크탑에서 한 번 수집해 둔 `mec_sim_ice/records/` 폴더만 있으면, **아무 PC에서나
(ROS/Isaac Sim/플랫폼 전부 불필요)** Stage 1 파이프라인을 재실행하고 정답과 비교할 수
있다. 파이프라인 코드를 고칠 때마다 데스크탑에서 플랫폼을 다시 띄울 필요가 없어진다.

## 1. record 자료가 뭔가

라이브 실행 시 `MARC_RECORD_DIR` 를 주면 라운드마다 이렇게 덤프된다
(수집 방법 자체는 `RUNBOOK_record.md` 3절 참고 — 이 문서는 **수집된 뒤** 얘기다):

```
mec_sim_ice/records/
  s1_001_<타임스탬프>/
    rig_1_a.png        # 그 라운드의 CCTV 프레임 (6대 전부)
    rig_2_b.png
    ...
    meta.json          # {command: 문제 문장,
                       #  cameras: {카메라별 K, R_cw, C_w, ground_z, width, height},
                       #  payload: 그때 제출했던 답}
  s1_002_<타임스탬프>/
  ...
```

핵심: **이미지 + 카메라 내부/외부 파라미터 + 문제 문장**이 다 들어 있으므로,
탐지 → 파싱 → 픽셀→월드 좌표 계산까지 전 과정을 오프라인에서 그대로 재현할 수 있다.

## 2. 준비물 (오프라인 PC)

| 준비물 | 얻는 법 |
|---|---|
| 레포 (records/ 포함) | `git clone` 또는 `git pull` — `mec_sim_ice/records/` 는 git 에 포함돼 있음 |
| `mec_sim_ice/demo_agents/weights/best.pt` | 학습 담당 팀원에게 파일로 받아 배치 (git 미포함) |
| Python 3.10+ | ROS 불필요. GPU 없어도 됨(느릴 뿐) |
| 패키지 | `pip install ultralytics pillow pyyaml` |

```bash
git pull                                  # ptj/stage1-vla-integration 브랜치
pip install ultralytics pillow pyyaml
mkdir -p mec_sim_ice/demo_agents/weights && cp <받은 best.pt> mec_sim_ice/demo_agents/weights/best.pt
```

## 3. 실행 — record 전용 파이프라인 (`mec_sim_ice/record_agent_vla.py`)

레포 루트에서:

```bash
# (a) 파이프라인 재실행만 — 기록 당시 제출(payload)과 나란히 출력
python3 mec_sim_ice/tools/run_offline_real_vla.py --records ./mec_sim_ice/records --weights mec_sim_ice/demo_agents/weights/best.pt

# (b) 좌표 포함 GT 와 비교 — 필드별 정답 여부 + 좌표 점수
python3 mec_sim_ice/tools/run_offline_real_vla.py --records ./mec_sim_ice/records --weights mec_sim_ice/demo_agents/weights/best.pt \
    --gt demo/mock_demo_data.yaml

# (c) 공식 시나리오 expected 와 비교 — camera/target/landmark/relation 일치 여부
python3 mec_sim_ice/tools/run_offline_real_vla.py --records ./mec_sim_ice/records --weights mec_sim_ice/demo_agents/weights/best.pt \
    --gt mec_sim_ice/vla_grounding/content-image-2026.R03/marc2026_chungmu.yaml
```

옵션: `--conf 0.35`(탐지 confidence), `--imgsz 1280`. Windows 는 `python3` 대신
`python`, 경로 구분자는 `\` 로.

### 출력 읽는 법

라운드마다:

```
round: s1_003_1787400698
command: Find the tumbler on the bench.
  ours    : {"camera_id": "rig_1_a", "interpretation": {...}, "grounding": {...}}   # 지금 코드의 답
  recorded: {...}                                # 기록 당시 제출했던 답 (코드 개선 전후 비교)
  vs GT   : {"camera": true, "target_type": true, "landmark": true, "relation": true,
             "situation": null, "anchor_score": 0.87, "target_score": 0.91, ...}
```

- `camera`/`target_type`/`landmark`/`relation`/`situation` — GT 와 일치 여부.
  `null` 은 그 GT 형식에 해당 필드가 없다는 뜻 (예: 공식 expected 엔 좌표가 없음).
- `anchor_score`/`target_score` — 좌표 점수 `max(0, 1 - 3D오차/5m)`
  (`mec_sim_ice/vla_grounding/validation/score_report.py` 와 같은 실측 역산식). `*_err_m` 은 오차 미터.
- 마지막 `summary` — 전체 라운드의 필드별 정답률과 좌표 점수 평균.

### 개선 루프

1. `command_parser` / `record_agent_vla.py` / weights 수정
2. 위 (b) 재실행 → summary 비교
3. 어느 필드에서 갈렸는지(`vs GT` 라인)로 원인 분리: 탐지 실패인지, 파싱 오답인지,
   카메라 오선택인지, 좌표 오차인지

**같은 records 를 계속 재사용**하므로 실험이 몇 초~몇 분이면 끝난다.

## 4. 다른 파이프라인도 같은 records 를 쓴다

| 도구 | 재실행 대상 |
|---|---|
| `mec_sim_ice/tools/run_offline_real_vla.py` | `mec_sim_ice/record_agent_vla.py` (record 전용) |
| `mec_sim_ice/vla_grounding/run_offline.py` | vla_grounding 파이프라인 (command_parser → yolo_adapter → grounding) — 라운드별 payload만 출력 |
| `mec_sim_ice/vla_grounding/validation/offline_score.py` | vla_grounding 파이프라인 + **그 자리에서 채점까지** (camera/object_type/landmark/relation/anchor/target 표, `score_report.py` 와 동일 포맷) |

(`mec_sim_ice/stage1/` 파이프라인과 그 재실행 도구 `tools/eval_offline.py` 는 제출 경로에서
전혀 안 쓰이는 죽은 코드라 2026-08-31에 삭제했다.)

예) vla_grounding 파이프라인 (payload만, 채점 없음):

```bash
cd mec_sim_ice/vla_grounding
python3 run_offline.py --records ../records --weights ../demo_agents/weights/best.pt
```

예) vla_grounding 파이프라인 + 즉시 채점 (권장 — chungmu 39문항 정답이 `validation/runs/`에 이미 있음):

```bash
cd mec_sim_ice/vla_grounding/validation
python3 offline_score.py --weights ../../demo_agents/weights/best.pt
```

anchor/카테고리 필드 점수는 과거 39건 전수 대조로 검증된 정확한 공식이고, target(좌표) 점수는
근사치다 (자세한 이유는 스크립트 docstring 참고). 결과 표는 문제번호·카메라·target·landmark·
relation·anchor·target 좌표 정답 여부를 `score_report.py` 와 같은 포맷으로 터미널에 바로 찍는다.

## 4-1. Stage2 채점 (2026-08-27 추가)

`score.py` / `offline_score.py` 는 이제 **Stage1 과 Stage2 점수를 따로 낸다.**

```
Stage1 평균          :  51.14   (33문항)
Stage2 grounding 평균 :  65.85   (6문항)
```

- 단계 구분은 기록 폴더 이름(`s1_*` / `s2_*`)이 아니라 **시나리오의 문항 종류**(`kind:
  grounding` / `kind: retrieval`)로 한다. 2026-08-25 기록 세션이 39문항을 전부 Stage1
  라운드로 출제해서 Stage2 문항(rt01~rtC2)까지 `s1_034`~`s1_039` 에 들어가 있기 때문이다.
- 정답은 `chungmu_gt.py` 가 `marc2026_chungmu.yaml` 에서 직접 유도한다(유도 규칙은 그
  파일 docstring 참고). 라이브 정답지(`runs/*.jsonl`)가 있는 문항은 그쪽을 우선 쓰고,
  없는 문항만 yaml 유도값으로 채운다 — 그래서 **문항을 새로 켜도 바로 채점된다.**
- 같은 문항이 여러 폴더에 잡히면(rt01 은 `s1_034`/`s2_004`/`s2_040` 세 개) 전부 채점해
  보여주되 평균에는 한 번만 반영한다(`[중복 레코드]` 로 표시).

~~**Stage2 는 grounding(인식) 부분만 채점된다.**~~ → **2026-08-27 갱신: 주행·집기·배달도
같이 나온다.** `score.py` 가 §4-2 의 실행 시뮬을 이어서 돌린다. 아래처럼 공식 3요소가
한 번에 찍힌다:

```
### Stage 2 실행 (주행·집기·배달) — 6문항   [오프라인 시뮬 / 가정 포함]
  (1) VLA grounding      평균 65.85 / 100점
  (2) 물품 집기          3/6 문항 성공
  (3) 지정 장소로 이동   3/6 문항 성공
```

`--no-stage2-exec` 로 끌 수 있다(실행 시뮬은 ~15초 더 걸린다).

**총점 환산은 여전히 안 한다 — 가중치가 비공개라서다.** 공식 개발자 가이드가 Stage2
채점을 세 가지(VLA grounding / 물품 집기 / 지정된 장소로 이동)로 명시하면서 동시에
"정확한 임계값·가중치는 공개하지 않습니다" 라고 못박고 있다.

리포 안의 관측치로 역산해 봐도 안 맞는다. 라이브 최종 `stage2 = 87.0`(만점 100, `total`
131.03 = stage1 44.03 + 87.0)이고, 08-22 세션 로그에 `"Stage 2 recovery: 0/80"` 이라는
문자열이 있어 실행 배점이 80 처럼 보이지만 — 그날 실제로 제출한 Stage2 grounding
(`records/s2_040`: `camera_id=""`, `landmark=""`, 좌표 전부 0)은 **25.00점**이라
`0.25 × 20 + 80 = 85 ≠ 87` 이다. 어떤 단순 조합으로도 87 이 나오지 않으므로 배점
가설을 세우지 않는다.

**집기/배달 판정은 가정이 섞여 있다.** 접촉 솔버가 없어 "손가락이 진짜 물었나" 는
원리적으로 알 수 없다. `집기 OK` 는 **명령 궤적이 물체 위치까지 EE 를 데려갔다** 는
뜻이다 (§4-2 참고).

## 4-2. Stage2 **실행** 시뮬 (2026-08-27 추가)

`vla_grounding/validation/stage2_sim.py` 는 `demo/participant_app.py` 를 **원본 그대로 실행한다.**
가짜 `MARCClient`(`validation/fake_client.py`)를 `MARCClient.from_env()` 자리에 끼워 넣는
방식이라, ROS·Isaac Sim 없이 주행 제어루프와 팔 서보가 실제로 돈다.

`score.py` 가 이 시뮬을 자동으로 이어서 돌리므로(§4-1) 보통은 따로 부를 일이 없다.
개별 문항을 파고들거나 대조군을 돌릴 때만 직접 쓴다.

```bash
python3 mec_sim_ice/vla_grounding/validation/stage2_sim.py                      # 6문항 전부
python3 ... --problem rt01_parking_tumbler --verbose                       # 한 문항 + 앱 로그
python3 ... --occupancy mec_sim_ice/records/occupancy.npz                  # 실제 지도가 있으면
```

문항당 벽시계 1~3초다(시뮬 시간으로 60~160초 분량). 시뮬레이션은 앱이 `client.sleep()` 을
부를 때만 전진하므로 단일 스레드로 결정론적이다.

|  | `stage2_offline.py` | `stage2_sim.py` |
|---|---|---|
| 하는 일 | 계산이 맞나 (정적 검사) | 코드가 굴러가나 (동적 실행) |
| 예 | A* 가 경로를 뱉나 | 그 경로를 따라 실제로 도착하나 |

### 무엇이 진짜 검증되나

- **진짜** — 차체 운동학, A*, PathFollower 제어루프, **placo FK/IK/DLS 팔 서보**(진짜 gen3
  URDF), 시간 예산. `arm_pick.py`/`arm_kin.py`/`mock_agent_navigation.py` 는 한 줄도 안 고치고 돈다.
- **가정** — 파지 포획 판정. 접촉 솔버가 없으니 "손가락이 진짜 물었나"는 여전히 모른다.
  여기서 말하는 `포획` 은 **명령 궤적이 물체 위치까지 EE 를 데려갔다**는 뜻이고,
  이게 `stage2_offline.py [3]` 이 열어둔 기하 질문에 대한 답이다.
- **불가** — 물체가 밀림/넘어짐, 차체 전복, 라이다 지역회피, 합성 지도의 실제 통행성,
  플랫폼 Stage2 배점.

모델 상수(가속·관절속도·포획반경)는 실측이 아니라 추정이므로 `fake_client.py` 상단에
모아뒀다. 절대 판정이 아니라 **회귀 검출과 실패 원인 분리**에 쓴다.

### 첫 실행에서 나온 것 (2026-08-27, 합성 지도 기준)

**주행+파지 기하 통과 3/6.** 실패 4건(rt01/rt02/rt04b, rt04 는 아슬하게 성공)이 전부
같은 원인이었다 — `_approach_straight()` 가 **20.3초 동안 3cm 전진**하고 끝난다.

`participant_app.py:298-300` 은 `lin=0.06, ang=±0.4` 까지 내보내는데, 이 4륜조향 차체는
`2v <= w*T` 인 명령에서 정지한다(`mock_agent_navigation.py:51-53` — "Same value as the
platform's chassis steering track"). `|lat| >= 0.211` 이면 정확히 그 조건에 걸린다.
같은 파일의 `GoalSeekController` 는 `w_max = 1.8*linear/STEERING_TRACK` 으로 이 상한을
지키지만, 나중에 쓰인 `_approach_straight` 에는 그 상한이 없다.

연쇄는 이렇다:

1. leg1 의 `tol=0.45` 가 사실상 무효다 — `PathFollower` 는 자기 `GOAL_TOLERANCE=0.7` 로
   완료를 알리고 A* 마지막 웨이포인트도 격자에 스냅돼 있어, 실제 정지 오차는 0.45~1.29m 다.
2. 그 오차를 메워야 할 `_approach_straight` 가 위 이유로 아무것도 못 한다.
3. 남은 거리를 팔이 감당해야 하는데 0.99~1.17m 라 도달범위(~0.75m)를 넘는다 → 파지 실패.

검증은 두 대조군으로 했다:

```bash
python3 ... --no-stall-rule        # 정지규칙만 끔        -> 6/6 (결론이 이 규칙에 달려 있음을 확인)
python3 ... --fix-approach-cap     # 정지규칙 ON + 상한 적용 -> 6/6 (수정안이 원인을 짚음을 확인)
```

`--fix-approach-cap` 은 `participant_app.py` 를 고치지 않고, 직진접근 구간의 `cmd_vel` 에만
`GoalSeekController` 와 같은 상한을 씌워보는 A/B 다. rt01 의 `lat` 이 0.824 → 0.003 으로
잡히고 4건 전부 뒤집힌다.

**단, 차체 정지 규칙 자체는 모델이지 실측이 아니다.** 근거는 스타터킷 주석 한 줄이다.
라이브에서 확인하기 전까지는 "유력한 원인" 이지 확정이 아니다.

## 5. 한계 — 오프라인 점수는 근사치다

- 공식 항목별 가중치·채점 세부는 플랫폼만 안다. 오프라인 결과는 **개선 전후의 상대
  비교와 실패 원인 분리**에 쓰고, 최종 점수 확인은 라이브 실행(`results/scores.jsonl`
  → `mec_sim_ice/vla_grounding/validation/score_report.py`)으로 한다.
- record 는 수집 시점의 씬 스냅샷이다. 시나리오/에셋이 갱신되면 데스크탑에서 다시
  수집해야 한다 (`RUNBOOK_record.md` 3절, `MARC_RECORD_ONLY=1` 이면 몇 분).
- `mock_demo_data.yaml` GT 는 시나리오에서 추출된 값(+노이즈 가능성)이라 좌표 점수가
  실제 채점과 1:1 은 아니다. 공식 expected(`marc2026_chungmu.yaml`)는 해석 필드만 있다.
