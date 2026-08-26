# 오프라인 검증 가이드 — Isaac Sim 없이 record 자료로 파이프라인 검증하기

Isaac Sim 데스크탑에서 한 번 수집해 둔 `mec_sim_ice/records/` 폴더만 있으면, **아무 PC에서나
(ROS/Isaac Sim/플랫폼 전부 불필요)** Stage 1 파이프라인을 재실행하고 정답과 비교할 수
있다. 파이프라인 코드를 고칠 때마다 데스크탑에서 플랫폼을 다시 띄울 필요가 없어진다.

## 1. record 자료가 뭔가

라이브 실행 시 `MARC_RECORD_DIR` 를 주면 라운드마다 이렇게 덤프된다
(수집 방법 자체는 `RUNBOOK_real_vla.md` 3절 참고 — 이 문서는 **수집된 뒤** 얘기다):

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

## 3. 실행 — RealVLA 파이프라인 (`mec_sim_ice/real_agent_vla.py`)

레포 루트에서:

```bash
# (a) 파이프라인 재실행만 — 기록 당시 제출(payload)과 나란히 출력
python3 mec_sim_ice/tools/run_offline_real_vla.py --records ./mec_sim_ice/records --weights mec_sim_ice/demo_agents/weights/best.pt

# (b) 좌표 포함 GT 와 비교 — 필드별 정답 여부 + 좌표 점수
python3 mec_sim_ice/tools/run_offline_real_vla.py --records ./mec_sim_ice/records --weights mec_sim_ice/demo_agents/weights/best.pt \
    --gt demo/mock_demo_data.yaml

# (c) 공식 시나리오 expected 와 비교 — camera/target/landmark/relation 일치 여부
python3 mec_sim_ice/tools/run_offline_real_vla.py --records ./mec_sim_ice/records --weights mec_sim_ice/demo_agents/weights/best.pt \
    --gt mec_sim_ice/taejeong/content-image-2026.R03/marc2026_chungmu.yaml
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
  (`mec_sim_ice/taejeong/validation/score_report.py` 와 같은 실측 역산식). `*_err_m` 은 오차 미터.
- 마지막 `summary` — 전체 라운드의 필드별 정답률과 좌표 점수 평균.

### 개선 루프

1. `command_parser` / `real_agent_vla.py` / weights 수정
2. 위 (b) 재실행 → summary 비교
3. 어느 필드에서 갈렸는지(`vs GT` 라인)로 원인 분리: 탐지 실패인지, 파싱 오답인지,
   카메라 오선택인지, 좌표 오차인지

**같은 records 를 계속 재사용**하므로 실험이 몇 초~몇 분이면 끝난다.

## 4. 다른 파이프라인도 같은 records 를 쓴다

| 도구 | 재실행 대상 |
|---|---|
| `mec_sim_ice/tools/run_offline_real_vla.py` | `mec_sim_ice/real_agent_vla.py` (RealVLA) |
| `mec_sim_ice/taejeong/run_offline.py` | taejeong 파이프라인 (command_parser → yolo_adapter → grounding) — 라운드별 payload만 출력 |
| `mec_sim_ice/taejeong/validation/offline_score.py` | taejeong 파이프라인 + **그 자리에서 채점까지** (camera/object_type/landmark/relation/anchor/target 표, `score_report.py` 와 동일 포맷) |
| `mec_sim_ice/tools/eval_offline.py` | `mec_sim_ice/stage1/` 파이프라인 (Grounding DINO 실험 계열) |

예) taejeong 파이프라인 (payload만, 채점 없음):

```bash
cd mec_sim_ice/taejeong
python3 run_offline.py --records ../records --weights ../demo_agents/weights/best.pt
```

예) taejeong 파이프라인 + 즉시 채점 (권장 — chungmu 39문항 정답이 `validation/runs/`에 이미 있음):

```bash
cd mec_sim_ice/taejeong/validation
python3 offline_score.py --weights ../../demo_agents/weights/best.pt
```

anchor/카테고리 필드 점수는 과거 39건 전수 대조로 검증된 정확한 공식이고, target(좌표) 점수는
근사치다 (자세한 이유는 스크립트 docstring 참고). 결과 표는 문제번호·카메라·target·landmark·
relation·anchor·target 좌표 정답 여부를 `score_report.py` 와 같은 포맷으로 터미널에 바로 찍는다.

## 5. 한계 — 오프라인 점수는 근사치다

- 공식 항목별 가중치·채점 세부는 플랫폼만 안다. 오프라인 결과는 **개선 전후의 상대
  비교와 실패 원인 분리**에 쓰고, 최종 점수 확인은 라이브 실행(`results/scores.jsonl`
  → `mec_sim_ice/taejeong/validation/score_report.py`)으로 한다.
- record 는 수집 시점의 씬 스냅샷이다. 시나리오/에셋이 갱신되면 데스크탑에서 다시
  수집해야 한다 (`RUNBOOK_real_vla.md` 3절, `MARC_RECORD_ONLY=1` 이면 몇 분).
- `mock_demo_data.yaml` GT 는 시나리오에서 추출된 값(+노이즈 가능성)이라 좌표 점수가
  실제 채점과 1:1 은 아니다. 공식 expected(`marc2026_chungmu.yaml`)는 해석 필드만 있다.
