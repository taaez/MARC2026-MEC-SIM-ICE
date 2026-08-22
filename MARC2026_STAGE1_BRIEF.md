# MARC 2026 Stage 1 — `mock_agent_vla.py` 실제 구현 브리프 (Claude Code용)

팀: **mec_sim_ice** (세종대, ICE 3 + MRE 2). 이 문서는 스타터킷 `demo/mock_agent_vla.py`를 실제 인식 코드로 교체하는 작업의 단일 기준 문서다. 아래 내용은 개발자 가이드(api-reference / submit-guide / faq, 2026-08-22 확인)와 스타터킷 소스(`marc_sdk/client.py`, `types.py`, `demo/participant_app.py`, `demo/mock_demo_data.yaml`, `problems.yaml`)를 직접 읽고 확인한 것이다. "실측 필요"라고 표시된 항목은 아직 확인되지 않았으니 추측으로 메우지 말 것.

---

## 0. 한 줄 목표

플랫폼이 보내는 영어 자연어 명령(`mission.voice_command`)과 고정 CCTV 영상을 입력받아, 대상의 world 좌표와 해석 결과를 `GroundingResult`로 만들어 `client.submit_grounding()`에 넘긴다. 제한시간 안에, 인터넷 없이.

---

## 1. 실제 하드웨어 구성 (중요 — 이전 버전의 "공유 GPU 서버" 서술은 틀렸다. 정정함)

**Isaac Sim 실행 머신**: 기계과 연구실 데스크탑 1대. **RTX 5060 Ti, VRAM 8GB.** Isaac Sim 권장 최소(16GB)에 못 미쳐 해상도를 낮춰서 돌리는 중. 물리적 접근 + 원격 데스크탑/SSH 가능, 단 항상 켜둬야 접근된다. **팀에 이 GPU 머신은 이거 한 대뿐이다.**

**개발 머신**: 팀원 개인 노트북(사양 제각각, **대부분 GPU 없음**).

```
Isaac Sim + 우리 추론 모델    ⇄ (같은 머신 localhost, 또는 LAN 너머)   개인 노트북(대부분 GPU 없음)
   데스크탑 1대, VRAM 8GB 공유           ROS 2 / DDS
```

- **명령은 플랫폼이 보낸다.** 사람이 치는 게 아니다. `problems.yaml`의 문항을 라운드마다 `MISSION_COMMAND(201)`로 던진다. 우리는 `@client.on_mission` 콜백으로 받는다.
- 흐름: `201 MISSION_COMMAND` → 우리 `301 GROUNDING_RESULT` → `401 SCORE_RESULT`. Stage 1 전체는 문항 수만큼 반복, 최종 점수 = 라운드 평균.
- **주최측 채점 하드웨어(RTX PRO 5000 48GB, i7-12700K, 125GiB — `submit-guide` 명시)는 우리가 접근할 수 없다.** 채점 시점에 주최측이 우리 Docker 이미지를 그 위에서 돌릴 뿐이다. 우리 개발·실험 하드웨어와는 완전히 별개다 — 이걸 우리 서버로 착각하면 안 된다.
- **VRAM 병목이 실질적 제약이다.** 8GB 카드 하나로 Isaac Sim(렌더링)과 우리 탐지·VLM 추론을 같이 감당해야 한다. Isaac Sim이 이미 몇 GB를 쓰므로 남는 VRAM은 실측 전까지 알 수 없다 — 매 실행 전 `nvidia-smi`로 확인한다. 노트북 대부분은 GPU가 없어 코드 작성·기하/규칙 파서 테스트까지만 가능하다.
- **인터넷**: 개발 중·Docker **빌드 시점**은 허용. **채점 런타임만 차단.** 가중치는 Dockerfile에서 받아 이미지에 굽는다. 런타임 다운로드 코드가 있으면 0점.

### VRAM 예산 (개발용, 대략치 — 실측 필요)

| 모델 | 정밀도 | 대략 VRAM |
|---|---|---|
| Grounding DINO base | fp16 | ~1GB |
| Qwen2.5-VL-3B | 4bit | ~2.5~3GB |
| Qwen2.5-VL-3B | fp16 | ~6~7GB |
| Qwen2.5-VL-7B | 4bit | ~5~6GB |
| Qwen2.5-VL-7B | fp16(제출 이미지 기본값, 48GB 채점 머신 기준) | ~14~16GB |

### 8GB 한 장으로 VLM을 실제로 테스트하는 방법

Isaac Sim과 VLM을 동시에 켜면 둘 다 여유가 없다. 대신 두 단계로 나눈다.

1. **1차: Isaac Sim 켠 채, `MARC_DISABLE_VLM=1`로 탐지기만.** `MARC_RECORD_DIR`에 CCTV 이미지·카메라 파라미터·명령을 기록만 해둔다(§9 참고). 이 단계는 VLM이 없으니 남는 몇 GB로도 충분.
2. **2차: Isaac Sim을 끄고, 기록을 `tools/eval_offline.py`로 재실행.** 이제 8GB 전체가 비어 VLM(3B-4bit는 물론 상황에 따라 그 이상)을 붙여볼 여지가 생긴다. `eval_offline.py`는 시뮬레이터 없이 순수 파이썬 함수로 돈다 — §9 참고.

**2차는 연구실 데스크탑이 아니어도 된다.** `records/` 폴더(라운드당 ~10MB, 33라운드 ~350MB)는 이미지+카메라 파라미터+명령이 전부라 USB나 클라우드로 그대로 옮길 수 있다. **GPU가 있는 팀원의 개인 노트북/데스크탑**(예: 집의 RTX 3050 8GB급)으로 옮겨서 `eval_offline.py`를 돌리면, 연구실 데스크탑을 다시 예약할 필요 없이 VLM·파서 튜닝을 독립적으로 반복할 수 있다. 연구실 데스크탑은 "새 기록이 필요할 때"만 다시 쓰면 된다 — 팀 병목을 크게 줄이는 방법이므로 GPU 있는 팀원은 이 역할을 맡는 게 좋다.

---

## 2. 반드시 지킬 인터페이스 (participant_app.py 와의 계약)

`demo/participant_app.py`는 다음만 기대한다. **이 시그니처를 바꾸지 마라.**

```python
class XXX:
    def __init__(self, ...): ...                       # participant_app 은 connect() 전에 생성한다 → 모델 로드는 여기서
    def process(self, voice_command: str, camera_ids: list = None) -> GroundingResult   # Stage 1 (msg 301)
    def process_stage2(self, task_description: str) -> GroundingResult                  # Stage 2 (msg 311)
```

`participant_app.py`에서 바꿀 곳은 **정확히 세 줄**이다:

```python
from agent_vla import VLAGrounding                 # was: from mock_agent_vla import MockVLAGrounding
self.vla = VLAGrounding(self.client)               # was: MockVLAGrounding()  — client 가 필요(이미지/K/지면높이 조회)
self.vla.set_time_limit(mission.time_limit)        # _on_mission 첫 줄에 추가 (on_stage2_mission 에도)
```

`GroundingResult` (정본 `marc_sdk/types.py`):

```python
GroundingResult(camera_id: str, target_type: str, anchor_coord: [x,y,z], target_coord: [x,y,z],
                landmark: str = "", relation: str|None = None, situation: str|None = None)
GroundingResult.from_payload(dict)   # payload dict → 객체
```

제출 payload 규격:

```json
{"camera_id": "rig_1_a",
 "interpretation": {"target_type": "mug", "landmark": "picnic_table", "relation": "on", "situation": null},
 "grounding": {"anchor_coord": [-62.3, 151.2, 16.36], "target_coord": [-62.1, 151.5, 17.45]}}
```

`relation` ∈ {near, beside, behind, front, left, right, on, above}, `situation` ∈ {accident, emergency, abnormal, normal}. 사람 문제는 `target_type="person"`, `relation=None`, `situation` 필수, **landmark 는 여전히 있다**.

---

## 3. SDK 에서 쓰는 것 (정본 `marc_sdk/client.py`)

| 호출 | 반환 | 비고 |
|---|---|---|
| `client.list_cctv()` | `list[str]` | 자동 탐색된 카메라 id (2초 주기 디스커버리) |
| `client.get_cctv_image(id)` | `sensor_msgs/Image` rgb8 1280×720 또는 None | QoS BEST_EFFORT depth 1 → None/옛 프레임 가능. 재시도 필요 |
| `client.get_cctv_info(id)` | `sensor_msgs/CameraInfo` | `.k` (소문자) 9개 float row-major |
| `client.get_cctv_ground_height(id)` | `float` 또는 None | **카메라별** 지면 world z. latched |
| `client.subscribe(topic, msg_type, cb, qos)` | Subscription | `/tf_static` 은 SDK가 안 감싼다 → 이걸로 직접 구독 |
| `client.now_s()` | float | **시뮬레이션 시간**(/clock). 데드라인은 반드시 이걸로 |
| `client.submit_grounding(GroundingResult)` | seq | participant_app 이 호출. 우리는 GroundingResult 만 반환 |

`/tf_static` 구독 QoS: `RELIABLE + TRANSIENT_LOCAL + KEEP_LAST(≥10)`. 안 맞추면 영원히 못 받는다. parent `world` → child `{camera_id}` (예: `rig_1_a`).

**`marc_sdk/` 폴더는 수정 금지.** 주최측 배포 코드이며 `git pull`로 덮인다.

---

## 4. 좌표계와 기하 (검증 완료)

- **world**: REP-103, X앞 Y왼쪽 Z위, 오른손. 제출 좌표 프레임.
- **CCTV 프레임**: ROS optical. +Z 전방, +X 이미지 오른쪽, +Y 이미지 아래. `CameraInfo.K`와 같은 규약이라 **축 뒤집기 불필요**.
- `/tf_static`의 transform(world→cam)은 tf2 의미로 `p_world = R(q) @ p_cam + t`. 따라서 `C = t`(카메라 world 위치), `R_cw = R(q)`.

역투영:
```
d_cam = K^-1 [u, v, 1]^T
d_w   = R_cw @ d_cam
t     = (plane_z - C.z) / d_w.z        (t <= 0 이면 카메라 뒤 → 무효)
P     = C + t * d_w
```

- **bbox 중심이 아니라 바닥 중앙점 `((x1+x2)/2, y2)`을 역투영한다.** 테스트 결과: 1.7m 사람을 중심으로 투영하면 1.73m 오차, 바닥점이면 0.
- 지면은 단일 평면이 아니다(도로/잔디 10cm 단차). **카메라별 `ground_height`를 각각 쓴다.**
- **target z는 물체가 놓인 면의 높이다.** `mock_demo_data.yaml` 실측: 피크닉 테이블 위 머그 z = 지면+1.09, 벤치 위 텀블러 +0.37, 자판기 위 사람 +1.74. `relation == on`(또는 사람 문제에서 "on top of/climbed/lying on the bench") 이면 `plane_z = ground_z + LANDMARK_TOP_M[landmark]`로 역투영한다.
- anchor z는 자산 원점이라 제각각(hydrant ≈ 지면, bicycle ≈ 지면+0.3). 지면 투영으로 낸다. 오차 0~0.3m. (GT가 바닥점인지 중심인지 — **실측 필요**)
- 공간관계 판정 기준축은 **카메라 시점**으로 구현(front = 카메라에 더 가까움, left = z×forward). 채점 정의와 같은지 **실측 필요**.

구현체: `stage1/geometry.py` (numpy만 의존, `tests/test_geometry.py` 통과). 그대로 쓴다.

---

## 5. 문항 구성 (`problems.yaml`, 33문항 전부 enabled)

| 접두사 | 수 | 유형 | 예 | 핵심 난점 |
|---|---|---|---|---|
| `lf01~11` | 11 | 직설 분실물 | "Find the mug on the picnic table." | 탐지·좌표 |
| `hf01~08` | 8 | 서사 분실물 | "I was taking notes at an outdoor table and left my pencil case behind." | "the machine"→vending_machine 같은 간접 표현. **VLM 파싱이 주력** |
| `sar01~09` (8, sar05 없음) | 8 | 사람·상황 직설 | "Find the person collapsed on the parking lot floor." | situation 분류, 명령에 랜드마크 없을 수 있음 |
| `sarH1~H6` | 6 | 사람·상황 서사 | "Several people are in the parking lot; one is completely face-down and immobile…" | 여러 사람 중 고르기 |

### 어휘 (실측. 키 표기 = 채점 기준이므로 절대 바꾸지 말 것)

- target_type: `mug, tumbler, sunblock, umbrella, pencilcase, cola_can, person`
- landmark: `picnic_table, bench, hydrant, vending_machine, trash_can, police_car, taxi, bicycle` (언더스코어 주의: `trash_can`)
- relation 실제 등장: `on, near, beside` (나머지 4개는 규격상 허용)
- situation: 경계가 흐리다. mock에서 "lying face-down on the bench" = abnormal, "lying face-down and motionless in the parking lot" = accident, "collapsed" = emergency. **정의 실측 필요(문항별 세부 점수로 역산).**

### mock_demo_data.yaml 취급

GT에 **의도적 오답**이 섞여 있다(docstring 명시). 예: lf08 "umbrella dropped…" → `target_type: cola_can`. lf/hf 쌍은 같은 정답을 공유하므로 교차 비교해 노이즈를 거른다. 절대 좌표 룩업 테이블로 쓰지 마라 — 실제 대회 배경은 바뀔 수 있다(NOTICES.md).

---

## 6. 파이프라인 설계 (구현체: `stage1/pipeline.py`, `demo/agent_vla.py`)

```
voice_command ─► CommandParser(규칙 + Qwen2.5-VL 텍스트 파싱; 서사형은 VLM 주력)
                    └► Interpretation{target_type, landmark, relation, situation}
각 CCTV ───────► CameraModel(K, R_cw, C_w, ground_z)  [tf_static + info + ground_height 캐시]
각 카메라:       HybridDetector(파인튜닝 YOLO 있으면 1순위 → Grounding DINO) → target/anchor bbox
                 detection_to_world(바닥점, 지면/상면 평면) → A_w, T_w
                 relation_scores(A_w, T_w) 로 명령의 relation 과 일관성 점수
카메라 선택 = argmax( (1-w)·탐지신뢰도 + w·관계일관성 ), w=0.35 (실측 튜닝)
사람 문제: 최종 카메라 crop → VLM situation 분류 (시간 남을 때만)
명령에 랜드마크가 없으면: 대상에서 가장 가까운 탐지 랜드마크(<6m)를 anchor 로 채택
아무 카메라에서도 못 찾으면: fallback payload (첫 카메라 화면 중앙 지면점) — 무제출보다 부분점수
```

### 모델 선택과 근거

| 역할 | 제출 이미지(주최측 48GB 채점 머신용, 우리는 그 하드웨어에 접근 못함) | 우리 개발 하드웨어(연구실 데스크탑, 8GB 1장, Isaac Sim과 공유) | 근거 |
|---|---|---|---|
| 탐지 | 파인튜닝 YOLO (있으면) + `IDEA-Research/grounding-dino-base` | 동일 (Isaac Sim과 동시 실행 가능한 크기) | 대상 어휘가 닫힌 집합이고 주최측 데이터셋 생성기가 정답 bbox를 준다. GDINO는 배경 변경 안전망 |
| 파싱·situation | `Qwen/Qwen2.5-VL-7B-Instruct` bf16 | `Qwen2.5-VL-3B-Instruct` + 4bit, **Isaac Sim을 끄고** eval_offline.py로만 검증 | 좌표는 VLM에 맡기지 않는다(분산 큼). 파싱·분류만. 8GB 한 장으로 Isaac Sim과 VLM을 동시에 못 돌리므로 기록-재실행으로 분리(§1, §9) |

환경변수: `MARC_VLM_MODEL`, `MARC_VLM_4BIT`, `MARC_DET_MODEL`, `MARC_YOLO_WEIGHTS`, `MARC_DISABLE_VLM`, `MARC_RECORD_DIR`.

---

## 7. 절대 규칙 (위반 시 0점 또는 디버깅 불능)

1. **`time.time()` / `time.sleep()` 금지.** 데드라인·타이밍은 `client.now_s()` / `client.sleep()`. 시뮬레이션이 실시간보다 느리면(RTF<1) 벽시계는 틀린다. `pipeline.run()`은 `now` 콜러블을 주입받는다 — `build_pipeline(now=client.now_s)`.
2. **런타임 다운로드 금지.** 이미지에 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 고정. 로컬에서도 채점과 같은 경로로 돌게.
3. **`marc_sdk/` 수정 금지. `participant_app.py` 는 위 3줄만.**
4. **씬 의존 하드코딩 금지.** 카메라 id↔장소, 랜드마크 좌표, 카메라 수 가정 전부 금지.
5. **모델 로드는 생성자에서**(connect 전). 첫 라운드에 로딩 지연이 들어가면 안 된다.
6. **`on_mission` 재진입 방어.** 제한시간 초과 시 다음 미션이 겹칠 수 있다 → busy 락, 스킵.
7. 제출 compose의 `MARC_TOKEN: "${MARC_TOKEN:-<토큰>}"` 기본값이 곧 제출값. Private 리포 필수. 태그는 `git push origin qualifier` 따로.
8. PyTorch는 **cu128 이상** wheel (RTX PRO 5000 / 5060 Ti 모두 Blackwell sm_120).

---

## 8. 연동 체크

두 가지 실행 모드가 있다. 실제 GPU 머신은 연구실 데스크탑 1대뿐이므로 상황에 맞게 고른다.

**모드 A — 데스크탑 위에서 직접 (localhost).** SSH 또는 원격 데스크탑으로 그 머신에 들어가 플랫폼과 에이전트를 같은 곳에서 돌린다. 네트워크 변수가 없어 디버깅이 쉽다. `bash scripts/diag_link.sh`(IP 인자 없이).

**모드 B — 노트북에서 LAN 너머로.** 노트북에 GPU가 없어도 `MARC_DISABLE_VLM=1` + 탐지기까지도 생략하고 **규칙 파서만** 시험하거나, 데스크탑에서 이미 만든 `records/`를 복사해 `eval_offline.py --dry`로 로직만 검증하는 용도에 적합하다. 실제 CCTV·모델을 쓰는 풀 실행은 데스크탑 자원을 그대로 쓰므로 사실상 A와 자원 경합은 같다. `bash scripts/diag_link.sh <데스크탑IP>`.

공통 확인 사항:
- `ROS_DOMAIN_ID` 동일, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, 둘 다 **Humble**(다른 배포판이 들어오면 플랫폼이 크래시), `ROS_LOCALHOST_ONLY` 미설정, Docker는 `network_mode: host`, ufw 허용.
- `MARC_TEAM_ID=mec_sim_ice` 어디서든 동일. **같은 team_id로 에이전트를 동시에 두 개 이상 띄우지 마라** — GPU도 하나뿐이라 어차피 순서대로 써야 한다. 슬랙 등으로 "지금 데스크탑 씀" 표시를 남겨라.
- 확인 토픽: `/marc/ops/announce`, `/marc/env/cctv/*/image`, `/tf_static`, `/clock`. `ros2 run tf2_ros tf2_echo world rig_1_a`.
- 실행 전 `nvidia-smi --query-gpu=memory.free --format=csv`로 Isaac Sim이 쓰고 남은 VRAM을 확인하고, §1의 VRAM 예산표와 대조해 VLM on/off·양자화 여부를 정한다.

---

## 9. 검증 3층

1. **단위** (ROS/GPU 없음, 1초): `python tests/test_geometry.py` — 왕복 투영, 바닥점, relation. 파서는 33문항 문장으로.
2. **기록 재실행** (연구실 데스크탑 또는 GPU 있는 팀원 개인 PC, 반복): 서버에서 1회 `MARC_RECORD_DIR=./records`로 33라운드 기록 → `python tools/eval_offline.py --records ./records --gt demo/mock_demo_data.yaml`. 파이프라인은 `f(명령, 이미지, K/extrinsic/ground_z)`의 순수 함수라 재현 가능. `records/`만 옮기면 연구실 데스크탑을 다시 예약하지 않고도 GPU 있는 팀원 누구든 독립적으로 반복할 수 있다(§1 참고). 해석 정확도·xy/z 오차·지연 표 출력. (`--dry`는 모델 없이 기하·파서만)
3. **플랫폼 자가 채점** (서버, 하루 1~2회): `problems.yaml`을 줄여 반복(lf 2 + hf 1 + sar 1), 확신 후 전체. `_on_score`의 `score.scores` dict를 **문항 id와 함께 JSONL로 저장**(현재 participant_app은 total만 로그).
- 제출 전: 새 디렉토리에 `qualifier` 태그 clone → `scripts/verify_offline.sh <LAN_CIDR>`(iptables로 외부만 차단; `network_mode: host`라 `--network none` 불가) → 층 3 한 바퀴.

---

## 10. 실측 필요 (추측 금지)

- **Isaac Sim 기동 후 실제 남는 VRAM** (`nvidia-smi`, 해상도 낮춘 상태 기준). §1 VRAM 예산표는 어림값이라 이 값이 나와야 VLM on/off·양자화 수준을 확정할 수 있다.
- `mission.time_limit` 실제 값, CCTV 대수
- GT 좌표 정의(바닥점 vs 중심), relation 기준 시점(카메라/랜드마크), situation 라벨 경계
- `requirements.txt` 버전 조합(transformers 4.53.2 + Qwen2.5-VL + Grounding DINO) 빌드 성공 여부. `post_process_grounded_object_detection` 반환 키가 버전따라 `labels`/`text_labels` → 둘 다 처리해 둠
- Grounding DINO base의 소형물(필통·선크림) 재현율 @1280×720, 카메라당 추론 시간
- `marc2026_chungmu` vs `marc2026_demo`: 공개 USD가 chungmu 하나라 같은 배경으로 추정. 카메라 id 비교로 확정
- 2026.R03 패치로 데이터셋 생성기 bbox GT가 수정됨 → **R03 이전 생성 데이터 폐기·재생성** (`TRAINER_AUTO_SCENES`로 대량 생성)

---

## 11. 제공된 코드 (tar: `marc_stage1_v3.tar.gz`) — 이걸 출발점으로 쓴다

```
stage1/geometry.py      역투영·관계 (검증 완료)
stage1/pipeline.py      파서·탐지기·카메라 선택·payload. now 주입
demo/agent_vla.py       VLAGrounding: mock drop-in. tf_static 캐시, 기록
demo/participant_app.patch   3줄 변경
tests/test_geometry.py
tools/eval_offline.py   기록 재실행
scripts/setup_checklist.sh / diag_link.sh / verify_offline.sh
docker/Dockerfile, requirements.txt, entrypoint.sh, docker-compose.yml
```

**검증 상태**: geometry·파서·컴파일은 통과. `agent_vla.py`는 SDK 없이 컴파일만 됨 — 서버에서 `MARC_DISABLE_VLM=1 ./launch.sh mec_sim_ice`로 첫 접속이 되기 전까지 "동작한다"고 말할 수 없다. Dockerfile의 `COPY marc_sdk` 경로는 스타터킷 `demo/Dockerfile`(이 프로젝트 파일에는 없음)을 보고 맞출 것.

## 12. Claude Code 작업 순서 제안

1. 스타터킷 루트에 tar 풀기. `demo/agent_vla.py`, `stage1/`, `tests/`, `tools/` 배치. `python tests/test_geometry.py` 통과 확인.
2. `participant_app.py` 3줄 패치 + `_on_score` JSONL 로깅 추가.
3. `demo/Dockerfile` 읽고 `docker/Dockerfile`의 SDK 경로·베이스 이미지 맞추기. `MARC_DISABLE_VLM=1`로 먼저 빌드해 GDINO만 굽고 오프라인 로드 확인.
4. 서버에서 lf 2문항만 켜고 첫 접속. `diag_link.sh` 통과 → 기록 생성 → 노트북에서 재실행.
5. VLM 켜고 hf → sar 순으로 확장. 각 단계에서 층 2 표를 남긴다.
