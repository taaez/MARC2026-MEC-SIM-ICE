# SESSION REPORT 2026-08-31 — 라이브 검증: CCTV 워밍업 레이스 + umbrella 타일링 회귀

작성: 2026-08-31. `MARC_DISABLE_VLM=1` + DINO ON 조건으로 `bash launch.sh
mec_sim_ice` (headless, `marc2026_chungmu`) 를 실제 기동해 라이브 경로를 검증.
가중치 4종 로드는 정상 확인. 그 과정에서 **라이브에서만** 드러난 문제 2건:
매 런 앞 2~4라운드가 CCTV 미수신으로 blind(§1), umbrella 문항에서 라운드
핸들러가 크래시(§2). §2 는 소형객체 서브모델 라우팅 머지 시의 회귀로 원인
확정. §3 은 검증 스크립트/커밋/브랜치 전수 추적 근거.

관련 로그·산출물:
- 정적 로드 검증: `scratchpad/verify_models.py`
- 라이브 런 로그: `scratchpad/launch_dino_full.log`, `launch_dino_on.log`
- 채점 결과: `results/marc2026_chungmu/mec_sim_ice_2026083*_*.json`,
  `demo/results/scores.jsonl`

---

## 0. 가중치 로드 — 4종 전부 정상

라이브 기동 로그에서 직접 확인:

```
[yolo] routed B+small (small_conf=0.25, small_t640_r2)
[dino-fallback] loaded IDEA-Research/grounding-dino-base on cuda (batch=2)
[pose-kp] loaded .../demo_agents/weights/yolo11m-pose.pt
```

| 모델 | 경로 | 상태 |
|---|---|---|
| YOLO11 B (34클래스) | `demo_agents/weights/best.pt` | 로드 OK, `DET_CLASSES` 34개 정확히 일치 (`verify_model_names` 통과) |
| 소형 서브모델 (3클래스) | `training/small_object_detection/runs/detect/small_t640_r2/weights/best.pt` | 로드 OK, `RoutedDetector` 활성, `small_to_det` = [2,8,7] (sunblock/tissue/disposable_cup) |
| YOLO11m-pose | `demo_agents/weights/yolo11m-pose.pt` | 로드 OK |
| Grounding DINO | HF 캐시 `models--IDEA-Research--grounding-dino-base` | 로드 OK (로컬 캐시, 네트워크 안 탐, fp32/cuda) |
| Qwen 텍스트 폴백 | — | `MARC_DISABLE_VLM=1` 로 의도대로 OFF |

주의: **8GB 개발 카드 + headless Isaac Sim(~3.3GB) 동시 상주에서 DINO forward
는 OOM 난다** (§2-B). 채점 서버(48GB)는 무관. 개발 검증은 Isaac Sim 내리고
`offline_score.py` 로 하거나, DINO OFF 로 라이브를 본다.

---

## 1. CCTV 워밍업 레이스 — 매 런 앞 2~4라운드 blind

### 1.1 증상

라이브 런 6회의 라운드 1~5 점수 (35.0 = "no camera matched" → `fallback_submission`):

| 런 (시각) | 라우팅 | r1 | r2 | r3 | r4 | r5 | 첫 실탐 |
|---|---|---|---|---|---|---|---|
| 08-30 22:45 | ✗ plain | 35 | 35 | 59 | 58 | 99 | **r3** |
| 08-30 22:58 | ✗ plain | 58 | 35 | 59 | 58 | 99 | **r1** |
| 08-30 23:13 | ✗ plain | 35 | 35 | 59 | 98 | 99 | **r3** |
| 08-31 00:08 | ✗ plain | 35 | 35 | 35 | 98 | 99 | **r4** |
| 08-31 00:44 | ✓ routed | 35 | 35 | 59 | 98 | 99 | **r3** |
| 08-31 01:00 (headless) | ✓ routed | 35 | 35 | 35 | 35 | 99 | **r5** |

고정 숫자가 아니라 **0~4라운드로 매번 다름**. 이전 `SESSION_REPORT` 들의
`stage1_round_scores` 도 첫 1~2개가 35.x 로 시작하는 패턴이 있었으나, 원인을
"탐지 recall 병목"으로 뭉뚱그려 읽어 왔음 — 실제로는 **탐지 이전에 이미지가
없어서**임.

### 1.2 근거 (점수 추론 아님, 로그)

SDK 가 카메라 구독을 실제 생성하는 순간 `[SUB] CCTV <cam>` 를 찍는다
(`marc_sdk/client.py:471`). 08-31 01:00 런:

```
01:03:38  round 1  [taejeong] no camera matched ...   score 35
01:03:40  round 2  no camera matched                  score 35
01:03:43  round 3  no camera matched                  score 35
01:03:45  round 4  no camera matched                  score 35
01:03:45  [SUB] CCTV rig_1_a ... rig_7_a              ← 여기서 처음 구독
01:03:47  round 5  [fallback] ... 검출 1/4 카메라로 제한   score 99.1  ← 첫 실탐
```

3개 신호가 전부 일치: ① `[SUB] CCTV` 타임스탬프, ② `[taejeong] no camera
matched` WARNING (`_ground()` 의 `images` 가 빈 dict), ③ 점수 35.

### 1.3 메커니즘 (대회 환경 명세 기준)

- 참가자 앱과 플랫폼은 **별도 머신**, 같은 LAN·같은 `ROS_DOMAIN_ID`, DDS 는
  **UDP 로 네트워크를 탄다** (`demo/README.md`). `network_mode: host` →
  디스커버리가 NIC 에서. 콜드 스타트 시 원격 participant/endpoint 디스커버리에
  수 초.
- **CCTV 이미지 QoS = `BEST_EFFORT / VOLATILE / depth=1`** (`marc_sdk/
  protocol.py:69`, `MARC2026_요약_v3.md:463`). TRANSIENT_LOCAL 이 아니라
  late-joiner 보존이 **전혀 없다** — 구독 성립 전 프레임은 못 받고, 구독 이후
  첫 publish 부터 받는다.
- SDK 카메라 발견은 **2초 폴링 타이머** (`client.py:316` `_discover_cctv`) 로
  ROS 그래프를 훑어 `/marc/env/cctv/*/image` 가 보이면 그때 구독.
  `list_cctv()` 는 이미 구독된 것만 반환.
- 플랫폼은 `"Auto-plan: waiting for a participant to register"` 상태에서
  **등록 ACK 떨어지자마자 STAGE1_RUN + round 1** 을 쏜다. 그 사이 위 항목이
  안 끝나 있으면 `_ground()` 의 `camera_ids = list_cctv()` 가 `[]` → `images
  = {}` → 35점.
- 편차 원인: 직전 세션의 DDS 캐시가 살아있으면(같은 도메인 재기동) r1 에 바로
  잡히고(08-30 22:58), 콜드면 2~4라운드 blind. blind 라운드는 즉시 제출돼
  ~2초 간격으로 흘러가 그만큼 더 많은 라운드가 blind 로 지나간다.

### 1.4 해결 방향 — "1라운드를 버리더라도 토픽 수신을 확인하고 들어간다"

대회 라운드는 **라운드당 30초 sim-time 예산**이고(플랫폼 로그 `"Stage1 round
timer started (30s sim-time)"`), 플랫폼은 **참가자 제출을 받거나 30초가
지나기 전엔 다음 라운드로 안 넘어간다**(로그상 `[SUBMIT] grounding` 직후에만
`STAGE1_RUN → STAGE1_SCORE`). RTF<1 이면 실시간으론 더 김.

→ **첫 `on_mission` 핸들러 안에서 CCTV 프레임이 실제로 들어올 때까지 블로킹**한다.
핸들러를 잡고 있으면 플랫폼이 대기하므로 손해는 최대 라운드 1 하나(이미 잃고
있음). 어차피 핸들러는 YOLO×6 + DINO 로 수 초 쓰는 자리라 성격이 같다.

`participant_app.py` (SDK 는 대회측 소스라 안 건드림):

```python
def _ensure_cctv_warm(self, settle=1.0, timeout=25.0):
    """카메라 수가 settle초간 안 늘고(발견 종료) + 전원 non-None 프레임일
    때까지 대기. 30초 예산 안. 타임아웃되면 있는 것만으로 진행(현재 동작)."""
    deadline = time.time() + timeout
    stable_since, last_n = None, -1
    while time.time() < deadline:
        cams = self.client.list_cctv(); n = len(cams)
        if n > 0 and n == last_n:
            if stable_since is None:
                stable_since = time.time()
            elif (time.time() - stable_since >= settle and
                  all(self.client.get_cctv_image(c) is not None for c in cams)):
                return True
        else:
            stable_since = None
        last_n = n
        time.sleep(0.2)
    log.warning("[warmup] CCTV 미완: %s", self.client.list_cctv())
    return False

def _on_mission(self, mission):
    if not self._cctv_warm:
        self._ensure_cctv_warm(); self._cctv_warm = True
    ...
```

핵심:
- **하드코딩된 카메라 수(chungmu=6: rig_1_a/2_b/2_c/3_b/3_c/7_a)에 의존 안 함** —
  `list_cctv()` 개수가 `settle`초간 안 늘면 "발견 종료"로 판단.
- **`get_cctv_image() is not None`** 을 조건에 넣는 게 중요 — 토픽 이름 발견
  (`list_cctv()` non-empty)만으로는 부족하다. BEST_EFFORT/depth=1 라 구독
  직후 첫 프레임 도착 전까지는 여전히 blind.
- 타임아웃 25초(30초 예산 내). 절대 무한 대기 안 함.

부수: `docker compose up`(제출 표준 경로) 리허설 시 `MARC_LOG_LEVEL=INFO`
로 `[SUB] CCTV` 타임스탬프와 첫 실탐 라운드를 항상 확인.

---

## 2. umbrella 문항 라운드 핸들러 크래시

라이브 33라운드 런에서 관측된 2개의 서로 다른 예외:

```
round 7  "Find the pencil case near the picnic table."
  [ERROR] handler exception: CUDA out of memory. Tried to allocate 328 MiB.
round 8  "Find the umbrella dropped in the empty parking space."   (lf08)
  [ERROR] handler exception: RoutedDetector.predict() got an unexpected keyword argument 'conf'
```

핸들러 예외는 SDK 가 잡아 삼키지만(`_dispatch`), **그 라운드는 제출 0 → 30초
타임아웃 → 0점**. 크래시한 라운드는 `scores.jsonl` 에 줄 자체가 안 남는다.

### 2-A. `RoutedDetector.predict()` TypeError (구조 버그, DINO 무관)

**증상.** `parsed.target_type == "umbrella"` + landmark 가 카메라 정확히 1대로
좁혀짐 + whole-frame 이 그 카메라에서 umbrella 미검출 → `yolo_tile_augment`
가 `_yolo_best_box_tiled(model, ...)` 호출 → 즉시 `TypeError`.

**원인.** `fallback._yolo_best_box_tiled` (fallback.py:658, 680) 은
`model.predict(img, conf=0.001, imgsz=1280, verbose=False)` 를 호출한다 —
**ultralytics YOLO 의 `.predict()` API 가정**. 그러나 소형객체 서브모델
라우팅 통합 이후 `self._det` 은 `RoutedDetector` 래퍼이고, 그
`.predict(self, image)` 는 그 kwargs 를 안 받는다.

**회귀 지점 (커밋·브랜치 전수 추적).**

| 브랜치 | `_yolo_best_box_tiled` | `self._det` | 정합성 |
|---|---|---|---|
| `origin/bongbak` (타일링 발원지) | 있음, `model.predict(conf=,imgsz=,verbose=)` | **`YOLO()` plain** | ✅ plain YOLO API |
| `origin/feature/init` (라우팅 발원지) | **없음** | `load_detector()` | ✅ 타일링 augment 자체가 없음 |
| `origin/joy`, `origin/jumon/*`, `origin/main` | 없음 | — | — |
| **`ptj/stage1-vla-integration` (현재)** | **있음** | **`load_detector()`** | ❌ **충돌** |

- `21c838a` (08-29 19:02, *bongbak*): `yolo_tile_augment` / `_yolo_best_box_tiled`
  신설. 커밋 원문 — *"DINO 의 `_dino_best_box_tiled` 와 같은 원리를 YOLO 에도
  적용한다 -- 이 원리 자체는 검출기 종류를 안 탄다."* 즉 **처음부터 YOLO 대상**
  설계이며 DINO 아이디어만 차용한 독립 경로. 이때 `self._det = YOLO(...)`
  plain 이라 정상.
- `a3931f0` (08-30 16:24): 라우팅 통합. `yolo_adapter.infer_cameras` /
  `real_agent_vla.py` 추론부를 `predict_result()`(RoutedDetector 호환 셸)
  경유로 교체. **`fallback._yolo_best_box_tiled` 는 대상에서 누락.**
  (`real_agent_vla.py` 는 애초에 `yolo_tile_augment` 를 안 씀 — 전수 확인)
- `f3f502f` (08-30 16:35): 라이브 진입점 `taejeong_agent_vla.py` 에서
  `self._det = YOLO()` → `load_detector()`. **이 파일 1개만 수정(+19/−5)**,
  `yolo_tile_augment(parsed, images, detections, self._det)` 호출은 그대로.
  검증은 `test_command_parser/test_situation_resolver/test_grounding` PASS 만
  — umbrella 타일링 라운드는 안 탐.
- `SESSION_REPORT_2026-08-30.md §4` 가 `predict_result()` 로 전환한 호출부를
  나열(`infer_cameras`, `real_agent_vla`)하는데 **`_yolo_best_box_tiled` 는
  어디에도 없음** + "나빠지는 경로 없음" 단언 → 누락이 문서로도 확인.

**간헐성.** whole-frame YOLO 가 umbrella 를 문턱 이상으로 잡으면
`yolo_tile_augment` 의 "이미 찾음 → skip" 에 걸려 `_yolo_best_box_tiled` 가
호출조차 안 됨. 그래서 라우팅 활성 런에서도 lf08 이 96.9(성공) ↔ 0.0(크래시)
로 갈렸다:

| 런 | 라우팅 | r8 (lf08) |
|---|---|---|
| 08-30 22:45 / 22:58 / 00:08 | ✗ plain | 96.9 / 96.9 / 96.4 |
| 08-30 23:13 | ✗ plain | 35.0 (타일링 돌았으나 구제 실패, 크래시 아님) |
| **08-31 00:44** | **✓ routed** | **0.0 (크래시)** |

### 2-B. DINO whole-frame OOM (개발 8GB 제약 + 가드 누락)

`_dino_best_box_tiled` (fallback.py:222) 이 whole-frame 을 먼저
`dino.best_box()` → `_detect_one()` 로 태우는데, `_detect_one` 에는 **CUDA OOM
가드가 없다** (`_detect_batch` 의 배치 절반 축소 로직은 있지만 그 앞의
whole-frame 호출에서 먼저 터짐). headless Isaac Sim(~3.3GB) + YOLO + pose +
DINO 가중치가 8GB 를 채워 DINO forward(fp32, 최소 ~330MiB 단위)가 안 들어감.

채점 서버(48GB)에서는 안 나지만, **혹시라도 transient OOM 이 나면 degrade 가
아니라 라운드 크래시**로 이어진다.

### 2-C. 오프라인 채점도 동일 버그

`offline_score.py:623` 이 같은 `yolo_tile_augment(parsed, images, dets,
model)` 을 `model = load_detector(...)` 로 부른다. 서브모델 산출물이 있고
`--no-small` 을 안 주고 hard umbrella 문항을 만나면 동일하게 `TypeError`.

---

## 3. umbrella 타일링 제거 시 영향 분석

### 3.1 배경 — 애초에 설계 원칙과도 충돌

`SESSION_REPORT_2026-08-27.md:231`, `08-28:33` — **"메인 검출기 B 를 타일링
한다"는 안은 08-27 에 철회**. 렌더가 이미 1280×720 = `imgsz` 라 타일링해봐야
보간 업스케일만, 새 정보 없음(과설계). 서브모델 타일링(SAHI)은 "B 와 격리된
별도 모델"로 3개 소형 클래스에만 부활.

`yolo_tile_augment`(08-29)는 그 철회된 "메인 B 타일링"을 umbrella 한정으로
다시 넣은 것. `21c838a` 커밋 메시지 스스로도 결과가 들쭉날쭉이라고 인정:

| 케이스 | whole → 8타일 | 판정 |
|---|---|---|
| umbrella@taxi (rig_3_c, lf11/rt04b) | 0.09 → 0.33~0.46 | 효과 있음 |
| pencilcase@picnic (rig_3_b) | 0.0 → 0.065~0.108 | 문턱 0.20 밑, 무효 (그래서 제외됨) |
| umbrella@trash_can (rig_3_b) | 0.001 → 0.000 | 전혀 안 뜸 |

추가로, `SMALL_TARGET_MIN_CONF['umbrella']` 0.25→0.20 하향(같은 커밋)이
lf08(conf 0.249)을 이미 whole-frame 에서 통과시켜, 실측상 타일링은 대부분
호출조차 안 된다(plain 4런 r8 = 96.9/96.9/35/96.4, 타일링 실기여 ≈ 0).

### 3.2 제거 대상 (umbrella 타일링 전용 — 삭제 안전)

| 심볼 | 위치 | 참조처 |
|---|---|---|
| `yolo_tile_augment` | fallback.py:694 | `taejeong_agent_vla.py:134`, `offline_score.py:623` 뿐 |
| `_yolo_best_box_tiled` | fallback.py:658 | `yolo_tile_augment` 뿐 |
| `_YOLO_TILE_CLASSES` | fallback.py:626 | `yolo_tile_augment` 뿐 |
| `_target_scope_cams` | fallback.py:629 | `yolo_tile_augment` 뿐 (dino_augment 는 자체 인라인 사본 사용, "의도적 중복") |
| `SMALL_TARGET_MIN_CONF` **import** | fallback.py:29, 35 | 제거 후 fallback.py 내 미사용 |

호출부 정리:
- `taejeong_agent_vla.py`: import(31-32) + call(134) 제거
- `offline_score.py`: import(97-98) + call(623) + `t["yolo_tile"]` 타이밍(624)
  제거 (`t["total"] = sum(t.values())` 이므로 키 하나 빠져도 무해)

### 3.3 유지 대상 (공유 — 삭제 금지)

| 심볼 | 이유 |
|---|---|
| `_tile_grid` (fallback.py:169) | DINO `_dino_best_box_tiled` 가 사용 (fallback.py:236, 269) |
| `SMALL_TARGET_MIN_CONF` **dict** (yolo_adapter.py:44) | plain-YOLO 경로의 클래스별 conf 하한(umbrella/tumbler/pencilcase=0.20). `detections_from_result`/`infer_cameras` 기본 인자. **lf08 을 실제로 구제하는 건 이 하한이지 타일링이 아님.** 타일링과 완전 독립 |

### 3.4 동작 영향

- `dino_augment`: "이미 찾음 → 그 카메라 DINO 스킵" 미세 최적화만 잃음.
  기능은 그대로 — DINO 켜져 있으면 umbrella 카메라에 자기 타일링
  (`_dino_best_box_tiled`)을 돌린다. `21c838a` 실측상 DINO 타일링 ≥ YOLO
  타일링(umbrella@taxi: YOLO 0.33 / DINO 도 뜸). 속도만 소폭 손해.
- DINO OFF(현재 기본 + 8GB): whole-frame YOLO conf 가 [0.001, 0.20) 인 umbrella
  라운드는 유일한 구제를 잃음. 실측 코퍼스(충무)에서 이 구간은 0.20 하한
  도입 후 사실상 안 나옴(lf08=0.249 통과, lf11/hf08 은 ≥0.20 이거나 =0).
- `resolve()` / `pose_augment` / `situation_resolver` / 테스트: **구조 의존
  0건**. `detections` dict 를 소비할 뿐, 제거되는 코드 경로 없음. 테스트 파일
  중 이 심볼들을 참조하는 것 없음.

### 3.5 권장 절차

1. `offline_score.py --no-dino --no-vlm` 로 현재 vs `_YOLO_TILE_CLASSES` 를
   빈 set 으로 둔 버전을 **umbrella 문항(lf08/lf11/hf08)만** A/B.
2. 차이 없거나 개선이면 §3.2 대로 제거.
3. 회귀 여부는 39문항 전수 `offline_score.py --dino --no-vlm` (48GB 환경 또는
   Isaac Sim 내린 8GB) 으로 재확인 — `21c838a` 가 걸었던 것과 같은 기준.

---

## 4. 조치 요약

| # | 항목 | 상태 | 다음 |
|---|---|---|---|
| 1 | CCTV 워밍업 레이스 | 원인 확정 (로그 근거) | `participant_app._ensure_cctv_warm()` 추가 (§1.4) |
| 2-A | `RoutedDetector.predict()` TypeError | 회귀 지점 확정 (`f3f502f` 머지 시 `_yolo_best_box_tiled` 미갱신) | §3 A/B 후 제거, 또는 `predict_result()` 경유로 수정 |
| 2-B | DINO whole-frame OOM 시 라운드 크래시 | 8GB 전용 + 가드 누락 | `_detect_one` 에 `_is_oom` 캐치 추가 (`_detect_batch` 와 동일) |
| 2-C | 오프라인 채점도 2-A 동일 | 확인됨 | 2-A 와 함께 해소 |

---

## 5. 변경 적용 (2026-08-31)

### 5.1 CCTV 워밍업 게이트 — `demo/participant_app.py`

`DemoParticipant._ensure_cctv_warm(settle=1.5, timeout=25.0)` 신규. 첫 Stage 1
mission 핸들러에서 1회 호출(`self._cctv_warm` 플래그):

- 준비 판정 = `list_cctv()` 개수가 `settle`초간 안 변함(디스커버리 종료) **AND**
  발견된 카메라 전부 `get_cctv_image() is not None`(첫 프레임 수신).
- `timeout` 초과 시 경고 로그 후 있는 것만으로 진행 → **기존 동작 보존**
  (죽은 피드가 런을 못 멈춤).
- 근거: 라운드당 30초 sim-time 예산 + 플랫폼은 제출 전까지 다음 라운드로 안
  넘어감 → 핸들러 블로킹은 플랫폼을 페이싱할 뿐, 손해는 최대 라운드 1
  (이미 레이스로 잃던 것).
- SDK(`marc_sdk`)는 대회측 소스라 미수정.

### 5.2 umbrella YOLO 타일링 제거

| 파일 | 변경 |
|---|---|
| `taejeong/fallback.py` | `_YOLO_TILE_CLASSES`, `_target_scope_cams`, `_yolo_best_box_tiled`, `yolo_tile_augment` + 섹션 주석 삭제 (−153줄). `SMALL_TARGET_MIN_CONF` import 제거(미사용) |
| `demo_agents/taejeong_agent_vla.py` | `yolo_tile_augment` import + `_ground()` 호출부 제거 |
| `taejeong/validation/offline_score.py` | `yolo_tile_augment` import + 호출부 + `t["yolo_tile"]` 타이밍 제거 |

유지: `_tile_grid`(DINO `_dino_best_box_tiled` 가 사용), `yolo_adapter.py` 의
`SMALL_TARGET_MIN_CONF` dict(plain-YOLO 클래스별 conf 하한 0.20 — lf08 을 실제로
통과시키는 것). §3.4 대로 `resolve`/`pose_augment`/`situation_resolver`/테스트에
구조 의존 없음.

### 5.3 미적용 (별건)

- **2-B** DINO `_detect_one` OOM 가드: 이번 변경에 미포함. DINO 기본 OFF
  (`68c2469`)라 기본 실행에는 안 걸림. 8GB 에서 `MARC_ENABLE_DINO_FALLBACK=1`
  로 켤 때만 문제 → 별도 커밋에서 `_detect_batch` 와 같은 `_is_oom` 캐치 추가.

---

## 6. 테스트 (2026-08-31)

순서: 로컬 유닛 → 데모 시나리오 → 충무 시나리오. DINO 기본 OFF,
`MARC_DISABLE_VLM=1`.

### 6.1 로컬 유닛 — 전부 PASS

| 테스트 | 결과 |
|---|---|
| `taejeong/test_command_parser.py` | PASS (object 25 / SAR 14 / novel 10) |
| `taejeong/test_situation_resolver.py` | PASS (golden 28/28, exhaustive 171) |
| `taejeong/test_grounding.py` | PASS |
| `tests/test_geometry.py` | 5 passed (pytest) |
| import smoke | `fallback` OK (`yolo_tile_augment` 제거 확인, `_tile_grid` 유지), `offline_score` import OK |

### 6.2 데모 시나리오 (`marc2026_demo`, headless, DINO OFF)

로그: `scratchpad/test_demo2.log` / 플랫폼 `results/marc2026_demo/`.

**CCTV 워밍업 — 성공.** 등록 02:25:38 → round 1 즉시 출제 →
`_handle_mission_deferred` 워커 스레드가 디스커버리 대기 →
`[warmup] CCTV ready: 6 cameras in 12.6s` (02:25:51) → round 1 정상 처리.

| 항목 | 결과 | 비고 |
|---|---|---|
| Stage1 39라운드 | 평균 **73.44**, 최저 25.0 | `--no-fallback` 하한(08-30 리포트) 73.16 과 동급 |
| **round 1** | **58.07** | blind 아님 (이전 런들: 35.0 고정) |
| "no camera matched" | **0회** / 39 | 이전: 앞 2~4라운드 |
| 35.0 라운드 | 6개 | 전부 실제 미탐(레코딩/각도), blind 아님 |
| **round 8 (lf08 umbrella@parking)** | **97.17** | umbrella 타일링 제거 후 크래시 없음 |
| **round 19 (hf08 umbrella)** | **97.13** | 〃 |
| handler exception | **0회** | (이전 라우팅 런: r7 OOM, r8 TypeError) |
| Stage2 grounding + 주행 | 정상 (leg1 목표 도달, pick 시퀀스 진입) | mock nav/manip, 이번 변경과 무관 |

경고 1건: `Exception ignored in TransformListener.__del__` (startup) — **기존
이슈**, 이번 변경 무관. `participant_app.__init__` 의 tf2 로드 `try/except` 가
`rclpy.init()` 전이라 반쯤 생성된 객체의 `__del__` 이 뱉는 것. 무해(로그 오염).

### 6.3 충무 시나리오 (`marc2026_chungmu`, headless, DINO OFF, stage1 33문항)

로그: `scratchpad/test_chungmu.log`. Stage2 는 이번 검증과 무관해 미실행(시간).

**CCTV 워밍업 — 성공.** `[warmup] CCTV ready: 6 cameras in 14.6s` → **round 1 =
58.08** (이전 routed 라이브 런들: r1=35 blind).

| 항목 | 이번 (타일링 제거 + 워밍업) | 이전 routed 라이브 (08-31 00:44) |
|---|---|---|
| Stage1 평균 | **70.30** / 33 | 67.49 / 33 |
| round 1 | 58.08 (실탐) | 35.0 (blind) |
| "no camera matched" | 0회 | 앞 2라운드 |
| handler exception | **0회** | r8 크래시(→0.0), r19 크래시(누락) |
| round 8 lf08 (umbrella@parking) | 35.0 | **0.0 (크래시)** |
| round 19 hf08 (umbrella@parking) | 35.0 | 크래시(로그 누락) |

전체 33: `[58,35,59,98,99,35,35,35,75,97,57,99,25,58,98,99,35,75,35,98,99,40,98,49,99,100,47,47,99,99,98,40,99]`

**관측: umbrella 문항 편차.** r5=99 / r6=35 / r8=35 / r11=57 / r16=99 / r17=35
/ r19=35. 타일링을 지웠으니 이 편차가 예상되는데,
plain-YOLO 런에서는 lf08 이 96.9(2회)/35(1회)/96.4(1회)로 **원래도 50:50**이었고
(타일링은 whole-frame 이 0.20 을 넘으면 호출조차 안 됨), 지금 sim 렌더에서
whole-frame conf 가 0.20 밑으로 떨어진 것으로 보인다. 이전 routed 런은 같은
문항이 **크래시(0.0)** 였으므로 35.0 이 더 낫다.

→ **§3.5 A/B 필요**: `offline_score.py --no-dino --no-vlm` 로 우산 문항만
현재(타일링 없음) vs `21c838a` 직전(타일링, plain YOLO) 비교해 실제 손실폭
정량화. 손실이 유의미하면 타일링을 `predict_result()` 경유로 되살리는 옵션도
있으나(§4 2-A), routed conf 고정 제약은 그대로.

### 6.4 종합

| 테스트 | 결과 |
|---|---|
| 로컬 유닛 4종 | ✅ PASS |
| 데모 39라운드 | ✅ 크래시 0, blind 0, S1 73.44 + S2 95.0 = 168.4/200 |
| 충무 33라운드 | ✅ 크래시 0, blind 0, S1 70.30 (이전 routed 라이브 67.49 대비 +2.8, 주로 r1 blind 해소) |

- **CCTV 워밍업**: 데모/충무 둘 다 round 1 실탐, "no camera matched" 0회.
  워밍업 소요 12~15초(30초 예산 내).
- **umbrella 타일링 제거**: 라우팅 활성 상태에서 umbrella 라운드 크래시 해소.
  타일링의 점수 기여는 A/B 로 별도 정량화 필요(§6.3).
- **다른 코드 연결성**: `resolve`/`pose_augment`/`situation_resolver`/Stage2
  경로 전부 정상. 신규 회귀 없음. 기존 `TransformListener.__del__` 경고는
  이번 변경과 무관.
