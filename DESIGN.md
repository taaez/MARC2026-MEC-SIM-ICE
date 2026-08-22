# MARC 2026 Stage 1 — 설계 결정 기록 (mec_sim_ice)

문서 기준: marc-dev-guide 2026.1.0 (api-reference / submit-guide / faq, 2026-08-22 확인)

## 0. 문서에서 확인한, 설계를 바꾸는 사실

| 사실 | 출처 | 영향 |
|---|---|---|
| 채점 시 에이전트는 플랫폼과 **별도 머신**(RTX PRO 5000 48GB, i7-12700K, 125GiB RAM)에서 실행 | submit-guide "예선 채점 하드웨어" | 제출 모델은 7B bf16 까지 여유. 8GB 는 개발 제약일 뿐 |
| **빌드 시점엔 인터넷 허용**, 런타임만 차단 | faq 공지 1 | 가중치는 Dockerfile `RUN snapshot_download` 로 baking. git LFS 불필요 |
| 공개 배경 USD 는 연습용 1종, 실제 대회는 **다른 배경일 수 있음** | faq 공지 3 | 랜드마크 좌표 룩업테이블·씬 의존 하드코딩 금지 |
| 2026.R03: 데이터셋 생성기 bbox GT 오류 수정(바닥에 놓인 캔·넘어진 자전거류의 좌표가 위로 치우침) | faq 변경이력 | R03 이전에 만든 학습 데이터 폐기·재생성. `TRAINER_AUTO_SCENES` 로 대량 생성 가능 |
| CCTV 프레임은 ROS optical 규약, `/tf_static` 의 `world→camera_id` 를 K 와 그대로 결합 | api-reference "CCTV 카메라 프레임" | 축 뒤집기 불필요 (geometry.py 가 그대로 구현) |
| 지면은 단일 평면이 아님(도로/잔디 10cm 단차) → 카메라별 `ground_height` | api-reference | `get_cctv_ground_height(id)` 를 카메라마다 사용 |
| 이미지 QoS BEST_EFFORT depth 1 / `/tf_static` TRANSIENT_LOCAL | api-reference QoS | TF 는 latched 구독으로 즉시 수신. 이미지는 최신 1장만 |

## 1. 전체 파이프라인

```
voice_command ──► CommandParser(규칙 → VLM 폴백) ──► Interpretation{target, landmark, relation, situation}
                                                         │
CCTV[i] image + K + tf_static + ground_height ──► CameraModel[i]
                                                         │
   각 카메라: HybridDetector(YOLO-ft → GroundingDINO) ──► target/anchor bbox
              detection_to_world(bbox 바닥점, 지면평면) ──► A_w, T_w
              relation_scores(A_w, T_w) 로 명령의 relation 과 일관성 점수
                                                         │
   카메라 선택 = argmax( 탐지신뢰도 ⊕ 관계일관성 )       ▼
   사람 문제면 SituationClassifier(VLM, crop)  ──► payload ──► GroundingResult.from_payload ──► submit_grounding
```

## 2. 결정과 근거

### 2.1 탐지: 파인튜닝 YOLO 를 1순위, Grounding DINO 를 2순위 (Hybrid)
- **왜**: 대상 어휘는 닫힌 집합(cola_can, mug, cracker_box, umbrella, pencilcase, sunblock, tissue, bicycle, kickboard, trashcan, person + 랜드마크)이고, 주최측이 **정답 bbox 가 붙은 합성 데이터 생성기**를 제공한다. 같은 렌더러에서 나온 학습 데이터로 튜닝한 닫힌 집합 탐지기가, 제로샷 개방어휘 탐지기보다 그 클래스에서는 정밀도·속도 모두 앞서는 것이 일반적이다. 특히 1280×720 원경에서 "필통", "선크림" 같은 소형 물체는 Grounding DINO 제로샷이 약한 지점이다(실측 필요).
- **왜 Grounding DINO 를 버리지 않나**: (a) 대회 배경이 바뀔 수 있음, (b) 명령에 어휘 밖 표현이 올 수 있음, (c) YOLO 학습이 마감 전 안 끝나도 제출 가능. 안전망이다.
- **대안**: OWLv2(속도 ↑, 정밀도 ↓), YOLO-World(개방어휘+속도). 셋 다 VRAM 은 2GB 이하라 제약 아님.
- **리스크**: 합성→합성이라 도메인 갭은 작지만, R03 이전 데이터로 학습했다면 z 가 위로 치우친 라벨을 배웠을 수 있다 → 반드시 재생성.

### 2.2 VLM: Qwen2.5-VL-7B-Instruct (제출), 3B + 4bit (로컬 개발)
- **역할을 제한**: 명령 파싱 폴백 + 사람 문제 situation 분류. 좌표 추정에는 쓰지 않는다. VLM 의 픽셀 좌표 출력은 분산이 크고, 기하 역투영이 결정적·검증 가능하기 때문이다.
- **왜 7B**: 채점 머신 48GB 에서 bf16 약 16GB. 여유가 크므로 양자화로 정확도를 깎을 이유가 없다. 로컬 8GB 에선 3B-4bit(≈3GB) 로 같은 코드 경로를 테스트하고, 최종 검증만 공유 서버에서 7B 로 한 번 돌린다.
- **왜 InternVL2 가 아닌가**: transformers 네이티브 지원 여부와 오프라인 로딩 편의(trust_remote_code 없이 로드)에서 Qwen2.5-VL 이 유리. 성능 차이는 **실측 전까지 근거 없음**.
- **리스크**: `transformers` 버전에 따라 Grounding DINO post-process 반환 키(`labels`/`text_labels`)와 Qwen 클래스명이 달라진다. requirements 를 고정했고 pipeline.py 가 두 키를 모두 처리한다. **requirements 의 버전 조합은 빌드로 실측해야 한다** — 내가 확신하는 것은 "4.5x 대에 둘 다 있다" 까지다.

### 2.3 좌표: bbox 중심이 아닌 **바닥 중앙점** + 카메라별 지면 평면
- 테스트(`tests/test_geometry.py`)로 확인: 1.7m 사람을 bbox 중심으로 역투영하면 1.7m 오차, 바닥점이면 0.
- **미확정 사항(실측 필요)**: GT 의 `target_coord` 가 **물체 바닥점**인지 **중심**인지. R03 공지 문구("바닥에 떨어뜨려 배치하는 물체의 좌표가 실제 위치보다 위쪽으로 기록")는 GT 가 물체 중심(prim 원점)일 가능성을 시사한다. 데이터셋 생성기 GT 와 우리 역투영 결과를 비교해 `object_height_m` 사용 여부를 정한다. 캔·머그 같은 소형물은 어느 쪽이든 10cm 이내라 영향이 작고, 사람(≈0.85m 차이)·자전거가 문제다.
- **미확정 사항 2**: left/right/front/behind 의 기준 시점. 구현은 **카메라 시점**(front = 카메라에 더 가까움). 랜드마크 정면 기준일 가능성도 있다. 연습 플랫폼 채점 결과(`on_score` 의 `scores` dict)를 relation 별로 로깅해 확인한다.

### 2.4 카메라 선택: 탐지 신뢰도 + 관계 일관성 결합
- 같은 이름 랜드마크가 여러 카메라에 있다(R02 패치 근거). 대상 신뢰도만으로 고르면 엉뚱한 카메라를 고를 수 있다 → 명령의 relation 과 기하적으로 맞는 (anchor,target) 쌍이 있는 카메라에 가중치.
- 랜드마크를 못 찾은 카메라는 0.6 감점. 값은 임의이며 실측 튜닝 대상.

### 2.5 시간 제한 대응
- `mission.time_limit` 은 값이 공개돼 있지 않다. 파이프라인은 deadline 을 계산해 카메라 순회를 중단하고, situation VLM 은 2초 이상 남을 때만 돈다. **미제출보다 부분점수 제출이 낫다**는 가정으로 fallback payload 를 낸다(카메라·해석 항목은 맞을 수 있음).
- 모델은 `connect()` **전에** 로드한다. 첫 라운드에 로딩 지연이 들어가면 안 된다.
- 카메라 수 × 탐지 1회가 주 비용. CCTV 가 8대 이상이면 배치 추론으로 바꿀 필요 (실측 후 결정).

### 2.6 오프라인
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 을 이미지에 박아, 네트워크가 있어도 다운로드를 시도하지 않게 한다 → 로컬에서도 채점과 동일 경로.
- Dockerfile 마지막에 프로세서 로드 자가검증 RUN. baking 실패 시 빌드가 실패해 제출 전에 드러난다.
- `network_mode: host` 라 `--network none` 을 그대로 못 쓴다 → `scripts/verify_offline.sh` 가 iptables 로 외부만 차단.

## 3. 우리가 놓치기 쉬운 리스크 (우선순위순)
1. **토큰 하드코딩 위치**: compose 의 `MARC_TOKEN: "${MARC_TOKEN:-...}"` 기본값이 제출값이다. 비워두면 0점. Private 리포 확인.
2. **태그 push**: `git push origin qualifier` 를 따로 해야 한다. 일반 push 로는 안 올라간다.
3. **R03 데이터 재생성** 누락.
4. **Blackwell CUDA wheel**: RTX PRO 5000/5060 Ti 는 sm_120 → cu128 이상 torch 필요. cu121/cu124 wheel 은 커널이 없어 실행 시 실패한다.
5. **ROS_DOMAIN_ID / RMW**: 플랫폼과 동일해야 하고 RMW 는 Fast DDS 고정. Cyclone 으로 바꾸면 디스커버리 실패.
6. **Python 3.10 vs 3.11**: 에이전트 컨테이너는 ros:humble(3.10). Isaac Sim 쪽 3.11 과 섞지 말 것.
7. **이미지 용량**: 7B 가중치 포함 시 25GB 내외. 주최측 빌드 시간이 길어질 뿐 규정 위반은 아니지만, 불필요한 패키지(ultralytics 미사용 시) 는 빼라.
8. **`on_mission` 재진입**: 라운드 제한시간을 넘기면 다음 미션이 겹쳐 올 수 있다 → agent.py 는 busy 락으로 스킵한다. 스킵 = 그 라운드 0점이므로 time_margin 을 보수적으로.

## 4. 실측이 필요한 항목 (추측하지 않은 것)
- `mission.time_limit` 실제 값, CCTV 대수
- GT 좌표 정의(바닥점/중심), relation 기준 시점
- Grounding DINO base 의 소형물 재현율 @1280×720, 카메라당 추론 시간(48GB 머신/8GB 머신)
- Qwen2.5-VL-7B situation 분류 정확도 (accident/emergency/abnormal 구분은 라벨 정의 자체가 모호)
- requirements.txt 버전 조합의 실제 빌드 성공 여부
