# YOLO11 학습 가이드 (test/ 스테이징)

`training/train_yolo11.py` + `training/det_classes.py`를 실제로 돌려서 데이터셋 생성 → 라벨 변환 →
학습 → 스모크 테스트까지 검증하는 순서. **이 범위까지만 구현돼 있음** — `real_agent_vla.py`
(명령 해석/좌표 변환/GroundingResult 제출)와 `participant_app.py`/`launch.sh`/`docker compose`
연동은 아직 구현 전이라 이 가이드에 없음. 그건 `test/README.md`의 "다음 단계"에 이어서 진행.

전제: Ubuntu 22.04 네이티브 + 시뮬레이션 플랫폼 빌드 완료(`정리폴더/MARC2026_요약_v3.md` §0).
Windows 개발 PC에서는 이 문서의 명령을 그대로 실행할 수 없음(Docker 자체가 없음, 이미 확인됨) —
대회 지정 환경(또는 그 환경을 재현한 Ubuntu 머신)에서 실행해야 함.

---

## 0-0. 스톡 YOLO11 대비 무엇이 다른가

배포 모델 B가 스톡 Ultralytics와 어디가 같고 어디가 다른지. **결론부터: 모델 쪽 변경은 0이다.**

| 항목 | 스톡 YOLO11 | 모델 B | 비고 |
|---|---|---|---|
| 백본 / 넥 / 헤드 | `yolo11s.yaml` 그대로 | **그대로 (무수정)** | 커스텀 yaml 없음 |
| stride | [8, 16, 32] | **[8, 16, 32]** | P2/stride-4 헤드는 시도 후 폐기 (§3-1) |
| loss | CIoU + DFL + BCE | **그대로** | `loss.py` / `tal.py` 미수정 |
| 스케일 | n (기본) | **s** | 채널 폭·깊이만 증가, 격자 해상도는 동일 |
| 클래스 수 | 80 (COCO) | 34 (`DET_CLASSES`) | 헤드 nc만 교체 |
| imgsz | 640 | **1280** | CCTV 원본 1280×720에 맞춘 값 |
| 증강 | Ultralytics 기본 | **기본 그대로** | 오버샘플·GT copy-paste는 전부 실패해서 미적용 |

즉 B는 **"스톡 yolo11s를 imgsz 1280으로 34클래스에 파인튜닝한 것"** 이상도 이하도 아니다.
데이터·구조·loss 어느 쪽으로도 개입하지 않은 이 버전이, 시도한 6개 실험 중 유일하게
34클래스 전부 무손실이었다 (A/C 오버샘플, D/E GT샘플링, P2 헤드는 전부 다른 클래스를 깎았다).

**의도적으로 안 한 것**:

- **P2/stride-4 헤드** — 2026-08-25 시도, 34클래스 전부 하락으로 폐기 (§3-1)
- **오버샘플링 / GT copy-paste** — 타겟 3클래스 ±0.02 노이즈에 다른 클래스 9~21개 하락
- **loss 교체(NWD 등)** — B에는 적용하지 않는다. 소형 객체 문제는 B를 건드리지 않고
  **별도 서브모델**로 분리해서 다룬다 → `small_object_detection/소형객체_서브모델_가이드.md`

이 "B는 손대지 않는다"는 원칙이 지금 프로젝트의 기본 전제다. B를 개선하려는 시도는 지금까지
전부 다른 클래스로 손해가 번졌다.

---

## 0. 실행 전 체크리스트

- [ ] `bash simulation-platform/marc.sh setup` 완료 (base 이미지 빌드)
- [ ] `pip3 install ultralytics opencv-python-headless` (호스트에서 학습할 경우)
- [x] `training/det_classes.py`의 `DET_CLASSES`(34종)는 **`marc2026_demo` 501씬(11,477 detection)
      전수 대조 완료** (2026-08-25, `정리폴더/작업기록_2026-08-25.md` §1) — 데이터에 있는데
      목록에 없는 클래스/그 반대 모두 0개. `marc2026_chungmu`(실제 쓸 시나리오)로도 한 번 더
      대조 권장하지만, 아래 2단계에서 미확인 클래스가 나오면 스크립트가 경고로 알려주니
      그때 여기 추가해도 됨.

---

## 1. 데이터셋 생성 (dataset-gen)

```bash
cd simulation-platform
cp .env.example .env   # ENV_MARC_SCENARIO=marc2026_chungmu 로 수정 (기본값은 marc2026_demo)
bash marc.sh dataset-gen
# 또는 배치(헤드리스) 생성: TRAINER_AUTO_SCENES=100 HEADLESS=true docker compose --profile dataset-gen up
```

결과 위치: **`<레포 루트>/trainer_output/<시나리오>/scene_<번호>/<카메라>.json`** + `.png` +
`_overlay.png` — `simulation-platform/trainer_output/`이 아니라 **레포 루트** 기준임에 주의
(`docker-compose.yml`의 dataset-gen 볼륨이 `simulation-platform/`의 상위를 가리킴).

**주의 (2026.R03 패치)**: 이 패치 이전 이미지는 눕혀지는 물체(캔·티슈)와 넘어지는 물체(자전거·
킥보드·쓰레기통)의 bbox가 틀려있었음. `git pull`로 최신 패치를 받은 뒤 생성한 이미지인지 확인.

---

## 2. 라벨 변환 (JSON → YOLO txt, 이미 구현됨)

```bash
cd training
python3 train_yolo11.py convert --scenes ../trainer_output/marc2026_chungmu --out dataset
```

`convert`가 하는 일: 씬별 JSON의 `detections[].bbox`(픽셀 좌표)를 YOLO 포맷(`class_id x_center
y_center width height`, 0~1 정규화)으로 변환해서 `dataset/labels/train|val/*.txt`에 저장하고,
이미지는 `dataset/images/train|val/`로 복사, `dataset/data.yaml`도 자동 생성함. 별도 변환
스크립트 필요 없음 — 이 한 줄로 끝.

확인할 것:
- 출력 로그의 `N train / M val images` 개수가 scene 수 × 카메라 수와 대략 맞는지 (0/0이 나오면
  이미지 해상도를 못 찾아서 전부 스킵된 것 — 이미 수정된 버그이니 최신 코드인지 확인)
- **`WARNING: N class(es) ... NOT in det_classes.DET_CLASSES`** 로그가 뜨는지 — 뜨면 그 클래스를
  `det_classes.py`에 추가하고 **`convert`부터 다시 실행** (클래스 순서가 `data.yaml`의 id와
  직결되므로, 이미 변환/학습을 시작했어도 클래스 목록을 바꾸면 처음부터 다시 해야 함)
- 라벨 줄 수 합계: `wc -l dataset/labels/train/*.txt | tail -1`

---

## 3. YOLO11 학습 (이미 구현됨)

```bash
python3 train_yolo11.py train --data dataset/data.yaml --epochs 200 --imgsz 1280
```

- 기본 가중치 `yolo11n.pt` (nano, 기본 커널/채널 — 아키텍처 수정 없음). 최초 실행 시 자동
  다운로드되므로 그 순간만 인터넷 필요 (런타임 인터넷 차단 규정은 채점 시점에만 적용, 학습은 무관).
- `imgsz=1280`은 CCTV 원본 해상도(1280×720)에 맞춘 것 — 임의값 아님, 나중에 추론 코드에서도
  같은 값을 유지해야 함.
- 데이터가 적어 정확도가 낮으면 `--weights yolo11s.pt`로 스케일만 올려서 재시도 가능.
- 결과: `runs/detect/train/weights/best.pt`, `results.png`(loss/mAP 곡선), `val_batch0_pred.jpg`
  (검증 이미지 예측 시각화 — 클래스가 제대로 잡히는지 눈으로 먼저 확인).
- **모든 `train` 실행은 전체 stdout/stderr을 `training_results/yolo_main_logs/ (또는 스크립트 위치 기준 training/logs/, 아래 참고)`에도 자동 저장함**
  (ANSI 코드/이모지/진행바 잔상은 파일에서 제거되고, 터미널에는 원래대로 컬러/진행바 그대로
  보임). 터미널 스크롤이 끝난 뒤 경고문구 다시 확인할 때 여기서 grep하면 됨.

**메모리 부족(`ConnectionResetError: [Errno 104]`, dataloader/`pin_memory` 스택트레이스)**:
데이터로더 워커가 OOM으로 강제종료된 것. `--batch`/`--workers`를 낮춰서 재시도:

```bash
python3 train_yolo11.py train --data dataset/data.yaml --epochs 200 --imgsz 1280 --batch 4 --workers 2
# 그래도 죽으면: --batch 2 --workers 0
```
`dmesg | tail -30`에 `Out of memory: Killed process`가 있으면 OOM 확정.

### 3-1. 소형 객체 대응 — P2 헤드는 시도했다가 폐기함

`tissue`/`sunblock`/`disposable_cup`/`tumbler`/`umbrella`/`pencilcase` 같은 소형 객체의
mAP50-95가 낮은 문제(§5 참고)에 P2/4 탐지 헤드 추가(`--p2` 옵션, `yolo11n-p2.yaml`)를
2026-08-25에 시도했으나 **효과가 없어서 코드에서 제거함** — 자세한 경과/수치는
`정리폴더/작업기록_2026-08-25.md` §4~7 참고. 요지: 사전학습 가중치 전이까지 다 고쳐서 재시도한
버전(v2)도 34개 클래스 **전부**(목표였던 소형 클래스 포함) 스톡보다 낮게 나옴 — 데이터량
대비 새 헤드를 처음부터 학습시키는 부담이 P2가 주는 이득보다 큰 것으로 판단, 다시 시도하지
않기로 함. 소형 객체는 대신 §5의 데이터 추가 쪽으로 접근.

---

## 4. 스모크 테스트 — 모델 단독 확인 (플랫폼/참가자 프로그램 불필요)

`real_agent_vla.py`/`participant_app.py`가 아직 없어도, 학습된 모델이 이미지 한 장에서 뭘
잡아내는지는 바로 확인할 수 있음:

```python
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
result = model.predict("dataset/images/val/<파일명>.png", imgsz=1280, conf=0.35)[0]
result.save()   # 박스 그려진 이미지 저장해서 확인 (result.show()는 디스플레이 필요)
print(result.names)
```

타겟/랜드마크/사람 자세 박스가 잘 잡히는지, 특히 소형 객체(`pencilcase` 등)가 아예 안 잡히는지
확인. 소형 객체가 계속 안 잡히면 §5, `정리폴더/작업기록_2026-08-25.md` §4~7(P2 헤드 시도했다가
폐기한 경과) 참고 — 구조 수정보다 데이터 추가 쪽이 지금은 더 안전한 방향으로 판단됨.

---

## 5. 다시 학습이 필요한 신호

- 특정 클래스가 스모크 테스트에서 계속 안 잡힘 → 데이터 부족, 1단계 반복(씬 더 생성 —
  `TRAINER_AUTO_SCENES` 늘리거나 여러 번 실행) 후 재학습
- 특정 각도·거리의 카메라에서만 실패 → 그 카메라가 잘 나오는 씬 위주로 추가 생성
- 눕혀진/넘어진 물체 인식률이 유난히 낮음 → 1단계의 R03 패치 확인부터
- 소형 객체(`pencilcase`, `cracker_box`, `cola_can` 등) 인식률이 유난히 낮음 → 구조 수정(P2
  헤드)은 시도했다가 효과 없어서 폐기함(§3-1). 씬 더 생성해서 데이터로 밀어붙이는 쪽으로 접근
  (`TRAINER_AUTO_SCENES` 늘리거나 여러 번 실행 후 재학습, 위 항목과 동일한 처방)

---

## 다음 단계 (이 가이드 범위 밖 — `test/README.md` 참고)

- [ ] `det_classes.py`/`train_yolo11.py` 검증되면 `demo/`로 이동
- [ ] `demo/real_agent_vla.py` 작성 (명령 파싱 + 픽셀→world 좌표 변환 + `GroundingResult` 구성)
- [ ] `weights/best.pt`를 `demo/weights/best.pt`로 복사
- [ ] `demo/launch.sh` / `docker compose`로 실제 플랫폼과 연동 테스트 → 이때부터 로그 기반
      원인 분리, 채점 결과 확인, 도커 격리 테스트 같은 절차가 필요해짐 (별도 가이드로 작성 예정)
