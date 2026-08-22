# YOLO11 학습 → 실행 → 테스트 가이드

`demo/real_agent_vla.py`(RealVLAGrounding) + `demo/train_yolo11.py`를 실제로 돌려서
검증하는 순서. 전제: Ubuntu 22.04 네이티브 + ROS2 Humble + 시뮬레이션 플랫폼 빌드 완료
(`MARC2026_요약_v2.md` §0). Windows 개발 PC에서는 이 문서의 명령을 그대로 실행할 수
없고, 대회 지정 환경(또는 그 환경을 재현한 머신)에서 실행해야 함.

---

## 0. 실행 전 체크리스트

- [ ] `bash simulation-platform/marc.sh setup` 완료 (base 이미지 빌드)
- [ ] `pip3 install ultralytics opencv-python-headless` (호스트에서 학습/개발할 경우. 도커
      제출 이미지는 `demo/Dockerfile`에 이미 반영되어 있음)
- [ ] `demo/real_agent_vla.py`의 `DET_CLASSES`가 실제 데이터와 맞는지 아직 검증 전이면
      1단계 끝나고 바로 확인할 것 (아래 1-3)

---

## 1. 데이터셋 생성 (dataset-gen)

```bash
cd simulation-platform
cp .env.example .env   # ENV_MARC_SCENARIO=marc2026_chungmu 로 설정
bash marc.sh dataset-gen
# 또는: TRAINER_AUTO_SCENES=100 HEADLESS=true docker compose --profile dataset-gen up
```

결과: `trainer_output/marc2026_chungmu/scene_<번호>/<카메라>.json` + `.png` + `_overlay.png`

**주의 (요약 문서 §9 재확인)**: 2026.R03 패치 이전 이미지는 눕혀지는 물체(캔·티슈)와
넘어지는 물체(자전거·킥보드·쓰레기통)의 bbox가 틀려있었음. `git pull`로 최신 패치를
받은 뒤 생성한 이미지인지 확인.

### 1-1. 클래스 이름 검증 (학습 돌리기 전 필수)

`real_agent_vla.py`의 `TARGET_CLASSES`/`LANDMARK_CLASSES`는 `problems.yaml`의 문항 id에서
추론한 것이지 실측 확인이 아님. 아무 `scene_*/*.json` 하나를 열어서 `detections[].class`
값이 그 목록과 정확히 일치하는지 먼저 확인:

```bash
python3 -c "
import json, glob
p = glob.glob('trainer_output/marc2026_chungmu/scene_*/*.json')[0]
data = json.load(open(p))
print(sorted({d['class'] for d in data['detections']}))
"
```

목록이 다르면 `demo/real_agent_vla.py`의 `TARGET_CLASSES`/`LANDMARK_CLASSES`를 실제 값으로
고치고, **이미 변환/학습을 시작했다면 처음부터 다시** (클래스 순서가 `data.yaml`의 id와
직결되므로 중간에 바꾸면 기존 라벨이 다 어긋남).

---

## 2. 라벨 변환

```bash
cd demo
python3 train_yolo11.py convert \
  --scenes ../simulation-platform/trainer_output/marc2026_chungmu \
  --out dataset
```

확인할 것:
- 출력 로그의 `N train / M val images` 개수가 scene 수 × 카메라 수와 대략 맞는지
- `dataset/labels/train/*.txt` 몇 개를 열어 라인이 있는지 (전부 빈 파일이면 클래스 이름이
  안 맞아서 `_to_yolo_line`이 전부 스킵한 것 — 1-1로 돌아가서 재확인)
- `dataset/data.yaml`의 `names:` 목록이 의도한 순서인지

---

## 3. YOLO11 학습

```bash
python3 train_yolo11.py train --data dataset/data.yaml --epochs 100 --imgsz 1280
```

- 기본 가중치는 `yolo11n.pt` (nano, 기본 커널/채널 — 아키텍처 수정 없음). 첫 실행 시
  자동 다운로드되므로 최초 1회는 인터넷 필요.
- 데이터가 적어서(수백~수천 장) 정확도가 낮으면 `--weights yolo11s.pt`로 스케일만 올려서
  재시도 (커맨드 한 줄만 바꾸면 됨).
- 학습 결과: `runs/detect/train/weights/best.pt`, `runs/detect/train/results.png`(loss/mAP
  곡선), `runs/detect/train/val_batch0_pred.jpg`(검증 이미지 예측 시각화 — 클래스가
  제대로 잡히는지 눈으로 먼저 확인).

### 3-1. 가중치 배치

```bash
mkdir -p weights
cp ../runs/detect/train/weights/best.pt weights/best.pt   # train_yolo11.py 실행 위치 기준 상대경로 주의
```

`real_agent_vla.py`의 `DET_WEIGHTS` 기본값이 `demo/weights/best.pt`이므로 여기 없으면
다음 단계에서 바로 실패함.

---

## 4. 코드 실행

### 4-1. 플랫폼 먼저 기동

```bash
cd simulation-platform
xhost +local:root   # GUI 모드 최초 1회
bash marc.sh platform
# "Auto-plan: waiting for a participant to register..." 로그까지 기다릴 것
```

### 4-2. 참가자 프로그램 (호스트에서, 반복 개발용)

```bash
cd demo
export MARC_TEAM_ID=u1
export MARC_TOKEN=<team-token>
./launch.sh u1
```

이 시점에 뜰 수 있는 초기 실패들:
- `FileNotFoundError: weights/best.pt` → 3-1을 건너뜀
- `ModuleNotFoundError: ultralytics` / `tf2_ros` → 호스트에 `pip3 install ultralytics
  opencv-python-headless` 안 함 / ROS2 Humble의 tf2_ros 패키지 미설치
- tf lookup 관련 경고(`tf lookup world<-... failed`) → `_pixel_to_world`가 가정한 프레임
  이름(`"world"`, `cam_id` 그대로)이 실제와 다름. 아래 5-3 참고

### 4-3. 도커로 (제출과 동일한 방식)

```bash
cd demo
docker compose up
```

---

## 5. 테스트

### 5-1. 스모크 테스트 — 카메라/모델만 단독 확인

플랫폼·참가자 프로그램 전체를 띄우기 전에, 모델이 이미지 한 장에서 뭘 잡아내는지부터
확인하면 원인 분리가 쉬움:

```python
from ultralytics import YOLO
model = YOLO("demo/weights/best.pt")
result = model.predict("trainer_output/marc2026_chungmu/scene_0/cam01.png",
                        imgsz=1280, conf=0.35)[0]
result.show()   # 또는 result.save() 로 박스 그려진 이미지 저장해서 확인
print(result.names)
```

여기서 타겟/랜드마크 박스가 안 잡히면 학습(2~3단계)의 문제고, 여기까지 잘 나오면
파이프라인(명령 파싱/카메라 선택/좌표 변환) 쪽 문제로 좁혀짐.

### 5-2. 문항 수 줄여서 반복 연습

전체 문항을 매번 다 돌리면 원인 파악이 느림. `marc.sh select`로 1~2문항만 골라서 반복:

```bash
cd simulation-platform
bash marc.sh select marc2026_chungmu   # 체크리스트에서 문항 1~2개만 선택
```

GUI 모드면 `MARC Runtime` 패널에 팀 총점/Stage1 점수가 실시간으로 뜸. 문항 바꾸면 그
시나리오로 `select`를 다시 돌려야 함(안 하면 전체 문항 폴백).

### 5-3. 라운드별 로그로 원인 분리

`participant_app.py`가 매 라운드 `voice_command`를 로그로 찍고(§4 흐름), `real_agent_vla.py`
는 카메라를 못 찾으면 `[vla] no camera matched target=... landmark=... situation=...`
경고를 남김. 로그를 보고 아래 표로 원인을 좁힌다.

| 증상 | 원인 후보 | 확인 위치 |
|---|---|---|
| 매 라운드 `no camera matched`, target_type 빈 문자열 | `_parse_command`가 명령을 못 읽음 | 로그의 실제 `voice_command` 문장 vs `TARGET_CLASSES`/`RELATION_WORDS`/`SITUATION_WORDS` 키워드 대조 |
| target_type은 잡히는데 카메라 계속 못 찾음 | 학습된 모델이 그 클래스를 실제로 못 잡음 | 5-1 스모크 테스트로 해당 카메라 이미지 단독 확인 |
| 카메라는 찾는데 좌표가 이상함(원점 근처 `[0,0,ground_z]` 반복) | tf lookup 실패 — 프레임 이름 불일치 | `ros2 run tf2_tools view_frames` (플랫폼 실행 중 별도 터미널)로 실제 프레임 트리 확인 후 `_pixel_to_world`의 `"world"`/`cam_id` 수정 |
| person 문항에서 situation이 항상 틀림 | pose 휴리스틱 임계값이 실제 자세와 안 맞음 | `_classify_situation`이 반환하는 값을 로그로 찍어서 실제 라운드의 상황과 비교, `_situation_from_keypoints` 임계값 조정 |
| 좌표는 대충 맞는데 오차가 큼(수십 cm~미터) | ground_height 평면 가정이 카메라마다 다른데 잘못 씀 | `get_cctv_ground_height(cam_id)`가 카메라별로 다른 값을 주는지 로그 확인 (요약 문서 §4: 배경 전체를 하나의 평면으로 가정하면 안 됨) |

### 5-4. 채점 결과로 정량 확인

`on_score` 콜백(`participant_app.py`의 `_on_score`)이 라운드별 점수를 로그로 남김. 로컬
채점은 참고용이지만(공식 채점은 비공개 시나리오), 상대적인 개선/퇴보는 여기서 판단.
개선 반복: 문항 좁혀서(5-2) → 로그로 원인 분리(5-3) → 코드/데이터 수정 → 재실행 → 점수 확인.

### 5-5. 도커 격리 테스트 (제출 전 필수)

로컬 개발 환경에서는 되는데 도커에서 안 되는 경우가 잦음(가중치 baking 누락, PYTHONPATH
문제 등). 제출 전 반드시:

```bash
# 1) 새 위치에 깨끗이 clone
git clone <내 저장소> /tmp/marc-check && cd /tmp/marc-check/demo

# 2) 인터넷 끊고 (또는 방화벽으로 차단) 빌드+실행
docker compose up
```

빌드 단계에서 `ultralytics` pip 설치·`yolo11n-pose.pt` 다운로드가 다 끝나 있어야 하고,
`demo/weights/best.pt`가 git에 커밋되어 있어야 함(용량 크면 Git LFS 고려 — `.gitattributes`
에 이미 LFS 설정이 일부 있으니 `simulation-platform/.gitattributes` 참고).

---

## 6. 다시 학습이 필요한 신호

- 5-3에서 "모델이 클래스를 못 잡음"으로 원인이 좁혀짐 → 데이터 더 생성(1단계 반복,
  `TRAINER_AUTO_SCENES` 늘리기) 후 재학습
- 특정 각도·거리의 카메라에서만 실패 → 그 카메라가 잘 나오는 씬 위주로 추가 생성
- 눕혀진/넘어진 물체(우산·자전거 등) 인식률이 유난히 낮음 → 1단계의 R03 패치 확인부터
