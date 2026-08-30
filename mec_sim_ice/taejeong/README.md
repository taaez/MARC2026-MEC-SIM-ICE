# taejeong — Stage 1 그라운딩 파이프라인 (일반화 설계)

명령 텍스트 → 파싱(target/landmark/relation 힌트) → 탐지 결과에 기하 계산 →
카메라·랜드마크 인스턴스·relation·**중심점 좌표** 확정 → GroundingResult payload.

## 설계 원칙 (대회 요구사항: 문제·배경이 바뀜, 씬 하드코딩 금지)

- **특정 시나리오/문항/좌표/씬 배치를 코드가 참조하지 않는다.** chungmu 는 테스트
  말뭉치로만 쓰고, 기하 테스트는 아예 **합성 가상 씬**(카메라를 수학적으로 만들어
  물체를 투영)으로 검증한다.
- **클래스별 크기·높이 상수 없음** — 물체/랜드마크의 높이·footprint 는 매 라운드
  그 자리에서 bbox + 카메라 기하로 추정한다.
- 어휘(클래스 목록)만 공식 로스터(config.yaml)에 닫혀 있고, 판정 로직은 전부 기하.
- 임계값은 `GroundingConfig` 로 외부화 — 연습 채점 역산으로 값만 보정, 코드 불변.
- 탐지기 무관 — `Detection(label, bbox, score)` 만 맞추면 YOLO/DINO 어느 쪽이든 연결.

## 모듈

| 파일 | 역할 |
|---|---|
| `command_parser.py` | 닫힌 어휘 텍스트 파싱. target 확정, landmark 타입 힌트, relation 힌트+confident, 사람 문제 pose 버킷, stage2 문형 |
| `grounding.py` | 기하 엔진: 역투영(중심점), 높이 추정, 'on' 지지면 재투영, relation 기하 판정, 최근접 랜드마크 인스턴스 선택, 카메라 선택 |
| `situation_resolver.py` | 공식 채점 규칙 situation = f(pose × support) 1:1 구현 (사람 문제) |
| `content-image-2026.R03/` | 공식 content 이미지에서 추출한 참조 파일 (규칙표·로스터·chungmu GT) |

## 파이프라인 사용 흐름

```python
from command_parser import parse_command
from grounding import Detection, CameraModel, camera_from_ros, resolve

parsed = parse_command(voice_command)
# 탐지기(YOLO 등)를 카메라별로 돌려 Detection 리스트 구성
dets = {cam_id: [Detection(label, (x1,y1,x2,y2), conf), ...] for ...}
cams = {cam_id: camera_from_ros(cam_id, cctv_info, tf_static, ground_h) for ...}

decision = resolve(parsed, dets, cams)
if decision is None:
    ...  # target 미탐지 → (예정) Grounding DINO 폴백 → 그래도 없으면 부분점수 payload
payload = decision.to_payload()   # GroundingResult 필드 구조 그대로
```

## 핵심 결정 로직 (전부 기하, 하드코딩 없음)

- **좌표 = 중심점(팀 결정)**: bbox 바닥중앙 역투영은 '카메라 쪽 가장자리'가 잡히므로
  footprint 반경만큼 밀어 중심을 얻고, z 는 (바닥 + 기하 추정 꼭대기)/2.
- **높이 추정**: bbox 상단 광선을 footprint 먼쪽 가장자리에서 매칭 — CCTV(물체보다
  높은 카메라)에서 bbox 상단은 먼쪽 윗모서리이기 때문. 클래스 상수 불필요.
- **랜드마크 확정(팀 결정)**: 명령에 타입이 있으면 그 타입의 3D 최근접 인스턴스,
  없으면 전체 랜드마크 중 3D 최근접. "car" 힌트는 차량 변형 전체 허용.
- **'on' 판정**: 이미지상 겹침(저비용 사전판정) → anchor 상면 평면으로 재투영 →
  **재투영이 target 을 anchor 중심으로 끌어당길 때만 채택** (앞/뒤 지면에 서 있는
  물체는 재투영하면 오히려 멀어지므로 자동으로 걸러짐 — 합성 씬으로 검증).
- **relation**: 텍스트 confident 면 텍스트 우선 → 명령이 '빈 자리/열린 바닥'이라는
  영역을 가리키면 near(확인된 의미 예외) → 그 외에는 기하(on/above/beside/near).
  **near/beside 는 XY 평면 OBB 표면거리로 가른다** (중심거리 아님, 임계값 0.63m).
  규칙·유도는 리포 루트의 `MARC2026_Chungmu_near_beside_OBB_analysis.md`. 문서의
  0.58m 는 공식 OBB 척도의 값이고, 우리는 같은 쌍을 단안으로 0.61m 로 재므로
  **우리 척도의 경계인 0.63m 로 재보정**했다 (2026-08-30, 39라운드 재채점 확인).
  방향(behind/front/left/right)은 텍스트가 방향을 말할 때만 기하로 확인 — 기준
  시점은 카메라 광축의 지면 투영으로 **가정**(채점 정의 미공개, 실측 검증 항목).
- **사람 문제**: pose 버킷으로 후보 자세 좁혀 사람 선택 → landmark 위('on')면
  landmark 를 support 로 공식 규칙표 조회, 아니면 fallback(자세별 기본값).
  relation 은 None 제출.

## 테스트

```bash
python3 test_command_parser.py            # chungmu 39 말뭉치 + 신규 일반화 문장 7
<venv>/python test_situation_resolver.py --verify-yaml   # 골든 28 + 전수 171 + 공식 YAML 대조
<venv>/python test_grounding.py           # 합성 가상 씬 12종 (numpy 필요)
```

합성 씬 테스트가 검증하는 것: 역투영 왕복 정합, 중심점 복원(±수 cm), 동일 클래스
다중 인스턴스 중 최근접 선택, 텍스트 타입 우선, 'on' 재투영 채택/기각(가짜 on 거부),
beside/near 기하 구분, 확정 텍스트 relation 우선, 방향 relation, pose 필터+situation
연동, prone=accident, 카메라 선택, payload 형식, 미탐지시 None.

## 남은 일 (연구실 데스크탑)

1. 탐지기 어댑터: YOLO 출력 → `Detection` 변환 + participant_app 배선
2. 미탐지 폴백: Grounding DINO 연결 (뒤로 미룸 — 사용자 결정)
3. `GroundingConfig` 임계값·방향 기준시점·"person vs pose클래스 제출 문자열"을
   연습 자가 채점 역산으로 보정 (코드 로직은 그대로)
4. 사람 문제의 지면 토큰(grass/road) 판정 수단 — 현재는 미판정 시 공식 fallback 경로
