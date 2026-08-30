# training/ — YOLO11 학습 코드

`det_classes.py` / `train_yolo11.py` (2026-08-29 `test/`에서 `training/`으로 이동, 데이터셋·결과물은 `datasets/`·`training_results/`로 분리)
(정식 제출 구조인 `demo/`를 아직 확정 안 된 코드로 건드리지 않기 위함).

## 여기 있는 것
- `det_classes.py` — `TARGET_CLASSES`/`LANDMARK_CLASSES`/`POSE_CLASSES`/`DET_CLASSES`.
  근거: `../docs/MARC2026_객체_상태_분류.md` (아직 완결 아님, 계속 갱신 중).
- `train_yolo11.py` — dataset-gen JSON → YOLO 라벨 변환 + 학습. 실행법/경로는 파일 상단 docstring 참고.

## 다음 단계 (검증되면 `demo/`로 이동)
- [ ] Ubuntu 환경에서 `bash simulation-platform/marc.sh dataset-gen`으로 씬 생성
- [ ] `python3 train_yolo11.py convert --scenes ../trainer_output/<시나리오> --out dataset` 실행,
      "WARNING: ... class(es) ... NOT in det_classes.DET_CLASSES" 로그 없는지 확인 (있으면
      `det_classes.py`에 클래스 추가 후 재실행)
  - [ ] 학습 1회 시험 (`python3 train_yolo11.py train --data dataset/data.yaml`) 정상 동작 확인
- [ ] 위 확인되면 `det_classes.py` + `train_yolo11.py`를 `demo/`로 이동, `real_agent_vla.py`가
      `det_classes`를 import하도록 연결
