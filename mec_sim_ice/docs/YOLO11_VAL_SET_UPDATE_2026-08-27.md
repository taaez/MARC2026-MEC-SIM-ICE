# YOLO11 고정 val 세트 도입 (2026-08-27)

## 배경

지금까지 `test/train_yolo11.py convert --scenes <dir> --val-ratio 0.1`은 **train과 val을 같은
dataset-gen 생성 배치에서 무작위로 나눴다.** 즉 val이 train과 같은 시드/분포에서 뽑혀서, val
AP가 올라도 실제 일반화(미션 recall)를 보장하지 않는 문제가 있었다. 이를 고치기 위해 train과
val을 **완전히 분리된 두 생성 배치**로 두는 방식으로 바꿨다 — train은 기존 `marc2026_demo`
누적분, val은 새로 생성 중인 `marc2026_validation`.

## `test/train_yolo11.py` 변경 사항

`convert` 서브커맨드에 `--train-scenes` / `--val-scenes` 옵션 쌍을 추가했다. 기존 `--scenes`
(단일 디렉토리 무작위 분할) 방식은 그대로 남아있다 — 하위 호환, 다른 용도로 계속 쓸 수 있음.

```
# 기존 방식 (그대로 동작, 변경 없음)
python3 train_yolo11.py convert --scenes ../trainer_output/marc2026_chungmu --out dataset

# 새 방식: train/val을 서로 다른 생성 배치에서 각각 전량 사용 (무작위 분할 없음, 리키지 없음)
python3 train_yolo11.py convert \
  --train-scenes ../trainer_output/marc2026_demo \
  --val-scenes   ../trainer_output/marc2026_validation \
  --out dataset
```

두 옵션은 함께 줘야 하고(`--train-scenes`만 주면 에러), `--scenes`와는 상호 배타적이다.

내부적으로 씬 하나를 YOLO 라벨로 바꾸는 로직을 `_process_scenes()`로 분리했다(기존 코드와
동일한 로직, 위치만 옮김). `convert()`는 이제 이 함수를 소스 디렉토리당 한 번씩(레거시 모드는
train/val 분할된 부분집합에 대해, 새 모드는 각 디렉토리 전체에 대해) 호출하는 형태다.

동작 검증: `trainer_output/temp`(백업된 기존 씬)에서 2씬/1씬짜리 미니 소스로 `--train-scenes`/
`--val-scenes` 조합을 돌려봤고, `images/train`에 2장 + `images/val`에 1장이 정확히 나뉘어
들어가는 것과 `data.yaml`이 정상 생성되는 것을 확인했다.

## val 세트 생성 완료 (2026-08-27 12:17)

자유배치(`marc2026_demo` 시나리오) 모드로 500씬 생성 완료. 500/500 전부 유효(json+png 누락
없음).

- 왜 500장인가: 카메라별 지분이 가장 적은 카메라(rig_1_a/rig_7_a, 전체의 ~10%)도 클래스당
  30개 이상 인스턴스를 확보하는 선.
- 두 번 크래시/버그를 거쳤다:
  1. 첫 시도(500장 한 번에 요청)는 GPU VRAM 부족으로 크래시 — 36시간째 idle 상태로 떠 있던
     `platform` 컨테이너가 VRAM 5GB를 이미 점유하고 있었음. 정지 후 재시도로 해결.
  2. 그 다음 짠 "배치 끊기면 병합 후 재요청" 자동 재시도 스크립트에 버그가 있었음: 씬 폴더가
     컨테이너(root)로 생성돼서 일반 계정 `mv`/`rm`이 Permission denied로 실패하는데 exit
     status를 체크 안 해서 "성공"으로 오인. 로그엔 "200장 병합 완료"라 찍혔지만 실제로는
     0장이었음 — 즉시 중단하고 재설계.
  3. 재설계: 씬 번호가 기존 개수에 이어서 붙는다는 걸 확인했으므로(0부터 재시작하지 않음)
     병합/리네임 자체가 불필요했음. 매 배치 종료 후 `docker run --rm --entrypoint chown ...`
     (이미 로컬에 있는 이미지 재사용, sudo 불필요)으로 소유권만 고치고, 남은 개수만 다시
     `TRAINER_AUTO_SCENES`로 요청하는 것으로 단순화. 214 → 414 → 500으로 3번 만에 완료.

## 현재 파일 위치 (중요 — 이름이 학습 데이터와 같음)

- **`trainer_output/marc2026_demo/`**: 방금 만든 새 500씬 (val 후보). **이 이름은 원래 학습
  데이터가 쓰던 이름과 같다** — 원본 601씬 학습 데이터는 이 작업을 시작하기 전에 사용자가
  미리 `trainer_output/temp/`로 백업해둔 상태다.
- **`trainer_output/temp/`**: 원본 601씬 학습 데이터 (백업본).

즉 지금 상태로 바로 `convert`를 돌리면 안 된다 — `--train-scenes ../trainer_output/marc2026_demo`가
가리키는 게 새로 만든 val 후보 500씬이지 원래 학습 데이터가 아니기 때문. 두 폴더를 제자리로
돌려놔야 함 (파일 이동은 사용자가 직접 처리하기로 함):

```bash
mv trainer_output/marc2026_demo trainer_output/marc2026_validation
mv trainer_output/temp trainer_output/marc2026_demo
```

## 다음 할 일

1. [ ] 위 두 `mv`로 폴더 제자리 복구 (val 후보 → `marc2026_validation`, 백업 → `marc2026_demo`)
2. [ ] `convert --train-scenes ../trainer_output/marc2026_demo --val-scenes ../trainer_output/marc2026_validation --out dataset` 실행
3. [ ] `train` 서브커맨드로 재학습, 기존 `best.pt`와 `mec_sim_ice/score.py` 등으로 전후 비교
   (구 가중치는 보관해둘 것)
