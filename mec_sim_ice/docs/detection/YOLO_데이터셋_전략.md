# YOLO11 탐지기 데이터셋 전략 브리프 (2026-08-27)

Stage 1 탐지기(YOLO11 + `DET_CLASSES` 34종)의 **데이터셋 개선 방향**을 정리한 문서.
새 세션이 이 문서만 읽고 바로 논의에 참여할 수 있도록 사실·진단·결정·미확인을 분리해 적었다.

관련 문서: `mec_sim_ice/training/YOLO11_학습_가이드.md` (학습 실행 절차),
`mec_sim_ice/training/det_classes.py` (34종 정의).

---

## 0. 한 줄 요약

**Stage 1 점수의 상한을 막고 있는 것은 로직이 아니라 탐지 recall이고, 그 원인은
"데이터가 적다"가 아니라 "클래스 × 카메라 × 크기 조합에 구멍이 있고, val 세트가 그걸
측정하지 못한다"일 가능성이 크다.** 다음 작업은 실측 → 구멍 메우기 → 고정 val로 검증.

---

## 1. 현재 상태 (실측 수치)

| 항목 | 값 | 출처 |
|---|---|---|
| Stage1 오프라인 평균 (39문항) | **57.80** | `mec_sim_ice/score.py`, 2026-08-27 |
| Stage1 라이브 기준선 | 44.03 (2026-08-25) | `taejeong/validation/runs/chungmu_full39_2026-08-25.jsonl` |
| GT 카메라에서 target 검출 | **23/35** (이미지 없는 4라운드 제외) | RUNBOOK 2026-08-27 측정 |
| GT 카메라에서 landmark 검출 | 29/33 | 동일 |

클래스별 편중:

| 클래스 | 검출 |
|---|---|
| `tumbler` | 5/5 |
| `umbrella` | 5/11 |
| `pencilcase` | **0/3** |
| `sunblock` | **0/1** |

**카메라 오선택 5건(lf06 / lf07 / hf02 / hf06 / rtC2)은 전부 GT = `rig_3_b`.**
확인된 것:

- GT target을 투영하면 정답 카메라 **화면 한복판**에 있다 (lf07: px=(947, 233))
- conf를 **0.03까지 낮춰도** 안 나옴 → 임계값 문제 아님
- imgsz를 **2560까지 올려도** 안 나옴 → 해상도 문제 아님
- → 모델의 **응답 자체가 0**. 피처가 없다.

`lf11` / `rt04b` (카메라는 맞는데 target 40점 0점)도 같은 원인 (umbrella 미검출).

---

## 2. 카메라 6대 실측 (`marc2026_chungmu.yaml`에서 추출)

rig 4개 → 미션 카메라 6대. 전부 `focal_length: 5.0`, 1280×720.

| 카메라 | rig 타입 | 카메라 z | 지면 z | **지면 위 높이** | 랜드마크 수 | 객체 종류(인스턴스) |
|---|---|---|---|---|---|---|
| **rig_3_b** | post_dual | 18.775 | 16.1 | **2.68m** (최저) | **3** (최소) | 7 (7) |
| rig_3_c | post_dual | 18.783 | 16.1 | 2.68m | 5 | 4 (4) |
| rig_1_a | post_single | 19.546 | 16.6 | 2.95m | 7 | 11 (14) |
| rig_7_a | wall_single | 20.609 | 16.8 | 3.81m | 8 (최다) | 9 (13) |
| rig_2_b | wall_dual | 21.437 | 16.4 | **5.04m** (최고) | 5 | 10 (18) |
| rig_2_c | wall_dual | 21.436 | 16.4 | 5.04m | 7 | 3 (3) |

**핵심**: `rig_3_b`가 유일한 2.68m 저설치 post → **물체가 가장 크게 잡힌다.**
(팀이 관찰한 "rig_3_b 근접 촬영 효과"의 정체.) 반대로 `rig_2_b/c`는 5.04m 벽설치라
가장 작게 잡힌다. **지면 높이가 카메라마다 다르다**(16.1~16.8) — 역투영 시 카메라별
`ground_height`를 써야 하는 이유.

`rig_3_b`는 랜드마크가 3개뿐이라 카메라 게이트가 쓸 단서도 가장 빈약하다.
오선택 5건이 전부 여기 몰린 것과 무관하지 않다.

---

## 3. 시나리오의 클래스 → 카메라 결합 (중요)

`marc2026_chungmu.yaml`의 `missions:`는 **카메라마다 어떤 객체/랜드마크가 놓이는지를
고정**하고 있다.

| 카메라 | 랜드마크 | 객체 |
|---|---|---|
| rig_7_a | normal_car_1, sports_car, kick_scooter, bicycle, trash_can, hydrant, police_car, suv | tumbler×3, disposable_cup×3, cracker_box, cola_can, umbrella, person_lying_down, person_reaching, person_prone, person_sitting |
| rig_2_b | bench×2, kick_scooter, normal_car_1, trash_can | cola_can×4, tissue×3, juice×2, disposable_cup×2, person_walking×2, cracker_box, umbrella, person_prone, person_sitting, person_reclining |
| rig_2_c | suv, bicycle×4, vending_machine, trash_can | umbrella, person_walking, person_sitting |
| rig_1_a | bench×4, picnic_table, bicycle, trash_can | **mug**, **sunblock**, cola_can×2, juice×2, tissue×2, tumbler, umbrella, person_sitting×2, person_lying_down, person_bending, person_walking_away |
| **rig_3_b** | trash_can, bicycle, picnic_table | **pencilcase**, umbrella, cola_can, disposable_cup, cracker_box, person_bending, person_lying_down |
| rig_3_c | kick_scooter×2, trash_can, taxi, bench | tumbler, umbrella, person_reaching, person_bending |

결합이 강한 클래스:

- **`pencilcase` → `rig_3_b`에만, 1개**
- **`sunblock` → `rig_1_a`에만, 1개**
- **`mug` → `rig_1_a`에만, 1개**
- `trash_can` → **6대 전부** (그래서 카메라 특정에 도움이 안 됨)

### chungmu 미션이 전혀 쓰지 않는 클래스 4종

`normal_car_2`, `sports_car_2`, `postbox`, `person_crouching` — 시나리오 전체 등장 0회.
채점이 안 묻는다는 뜻이지 생성이 불가능하다는 뜻은 아니다 (§9-1 참고).

### 39문항의 target 분포

`umbrella` 11 / `tumbler` 7 / `person_bending` 5 / `person_sitting` 4 /
`person_prone` 3 / `pencilcase` 3 / `sunblock` 2 / `person_lying_down` 2 / `mug` 2

---

## 4. 현재 학습 데이터셋 구성 (약 600씬)

`marc2026_chungmu`로 dataset-gen 생성. 4개 블록의 커리큘럼:

| 구간 | 씬 수 | 의도 |
|---|---|---|
| 1 ~ 246 | 246 | 베이스라인 — 전체 객체 랜덤 생성 |
| 247 ~ 349 | 103 | 소형 물체 6종 mAP 개선 — `rig_3_b` 근접 효과 활용, 소형 객체 + (차량 제외) 랜드마크만 생성 |
| +150 | 150 | 사람 캐릭터 외형 다양성 (female / professor / student_weight) |
| +100 | 100 | `person_*` 9종 pose 인스턴스 수·크기 확보 (`rig_3_b` 재활용) |
| **합계** | **599~600** | |

**train/val 분할은 `train_yolo11.py convert`의 내부 비율로 갈렸다.**
`n_val = max(1, int(600 × 0.1))` = **val 60씬 / train 540씬**, seed 0 무작위 셔플.

이 데이터셋으로 학습한 가중치가 현재 `best.pt`이고, §1의 수치가 그 모델의 성적이다.
학습 결과 **val AP는 크게 올랐다** (팀 보고).

> 가중치·데이터셋은 Isaac Sim 데스크탑에만 있다. `.gitignore`가 `*.pt`, `/trainer_output/`,
> `demo/weights/`를 제외하므로 Windows 개발 PC에는 없다 (2026-08-27 확인).

---

## 5. 진단 — val AP ↑ 인데 미션 recall은 23/35인 이유

두 숫자는 모순이 아니다. **val이 다른 질문에 답하고 있었다.**

1. **val이 train과 같은 분포다.** 같은 600씬에서 무작위 10%를 뽑았으므로, val 60씬에도
   4개 블록이 같은 비율로 들어간다. 특히 `rig_3_b` 근접(= 물체가 크게 잡히는 쉬운) 씬이
   val에도 섞인다. → val AP는 "학습한 것과 같은 종류를 잡나"를 측정하지, 일반화를
   측정하지 않는다.
2. **희소 클래스는 통계적으로 부족하다.** `pencilcase`는 씬당 1개(rig_3_b)이므로
   val 60씬 = 인스턴스 약 60개. 클래스별 AP가 안정되려면 100+ 필요하고 30 밑은
   무의미하다. **60은 노이즈 바닥** — "pencilcase AP 올랐다"는 상당 부분 분할 운일 수 있다.
3. **전체 AP 하나로는 스케일 결손이 안 보인다.** 큰 인스턴스를 늘리면 전체 AP는
   자동으로 오른다. 실패는 원거리 소형(lf07 rig_3_b px y=233 = 화면 상단 = 원경)에서 난다.
   COCO식 `AP_small`(<32²px) / `AP_medium` / `AP_large`로 나눠 봐야 잡힌다.

### 의심되는 근본 결함: 클래스 × 카메라 결합

247~349에서 소형 객체를 `rig_3_b`(가장 크게 잡히는 카메라)에 집중 생성했다.
배치가 고정이라는 전제에선 합리적이지만, **배치가 바뀐다면 결합을 강화한 것**이다 —
모델이 `pencilcase`를 "rig_3_b 시점 · 그 스케일"에서만 배웠을 수 있다.

**이것이 §9-1로 반드시 확인해야 할 첫 번째 사항이다.**

---

## 6. 폐기한 접근 — 2D GT 샘플링 (copy-paste 증강)

GT 라벨로 인스턴스를 크롭해 다른 학습 이미지에 랜덤 위치·스케일로 붙여넣는
오프라인 증강(`gt-sample` 서브커맨드)을 검토했으나 **채택하지 않기로 했다.**

폐기 사유:

- **잃어버린 정보를 복원할 수 없다.** 그림자·접지, 원근 스케일, 광원 일관성, 가림 관계가
  전부 빠진다. 이걸 다 보정하려면 간이 렌더러를 다시 짜는 꼴인데, 진짜 렌더러
  (`marc.sh dataset-gen`)가 옆에 있다.
- **하드 사각 seam을 train에만 주입** → sim→sim인데 분포 이동이 생겨 clean AP를 깎을 수 있다.
- **과적합**: 희소 클래스일수록 크롭 소스가 1~3장 → 그 특정 인스턴스 픽셀을 암기.
  `NOTICES.md`의 "대회 배경이 바뀔 수 있음"과 정면 충돌.
- 검토한 구현의 겹침 게이트(`IoU <= 0.05`)는 **큰 붙여넣기가 소형 GT를 완전히 덮어도
  통과**시킨다 → 라벨 오염(false negative 샘플) 발생.
- 목표가 "원거리 소형 보강"인데 기본 `scale_range`가 0.8~1.8(확대 위주)이라 방향이 반대였다.

**대안**: 씬 추가 생성(실제 텍스처·그림자·원근 정합·인도메인) + Ultralytics 기본 증강
(`scale` 상향, `mosaic` 유지 — `close_mosaic` 늦추기). 참고로 Ultralytics의 `copy_paste`는
세그멘테이션 마스크가 필요해 bbox 라벨에는 **무효**다.

과거 경위: P2/4 탐지 헤드 추가(구조 수정)도 2026-08-25에 시도했다가 34개 클래스 전부
스톡보다 낮게 나와 폐기했다 (`YOLO11_학습_가이드.md` §3-1). **구조가 아니라 데이터로
접근한다**는 것이 팀 결론이다.

---

## 7. 지표 체계 — 무엇을 최대화할 것인가

**`records/` 39라운드 점수(57.80)는 목표 함수가 아니다.**
실제 대회는 **같은 카메라·같은 환경이지만 객체 위치가 전부 바뀌고 명령도 다르다.**
`records/`는 배치를 한 번 뽑은 실현값이므로, 그걸 최대화하면 그 배치에 과적합한다.

| 지표 | 역할 | 상태 |
|---|---|---|
| **고정 val 세트** (클래스 × 카메라 × 크기버킷) | **주지표.** 배치가 바뀌어도 유지되는 능력 | **없음 — 만들어야 함** |
| `records/` 39라운드 (`score.py`) | **회귀 가드.** "나빠지지 않았나"만 확인 | 있음 |

lf06/lf07/hf02/hf06/rtC2가 뒤집히는지는 **증상 확인**이지 목표가 아니다.
올바른 해석: "탐지기가 소형 클래스를 특정 스케일에서 전혀 못 잡는다"는 일반 결함의 표본.

명령 변화는 **탐지기가 아니라 파서** 문제다 (`taejeong/command_parser.py`의
`OBJECT_SYNONYMS`/`LANDMARK_SYNONYMS`, `situation_resolver.py`). 데이터셋 작업과 독립된 축.

---

## 8. 할 일 (순서대로)

### 1) 현재 600씬 실측 ← 선행 필수

`test/dataset/labels/`에서 뽑을 표 두 개:

- **클래스 × 카메라 교차표** — 결합이 데이터에 박혔는지. 0인 칸이 곧 구멍
- **클래스 × 크기버킷 교차표** (`<32²` / `32²~96²` / `>96²` px) — 스케일 커버리지

부수적으로: 클래스별 총 인스턴스 수, 라벨 없는 배경 이미지 비율, 화면밖·degenerate bbox,
`DET_CLASSES` 미등장 클래스, 600이 씬 수인지 이미지 수인지.

**이 표가 "무엇을 더 생성할지"를 그대로 지정한다. 추측 불필요.**

### 2) 고정 val 세트 생성 + 동결

- dataset-gen **별도 실행 → 별도 출력 디렉토리**
- New Scene 패널에서 **객체·랜드마크 전부 체크** (34종 커버, §3의 4종 포함)
- **6대 카메라에 고르게** 배치 — 시나리오의 클래스→카메라 결합을 **의도적으로 깬다**
- 분량: 최희소 클래스 **100 인스턴스 이상**. 100~150씬으로 시작해 1)의 교차표로 확인 후 증량
- **철칙**: ① 한 번 만들고 동결(학습 데이터 늘릴 때마다 재생성 금지) ② train에 절대 안 섞기
  ③ R03 이후 플랫폼인지 확인

### 3) val 교체 — 코드 변경 불필요

```bash
# train: 600씬 거의 전부 (val 로는 1씬만 빠짐)
python3 train_yolo11.py convert --scenes ../trainer_output/marc2026_chungmu --out dataset --val-ratio 0

# 고정 val: 전량 val 쪽으로
python3 train_yolo11.py convert --scenes ../trainer_output/<val세트> --out valset --val-ratio 1.0

# val 교체
rm -rf dataset/images/val dataset/labels/val
cp -r valset/images/val dataset/images/val
cp -r valset/labels/val dataset/labels/val
```

`data.yaml`은 이미 `val: images/val`이라 수정 불필요.
**부수 이득**: 지금 학습에서 빠져 있는 60씬이 회수된다.

### 4) 부족분 타겟 생성 → 재학습

1)의 교차표에서 빈 칸을 메운다. 현재 예상 방향:

- 실패 클래스(`pencilcase`/`sunblock`/`umbrella`)를 **전 카메라 × 전 거리대**로 생성.
  `rig_3_b`(2.68m, 크게)부터 `rig_2_b/c`(5.04m, 작게)까지 **스케일 스펙트럼**을 채운다
  (247~349가 한쪽 끝만 채웠음)
- 247~349에서 빠진 차량 클래스 결손 확인
- 배경 이미지 비율 과다 여부 확인

### 5) 평가

- **주**: 고정 val의 **크기버킷별** AP/recall
- **보조**: `records/` 39라운드 회귀 확인

```bash
python3 mec_sim_ice/score.py --weights <구 best.pt> --json before.json
python3 mec_sim_ice/score.py --weights <새 best.pt> --json after.json
python3 mec_sim_ice/taejeong/validation/cmp_reports.py before.json after.json
```

**구 `best.pt`는 반드시 보관.** 전후 비교하려면 같은 방식으로 다시 돌려야 한다.

---

## 9. 미확인 — 추측으로 메우지 말 것

1. **dataset-gen이 미션 로스터를 따르는가, 자유 배치인가?**
   `missions:`는 39개 채점 문항용 로스터이고, dataset-gen의 New Scene 패널은
   10 objects + 15 landmarks를 독립적으로 고를 수 있다 (`YOLO11_학습_가이드.md` §0).
   **배치 도메인이 시나리오 존을 따르는지는 미확인.**
   → §8-1의 클래스 × 카메라 교차표가 이걸 그대로 답한다. **가장 중요한 미확인 사항.**
2. **600이 씬 수인가 이미지 수인가.** 씬이면 카메라 수만큼(최대 6배) 이미지가 나온다.
3. **1~246 구간이 2026.R03 패치 이후 생성인가?** 이전이면 눕힌 물체(캔·티슈)·넘어진 물체
   (자전거·킥보드·쓰레기통)의 bbox GT가 위로 치우친 오염 라벨이다 → 그 구간 폐기·재생성.
4. **구 `best.pt`가 보관돼 있는가** (전후 비교에 필요).
5. **대회 배경이 chungmu 그대로인가.** `NOTICES.md`는 "배경이 바뀔 수 있다"고 명시.
   팀 가정은 "같은 카메라·환경, 객체 위치와 명령만 변경". 더 엄한 쪽으로 설계해도 비용은 작다.

---

## 10. 참고 — 검증 도구 현황

| 도구 | 용도 | 비고 |
|---|---|---|
| `mec_sim_ice/score.py` | 39라운드 오프라인 채점 | `--weights` `--conf` `--imgsz` `--json` |
| `taejeong/validation/cmp_reports.py` | 리포트 2개 문항 단위 비교 | 개선/악화 분해 |
| `taejeong/check_interfaces.py` | `DET_CLASSES` ↔ 가중치 ↔ 파서 어휘 정합성 | 재학습 후 필수 |
| `tools/run_offline_real_vla.py` | RealVLA 파이프라인 오프라인 재실행 | |

**없는 것 (필요하면 새로 만들어야 함)**:

- 데이터셋 구성 분석 (§8-1의 교차표)
- 클래스 × 크기버킷 AP/recall 리포트 (§8-5 주지표)
- 미션 레벨 recall 측정 — §1의 23/35는 2026-08-27에 애드혹으로 측정됐고
  **커밋된 스크립트가 없어 재현 불가**

`train_yolo11.py`는 건드리지 않는 방침 (§8-3이 코드 변경 없이 되는 이유).
