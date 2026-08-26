# 작업 기록 — 2026-08-27 (작성: jo2june)

브랜치 `ptj/stage1-vla-integration`, 커밋 `86f2528`. Windows PC 에서 오프라인
채점 루프(`mec_sim_ice/score.py`)만으로 진행 — Isaac Sim 안 씀.

**요약: chungmu 39문항 Stage1 평균 53.03 → 57.80 (+4.77). 라이브 경로는
`DET_CONF` 수정으로 추가 +4.3 이 예상된다(오프라인 채점엔 이미 반영된 값).**
같은 records, 같은 채점식으로 이전 코드와 비교한 실측이다 (개선 19건 / 악화
5건 / 동일 15건, 악화는 전부 좌표 미세 변화로 판정 뒤집힘 없음).

> **팀원 확인 요청** — 아래 4개 변경과 "다음 할 일"을 한번 검토해 주세요.
> 특히 ③(과적합 방지 가드)이 대회 취지("씬 하드코딩 금지")에 맞게 들어갔는지,
> 그리고 ⑤(탐지기 재학습 필요)는 학습 담당의 판단이 필요합니다.

---

## 무엇을 바꿨나 (4개, 모두 `86f2528`)

### ① anchor 좌표: 중심점 → **지면 접촉점** (`grounding.py` `Located.ground_point`)

- anchor 만점이 0/39 인 것에서 출발. camera/landmark/relation 이 전부 정답인
  clean 라운드 12개만 골라 `locate()` 의 파라미터(footprint push × 제출 z)를
  스윕한 결과, anchor 는 `base` z 가 압도적: **평균 오차 1.00m → 0.64m**.
- 원인은 기하적으로 자명하다: 자판기·택시처럼 큰 랜드마크는 center 를 쓰면
  물체 높이의 절반이 그대로 z 오차가 된다. **채점 GT 의 정의에 관한 사실**이라
  씬이 바뀌어도 성립한다.
- target(작은 물체)은 반대로 center 가 맞아서 그대로 뒀다.

### ② 카메라 게이트: 랜드마크 ∩ 구역 **교집합** (`grounding.py` `resolve()`)

- 카메라 오선택 9건 중 4건이 같은 형태였다: 명령이 랜드마크와 구역을 **둘 다**
  말하는데("Find the person sitting on top of the trash can **in the parking
  lot**") 기존 코드는 랜드마크 게이트가 걸리면 `area_hint` 를 버렸다.
- trash_can 은 3대 카메라에 보여서 랜드마크만으로 안 좁혀진다 → 구역이 그중
  하나로 갈라 준다. 두 단서는 독립이므로 교집합. **교집합이 비면 랜드마크
  게이트로 폴백하므로 나빠지는 경로가 없다.**
- 이 4건이 전부 뒤집혔다: hf08 25→94.45, sarH2 0→55.41, sar01/sarH4 15→34.8.

### ③ 새 배경 안전장치: `AREA_DOMINANCE_RATIO = 2.0`

- ②로 구역 게이트가 랜드마크가 있을 때도 후보를 걸러내게 되면서, **새 배경에서
  구역 지표가 오도하면 정답 카메라를 배제할 수 있는 경로가 새로 생겼다.**
- 그래서 "밀집" 판정에 명시 기준을 뒀다: 지표 1등 카메라가 2등의 **2배 이상**일
  때만 그 구역으로 인정. 근거는 씬이 아니라 말의 의미다 — 차가 4대 대 3대로
  갈리면 어느 쪽도 "주차장"이라 단정할 수 없다. 애매하면 게이트가 풀린다.
- **이 코퍼스에선 가드 유무로 점수가 완전히 동일** = 지금 비용 0, 새 배경 전용
  보험. `test_grounding.py` T18(교집합 동작)/T19(애매하면 해제)가 이 성질을
  고정한다. T6-table-z 는 새 anchor 규약으로 갱신했다. 3개 테스트 스위트
  전부 PASS.

### ④ 라이브 `DET_CONF` 0.35 → 0.25 (`demo_agents/taejeong_agent_vla.py`)

- 임계값만 바꿔 39문항 재채점: **0.35 는 53.48, 0.10/0.15/0.25 는 전부 57.80.**
- 즉 라이브 경로가 쓰던 0.35 가 그 자체로 4.3점을 깎고 있었다 (오프라인 채점
  스크립트는 기본이 0.25 라 이 손실이 안 보였다). 셋이 동일하므로 오검출
  여지가 가장 적은 0.25 선택.

## 새 도구

- `taejeong/validation/cmp_reports.py` — 채점 리포트(JSON) 2개를 문항 단위로
  비교. "평균은 올랐는데 뭐가 대신 나빠졌나"를 매 수정마다 확인하는 용도.
  ```bash
  python3 mec_sim_ice/score.py --json before.json
  # ... 수정 ...
  python3 mec_sim_ice/score.py --json after.json
  python3 mec_sim_ice/taejeong/validation/cmp_reports.py before.json after.json
  ```

## 측정으로 확인한 사실: 이제 상한을 막는 건 로직이 아니라 **탐지 recall**

39라운드 전수로 "GT 카메라에서 GT target 이 검출되는가"(같은 클래스 탐지가 GT
좌표 2m 이내)를 측정했다:

- **GT 카메라에서 target 검출 23/35** (이미지 없는 4라운드 제외)
- 클래스 편중이 뚜렷: `tumbler` 5/5 인데 **`pencilcase` 0/3, `sunblock` 0/1,
  `umbrella` 5/11**
- 남은 카메라 오선택 5건(lf06/lf07/hf02/hf06/rtC2, 전부 GT=rig_3_b)은 **전부 이
  문제다**: GT target 을 투영하면 정답 카메라 화면 한복판인데(예: lf07
  px=(947,233)) 탐지기가 놓친다. **conf 0.03 까지 낮춰도, imgsz 2560 까지 올려도
  안 나온다.** 정답 카메라에 target 이 없고 다른 카메라에 디코이 쌍이 있으니
  로직으로 구별할 신호 자체가 없다.
- lf11/rt04b(카메라 정답인데 target 40점이 0점)도 같은 원인.

## 앞으로 할 일 (우선순위순)

1. **탐지기 재학습** — `pencilcase` / `sunblock` / `umbrella` 중심으로 작은
   물체·원거리 샘플 보강. 위 측정이 근거. **가장 큰 지렛대**이고 로직으론 못
   푼다. → 학습 담당 확인 필요
2. **중복 인스턴스 오선택 (landmark 오답 4건, 약 +5 여지)** — 씬에 같은 클래스가
   여러 개면 "가장 가까운 쌍"이 디코이에 진다(rt01: 진짜 tumbler conf 0.33/GT
   0.72m vs 디코이 conf 0.77/GT 10.45m). 문구가 "**empty** parking space" 라
   "빈 자리 = 차 옆이 아니다"로 읽으면 풀리지만, 근접 선호를 뒤집는 규칙이라
   **과적합 위험이 커서 이번엔 안 넣었다.** 상세는 RUNBOOK "남은 개선 후보".
   → 넣을지 팀 판단 필요
3. **재수집** — s1_001~004(이미지 없는 라운드, 빈 제출 4건 확정)와
   sar04(GT 가 6대 어디에도 안 보이는 데이터 이상). 데스크탑에서
   `MARC_RECORD_ONLY=1` 로 몇 분이면 된다.
4. **라이브 검증** — 오프라인 target 점수는 근사치이므로, 위 변경들을 Isaac Sim
   라이브로 한 번 확인 (기존 라이브 기준선 44.03 대비).
5. Docker 빌드 검증(전날 리포트의 미완 항목) 그대로 남아 있음.

## 검증 방법 (재현)

```bash
git pull
python3 mec_sim_ice/score.py            # Stage1 평균 : 57.80 이 나와야 함
cd mec_sim_ice/taejeong
python3 test_grounding.py               # PASS
python3 test_command_parser.py          # PASS
python3 test_situation_resolver.py      # PASS
```

상세 근거·표는 `taejeong/RUNBOOK.md` 의 "2026-08-27 개선" / "지금 상한을 막고
있는 것은 로직이 아니라 탐지 recall" 절에 정리돼 있다.
