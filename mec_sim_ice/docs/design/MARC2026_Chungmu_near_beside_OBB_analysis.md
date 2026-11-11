# MARC 2026 Chungmu `near`·`beside` 판단 기준 분석 기록

- 작성일: 2026-08-30
- 대상: `marc2026_chungmu.yaml`의 분실물–랜드마크 위치 관계
- 분석 목적: 명령문에 정답 관계어 `near` 또는 `beside`가 직접 나오지 않는 경우에도 사용할 수 있는 기하·언어 결합 규칙 수립
- 최종 권고안: **XY 평면 OBB 표면거리 임계값 `0.58 m` + 확인된 의미 예외 규칙**
- 상태: 공식 R03 에셋과 시나리오 변환을 이용한 오프라인 재구성 결과. 실제 실행 중인 Isaac Sim stage의 최종 OBB CSV로 한 번 더 확인할 필요가 있음.

> 이 문서는 대화의 축어록이 아니라, 대화에서 오간 질문·답변과 이후의 검증 및 정정 내용을 기술적으로 재구성한 기록이다. 수치가 바뀐 지점은 이전 판단을 삭제하지 않고 `초기 가설 → 추가 검증 → 최종 판단` 순서로 남겼다.

---

## 1. 결론부터 요약

### 1.1 최종 결론

Chungmu의 9개 고유 분실물–랜드마크 쌍은 **거리 하나만으로 `near`와 `beside`를 9개 모두 정확히 분리할 수 없다.** 특히 `police_car–umbrella`는 공식 OBB 계산에서 표면거리가 `0 m`인데도 정답은 `near`다. 명령문은 물체가 경찰차 바로 옆에 있다고 표현하지 않고, **“빈 주차 공간 안에 떨어진 우산”**이라는 영역 관계를 표현한다.

따라서 실제 구현 규칙은 다음처럼 정한다.

1. 명령문에 canonical label인 `near` 또는 `beside`가 명시되면 그 표현을 우선한다.
2. `next to`, `leaning against`, `propped against` 같은 유사 표현은 단독으로 `near` 또는 `beside`에 고정 매핑하지 않는다.
3. 명령문이 `in the empty parking space`처럼 **랜드마크 주변의 영역/공간을 지시하는 것으로 확인된 경우** `near`를 사용한다.
4. 그 외에는 지면상의 **XY OBB 표면거리**를 계산한다.
   - `d_xy <= 0.58 m` → `beside`
   - `d_xy > 0.58 m` → `near`

이 규칙에서 거리 부분만 적용하면 고유 9쌍 중 8쌍을 맞히며, `police_car–umbrella` 한 쌍만 틀린다. 확인된 영역 의미 예외까지 적용하면 현재 공개 Chungmu 9쌍을 모두 설명할 수 있다.

### 1.2 `0.58 m`의 의미

`0.58 m`는 주최 측이 공표한 공식 임계값이 아니다. 공식 R03 시나리오에 놓인 실제 에셋의 OBB를 재구성하여 다음 두 경계 사례의 중간값으로 정한 **Chungmu 전용 경험적 임계값**이다.

| 경계 사례 | 정답 | 공식 Isaac 방식 XY OBB 거리 |
|---|---:|---:|
| vending_machine–umbrella | beside | 0.469896 m |
| trash_can–umbrella | near | 0.696786 m |

중간값은 다음과 같다.

\[
\tau=\frac{0.469895694+0.696785604}{2}=0.583340649\ \text{m}
\]

구현에서는 이를 `0.58 m`로 사용한다. 회전 쿼터니언을 의도된 intrinsic XYZ로 해석한 대안 계산에서는 중간값이 `0.572715 m`였으므로, 반올림한 `0.58 m`는 두 해석 모두에서 같은 경계 결정을 낸다.

### 1.3 반드시 기억할 제한

- `0.58 m`는 보편적인 자연어 의미 경계가 아니다.
- 공개 Chungmu에는 고유 `beside` 표본이 2개뿐이다.
- OBB는 물체의 실제 메시 표면이 아니라 메시를 둘러싼 직육면체이므로 빈 공간을 포함한다.
- 아래 계산은 공식 R03 USD 에셋과 YAML 변환의 재구성이다. PhysX가 안정화된 **실행 중 stage**에서 직접 뽑은 값은 아니다.
- 따라서 최종 코드를 고정하기 전에 실행 중 Isaac Sim에서 9쌍의 OBB corner를 CSV로 추출해 재검증해야 한다.

---

## 2. 대화에서 질문이 발전한 순서

## 2.1 첫 질문: 명시되지 않은 `near`와 `beside`를 무엇으로 나눌 것인가

사용자 질문:

> “너가 생각하기에 chungmu 문제에서 near, beside가 갈리는 기준이 뭐인거 같아 명령에서 near, beside가 등장해서 명시된 경우를 제외하고, 텍스트 규칙과 Isaac sim상에서의 객체와 랜드마크 기하, 위치 관계를 모두 고려해서 골몰해줘”

당시 답변의 핵심은 다음과 같았다.

- 단순히 명령문에 들어간 영어 표현만 치환해서는 부족하다.
- `near`는 비교적 넓은 근방, `beside`는 랜드마크의 옆면 가까이에 붙은 국소 배치를 뜻할 가능성이 있다.
- 기하적으로는 prim 원점 간 거리보다 물체 외곽 사이의 거리가 더 적절하다.
- 후보 변수로 다음을 제시했다.
  - 객체–랜드마크 중심/원점 거리
  - 크기를 고려한 표면거리
  - 랜드마크 옆면 방향성
  - 같은 공간/영역 안에 있는지
  - 접촉 또는 기대어 있음 여부
- 결론적으로 **텍스트 의미를 먼저 사용하고, 애매할 때 OBB 표면거리를 사용해야 한다**는 방향으로 논의를 시작했다.

이 단계에서는 실제 9개 에셋의 OBB가 모두 확보되지 않았으므로 정확한 임계값을 확정할 수 없었다.

## 2.2 두 번째 질문: prim 기준점 거리, OBB, 표면거리가 정확히 무엇인가

사용자 질문:

> “USD prim 기준점 사이 거리 라는게 무슨 말이지 어떤 거리라는 거야 그리고 Isaac Sim에서 OBB와 표면거리가 정확히 뭘 얘기하는거지 예시를 들어서 자세하게 설명해줘”

당시 설명의 핵심은 다음과 같다.

### Prim 기준점 사이 거리

USD prim에는 world transform의 translation이 있고, 이를 그 prim의 원점 또는 기준점처럼 사용할 수 있다. 두 prim의 translation을 각각

\[
\mathbf{p}_A=(x_A,y_A,z_A),\quad \mathbf{p}_B=(x_B,y_B,z_B)
\]

라고 하면 원점 간 3차원 거리는

\[
d_{origin}=\|\mathbf{p}_A-\mathbf{p}_B\|_2
\]

이고, 지면 관계만 볼 때의 XY 거리는

\[
d_{origin,xy}=\sqrt{(x_A-x_B)^2+(y_A-y_B)^2}
\]

이다.

문제는 이 원점이 반드시 물체의 기하 중심이 아니라는 것이다. 에셋 제작자가 원점을 바닥 중앙, 메시 중심, 한쪽 모서리 또는 조립체 루트에 둘 수 있다. 큰 자판기와 작은 우산은 표면이 가까워도 원점끼리는 멀 수 있다. 따라서 원점 거리는 계산은 간단하지만, `beside` 판단에는 물체 크기가 반영되지 않는다는 한계가 있다.

### AABB와 OBB

- **AABB(Axis-Aligned Bounding Box)**: 월드 좌표축 X/Y/Z에 항상 평행한 경계 상자다. 물체가 회전하면 상자가 불필요하게 커질 수 있다.
- **OBB(Oriented Bounding Box)**: 물체의 방향을 따라 회전할 수 있는 경계 상자다. 회전된 자동차나 테이블의 외형을 AABB보다 더 잘 근사한다.

다만 Isaac Sim의 `compute_obb`는 항상 최소 부피 OBB를 최적화하는 함수라고 보면 안 된다. Isaac Sim 5.1의 구현은 `UsdGeom.BBoxCache.ComputeWorldBound`가 만든 bound를 사용하고, 그 범위·변환으로 OBB 중심, 축과 반길이를 구성한다. 즉 실제 메시를 둘러싼 실용적인 oriented bound이지, 모든 가능한 방향을 탐색한 수학적 최소 OBB라는 보장은 없다.

### OBB 표면거리

두 OBB의 중심 사이 거리가 아니라 **두 경계 상자에 속한 점들 사이의 최소 거리**를 뜻한다.

\[
d_{OBB}(A,B)=\min_{\mathbf{a}\in A,\mathbf{b}\in B}\|\mathbf{a}-\mathbf{b}\|_2
\]

- 상자가 떨어져 있으면 가장 가까운 두 표면 사이의 간격이다.
- 상자가 닿거나 겹치면 `0`이다.
- 물체의 크기를 자동으로 반영한다.

예를 들어 폭 2 m인 자판기 중심과 우산 중심이 1.8 m 떨어져 있어도, 자판기 반폭 1 m와 우산 반폭을 빼면 실제 경계 상자 사이 간격은 약 0.8 m 이하가 된다. 반대로 작은 쓰레기통과 우산 중심이 1.1 m 떨어져 있으면 표면 간격은 중심거리와 크게 다르지 않을 수 있다.

### 이 과제에서는 3D보다 XY가 우선

`near`와 `beside`는 주로 지면 위 평면 관계다. 그러므로 OBB의 8개 corner를 XY 평면으로 투영하고 두 footprint의 최소 간격을 구하는 방식이 더 직접적이다.

\[
d_{xy}(A,B)=\min_{\mathbf{a}\in P_{xy}(A),\mathbf{b}\in P_{xy}(B)}\|\mathbf{a}-\mathbf{b}\|_2
\]

여기서 `P_xy`는 OBB를 XY로 투영한 다각형이다. 높이가 다른 물체 때문에 3D 거리가 커지는 현상을 피할 수 있다. 이 문서에서는 `d_xy`를 주 판단값으로, 3D OBB 거리를 진단값으로 사용했다.

## 2.3 세 번째 질문: 실제 OBB를 알 수 있는가

사용자 질문:

> “너가 지금 각 객체 그니까 랜드마크나 타겟의 OBB를 알 수 있나? OBB 표면거리로 계산하면 beside, near이 설명되는지 검증해보고 싶은데”

당시 확인 결과는 두 단계로 나뉘었다.

### 처음 확보된 에셋

초기 작업 환경에서는 다음 에셋만 직접 확인할 수 있었다.

- picnic_table
- police_car
- taxi
- umbrella
- pencilcase

초기 로컬 bound 크기는 다음과 같았다.

| 에셋 | X 크기 (m) | Y 크기 (m) | Z 크기 (m) |
|---|---:|---:|---:|
| picnic_table | 2.830520 | 2.832985 | 1.029803 |
| police_car | 1.777625 | 3.730490 | 1.238751 |
| taxi | 1.807360 | 4.220717 | 1.310464 |
| umbrella | 0.070762 | 0.211211 | 0.050494 |
| pencilcase | 0.043894 | 0.171942 | 0.049884 |

그러나 `vending_machine`, `trash_can`, `hydrant`, `tumbler`, `sunblock` 에셋이 없어서 9개 전체를 동일한 기준으로 계산할 수 없었다. 이 때문에 일부 쌍은 대상 물체를 점으로 근사하는 proxy 계산을 사용했고, 그 결과를 확정값으로 취급하면 안 된다고 구분했다.

### 초기 부분 계산

초기 root-aligned 계산 또는 target-point proxy에서 다음 값들이 나왔다.

| 쌍 | 정답 | 초기 값/상태 |
|---|---:|---:|
| picnic_table–sunblock | beside | target 원점이 table footprint 안에 있어 proxy 0 m |
| picnic_table–pencilcase | near | 약 1.879580 m |
| police_car–umbrella | near | 약 0.439886 m |
| police_car–tumbler | near | target-point proxy 약 2.250886 m |
| taxi–tumbler | near | target-point proxy 약 3.978532 m |
| taxi–umbrella | near | 약 3.853471 m |

이 시점의 증거만 보면 `beside`가 매우 가까운 범위이고 `near`는 그보다 멀다는 가설은 가능했지만, 가장 중요한 `vending_machine–umbrella`와 `trash_can–umbrella`가 비어 있었다.

## 2.4 네 번째 질문: 누락된 5종 에셋을 구할 수 있는가, 기준거리는 얼마인가

사용자 질문:

> “vending_machine, trash_can, hydrant, tumbler, sunblock 이 5종에 대한 에셋을 나는 구할 수 있는 거야? 아님 나도 못구해? 실행 중인 Isaac Sim에서 9개 쌍의 OBB를 CSV로 자동 추출해서 본다고 했을 때 너가 생각했을 때 OBB 표면거리에 대해 near, beside를 가르는 기준점, 그니까 기준거리를 어느정도로하면 될 거 같아?”

### 에셋 확보 결론

5종은 모두 공식 MARC R03 content image 안의 runtime USD 에셋으로 확보할 수 있었다. 원본 제작 소스인 `.glb`가 그대로 남아 있다는 뜻은 아니지만, Isaac Sim이 실제 scene 생성에 쓰는 USD와 종속 파일은 존재했다.

| 종류 | 공식 content 내부 경로 |
|---|---|
| vending_machine | `/metacom2026/resources/assets/landmark/street/vending_machine/vending_machine.usd` |
| trash_can | `/metacom2026/resources/assets/landmark/street/trash_can/trash_can.usd` |
| hydrant | `/metacom2026/resources/assets/landmark/street/hydrant/hydrant.usd` |
| tumbler | `/metacom2026/resources/assets/object/tumbler/tumbler/tumbler.usd` |
| sunblock | `/metacom2026/resources/assets/object/etc/sunblock/sunblock.usd` |

주의할 점은 USD 파일 하나만 복사하면 텍스처나 하위 USD reference가 끊길 수 있으므로, 실제 환경으로 가져갈 때는 관련 에셋 디렉터리 전체를 보존해야 한다는 것이다.

### 초기 임계값 제안과 정정

누락 에셋을 확보하기 전에는 일부 값만 보고 **`0.25 m` 안팎**을 임시 경계로 제안한 적이 있다. 이는 완전한 9쌍 검증 전의 가설이었다.

이후 실제 `vending_machine–umbrella`의 OBB 표면거리가 약 `0.47~0.49 m`인데 정답이 `beside`라는 사실을 확인했다. 따라서 `0.25 m`를 쓰면 이 쌍을 `near`로 오분류한다. **초기 `0.25 m` 제안은 폐기하고 최종 규칙에 사용하지 않는다.**

## 2.5 다섯 번째 질문: 두 경계 쌍만 우선 확인

사용자 질문:

> “vending_machine–umbrella, trash_can–umbrella에 대해서만 너가 해봐”

이 두 쌍을 먼저 계산한 이유는 텍스트만 보면 오히려 모순적으로 보였기 때문이다.

- vending machine 쪽은 `beside`, `standing next to it`, `propped against its side`로 쓰였고 정답은 `beside`다.
- trash can 쪽은 `next to the trash can`, `leaning against the trash can`으로 쓰였지만 정답은 `near`다.

즉 `next to`나 `leaning against`를 단순 사전 규칙으로 변환하면 틀린다. 실제 기하가 두 정답을 분리하는지 확인해야 했다.

공식 Isaac 방식의 결과는 다음과 같았다.

| 문제 | 정답 | 원점 XY 거리 | OBB XY 거리 | OBB 3D 거리 |
|---|---:|---:|---:|---:|
| LF05 vending_machine–umbrella | beside | 1.847382 m | 0.469896 m | 0.469896 m |
| LF06 trash_can–umbrella | near | 1.148257 m | 0.696786 m | 0.696786 m |

여기서 중요한 점은 다음 두 가지다.

1. **원점 거리만 보면 관계가 뒤집힌다.** `beside`인 자판기 쌍의 원점 거리가 `near`인 쓰레기통 쌍보다 더 멀다.
2. **OBB 표면거리로 바꾸면 기대한 순서가 된다.** 큰 자판기의 크기를 반영하므로 우산과 자판기 표면 간격이 쓰레기통–우산보다 더 짧아진다.

따라서 이 두 쌍은 prim 원점 거리보다 OBB 표면거리가 `near`/`beside` 구분에 적합하다는 직접적인 근거가 됐다.

## 2.6 여섯 번째 질문: 남은 hydrant–tumbler를 확인한 뒤 임계값 결정

사용자 질문:

> “이 두개가 판단의 예외가 될 수도 있었던 거고 9개 중 아까 6개는 했잖아 그럼 남은건 소화전-텀블러에 대해서만 확인해보면 되는거 아니야? 얘까지 확인하고 기준거리 어떻게 할지 정해보자”

`LF03 hydrant–tumbler`를 공식 에셋으로 계산한 결과는 다음과 같았다.

| 문제 | 정답 | 원점 XY 거리 | 공식 OBB XY 거리 | 공식 OBB 3D 거리 |
|---|---:|---:|---:|---:|
| LF03 hydrant–tumbler | near | 2.915349 m | 2.627796 m | 2.627993 m |

이 값은 경계 근처가 아니라 확실히 큰 값이므로 `near` 사례로 자연스럽게 설명된다. 즉 남은 핵심은 `vending_machine–umbrella`와 `trash_can–umbrella` 사이 경계를 어디에 둘지, 그리고 거리 예외가 있는지 확인하는 것이었다.

---

## 3. 분석 대상 9개 고유 기하 쌍

Chungmu YAML에는 같은 배치를 직접형(LF), 서사형(HF), retrieval(RT) 명령으로 반복한 문제가 있다. 반복된 명령을 별도 데이터로 세면 표본 수를 과대평가하게 된다. 실제 기하 배치 기준으로는 다음 9개가 고유 표본이다.

| # | 대표 문제 | 랜드마크–타깃 | 정답 | 같은 기하를 재사용하는 문제 |
|---:|---|---|---:|---|
| 1 | LF03 | hydrant–tumbler | near | LF03 |
| 2 | LF04 | picnic_table_01–sunblock | beside | LF04, HF04 |
| 3 | LF05 | vending_machine_01–umbrella | beside | LF05, HF01, HF05, RT03 |
| 4 | LF06 | trash_can_01–umbrella | near | LF06, HF06 |
| 5 | LF07 | picnic_table_02–pencilcase | near | LF07, HF02, RTC2 |
| 6 | LF08 | police_car_01–umbrella | near | LF08, HF08, RT02 |
| 7 | LF09 | police_car_01–tumbler | near | LF09, HF07, RT01 |
| 8 | LF10 | taxi_01–tumbler | near | LF10, RT04 |
| 9 | LF11 | taxi_01–umbrella | near | LF11, RT04B |

따라서 통계적으로는 `beside` 2쌍, `near` 7쌍이다. HF/RT 변형은 언어 규칙을 확인하는 데는 유용하지만, 거리 임계값을 학습하는 독립 표본은 아니다.

---

## 4. 공식 R03 에셋과 크기 확인

공식 content layer에서 9개 쌍에 필요한 모든 에셋을 확보한 뒤, 변환이 적용되지 않은 로컬 bound 크기를 확인했다.

| 에셋 | X 크기 (m) | Y 크기 (m) | Z 크기 (m) |
|---|---:|---:|---:|
| hydrant | 0.320754 | 0.230929 | 0.672998 |
| vending_machine | 1.922631 | 2.041883 | 2.497070 |
| trash_can | 0.709367 | 0.709367 | 1.072035 |
| picnic_table | 2.830520 | 2.832985 | 1.029803 |
| police_car | 1.777625 | 3.730490 | 1.238751 |
| taxi | 1.807360 | 4.220717 | 1.310464 |
| tumbler | 0.065743 | 0.159117 | 0.065743 |
| sunblock | 0.046905 | 0.128929 | 0.030700 |
| umbrella | 0.070762 | 0.211211 | 0.050494 |
| pencilcase | 0.043894 | 0.171942 | 0.049884 |

이 표가 보여주는 가장 중요한 사실은 랜드마크 크기의 차이다. 자판기는 쓰레기통보다 훨씬 크기 때문에 동일한 중심거리 또는 오히려 더 큰 중심거리에서도 표면은 더 가까울 수 있다. 이것이 원점 거리 대신 OBB 표면거리를 사용해야 하는 이유다.

---

## 5. 재현에 사용한 데이터와 무결성 확인

### 5.1 사용한 공식 R03 데이터

- 시나리오: `metacom2026/scenarios/marc2026_chungmu.yaml`
- 설정: `metacom2026/scenarios/config.yaml`
- content tag: `ghcr.io/marc-challenge/marc-platform-content:2026.R03`
- registry manifest에 기록된 content layer 크기: `657,109,359 bytes`
- registry manifest에 기록된 digest:

```text
d5f1be542749458c6a6faac5130cdce7108f3e120fdba0cd9fec40049374f49b
```

현재 scratch에 남은 `content-layer.tar.gz`는 후속 재다운로드가 중간에 끊긴 `312,442,880 bytes`짜리 파일이며, SHA-256은 `c80e6859...`이고 `gzip -t`도 실패했다. 따라서 이 현재 임시파일을 위 manifest digest의 재검증 근거로 사용하지 않았다. 분석에 사용한 `extracted/` 디렉터리는 그보다 앞선 시각에 생성됐으며, 실제 계산 입력의 재현성을 위해 추출된 YAML과 각 USD 파일을 별도로 해시 확인했다.

- 추출된 `marc2026_chungmu.yaml` SHA-256: `8b824346cbf05b1c8d5626bbee6ae7f82d7fd55c1c55d66f80ff02516a2574a4`
- 추출된 `config.yaml` SHA-256: `387ecdf5dab9f4b3e7688a77e6a9d61c24a2f85c02c53a24efc9ae3cd1f8abfe`

즉 이 문서의 거리 결과는 현재 손상된 압축파일을 다시 읽어 만든 것이 아니라, 이미 추출되어 열리는 시나리오와 USD들을 입력으로 한 결과다. 다만 전체 레이어의 현재 로컬 사본을 다시 완전하게 해시 검증했다는 주장까지는 하지 않는다.

### 5.2 사용한 Isaac Sim 5.1 구현

- bounds 구현: <https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/source/extensions/isaacsim.core.utils/python/impl/bounds.py>
- rotation 구현: <https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/source/extensions/isaacsim.core.utils/python/impl/numpy/rotations.py>
- Isaac Sim bounds 문서: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.utils/docs/index.html>

### 5.3 공식 MARC 관련 참고 링크

- Challenge: <https://marc-challenge.github.io/en/challenge/>
- Development Guide: <https://marc-challenge.github.io/marc-dev-guide/en/>
- Starter Kit: <https://github.com/marc-challenge/marc-starter-kit>
- Notices: <https://github.com/marc-challenge/marc-starter-kit/blob/main/NOTICES.md>

---

## 6. OBB 거리 계산 방법

## 6.1 Stage transform 재구성

각 랜드마크와 타깃에 대해 다음 정보를 사용했다.

1. 공식 USD 에셋의 로컬 기하 bound
2. Chungmu YAML에 저장된 translation
3. YAML에 저장된 rotation/Euler 값
4. YAML에 저장된 scale
5. Isaac Sim이 적용하는 xform op 순서

그 뒤 Isaac Sim 5.1 `compute_obb`가 사용하는 방식에 맞춰 world bound의 centroid, 축, half-extent를 구성했다.

## 6.2 OBB corner 생성

OBB를 중심 `c`, 단위축 `u_1,u_2,u_3`, 각 축의 반길이 `h_1,h_2,h_3`로 나타내면 8개 corner는 다음과 같다.

\[
\mathbf{q}_{s_1s_2s_3}=\mathbf{c}
+s_1h_1\mathbf{u}_1
+s_2h_2\mathbf{u}_2
+s_3h_3\mathbf{u}_3,
\quad s_i\in\{-1,+1\}
\]

## 6.3 XY footprint 거리

1. 8개 corner에서 `(x,y)`만 취한다.
2. 중복점을 제거하고 convex hull을 만든다.
3. 두 hull이 겹치거나 닿으면 `d_xy=0`이다.
4. 떨어져 있으면 모든 변–변 조합의 최소 거리를 구한다.

이 값이 지면상 OBB 표면거리다.

## 6.4 3D OBB 거리

3D 진단값은 두 OBB 내부의 점을 각각

\[
\mathbf{x}_A=\mathbf{c}_A+U_A\mathbf{s},\quad
\mathbf{x}_B=\mathbf{c}_B+U_B\mathbf{t}
\]

로 두고, 각 좌표에 `-h <= s,t <= h` 제한을 둔 상태에서

\[
\min_{\mathbf{s},\mathbf{t}}\|\mathbf{x}_A-\mathbf{x}_B\|_2
\]

를 푸는 방식으로 교차검증했다.

경계를 결정한 `vending_machine–umbrella`, `trash_can–umbrella`에서는 수직 범위가 겹치므로 3D 거리와 XY 거리가 사실상 동일했다.

## 6.5 OBB 거리와 실제 메시 거리의 차이

OBB는 직육면체이므로 원통형 쓰레기통, 우산의 가는 손잡이, 자동차의 곡면 주변에 실제 메시가 없는 공간도 상자 안에 포함한다. 따라서:

- OBB 거리 `0 m`는 두 실제 메시가 물리적으로 접촉했다는 뜻이 아니다.
- 물리 접촉 여부를 판단하려면 collision shape 또는 mesh-to-mesh distance가 필요하다.
- 이 분석의 목적은 자연어 관계 분류를 위한 안정적인 크기 보정 거리이므로 OBB가 적합하지만, 접촉 판정용으로 해석하면 안 된다.

---

## 7. 팩트 체크 및 교차검증 절차

수치는 다음 순서로 두 번 이상 확인했다.

### 7.1 첫 번째 검증: 공식 파일과 에셋

1. 공식 GHCR R03 content image에서 시나리오와 에셋을 추출했다.
2. registry manifest에 기록된 layer digest와 크기를 확인했다.
3. 현재 scratch의 후속 불완전 다운로드 파일은 manifest 검증 자료에서 제외했다.
4. 실제 계산에 사용한 추출 YAML/config와 개별 USD의 SHA-256을 별도로 기록했다.
5. 5개 누락 에셋의 USD 경로와 종속 디렉터리 존재를 확인했다.
6. USD core로 각 stage를 열어 로컬 bound가 실제로 계산되는지 확인했다.

### 7.2 두 번째 검증: 거리 알고리즘 이중 계산

1. XY에서는 convex hull의 polygon edge-to-edge 최소거리로 계산했다.
2. 3D에서는 두 OBB 내부 점의 bounded least-squares 최소거리로 계산했다.
3. 높이 구간이 겹치는 결정적 두 쌍에서 두 계산값이 부동소수점 오차 범위 내에서 일치하는지 확인했다.
4. `BBoxCache`의 `useExtentsHint=True/False`에서도 결정적 결과가 바뀌지 않는지 확인했다.

### 7.3 세 번째 검증: 회전 해석 민감도

회전 변환을 검토하는 과정에서 주의할 부분이 발견됐다.

- Isaac Sim 5.1의 `euler_angles_to_quats`는 scalar-first `[w,x,y,z]`를 반환한다.
- 현재 확인한 MARC R03 starter-kit patch/comment는 반환값을 `[x,y,z,w]`처럼 취급하는 것으로 읽힌다.

따라서 두 경우를 모두 계산했다.

1. **literal R03 patch**: 현재 코드가 문자 그대로 authoring하는 회전
2. **intended intrinsic XYZ**: 작성자가 의도했을 것으로 보이는 intrinsic XYZ 회전

두 해석 사이에 수 cm 차이는 있었지만 최종 경계와 예외 결론은 바뀌지 않았다.

| 경계 쌍 | literal R03 | intended XYZ |
|---|---:|---:|
| vending_machine–umbrella, beside | 0.469896 m | 0.489054 m |
| trash_can–umbrella, near | 0.696786 m | 0.656376 m |
| 두 값의 중간 | 0.583341 m | 0.572715 m |

이 교차검증 때문에 최종값을 지나치게 정밀한 `0.583340649 m`로 고정하지 않고 `0.58 m`로 정했다.

---

## 8. 공식 Isaac `compute_obb` 방식으로 계산한 9쌍

다음 표는 동일한 `literal R03` 회전 해석과 공식 Isaac `ComputeWorldBound` 기반 OBB 정의를 9개 모두에 일관되게 적용한 결과다.

| 문제 | 랜드마크–타깃 | 정답 | 원점 XY (m) | OBB XY (m) | OBB 3D (m) |
|---|---|---:|---:|---:|---:|
| LF03 | hydrant–tumbler | near | 2.915349 | 2.627796 | 2.627993 |
| LF04 | picnic_table–sunblock | beside | 1.429680 | 0.000000 | 0.000000 |
| LF05 | vending_machine–umbrella | beside | 1.847382 | 0.469896 | 0.469896 |
| LF06 | trash_can–umbrella | near | 1.148257 | 0.696786 | 0.696786 |
| LF07 | picnic_table–pencilcase | near | 3.471401 | 1.871464 | 1.871464 |
| LF08 | police_car–umbrella | near | 1.998104 | 0.000000 | 0.000000 |
| LF09 | police_car–tumbler | near | 3.306369 | 0.759093 | 0.759390 |
| LF10 | taxi–tumbler | near | 5.477740 | 2.628534 | 2.628534 |
| LF11 | taxi–umbrella | near | 5.590001 | 2.894389 | 2.894389 |

### 8.1 표에서 확인되는 사실

- 원점 거리는 랜드마크 크기를 반영하지 못한다. LF05/LF06에서 정답 순서와 반대로 나타난다.
- 일반적인 표본에서는 `beside`가 짧고 `near`가 길다.
- `LF08 police_car–umbrella`만 거리 규칙과 충돌한다.
- 공식 OBB에서는 LF08이 `0 m`이므로 어떤 양의 거리 임계값을 사용해도 `beside` 쪽에 놓인다.
- 따라서 공식 OBB 거리 하나만으로 9개를 완전 분리하는 임계값은 존재하지 않는다.

### 8.2 intended XYZ 회전으로 계산한 민감도 표

| 문제 | 정답 | OBB XY (m) |
|---|---:|---:|
| LF03 | near | 2.671073 |
| LF04 | beside | 0.000000 |
| LF05 | beside | 0.489054 |
| LF06 | near | 0.656376 |
| LF07 | near | 1.877756 |
| LF08 | near | 약 0.000000 |
| LF09 | near | 0.739065 |
| LF10 | near | 2.609084 |
| LF11 | near | 2.894314 |

회전 해석을 바꿔도 LF08 예외와 `0.58 m` 경계의 유효성은 그대로다.

---

## 9. 왜 다른 OBB 계산에서는 자동차 관련 거리가 달라졌는가

분석 중에는 공식 `ComputeWorldBound` 기반 OBB 외에, 에셋의 로컬 bound를 루트 방향에 맞춰 더 타이트하게 회전시킨 **root-aligned OBB**도 계산했다.

### 9.1 root-aligned 결과

| 문제 | 정답 | literal root-aligned XY (m) | intended root-aligned XY (m) |
|---|---:|---:|---:|
| LF03 | near | 2.627796 | 2.671073 |
| LF04 | beside | 0.000000 | 0.000000 |
| LF05 | beside | 0.471878 | 0.489588 |
| LF06 | near | 0.700885 | 0.664412 |
| LF07 | near | 1.872491 | 1.879580 |
| LF08 | near | 0.459983 | 0.439886 |
| LF09 | near | 2.179405 | 2.167609 |
| LF10 | near | 3.900555 | 3.885700 |
| LF11 | near | 3.905927 | 3.853471 |

### 9.2 차이가 생기는 이유

Isaac Sim `compute_obb`는 내부적으로 `BBoxCache.ComputeWorldBound`를 호출한다. 자동차처럼 하위 prim과 복합 geometry가 많은 에셋은 child geometry의 world bound가 합쳐지는 과정에서 루트 방향 기준의 타이트한 상자보다 느슨하거나 사실상 world-aligned에 가까운 상자가 될 수 있다.

그 결과:

- 공식 방식의 `police_car–umbrella`는 OBB가 겹쳐 `0 m`가 된다.
- root-aligned 방식은 약 `0.44~0.46 m`의 간격을 남긴다.
- taxi와 police_car 관련 몇몇 값도 크게 달라진다.

그러나 어느 정의에서도 단일 임계값으로 9개가 완전 분리되지는 않는다.

- 공식 방식: `near`의 최소값 `0 m` < `beside`의 최대값 약 `0.47 m`
- root-aligned 방식: `near`의 최소값 약 `0.44~0.46 m` < `beside`의 최대값 약 `0.47~0.49 m`

즉 OBB 정의 선택으로 LF08의 의미적 예외가 사라지는 것은 아니다.

실제 코드가 Isaac Sim의 `compute_obb`를 호출한다면 임계값 분석도 공식 방식 결과를 기준으로 유지하는 것이 일관적이다.

---

## 10. 텍스트 규칙 검증

## 10.1 동의어 단순 매핑이 실패하는 사례

공개 YAML의 정답을 보면 다음 표현들이 서로 충돌한다.

| 배치 | 명령 표현 | YAML 정답 |
|---|---|---:|
| vending_machine–umbrella | `beside the vending machine` | beside |
| vending_machine–umbrella | `standing next to it` | beside |
| vending_machine–umbrella | `propped against its side` | beside |
| trash_can–umbrella | `next to the trash can` | near |
| trash_can–umbrella | `leaning against the trash can` | near |

따라서 다음과 같은 규칙은 사용하면 안 된다.

```text
next to          -> beside
leaning against  -> beside
against          -> beside
```

같은 표현 계열이 자판기에서는 `beside`, 쓰레기통에서는 `near`로 정답 처리되기 때문이다. 이 경우에는 기하 거리 또는 장면 ID/영역 문맥이 필요하다.

## 10.2 확인된 의미 예외: 빈 주차 공간

LF08의 명령은 다음과 같다.

```text
Find the umbrella dropped in the empty parking space.
```

정답 landmark는 `police_car_01`, relation은 `near`다. 여기서 문장은 우산이 경찰차의 표면 가까이에 있다고 말하지 않는다. 경찰차가 속한 주차 맥락을 이용해 **빈 주차 칸이라는 영역**을 지시한다.

같은 기하를 쓰는 HF08과 RT02도 정답이 `near`다. 따라서 다음 해석이 타당하다.

- geometric beside: 랜드마크 표면 옆에 놓인 물체
- regional near: 랜드마크를 anchor로 하여 주변 영역·주차 칸·도로 구간에 있는 물체

다만 `road`, `parking lot`, `space` 같은 단어가 나오면 언제나 무조건 `near`라고 일반화하면 안 된다. 현재 실제로 거리 임계값과 충돌하여 별도 처리해야 한다고 확인된 것은 **police_car를 anchor로 하는 empty parking space–umbrella 배치**다.

LF09의 police_car–tumbler도 빈 주차 공간 문맥이며 정답 `near`지만, 공식 OBB 거리가 `0.759093 m`이므로 임계값만으로도 이미 `near`가 된다. LF10/LF11의 taxi–object 역시 road 문맥이지만 거리가 충분히 커서 별도 예외가 필요하지 않다.

---

## 11. 임계값 도출 과정

## 11.1 예외를 포함하면 임계값이 존재하지 않음

공식 OBB 결과에서:

\[
\min(d_{near})=d_{LF08}=0
\]

\[
\max(d_{beside})=d_{LF05}=0.469895694
\]

`near`의 최소값이 `beside`의 최대값보다 작으므로, `d <= tau`를 `beside`로 두는 어떤 `tau`도 전체를 완전 분리할 수 없다.

## 11.2 확인된 영역 예외 LF08을 제외한 경계

literal R03 공식 계산:

\[
\max(d_{beside})=0.469895694
\]

\[
\min(d_{ordinary\ near})=0.696785604
\]

따라서 두 클래스를 나누는 가능한 구간은 대략 다음과 같다.

\[
0.469895694 < \tau < 0.696785604
\]

intended XYZ 계산에서는:

\[
0.489053683 < \tau < 0.656376439
\]

두 해석을 동시에 안전하게 만족시키는 공통 구간은 대략:

\[
0.4891 < \tau < 0.6564\ \text{m}
\]

이다.

## 11.3 최종 `0.58 m` 선택

literal R03 경계 중간값:

\[
\tau_{literal}=0.583340649\ \text{m}
\]

intended XYZ 경계 중간값:

\[
\tau_{intended}=0.572715061\ \text{m}
\]

root-aligned 계산에서도 중간값이 대략 `0.577~0.586 m` 범위에 놓였다. 따라서 여러 계산 정의에 공통으로 안정적인 대표값으로:

```text
tau_xy = 0.58 m
```

를 정했다.

0.58은 지나친 자릿수 정밀도를 피하면서도 다음을 만족한다.

- LF05 vending_machine–umbrella 약 0.47~0.49 m → beside
- LF06 trash_can–umbrella 약 0.66~0.70 m → near
- 나머지 일반 near 표본 → near
- LF04 table–sunblock 0 m → beside

---

## 12. 최종 운영 규칙

### 12.1 우선순위

```text
1. canonical relation 명시 여부 확인
2. 확인된 영역/공간 의미 예외 확인
3. XY OBB 표면거리 계산
4. 0.58 m 임계값 적용
5. 낮은 확신 또는 경계 구간이면 로그/후보 유지
```

### 12.2 권장 의사코드

```python
TAU_BESIDE_M = 0.58

def infer_near_or_beside(command, target_obb, landmark_obb, context):
    explicit = extract_canonical_relation(command)
    if explicit in {"near", "beside"}:
        return explicit

    # 확인된 Chungmu 의미 예외.
    # 단순히 parking이라는 단어 하나만 보지 말고,
    # empty parking space를 지시하며 police_car가 anchor인 상황인지 확인한다.
    if refers_to_empty_parking_space(command, context):
        return "near"

    d_xy = obb_footprint_surface_distance(target_obb, landmark_obb)

    if d_xy <= TAU_BESIDE_M:
        return "beside"
    return "near"
```

### 12.3 경계값 주변의 보수적 처리

실행 중 PhysX 변위나 OBB 구성 차이를 감안하면 `0.58 m` 주변에서 딱 잘라 확신하는 것보다 경계 밴드를 두는 편이 낫다.

예시:

```python
LOW = 0.53
HIGH = 0.63

if d_xy < LOW:
    relation = "beside"
elif d_xy > HIGH:
    relation = "near"
else:
    relation = resolve_with_text_and_scene_context(...)
```

다만 공개 Chungmu의 LF05 약 `0.47~0.49 m`, LF06 약 `0.66~0.70 m`는 이 밴드 바깥에 있어 서로 분리된다. 실제 채점 구현이 반드시 단일 label을 즉시 내야 한다면 `0.58 m`를 사용하면 된다.

---

## 13. 실행 중 Isaac Sim에서 CSV로 최종 확인하는 방법

## 13.1 왜 live-stage CSV가 필요한가

현재 수치는 공식 R03 에셋과 YAML transform을 재구성한 것이다. 다음 요인은 실제 실행 stage에서 차이를 만들 수 있다.

- PhysX settle 후 작은 translation/rotation 변화
- parent prim transform 또는 scene root scale
- 실제 spawn 코드의 xform op 순서
- asset reference가 합쳐진 뒤의 `ComputeWorldBound`
- collider와 render mesh의 차이

따라서 Isaac Sim timeline이 안정화된 뒤 실제 prim path에서 OBB corner를 직접 얻는 것이 최종 확인이다.

## 13.2 권장 CSV 열

```text
problem_id
expected_relation
target_path
landmark_path
origin_xy_m
obb_xy_gap_m
obb_3d_gap_m
signed_xy_gap_m
target_inside_landmark_xy
quaternion_convention
bbox_mode
```

추가로 각 OBB의 center, axes, half-extents 또는 8개 corner를 별도 JSON 열/파일로 저장하면 재계산이 가능하다.

## 13.3 추출 시점

1. Chungmu scene 로딩 완료 대기
2. 필요한 reference와 texture 로딩 완료 확인
3. timeline 재생 및 PhysX settle 대기
4. timeline을 멈추거나 동일 frame에서 9개를 연속 추출
5. `compute_obb`와 `compute_obb_corners` 호출
6. XY polygon 거리와 3D 거리를 계산해 CSV 저장

## 13.4 CSV에서 확인할 기준

최종 live CSV에서 다음을 확인한다.

- LF05가 LF06보다 여전히 짧은가?
- LF05가 `0.58 m`보다 작은가?
- LF06가 `0.58 m`보다 큰가?
- LF08이 여전히 `0.58 m`보다 작거나 OBB overlap인가?
- 물리 안정화 전후 값이 몇 cm 변하는가?

LF05/LF06이 0.58을 기준으로 그대로 분리되고 LF08만 예외라면 현재 규칙을 유지하면 된다. 두 경계 쌍이 임계값을 넘어가면, 두 live 값의 중간으로 `tau`를 다시 잡아야 한다.

---

## 14. 초기 판단에서 최종 판단까지의 정정 이력

| 단계 | 당시 판단 | 추가로 확인된 사실 | 최종 처리 |
|---|---|---|---|
| 1 | prim 원점 거리도 후보 | LF05/LF06에서 원점 거리 순서가 정답과 반대 | 원점 거리는 진단값으로만 사용 |
| 2 | `beside`는 매우 가까울 것이므로 약 0.25 m 가능성 | LF05 beside의 OBB 간격이 약 0.47~0.49 m | 0.25 m 폐기 |
| 3 | 유사 표현을 텍스트 규칙으로 매핑 가능성 | `next to`, `leaning against`가 near와 beside 양쪽에 등장 | 동의어 단독 매핑 금지 |
| 4 | OBB 거리 하나로 전체 분리 가능성 | LF08 near가 공식 OBB 0 m | 거리 단독 9/9 불가능 |
| 5 | 경계 두 쌍의 중간 약 0.58 m | 회전·OBB 정의를 바꿔도 중간값 약 0.57~0.59 m | 0.58 m 채택 |
| 6 | 모든 공간 표현을 예외 처리할 가능성 | taxi/road 및 police/tumbler는 거리만으로 이미 near | 확인된 empty-parking-space 예외만 제한적으로 적용 |

가장 중요한 정정은 **`0.25 m`가 최종 기준이 아니라는 점**이다. 공식 에셋 전체를 확보한 이후의 최종 권고값은 `0.58 m`다.

---

## 15. 최종 성능 해석

### 거리 규칙만 사용

```text
d_xy <= 0.58 m -> beside
d_xy > 0.58 m  -> near
```

공식 literal R03 9쌍에 적용하면:

- 정답: 8쌍
- 오답: 1쌍 (`LF08 police_car–umbrella`)
- 고유 기하 쌍 기준 정확도: `8/9 = 88.9%`

### 의미 예외까지 사용

`empty parking space` 영역 관계를 `near`로 우선 처리하면 현재 공개 9쌍을 모두 설명한다.

그러나 이는 공개 Chungmu에 맞춘 규칙이다. 숨겨진 scene이나 향후 다른 도시 scenario에서 같은 표현·배치가 항상 같은 방식으로 채점된다는 보장은 없다. 그러므로 코드에서는 문제 ID 하드코딩보다 `영역 관계인지`, `직접 표면 인접 관계인지`를 판단하는 일반화된 semantic feature로 구현하는 편이 낫다.

---

## 16. 구현 권고안

### 16.1 최소 구현

- 명령문 canonical relation 추출
- landmark 및 target prim 확보
- Isaac `compute_obb_corners`로 corner 추출
- XY convex hull 표면거리 계산
- `0.58 m` threshold
- empty-parking-space semantic override

### 16.2 더 안정적인 구현

- `0.53~0.63 m`를 불확실 구간으로 둔다.
- 불확실 구간에서는 다음 feature를 함께 본다.
  - 직접 옆면을 지시하는 표현인지
  - 영역/주차 칸/도로를 지시하는 표현인지
  - target 중심이 landmark의 side band에 있는지
  - landmark와 target의 상대 방위 및 footprint overlap
- OBB뿐 아니라 collider distance를 보조값으로 로그한다.
- 시작 시 9쌍 CSV를 자동 생성해 환경 버전이 바뀌면 threshold drift를 탐지한다.

### 16.3 피해야 할 구현

- prim translation 거리만으로 분류
- `next to`를 무조건 `beside`로 치환
- `leaning against`를 무조건 `beside`로 치환
- `parking`이 들어가면 전부 `near`로 치환
- 초기 proxy 값과 최종 공식 OBB 값을 한 표본 집합처럼 혼합
- OBB overlap `0 m`를 실제 메시 접촉이라고 해석

---

## 17. 남아 있는 불확실성

1. **실행 stage 미추출**  
   현재 분석은 공식 R03 파일 기반 재구성이다. 실행 중 Isaac Sim에서 PhysX settle 후 얻은 CSV가 최종 근거가 되어야 한다.

2. **표본 부족**  
   고유 `beside`가 두 쌍뿐이므로 0.58 m를 다른 scenario에 일반화할 통계적 근거는 부족하다.

3. **쿼터니언 해석 문제**  
   starter-kit patch의 quaternion 순서에 해석상 불일치가 보인다. 실제 stage가 어떤 orientation을 authoring했는지는 live prim transform으로 확인해야 한다.

4. **OBB 정의 차이**  
   공식 `ComputeWorldBound` OBB와 root-aligned 타이트 OBB는 복합 자동차 에셋에서 크게 다르다. 참가자 코드가 어느 함수를 쓰는지에 따라 수치가 바뀐다.

5. **OBB의 빈 공간**  
   OBB gap은 실제 mesh gap과 다르다. 자연어 관계에는 유용하지만 물리 접촉의 증거가 아니다.

6. **공식 기준 여부**  
   `0.58 m`는 주최 측 문서에서 발견된 공식 threshold가 아니라 공개 Chungmu 샘플로 역추정한 공학적 기준이다.

---

## 18. 최종 결정 체크리스트

- [x] prim 원점 거리 대신 물체 크기를 반영하는 OBB 표면거리 사용
- [x] 지면 관계이므로 XY footprint 거리를 주 판단값으로 사용
- [x] 3D OBB 거리는 진단 및 교차검증용으로 저장
- [x] 공식 R03에서 누락 5개 에셋 확보
- [x] vending_machine–umbrella와 trash_can–umbrella 직접 비교
- [x] hydrant–tumbler 추가 확인
- [x] 9개 고유 쌍을 동일 방식으로 재계산
- [x] 초기 `0.25 m` 가설 폐기
- [x] 최종 nominal threshold `0.58 m` 채택
- [x] `police_car–umbrella / empty parking space` 의미 예외 반영
- [ ] 실행 중 Isaac Sim에서 9쌍 OBB CSV 최종 추출
- [ ] live CSV로 threshold 및 예외 재확인

---

## 19. 한 문장으로 정리

**Chungmu에서는 `near`/`beside`를 prim 원점 거리나 영어 동의어 하나로 나누지 말고, 명시 관계어와 영역 의미를 먼저 해석한 뒤 XY OBB 표면거리 `0.58 m`를 기본 경계로 사용하되, `empty parking space`의 police_car–umbrella처럼 표면 인접이 아니라 주변 영역을 가리키는 사례는 `near`로 별도 처리한다.**
