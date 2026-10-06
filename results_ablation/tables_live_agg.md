# Stage 1 ablation: 반복 실행 집계 (live)

- **최빈값 표**: 문항마다 (카메라·유형·랜드마크·관계 정답 여부, target 득점 여부)가 가장 자주 나온 결과 묶음을 고르고, 그 묶음 안에서 총점이 중앙인 실제 기록 한 줄을 대표로 써서 표를 만든다. 동률이면 총점 중앙값이 낮은 쪽을 쓴다. 한 번도 실제로 나오지 않은 조합일 수 있다.
- **평균 표**: 실행마다 지표를 따로 계산한 뒤 `평균 ± 표준편차 [최소–최대]`로 적는다.
- G4와 C5는 설정이 같아 실행을 합쳐서 집계한다.

## 표 A (최빈값 기준)

| ID | Configuration | N | Cam acc. (%) | Stage1 | Object | Person | 동률 문항 수 |
|---|---|---:|---:|---:|---:|---:|---:|
| G0 | No gate, confidence ranking | 7 | 33.3 | 44.58 | 53.88 | 31.96 | 0 |
| G1 | + Landmark gate | 7 | 39.4 | 48.29 | 59.32 | 33.30 | 0 |
| G2 | + Area gate (intersection) | 7 | 75.8 | 68.76 | 65.64 | 72.99 | 0 |
| G3 | + Relation-fit ranking (bucket-first) | 7 | 84.8 | 74.66 | 70.14 | 80.79 | 0 |
| G4 | + Person proximity exception (full) | 7 | 87.9 | 73.77 | 66.91 | 83.07 | 0 |

## 표 A (평균 ± 표준편차 [범위])

| ID | Configuration | N | Cam acc. (%) | Stage1 | Object | Person |
|---|---|---:|---:|---:|---:|---:|
| G0 | No gate, confidence ranking | 7 | 33.8 ± 1.1 [33.3–36.4] | 44.80 ± 0.45 [44.58–45.79] | 54.25 ± 0.79 [53.88–55.98] | 31.96 ± 0.00 [31.96–31.96] |
| G1 | + Landmark gate | 7 | 40.7 ± 1.6 [39.4–42.4] | 49.11 ± 1.05 [48.28–51.01] | 60.75 ± 1.82 [59.32–64.06] | 33.30 ± 0.00 [33.30–33.30] |
| G2 | + Area gate (intersection) | 7 | 75.8 ± 0.0 [75.8–75.8] | 70.09 ± 1.83 [67.55–72.48] | 67.95 ± 3.18 [63.54–72.10] | 72.99 ± 0.00 [72.99–73.00] |
| G3 | + Relation-fit ranking (bucket-first) | 7 | 84.8 ± 0.0 [84.8–84.8] | 75.54 ± 1.72 [73.17–77.73] | 71.66 ± 2.61 [70.14–75.48] | 80.81 ± 3.53 [77.28–84.43] |
| G4 | + Person proximity exception (full) | 7 | 87.9 ± 0.0 [87.9–87.9] | 75.80 ± 0.61 [75.20–76.84] | 69.71 ± 1.60 [66.91–72.25] | 84.07 ± 1.71 [83.07–86.57] |

## 표 B (최빈값 기준)

| ID | Configuration | N | Stage1 | 'on' subset | Person | Anchor err obj, cam-correct (m) | Target err obj, cam-correct (m) | Target err person (m) | Cam acc. (%) | 동률 문항 수 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | BBox-center back-projection | 7 | 58.93 | 37.49 | 53.90 | 1.75 | 0.78 | 1.99 | 78.8 | 0 |
| C1 | Bottom-center contact point | 7 | 64.16 | 47.28 | 66.59 | 1.74 | 1.29 | 1.49 | 78.8 | 0 |
| C2 | + Footprint-center push | 7 | 62.18 | 43.72 | 60.35 | 0.64 | 0.84 | 1.46 | 78.8 | 0 |
| C3 | + 'on' support-plane correction | 7 | 72.17 | 68.09 | 79.50 | 0.64 | 0.59 | 0.93 | 84.8 | 0 |
| C4 | + 'on' low-conf target probe (detection) | 7 | 73.46 | 77.56 | 79.50 | 0.64 | 0.65 | 0.93 | 87.9 | 0 |
| C5 | + Ankle keypoint (full) | 7 | 73.77 | 73.92 | 83.07 | 0.64 | 0.64 | 0.75 | 87.9 | 0 |

## 표 B (평균 ± 표준편차 [범위])

| ID | Configuration | N | Stage1 | 'on' subset | Person | Anchor err obj, cam-correct (m) | Target err obj, cam-correct (m) | Target err person (m) | Cam acc. (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | BBox-center back-projection | 7 | 59.87 ± 1.74 [57.70–62.82] | 37.49 ± 0.00 [37.48–37.49] | 53.90 ± 0.01 [53.89–53.91] | 1.35 ± 0.50 [0.77–1.75] | 0.71 ± 0.17 [0.50–1.00] | 2.00 ± 0.01 [1.99–2.01] | 78.8 ± 0.0 [78.8–78.8] |
| C1 | Bottom-center contact point | 7 | 64.74 ± 0.84 [64.16–65.93] | 47.28 ± 0.00 [47.27–47.28] | 66.59 ± 0.00 [66.59–66.59] | 1.74 ± 0.00 [1.74–1.74] | 1.12 ± 0.28 [0.71–1.29] | 1.49 ± 0.00 [1.49–1.49] | 78.8 ± 0.0 [78.8–78.8] |
| C2 | + Footprint-center push | 7 | 63.25 ± 1.00 [62.18–64.05] | 43.72 ± 0.00 [43.72–43.72] | 60.35 ± 0.00 [60.35–60.35] | 0.64 ± 0.00 [0.64–0.64] | 0.74 ± 0.10 [0.65–0.85] | 1.46 ± 0.00 [1.46–1.46] | 78.8 ± 0.0 [78.8–78.8] |
| C3 | + 'on' support-plane correction | 7 | 72.26 ± 1.40 [70.31–74.03] | 68.10 ± 0.00 [68.09–68.10] | 79.50 ± 0.00 [79.50–79.50] | 0.64 ± 0.00 [0.64–0.64] | 0.58 ± 0.03 [0.54–0.62] | 0.93 ± 0.00 [0.93–0.94] | 84.8 ± 0.0 [84.8–84.8] |
| C4 | + 'on' low-conf target probe (detection) | 7 | 73.83 ± 1.42 [72.25–75.97] | 75.48 ± 1.95 [73.92–77.56] | 79.50 ± 0.00 [79.49–79.50] | 0.64 ± 0.00 [0.64–0.64] | 0.59 ± 0.07 [0.49–0.65] | 0.93 ± 0.00 [0.93–0.94] | 87.9 ± 0.0 [87.9–87.9] |
| C5 | + Ankle keypoint (full) | 7 | 75.80 ± 0.61 [75.20–76.84] | 76.05 ± 3.37 [72.64–82.01] | 84.07 ± 1.71 [83.07–86.57] | 0.64 ± 0.00 [0.64–0.64] | 0.59 ± 0.05 [0.49–0.65] | 0.75 ± 0.00 [0.75–0.76] | 87.9 ± 0.0 [87.9–87.9] |

## 구성별 실행 목록과 결과가 뒤집힌 문항

### G0: 7회 (G0, G0_r2, G0_r3, G0_r4, G0_r5, G0_r6, G0_r7)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf01_mug_table | OOOOx×6, OOOOO×1 |
| hf07_parking_tumbler | xOxOx×6, OOxOx×1 |

### G1: 7회 (G1, G1_r2, G1_r3, G1_r4, G1_r5, G1_r6, G1_r7)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf01_mug_table | OOOOx×4, OOOOO×3 |
| lf09_tumbler_parking | xOxOx×5, OOxOO×1, OOxOx×1 |
| hf03_table_mug | OOOOO×6, OOOOx×1 |
| hf07_parking_tumbler | xOxOx×6, OOxOO×1 |

### G2: 7회 (G2, G2_r2, G2_r3, G2_r4, G2_r5, G2_r6, G2_r7)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf01_mug_table | OOOOx×4, OOOOO×3 |
| lf08_umbrella_parking | OOxOx×4, OOOOO×3 |
| lf09_tumbler_parking | OOxOO×6, OOxOx×1 |
| hf03_table_mug | OOOOO×5, OOOOx×2 |
| hf08_parking_umbrella | OOxOx×5, OOOOO×2 |

### G3: 7회 (G3, G3_r2, G3_r3, G3_r4, G3_r5, G3_r6, G3_r7)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf01_mug_table | OOOOx×5, OOOOO×2 |
| lf08_umbrella_parking | OOOOO×6, OOxOx×1 |
| hf08_parking_umbrella | OOxOx×4, OOOOO×3 |
| sar03_bench_lie | OOOxO×4, OOOxx×3 |
| sarH5_taxi_benchlie | OOOxx×4, OOOxO×3 |

### G4 = C5: 7회 (G4, G4_r2, G4_r3, G4_r4, G4_r5, G4_r6, C5)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf01_mug_table | OOOOx×5, OOOOO×2 |
| lf08_umbrella_parking | OOxOx×5, OOOOO×2 |
| hf08_parking_umbrella | OOxOx×4, OOOOO×3 |
| sar03_bench_lie | OOOxx×6, OOOxO×1 |
| sarH5_taxi_benchlie | OOOxx×6, OOOxO×1 |

### C0: 7회 (C0, C0_r2, C0_r3, C0_r4, C0_r5, C0_r6, C0_r7)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf08_umbrella_parking | OOxOx×4, OOOOO×3 |
| hf07_parking_tumbler | OOxOO×6, OOxOx×1 |
| hf08_parking_umbrella | OOxOx×6, OOOOO×1 |

### C1: 7회 (C1, C1_r2, C1_r3, C1_r4, C1_r5, C1_r6, C1_r7)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf08_umbrella_parking | OOxOx×5, OOOOO×2 |
| hf07_parking_tumbler | OOxOO×6, OOxOx×1 |
| hf08_parking_umbrella | OOxOx×6, OOOOO×1 |

### C2: 7회 (C2, C2_r2, C2_r3, C2_r4, C2_r5, C2_r6, C2_r7)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf08_umbrella_parking | OOxOx×4, OOOOO×3 |
| hf08_parking_umbrella | OOxOx×6, OOOOO×1 |

### C3: 7회 (C3, C3_r2, C3_r3, C3_r4, C3_r5, C3_r6, C3_r7)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf08_umbrella_parking | OOOOO×6, OOxOx×1 |
| lf09_tumbler_parking | OOxOO×6, OOxOx×1 |
| hf08_parking_umbrella | OOxOx×5, OOOOO×2 |

### C4: 7회 (C4, C4_r2, C4_r3, C4_r4, C4_r5, C4_r6, C4_r7)

| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |
|---|---|
| lf01_mug_table | OOOOO×4, OOOOx×3 |
| lf08_umbrella_parking | OOxOx×4, OOOOO×3 |
| hf03_table_mug | OOOOO×6, OOOOx×1 |
| hf08_parking_umbrella | OOxOx×6, OOOOO×1 |

