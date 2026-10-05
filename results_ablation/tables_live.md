## 표 A. Camera gating (live)

| ID | Configuration | Cam acc. (%) | Cam acc. obj / person (%) | Stage1 | Object | Person | Gated rounds | GT cam excluded |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | No gate, confidence ranking | 36.4 | 47.4 / 21.4 | 44.88 | 54.40 | 31.96 | 0/33 | 0 |
| G1 | + Landmark gate | 42.4 | 57.9 / 21.4 | 49.80 | 61.96 | 33.30 | 27/33 | 0 |
| G2 | + Area gate (intersection) | 75.8 | 73.7 / 78.6 | 68.76 | 65.64 | 73.00 | 33/33 | 0 |
| G3 | + Relation-fit ranking (bucket-first) | 84.8 | 78.9 / 92.9 | 76.20 | 70.14 | 84.43 | 33/33 | 0 |
| G4 | + Person proximity exception (full) | 87.9 | 78.9 / 100.0 | 75.20 | 69.40 | 83.07 | 33/33 | 0 |
|  | ↳ 복제 resolve 대조 불일치 0회 | | | | | | | |

## 표 B. Coordinate correction (live)

| ID | Configuration | Stage1 | 'on' subset | Person | Anchor err obj (m) | Target err obj (m) | Anchor / Target err obj, cam-correct (m) | Target err person (m) | Target hit obj / person | Cam acc. (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | BBox-center back-projection | 57.70 | 37.48 | 53.89 | 2.67 | 1.80 | 1.75 / 1.00 | 2.01 | 7/19 / 6/14 | 78.8 |
| C1 | Bottom-center contact point | 64.16 | 47.28 | 66.59 | 1.93 | 1.87 | 1.74 / 1.29 | 1.49 | 8/19 / 8/14 | 78.8 |
| C2 | + Footprint-center push | 62.18 | 43.72 | 60.35 | 1.08 | 1.82 | 0.64 / 0.84 | 1.46 | 8/19 / 6/14 | 78.8 |
| C3 | + 'on' support-plane correction | 72.17 | 68.09 | 79.50 | 1.08 | 0.66 | 0.64 / 0.59 | 0.94 | 9/19 / 9/14 | 84.8 |
| C4 | + 'on' low-conf target probe (detection) | 72.25 | 73.92 | 79.50 | 1.08 | 1.83 | 0.64 / 0.65 | 0.93 | 9/19 / 9/14 | 87.9 |
| C5 | + Ankle keypoint (full) | 76.46 | 82.01 | 86.57 | 1.08 | 1.82 | 0.64 / 0.65 | 0.75 | 10/19 / 11/14 | 87.9 |
|  | ↳ 복제 resolve 대조 불일치 0회 | | | | | | | | | |

오차는 중앙값(m), 제출 좌표와 공식 GT 좌표의 3D 거리. cam-correct = 카메라 정답 라운드만. Target hit = target 점수 ≥ 0.5 인 라운드 수.
