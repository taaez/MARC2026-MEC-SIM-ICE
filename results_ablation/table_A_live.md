| ID | Configuration | Cam acc. (%) | Cam acc. obj / person (%) | Stage1 | Object | Person | Gated rounds | GT cam excluded |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | No gate, confidence ranking | 36.4 | 47.4 / 21.4 | 44.88 | 54.40 | 31.96 | 0/33 | 0 |
| G1 | + Landmark gate | 42.4 | 57.9 / 21.4 | 49.80 | 61.96 | 33.30 | 27/33 | 0 |
| G2 | + Area gate (intersection) | 75.8 | 73.7 / 78.6 | 68.76 | 65.64 | 73.00 | 33/33 | 0 |
| G3 | + Relation-fit ranking (bucket-first) | 84.8 | 78.9 / 92.9 | 76.20 | 70.14 | 84.43 | 33/33 | 0 |
| G4 | + Person proximity exception (full) | 87.9 | 78.9 / 100.0 | 75.20 | 69.40 | 83.07 | 33/33 | 0 |
|  | ↳ 복제 resolve 대조 불일치 0회 | | | | | | | |
