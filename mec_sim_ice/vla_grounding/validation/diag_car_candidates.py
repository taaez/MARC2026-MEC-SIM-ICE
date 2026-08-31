"""D 진단: lf09/rt01 이 Mac(MPS)에서는 police_car 를 맞게 고르는데 CUDA
데스크탑에서는 못 고르는 이유를 찾는다.

Mac 실측(2026-08-29): _open_space_pick() 의 clearance 마진이 3.5m vs 0.9m 로
압도적이라 하드웨어 수치 차이로 뒤집힐 만큼 근소하지 않다. 그런데도 CUDA
report(runs/report_cuda.json)에서는 landmark=0 으로 나왔다 -- 즉 clearance
로직 자체의 문제가 아니라, 그 앞단(YOLO raw 검출)이 CUDA 에서 다르게
나왔을 가능성이 높다. 이 스크립트는 그 raw 검출을 그대로 찍어서 Mac 과
바로 대조할 수 있게 한다.

데스크탑(CUDA)에서 실행:
    docker run --rm --gpus all -v "$(pwd)":/workspace -w /workspace \
      marc-mec-sim-ice-stage1:qualifier \
      python3 mec_sim_ice/vla_grounding/validation/diag_car_candidates.py

출력을 그대로 복사해서 Mac 쪽 실측과 대조한다:
    tumbler conf=0.767  clearance=  0.935m  (최근접=normal_car_1, gap=0.935)
    tumbler conf=0.331  clearance=  3.500m  (최근접=police_car, gap=3.500)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np
from ultralytics import YOLO

from grounding import CAR_CLASSES, Detection, GroundingConfig, _landmark_ok, locate
from run_offline import load_round

_ROOT = os.path.join(_HERE, "..", "..", "..")
WEIGHTS = os.environ.get("VLA_DET_WEIGHTS",
                         os.path.join(_ROOT, "mec_sim_ice", "demo_agents",
                                      "weights", "best.pt"))
RECORDS = os.path.join(_ROOT, "mec_sim_ice", "records")

CASES = [
    ("s1_009_1787669024", "tumbler", "lf09_tumbler_parking"),
    ("s2_040_1787669145", "tumbler", "rt01_parking_tumbler"),
    ("s1_008_1787669020", "umbrella", "lf08_umbrella_parking"),
]


def main() -> int:
    model = YOLO(WEIGHTS)
    cfg = GroundingConfig()
    for rec, tcls, pid in CASES:
        d = os.path.join(RECORDS, rec)
        if not os.path.isdir(d):
            print(f"[{pid}] record {rec} 없음 -- 건너뜀")
            continue
        meta, cams, images = load_round(d)
        if "rig_7_a" not in images:
            print(f"[{pid}] rig_7_a 이미지 없음")
            continue
        cam = cams["rig_7_a"]
        res = model.predict(images["rig_7_a"], conf=0.25, imgsz=1280, verbose=False)[0]
        cars, tgts = [], []
        for b in res.boxes:
            lab = res.names[int(b.cls[0])]
            det = Detection(lab, tuple(float(x) for x in b.xyxy[0]), float(b.conf[0]))
            loc = locate(cam, det, push=cfg.footprint_push)
            if loc is None:
                continue
            if lab in CAR_CLASSES or _landmark_ok(lab, "car"):
                cars.append((lab, det.score, loc))
            if lab == tcls:
                tgts.append((lab, det.score, loc))
        print(f"\n=== {pid}  record={rec} ===")
        print(f"  차량 후보 {len(cars)}개:")
        for lab, sc, loc in cars:
            print(f"    {lab:16} conf={sc:.3f}  world={tuple(round(v,2) for v in loc.ground_point)}")
        print(f"  {tcls} 후보 {len(tgts)}개:")
        for lab, sc, t0 in tgts:
            nearest = None
            for clab, csc, a_loc in cars:
                gap = (float(np.linalg.norm(t0.base[:2] - a_loc.base[:2]))
                       - a_loc.radius - t0.radius)
                if nearest is None or gap < nearest[0]:
                    nearest = (gap, clab)
            clearance = max(nearest[0], 0.0) if nearest else None
            print(f"    conf={sc:.3f}  clearance={clearance:.3f}m  "
                  f"(최근접={nearest[1] if nearest else '-'}, gap={nearest[0] if nearest else float('nan'):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
