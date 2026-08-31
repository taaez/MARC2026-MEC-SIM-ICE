"""진단: tumbler/pencilcase/umbrella 21문항에서 YOLO whole-frame raw confidence
분포를 conf=0(사실상 문턱 없음)로 찍어본다.

목적: "클래스별 min_conf 를 낮추면 구제되는가" 대 "문턱을 아무리 낮춰도 애초에
박스 자체가 안 나오는가"를 구분한다 -- 전자만 min_conf 차등으로 풀리고, 후자는
타일링(장면 대비 물체 비중을 키우는 것)만 듣는다. 숫자로 확인 없이 문턱만
낮추면 하드코딩 회피 원칙(근거 없는 상수) 위반이라 먼저 찍어본다.

실행 (Mac):
    .venv-umbrella/bin/python3 mec_sim_ice/vla_grounding/validation/diag_small_object_conf.py
"""
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from chungmu_gt import DEFAULT_SCENARIO, load_gt
from run_offline import load_round
from ultralytics import YOLO

_ROOT = os.path.join(_HERE, "..", "..", "..")
WEIGHTS = os.environ.get("VLA_DET_WEIGHTS",
                         os.path.join(_ROOT, "mec_sim_ice", "demo_agents",
                                      "weights", "best.pt"))
RECORDS = os.environ.get("VLA_RECORDS",
                         os.path.join(_ROOT, "mec_sim_ice", "records"))

SMALL_CLASSES = ("tumbler", "pencilcase", "umbrella")


def main() -> int:
    gt = load_gt(DEFAULT_SCENARIO)
    answer_key = {cmd: hit for cmd, hit in gt.items()}
    model = YOLO(WEIGHTS)

    round_dirs = sorted(glob.glob(os.path.join(RECORDS, "s[12]_*")))
    n_checked = 0
    for d in round_dirs:
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta, cams, images = load_round(d)
        hit = answer_key.get(meta["command"])
        if hit is None:
            continue
        pid, expected, weights, kind, _owner = hit
        tcls = expected.get("target_type")
        if tcls not in SMALL_CLASSES:
            continue
        n_checked += 1
        gate_cam = expected.get("camera_id")
        print(f"\n=== {pid}  target={tcls}  record={os.path.basename(d)}  "
              f"GT camera={gate_cam} ===")
        any_hit = False
        for cam_id, img in images.items():
            res = model.predict(img, conf=0.001, imgsz=1280, verbose=False)[0]
            boxes = [(res.names[int(b.cls[0])], float(b.conf[0]))
                    for b in res.boxes]
            hits = sorted([c for lab, c in boxes if lab == tcls], reverse=True)
            if not hits:
                continue
            any_hit = True
            mark = " <-- GT camera" if cam_id == gate_cam else ""
            below = sum(1 for c in hits if c < 0.25)
            print(f"    {cam_id:10} {tcls} confs={[round(c,3) for c in hits]}"
                  f"  (0.25 미만 {below}개){mark}")
        if not any_hit:
            print("    (모든 카메라에서 conf=0.001 로도 박스 자체가 0개 -- "
                  "문턱 조정으로는 구제 불가, 타일링만 듣는 케이스)")
    print(f"\n총 {n_checked}문항 점검")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
