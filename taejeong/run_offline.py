"""오프라인 엔드투엔드 실행: 기록된 CCTV 라운드 → YOLO → 파싱 → 기하 → payload.

입력은 데스크탑 세션이 남긴 records/s1_* 폴더 (ROS/Isaac Sim 무의존 순수 데이터):
  records/s1_*/
    meta.json   - {"command": ..., "cameras": {cam_id: {K, R_cw, C_w, ground_z,
                   width, height}}, "payload": <그때 제출했던 payload>}
    <cam_id>.png  카메라별 CCTV 프레임

사용:
  <python-with-ultralytics> run_offline.py --records <records 폴더> \
      --weights weights/best.pt [--conf 0.25] [--imgsz 1280]

출력: 라운드별 카메라 탐지 요약 + 우리 파이프라인의 payload
(기록된 당시 payload 와 나란히 비교 출력).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from command_parser import parse_command
from grounding import CameraModel, resolve
from yolo_adapter import infer_cameras


def load_round(d: str):
    meta = json.load(open(os.path.join(d, "meta.json")))
    cams, images = {}, {}
    from PIL import Image
    for cid, c in meta["cameras"].items():
        p = os.path.join(d, f"{cid}.png")
        if not os.path.exists(p):
            continue
        cams[cid] = CameraModel(
            cid, np.array(c["K"], dtype=float), np.array(c["R_cw"], dtype=float),
            np.array(c["C_w"], dtype=float), float(c["ground_z"]),
            int(c["width"]), int(c["height"]))
        rgb = np.array(Image.open(p).convert("RGB"))
        images[cid] = rgb[:, :, ::-1].copy()      # ultralytics 는 BGR 해석
    return meta, cams, images


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    rounds = sorted(glob.glob(os.path.join(args.records, "s1_*")))
    if not rounds:
        rounds = [args.records]           # 단일 라운드 폴더를 직접 준 경우
    for d in rounds:
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta, cams, images = load_round(d)
        cmd = meta["command"]
        print("=" * 78)
        print(f"round: {os.path.basename(d)}")
        print(f"command: {cmd}")

        parsed = parse_command(cmd)
        print(f"parsed: target={parsed.target_type} landmark={parsed.landmark_type} "
              f"relation={parsed.relation_hint}(conf={parsed.relation_confident}) "
              f"person={parsed.is_person} pose={parsed.pose_hint}")

        dets = infer_cameras(model, images, min_conf=args.conf, imgsz=args.imgsz)
        for cid in sorted(dets):
            labels = {}
            for det in dets[cid]:
                labels[det.label] = labels.get(det.label, 0) + 1
            print(f"  {cid}: {sum(labels.values())} dets  {dict(sorted(labels.items()))}")

        decision = resolve(parsed, dets, cams)
        if decision is None:
            print("  -> target 미탐지 (폴백 필요)")
            continue
        print("  ours   :", json.dumps(decision.to_payload()))
        if meta.get("payload"):
            print("  recorded:", json.dumps(meta["payload"]))
        print("  debug  :", json.dumps(decision.debug))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
