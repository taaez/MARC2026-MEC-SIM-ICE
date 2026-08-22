"""Dataset-gen JSON -> YOLO labels -> YOLO11 detection training.

Two steps, run once each:

    python3 train_yolo11.py convert  --scenes ../simulation-platform/trainer_output/marc2026_chungmu --out dataset
    python3 train_yolo11.py train    --data dataset/data.yaml

``convert`` reads every scene_*/*.json + matching .png from dataset-gen output (see
MARC2026_요약_v2.md #9) and writes a YOLO-format dataset (images/ + labels/ + data.yaml),
90/10 train/val split by scene. ``train`` fine-tunes the stock ``yolo11n.pt`` (default
Ultralytics kernel/channel widths -- no architecture changes) on it.

Class list mirrors demo/real_agent_vla.py's DET_CLASSES. If dataset-gen's
``detections[].class`` values differ from these names, fix DET_CLASSES in *both* files
before running convert (the label ids depend on this exact order).
"""

import argparse
import json
import os
import random
import shutil

from real_agent_vla import DET_CLASSES

CLASS_INDEX = {name: i for i, name in enumerate(DET_CLASSES)}


def _to_yolo_line(det, img_w, img_h):
    cls = det.get("class")
    if cls not in CLASS_INDEX:
        return None
    x_min, y_min, x_max, y_max = det["bbox"]
    x_c = (x_min + x_max) / 2 / img_w
    y_c = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return f"{CLASS_INDEX[cls]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def convert(scenes_dir: str, out_dir: str, val_ratio: float = 0.1, seed: int = 0):
    scene_dirs = sorted(
        d for d in os.listdir(scenes_dir)
        if d.startswith("scene_") and os.path.isdir(os.path.join(scenes_dir, d))
    )
    if not scene_dirs:
        raise SystemExit(f"no scene_* directories under {scenes_dir} -- run dataset-gen first")

    random.Random(seed).shuffle(scene_dirs)
    n_val = max(1, int(len(scene_dirs) * val_ratio))
    splits = {"val": set(scene_dirs[:n_val]), "train": set(scene_dirs[n_val:])}

    counts = {"train": 0, "val": 0}

    for split, dirs in splits.items():
        img_out = os.path.join(out_dir, "images", split)
        lbl_out = os.path.join(out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for scene in dirs:
            scene_path = os.path.join(scenes_dir, scene)
            for fname in os.listdir(scene_path):
                if not fname.endswith(".json") or fname.endswith("_overlay.json"):
                    continue
                cam_id = fname[:-len(".json")]
                png_path = os.path.join(scene_path, cam_id + ".png")
                json_path = os.path.join(scene_path, fname)
                if not os.path.exists(png_path):
                    continue

                with open(json_path) as f:
                    payload = json.load(f)
                cam = payload.get("camera", {})
                img_w = cam.get("width") or cam.get("intrinsic", {}).get("width")
                img_h = cam.get("height") or cam.get("intrinsic", {}).get("height")
                if not img_w or not img_h:
                    continue  # camera block missing resolution -- skip rather than guess

                lines = []
                for det in payload.get("detections", []):
                    line = _to_yolo_line(det, img_w, img_h)
                    if line is not None:
                        lines.append(line)

                stem = f"{scene}_{cam_id}"
                shutil.copy(png_path, os.path.join(img_out, stem + ".png"))
                with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                counts[split] += 1

    data_yaml = os.path.join(out_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(DET_CLASSES)}\n")
        f.write(f"names: {DET_CLASSES!r}\n")

    print(f"wrote {counts['train']} train / {counts['val']} val images -> {data_yaml}")


def train(data_yaml: str, epochs: int, imgsz: int, weights: str):
    from ultralytics import YOLO
    model = YOLO(weights)  # stock yolo11n.pt -- default kernel/channel widths, no custom arch
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert")
    c.add_argument("--scenes", required=True, help="dataset-gen trainer_output/<scenario> dir")
    c.add_argument("--out", default="dataset")
    c.add_argument("--val-ratio", type=float, default=0.1)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="path to data.yaml from convert")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--weights", default="yolo11n.pt")

    args = ap.parse_args()
    if args.cmd == "convert":
        convert(args.scenes, args.out, args.val_ratio)
    elif args.cmd == "train":
        train(args.data, args.epochs, args.imgsz, args.weights)


if __name__ == "__main__":
    main()
