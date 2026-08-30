"""Dataset-gen JSON -> YOLO labels -> YOLO11 detection training.

STATUS: staged in test/ (repo root, sibling of demo/) while the class list/pipeline is
still being shaken out -- see test/README.md. Once verified, det_classes.py and this file
move into demo/ so record_agent_vla.py (also in demo/) can import det_classes directly.

Two steps, run once each (from this test/ directory):

    python3 train_yolo11.py convert  --scenes ../trainer_output/marc2026_chungmu --out dataset
    python3 train_yolo11.py train    --data dataset/data.yaml

``convert`` reads every scene_*/*.json + matching .png from dataset-gen output and writes a
YOLO-format dataset (images/ + labels/ + data.yaml). Two ways to pick the split:

  - ``--scenes DIR``: one dataset-gen output dir, split 90/10 train/val by scene
    (random, seeded -- val comes from the same generation batch as train).
  - ``--train-scenes DIR --val-scenes DIR``: two SEPARATE dataset-gen output dirs, one used
    entirely for train and the other entirely for val -- no random split, no leakage between
    them. Use this for a genuinely held-out val set (e.g. train from marc2026_demo, val from
    a separately-generated marc2026_validation) instead of a random split of the same batch:

    python3 train_yolo11.py convert \
      --train-scenes ../trainer_output/marc2026_demo \
      --val-scenes   ../trainer_output/marc2026_validation \
      --out dataset

``train`` fine-tunes the stock ``yolo11n.pt`` (default Ultralytics kernel/channel widths --
no architecture changes) on it.

WHERE THINGS LAND (double-checked against simulation-platform/docker-compose.yml, since an
earlier draft of this comment pointed at the wrong path):
  - dataset-gen output (source of --scenes): <repo-root>/trainer_output/<scenario>/scene_*/
    -- NOT simulation-platform/trainer_output/. The dataset-gen service's bind mount is
    ``../trainer_output:/metacom2026/trainer_output`` relative to simulation-platform/, i.e.
    one level ABOVE simulation-platform/, at the repo root (matches .gitignore's
    ``/trainer_output/``, which is root-anchored). test/ is also a repo-root sibling, so the
    same ``../trainer_output`` relative path works unchanged from here.
  - converted YOLO dataset (--out, default "dataset"): test/dataset/ (relative to CWD when
    you run this script -- run it from test/ so it lands next to this file).
  - training run artifacts: ./runs/detect/train/ (Ultralytics default, relative to CWD) --
    test/runs/ if you ran `train` from test/. weights/best.pt lives under there; once the
    pipeline is verified, copy the final one to demo/weights/best.pt for record_agent_vla.py /
    submission (see the guide, step 3-1) -- note the demo/ prefix, this file's own directory
    changes. test/dataset/ and test/runs/ are gitignored -- do not commit raw images or
    training logs, only the final demo/weights/best.pt once it's placed there.

Class list mirrors demo/det_classes.py's DET_CLASSES. If dataset-gen's ``detections[].class``
values differ from these names, fix DET_CLASSES in det_classes.py before running convert (the
label ids depend on this exact order -- see that file's docstring).
"""

import argparse
import contextlib
import datetime
import json
import os
import random
import re
import shutil
import sys

import yaml

from det_classes import DET_CLASSES


class _Tee:
    """Mirror writes to multiple file-like streams (used to echo stdout/stderr to a log
    file while still printing to the terminal -- ultralytics prints warnings/progress
    straight to stdout/stderr, not just through its LOGGER, so wrapping the streams is
    the only way to capture everything a `train` run prints)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# emoji / pictograph / dingbat ranges ultralytics likes to sprinkle in (checks passed ✅,
# WARNING ⚠️, "Ultralytics 8.4.126 🚀 ..."), plus the variation-selector that often
# trails them -- fine on a live terminal, just noise in a saved text log.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF️]"
)


class _CleanLogFile:
    """Write-only wrapper around a log file that makes saved training logs readable:
    strips ANSI color codes and emoji, and collapses carriage-return progress-bar
    redraws (ultralytics' TQDM rewrites the same line dozens of times via '\\r') down to
    just their final state, so one epoch takes one line instead of a wall of box-drawing
    garbage. Purely cosmetic for the log file -- the terminal's own live output (the
    other stream in the _Tee) is untouched."""

    def __init__(self, f):
        self._f = f
        self._buf = ""

    def write(self, data):
        self._buf += data
        while True:
            r = self._buf.find("\r")
            n = self._buf.find("\n")
            if r != -1 and (r < n or n == -1):
                self._buf = self._buf[r + 1:]  # drop the redrawn-over partial line
                continue
            if n != -1:
                line, self._buf = self._buf[:n], self._buf[n + 1:]
                clean = _EMOJI_RE.sub("", _ANSI_RE.sub("", line)).rstrip()
                if clean:
                    self._f.write(clean + "\n")
                continue
            break

    def flush(self):
        self._f.flush()

    def close(self):
        """Flush whatever's left in the buffer (e.g. the final progress-bar line, which
        never gets a trailing newline) -- call once after training finishes."""
        clean = _EMOJI_RE.sub("", _ANSI_RE.sub("", self._buf)).rstrip()
        if clean:
            self._f.write(clean + "\n")
        self._buf = ""
        self._f.flush()


CLASS_INDEX = {name: i for i, name in enumerate(DET_CLASSES)}


def _to_yolo_line(det, img_w, img_h):
    cls = det.get("class")
    # Unknown class (not in det_classes.DET_CLASSES) -- skip rather than guess. Common
    # cause: a new class showed up in a fresh dataset-gen batch that hasn't been added to
    # det_classes.py yet (see that file's docstring -- the taxonomy is not final).
    if cls not in CLASS_INDEX:
        return None
    x_min, y_min, x_max, y_max = det["bbox"]
    x_c = (x_min + x_max) / 2 / img_w
    y_c = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return f"{CLASS_INDEX[cls]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def _process_scenes(scenes_dir: str, img_out: str, lbl_out: str):
    """Convert every scene_*/*.json + matching .png under scenes_dir into one YOLO split
    (img_out/lbl_out). Returns (count, unknown_classes)."""
    scene_dirs = sorted(
        d for d in os.listdir(scenes_dir)
        if d.startswith("scene_") and os.path.isdir(os.path.join(scenes_dir, d))
    )
    if not scene_dirs:
        raise SystemExit(f"no scene_* directories under {scenes_dir} -- run dataset-gen first "
                          f"(bash simulation-platform/marc.sh dataset-gen)")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    count = 0
    unknown_classes = set()

    for scene in scene_dirs:
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
            img_info = payload.get("image", {})
            cam = payload.get("camera", {})
            img_w = img_info.get("width") or cam.get("width") or cam.get("intrinsic", {}).get("width")
            img_h = img_info.get("height") or cam.get("height") or cam.get("intrinsic", {}).get("height")
            if not img_w or not img_h:
                continue  # neither image nor camera block has resolution -- skip rather than guess

            lines = []
            for det in payload.get("detections", []):
                cls = det.get("class")
                if cls and cls not in CLASS_INDEX:
                    unknown_classes.add(cls)
                line = _to_yolo_line(det, img_w, img_h)
                if line is not None:
                    lines.append(line)

            stem = f"{scene}_{cam_id}"
            shutil.copy(png_path, os.path.join(img_out, stem + ".png"))
            with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            count += 1

    return count, unknown_classes


def convert(out_dir: str, scenes_dir: str = None, val_ratio: float = 0.1, seed: int = 0,
            train_scenes_dir: str = None, val_scenes_dir: str = None):
    if train_scenes_dir or val_scenes_dir:
        if not (train_scenes_dir and val_scenes_dir):
            raise SystemExit("--train-scenes and --val-scenes must be given together "
                              "(use --scenes alone for a random split of one dir)")
        # Two separate dataset-gen output dirs -- train and val never come from the same
        # generation batch, so there's no leakage from a random split of shared scenes.
        n_train, unk_train = _process_scenes(
            train_scenes_dir, os.path.join(out_dir, "images", "train"), os.path.join(out_dir, "labels", "train"))
        n_val, unk_val = _process_scenes(
            val_scenes_dir, os.path.join(out_dir, "images", "val"), os.path.join(out_dir, "labels", "val"))
        counts = {"train": n_train, "val": n_val}
        unknown_classes = unk_train | unk_val
    else:
        if not scenes_dir:
            raise SystemExit("convert needs either --scenes, or both --train-scenes and --val-scenes")
        scene_dirs = sorted(
            d for d in os.listdir(scenes_dir)
            if d.startswith("scene_") and os.path.isdir(os.path.join(scenes_dir, d))
        )
        if not scene_dirs:
            raise SystemExit(f"no scene_* directories under {scenes_dir} -- run dataset-gen first "
                              f"(bash simulation-platform/marc.sh dataset-gen)")

        random.Random(seed).shuffle(scene_dirs)
        n_val = max(1, int(len(scene_dirs) * val_ratio))
        splits = {"val": set(scene_dirs[:n_val]), "train": set(scene_dirs[n_val:])}

        counts = {"train": 0, "val": 0}
        unknown_classes = set()

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
                    img_info = payload.get("image", {})
                    cam = payload.get("camera", {})
                    img_w = img_info.get("width") or cam.get("width") or cam.get("intrinsic", {}).get("width")
                    img_h = img_info.get("height") or cam.get("height") or cam.get("intrinsic", {}).get("height")
                    if not img_w or not img_h:
                        continue  # neither image nor camera block has resolution -- skip rather than guess

                    lines = []
                    for det in payload.get("detections", []):
                        cls = det.get("class")
                        if cls and cls not in CLASS_INDEX:
                            unknown_classes.add(cls)
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
    if unknown_classes:
        print(f"WARNING: {len(unknown_classes)} class(es) seen in the data but NOT in "
              f"det_classes.DET_CLASSES (their boxes were dropped, not trained on): "
              f"{sorted(unknown_classes)}")
        print("         Add them to demo/det_classes.py and re-run convert if they matter.")


def oversample(data_yaml: str, out_dir: str, classes: list, factor: int):
    """Duplicate train images/labels that contain any of `classes` `factor` times (val is
    left untouched -- copies must never leak into eval). Duplicates are symlinks to the
    original image (no extra disk for pixels) plus a copied label .txt (tiny), under a
    `__ovN` suffix so filenames stay unique. This is a data-loader-level substitute for
    Ultralytics' old YOLOv5 `--image-weights` flag, which 8.4.126 doesn't expose."""
    with open(data_yaml) as f:
        src_cfg = yaml.safe_load(f)
    src_root = src_cfg["path"]
    target_ids = {CLASS_INDEX[c] for c in classes if c in CLASS_INDEX}
    missing = [c for c in classes if c not in CLASS_INDEX]
    if missing:
        raise SystemExit(f"unknown class(es), not in det_classes.DET_CLASSES: {missing}")

    img_out = os.path.join(out_dir, "images", "train")
    lbl_out = os.path.join(out_dir, "labels", "train")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    src_img_dir = os.path.join(src_root, "images", "train")
    src_lbl_dir = os.path.join(src_root, "labels", "train")
    total_written, hit_written = 0, 0
    for fname in sorted(os.listdir(src_img_dir)):
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(src_lbl_dir, stem + ".txt")
        with open(lbl_path) as f:
            lines = f.read().splitlines()
        has_target = any(lines and int(l.split()[0]) in target_ids for l in lines)
        copies = factor if has_target else 1
        for i in range(copies):
            suffix = f"__ov{i}" if i > 0 else ""
            os.symlink(os.path.abspath(os.path.join(src_img_dir, fname)),
                        os.path.join(img_out, stem + suffix + ".png"))
            shutil.copy(lbl_path, os.path.join(lbl_out, stem + suffix + ".txt"))
            total_written += 1
        if has_target:
            hit_written += copies

    # val: symlinked through unchanged (1:1, never duplicated -- eval must stay honest)
    val_img_out = os.path.join(out_dir, "images", "val")
    val_lbl_out = os.path.join(out_dir, "labels", "val")
    os.makedirs(val_img_out, exist_ok=True)
    os.makedirs(val_lbl_out, exist_ok=True)
    src_val_img_dir = os.path.join(src_root, "images", "val")
    src_val_lbl_dir = os.path.join(src_root, "labels", "val")
    n_val = 0
    for fname in sorted(os.listdir(src_val_img_dir)):
        stem = os.path.splitext(fname)[0]
        os.symlink(os.path.abspath(os.path.join(src_val_img_dir, fname)),
                    os.path.join(val_img_out, fname))
        shutil.copy(os.path.join(src_val_lbl_dir, stem + ".txt"), os.path.join(val_lbl_out, stem + ".txt"))
        n_val += 1

    data_yaml_out = os.path.join(out_dir, "data.yaml")
    with open(data_yaml_out, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(DET_CLASSES)}\n")
        f.write(f"names: {DET_CLASSES!r}\n")

    print(f"wrote {total_written} train images ({hit_written} from {classes} classes, "
          f"factor={factor}) / {n_val} val images -> {data_yaml_out}")


def train(data_yaml: str, epochs: int, imgsz: int, weights: str, batch: int, workers: int,
          project: str = None, name: str = None, resume: bool = False):
    # Full stdout/stderr tee'd to a timestamped file, in addition to the normal terminal
    # output. ultralytics prints its warnings (unknown-arg notices, AMP check results,
    # OOM tracebacks) and PyTorch's own UserWarnings straight to stdout/stderr rather
    # than only through its LOGGER, so wrapping the raw streams is the only way to catch
    # everything -- this is what you grep after the fact instead of having to copy-paste
    # from a terminal that already scrolled away or a background run you can't see.
    #
    # `import ultralytics` MUST happen only after the redirect is active (not at module
    # top-level, and not before entering the `with` below): ultralytics.utils.LOGGER binds
    # a StreamHandler directly to whatever object `sys.stdout` IS at import time
    # (`logging.StreamHandler(sys.stdout)`), once, and keeps that reference forever --
    # importing before the redirect makes every LOGGER.info/.warning call (which is most
    # of the interesting text: AMP checks, weight-transfer counts, class warnings) bypass
    # the tee even though TQDM's progress bars (which re-read sys.stdout on the fly) still
    # get captured fine. Import earlier than this function only if you also move this
    # redirect earlier with it.
    # mec_sim_ice/training/train_yolo11.py -> mec_sim_ice/training_results/yolo_main_logs
    # (2026-08-29 reorg: logs/datasets/results split out of test/ -- see
    # mec_sim_ice/docs/폴더_정리_기록_2026-08-29.md)
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "training_results", "yolo_main_logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Tag the file with --name too. A directory of timestamp-only logs is unreadable once
    # several experiments are in flight -- you end up grepping each one for `name=` just to
    # tell which run it was.
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", name) if name else ""
    log_path = os.path.join(logs_dir, f"train_{stamp}{'_' + tag if tag else ''}.log")
    print(f"logging full output to {log_path}")

    with open(log_path, "w") as log_file:
        clean_log = _CleanLogFile(log_file)
        tee_out = _Tee(sys.stdout, clean_log)
        tee_err = _Tee(sys.stderr, clean_log)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            from ultralytics import YOLO  # see note above -- must import after redirect starts

            model = YOLO(weights)  # stock yolo11n.pt -- default kernel/channel widths, no custom arch
            if resume:
                # Ultralytics restores epochs/data/batch/optimizer/scheduler state from the
                # checkpoint's own args, so passing them again would be ignored at best and
                # conflicting at worst -- hand it only `resume=True`. `weights` must be the
                # interrupted run's weights/last.pt (best.pt has no optimizer state).
                model.train(resume=True)
            else:
                model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
                            project=project, name=name)
        clean_log.close()  # flush the final progress-bar line, which never got a trailing \n

    print(f"full output logged to {log_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert")
    c.add_argument("--scenes",
                   help="single dataset-gen output dir, split --val-ratio randomly into train/val "
                        "(repo-root/trainer_output/<scenario>, NOT under simulation-platform/). "
                        "Mutually exclusive with --train-scenes/--val-scenes.")
    c.add_argument("--train-scenes",
                   help="dataset-gen output dir used entirely for train (e.g. "
                        "../trainer_output/marc2026_demo). Must be given together with --val-scenes.")
    c.add_argument("--val-scenes",
                   help="dataset-gen output dir used entirely for val -- a genuinely separate "
                        "generation batch, not a random split of --train-scenes (e.g. "
                        "../trainer_output/marc2026_validation). Must be given together with --train-scenes.")
    c.add_argument("--out", default="dataset")
    c.add_argument("--val-ratio", type=float, default=0.1,
                   help="only used with --scenes (ignored for --train-scenes/--val-scenes)")

    o = sub.add_parser("oversample",
                        help="duplicate train images containing given classes N times "
                             "(val untouched) -- a stand-in for the class-balanced sampling "
                             "Ultralytics 8.4.126 doesn't expose")
    o.add_argument("--data", required=True, help="path to data.yaml from convert")
    o.add_argument("--out", required=True)
    o.add_argument("--classes", required=True, help="comma-separated class names, e.g. sunblock,tissue,disposable_cup")
    o.add_argument("--factor", type=int, default=4, help="total copies of each matching train image (1 = no oversampling)")

    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="path to data.yaml from convert")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--imgsz", type=int, default=1280)  # matches CCTV native resolution (1280x720)
    t.add_argument("--weights", default="yolo11n.pt")
    t.add_argument("--batch", type=int, default=16, help="lower this if training crashes with an OOM/dataloader error")
    t.add_argument("--workers", type=int, default=8, help="dataloader worker processes; lower this if training crashes with an OOM/dataloader error")
    t.add_argument("--project", default=None, help="Ultralytics run output dir (default: runs/detect, relative to CWD)")
    t.add_argument("--name", default=None, help="Ultralytics run name (default: auto-incrementing train, train2, ...)")
    t.add_argument("--resume", action="store_true",
                   help="continue an interrupted run: pass its weights/last.pt as --weights; "
                        "every other setting is restored from that checkpoint")

    args = ap.parse_args()
    if args.cmd == "convert":
        convert(args.out, args.scenes, args.val_ratio,
                train_scenes_dir=args.train_scenes, val_scenes_dir=args.val_scenes)
    elif args.cmd == "oversample":
        oversample(args.data, args.out, args.classes.split(","), args.factor)
    elif args.cmd == "train":
        train(args.data, args.epochs, args.imgsz, args.weights, args.batch, args.workers,
              args.project, args.name, args.resume)


if __name__ == "__main__":
    main()
