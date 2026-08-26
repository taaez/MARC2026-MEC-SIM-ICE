"""Dataset-gen JSON -> YOLO labels -> YOLO11 detection training.

STATUS: staged in test/ (repo root, sibling of demo/) while the class list/pipeline is
still being shaken out -- see test/README.md. Once verified, det_classes.py and this file
move into demo/ so real_agent_vla.py (also in demo/) can import det_classes directly.

Two steps, run once each (from this test/ directory):

    python3 train_yolo11.py convert  --scenes ../trainer_output/marc2026_chungmu --out dataset
    python3 train_yolo11.py train    --data dataset/data.yaml

``convert`` reads every scene_*/*.json + matching .png from dataset-gen output and writes a
YOLO-format dataset (images/ + labels/ + data.yaml), 90/10 train/val split by scene.
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
    pipeline is verified, copy the final one to demo/weights/best.pt for real_agent_vla.py /
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


def convert(scenes_dir: str, out_dir: str, val_ratio: float = 0.1, seed: int = 0):
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


def train(data_yaml: str, epochs: int, imgsz: int, weights: str, batch: int, workers: int):
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
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"train_{stamp}.log")
    print(f"logging full output to {log_path}")

    with open(log_path, "w") as log_file:
        clean_log = _CleanLogFile(log_file)
        tee_out = _Tee(sys.stdout, clean_log)
        tee_err = _Tee(sys.stderr, clean_log)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            from ultralytics import YOLO  # see note above -- must import after redirect starts

            model = YOLO(weights)  # stock yolo11n.pt -- default kernel/channel widths, no custom arch
            model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, workers=workers)
        clean_log.close()  # flush the final progress-bar line, which never got a trailing \n

    print(f"full output logged to {log_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert")
    c.add_argument("--scenes", required=True,
                   help="dataset-gen output dir, e.g. ../trainer_output/marc2026_chungmu "
                        "(repo-root/trainer_output/<scenario>, NOT under simulation-platform/)")
    c.add_argument("--out", default="dataset")
    c.add_argument("--val-ratio", type=float, default=0.1)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="path to data.yaml from convert")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--imgsz", type=int, default=1280)  # matches CCTV native resolution (1280x720)
    t.add_argument("--weights", default="yolo11n.pt")
    t.add_argument("--batch", type=int, default=16, help="lower this if training crashes with an OOM/dataloader error")
    t.add_argument("--workers", type=int, default=8, help="dataloader worker processes; lower this if training crashes with an OOM/dataloader error")

    args = ap.parse_args()
    if args.cmd == "convert":
        convert(args.scenes, args.out, args.val_ratio)
    elif args.cmd == "train":
        train(args.data, args.epochs, args.imgsz, args.weights, args.batch, args.workers)


if __name__ == "__main__":
    main()
