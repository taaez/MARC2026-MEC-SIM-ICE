"""Train the small-object sub-model on a tiled dataset. Step 2 of the pipeline.

    python3 train_small.py --data dataset_small_t640 --weights yolo11s.pt --name small_t640

Deliberately thin: it is stock Ultralytics training on a stock yolo11s, no architecture
changes and no custom loss. That is the plan's "안 B" -- see README. The value it adds over
calling `yolo train` directly is three things that have each already cost this project a run:

  1. It reads `tiling.json` and uses ITS `train_imgsz`, rather than letting imgsz be retyped
     and silently disagree with the tiling the dataset was built for. imgsz here IS the
     upscale factor; getting it wrong makes the tiling pointless while still looking like a
     normal training run.
  2. It refuses to start at upscale <= 1.0, which is that same mistake in its terminal form.
  3. It tees the full stdout/stderr to test/logs/ the way train_yolo11.py does, so the run
     can be grepped afterwards instead of being lost to a scrolled-away terminal.

The 3-class head is trained from the same pretrained yolo11s.pt as model B. Only 3 of the 80
COCO rows can be name-matched, so effectively the head is fresh while the backbone is not --
which is what we want: the backbone's low-level features transfer, the class semantics don't.

NOTE on OOM: model B needed batch 16 -> 4 at imgsz 1280 on the 7.7 GB RTX 5060 Ti. Tiles are
square 640/320 crops fed at 1280, so the per-image cost is comparable; expect the same
auto-reduction. Pass --batch explicitly to skip the retry cycle.
"""

import argparse
import contextlib
import datetime
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_yolo11 import _CleanLogFile, _Tee  # noqa: E402  (log tee, kept in one place)

from common import TilingConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="tiled dataset dir from make_tiled_dataset.py (holds data.yaml + tiling.json)")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16,
                    help="lower this if training crashes or auto-reduces repeatedly")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override tiling.json's train_imgsz. Only pass this if you mean to "
                         "train at a different upscale than the dataset was planned for.")
    ap.add_argument("--project", default=None, help="default: test/runs/detect")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: pass its weights/last.pt as --weights; "
                         "every other setting is restored from that checkpoint")
    args = ap.parse_args()

    data_yaml = os.path.join(args.data, "data.yaml") if os.path.isdir(args.data) else args.data
    if not os.path.exists(data_yaml):
        raise SystemExit(f"no data.yaml at {data_yaml}")

    cfg = TilingConfig.load(os.path.dirname(os.path.abspath(data_yaml)))
    imgsz = args.imgsz or cfg.train_imgsz
    upscale = imgsz / cfg.tile
    print(f"tiling: tile={cfg.tile} overlap={cfg.overlap:.0%} classes={cfg.classes}")
    print(f"training at imgsz={imgsz} -> upscale {upscale:.2f}x "
          f"(a 6 px object is seen as {6 * upscale:.1f} px, {6 * upscale / 8:.2f} stride-8 cells)")
    if upscale <= 1.0 and not args.resume:
        raise SystemExit(
            f"upscale {upscale:.2f}x buys nothing -- at imgsz {imgsz} a {cfg.tile} px tile is fed "
            f"at its own size, so objects stay their original pixel size and still fall inside a "
            f"single stride-8 cell. Raise --imgsz or rebuild the dataset with a smaller --tile.")

    # mec_sim_ice/training/small_object_detection/train_small.py ->
    # mec_sim_ice/training_results/yolo_main_logs (shared with train_yolo11.py -- same
    # convention as before the 2026-08-29 reorg, see mec_sim_ice/docs/폴더_정리_기록_2026-08-29.md)
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "training_results", "yolo_main_logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", args.name) if args.name else "small"
    log_path = os.path.join(logs_dir, f"train_{stamp}_{tag}.log")
    print(f"logging full output to {log_path}")

    with open(log_path, "w") as log_file:
        clean_log = _CleanLogFile(log_file)
        with contextlib.redirect_stdout(_Tee(sys.stdout, clean_log)), \
                contextlib.redirect_stderr(_Tee(sys.stderr, clean_log)):
            # Import only after the redirect is live: ultralytics' LOGGER binds a
            # StreamHandler to whatever sys.stdout IS at import time and keeps that
            # reference forever, so importing earlier makes every LOGGER call bypass the
            # tee. Same constraint as train_yolo11.train() -- see its comment.
            from ultralytics import YOLO

            model = YOLO(args.weights)
            if args.resume:
                model.train(resume=True)
            else:
                model.train(data=data_yaml, epochs=args.epochs, imgsz=imgsz, batch=args.batch,
                            workers=args.workers, project=args.project, name=args.name)
        clean_log.close()

    print(f"full output logged to {log_path}")
    print("\nnext: evaluate on FULL images, not tiles --\n"
          f"  python3 evaluate.py --base ../dataset_v2 --split val \\\n"
          f"      --general <B best.pt> --small <this run's best.pt> --tiling {args.data}")


if __name__ == "__main__":
    main()
