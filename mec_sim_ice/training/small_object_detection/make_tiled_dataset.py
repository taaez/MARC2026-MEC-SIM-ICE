"""Slice an existing YOLO dataset into overlapping tiles for the small-object sub-model.

Takes the dataset `train_yolo11.py convert` already produced (e.g. test/dataset_v2, 34
classes, 1280x720 renders), keeps only the target classes, and writes a NEW dataset whose
images are overlapping crops of the originals. This is step 1 of the sub-model pipeline --
see README.md for the whole sequence and why the sub-model exists at all.

    python3 make_tiled_dataset.py --base ../dataset_v2 --out dataset_small_t640 \
        --tile 640 --train-imgsz 1280

Alongside the dataset it writes `tiling.json` (common.TilingConfig): the tile size, overlap
and train imgsz that produced it. train_small.py and predict.py read that file instead of
taking the numbers again on the command line, because inference sliced differently from
training fails silently -- the model simply under-performs at a scale it never saw.

WHY TILING (the part that actually matters)
-------------------------------------------
YOLO11 n/s/m/l/x all use stride [8, 16, 32]; the finest feature map has one grid cell per
8x8 px of MODEL INPUT. The three target classes average 5-8 px per side, i.e. a whole object
fits inside one grid cell, which is why IoU-based assignment and CIoU loss are so unstable
for them.

Tiling helps ONLY through the upscale it enables:

    effective object size = original px  x  (train_imgsz / tile)

A 640 px tile trained at imgsz=1280 doubles every object: a 6 px cup becomes 12 px, ~1.9
grid cells. A 320 px tile quadruples it (~24 px, ~3.7 cells), where IoU stops being
degenerate -- at ~2.5x the tile count. Cropping WITHOUT that upscale (tile 640 trained at
imgsz 640) changes nothing: still 6 px against an 8 px cell. This script prints the achieved
factor and resulting box sizes so the choice gets checked rather than assumed.

NOT THE SAME AS THE GT-CROP EXPERIMENT THAT FAILED
--------------------------------------------------
The 2026-08-27 "GT crop" attempt cropped around each ground-truth box, centering every
object and destroying both the object-position prior and the surrounding context. This cuts
a REGULAR GRID out of the untouched render: shadows, ground contact, perspective and
lighting all survive, objects land wherever they happen to land, and empty tiles become
background negatives -- which the source dataset has none of (601 train images, 0
backgrounds). Different method, different failure mode.

Validation produced here is a TRAINING MONITOR ONLY: it scores the model on the same tile
distribution it trained on. The authoritative number comes from evaluate.py, which runs the
full routed pipeline over original full-size images at the deployment conf cut.
"""

import argparse
import os
import random
import sys
from collections import Counter

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from det_classes import DET_CLASSES  # noqa: E402  (needs the path shim above)

from common import SMALL_CLASSES, TilingConfig, tile_grid  # noqa: E402

CLASS_INDEX = {name: i for i, name in enumerate(DET_CLASSES)}


def _read_labels(path: str):
    """YOLO label file -> [(class_id, cx, cy, w, h)] normalized. Missing file = background
    image, which is legitimate input rather than an error."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            out.append((int(parts[0]), *(float(p) for p in parts[1:])))
    return out


def _tile_boxes(boxes_px, x0, y0, tile, min_vis, min_box):
    """Clip absolute-pixel boxes to one tile -> tile-local normalized YOLO rows.

    Two filters, both about not teaching the model garbage at tile seams:
      - `min_vis`: a box cut by a tile edge is dropped unless this fraction of its area
        survives. Overlap means the dropped sliver is almost always whole in a neighbouring
        tile. At 5-8 px against a 640 px tile this fires rarely -- the count is reported.
      - `min_box`: drops degenerate slivers that pass min_vis but are a pixel wide.
    """
    rows, dropped = [], 0
    x1t, y1t = x0 + tile, y0 + tile
    for cls, bx0, by0, bx1, by1 in boxes_px:
        ix0, iy0 = max(bx0, x0), max(by0, y0)
        ix1, iy1 = min(bx1, x1t), min(by1, y1t)
        iw, ih = ix1 - ix0, iy1 - iy0
        if iw <= 0 or ih <= 0:
            continue
        box_area = (bx1 - bx0) * (by1 - by0)
        if box_area <= 0:
            continue
        if (iw * ih) / box_area < min_vis or iw < min_box or ih < min_box:
            dropped += 1
            continue
        rows.append((cls, ((ix0 + ix1) / 2 - x0) / tile, ((iy0 + iy1) / 2 - y0) / tile,
                     iw / tile, ih / tile))
    return rows, dropped


def _plan_split(base, split, keep_ids, remap, cfg, min_vis, min_box):
    """Enumerate every tile of every image in a split without writing anything.

    Planning separately from writing is what lets negatives be sampled at a ratio to the
    positives actually found -- those counts aren't known until the split has been walked.
    """
    img_dir = os.path.join(base, "images", split)
    lbl_dir = os.path.join(base, "labels", split)
    if not os.path.isdir(img_dir):
        raise SystemExit(f"missing split dir: {img_dir}")

    positives, negatives = [], []
    stats = {"images": 0, "tiles": 0, "dropped_boxes": 0, "src_sizes": [],
             "per_class": Counter()}

    for fname in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        img_path = os.path.join(img_dir, fname)
        with Image.open(img_path) as im:
            W, H = im.size
        stats["images"] += 1

        # Absolute-pixel boxes, target classes only. Source labels are normalized against
        # the full image; every decision below is in pixels because tile membership and
        # visibility have no meaning in normalized space.
        boxes_px = []
        for cls, cx, cy, w, h in _read_labels(os.path.join(lbl_dir, stem + ".txt")):
            if cls not in keep_ids:
                continue
            bw, bh = w * W, h * H
            boxes_px.append((remap[cls], cx * W - bw / 2, cy * H - bh / 2,
                             cx * W + bw / 2, cy * H + bh / 2))
            stats["src_sizes"].append(max(bw, bh))

        for ix, iy, x0, y0 in tile_grid(W, H, cfg.tile, cfg.stride):
            rows, dropped = _tile_boxes(boxes_px, x0, y0, cfg.tile, min_vis, min_box)
            stats["dropped_boxes"] += dropped
            stats["tiles"] += 1
            entry = (img_path, f"{stem}__t{ix}_{iy}", x0, y0, rows)
            (positives if rows else negatives).append(entry)
            for r in rows:
                stats["per_class"][r[0]] += 1

    return positives, negatives, stats


def _write_split(entries, out_dir, split, tile):
    """Crop and write the chosen tiles. Entries are sorted by source path so a 15-tile
    image is decoded once instead of 15 times."""
    img_out = os.path.join(out_dir, "images", split)
    lbl_out = os.path.join(out_dir, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    cur_path, cur_im = None, None
    try:
        for img_path, stem, x0, y0, rows in sorted(entries, key=lambda e: e[0]):
            if img_path != cur_path:
                if cur_im is not None:
                    cur_im.close()
                cur_im = Image.open(img_path).convert("RGB")
                cur_path = img_path
            cur_im.crop((x0, y0, x0 + tile, y0 + tile)).save(
                os.path.join(img_out, stem + ".png"))
            with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
                f.write("\n".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}"
                                  for c, a, b, w, h in rows))
    finally:
        if cur_im is not None:
            cur_im.close()


def _pct(values, q):
    if not values:
        return 0.0
    return sorted(values)[min(len(values) - 1, int(q * len(values)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="existing YOLO dataset from `train_yolo11.py convert` (e.g. ../dataset_v2)")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--classes", default=",".join(SMALL_CLASSES),
                    help="comma-separated class names to keep, reindexed to 0..N-1 in this order")
    ap.add_argument("--tile", type=int, default=640, help="tile size in source pixels")
    ap.add_argument("--overlap", type=float, default=0.25,
                    help="fraction shared with the neighbouring tile (stride = tile * (1 - overlap))")
    ap.add_argument("--train-imgsz", type=int, default=1280,
                    help="imgsz the sub-model will be TRAINED at. Nothing is resized here -- this "
                         "is recorded in tiling.json and used to report the upscale factor, which "
                         "is the only thing tiling actually buys.")
    ap.add_argument("--min-vis", type=float, default=0.5,
                    help="drop a tile-edge-cut box unless this fraction of its area survives")
    ap.add_argument("--min-box", type=float, default=2.0,
                    help="drop clipped boxes thinner than this many source px on either side")
    ap.add_argument("--neg-per-pos", type=float, default=0.5,
                    help="background tiles kept per positive tile (0 = none). The source dataset "
                         "has zero background images, so some negatives help hold false positives "
                         "down -- but for classes this sparse nearly every tile is empty, so they "
                         "are sampled rather than all kept.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    missing = [c for c in names if c not in CLASS_INDEX]
    if missing:
        raise SystemExit(f"unknown class(es), not in det_classes.DET_CLASSES: {missing}")
    keep_ids = {CLASS_INDEX[c] for c in names}
    remap = {CLASS_INDEX[c]: i for i, c in enumerate(names)}

    cfg = TilingConfig(tile=args.tile, overlap=args.overlap, train_imgsz=args.train_imgsz,
                       classes=names, source=os.path.abspath(args.base))
    rng = random.Random(args.seed)
    print(f"tile={cfg.tile} stride={cfg.stride} (overlap {cfg.overlap:.0%})  classes={names}")

    all_src, all_eff = [], []
    for split in ("train", "val"):
        pos, neg, st = _plan_split(args.base, split, keep_ids, remap, cfg,
                                   args.min_vis, args.min_box)
        n_neg = min(len(neg), int(round(len(pos) * args.neg_per_pos)))
        _write_split(pos + rng.sample(neg, n_neg), args.out, split, cfg.tile)

        all_src += st["src_sizes"]
        all_eff += [s * cfg.upscale for s in st["src_sizes"]]
        print(f"\n[{split}] {st['images']} source images -> {st['tiles']} tiles enumerated, "
              f"{len(pos) + n_neg} written ({len(pos)} positive + {n_neg} background "
              f"of {len(neg)} available)")
        for i, name in enumerate(names):
            print(f"    {name:<18} {st['per_class'][i]:>6} instances")
        if st["dropped_boxes"]:
            print(f"    dropped at tile seams: {st['dropped_boxes']} "
                  f"(min_vis={args.min_vis}, min_box={args.min_box})")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"names: {names!r}\n")
    cfg.save(args.out)

    # The whole point, stated as a number: does a stride-8 grid cell (8 px of model input)
    # now sit INSIDE the object instead of swallowing it whole?
    print(f"\nupscale factor at train imgsz={cfg.train_imgsz}: {cfg.upscale:.2f}x")
    print(f"  box longest side, source px : p10={_pct(all_src, .10):.1f}  "
          f"p50={_pct(all_src, .50):.1f}  p90={_pct(all_src, .90):.1f}")
    print(f"  box longest side, model px  : p10={_pct(all_eff, .10):.1f}  "
          f"p50={_pct(all_eff, .50):.1f}  p90={_pct(all_eff, .90):.1f}")
    print(f"  stride-8 cells across a median box : {_pct(all_eff, .50) / 8:.2f}")
    if cfg.upscale <= 1.0:
        print("  WARNING: upscale <= 1.0 -- tiling buys nothing at this train-imgsz. "
              "Raise --train-imgsz or lower --tile.")
    print(f"\nwrote {os.path.join(args.out, 'data.yaml')} and {os.path.join(args.out, 'tiling.json')}")


if __name__ == "__main__":
    main()
