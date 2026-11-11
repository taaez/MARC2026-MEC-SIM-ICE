"""GT sampling (copy-paste augmentation) for the tiny pickable classes.

Motivation: sunblock / tissue /
disposable_cup sit far below every other class, and image-level oversampling
(``train_yolo11.py oversample``) only partly helps -- it duplicates whole images, so the
target share of all instances stays pinned at ~7.5% no matter how high the duplication
factor goes, and it dragged 9 unrelated classes DOWN in experiment A.

This does the thing oversampling cannot: raise the *instance density* of the target
classes without duplicating a single non-target instance, by pasting real target crops
into other scenes -- the 2D analogue of GT/database sampling from 3D detection
(SECOND / PointPillars).

Ultralytics' own ``copy_paste=`` flag is a no-op here: augment.py's CopyPaste bails out on
``len(labels["instances"].segments) == 0``, and dataset-gen emits bounding boxes only (no
segmentation masks). Hence this offline pass.

WHY BBOX CROPS ARE GOOD ENOUGH: the target objects average ~5-10px per side in these
renders, so a bbox is very nearly a mask -- there is almost no background inside an 8x8
box to leak in. The same would not hold for large classes, which is why only the small
pickables are pasted.

PLACEMENT / SCALE: rather than unprojecting through the camera (Isaac Sim's euler
convention is easy to get wrong, and a silent sign error would poison every pasted box),
the apparent size is calibrated empirically per (camera, class): across the real scenes,
object size vs. image row is essentially linear, r = +0.79..+0.99. So a donor pasted at
row y is resized to ``a + b*y`` for that camera/class, and dropped only on ground spots
where dataset-gen actually placed objects for that camera. Nothing depends on extrinsics.

Only the TRAIN split is touched. val is passed through untouched from the base dataset --
pasted pixels must never reach evaluation.

By default each train image is REPLACED by its augmented version (same image count, same
epochs, same compute as the baseline), so a run on this dataset is directly comparable to
one on the base dataset with no "it just trained longer" confound -- the flaw that made
experiment A hard to interpret. ``--keep-original`` opts into keeping both copies instead.

Usage (from this test/ directory):

    python3 gt_sample.py stats --scenes ../../trainer_output/marc2026_demo

    python3 gt_sample.py build \
      --scenes    ../../trainer_output/marc2026_demo \
      --base      dataset_v2 \
      --out       dataset_v4_gtsample \
      --per-image 3

Then train on it exactly like any other dataset:

    python3 train_yolo11.py train --data dataset_v4_gtsample/data.yaml \
      --project <abs path>/runs/detect --name train_D_gtsample
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

import numpy as np
import yaml
from PIL import Image

from det_classes import DET_CLASSES

CLASS_INDEX = {name: i for i, name in enumerate(DET_CLASSES)}

DEFAULT_CLASSES = ["sunblock", "tissue", "disposable_cup"]

# dataset-gen writes -1.0 when it did not compute occlusion for an instance (every
# sunblock is like this), and 0.0 for "fully visible". Anything above 0 is genuinely
# occluded and makes a bad donor -- a crop of a half-hidden object teaches the wrong shape.
CLEAN_OCCLUSION = (0.0, -1.0)

# Landmarks a dropped object cannot rest on: tall things that stand ON the ground, so their
# bbox already reaches the ground line. A ground contact point INSIDE such a silhouette means
# the object is on the bodywork / on top of the machine / behind it -- never a valid resting
# place. (An object standing in FRONT of one is fine and stays allowed: being nearer the
# camera, its contact point falls BELOW the landmark's box, not inside it.)
#
# picnic_table and bench are deliberately absent -- they are flat surfaces meant to hold
# things, and the competition places objects on them outright ("Find the mug on the picnic
# table"), so a contact inside those boxes is legitimate.
NO_STAND_CLASSES = frozenset({
    "normal_car_1", "normal_car_2", "sports_car", "sports_car_2", "suv", "taxi",
    "police_car", "bicycle", "kick_scooter",
    "vending_machine", "postbox", "trash_can", "hydrant",
    # People too: a contact inside someone's silhouette means the object is resting on them.
    "person_walking", "person_walking_away", "person_crouching", "person_bending",
    "person_reclining", "person_prone", "person_lying_down", "person_reaching",
    "person_sitting",
})

# Where a small dropped object may be placed. Ground spots are taken only from instances of
# these classes -- i.e. "somewhere dataset-gen actually put a litter-sized object" -- rather
# than from every movable instance. Sampling people's foot positions too (they are also
# `kind == "object"`) put pastes on top of pedestrians, and biased the row distribution the
# apparent-size model is evaluated at.
PLACEABLE_CLASSES = frozenset({
    "mug", "tumbler", "sunblock", "umbrella", "pencilcase", "cola_can", "cracker_box",
    "disposable_cup", "tissue", "juice",
})

MIN_FIT_SAMPLES = 8      # below this the per-(camera,class) size fit is not trustworthy
MIN_PASTE_PX = 2         # a box under ~2px carries no signal and confuses the loss
MAX_PASTE_PX = 80
MAX_PLACE_TRIES = 25     # attempts to find a non-overlapping spot before giving up


def _ring_bg(img: np.ndarray, box, pad: int = 3):
    """Mean RGB of the ground immediately around a box (the box itself excluded).

    Serves two jobs: matching a donor's surroundings to the spot it is pasted into, and
    telling the paste mask what counts as "background" inside the donor crop. A crop carries
    some of its original ground in the bbox corners; dropping a dark-asphalt crop onto bright
    concrete leaves a visible rectangle, and a detector will happily learn that rectangle
    instead of the object -- a shortcut that does not exist in real footage.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = img.shape[:2]
    ox0, oy0 = max(0, x0 - pad), max(0, y0 - pad)
    ox1, oy1 = min(w, x1 + pad), min(h, y1 + pad)
    outer = img[oy0:oy1, ox0:ox1]
    if outer.size == 0:
        return None
    total = outer.reshape(-1, outer.shape[-1]).astype(np.float64).sum(axis=0)
    n = outer.shape[0] * outer.shape[1]
    inner = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if inner.size:
        total -= inner.reshape(-1, inner.shape[-1]).astype(np.float64).sum(axis=0)
        n -= inner.shape[0] * inner.shape[1]
    return (total / n).astype(np.float32) if n > 0 else None


def _iter_scenes(scenes_dir: str):
    """Yield (scene, cam_id, png_path, payload) for every scene_*/<cam>.json + .png pair."""
    scene_dirs = sorted(
        d for d in os.listdir(scenes_dir)
        if d.startswith("scene_") and os.path.isdir(os.path.join(scenes_dir, d))
    )
    if not scene_dirs:
        raise SystemExit(f"no scene_* directories under {scenes_dir} -- run dataset-gen first")

    for scene in scene_dirs:
        scene_path = os.path.join(scenes_dir, scene)
        for fname in sorted(os.listdir(scene_path)):
            if not fname.endswith(".json") or fname.endswith("_overlay.json"):
                continue
            cam_id = fname[:-len(".json")]
            png_path = os.path.join(scene_path, cam_id + ".png")
            if not os.path.exists(png_path):
                continue
            with open(os.path.join(scene_path, fname)) as f:
                payload = json.load(f)
            yield scene, cam_id, png_path, payload


def _fit_line(xs, ys):
    """Least-squares fit y = a + b*x. Falls back to a flat mean when x has no spread."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


class Banks:
    """Donor crops, per-(camera,class) size models, and valid ground spots per camera."""

    def __init__(self):
        self.donors = defaultdict(list)     # (cam, cls) -> [(png_path, bbox, bg_luma)]
        self.fits = {}                      # (cam, cls) -> (a, b) for size = a + b*row
        self.cam_fits = {}                  # cam -> (a, b) pooled over target classes
        self.spots = defaultdict(list)      # cam -> [(x_center, y_bottom)]

    def scale_at(self, cam: str, cls: str, y_bottom: float):
        """Apparent object size (px per side) for this camera/class at image row y_bottom."""
        fit = self.fits.get((cam, cls)) or self.cam_fits.get(cam)
        if fit is None:
            return None
        size = fit[0] + fit[1] * y_bottom
        return size if size >= MIN_PASTE_PX else None


def build_banks(scenes_dir: str, classes) -> Banks:
    banks = Banks()
    rows = defaultdict(list)        # (cam, cls) -> [(y_bottom, size)]
    cam_rows = defaultdict(list)    # cam -> [(y_bottom, size)] pooled over target classes

    for _scene, cam_id, png_path, payload in _iter_scenes(scenes_dir):
        img = None      # loaded only if this scene actually contributes a donor
        for det in payload.get("detections", []):
            cls = det.get("class")
            x0, y0, x1, y1 = det["bbox"]
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue

            # Ground spots: anywhere dataset-gen actually put a litter-sized object for this
            # camera is, by construction, a plausible place to drop another one.
            if cls in PLACEABLE_CLASSES:
                banks.spots[cam_id].append(((x0 + x1) / 2, y1))

            if cls not in classes:
                continue
            size = (w + h) / 2
            rows[(cam_id, cls)].append((y1, size))
            cam_rows[cam_id].append((y1, size))
            if det.get("occlusion") in CLEAN_OCCLUSION:
                if img is None:
                    img = np.asarray(Image.open(png_path).convert("RGB"))
                bg = _ring_bg(img, (x0, y0, x1, y1))
                if bg is not None:
                    banks.donors[(cam_id, cls)].append((png_path, (x0, y0, x1, y1), bg))

    for key, pts in rows.items():
        if len(pts) >= MIN_FIT_SAMPLES:
            banks.fits[key] = _fit_line([p[0] for p in pts], [p[1] for p in pts])
    for cam, pts in cam_rows.items():
        if len(pts) >= MIN_FIT_SAMPLES:
            banks.cam_fits[cam] = _fit_line([p[0] for p in pts], [p[1] for p in pts])

    return banks


def _overlaps(box, existing, thresh=0.3):
    """True if box overlaps an existing box too much to be a believable new object.

    Against a large landmark the IoU of a ~6px box is negligible, so in practice this only
    rejects collisions with other small objects -- which is what we want: pasting *near*
    a bench or table is realistic, pasting on top of another cup is not.
    """
    ax0, ay0, ax1, ay1 = box
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    a_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    for bx0, by0, bx1, by1 in existing:
        b_area = max(1.0, (bx1 - bx0) * (by1 - by0))
        iw = min(ax1, bx1) - max(ax0, bx0)
        ih = min(ay1, by1) - max(ay0, by0)
        if iw > 0 and ih > 0:
            inter = iw * ih
            if inter / (a_area + b_area - inter) > thresh:
                return True
        # Also refuse to land inside another *small* object, which IoU alone would allow
        # when the two boxes differ a lot in size.
        if b_area < 4 * a_area and bx0 <= acx <= bx1 and by0 <= acy <= by1:
            return True
    return False


def _alpha_mask(patch: np.ndarray, blend: str, donor_bg) -> np.ndarray:
    """Per-pixel blend weight for a pasted crop.

    'auto' (default) keeps the pixels that differ from the ground the donor was sitting on
    and drops the ones that look like that ground, so only the object travels and the
    destination keeps its own background. The threshold is relative to the crop's own
    maximum contrast, which matters because these objects are 5-15px and their absolute
    contrast varies hugely between asphalt and concrete.

    'hard' pastes the raw rectangle -- fine when donor and destination ground match, but it
    leaves a visible square when they do not.

    'ellipse' fades the corners geometrically. Rejected as a default: at these sizes it
    rounded thin, angular objects into featureless blobs, erasing the very shape the
    detector needs to learn.
    """
    h, w = patch.shape[:2]
    if blend == "hard":
        return np.ones((h, w, 1), dtype=np.float32)
    if blend == "ellipse":
        yy, xx = np.mgrid[0:h, 0:w]
        ny = (yy + 0.5) / h * 2 - 1
        nx = (xx + 0.5) / w * 2 - 1
        r = np.sqrt(nx ** 2 + ny ** 2)
        return np.clip((1.15 - r) / 0.35, 0.0, 1.0).astype(np.float32)[:, :, None]

    if donor_bg is None:
        return np.ones((h, w, 1), dtype=np.float32)
    dist = np.linalg.norm(patch - donor_bg.reshape(1, 1, 3), axis=2)
    peak = float(dist.max())
    if peak < 1e-3:
        return np.ones((h, w, 1), dtype=np.float32)
    a = np.clip((dist / peak - 0.25) / 0.35, 0.0, 1.0)
    return a.astype(np.float32)[:, :, None]


def _paste(dst: np.ndarray, crop: np.ndarray, box, blend: str, donor_bg=None):
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    patch = np.asarray(
        Image.fromarray(crop).resize((w, h), Image.LANCZOS), dtype=np.float32)
    a = _alpha_mask(patch, blend, donor_bg)
    region = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = np.clip(patch * a + region * (1 - a), 0, 255).astype(np.uint8)


def build(scenes_dir: str, base_dir: str, out_dir: str, classes, per_image: int,
          blend: str, seed: int, keep_original: bool):
    rng = random.Random(seed)
    classes = set(classes)

    print(f"[gt_sample] scanning {scenes_dir} ...")
    banks = build_banks(scenes_dir, classes)
    n_donors = sum(len(v) for v in banks.donors.values())
    if not n_donors:
        raise SystemExit("no clean donor crops found -- check --classes against the scene JSON")
    print(f"[gt_sample] {n_donors} clean donor crops, "
          f"{sum(len(v) for v in banks.spots.values())} ground spots")

    img_out = os.path.join(out_dir, "images", "train")
    lbl_out = os.path.join(out_dir, "labels", "train")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # Donor pixels are read straight off disk repeatedly; cache the decoded PNGs since the
    # same few hundred scene images get sampled over and over.
    cache = {}

    def load(path):
        if path not in cache:
            cache[path] = np.asarray(Image.open(path).convert("RGB"))
        return cache[path]

    n_img = 0
    n_pasted = 0
    n_failed = 0
    per_class = defaultdict(int)

    for scene, cam_id, png_path, payload in _iter_scenes(scenes_dir):
        info = payload.get("image", {})
        img_w, img_h = info.get("width"), info.get("height")
        if not img_w or not img_h:
            continue

        lines = []
        existing = []
        no_stand = []
        for det in payload.get("detections", []):
            if det.get("class") not in CLASS_INDEX:
                continue
            x0, y0, x1, y1 = det["bbox"]
            existing.append((x0, y0, x1, y1))
            if det.get("class") in NO_STAND_CLASSES:
                no_stand.append((x0, y0, x1, y1))
            lines.append((CLASS_INDEX[det["class"]],
                          (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h,
                          (x1 - x0) / img_w, (y1 - y0) / img_h))

        orig_lines = list(lines)
        spots = banks.spots.get(cam_id)
        arr = load(png_path).copy()

        for _ in range(per_image):
            cand = [c for c in classes if banks.donors.get((cam_id, c))]
            if not cand or not spots:
                n_failed += 1
                continue
            cls = rng.choice(cand)

            placed = False
            for _try in range(MAX_PLACE_TRIES):
                cx, y_bottom = rng.choice(spots)
                # Jitter off the exact spot so pastes do not stack on the same few pixels.
                # Kept small on purpose: the spot pool is ground truth for "an object stood
                # here", and a wide jitter walks off it -- an early version with +-40px
                # dropped cups onto car bodies and other raised surfaces.
                cx = cx + rng.uniform(-12, 12)
                y_bottom = y_bottom + rng.uniform(-4, 4)
                size = banks.scale_at(cam_id, cls, y_bottom)
                if size is None:
                    continue
                size = float(np.clip(size * rng.uniform(0.85, 1.15),
                                     MIN_PASTE_PX, MAX_PASTE_PX))

                x0 = int(round(cx - size / 2))
                x1 = int(round(cx + size / 2))
                y1 = int(round(y_bottom))
                y0 = int(round(y_bottom - size))
                if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h or x1 - x0 < MIN_PASTE_PX \
                        or y1 - y0 < MIN_PASTE_PX:
                    continue
                # Test the GROUND CONTACT point, not the box: an object standing in front of
                # a car legitimately covers part of it, so rejecting on box overlap would
                # throw away valid placements. Uses the final integer box (not the pre-round
                # float) -- verifying against the rounded label is what the check has to
                # agree with, and an earlier float-only version left contacts sitting exactly
                # on a box edge after rounding.
                contact_x, contact_y = (x0 + x1) // 2, y1
                if any(bx0 <= contact_x <= bx1 and by0 <= contact_y <= by1
                       for bx0, by0, bx1, by1 in no_stand):
                    continue
                if _overlaps((x0, y0, x1, y1), existing):
                    continue

                # Prefer a donor at least as big as the slot so the crop is downscaled
                # rather than upscaled -- upscaling a 3px crop invents detail that is not
                # there and makes the pasted object blurrier than a real one.
                pool = banks.donors[(cam_id, cls)]
                want = [d for d in pool
                        if (d[1][2] - d[1][0] + d[1][3] - d[1][1]) / 2 >= size] or pool
                # ...then, among those, the ones whose original surroundings look like the
                # spot we are dropping into, so the crop's bbox corners blend in instead of
                # leaving a rectangle. Keep a few candidates rather than the single best so
                # the same donor does not get reused everywhere.
                dest_bg = _ring_bg(arr, (x0, y0, x1, y1))
                if dest_bg is not None:
                    dl = float(dest_bg.mean())
                    want = sorted(want, key=lambda d: abs(float(d[2].mean()) - dl))[
                        :max(3, len(want) // 8)]
                dpng, dbox, dbg = rng.choice(want)
                crop = load(dpng)[dbox[1]:dbox[3], dbox[0]:dbox[2]]
                if crop.size == 0:
                    continue

                _paste(arr, crop, (x0, y0, x1, y1), blend, dbg)
                existing.append((x0, y0, x1, y1))
                lines.append((CLASS_INDEX[cls],
                              (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h,
                              (x1 - x0) / img_w, (y1 - y0) / img_h))
                per_class[cls] += 1
                n_pasted += 1
                placed = True
                break
            if not placed:
                n_failed += 1

        stem = f"{scene}_{cam_id}"
        if keep_original:
            # Untouched pair keeps the plain stem; the pasted one gets its own so both
            # survive in the same split.
            shutil.copy(png_path, os.path.join(img_out, stem + ".png"))
            with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
                f.write("\n".join(_fmt(l) for l in orig_lines))
            stem = stem + "__gt"

        Image.fromarray(arr).save(os.path.join(img_out, stem + ".png"))
        with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
            f.write("\n".join(_fmt(l) for l in lines))
        n_img += 1

    # val is reused verbatim -- symlinked, never regenerated, so pasted pixels cannot leak
    # into evaluation and the split stays byte-identical to the base dataset.
    base_dir = os.path.abspath(base_dir)
    for sub in ("images", "labels"):
        src = os.path.join(base_dir, sub, "val")
        dst = os.path.join(out_dir, sub, "val")
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(src):
            link = os.path.join(dst, name)
            if not os.path.exists(link):
                os.symlink(os.path.join(src, name), link)

    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": os.path.abspath(out_dir),
                        "train": "images/train", "val": "images/val",
                        "nc": len(DET_CLASSES), "names": list(DET_CLASSES)},
                       f, sort_keys=False, allow_unicode=True)

    print(f"[gt_sample] wrote {n_img} train images -> {out_dir}")
    print(f"[gt_sample] pasted {n_pasted} instances "
          f"({', '.join(f'{k}={v}' for k, v in sorted(per_class.items()))})")
    if n_failed:
        print(f"[gt_sample] {n_failed} paste slots skipped (no free spot / no size model)")


def _fmt(l):
    return f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}"


def stats(scenes_dir: str, classes):
    """Print the donor bank and size models without writing a dataset."""
    banks = build_banks(scenes_dir, set(classes))
    print(f"{'camera':10s} {'class':16s} {'donors':>7s} {'size@row200':>12s} {'size@row600':>12s}")
    for cam in sorted(banks.spots):
        for cls in classes:
            n = len(banks.donors.get((cam, cls), []))
            s200 = banks.scale_at(cam, cls, 200)
            s600 = banks.scale_at(cam, cls, 600)
            print(f"{cam:10s} {cls:16s} {n:7d} "
                  f"{(f'{s200:.1f}px' if s200 else '-'):>12s} "
                  f"{(f'{s600:.1f}px' if s600 else '-'):>12s}")
    print(f"\nground spots per camera: "
          f"{ {c: len(v) for c, v in sorted(banks.spots.items())} }")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = {"--classes": dict(default=",".join(DEFAULT_CLASSES),
                                help="comma-separated classes to paste")}

    s = sub.add_parser("stats", help="inspect donor bank / size models only")
    s.add_argument("--scenes", required=True)
    for flag, kw in common.items():
        s.add_argument(flag, **kw)

    b = sub.add_parser("build", help="write a GT-sampled copy of the train split")
    b.add_argument("--scenes", required=True, help="dataset-gen output dir (train scenes)")
    b.add_argument("--base", required=True, help="existing dataset dir to take val from")
    b.add_argument("--out", required=True)
    b.add_argument("--per-image", type=int, default=3,
                   help="target instances to paste per train image (default 3)")
    b.add_argument("--blend", choices=("auto", "hard", "ellipse"), default="auto")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--keep-original", action="store_true",
                   help="also keep the un-pasted image (doubles the split; off by default so "
                        "compute stays matched to the base dataset)")
    for flag, kw in common.items():
        b.add_argument(flag, **kw)

    a = ap.parse_args()
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    unknown = [c for c in classes if c not in CLASS_INDEX]
    if unknown:
        raise SystemExit(f"unknown class(es): {unknown} -- must be in det_classes.DET_CLASSES")

    if a.cmd == "stats":
        stats(a.scenes, classes)
    else:
        build(a.scenes, a.base, a.out, classes, a.per_image, a.blend, a.seed,
              a.keep_original)


if __name__ == "__main__":
    main()
