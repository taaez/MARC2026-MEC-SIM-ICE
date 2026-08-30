"""Shared pieces of the small-object sub-model: tiling geometry, box ops, run config.

Everything here exists to keep the TRAINING tiler and the INFERENCE tiler byte-identical.
If a model is trained on 640 px tiles at 25% overlap upscaled 2x, and inference then slices
at 512/0.2, the model is being asked about a scale it never saw -- and the failure is silent,
showing up only as unexplained recall loss. So the grid math lives in exactly one function
(`tile_grid`), and the parameters that produced a dataset are written next to it as
`tiling.json` (`TilingConfig`) for the trainer and the predictor to read back rather than
having them re-specified by hand at each step.

Nothing in this module imports ultralytics or torch -- it stays importable (and testable)
on a machine with no weights and no CUDA.
"""

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

# The three classes the sub-model owns, in the order they are indexed in the tiled dataset.
# The deployed 34-class model B keeps predicting these too; the routing step in predict.py
# discards B's copies and takes the sub-model's instead. Order matters -- it fixes the label
# ids in the generated dataset, so changing it invalidates an existing dataset/weights pair.
SMALL_CLASSES = ["sunblock", "tissue", "disposable_cup"]

CONFIG_NAME = "tiling.json"


@dataclass
class TilingConfig:
    """How a tiled dataset was cut, carried alongside it so training and inference agree.

    `train_imgsz` is the one number that actually buys anything: a tile is fed to the model
    at this size, so every object is scaled by `train_imgsz / tile` before the network sees
    it. At 1.0 the tiling is pure bookkeeping and gains nothing (see README).
    """

    tile: int = 640
    overlap: float = 0.25
    train_imgsz: int = 1280
    classes: list = field(default_factory=lambda: list(SMALL_CLASSES))
    source: str = ""

    @property
    def stride(self) -> int:
        return max(1, int(round(self.tile * (1.0 - self.overlap))))

    @property
    def upscale(self) -> float:
        return self.train_imgsz / self.tile

    def save(self, dataset_dir: str):
        with open(os.path.join(dataset_dir, CONFIG_NAME), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "TilingConfig":
        """Accepts either the json itself or the dataset dir holding it."""
        if os.path.isdir(path):
            path = os.path.join(path, CONFIG_NAME)
        if not os.path.exists(path):
            raise SystemExit(
                f"no {CONFIG_NAME} at {path} -- it is written by make_tiled_dataset.py next to "
                f"the dataset it produces. Inference must reuse the training tiling; passing "
                f"the numbers again by hand is exactly the mismatch this file prevents.")
        with open(path) as f:
            return TilingConfig(**json.load(f))


def tile_starts(total: int, tile: int, stride: int):
    """Tile origins along one axis, always yielding full-size tiles.

    The last origin snaps back to `total - tile` instead of running off the edge, so every
    tile is exactly `tile` px and nothing is ever padded. That snap can make the final step
    shorter than `stride` -- on a 720 px axis with tile=640 the origins are [0, 80] -- which
    is fine: extra overlap costs a little compute, a partial tile costs correctness.
    """
    if total <= tile:
        return [0]
    xs = list(range(0, total - tile + 1, stride))
    if xs[-1] != total - tile:
        xs.append(total - tile)
    return xs


def tile_grid(width: int, height: int, tile: int, stride: int):
    """(ix, iy, x0, y0) for every tile. THE single definition of the grid -- both the
    dataset builder and the predictor call this, so they cannot drift apart."""
    return [(ix, iy, x0, y0)
            for iy, y0 in enumerate(tile_starts(height, tile, stride))
            for ix, x0 in enumerate(tile_starts(width, tile, stride))]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b)).

    Note for this project specifically: at 5-8 px, IoU is a brutal and twitchy metric --
    a two-pixel offset on a 6 px box drops IoU below 0.5 even though the detection is
    plainly correct. That is the whole reason the sub-model exists, and it is also why
    evaluate.py offers center-distance matching as well as IoU matching.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax0, ay0, ax1, ay1 = (a[:, i][:, None] for i in range(4))
    bx0, by0, bx1, by1 = (b[:, i][None, :] for i in range(4))
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    area_a = np.clip(ax1 - ax0, 0, None) * np.clip(ay1 - ay0, 0, None)
    area_b = np.clip(bx1 - bx0, 0, None) * np.clip(by1 - by0, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float = 0.5):
    """Class-wise NMS. Returns the indices to keep, in descending score order.

    Used to collapse the duplicate detections that overlapping tiles necessarily produce:
    an object in the shared strip between two tiles gets predicted twice. Highest-confidence
    box wins rather than an average, because a box may be truncated by a tile edge in one
    tile and whole in the other -- averaging their coordinates would shrink the survivor
    toward the truncated one.
    """
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        idx = idx[np.argsort(-scores[idx])]
        while len(idx):
            i, idx = idx[0], idx[1:]
            keep.append(i)
            if not len(idx):
                break
            ious = iou_matrix(boxes[i][None, :], boxes[idx])[0]
            idx = idx[ious < iou_thr]
    keep = np.array(keep, dtype=int)
    return keep[np.argsort(-scores[keep])] if len(keep) else keep


def wbf(box_sets, score_sets, class_sets, weights=None, iou_thr: float = 0.55,
        n_models: int = None):
    """Weighted Boxes Fusion (Solovyev et al.) over predictions from several models.

    Unlike NMS, agreeing boxes are averaged (coordinates weighted by confidence) and the
    fused score reflects how many models agreed -- which is the point of using it between
    two detectors rather than just concatenating.

    Only used for the optional second stage inside the three small classes, where both the
    sub-model and B have an opinion. The default routing path does NOT call this: each class
    is owned by exactly one model, so there is nothing to fuse and nothing to re-calibrate.
    Be aware the score rescale below can push a box under a deployment conf cut that a raw
    prediction would have cleared -- the trap called out in the plan doc.
    """
    boxes = np.concatenate([b for b in box_sets if len(b)]) if any(len(b) for b in box_sets) \
        else np.zeros((0, 4), dtype=np.float32)
    if not len(boxes):
        return boxes, np.zeros(0, np.float32), np.zeros(0, int)
    if weights is None:
        weights = [1.0] * len(box_sets)
    scores = np.concatenate([s * w for s, w in zip(score_sets, weights) if len(s)])
    classes = np.concatenate([c for c in class_sets if len(c)])
    src = np.concatenate([np.full(len(b), i) for i, b in enumerate(box_sets) if len(b)])
    n_models = n_models or len(box_sets)

    out_b, out_s, out_c = [], [], []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        idx = idx[np.argsort(-scores[idx])]
        clusters = []  # each: [list of member indices]
        for i in idx:
            placed = False
            for cl in clusters:
                # Match against the cluster's current fused box, not its seed, so a cluster
                # that has drifted while absorbing members keeps attracting the right boxes.
                fused = np.average(boxes[cl], axis=0, weights=scores[cl])
                if iou_matrix(fused[None, :], boxes[i][None, :])[0, 0] > iou_thr:
                    cl.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
        for cl in clusters:
            w = scores[cl]
            out_b.append(np.average(boxes[cl], axis=0, weights=w))
            # Standard WBF rescale: a box only one model saw is damped by how many models
            # could have seen it, so unanimous detections outrank lone ones.
            out_s.append(float(w.mean() * min(len(np.unique(src[cl])), n_models) / n_models))
            out_c.append(int(c))
    order = np.argsort(-np.array(out_s))
    return (np.array(out_b, dtype=np.float32)[order],
            np.array(out_s, dtype=np.float32)[order],
            np.array(out_c, dtype=int)[order])


def read_yolo_labels(path: str, width: int, height: int):
    """YOLO label file -> (xyxy boxes in pixels, class ids). A missing file means a
    background image, which is legitimate input, not an error."""
    boxes, classes = [], []
    if not os.path.exists(path):
        return np.zeros((0, 4), np.float32), np.zeros(0, int)
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            cid, cx, cy, w, h = int(p[0]), *(float(v) for v in p[1:])
            bw, bh = w * width, h * height
            boxes.append([cx * width - bw / 2, cy * height - bh / 2,
                          cx * width + bw / 2, cy * height + bh / 2])
            classes.append(cid)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(classes, dtype=int))


def longest_side(boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0, np.float32)
    return np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
