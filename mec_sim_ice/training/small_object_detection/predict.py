"""Tiled inference for the sub-model + class routing with model B. Step 3 of the pipeline.

Two pieces, usable as a library (evaluate.py imports both) or from the command line:

  TiledPredictor  -- runs the 3-class sub-model over overlapping tiles of a full-size image
                     and merges the results back into original-image coordinates (the SAHI
                     scheme). Reads tiling.json so the slicing matches training exactly.
  RoutedDetector  -- runs model B for its 31 classes and the sub-model for its 3, and
                     concatenates. B is loaded and used unchanged; nothing retrains it.

    python3 predict.py --images ../dataset_v2/images/val \
        --general ../runs/detect/train_B_yolo11s/weights/best.pt \
        --small   ../runs/detect/small_t640/weights/best.pt \
        --tiling  dataset_small_t640 --out preds.json

WHY ROUTING RATHER THAN FUSION
------------------------------
The two models' outputs barely overlap: B owns 31 classes, the sub-model owns 3. Giving each
class exactly one owner means there is nothing to fuse, and -- more importantly -- no need to
reconcile two confidence scales. That matters because the deployment pipeline applies a hard
conf=0.35 cut: any fusion that averages scores can push a correct box under the cut and lose
points that the raw prediction would have scored. So routing is the default and WBF (--fuse)
is opt-in, applied only inside the 3 small classes where both models have an opinion.

The sub-model gets its OWN conf threshold (--small-conf). It saw only 3 tiny classes on
square crops, so its score distribution has no reason to line up with B's; reusing B's 0.35
would be an assumption, not a default.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from det_classes import DET_CLASSES  # noqa: E402

from common import TilingConfig, longest_side, nms, tile_grid, wbf  # noqa: E402


class TiledPredictor:
    """The 3-class sub-model, applied to a full-size image through overlapping tiles."""

    def __init__(self, weights, tiling, conf=0.25, iou=0.5, max_box_px=64.0,
                 edge_margin=2.0, batch=8, device=None, full_pass=False, model=None):
        # `model` accepts an already-loaded detector instead of a path -- used by selftest.py
        # to exercise the tiling/merging logic against scripted detections, and available if
        # a caller wants to share one loaded model across several predictors.
        if model is None:
            from ultralytics import YOLO  # lazy, so common/evaluate import without torch
            model = YOLO(weights)
        self.model = model
        self.cfg = tiling if isinstance(tiling, TilingConfig) else TilingConfig.load(tiling)
        self.conf, self.iou = conf, iou
        self.max_box_px, self.edge_margin = max_box_px, edge_margin
        self.batch, self.device, self.full_pass = batch, device, full_pass
        self.names = list(self.cfg.classes)

    def _run(self, images):
        out = []
        for i in range(0, len(images), self.batch):
            chunk = images[i:i + self.batch]
            out += self.model.predict(chunk, imgsz=self.cfg.train_imgsz, conf=self.conf,
                                      iou=self.iou, device=self.device, verbose=False)
        return out

    def predict(self, image: np.ndarray):
        """image: HxWx3 BGR. Returns (xyxy boxes in image coords, scores, class ids 0..2)."""
        H, W = image.shape[:2]
        grid = tile_grid(W, H, self.cfg.tile, self.cfg.stride)
        # ascontiguousarray, not a bare slice: a crop out of the middle of a frame is a
        # non-contiguous view, and the preprocessing path (cv2 resize / torch.from_numpy)
        # is only reliably happy with contiguous buffers. It copies anyway downstream, so
        # this costs nothing and removes a class of environment-dependent breakage.
        tiles = [np.ascontiguousarray(image[y0:y0 + self.cfg.tile, x0:x0 + self.cfg.tile])
                 for _, _, x0, y0 in grid]

        boxes, scores, classes = [], [], []
        for (_, _, x0, y0), res in zip(grid, self._run(tiles)):
            b = res.boxes
            if b is None or not len(b):
                continue
            xyxy = b.xyxy.cpu().numpy()
            keep = np.ones(len(xyxy), dtype=bool)

            # Drop detections hugging an INNER tile edge -- an object straddling the seam is
            # cut here but whole in the neighbouring tile, and its truncated copy can easily
            # outscore the complete one, which plain NMS would then keep. Edges that are also
            # image borders are left alone: nothing else covers those.
            if self.edge_margin > 0:
                m = self.edge_margin
                if x0 > 0:
                    keep &= xyxy[:, 0] > m
                if y0 > 0:
                    keep &= xyxy[:, 1] > m
                if x0 + self.cfg.tile < W:
                    keep &= xyxy[:, 2] < self.cfg.tile - m
                if y0 + self.cfg.tile < H:
                    keep &= xyxy[:, 3] < self.cfg.tile - m

            xyxy = xyxy[keep]
            if not len(xyxy):
                continue
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0
            boxes.append(xyxy)
            scores.append(b.conf.cpu().numpy()[keep])
            classes.append(b.cls.cpu().numpy()[keep].astype(int))

        # Optional whole-image pass at the same imgsz. Off by default: this sub-model was
        # trained only on upscaled tiles, so a downscaled full frame is out of distribution
        # for it. Worth enabling only if tiling turns out to miss objects at tile seams.
        if self.full_pass:
            for res in self._run([image]):
                b = res.boxes
                if b is not None and len(b):
                    boxes.append(b.xyxy.cpu().numpy())
                    scores.append(b.conf.cpu().numpy())
                    classes.append(b.cls.cpu().numpy().astype(int))

        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        boxes = np.concatenate(boxes).astype(np.float32)
        scores = np.concatenate(scores).astype(np.float32)
        classes = np.concatenate(classes).astype(int)

        # Size gate: these three classes top out around 40 px per side in this scenario, so
        # anything much larger is the sub-model mistaking a fragment of a big object for one
        # of its own. Cheap, and it only ever removes predictions B is already covering.
        if self.max_box_px:
            k = longest_side(boxes) <= self.max_box_px
            boxes, scores, classes = boxes[k], scores[k], classes[k]

        keep = nms(boxes, scores, classes, self.iou)
        return boxes[keep], scores[keep], classes[keep]


class RoutedDetector:
    """Model B for its 31 classes + the sub-model for its 3. Output ids are DET_CLASSES ids."""

    def __init__(self, general_weights, small: TiledPredictor, conf=0.35, iou=0.7,
                 imgsz=1280, device=None, fuse=False, fuse_iou=0.55):
        from ultralytics import YOLO

        self.general = YOLO(general_weights)
        self.small = small
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.fuse, self.fuse_iou = fuse, fuse_iou
        # Map the sub-model's 0..2 onto the shared 34-class taxonomy so downstream code
        # (score.py, record_agent_vla.py) never learns there were two models.
        self.small_to_det = np.array([DET_CLASSES.index(n) for n in small.names], dtype=int)
        self.owned = set(self.small_to_det.tolist())

    def predict(self, image: np.ndarray):
        res = self.general.predict(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                   device=self.device, verbose=False)[0]
        b = res.boxes
        if b is not None and len(b):
            g_box = b.xyxy.cpu().numpy().astype(np.float32)
            g_score = b.conf.cpu().numpy().astype(np.float32)
            g_cls = b.cls.cpu().numpy().astype(int)
        else:
            g_box = np.zeros((0, 4), np.float32)
            g_score, g_cls = np.zeros(0, np.float32), np.zeros(0, int)

        s_box, s_score, s_cls = self.small.predict(image)
        s_cls = self.small_to_det[s_cls] if len(s_cls) else s_cls.astype(int)

        owned = np.isin(g_cls, list(self.owned)) if len(g_cls) else np.zeros(0, bool)
        if not self.fuse:
            # Routing: B's opinion on the three small classes is discarded outright.
            k = ~owned
            return (np.concatenate([g_box[k], s_box]),
                    np.concatenate([g_score[k], s_score]),
                    np.concatenate([g_cls[k], s_cls]))

        # Optional stage 2: fuse inside the small classes only. B still detects them at
        # R ~= 0.4, so agreement is real evidence -- but WBF's score rescale can drop a box
        # below the deployment conf cut, so this is opt-in and must be re-checked at 0.35.
        fb, fs, fc = wbf([s_box, g_box[owned]], [s_score, g_score[owned]],
                         [s_cls, g_cls[owned]], iou_thr=self.fuse_iou, n_models=2)
        k = ~owned
        return (np.concatenate([g_box[k], fb]),
                np.concatenate([g_score[k], fs]),
                np.concatenate([g_cls[k], fc]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory of full-size images")
    ap.add_argument("--general", required=True, help="model B weights (34-class best.pt)")
    ap.add_argument("--small", required=True, help="sub-model weights (3-class best.pt)")
    ap.add_argument("--tiling", required=True, help="tiled dataset dir or tiling.json")
    ap.add_argument("--out", default="preds.json")
    ap.add_argument("--conf", type=float, default=0.35, help="model B conf (deployment value)")
    ap.add_argument("--small-conf", type=float, default=0.25,
                    help="sub-model conf -- tuned separately, its scores are not on B's scale")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU for tile merging")
    ap.add_argument("--max-box-px", type=float, default=64.0, help="size gate; 0 disables")
    ap.add_argument("--edge-margin", type=float, default=2.0, help="inner-tile-edge reject band")
    ap.add_argument("--tile-batch", type=int, default=8, help="tiles per forward pass")
    ap.add_argument("--imgsz", type=int, default=1280, help="model B imgsz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fuse", action="store_true", help="WBF inside the 3 small classes")
    args = ap.parse_args()

    import cv2

    small = TiledPredictor(args.small, args.tiling, conf=args.small_conf, iou=args.iou,
                           max_box_px=args.max_box_px, edge_margin=args.edge_margin,
                           batch=args.tile_batch, device=args.device)
    det = RoutedDetector(args.general, small, conf=args.conf, iou=0.7, imgsz=args.imgsz,
                         device=args.device, fuse=args.fuse)

    files = sorted(f for f in os.listdir(args.images)
                   if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg"))
    out = []
    for i, f in enumerate(files, 1):
        img = cv2.imread(os.path.join(args.images, f))
        if img is None:
            continue
        boxes, scores, classes = det.predict(img)
        out.append({"file": f, "detections": [
            {"class": DET_CLASSES[c], "conf": round(float(s), 4),
             "bbox": [round(float(v), 2) for v in b]}
            for b, s, c in zip(boxes, scores, classes)]})
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} images -> {args.out}")


if __name__ == "__main__":
    main()
