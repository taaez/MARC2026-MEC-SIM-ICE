"""Score model B alone vs B + sub-model, on FULL images. Step 4 -- the deciding number.

    python3 evaluate.py --base ../dataset_v2 --split val \
        --general ../runs/detect/train_B_yolo11s/weights/best.pt \
        --small   ../runs/detect/small_t640/weights/best.pt \
        --tiling  dataset_small_t640

Everything else in this folder reports numbers that are easy to improve and easy to fool.
The tiled val split scores the sub-model on the distribution it trained on. Ultralytics' own
mAP is dominated by classes that were never the problem. This script exists to answer the one
question that decides whether the sub-model ships:

    at the deployment conf cut, on original 1280x720 frames, does routing recover
    sunblock / tissue / disposable_cup without disturbing anything else?

Three deliberate choices, each of which has burned this project before:

  1. BOTH configurations come out of ONE inference pass, run at a low conf floor, with the
     thresholds applied afterwards in the metric code. So the B-alone baseline is measured,
     not quoted from an old log, and the sub-model's threshold can be swept without re-running
     the network.
  2. Results are broken down BY GT SIZE BUCKET. Whole-class AP hides the failure: the misses
     are concentrated in the far-field tiny instances (the 5.04 m wall rigs), and a metric
     averaged over every instance moves too little to read.
  3. Matching is available by CENTER DISTANCE as well as IoU. At 5-8 px, IoU@0.5 is a
     coin-flip on detections that are visibly correct -- a 2 px offset on a 6 px box fails it.
     IoU stays the default for comparability with the training logs; center matching is what
     to look at when deciding whether a detection would actually have grounded the object.
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from det_classes import DET_CLASSES  # noqa: E402

from common import (TilingConfig, iou_matrix, longest_side,  # noqa: E402
                    read_yolo_labels)
from predict import TiledPredictor  # noqa: E402

BUCKETS = [(0, 8), (8, 16), (16, 32), (32, 10 ** 9)]
BUCKET_NAMES = ["<8px", "8-16px", "16-32px", ">=32px"]


def _match(pred_boxes, pred_scores, gt_boxes, iou_thr, mode, center_px):
    """Greedy score-ordered matching -> (tp flags aligned to score-sorted preds, those
    sorted scores, matched-GT mask).

    Matching once and thresholding afterwards is valid precisely because the greedy order is
    by score: raising the conf cut only truncates the tail, it never changes who matched whom.
    That is what lets one inference pass produce a whole threshold sweep.
    """
    order = np.argsort(-pred_scores)
    pb, ps = pred_boxes[order], pred_scores[order]
    taken = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pb), dtype=bool)

    if len(gt_boxes) and len(pb):
        if mode == "iou":
            score_mat = iou_matrix(pb, gt_boxes)
            ok = score_mat >= iou_thr
        else:
            # Center distance, tolerant to the object's own size: a 6 px object matched
            # within `center_px` is the same object by any sane reading.
            pc = np.stack([(pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2], 1)
            gc = np.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                           (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2], 1)
            dist = np.linalg.norm(pc[:, None, :] - gc[None, :, :], axis=2)
            tol = np.maximum(center_px, longest_side(gt_boxes))[None, :]
            score_mat, ok = -dist, dist <= tol
        for i in range(len(pb)):
            cand = np.where(ok[i] & ~taken)[0]
            if not len(cand):
                continue
            j = cand[np.argmax(score_mat[i, cand])]
            taken[j] = True
            tp[i] = True
    return tp, ps, taken


def _ap(tp, n_gt):
    """All-point-interpolated AP from score-ordered TP flags."""
    if n_gt == 0 or not len(tp):
        return 0.0
    ctp = np.cumsum(tp)
    rec = ctp / n_gt
    prec = ctp / np.arange(1, len(tp) + 1)
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _bucket(size):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= size < hi:
            return i
    return len(BUCKETS) - 1


def classify_small_box(box, cid, gt_boxes, gt_cls, b_boxes, b_cls,
                       match="iou", iou_thr=0.5, center_px=8.0):
    """One shipped sub-model box -> (is_tp, max_iou_with_a_b_box, that_b_class_or_None).

    `is_tp`: does it match a GT of its own class. `max_iou`/`b_hit`: its strongest overlap
    with the B boxes that survive routing (caller passes only non-owned, above-cut B boxes).
    Pulled out of the main loop so selftest can pin the overlap accounting -- the whole
    reason this report exists is to decide a veto, and a miscounted overlap would send that
    decision the wrong way silently.
    """
    g = gt_boxes[gt_cls == cid] if len(gt_cls) else np.zeros((0, 4), np.float32)
    if not len(g):
        is_tp = False
    elif match == "iou":
        is_tp = bool((iou_matrix(box[None, :], g)[0] >= iou_thr).any())
    else:
        pc = (box[:2] + box[2:]) / 2
        gc = (g[:, :2] + g[:, 2:]) / 2
        tol = np.maximum(center_px, longest_side(g))
        is_tp = bool((np.linalg.norm(gc - pc, axis=1) <= tol).any())

    max_iou, b_hit = 0.0, None
    if len(b_boxes):
        ious = iou_matrix(box[None, :], b_boxes)[0]
        j = int(np.argmax(ious))
        max_iou, b_hit = float(ious[j]), int(b_cls[j])
    return is_tp, max_iou, b_hit


def _report(title, per_class, conf, small_names, small_conf):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'class':<18}{'GT':>6}{'TP':>6}{'FP':>6}{'P':>8}{'R':>8}{'F1':>8}{'AP50':>8}")
    tot = defaultdict(float)
    for name in per_class:
        d = per_class[name]
        t = small_conf if name in small_names else conf
        keep = d["scores"] >= t
        tp = int(d["tp"][keep].sum())
        fp = int(keep.sum()) - tp
        n_gt = d["n_gt"]
        p = tp / max(tp + fp, 1)
        r = tp / max(n_gt, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"{name:<18}{n_gt:>6}{tp:>6}{fp:>6}{p:>8.3f}{r:>8.3f}{f1:>8.3f}"
              f"{_ap(d['tp'], n_gt):>8.3f}")
        tot["gt"] += n_gt
        tot["tp"] += tp
        tot["fp"] += fp
    p = tot["tp"] / max(tot["tp"] + tot["fp"], 1)
    r = tot["tp"] / max(tot["gt"], 1)
    print(f"{'-' * 78}\n{'TOTAL':<18}{int(tot['gt']):>6}{int(tot['tp']):>6}"
          f"{int(tot['fp']):>6}{p:>8.3f}{r:>8.3f}{2 * p * r / max(p + r, 1e-9):>8.3f}")


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--base", required=True, help="original YOLO dataset (e.g. ../dataset_v2)")
    ap_.add_argument("--split", default="val")
    ap_.add_argument("--general", required=True, help="model B weights")
    ap_.add_argument("--small", default=None,
                     help="sub-model weights. Omit to measure the B-alone baseline only.")
    ap_.add_argument("--tiling", default=None, help="tiled dataset dir or tiling.json")
    ap_.add_argument("--conf", type=float, default=0.35, help="deployment conf cut for B")
    ap_.add_argument("--small-conf", type=float, default=0.25, help="conf cut for the sub-model")
    ap_.add_argument("--conf-floor", type=float, default=0.05,
                     help="inference floor; thresholds above are applied in the metric code")
    ap_.add_argument("--match", choices=["iou", "center"], default="iou")
    ap_.add_argument("--iou-thr", type=float, default=0.5)
    ap_.add_argument("--center-px", type=float, default=8.0,
                     help="center-match tolerance in px (or the GT's own size, whichever is larger)")
    ap_.add_argument("--max-box-px", type=float, default=64.0)
    ap_.add_argument("--imgsz", type=int, default=1280, help="model B imgsz")
    ap_.add_argument("--tile-batch", type=int, default=8)
    ap_.add_argument("--device", default=None)
    ap_.add_argument("--limit", type=int, default=0, help="only the first N images (smoke runs)")
    args = ap_.parse_args()

    import cv2
    from ultralytics import YOLO

    img_dir = os.path.join(args.base, "images", args.split)
    lbl_dir = os.path.join(args.base, "labels", args.split)
    files = sorted(f for f in os.listdir(img_dir)
                   if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg"))
    if args.limit:
        files = files[:args.limit]

    general = YOLO(args.general)
    small = None
    small_names = []
    if args.small:
        if not args.tiling:
            raise SystemExit("--small needs --tiling (the dataset dir holding tiling.json)")
        cfg = TilingConfig.load(args.tiling)
        small_names = list(cfg.classes)
        small = TiledPredictor(args.small, cfg, conf=args.conf_floor, iou=0.5,
                               max_box_px=args.max_box_px, batch=args.tile_batch,
                               device=args.device)
        print(f"sub-model: tile={cfg.tile} overlap={cfg.overlap:.0%} "
              f"imgsz={cfg.train_imgsz} upscale={cfg.upscale:.2f}x classes={small_names}")

    # gathered[config][class] = {"boxes","scores"} accumulated per image, matched per image
    acc = {k: defaultdict(lambda: {"tp": [], "scores": [], "n_gt": 0,
                                   "buckets": np.zeros((len(BUCKETS), 2))})
           for k in ("baseline", "routed")}

    # One row per shipped sub-model box: (is_tp, max_iou_with_a_shipped_B_box, that_B_class).
    # Routing keeps every sub-model box regardless of what B saw at the same spot; this is
    # how we find out whether that is actually costing anything -- a run of FP boxes sitting
    # on top of confident B detections would argue for a one-way veto, whereas TP boxes that
    # overlap B (a cup inside a picnic_table bbox) say a veto would delete real detections.
    overlap_rows = []

    for n, fname in enumerate(files, 1):
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt_boxes, gt_cls = read_yolo_labels(
            os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt"), W, H)

        res = general.predict(img, imgsz=args.imgsz, conf=args.conf_floor, iou=0.7,
                              device=args.device, verbose=False)[0]
        b = res.boxes
        if b is not None and len(b):
            gb = b.xyxy.cpu().numpy().astype(np.float32)
            gs = b.conf.cpu().numpy().astype(np.float32)
            gc = b.cls.cpu().numpy().astype(int)
        else:
            gb, gs, gc = np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)

        preds = {"baseline": (gb, gs, gc)}
        if small is not None:
            sb, ss, sc = small.predict(img)
            sc = np.array([DET_CLASSES.index(small_names[i]) for i in sc], dtype=int) \
                if len(sc) else sc.astype(int)
            owned = np.isin(gc, [DET_CLASSES.index(x) for x in small_names]) if len(gc) \
                else np.zeros(0, bool)
            k = ~owned
            preds["routed"] = (np.concatenate([gb[k], sb]), np.concatenate([gs[k], ss]),
                               np.concatenate([gc[k], sc]))

            # Overlap bookkeeping, at the thresholds that actually ship: sub-model boxes
            # above --small-conf vs the B boxes that survive routing (non-owned) and the
            # B conf cut.
            keep_s = ss >= args.small_conf
            b_ship = ((~owned) & (gs >= args.conf)) if len(gc) else np.zeros(0, bool)
            gb_s, gc_s = gb[b_ship], gc[b_ship]
            for si in np.where(keep_s)[0]:
                overlap_rows.append(classify_small_box(
                    sb[si], int(sc[si]), gt_boxes, gt_cls, gb_s, gc_s,
                    args.match, args.iou_thr, args.center_px))

        for cfg_name, (pb, ps, pc) in preds.items():
            for cid in np.unique(np.concatenate([gt_cls, pc]) if len(gt_cls) or len(pc)
                                 else np.zeros(0, int)):
                name = DET_CLASSES[cid]
                g = gt_boxes[gt_cls == cid]
                m = pc == cid
                tp, sc_sorted, taken = _match(
                    pb[m], ps[m], g, args.iou_thr, args.match, args.center_px)
                d = acc[cfg_name][name]
                d["tp"].append(tp)
                d["scores"].append(sc_sorted)
                d["n_gt"] += len(g)
                # Recall by GT size needs the GT side, not the prediction side: count how
                # many GT boxes in each bucket existed and how many got taken.
                for j, sz in enumerate(longest_side(g)):
                    bi = _bucket(sz)
                    d["buckets"][bi, 0] += 1
                    d["buckets"][bi, 1] += bool(taken[j])
        if n % 25 == 0:
            print(f"  {n}/{len(files)}")

    for cfg_name in acc:
        for name, d in acc[cfg_name].items():
            d["tp"] = np.concatenate(d["tp"]) if d["tp"] else np.zeros(0, bool)
            d["scores"] = np.concatenate(d["scores"]) if len(d["scores"]) else np.zeros(0, np.float32)
            order = np.argsort(-d["scores"])
            d["tp"], d["scores"] = d["tp"][order], d["scores"][order]

    tag = f"match={args.match}" + (f"@IoU{args.iou_thr}" if args.match == "iou"
                                   else f"@{args.center_px}px")
    _report(f"BASELINE  model B alone  (conf={args.conf}, {tag})",
            acc["baseline"], args.conf, [], args.conf)
    if small is not None:
        _report(f"ROUTED  B(31) + sub-model(3)  (conf={args.conf}, "
                f"small_conf={args.small_conf}, {tag})",
                acc["routed"], args.conf, small_names, args.small_conf)

        print(f"\n{'=' * 78}\nSMALL CLASSES BY GT SIZE  (recall, {tag})\n{'=' * 78}")
        print(f"{'class':<18}{'config':<10}" + "".join(f"{b:>12}" for b in BUCKET_NAMES))
        for name in small_names:
            for cfg_name in ("baseline", "routed"):
                bk = acc[cfg_name][name]["buckets"]
                cells = "".join(
                    f"{(bk[i, 1] / bk[i, 0] if bk[i, 0] else 0):>7.3f}({int(bk[i, 0]):>3})"
                    for i in range(len(BUCKETS)))
                print(f"{name if cfg_name == 'baseline' else '':<18}{cfg_name:<10}{cells}")

        print(f"\n{'=' * 78}\nSUB-MODEL CONF SWEEP  (its scores are not on B's scale)\n{'=' * 78}")
        print(f"{'conf':<8}" + "".join(f"{n:>26}" for n in small_names))
        print(f"{'':<8}" + "".join(f"{'P / R / F1':>26}" for _ in small_names))
        for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
            row = f"{t:<8.2f}"
            for name in small_names:
                d = acc["routed"][name]
                keep = d["scores"] >= t
                tp = int(d["tp"][keep].sum())
                fp = int(keep.sum()) - tp
                p = tp / max(tp + fp, 1)
                r = tp / max(d["n_gt"], 1)
                row += f"{p:>8.3f}{r:>9.3f}{2 * p * r / max(p + r, 1e-9):>9.3f}"
            print(row)

        print(f"\n{'=' * 78}")
        print(f"SUB-MODEL vs B SPATIAL OVERLAP  "
              f"(routed; small_conf>={args.small_conf}, B conf>={args.conf})")
        print(f"{'=' * 78}")
        n_ship = len(overlap_rows)
        n_tp = sum(r[0] for r in overlap_rows)
        print(f"  shipped sub-model boxes: {n_ship}  ({n_tp} TP / {n_ship - n_tp} FP)")
        for thr in (0.1, 0.3, 0.5):
            over = [r for r in overlap_rows if r[1] >= thr]
            otp = sum(r[0] for r in over)
            pct = 100 * len(over) / max(n_ship, 1)
            print(f"    overlap a shipped B box @ IoU>={thr}: {len(over):>4} ({pct:4.1f}%)  "
                  f"-> {otp} TP, {len(over) - otp} FP")
        cc = Counter(DET_CLASSES[r[2]] for r in overlap_rows
                     if r[2] is not None and r[1] >= 0.3)
        if cc:
            print("    B classes overlapped @ IoU>=0.3: "
                  + ", ".join(f"{k}={v}" for k, v in cc.most_common(10)))
        print("  read: mostly FP + overlapping a confident B box -> a one-way veto against B "
              "would remove them cheaply.")
        print("        mostly TP + overlapping (cup inside a picnic_table bbox) -> a veto "
              "would delete real detections; keep plain concat.")

    print("\nreminder: the 31 non-small classes come straight from B and cannot regress -- "
          "if they differ between the two tables above, the routing code is wrong, not the model.")


if __name__ == "__main__":
    main()
