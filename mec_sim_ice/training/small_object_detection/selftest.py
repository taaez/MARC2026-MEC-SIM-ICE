"""Self-checks for the parts of this pipeline that fail silently. Run: python3 selftest.py

No weights, no dataset, no GPU needed -- the detector is scripted, so the assertions are
about OUR arithmetic rather than a model's accuracy. The bugs this catches are the ones that
don't announce themselves: a tile offset applied in the wrong direction still produces
plausible boxes, a size gate with an off-by-one still returns detections, and a routing step
that drops the wrong class still returns a full-looking result. Every one of those would
show up only as unexplained recall loss weeks later.

Covered: grid coverage and determinism, tile-local -> image coordinate lifting, inner-edge
rejection, size gating, cross-tile NMS, WBF agreement/score behaviour, greedy matching and
AP, and label round-tripping.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (TilingConfig, iou_matrix, longest_side, nms,  # noqa: E402
                    read_yolo_labels, tile_grid, tile_starts, wbf)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# --------------------------------------------------------------------------- grid geometry
def test_grid():
    print("\ntile geometry")
    # Every pixel of a 1280x720 frame must be inside at least one 640 tile: a gap is a blind
    # spot that no amount of model quality can recover.
    W, H, tile = 1280, 720, 640
    stride = TilingConfig(tile=tile, overlap=0.25).stride
    grid = tile_grid(W, H, tile, stride)
    cover = np.zeros((H, W), dtype=bool)
    for _, _, x0, y0 in grid:
        cover[y0:y0 + tile, x0:x0 + tile] = True
        check_bounds = 0 <= x0 <= W - tile and 0 <= y0 <= H - tile
        if not check_bounds:
            break
    check("every pixel covered by some tile", cover.all())
    check("all tiles fully inside the image",
          all(0 <= x0 <= W - tile and 0 <= y0 <= H - tile for _, _, x0, y0 in grid))
    check("1280x720 @ tile640/stride480 -> 6 tiles", len(grid) == 6, f"got {len(grid)}")
    check("axis shorter than tile yields one origin", tile_starts(400, 640, 480) == [0])
    check("last origin snaps to total-tile", tile_starts(720, 640, 480)[-1] == 80)
    check("grid is deterministic", tile_grid(W, H, tile, stride) == grid)

    # 320 tiles: the 4x-upscale option.
    s320 = TilingConfig(tile=320, overlap=0.25).stride
    g320 = tile_grid(W, H, 320, s320)
    cover = np.zeros((H, W), dtype=bool)
    for _, _, x0, y0 in g320:
        cover[y0:y0 + 320, x0:x0 + 320] = True
    check("tile320 also covers every pixel", cover.all())
    check("upscale reported correctly",
          abs(TilingConfig(tile=320, train_imgsz=1280).upscale - 4.0) < 1e-9)


# ------------------------------------------------------------------- scripted tiled predict
class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _T(np.asarray(xyxy, np.float32).reshape(-1, 4))
        self.conf = _T(np.asarray(conf, np.float32))
        self.cls = _T(np.asarray(cls, np.float32))

    def __len__(self):
        return len(self.xyxy.v)


class _T:
    """Stands in for a torch tensor: only .cpu().numpy() is ever called on these."""

    def __init__(self, v):
        self.v = v

    def cpu(self):
        return self

    def numpy(self):
        return self.v


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Returns whatever detections the test scripted for each tile, by call order."""

    def __init__(self, per_tile):
        self.per_tile = list(per_tile)
        self.calls = 0

    def predict(self, chunk, **kw):
        out = []
        for _ in chunk:
            out.append(_Res(self.per_tile[self.calls] if self.calls < len(self.per_tile) else None))
            self.calls += 1
        return out


def _predictor(per_tile, **kw):
    from predict import TiledPredictor
    cfg = TilingConfig(tile=640, overlap=0.25, train_imgsz=1280)
    return TiledPredictor(None, cfg, model=_FakeModel(per_tile), **kw)


def test_tiled_predict():
    print("\ntiled inference -> image coordinates")
    img = np.zeros((720, 1280, 3), np.uint8)
    grid = tile_grid(1280, 720, 640, 480)  # 6 tiles: x0 in {0,480,640}, y0 in {0,80}

    # One detection in the middle of tile 0 (origin 0,0) -- must come back unshifted.
    # One in the middle of a later tile -- must come back shifted by that tile's origin.
    per_tile = [None] * len(grid)
    per_tile[0] = _FakeBoxes([[300, 300, 310, 310]], [0.9], [0])
    idx = next(i for i, g in enumerate(grid) if (g[2], g[3]) == (640, 80))
    per_tile[idx] = _FakeBoxes([[100, 100, 108, 108]], [0.8], [1])

    boxes, scores, classes = _predictor(per_tile, edge_margin=0, max_box_px=0).predict(img)
    check("2 detections survive", len(boxes) == 2, f"got {len(boxes)}")
    got = {tuple(np.round(b).astype(int)) for b in boxes}
    check("tile-0 box unshifted", (300, 300, 310, 310) in got, str(got))
    check("tile(640,80) box lifted to image coords", (740, 180, 748, 188) in got, str(got))
    check("scores/classes carried through",
          sorted(np.round(scores, 2)) == [0.8, 0.9] and sorted(classes) == [0, 1])

    print("\ninner-edge rejection")
    # A box hard against the RIGHT edge of the tile at x0=0. That edge is interior (the
    # neighbour tile covers it), so the truncated copy must be dropped.
    per_tile = [None] * len(grid)
    per_tile[0] = _FakeBoxes([[630, 300, 640, 310]], [0.99], [0])
    b, _, _ = _predictor(per_tile, edge_margin=2, max_box_px=0).predict(img)
    check("box on an interior tile edge is dropped", len(b) == 0, f"got {len(b)}")

    # The same box against the LEFT edge of the tile at x0=0 -- that IS the image border,
    # nothing else covers it, so it must be kept.
    per_tile = [None] * len(grid)
    per_tile[0] = _FakeBoxes([[0, 300, 10, 310]], [0.99], [0])
    b, _, _ = _predictor(per_tile, edge_margin=2, max_box_px=0).predict(img)
    check("box on the image border is kept", len(b) == 1, f"got {len(b)}")

    print("\nsize gate")
    per_tile = [None] * len(grid)
    per_tile[0] = _FakeBoxes([[100, 100, 110, 110], [100, 200, 300, 400]], [0.9, 0.9], [0, 0])
    b, _, _ = _predictor(per_tile, edge_margin=0, max_box_px=64).predict(img)
    check("oversized detection gated out", len(b) == 1 and longest_side(b)[0] <= 64)

    print("\ncross-tile duplicate merge")
    # The same physical object seen in two overlapping tiles must collapse to one box.
    per_tile = [None] * len(grid)
    per_tile[0] = _FakeBoxes([[500, 300, 512, 312]], [0.9], [0])           # image 500..512
    j = next(i for i, g in enumerate(grid) if (g[2], g[3]) == (480, 0))
    per_tile[j] = _FakeBoxes([[20, 300, 32, 312]], [0.7], [0])            # image 500..512 too
    b, s, _ = _predictor(per_tile, edge_margin=0, max_box_px=0, iou=0.5).predict(img)
    check("duplicate across tiles merged to one", len(b) == 1, f"got {len(b)}")
    check("higher-confidence copy survives", len(s) == 1 and abs(s[0] - 0.9) < 1e-6)


# ---------------------------------------------------------------------------- box utilities
def test_boxops():
    print("\nbox ops")
    a = np.array([[0, 0, 10, 10]], np.float32)
    check("iou(self) == 1", abs(iou_matrix(a, a)[0, 0] - 1.0) < 1e-6)
    b = np.array([[20, 20, 30, 30]], np.float32)
    check("iou(disjoint) == 0", iou_matrix(a, b)[0, 0] == 0.0)
    half = np.array([[5, 0, 15, 10]], np.float32)
    check("iou(half overlap) == 1/3", abs(iou_matrix(a, half)[0, 0] - 1 / 3) < 1e-6)

    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7], np.float32), np.zeros(3, int), 0.5)
    check("nms drops the overlapping duplicate", len(keep) == 2, f"got {len(keep)}")
    check("nms keeps the highest score first", keep[0] == 0)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7], np.float32), np.array([0, 1, 0]), 0.5)
    check("nms is class-wise (different classes both kept)", len(keep) == 3, f"got {len(keep)}")

    print("\nwbf")
    box_a = np.array([[10, 10, 20, 20]], np.float32)
    box_b = np.array([[11, 11, 21, 21]], np.float32)
    fb, fs, fc = wbf([box_a, box_b], [np.array([0.8], np.float32), np.array([0.6], np.float32)],
                     [np.zeros(1, int), np.zeros(1, int)], iou_thr=0.5, n_models=2)
    check("agreeing boxes fuse to one", len(fb) == 1, f"got {len(fb)}")
    check("fused box lies between the inputs", 10 < fb[0][0] < 11, str(fb))
    lone_b, lone_s, _ = wbf([box_a, np.zeros((0, 4), np.float32)],
                            [np.array([0.8], np.float32), np.zeros(0, np.float32)],
                            [np.zeros(1, int), np.zeros(0, int)], iou_thr=0.5, n_models=2)
    check("a lone detection is damped vs an agreed one", lone_s[0] < fs[0],
          f"lone={lone_s[0]:.3f} agreed={fs[0]:.3f}")
    # This damping is exactly the conf-cut hazard the plan warns about -- assert it is real
    # so nobody enables --fuse assuming scores are preserved.
    check("lone WBF score is halved with 2 models", abs(lone_s[0] - 0.4) < 1e-6,
          f"got {lone_s[0]:.3f}")


# ---------------------------------------------------------------------- matching / AP / IO
def test_eval():
    print("\nmatching and AP")
    from evaluate import _ap, _match

    gt = np.array([[100, 100, 106, 106], [200, 200, 206, 206]], np.float32)
    exact = np.array([[100, 100, 106, 106]], np.float32)
    tp, _, taken = _match(exact, np.array([0.9], np.float32), gt, 0.5, "iou", 8.0)
    check("exact box matches under IoU", tp[0] and taken[0] and not taken[1])

    # A 2 px shift on a 6 px box: IoU says miss, center distance says hit. This asymmetry is
    # the reason evaluate.py offers both -- assert it rather than trusting the description.
    shifted = np.array([[102, 102, 108, 108]], np.float32)
    tp_iou, _, _ = _match(shifted, np.array([0.9], np.float32), gt, 0.5, "iou", 8.0)
    tp_ctr, _, _ = _match(shifted, np.array([0.9], np.float32), gt, 0.5, "center", 8.0)
    check("2px shift on a 6px box FAILS IoU@0.5", not tp_iou[0])
    check("2px shift on a 6px box PASSES center match", tp_ctr[0])

    two = np.array([[100, 100, 106, 106], [100, 100, 106, 106]], np.float32)
    tp, _, _ = _match(two, np.array([0.9, 0.8], np.float32), gt, 0.5, "iou", 8.0)
    check("one GT is claimed only once", tp[0] and not tp[1])

    check("AP of a perfect ranking == 1", abs(_ap(np.array([True, True]), 2) - 1.0) < 1e-6)
    check("AP with no GT == 0", _ap(np.array([True]), 0) == 0.0)
    check("AP of half-recall ranking == 0.5", abs(_ap(np.array([True, False]), 2) - 0.5) < 1e-6)

    print("\nsub-model / B overlap accounting")
    from evaluate import classify_small_box

    gt_b = np.array([[100, 100, 108, 108]], np.float32)   # one disposable_cup GT, class 7
    gt_c = np.array([7], int)
    # A correct sub-model box that also sits inside a big B 'picnic_table' box (class 10):
    # this is the case a naive veto would wrongly delete.
    b_box = np.array([[60, 60, 300, 300]], np.float32)
    b_cls = np.array([10], int)
    is_tp, miou, bhit = classify_small_box(np.array([101, 101, 107, 107], np.float32), 7,
                                           gt_b, gt_c, b_box, b_cls, "iou", 0.5, 8.0)
    check("correct box on a GT is TP", is_tp)
    check("overlap with the containing B box is recorded", bhit == 10 and miou > 0)
    check("small box inside a big box has low IoU with it", miou < 0.1, f"iou={miou:.3f}")

    # An FP sub-model box squarely on top of a same-size B 'cola_can' box (class 5):
    # high IoU, no matching GT -> exactly what a veto should catch.
    fp = np.array([[400, 400, 410, 410]], np.float32)
    is_tp, miou, bhit = classify_small_box(fp[0], 7, gt_b, gt_c,
                                           np.array([[401, 401, 411, 411]], np.float32),
                                           np.array([5], int), "iou", 0.5, 8.0)
    check("FP box with no GT is not TP", not is_tp)
    check("FP box coincident with a B box has high IoU", miou > 0.5, f"iou={miou:.3f}")
    check("no B boxes -> b_hit is None",
          classify_small_box(fp[0], 7, gt_b, gt_c, np.zeros((0, 4), np.float32),
                             np.zeros(0, int))[2] is None)

    print("\nlabel round-trip")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.txt")
        with open(p, "w") as f:
            f.write("2 0.5 0.5 0.0125 0.0125\n")  # 16x9 px box centred in 1280x720
        boxes, cls = read_yolo_labels(p, 1280, 720)
        check("label decodes to pixel xyxy",
              np.allclose(boxes[0], [632, 355.5, 648, 364.5]) and cls[0] == 2, str(boxes))
        check("missing label file reads as background",
              len(read_yolo_labels(os.path.join(d, "nope.txt"), 1280, 720)[0]) == 0)


if __name__ == "__main__":
    test_grid()
    test_boxops()
    test_tiled_predict()
    test_eval()
    print(f"\n{'=' * 60}")
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
    print("all checks passed")
