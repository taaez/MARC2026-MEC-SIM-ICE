"""Offline curb / step analysis of survey runs (live_survey_app.py output).

    python3 curb_analysis.py <run_dir> [<run_dir> ...]  -> curb_report.json (+ printout)

For every road / off-road boundary cell of the occupancy map:
  - lidar: in each scan, the boundary sample's ground height on the road side vs the off-road
    side, measured inside that one scan (pose / terrain-slope drift between scans cancels);
    median over scans
  - depth: the same from the frames captured while facing a boundary
  dz = largest step in a +-1.5 m profile across the boundary, signed towards off-road
  (+ = kerb up onto the off-road side, - = drop)

Classes: flush |dz| < FLUSH_M, low < LOW_M, high, drop (dz < -FLUSH_M).
Entries: runs of contiguous flush boundary cells at least ENTRY_MIN_M long (robot can drive in
without a kerb). All thresholds are here; nothing references a particular place.
"""

import glob
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage

CELL_M = 0.2
PROFILE_M = 1.5                     # profile half-length along the boundary normal
BIN_M = 0.1
STEP_W = 0.4                        # m averaged on each side of a candidate step
MIN_COVER = 0.6                     # fraction of the profile that must have data
HALF_ALONG = 0.3                    # m along the boundary
MIN_PTS = 4
RANGE_MIN, RANGE_MAX = 1.0, 5.5     # lidar ground ring radii that see the floor
FLUSH_M, LOW_M = 0.03, 0.10
ENTRY_MIN_M = 1.0
DEPTH_MAX = 4.0


def load_map(run):
    m = np.load(os.path.join(run, "map.npz"))
    grid, (ox, oy), res = m["grid"], m["origin"], float(m["res"])
    f = max(1, int(round(CELL_M / res)))
    h, w = grid.shape[0] // f, grid.shape[1] // f
    sub = grid[:h * f, :w * f].reshape(h, f, w, f)
    road = (sub == 0).mean(axis=(1, 3)) > 0.5
    unknown = (sub < 0).mean(axis=(1, 3)) > 0.5
    boundary = road & ndimage.binary_dilation(unknown, structure=np.ones((3, 3), bool))
    # outward normal (towards off-road): -grad of a smoothed road mask
    sm = ndimage.gaussian_filter(road.astype(float), 1.5)
    gy, gx = np.gradient(sm)
    bi, bj = np.nonzero(boundary)
    nx, ny = -gx[bi, bj], -gy[bi, bj]
    nn = np.hypot(nx, ny) + 1e-9
    xs = ox + (bj + 0.5) * CELL_M
    ys = oy + (bi + 0.5) * CELL_M
    return dict(bi=bi, bj=bj, x=xs, y=ys, nx=nx / nn, ny=ny / nn, shape=(h, w),
                origin=(float(ox), float(oy)))


def side_heights(px, py, pz, bx, by, nx, ny):
    """Per boundary sample: (step dz, offset of the step along the normal) from one point set.

    A ground-height profile along the boundary normal over +-PROFILE_M (0.1 m bins), then the
    split position with the largest |mean(next STEP_W) - mean(previous STEP_W)|. Searching the
    profile instead of a fixed window at the map boundary matters: the map's road edge can sit
    > 0.8 m from the real kerb, and a fixed window then sees road on both sides -> a false
    "flush" (seen in the RGB check of survey_rtC2_p1 entries 2/3).
    """
    from scipy.spatial import cKDTree
    out = np.full((len(bx), 2), np.nan)
    if len(px) < MIN_PTS:
        return out
    tree = cKDTree(np.stack([px, py], 1))
    near = tree.query_ball_point(np.stack([bx, by], 1), r=math.hypot(PROFILE_M, HALF_ALONG))
    nb = int(round(2 * PROFILE_M / BIN_M))
    w = int(round(STEP_W / BIN_M))
    for k in range(len(bx)):
        ii = np.asarray(near[k], dtype=np.int64)
        if len(ii) < 2 * MIN_PTS:
            continue
        dx, dy = px[ii] - bx[k], py[ii] - by[k]
        d_n = dx * nx[k] + dy * ny[k]                 # + = off-road side
        d_t = -dx * ny[k] + dy * nx[k]
        sel = (np.abs(d_t) <= HALF_ALONG) & (np.abs(d_n) < PROFILE_M)
        if sel.sum() < 2 * MIN_PTS:
            continue
        zz, dn = pz[ii][sel], d_n[sel]
        lo_ = np.percentile(zz, 10)
        keep = zz < lo_ + 0.45                        # drop objects above the local floor
        zz, dn = zz[keep], dn[keep]
        b_ = np.clip(((dn + PROFILE_M) / BIN_M).astype(int), 0, nb - 1)
        prof = np.full(nb, np.nan)
        for q in np.unique(b_):
            v = zz[b_ == q]
            if len(v) >= 2:
                prof[q] = np.percentile(v, 30)
        best = None
        for s_ in range(w, nb - w + 1):
            a_, c_ = prof[s_ - w:s_], prof[s_:s_ + w]
            if np.isfinite(a_).sum() >= 2 and np.isfinite(c_).sum() >= 2:
                dz = np.nanmean(c_) - np.nanmean(a_)
                if best is None or abs(dz) > abs(best[0]):
                    best = (dz, -PROFILE_M + s_ * BIN_M)
        if best is not None:
            out[k] = best
    return out


def lidar_steps(run, B):
    meas = [[] for _ in range(len(B["x"]))]
    for fn in sorted(glob.glob(os.path.join(run, "lidar_*.npz"))):
        z = np.load(fn, allow_pickle=True)
        for pose, mount, pts in zip(z["pose"], z["mount"], z["pts"]):
            pts = pts.astype(np.float32)
            r = np.hypot(pts[:, 0], pts[:, 1])
            keep = (r >= RANGE_MIN) & (r <= RANGE_MAX)
            pts = pts[keep]
            x, y, yaw = [float(v) for v in pose]
            c, s = math.cos(yaw), math.sin(yaw)
            sx, sy = pts[:, 0] + mount[0], pts[:, 1] + mount[1]
            px, py, pz = x + sx * c - sy * s, y + sx * s + sy * c, pts[:, 2]
            d = np.hypot(B["x"] - x, B["y"] - y)
            sel = np.nonzero((d >= RANGE_MIN + 0.3) & (d <= RANGE_MAX - 0.8))[0]
            if not len(sel):
                continue
            hz = side_heights(px, py, pz, B["x"][sel], B["y"][sel], B["nx"][sel], B["ny"][sel])
            for k, (dz, off) in zip(sel, hz):
                if np.isfinite(dz):
                    meas[k].append(dz)
    return meas


def depth_steps(run, B):
    cam = json.load(open(os.path.join(run, "camera.json")))
    K = np.array(cam["K"]).reshape(3, 3) / np.array([[2, 1, 2], [1, 2, 2], [1, 1, 1]])  # stride 2
    R, t = np.array(cam["mount_R"]), np.array(cam["mount_t"])
    meas = [[] for _ in range(len(B["x"]))]
    for fn in sorted(glob.glob(os.path.join(run, "cam_*.npz"))):
        z = np.load(fn, allow_pickle=True)
        for pose, dep, facing in zip(z["pose"], z["depth"], z["facing"]):
            if not facing:
                continue
            d = dep.astype(np.float32)
            h, w = d.shape
            vs, us = np.mgrid[0:h:2, 0:w:2]
            dd = d[vs, us]
            ok = np.isfinite(dd) & (dd > 0.1) & (dd < DEPTH_MAX)
            pc = np.stack([(us - K[0, 2]) / K[0, 0] * dd, (vs - K[1, 2]) / K[1, 1] * dd, dd], -1)[ok]
            pb = pc @ R.T + t
            x, y, yaw = [float(v) for v in pose]
            c, s = math.cos(yaw), math.sin(yaw)
            px, py, pz = x + pb[:, 0] * c - pb[:, 1] * s, y + pb[:, 0] * s + pb[:, 1] * c, pb[:, 2]
            dist = np.hypot(B["x"] - x, B["y"] - y)
            sel = np.nonzero((dist >= 0.5) & (dist <= DEPTH_MAX - 0.8))[0]
            if not len(sel):
                continue
            hz = side_heights(px, py, pz, B["x"][sel], B["y"][sel], B["nx"][sel], B["ny"][sel])
            for k, (dz, off) in zip(sel, hz):
                if np.isfinite(dz):
                    meas[k].append(dz)
    return meas


def classify(dz):
    if not np.isfinite(dz):
        return "unseen"
    if dz < -FLUSH_M:
        return "drop"
    if abs(dz) < FLUSH_M:
        return "flush"
    return "low" if dz < LOW_M else "high"


def entries(B, cls):
    """Contiguous flush boundary runs >= ENTRY_MIN_M (8-connected on the boundary grid)."""
    h, w = B["shape"]
    m = np.zeros((h, w), bool)
    idx = np.full((h, w), -1)
    for k, (i, j) in enumerate(zip(B["bi"], B["bj"])):
        idx[i, j] = k
        m[i, j] = cls[k] == "flush"
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    out = []
    for L in range(1, n + 1):
        ks = idx[lab == L]
        length = len(ks) * CELL_M
        if length >= ENTRY_MIN_M:
            out.append({"x": float(B["x"][ks].mean()), "y": float(B["y"][ks].mean()),
                        "length_m": round(length, 1), "cells": int(len(ks))})
    return sorted(out, key=lambda e: -e["length_m"])


def main(runs):
    B = load_map(runs[0])
    n = len(B["x"])
    lid = [[] for _ in range(n)]
    dep = [[] for _ in range(n)]
    for run in runs:
        for k, v in enumerate(lidar_steps(run, B)):
            lid[k] += v
        if os.path.exists(os.path.join(run, "camera.json")):
            for k, v in enumerate(depth_steps(run, B)):
                dep[k] += v
    dz_l = np.array([np.median(v) if len(v) >= 2 else np.nan for v in lid])
    dz_d = np.array([np.median(v) if len(v) >= 1 else np.nan for v in dep])
    dz = np.where(np.isfinite(dz_l), dz_l, dz_d)
    cls = [classify(v) for v in dz]
    seen = np.isfinite(dz)
    rep = {"runs": runs, "boundary_cells": n, "boundary_len_m": n * CELL_M,
           "seen_cells": int(seen.sum()), "seen_frac": float(seen.mean()),
           "classes": {c: cls.count(c) for c in ("flush", "low", "high", "drop", "unseen")}}
    up = dz[seen & (dz >= FLUSH_M)]
    dn = dz[seen & (dz <= -FLUSH_M)]
    rep["kerb_up_m"] = ({"n": int(len(up)), "mean": float(up.mean()), "median": float(np.median(up)),
                         "p10": float(np.percentile(up, 10)), "p90": float(np.percentile(up, 90))}
                        if len(up) else None)
    rep["drop_m"] = ({"n": int(len(dn)), "mean": float(dn.mean()), "median": float(np.median(dn)),
                      "p10": float(np.percentile(dn, 10)), "p90": float(np.percentile(dn, 90))}
                     if len(dn) else None)
    both = np.isfinite(dz_l) & np.isfinite(dz_d)
    rep["lidar_vs_depth"] = ({"n": int(both.sum()), "mean_abs_diff_m": float(np.abs(dz_l - dz_d)[both].mean()),
                              "corr": float(np.corrcoef(dz_l[both], dz_d[both])[0, 1]) if both.sum() > 2 else None}
                             if both.any() else None)
    rep["entries"] = entries(B, cls)
    rep["cells"] = [{"x": round(float(B["x"][k]), 2), "y": round(float(B["y"][k]), 2),
                     "dz_lidar": None if not np.isfinite(dz_l[k]) else round(float(dz_l[k]), 3),
                     "dz_depth": None if not np.isfinite(dz_d[k]) else round(float(dz_d[k]), 3),
                     "n_lidar": len(lid[k]), "cls": cls[k]} for k in range(n) if seen[k]]
    out = os.path.join(runs[-1], "curb_report.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, indent=1)[:3000])
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1:])
