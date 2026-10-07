"""Ground-truth kerb / step statistics from the scene height field (analysis only -- never used
by the agent at runtime). Usage: python3 gt_analysis.py hf.npz map.npz [curb_report.json]"""
import json, math, sys
import numpy as np
from scipy import ndimage

hf = np.load(sys.argv[1]); mp = np.load(sys.argv[2])
Z, KIND, X0, Y0, R = hf["z"], hf["kind"], float(hf["x0"]), float(hf["y0"]), float(hf["res"])
KINDS = [str(k) for k in hf["kinds"]]
H, W = Z.shape


def zat(x, y):
    j = np.floor((np.asarray(x) - X0) / R).astype(int)
    i = np.floor((np.asarray(y) - Y0) / R).astype(int)
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    out = np.full(np.shape(x), np.nan)
    out[ok] = Z[i[ok], j[ok]]
    return out


def profile_step(bx, by, nx, ny, half=1.5, w=0.4, binm=0.1):
    """Same estimator as curb_analysis.side_heights, on the true surface."""
    s = np.arange(-half, half, binm) + binm / 2
    prof = zat(bx + s * nx, by + s * ny)
    k = int(round(w / binm))
    best = (np.nan, np.nan)
    for c in range(k, len(s) - k + 1):
        a, b = prof[c - k:c], prof[c:c + k]
        if np.isfinite(a).sum() >= 2 and np.isfinite(b).sum() >= 2:
            dz = np.nanmean(b) - np.nanmean(a)
            if not np.isfinite(best[0]) or abs(dz) > abs(best[0]):
                best = (dz, -half + c * binm)
    return best


# ---- 1. map boundary (same cells as curb_analysis) ----
grid, (ox, oy), res = mp["grid"], mp["origin"], float(mp["res"])
f = 4
h, w = grid.shape[0] // f, grid.shape[1] // f
sub = grid[:h * f, :w * f].reshape(h, f, w, f)
road = (sub == 0).mean(axis=(1, 3)) > 0.5
unk = (sub < 0).mean(axis=(1, 3)) > 0.5
bnd = road & ndimage.binary_dilation(unk, structure=np.ones((3, 3), bool))
sm = ndimage.gaussian_filter(road.astype(float), 1.5)
gy, gx = np.gradient(sm)
bi, bj = np.nonzero(bnd)
nx, ny = -gx[bi, bj], -gy[bi, bj]
nn = np.hypot(nx, ny) + 1e-9
nx, ny = nx / nn, ny / nn
bx, by = ox + (bj + 0.5) * 0.2, oy + (bi + 0.5) * 0.2
steps = np.array([profile_step(bx[k], by[k], nx[k], ny[k])[0] for k in range(len(bx))])
fin = np.isfinite(steps)


def stats(v):
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "p10": round(float(np.percentile(v, 10)), 4), "p90": round(float(np.percentile(v, 90)), 4),
            "max": round(float(v.max()), 4), "min": round(float(v.min()), 4)}


rep = {"boundary_cells": int(len(bx)), "boundary_with_gt": int(fin.sum())}
cls = np.full(len(bx), "nogt", dtype=object)
cls[fin & (np.abs(steps) < 0.03)] = "flush"
cls[fin & (steps >= 0.03) & (steps < 0.10)] = "low"
cls[fin & (steps >= 0.10)] = "high"
cls[fin & (steps <= -0.03)] = "drop"
rep["boundary_classes"] = {c: int((cls == c).sum()) for c in ("flush", "low", "high", "drop", "nogt")}
rep["boundary_up"] = stats(steps[fin & (steps >= 0.03)])
rep["boundary_drop"] = stats(steps[fin & (steps <= -0.03)])
rep["boundary_abs_all"] = stats(np.abs(steps))

# ---- 2. surface-kind transitions: step across each 4-neighbour edge where the kind changes ----
pairs = {}
for di, dj in ((0, 1), (1, 0)):
    k1, k2 = KIND[:H - di, :W - dj], KIND[di:, dj:]
    # compare heights 0.15 m away from the edge on each side (skip the vertical face cells)
    off = 3
    za = Z[max(0, -off * di):, :][:H - di, :W - dj] if False else None
    m = (k1 != k2) & (k1 >= 0) & (k2 >= 0)
    ii, jj = np.nonzero(m)
    i_a, j_a = np.clip(ii - off * di, 0, H - 1), np.clip(jj - off * dj, 0, W - 1)
    i_b, j_b = np.clip(ii + di + off * di, 0, H - 1), np.clip(jj + dj + off * dj, 0, W - 1)
    dz = Z[i_b, j_b] - Z[i_a, j_a]
    ka, kb = KIND[i_a, j_a], KIND[i_b, j_b]
    for a, b, d in zip(ka, kb, dz):
        if a < 0 or b < 0 or a == b or not np.isfinite(d):
            continue
        lo, hi = sorted((int(a), int(b)))
        sgn = d if a == lo else -d            # height of the second kind minus the first
        pairs.setdefault((KINDS[lo], KINDS[hi]), []).append(sgn)
rep["kind_steps"] = {f"{a}->{b} (z_{b} - z_{a})": stats(np.array(v)) for (a, b), v in pairs.items()}

# ---- 3. curb-free reachability: cells connected by steps <= thr from the spawn ----
spawn = (-64.69, 123.65)
targets = {"rtC2_object": (-77.26, 97.12), "owner_zone": (-54.57, 142.79)}
g = np.isfinite(Z)
reach_rep = {}
for thr in (0.03, 0.06, 0.10):
    ok_r = np.abs(np.diff(Z, axis=1)) <= thr      # edge (i,j)-(i,j+1)
    ok_d = np.abs(np.diff(Z, axis=0)) <= thr      # edge (i,j)-(i+1,j)
    # label via union-find over passable edges (scipy: build graph)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.arange(H * W).reshape(H, W)
    r1 = idx[:, :-1][ok_r & g[:, :-1] & g[:, 1:]]; c1 = idx[:, 1:][ok_r & g[:, :-1] & g[:, 1:]]
    r2 = idx[:-1, :][ok_d & g[:-1, :] & g[1:, :]]; c2 = idx[1:, :][ok_d & g[:-1, :] & g[1:, :]]
    A = coo_matrix((np.ones(len(r1) + len(r2)), (np.r_[r1, r2], np.r_[c1, c2])), shape=(H * W, H * W))
    n, lab = connected_components(A, directed=False)
    lab = lab.reshape(H, W)

    def cell(p):
        return int((p[1] - Y0) / R), int((p[0] - X0) / R)
    s_lab = lab[cell(spawn)]
    comp = lab == s_lab
    out = {"area_m2": round(float(comp.sum() * R * R), 1)}
    for name, p in targets.items():
        i, j = cell(p)
        # target reachable if any cell within 0.6 m of it is in the spawn component
        win = comp[max(i - 12, 0):i + 13, max(j - 12, 0):j + 13]
        out[name] = bool(win.any())
    reach_rep[f"max_step_{thr:.2f}"] = out
    if thr == 0.03:
        comp03 = comp
rep["curb_free_reach_from_spawn"] = reach_rep

# entries into grass/sidewalk areas outside the map road, reachable with no step (<=3 cm)
roadmask_hf = np.zeros_like(g)
# map road -> heightfield grid
yy, xx = np.mgrid[0:H, 0:W]
mi = ((Y0 + (yy + 0.5) * R - oy) / res).astype(int)
mj = ((X0 + (xx + 0.5) * R - ox) / res).astype(int)
okm = (mi >= 0) & (mi < grid.shape[0]) & (mj >= 0) & (mj < grid.shape[1])
roadmask_hf[okm] = grid[mi[okm], mj[okm]] == 0
outside = comp03 & ~roadmask_hf
lab2, n2 = ndimage.label(outside)
ents = []
for L in range(1, n2 + 1):
    m = lab2 == L
    a = m.sum() * R * R
    if a < 2.0:
        continue
    touch = m & ndimage.binary_dilation(roadmask_hf, iterations=2)
    ii, jj = np.nonzero(touch)
    kinds = np.bincount(KIND[m][KIND[m] >= 0], minlength=len(KINDS))
    ents.append({"area_m2": round(float(a), 1),
                 "entry_xy": [round(float(X0 + (jj.mean() + 0.5) * R), 2), round(float(Y0 + (ii.mean() + 0.5) * R), 2)] if len(ii) else None,
                 "entry_width_m": round(float(len(ii) * R / 2), 1),
                 "kinds": {KINDS[k]: round(float(c * R * R), 1) for k, c in enumerate(kinds) if c}})
rep["curb_free_offroad_areas"] = sorted(ents, key=lambda e: -e["area_m2"])[:15]

# ---- 4. sensor vs truth on the surveyed cells ----
if len(sys.argv) > 3:
    cells = json.load(open(sys.argv[3]))["cells"]
    from scipy.spatial import cKDTree
    t = cKDTree(np.stack([bx, by], 1))
    pairs_l, pairs_d = [], []
    for c in cells:
        d, k = t.query([c["x"], c["y"]])
        if d < 0.15 and np.isfinite(steps[k]):
            if c["dz_lidar"] is not None:
                pairs_l.append((c["dz_lidar"], steps[k]))
            if c["dz_depth"] is not None:
                pairs_d.append((c["dz_depth"], steps[k]))
    for name, pr in (("lidar", pairs_l), ("depth", pairs_d)):
        if len(pr) > 2:
            a = np.array(pr)
            e = a[:, 0] - a[:, 1]
            rep[f"sensor_vs_gt_{name}"] = {"n": len(a), "bias_m": round(float(e.mean()), 4),
                                          "mae_m": round(float(np.abs(e).mean()), 4),
                                          "corr": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3),
                                          "class_agree": round(float(np.mean([
                                              (abs(s) < .03) == (abs(g_) < .03) for s, g_ in a])), 3)}
np.savez_compressed("/o/gt_boundary_steps.npz", x=bx, y=by, nx=nx, ny=ny, step=steps)
json.dump(rep, open("/o/gt_curb_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
