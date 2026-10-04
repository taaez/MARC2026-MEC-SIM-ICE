"""Offline A/B for demo/grasp_shape.py -- plain blob mean (old) vs shape-based grasp (new).

Renders synthetic base-camera depth frames of known objects (ray-cast against analytic
cylinders/boxes on a floor plane), runs the SAME raised-pixel extraction that
participant_app._locate_object_by_depth uses, then compares each estimator's grasp point
and gripper yaw to the ground truth.

    python3 mec_sim_ice/vla_grounding/validation/grasp_shape_eval.py [--trials 200]

What is real vs assumed -- read before trusting a number:
  real     the extraction pipeline (copied from participant_app, keep in sync), the
           estimator code under test, the yaw<->closing-axis relation (placo FK, see
           grasp_shape.py docstring).
  assumed  camera mount/intrinsics (no recorded Base_Camera_Center tf in the repo -- three
           mounts are swept), object dimensions, depth noise, and the capture model
           (2F-140 stroke 0.14 m, pad strip half-width 0.015 m, <=30deg yaw error on an
           elongated body). Use the numbers to compare old vs new, not as absolute rates.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "demo"))
import grasp_shape  # noqa: E402

FLOOR_Z = -0.30            # arm_base z of the ground (tuned tumbler grasp_down is -0.28)
W, H, HFOV_DEG = 320, 240, 87.0
DEFAULT_YAW = 0.52         # pnp_params default grasp_yaw
STROKE = 0.14
PAD_HALF = 0.015
MAX_YAW_ERR_DEG = 30.0
REACH = 0.72               # participant_app.ARM_REACH_FWD_M
EXPECT_WINDOW = 0.5        # participant_app.DEPTH_EXPECT_WINDOW_M

MOUNTS = {                 # name: (cam fwd, cam z, pitch down deg) in arm_base frame
    "mount_A": (0.10, -0.05, 30.0),
    "mount_B": (0.00, 0.10, 40.0),
    "mount_C": (0.15, -0.12, 22.0),
}


# ---------------------------------------------------------------- rendering --

def camera(mount):
    f, z, pitch = mount
    p = math.radians(pitch)
    zc = np.array([math.cos(p), 0.0, -math.sin(p)])   # optical z (forward, tilted down)
    xc = np.array([0.0, -1.0, 0.0])                   # optical x = right = -lat
    yc = np.cross(zc, xc)                             # optical y = down
    R = np.stack([xc, yc, zc], axis=1)
    C = np.array([f, 0.0, z])
    fx = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    return R, C, K


def _rays(R, K):
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    return d_cam @ R.T            # (H, W, 3); t along this ray == optical depth


def _hit_cylinder(C, D, p0, a, r, L):
    o = C - p0
    da, oa = D @ a, float(o @ a)
    dp = D - da[..., None] * a
    op = o - oa * a
    A = np.sum(dp * dp, -1)
    B = 2 * (dp @ op)
    Cc = float(op @ op) - r * r
    disc = B * B - 4 * A * Cc
    t = np.full(D.shape[:2], np.inf)
    ok = (disc >= 0) & (A > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0))
    for sgn in (-1, 1):
        tt = np.where(ok, (-B + sgn * sq) / np.where(A > 1e-12, 2 * A, 1), np.inf)
        s = oa + tt * da
        good = ok & (tt > 0) & (s >= 0) & (s <= L)
        t = np.where(good & (tt < t), tt, t)
    for s_cap in (0.0, L):
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (s_cap - oa) / da
        pp = op + tt[..., None] * dp
        good = np.isfinite(tt) & (tt > 0) & (np.sum(pp * pp, -1) <= r * r)
        t = np.where(good & (tt < t), tt, t)
    return t


def _hit_box(C, D, center, yaw, dims):
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    o = (C - center) @ Rz
    d = D @ Rz
    half = np.asarray(dims) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / d
        t2 = (half - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), -1)
    tmax = np.nanmin(np.maximum(t1, t2), -1)
    good = (tmax >= tmin) & (tmin > 0)
    return np.where(good, tmin, np.inf)


def render(objs, R, C, K, rng):
    D = _rays(R, K)
    with np.errstate(divide="ignore"):
        t = np.where(D[..., 2] < 0, (FLOOR_Z - C[2]) / D[..., 2], np.inf)
    for o in objs:
        if o["kind"] == "cyl":
            t = np.minimum(t, _hit_cylinder(C, D, o["p0"], o["axis"], o["r"], o["L"]))
        else:
            t = np.minimum(t, _hit_box(C, D, o["center"], o["yaw"], o["dims"]))
    depth = t.astype(np.float32)
    depth = depth + rng.normal(0, 1, depth.shape).astype(np.float32) * (0.001 + 0.0015 * depth ** 2)
    depth[rng.random(depth.shape) < 0.01] = np.nan
    depth[~np.isfinite(depth)] = np.nan
    return depth


# ------------------------------------------- extraction (== participant_app) --

def extract(depth, K, R, C, expected_xy):
    """Copy of _locate_object_by_depth's point extraction. Returns (fwd, lat, up, floor_z) or None."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    depth = np.where(valid, depth, 0.0)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([(us - cx) / fx * depth, (vs - cy) / fy * depth, depth], -1) @ R.T + C
    fwd, lat, up = pts[..., 0], pts[..., 1], pts[..., 2]
    floor_z = float(np.median(up[valid]))
    cand = valid & (up > floor_z + 0.03) & (fwd > 0.0) & (fwd < 1.2)
    ef, el = expected_xy
    near = cand & ((fwd - ef) ** 2 + (lat - el) ** 2 <= EXPECT_WINDOW ** 2)
    if int(near.sum()) < 20:
        return None
    return fwd[near], lat[near], up[near], floor_z, (ef, el)


def old_estimate(fwd, lat, up, floor_z, expected_xy=None):
    return {"fwd": float(fwd.mean()), "lat": float(lat.mean()),
            "down": max(float(up.mean()) - 0.01, floor_z + 0.012), "yaw": None}


def new_estimate(fwd, lat, up, floor_z, expected_xy=None):
    est = grasp_shape.estimate_grasp(fwd, lat, up, floor_z, expected_xy=expected_xy,
                                     reach_fwd=REACH)
    return est if est is not None else old_estimate(fwd, lat, up, floor_z)


# ----------------------------------------------------------------- scenes --

def make_target(kind, rng):
    yaw = rng.uniform(0, math.pi)
    if kind == "standing_tumbler":
        x, y = rng.uniform(0.38, 0.50), rng.uniform(-0.08, 0.08)
        o = {"kind": "cyl", "p0": np.array([x, y, FLOOR_Z]), "axis": np.array([0, 0, 1.0]),
             "r": 0.035, "L": 0.20}
        gt = {"type": "disc", "xy": np.array([x, y]), "r": 0.035, "elong": False}
    elif kind in ("lying_tumbler", "lying_umbrella"):
        r, L = (0.035, 0.20) if kind == "lying_tumbler" else (0.03, 0.85)
        if kind == "lying_tumbler":
            x, y = rng.uniform(0.38, 0.50), rng.uniform(-0.08, 0.08)
        else:
            x, y = rng.uniform(0.35, 0.60), rng.uniform(-0.15, 0.15)
        a = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        p0 = np.array([x, y, FLOOR_Z + r]) - a * L / 2
        o = {"kind": "cyl", "p0": p0, "axis": a, "r": r, "L": L}
        gt = {"type": "strip", "p0": p0[:2], "a": a[:2], "L": L, "r": r, "xy": np.array([x, y]),
              "elong": True, "theta": yaw}
    else:   # standing boxes
        dims = (0.16, 0.06, 0.21) if kind == "standing_box" else (0.05, 0.035, 0.15)
        x, y = rng.uniform(0.38, 0.50), rng.uniform(-0.08, 0.08)
        o = {"kind": "box", "center": np.array([x, y, FLOOR_Z + dims[2] / 2]), "yaw": yaw,
             "dims": dims}
        gt = {"type": "rect", "xy": np.array([x, y]), "yaw": yaw, "dims": dims[:2],
              "elong": dims[0] > 1.4 * dims[1], "theta": yaw}
    return o, gt


def distractor(gt, rng):
    """A bin-sized box 0.45-0.7 m from the target (partly inside the 0.5 m window)."""
    for _ in range(50):
        ang = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0.45, 0.7)
        c = gt["xy"] + d * np.array([math.cos(ang), math.sin(ang)])
        if c[0] <= 0.25:
            continue
        box = {"kind": "box", "center": np.array([c[0], c[1], FLOOR_Z + 0.3]),
               "yaw": rng.uniform(0, math.pi), "dims": (0.4, 0.4, 0.6)}
        # Reject placements where the target pierces or touches the box (>= 5 cm gap):
        # a long umbrella would otherwise often be rendered *through* the bin. Touching
        # targets are a known limitation of top-down blob splitting (measured separately:
        # lying umbrella within 5 cm of the box -> ~18% capture, vs ~89% when clear).
        cy, sy = math.cos(box["yaw"]), math.sin(box["yaw"])
        rel = (footprint(gt) - c) @ np.array([[cy, -sy], [sy, cy]])
        if np.any(np.max(np.abs(rel) / (np.array(box["dims"][:2]) / 2 + 0.05), axis=1) <= 1.0):
            continue
        return box
    return None


# ------------------------------------------------------------------ scoring --

def footprint(gt, n=60):
    if gt["type"] == "disc":
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        rr = np.sqrt(np.linspace(0, 1, 8))[:, None]
        pts = gt["xy"] + gt["r"] * rr[..., None] * np.stack([np.cos(t), np.sin(t)], -1)[None]
        return pts.reshape(-1, 2)
    if gt["type"] == "strip":
        s = np.linspace(0, gt["L"], 4 * n)[:, None]
        w = np.linspace(-gt["r"], gt["r"], 7)[None, :]
        perp = np.array([-gt["a"][1], gt["a"][0]])
        pts = gt["p0"][None, None] + s[..., None] * gt["a"] + w[..., None] * perp
        return pts.reshape(-1, 2)
    c, s = math.cos(gt["yaw"]), math.sin(gt["yaw"])
    gx, gy = np.meshgrid(np.linspace(-0.5, 0.5, 25) * gt["dims"][0],
                         np.linspace(-0.5, 0.5, 15) * gt["dims"][1])
    loc = np.stack([gx.ravel(), gy.ravel()], -1)
    return gt["xy"] + loc @ np.array([[c, s], [-s, c]])


def yaw_err_deg(gt, yaw):
    """Angle between the closing axis and the object's short direction (0 = ideal).

    Only a lying cylinder longer than the stroke has a stability requirement on top of
    the geometric fit; for boxes the footprint test in captured() already decides whether
    the object fits across the closing axis at that yaw.
    """
    if not gt["elong"] or gt["type"] != "strip":
        return 0.0
    closing = math.pi / 2 + yaw
    ideal = gt["theta"] + math.pi / 2
    return abs(math.degrees(grasp_shape.wrap_half_pi(closing - ideal)))


def captured(gt, est):
    yaw = DEFAULT_YAW if est["yaw"] is None else est["yaw"]
    if yaw_err_deg(gt, yaw) > MAX_YAW_ERR_DEG:
        return False
    phi = math.pi / 2 + yaw
    u = np.array([math.cos(phi), math.sin(phi)])
    wv = np.array([-u[1], u[0]])
    rel = footprint(gt) - np.array([est["fwd"], est["lat"]])
    with np.errstate(all="ignore"):   # numpy+Accelerate on macOS warns spuriously here
        pu, pw = rel @ u, rel @ wv
    strip = np.abs(pw) <= PAD_HALF
    if not np.any(strip):
        return False                                  # pads close on air
    return bool(np.max(np.abs(pu[strip])) <= STROKE / 2)   # a finger lands on the object


def center_err(gt, est):
    g = np.array([est["fwd"], est["lat"]])
    if gt["type"] == "strip":                         # distance to the reachable axis segment
        rel = g - gt["p0"]
        s = float(np.clip(rel @ gt["a"], 0.0, gt["L"]))
        return float(np.linalg.norm(rel - s * gt["a"]))
    return float(np.linalg.norm(g - gt["xy"]))


# --------------------------------------------------------------------- main --

def run(kind, mount, trials, with_distractor, seed):
    rng = np.random.default_rng(seed)
    R, C, K = camera(MOUNTS[mount])
    res = {"old": [], "new": []}
    skipped = 0
    for _ in range(trials):
        obj, gt = make_target(kind, rng)
        objs = [obj]
        if with_distractor:
            d = distractor(gt, rng)
            if d is not None:
                objs.append(d)
        depth = render(objs, R, C, K, rng)
        nav_err = rng.uniform(0, 0.15) * np.array([math.cos(a := rng.uniform(0, 6.283)), math.sin(a)])
        pts = extract(depth, K, R, C, tuple(gt["xy"] + nav_err))
        if pts is None:
            skipped += 1
            continue
        for name, fn in (("old", old_estimate), ("new", new_estimate)):
            est = fn(*pts)
            if est["fwd"] > REACH:
                est["fwd"] = REACH
            yaw = DEFAULT_YAW if est["yaw"] is None else est["yaw"]
            res[name].append((center_err(gt, est), yaw_err_deg(gt, yaw), captured(gt, est)))
    return res, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--off", nargs="*", default=[],
                    help="grasp_shape switches to turn off, e.g. --off USE_YAW USE_CLUSTER")
    args = ap.parse_args()
    kinds = ["standing_tumbler", "lying_tumbler", "lying_umbrella", "standing_box",
             "standing_sunblock"]
    for sw in args.off:
        setattr(grasp_shape, sw, False)
    print("switches: cluster=%s top_band=%s lying_axis=%s yaw=%s" % (
        grasp_shape.USE_CLUSTER, grasp_shape.USE_TOP_BAND_CENTER,
        grasp_shape.USE_LYING_AXIS, grasp_shape.USE_YAW))
    print("%-18s %-8s %-5s | %-27s | %-27s" % ("object", "mount", "dist",
          "OLD  err_p50/p90  yawErr  cap", "NEW  err_p50/p90  yawErr  cap"))
    total = {"old": [0, 0], "new": [0, 0]}
    for kind in kinds:
        for mount in MOUNTS:
            for dist in (False, True):
                res, skipped = run(kind, mount, args.trials, dist, args.seed)
                cols = []
                for name in ("old", "new"):
                    a = np.array(res[name], dtype=float)
                    if a.size == 0:
                        cols.append("no points")
                        continue
                    total[name][0] += int(a[:, 2].sum())
                    total[name][1] += len(a)
                    cols.append("%5.1f/%5.1fcm %5.1fdeg %4.0f%%" % (
                        100 * np.median(a[:, 0]), 100 * np.percentile(a[:, 0], 90),
                        np.median(a[:, 1]), 100 * a[:, 2].mean()))
                print("%-18s %-8s %-5s | %-27s | %-27s%s" % (
                    kind, mount, "yes" if dist else "no", cols[0], cols[1],
                    "  (skipped %d: <20 px)" % skipped if skipped else ""))
    for name in ("old", "new"):
        c, n = total[name]
        print("TOTAL %s capture %d/%d = %.1f%%" % (name, c, n, 100.0 * c / max(n, 1)))


if __name__ == "__main__":
    main()
