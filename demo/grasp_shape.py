"""Shape-based grasp point from the raised depth blob -- replaces the plain blob mean.

[팀 신규 추가] participant_app._locate_object_by_depth() finds the pixels raised above the
floor near the nav estimate and used to grasp at their plain mean. Two systematic errors
come with that:

  (a) The base camera only sees the near side of an object, so the mean of the visible
      surface sits between the true center and the robot -- about a radius short on a
      cylinder, and pulled sideways by the visible side wall.
  (b) The gripper yaw is a fixed constant (grasp_yaw 0.52 -> closing axis at 120deg in
      arm_base XY). On a lying umbrella / tumbler of random orientation the fingers can
      close along the shaft, or diagonally across it (unstable, or wider than the stroke).

This module only re-reads the same points (arm_base frame, metres) and returns a better
(fwd, lat, down, yaw). Each correction is behind its own switch so a change that does not
help can be dropped on its own -- see grasp_shape_eval.py for the A/B numbers.

Gripper yaw convention (measured with placo FK of the URDF, arm_pick.run_motion's
hold_quat = base_quat * Rz(grasp_yaw)): the line between the two finger pads lies at
    closing_axis_angle = 90deg + grasp_yaw      (arm_base XY, mod 180deg)
so to close *across* an object whose long axis is at theta, grasp_yaw = theta (mod pi).
"""

from __future__ import annotations

import math

import numpy as np

# -- switches (adopt / drop independently) --
# Split the windowed points into connected blobs (top-down, numpy only -- no scipy on the
# live host) and keep the one whose centroid is nearest the nav estimate. The 0.5 m window
# alone still lets the near face of a parked car / vending machine into the set, and both
# the plain mean and any shape fit are then dragged toward it.
USE_CLUSTER = True
# Standing/compact objects: center from the top band's bounding box instead of the mean of
# every visible pixel (fixes the near-side bias (a)).
USE_TOP_BAND_CENTER = True
# Lying/elongated objects: axis from PCA of the upper ridge, grasp at the middle of the
# visible extent, height at the axis (half the measured height).
USE_LYING_AXIS = True
# Turn the gripper so it closes across the long axis (fixes (b)). Applies to lying objects
# and to standing objects with a clearly elongated top face (a box's narrow side).
USE_YAW = True

MIN_POINTS = 20
CLUSTER_CELL_M = 0.02        # top-down grid for blob splitting
CLUSTER_LINK_CELLS = 2       # cells within this Chebyshev distance join (bridges ~4 cm gaps)
TOP_BAND_M = 0.02            # top band = within this of the 95th-percentile height
MIN_TOP_POINTS = 15
# Lying test: visible footprint much longer than it is tall, and clearly anisotropic.
LYING_EXTENT_OVER_HEIGHT = 2.0
LYING_MIN_ANISOTROPY = 1.8
# Ridge = points above this fraction of the measured height (upper part of a lying cylinder).
RIDGE_FRACTION = 0.6
# Top-face anisotropy needed before a standing object gets a yaw (a round tumbler top
# should keep the tuned default yaw).
BOX_YAW_MIN_ANISOTROPY = 1.4
BOX_YAW_MIN_EXTENT_M = 0.08
# A corrected center further than this from the plain mean is not trusted (bad fit).
MAX_SHIFT_M = 0.12
FLOOR_CLEARANCE_M = 0.012    # same floor clamp the old height rule used


def _pca2(xy):
    """(center, e1 major unit vector, e2, sqrt-eigvals) of 2D points."""
    c = xy.mean(axis=0)
    cov = np.cov((xy - c).T) if len(xy) > 1 else np.eye(2) * 1e-9
    w, v = np.linalg.eigh(cov)
    e1, e2 = v[:, 1], v[:, 0]
    return c, e1, e2, np.sqrt(np.maximum(w[::-1], 1e-12))


def _extent(s, lo=2.0, hi=98.0):
    a, b = np.percentile(s, [lo, hi])
    return float(a), float(b)


def wrap_half_pi(a):
    """Angle -> (-pi/2, pi/2]: the gripper is symmetric, so yaw is defined mod pi."""
    a = math.fmod(a, math.pi)
    if a <= -math.pi / 2:
        a += math.pi
    elif a > math.pi / 2:
        a -= math.pi
    return a


def select_cluster(xy, expected_xy):
    """Boolean mask of the connected blob (top-down) whose centroid is nearest expected_xy."""
    ij = np.floor((xy - xy.min(axis=0)) / CLUSTER_CELL_M).astype(np.int64)
    cells, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = inv.ravel()
    index = {(int(a), int(b)): k for k, (a, b) in enumerate(cells)}
    label = np.full(len(cells), -1, dtype=np.int64)
    n_lab = 0
    rng = range(-CLUSTER_LINK_CELLS, CLUSTER_LINK_CELLS + 1)
    for start in range(len(cells)):
        if label[start] >= 0:
            continue
        label[start] = n_lab
        stack = [start]
        while stack:
            k = stack.pop()
            a, b = int(cells[k][0]), int(cells[k][1])
            for da in rng:
                for db in rng:
                    j = index.get((a + da, b + db))
                    if j is not None and label[j] < 0:
                        label[j] = n_lab
                        stack.append(j)
        n_lab += 1
    if n_lab == 1:
        return np.ones(len(xy), dtype=bool)
    pt_label = label[inv]
    best, best_d = None, float("inf")
    for lab in range(n_lab):
        m = pt_label == lab
        if int(m.sum()) < MIN_POINTS:
            continue
        d = float(np.hypot(*(xy[m].mean(axis=0) - np.asarray(expected_xy, dtype=np.float64))))
        if d < best_d:
            best, best_d = m, d
    return best if best is not None else np.ones(len(xy), dtype=bool)


def estimate_grasp(fwd, lat, up, floor_z, expected_xy=None, reach_fwd=None):
    """See _estimate_grasp. Wrapped so numpy's spurious matmul FP warnings (numpy 1.x +
    Accelerate on macOS) stay out of the run log; a non-finite result is rejected instead."""
    with np.errstate(all="ignore"):
        est = _estimate_grasp(fwd, lat, up, floor_z, expected_xy, reach_fwd)
    if est is None or not all(math.isfinite(est[k]) for k in ("fwd", "lat", "down")):
        return None
    if est["yaw"] is not None and not math.isfinite(est["yaw"]):
        est["yaw"] = None
    return est


def _estimate_grasp(fwd, lat, up, floor_z, expected_xy=None, reach_fwd=None):
    """Grasp estimate from the raised points near the nav estimate (arm_base frame).

    Args:
        fwd, lat, up: 1-D arrays of raised points (already inside the distractor window).
        floor_z: measured floor height (same frame).
        expected_xy: nav-estimated (fwd, lat) -- picks the blob when USE_CLUSTER.
        reach_fwd: arm forward reach cap -- a lying object's grasp slides along its axis
            to stay inside it.

    Returns dict(fwd, lat, down, yaw, shape, length) -- length = visible extent along the
    axis for a lying object (None otherwise; arm_pick uses it to pick the long-object
    transport); yaw None = keep the tuned default; shape
    "mean" = no shape rule applied (plain mean of the selected blob, i.e. the old answer
    when clustering is off) -- or None when there are too few points.
    """
    fwd, lat, up = (np.asarray(a, dtype=np.float64).ravel() for a in (fwd, lat, up))
    if fwd.size < MIN_POINTS:
        return None
    xy = np.stack([fwd, lat], axis=1)
    if USE_CLUSTER and expected_xy is not None:
        keep = select_cluster(xy, expected_xy)
        xy, up = xy[keep], up[keep]
    mean_xy = xy.mean(axis=0)
    old_down = max(float(up.mean()) - 0.01, floor_z + FLOOR_CLEARANCE_M)
    fallback = {"fwd": float(mean_xy[0]), "lat": float(mean_xy[1]), "down": old_down,
                "yaw": None, "shape": "mean", "length": None}

    top = float(np.percentile(up, 95))
    h = max(top - floor_z, 0.005)

    c, e1, e2, sd = _pca2(xy)
    s1_lo, s1_hi = _extent((xy - c) @ e1)
    ext1 = s1_hi - s1_lo
    lying = (USE_LYING_AXIS and ext1 > LYING_EXTENT_OVER_HEIGHT * h
             and sd[0] / sd[1] > LYING_MIN_ANISOTROPY)

    yaw = None
    length = None
    shift_dir = None          # lying: only the cross-axis shift is checked (see below)
    if lying:
        # Axis from the ridge: on a lying cylinder the upper strip runs along the axis
        # line itself, while the full visible set is biased toward the camera side.
        ridge = up >= floor_z + RIDGE_FRACTION * h
        rxy = xy[ridge] if int(ridge.sum()) >= MIN_TOP_POINTS else xy
        rc, re1, re2, _ = _pca2(rxy)
        # Middle of the visible extent along the axis, on the ridge line.
        s_lo, s_hi = _extent((xy - rc) @ re1)
        s_mid = 0.5 * (s_lo + s_hi)
        g = rc + re1 * s_mid
        if reach_fwd is not None and g[0] > reach_fwd and abs(re1[0]) > 1e-3:
            # Slide toward the robot along the axis, but not past the visible end.
            ds = (reach_fwd - g[0]) / re1[0]
            s_new = float(np.clip(s_mid + ds, s_lo, s_hi))
            g = rc + re1 * s_new
        center = g
        down = max(floor_z + 0.5 * h, floor_z + FLOOR_CLEARANCE_M)
        if USE_YAW:
            yaw = wrap_half_pi(math.atan2(re1[1], re1[0]))
        shape = "lying"
        length = float(s_hi - s_lo)
        shift_dir = re2
    else:
        center = mean_xy
        band = up >= top - TOP_BAND_M
        if USE_TOP_BAND_CENTER and int(band.sum()) >= MIN_TOP_POINTS:
            txy = xy[band]
            tc, te1, te2, tsd = _pca2(txy)
            a_lo, a_hi = _extent((txy - tc) @ te1)
            b_lo, b_hi = _extent((txy - tc) @ te2)
            # Bounding-box middle, not the mean: the visible front wall adds density on
            # the near edge but does not move the far edge.
            center = tc + te1 * 0.5 * (a_lo + a_hi) + te2 * 0.5 * (b_lo + b_hi)
            if (USE_YAW and (a_hi - a_lo) > BOX_YAW_MIN_EXTENT_M
                    and (a_hi - a_lo) > BOX_YAW_MIN_ANISOTROPY * max(b_hi - b_lo, 1e-3)):
                yaw = wrap_half_pi(math.atan2(te1[1], te1[0]))
        down = old_down          # standing height rule unchanged (tuned constants win anyway)
        shape = "standing"

    # Sanity: a fit that moved the point far from the blob mean is a bad fit. Along a long
    # axis the grasp may legitimately slide (reach clamp), so there only the cross-axis
    # component is checked.
    delta = center - mean_xy
    shift = abs(float(delta @ shift_dir)) if shift_dir is not None else float(np.hypot(*delta))
    if shift > MAX_SHIFT_M:
        return fallback
    return {"fwd": float(center[0]), "lat": float(center[1]), "down": float(down),
            "yaw": yaw, "shape": shape, "length": length}
