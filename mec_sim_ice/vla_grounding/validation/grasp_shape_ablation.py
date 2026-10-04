"""Stage 2 grasp ablation -- grasp_shape.py full system minus one switch at a time.

Paired design: every synthetic depth frame (grasp_shape_eval.py renderer) is rendered once
and every config is evaluated on that same frame, so config differences are not scene luck.

    python3 mec_sim_ice/vla_grounding/validation/grasp_shape_ablation.py [--trials 200]
        -> runs/stage2_grasp_ablation.json

Configs: full / -USE_CLUSTER / -USE_TOP_BAND_CENTER / -USE_LYING_AXIS / -USE_YAW /
baseline (all switches off == the qualifier-time plain blob mean + fixed yaw 0.52; the eval
asserts it is identical to grasp_shape_eval.old_estimate on every frame).
Same assumptions as grasp_shape_eval.py (camera mounts, object sizes, capture model).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import grasp_shape_eval as E                                       # noqa: E402

G = E.grasp_shape
SWITCHES = ["USE_CLUSTER", "USE_TOP_BAND_CENTER", "USE_LYING_AXIS", "USE_YAW"]
LABELS = {
    "USE_CLUSTER": "Blob split + nearest-to-nav-estimate selection",
    "USE_TOP_BAND_CENTER": "Top-band bounding-box center (standing objects)",
    "USE_LYING_AXIS": "Ridge-PCA axis grasp (lying objects)",
    "USE_YAW": "Gripper yaw across the long axis",
}
KINDS = ["standing_tumbler", "lying_tumbler", "lying_umbrella", "standing_box",
         "standing_sunblock"]


def set_switches(disabled):
    for s in SWITCHES:
        setattr(G, s, s not in disabled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    configs = [("full", [])] + [("-" + s, [s]) for s in SWITCHES] + [("baseline", list(SWITCHES))]
    # recs[config] -> list of (kind, distractor, center_err_m, yaw_err_deg, captured)
    recs = {name: [] for name, _ in configs}
    frames = skipped = 0
    for ki, kind in enumerate(KINDS):
        for mi, mount in enumerate(E.MOUNTS):
            R, C, K = E.camera(E.MOUNTS[mount])
            for di, dist in enumerate((False, True)):
                rng = np.random.default_rng(args.seed + 1000 * ki + 100 * mi + di)
                for _ in range(args.trials):
                    obj, gt = E.make_target(kind, rng)
                    objs = [obj]
                    if dist:
                        d = E.distractor(gt, rng)
                        if d is not None:
                            objs.append(d)
                    depth = E.render(objs, R, C, K, rng)
                    a = rng.uniform(0, 2 * math.pi)
                    nav_err = rng.uniform(0, 0.15) * np.array([math.cos(a), math.sin(a)])
                    pts = E.extract(depth, K, R, C, tuple(gt["xy"] + nav_err))
                    if pts is None:
                        skipped += 1
                        continue
                    frames += 1
                    old = E.old_estimate(*pts)
                    for name, disabled in configs:
                        set_switches(disabled)
                        est = E.new_estimate(*pts)
                        if name == "baseline":
                            assert (abs(est["fwd"] - old["fwd"]) < 1e-9
                                    and abs(est["lat"] - old["lat"]) < 1e-9
                                    and est["yaw"] is None), "baseline != old estimator"
                        est["fwd"] = min(est["fwd"], E.REACH)
                        yaw = E.DEFAULT_YAW if est["yaw"] is None else est["yaw"]
                        recs[name].append((kind, dist, E.center_err(gt, est),
                                           E.yaw_err_deg(gt, yaw), E.captured(gt, est)))
    set_switches([])

    def agg(rows):
        a = np.array([(r[2], r[3], r[4]) for r in rows], dtype=float)
        elong = [r[3] for r in rows if r[0] in ("lying_tumbler", "lying_umbrella")]
        return {"n": len(rows), "captured": int(a[:, 2].sum()),
                "capture_rate": float(a[:, 2].mean()),
                "center_err_p50_cm": float(100 * np.median(a[:, 0])),
                "center_err_p90_cm": float(100 * np.percentile(a[:, 0], 90)),
                "yaw_err_p50_deg_lying": float(np.median(elong)) if elong else None}

    out = {"switches": SWITCHES, "labels": LABELS, "kinds": KINDS, "trials": args.trials,
           "frames": frames, "skipped_frames": skipped, "mounts": E.MOUNTS, "configs": {}}
    for name, disabled in configs:
        rows = recs[name]
        res = {"disabled": disabled, "all": agg(rows),
               "no_distractor": agg([r for r in rows if not r[1]]),
               "distractor": agg([r for r in rows if r[1]]),
               "by_kind": {k: agg([r for r in rows if r[0] == k]) for k in KINDS}}
        out["configs"][name] = res
        print("%-22s capture %5.1f%%  (no dist %5.1f%%, dist %5.1f%%)  err p50 %.1fcm  yaw(lying) %.1fdeg" % (
            name, 100 * res["all"]["capture_rate"], 100 * res["no_distractor"]["capture_rate"],
            100 * res["distractor"]["capture_rate"], res["all"]["center_err_p50_cm"],
            res["all"]["yaw_err_p50_deg_lying"]))
    path = os.path.join(_HERE, "runs", "stage2_grasp_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("frames %d (skipped %d) -> wrote %s" % (frames, skipped, path))


if __name__ == "__main__":
    main()
