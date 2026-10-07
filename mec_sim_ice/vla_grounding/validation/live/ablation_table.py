"""Planner ablation table from live run directories.

    python3 ablation_table.py runs/rt01_r2_legacy runs/rt01_r2_mppi ...  -> JSON lines + markdown

Per run: Stage 2 score and its parts (platform_result.json), leg times / brakes / stalls
(summary.json from analyze_run.py), and from telemetry.csv between the first "leg1 ->" event
and the end of driving: distance driven, mean speed while moving, max |roll| / |pitch|.
A run stopped by the watchdog has no platform score -> counted as 0 (failure).
"""

import csv
import json
import math
import os
import re
import sys


def events(run):
    out = []
    p = os.path.join(run, "events.jsonl")
    if os.path.exists(p):
        for ln in open(p, errors="replace"):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def first_st(ev, pat):
    for e in ev:
        if e.get("st") is not None and re.search(pat, e.get("msg", "")):
            return float(e["st"])
    return None


def telemetry(run, t0, t1):
    """Distance driven between t0 and t1 (sim s) and max |roll| / |pitch|. The sim clock ticks
    slower than the 20 Hz telemetry (most rows share one sim time, some even step back), so the
    distance is accumulated pose to pose without any dt condition; only jumps >= 3 m are dropped."""
    p = os.path.join(run, "telemetry.csv")
    if not os.path.exists(p) or t0 is None:
        return {}
    dist, roll, pitch, prev = 0.0, 0.0, 0.0, None
    for r in csv.DictReader(open(p)):
        try:
            st, x, y = float(r["st"]), float(r["x"]), float(r["y"])
        except (TypeError, ValueError):
            continue
        if st < t0 or (t1 is not None and st > t1):
            continue
        if r["roll"] and r["pitch"]:
            roll = max(roll, abs(math.degrees(float(r["roll"]))))
            pitch = max(pitch, abs(math.degrees(float(r["pitch"]))))
        if prev is not None:
            d = math.hypot(x - prev[0], y - prev[1])
            if d < 3.0:
                dist += d
        prev = (x, y)
    return {"dist_m": round(dist, 1), "max_roll_deg": round(roll, 1), "max_pitch_deg": round(pitch, 1)}


def row(run):
    s = json.load(open(os.path.join(run, "summary.json"))) if os.path.exists(os.path.join(run, "summary.json")) else {}
    pr = json.load(open(os.path.join(run, "platform_result.json"))) if os.path.exists(os.path.join(run, "platform_result.json")) else {}
    d = pr.get("stage2_detail") or {}
    ev = events(run)
    t0 = first_st(ev, r"leg1 ->")
    t1 = first_st(ev, r"driving finished|task_complete")
    tel = telemetry(run, t0, t1)
    cfg = json.load(open(os.path.join(run, "config.json"))) if os.path.exists(os.path.join(run, "config.json")) else {}
    log = open(os.path.join(run, "agent.log"), errors="replace").read() if os.path.exists(os.path.join(run, "agent.log")) else ""
    planner = "mppi" if "mppi pose=" in log else "legacy"
    legs = [s.get("t_leg1_s"), s.get("t_leg2_s")]
    t_drive = round(sum(legs), 1) if all(v is not None for v in legs) else None
    out = {
        "run": os.path.basename(run.rstrip("/")), "planner": planner,
        "stage2": pr.get("scores", {}).get("stage2") if pr else 0.0,
        "finished": bool(pr), "picked": d.get("picked"), "delivered": d.get("delivered"),
        "pick_score": d.get("pick_score"), "delivery_score": d.get("delivery_score"),
        # planner-relevant part: pick + delivery (80 max). Stage2 also adds a grounding-score
        # share (VLA perception, varies run to run independently of the planner)
        "task_score": (d.get("pick_score") or 0.0) + (d.get("delivery_score") or 0.0) if pr else 0.0,
        "robot_to_pickup_m": d.get("robot_to_pickup"),
        "t_leg1_s": s.get("t_leg1_s"), "t_pick_s": s.get("t_pick_s"), "t_leg2_s": s.get("t_leg2_s"),
        "t_drive_s": t_drive,
        "pick_attempts": s.get("pick_attempts"), "brakes": s.get("brakes"), "stalls": s.get("stalls"),
        "reverse_escape": s.get("reverse_escape"), "astar_failed": log.count("A* failed"),
        "tilt_guard": log.count("tilt guard: roll"), "reapproach": log.count("-> re-approach") + log.count("grasp near reach -> turned"),
        "watchdog": s.get("watchdog_abort"),
        "imu_horiz_acc_p99": s.get("imu_horiz_acc_p99"), **tel,
    }
    # distance includes the pick-phase creep (align + straight approach, ~1 m); speed = drive legs
    out["mean_speed_mps"] = (round(out["dist_m"] / t_drive, 2)
                             if t_drive and out.get("dist_m") else None)
    return out


if __name__ == "__main__":
    rows = [row(r) for r in sys.argv[1:]]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
