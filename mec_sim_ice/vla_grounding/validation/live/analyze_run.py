"""Summarize one live Stage2 run dir (events.jsonl + telemetry.csv + platform result).

    python3 analyze_run.py runs/<tag> [runs/<tag> ...]   -> prints + writes <tag>/summary.json
"""

import csv
import glob
import json
import math
import os
import re
import sys


def _events(d):
    out = []
    with open(os.path.join(d, "events.jsonl")) as f:
        for line in f:
            try:
                e = json.loads(line)
                # sim time when the run recorded it (wall-clock fallback before /clock is huge)
                if e.get("st") is not None and e["st"] < 1e8:
                    e["t"] = e["st"]
                out.append(e)
            except ValueError:
                pass
    return out


def _telemetry(d):
    p = os.path.join(d, "telemetry.csv")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                row = {k: (float(v) if v not in ("", "None") else None) for k, v in r.items()}
                if row.get("st") is not None and row["st"] < 1e8:
                    row["t"] = row["st"]
                rows.append(row)
            except ValueError:
                pass
    return rows


def _first(ev, pat):
    for e in ev:
        if re.search(pat, e["msg"]):
            return e["t"]
    return None


def _count(ev, pat):
    return sum(1 for e in ev if re.search(pat, e["msg"]))


def _speed_series(tel):
    """|odom vx| per sample (/clock is coarser than the 20 Hz sampling, so pose
    differences over sim time are unreliable)."""
    return [(r["t"], abs(r["odom_vx"]), r) for r in tel
            if r.get("odom_vx") is not None and r.get("x") is not None]


def _brakes(ev, tel):
    """Per brake event: commanded v, speed at brake, distance until the chassis is below 0.05 m/s."""
    res = []
    sp = _speed_series(tel)
    for e in ev:
        m = re.search(r"obstacle within ([\d.]+)m ahead \(v=([\d.]+)\) -> braking", e["msg"])
        if not m:
            continue
        t0 = e["t"]
        seg = [s for s in sp if t0 - 0.06 <= s[0] <= t0 + 3.0]
        if not seg:
            continue
        v0 = seg[0][1]
        stop = next((s for s in seg if s[1] < 0.05), None)
        if stop is None:
            continue
        p0, p1 = seg[0][2], stop[2]
        res.append({"t": t0, "zone_m": float(m.group(1)), "cmd_v": float(m.group(2)),
                    "v_meas": round(v0, 3),
                    "stop_dist_m": round(math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"]), 3),
                    "stop_time_s": round(stop[0] - t0, 2)})
    return res


def summarize(d):
    ev = _events(d)
    tel = _telemetry(d)
    t_start = _first(ev, r"\[live\] drive start")
    t_leg1 = _first(ev, r"leg1 ->")
    t_standoff = _first(ev, r"standoff stop")
    t_leg2 = _first(ev, r"leg2 ->")
    t_done = _first(ev, r"driving finished")
    s = {"run": os.path.basename(d.rstrip("/"))}
    try:
        s["config"] = json.load(open(os.path.join(d, "config.json")))
    except Exception:
        pass

    def dt(a, b):
        return round(b - a, 1) if a and b else None

    s["t_leg1_s"] = dt(t_leg1, t_standoff)
    s["t_pick_s"] = dt(t_standoff, t_leg2)
    s["t_leg2_s"] = dt(t_leg2, t_done)
    s["t_drive_total_s"] = dt(t_start, t_done)
    wd = os.path.join(d, "watchdog.txt")
    s["watchdog_abort"] = open(wd).read().strip() if os.path.exists(wd) else None
    s["brakes"] = _count(ev, r"-> braking")
    s["stalls"] = _count(ev, r"stall detected|brake held|\] pinned:")
    s["follower_done"] = _count(ev, r"follower done")
    s["reroutes"] = _count(ev, r"rerouted \d+ waypoint")
    s["reverse_escape"] = _count(ev, r"consecutive stall at the same spot|chassis wedged \(")
    s["pinned"] = _count(ev, r"\] pinned:")
    s["mppi_all_hit"] = _count(ev, r"rollouts collide -> stopping")
    s["navfield_unreachable"] = _count(ev, r"navfield: goal unreachable|navfield: stage ")
    s["navfield_last_resort"] = _count(ev, r"all off-road passable|stage last-resort")
    s["navfield_tight_ok"] = _count(ev, r"stage tight .*V\(robot\)=[0-9]")
    s["navfield_hold"] = _count(ev, r"holding stage")
    s["pick_attempts"] = _count(ev, r"object pick sequence attempt")
    s["pick_results"] = [e["msg"] for e in ev if re.search(r"pick attempt \d+ (ok|failed) \(grip", e["msg"])]
    s["grasp_shape"] = [e["msg"].split("] ", 1)[1] for e in ev if "depth grasp shape" in e["msg"]]
    s["grasp_yaw"] = [e["msg"].strip() for e in ev if "[grasp-yaw]" in e["msg"]]
    s["delivered_log"] = next((e["msg"] for e in ev if "driving finished" in e["msg"]), None)

    if tel:
        rp = [(r["roll"], r["pitch"]) for r in tel if r.get("roll") is not None]
        if rp:
            s["max_abs_roll_deg"] = round(max(abs(math.degrees(a)) for a, _ in rp), 2)
            s["max_abs_pitch_deg"] = round(max(abs(math.degrees(b)) for _, b in rp), 2)
        acc = [math.hypot(r["ax"], r["ay"]) for r in tel if r.get("ax") is not None]
        if acc:
            acc.sort()
            s["imu_horiz_acc_p99"] = round(acc[int(0.99 * (len(acc) - 1))], 2)
        sp = _speed_series(tel)
        if sp:
            s["max_ground_speed"] = round(max(v for _, v, _ in sp if v < 3.0), 2)
    s["brake_events"] = _brakes(ev, tel)

    res = sorted(glob.glob(os.path.join(d, "platform_result*.json")))
    if res:
        r = json.load(open(res[-1]))
        s["platform_stage2"] = r.get("scores", {}).get("stage2")
        s["stage2_detail"] = r.get("stage2_detail")
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(s, f, indent=1, ensure_ascii=False)
    return s


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(summarize(d), indent=1, ensure_ascii=False))
