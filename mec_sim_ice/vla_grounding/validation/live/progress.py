"""Watchdog progress check for one live run (called by run_one.sh every WATCHDOG_S).

    python3 progress.py <agent.log> <state.json>

Compares the agent log against the state saved at the previous check and prints
"progress <why>" or "stuck <why>" (exit 0 / 1). Progress = any of:
  - a milestone line (leg start, pick attempt/phase, reached/accepting, state change)
  - the best distance-to-goal of the current leg improved by >= DIST_GAIN_M
No new log line at all = stuck (the agent hung, as rt01_ab_legacy's pick did).
A window with no progress but with recovery actions (stall/pinned/reverse/tilt guard) gets
MAX_GRACE grace windows: rt01_mppi_r4 sat ~140 sim s (~13 min wall) at one spot, recovered
through stall marks + reverse escapes, and scored 95 -- a plain 3 min window killed it at +6 min.
"""
import json
import os
import re
import sys

DIST_GAIN_M = 0.3
MAX_GRACE = int(os.environ.get("WATCHDOG_GRACE", "3"))
RECOVERY = re.compile(r"stall detected|\] pinned:|wedged|reversing|tilt guard|brake held")
MILESTONE = re.compile(r"leg[12] ->|pick attempt|object pick sequence|at default pose|"
                       r"traversing waypoints|DONE \(holding|reached|accepting|\[STATE\]|"
                       r"driving finished|approach_straight|align|\[survey\] wp")
LEG = re.compile(r"leg[12] ->")
DIST = re.compile(r"\bpose=.*?dist=([\d.]+)m")

log_path, state_path = sys.argv[1], sys.argv[2]
lines = open(log_path, errors="replace").read().splitlines() if os.path.exists(log_path) else []
st = json.load(open(state_path)) if os.path.exists(state_path) else {"n": 0, "best": None}
grace = st.get("grace", 0)
new = lines[st["n"]:]
best0 = st["best"]
best = best0
milestones = []
recoveries = 0
for ln in new:
    if LEG.search(ln):
        best = best0 = None            # new leg: distances restart
    if RECOVERY.search(ln):
        recoveries += 1
    if MILESTONE.search(ln):
        milestones.append(ln)
    m = DIST.search(ln)
    if m:
        d = float(m.group(1))
        best = d if best is None else min(best, d)

def done(code, msg, grace):
    json.dump({"n": len(lines), "best": best, "grace": grace}, open(state_path, "w"))
    print(msg)
    sys.exit(code)


if not new:
    done(1, "stuck no new agent log lines", grace)
if milestones:
    done(0, "progress milestone: " + milestones[-1].split("INFO:")[-1].strip()[:100], 0)
if best is not None and (best0 is None or best0 - best >= DIST_GAIN_M):
    done(0, f"progress best dist {best0} -> {best:.2f}", 0)
if recoveries and grace < MAX_GRACE:
    done(0, f"grace {grace + 1}/{MAX_GRACE}: no progress but {recoveries} recovery action(s)",
         grace + 1)
done(1, f"stuck best dist {best0} -> {best} (< {DIST_GAIN_M} m gain), no milestone, "
        f"{recoveries} recovery action(s), grace used {grace}/{MAX_GRACE}", grace)
