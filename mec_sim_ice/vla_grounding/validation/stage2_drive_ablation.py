"""Stage 2 driving ablation -- full system minus one component at a time (offline harness).

Each component added after the `qualifier` tag is switched back to its qualifier-time
behaviour by monkeypatching *inside this process only* (participant_app.py itself is not
edited), then the 6 retrieval problems run through stage2_sim.run_execution on the
plain map and with --hidden-obstacles (obstacles only as chassis colliders + lidar).

    python3 mec_sim_ice/vla_grounding/validation/stage2_drive_ablation.py
        -> runs/stage2_drive_ablation.json

Faithfulness check: the "baseline" config (every component reverted at once) must
reproduce the pre-change qualifier totals measured on the untouched qualifier code
(plain 634.9 s, hidden 659.8 s). If it does not, a revert patch is wrong.

Components (ids used in the JSON / Excel):
  speed     MAX_LINEAR 1.1 m/s           (revert: 0.8)
  profile   sqrt decel on remaining path  (revert: K_LINEAR*dist on final wp, cruise others)
  long_acc  longitudinal accel limiter    (revert: none)
  lat_acc   lateral accel cap |v*w|<=1.0  (revert: none)
  brake     speed-proportional stop dist  (revert: fixed 0.6 m)
  approach  distance-proportional creep   (revert: fixed 0.12 / 0.06 m/s)
  halt      confirmed-stop _halt()        (revert: fixed 5x0.05 s + 5x0.1 s after approach)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import fake_client                                                  # noqa: E402
import stage2_sim                                                   # noqa: E402

fake_client.install_shims()
import mock_agent_navigation as nav                                 # noqa: E402
import participant_app as app_mod                                   # noqa: E402

COMPONENTS = ["speed", "profile", "long_acc", "lat_acc", "brake", "approach", "halt"]
LABELS = {
    "speed": "Max speed 1.1 m/s (vs 0.8)",
    "profile": "sqrt decel profile on remaining path",
    "long_acc": "Longitudinal accel limit (0.6 / 1.0 m/s^2)",
    "lat_acc": "Lateral accel cap |v*w| <= 1.0 m/s^2",
    "brake": "Speed-proportional reactive brake distance",
    "approach": "Distance-proportional straight approach",
    "halt": "Confirmed-stop halt (replaces fixed waits)",
}

# ---------------------------------------------------------------- originals --
_ORIG = {
    "MAX_LINEAR": nav.GoalSeekController.MAX_LINEAR,
    "MAX_LAT_ACCEL": nav.GoalSeekController.MAX_LAT_ACCEL,
    "gs_compute": nav.GoalSeekController.compute,
    "pf_compute": nav.PathFollower.compute,
    "limit_accel": app_mod.DemoParticipant.__dict__["_limit_accel"],
    "react_stop": app_mod.DemoParticipant.__dict__["_react_stop_dist"],
    "halt": app_mod.DemoParticipant._halt,
    "approach": app_mod.DemoParticipant._approach_straight,
    "APPROACH": (app_mod.APPROACH_K, app_mod.APPROACH_V_MIN, app_mod.APPROACH_V_MAX),
}


def _old_gs_compute(self, pose, goal, cruise=False, remaining=None):
    """GoalSeekController.compute at the qualifier tag (K_LINEAR=0.5 proportional decel),
    keeping the lateral cap so that component can be ablated separately."""
    x, y, yaw = pose
    dx, dy = goal[0] - x, goal[1] - y
    dist = math.hypot(dx, dy)
    if dist < self.GOAL_TOLERANCE:
        return 0.0, 0.0, True
    heading_err = nav._wrap_pi(math.atan2(dy, dx) - yaw)
    angular = nav._clamp(self.K_ANGULAR * heading_err, -self.MAX_ANGULAR, self.MAX_ANGULAR)
    if abs(heading_err) > self.HEADING_TOLERANCE:
        linear = 0.0
    elif cruise:
        linear = self.MAX_LINEAR
    else:
        linear = nav._clamp(0.5 * dist, 0.0, self.MAX_LINEAR)
    if linear > 0.0 and abs(angular) > 1e-6:
        linear = min(linear, self.MAX_LAT_ACCEL / abs(angular))
    angular = nav.cap_angular(linear, angular, self.STEERING_TRACK)
    return linear, angular, False


def _old_pf_compute(self, pose):
    """PathFollower.compute at the qualifier tag (intermediate waypoints cruise)."""
    x, y, _ = pose
    while self.idx < len(self.path) - 1:
        wx, wy = self.path[self.idx]
        if math.hypot(wx - x, wy - y) < self.WAYPOINT_TOLERANCE:
            self.idx += 1
        else:
            break
    wx, wy = self.path[self.idx]
    final = self.idx >= len(self.path) - 1
    cruise = (not final) or self._passthrough
    linear, angular, reached = self.ctrl.compute(pose, (wx, wy), cruise=cruise)
    info = {"idx": self.idx, "total": len(self.path), "waypoint": (wx, wy)}
    return linear, angular, final and reached, info


def _old_halt(self, *a, **kw):
    """qualifier: 5 zero commands x 0.05 s after align / approach."""
    for _ in range(5):
        self.client.send_cmd_vel(0.0, 0.0)
        self.client.sleep(0.05)
    return False


def _approach_with_old_wait(self, *a, **kw):
    """qualifier: _drive added 5 zero commands x 0.1 s after every straight approach."""
    r = _ORIG["approach"](self, *a, **kw)
    for _ in range(5):
        self.client.send_cmd_vel(0.0, 0.0)
        self.client.sleep(0.1)
    return r


def apply(disabled):
    """Restore originals, then revert each component in `disabled`."""
    G, D = nav.GoalSeekController, app_mod.DemoParticipant
    G.MAX_LINEAR = _ORIG["MAX_LINEAR"]
    G.MAX_LAT_ACCEL = _ORIG["MAX_LAT_ACCEL"]
    G.compute = _ORIG["gs_compute"]
    nav.PathFollower.compute = _ORIG["pf_compute"]
    D._limit_accel = _ORIG["limit_accel"]
    D._react_stop_dist = _ORIG["react_stop"]
    D._halt = _ORIG["halt"]
    D._approach_straight = _ORIG["approach"]
    app_mod.APPROACH_K, app_mod.APPROACH_V_MIN, app_mod.APPROACH_V_MAX = _ORIG["APPROACH"]

    if "speed" in disabled:
        G.MAX_LINEAR = 0.8
    if "profile" in disabled:
        G.compute = _old_gs_compute
        nav.PathFollower.compute = _old_pf_compute
    if "long_acc" in disabled:
        D._limit_accel = staticmethod(lambda linear, prev_linear, dt=0.1: linear)
    if "lat_acc" in disabled:
        G.MAX_LAT_ACCEL = float("inf")
    if "brake" in disabled:
        D._react_stop_dist = staticmethod(lambda speed: app_mod.REACT_STOP_DIST_M)
    if "approach" in disabled:
        # max(0.06, min(0.12, huge*rem)) == 0.12 while rem > 0, 0.06 when |lat| >= 0.15
        app_mod.APPROACH_K, app_mod.APPROACH_V_MIN, app_mod.APPROACH_V_MAX = 1e9, 0.06, 0.12
    if "halt" in disabled:
        D._halt = _old_halt
        D._approach_straight = _approach_with_old_wait


# ------------------------------------------------------------- instruments --
class _Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.peak_lat = 0.0          # actual |v*w| of the simulated chassis (m/s^2)
        self.peak_cmd_acc = 0.0      # largest commanded speed-up between nav ticks (m/s^2),
                                     # measured inside _navigate_to only (driving legs)
        self.in_nav = False
        self.brakes = 0
        self.stalls = 0
        self._last_v = None
        self._last_t = None


METER = _Meter()
_chassis_step = fake_client.Chassis.step
_send = fake_client.FakeMARCClient.send_cmd_vel


def _step(self, dt):
    _chassis_step(self, dt)
    METER.peak_lat = max(METER.peak_lat, abs(self.v * self.w))


def _send_cmd(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, twist=None):
    t = self.now_s()
    if not METER.in_nav:
        METER._last_t = None
        return _send(self, linear_x, linear_y, angular_z, twist)
    if METER._last_t is not None and t - METER._last_t >= 0.05:
        acc = (float(linear_x) - METER._last_v) / (t - METER._last_t)
        METER.peak_cmd_acc = max(METER.peak_cmd_acc, acc)
    METER._last_v, METER._last_t = float(linear_x), t
    return _send(self, linear_x, linear_y, angular_z, twist)


class _Count(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "-> braking" in msg:
            METER.brakes += 1
        if "stall detected" in msg or "brake held" in msg:
            METER.stalls += 1


_nav_orig = app_mod.DemoParticipant._navigate_to


def _nav_metered(self, *a, **kw):
    METER.in_nav, METER._last_t = True, None
    try:
        return _nav_orig(self, *a, **kw)
    finally:
        METER.in_nav = False


app_mod.DemoParticipant._navigate_to = _nav_metered
fake_client.Chassis.step = _step
fake_client.FakeMARCClient.send_cmd_vel = _send_cmd
logging.getLogger("participant_app").addHandler(_Count())
logging.getLogger("participant_app").setLevel(logging.INFO)


def _success(r):
    legs = r["legs"]
    return bool(r["error"] is None and not r["hit_cap"] and len(legs) >= 2
                and all(l["reached"] for l in legs) and r["capture"] and r["in_basket"]
                and r["task_complete"])


def run_config(disabled, hidden):
    apply(disabled)
    METER.reset()
    rows, _ = stage2_sim.run_execution(hidden_obstacles=hidden, progress=False)
    return {
        "total_s": round(sum(r["sim_s"] for r in rows), 1),
        "leg1_s": round(sum(r["legs"][0]["sim_s"] for r in rows if r["legs"]), 1),
        "leg2_s": round(sum(r["legs"][1]["sim_s"] for r in rows if len(r["legs"]) > 1), 1),
        "approach_s": round(sum(r["approach"].get("sim_s", 0.0) for r in rows), 1),
        "success": sum(_success(r) for r in rows),
        "n": len(rows),
        "contacts": sum(r["blocked_ticks"] for r in rows),
        "invalid_steer_ticks": sum(r["stall_ticks"] for r in rows),
        "brake_events": METER.brakes,
        "stall_replans": METER.stalls,
        "peak_lat_acc": round(METER.peak_lat, 3),
        "peak_cmd_acc": round(METER.peak_cmd_acc, 2),
        "per_problem_s": {r["pid"]: round(r["sim_s"], 1) for r in rows},
    }


def main():
    configs = [("full", [])] + [("-" + c, [c]) for c in COMPONENTS] + [("baseline", list(COMPONENTS))]
    out = {"components": COMPONENTS, "labels": LABELS, "configs": {}}
    for name, disabled in configs:
        out["configs"][name] = {"disabled": disabled,
                                "plain": run_config(disabled, hidden=False),
                                "hidden": run_config(disabled, hidden=True)}
        p, h = out["configs"][name]["plain"], out["configs"][name]["hidden"]
        print("%-10s plain %6.1fs (%d/%d)  hidden %6.1fs (%d/%d) contacts=%d brakes=%d "
              "peak_lat=%.2f peak_cmd_acc=%.1f" % (
                  name, p["total_s"], p["success"], p["n"], h["total_s"], h["success"], h["n"],
                  h["contacts"], h["brake_events"], max(p["peak_lat_acc"], h["peak_lat_acc"]),
                  max(p["peak_cmd_acc"], h["peak_cmd_acc"])), flush=True)
    apply([])
    path = os.path.join(_HERE, "runs", "stage2_drive_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
