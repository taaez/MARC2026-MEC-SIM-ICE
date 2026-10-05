"""Live (Isaac Sim) Stage2 ablation entry point -- wraps demo/participant_app.py.

Runs the real participant app against the platform, with post-qualifier components
switched off by the same reverts as the offline harness (stage2_drive_ablation.apply()),
and records telemetry for the Isaac_Drive / Isaac_Grasp sheets.

Env:
  ABL_OFF     comma list of drive components to revert: speed,profile,long_acc,lat_acc,
              brake,approach,halt  (or "all" = qualifier baseline)
  GRASP_OFF   comma list of grasp_shape switches to turn off: USE_CLUSTER,USE_YAW,...
              (or "all" = qualifier mean-of-points grasp + yaw 0.52)
  LIVE_OUT    output dir (events.jsonl, telemetry.csv, config.json)

The revert functions are copied from stage2_drive_ablation.py, which cannot be imported
here: it installs the offline fake-client shims at import time.
"""

import csv
import json
import logging
import math
import os
import threading
import time

import mock_agent_navigation as nav
import participant_app as app_mod
import grasp_shape

COMPONENTS = ["speed", "profile", "long_acc", "lat_acc", "brake", "approach", "halt"]
GRASP_SWITCHES = ["USE_CLUSTER", "USE_TOP_BAND_CENTER", "USE_LYING_AXIS", "USE_YAW"]

_ORIG_APPROACH = app_mod.DemoParticipant._approach_straight


def _old_gs_compute(self, pose, goal, cruise=False, remaining=None):
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
    for _ in range(5):
        self.client.send_cmd_vel(0.0, 0.0)
        self.client.sleep(0.05)
    return False


def _approach_with_old_wait(self, *a, **kw):
    r = _ORIG_APPROACH(self, *a, **kw)
    for _ in range(5):
        self.client.send_cmd_vel(0.0, 0.0)
        self.client.sleep(0.1)
    return r


def apply_drive(disabled):
    G, D = nav.GoalSeekController, app_mod.DemoParticipant
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
        app_mod.APPROACH_K, app_mod.APPROACH_V_MIN, app_mod.APPROACH_V_MAX = 1e9, 0.06, 0.12
    if "halt" in disabled:
        D._halt = _old_halt
        D._approach_straight = _approach_with_old_wait


def apply_grasp(disabled):
    for s in disabled:
        setattr(grasp_shape, s, False)


def _parse(env, universe):
    raw = os.environ.get(env, "").strip()
    if raw == "all":
        return list(universe)
    out = [s.strip() for s in raw.split(",") if s.strip()]
    bad = [s for s in out if s not in universe]
    if bad:
        raise SystemExit("%s: unknown %s (choose from %s)" % (env, bad, universe))
    return out


class _EventLog(logging.Handler):
    """Every log record -> events.jsonl with wall time (joins with telemetry.csv)."""

    def __init__(self, path):
        super().__init__()
        self.f = open(path, "a", buffering=1)
        self.sim_now = None   # set to client.now_s once the client exists

    def emit(self, record):
        try:
            st = self.sim_now() if self.sim_now else None
            self.f.write(json.dumps({"t": record.created, "st": st, "lvl": record.levelname,
                                     "name": record.name, "msg": record.getMessage()},
                                    ensure_ascii=False) + "\n")
        except Exception:
            pass


def _quat_rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _dump_camera(app, out):
    """Once both are known: base depth camera_info + the camera mount the app resolved
    (for grasp_shape_eval.py MOUNTS / HFOV)."""
    info, R, t = app._depth_cam_info, app._cam_mount_R, app._cam_mount_t
    if info is None or R is None or t is None:
        return False
    import numpy as np
    K = list(info.k)
    d = {"width": info.width, "height": info.height, "K": K, "frame": info.header.frame_id,
         "hfov_deg": math.degrees(2 * math.atan2(info.width / 2.0, K[0])),
         "mount_R": np.asarray(R).tolist(), "mount_t": np.asarray(t).tolist()}
    with open(os.path.join(out, "camera.json"), "w") as f:
        json.dump(d, f, indent=1)
    logging.getLogger("live").info("[live] camera %s", d)
    return True


def _telemetry(client, path, stop, app=None, out=None):
    """20 Hz: world pose, odom twist, IMU roll/pitch + linear accel."""
    cam_done = app is None
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "st", "x", "y", "yaw", "odom_vx", "odom_wz",
                    "roll", "pitch", "ax", "ay", "az", "gx", "gy", "gz"])
        while not stop.is_set():
            row = [time.time(), client.now_s()]
            try:
                p = client.get_world_pose()
                row += list(p) if p else [None] * 3
                o = client.get_odom()
                row += ([o.twist.twist.linear.x, o.twist.twist.angular.z] if o else [None] * 2)
                m = client.get_imu()
                if m is not None:
                    r, pi, _ = _quat_rpy(m.orientation)
                    a, g = m.linear_acceleration, m.angular_velocity
                    row += [r, pi, a.x, a.y, a.z, g.x, g.y, g.z]
                else:
                    row += [None] * 8
                w.writerow(row)
            except Exception as e:
                logging.getLogger("live").debug("telemetry: %s", e)
            if not cam_done:
                try:
                    cam_done = _dump_camera(app, out)
                except Exception as e:
                    logging.getLogger("live").warning("[live] camera dump: %s", e)
                    cam_done = True
            stop.wait(0.05)


def install_lidar_debug(out, sim_now):
    """LIDAR_DEBUG=1: per scan, what the forward cone holds (layer-band points vs points too
    low for the band, i.e. kerb candidates); per tick, the brake decision with and without
    the goal exclusion; every 5th raw scan + pose saved to lidar_scans.npz."""
    import numpy as np
    import lidar_obstacles as lo
    f = open(os.path.join(out, "lidar_dbg.jsonl"), "a", buffering=1)
    scans, state = [], {"n": 0}
    orig_update, orig_danger = lo.LidarObstacleLayer.update, lo.LidarObstacleLayer.danger_ahead

    def rel_height(pts):
        # same sector-wise linear ground fit as LidarObstacleLayer.update
        gz = float(np.percentile(pts[:, 2], lo.GROUND_PERCENTILE))
        az = np.arctan2(pts[:, 1], pts[:, 0])
        sec = np.clip(((az + math.pi) / (2 * math.pi) * lo.GROUND_SECTORS).astype(np.int64),
                      0, lo.GROUND_SECTORS - 1)
        rng = np.hypot(pts[:, 0], pts[:, 1])
        ground = np.full(len(pts), gz)
        for k in range(lo.GROUND_SECTORS):
            m = sec == k
            if m.sum() < lo.GROUND_MIN_BIN_PTS:
                continue
            zs, rs = pts[m, 2], rng[m]
            rough = zs <= np.percentile(zs, lo.GROUND_PERCENTILE) + lo.GROUND_ROUGH_BAND_M
            if rough.sum() < lo.GROUND_MIN_BIN_PTS:
                continue
            a, b = np.polyfit(rs[rough], zs[rough], 1)
            if abs(a) <= lo.GROUND_MAX_SLOPE:
                ground[m] = a * rs + b
        return pts[:, 2] - ground, az, rng

    def update(self, msg, pose, mount_offset=(0.0, 0.0)):
        n = orig_update(self, msg, pose, mount_offset)
        try:
            pts = lo.lidar_xyz(msg)
            if pts is not None and pose is not None:
                zr, az, rng = rel_height(pts)
                fwd = (np.abs(np.degrees(az)) < 40) & (rng < 3.0)
                band = fwd & (zr > lo.GROUND_BAND_MIN_M) & (zr < lo.GROUND_BAND_MAX_M)
                low = fwd & (zr > 0.04) & (zr <= lo.GROUND_BAND_MIN_M)
                f.write(json.dumps({"k": "scan", "st": sim_now(), "pose": list(pose),
                    "fwd_band_n": int(band.sum()),
                    "fwd_band_minr": float(rng[band].min()) if band.any() else None,
                    "fwd_low_n": int(low.sum()),
                    "fwd_low_minr": float(rng[low].min()) if low.any() else None,
                    "fwd_low_maxz": float(zr[low].max()) if low.any() else None,
                    "fwd_ground_minr": float(rng[fwd].min()) if fwd.any() else None}) + "\n")
                state["n"] += 1
                if state["n"] % 5 == 0:
                    scans.append((sim_now(), np.asarray(pose, np.float32), pts.astype(np.float16)))
        except Exception as e:
            logging.getLogger("live").debug("lidar dbg: %s", e)
        return n

    def danger_ahead(self, pose, max_dist, cone_deg=70.0, goal=None, goal_exclude_m=0.0):
        r = orig_danger(self, pose, max_dist, cone_deg, goal, goal_exclude_m)
        try:
            raw = orig_danger(self, pose, max_dist, cone_deg, None, 0.0)
            if raw or r:
                f.write(json.dumps({"k": "brake", "st": sim_now(), "pose": list(pose),
                    "stop_dist": max_dist, "fired": bool(r), "fired_without_exclusion": bool(raw),
                    "goal": list(goal) if goal is not None else None,
                    "exclude_m": goal_exclude_m}) + "\n")
        except Exception:
            pass
        return r

    lo.LidarObstacleLayer.update = update
    lo.LidarObstacleLayer.danger_ahead = danger_ahead

    def dump():
        if scans:
            np.savez_compressed(os.path.join(out, "lidar_scans.npz"),
                                st=np.array([s[0] for s in scans]),
                                pose=np.stack([s[1] for s in scans]),
                                pts=np.array([s[2] for s in scans], dtype=object))
    return dump


def main():
    out = os.environ.get("LIVE_OUT", "./live_out")
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    evlog = _EventLog(os.path.join(out, "events.jsonl"))
    logging.getLogger().addHandler(evlog)

    drive_off = _parse("ABL_OFF", COMPONENTS)
    grasp_off = _parse("GRASP_OFF", GRASP_SWITCHES)
    apply_drive(drive_off)
    apply_grasp(grasp_off)
    cfg = {"drive_off": drive_off, "grasp_off": grasp_off,
           "MAX_LINEAR": nav.GoalSeekController.MAX_LINEAR,
           "grasp_switches": {s: getattr(grasp_shape, s) for s in GRASP_SWITCHES},
           "started": time.time()}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    logging.getLogger("live").info("[live] config %s", cfg)

    app = app_mod.DemoParticipant()
    evlog.sim_now = app.client.now_s
    orig_drive = app._drive
    stop = threading.Event()

    def drive_with_telemetry(*a, **kw):
        th = threading.Thread(target=_telemetry, daemon=True,
                              args=(app.client, os.path.join(out, "telemetry.csv"), stop, app, out))
        th.start()
        logging.getLogger("live").info("[live] drive start")
        try:
            return orig_drive(*a, **kw)
        finally:
            logging.getLogger("live").info("[live] drive end")
            stop.set()

    # _drive was bound as the stage2_run callback in __init__ -- rebind it there.
    app._drive = drive_with_telemetry
    app.client.on_stage2_run(drive_with_telemetry)
    if os.environ.get("LIDAR_DEBUG") == "1":
        dump = install_lidar_debug(out, app.client.now_s)
        import atexit
        atexit.register(dump)
        _orig = drive_with_telemetry

        def drive_dbg(*a, **kw):
            try:
                return _orig(*a, **kw)
            finally:
                dump()
        app._drive = drive_dbg
        app.client.on_stage2_run(drive_dbg)
    app.run()


if __name__ == "__main__":
    main()
