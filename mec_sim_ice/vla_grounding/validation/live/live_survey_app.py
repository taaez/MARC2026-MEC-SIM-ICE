"""Live curb survey -- wraps demo/participant_app.py, replaces the Stage 2 drive with a
road-edge survey that records lidar / depth / RGB / IMU for offline curb analysis.

Stage 1 runs as usual (the platform needs it to reach STAGE2_RUN). In STAGE2_RUN:
  1. save the occupancy map (map.npz)
  2. build survey waypoints from the map alone (nothing scene-specific): road cells
     SURVEY_OFFSET_M inside the road / off-road (-1) boundary, spaced SURVEY_SPACING_M,
     in the road component the robot stands in, optionally one angular sector of them
     around the robot (SURVEY_PART=k/N), ordered as a greedy nearest-neighbour tour
  3. drive the tour with the normal navigation (_navigate_to; use MARC_LOCAL_PLANNER=mppi,
     off-road stays lethal so the robot does not cross any kerb); at every waypoint turn to
     face the nearest boundary point and hold, so the forward depth camera sees the kerb
  4. task_complete when the budget runs out

Recorded (LIVE_OUT):
  map.npz            grid (int8, row 0 = bottom), origin, res
  survey_plan.json   waypoints + facing targets
  lidar_*.npz        merged lidar clouds (sensor frame, float16) + pose + mount offset
  cam_*.npz          depth (float16, stride 2), RGB (uint8, stride 4), pose, IMU roll/pitch,
                     flag "facing" = captured while held facing a boundary
  camera.json, telemetry.csv, events.jsonl (as live_stage2_app)

No kerb is crossed here -- crossing tests come after the heights are known.
"""

import json
import logging
import math
import os
import threading
import time

import numpy as np

import participant_app as app_mod
import lidar_obstacles as lo
from mock_agent_navigation import OccupancyGridPlanner
from live_stage2_app import _EventLog, _telemetry

log = logging.getLogger("survey")

BUDGET_S = float(os.environ.get("SURVEY_BUDGET_S", "560"))   # sim s (Stage 2 limit is 600)
OFFSET_M = float(os.environ.get("SURVEY_OFFSET_M", "1.5"))
SPACING_M = float(os.environ.get("SURVEY_SPACING_M", "6.0"))
WP_BUDGET_S = 45.0
LIDAR_EVERY = 2            # keep every Nth merged cloud the obstacle layer receives (~2.5 Hz)
CAM_PERIOD_S = 1.0         # sim s between camera captures while driving
CHUNK = 60                 # frames per npz chunk
CELL_M = 0.2


def _part():
    raw = os.environ.get("SURVEY_PART", "0/1")
    k, n = raw.split("/")
    return int(k), int(n)


# ---------------------------------------------------------------- planning --
def plan_survey(grid, origin, res, robot_xy):
    from scipy import ndimage
    f = max(1, int(round(CELL_M / res)))
    h, w = grid.shape[0] // f, grid.shape[1] // f
    sub = grid[:h * f, :w * f].reshape(h, f, w, f)
    road = (sub == 0).mean(axis=(1, 3)) > 0.5
    unknown = (sub < 0).mean(axis=(1, 3)) > 0.5
    ox, oy = origin

    def ij(x, y):
        return int((y - oy) / (res * f)), int((x - ox) / (res * f))

    def xy(i, j):
        return ox + (j + 0.5) * res * f, oy + (i + 0.5) * res * f

    lab, _ = ndimage.label(road)
    ri, rj = ij(*robot_xy)
    comp_id = lab[min(max(ri, 0), h - 1), min(max(rj, 0), w - 1)]
    if comp_id == 0:   # robot cell not road: nearest road component
        ii, jj = np.nonzero(road)
        k = int(np.argmin((ii - ri) ** 2 + (jj - rj) ** 2))
        comp_id = lab[ii[k], jj[k]]
    comp = lab == comp_id
    # boundary = component cells touching off-road
    near_unknown = ndimage.binary_dilation(unknown, structure=np.ones((3, 3), bool))
    boundary = comp & near_unknown
    dist = ndimage.distance_transform_edt(comp) * CELL_M
    band = comp & (np.abs(dist - OFFSET_M) <= CELL_M)
    bi, bj = np.nonzero(boundary)
    ci, cj = np.nonzero(band)
    # spread candidates: greedy, keep a cell only if >= SPACING_M from every kept one
    order = np.random.default_rng(0).permutation(len(ci))
    kept = []
    for k in order:
        p = np.array(xy(ci[k], cj[k]))
        if all(np.hypot(*(p - q)) >= SPACING_M for q in kept):
            kept.append(p)
    kept = np.array(kept) if kept else np.zeros((0, 2))
    seen_path = os.environ.get("SURVEY_SEEN", "")
    if seen_path and os.path.exists(seen_path) and len(kept):
        # skip waypoints whose stretch of boundary an earlier survey already measured
        cells = json.load(open(seen_path))["cells"]
        sxy = np.array([[c["x"], c["y"]] for c in cells]) if cells else np.zeros((0, 2))
        if len(sxy):
            from scipy.spatial import cKDTree
            tree = cKDTree(sxy)
            n_seen = np.array([len(v) for v in tree.query_ball_point(kept, r=SPACING_M / 2)])
            kept = kept[n_seen < 5]
    part, nparts = _part()
    if nparts > 1 and len(kept):
        ang = np.arctan2(kept[:, 1] - robot_xy[1], kept[:, 0] - robot_xy[0])
        sec = ((ang + math.pi) / (2 * math.pi) * nparts).astype(int) % nparts
        kept = kept[sec == part]
    # greedy nearest-neighbour tour by ROAD distance (Dijkstra on the road component), not
    # straight-line: survey_rtC2_p2 picked a waypoint 35 m away in a straight line that was an
    # ~90 m detour by road
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    idx = -np.ones((h, w), np.int64)
    ii, jj = np.nonzero(comp)
    idx[ii, jj] = np.arange(len(ii))
    rows, cols, wts = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a_ = idx[max(0, -di):h - max(0, di), max(0, -dj):w - max(0, dj)]
        b_ = idx[max(0, di):h + min(0, di) or None, max(0, dj):w + min(0, dj) or None]
        m = (a_ >= 0) & (b_ >= 0)
        step = CELL_M * (math.sqrt(2.0) if di and dj else 1.0)
        rows += [a_[m], b_[m]]
        cols += [b_[m], a_[m]]
        wts += [np.full(m.sum(), step)] * 2
    G = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(ii), len(ii)))

    def node(p):
        i, j = ij(*p)
        i, j = min(max(i, 0), h - 1), min(max(j, 0), w - 1)
        if idx[i, j] >= 0:
            return idx[i, j]
        k = int(np.argmin((ii - i) ** 2 + (jj - j) ** 2))
        return k

    knodes = np.array([node(p) for p in kept], dtype=np.int64)
    tour, cur, left = [], node(robot_xy), list(range(len(kept)))
    while left:
        d = dijkstra(G, indices=int(cur))[knodes[left]]
        k = left[int(np.argmin(d))]
        left.remove(k)
        tour.append(kept[k])
        cur = knodes[k]
    # facing target: nearest boundary cell to each waypoint
    bxy = np.array([xy(i, j) for i, j in zip(bi, bj)]) if len(bi) else np.zeros((0, 2))
    plan = []
    for p in tour:
        if len(bxy):
            q = bxy[int(np.argmin(np.hypot(bxy[:, 0] - p[0], bxy[:, 1] - p[1])))]
        else:
            q = p
        plan.append({"wp": [float(p[0]), float(p[1])], "face": [float(q[0]), float(q[1])]})
    stats = {"road_cells": int(comp.sum()), "boundary_cells": int(boundary.sum()),
             "boundary_len_m": float(boundary.sum() * CELL_M), "candidates": int(len(kept)),
             "tour_len_m": float(sum(np.hypot(*(np.array(plan[i + 1]["wp"]) - plan[i]["wp"]))
                                     for i in range(len(plan) - 1))) if len(plan) > 1 else 0.0,
             "part": "%d/%d" % _part()}
    return plan, stats


# ---------------------------------------------------------------- recording --
class Recorder:
    def __init__(self, out, client):
        self.out, self.c = out, client
        self.lock = threading.Lock()
        self.lidar, self.cam = [], []
        self.n_lidar = self.n_cam = 0
        self.k_lidar = self.k_cam = 0
        self.facing = False
        self.app = None

    def on_lidar(self, msg, pose, mount):
        self.n_lidar += 1
        if self.n_lidar % LIDAR_EVERY:
            return
        pts = lo.lidar_xyz(msg)
        if pts is None or pose is None:
            return
        with self.lock:
            self.lidar.append((self.c.now_s(), np.asarray(pose, np.float32),
                               np.asarray(mount, np.float32), pts.astype(np.float16)))
            if len(self.lidar) >= CHUNK:
                self._flush_lidar()

    def _flush_lidar(self):
        if not self.lidar:
            return
        np.savez_compressed(os.path.join(self.out, "lidar_%03d.npz" % self.k_lidar),
                            st=np.array([r[0] for r in self.lidar]),
                            pose=np.stack([r[1] for r in self.lidar]),
                            mount=np.stack([r[2] for r in self.lidar]),
                            pts=np.array([r[3] for r in self.lidar], dtype=object))
        self.k_lidar += 1
        self.lidar = []

    def capture_cam(self, facing=False):
        c = self.c
        pose = c.get_world_pose()
        dm = c.get_robot_depth("base")
        im = c.get_robot_image("base_left")
        if pose is None or dm is None:
            return False
        depth = np.frombuffer(dm.data, np.float32).reshape(dm.height, dm.width)[::2, ::2]
        rgb = None
        if im is not None:
            ch = 3 if im.encoding in ("rgb8", "bgr8") else 4 if im.encoding in ("rgba8", "bgra8") else 0
            if ch:
                a = np.frombuffer(im.data, np.uint8).reshape(im.height, im.width, ch)[::4, ::4, :3]
                rgb = a[..., ::-1] if im.encoding.startswith("bgr") else a
        rp = app_mod.imu_roll_pitch(c.get_imu())
        with self.lock:
            self.cam.append((c.now_s(), np.asarray(pose, np.float32), depth.astype(np.float16),
                             rgb, bool(facing), np.asarray(rp if rp else (np.nan, np.nan), np.float32)))
            if len(self.cam) >= CHUNK:
                self._flush_cam()
        return True

    def _flush_cam(self):
        if not self.cam:
            return
        rgbs = [r[3] for r in self.cam]
        np.savez_compressed(os.path.join(self.out, "cam_%03d.npz" % self.k_cam),
                            st=np.array([r[0] for r in self.cam]),
                            pose=np.stack([r[1] for r in self.cam]),
                            depth=np.stack([r[2] for r in self.cam]),
                            rgb=np.array(rgbs, dtype=object),
                            facing=np.array([r[4] for r in self.cam]),
                            rollpitch=np.stack([r[5] for r in self.cam]))
        self.k_cam += 1
        self.cam = []

    def cam_loop(self, stop):
        last = -1e9
        while not stop.is_set():
            try:
                t = self.c.now_s()
                if t - last >= CAM_PERIOD_S:
                    if self.capture_cam(False):
                        last = t
            except Exception as e:
                log.debug("cam: %s", e)
            stop.wait(0.05)

    def flush(self):
        with self.lock:
            self._flush_lidar()
            self._flush_cam()


def install_lidar_tap(rec):
    orig = lo.LidarObstacleLayer.update

    def update(self, msg, pose, mount_offset=(0.0, 0.0)):
        n = orig(self, msg, pose, mount_offset)
        try:
            rec.on_lidar(msg, pose, mount_offset)
        except Exception as e:
            log.debug("lidar tap: %s", e)
        return n
    lo.LidarObstacleLayer.update = update


# ---------------------------------------------------------------- driving --
def face_point(app, target, max_s=4.0):
    c = app.client
    t_end = app._clock.tick() + max_s
    while c.is_running and app._clock.tick() < t_end:
        p = c.get_world_pose()
        if p is None:
            c.sleep(0.1)
            continue
        err = math.atan2(math.sin(math.atan2(target[1] - p[1], target[0] - p[0]) - p[2]),
                         math.cos(math.atan2(target[1] - p[1], target[0] - p[0]) - p[2]))
        if abs(err) < 0.08:
            break
        c.send_cmd_vel(0.0, max(-1.0, min(1.0, 1.5 * err)))
        c.sleep(0.1)
    app._halt()


def survey_drive(app, rec, out):
    c = app.client
    app._clock = app_mod.ElapsedClock(c)
    app._lidar_layer = lo.LidarObstacleLayer()
    app._lidar_acc = app._lidar_accumulator()
    app._lidar_mount_xy = app._lidar_mount_offset()
    app._cam_mount_R, app._cam_mount_t = app._camera_mount_transform()
    app._stow_arm()

    planner = OccupancyGridPlanner()
    while c.is_running and not planner.ready:
        occ = c.get_occupancy_map()
        if occ is not None:
            planner.update_from_msg(occ)
        else:
            c.sleep(0.2)
    np.savez_compressed(os.path.join(out, "map.npz"), grid=planner._grid,
                        origin=np.array([planner._ox, planner._oy]), res=planner._res)
    pose = None
    while pose is None and c.is_running:
        pose = c.get_world_pose()
        c.sleep(0.1)
    plan, stats = plan_survey(planner._grid, (planner._ox, planner._oy), planner._res, pose[:2])
    with open(os.path.join(out, "survey_plan.json"), "w") as f:
        json.dump({"stats": stats, "plan": plan, "start": list(pose)}, f, indent=1)
    log.info("[survey] plan %s", stats)

    for i, step in enumerate(plan):
        if not c.is_running or app._clock.tick() > BUDGET_S:
            break
        # budget grows with the distance: after skipping already-surveyed stretches the first
        # waypoint can be ~40 m away (survey_rtC2_p2: a flat 45 s ran out 13 times in a row)
        p0 = c.get_world_pose()
        far = math.hypot(step["wp"][0] - p0[0], step["wp"][1] - p0[1]) if p0 else 0.0
        dl = min(app._clock.tick() + max(WP_BUDGET_S, 2.5 * far), BUDGET_S)
        ok = app._navigate_to(tuple(step["wp"]), dl, tol=1.0)
        p = c.get_world_pose()
        log.info("[survey] wp %d/%d %s reached=%s pose=%s t=%.0f", i + 1, len(plan), step["wp"], ok,
                 None if p is None else tuple(round(v, 2) for v in p), app._clock.tick())
        if not ok:
            continue
        face_point(app, step["face"])
        c.sleep(0.5)
        rec.capture_cam(facing=True)
    c.send_cmd_vel(0.0, 0.0)
    rec.flush()
    log.info("[survey] done t=%.0f", app._clock.tick())
    c.task_complete()


def main():
    out = os.environ.get("LIVE_OUT", "./live_out")
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    evlog = _EventLog(os.path.join(out, "events.jsonl"))
    logging.getLogger().addHandler(evlog)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"survey": True, "budget_s": BUDGET_S, "offset_m": OFFSET_M,
                   "spacing_m": SPACING_M, "part": "%d/%d" % _part(),
                   "planner": app_mod.LOCAL_PLANNER, "started": time.time()}, f, indent=1)

    app = app_mod.DemoParticipant()
    evlog.sim_now = app.client.now_s
    rec = Recorder(out, app.client)
    install_lidar_tap(rec)
    stop = threading.Event()

    def drive(*a, **kw):
        th = threading.Thread(target=_telemetry, daemon=True,
                              args=(app.client, os.path.join(out, "telemetry.csv"), stop, app, out))
        th.start()
        cam_th = threading.Thread(target=rec.cam_loop, args=(stop,), daemon=True)
        cam_th.start()
        try:
            survey_drive(app, rec, out)
        except Exception:
            log.exception("[survey] failed")
        finally:
            stop.set()
            rec.flush()

    app._drive = drive
    app.client.on_stage2_run(drive)
    app.run()


if __name__ == "__main__":
    main()
