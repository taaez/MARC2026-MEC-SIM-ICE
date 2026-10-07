"""Replay survey recordings through CurbLayer and score against the ground-truth height field."""
import sys, glob, json, math, os
import numpy as np
sys.path[:0] = ["/w/mec_sim_ice/demo_agents"]
import curb_layer as cl
import depth_ground_veto as dgv
runs = sys.argv[1:]
hf = np.load("/o/hf.npz"); Z, X0, Y0, R = hf["z"], float(hf["x0"]), float(hf["y0"]), float(hf["res"])
for k in ("STEP_UP_M", "STEP_DROP_M", "SLOPE_TOL", "NEIGH_CELLS", "MIN_OBS", "FLAT_M", "STEP_MID_M"):
    if k in os.environ:
        setattr(cl, k, type(getattr(cl, k))(os.environ[k]))
L = cl.CurbLayer()
for run in runs:
    for fn in sorted(glob.glob(os.path.join(run, "lidar_*.npz"))):
        z = np.load(fn, allow_pickle=True)
        for pose, mount, pts in zip(z["pose"], z["mount"], z["pts"]):
            wx, wy, zz = cl.lidar_world(pts.astype(np.float32), pose, mount)
            L.update(wx, wy, zz)
    nl = L.frames
    if os.environ.get("DEPTH", "1") == "1" and os.path.exists(os.path.join(run, "camera.json")):
        cam = json.load(open(os.path.join(run, "camera.json")))
        K = np.array(cam["K"]).reshape(3, 3).copy(); K[0, 0] /= 2; K[1, 1] /= 2; K[0, 2] /= 2; K[1, 2] /= 2
        Rm, t = np.array(cam["mount_R"]), np.array(cam["mount_t"])
        for fn in sorted(glob.glob(os.path.join(run, "cam_*.npz"))):
            z = np.load(fn, allow_pickle=True)
            for pose, dep in zip(z["pose"], z["depth"]):
                f, l, u = dgv.base_points(dep.astype(np.float32), K, Rm, t, stride=2)
                x, y, yaw = [float(v) for v in pose]; c, s = math.cos(yaw), math.sin(yaw)
                L.update(x + f * c - l * s, y + f * s + l * c, u)
    0 and print(run, "lidar frames", nl, "total frames", L.frames)
print(L.summary())
# ground truth at the layer's 0.2 m cells: same neighbour-step rule on the true surface
keys, up, dn = L._stats()
xy = L.cells_xy(keys)
def gz(x, y):
    j = np.floor((x - X0) / R).astype(int); i = np.floor((y - Y0) / R).astype(int)
    ok = (i >= 0) & (i < Z.shape[0]) & (j >= 0) & (j < Z.shape[1])
    o = np.full(len(x), np.nan); o[ok] = Z[i[ok], j[ok]]; return o
gt_up = np.full(len(xy), -np.inf); gt_dn = np.full(len(xy), np.inf)
z0 = gz(xy[:, 0], xy[:, 1])
for di in range(-3, 4):
    for dj in range(-3, 4):
        if di == dj == 0: continue
        d = math.hypot(di, dj) * 0.2
        if d > 0.6 + 1e-6: continue
        zn = gz(xy[:, 0] + dj * 0.2, xy[:, 1] + di * 0.2)
        dd = zn - z0
        gt_up = np.maximum(gt_up, np.where(np.isfinite(dd), dd - cl.SLOPE_TOL * d, -np.inf))
        gt_dn = np.minimum(gt_dn, np.where(np.isfinite(dd), dd + cl.SLOPE_TOL * d, np.inf))
has = np.isfinite(z0)
gt_step = has & ((gt_up >= cl.STEP_UP_M) | (gt_dn <= -cl.STEP_DROP_M))
gt_flat = has & (gt_up < cl.FLAT_M) & (gt_dn > -cl.FLAT_M)
sd = np.array(L.step_discs())
from scipy.spatial import cKDTree as _T
s_step = np.zeros(len(xy), bool)
if len(sd):
    _d, _ = _T(sd[:, :2]).query(xy); s_step = _d < 0.05
s_flat = (up < cl.FLAT_M) & (dn > -cl.FLAT_M)
def pr(pred, truth, name):
    tp = (pred & truth).sum(); p = pred.sum(); t = truth.sum()
    print(f"{name}: predicted {p}, truth {t}, precision {tp/max(p,1):.3f}, recall {tp/max(t,1):.3f}")
pr(s_step[has], gt_step[has], "STEP"); pr(s_flat[has], gt_flat[has], "FLAT")
# dangerous error: sensor says flat where truth has a step
print("flat-but-truth-step (dangerous):", int((s_flat & gt_step).sum()), "of", int(s_flat.sum()))
from scipy.spatial import cKDTree
pred_xy = xy[s_step]
if len(pred_xy):
    t = cKDTree(pred_xy)
    near_pred = np.array([len(v) > 0 for v in t.query_ball_point(xy, r=0.41)])
else:
    near_pred = np.zeros(len(xy), bool)
cov = (gt_step & near_pred).sum() / max(gt_step.sum(), 1)
danger = s_flat & gt_step & ~near_pred
print(f"curb coverage (truth step with a predicted step within 0.4 m): {cov:.3f}")
print(f"dangerous flat (truth step, no predicted step within 0.4 m): {int(danger.sum())} of {int(s_flat.sum())} flat ({danger.sum()/max(s_flat.sum(),1):.4f})")
