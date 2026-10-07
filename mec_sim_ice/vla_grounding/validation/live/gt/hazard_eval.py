import sys, glob, math, time, types, numpy as np
sys.path[:0] = ["/w/mec_sim_ice/demo_agents", "/w/demo"]
import curb_layer as cl
from scipy.spatial import cKDTree
from scipy import ndimage
run = "/w/mec_sim_ice/vla_grounding/validation/live/runs/survey_rtC2_p1"
hf = np.load("/o/hf.npz"); Z, X0, Y0, R = hf["z"], float(hf["x0"]), float(hf["y0"]), float(hf["res"])
L = cl.CurbLayer()
for fn in sorted(glob.glob(run + "/lidar_*.npz")):
    z = np.load(fn, allow_pickle=True)
    for pose, mount, pts in zip(z["pose"], z["mount"], z["pts"]):
        L.update(*cl.lidar_world(pts.astype(np.float32), pose, mount))
m = np.load(run + "/map.npz")
planner = types.SimpleNamespace(_grid=m["grid"], _res=float(m["res"]), _ox=float(m["origin"][0]), _oy=float(m["origin"][1]))
import importlib.util
t0 = time.time()
g, res = planner._grid, planner._res
edge = (g == 0) & ndimage.binary_dilation(g < 0)
zone = ndimage.distance_transform_edt(~edge) * res <= 1.5
def near_edge(xy):
    j = ((xy[:, 0] - planner._ox) / res).astype(int); i = ((xy[:, 1] - planner._oy) / res).astype(int)
    return zone[i, j]
print("edge zone build %.2f s" % (time.time() - t0))
t0 = time.time(); hz = L.hazard_xy(near_edge=near_edge); print("hazard_xy %.1f ms, %d cells (all step cells %d)" % (1000 * (time.time() - t0), len(hz), int(L._is_step[:L._n].sum())))
# truth: 5 cm jumps >= 6 cm (kerbs) and >= 10 cm
def truth(th):
    dx = np.abs(np.diff(Z, axis=1)); dy = np.abs(np.diff(Z, axis=0))
    a, b = np.nonzero(np.nan_to_num(dx) >= th); c, d = np.nonzero(np.nan_to_num(dy) >= th)
    return np.r_[np.c_[X0 + (b + 1) * R, Y0 + (a + .5) * R], np.c_[X0 + (d + .5) * R, Y0 + (c + 1) * R]]
for th in (0.06, 0.10):
    T = truth(th); dist, _ = cKDTree(T).query(hz)
    print(f"hazard precision vs truth jumps >= {th*100:.0f} cm within 0.3 m: {np.mean(dist < 0.3):.3f}")
# coverage of truth >=10 cm jumps that the survey observed (near a confirmed cell)
keys, up, dn = L._stats(); obs = L.cells_xy(keys); To = truth(0.10)
seen = cKDTree(obs).query(To)[0] < 0.2
cov = cKDTree(hz).query(To[seen])[0] < 0.4
print(f"truth >= 10 cm jumps in observed area: {seen.sum()}, covered by a hazard within 0.4 m: {cov.mean():.3f}")
