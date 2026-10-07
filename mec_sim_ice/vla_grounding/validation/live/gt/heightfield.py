"""Rasterize chungmu ground meshes (/Root/S1/_2_Site) into a top-surface height field.
Output /gt/hf.npz: z (H,W) float32 m (nan = no ground), kind (H,W) int8, x0, y0, res, kinds."""
from pxr import Usd, UsdGeom
import numpy as np

RES = 0.05
st = Usd.Stage.Open("/metacom2026/resources/playgrounds/chungmu/scene.usd")
cache = UsdGeom.XformCache()
KINDS = ["CarRoad", "SideWalk", "Grass", "Ground", "other"]
tris = []
for p in st.Traverse():
    path = str(p.GetPath())
    if not (p.IsA(UsdGeom.Mesh) and "/_2_Site" in path):
        continue
    m = UsdGeom.Mesh(p)
    pts = m.GetPointsAttr().Get()
    cnt = m.GetFaceVertexCountsAttr().Get()
    idx = m.GetFaceVertexIndicesAttr().Get()
    if not pts or not cnt or not idx:
        continue
    pts = np.array(pts, float)
    M = np.array(cache.GetLocalToWorldTransform(p))
    w = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3] / 100.0
    kind = next((i for i, k in enumerate(KINDS[:-1]) if k in path), len(KINDS) - 1)
    idx = np.array(idx)
    o = 0
    for c in cnt:                       # fan-triangulate polygons
        f = idx[o:o + c]
        for t in range(1, c - 1):
            tris.append((w[f[0]], w[f[t]], w[f[t + 1]], kind))
        o += c
print("triangles", len(tris))
allp = np.array([v for t in tris for v in t[:3]])
x0, y0 = allp[:, 0].min(), allp[:, 1].min()
W = int((allp[:, 0].max() - x0) / RES) + 2
H = int((allp[:, 1].max() - y0) / RES) + 2
Z = np.full((H, W), -np.inf, np.float32)
K = np.full((H, W), -1, np.int8)
for a, b, c, kind in tris:
    xs, ys = np.array([a[0], b[0], c[0]]), np.array([a[1], b[1], c[1]])
    j0, j1 = int((xs.min() - x0) / RES), int((xs.max() - x0) / RES) + 1
    i0, i1 = int((ys.min() - y0) / RES), int((ys.max() - y0) / RES) + 1
    jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
    px, py = x0 + (jj + 0.5) * RES, y0 + (ii + 0.5) * RES
    d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(d) < 1e-12:
        continue
    l1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d
    l2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d
    l3 = 1 - l1 - l2
    inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
    if not inside.any():
        continue
    z = l1 * a[2] + l2 * b[2] + l3 * c[2]
    ii, jj, z = ii[inside], jj[inside], z[inside]
    ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
    ii, jj, z = ii[ok], jj[ok], z[ok]
    upd = z > Z[ii, jj]
    Z[ii[upd], jj[upd]] = z[upd]
    K[ii[upd], jj[upd]] = kind
Z[~np.isfinite(Z)] = np.nan
np.savez_compressed("/gt/hf.npz", z=Z, kind=K, x0=x0, y0=y0, res=RES, kinds=np.array(KINDS))
print("grid", Z.shape, "covered", float(np.isfinite(Z).mean()),
      {k: int((K == i).sum()) for i, k in enumerate(KINDS)})
