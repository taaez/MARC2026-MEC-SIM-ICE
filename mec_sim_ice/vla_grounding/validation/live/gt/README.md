# Scene ground-truth kerb analysis (analysis only -- the agent never reads any of this)

1. `heightfield.py` -- run in the platform image with the USD libs on the path:
   ```
   L=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
   docker run --rm -e PYTHONPATH=$L -e LD_LIBRARY_PATH=$L/bin:$L/lib -v $PWD:/gt \
     --entrypoint /isaac-sim/python.sh marc-platform:practice /gt/heightfield.py
   ```
   -> `hf.npz` (0.05 m top-surface height + surface kind of `/Root/S1/_2_Site`, ~9 MB, not committed)
2. `gt_analysis.py hf.npz <survey run>/map.npz [<survey run>/curb_report.json]` (agent image; writes to /o)
   -> `gt_curb_report.json`: boundary step stats, steps per surface pair, curb-free reach, sensor vs truth
3. `replay_eval.py <survey runs>` / `hazard_eval.py` -- replay recorded lidar/depth through `curb_layer.py`
   and score step / flat / hazard cells against `hf.npz` (expects /w = repo, /o = this folder).
