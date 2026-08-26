"""
tools/eval_offline.py  --  MARC_RECORD_DIR 로 덤프한 라운드를 Isaac Sim 없이 재실행한다.

사용 (리포 루트에서):
  python mec_sim_ice/tools/eval_offline.py --records ./mec_sim_ice/records         # 파이프라인 재실행, payload 비교
  python mec_sim_ice/tools/eval_offline.py --records ./mec_sim_ice/records --gt demo/mock_demo_data.yaml
        # mock 데이터(GT+노이즈)와 해석/좌표를 비교. lf/hf 쌍 교차로 노이즈 의심 항목은 표시만 한다.
  python mec_sim_ice/tools/eval_offline.py --records ./mec_sim_ice/records --dry   # 모델 로드 없이 기하/파서만 검사

기록 구조 (agent_vla._record):
  mec_sim_ice/records/s1_001_<ts>/{<camera_id>.png ..., meta.json{command, cameras{K,R_cw,C_w,ground_z}, payload}}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage1.geometry import CameraModel                      # noqa: E402
from stage1.pipeline import CameraObservation, RuleParser    # noqa: E402


def load_record(d: str):
    from PIL import Image
    meta = json.load(open(os.path.join(d, "meta.json")))
    obs = []
    for cid, c in meta["cameras"].items():
        p = os.path.join(d, f"{cid}.png")
        if not os.path.exists(p):
            continue
        cam = CameraModel(cid, np.array(c["K"]), np.array(c["R_cw"]), np.array(c["C_w"]),
                          float(c["ground_z"]), int(c["width"]), int(c["height"]))
        obs.append(CameraObservation(cam, np.array(Image.open(p).convert("RGB"))))
    return meta, obs


def load_gt(path: str) -> dict:
    import yaml
    d = yaml.safe_load(open(path))
    return {k.strip().lower(): v for k, v in (d.get("stage1_answers") or {}).items()}


def compare(payload: dict, gt: dict) -> dict:
    pi, gi = payload["interpretation"], gt["interpretation"]
    pa, pt = np.array(payload["grounding"]["anchor_coord"]), np.array(payload["grounding"]["target_coord"])
    ga, gtc = np.array(gt["grounding"]["anchor_coord"]), np.array(gt["grounding"]["target_coord"])
    return {
        "camera": payload["camera_id"] == gt["camera_id"],
        "target_type": pi.get("target_type") == gi.get("target_type"),
        "landmark": pi.get("landmark") == gi.get("landmark"),
        "relation": pi.get("relation") == gi.get("relation"),
        "situation": pi.get("situation") == gi.get("situation"),
        "anchor_err_xy": float(np.linalg.norm((pa - ga)[:2])), "anchor_err_z": float(abs(pa[2] - ga[2])),
        "target_err_xy": float(np.linalg.norm((pt - gtc)[:2])), "target_err_z": float(abs(pt[2] - gtc[2])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--time-limit", type=float, default=60.0)
    a = ap.parse_args()

    gt = load_gt(a.gt) if a.gt else {}
    pipe = None
    if not a.dry:
        from stage1.pipeline import build_pipeline
        pipe = build_pipeline()
    rule = RuleParser()

    rows = []
    for d in sorted(glob.glob(os.path.join(a.records, "s1_*"))):
        meta, obs = load_record(d)
        cmd = meta["command"]
        print(f"\n=== {os.path.basename(d)} | {cmd}")
        if a.dry:
            print("  rule:", rule.parse(cmd))
            print("  recorded payload:", json.dumps(meta["payload"]))
            payload = meta["payload"]
            dt = 0.0
        else:
            t0 = time.time()
            payload = pipe.run(cmd, obs, time_limit_s=a.time_limit)
            dt = time.time() - t0
            print(f"  new payload ({dt:.1f}s):", json.dumps(payload))
        g = gt.get(cmd.strip().lower())
        if g:
            r = compare(payload, g)
            r["cmd"], r["dt"] = cmd[:50], dt
            rows.append(r)
            flags = " ".join(k for k in ("camera", "target_type", "landmark", "relation", "situation") if not r[k])
            print(f"  vs mock-GT: anchor xy {r['anchor_err_xy']:.2f} z {r['anchor_err_z']:.2f} | "
                  f"target xy {r['target_err_xy']:.2f} z {r['target_err_z']:.2f} | mismatch: {flags or '-'}")

    if rows:
        print("\n===== summary (n=%d) =====" % len(rows))
        for k in ("camera", "target_type", "landmark", "relation", "situation"):
            print(f"  {k:12s} acc {np.mean([r[k] for r in rows]):.2f}")
        for k in ("anchor_err_xy", "anchor_err_z", "target_err_xy", "target_err_z"):
            v = [r[k] for r in rows]
            print(f"  {k:14s} mean {np.mean(v):.2f}  median {np.median(v):.2f}  max {np.max(v):.2f}")
        print(f"  latency mean {np.mean([r['dt'] for r in rows]):.1f}s")
        print("  주의: mock_demo_data 는 GT 에 의도적 오답이 섞여 있다(예: umbrella→cola_can). "
              "mismatch 가 한 항목에만 나고 lf/hf 쌍의 다른 쪽과는 일치하면 GT 노이즈로 본다.")


if __name__ == "__main__":
    main()
