"""MARC_RECORD_DIR 덤프를 Isaac Sim/ROS 없이 재실행 -- RealVLAGrounding 오프라인 자가 채점.

데스크탑 라이브 실행이 남긴 records/s1_* 폴더(라운드별 CCTV PNG + meta.json)를 읽어,
루트 real_agent_vla.py 의 라운드 결정 로직을 기록된 카메라 파라미터(K/R_cw/C_w/ground_z)로
그대로 재현한다. 탐지 헬퍼/클래스 상수는 real_agent_vla 에서 직접 import 하므로
(그 파일의 ROS 의존은 지연 import) 로직이 어긋날 일이 없고, 카메라 루프와
pixel->world 만 여기서 미러링한다 (`keep in sync with RealVLAGrounding._ground`).

사용 (ultralytics + PyYAML + Pillow 만 있으면 됨 -- ROS/Isaac Sim 불필요):
  python tools/run_offline_real_vla.py --records ./records \
      --weights demo/weights/best.pt \
      [--gt demo/mock_demo_data.yaml]                # 좌표 포함 GT (payload 형식)
      [--gt taejeong/content-image-2026.R03/marc2026_chungmu.yaml]  # 공식 expected (해석 필드만)
      [--conf 0.35] [--imgsz 1280]

출력: 라운드별 제출 payload + (GT 지정 시) 필드별 정답 여부와 좌표 오차
(좌표 점수 = max(0, 1 - 3D오차/5m), taejeong/validation/score_report.py 와 같은 실측 역산식),
마지막에 전체 요약.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from real_agent_vla import (  # noqa: E402
    CAR_CLASSES, DET_CONF, DET_IMGSZ, POSE_BUCKET_CLASSES, RealVLAGrounding,
    parse_command, resolve_situation,
)

COORD_TOLERANCE_M = 5.0     # 실측 역산값 (score = 1 - err/5) -- score_report.py 와 동일


def load_round(d: str):
    """meta.json + PNG -> (meta, cams{K,R_cw,C_w np.ndarray...}, images{BGR ndarray})."""
    from PIL import Image
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    cams, images = {}, {}
    for cid, c in meta["cameras"].items():
        p = os.path.join(d, f"{cid}.png")
        if not os.path.exists(p):
            continue
        cams[cid] = {"K": np.array(c["K"], dtype=float),
                     "R_cw": np.array(c["R_cw"], dtype=float),
                     "C_w": np.array(c["C_w"], dtype=float),
                     "ground_z": float(c["ground_z"])}
        rgb = np.array(Image.open(p).convert("RGB"))
        images[cid] = rgb[:, :, ::-1].copy()      # ultralytics 는 BGR 해석
    return meta, cams, images


def pixel_to_world(cam: dict, pixel):
    """RealVLAGrounding._pixel_to_world 와 동일한 지면 역투영 (기록된 파라미터 사용)."""
    K, R, C = cam["K"], cam["R_cw"], cam["C_w"]
    gz = cam["ground_z"]
    u, v = pixel
    ray = R @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    if abs(ray[2]) < 1e-6:
        return [float(C[0]), float(C[1]), gz]
    s = (gz - C[2]) / ray[2]
    p = C + s * ray
    return [float(p[0]), float(p[1]), gz]


def to_payload(camera_id, target_type, landmark, relation, situation, anchor, target):
    """GroundingResult.to_payload() 와 같은 구조 (relation/situation 은 None 이면 생략)."""
    interp = {"target_type": target_type, "landmark": landmark}
    if relation is not None:
        interp["relation"] = relation
    if situation is not None:
        interp["situation"] = situation
    return {"camera_id": camera_id, "interpretation": interp,
            "grounding": {"anchor_coord": anchor, "target_coord": target}}


def ground_offline(model, command: str, cams: dict, images: dict,
                   conf: float, imgsz: int) -> dict:
    """RealVLAGrounding._ground 의 오프라인 미러 -- keep in sync."""
    parsed = parse_command(command)
    target_type = parsed.target_type or ("person" if parsed.is_person else "")
    landmark = parsed.landmark_type or ""
    G = RealVLAGrounding    # static 탐지 헬퍼 재사용

    for cam_id, img in images.items():
        det = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        if parsed.is_person:
            target_box, pose_class = G._best_person_box(det, parsed.pose_hint)
        else:
            target_box, pose_class = G._best_box(det, target_type), None
        if target_box is None:
            continue
        anchor_box = None
        if landmark:
            anchor_box = G._best_box(det, CAR_CLASSES if landmark == "car" else landmark)
            if anchor_box is None:
                continue
        relation, situation = parsed.relation_hint, None
        if parsed.is_person:
            situation = resolve_situation(pose_class, landmark or parsed.area_hint).situation
            relation = None
        target = pixel_to_world(cams[cam_id], G._box_center(target_box))
        anchor = (pixel_to_world(cams[cam_id], G._box_center(anchor_box))
                  if anchor_box is not None else target)
        return to_payload(cam_id, target_type, landmark, relation, situation,
                          anchor, target)

    situation = None
    if parsed.is_person:
        guess_pose = POSE_BUCKET_CLASSES.get(parsed.pose_hint, ("person_walking",))[0]
        situation = resolve_situation(guess_pose, landmark or parsed.area_hint).situation
    return to_payload("", target_type, landmark,
                      None if parsed.is_person else parsed.relation_hint,
                      situation, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


# -- GT 로딩/비교 (두 형식 지원) ------------------------------------------------

def load_gt(path: str) -> dict:
    """command(소문자) -> {"payload": ...} (mock_demo_data) 또는 {"expected": ...} (공식)."""
    import yaml
    d = yaml.safe_load(open(path, encoding="utf-8"))
    out = {}
    for k, v in (d.get("stage1_answers") or {}).items():
        out[k.strip().lower()] = {"payload": v}
    for p in (d.get("challenge_problems") or []):
        if p.get("kind") == "grounding" and p.get("voice_command"):
            out[p["voice_command"].strip().lower()] = {"expected": p["expected"],
                                                       "id": p.get("id")}
    return out


def _lm_class(name: str | None) -> str:
    """landmark_name 인스턴스 접미사 제거: picnic_table_01 -> picnic_table."""
    return re.sub(r"_\d+$", "", (name or "").strip().lower())


def coord_score(err_3d: float) -> float:
    return max(0.0, 1.0 - err_3d / COORD_TOLERANCE_M)


def compare(ours: dict, gt: dict) -> dict:
    """필드별 일치 여부(+payload GT 면 좌표 점수). 값 None 인 항목은 GT 에 없는 필드."""
    oi = ours["interpretation"]
    row = {"camera": None, "target_type": None, "landmark": None,
           "relation": None, "situation": None,
           "anchor_score": None, "target_score": None}
    if "payload" in gt:
        g, gi = gt["payload"], gt["payload"]["interpretation"]
        row["camera"] = ours["camera_id"] == g["camera_id"]
        row["target_type"] = oi.get("target_type") == gi.get("target_type")
        row["landmark"] = oi.get("landmark") == gi.get("landmark")
        row["relation"] = oi.get("relation") == gi.get("relation")
        row["situation"] = oi.get("situation") == gi.get("situation")
        for key, score_key in (("anchor_coord", "anchor_score"),
                               ("target_coord", "target_score")):
            err = float(np.linalg.norm(
                np.array(ours["grounding"][key]) - np.array(g["grounding"][key])))
            row[score_key] = round(coord_score(err), 3)
            row[key.replace("_coord", "_err_m")] = round(err, 2)
    else:
        e = gt["expected"]
        row["camera"] = ours["camera_id"] == e.get("camera_id")
        row["target_type"] = oi.get("target_type") == e.get("target_type")
        row["landmark"] = oi.get("landmark") == _lm_class(e.get("landmark_name"))
        if e.get("relation") is not None:
            row["relation"] = oi.get("relation") == e.get("relation")
        if e.get("situation") is not None:
            row["situation"] = oi.get("situation") == e.get("situation")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--weights", default=os.path.join("demo", "weights", "best.pt"))
    ap.add_argument("--gt", default=None,
                    help="mock_demo_data.yaml(좌표 포함) 또는 marc2026_chungmu.yaml(해석 필드만)")
    ap.add_argument("--conf", type=float, default=DET_CONF)
    ap.add_argument("--imgsz", type=int, default=DET_IMGSZ)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    gt_table = load_gt(args.gt) if args.gt else {}

    rounds = sorted(glob.glob(os.path.join(args.records, "s[12]_*")))
    if not rounds:
        rounds = [args.records]           # 단일 라운드 폴더를 직접 준 경우
    rows = []
    for d in rounds:
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta, cams, images = load_round(d)
        cmd = meta["command"]
        print("=" * 78)
        print(f"round: {os.path.basename(d)}")
        print(f"command: {cmd}")

        ours = ground_offline(model, cmd, cams, images, args.conf, args.imgsz)
        print("  ours    :", json.dumps(ours, ensure_ascii=False))
        if meta.get("payload"):
            print("  recorded:", json.dumps(meta["payload"], ensure_ascii=False))

        gt = gt_table.get(cmd.strip().lower())
        if gt:
            row = compare(ours, gt)
            rows.append(row)
            print("  vs GT   :", json.dumps(row, ensure_ascii=False))
        elif gt_table:
            print("  vs GT   : (GT 에 이 command 없음)")

    if rows:
        print("=" * 78)
        print(f"summary ({len(rows)} rounds with GT):")
        for f in ("camera", "target_type", "landmark", "relation", "situation"):
            vals = [r[f] for r in rows if r[f] is not None]
            if vals:
                print(f"  {f:12s}: {sum(vals)}/{len(vals)} ({sum(vals)/len(vals):.3f})")
        for f in ("anchor_score", "target_score"):
            vals = [r[f] for r in rows if r[f] is not None]
            if vals:
                print(f"  {f:12s}: mean {sum(vals)/len(vals):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
