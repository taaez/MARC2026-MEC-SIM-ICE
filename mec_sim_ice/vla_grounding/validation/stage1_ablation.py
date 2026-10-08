"""Stage 1 ablation -- 카메라 게이팅 / 좌표 보정을 단계별로 끄고 켜며 오프라인 채점.

기존 코드(grounding.py, offline_score.py 등)는 **한 줄도 고치지 않는다**. 이 스크립트가
offline_score 를 import 한 뒤 몇 개 함수만 런타임에 바꿔 끼우고 main() 을 그대로 돌린다.
조합마다 별도 프로세스로 실행하므로 패치가 서로 새지 않는다.

사용 (리포 루트에서):
  python3 mec_sim_ice/vla_grounding/validation/stage1_ablation.py --all
  python3 mec_sim_ice/vla_grounding/validation/stage1_ablation.py --configs G0,G4
  python3 mec_sim_ice/vla_grounding/validation/stage1_ablation.py --summarize <out_dir>
  python3 mec_sim_ice/vla_grounding/validation/stage1_ablation.py --list
  # offline_score 옵션 전달: 맨 뒤 -- 다음에 적는다 (예: -- --records /path/records)

결과: <out_dir>/<CONFIG>.json (offline_score 리포트), <CONFIG>.side.json (라운드별
게이트/오차), <CONFIG>.log, 그리고 summary.md / summary.csv (논문용 표).
자세한 설명은 mec_sim_ice/docs/작업기록_2026-10-05.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time

_VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
_VLA_GROUNDING_DIR = os.path.dirname(_VALIDATION_DIR)
RUNS_DIR = os.path.join(_VALIDATION_DIR, "runs")

# 모든 조합 공통: DINO/VLM 은 끈다(future work), Stage2 실행 시뮬은 Stage1 표와 무관.
BASE_ARGS = ["--no-stage2-exec", "--no-dino", "--no-vlm"]

# ── 조합 정의 ────────────────────────────────────────────────────────────────
# gate:   landmark / area 게이트 사용 여부
# rank:   카메라 간 최종 선택 키
#         "conf"     = target 탐지 confidence 최대
#         "bucket"   = relation 버킷 일치 우선 (사람 문항도 동일)
#         "full"     = 현재 코드 (사물은 버킷 우선, 사람은 근접도 우선)
# coord:  제출 좌표 계산
#         "center"   = bbox 중심 픽셀 역투영
#         "contact"  = bbox 아랫변 중앙(접지선) 역투영, footprint 밀기 없음
#         "push"     = 접지선 + footprint 중심으로 밀기 (= Located.ground_point)
#         "full"     = 현재 코드 (push + 사람이면 발목 keypoint 보정)
# on:     'on' 지지면 보정 (재투영 + target 미탐지 시 상판 높이)
# probe:  'on' 저신뢰 target 재탐색 (conf 0.001, 탐지 보강)
# ankle:  발목 keypoint 모델 로드 여부
FULL = dict(landmark_gate=True, area_gate=True, rank="full",
            coord="full", on=True, probe=True, ankle=True)

CONFIGS = {
    # 표 1 -- 카메라 게이팅 (좌표는 전부 현재 코드)
    "G0": dict(FULL, landmark_gate=False, area_gate=False, rank="conf"),
    "G1": dict(FULL, landmark_gate=True, area_gate=False, rank="conf"),
    "G2": dict(FULL, landmark_gate=True, area_gate=True, rank="conf"),
    "G3": dict(FULL, rank="bucket"),
    "G4": dict(FULL),
    # 표 2 -- 좌표 보정 (게이팅은 전부 현재 코드)
    "C0": dict(FULL, coord="center", on=False, probe=False, ankle=False),
    "C1": dict(FULL, coord="contact", on=False, probe=False, ankle=False),
    "C2": dict(FULL, coord="push", on=False, probe=False, ankle=False),
    "C3": dict(FULL, coord="push", on=True, probe=False, ankle=False),
    "C4": dict(FULL, coord="push", on=True, probe=True, ankle=False),
    "C5": dict(FULL),
}

LABELS = {
    "G0": "No gate, confidence ranking",
    "G1": "+ Landmark gate",
    "G2": "+ Area gate (intersection)",
    "G3": "+ Relation-fit ranking (bucket-first)",
    "G4": "+ Person proximity exception (full)",
    "C0": "BBox-center back-projection",
    "C1": "Bottom-center contact point",
    "C2": "+ Footprint-center push",
    "C3": "+ 'on' support-plane correction",
    "C4": "+ 'on' low-conf target probe (detection)",
    "C5": "+ Ankle keypoint (full)",
}

TABLES = {"gating": ["G0", "G1", "G2", "G3", "G4"],
          "coord": ["C0", "C1", "C2", "C3", "C4", "C5"]}


# ── 단일 조합 실행 (패치 + offline_score.main) ────────────────────────────────
def run_single(name: str, out_dir: str, passthrough: list) -> int:
    cfg = CONFIGS[name]
    if not cfg["ankle"]:
        os.environ["MARC_DISABLE_POSE_KEYPOINTS"] = "1"

    sys.path.insert(0, _VALIDATION_DIR)
    sys.path.insert(0, _VLA_GROUNDING_DIR)
    import offline_score as OS          # noqa: E402
    import grounding as G               # noqa: E402

    orig_resolve = G.resolve
    orig_build = G._build_decision
    orig_support = G._support_adjusted
    state = {"last": None, "mismatch": 0, "checked": 0}

    if not cfg["probe"]:
        OS.on_target_probe_augment = lambda *a, **k: None
    if not cfg["on"]:
        G._support_adjusted = lambda cam, tdet, t_loc, a_loc, c: t_loc

    if cfg["coord"] != "full" or not cfg["on"]:
        def build(parsed, cam, cam_id, t_loc, a_loc, gcfg, token, extra):
            d = orig_build(parsed, cam, cam_id, t_loc, a_loc, gcfg, token, extra)
            if a_loc is not None:
                d.anchor_coord = _coord(cam, a_loc, cfg["coord"])
            if t_loc is not None:
                if cfg["coord"] != "full":
                    d.target_coord = _coord(cam, t_loc, cfg["coord"])
            elif not cfg["on"] or cfg["coord"] != "full":
                # target 미탐지: on 보정이 꺼져 있으면 f6ed1a3 이전처럼 anchor 좌표를 그대로
                use_top = (cfg["on"] and a_loc is not None
                           and parsed.relation_hint == "on")
                d.target_coord = (_on_top(cam, a_loc, cfg["coord"]) if use_top
                                  else list(d.anchor_coord))
            return d
        G._build_decision = build

    def resolve(parsed, dets, cams, *a, **k):
        dec = resolve_ablation(G, parsed, dets, cams, cfg)
        if name == "G4":
            # 복제한 resolve 가 원본과 같은지 매 라운드 대조 (드리프트 감지)
            ref = orig_resolve(parsed, dets, cams)
            state["checked"] += 1
            if not _same(dec, ref):
                state["mismatch"] += 1
        state["last"] = {
            "text_relation": parsed.relation_hint,
            "is_person": bool(parsed.is_person),
            "decided": dec is not None,
            "gate_applied": (dec.debug.get("gate_applied") if dec else None),
            "gate_kind": (dec.debug.get("gate_kind") if dec else None),
            "gate_cameras": (dec.debug.get("gate_cameras") if dec else None),
            "selection": (dec.debug.get("selection") if dec else None),
        }
        return dec
    OS.resolve = resolve

    orig_score = OS.score_round

    def score_round(problem_id, weights, expected, submission):
        r = orig_score(problem_id, weights, expected, submission)
        r["_ablation"] = state["last"]
        state["last"] = None
        return r
    OS.score_round = score_round

    orig_report = OS.report

    def report(rounds, final, json_path):
        side = []
        for r in rounds:
            s = r["scores"]
            exp = s.get("expected") or {}
            ab = s.get("_ablation") or {}
            gc = ab.get("gate_cameras")
            excluded = (bool(ab.get("gate_applied")) and gc is not None
                        and exp.get("camera_id") not in gc)
            side.append({
                "problem_id": s["problem_id"], "stage": r.get("stage"),
                "record": r.get("record"), "borrowed": r.get("borrowed"),
                "is_person": exp.get("target_kind") == "person",
                "text_relation": ab.get("text_relation"),
                "total": s["total"],
                "fields": {k: s[k] for k in ("camera", "object_type", "landmark",
                                             "relation", "anchor", "target")},
                "anchor_err_m": s.get("anchor_err_m"),
                "target_err_m": s.get("target_err_m"),
                "gt_camera": exp.get("camera_id"),
                "sub_camera": (s.get("submission") or {}).get("camera_id"),
                "gate_applied": ab.get("gate_applied"), "gate_kind": ab.get("gate_kind"),
                "gate_cameras": gc, "gate_excluded_gt": excluded,
                "selection": ab.get("selection"),
            })
        meta = {"config": name, "label": LABELS[name], "settings": cfg,
                "fallback": final.get("fallback"),
                "equivalence_check": ({"checked": state["checked"],
                                       "mismatch": state["mismatch"]}
                                      if name == "G4" else None)}
        with open(os.path.join(out_dir, f"{name}.side.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "rounds": side}, f, ensure_ascii=False, indent=2)
        return orig_report(rounds, final, json_path)
    OS.report = report

    sys.argv = (["offline_score.py"] + BASE_ARGS + passthrough
                + ["--json", os.path.join(out_dir, f"{name}.json")])
    rc = OS.main()
    if name == "G4":
        print(f"[ablation] 복제 resolve 대조: {state['checked']}회 중 불일치 "
              f"{state['mismatch']}회", flush=True)
    return rc


def _coord(cam, loc, mode: str) -> list:
    """제출 좌표 계산 방식만 바꾼다 (선택 기하는 건드리지 않는다)."""
    import grounding as G
    x1, y1, x2, y2 = loc.det.bbox
    z = float(loc.base[2])
    if mode in ("push", "full"):
        return loc.ground_point
    v = (y1 + y2) / 2.0 if mode == "center" else y2
    p = G.pixel_to_plane(cam, (x1 + x2) / 2.0, v, plane_z=z)
    return loc.ground_point if p is None else [float(p[0]), float(p[1]), float(p[2])]


def _on_top(cam, a_loc, mode: str) -> list:
    """target 미탐지 + on: anchor 의 (x, y) 에 상판 높이."""
    xy = _coord(cam, a_loc, mode)
    return [xy[0], xy[1], float(a_loc.top_z)]


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    pa, pb = a.to_payload(), b.to_payload()
    if pa["camera_id"] != pb["camera_id"] or pa["interpretation"] != pb["interpretation"]:
        return False
    for k in ("anchor_coord", "target_coord"):
        if any(abs(x - y) > 1e-6 for x, y in zip(pa["grounding"][k], pb["grounding"][k])):
            return False
    return True


# ── grounding.resolve 복제 (게이트/순위 스위치 추가) ──────────────────────────
def resolve_ablation(G, parsed, detections, cameras, ab):
    """grounding.resolve 와 같은 로직에 스위치 3개만 추가한 복제본.

    전부 켜면(G4) 원본과 같은 결과를 내야 하고, G4 실행이 그것을 라운드마다 대조한다.
    원본 주석/근거는 grounding.py:820 의 resolve() 를 볼 것.
    """
    cfg = G.GroundingConfig()
    ground_surface_token = parsed.area_hint

    per_cam = {}
    for cam_id, dets in detections.items():
        cam = cameras.get(cam_id)
        if cam is None:
            continue
        per_cam[cam_id] = (cam, dets,
                           [d for d in dets if G._landmark_ok(d.label, parsed.landmark_type)])
    if not per_cam:
        return None

    landmark_gated = ([cid for cid, (_, _, adets) in per_cam.items() if adets]
                      if (ab["landmark_gate"] and parsed.landmark_type is not None) else [])

    area_gated = []
    indicators = G.AREA_INDICATOR_CLASSES.get(parsed.area_hint or "")
    if ab["area_gate"] and indicators:
        counts = {cid: sum(1 for d in dets if d.label in indicators)
                  for cid, (_, dets, _) in per_cam.items()}
        top = max(counts.values(), default=0)
        if top > 0:
            leaders = [cid for cid, n in counts.items() if n == top]
            runner_up = max((n for cid, n in counts.items() if cid not in leaders),
                            default=0)
            if top >= G.AREA_DOMINANCE_RATIO * runner_up:
                area_gated = leaders

    if landmark_gated and area_gated:
        both = [cid for cid in landmark_gated if cid in area_gated]
        gated, gate_kind = ((both, "landmark+area") if both
                            else (landmark_gated, "landmark"))
    elif landmark_gated:
        gated, gate_kind = landmark_gated, "landmark"
    elif area_gated:
        gated, gate_kind = area_gated, "area"
    else:
        gated, gate_kind = [], None

    gate_applied = bool(gated)
    if not gated:
        gated = list(per_cam)

    a_locs = {cid: [(adet, loc) for adet, loc in
                    ((a, G.locate(per_cam[cid][0], a, push=cfg.footprint_push))
                     for a in per_cam[cid][2])
                    if loc is not None]
              for cid in gated}

    open_space = parsed.landmark_type is None and parsed.open_space
    all_candidates = []
    hint_nearish = parsed.relation_hint in G.NEARISH_RELATIONS

    def search(target_ok):
        if open_space:
            picked = G._open_space_pick(gated, per_cam, a_locs, target_ok, cfg)
            if picked is not None:
                return picked

        per_cam_best = {}
        for cam_id in gated:
            cam, dets, _adets = per_cam[cam_id]
            cam_best = None
            for tdet in (d for d in dets if target_ok(d)):
                t0 = G.locate(cam, tdet, push=cfg.footprint_push)
                if t0 is None:
                    continue
                paired = False
                for adet, a_loc in a_locs[cam_id]:
                    if adet is tdet:
                        continue
                    paired = True
                    t_loc = G._support_adjusted(cam, tdet, t0, a_loc, cfg)
                    fit = G._relation_fit(cam, a_loc, t_loc, cfg, parsed.relation_hint)
                    cand = (fit[0], fit[1], tdet.score, cam_id, t_loc, a_loc)
                    all_candidates.append(cand)
                    if cam_best is None or cand[:3] > cam_best[:3]:
                        cam_best = cand
                if not paired:
                    cand = (0.0, 0.0, tdet.score, cam_id, t0, None)
                    all_candidates.append(cand)
                    if cam_best is None or cand[:3] > cam_best[:3]:
                        cam_best = cand
            if cam_best is not None:
                per_cam_best[cam_id] = cam_best

        if not per_cam_best:
            return None

        def cross_key(cand):
            fit_a, fit_b, conf = cand[0], cand[1], cand[2]
            if ab["rank"] == "conf":
                return (conf,)
            bucket_val, prox_val = (fit_b, fit_a) if hint_nearish else (fit_a, fit_b)
            if ab["rank"] == "full" and parsed.is_person:
                return (prox_val, bucket_val, conf)
            return (bucket_val, prox_val, conf)
        best_cam_id = max(per_cam_best, key=lambda cid: cross_key(per_cam_best[cid]))
        return per_cam_best[best_cam_id]

    best = search(lambda d: G._target_ok(d.label, parsed))
    relaxed = False
    if best is None and parsed.is_person and parsed.pose_hint:
        best = search(lambda d: d.label in G.PERSON_POSE_CLASSES or d.label == "person")
        relaxed = best is not None

    candidates_summary = [(cid, round(fa, 3), round(fb, 3), round(c, 3))
                          for fa, fb, c, cid, _t, _a in all_candidates]
    gate_info = {"gate_applied": gate_applied, "gate_kind": gate_kind,
                 "gate_cameras": sorted(gated), "candidates": candidates_summary}
    if best is not None:
        fit_a, fit_b, _, cam_id, t_loc, a_loc = best
        return G._build_decision(
            parsed, per_cam[cam_id][0], cam_id, t_loc, a_loc, cfg,
            ground_surface_token,
            dict(gate_info, selection="pair", relation_fit=(fit_a, fit_b),
                 pose_relaxed=relaxed))

    indicators = G.AREA_INDICATOR_CLASSES.get(parsed.area_hint or "")
    anchor_best = None
    anchor_best_indicator = None
    for cam_id in gated:
        for adet, a_loc in a_locs[cam_id]:
            if anchor_best is None or adet.score > anchor_best[0]:
                anchor_best = (adet.score, cam_id, a_loc)
            if indicators and adet.label in indicators:
                if anchor_best_indicator is None or adet.score > anchor_best_indicator[0]:
                    anchor_best_indicator = (adet.score, cam_id, a_loc)
    anchor_best = anchor_best_indicator or anchor_best
    if anchor_best is None:
        return None
    _, cam_id, a_loc = anchor_best
    return G._build_decision(
        parsed, per_cam[cam_id][0], cam_id, None, a_loc, cfg,
        ground_surface_token,
        dict(gate_info, selection="anchor_only",
             landmark_discriminative=gate_applied and len(gated) == 1))


# ── 집계 ─────────────────────────────────────────────────────────────────────
def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _med(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else None


def summarize_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        side = json.load(f)
    rs = [r for r in side["rounds"] if r["stage"] == 1]
    obj = [r for r in rs if not r["is_person"]]
    per = [r for r in rs if r["is_person"]]
    on = [r for r in rs if r["text_relation"] == "on"]
    gated = [r for r in rs if r["gate_applied"]]
    return {
        "config": side["meta"]["config"], "label": side["meta"]["label"],
        "n": len(rs), "n_obj": len(obj), "n_person": len(per), "n_on": len(on),
        "stage1": _avg([r["total"] for r in rs]),
        "obj": _avg([r["total"] for r in obj]),
        "person": _avg([r["total"] for r in per]),
        "on_subset": _avg([r["total"] for r in on]),
        "cam_acc": _avg([r["fields"]["camera"] for r in rs]),
        "cam_acc_obj": _avg([r["fields"]["camera"] for r in obj]),
        "cam_acc_person": _avg([r["fields"]["camera"] for r in per]),
        "gate_applied": len(gated),
        "gate_excluded_gt": sum(1 for r in gated if r["gate_excluded_gt"]),
        "anchor_med_obj": _med([r["anchor_err_m"] for r in obj]),
        "target_med_obj": _med([r["target_err_m"] for r in obj]),
        "anchor_med_person": _med([r["anchor_err_m"] for r in per]),
        "target_med_person": _med([r["target_err_m"] for r in per]),
        "equivalence_check": side["meta"].get("equivalence_check"),
        "fallback": side["meta"].get("fallback"),
    }


def _f(x, nd=2, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.{nd}f}"


def write_summary(out_dir: str) -> None:
    rows = {}
    for name in CONFIGS:
        p = os.path.join(out_dir, f"{name}.side.json")
        if os.path.exists(p):
            rows[name] = summarize_config(p)
    if not rows:
        print(f"집계할 결과가 없습니다: {out_dir}")
        return

    lines = [f"# Stage 1 ablation summary", "",
             f"- 결과 폴더: `{out_dir}`",
             "- 공통 조건: DINO/VLM off, records 기반 오프라인 채점 (target 좌표 점수는 근사치)",
             ""]
    any_row = next(iter(rows.values()))
    lines.append(f"- 문항 수: Stage1 {any_row['n']} (사물 {any_row['n_obj']}, "
                 f"사람 {any_row['n_person']}, 'on' 명령 {any_row['n_on']})")
    eq = rows.get("G4", {}).get("equivalence_check")
    if eq:
        lines.append(f"- 복제 resolve 대조(G4): {eq['checked']}회 중 불일치 {eq['mismatch']}회 "
                     "(0 이어야 함)")
    lines.append("")

    g = [n for n in TABLES["gating"] if n in rows]
    if g:
        lines += ["## Table A. Camera gating (cumulative)", "",
                  "| ID | Configuration | Cam acc. (%) | Cam acc. obj / person (%) | "
                  "Stage1 | Object | Person | Gated rounds | GT cam excluded |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for n in g:
            r = rows[n]
            lines.append(
                f"| {n} | {r['label']} | {_f(r['cam_acc'], pct=True)} | "
                f"{_f(r['cam_acc_obj'], pct=True)} / {_f(r['cam_acc_person'], pct=True)} | "
                f"{_f(r['stage1'])} | {_f(r['obj'])} | {_f(r['person'])} | "
                f"{r['gate_applied']}/{r['n']} | {r['gate_excluded_gt']} |")
        lines.append("")

    c = [n for n in TABLES["coord"] if n in rows]
    if c:
        lines += ["## Table B. Coordinate correction (cumulative)", "",
                  "| ID | Configuration | Stage1 | 'on' subset | Person | "
                  "Anchor err obj (m) | Target err obj (m) | Target err person (m) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for n in c:
            r = rows[n]
            lines.append(
                f"| {n} | {r['label']} | {_f(r['stage1'])} | {_f(r['on_subset'])} | "
                f"{_f(r['person'])} | {_f(r['anchor_med_obj'])} | "
                f"{_f(r['target_med_obj'])} | {_f(r['target_med_person'])} |")
        lines += ["", "오차는 중앙값(m). 'on' subset = 명령 텍스트 relation 이 on 인 문항."]

    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    keys = [k for k in any_row if k not in ("equivalence_check", "fallback")]
    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for n in rows:
            w.writerow(rows[n])
    print("\n".join(lines))
    print(f"\n저장: {os.path.join(out_dir, 'summary.md')}, summary.csv")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> int:
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="모든 조합 실행 후 표 생성")
    ap.add_argument("--configs", default=None, help="쉼표로 구분한 조합 ID (예: G0,G4,C0)")
    ap.add_argument("--table", choices=list(TABLES), default=None,
                    help="한 표의 조합만 실행 (gating / coord)")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본 runs/stage1_ablation_<시각>)")
    ap.add_argument("--summarize", default=None, metavar="DIR", help="실행 없이 표만 다시 생성")
    ap.add_argument("--list", action="store_true", help="조합 목록 출력")
    ap.add_argument("--single", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.list:
        for n, c in CONFIGS.items():
            print(f"{n}  {LABELS[n]:44s} {c}")
        return 0
    if args.summarize:
        write_summary(os.path.abspath(args.summarize))
        return 0
    if args.single:
        return run_single(args.single, os.path.abspath(args.out), passthrough)

    if args.configs:
        names = [n.strip() for n in args.configs.split(",") if n.strip()]
    elif args.table:
        names = TABLES[args.table]
    elif args.all:
        names = list(CONFIGS)
    else:
        ap.print_help()
        return 2
    bad = [n for n in names if n not in CONFIGS]
    if bad:
        print(f"알 수 없는 조합: {bad}. --list 로 확인.", file=sys.stderr)
        return 2

    out = os.path.abspath(args.out or os.path.join(
        RUNS_DIR, time.strftime("stage1_ablation_%Y-%m-%d_%H%M%S")))
    os.makedirs(out, exist_ok=True)
    failed = []
    for n in names:
        t0 = time.time()
        print(f"=== [{n}] {LABELS[n]}", flush=True)
        log_path = os.path.join(out, f"{n}.log")
        with open(log_path, "w", encoding="utf-8") as lf:
            rc = subprocess.call(
                [sys.executable, os.path.abspath(__file__), "--single", n, "--out", out,
                 "--"] + passthrough, stdout=lf, stderr=subprocess.STDOUT)
        print(f"    rc={rc}  {time.time() - t0:.0f}s  log={log_path}", flush=True)
        if rc != 0:
            failed.append(n)
    write_summary(out)
    if failed:
        print(f"실패한 조합: {failed} -- 각 .log 확인", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
