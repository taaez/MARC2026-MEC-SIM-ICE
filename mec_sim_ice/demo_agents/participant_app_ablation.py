"""MARC 2026 participant app -- Stage 1 ablation (표 A 게이팅 / 표 B 좌표 보정) 라이브 실행 진입점.

participant_app.py 와 완전히 같고, 조합 설정에 따라 아래만 런타임에 바꿔 끼운다.
기존 코드는 수정하지 않는다. 조합 정의는 stage1_ablation.CONFIGS 를 그대로 쓰고,
패치 내용은 stage1_ablation.run_single 과 같다 (대상 모듈만 라이브용 vla_grounding.grounding).

| 대상                                   | 목적                                   |
|----------------------------------------|----------------------------------------|
| vla_grounding_agent.resolve            | resolve_ablation (게이트/순위 스위치)  |
| grounding._support_adjusted            | 'on' 지지면 재투영 끄기                |
| grounding._build_decision              | 제출 좌표 계산 방식, 'on' 상판 fallback|
| vla_grounding_agent.on_target_probe_augment | 'on' 재탐색 끄기                  |
| MARC_DISABLE_POSE_KEYPOINTS=1          | 발목 keypoint 끄기                     |

실행 (demo/ 에서, 플랫폼이 떠 있는 상태):
    MARC_ABLATION_CONFIG=C0 MARC_ENTRYPOINT=participant_app_ablation.py \
    MARC_SCORE_LOG=../results_ablation/C0.scores.jsonl ./launch.sh <team_id>

라운드별 게이트 정보는 MARC_ABLATION_SIDE_LOG (기본: 점수 로그 옆 <ID>.side.jsonl) 에 쌓인다.
표 정리:
    python3 mec_sim_ice/demo_agents/participant_app_ablation.py --summarize results_ablation

G4 / C5 (= 현재 코드) 는 매 라운드 원본 grounding.resolve 와 결과를 대조해 불일치 횟수를 남긴다.

반복 측정 (같은 조합을 여러 번):
    MARC_ABLATION_CONFIG=G4 MARC_ABLATION_RUN=2 MARC_ENTRYPOINT=participant_app_ablation.py \
    ./launch.sh <team_id>
  - MARC_ABLATION_RUN 이 없으면 파일 이름은 <ID>.* (1회차), 있으면 <ID>_r<N>.* 이다.
  - 로그 경로(MARC_SCORE_LOG 등)를 따로 주지 않아도 results_ablation/ 에 저장한다.
  - 같은 이름의 로그가 이미 있으면 이어붙이지 않고 시작 전에 멈춘다 (실행이 섞이는 것을 막음).
  - 여러 실행을 합친 표: --aggregate results_ablation  (문항별 최빈값 + 평균/범위)
"""

import json
import logging
import math
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VALIDATION_DIR = os.path.join(os.path.dirname(_HERE), "vla_grounding", "validation")
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _VALIDATION_DIR)

from stage1_ablation import CONFIGS, LABELS, TABLES, _same, resolve_ablation  # noqa: E402

log = logging.getLogger("participant_app")

GATING = TABLES["gating"]
COORD = TABLES["coord"]
FULL_IDS = ("G4", "C5")


def _coord(G, cam, loc, mode: str) -> list:
    """제출 좌표 계산 방식만 바꾼다 (stage1_ablation._coord 와 같음)."""
    x1, y1, x2, y2 = loc.det.bbox
    z = float(loc.base[2])
    if mode in ("push", "full"):
        return loc.ground_point
    v = (y1 + y2) / 2.0 if mode == "center" else y2
    p = G.pixel_to_plane(cam, (x1 + x2) / 2.0, v, plane_z=z)
    return loc.ground_point if p is None else [float(p[0]), float(p[1]), float(p[2])]


def _install(name: str, side_path: str) -> None:
    import vla_grounding.grounding as G
    import vla_grounding_agent as VA

    cfg = CONFIGS[name]
    orig_resolve = VA.resolve
    orig_build = G._build_decision
    state = {"checked": 0, "mismatch": 0}

    if not cfg["ankle"]:
        # VLAGroundingAgent.__init__ 이 is_disabled() 를 보고 모델 로드를 건너뛴다.
        os.environ["MARC_DISABLE_POSE_KEYPOINTS"] = "1"
    if not cfg["probe"]:
        VA.on_target_probe_augment = lambda *a, **k: None
    if not cfg["on"]:
        G._support_adjusted = lambda cam, tdet, t_loc, a_loc, c: t_loc

    if cfg["coord"] != "full" or not cfg["on"]:
        def build(parsed, cam, cam_id, t_loc, a_loc, gcfg, token, extra):
            d = orig_build(parsed, cam, cam_id, t_loc, a_loc, gcfg, token, extra)
            if a_loc is not None:
                d.anchor_coord = _coord(G, cam, a_loc, cfg["coord"])
            if t_loc is not None:
                if cfg["coord"] != "full":
                    d.target_coord = _coord(G, cam, t_loc, cfg["coord"])
            elif not cfg["on"] or cfg["coord"] != "full":
                # target 미탐지: on 보정이 꺼져 있으면 f6ed1a3 이전처럼 anchor 좌표를 그대로
                use_top = (cfg["on"] and a_loc is not None
                           and parsed.relation_hint == "on")
                if use_top:
                    xy = _coord(G, cam, a_loc, cfg["coord"])
                    d.target_coord = [xy[0], xy[1], float(a_loc.top_z)]
                else:
                    d.target_coord = list(d.anchor_coord)
            return d
        G._build_decision = build

    def resolve(parsed, dets, cams, *a, **k):
        dec = resolve_ablation(G, parsed, dets, cams, cfg)
        entry = {
            "config": name,
            "text_relation": parsed.relation_hint,
            "is_person": bool(parsed.is_person),
            "decided": dec is not None,
            "sub_camera": dec.camera_id if dec is not None else None,
            "gate_applied": dec.debug.get("gate_applied") if dec else None,
            "gate_kind": dec.debug.get("gate_kind") if dec else None,
            "gate_cameras": dec.debug.get("gate_cameras") if dec else None,
            "selection": dec.debug.get("selection") if dec else None,
        }
        if name in FULL_IDS:
            ref = orig_resolve(parsed, dets, cams)
            state["checked"] += 1
            same = _same(dec, ref)
            state["mismatch"] += 0 if same else 1
            entry["matches_original"] = same
            log.info("[ablation] %s 복제 resolve 대조: %d회 중 불일치 %d회",
                     name, state["checked"], state["mismatch"])
        log.info("[ablation] %s cam=%s gate=%s %s", name, entry["sub_camera"],
                 entry["gate_kind"], entry["gate_cameras"])
        try:
            with open(side_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("[ablation] side log 기록 실패: %s", e)
        return dec

    # VLAGroundingAgent._ground 는 모듈 전역 이름 resolve 를 호출 시점에 찾는다.
    VA.resolve = resolve


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    name = os.environ.get("MARC_ABLATION_CONFIG", "")
    if name not in CONFIGS:
        raise SystemExit(f"MARC_ABLATION_CONFIG 는 {list(CONFIGS)} 중 하나여야 합니다 (받은 값: {name!r})")
    run = os.environ.get("MARC_ABLATION_RUN", "").strip()
    stem = f"{name}_r{run}" if run else name
    out_dir = os.environ.get("MARC_ABLATION_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(_HERE)), "results_ablation")
    os.makedirs(out_dir, exist_ok=True)
    # participant_app 이 생성 시점에 환경변수를 읽으므로 여기서 먼저 정해 둔다.
    score_log = os.environ.setdefault("MARC_SCORE_LOG", os.path.join(out_dir, f"{stem}.scores.jsonl"))
    os.environ.setdefault("MARC_REVEAL_LOG", os.path.join(out_dir, f"{stem}.reveals.jsonl"))
    side_path = os.environ.get(
        "MARC_ABLATION_SIDE_LOG",
        os.path.join(os.path.dirname(score_log) or ".", f"{stem}.side.jsonl"))
    os.makedirs(os.path.dirname(side_path) or ".", exist_ok=True)
    # 로그는 추가 모드로 쓰이므로, 이미 내용이 있으면 실행이 섞인다. 시작 전에 막는다.
    for path in (score_log, side_path):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            raise SystemExit(f"이미 로그가 있습니다: {path}\n"
                             f"다른 실행 번호(MARC_ABLATION_RUN)를 쓰거나 파일을 옮긴 뒤 다시 실행하세요.")
    try:
        import socket
        import subprocess
        import time
        git = subprocess.run(["git", "-C", _HERE, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        with open(os.path.join(out_dir, f"{stem}.meta.json"), "w", encoding="utf-8") as f:
            json.dump({"config": name, "run": run or "1", "host": socket.gethostname(),
                       "git": git, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "settings": CONFIGS[name]}, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001  -- 메타 기록 실패가 실험을 막으면 안 된다
        log.warning("[entry] meta 기록 실패: %s", e)

    import participant_app
    _install(name, side_path)
    log.info("[entry] Stage 1 ablation %s (%s) -- %s", name, LABELS[name], CONFIGS[name])
    log.info("[entry] run=%s score log=%s  side log=%s  pose_disabled_env=%s", run or "1",
             score_log, side_path, os.environ.get("MARC_DISABLE_POSE_KEYPOINTS", ""))
    participant_app.DemoParticipant().run()




# ── 표 정리 (라이브 점수 로그 + side 로그) ───────────────────────────────────
N_ROUNDS = 33          # Stage 1 문항 수. 이보다 적게 끝난 실행은 집계에서 뺀다.
POOL = {"G4": "FULL", "C5": "FULL"}   # G4 와 C5 는 설정이 같다 (stage1_ablation.CONFIGS)


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _err(sub, exp, key):
    a, b = (sub or {}).get(key), (exp or {}).get(key)
    if not a or not b or len(a) != 3 or len(b) != 3:
        return None
    return math.dist(a, b)   # offline_score 와 같은 3D 거리


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def _load_rows(out_dir: str, stem: str):
    """한 실행의 Stage 1 라운드별 기록. (rows, side 줄 수). 점수 로그가 없으면 (None, 0)."""
    scores = [r for r in _load(os.path.join(out_dir, f"{stem}.scores.jsonl"))
              if r.get("stage") == 1]
    side = _load(os.path.join(out_dir, f"{stem}.side.jsonl"))
    if not scores:
        return None, 0
    rows = []
    for i, r in enumerate(scores):
        s = r["scores"]
        exp = s.get("expected") or {}
        sub = s.get("submission") or {}
        ab = side[i] if i < len(side) else {}
        gc = ab.get("gate_cameras")
        rows.append({
            "problem_id": s.get("problem_id"),
            "is_person": exp.get("target_kind") == "person",
            "text_relation": ab.get("text_relation"),
            "total": s.get("total", 0.0),
            "camera": s.get("camera", 0.0),
            "object_type": s.get("object_type", 0.0),
            "landmark": s.get("landmark", 0.0),
            "relation": s.get("relation", 0.0),
            "anchor": s.get("anchor", 0.0),
            "target": s.get("target", 0.0),
            "anchor_err": _err(sub, exp, "anchor_coord"),
            "target_err": _err(sub, exp, "target_coord"),
            "gt_camera": exp.get("camera_id"),
            "gate_applied": bool(ab.get("gate_applied")),
            "excluded": bool(ab.get("gate_applied")) and gc is not None
                        and exp.get("camera_id") not in gc,
            "matches_original": ab.get("matches_original"),
        })
    return rows, len(side)


def _metrics(rows):
    obj = [r for r in rows if not r["is_person"]]
    per = [r for r in rows if r["is_person"]]
    on = [r for r in rows if r["text_relation"] == "on"]

    def pct(rs):
        return 100.0 * sum(1 for r in rs if r["camera"] >= 1.0) / len(rs) if rs else float("nan")

    def avg(rs):
        return sum(r["total"] for r in rs) / len(rs) if rs else float("nan")

    cc = [r for r in obj if r["camera"] >= 1.0]
    return {
        "n": len(rows), "n_obj": len(obj), "n_person": len(per), "n_on": len(on),
        "cam_acc": pct(rows), "cam_obj": pct(obj), "cam_per": pct(per),
        "stage1": avg(rows), "object": avg(obj), "person": avg(per), "on_subset": avg(on),
        "anchor_med_obj": _med([r["anchor_err"] for r in obj]),
        "target_med_obj": _med([r["target_err"] for r in obj]),
        "target_med_person": _med([r["target_err"] for r in per]),
        # 사물 오차는 0.3~2m 군과 10m+ 군(카메라/인스턴스 오답)으로 갈려 중앙값이 군 사이에 걸리면
        # 한 문항으로 크게 뛴다. 카메라 정답 라운드만 본 값을 함께 둔다.
        "anchor_med_obj_cc": _med([r["anchor_err"] for r in cc]),
        "target_med_obj_cc": _med([r["target_err"] for r in cc]),
        "target_hit_obj": sum(1 for r in obj if r["target"] >= 0.5),
        "target_hit_person": sum(1 for r in per if r["target"] >= 0.5),
        "gated": sum(r["gate_applied"] for r in rows),
        "excluded": [r["problem_id"] for r in rows if r["excluded"]],
        "mismatch": sum(1 for r in rows if r["matches_original"] is False),
    }


def summarize_config(out_dir: str, name: str, stem: str = None):
    """한 실행(기본: 1회차 <ID>.*)의 요약. 표 A·B 단일 실행 출력과 외부 호출용."""
    rows, n_side = _load_rows(out_dir, stem or name)
    if rows is None:
        return None
    s = _metrics(rows)
    s.update({"config": name, "label": LABELS[name], "n_side": n_side, "rows": rows})
    return s


def _notes(s, name):
    notes = []
    # Stage 2 도 grounding 을 1회 호출하므로 side 로그가 1줄 더 많은 것은 정상이다.
    if s["n_side"] < s["n"]:
        notes.append(f"side 로그 {s['n_side']}줄 < 점수 {s['n']}라운드 -- 대응이 어긋났을 수 있음")
    if s["excluded"]:
        notes.append("GT cam excluded: " + ", ".join(s["excluded"]))
    if name in FULL_IDS:
        notes.append(f"복제 resolve 대조 불일치 {s['mismatch']}회")
    return notes


def summarize(out_dir: str) -> None:
    a = ["## 표 A. Camera gating (live, 1회차)", "",
         "| ID | Configuration | Cam acc. (%) | Cam acc. obj / person (%) | Stage1 | Object"
         " | Person | Gated rounds | GT cam excluded |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in GATING:
        s = summarize_config(out_dir, name)
        if s is None:
            a.append(f"| {name} | {LABELS[name]} | (아직 실행 안 함) | | | | | | |")
            continue
        a.append(
            f"| {name} | {s['label']} | {s['cam_acc']:.1f} | {s['cam_obj']:.1f} / {s['cam_per']:.1f}"
            f" | {s['stage1']:.2f} | {s['object']:.2f} | {s['person']:.2f}"
            f" | {s['gated']}/{s['n']} | {len(s['excluded'])} |")
        a += [f"|  | ↳ {n} | | | | | | | |" for n in _notes(s, name)]

    b = ["## 표 B. Coordinate correction (live, 1회차)", "",
         "| ID | Configuration | Stage1 | 'on' subset | Person | Anchor err obj (m)"
         " | Target err obj (m) | Anchor / Target err obj, cam-correct (m)"
         " | Target err person (m) | Target hit obj / person | Cam acc. (%) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in COORD:
        s = summarize_config(out_dir, name)
        if s is None:
            b.append(f"| {name} | {LABELS[name]} | (아직 실행 안 함) | | | | | | | | |")
            continue
        b.append(
            f"| {name} | {s['label']} | {s['stage1']:.2f} | {s['on_subset']:.2f} | {s['person']:.2f}"
            f" | {s['anchor_med_obj']:.2f} | {s['target_med_obj']:.2f}"
            f" | {s['anchor_med_obj_cc']:.2f} / {s['target_med_obj_cc']:.2f}"
            f" | {s['target_med_person']:.2f}"
            f" | {s['target_hit_obj']}/{s['n_obj']} / {s['target_hit_person']}/{s['n_person']}"
            f" | {s['cam_acc']:.1f} |")
        b += [f"|  | ↳ {n} | | | | | | | | | |" for n in _notes(s, name)]
    b += ["", "오차는 중앙값(m), 제출 좌표와 공식 GT 좌표의 3D 거리. "
          "cam-correct = 카메라 정답 라운드만. Target hit = target 점수 ≥ 0.5 인 라운드 수."]

    text = "\n".join(a + [""] + b)
    print(text)
    with open(os.path.join(out_dir, "tables_live.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


# ── 여러 번 실행한 결과 합치기 ───────────────────────────────────────────────
def _stems(out_dir: str, name: str):
    """<ID> (1회차)와 <ID>_r<N> 실행의 stem 목록."""
    found = []
    if os.path.exists(os.path.join(out_dir, f"{name}.scores.jsonl")):
        found.append(name)
    pre = f"{name}_r"
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith(pre) and fn.endswith(".scores.jsonl"):
            tail = fn[len(pre):-len(".scores.jsonl")]
            if tail.isdigit():
                found.append(f"{name}_r{tail}")
    return sorted(found, key=lambda x: (len(x), x))


def _outcome_key(r):
    """문항 한 번의 결과를 이산 묶음으로 본다: 카메라·유형·랜드마크·관계 정답 여부 + target 득점 여부."""
    return (r["camera"] >= 1.0, r["object_type"] >= 1.0, r["landmark"] >= 1.0,
            r["relation"] >= 1.0, r["target"] >= 0.5)


def _modal_rows(runs):
    """문항별 최빈 결과 묶음을 골라, 그 묶음 안에서 총점이 중앙인 실제 기록 한 줄을 대표로 쓴다.
    동률이면 묶음 안 총점 중앙값이 낮은 쪽(보수적)을 고른다. (대표 행들, 동률 문항 목록, 뒤집힌 문항 정보)"""
    rep, ties, flips = [], [], []
    for i in range(N_ROUNDS):
        cands = [run[i] for run in runs]
        groups = {}
        for r in cands:
            groups.setdefault(_outcome_key(r), []).append(r)
        best = max(len(g) for g in groups.values())
        top = [g for g in groups.values() if len(g) == best]
        if len(top) > 1:
            ties.append(cands[0]["problem_id"])
        g = min(top, key=lambda g: statistics.median(r["total"] for r in g))
        totals = sorted(r["total"] for r in g)
        mid = totals[(len(totals) - 1) // 2]
        rep.append(next(r for r in g if r["total"] == mid))
        if len(groups) > 1:
            flips.append((cands[0]["problem_id"], len(runs),
                          sorted(((k, len(v)) for k, v in groups.items()), key=lambda kv: -kv[1])))
    return rep, ties, flips


def _stat(vals, d=2, pct=False):
    vals = [v for v in vals if v == v]   # NaN 제외
    if not vals:
        return "–"
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f"{m:.{d}f}"
    sd = statistics.stdev(vals)
    return f"{m:.{d}f} ± {sd:.{d}f} [{min(vals):.{d}f}–{max(vals):.{d}f}]"


def aggregate(out_dir: str, pool: bool = True) -> None:
    """여러 실행을 합쳐 (가) 문항별 최빈값 기준 표, (나) 실행 간 평균 ± 표준편차 [범위] 표를 만든다.
    G4 와 C5 는 설정이 같아 pool=True 면 두 이름의 실행을 합쳐 양쪽 표에 같은 값을 쓴다."""
    data = {}
    for name in GATING + COORD:
        names = [n for n in (GATING + COORD) if POOL.get(n) == POOL.get(name)] \
            if (pool and name in POOL) else [name]
        stems, seen = [], set()
        for n in names:
            for st in _stems(out_dir, n):
                if st not in seen:
                    seen.add(st)
                    stems.append(st)
        runs, used, skipped = [], [], []
        for st in stems:
            rows, _ = _load_rows(out_dir, st)
            if rows is None or len(rows) < N_ROUNDS:
                skipped.append(f"{st}({0 if rows is None else len(rows)}줄)")
                continue
            runs.append(rows[:N_ROUNDS])
            used.append(st)
        data[name] = (runs, used, skipped)

    out = ["# Stage 1 ablation: 반복 실행 집계 (live)", "",
           "- **최빈값 표**: 문항마다 (카메라·유형·랜드마크·관계 정답 여부, target 득점 여부)가 가장 자주 나온 "
           "결과 묶음을 고르고, 그 묶음 안에서 총점이 중앙인 실제 기록 한 줄을 대표로 써서 표를 만든다. "
           "동률이면 총점 중앙값이 낮은 쪽을 쓴다. 한 번도 실제로 나오지 않은 조합일 수 있다.",
           "- **평균 표**: 실행마다 지표를 따로 계산한 뒤 `평균 ± 표준편차 [최소–최대]`로 적는다.",
           f"- G4와 C5는 설정이 같아 {'실행을 합쳐서' if pool else '따로'} 집계한다.", ""]

    def runs_of(n):
        return data[n][0]

    def table_mode(title, names, cols):
        out.append(f"## {title}")
        out.append("")
        out.append("| ID | Configuration | N | " + " | ".join(c[0] for c in cols) + " | 동률 문항 수 |")
        out.append("|---|---|---:|" + "---:|" * len(cols) + "---:|")
        for n in names:
            runs, used, skipped = data[n]
            if not runs:
                out.append(f"| {n} | {LABELS[n]} | 0 | " + " | ".join("–" for _ in cols) + " | – |")
                continue
            rep, ties, _ = _modal_rows(runs)
            m = _metrics(rep)
            out.append(f"| {n} | {LABELS[n]} | {len(runs)} | "
                       + " | ".join(c[1](m) for c in cols) + f" | {len(ties)} |")
        out.append("")

    def table_mean(title, names, cols):
        out.append(f"## {title}")
        out.append("")
        out.append("| ID | Configuration | N | " + " | ".join(c[0] for c in cols) + " |")
        out.append("|---|---|---:|" + "---:|" * len(cols))
        for n in names:
            runs = runs_of(n)
            if not runs:
                out.append(f"| {n} | {LABELS[n]} | 0 | " + " | ".join("–" for _ in cols) + " |")
                continue
            ms = [_metrics(r) for r in runs]
            out.append(f"| {n} | {LABELS[n]} | {len(runs)} | "
                       + " | ".join(_stat([m[c[2]] for m in ms], c[3]) for c in cols) + " |")
        out.append("")

    f1 = lambda k: (lambda m: f"{m[k]:.1f}")
    f2 = lambda k: (lambda m: f"{m[k]:.2f}")
    colsA = [("Cam acc. (%)", f1("cam_acc"), "cam_acc", 1), ("Stage1", f2("stage1"), "stage1", 2),
             ("Object", f2("object"), "object", 2), ("Person", f2("person"), "person", 2)]
    colsB = [("Stage1", f2("stage1"), "stage1", 2), ("'on' subset", f2("on_subset"), "on_subset", 2),
             ("Person", f2("person"), "person", 2),
             ("Anchor err obj, cam-correct (m)", f2("anchor_med_obj_cc"), "anchor_med_obj_cc", 2),
             ("Target err obj, cam-correct (m)", f2("target_med_obj_cc"), "target_med_obj_cc", 2),
             ("Target err person (m)", f2("target_med_person"), "target_med_person", 2),
             ("Cam acc. (%)", f1("cam_acc"), "cam_acc", 1)]
    table_mode("표 A (최빈값 기준)", GATING, colsA)
    table_mean("표 A (평균 ± 표준편차 [범위])", GATING, colsA)
    table_mode("표 B (최빈값 기준)", COORD, colsB)
    table_mean("표 B (평균 ± 표준편차 [범위])", COORD, colsB)

    out += ["## 구성별 실행 목록과 결과가 뒤집힌 문항", ""]
    seen_pool = set()
    for n in GATING + COORD:
        key = POOL.get(n) if (pool and n in POOL) else n
        if key in seen_pool:
            continue
        seen_pool.add(key)
        runs, used, skipped = data[n]
        label = "G4 = C5" if (pool and n in POOL) else n
        out.append(f"### {label}: {len(runs)}회 ({', '.join(used) or '없음'})"
                   + (f" -- 제외: {', '.join(skipped)}" if skipped else ""))
        if len(runs) < 2:
            out += ["", "- 반복이 없어 뒤집힘을 알 수 없음", ""]
            continue
        _, ties, flips = _modal_rows(runs)
        out.append("")
        if not flips:
            out += ["- 모든 문항이 모든 실행에서 같은 결과 묶음", ""]
            continue
        out.append("| 문항 | 결과 묶음 분포 (카메라/유형/랜드마크/관계/target 득점 → 횟수) |")
        out.append("|---|---|")
        for pid, nr, groups in flips:
            desc = ", ".join("".join("O" if x else "x" for x in k) + f"×{c}" for k, c in groups)
            out.append(f"| {pid} | {desc} |")
        out.append("")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(out_dir, "tables_live_agg.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--aggregate":
        aggregate(sys.argv[2], pool="--no-pool" not in sys.argv[3:])
    else:
        main()
