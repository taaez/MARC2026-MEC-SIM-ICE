"""Build the Stage 2 ablation workbook (paper tables) from the two ablation JSONs.

    python3 mec_sim_ice/vla_grounding/validation/stage2_drive_ablation.py
    python3 mec_sim_ice/vla_grounding/validation/grasp_shape_ablation.py
    python3 mec_sim_ice/vla_grounding/validation/make_stage2_ablation_xlsx.py
        -> results_ablation/stage2_ablation.xlsx (not tracked)

Measured values are written as numbers (blue); every derived column (deltas, rates) is an
Excel formula (black). The workbook is flagged to recalculate fully on open.
"""

from __future__ import annotations

import json
import os

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
OUT = os.path.join(_REPO, "results_ablation", "stage2_ablation.xlsx")

FONT = "Arial"
F_TITLE = Font(name=FONT, size=13, bold=True)
F_HEAD = Font(name=FONT, size=10, bold=True, color="FFFFFF")
F_IN = Font(name=FONT, size=10, color="0000FF")         # measured / hardcoded input
F_CALC = Font(name=FONT, size=10, color="000000")       # formula
F_LINK = Font(name=FONT, size=10, color="008000")       # cross-sheet link
F_TXT = Font(name=FONT, size=10)
F_BOLD = Font(name=FONT, size=10, bold=True)
F_NOTE = Font(name=FONT, size=9, italic=True, color="555555")
FILL_HEAD = PatternFill("solid", fgColor="305496")
FILL_FULL = PatternFill("solid", fgColor="E2EFDA")
FILL_BASE = PatternFill("solid", fgColor="F2F2F2")
FILL_INPUT = PatternFill("solid", fgColor="FFFF00")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _load(name):
    with open(os.path.join(_HERE, "runs", name), encoding="utf-8") as f:
        return json.load(f)


def _header(ws, row, headers, widths=None):
    for j, h in enumerate(headers, 1):
        c = ws.cell(row=row, column=j, value=h)
        c.font, c.fill, c.alignment, c.border = F_HEAD, FILL_HEAD, CENTER, BOX
    ws.row_dimensions[row].height = 54
    if widths:
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w


def _put(ws, row, col, value, font=F_IN, fmt=None, fill=None, align=None):
    c = ws.cell(row=row, column=col, value=value)
    c.font, c.border = font, BOX
    if fmt:
        c.number_format = fmt
    if fill:
        c.fill = fill
    c.alignment = align or Alignment(vertical="top", wrap_text=isinstance(value, str))
    return c


def _notes(ws, row, lines, ncols):
    for i, line in enumerate(lines):
        ws.cell(row=row + i, column=1, value=line).font = F_NOTE
        ws.merge_cells(start_row=row + i, start_column=1, end_row=row + i, end_column=ncols)
        ws.cell(row=row + i, column=1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row + i].height = 28


# ---------------------------------------------------------------- driving --

DRIVE_NOTES = {
    "baseline": "qualifier tag behaviour (all 7 components reverted). Reproduces the totals "
                "measured on the untouched qualifier code exactly (634.9 s / 659.8 s).",
    "full": "Proposed system (all components on).",
    "-speed": "Largest time contributor. Without it the robot also brakes less (lower speed).",
    "-profile": "No time gain offline (removal is 1.3 s / 4.8 s faster). Kept for "
                "controlled arrival; re-check in Isaac Sim.",
    "-long_acc": "Removal is faster but commanded accel jumps to 11 m/s^2 (vs 0.6) and brake "
                 "events triple -- a safety/time trade-off (tip-over not modelled offline).",
    "-lat_acc": "Identical to Full: the cap never binds on these paths (safety guard only).",
    "-brake": "Removal is 9.7 s faster in hidden-obstacle runs with 0 contacts either way; "
              "its safety benefit is not measurable offline (sim chassis decel 2 m/s^2).",
    "-approach": "Within the approach phase the component is 0.4 s slower (plain) and 2.5 s "
                 "faster (hidden): faster far out, slower in the last ~15 cm. The hidden total "
                 "differs mostly through leg2 path changes (chaotic, not this component).",
    "-halt": "Saves 3.6 s of approach/pick-phase waiting (plain).",
}


def build_drive(wb, d):
    ws = wb.active
    ws.title = "Drive_Ablation"
    ws["A1"] = "Table A. Stage 2 driving ablation (offline harness, 6 retrieval problems, deterministic)"
    ws["A1"].font = F_TITLE
    ws["A2"] = ("Full system minus one component. Δ = (variant − Full); a positive Δ time means "
                "the removed component was saving that much time. Blue = measured, black = formula.")
    ws["A2"].font = F_NOTE
    headers = ["Configuration", "Removed component",
               "Success\nplain map\n(/6)", "Total time\nplain (s)", "Δ time vs Full\nplain (s)",
               "Δ time vs Full\nplain (%)",
               "Success\nhidden obstacles\n(/6)", "Total time\nhidden (s)",
               "Δ time vs Full\nhidden (s)", "Δ time vs Full\nhidden (%)",
               "Nav legs time\nleg1+leg2\nhidden (s)", "Approach+halt\nphase plain (s)",
               "Brake events\nhidden", "Stall replans\nhidden", "Contacts\n(collider ticks)\nhidden",
               "Peak commanded\naccel in nav\n(m/s²)", "Peak lateral\naccel |v·ω|\n(m/s²)",
               "Interpretation"]
    _header(ws, 4, headers, [20, 34, 9, 10, 11, 10, 11, 10, 11, 10, 12, 12, 9, 9, 11, 12, 11, 60])
    order = ["baseline", "full"] + ["-" + c for c in d["components"]]
    r0 = 5
    full_row = r0 + 1
    for i, name in enumerate(order):
        r = r0 + i
        cfg = d["configs"][name]
        p, h = cfg["plain"], cfg["hidden"]
        fill = FILL_FULL if name == "full" else (FILL_BASE if name == "baseline" else None)
        label = {"baseline": "Baseline (qualifier)", "full": "Full (proposed)"}.get(
            name, "w/o " + name[1:])
        removed = ("all 7 components" if name == "baseline" else
                   "—" if name == "full" else d["labels"][name[1:]])
        _put(ws, r, 1, label, F_BOLD, fill=fill)
        _put(ws, r, 2, removed, F_TXT, fill=fill)
        _put(ws, r, 3, p["success"], fmt="0", fill=fill)
        _put(ws, r, 4, p["total_s"], fmt="0.0", fill=fill)
        _put(ws, r, 5, f"=D{r}-D${full_row}", F_CALC, "+0.0;-0.0;0.0", fill)
        _put(ws, r, 6, f"=IF(D${full_row}=0,0,(D{r}-D${full_row})/D${full_row})", F_CALC,
             "+0.0%;-0.0%;0.0%", fill)
        _put(ws, r, 7, h["success"], fmt="0", fill=fill)
        _put(ws, r, 8, h["total_s"], fmt="0.0", fill=fill)
        _put(ws, r, 9, f"=H{r}-H${full_row}", F_CALC, "+0.0;-0.0;0.0", fill)
        _put(ws, r, 10, f"=IF(H${full_row}=0,0,(H{r}-H${full_row})/H${full_row})", F_CALC,
             "+0.0%;-0.0%;0.0%", fill)
        _put(ws, r, 11, round(h["leg1_s"] + h["leg2_s"], 1), fmt="0.0", fill=fill)
        _put(ws, r, 12, p["approach_s"], fmt="0.0", fill=fill)
        _put(ws, r, 13, h["brake_events"], fmt="0", fill=fill)
        _put(ws, r, 14, h["stall_replans"], fmt="0", fill=fill)
        _put(ws, r, 15, h["contacts"], fmt="0", fill=fill)
        _put(ws, r, 16, max(p["peak_cmd_acc"], h["peak_cmd_acc"]), fmt="0.0", fill=fill)
        _put(ws, r, 17, max(p["peak_lat_acc"], h["peak_lat_acc"]), fmt="0.00", fill=fill)
        _put(ws, r, 18, DRIVE_NOTES.get(name, ""), F_TXT, fill=fill)
        ws.row_dimensions[r].height = 42
    last = r0 + len(order) - 1
    s = last + 2
    ws.cell(row=s, column=1, value="Full vs Baseline").font = F_BOLD
    ws.cell(row=s, column=2, value="time saved, plain (s)").font = F_TXT
    _put(ws, s, 3, f"=D{r0}-D{full_row}", F_CALC, "0.0")
    _put(ws, s, 4, f"=(D{r0}-D{full_row})/D{r0}", F_CALC, "0.0%")
    ws.cell(row=s + 1, column=2, value="time saved, hidden obstacles (s)").font = F_TXT
    _put(ws, s + 1, 3, f"=H{r0}-H{full_row}", F_CALC, "0.0")
    _put(ws, s + 1, 4, f"=(H{r0}-H{full_row})/H{r0}", F_CALC, "0.0%")
    _notes(ws, s + 3, [
        "Plain map: obstacles are in the occupancy map (A* routes around them). Hidden obstacles: obstacles are removed "
        "from the map and exist only as chassis colliders seen by the simulated lidar (the live-platform situation).",
        "Time = simulated seconds summed over 6 problems (rt01, rt02, rt03, rt04, rt04b, rtC2), including pick. "
        "Success = both legs reached + grasp captured + object in basket + task_complete.",
        "Peak commanded accel: largest speed-up between consecutive cmd_vel inside _navigate_to (the driving legs). "
        "Peak lateral accel: actual |v·ω| of the simulated chassis.",
        "Components interact (a different arrival pose changes the next leg's path), so single-removal deltas "
        "do not sum to the Full−Baseline gap. Offline model: no tip-over, no contact physics -- see Setup sheet.",
    ], 18)
    ws.freeze_panes = "C5"
    return full_row, r0


# ---------------------------------------------------------------- grasping --

def build_grasp(wb, g):
    raw = wb.create_sheet("Grasp_Raw")
    raw["A1"] = "Grasp ablation raw counts (paired synthetic depth frames) -- source for Grasp_Ablation formulas"
    raw["A1"].font = F_TITLE
    order = ["baseline", "full"] + ["-" + s for s in g["switches"]]
    slices = ["all", "no_distractor", "distractor"] + ["kind:" + k for k in g["kinds"]]
    hdr = ["Configuration"]
    for sl in slices:
        hdr += [sl + "\ncaptured", sl + "\nn"]
    hdr += ["center err p50 (cm)", "center err p90 (cm)", "yaw err p50, lying (deg)"]
    _header(raw, 3, hdr, [22] + [11] * (len(hdr) - 1))
    raw_row = {}
    for i, name in enumerate(order):
        r = 4 + i
        raw_row[name] = r
        cfg = g["configs"][name]
        _put(raw, r, 1, name, F_BOLD)
        col = 2
        for sl in slices:
            a = cfg["by_kind"][sl[5:]] if sl.startswith("kind:") else cfg[sl]
            _put(raw, r, col, a["captured"], fmt="0")
            _put(raw, r, col + 1, a["n"], fmt="0")
            col += 2
        _put(raw, r, col, round(cfg["all"]["center_err_p50_cm"], 2), fmt="0.00")
        _put(raw, r, col + 1, round(cfg["all"]["center_err_p90_cm"], 2), fmt="0.00")
        _put(raw, r, col + 2, round(cfg["all"]["yaw_err_p50_deg_lying"], 2), fmt="0.00")
    raw.freeze_panes = "B4"

    ws = wb.create_sheet("Grasp_Ablation", 1)
    ws["A1"] = ("Table B. Stage 2 grasp-point ablation (synthetic base-camera depth, "
                f"{g['frames']} paired frames per configuration)")
    ws["A1"].font = F_TITLE
    ws["A2"] = ("Full system minus one switch. Δ = (variant − Full) in percentage points; a negative Δ "
                "means the removed component was contributing that much capture rate.")
    ws["A2"].font = F_NOTE
    kind_names = {"standing_tumbler": "Standing\ntumbler", "lying_tumbler": "Lying\ntumbler",
                  "lying_umbrella": "Lying\numbrella", "standing_box": "Standing box\n(cracker)",
                  "standing_sunblock": "Standing\nsunblock"}
    headers = ["Configuration", "Removed component", "Capture rate\n(all)", "Δ vs Full\n(pp)",
               "Capture\nno distractor", "Capture\nwith distractor"] + \
              [kind_names[k] for k in g["kinds"]] + \
              ["Center err\np50 (cm)", "Center err\np90 (cm)", "Yaw err p50\nlying (deg)",
               "Interpretation"]
    _header(ws, 4, headers, [20, 36, 11, 10, 11, 11] + [10] * len(g["kinds"]) + [10, 10, 11, 55])
    notes = {
        "baseline": "qualifier: plain mean of all raised pixels in the 0.5 m window + fixed yaw 0.52 rad. "
                    "Asserted identical to the old estimator on every frame.",
        "full": "Proposed shape-based estimator (all switches on).",
        "-USE_CLUSTER": "Largest single contributor: a nearby large object drags every other rule "
                        "(with-distractor capture collapses).",
        "-USE_TOP_BAND_CENTER": "Near-side bias on standing objects (box / tumbler center ~2-4 cm short).",
        "-USE_LYING_AXIS": "Lying objects grasped at the visible-surface mean instead of the axis.",
        "-USE_YAW": "Fixed yaw closes along/diagonally across long objects (median 44° error) and along "
                    "a box's long side (> 0.14 m stroke).",
    }
    full_row = 6
    col_of = {}   # slice -> captured column letter in Grasp_Raw
    col = 2
    for sl in slices:
        col_of[sl] = (get_column_letter(col), get_column_letter(col + 1))
        col += 2
    err_cols = [get_column_letter(col), get_column_letter(col + 1), get_column_letter(col + 2)]
    for i, name in enumerate(order):
        r = 5 + i
        rr = raw_row[name]
        fill = FILL_FULL if name == "full" else (FILL_BASE if name == "baseline" else None)
        label = {"baseline": "Baseline (qualifier)", "full": "Full (proposed)"}.get(
            name, "w/o " + name[1:])
        removed = ("all 4 components" if name == "baseline" else
                   "—" if name == "full" else g["labels"][name[1:]])
        _put(ws, r, 1, label, F_BOLD, fill=fill)
        _put(ws, r, 2, removed, F_TXT, fill=fill)

        def rate(sl):
            cc, nc = col_of[sl]
            return f"=IF(Grasp_Raw!{nc}{rr}=0,0,Grasp_Raw!{cc}{rr}/Grasp_Raw!{nc}{rr})"
        _put(ws, r, 3, rate("all"), F_LINK, "0.0%", fill)
        _put(ws, r, 4, f"=(C{r}-C${full_row})*100", F_CALC, "+0.0;-0.0;0.0", fill)
        _put(ws, r, 5, rate("no_distractor"), F_LINK, "0.0%", fill)
        _put(ws, r, 6, rate("distractor"), F_LINK, "0.0%", fill)
        for j, k in enumerate(g["kinds"]):
            _put(ws, r, 7 + j, rate("kind:" + k), F_LINK, "0.0%", fill)
        c0 = 7 + len(g["kinds"])
        for j in range(3):
            _put(ws, r, c0 + j, f"=Grasp_Raw!{err_cols[j]}{rr}", F_LINK, "0.0", fill)
        _put(ws, r, c0 + 3, notes.get(name, ""), F_TXT, fill=fill)
        ws.row_dimensions[r].height = 40
    last = 5 + len(order) - 1
    _notes(ws, last + 2, [
        "Capture model (assumed): gripper stroke 0.14 m (Robotiq 2F-140), pad strip half-width 0.015 m; lying "
        "cylinders additionally need ≤ 30° yaw error. It is a geometric proxy -- contact physics is NOT simulated.",
        "Each of 5 objects × 3 camera mounts × {no distractor, distractor 0.45–0.7 m away} × "
        f"{g['trials']} trials; nav-estimate error uniform in a 0–0.15 m disc. Same frames for every row (paired).",
        "Known limitation not in this table: a target touching a large object (< 5 cm) merges into one blob -- "
        "measured separately ≈ 18% capture for a lying umbrella against a bin.",
    ], len(headers))
    ws.freeze_panes = "C5"


# ------------------------------------------------------------- setup sheet --

def build_setup(wb, d, g):
    ws = wb.create_sheet("Setup_Assumptions")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 110
    rows = [
        ("Stage 2 ablation -- experimental setup", None),
        ("Code base", "qualifier tag ed416bb + post-qualifier Stage 2 changes (this workbook's commit)."),
        ("Driving harness", "mec_sim_ice/vla_grounding/validation/stage2_sim.py (fake MARCClient: unicycle chassis "
                            "with accel limits 2.0 m/s² / 3.0 rad/s², Ackermann invalid-command stall rule, VLP-16-like "
                            "lidar ray-cast, placo arm servo). participant_app.py runs unmodified."),
        ("Driving ablation script", "stage2_drive_ablation.py -- reverts one component at a time by monkeypatching to "
                                    "the qualifier-time behaviour; output runs/stage2_drive_ablation.json."),
        ("Driving scenarios", "6 retrieval problems of marc2026_chungmu (rt01, rt02, rt03, rt04, rt04b, rtC2); plain "
                              "occupancy map and --hidden-obstacles (obstacles only as colliders + lidar)."),
        ("Driving determinism", "Fully deterministic (single run per config is exact)."),
        ("Grasp evaluation", "grasp_shape_eval.py renders base-camera depth by ray-casting analytic cylinders/boxes "
                             "on a floor; the raised-pixel extraction is copied from participant_app."
                             "_locate_object_by_depth. grasp_shape_ablation.py evaluates all configs on the same frames."),
        ("Camera (assumed)", "320×240, HFOV 87°, depth noise 1 mm + 1.5 mm·d², 1% dropouts. Mounts (fwd, z, pitch) "
                             "in arm_base: A (0.10, −0.05, 30°), B (0.00, 0.10, 40°), C (0.15, −0.12, 22°). "
                             "No recorded Base_Camera_Center tf exists in the repo."),
        ("Objects (assumed)", "standing tumbler r 0.035 h 0.20; lying tumbler r 0.035 L 0.20; lying umbrella r 0.03 "
                              "L 0.85; cracker box 0.16×0.06×0.21; sunblock 0.05×0.035×0.15; floor at arm_base z −0.30."),
        ("Yaw ↔ finger axis", "Measured with placo FK of the URDF: closing axis angle = 90° + grasp_yaw (mod 180°)."),
        ("Arm feasibility", "Pick sequence completes for yaw −89…+89° at (fwd, lat) ∈ {(0.40,0), (0.47,−0.10), "
                            "(0.60,0.12), (0.70,−0.15)}; worst IK orientation error 0.36°."),
        ("Not modelled offline", "Tip-over, wheel slip, contact/grasp physics, real camera mount, real object meshes. "
                                 "Use the Isaac_* template sheets to record Isaac Sim measurements."),
        ("Reproduce", "python3 mec_sim_ice/vla_grounding/validation/stage2_drive_ablation.py ; "
                      "python3 mec_sim_ice/vla_grounding/validation/grasp_shape_ablation.py --trials 200 ; "
                      "python3 mec_sim_ice/vla_grounding/validation/make_stage2_ablation_xlsx.py"),
    ]
    for i, (k, v) in enumerate(rows, 1):
        a = ws.cell(row=i, column=1, value=k)
        a.font = F_TITLE if v is None else F_BOLD
        a.alignment = WRAP
        if v is not None:
            b = ws.cell(row=i, column=2, value=v)
            b.font, b.alignment = F_TXT, WRAP
            ws.row_dimensions[i].height = 32


# ----------------------------------------------------------- Isaac sheets --

def build_isaac(wb, title, sheet, configs, metrics):
    ws = wb.create_sheet(sheet)
    ws["A1"] = title
    ws["A1"].font = F_TITLE
    ws["A2"] = ("Legend: fill only the YELLOW cells (one value per run); Mean/Std are formulas. "
                "Example format for a time cell: 531.0 (seconds). Leave a run blank if it was not executed.")
    ws["A2"].font = F_NOTE
    runs = 3
    headers = ["Configuration"]
    for m in metrics:
        headers += [f"{m}\nrun {k}" for k in range(1, runs + 1)] + [f"{m}\nmean", f"{m}\nstd"]
    _header(ws, 4, headers, [24] + [10] * (len(headers) - 1))
    for i, name in enumerate(configs):
        r = 5 + i
        _put(ws, r, 1, name, F_BOLD)
        col = 2
        for _ in metrics:
            first, lastc = get_column_letter(col), get_column_letter(col + runs - 1)
            for k in range(runs):
                _put(ws, r, col + k, None, F_IN, "0.0", FILL_INPUT)
            _put(ws, r, col + runs, f'=IF(COUNT({first}{r}:{lastc}{r})=0,"",AVERAGE({first}{r}:{lastc}{r}))',
                 F_CALC, "0.0")
            _put(ws, r, col + runs + 1, f'=IF(COUNT({first}{r}:{lastc}{r})<2,"",STDEV({first}{r}:{lastc}{r}))',
                 F_CALC, "0.0")
            col += runs + 2
    ws.freeze_panes = "B5"


def main():
    d = _load("stage2_drive_ablation.json")
    g = _load("stage2_grasp_ablation.json")
    wb = Workbook()
    build_drive(wb, d)
    build_grasp(wb, g)
    build_setup(wb, d, g)
    build_isaac(wb, "Isaac Sim measurements -- driving (fill in)", "Isaac_Drive",
                ["Baseline (qualifier)", "Full (proposed)"] + ["w/o " + c for c in d["components"]],
                ["Total time (s)", "Success (/6)", "Contacts", "Tip-over (0/1)"])
    build_isaac(wb, "Isaac Sim measurements -- grasping (fill in)", "Isaac_Grasp",
                ["Baseline (qualifier)", "Full (proposed)"] + ["w/o " + s for s in g["switches"]],
                ["Grasp success (%)", "Pick attempts", "Center err (cm)"])
    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
    wb.calculation.fullCalcOnLoad = True
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wb.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
