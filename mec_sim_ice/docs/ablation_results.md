# Ablation Results (Final)

Final experimental results of the MARC 2026 agent, identical to Tables I–IV of the paper
([`paper/MARC2026_paper_EN.pdf`](../../paper/MARC2026_paper_EN.pdf), Section V).
All runs are live, network-disabled runs on the public Chungmu playground.

The comparisons are **cumulative**: each row adds one component on top of the previous row, so a
row's gain is conditional on the components before it, not an isolated effect.

---

## Stage 1 — Grounding and Localization

**Setup**: all 33 public Chungmu missions (14 person, 19 lost-item), 7 live runs per configuration,
mean ± std. Semantic fields use exact matching; coordinates use XY error.
Detector: mAP50 = 0.841 on 500 platform-generated validation scenes; tiling raises small-object
recall from 0.42–0.50 to 0.58–0.71.

### Table I. Cumulative camera-selection results

| Configuration | Camera (%) | Score /100 |
|---|---:|---:|
| G0 Confidence only | 33.8 ± 1.1 | 44.8 ± 0.5 |
| G1 + Landmark gate | 40.7 ± 1.6 | 49.1 ± 1.0 |
| G2 + Area gate | 75.8 ± 0.0 | 70.1 ± 1.8 |
| G3 + Relation ranking | 84.8 ± 0.0 | 75.5 ± 1.7 |
| **G4 + Person proximity (final)** | **87.9 ± 0.0** | **75.8 ± 0.6** |

The final configuration selects the correct camera in 29/33 rounds in every run. The four camera
failures involve targets missed in the correct view. Area gating gives the largest camera gain.

### Table II. Cumulative localization results

| Configuration | Score | On score | Person error (m) |
|---|---:|---:|---:|
| C0 Box center | 59.9 ± 1.7 | 37.5 ± 0.0 | 2.00 ± 0.01 |
| C1 Box bottom | 64.7 ± 0.8 | 47.3 ± 0.0 | 1.49 ± 0.00 |
| C2 + Footprint | 63.2 ± 1.0 | 43.7 ± 0.0 | 1.46 ± 0.00 |
| C3 + Support height | 72.3 ± 1.4 | 68.1 ± 0.0 | 0.93 ± 0.00 |
| C4 + Re-detection | 73.8 ± 1.4 | 75.5 ± 1.9 | 0.93 ± 0.00 |
| **C5 + Ankles (final)** | **75.8 ± 0.6** | **76.0 ± 3.4** | **0.75 ± 0.00** |

*On score* averages the 11 "on" commands. *Person error* is the per-run median over the 14 person
missions, averaged across runs. C4 changes detection, not only geometry. G4 and C5 are the same
configuration (the final code).

Footprint correction reduces anchor error from 1.74 m to 0.64 m in correctly selected object rounds;
support-height correction gives the largest total-score increase. About half of the zero-target-score
rounds come from targets missed in the correct camera, which geometric correction alone cannot fix.

---

## Stage 2 — Retrieval and Obstacle Representation

**Setup**: five public retrieval missions (tumblers and umbrellas, two in a dense obstacle area
around a taxi), 4 rounds per mission = 20 runs per configuration. Retrieval subtotal = pick (50) +
delivery (30). A run is stopped with zero score when a watchdog window shows neither a task milestone
nor a 0.3 m gain in straight-line goal distance.

### Table III. Retrieval results on five missions

| Metric | A* + P control (discs) | MPPI (discs) | **MPPI (depth grid, final)** |
|---|---:|---:|---:|
| Completed / runs | 9/20 | 18/20 | **20/20** |
| Retrieval score /80 | 36.0 ± 40.8 | 69.5 ± 26.3 | **75.0 ± 15.4** |
| Stage 2 score /100 | 42.8 ± 48.7 | 82.5 ± 30.2 | **90.1 ± 14.5** |
| Correct grasps¹ | 9/20 | 17/20 | **18/20** |
| Drive time (s)² | 107.1 | 117.9 | **85.0** |
| Brakes per run | 15.0 | 4.6 | **1.0** |
| Path-loss runs | – | 7 | **0** |

¹ Confirmed by the platform, which scores only completed runs.
² Mean of per-mission means over missions completed by all three configurations.
Brakes: LiDAR proximity stops. Path loss: V = ∞ at the robot in some field build.

With the same disc obstacles, cost-to-go-guided MPPI raised completion from 9/20 to 18/20 and cut
brakes from 15.0 to 4.6 per run. The depth-prioritized grid removed the remaining path loss.
The remaining grasp failures (2 of 4 final runs) occur with the vending-machine umbrella, where curb
fragments remain in the depth window.

### Table IV. Dense-area obstacle-representation results (MPPI fixed)

| Representation | Completed¹ | Path lost | Relaxed builds² | Time (s)³ |
|---|---:|---:|---:|---:|
| Discs | 3/4; 3/4 | 5/8 | 114.4 | 145.4; 172.8 |
| Occupancy grid (log-odds) | 0/1; 0/1 | 2/2 | 70.0 | — |
| **+ Depth priority (final)** | **4/4; 4/4** | **0/8** | **0** | **107.2; 110.8** |

¹ Tumbler; umbrella. ² Field builds per run with relaxed obstacle constraints.
³ Means for completed tumbler; umbrella trials.

A conventional occupancy grid failed because slope-induced LiDAR errors recur in the same cells and
accumulate; letting depth-confirmed floor override the grid removes them.

---

## Reproducing

Raw run logs are not tracked in this repository.

**Stage 1** (configs `G0`–`G4`, `C0`–`C5`):

```bash
MARC_ABLATION_CONFIG=C0 MARC_ENTRYPOINT=participant_app_ablation.py \
MARC_SCORE_LOG=../results_ablation/C0.scores.jsonl ./launch.sh <team_id>
```

- Wrapper: [`demo_agents/participant_app_ablation.py`](../demo_agents/participant_app_ablation.py)
- 7-run repeat runner: [`scripts/run_ablation_repeats.sh`](../scripts/run_ablation_repeats.sh)

**Stage 2** (switches in [`demo/participant_app.py`](../../demo/participant_app.py)):

| Column | Environment |
|---|---|
| A* + P control (discs) | `MARC_LOCAL_PLANNER=legacy` |
| MPPI (discs) | `MARC_LOCAL_PLANNER=mppi MARC_MPPI_OCC_GRID=0` |
| MPPI (depth grid, final) | defaults (`mppi`, `MARC_MPPI_OCC_GRID=1`) |
