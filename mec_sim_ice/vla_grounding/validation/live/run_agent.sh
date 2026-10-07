#!/usr/bin/env bash
# Run the team agent image with the working-tree code bind-mounted (no rebuild).
#   run_agent.sh <tag> [ABL_OFF] [GRASP_OFF]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TAG="$1"; ABL="${2:-}"; GRASP="${3:-}"
OUT="$ROOT/mec_sim_ice/vla_grounding/validation/live/runs/$TAG"
mkdir -p "$OUT"
docker rm -f live-agent >/dev/null 2>&1 || true
exec docker run --name live-agent --rm --gpus all --network host \
  -e MARC_TEAM_ID=mec_sim_ice -e MARC_TOKEN=2a5b6258f40b17057ba4a863529c0e2d \
  -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e MARC_DISABLE_DINO_FALLBACK=1 -e MARC_LOG_LEVEL=INFO \
  -e MARC_LOCAL_PLANNER="${MARC_LOCAL_PLANNER:-legacy}" -e SURVEY_PART="${SURVEY_PART:-0/1}" -e SURVEY_BUDGET_S="${SURVEY_BUDGET_S:-560}" -e SURVEY_SEEN="${SURVEY_SEEN:-}" -e LIDAR_DEBUG="${LIDAR_DEBUG:-0}" -e ABL_OFF="$ABL" -e GRASP_OFF="$GRASP" -e LIVE_OUT=/agent/live_out \
  -e MARC_SCORE_LOG=/agent/live_out/scores.jsonl -e MARC_REVEAL_LOG=/agent/live_out/stage2_reveals.jsonl \
  -v "$ROOT/marc_sdk:/agent/marc_sdk:ro" -v "$ROOT/demo:/agent/demo:ro" \
  -v "$ROOT/manipulation:/agent/manipulation:ro" -v "$ROOT/mec_sim_ice:/agent/mec_sim_ice:ro" \
  -v "$OUT:/agent/live_out" \
  marc-mec-sim-ice-stage1:qualifier \
  python3 /agent/mec_sim_ice/vla_grounding/validation/live/"${LIVE_APP:-live_stage2_app.py}"
