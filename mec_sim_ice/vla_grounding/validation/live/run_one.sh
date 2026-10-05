#!/usr/bin/env bash
# One live Stage2 run: fresh platform (GUI) -> agent -> wait FINISHED -> collect + summarize.
#   run_one.sh <problem_id> <tag> [ABL_OFF] [GRASP_OFF]
# e.g. run_one.sh rt01_parking_tumbler rt01_base_r1 all all
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PROB="$1"; TAG="$2"; ABL="${3:-}"; GRASP="${4:-}"
[ "$ABL" = "-" ] && ABL=""; [ "$GRASP" = "-" ] && GRASP=""
RUNS="$HERE/runs"; OUT="$RUNS/$TAG"; mkdir -p "$OUT"
PLOG="$OUT/platform.log"; ALOG="$OUT/agent.log"
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority

cd "$ROOT"
python3 "$HERE/make_problems.py" "$PROB" || exit 1
docker rm -f live-agent >/dev/null 2>&1
ENV_MARC_SCENARIO=marc2026_chungmu HEADLESS=false \
  docker compose -f simulation-platform/docker-compose.yml --profile platform up -d --force-recreate platform
docker logs -f simulation-platform-platform-1 > "$PLOG" 2>&1 &
LOGPID=$!

until grep -q "READY — waiting" "$PLOG"; do sleep 3; done
sleep 20   # scene still settling right after READY (first HELLO was queued too long once)

for attempt in 1 2 3; do
  "$HERE/run_agent.sh" "$TAG" "$ABL" "$GRASP" > "$ALOG" 2>&1 &
  until grep -qE "registration succeeded|Registration failed|Error|unknown" "$ALOG"; do sleep 2; done
  grep -q "registration succeeded" "$ALOG" && break
  echo "[run_one] registration failed (attempt $attempt), retrying"; sleep 10
done

# Watchdog: every WATCHDOG_S (wall) check progress.py; no progress -> stop this run. The sim runs
# at RTF ~0.18 here, so a stuck run would otherwise sit out the whole 600 s (sim) Stage 2 limit
# = ~55 min wall.
WATCHDOG_S="${WATCHDOG_S:-180}"
rm -f "$OUT/.wd_state.json" "$OUT/watchdog.txt"
last_wd=$(date +%s)
until grep -q "STAGE2_RUN → FINISHING\|FINISHED" "$PLOG" || ! docker ps -q -f name=live-agent | grep -q .; do
  sleep 5
  if [ $(( $(date +%s) - last_wd )) -ge "$WATCHDOG_S" ]; then
    last_wd=$(date +%s)
    if ! why=$(python3 "$HERE/progress.py" "$ALOG" "$OUT/.wd_state.json"); then
      echo "$(date +%T) $why" > "$OUT/watchdog.txt"
      echo "[run_one] watchdog: $why -> stopping $TAG"
      docker rm -f live-agent >/dev/null 2>&1
      break
    fi
  fi
done
sleep 5
RES=$(ls -t results/marc2026_chungmu/mec_sim_ice_*.json 2>/dev/null | head -1)
[ -n "$RES" ] && [ "$RES" -nt "$OUT/config.json" ] && cp "$RES" "$OUT/platform_result.json"
docker rm -f live-agent >/dev/null 2>&1
kill $LOGPID 2>/dev/null
python3 "$HERE/analyze_run.py" "$OUT" > /dev/null && echo "[run_one] $TAG done -> $OUT/summary.json"
