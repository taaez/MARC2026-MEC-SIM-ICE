#!/usr/bin/env bash
# Run every line of a queue file "<problem_id> <tag> <ABL_OFF|-> <GRASP_OFF|-> [planner]", skipping tags
# that already have a summary.json (so the queue can be restarted / appended to).
HERE="$(cd "$(dirname "$0")" && pwd)"
Q="${1:-$HERE/queue.txt}"
while true; do
  line=$(grep -v '^\s*#' "$Q" | while read -r p t a g pl; do
           [ -n "$p" ] && [ ! -f "$HERE/runs/$t/summary.json" ] && [ ! -f "$HERE/runs/$t/.skip" ] && { echo "$p $t $a $g ${pl:-legacy}"; break; }
         done)
  [ -z "$line" ] && break
  read -r p t a g pl <<< "$line"
  [ "$a" = "-" ] && a=""; [ "$g" = "-" ] && g=""
  echo "[queue] $(date +%T) start $t ($p) ABL_OFF=$a GRASP_OFF=$g planner=$pl"
  MARC_LOCAL_PLANNER="$pl" timeout 5400 "$HERE/run_one.sh" "$p" "$t" "$a" "$g" || { echo "[queue] $t failed/timeout"; touch "$HERE/runs/$t/.skip"; }
done
echo "[queue] all done $(date +%T)"
