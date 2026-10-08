#!/bin/bash
# Stage 1 ablation 반복 측정 러너 (조합당 7회, 라운드 단위로 조합을 번갈아 실행).
#   - 매 실행 전에 플랫폼 컨테이너를 재시작해 같은 초기 장면에서 시작한다 (GUI 유지).
#   - 에이전트는 Stage 1 33라운드가 끝나 STAGE2_RUN 으로 넘어가면 바로 끊는다.
#   - 이미 33라운드가 다 기록된 실행은 건너뛴다 (중간에 멈췄다 다시 돌려도 이어서 진행).
#   - 실패한 실행의 로그는 results_ablation/failed/ 로 옮기고 최대 3번까지 다시 시도한다.
set -u
REPO="$HOME/ablation_test"
C=simulation-platform-platform-1
OUT="$REPO/results_ablation"
LOG="$OUT/repeat_runner.log"
WRAP="$REPO/mec_sim_ice/demo_agents/participant_app_ablation.py"
TOK=$(grep -oP 'MARC_TOKEN:\s*"\$\{MARC_TOKEN:-\K[^}]+' "$REPO/docker-compose.yml")
CONFIGS=(G0 G1 G2 G3 G4 C0 C1 C2 C3 C4)   # C5 = G4 (같은 설정, 집계에서 합침)
FIRST_RUN=2
LAST_RUN=7
G4_LAST_RUN=6                              # G4·C5 1회씩 있으므로 G4_r2..r6 이면 합쳐서 7회

mkdir -p "$OUT/failed"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

n_stage1() { local n=0; [ -f "$1" ] && n=$(grep -c '"stage": 1,' "$1"); echo "${n:-0}"; }

restart_platform() {
    local since
    since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    docker restart -t 20 "$C" >/dev/null 2>&1 || return 1
    for _ in $(seq 1 120); do            # 최대 10분
        if docker logs --since "$since" "$C" 2>&1 | grep -q "Auto-plan: waiting for a participant"; then
            return 0
        fi
        sleep 5
    done
    return 1
}

run_one() {   # $1=config $2=run
    local cfg=$1 run=$2 stem="${1}_r${2}" pid t0 n
    local scores="$OUT/$stem.scores.jsonl"
    if [ "$(n_stage1 "$scores")" -ge 33 ]; then
        log "건너뜀 $stem (이미 33라운드)"
        return 0
    fi
    # 미완료 흔적은 옮겨 둔다 (래퍼는 기존 로그가 있으면 시작을 거부한다)
    local ts; ts=$(date +%m%d_%H%M%S)
    for f in "$OUT/$stem".*; do [ -e "$f" ] && mv "$f" "$OUT/failed/$(basename "$f").$ts"; done

    if ! restart_platform; then
        log "실패 $stem: 플랫폼이 10분 안에 준비되지 않음"
        return 1
    fi
    # 플랫폼이 Stage 1 33문항을 내는지 확인 (다른 폴더의 problems.yaml 이 마운트되면 문항 수가 달라진다)
    local n_prob
    n_prob=$(docker exec "$C" python3 -c "import yaml;d=yaml.safe_load(open('/metacom2026/problems.yaml'));print(sum(1 for x in d.get('stage1',[]) if x.get('enabled')))" 2>/dev/null)
    if [ "$n_prob" != "33" ]; then
        log "중단: 플랫폼 Stage1 문항 수가 33이 아님 (${n_prob:-확인불가}), 마운트=$(docker inspect "$C" --format '{{range .Mounts}}{{if eq .Destination "/metacom2026/problems.yaml"}}{{.Source}}{{end}}{{end}}')"
        exit 2
    fi
    t0=$(date +%s)
    ( cd "$REPO/demo" && MARC_TOKEN="$TOK" MARC_ABLATION_CONFIG="$cfg" MARC_ABLATION_RUN="$run" \
        MARC_ENTRYPOINT=participant_app_ablation.py \
        exec setsid ./launch.sh mec_sim_ice > "$OUT/$stem.log" 2>&1 ) &
    pid=$!
    while :; do
        sleep 5
        if grep -q "STAGE1_SCORE -> STAGE2_RUN" "$OUT/$stem.log" 2>/dev/null; then break; fi
        if ! kill -0 "$pid" 2>/dev/null; then break; fi
        if [ $(( $(date +%s) - t0 )) -gt 900 ]; then log "시간 초과 $stem (15분)"; break; fi
    done
    sleep 2
    kill -INT -- "-$pid" 2>/dev/null; sleep 5; kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    n=$(n_stage1 "$scores")
    local mism; mism=$(grep -o "불일치 [0-9]*회" "$OUT/$stem.log" | tail -1)
    local pose; pose=$(grep -c "pose-kp\] loaded" "$OUT/$stem.log")
    if [ "$n" -ge 33 ]; then
        log "완료 $stem: 33라운드, $(( $(date +%s) - t0 ))초, pose 로드 ${pose}회${mism:+, $mism}"
        return 0
    fi
    if grep -q "STAGE1_SCORE -> STAGE2_RUN" "$OUT/$stem.log" 2>/dev/null; then
        log "중단: $stem 이 Stage1 ${n}라운드만 하고 Stage2 로 넘어감 (문항 설정 문제로 보고 전체 정지)"
        for f in "$OUT/$stem".*; do [ -e "$f" ] && mv "$f" "$OUT/failed/$(basename "$f").$ts"; done
        exit 3
    fi
    log "실패 $stem: Stage1 ${n}라운드만 기록됨 (마지막 로그: $(grep -vE '^\s*$' "$OUT/$stem.log" | tail -1 | cut -c1-120))"
    for f in "$OUT/$stem".*; do [ -e "$f" ] && mv "$f" "$OUT/failed/$(basename "$f").$ts"; done
    return 1
}

log "===== 시작: 조합 ${CONFIGS[*]}, 회차 r$FIRST_RUN~r$LAST_RUN (G4 는 r$G4_LAST_RUN 까지) ====="
for run in $(seq $FIRST_RUN $LAST_RUN); do
    log "----- ${run}회차 -----"
    for cfg in "${CONFIGS[@]}"; do
        if [ "$cfg" = G4 ] && [ "$run" -gt $G4_LAST_RUN ]; then continue; fi
        ok=0
        for attempt in 1 2 3; do
            if run_one "$cfg" "$run"; then ok=1; break; fi
            log "재시도 ${cfg}_r${run} ($attempt/3 실패)"
        done
        [ $ok = 1 ] || log "포기 ${cfg}_r${run}: 3번 실패"
    done
    python3 "$WRAP" --aggregate "$OUT" > /dev/null 2>>"$LOG" && log "${run}회차 집계 갱신 → tables_live_agg.md"
done
restart_platform && log "플랫폼을 대기 상태로 재시작해 둠"
log "===== 종료 ====="
