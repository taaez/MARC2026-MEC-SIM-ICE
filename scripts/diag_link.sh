#!/usr/bin/env bash
# =============================================================================
# 2대 PC(플랫폼 / 에이전트) 연동 진단.  에이전트 PC 에서 실행한다.
#   bash scripts/diag_link.sh
# 플랫폼 PC 가 이미 떠 있어야 한다("Auto-plan: waiting for a participant" 로그 확인).
# =============================================================================
set -uo pipefail
source /opt/ros/humble/setup.bash 2>/dev/null || { echo "[FAIL] /opt/ros/humble 없음"; exit 1; }
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
PLATFORM_IP="${1:-}"

ok(){ echo "  [OK]   $*"; }; warn(){ echo "  [WARN] $*"; }; bad(){ echo "  [FAIL] $*"; }

echo "== 1. 로컬 설정 =="
echo "  ROS_DOMAIN_ID=$ROS_DOMAIN_ID  RMW=$RMW_IMPLEMENTATION  ROS_DISTRO=${ROS_DISTRO:-?}"
[[ "$RMW_IMPLEMENTATION" == "rmw_fastrtps_cpp" ]] && ok "Fast DDS" || bad "플랫폼은 Fast DDS 고정. Cyclone 이면 디스커버리 실패"
[[ "${ROS_DISTRO:-}" == "humble" ]] && ok "Humble" || bad "다른 배포판이 같은 도메인에 들어오면 런타임이 크래시할 수 있다"
[[ -z "${ROS_LOCALHOST_ONLY:-}" || "${ROS_LOCALHOST_ONLY}" == "0" ]] && ok "ROS_LOCALHOST_ONLY 미설정" || bad "ROS_LOCALHOST_ONLY=1 이면 다른 PC 를 못 본다"

echo "== 2. 네트워크 =="
if command -v ufw >/dev/null && sudo ufw status 2>/dev/null | grep -q "Status: active"; then
  warn "ufw 활성. DDS(UDP 7400-7600 대역 + 멀티캐스트) 허용 필요: sudo ufw allow from <상대IP>"
else ok "방화벽 비활성 또는 ufw 없음"; fi
if [[ -n "$PLATFORM_IP" ]]; then
  ping -c1 -W2 "$PLATFORM_IP" >/dev/null 2>&1 && ok "플랫폼 PC 도달 ($PLATFORM_IP)" || bad "ping 실패"
  SPD=$(ethtool "$(ip route get "$PLATFORM_IP" | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1); exit}')" 2>/dev/null | awk -F: '/Speed/{print $2}')
  [[ -n "${SPD:-}" ]] && echo "  링크 속도:${SPD} (CCTV 다중 스트림에는 1Gbps 이상 권장)"
fi
ip -o link | grep -q MULTICAST && ok "멀티캐스트 가능 인터페이스 존재" || warn "멀티캐스트 미지원 NIC 이면 디스커버리 실패"

echo "== 3. 토픽 디스커버리 (최대 20초 대기) =="
for i in $(seq 20); do
  TOPICS=$(timeout 5 ros2 topic list 2>/dev/null)
  echo "$TOPICS" | grep -q "/marc/ops/announce" && break
  sleep 1
done
echo "$TOPICS" | grep -q "/marc/ops/announce" && ok "ops/announce 보임" || { bad "플랫폼 토픽이 안 보인다 → 도메인/RMW/방화벽/LAN 확인"; exit 1; }
CCTV=$(echo "$TOPICS" | grep -c "/marc/env/cctv/.*/image")
(( CCTV > 0 )) && ok "CCTV 카메라 ${CCTV}대" || bad "CCTV 토픽 없음 → 플랫폼이 WSL2/가상화 위에서 돌거나 렌더러 실패"
echo "$TOPICS" | grep -q "^/tf_static$" && ok "/tf_static 있음" || bad "/tf_static 없음 → 카메라 extrinsic 을 못 얻는다(2026.R01 이상 필요)"
echo "$TOPICS" | grep -q "^/clock$" && ok "/clock 있음(시뮬레이션 시간)" || warn "/clock 없음 → now_s() 가 벽시계로 폴백"

echo "== 4. 실제 수신 확인 =="
FIRST=$(echo "$TOPICS" | grep "/marc/env/cctv/.*/image" | head -1)
if [[ -n "$FIRST" ]]; then
  CID=$(echo "$FIRST" | sed 's|/marc/env/cctv/||; s|/image||')
  timeout 10 ros2 topic echo --once "$FIRST" --field header.frame_id >/dev/null 2>&1 \
    && ok "이미지 수신 OK ($CID)" || bad "이미지 미수신 → QoS(BEST_EFFORT) 또는 대역폭 문제"
  timeout 10 ros2 topic echo --once "/marc/env/cctv/$CID/ground_height" >/dev/null 2>&1 \
    && ok "ground_height 수신 OK" || bad "ground_height 미수신"
  timeout 10 ros2 run tf2_ros tf2_echo world "$CID" 2>&1 | head -5 | grep -q Translation \
    && ok "world→$CID TF 조회 OK" || bad "TF 조회 실패"
fi
echo "== 5. 시뮬레이션 속도(RTF) 참고 =="
T0=$(timeout 5 ros2 topic echo --once /clock --field clock.sec 2>/dev/null); W0=$(date +%s)
sleep 5
T1=$(timeout 5 ros2 topic echo --once /clock --field clock.sec 2>/dev/null); W1=$(date +%s)
if [[ -n "${T0:-}" && -n "${T1:-}" ]]; then
  echo "  sim $((T1-T0))s / wall $((W1-W0))s  → RTF = sim/wall, 1.0 미만이면 시뮬레이션이 실시간보다 느리다"
fi
echo "진단 끝."
