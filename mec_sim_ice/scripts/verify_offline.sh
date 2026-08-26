#!/usr/bin/env bash
# =============================================================================
# 제출 전 오프라인 동작 검증.
#   network_mode: host 라 `--network none` 을 compose 에 그대로 쓸 수 없다(플랫폼과 DDS 통신이 끊긴다).
#   대신 host 에서 '외부(인터넷)' 만 iptables 로 막고, LAN(플랫폼 머신) 은 열어 둔 채 실행한다.
#
#   bash scripts/verify_offline.sh <LAN_CIDR 예: 192.168.0.0/24>
#   끝나면 규칙은 자동 복구된다.
# =============================================================================
set -euo pipefail
LAN="${1:?LAN CIDR 필요 (예: 192.168.0.0/24)}"
REPO_URL="${2:-}"   # 지정하면 새 clone 에서 테스트 (주최측과 동일 조건)

WORK=$(mktemp -d)
if [[ -n "$REPO_URL" ]]; then
  git clone --branch qualifier --depth 1 "$REPO_URL" "$WORK/repo"; cd "$WORK/repo"
else
  cd "$(git rev-parse --show-toplevel)"
fi

echo "[1/4] 이미지 빌드 (인터넷 허용 — 주최측도 빌드 시점엔 허용)"
docker compose build

echo "[2/4] 외부 네트워크 차단 (LAN $LAN, loopback 만 허용)"
sudo iptables -I OUTPUT 1 -o lo -j ACCEPT
sudo iptables -I OUTPUT 2 -d "$LAN" -j ACCEPT
sudo iptables -I OUTPUT 3 -d 224.0.0.0/4 -j ACCEPT     # DDS 멀티캐스트 디스커버리
sudo iptables -I OUTPUT 4 -j REJECT
restore() { echo "[4/4] 네트워크 규칙 복구"; sudo iptables -D OUTPUT -j REJECT || true
            sudo iptables -D OUTPUT -d 224.0.0.0/4 -j ACCEPT || true
            sudo iptables -D OUTPUT -d "$LAN" -j ACCEPT || true
            sudo iptables -D OUTPUT -o lo -j ACCEPT || true; }
trap restore EXIT

echo "[3/4] 오프라인 상태로 에이전트 기동. entrypoint 가 'offline OK' 를 찍어야 하고,"
echo "      모델 로드 중 HF 다운로드 시도/에러가 없어야 한다. 플랫폼을 먼저 띄워 둘 것."
docker compose up --abort-on-container-exit
