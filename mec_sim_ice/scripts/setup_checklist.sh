#!/usr/bin/env bash
# =============================================================================
# MARC 2026 환경 구축 체크리스트 (개발자 가이드 3.2 ~ 3.6)
#   bash scripts/setup_checklist.sh [check|clone|login|setup|platform|demo|all]
# 각 단계는 멱등(여러 번 실행해도 안전). 실패 지점마다 원인과 대처를 출력한다.
# =============================================================================
set -euo pipefail
STEP="${1:-check}"
KIT_DIR="${MARC_KIT_DIR:-$HOME/marc-starter-kit}"
TEAM_ID="${MARC_TEAM_ID:-mec_sim_ice}"

ok()   { echo -e "  [OK]   $*"; }
warn() { echo -e "  [WARN] $*"; }
fail() { echo -e "  [FAIL] $*"; exit 1; }

# -----------------------------------------------------------------------------
step_check() {
  echo "== 0. 사전 점검 (3.1 필수 요구사항) =="
  # OS
  . /etc/os-release; [[ "$VERSION_ID" == "22.04" ]] && ok "Ubuntu 22.04" || fail "Ubuntu 22.04 필요 (현재 $PRETTY_NAME)"
  # WSL2 금지 (FAQ: 렌더러가 동작하지 않아 CCTV 토픽이 안 나온다)
  if uname -r | grep -qi microsoft || [[ -e /dev/dxg ]]; then fail "WSL2 감지됨. 네이티브 Ubuntu 필요"; else ok "네이티브 Linux"; fi
  # 드라이버: production 580.x (최소 570). 590+ 베타는 RTX 렌더러 segfault
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)
  [[ -z "$DRV" ]] && fail "nvidia-smi 실패. 드라이버 미설치 또는 MOK/Secure Boot 문제"
  MAJ=${DRV%%.*}
  if (( MAJ >= 590 )); then fail "드라이버 $DRV: 590+ 베타 → RTX 크래시 위험. 580.x 로 교체 후 셰이더 캐시(.runtime-data/cache) 삭제"
  elif (( MAJ < 570 )); then fail "드라이버 $DRV: 570 미만. 580.x 권장"
  else ok "드라이버 $DRV"; fi
  # Blackwell(RTX 50xx) 은 open 커널 모듈이어야 GPU 할당이 된다
  if nvidia-smi -L | grep -qi "RTX 50"; then
    if modinfo nvidia 2>/dev/null | grep -qi "license:.*GPL"; then ok "open 커널 모듈 (Blackwell 필수)"; else warn "RTX 50 계열인데 open 커널 모듈이 아닐 수 있음 → nvidia-driver-580-open 확인"; fi
  fi
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  (( VRAM >= 16000 )) && ok "VRAM ${VRAM}MiB" || warn "VRAM ${VRAM}MiB: Isaac Sim 캠퍼스 씬에는 부족할 가능성 높음. 이 PC 는 에이전트 개발용으로 쓰고 시뮬레이션은 공유 서버에서"
  # Docker + compose v2 + nvidia runtime
  command -v docker >/dev/null || fail "docker 없음"
  docker compose version >/dev/null 2>&1 && ok "docker compose (v2 플러그인형): $(docker compose version --short)" || fail "'docker compose'(공백) 플러그인 필요. docker-compose(하이픈) 는 v1"
  docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia" && ok "nvidia runtime 등록됨" || {
    warn "nvidia runtime 미등록 → 아래 실행"; echo "    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"; }
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 && ok "컨테이너 내 GPU 인식" || warn "컨테이너 내 nvidia-smi 실패 (NVIDIA Container Toolkit 확인)"
  # NGC 로그인 여부
  grep -q '"nvcr.io"' ~/.docker/config.json 2>/dev/null && ok "nvcr.io 로그인 정보 있음" || warn "nvcr.io 미로그인 → 'login' 단계 실행"
  # 만료된 ghcr 자격증명은 public content 이미지 pull 을 401 로 막는다
  grep -q '"ghcr.io"' ~/.docker/config.json 2>/dev/null && warn "ghcr.io 자격증명이 남아있음. 빌드 401 시 'docker logout ghcr.io'" || true
  # ROS 2 Humble (에이전트 개발용; 플랫폼은 컨테이너 안에 있음)
  [[ -f /opt/ros/humble/setup.bash ]] && ok "ROS 2 Humble" || warn "ROS 2 Humble 미설치 (호스트에서 ros2 topic list 디버깅용으로 권장)"
  python3 --version | grep -q "3.10" && ok "Python 3.10" || warn "Python $(python3 --version): SDK 는 3.10 기준"
  # 디스크
  AVAIL=$(df -BG "$HOME" | awk 'NR==2{gsub("G","",$4); print $4}')
  (( AVAIL >= 120 )) && ok "여유 디스크 ${AVAIL}G" || warn "여유 디스크 ${AVAIL}G: base 이미지 + 콘텐츠 + 에이전트 이미지에 100GB 이상 권장"
}

# -----------------------------------------------------------------------------
step_clone() {
  echo "== 3.2 스타터킷 =="
  if [[ -d "$KIT_DIR/.git" ]]; then
    (cd "$KIT_DIR" && git pull --ff-only) && ok "갱신: $KIT_DIR (패치 R01/R02/R03 반영 여부는 FAQ 변경이력 확인)"
  else
    git clone https://github.com/marc-challenge/marc-starter-kit.git "$KIT_DIR" && ok "clone: $KIT_DIR"
  fi
  ls "$KIT_DIR"
}

# -----------------------------------------------------------------------------
step_login() {
  echo "== 3.3 NGC 로그인 (marc.sh setup 이 대신 해주지 않음) =="
  echo "  1) https://ngc.nvidia.com → 프로필 → Account Settings → Generate Personal Key"
  echo "     * Personal Key (여러 개 동시 유효) 를 쓰고, Legacy API Key 는 쓰지 않는다"
  echo "     * 'NGC Catalog' 서비스 스코프 체크 필수, 키는 1회만 노출"
  echo "     * build.nvidia.com 의 nvapi- 키는 docker login 에 쓸 수 없다"
  echo "  2) Username 은 문자 그대로 \$oauthtoken"
  read -rsp "  NGC Personal Key 붙여넣기: " KEY; echo
  echo "$KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin && ok "nvcr.io 로그인"
  # 만료 ghcr 토큰 정리(있으면)
  docker logout ghcr.io >/dev/null 2>&1 || true
}

# -----------------------------------------------------------------------------
step_setup() {
  echo "== 3.4 base 이미지 빌드 (최초 1회, 수십 분~수 시간) =="
  cd "$KIT_DIR"
  if docker image ls marc-base:ros2-isaacsim-5.1 --format '{{.Repository}}' | grep -q marc-base; then
    ok "marc-base:ros2-isaacsim-5.1 이미 존재 (재빌드하려면 docker rmi 후 재실행)"
  else
    bash simulation-platform/marc.sh setup
    docker image ls marc-base:ros2-isaacsim-5.1 | grep -q marc-base && ok "base 이미지 빌드 완료" || fail "base 이미지 없음. 'pull access denied' 면 NGC 로그인 확인"
  fi
}

# -----------------------------------------------------------------------------
step_platform() {
  echo "== 3.5 플랫폼 실행 =="
  cd "$KIT_DIR/simulation-platform"
  [[ -f .env ]] || { cp .env.example .env; sed -i 's/^ENV_MARC_SCENARIO=.*/ENV_MARC_SCENARIO=marc2026_demo/' .env; ok ".env 생성 (scenario=marc2026_demo)"; }
  grep -q "ENV_MARC_SCENARIO=marc2026_demo" .env || warn ".env 의 ENV_MARC_SCENARIO 확인"
  # 갱신 공지 대응: 같은 태그라도 최신 digest 로
  docker pull ghcr.io/marc-challenge/marc-platform-content:2026 || warn "콘텐츠 이미지 pull 실패 (401 이면 docker logout ghcr.io)"
  [[ -n "${DISPLAY:-}" ]] && xhost +local:root >/dev/null 2>&1 || true
  echo "  플랫폼 기동 중. '[Runtime] Startup complete' 와 'Auto-plan: waiting for a participant' 가 뜰 때까지 기다릴 것."
  echo "  뷰포트가 검게/멈춘 것처럼 보여도(2~5분, 첫 실행 더 오래) 강제 종료 금지."
  docker compose --profile platform up
}

# -----------------------------------------------------------------------------
step_demo() {
  echo "== 3.6 데모(베이스라인) 실행 — 흐름 확인용, 실제 추론 아님 =="
  [[ -n "${MARC_TOKEN:-}" ]] || fail "MARC_TOKEN 환경변수 필요 (팀 토큰)"
  export MARC_TEAM_ID="$TEAM_ID"
  cd "$KIT_DIR/demo"
  echo "  (다른 터미널에서) ros2 topic list | grep /marc/env/cctv  → CCTV 토픽이 보여야 한다"
  docker compose up
}

case "$STEP" in
  check) step_check ;;
  clone) step_clone ;;
  login) step_login ;;
  setup) step_setup ;;
  platform) step_platform ;;
  demo) step_demo ;;
  all) step_check; step_clone; step_login; step_setup; echo "다음: 터미널 A 에서 'platform', 터미널 B 에서 'demo'" ;;
  *) echo "usage: $0 [check|clone|login|setup|platform|demo|all]"; exit 2 ;;
esac
