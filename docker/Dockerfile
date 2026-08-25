# ---------------------------------------------------------------------------
# MARC 2026 Stage 1 agent — 빌드 시점에만 인터넷 사용, 런타임은 완전 오프라인
#
# 스타터킷 demo/Dockerfile 과 같은 구조를 따른다:
#   베이스 ros:humble-ros-base, 루트 /agent, COPY marc_sdk + demo,
#   PYTHONPATH=/agent:/agent/demo, ROS 소싱은 /ros_entrypoint.sh (베이스 제공).
# 거기에 stage1/ 파이프라인, 모델 가중치 baking, HF 오프라인 고정을 얹는다.
#
# Build context = 스타터킷 루트 (compose build.context=. , dockerfile=docker/Dockerfile)
# ---------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 + ROS2 Humble + Python 3.10 (rclpy / *_msgs 포함, 기본 RMW=FastDDS)
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# 원본 demo/Dockerfile 과 동일한 기본 의존성 (numpy/yaml/pip + placo — participant_app 이
# arm_pick(placo 기반 IK) 을 import 한다) + opencv 런타임 라이브러리.
# 네트워크는 빌드 시점에만 사용 — 채점 런타임은 인터넷 차단.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy python3-yaml python3-pip python3-dev \
        libgl1 libglib2.0-0 \
    && pip3 install --no-cache-dir placo \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch (CUDA 12.x wheel; 채점 머신 RTX PRO 5000 = Blackwell → cu128 이상 필요) ---
# 근거: Blackwell(sm_120) 은 cu128 부터 지원. 개발 PC 의 RTX 5060 Ti 도 동일 아키텍처.
RUN pip3 install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchvision==0.22.1

# --- 추론 스택 ---
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# --- 모델 가중치 baking (여기가 핵심: 런타임 다운로드 금지) ---
# 제출 기본값 = 7B (48GB 채점 머신 기준). 로컬 8GB 개발 빌드는 ARG 로 3B 지정.
ARG VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
ARG DET_MODEL=IDEA-Research/grounding-dino-base
ENV HF_HOME=/opt/hf
RUN python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo in [os.environ.get("DET_MODEL", "IDEA-Research/grounding-dino-base"),
             os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")]:
    print("downloading", repo, flush=True)
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken"])
EOF
# (ARG 는 같은 stage 의 RUN 에 환경변수로 자동 노출된다)

# 런타임: 허브 접근 완전 차단 (네트워크가 있어도 시도조차 안 함 → 채점 환경과 동일 동작)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    MARC_VLM_MODEL=${VLM_MODEL} MARC_DET_MODEL=${DET_MODEL}

# --- [SDK] marc_sdk + 에이전트 코드 (원본 demo/Dockerfile 과 동일한 /agent 레이아웃) ---
WORKDIR /agent
COPY marc_sdk /agent/marc_sdk
COPY stage1   /agent/stage1
COPY taejeong /agent/taejeong
COPY demo     /agent/demo
COPY models   /agent/models

# 런타임 플랫폼과 같은 RMW (FastDDS). import 경로: marc_sdk·stage1(/agent) + demo 로컬 모듈(/agent/demo).
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV PYTHONPATH=/agent:/agent/demo

# 빌드 단계에서 모델이 오프라인으로 로드되는지 자가 검증 (실패하면 빌드 실패 → 제출 전 발견)
RUN python3 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('${DET_MODEL}'); print('det ok')"

WORKDIR /agent/demo

# ROS 소싱은 원본과 동일하게 /ros_entrypoint.sh (베이스 이미지 제공) 를 쓴다.
# entrypoint.sh 는 오프라인 상태 로그만 남기고 /ros_entrypoint.sh 로 체이닝한다.
COPY docker/entrypoint.sh /agent/entrypoint.sh
RUN chmod +x /agent/entrypoint.sh
ENTRYPOINT ["/agent/entrypoint.sh"]
# participant_app.py (팀 기본) 는 taejeong 파이프라인(YOLO + 명령파싱 + 기하)으로 연결돼 있다.
# 이 이미지는 GroundingDINO+Qwen2.5-VL 구현용이라 별도 진입점을 쓴다.
#
# ⚠️ 제출 전 결정 필요: 루트 docker-compose.yml 이 이 Dockerfile 을 빌드하므로,
#    지금 상태로 제출하면 DINO+Qwen 경로가 실행된다. taejeong 파이프라인을 제출하려면
#    (a) 아래 CMD 를 participant_app.py 로 바꾸고,
#    (b) YOLO 가중치(demo/weights/best.pt)를 이미지에 baking 하고,
#    (c) 불필요해진 VLM/DET 빌드 ARG·전용 의존성을 정리해야 한다.
CMD ["python3", "participant_app_agent_vla.py"]
