# ---------------------------------------------------------------------------
# MARC 2026 Stage 1 agent — 빌드 시점에만 인터넷 사용, 런타임은 완전 오프라인
#
# 스타터킷 demo/Dockerfile 과 같은 구조를 따른다:
#   베이스 ros:humble-ros-base, 루트 /agent, COPY marc_sdk + demo,
#   PYTHONPATH=/agent:/agent/demo, ROS 소싱은 /ros_entrypoint.sh (베이스 제공).
# 거기에 vla_grounding/ 파이프라인, 모델 가중치 baking, HF 오프라인 고정을 얹는다.
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
        libgl1 libglib2.0-0 ros-humble-tf2-ros \
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
# 2026-08-29: VL(비전-언어) 모델을 완전히 뺐다(vla_grounding/vlm_fallback.py 참고) --
# 이 파이프라인의 VLM 은 텍스트 분류 두 건(OOV 물체명 재해석, "behind" 구동사/
# 공간관계 중의성 판정)뿐이라 이미지 인코더가 필요 없다. 기본값을 텍스트 전용
# 소형 LM(1.5B, ~3GB)으로 바꿔서 DINO(~1GB)와 합쳐도 여유 있게 들어간다.
# 필요하면 여기 ARG 또는 --build-arg VLM_MODEL=... 로 다른 텍스트 LM 도 가능
# (VL 모델은 더 이상 코드가 지원하지 않는다 -- 넣으면 아래 자가검증에서 바로 걸림).
ARG VLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ARG DET_MODEL=IDEA-Research/grounding-dino-base
ENV HF_HOME=/opt/hf
RUN python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo in [os.environ.get("DET_MODEL", "IDEA-Research/grounding-dino-base"),
             os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")]:
    print("downloading", repo, flush=True)
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken"])
EOF
# (ARG 는 같은 stage 의 RUN 에 환경변수로 자동 노출된다)

# 런타임: 허브 접근 완전 차단 (네트워크가 있어도 시도조차 안 함 → 채점 환경과 동일 동작)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    MARC_VLM_MODEL=${VLM_MODEL} MARC_DET_MODEL=${DET_MODEL}

# --- [SDK] marc_sdk + 에이전트 코드 (원본 demo/Dockerfile 과 동일한 /agent 레이아웃) ---
# 팀 제작 코드는 리포 정리로 mec_sim_ice/ 아래에 있다 (organizer 원본 demo/ 와는 별도).
# mec_sim_ice/ 를 로컬과 같은 상대구조 그대로 통째로 COPY 한다 (demo/launch.sh 의
# SDK_ROOT:SCRIPT_DIR:TEAM_ROOT:TEAM_AGENTS_DIR 배선과 동일한 의미). vla_grounding/
# 안의 pose_keypoints.py·yolo_adapter.py 는 dirname(dirname(__file__)) 로 형제
# 디렉토리(demo_agents/weights/, training/, datasets/)를 찾는데, 예전처럼
# demo_agents/ 를 /agent/demo 로 병합해버리면 이 형제 관계가 깨져서 가중치/서브모델
# 경로 계산이 컨테이너 안에서만 조용히 어긋난다 (로컬 실행은 멀쩡해서 못 알아챘던 버그).
# mec_sim_ice/ 를 그대로 두면 이 계산이 로컬과 컨테이너에서 완전히 동일하게 풀린다.
# records/(264MB, 오프라인 채점 전용)와 docs/ 는 .dockerignore 로 이미지에서 뺀다.
WORKDIR /agent
COPY marc_sdk     /agent/marc_sdk
COPY demo         /agent/demo
COPY manipulation /agent/manipulation
COPY mec_sim_ice  /agent/mec_sim_ice

# 런타임 플랫폼과 같은 RMW (FastDDS). import 경로: marc_sdk(/agent) + demo 로컬 모듈
# (/agent/demo) + vla_grounding 패키지·demo_agents 플랫 모듈(/agent/mec_sim_ice 이하).
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV PYTHONPATH=/agent:/agent/demo:/agent/mec_sim_ice:/agent/mec_sim_ice/demo_agents

# 빌드 단계에서 모델이 오프라인으로 로드되는지 자가 검증 (실패하면 빌드 실패 → 제출 전 발견)
RUN python3 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('${DET_MODEL}'); print('det ok')"
# VLM 은 이제 텍스트 전용(SmallLmBackend, vla_grounding/vlm_fallback.py) -- VL
# 모델을 여기 넣으면(예: 실수로 되돌리면) AutoTokenizer 로드 자체가 chat
# template 없이 깨지거나 여기서 걸린다. AutoProcessor 가 아니라 실제 코드가
# 쓰는 클래스(AutoTokenizer)로 검증해야 이 불일치를 잡는다.
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${VLM_MODEL}'); print('vlm ok')"
# 위 두 줄은 processor/tokenizer 만 확인한다 -- 그건 가중치 파일을 안 건드리므로,
# snapshot_download 의 allow_patterns 에 안 걸린 형식(*.bin 등)으로만 올라온
# repo 였다면 **빌드는 통과하고 런타임에 조용히 폴백이 꺼진다**(로드 실패를
# vla_grounding_agent.py 가 경고만 남기고 삼킨다). 실제 모델 클래스로 한 번 더
# 로드해서 그 경우를 빌드에서 잡는다.
RUN python3 -c "from transformers import AutoModelForZeroShotObjectDetection as M; M.from_pretrained('${DET_MODEL}'); print('det weights ok')"
RUN python3 -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('${VLM_MODEL}'); print('vlm weights ok')"

WORKDIR /agent/demo

# ROS 소싱은 원본과 동일하게 /ros_entrypoint.sh (베이스 이미지 제공) 를 쓴다.
# entrypoint.sh 는 오프라인 상태 로그만 남기고 /ros_entrypoint.sh 로 체이닝한다.
COPY docker/entrypoint.sh /agent/entrypoint.sh
RUN chmod +x /agent/entrypoint.sh
ENTRYPOINT ["/agent/entrypoint.sh"]
# participant_app.py (팀 기본) 는 vla_grounding 파이프라인(YOLO + 명령파싱 + 기하)으로
# 연결돼 있다 -- demo/participant_app.py 가 `from vla_grounding_agent import
# VLAGroundingAgent` 를 하고, mec_sim_ice/demo_agents/ 가 /agent/mec_sim_ice/demo_agents
# 로 baking되며 PYTHONPATH 에 포함되므로 vla_grounding_agent.py 가 flat import 로
# 그대로 잡힌다. YOLO 가중치(best.pt)는 .gitignore 예외로 강제 커밋돼 있다.
CMD ["python3", "participant_app.py"]
