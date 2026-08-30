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
# 2026-08-29: VL(비전-언어) 모델을 완전히 뺐다(taejeong/vlm_fallback.py 참고) --
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
# demo/ 와 mec_sim_ice/demo_agents/ 를 같은 /agent/demo 로 순서대로 COPY 해서 원래
# 하나였던 디렉토리처럼 합친다 (agent_vla.py 등 로컬 임포트가 그대로 동작).
WORKDIR /agent
COPY marc_sdk           /agent/marc_sdk
COPY mec_sim_ice/stage1   /agent/stage1
COPY mec_sim_ice/taejeong /agent/taejeong
COPY demo                /agent/demo
COPY mec_sim_ice/demo_agents /agent/demo
COPY manipulation        /agent/manipulation
COPY mec_sim_ice/models   /agent/models

# 런타임 플랫폼과 같은 RMW (FastDDS). import 경로: marc_sdk·stage1(/agent) + demo 로컬 모듈(/agent/demo).
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV PYTHONPATH=/agent:/agent/demo

# 빌드 단계에서 모델이 오프라인으로 로드되는지 자가 검증 (실패하면 빌드 실패 → 제출 전 발견)
RUN python3 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('${DET_MODEL}'); print('det ok')"
# VLM 은 이제 텍스트 전용(SmallLmBackend, taejeong/vlm_fallback.py) -- VL
# 모델을 여기 넣으면(예: 실수로 되돌리면) AutoTokenizer 로드 자체가 chat
# template 없이 깨지거나 여기서 걸린다. AutoProcessor 가 아니라 실제 코드가
# 쓰는 클래스(AutoTokenizer)로 검증해야 이 불일치를 잡는다.
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${VLM_MODEL}'); print('vlm ok')"
# 위 두 줄은 processor/tokenizer 만 확인한다 -- 그건 가중치 파일을 안 건드리므로,
# snapshot_download 의 allow_patterns 에 안 걸린 형식(*.bin 등)으로만 올라온
# repo 였다면 **빌드는 통과하고 런타임에 조용히 폴백이 꺼진다**(로드 실패를
# taejeong_agent_vla.py 가 경고만 남기고 삼킨다). 실제 모델 클래스로 한 번 더
# 로드해서 그 경우를 빌드에서 잡는다.
RUN python3 -c "from transformers import AutoModelForZeroShotObjectDetection as M; M.from_pretrained('${DET_MODEL}'); print('det weights ok')"
RUN python3 -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('${VLM_MODEL}'); print('vlm weights ok')"

WORKDIR /agent/demo

# ROS 소싱은 원본과 동일하게 /ros_entrypoint.sh (베이스 이미지 제공) 를 쓴다.
# entrypoint.sh 는 오프라인 상태 로그만 남기고 /ros_entrypoint.sh 로 체이닝한다.
COPY docker/entrypoint.sh /agent/entrypoint.sh
RUN chmod +x /agent/entrypoint.sh
ENTRYPOINT ["/agent/entrypoint.sh"]
# participant_app.py (팀 기본) 는 taejeong 파이프라인(YOLO + 명령파싱 + 기하)으로
# 연결돼 있다 -- demo/participant_app.py 가 `from taejeong_agent_vla import
# TaejeongVLAGrounding` 을 하고, mec_sim_ice/demo_agents/ 가 COPY 로 같은
# /agent/demo 에 합쳐지므로 taejeong_agent_vla.py(=demo_agents/ 안의 그 파일)
# 가 그대로 잡힌다.
#
# 2026-08-29 결정 완료(이전엔 ⚠️ 로 표시돼 있었다): taejeong 파이프라인으로
# 확정한다. 이전엔 이 CMD 가 GroundingDINO+Qwen2.5-VL 전용 진입점
# (participant_app_agent_vla.py, agent_vla.py 기반 -- taejeong 과 다른 구현체)
# 을 가리키고 있어서, 지금까지 이 리포에서 검증해 온 taejeong 개선사항(YOLO
# 타일링/명령파싱/기하 판정 등)이 실제로는 하나도 제출되지 않는 상태였다.
# YOLO 가중치는 best.pt 를 강제로 커밋해서 해결했다(.gitignore 의 `*.pt` 에
# 걸려 그동안 아무 브랜치에도 안 올라가 있었다 -- 새로 clone 하면 파일 자체가
# 없어서 런타임에 크래시났을 것).
CMD ["python3", "participant_app.py"]
