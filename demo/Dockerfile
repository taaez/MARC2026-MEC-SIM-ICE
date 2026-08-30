# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - Participant demo application (Docker submission template)
# ---------------------------------------------------------------------------
# The participant application must be developed/submitted as a Docker image (required).
# The standard entry point is `docker compose up`, and the team ID/token are
# entered in docker-compose.yml.
#
# Build context = the starter kit root (copies marc_sdk + demo together).
#   -> see compose build.context=.. , dockerfile=demo/Dockerfile.
# ---------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 + ROS2 Humble + Python 3.10 (provides rclpy / *_msgs, default RMW=FastDDS)
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Participant algorithm dependencies. numpy + pyyaml (stdlib/ROS2 for the rest), placo for
# the closed-loop arm IK pick (Pinocchio-based), ultralytics for YOLO11 detection/pose, and
# ros-humble-tf2-ros for the CCTV extrinsic (world<-camera) lookups in vla_grounding_agent.py.
# The network is used at build time only -- the judging runtime environment has no internet
# (no runtime dependency) -- so best.pt and the yolo11n-pose.pt COCO weights must both be
# present in the image after this step.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy python3-yaml python3-pip ros-humble-tf2-ros \
    && pip3 install --no-cache-dir placo ultralytics opencv-python-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /agent

# context = starter kit root -> copy marc_sdk (library) + manipulation (arm kit, for the
# world->arm_base tf helper) + demo (organizer agent) + mec_sim_ice (team code: the
# vla_grounding/ Stage 1 pipeline package + demo_agents/ local modules, incl. weights/).
# mec_sim_ice/ is copied wholesale, preserving its own local layout under /agent, so
# vla_grounding_agent.py's flat `from vla_grounding.grounding import ...`-style imports and
# vla_grounding/'s own dirname(dirname(__file__))-based weight-path lookups resolve exactly
# like they do locally (see docker/Dockerfile for the same fix, applied there first).
COPY marc_sdk     /agent/marc_sdk
COPY manipulation /agent/manipulation
COPY demo         /agent/demo
COPY mec_sim_ice  /agent/mec_sim_ice

# Same RMW as the runtime platform (FastDDS). Import paths for marc_sdk (/agent) + demo
# local modules (/agent/demo) + vla_grounding package/demo_agents flat modules (/agent/mec_sim_ice).
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV PYTHONPATH=/agent:/agent/demo:/agent/mec_sim_ice:/agent/mec_sim_ice/demo_agents

WORKDIR /agent/demo

# Bake the COCO-pretrained pose weights into the image at build time (runtime has no
# internet). Downloaded into this WORKDIR so YOLO('yolo11n-pose.pt') finds it locally
# at runtime instead of retrying the download. mec_sim_ice/demo_agents/weights/best.pt
# (the fine-tuned detector) must already exist before building -- see
# mec_sim_ice/training/train_yolo11.py.
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

# The ros:humble ENTRYPOINT (/ros_entrypoint.sh) sources /opt/ros/humble/setup.bash
# and then runs the CMD below. team_id/token are injected via the compose env (from_env).
CMD ["python3", "participant_app.py"]
