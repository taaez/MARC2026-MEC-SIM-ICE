#!/usr/bin/env bash
set -e
# 런타임 네트워크 상태를 로그로 남긴다 (채점 환경 재현 확인용). 실패해도 무시.
( timeout 2 python3 -c "import socket; socket.create_connection(('8.8.8.8',53),1)" && echo "[entrypoint] WARNING: internet reachable (not the judge env)" ) 2>/dev/null || echo "[entrypoint] offline OK"
# ROS 소싱은 원본 demo/Dockerfile 과 동일하게 베이스 이미지의 /ros_entrypoint.sh 에 맡긴다
# (/opt/ros/humble/setup.bash 를 source 한 뒤 CMD 를 exec 한다).
exec /ros_entrypoint.sh "$@"
