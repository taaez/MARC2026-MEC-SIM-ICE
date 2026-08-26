# mec_sim_ice — MARC 2026 Stage 1

```
docker-compose.yml        # 리포 루트. 주최측은 `docker compose up` 만 실행
docker/                   # Dockerfile(가중치 baking), requirements, entrypoint
stage1/geometry.py        # 픽셀→world 역투영, 관계 점수 (ROS/torch 불필요)
stage1/pipeline.py        # 명령 파서, 탐지기, 카메라 선택, payload 생성
stage1/agent.py           # marc_sdk 연동 진입점 (mock_agent_vla.py 대체)
tests/test_geometry.py    # 합성 카메라 왕복 투영 검증
scripts/setup_checklist.sh# 환경 구축 3.2~3.6
scripts/verify_offline.sh # 제출 전 오프라인 검증
models/                   # 파인튜닝 YOLO 가중치(yolo_marc.pt) 위치. 없으면 Grounding DINO 만 사용
DESIGN.md                 # 결정·대안·리스크
```

통합 절차: 스타터킷 `demo/` 를 복사한 리포에 위 파일을 얹고, `marc_sdk/` 디렉토리가
`docker/Dockerfile` 의 `COPY marc_sdk /agent/marc_sdk` 경로에 있도록 맞춘다.

로컬(8GB) 개발:
```
MARC_VLM_4BIT=1 docker compose build --build-arg VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
MARC_VLM_4BIT=1 MARC_TOKEN=... docker compose up
```
