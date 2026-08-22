너는 marc-starter-kit 리포지토리(MARC 2026 대회, 팀 mec_sim_ice) 루트에서 작업한다.
먼저 `MARC2026_STAGE1_BRIEF.md`를 전부 읽어라. 이 문서가 유일한 기준 문서다. 이 프롬프트와
문서 내용이 다르면 문서를 따르고, 그래도 애매하면 추측하지 말고 나에게 물어봐라.

# 지금 할 일 (범위를 좁혀서 명확히 한다)

목표는 딱 하나다: `demo/mock_agent_vla.py`를 실제 인식 코드로 교체하고, **연구실 데스크탑에서
CCTV 기록을 한 번 받아오는 것**까지. VLM 튜닝이나 반복 실험은 이 프롬프트의 범위가 아니다 —
기록을 받아온 뒤에는 GPU 있는 팀원 각자의 PC에서 별도로 진행한다(BRIEF.md §1, §9 참고).

# 하드웨어 현실 (중요 — 오해하기 쉬운 부분, 반드시 이해하고 시작해라)

Isaac Sim이 도는 GPU 머신은 **기계과 연구실 데스크탑 딱 한 대**뿐이다(RTX 5060 Ti, VRAM 8GB —
Isaac Sim 권장 최소 16GB에 못 미쳐 해상도를 낮춰 돌리는 중). 팀원 개인 노트북은 대부분
GPU가 없다. 주최측 채점 하드웨어(48GB급)는 우리가 접근할 수 있는 게 절대 아니다 — 그건
채점 시점에 주최측이 우리 Docker 이미지를 돌리는 자기네 장비이고, 스펙은 참고용일 뿐이다.

GPU를 실제로 쓸 수 있는 지점은 그 데스크탑 하나뿐이고, 그 위에서도 Isaac Sim과 우리
모델이 8GB를 나눠 써야 한다. 그래서 이번 세션에서는 **VLM 없이 탐지기만 켜고** 기록만
남긴다 — 그 이유는 아래 2단계에서 다시 설명한다.

# 0단계 — 통합 (GPU 불필요, 지금 있는 아무 머신에서나)
1. `marc_stage1_v3.tar.gz`를 리포 루트에 풀어라. `stage1/`, `demo/agent_vla.py`,
   `tests/`, `tools/`, `scripts/`, `docker/`가 생긴다.
2. `python tests/test_geometry.py`를 돌려 통과하는지 확인해라. 이 단계는 GPU도
   marc_sdk도 필요 없다 — GPU 없는 노트북에서 진행해도 된다.
3. `demo/participant_app.py`를 열어 `demo/participant_app.patch`의 3줄 변경을 적용해라
   (import, `self.vla` 생성, `_on_mission`에 `set_time_limit` 호출). Stage 2 쪽
   `_on_stage2_mission`에도 동일하게 `self.vla.set_time_limit(mission.time_limit)`을 추가해라.
4. `participant_app.py`의 `_on_score`에 문항별 세부 점수를 JSONL로 남기는 로깅을 추가해라
   (라운드 번호, 그 라운드의 voice_command, score.scores 전체를 한 줄씩 append). 파일 경로는
   `MARC_SCORE_LOG` 환경변수로 받고 기본값은 `./results/scores.jsonl`.
5. `demo/mock_agent_vla.py`는 지우지 말고 그대로 남겨라 — 안전망 폴백 참고용이다.

이 단계가 끝나면 무엇을 했고 무엇을 확인했는지 요약해서 나에게 보고해라. 다음 단계로
넘어가기 전에 내 확인을 받아라.

# 1단계 — Docker 통합 (GPU 불필요)
1. 기존 `demo/Dockerfile`(이 리포에 있는 원본)을 읽고, 어떤 베이스 이미지를 쓰는지,
   `marc_sdk`를 어떤 경로에 COPY하는지, PYTHONPATH를 어떻게 잡는지 파악해라.
2. `docker/Dockerfile`(내가 만든 것)을 그 구조에 맞춰 고쳐라. marc_sdk COPY 경로,
   PYTHONPATH, ROS 소싱 방식을 원본 Dockerfile과 일치시키되, 가중치 baking과
   HF_HUB_OFFLINE 설정은 유지해라. 이미지에 굽는 모델은 제출용 크기(7B급)를 기본으로
   유지해라 — 이건 채점 머신(48GB)에서 도는 것이지 우리 8GB 데스크탑에서 도는 게 아니다.
3. `docker-compose.yml`도 스타터킷 기존 `demo/docker-compose.yml`의 환경변수 이름
   (MARC_TEAM_ID, MARC_TOKEN, ROS_DOMAIN_ID 등)과 정확히 맞춰라.
4. 지금은 Docker 이미지를 굳이 빌드하지 않아도 된다. 이번 세션의 실제 실행은
   `launch.sh`(호스트 직접 실행)로 한다 — 2단계 참고. Dockerfile은 통합만 해두고
   내가 별도로 "Docker 빌드해라"라고 하기 전까지는 빌드를 시도하지 마라(오래 걸린다).

# 2단계 — 연구실 데스크탑에서 첫 기록 세션 (GPU 필요, VLM은 끈다)

이 단계부터는 연구실 데스크탑(SSH 또는 원격 데스크탑, 물리 접근도 가능)에서 진행한다.

**왜 VLM을 끄는가**: 8GB를 Isaac Sim과 나눠 써야 해서 이번 세션에 VLM까지 올리면
OOM 위험이 크다. 대신 이번엔 **탐지기(Grounding DINO)만 켜고 CCTV 기록만 남긴다.**
VLM 검증은 이 기록을 GPU 있는 팀원 개인 PC로 옮겨서 Isaac Sim 없이 따로 한다
(BRIEF.md §1 "8GB 한 장으로 VLM을 실제로 테스트하는 방법" 참고) — 그건 다음 프롬프트다.

시작 전 반드시:
- 그 데스크탑을 지금 다른 팀원이 쓰고 있지 않은지 확인해라(GPU가 한 장뿐이라 동시
  사용 시 충돌하거나 OOM 난다). 확인 안 됐으면 나에게 먼저 물어라.
- 팀원이 Isaac Sim 플랫폼을 이미 띄워뒀는지 확인해라. **너는 `simulation-platform/`을
  기동·재기동·중지하지 마라.** 로그에 `[Runtime] Startup complete`와
  `Auto-plan: waiting for a participant to register...`가 뜬 것을 팀원이 확인해줘야
  다음으로 넘어간다. 안 떠 있으면 실행하지 말고 나에게 알려라.

절차:
1. `problems.yaml`에서 `lf01`, `lf02`, `sar01` 세 개만 `enabled: true`로 남기고
   나머지는 `false`로 바꿔라 (원본은 `problems.yaml.bak`으로 백업). 처음부터 33문항을
   전부 돌리지 마라 — 코드에 문제가 있으면 서버 시간을 통째로 날린다.
2. `bash scripts/diag_link.sh`를 실행해 `/marc/ops/announce`, CCTV 이미지, `/tf_static`,
   `/clock`이 다 보이는지 확인해라. 하나라도 안 보이면 여기서 멈추고 나에게 보고해라 —
   코드를 건드리지 마라, 대개 플랫폼이 아직 다 안 떴거나 ROS 환경 문제다.
3. `nvidia-smi --query-gpu=memory.free --format=csv`로 남은 VRAM을 확인하고 나에게
   보고해라. `MARC_DISABLE_VLM=1`로 진행하므로 이 세션에서는 몇 GB든 상관없이
   진행 가능하지만, 숫자를 남겨두면 다음 VLM 세션 계획에 참고가 된다.
4. 접속 명령을 정리해서 제시해라. 예:
   ```
   MARC_DISABLE_VLM=1 MARC_RECORD_DIR=./records MARC_TEAM_ID=mec_sim_ice \
   MARC_TOKEN=... ./launch.sh mec_sim_ice
   ```
   내 확인 후 실행하고, 3라운드가 제출되는지 로그로 확인해라. `_on_score` JSONL
   로그(`./results/scores.jsonl`)도 남는지 확인해라.
5. `./records/` 아래 라운드별 폴더(이미지 + meta.json)가 생겼는지 확인하고, 폴더
   목록과 대략적인 총 용량을 나에게 보고해라. 이게 이번 세션의 최종 산출물이다.
6. `problems.yaml.bak`을 원래대로 복구할지는 나와 상의해라 — 이후 33문항 전체
   기록은 별도로 진행할 것이므로 아직 복구하지 않아도 된다.

**여기서 멈춰라.** VLM을 켜거나, `records/`를 옮기거나, `eval_offline.py`로 결과를
분석하는 건 다음 단계다. 이번 세션은 "코드 통합 + 최소 기록 확보"까지만 한다.

# 절대 규칙
- `time.time()` / `time.sleep()`을 새로 작성하는 코드에 쓰지 마라. 시간은 전부
  `client.now_s()` 또는 `pipeline`에 주입된 `now` 콜러블을 통해서만 다뤄라.
- `marc_sdk/` 폴더 안의 어떤 파일도 수정하지 마라. 필요한 게 있으면 나에게 먼저 물어라.
- `stage1/pipeline.py`나 `stage1/geometry.py`의 로직을 바꿔야 할 이유가 생기면,
  바꾸기 전에 왜 바꿔야 하는지와 어떤 대안을 검토했는지 먼저 설명하고 내 승인을 받아라.
  특히 `LANDMARK_VOCAB`, `OBJECT_VOCAB`의 키 이름(예: `trash_can`)은 채점 어휘와
  맞춰둔 것이니 함부로 바꾸지 마라.
- 런타임에 인터넷에 접속하는 코드(모델 자동 다운로드 등)를 추가하지 마라. 빌드 시점은 허용.
- `simulation-platform/`을 기동·재기동·중지하는 명령을 절대 실행하지 마라. 그건 팀원의
  책임이다. GPU가 한 장뿐이라 잘못 건드리면 팀 전체 작업이 멈춘다.
- GPU 메모리를 확인하지 않고 모델을 로드하지 마라. 항상 `nvidia-smi`로 여유를
  먼저 보고, 그 근거를 보고에 남겨라.
- 이번 세션에서는 VLM을 켜지 마라(`MARC_DISABLE_VLM=1` 고정). VLM 검증은 별도 세션이다.
- 확실하지 않은 것(모델 성능, 파라미터 값, GT 정의)은 추측해서 코드에 하드코딩하지 말고
  "실측 필요"라고 주석으로 남기고 나에게 보고해라.

# 보고 방식
각 단계가 끝날 때마다: 무엇을 바꿨는지(파일 목록), 무엇을 테스트했는지, 남은 VRAM이
얼마였는지, 무엇이 아직 확인 안 됐는지를 세 줄 이내로 요약해라. 장황한 설명 대신
다음에 내가 뭘 결정해줘야 하는지를 명확히 물어봐라. 데스크탑 사용 여부를 다른
팀원과 조율해야 하는 시점은 반드시 먼저 말하고 내 지시를 기다려라.
