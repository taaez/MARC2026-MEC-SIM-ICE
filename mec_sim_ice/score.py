#!/usr/bin/env python3
"""chungmu 오프라인 자가 채점 단축 실행기.

긴 경로를 타고 들어가지 않고 리포 루트에서 바로 돌리기 위한 얇은 래퍼다.
실제 로직은 vla_grounding/validation/offline_score.py 에 있다 (옵션도 그대로 전달된다).

    python3 mec_sim_ice/score.py
    python3 mec_sim_ice/score.py --conf 0.35 --json report.json

VRAM 8GB 한 장에서 나눠 돌리기 (모델별 스위치 -- offline_score.py 참고):

    python3 mec_sim_ice/score.py --no-dino --no-vlm      # YOLO+기하만 (기준선)
    python3 mec_sim_ice/score.py --dino --no-vlm         # +DINO (제출 기본 구성)
    MARC_DINO_BATCH=2 python3 mec_sim_ice/score.py --dino --no-vlm   # 8GB 안전
    python3 mec_sim_ice/score.py --no-dino --vlm         # +소형 LM 만 단독
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "vla_grounding", "validation"))

from offline_score import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
