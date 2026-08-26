#!/usr/bin/env python3
"""chungmu 오프라인 자가 채점 단축 실행기.

긴 경로를 타고 들어가지 않고 리포 루트에서 바로 돌리기 위한 얇은 래퍼다.
실제 로직은 taejeong/validation/offline_score.py 에 있다 (옵션도 그대로 전달된다).

    python3 mec_sim_ice/score.py
    python3 mec_sim_ice/score.py --conf 0.35 --json report.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "taejeong", "validation"))

from offline_score import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
