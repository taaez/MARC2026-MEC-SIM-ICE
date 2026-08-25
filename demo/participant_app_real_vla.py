"""MARC 2026 participant app -- RealVLAGrounding entry point (root real_agent_vla.py).

Identical to participant_app.py (protocol wiring, Stage 2 A* nav + placo pick,
per-round score JSONL logging) except Stage 1/2 grounding runs
``RealVLAGrounding`` from the starter-kit root ``real_agent_vla.py`` instead of
the taejeong pipeline. Kept as a separate entry point -- selected via
``MARC_ENTRYPOINT`` like the Grounding DINO experiment -- so the team default
``participant_app.py`` wiring stays untouched.

The swap works by replacing the ``TaejeongVLAGrounding`` name inside the
imported ``participant_app`` module BEFORE constructing ``DemoParticipant``
(its ``__init__`` resolves that name at call time), so the YOLO weights load
exactly once and every other behavior is inherited as-is.

Run (from demo/, platform already up and playing):
    MARC_ENTRYPOINT=participant_app_real_vla.py ./launch.sh <team_id>

Scores land in ./results/scores.jsonl exactly like the default entry point,
so taejeong/validation/verify_run.py + score_report.py work unchanged.
"""

import logging
import os
import sys

# real_agent_vla.py lives at the starter-kit root (parent of demo/). launch.sh
# already puts the root on PYTHONPATH; this insert keeps a bare
# `python3 participant_app_real_vla.py` working too.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import participant_app
from real_agent_vla import RealVLAGrounding

# DemoParticipant.__init__ does `self.vla = TaejeongVLAGrounding(self.client)`;
# the name is looked up in participant_app's module globals at call time, so
# rebinding it here swaps the grounding implementation without a double model
# load and without touching participant_app.py.
participant_app.TaejeongVLAGrounding = RealVLAGrounding


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("participant_app").info(
        "[entry] Stage 1/2 grounding = RealVLAGrounding (real_agent_vla.py)")
    participant_app.DemoParticipant().run()


if __name__ == "__main__":
    main()
