"""MARC 2026 participant app -- RecordOnlyVLAGrounding entry point (mec_sim_ice/record_agent_vla.py).

Identical to participant_app.py (protocol wiring, Stage 2 A* nav + placo pick,
per-round score JSONL logging) except Stage 1/2 grounding runs
``RecordOnlyVLAGrounding`` from ``record_agent_vla.py`` instead of the
vla_grounding pipeline. Kept as a separate entry point -- routing recording
through the exact same code path as the live submission would slow down data
collection -- so the team default ``participant_app.py`` wiring stays untouched.

The swap works by replacing the ``VLAGroundingAgent`` name inside the
imported ``participant_app`` module BEFORE constructing ``DemoParticipant``
(its ``__init__`` resolves that name at call time), so the YOLO weights load
exactly once and every other behavior is inherited as-is.

Run (from demo/, platform already up and playing):
    MARC_ENTRYPOINT=participant_app_record.py ./launch.sh <team_id>

Scores land in ./results/scores.jsonl exactly like the default entry point,
so vla_grounding/validation/verify_run.py + score_report.py work unchanged.
"""

import logging
import os
import sys

# record_agent_vla.py lives at mec_sim_ice/ (this file's grandparent dir --
# mec_sim_ice/demo_agents/participant_app_record.py -> mec_sim_ice/). launch.sh
# already puts it on PYTHONPATH; this insert keeps a bare
# `python3 participant_app_record.py` working too.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import participant_app
from record_agent_vla import RecordOnlyVLAGrounding

# DemoParticipant.__init__ does `self.vla = VLAGroundingAgent(self.client)`;
# the name is looked up in participant_app's module globals at call time, so
# rebinding it here swaps the grounding implementation without a double model
# load and without touching participant_app.py.
participant_app.VLAGroundingAgent = RecordOnlyVLAGrounding


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("participant_app").info(
        "[entry] Stage 1/2 grounding = RecordOnlyVLAGrounding (record_agent_vla.py)")
    participant_app.DemoParticipant().run()


if __name__ == "__main__":
    main()
