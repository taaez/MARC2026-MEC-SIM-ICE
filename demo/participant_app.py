"""MARC 2026 participant app -- team default entry point.

Stage 1 grounding runs the vla_grounding pipeline (``vla_grounding_agent`` ->
``vla_grounding/``: closed-vocabulary command parsing, pure-geometry grounding, and
the organizer's official pose x support situation rules), on top of the team's
YOLO detector weights (``demo/weights/best.pt``, classes per
``test/det_classes.py``). Stage 2 is the starter kit's occupancy A* driving
(cmd_vel) + placo pick, unchanged.

``demo/mock_agent_vla.py`` is intentionally left untouched: the guide's
requirement is the ``process()`` / ``process_stage2()`` signature returning a
``GroundingResult``, which ``VLAGroundingAgent`` satisfies -- so this file
swaps the implementation without changing the protocol wiring.

Per-round scores are appended to ``./results/scores.jsonl``
(``MARC_SCORE_LOG``); ``vla_grounding/validation/`` reads that file for self-scoring
against marc2026_chungmu. See ``vla_grounding/RUNBOOK.md``.
The protocol/ROS wiring (register, subscribe/publish, world_pose, occupancy) is all
handled by ``MARCClient``, so only the participant logic (mock VLA + nav) is connected
through handlers.

Run:
    source /opt/ros/humble/setup.bash
    export MARC_TEAM_ID=u1
    export MARC_TOKEN=<token>            # your assigned token (required; issued by the organizers)
    python3 participant_app.py
  or
    ./launch.sh u1

Precondition: the simulation platform must be running first.
"""

import json
import logging
import math
import os
import threading
import time
from collections import deque

import numpy as np

from marc_sdk import MARCClient

from vla_grounding_agent import VLAGroundingAgent
from mock_agent_navigation import (
    OccupancyGridPlanner, PathFollower, GoalSeekController,
    cap_angular, obstacles_excluding_goal,
)
# Closed-loop placo pick from the manipulation kit (arm_pick.py): grasp_yaw wrist-roll +
# DLS servo, targeting the object position. Requires placo (installed in the demo image).
# The fixed-keyframe mock in mock_agent_manipulation.py remains as a dependency-free
# fallback reference.
from arm_pick import run_pick_sequence
from manipulation import transform_via_tf2
# Lidar obstacle layer -- the occupancy map holds static road only, so parked vehicles and
# the like have to be sensed. See mec_sim_ice/demo_agents/lidar_obstacles.py.
from lidar_obstacles import LidarObstacleLayer

log = logging.getLogger("participant_app")

# -- Stage 2 navigation parameters --
# Demo-internal giveup so the drive loop cannot spin forever if a goal is unreachable. This is NOT
# the competition timeout - the platform counts and enforces the real Stage 2 time limit.
STAGE2_NAV_TIMEOUT_S = 1800.0
STAGE2_NAV_RATE_HZ = 10.0
STAGE2_NAV_DT = 1.0 / STAGE2_NAV_RATE_HZ
STAGE2_INFLATION_M = 0.25      # erode free by the robot radius (avoids road<->grass step stalls)
# Stall test: total displacement over a trailing STAGE2_STUCK_TIME_S-second window must
# exceed STAGE2_STUCK_MOVE_M, or it is a stall. This used to compare consecutive ticks
# (0.1s apart) instead of a window -- since GoalSeekController's decel is
# linear=clamp(K_LINEAR*dist, ...) with K_LINEAR=0.5, speed drops under
# STAGE2_STUCK_MOVE_M/STAGE2_NAV_DT = 0.5 m/s the moment dist < 1.0m, so *every* tick of a
# normal, correctly-progressing final approach could fail that single-tick test and the
# stall clock would never reset. Chained short in-place rotations (heading_err > 45deg,
# linear=0 -- exactly 0 m/tick) between frequent lidar replans compound this: none of them
# individually looks stuck, but nothing between them resets the old per-tick check either.
# Measured live: this fired mid-lidar-replan-burst on rt02 while the chassis was still
# making net progress. A window fixes both: slow-but-real progress accumulates past the
# threshold, and a genuinely stationary chassis still gets caught after the same 8s.
STAGE2_STUCK_TIME_S = 8.0      # trailing window length (s) -- see above; not independently tuned
STAGE2_STUCK_MOVE_M = 0.05     # total displacement (m) required within that window
GOAL_TOLERANCE = 0.7           # reached test (m)

# -- Stall memory: obstacle evidence without a sensor --
# The occupancy map holds static road only -- parked vehicles/bicycles/people are absent
# (confirmed on the platform: the robot drives into a parked car and simply stops). A* then
# keeps returning the same path through it, so the existing stall->replan loop never escapes
# and burns the whole Stage 2 budget. Remembering where the chassis stopped turns that loop
# into a real recovery. All three values are robot-scale, not scene-scale.
# Lidar sweeps folded into the layer, in nav ticks. The nav loop runs at
# STAGE2_NAV_RATE_HZ (10 Hz) and a VLP-16 sweeps at ~10 Hz, so every other tick keeps up
# with the sensor without spending a scan parse on ticks that cannot carry new information.
LIDAR_UPDATE_EVERY_TICKS = 2

# A single lidar update failure must not permanently disable avoidance for the rest of the
# run -- a fresh subscription can throw (not just return None) for its first reading or two
# while ROS/DDS discovery settles, exactly the pattern _ensure_cctv_warm and
# _wait_basket_state already guard against for other topics. This many *consecutive*
# failures (~1s at LIDAR_UPDATE_EVERY_TICKS) still catches a genuinely broken/absent sensor
# without wasting the whole run parsing garbage.
LIDAR_MAX_CONSEC_FAILS = 5

ROBOT_RADIUS_M = STAGE2_INFLATION_M      # the radius the planner already erodes free space by

# Reactive near-field brake -- independent of A*/replanning, checked every tick. The
# lidar-triggered replan (blocks_path) and the stall-memory replan are both deliberative:
# they wait for a full cycle (grid refresh + A* + a new PathFollower) before the chassis
# command changes, which is at least one nav tick and often several. Something already
# close and roughly ahead cannot wait that long -- this just zeroes linear velocity for
# that one tick so the chassis stops before contact, then lets the existing stall/lidar
# replan machinery (already reacting to near-zero movement) route around it. It does not
# steer or replan by itself on purpose -- it only has to buy the time the deliberative
# path needs, not solve avoidance itself.
REACT_STOP_DIST_M = ROBOT_RADIUS_M + 0.35   # chassis radius + a few nav ticks of stopping slack
REACT_STOP_CONE_DEG = 40.0                  # narrow and close on purpose: A* already keeps its
                                             # own inflation clearance from known obstacles, so a
                                             # wide/generous cone here just re-questions routes A*
                                             # already judged clear and can deadlock a genuinely
                                             # passable tight corridor (measured: 70deg/0.9m timed
                                             # out rt02_parking_umbrella offline, which passes
                                             # without this brake -- see stage2_sim.py --hidden-obstacles)

# How long the brake may hold linear at 0 (while the follower wants to move) before this
# counts as a stall on its own, instead of waiting for the generic STAGE2_STUCK_TIME_S(8s)
# window to notice the same thing indirectly.
#
# Measured live (rt01_parking_tumbler, post-hang-fix): once something sits persistently
# within the brake's stop zone, the follower keeps recomputing a nonzero linear every tick
# (it does not know the brake exists) and the brake keeps clamping it back to 0 -- angular
# is untouched, so heading keeps refining toward the waypoint while XY stays frozen, which
# *looks* exactly like a slow-converging turn but is actually the brake holding the whole
# time. This repeated 4 times in one leg1 approach, ~30-40s lost per occurrence, before the
# generic stall detector's 8s window finally caught up and forced a replan. Since the brake
# already knows precisely why linear is 0 (unlike the generic stall test, which only sees
# XY not moving and has to infer a cause), there is no reason to wait the full 8s: treat a
# sustained brake-hold as an immediate stall -- same _remember_blockage + replan path, just
# triggered by the mechanism that actually caused the freeze instead of by its symptom.
BRAKE_STALL_S = 1.5
BLOCKAGE_RADIUS_M = 0.6                  # wider than the chassis, so A* must route around it
BLOCKAGE_GROWTH_M = 0.4                  # stalled inside a known blockage -> it was too small
BLOCKAGE_MAX_RADIUS_M = 2.0              # cap, so a bad mark cannot wall off the map

# -- Pick approach --
# Stop the chassis this far (m) in front of the object, then close the last stretch with a
# straight approach (the planner cannot enter a tight spot fully). Avoids driving over the object.
TUMBLER_STANDOFF_M = 0.58
# car_base(world_pose) -> arm_base forward offset. Used ONLY by the offline fallback in
# _object_arm_xy when tf2 is unavailable (no ROS / no live platform); live runs read the
# real offset from the world->arm_base_link tf instead. Unverified assumption -- see
# manipulation/world_to_arm_base.py for why the URDF cannot supply it.
ARM_BASE_FWD = 0.18
# Grasp point relative to the object along +fwd. Negative = slightly past the object center
# so the gripper closes around its body (a small backoff grabbed only the edge -> dropped in
# transit), tuned slightly forward for a secure grip.
GRASP_FWD_BACKOFF = -0.01

# Retry budget for a failed pick (empty grasp, or grasped but missed the rear basket on place).
MAX_PICK_ATTEMPTS = 3

# -- Stage 2 goals --
# The Stage 2 task drives to the target object and then delivers it to the owner zone.
# Both goals come from the runtime protocol only: the pickup goal from the reveal (msg 411)
# hint_center, with the Stage 2 grounding answer's target_coord as a fallback (see
# _wait_for_collect_goal), and the delivery goal from owner_position in the mission
# (msg 211). There is deliberately no mock_demo_data.yaml fallback for either -- that file
# is pre-extracted from one scenario, so on any other scenario it aims tens of metres away
# and fails silently. No owner_position -> no delivery leg (see _drive).

# Approximate-location radius (m) assumed only when a reveal never arrives. hint_radius is
# the platform's own statement of how uncertain hint_center is, so the goal-relaxation
# distances below (planner snap, obstacle exclusion) are derived from it rather than fixed
# per scene; this is just the no-reveal fallback.
DEFAULT_HINT_RADIUS_M = 1.0


class ElapsedClock:
    """Monotonic elapsed-seconds accumulator over `client.now_s()`, immune to a backward jump.

    Why this exists: `now_s()` returns `time.time()` (a huge wall-clock epoch value) until the
    first `/clock` message arrives, then switches to sim-time (a small value). Every deadline in
    this file used to be computed once as `now_s() + budget` and compared against later `now_s()`
    calls -- if that one computation happened to land in the wall-clock-fallback window, the
    deadline is a huge epoch-based number that no future sim-time `now_s()` can ever reach, so
    `now_s() < deadline` stays true forever and the loop never exits. This is the exact bug
    documented in STAGE2_SESSION_REPORT_2026-08-22.md for demo/agent_vla.py's _observe() (fixed
    there with an iteration-count budget instead); the same pattern reappeared here in at least
    six places and was reproduced live twice in a row (participant registering while STAGE1_RUN/
    STAGE2_RUN was already in progress -- exactly maximises the chance the first now_s() call
    lands before /clock).

    Rather than patch each call site with its own iteration-count workaround (loses the sim-time
    budget semantics, which matter here since STAGE2_NAV_TIMEOUT_S etc. are meant to scale with
    RTF), this tracks elapsed sim-time as a running sum of consecutive deltas: a delta is only
    ever added if it is a plausible forward step. A jump backward (the wall-clock -> sim-time
    switch) contributes 0 instead of going negative, and a jump forward larger than one plausible
    tick is capped at 1s so a single stray large delta cannot itself satisfy a whole timeout.
    Genuine sim-time -- even sped up by RTF -- always advances in small steps between our own
    poll calls, so this never treats real progress as 0.
    """

    _MAX_TICK_S = 1.0

    def __init__(self, client):
        self._client = client
        self._last = client.now_s()
        self.elapsed = 0.0

    def tick(self):
        """Advance and return the cumulative elapsed seconds since this clock was created."""
        now = self._client.now_s()
        delta = now - self._last
        self._last = now
        if delta > 0.0:
            self.elapsed += min(delta, self._MAX_TICK_S)
        return self.elapsed


class DemoParticipant:
    """SDK-based demo participant -- connects the VLA/nav logic through handlers."""

    def __init__(self):
        self.client = MARCClient.from_env()
        # Stage 1/2 grounding: vla_grounding pipeline (command_parser -> YOLO -> grounding
        # geometry -> situation_resolver). YOLO weights load here, once, before connect().
        self.vla = VLAGroundingAgent(self.client)
        # Per-round detail-score JSONL log (see _on_score). Round -> voice_command so each
        # score line can be joined back to the question that produced it; vla_grounding/validation
        # reads this file for self-scoring.
        self._round_commands = {}
        self._score_log_path = os.environ.get("MARC_SCORE_LOG", "./results/scores.jsonl")
        # world -> arm_base_link tf, used by _object_arm_xy for the real (not assumed)
        # chassis->arm mount offset/rotation, and by _locate_object_by_depth for the
        # camera->arm transform. Created in run() AFTER client.connect() -- NOT here:
        # rclpy.init() only happens inside connect(), so a TransformListener built at
        # construction time always died with "rclpy.init() has not been called", which
        # silently disabled BOTH the real-tf path of _object_arm_xy (falling back to the
        # unverified ARM_BASE_FWD assumption) and the depth-refinement pick correction,
        # on every live run (observed in every run4/run3/... startup log tonight).
        self._tf_buffer = self._tf_listener = None
        # Stage 2 pickup goal -- filled from the approximate location hint in the reveal (msg 411).
        # If not received, fall back to the Stage 2 answer target (_stage2_target_xy).
        self._collect_xy = None
        # Object type from the same reveal -- selects a pnp_params.json override in run_pick_sequence()
        # (see arm_pick.load_params()), since one grasp/threshold set doesn't fit every object shape.
        self._target_type = None
        # base_camera/depth/info -- subscribed lazily (once) on first _locate_object_by_depth()
        # call, not per pick attempt, so retries don't pile up duplicate raw subscriptions.
        self._depth_cam_info = None
        self._depth_cam_info_sub = None
        # Fallback pickup goal -- the Stage 2 grounding answer's target_coord (set in
        # _on_stage2_mission), used only if the reveal does not arrive.
        self._stage2_target_xy = None
        # Stage 2 delivery goal (owner) -- filled from owner_position in mission (msg 211).
        # Stays None when the scenario defines no owner_zone; there is no scenario-derived
        # fallback any more, so _drive skips the delivery leg instead of aiming elsewhere.
        self._pickup_xy = None
        self._occ_logged = False
        # CCTV warm-up gate (see _ensure_cctv_warm). The platform starts Stage 1 the
        # instant registration is ACKed, but the SDK discovers CCTV topics on a 2s timer
        # and the image QoS is BEST_EFFORT/VOLATILE/depth=1 (no retained frames), so the
        # first ~2-4 rounds used to run with an empty image set. Block once, on the first
        # mission, until frames are actually flowing.
        self._cctv_warm = False
        # Extra obstructions the occupancy map omits (vehicles/bicycles/people/objects).
        # The demo filled this from mock_demo_data.yaml, i.e. from scenario ground truth the
        # competition does not hand out -- on marc2026_chungmu that list even contained each
        # pick target itself as a blocked circle. Detecting them is a participant task
        # ("laser/camera sensing" per the dev guide), so this stays empty until the lidar
        # local map lands; obstacles_excluding_goal below already handles a populated list.
        self._obstacles = []
        # Stage 2 reveal state (msg 411) beyond hint_center: hint_radius is the platform's
        # own statement of how far off the hint may be, and it sizes the goal-relaxation
        # distances in _drive so no scene's spacing is baked in.
        self._hint_radius = None
        # Lidar obstacle layer, built per Stage 2 run and shared by both legs (what leg 1
        # bumps into is usually on leg 2's way back too). Set to None once the environment
        # turns out to have no lidar, so a missing sensor costs one attempt, not one per tick.
        self._lidar_layer = None
        # Base_LiDAR -> chassis mount offset (dx, dy). Static, so tf is asked once per run;
        # (0, 0) (sensor above the chassis origin) when tf cannot supply it.
        self._lidar_mount_xy = (0.0, 0.0)
        # Consecutive lidar update failures. Only a *sustained* failure (LIDAR_MAX_CONSEC_FAILS
        # in a row) disables the layer -- see _update_lidar_layer for why a one-shot disable
        # is dangerous.
        self._lidar_fail_streak = 0
        # Elapsed-time clock for this run's deadlines -- see ElapsedClock. Built fresh in
        # _drive() (like _lidar_layer) since "elapsed" only means something relative to when
        # this run's driving started.
        self._clock = None
        self._stage2_task = ""
        self._stage2_submitted_target = None
        self._reveal_log_path = os.environ.get(
            "MARC_REVEAL_LOG", "./results/stage2_reveals.jsonl")
        self._register_handlers()

    # -- handler registration --

    def _register_handlers(self):
        c = self.client
        c.on_mission(self._on_mission)
        c.on_stage2_mission(self._on_stage2_mission)
        c.on_stage2_reveal(self._on_stage2_reveal)
        c.on_stage2_run(self._drive)
        c.on_score(self._on_score)
        c.on_state_change(self._on_state_change)
        c.on_time_expired(self._on_time_expired)

    # -- Stage 1 --

    def _ensure_cctv_warm(self, settle: float = 1.0, timeout: float = 20.0,
                          poll: float = 0.2) -> bool:
        """Wait until every discovered CCTV camera has delivered at least one frame.

        MUST run on a worker thread, never inside an SDK callback: the SDK's
        ``_discover_cctv`` runs on a 2s timer in the node's default (MutuallyExclusive)
        callback group, so blocking a callback would also freeze discovery and this
        would spin to the timeout with an empty list. From a worker thread the executor
        keeps discovering while we poll.

        Why the wait is needed: the platform poses round 1 the instant registration is
        ACKed, but CCTV topics take a few seconds to discover and the image QoS is
        BEST_EFFORT / VOLATILE / depth=1 (no retained frames), so the first rounds ran
        with an empty image set ("no camera matched", ~35 pts). Each round has a 30s
        sim-time budget and the platform does not advance until we submit, so spending
        a few seconds here only paces the platform.

        Readiness = camera count stable for ``settle`` s (discovery settled) AND every
        discovered camera has a cached frame. Returns False on timeout (caller proceeds
        with whatever is available -- the prior behaviour; a dead feed cannot hang us).
        """
        deadline = time.time() + timeout
        stable_since = None
        last_n = -1
        while time.time() < deadline:
            cams = self.client.list_cctv()
            n = len(cams)
            if n > 0 and n == last_n:
                if stable_since is None:
                    stable_since = time.time()
                elif (time.time() - stable_since >= settle
                      and all(self.client.get_cctv_image(c) is not None for c in cams)):
                    log.info("[warmup] CCTV ready: %d cameras in %.1fs", n,
                             timeout - (deadline - time.time()))
                    return True
            else:
                stable_since = None
            last_n = n
            time.sleep(poll)
        log.warning("[warmup] CCTV not fully ready after %.0fs -- proceeding with %s",
                    timeout, self.client.list_cctv())
        return False

    def _on_mission(self, mission):
        self._round_commands[mission.round] = mission.voice_command
        # Until CCTV is confirmed flowing, handle the round on a worker thread so this
        # callback returns immediately and the SDK's discovery timer can run (see
        # _ensure_cctv_warm). The platform waits for our submission (30s budget) and does
        # not send the next round until this one is scored, so there is no round overlap.
        if not self._cctv_warm:
            threading.Thread(target=self._handle_mission_deferred, args=(mission,),
                             daemon=True).start()
            return
        self._handle_mission(mission)

    def _handle_mission_deferred(self, mission):
        self._cctv_warm = self._ensure_cctv_warm()
        self._handle_mission(mission)

    def _handle_mission(self, mission):
        log.info("[Stage1] round %s/%s: \"%s\"",
                 mission.round, mission.total_rounds, mission.voice_command)
        result = self.vla.process(mission.voice_command, self.client.list_cctv())
        # result is a GroundingResult -- submit it directly.
        self.client.submit_grounding(result)

    # -- Stage 2: submit interpretation -> drive --

    def _on_stage2_mission(self, mission):
        log.info("[Stage2] task: \"%s\" (limit %.0fs)",
                 mission.task_description, mission.time_limit)
        # Together with the VLA command, the platform provides the delivery goal (owner) coordinates (msg 211 owner_position).
        op = getattr(mission, "owner_position", None)
        if op and len(op) >= 2:
            self._pickup_xy = (float(op[0]), float(op[1]))
            log.info("[Stage2] owner_position(runtime-provided)=%s", self._pickup_xy)
        # Grounding: look up the pre-extracted Stage 2 answer for this task (mock VLA).
        self._stage2_task = mission.task_description
        result = self.vla.process_stage2(mission.task_description)
        self.client.submit_stage2_grounding(result)
        # Keep the answer's target as a fallback pickup goal (used if no reveal arrives).
        tc = result.target_coord
        if tc and len(tc) >= 2:
            self._stage2_target_xy = (float(tc[0]), float(tc[1]))
        self._stage2_submitted_target = list(tc) if tc else None
        log.info("[Stage2] submitted grounding target=%s, delivery point=%s",
                 self._stage2_target_xy, self._pickup_xy)

    def _on_stage2_reveal(self, reveal):
        """Stage 2 post-grounding reveal (msg 411) -- adopt the approximate location/type as the pick goal.

        Regardless of grounding accuracy, use the revealed approximate location
        (hint_center) as the goal of the pickup leg. There may be distractors within the
        radius, so use the type (target_type) to decide which object to pick up (the demo
        only demonstrates reaching the location).
        """
        log.info("[Stage2] reveal received -- score=%s, type=%s, center=%s, r=%s",
                 reveal.grounding_score, reveal.target_type,
                 reveal.hint_center, reveal.hint_radius)
        self._target_type = reveal.target_type or None
        if reveal.hint_center and len(reveal.hint_center) >= 2:
            self._collect_xy = (float(reveal.hint_center[0]), float(reveal.hint_center[1]))
        try:
            # 0.0 is not a usable radius -- treat it as "not stated" and fall back.
            self._hint_radius = float(reveal.hint_radius) or None
        except (TypeError, ValueError):
            self._hint_radius = None
        self._append_reveal_log(reveal)

    def _drive(self):
        """Called once in a separate thread on entering STAGE2_RUN -- pick up the object then deliver it.

        Flow: stow the arm to lower the CoG -> approach the object standoff -> straight final
        approach into arm reach -> pick (retry up to MAX_PICK_ATTEMPTS on an empty grasp or a
        missed basket) -> stow again if still empty-handed -> deliver to the owner zone ->
        task_complete. All legs share one sim-time budget.
        """
        c = self.client
        # Elapsed-time clock for this run -- see ElapsedClock. Every "deadline" from here on is
        # an elapsed-seconds budget measured against self._clock.tick(), NOT an absolute now_s()
        # value: a plain `now_s() + budget` computed here can land before the first /clock
        # message (a huge wall-clock fallback), making every later `now_s() < deadline` compare
        # against a value no sim-time now_s() can ever reach -- an infinite hang, reproduced live
        # twice (see STAGE2_SESSION_REPORT_2026-08-22.md for the same bug in agent_vla.py).
        self._clock = ElapsedClock(c)
        deadline = STAGE2_NAV_TIMEOUT_S

        # Poll briefly until the reveal (approximate location) arrives after scoring -> pickup goal.
        collect_xy = self._wait_for_collect_goal(timeout=5.0)
        # Delivery goal = runtime-provided owner_position (msg 211). None = the scenario
        # defines no owner_zone, so there is no delivery leg to run (handled at leg 2).
        pickup_xy = self._pickup_xy
        # How far the platform says hint_center may be off. Every goal-relaxation distance
        # below is derived from it, so nothing encodes one scene's spacing.
        hint_r = self._hint_radius if self._hint_radius is not None else DEFAULT_HINT_RADIUS_M
        log.info("[Stage2] navigation start -- collect=%s deliver=%s (timeout %.0fs)",
                 collect_xy, pickup_xy, STAGE2_NAV_TIMEOUT_S)

        # Fresh obstacle layer per run -- the scene is re-posed between rounds, so evidence
        # from the previous round is not evidence about this one.
        self._lidar_layer = LidarObstacleLayer()
        self._lidar_mount_xy = self._lidar_mount_offset()

        # Top-heavy robot tip-over prevention: fold the arm to HOME before driving -> lower CoG.
        self._stow_arm()

        picked = False
        # -- leg 1: approach the object front standoff, then straight-approach into arm reach, then pick --
        approach = self._standoff_goal(collect_xy, TUMBLER_STANDOFF_M)
        log.info("[Stage2] leg1 -> object %s front standoff %s",
                 tuple(round(v, 2) for v in collect_xy), tuple(round(v, 2) for v in approach))
        # Aggressive planner tuning for the tight pick approach (low inflation + goal-exclusion
        # and snap sized by the stated hint uncertainty rather than by this scene's spacing).
        self._navigate_to(approach, deadline, tol=0.45, inflation_m=0.10,
                          exclude_m=max(0.6, hint_r), snap_m=max(1.0, hint_r))
        if not c.is_running:
            return
        # Square up before creeping in -- see _align_to_object for why this can't be
        # skipped even though _approach_straight also steers.
        self._align_to_object(collect_xy, deadline)
        # Final straight approach: drive straight at the object until arm_base fwd reaches grasp reach.
        self._approach_straight(collect_xy, deadline, target_fwd=0.40)
        # Stop! the chassis driver keeps applying the last cmd_vel, so emit 0 before the pick
        # (otherwise residual speed overshoots the object).
        for _ in range(5):
            c.send_cmd_vel(0.0, 0.0)
            c.sleep(0.1)
        log.info("[Stage2] standoff stop -- ready to pick")

        # Retry: is_grasping() (via run_pick_sequence's returned held) catches an empty-handed
        # close; is_basket_occupied() additionally catches a held object that missed the rear
        # basket on place (grip alone would miss this, since the arm sequence has already
        # released the gripper by the time it returns).
        basket_ok = False
        for attempt in range(1, MAX_PICK_ATTEMPTS + 1):
            if not c.is_running or self._clock.tick() >= deadline:
                log.warning("[Stage2] pick retry budget exhausted (run stopped or deadline hit) "
                            "before attempt %d/%d", attempt, MAX_PICK_ATTEMPTS)
                break
            try:
                grasp_xy = self._object_arm_xy(collect_xy)   # actual object position in the arm_base frame
                depth_xy = self._locate_object_by_depth()    # camera+depth refinement, if available
                if depth_xy is not None:
                    log.info("[Stage2] depth refinement grasp=%s (nav estimate was %s)",
                             tuple(round(v, 3) for v in depth_xy),
                             tuple(round(v, 3) for v in grasp_xy) if grasp_xy else None)
                    grasp_xy = depth_xy
                log.info("[Stage2] object pick sequence attempt %d/%d (object-position grasp=%s, type=%s)",
                         attempt, MAX_PICK_ATTEMPTS,
                         tuple(round(v, 3) for v in grasp_xy) if grasp_xy else None, self._target_type)
                held = run_pick_sequence(c, log, grasp_xy, object_type=self._target_type)
                c.send_cmd_vel(0.0, 0.0)
                # Grip-hold feedback (participant style). None (no feedback) -> conservatively count
                # the completed sequence as success.
                picked = True if held is None else bool(held)
            except Exception as e:
                log.warning("[Stage2] pick attempt %d failed: %s", attempt, e)
                picked, held = False, False
                # run_pick_sequence's own abort recovery didn't run (it raised instead of
                # returning) -- the arm pose is unknown, so stow before moving the chassis again.
                self._stow_arm()

            basket_reading = self._wait_basket_state(timeout=1.0)
            # Same no-feedback convention as the grip check above: None (no basket topic in
            # this environment) -> conservatively count as pass, don't burn retries on it.
            basket_ok = True if basket_reading is None else bool(basket_reading)
            log.info("[Stage2] pick attempt %d %s (grip held=%s, basket=%s)",
                     attempt, "ok" if picked and basket_ok else "failed", held, basket_reading)
            if picked and basket_ok:
                break
            if attempt < MAX_PICK_ATTEMPTS:
                log.info("[Stage2] retrying pick (attempt %d/%d)", attempt + 1, MAX_PICK_ATTEMPTS)
                self._align_to_object(collect_xy, deadline)
                self._approach_straight(collect_xy, deadline, target_fwd=0.40)
                for _ in range(5):
                    c.send_cmd_vel(0.0, 0.0)
                    c.sleep(0.1)
        picked = picked and basket_ok

        # If the arm is left raised (front_high) without an object, driving the delivery leg with
        # that top-heavy CoG can tip the robot over on turns. Empty-handed -> fold to HOME.
        if not picked:
            self._stow_arm()

        # -- leg 2: deliver to the owner zone --
        reached = False
        if pickup_xy is None:
            log.error("[Stage2] no owner_position in the mission -- skipping the delivery leg")
        else:
            log.info("[Stage2] leg2 -> deliver to owner zone %s", pickup_xy)
            reached = self._navigate_to(pickup_xy, deadline)

        # No recovery attempted here (no fresh perception of where it fell) -- just surface it,
        # since a silent loss would otherwise look identical to a normal successful delivery.
        # `is False` (not a plain falsy check): None means no basket topic in this environment,
        # which is "unknown", not "confirmed empty" -- only warn on a real reading.
        if picked and self._wait_basket_state(timeout=1.0) is False:
            log.warning("[Stage2] object lost from basket in transit (picked ok, basket empty on arrival)")

        c.stop()
        log.info("[Stage2] driving finished (delivery_reached=%s) -> task_complete", reached)
        c.task_complete()

    def _stow_arm(self):
        """Fold the arm to the HOME pose to lower the center of gravity -- tip-over prevention.

        The arm driver re-applies the last joint_command every tick, so one publish holds the pose
        for the whole leg. (Not called after a successful pick -- leg 2 keeps the carry pose.)
        """
        try:
            from arm_pick import HOME, GRIPPER_OPEN, _make_js
            self.client.send_arm_command(_make_js(HOME, GRIPPER_OPEN))
            log.info("[Stage2] arm stow -> HOME (lower CoG, tip-over prevention)")
        except Exception as e:
            log.warning("[Stage2] arm stow failed: %s", e)

    def _standoff_goal(self, dest, standoff):
        """A point ``standoff`` metres in front of ``dest`` along the robot->dest line.

        Approaching a standoff (instead of the object itself) keeps the chassis from driving
        onto/over the object.
        """
        pose = self.client.get_world_pose()
        if pose is None:
            return dest
        x, y, _ = pose
        dx, dy = dest[0] - x, dest[1] - y
        d = math.hypot(dx, dy)
        if d <= standoff:
            return (x, y)
        f = (d - standoff) / d
        return (x + dx * f, y + dy * f)

    def _object_arm_xy(self, obj_xy):
        """Object world XY -> arm_base frame (fwd, lat).

        Preferred path is the real world->arm_base_link tf: it needs no ARM_BASE_FWD
        assumption, so the result stays correct whatever angle/offset the arm is actually
        mounted at on the chassis. The gen3 URDF has no arm_base link at all (the full
        robot assembly is platform-only), so the mount offset is *only* knowable at
        runtime -- see manipulation/world_to_arm_base.py.

        When tf2 is unavailable (offline harness: no ROS, no live platform) we fall back
        to the previous pose-based computation with the ARM_BASE_FWD assumption. That
        fallback is an approximation, but it lets vla_grounding/validation/stage2_sim.py run
        the whole Stage 2 flow without Isaac Sim. Live runs always take the tf2 path.
        """
        if self._tf_buffer is not None:
            try:
                fwd, lat = transform_via_tf2(self._tf_buffer, (obj_xy[0], obj_xy[1], 0.0))
                return (fwd - GRASP_FWD_BACKOFF, lat)
            except Exception as e:
                log.warning("[Stage2] tf2 world->arm_base_link transform failed: %s "
                            "-- falling back to pose math", e)

        pose = self.client.get_world_pose()
        if pose is None:
            return None
        x, y, yaw = pose
        dx, dy = obj_xy[0] - x, obj_xy[1] - y
        cc, ss = math.cos(yaw), math.sin(yaw)
        fwd = dx * cc + dy * ss            # car_base forward (+X)
        lat = -dx * ss + dy * cc           # car_base left (+Y)
        return (fwd - ARM_BASE_FWD - GRASP_FWD_BACKOFF, lat)

    def _align_to_object(self, obj_xy, deadline, max_t=6.0, tol_rad=math.radians(8)):
        """Rotate in place to face ``obj_xy`` before the final straight approach.

        ``_approach_straight`` only has authority to correct a *small* lateral offset
        while creeping forward -- its exit test is "close enough in depth" (fwd), not
        "centered" (lat), on the assumption that the standoff arrival is already
        roughly aligned. That assumption is not guaranteed: _navigate_to's goal test is
        pure XY distance with no heading term, so a standoff reached from an angle (an
        A* replan mid-corner, an off-path recovery, tracking error) can land with fwd
        already inside target_fwd while lat is still large -- the straight-approach loop
        then exits on its very first tick with the object off to the side and no time
        left to correct it. Measured while trialling a pure-pursuit path follower whose
        corner-cutting produced exactly this arrival (fwd=0.388/lat=0.917, broke in
        0.2s, pick missed by 0.251m) -- that follower was reverted for a net time loss,
        but the arrival-angle assumption it exposed was already latent, not new, so
        this fix stays.

        Squaring up first -- while there is still room in front to turn without
        overshooting past the object -- is what actually fixes that, independent of
        whatever put the chassis at an angle. Pure rotation, so it is safe to run even
        very close in (unlike the forward creep, which cannot correct a large angle
        without also shrinking the remaining approach room).
        """
        c = self.client
        t_end = min(deadline, self._clock.elapsed + max_t)
        while c.is_running and self._clock.tick() < t_end:
            gxy = self._object_arm_xy(obj_xy)
            if gxy is None:
                break
            fwd, lat = gxy
            # Bearing to the object in the arm_base frame (0 = dead ahead). atan2 (not the
            # raw lat used for centering while moving) so a badly angled arrival -- fwd
            # small or even negative, lat large -- still turns the right way instead of
            # just riding a small-angle approximation past its range.
            heading_err = math.atan2(lat, fwd) if fwd > 1e-3 else (
                math.copysign(math.pi / 2, lat) if lat else 0.0)
            if abs(heading_err) <= tol_rad:
                break
            ang = max(-0.5, min(0.5, 1.5 * heading_err))
            c.send_cmd_vel(linear_x=0.0, angular_z=ang)
            c.sleep(STAGE2_NAV_DT)
        for _ in range(5):
            c.send_cmd_vel(0.0, 0.0)
            c.sleep(0.05)
        gxy = self._object_arm_xy(obj_xy)
        log.info("[Stage2] align done -- object arm coords %s",
                 tuple(round(v, 3) for v in gxy) if gxy else None)
    def _locate_object_by_depth(self):
        """Refine the pick point using the base camera's own depth (floor-height segmentation).

        _object_arm_xy() gets the chassis into the right neighborhood from the (approximate,
        per SDK docs) reveal hint_center; this looks at what the base camera actually sees
        right now to correct the last bit of error before the fixed waypoint sequence commits
        to a grasp point. Back-projects the whole depth image to arm_base_link via that
        camera's own TF (Base_Camera_Center -> arm_base_link -- confirmed connected, distinct
        from the broken world->arm_base_link link), takes the median height as the floor
        (robust: most of the frame is floor), and grasps at the centroid of the largest
        connected blob raised above it.

        No learned model, so no CCTV-vs-robot-camera domain mismatch. Caveat: this finds
        "something raised off the floor", not object identity -- if a distractor sits in the
        same frame it could pick the wrong blob (e.g. the larger one). Validated so far only
        with one object in view (manipulation_trainer, 2026-09-01); watch this in a real
        Stage2 run where distractors are possible.

        Returns (fwd, lat) in arm_base_link frame, or None if depth/camera_info/TF isn't
        available, they don't agree on resolution, or no clear blob is found -- callers keep
        their existing _object_arm_xy() estimate in that case, so this can only refine, never
        regress below today's baseline.
        """
        if self._tf_buffer is None:
            return None
        try:
            import rclpy
            from sensor_msgs.msg import CameraInfo
        except Exception:
            return None

        c = self.client
        depth_msg = c.get_robot_depth("base")
        if depth_msg is None:
            return None

        if self._depth_cam_info_sub is None:
            def _on_depth_info(msg):
                self._depth_cam_info = msg
            self._depth_cam_info_sub = c.subscribe(
                c.topics.robot("base_camera/depth/info"), CameraInfo, _on_depth_info)
        t_end = self._clock.elapsed + 1.0
        while self._depth_cam_info is None and self._clock.tick() < t_end:
            c.sleep(0.05)
        info = self._depth_cam_info
        if info is None:
            return None
        if (info.width, info.height) != (depth_msg.width, depth_msg.height):
            log.warning("[Stage2] depth camera_info size (%dx%d) != depth image size (%dx%d) "
                        "-- skipping depth refinement", info.width, info.height,
                        depth_msg.width, depth_msg.height)
            return None
        K = np.array(info.k, dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        try:
            tf = self._tf_buffer.lookup_transform(
                "arm_base_link", "Base_Camera_Center", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:
            log.warning("[Stage2] depth refinement: tf lookup Base_Camera_Center -> "
                        "arm_base_link failed: %s", e)
            return None
        tr, rot = tf.transform.translation, tf.transform.rotation
        qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
        ])
        C = np.array([tr.x, tr.y, tr.z])

        h, w = depth_msg.height, depth_msg.width
        depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(h, w)
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
        if not np.any(valid):
            return None

        us_grid, vs_grid = np.meshgrid(np.arange(w), np.arange(h))
        x_cam = (us_grid - cx) / fx * depth
        y_cam = (vs_grid - cy) / fy * depth
        pts_cam = np.stack([x_cam, y_cam, depth], axis=-1)
        pts_arm = pts_cam @ R.T + C
        fwd, lat, up = pts_arm[..., 0], pts_arm[..., 1], pts_arm[..., 2]

        floor_z = float(np.median(up[valid]))
        candidate = valid & (up > floor_z + 0.03) & (fwd > 0.0) & (fwd < 1.2)
        n_cand = int(np.sum(candidate))
        if n_cand < 20:
            return None

        try:
            from scipy import ndimage
            labeled, n_blobs = ndimage.label(candidate)
            sizes = ndimage.sum(candidate, labeled, index=range(1, n_blobs + 1))
            blob_mask = labeled == (int(np.argmax(sizes)) + 1)
        except ImportError:
            blob_mask = candidate

        return (float(np.mean(fwd[blob_mask])), float(np.mean(lat[blob_mask])))

    def _approach_straight(self, obj_xy, deadline, target_fwd=0.40, max_t=20.0):
        """Drive straight at the object until the arm_base forward distance reaches target_fwd.

        The planner cannot enter a tight spot fully (it halts ~1 m out), so fill the remaining
        empty space by driving straight into the object, with a small lateral steering term to
        center it (lat -> 0). Call _align_to_object first -- this loop assumes it is already
        roughly facing the object and cannot recover a large initial heading error (see
        _align_to_object for why). SIM-time paced.
        """
        c = self.client
        t_end = min(deadline, self._clock.elapsed + max_t)
        # True while backing straight up because the chassis is past/over the object --
        # see below. Hysteresis: engage under target_fwd-0.03, release at target_fwd+0.10
        # so the normal forward creep (which also centers lat) finishes the last stretch.
        reversing = False
        while c.is_running and self._clock.tick() < t_end:
            gxy = self._object_arm_xy(obj_xy)
            if gxy is None:
                break
            fwd, lat = gxy
            if reversing:
                if fwd >= target_fwd + 0.10:
                    reversing = False
                else:
                    c.send_cmd_vel(linear_x=-0.10, angular_z=0.0)
                    c.sleep(STAGE2_NAV_DT)
                    continue
            if fwd <= target_fwd + 0.02:
                if fwd < target_fwd - 0.03:
                    # The chassis is past (or on top of) the object -- measured live on
                    # rt01: the leg arrived 0.22 m from the object (follower done fired at
                    # its own 0.7 m tolerance, past the 0.58 m standoff, plus residual
                    # 0.8 m/s coasting), so every pick ran at fwd 0.02~0.16 against a
                    # grasp tuned for 0.47 and failed 3/3. Backing straight up restores
                    # the tuned grasp distance and also brings the object back into the
                    # base camera's view for the depth refinement.
                    log.info("[Stage2] approach: object at fwd=%.3f < target %.2f -- "
                             "chassis past/over it, backing up", fwd, target_fwd)
                    reversing = True
                    continue
                break
            ang = max(-0.4, min(0.4, 1.5 * lat))         # center the object (lat -> 0)
            lin = 0.12 if abs(lat) < 0.15 else 0.06      # badly off -> creep slower while turning
            # Same chassis validity cap the path controllers use: without it this leg emits
            # 2v <= w*T whenever |lat| >= 0.211, and the chassis silently stops -- it burned
            # the whole 20 s budget moving 6 mm, leaving the object out of arm reach.
            ang = cap_angular(lin, ang)
            c.send_cmd_vel(linear_x=lin, angular_z=ang)
            c.sleep(STAGE2_NAV_DT)
        for _ in range(5):
            c.send_cmd_vel(0.0, 0.0)
            c.sleep(0.05)
        gxy = self._object_arm_xy(obj_xy)
        log.info("[Stage2] straight approach done -- object arm coords %s",
                 tuple(round(v, 3) for v in gxy) if gxy else None)

    def _wait_basket_state(self, timeout: float = 1.0):
        """Poll briefly for a settled is_basket_occupied() reading.

        Right after the first call in a run, the subscription is fresh and a read can return
        None for a tick or two before the first message arrives -- poll for a real reading.
        Returns True/False/None (None = no message arrived at all in this environment, distinct
        from a confirmed False) so callers apply their own no-feedback convention instead of this
        helper silently coercing "unknown" into "empty" for them.
        """
        c = self.client
        start = self._clock.elapsed
        val = c.is_basket_occupied()
        while val is None and self._clock.tick() - start < timeout:
            c.sleep(0.05)
            val = c.is_basket_occupied()
        return val

    def _wait_for_collect_goal(self, timeout: float = 5.0):
        """Poll briefly until the reveal (msg 411) approximate position arrives, then return the pickup goal (x, y).

        When the reveal arrives, _on_stage2_reveal fills self._collect_xy. If not received
        within the timeout, fall back to the Stage 2 grounding answer's target (_stage2_target_xy).
        """
        c = self.client
        start = self._clock.elapsed
        while self._clock.tick() - start < timeout:
            if self._collect_xy is not None:
                return self._collect_xy
            c.sleep(0.1)
        log.warning("[Stage2] reveal not received (%.0fs) -- falling back to grounding target %s",
                    timeout, self._stage2_target_xy)
        return self._stage2_target_xy

    def _lidar_mount_offset(self):
        """Base_LiDAR -> chassis (dx, dy) from tf, or (0, 0) when tf cannot answer.

        Static transform, so it is read once per run rather than per scan. The fallback
        assumes the sensor sits above the chassis origin; that is a few centimetres of
        error on a 0.25 m grid, which is why a failure here only degrades accuracy instead
        of disabling the layer.
        """
        if self._tf_buffer is None:
            return (0.0, 0.0)
        try:
            import rclpy
            tf = self._tf_buffer.lookup_transform(
                "base_link", "Base_LiDAR", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            t = tf.transform.translation
            log.info("[Stage2] lidar mount offset from tf: (%.3f, %.3f)", t.x, t.y)
            return (float(t.x), float(t.y))
        except Exception as e:
            log.info("[Stage2] lidar mount offset unavailable (%s) -- assuming (0, 0)", e)
            return (0.0, 0.0)

    def _update_lidar_layer(self, pose):
        """Fold one lidar sweep into the obstacle layer. Returns newly blocked cell count.

        A missing topic (get_lidar() -> None, not an exception -- a fresh subscription
        legitimately reads as "nothing yet") costs nothing here; lidar_xyz() already
        returns None for that and LidarObstacleLayer.update() no-ops on it. Only a raw
        exception (parse failure, comms fault) counts as a "fail" below, and only
        LIDAR_MAX_CONSEC_FAILS of those *in a row* disable the layer -- disabling on the
        first one, ever, meant a single transient hiccup (e.g. sensor_msgs_py missing so
        the byte-parse fallback runs and trips once on a malformed early packet) killed
        avoidance for the entire run with no way back, indistinguishable in the logs from
        "no lidar at all". Any success resets the streak.
        """
        if self._lidar_layer is None:
            return 0
        try:
            n = self._lidar_layer.update(self.client.get_lidar(), pose,
                                         self._lidar_mount_xy)
            self._lidar_fail_streak = 0
            return n
        except Exception as e:
            self._lidar_fail_streak += 1
            log.warning("[Stage2] lidar update failed (%d/%d consecutive): %s: %s",
                        self._lidar_fail_streak, LIDAR_MAX_CONSEC_FAILS,
                        type(e).__name__, e)
            if self._lidar_fail_streak >= LIDAR_MAX_CONSEC_FAILS:
                log.warning("[Stage2] lidar failed %d times in a row -- obstacle layer "
                            "off for the rest of this run, falling back to stall memory",
                            self._lidar_fail_streak)
                self._lidar_layer = None
            return 0

    def _plan_obstacles(self, goal, exclude_m):
        """Everything the planner should route around on this leg.

        Two sources, excluded near the goal by two DIFFERENT radii -- they are not the same
        kind of evidence:

        - Lidar discs are a sensor GUESS that can legitimately be the target object itself
          (or its landmark) -- excluding a wide radius (exclude_m, sized by the reveal's own
          hint_radius) around the goal is what keeps the leg plannable at all.
        - Stall-memory (self._obstacles) is not a guess -- it is "the chassis physically hit
          something at this exact point", recorded by _remember_blockage(), which already
          refuses to mark anything within BLOCKAGE_RADIUS_M of the goal so a normal
          approach-speed slowdown near the target is never misread as a wall. Applying the
          much larger exclude_m here too silently discards that evidence whenever the real
          obstruction happens to sit within the reveal's hint-uncertainty radius of the goal
          -- which, in a scenario like "object next to a parked car", is exactly where a
          real obstruction is likely to be. Measured live (rt01): a blockage grew to the
          BLOCKAGE_MAX_RADIUS_M cap (2.0m) over 3 stalls while the chassis sat pinned at the
          exact same spot the whole time -- A* kept returning the same 1-waypoint straight
          path because this exclusion was silently stripping that blockage out before the
          planner ever saw it (exclude_m for this leg is also 2.0m, from hint_radius, so
          every mark landed inside it). The lidar genuinely saw nothing there (raw sweep
          probe: no return within 3m of dead ahead) -- the mechanism the module's own
          docstring names for exactly this case (kerbs/low steps/wheel traps the lidar
          height-band cannot see) was firing correctly and then being discarded downstream.
        """
        lidar_discs = list(self._lidar_layer.discs()) if self._lidar_layer is not None else []
        lidar_discs = obstacles_excluding_goal(lidar_discs, goal, exclude_m=exclude_m)
        memory = obstacles_excluding_goal(self._obstacles, goal, exclude_m=BLOCKAGE_RADIUS_M)
        return lidar_discs + memory

    def _remember_blockage(self, pose, goal):
        """Mark the spot the chassis just stalled against, so the replan routes around it.

        This is the only obstacle evidence available without a sensor, and it costs nothing
        we are not already reading (the pose polled every tick). It is a recovery, not a
        substitute for sensing: it learns only after contact, one blockage at a time, and
        each lesson costs STAGE2_STUCK_TIME_S. The lidar local map supersedes it for
        anything visible ahead; this stays for what that misses (kerbs, low steps, wheel
        traps) and for the competition scenario, where no obstacle list exists at all.

        The disc is placed just *ahead* of the robot -- its near edge at the front of the
        footprint -- rather than on the robot, so the planner never has to snap the start
        cell out of blocked space. Stalling again inside a known disc grows that disc
        instead of stacking another: a first mark too narrow to force a detour would
        otherwise be re-hit forever.

        Returns True when the obstacle list changed (the caller then refreshes its
        per-leg view of it).
        """
        x, y, yaw = pose
        d = ROBOT_RADIUS_M + BLOCKAGE_RADIUS_M
        bx, by = x + d * math.cos(yaw), y + d * math.sin(yaw)

        # Never wall off the goal: closing on the target legitimately slows the robot to a
        # crawl, and a disc over the goal makes the rest of the leg unplannable.
        if math.hypot(bx - goal[0], by - goal[1]) <= BLOCKAGE_RADIUS_M:
            log.info("[Stage2] stalled next to the goal -- not marking it blocked")
            return False

        for i, (ox, oy, orad) in enumerate(self._obstacles):
            if math.hypot(bx - ox, by - oy) <= orad:
                if orad >= BLOCKAGE_MAX_RADIUS_M:
                    log.warning("[Stage2] stalled inside a maxed-out blockage at "
                                "(%.2f, %.2f) -- cannot grow it further", ox, oy)
                    return False
                grown = min(BLOCKAGE_MAX_RADIUS_M, orad + BLOCKAGE_GROWTH_M)
                self._obstacles[i] = (ox, oy, grown)
                log.info("[Stage2] stalled inside a known blockage -> grew (%.2f, %.2f) "
                         "to r=%.2fm", ox, oy, grown)
                return True

        self._obstacles.append((bx, by, BLOCKAGE_RADIUS_M))
        log.info("[Stage2] blockage remembered at (%.2f, %.2f) r=%.2fm (%d known)",
                 bx, by, BLOCKAGE_RADIUS_M, len(self._obstacles))
        return True

    def _navigate_to(self, goal, deadline, tol=GOAL_TOLERANCE,
                     inflation_m=STAGE2_INFLATION_M, exclude_m=0.6, cruise=False,
                     snap_m=None):
        """Drive to a single goal (x, y) by following occupancy A*. True if reached, False on timeout.

        Args:
            tol: reached-test XY distance (m). Tighter for a tight pick approach.
            inflation_m: erode the free area by this much (lower = enter tighter spots).
            exclude_m: drop scenario obstacles within this radius of the goal (keep it reachable).
            cruise: True -> pass-through waypoint, no near-goal deceleration.
            snap_m: how far the planner may move start/goal onto free space to make a plan.
                Defaults to the leg's own reached-test distance, since a goal snapped
                further than that is a goal we cannot finish the leg from. (It used to be a
                fixed 8 m, which can silently relocate the goal into a different area.)
        `deadline` is elapsed seconds on self._clock (see ElapsedClock) since _drive() started,
        shared across all Stage 2 legs -- not an absolute now_s() value.
        """
        c = self.client
        planner = OccupancyGridPlanner()
        follower = None
        fallback_ctrl = GoalSeekController()
        # The controller's class-level reached test (0.7 m) can be looser than this leg's
        # tol (e.g. the tight pick approach at 0.45) -- shadow it per-instance so the
        # fallback actually finishes the leg inside tol instead of declaring victory 0.7 m
        # out (see the follower-done handling below for the measured failure this caused).
        fallback_ctrl.GOAL_TOLERANCE = min(fallback_ctrl.GOAL_TOLERANCE, tol)

        # Obstacles for this leg -- lidar layer + stall memory, minus anything on the goal.
        leg_obstacles = self._plan_obstacles(goal, exclude_m)
        snap = max(1.0, float(tol)) if snap_m is None else float(snap_m)
        tick = 0

        # Sliding window of (t, x, y) for the stall test -- see STAGE2_STUCK_TIME_S above.
        pos_history = deque()
        last_log_t = 0.0
        reached_goal = False
        braking = False   # edge-tracked so the reactive brake logs once per onset, not every tick
        brake_held_since = None   # elapsed-clock value; see BRAKE_STALL_S
        closing_logged = False    # edge-tracked so the precise-close log fires once, not every tick
        force_replan = False      # set by the brake-stall escalation below, consumed next tick

        while c.is_running and self._clock.tick() < deadline:
            pose = c.get_world_pose()
            if pose is None:
                c.sleep(STAGE2_NAV_DT)
                continue
            x, y, yaw = pose
            tick += 1
            # Sense before hitting anything. The stall path below only reacts after contact.
            new_cells = 0
            if tick % LIDAR_UPDATE_EVERY_TICKS == 0:
                new_cells = self._update_lidar_layer(pose)

            dist = math.hypot(goal[0] - x, goal[1] - y)
            if dist < tol:
                reached_goal = True
                log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                break

            # Refresh the occupancy grid (from the occupancy map)
            occ = c.get_occupancy_map()
            if occ is not None:
                planner.update_from_msg(occ)
                # NOTE: the lidar layer deliberately keeps its OWN (coarser) cell size and is
                # no longer synced to the map's. This used to call set_resolution(occ.info.
                # resolution) so its discs would tile the planner grid exactly, but that is
                # cosmetic -- _block_obstacles paints world-coordinate circles, which need no
                # alignment with the layer's own grid -- and on the live map (0.05 m/cell) it
                # broke the layer three ways at once:
                #   1. MAX_BLOCKED_CELLS is a cell *count*, so the same 8000 cells fell from a
                #      500 m2 budget at 0.25 m to 20 m2 at 0.05 m. Measured live: saturated in
                #      ~8 s, after which update() drops every new cell -- the obstacle ahead
                #      never enters the layer, and new_blocked stays 0, which also permanently
                #      kills the `if new_cells` replan trigger in this loop.
                #   2. MIN_POINTS_PER_CELL=2 stops being reachable for most cells: at 5 cm the
                #      returns spread thin, so real obstacle surface goes unmarked. Measured on
                #      identical live sweeps: 0.05 m marked 777 cells / 1.9 m2, while 0.25 m
                #      marked 89 cells / 5.6 m2 -- fewer cells covering ~3x more real surface.
                #   3. discs() emits radius == cell size, so 0.05 m gave the planner a sparse
                #      dot pattern A* can thread between, instead of overlapping 0.25 m discs
                #      that form a continuous barrier around a car.
                if not self._occ_logged:
                    self._occ_logged = True
                    log.info("[Stage2] occupancy received: %dx%d res=%.3f origin=(%.1f, %.1f)",
                             occ.info.width, occ.info.height, occ.info.resolution,
                             occ.info.origin.position.x, occ.info.origin.position.y)

            # Replan triggers, in the order we want them to fire: a new obstacle across the
            # remaining path (before contact), then a stall (after contact, for whatever the
            # lidar missed). Replanning on a timer instead would rewrite the path every cycle
            # and make the follower jitter, so it is event-driven.
            need_replan = follower is None or force_replan
            force_replan = False
            if new_cells and follower is not None and self._lidar_layer is not None:
                remaining = follower.path[follower.idx:]
                if self._lidar_layer.blocks_path(remaining, start_xy=(x, y)):
                    log.info("[Stage2] lidar: %d new cell(s) across the remaining path "
                             "-> replanning (%d blocked, %.1f m2)", new_cells,
                             self._lidar_layer.stats()["blocked_cells"],
                             self._lidar_layer.stats()["blocked_area_m2"])
                    leg_obstacles = self._plan_obstacles(goal, exclude_m)
                    need_replan = True
            # Trim the window to the trailing STAGE2_STUCK_TIME_S seconds -- keep popping
            # while the *second* entry is still old enough, so pos_history[0] ends up as the
            # oldest sample that is >= the window length old (or the very first sample this
            # leg has seen, if the leg hasn't run that long yet).
            now_t = c.now_s()
            pos_history.append((now_t, x, y))
            while len(pos_history) >= 2 and now_t - pos_history[1][0] >= STAGE2_STUCK_TIME_S:
                pos_history.popleft()
            if now_t - pos_history[0][0] >= STAGE2_STUCK_TIME_S:
                moved = math.hypot(x - pos_history[0][1], y - pos_history[0][2])
                if moved <= STAGE2_STUCK_MOVE_M:
                    log.info("[Stage2] stall detected (%.3fm over %.1fs) -> replanning",
                             moved, now_t - pos_history[0][0])
                    # Replanning on an unchanged map returns the same path, so record what
                    # we just stopped against before asking for a new one.
                    if self._remember_blockage((x, y, yaw), goal):
                        leg_obstacles = self._plan_obstacles(goal, exclude_m)
                    need_replan = True
                    # Start a fresh window from here -- otherwise the very next tick sees
                    # the same near-zero displacement over the same stale window and fires
                    # again immediately, on top of whatever replan just happened.
                    pos_history.clear()
                    pos_history.append((now_t, x, y))

            if need_replan:
                if not planner.ready:
                    if not self._occ_logged:
                        log.warning("[Stage2] occupancy not received -> straight-line fallback")
                else:
                    # Increase the snap radius so start/goal off the road (free) are also corrected to nearby free.
                    # Inject scenario obstructions (vehicles/bicycles/people/objects) as obstacles to route around.
                    path = planner.plan((x, y), goal, inflation_m=inflation_m,
                                        snap_m=snap, extra_obstacles=leg_obstacles)
                    if path:
                        follower = PathFollower(path, passthrough=cruise)
                        log.info("[Stage2] occupancy A* path %d waypoints", len(path))
                    else:
                        sr, sc = planner._w2c(x, y)
                        gr, gc = planner._w2c(*goal)
                        sv = int(planner._grid[sr, sc]) if planner._in_bounds(sr, sc) else 99
                        gv = int(planner._grid[gr, gc]) if planner._in_bounds(gr, gc) else 99
                        log.warning("[Stage2] A* failed -> straight-line fallback "
                                    "(start cell=(%d,%d) val=%d, goal cell=(%d,%d) val=%d) "
                                    "[0=free 100=occ -1=unknown]", sr, sc, sv, gr, gc, gv)

            # Compute commands: follow the path if there is one, otherwise straight-line fallback
            wp_str = ""
            if follower is not None:
                linear, angular, done, info = follower.compute(pose)
                wp_str = " wp %d/%d=(%.1f, %.1f)" % (
                    info["idx"], info["total"], info["waypoint"][0], info["waypoint"][1])
                if done and dist >= tol:
                    # The follower's internal done test is its controller's own
                    # GOAL_TOLERANCE (0.7 m on the final waypoint) -- looser than this
                    # leg's tol, so "done" can mean anywhere within 0.7 m of the goal,
                    # including PAST it. Measured live on rt01's pick approach
                    # (tol=0.45, standoff only 0.58 m from the object): done fired
                    # 0.68 m past-side of the standoff, 0.22 m from the object, and the
                    # chassis ended up on top of the pick target. Override this tick's
                    # command with the fallback controller's precise close instead --
                    # deliberately NOT `follower = None` here: that would make
                    # `need_replan = follower is None` true on every remaining tick of
                    # this leg (re-running A* every 100 ms for no reason, confirmed
                    # offline: the very next tick re-triggers this same branch, over and
                    # over, until dist < tol). Overriding linear/angular in place needs
                    # no replan at all.
                    if not closing_logged:
                        log.info("[Stage2] follower done at %.2fm > tol %.2fm -- "
                                 "closing precisely with the fallback controller", dist, tol)
                        closing_logged = True
                    linear, angular, reached = fallback_ctrl.compute(pose, goal, cruise=cruise)
                    if reached:
                        reached_goal = True
                        log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                        break
                elif done:
                    reached_goal = True
                    log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                    break
            else:
                linear, angular, reached = fallback_ctrl.compute(pose, goal, cruise=cruise)
                if reached:
                    reached_goal = True
                    log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                    break

            # Reactive brake -- see REACT_STOP_DIST_M above. Runs every tick regardless of
            # whether this tick refreshed the lidar layer, so it acts on the freshest data
            # available without waiting for LIDAR_UPDATE_EVERY_TICKS. Only clamps linear
            # (still free to turn) -- steering out of the way is the replan's job, this only
            # has to prevent contact before that replan lands.
            danger = self._lidar_layer is not None and self._lidar_layer.danger_ahead(
                pose, REACT_STOP_DIST_M, REACT_STOP_CONE_DEG,
                goal=goal, goal_exclude_m=exclude_m)
            blocked_by_brake = danger and linear > 0.0
            if blocked_by_brake:
                if not braking:
                    log.info("[Stage2] lidar: obstacle within %.2fm ahead -> braking",
                             REACT_STOP_DIST_M)
                linear = 0.0
                # BRAKE_STALL_S escalation -- see the constant's comment. The follower keeps
                # wanting to move (that is what blocked_by_brake means) while heading keeps
                # refining toward the waypoint (angular is untouched), which looks exactly
                # like a slow-converging turn but is actually this brake holding the whole
                # time. Don't wait for the generic 8s stall window to notice the same freeze
                # indirectly -- treat a sustained hold as an immediate stall.
                if brake_held_since is None:
                    brake_held_since = self._clock.elapsed
                elif self._clock.elapsed - brake_held_since >= BRAKE_STALL_S:
                    log.info("[Stage2] brake held %.1fs -> treating as stall, replanning",
                             self._clock.elapsed - brake_held_since)
                    if self._remember_blockage((x, y, yaw), goal):
                        leg_obstacles = self._plan_obstacles(goal, exclude_m)
                    force_replan = True
                    brake_held_since = None
                    # Same reason as the generic stall path: start the window fresh so it
                    # does not also fire on this same freeze a few ticks later. pos_history
                    # is keyed on c.now_s() (see below), NOT self._clock.elapsed -- these are
                    # two different clocks and must not be mixed.
                    pos_history.clear()
                    pos_history.append((c.now_s(), x, y))
            else:
                brake_held_since = None
            braking = danger

            c.send_cmd_vel(linear_x=linear, angular_z=angular)

            # Progress log (throttled to ~1s SIM time) -- for tracking pose/dist/v/w
            now = c.now_s()
            if now - last_log_t >= 1.0:
                log.info("[STAGE2] pose=(%.2f, %.2f, yaw=%.0fdeg) dist=%.2fm%s v=%.2f w=%+.2f",
                         x, y, math.degrees(yaw), dist, wp_str, linear, angular)
                last_log_t = now

            c.sleep(STAGE2_NAV_DT)

        return reached_goal

    # -- notifications --

    def _on_score(self, score):
        if score.is_final:
            log.info("[FINAL] %s", score.scores)
        else:
            log.info("[Score] round %s: total=%s", score.round, score.total)
        self._append_score_log(score)

    def _append_score_log(self, score):
        """Append one JSONL line per score event (round, its voice_command, full scores dict).

        Path from MARC_SCORE_LOG (default ./results/scores.jsonl). Logging must never
        break the run loop, so failures are logged and swallowed.
        """
        try:
            entry = {
                "stage": score.stage,
                "round": score.round,
                "voice_command": self._round_commands.get(score.round),
                "scores": score.scores,
            }
            os.makedirs(os.path.dirname(self._score_log_path) or ".", exist_ok=True)
            with open(self._score_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("[Score] JSONL logging failed: %s", e)

    def _append_reveal_log(self, reveal):
        """Append one JSONL line per Stage 2 reveal (msg 411).

        Path from MARC_REVEAL_LOG (default ./results/stage2_reveals.jsonl). This is the only
        record of what the platform actually hands the driving legs, and nothing else in the
        pipeline captures it -- the offline harness has been assuming a perfect hint, so how
        far hint_center really sits from the object (and how large hint_radius is) is still
        unmeasured. It decides whether close-range re-detection is a precision nicety or a
        hard requirement, since the arm only reaches ~0.75 m. Logging must never break the
        run loop, so failures are logged and swallowed.
        """
        try:
            entry = {
                "sim_time": self.client.now_s(),
                "task_description": self._stage2_task,
                "grounding_score": reveal.grounding_score,
                "target_type": reveal.target_type,
                "hint_center": list(reveal.hint_center or []),
                "hint_radius": reveal.hint_radius,
                "submitted_target_coord": self._stage2_submitted_target,
            }
            os.makedirs(os.path.dirname(self._reveal_log_path) or ".", exist_ok=True)
            with open(self._reveal_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.info("[Stage2] reveal logged -> %s", self._reveal_log_path)
        except Exception as e:
            log.warning("[Stage2] reveal JSONL logging failed: %s", e)

    def _on_state_change(self, old, new):
        log.info("[STATE] %s -> %s", old, new)

    def _on_time_expired(self, which):
        log.info("[TIME] expired: %s", which)

    # -- run --

    def _setup_tf(self):
        """Create the tf listener -- must run after client.connect() (rclpy initialised).

        The listener must be kept alive (held as an attribute) or it stops receiving tf
        broadcasts. Offline harnesses (vla_grounding/validation/stage2_sim.py) never call
        run(), so there self._tf_buffer stays None and _object_arm_xy uses its pose-math
        fallback -- identical to the previous behaviour.
        """
        try:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self.client.node)
            log.info("[Stage2] tf2 listener up (world->arm_base_link / camera transforms live)")
        except Exception as e:
            log.warning("[Stage2] tf2 unavailable (%s) -- falling back to "
                        "ARM_BASE_FWD pose math in _object_arm_xy", e)
            self._tf_buffer = self._tf_listener = None

    def run(self):
        if not self.client.connect():
            raise SystemExit("Registration failed -- check the runtime and the token.")
        self._setup_tf()
        self.client.run()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    DemoParticipant().run()


if __name__ == "__main__":
    main()
