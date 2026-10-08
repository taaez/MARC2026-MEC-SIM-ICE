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

# ============================================================================
# [파일 출처 안내 -- starter kit 대비 무엇이 바뀌었나]
# starter kit 원본 demo/participant_app.py는 443줄짜리 아주 단순한 데모였다
# (Stage1: MockVLAGrounding 호출 -> submit, Stage2: standoff 접근 -> 직진 접근 ->
#  1회 픽 시도 -> 배달 -> task_complete, 회피는 "8초 정체 시 A* 재계획" 단 하나뿐).
# 이 파일은 그 골격(핸들러 등록 순서, _drive()의 큰 흐름, _standoff_goal 등)은
# 유지하면서 아래 4가지 축으로 확장한 것이다. 아래에서부터는 각 함수/클래스 정의
# 바로 위에 다음 3가지 중 하나로 표시해둔다.
#   [스타터킷 원본]        -- 로직 변경 없음 (또는 극히 사소한 변경만)
#   [스타터킷 골격 + 확장]  -- 원본 흐름은 유지하되 상당 부분을 팀이 추가/수정
#   [팀 신규 추가]          -- starter kit에는 아예 없던 것
#
# 확장 축 4가지:
#   1) 센서 기반 회피 (우리 팀 담당, 이번 대화의 핵심) -- lidar_obstacles /
#      waypoint_avoid / depth_ground_veto 를 실제 tf/토픽에 연결하고, _navigate_to()
#      안에 반응형 브레이크·정체 기억·후진 탈출까지 엮어 넣음.
#   2) ElapsedClock -- now_s() 의 wall-clock/sim-time 전환 버그 수정.
#   3) VLA grounding 교체 (MockVLAGrounding -> VLAGroundingAgent) + CCTV 워밍업 --
#      Stage1 담당 팀원 쪽 작업.
#   4) 픽업 재시도·뎁스 정밀 파지(_locate_object_by_depth)·바구니 확인 -- 매니퓰
#      레이션 담당 팀원 쪽 작업.
# 1)/2)가 이번에 설명한 "주행/회피" 범위이고, 3)/4)는 같은 파일 안에 있지만 다른
# 담당 영역이라 간단히만 표시한다.
# ============================================================================

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
    cap_angular, obstacles_excluding_goal, nearest_only_exclude, STEERING_TRACK,
)
# Closed-loop placo pick from the manipulation kit (arm_pick.py): grasp_yaw wrist-roll +
# DLS servo, targeting the object position. Requires placo (installed in the demo image).
# The fixed-keyframe mock in mock_agent_manipulation.py remains as a dependency-free
# fallback reference.
from arm_pick import run_pick_sequence
from manipulation import transform_via_tf2
# Lidar obstacle layer -- the occupancy map holds static road only, so parked vehicles and
# the like have to be sensed. See mec_sim_ice/demo_agents/lidar_obstacles.py.
from lidar_obstacles import LidarObstacleLayer, lidar_xyz
# [팀 추가] 연석·단차 레이어 (MPPI 모드의 NavField 에만 반영). 모듈 docstring 참고.
from curb_layer import CurbLayer, lidar_world
from lidar_accumulator import LidarSliceAccumulator, imu_roll_pitch   # [팀 추가] 조각 누적
# [팀 추가] 지역 내비게이션 함수 + MPPI (MARC_LOCAL_PLANNER=mppi). 모듈 docstring 참고.
from nav_field import NavField
from occupancy_layer import OccupancyLayer
from mppi import MPPI, V_MAX as V_MAX_MPPI, W_MAX as MPPI_W_MAX
# 기본값 mppi = 최종 제출 설정 (2026-10-08 finals). "legacy" 는 ablation 비교용 -- 라이브 실험
# 스크립트(validation/live/run_one.sh)는 이 변수를 항상 명시해서 넘긴다.
LOCAL_PLANNER = os.environ.get("MARC_LOCAL_PLANNER", "mppi").strip().lower()
NAVFIELD_REBUILD_S = 1.0       # sim s between cost-to-go rebuilds triggered by new lidar cells
MPPI_GOAL_CLEAR_M = 0.6        # lidar/stall discs dropped this close to the goal (pick target)
MPPI_ACCEPT_NEAR_M = 1.0       # tight legs: a stall this close to the goal counts as arrival
# Tilt guard: live rt01 2026-10-05 the chassis climbed onto off-road terrain and rolled over
# (roll 55 deg, pitch -54 deg). Normal driving stays within ~2 deg; kerb wedges reach ~10 deg.
TILT_GUARD_DEG = 12.0
# [팀 수정 2026-10-07] tilt guard: stop and settle first, back off slowly only if still tilted,
# and do not re-fire at the same spot unless the tilt got worse. Before: an immediate reverse on
# every tick the IMU read > 12 deg -- a robot resting tilted on a kerb re-fired every 8 s
# (rt01_ab_mppi 98 times, survey run 212 times, ending on its side at roll 101 deg).
TILT_SETTLE_S = 0.6            # stop, then re-read the tilt after this
TILT_REVERSE_V = 0.15          # slower than _reverse_escape's default 0.2
TILT_REPEAT_M = 0.5            # same spot = within this of the last trigger
TILT_REPEAT_DEG = 3.0          # ... re-fire there only if the tilt grew by this much
MPPI_OBJECT_RADIUS_M = 0.2     # pick target footprint kept in the navfield on the approach leg
# Pinned = driving forward but not moving. Live rt01_mppi_r4: v=1.1 commanded for 80 s against
# something below the lidar band while the yaw twisted -10 -> -45 deg, which the 8 s stall
# rule counted as "turning in place" progress. This catches it in PIN_WINDOW_S.
MPPI_PIN_WINDOW_S = 1.5
MPPI_PIN_CMD_V = 0.3           # mean commanded forward speed over the window
MPPI_PIN_MOVE_M = 0.08         # displacement below this over the window = pinned
MPPI_SPIN_CMD_V = 0.15         # stall-rule turning exemption only when commanding a spin
MPPI_ALL_HIT_FRAC = 0.98       # this share of rollouts colliding -> stop instead of MPPI output
# Goal unreachable in the navfield (live rt01_ab_mppi 2026-10-05: 2058 accumulated lidar discs,
# ghost walls included, cut the road; the old fallback then opened ALL off-road and the robot
# hung on a kerb). Escalation, each step only for THIS build -- the lidar layer itself keeps
# every cell (the reactive brake still sees them):
#   1) tight: chassis radius MPPI_TIGHT_INSCRIBED_M, all obstacles kept. Squeezing closer to
#      obstacles is less risky than ignoring some. Live rt04b_ab_mppi 2026-10-06: the pick
#      approach (radius 0.22) got in between kick scooters / trash can, then leg 2 (0.30)
#      found the gap it came through closed -> V(robot)=inf in every step, 15 min stuck.
#   2) fresh+near: lidar cells not re-observed for MPPI_DISC_STALE_S are left out (a real
#      obstacle in view is re-hit every sweep; a ghost from a past rotation is not), except
#      within MPPI_DISC_NEAR_M of the robot (too close for the lidar to re-hit)
#   3) near-only: only lidar cells within MPPI_DISC_NEAR_M of the robot
#   4) last resort: all off-road passable (the old fallback)
# Stall-memory discs (physical contact) are kept in every step. Steps 2-4 also use the tight
# radius.
# Stage hold: once a build needed step k, the next builds start at step k for
# MPPI_STAGE_HOLD_S. Live rt04_ab_mppi 2026-10-06: next to the target (kick scooters around
# it) builds alternated unreachable / near-only-reachable, V jumped 8 -> 17 -> 10 and the
# robot spun and backed up for minutes 4-6 m from the object.
MPPI_DISC_STALE_S = 10.0
MPPI_DISC_NEAR_M = 2.0
MPPI_TIGHT_INSCRIBED_M = 0.22
MPPI_STAGE_HOLD_S = 15.0
# [팀 추가 2026-10-07] Disc-set stabilization. Live rt04/rt04b (rounds 1, 4, 5): next to the
# target the lidar disc count changed every build (rt04_r5_mppi: 1813 -> 859 -> 706 -> 193 ->
# 153 ...), V jumped and MPPI flipped between turning left/right and forward/back for minutes.
# A lidar disc now enters the field only after MPPI_DISC_ADD_BUILDS consecutive builds that
# saw it and leaves only after MPPI_DISC_DROP_BUILDS consecutive builds that did not (per
# stage; a stage's first build takes everything). While a stage hold is active
# (MPPI_STAGE_HOLD_S) the held stage's disc set is frozen as it was when the hold began.
# [팀 추가 2026-10-08] MPPI obstacles from a log-odds occupancy grid (occupancy_layer.py) with
# soft costs in the navfield (nav_field.OCC_L_*) instead of lidar discs. Pass-through beams lower
# a cell's log-odds, so momentary returns clear by themselves; only confirmed cells are lethal.
# With it the lidar discs are not used by MPPI (disc stabilization / freezing below then only
# apply to the stall-memory discs). Legacy keeps the disc layer unchanged.
MPPI_OCC_GRID = os.environ.get("MARC_MPPI_OCC_GRID", "1") != "0"   # 0 = lidar discs (ablation only)
MPPI_DISC_ADD_BUILDS = 2
MPPI_DISC_DROP_BUILDS = 5
MPPI_DISC_KEY_M = 0.1                 # disc identity = its centre rounded to this grid
# [팀 추가 2026-10-07] Spin stall: rt04_r5_mppi's last ~6 s commanded w = +1.2 at v = 0 while
# the yaw only wandered between -21 and -8 deg. A pure spin (|v| < MPPI_SPIN_CMD_V,
# |w| >= MPPI_SPIN_STALL_W) held for MPPI_SPIN_STALL_S that turned less than
# MPPI_SPIN_STALL_DEG -> back straight up MPPI_SPIN_ESCAPE_M (the track just driven); if that
# does not move the chassis either, side-step (crab, linear_y) MPPI_SPIN_ESCAPE_M towards the
# side with the lower cost-to-go.
MPPI_SPIN_STALL_S = 1.0
MPPI_SPIN_STALL_W = 0.5
MPPI_SPIN_STALL_DEG = 10.0
MPPI_SPIN_ESCAPE_M = 0.3
MPPI_SPIN_ESCAPE_V = 0.2
# Kerb / step layer (curb_layer.py) in the navfield. Step cells within MPPI_STEP_NEAR_M of the
# robot (its footprint) are dropped -- the robot is standing there, and a stale band around a
# kerb right beside it (the spawn) boxed it in.
MPPI_STEP_NEAR_M = 0.6
# (was 1.0 m kept only within +-35 deg ahead -- heading-dependent, so the field changed as the
#  robot turned and MPPI spun in place: live rt04b/rtC2_curb2_mppi, V(robot) 7.6 -> 11.0 per turn)
# step cells this close to the goal are left out: the pick target itself (a lying umbrella, a
# tumbler) and what it lies against are low height jumps (live rt04b_curb_mppi: goal cell made
# lethal and snapped away, object never reached). The leg stops 0.58 m short anyway.
MPPI_STEP_GOAL_CLEAR_M = 0.9
# last-resort stage (off-road open, only sensed steps block): slower, so a kerb / drop the
# lidar first sees 1.2-5 m ahead is in the field before the robot gets there
MPPI_LAST_RESORT_V = 0.6
# [팀 추가 2026-10-07] Slow zone near kerbs / ledges / the road edge: MPPI top speed and turn
# rate are lowered while the robot is within MPPI_SLOW_HAZARD_M of a sensed kerb/ledge line
# (curb_layer hazard) or within MPPI_SLOW_EDGE_M of the map's road edge; released past the
# *_OFF distances (hysteresis). The limits are the controller's own (MPPI.v_max / w_max), so the
# rollouts plan with them instead of the command being clipped afterwards. Live round 4: the
# same 1.1 m/s / 1.2 rad/s limits as legacy, but MPPI turns while moving -- rt02 rolled over at
# a kerb next to the spawn (2.8 rad/s spin at 1.5-1.76 m/s, 7 earlier passes there were fine)
# and rt04 drove off the road ledge at (-70, 123) at 1.3 m/s.
MPPI_SLOW_V = 0.5
# [팀 추가 2026-10-07] Depth grasp: raised pixels within DEPTH_CURB_EXCLUDE_M of a sensed kerb /
# ledge line (curb_layer.hazard_xy, MPPI runs only -- legacy has no curb layer) are not object
# candidates. rt03 mppi (rounds 3, 4, 5): approaching from the north-east, the first depth
# measurement was a "lying object 0.68-0.69 m long" 0.05 m from the map road edge and aligned
# with it -- the kerb; the umbrella (0.21 m) lies 0.27 m from that edge.
DEPTH_CURB_EXCLUDE_M = 0.15
MPPI_SLOW_W = 0.6
MPPI_SLOW_HAZARD_M, MPPI_SLOW_HAZARD_OFF_M = 1.5, 2.0
# Road-edge distance was 1.0 / 1.5 m: the round-4 MPPI paths spent 3-68 % (rt03) of their
# samples within 1.0 m of the map's road edge, and rt01_r5_mppi crawled at 0.5 m/s from the spawn
# on. 0.5 / 1.0 m keeps the zone to the edge strip itself; sensed kerbs keep 1.5 / 2.0 m.
MPPI_SLOW_EDGE_M, MPPI_SLOW_EDGE_OFF_M = 0.5, 1.0
# last-resort stage with the curb layer: off-road open, but the strip of off-road along the
# road (this wide) only where it was observed flat -- leave the road only at a sensed flat entry
MPPI_EDGE_BAND_M = 0.0   # off: offline (survey_rtC2_p1) the band left the goal unreachable,
                         # while all-open + sensed steps found a 33 m path crossing at most 1.8 cm
# with the curb layer the last-resort stage is held longer: live rtC2_curb_mppi flipped between
# normal and last-resort every 15 s, each pointing elsewhere, and the robot danced 10 m short
MPPI_LAST_RESORT_HOLD_S = 15.0   # (60 s while kerbs were in the path planning; back to 15 s)
# Kerbs are NOT in the path planning (live rt04b/rtC2_curb*_mppi: false steps from objects and
# low kerbs walled the way, rt04b 91 -> stuck 3-5 m short in 3 of 3 runs). They are a rollout
# safety check only (NavField hazard_xy): lines near the map road edge (MPPI_CURB_EDGE_M) from
# curb_layer.CURB_M, lines elsewhere from LEDGE_M, blobs never.
MPPI_CURB_EDGE_M = 1.5
# Waypoint-circle reroute -- bends only the path segments an obstacle circle actually
# crosses, in place, no A*/follower cycle. See mec_sim_ice/demo_agents/waypoint_avoid.py.
from waypoint_avoid import relocate
# Depth-camera ghost-wall veto -- clears lidar cells the front depth camera confirms as
# floor this frame, correcting the rotation-induced ground misclassification the lidar
# layer's own single-percentile estimate cannot avoid. See
# mec_sim_ice/demo_agents/depth_ground_veto.py.
import depth_ground_veto
# [팀 추가] Shape-based grasp point / gripper yaw from the depth blob -- see demo/grasp_shape.py.
import grasp_shape

log = logging.getLogger("participant_app")

# [스타터킷 원본] STAGE2_NAV_TIMEOUT_S ~ STAGE2_INFLATION_M, GOAL_TOLERANCE(아래)는
# 원본 그대로. STAGE2_STUCK_TIME_S/STAGE2_STUCK_MOVE_M은 이름과 기본값은 같지만
# "판정 방식"이 팀에 의해 틱 비교 -> 슬라이딩 윈도우로 바뀜(바로 아래 긴 주석 참고).
# 이 섹션 이후 LIDAR_*/REACT_STOP_*/BRAKE_STALL_S/BLOCKAGE_*/RELOCATE_MARGIN_M/
# DEPTH_EXPECT_WINDOW_M/ARM_REACH_FWD_M/MAX_PICK_ATTEMPTS/DEFAULT_HINT_RADIUS_M은
# 전부 [팀 신규 추가] 상수다.
# -- Stage 2 navigation parameters --
# Demo-internal giveup so the drive loop cannot spin forever if a goal is unreachable. This is NOT
# the competition timeout - the platform counts and enforces the real Stage 2 time limit.
STAGE2_NAV_TIMEOUT_S = 1800.0
STAGE2_NAV_RATE_HZ = 10.0
STAGE2_NAV_DT = 1.0 / STAGE2_NAV_RATE_HZ
STAGE2_INFLATION_M = 0.25      # erode free by the robot radius (avoids road<->grass step stalls)
# Stall test: total displacement over a trailing STAGE2_STUCK_TIME_S-second window must
# exceed STAGE2_STUCK_MOVE_M, or it is a stall. This used to compare consecutive ticks
# (0.1s apart) instead of a window -- since GoalSeekController's decel was (before the sqrt profile)
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
# [팀 추가] 속도 비례 제동거리 -- REACT_STOP_DIST_M 은 저속용 하한으로 남기고, 그 위로
#   d = 반경 + 여유 + v*t_react + v^2/(2*a_brake)
# 를 쓴다. 0.6m 고정은 0.8 m/s 시절 값이라 1.1 m/s 에서는 반응 시간 동안 달리는 거리 +
# 제동거리를 못 덮는다. v ~0.45 m/s 이하에서는 하한이 이겨서 저속 동작(좁은 통로에서
# 넓은 브레이크가 데드락을 만든 실측 -- 아래 콘 각 주석)이 예전과 같다.
# t_react = 내비 1틱 + 라이다 갱신 주기(2틱) / a_brake 는 섀시 실측이 없어 보수적 추정.
REACT_T_REACT_S = 0.3
REACT_BRAKE_DECEL = 1.0                     # m/s^2
REACT_EXTRA_MARGIN_M = 0.10
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

# [팀 추가] 종방향 가속도 제한 -- 전복 방지. MAX_LINEAR 가 1.1 로 오르면서 0 -> 최고속
# 계단 명령이 상단 무거운 차체를 뒤로 젖힌다. 감속은 컨트롤러 프로파일(DECEL 0.5)이
# 이미 계획하므로, 여기 감속 한도는 계획 밖 급정지(제자리 회전 전환 등)만 완화한다.
# 반응형 브레이크는 이 제한을 거치지 않는다 -- 비상정지는 늦출 이유가 없다.
STAGE2_ACCEL_MPS2 = 0.6
STAGE2_DECEL_MPS2 = 1.0
BLOCKAGE_RADIUS_M = 0.6                  # wider than the chassis, so A* must route around it
BLOCKAGE_GROWTH_M = 0.4                  # stalled inside a known blockage -> it was too small
BLOCKAGE_MAX_RADIUS_M = 2.0              # cap, so a bad mark cannot wall off the map

# Waypoint-circle reroute (see waypoint_avoid.relocate) -- tried before a full A* replan
# when a new lidar cluster crosses the remaining path. inflation_m (the leg's own A* erosion
# radius) is already the robot-radius clearance; a touch more accounts for the follower's
# P-control not tracking the bent path exactly (it cuts corners toward the next waypoint).
RELOCATE_MARGIN_M = 0.10

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
# Depth-refinement distractor window (m): only raised pixels within this radius of the
# nav-estimated object position count. Wide enough to absorb hint_center error at pick
# range (~0.5m observed), narrow enough to exclude the bulk of a parked car / vending
# machine sharing the frame. See _locate_object_by_depth.
DEPTH_EXPECT_WINDOW_M = 0.5
# Arm forward reach cap (m) -- the offline harness scores reach against 0.75; command a
# grasp beyond it and the descend waypoint just times out "unreached" and the gripper
# closes wherever the servo stalled (a silent air-grab). Clamp with a loud warning instead.
ARM_REACH_FWD_M = 0.72
# [팀 추가] 직진 접근 속도 = 남은 거리(fwd - target_fwd) 비례. 0.12 m/s 고정이던 것을
# 멀 때는 빨리, 붙을수록 천천히. 하한 0.06 은 예전 "lat 크게 틀어졌을 때" 속도와 같고,
# 10Hz 에서 정지 판정 밴드(-0.03~+0.02)를 한 틱에 건너뛰지 않는 속도다.
APPROACH_K = 0.8
APPROACH_V_MIN = 0.06
APPROACH_V_MAX = 0.30

# Retry budget for a failed pick (empty grasp, or grasped but missed the rear basket on place).
MAX_PICK_ATTEMPTS = 3
# [팀 추가 2026-10-07] Re-approach when the depth grasp point is near the arm's reach limit.
# rt03 mppi (1st and 2nd run): the robot stopped where legacy did (9 cm apart) but with another
# heading, the lying umbrella's grasp point came out at fwd 0.70 m (reach 0.72) and it slipped
# out during transport 3/3; legacy's was 0.38 m and held. Creep straight ahead until the point
# is at GRASP_REAPPROACH_TARGET_M and measure again -- but never so far that the near end of a
# lying object (grasp point - length/2 along the axis) comes closer than GRASP_NEAR_END_MIN_M
# to the arm base (the chassis front would drive onto it).
GRASP_REAPPROACH_FWD_M = 0.60
GRASP_REAPPROACH_TARGET_M = 0.47
GRASP_REAPPROACH_MAX_M = 0.35
GRASP_NEAR_END_MIN_M = 0.25
# Lying objects: grasp_shape slides the grasp along the object's axis to stay within this
# forward distance. Was ARM_REACH_FWD_M (0.72) -> the slide stopped right AT the reach limit and
# the same 0.72 came back after every re-approach (rt03_r3_mppi). The final clamp to the true
# reach (ARM_REACH_FWD_M) below is unchanged.
GRASP_SLIDE_FWD_M = 0.62
# [팀 추가 2026-10-07] Retry: before, every retry re-aligned to the ORIGINAL hint and, with
# nothing raised in the camera window, grasped the nav estimate blind -- rt04_r3_mppi: attempts 2
# and 3 "empty hand at lift" with no depth at all. Now: fold the arm (it must not block the base
# camera), re-align and approach the last known object position, measure with a slightly wider
# window (a rt04 distractor umbrella sits 0.38 m from the tumbler, so it stays small); nothing
# there -> skip the remaining attempts and go deliver. (Round 4 also searched +-30 deg in place
# first; removed -- see PICK_SPIN_*.)
RETRY_WINDOW_M = 0.6
# Small pure-rotation commands do not reliably spin this chassis in the pick phase: the robot
# slid sideways instead (rt03_r4_mppi: 1.8 m during the re-approach turn, 2.9 m during the
# -30 deg re-localization turn with +8 deg of yaw; a +10 deg view turn slid 0.8 m in 6 s with
# the yaw unchanged -- the depth views confirmed the slide). The one remaining pick-phase turn
# (re-approach) uses a constant PICK_SPIN_W and stops as soon as the chassis centre has moved
# PICK_SPIN_MAX_SHIFT_M; the grasp point is then re-framed via world xy.
PICK_SPIN_W = 0.5
PICK_SPIN_MAX_SHIFT_M = 0.05

class DiscStabilizer:
    """Temporal filter for lidar discs (see MPPI_DISC_ADD_BUILDS / MPPI_DISC_DROP_BUILDS)."""

    def __init__(self, n_add=MPPI_DISC_ADD_BUILDS, n_drop=MPPI_DISC_DROP_BUILDS, key_m=MPPI_DISC_KEY_M):
        self.n_add, self.n_drop, self.key_m = n_add, n_drop, key_m
        self.seen, self.miss, self.active = {}, {}, {}
        self.builds = 0

    def update(self, discs):
        cur = {}
        for d in discs:
            cur[(int(round(d[0] / self.key_m)), int(round(d[1] / self.key_m)))] = d
        first = self.builds == 0
        self.builds += 1
        for k, d in cur.items():
            self.seen[k] = self.seen.get(k, 0) + 1
            self.miss[k] = 0
            if first or self.seen[k] >= self.n_add or k in self.active:
                self.active[k] = d
        for k in list(self.seen):
            if k in cur:
                continue
            self.seen[k] = 0
            self.miss[k] = self.miss.get(k, 0) + 1
            if self.miss[k] >= self.n_drop:
                self.active.pop(k, None)
                del self.seen[k], self.miss[k]
        return list(self.active.values())


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


# [팀 신규 추가] 클래스 전체가 starter kit에 없던 것 (starter kit은 그냥
# `now_s() + budget`을 데드라인으로 썼음 -- 아래 docstring이 설명하는 버그의 원인).
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

    # [스타터킷 골격 + 확장] self.client/self.vla와 목표 좌표 관련 속성
    # (_collect_xy, _stage2_target_xy, _pickup_xy, _occ_logged)은 원본에 있던
    # 것. 그 아래(_tf_buffer부터 _clock까지)는 전부 팀이 추가한 상태 -- 주행/회피
    # 쪽은 _lidar_layer, _lidar_mount_xy, _cam_mount_R/_t, _lidar_fail_streak,
    # _clock, _obstacles(원본엔 load_obstacles()로 채웠으나 실전용으로 빈 리스트로
    # 바꿈)가 해당.
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
        # Ground z at the pick site (reveal hint_center[2]) -- see _on_stage2_reveal for
        # why _object_arm_xy must not hand tf2 a z=0 point on this terrain.
        self._collect_z = 0.0
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
        self._lidar_acc = None        # LidarSliceAccumulator, built once on the first run
        self._curb = None             # CurbLayer, fresh per run (curb_layer.py)
        self._curb_edge_zone = None   # map road-edge test from the last MPPI build (hazard_xy)
        self._occ = None              # OccupancyLayer (MPPI_OCC_GRID), fresh per run
        # Base_LiDAR -> chassis mount offset (dx, dy). Static, so tf is asked once per run;
        # (0, 0) (sensor above the chassis origin) when tf cannot supply it.
        self._lidar_mount_xy = (0.0, 0.0)
        # base_link -> Base_Camera_Center mount rotation/translation, for depth_ground_veto.
        # Static (chassis-mounted, not arm-mounted), so tf is asked once per run -- same
        # reasoning as _lidar_mount_xy above, but a full rotation is needed (not just dx,dy)
        # since the camera is tilted to see the floor ahead. None -> veto is skipped for the
        # whole run (lidar-only fallback, unaffected).
        self._cam_mount_R = None
        self._cam_mount_t = None
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

    # [스타터킷 원본] 등록 순서/내용 모두 원본 그대로.
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

    # [팀 신규 추가] (Stage1 담당) -- CCTV 워밍업. starter kit에는 없던 메서드.
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

    # [스타터킷 골격 + 확장] (Stage1 담당) -- 원본은 곧바로 self.vla.process()를
    # 불렀고, CCTV 워밍업 전까지 워커 스레드로 미루는 분기는 팀이 추가.
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

    # [팀 신규 추가] (Stage1 담당) -- 아래 두 메서드 모두 starter kit에 없던 것.
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

    # [스타터킷 골격 + 확장] owner_position 읽기/grounding 제출 흐름은 원본과 동일.
    # _stage2_submitted_target(리비전 로그용)만 팀이 추가.
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

    # [스타터킷 골격 + 확장] hint_center(x,y)를 _collect_xy로 받는 부분은 원본과
    # 동일. target_type/collect_z(지면 높이)/hint_radius 저장, reveal 로그 남기기는
    # 전부 팀이 추가(주행 목표 정밀도를 높이기 위함).
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
            # Ground z at the pick site. chungmu terrain sits at world z~16.3-16.7, so a
            # transform of (x, y, 0.0) would hand tf2 a point ~16m under the ground -- any
            # non-yaw tilt in the target frame then leaks that huge z error into (fwd, lat)
            # (1deg of tilt x 16m = 0.29m). Latent today (world->arm_base_link never
            # resolves, see _object_arm_xy), but armed the moment that tf ever connects.
            if len(reveal.hint_center) >= 3:
                self._collect_z = float(reveal.hint_center[2])
        try:
            # 0.0 is not a usable radius -- treat it as "not stated" and fall back.
            self._hint_radius = float(reveal.hint_radius) or None
        except (TypeError, ValueError):
            self._hint_radius = None
        self._append_reveal_log(reveal)

    # [스타터킷 골격 + 확장] 큰 흐름(팔 접기 -> standoff 접근 -> 직진 접근 -> 픽 ->
    # 배달 -> task_complete)은 원본과 동일. 팀이 확장한 부분(주행/회피 쪽):
    #   - ElapsedClock 도입, lidar_layer/카메라 마운트 tf 셋업(_lidar_mount_offset,
    #     _camera_mount_transform)을 매 런마다 새로 함
    #   - _align_to_object() 호출 추가(standoff 도착 각도 보정)
    # 그 외(픽 재시도 루프, 뎁스 정밀 파지, 바구니 확인)는 매니퓰레이션 담당 쪽 확장.
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
        self._lidar_acc = self._lidar_accumulator()
        self._curb = CurbLayer() if LOCAL_PLANNER == "mppi" else None
        self._occ = OccupancyLayer() if LOCAL_PLANNER == "mppi" and MPPI_OCC_GRID else None
        self._lidar_mount_xy = self._lidar_mount_offset()
        self._cam_mount_R, self._cam_mount_t = self._camera_mount_transform()

        # Top-heavy robot tip-over prevention: fold the arm to HOME before driving -> lower CoG.
        self._stow_arm()

        picked = False
        # -- leg 1: drive at the object itself and stop the moment it is TUMBLER_STANDOFF_M away,
        # then straight-approach into arm reach, then pick --
        # [팀 수정] The goal used to be a standoff point fixed up front: TUMBLER_STANDOFF_M short
        # of the object on the straight start->object line, blind to what is around it. In rt01
        # that line crosses an SUV (object 1.68 m from it), so the goal landed inside the car
        # body. Now the planner routes to the object (the path arrives from whichever side is
        # free) and the leg ends when the object is TUMBLER_STANDOFF_M away (tol below).
        log.info("[Stage2] leg1 -> object %s, stop at %.2fm from it",
                 tuple(round(v, 2) for v in collect_xy), TUMBLER_STANDOFF_M)
        # Aggressive planner tuning for the tight pick approach (low inflation + goal-exclusion
        # and snap sized by the stated hint uncertainty rather than by this scene's spacing).
        self._navigate_to(collect_xy, deadline, tol=TUMBLER_STANDOFF_M, inflation_m=0.10,
                          exclude_m=max(0.6, hint_r), snap_m=max(1.0, hint_r),
                          avoid_xy=collect_xy, offroad_r=hint_r)
        if not c.is_running:
            return
        self._halt()
        # Square up before creeping in -- see _align_to_object for why this can't be
        # skipped even though _approach_straight also steers.
        self._align_to_object(collect_xy, deadline)
        # Final straight approach: drive straight at the object until arm_base fwd reaches grasp reach.
        # _approach_straight ends with _halt(), which confirms the chassis has actually
        # stopped (the driver keeps applying the last cmd_vel, so residual speed would
        # overshoot the object) -- no separate fixed stop wait needed here.
        self._approach_straight(collect_xy, deadline, target_fwd=0.40)
        log.info("[Stage2] standoff stop -- ready to pick")

        # Retry: is_grasping() (via run_pick_sequence's returned held) catches an empty-handed
        # close; is_basket_occupied() additionally catches a held object that missed the rear
        # basket on place (grip alone would miss this, since the arm sequence has already
        # released the gripper by the time it returns).
        basket_ok = False
        reapproached = 0.0      # total chassis creep for re-approach this pick (once, <= MAX)
        last_obj_world = tuple(collect_xy)   # last place the object was seen (world xy)
        for attempt in range(1, MAX_PICK_ATTEMPTS + 1):
            if not c.is_running or self._clock.tick() >= deadline:
                log.warning("[Stage2] pick retry budget exhausted (run stopped or deadline hit) "
                            "before attempt %d/%d", attempt, MAX_PICK_ATTEMPTS)
                break
            try:
                grasp_xy = self._object_arm_xy(collect_xy)   # actual object position in the arm_base frame
                # camera+depth refinement, if available -- anchored to the nav estimate so a
                # distractor sharing the frame cannot drag the grasp point (see the method).
                depth_xy = self._locate_object_by_depth(expected_xy=grasp_xy)
                if depth_xy is not None:
                    log.info("[Stage2] depth refinement grasp=%s (nav estimate was %s)",
                             tuple(round(v, 3) for v in depth_xy),
                             tuple(round(v, 3) for v in grasp_xy) if grasp_xy else None)
                    grasp_xy = depth_xy
                    moved, expect = ((self._reapproach_for_grasp(depth_xy) if reapproached == 0.0
                                      else (0.0, None)))
                    if moved:
                        reapproached = moved
                        # anchor = the grasp point just measured, moved by the turn + creep (the
                        # nav estimate of the object centre fell out of the camera window once
                        # the chassis was 0.25 m closer: rt03_r3_mppi re-measured None 3/3)
                        shifted = tuple(expect) + tuple(depth_xy[2:])
                        again = self._locate_object_by_depth(expected_xy=shifted[:2])
                        log.info("[Stage2] re-measured after re-approach: %s",
                                 tuple(round(v, 3) for v in again) if again else None)
                        # never fall back to the pre-creep point (it is now `moved` too far)
                        grasp_xy = again if again is not None else shifted
                    last_obj_world = self._arm_to_world(grasp_xy[0], grasp_xy[1]) or last_obj_world
                if grasp_xy is not None and grasp_xy[0] > ARM_REACH_FWD_M:
                    # Beyond reach the descend waypoint just times out "unreached" and the
                    # gripper closes mid-air wherever the servo stalled -- surface it loudly.
                    log.warning("[Stage2] grasp fwd %.2fm exceeds arm reach %.2fm -- "
                                "clamping (approach probably stopped short; pick may miss)",
                                grasp_xy[0], ARM_REACH_FWD_M)
                    # Preserve the depth-measured grasp height if present (3rd element).
                    grasp_xy = (ARM_REACH_FWD_M,) + tuple(grasp_xy[1:])
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
                self._stow_arm()      # the arm must not block the base camera
                self._align_to_object(last_obj_world, deadline)
                self._approach_straight(last_obj_world, deadline, target_fwd=0.40)
                exp = self._object_arm_xy(last_obj_world)
                found = (self._locate_object_by_depth(expected_xy=exp, window_m=RETRY_WINDOW_M)
                         if exp is not None and exp[0] > 0.0 else None)
                where = self._arm_to_world(found[0], found[1]) if found is not None else None
                if where is None:
                    log.warning("[Stage2] retry: nothing within %.1f m of the last known object "
                                "position -> skipping the remaining pick attempts", RETRY_WINDOW_M)
                    break
                log.info("[Stage2] retry: object at (%.2f, %.2f) (last seen (%.2f, %.2f))",
                         where[0], where[1], last_obj_world[0], last_obj_world[1])
                collect_xy = where
                last_obj_world = where
                self._align_to_object(collect_xy, deadline)
                self._approach_straight(collect_xy, deadline, target_fwd=0.40)
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

    # [스타터킷 원본] 로직 변경 없음.
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

    # [스타터킷 원본] 로직 변경 없음.
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

    # [스타터킷 골격 + 확장] 원본은 아래 pose-math 폴백 계산(ARM_BASE_FWD 가정)
    # 하나뿐이었다. 팀이 tf2 기반 실측 경로(world->arm_base_link)를 앞에 추가하고,
    # 그게 실패할 때만 원본 계산으로 떨어지도록 바꿨다.
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
                # NOTE: the z here must be the real ground height (reveal hint_center[2]),
                # NOT 0.0 -- chungmu ground sits at world z~16.5, and a point 16m below
                # ground leaks into (fwd, lat) through any tilt in the transform.
                fwd, lat = transform_via_tf2(
                    self._tf_buffer, (obj_xy[0], obj_xy[1], self._collect_z))
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

    # [팀 신규 추가] 메서드 전체가 starter kit에 없던 것 -- standoff 도착 각도가
    # 틀어졌을 때(A* 재계획/우회 후 등) 직진 접근 전에 먼저 제자리 회전으로 정렬.
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
        self._halt()
        gxy = self._object_arm_xy(obj_xy)
        log.info("[Stage2] align done -- object arm coords %s",
                 tuple(round(v, 3) for v in gxy) if gxy else None)
    # [팀 신규 추가, 매니퓰레이션 담당 쪽 확장] 메서드 전체가 starter kit에 없던
    # 것 -- 주행/회피와는 다른 목적(파지 정밀도)이지만 depth_ground_veto.py와
    # 같은 좌표 변환 방식(핀홀 역투영 + tf 회전행렬)을 재사용한다.
    def _locate_object_by_depth(self, expected_xy=None, window_m=None):
        """Refine the pick point using the base camera's own depth (floor-height segmentation).

        expected_xy: nav-estimated (fwd, lat) of the object in the arm_base frame. When
        given, only raised pixels within DEPTH_EXPECT_WINDOW_M of it count -- distractor
        rejection that needs no scipy and no identity model (see the candidate filter
        below for why "largest blob" is the wrong rule on exactly the frames that matter).

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
        candidate = self._exclude_kerb_pixels(candidate, fwd, lat)
        n_cand = int(np.sum(candidate))
        if n_cand < 20:
            return None

        if expected_xy is not None:
            # Distractor rejection anchored to the nav estimate. "Largest blob" -- and,
            # with scipy absent (true on the live host), the even blunter fallback of
            # averaging EVERY raised pixel in frame -- gets dragged by whatever else is
            # raised in view: the parked car on rt01, the vending machine on rt03.
            # Those are exactly the frames where this refinement matters most. The nav
            # estimate is only hint-grade, but the window merely has to exclude *other*
            # objects, not be precise. Nothing near the estimate -> None (callers keep
            # the nav estimate), never a polluted average.
            ef, el = float(expected_xy[0]), float(expected_xy[1])
            win = DEPTH_EXPECT_WINDOW_M if window_m is None else float(window_m)
            near = candidate & ((fwd - ef) ** 2 + (lat - el) ** 2 <= win ** 2)
            if int(np.sum(near)) >= 20:
                # [팀 추가] Shape-based grasp (demo/grasp_shape.py): picks the blob nearest
                # the nav estimate, corrects the near-side bias, and returns a gripper yaw
                # that closes across a long object. Each part has its own switch there.
                est = grasp_shape.estimate_grasp(
                    fwd[near], lat[near], up[near], floor_z,
                    expected_xy=(ef, el), reach_fwd=GRASP_SLIDE_FWD_M)
                if est is not None:
                    log.info("[Stage2] depth grasp shape=%s fwd=%.3f lat=%.3f down=%.3f yaw=%s len=%s",
                             est["shape"], est["fwd"], est["lat"], est["down"],
                             "default" if est["yaw"] is None else "%.2f" % est["yaw"],
                             "-" if est.get("length") is None else "%.2f" % est["length"])
                    out = (est["fwd"], est["lat"], est["down"])
                    if est["yaw"] is None:
                        return out
                    # 5th element = visible length -> arm_pick's long-object transport
                    return out + (est["yaw"], est["length"]) if est.get("length") else out + (est["yaw"],)
                # Measure the grasp HEIGHT too, not just (fwd, lat). grasp_down defaults
                # are tuned per object family for a STANDING tumbler; a lying umbrella's
                # shaft sits at floor level, so a fixed -0.28 closed the gripper ~5cm
                # above it (measured live rt02 2026-09-01: fingers brushed the shaft top,
                # inner_fk spread to 44-45deg, and the hold check misread it as empty).
                # Grab slightly below the blob's top surface, floored just above the
                # measured ground so the fingers cannot jam into the floor.
                obj_up = float(np.mean(up[near]))
                down = max(obj_up - 0.01, floor_z + 0.012)
                return (float(np.mean(fwd[near])), float(np.mean(lat[near])), down)
            log.info("[Stage2] depth refinement: no raised blob within %.2fm of the nav "
                     "estimate (%d raised px elsewhere in frame) -- keeping the nav "
                     "estimate", win, n_cand)
            return None

        try:
            from scipy import ndimage
            labeled, n_blobs = ndimage.label(candidate)
            sizes = ndimage.sum(candidate, labeled, index=range(1, n_blobs + 1))
            blob_mask = labeled == (int(np.argmax(sizes)) + 1)
        except ImportError:
            blob_mask = candidate

        return (float(np.mean(fwd[blob_mask])), float(np.mean(lat[blob_mask])))

    # [팀 신규 추가 2026-10-07] DEPTH_CURB_EXCLUDE_M 위 주석 참고.
    def _exclude_kerb_pixels(self, candidate, fwd, lat):
        """Drop candidate pixels within DEPTH_CURB_EXCLUDE_M of a sensed kerb / ledge line."""
        if self._curb is None or not np.any(candidate):
            return candidate
        pose = self.client.get_world_pose()
        if pose is None:
            return candidate
        try:
            hz = self._curb.hazard_xy(near_edge=self._curb_edge_zone)
        except Exception as e:
            log.warning("[Stage2] depth kerb exclusion: hazard_xy failed: %s", e)
            return candidate
        if not len(hz):
            return candidate
        hz = hz[np.hypot(hz[:, 0] - pose[0], hz[:, 1] - pose[1]) < 2.5]
        if not len(hz):
            return candidate
        # hazard cells -> raw arm frame (pose math, same convention as _world_to_arm_pose)
        ha = np.array([self._world_to_arm_pose(pose, hx, hy) for hx, hy in hz])
        ha[:, 0] += GRASP_FWD_BACKOFF
        idx = np.flatnonzero(candidate)
        cf, cl = fwd.ravel()[idx], lat.ravel()[idx]
        near = np.zeros(len(idx), bool)
        for hf, hl in ha:
            near |= (cf - hf) ** 2 + (cl - hl) ** 2 <= DEPTH_CURB_EXCLUDE_M ** 2
        if near.any():
            out = candidate.copy()
            out.ravel()[idx[near]] = False
            log.info("[Stage2] depth: %d/%d raised px within %.2fm of %d kerb/ledge cell(s) "
                     "-> not object candidates", int(near.sum()), len(idx), DEPTH_CURB_EXCLUDE_M,
                     len(hz))
            return out
        return candidate

    # [스타터킷 골격 + 확장] 직진 접근 + lat 중심맞춤 로직은 원본과 동일. 팀이
    # 추가한 것: cap_angular() 적용(섀시 정지 방지), 그리고 reversing 로직(과주행
    # 시 후진해서 파지 거리 복구 -- 아래 docstring의 rt01 실측 사례 참고).
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
            # Speed proportional to the remaining depth (see APPROACH_K); badly off-center
            # -> creep at the floor while turning, as before.
            lin = max(APPROACH_V_MIN, min(APPROACH_V_MAX, APPROACH_K * (fwd - target_fwd)))
            if abs(lat) >= 0.15:
                lin = APPROACH_V_MIN
            # Same chassis validity cap the path controllers use: without it this leg emits
            # 2v <= w*T whenever |lat| >= 0.211, and the chassis silently stops -- it burned
            # the whole 20 s budget moving 6 mm, leaving the object out of arm reach.
            ang = cap_angular(lin, ang)
            c.send_cmd_vel(linear_x=lin, angular_z=ang)
            c.sleep(STAGE2_NAV_DT)
        self._halt()
        gxy = self._object_arm_xy(obj_xy)
        log.info("[Stage2] straight approach done -- object arm coords %s",
                 tuple(round(v, 3) for v in gxy) if gxy else None)

    # [팀 신규 추가 2026-10-07] arm_base (fwd, lat) -> world xy, inverse of _object_arm_xy's
    # pose-math fallback (same ARM_BASE_FWD assumption). None without a pose.
    def _arm_to_world(self, fwd, lat):
        pose = self.client.get_world_pose()
        if pose is None:
            return None
        x, y, yaw = pose
        f = float(fwd) + ARM_BASE_FWD + GRASP_FWD_BACKOFF
        return (x + f * math.cos(yaw) - float(lat) * math.sin(yaw),
                y + f * math.sin(yaw) + float(lat) * math.cos(yaw))

    @staticmethod
    def _arm_pose_to_world(pose, fwd, lat):
        """Pose-math arm (fwd, lat) -> world xy for a given chassis pose (see _arm_to_world)."""
        x, y, yaw = pose
        f = float(fwd) + ARM_BASE_FWD + GRASP_FWD_BACKOFF
        return (x + f * math.cos(yaw) - float(lat) * math.sin(yaw),
                y + f * math.sin(yaw) + float(lat) * math.cos(yaw))

    @staticmethod
    def _world_to_arm_pose(pose, wx, wy):
        """Inverse of _arm_pose_to_world."""
        x, y, yaw = pose
        dx, dy = wx - x, wy - y
        f = dx * math.cos(yaw) + dy * math.sin(yaw)
        lat = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return (f - ARM_BASE_FWD - GRASP_FWD_BACKOFF, lat)

    # [팀 신규 추가 2026-10-07] PICK_SPIN_* 위 주석 참고.
    def _spin_checked(self, angle, max_t=6.0):
        """Turn in place by `angle` at a constant PICK_SPIN_W; stop early if the chassis centre
        moves more than PICK_SPIN_MAX_SHIFT_M. True = turned without sliding."""
        c = self.client
        p0 = c.get_world_pose()
        if p0 is None:
            return False
        target = p0[2] + angle
        ok = False
        t_end = self._clock.elapsed + max_t
        while c.is_running and self._clock.tick() < t_end:
            p = c.get_world_pose()
            if p is None:
                c.sleep(STAGE2_NAV_DT)
                continue
            shift = math.hypot(p[0] - p0[0], p[1] - p0[1])
            if shift > PICK_SPIN_MAX_SHIFT_M:
                log.warning("[Stage2] spin: chassis slid %.2f m during a %.0f deg turn (yaw %.1f "
                            "deg done) -> stop", shift, math.degrees(angle), math.degrees(
                                math.atan2(math.sin(p[2] - p0[2]), math.cos(p[2] - p0[2]))))
                break
            err = math.atan2(math.sin(target - p[2]), math.cos(target - p[2]))
            if abs(err) < math.radians(2):
                ok = True
                break
            c.send_cmd_vel(0.0, math.copysign(PICK_SPIN_W, err))
            c.sleep(STAGE2_NAV_DT)
        self._halt()
        p = c.get_world_pose()
        return ok and p is not None and math.hypot(p[0] - p0[0], p[1] - p0[1]) <= PICK_SPIN_MAX_SHIFT_M

    # [팀 신규 추가 2026-10-07] GRASP_REAPPROACH_* 위 주석 참고.
    def _reapproach_for_grasp(self, grasp):
        """Grasp point near the reach limit: turn to face it, then creep straight in.

        Returns (moved_m, expected_after) -- expected_after = the grasp point's predicted
        (fwd, lat) in the arm frame after the turn and the creep (anchor for re-measuring), or
        (0.0, None) when nothing was done. Turning first matters: rt03_r3_mppi had the point at
        fwd 0.73 / lat -0.34 with the umbrella lying across the robot -- a straight creep left it
        0.34 m to the side and the re-measure locked onto another object.
        """
        fwd, lat = float(grasp[0]), float(grasp[1])
        if fwd <= GRASP_REAPPROACH_FWD_M:
            return 0.0, None
        c = self.client
        p0 = c.get_world_pose()
        if p0 is None:
            return 0.0, None
        # 1) turn about the chassis centre (arm base is ARM_BASE_FWD ahead of it) to face the point
        cx_, cy_ = fwd + ARM_BASE_FWD, lat
        turn = max(-math.radians(35), min(math.radians(35), math.atan2(cy_, cx_)))
        pw = self._arm_pose_to_world(p0, fwd, lat)     # the point in world xy
        p0_turn = p0
        if abs(turn) > math.radians(4):
            # slide-guarded (see PICK_SPIN_W): rt03_r4_mppi's turn slid the chassis 1.8 m
            # sideways and turned only 9 of the 20 deg
            self._spin_checked(turn)
        p1 = c.get_world_pose() or p0
        turn = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
        # point in the arm frame after the turn (via world xy: also right if the chassis slid)
        fwd, lat = self._world_to_arm_pose(p1, *pw)
        want = min(fwd - GRASP_REAPPROACH_TARGET_M, GRASP_REAPPROACH_MAX_M)
        if len(grasp) >= 5 and grasp[4]:
            # lying object: keep its near end clear of the chassis front (axis yaw is relative
            # to the arm frame, so it turns with the chassis)
            near_end = fwd - 0.5 * float(grasp[4]) * abs(math.cos(float(grasp[3]) - turn))
            want = min(want, near_end - GRASP_NEAR_END_MIN_M)
        log.info("[Stage2] grasp near reach -> turned %.0f deg, point now fwd %.2f lat %.2f, "
                 "creep %.2f m", math.degrees(turn), fwd, lat, max(want, 0.0))
        moved = 0.0
        if want >= 0.05:
            p0 = c.get_world_pose() or p0
            t_end = self._clock.elapsed + 10.0
            while c.is_running and self._clock.tick() < t_end:
                p = c.get_world_pose()
                if p is not None:
                    moved = math.hypot(p[0] - p0[0], p[1] - p0[1])
                    if moved >= want:
                        break
                c.send_cmd_vel(linear_x=APPROACH_V_MIN, angular_z=0.0)
                c.sleep(STAGE2_NAV_DT)
            self._halt()
            p = c.get_world_pose()
            if p is not None:
                moved = math.hypot(p[0] - p0[0], p[1] - p0[1])
        if abs(turn) < math.radians(4) and moved < 0.05 and math.hypot(p1[0] - p0_turn[0], p1[1] - p0_turn[1]) < 0.02:
            return 0.0, None
        p = c.get_world_pose()
        return max(moved, 1e-3), (self._world_to_arm_pose(p, *pw) if p is not None else (fwd - moved, lat))

    # [팀 신규 추가, 매니퓰레이션 담당 쪽] starter kit에 없던 메서드.
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

    # [스타터킷 골격 + 확장] 폴링 로직은 원본과 동일, ElapsedClock으로 시간 기준만
    # (now_s() 절대값 -> 경과시간) 교체.
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

    # ========================================================================
    # [팀 신규 추가 구간] 여기서부터 _reverse_escape()까지 -- lidar_obstacles /
    # depth_ground_veto / waypoint_avoid 를 실제 tf·토픽에 연결하고, 회피 관련
    # 상태(장애물 원 목록, 정체 기억)를 만드는 메서드들. starter kit에는 전부
    # 없던 것들이며, 이번 대화에서 설명한 "주행/회피" 로직의 핵심부.
    # ========================================================================
    def _lidar_mount_offset(self):
        """Base_LiDAR -> chassis (dx, dy) from tf, or (0, 0) when tf cannot answer.

        Static transform, so it is read once per run rather than per scan. The fallback
        assumes the sensor sits above the chassis origin; that is a few centimetres of
        error on a 0.25 m grid, which is why a failure here only degrades accuracy instead
        of disabling the layer.

        Frame reality (probed live, 2026-09-01): the tf forest is split in two -- the
        chassis frame is `car_base_link` (there is NO `base_link`), and every sensor hangs
        under `arm_base_link`, which has no edge to the world/car tree at all. So the only
        resolvable transform is `arm_base_link -> Base_LiDAR`; the arm_base->chassis leg is
        bridged with the same ARM_BASE_FWD forward-offset assumption _object_arm_xy's
        fallback already relies on (one shared assumption, not a second independent one).
        """
        if self._tf_buffer is None:
            return (0.0, 0.0)
        try:
            import rclpy
            tf = self._tf_buffer.lookup_transform(
                "arm_base_link", "Base_LiDAR", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            t = tf.transform.translation
            dx, dy = float(t.x) + ARM_BASE_FWD, float(t.y)
            log.info("[Stage2] lidar mount offset from tf (arm_base + ARM_BASE_FWD): "
                     "(%.3f, %.3f)", dx, dy)
            return (dx, dy)
        except Exception as e:
            log.info("[Stage2] lidar mount offset unavailable (%s) -- assuming (0, 0)", e)
            return (0.0, 0.0)

    # [팀 신규 추가] Isaac Sim 라이다는 ~70도 조각을 따로 발행한다 -- lidar_accumulator.py.
    def _lidar_accumulator(self):
        """The slice accumulator (subscribed once per process; drained at each run start),
        or None when this client cannot subscribe (offline harness -> get_lidar() path)."""
        acc = getattr(self, "_lidar_acc", None)
        if acc is None:
            try:
                from sensor_msgs.msg import PointCloud2
                from marc_sdk import protocol as P
                c = self.client
                acc = LidarSliceAccumulator(c, c.topics.robot("lidar/points"), PointCloud2,
                                            P.QOS_IMAGE)
            except Exception as e:
                log.info("[Stage2] lidar slice accumulator unavailable (%s) -- "
                         "using get_lidar() latest slice", e)
                return None
            if not acc.ok:
                return None
            log.info("[Stage2] lidar slice accumulator on (all slices merged per update)")
        acc.take_merged((0.0, 0.0, 0.0))     # drop slices queued before this run
        return acc

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
            msg = None
            if self._lidar_acc is not None:
                msg = self._lidar_acc.take_merged(pose, self._lidar_mount_xy)
            if msg is None:
                msg = self.client.get_lidar()
            n = self._lidar_layer.update(msg, pose, self._lidar_mount_xy)
            if self._occ is not None:
                self._occ.update(msg, pose, self._lidar_mount_xy)
            if self._curb is not None:
                pts = lidar_xyz(msg)
                if pts is not None and len(pts):
                    self._curb.update(*lidar_world(pts, pose, self._lidar_mount_xy))
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

    def _camera_mount_transform(self):
        """chassis -> Base_Camera_Center (R, t) from tf, or (None, None) if unavailable.

        Static transform, read once per run -- same reasoning as _lidar_mount_offset, but
        a full rotation is needed here (not just dx, dy) since the camera is tilted to see
        the floor ahead. None -> depth ghost-veto is skipped for the whole run; the lidar
        layer works unaffected (this is a correction on top of it, not a dependency).

        Frame reality (probed live, 2026-09-01): there is no `base_link`, and the sensors
        hang under `arm_base_link`, which is disconnected from the world/car tree. So this
        resolves `arm_base_link -> Base_Camera_Center` (which does exist) and bridges
        arm_base->chassis with the ARM_BASE_FWD forward offset, assuming no relative
        rotation -- the exact assumption _object_arm_xy's pose-math fallback already makes.
        A ~0.1-0.2m residual mount error is fine here: the veto clears 0.25m grid cells in
        patches, not point-precise geometry.
        """
        if self._tf_buffer is None:
            return None, None
        try:
            import rclpy
            tf = self._tf_buffer.lookup_transform(
                "arm_base_link", "Base_Camera_Center", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            log.info("[Stage2] camera mount transform unavailable (%s) -- "
                     "depth ghost-veto off", e)
            return None, None
        tr, rot = tf.transform.translation, tf.transform.rotation
        R = depth_ground_veto.quat_to_rotmat(rot.x, rot.y, rot.z, rot.w)
        t = np.array([tr.x + ARM_BASE_FWD, tr.y, tr.z])
        log.info("[Stage2] camera mount transform acquired (arm_base + ARM_BASE_FWD) -- "
                 "depth ghost-veto active")
        return R, t

    def _update_depth_veto(self, pose):
        """Clear lidar cells the front depth camera confirms as floor this frame.

        See depth_ground_veto.py for why this fixes what the lidar layer's own ground
        estimate cannot: a single frame's plane fit does not accumulate the per-azimuth
        error that makes rotation a ghost-wall generator for a 360-degree sweep.

        Every failure mode here (no tf, no depth topic yet, camera_info/depth size
        mismatch, a raw exception) returns 0 silently -- this is a correction layered on
        top of an already fail-tolerant lidar layer, not a dependency it needs to keep
        working.
        """
        if self._lidar_layer is None or self._cam_mount_R is None:
            return 0
        c = self.client
        try:
            from sensor_msgs.msg import CameraInfo
        except Exception:
            return 0
        depth_msg = c.get_robot_depth("base")
        if depth_msg is None:
            return 0
        if self._depth_cam_info_sub is None:
            def _on_depth_info(msg):
                self._depth_cam_info = msg
            self._depth_cam_info_sub = c.subscribe(
                c.topics.robot("base_camera/depth/info"), CameraInfo, _on_depth_info)
        info = self._depth_cam_info
        if info is None or (info.width, info.height) != (depth_msg.width, depth_msg.height):
            return 0
        try:
            h, w = depth_msg.height, depth_msg.width
            depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(h, w)
            K = np.array(info.k, dtype=np.float64).reshape(3, 3)
            pts, obst = depth_ground_veto.floor_and_obstacle_world_xy(
                depth, K, self._cam_mount_R, self._cam_mount_t, pose)
            if self._curb is not None:
                f_, l_, u_ = depth_ground_veto.base_points(
                    depth, K, self._cam_mount_R, self._cam_mount_t)
                x_, y_, yaw_ = pose
                c_, s_ = math.cos(yaw_), math.sin(yaw_)
                self._curb.update(x_ + f_ * c_ - l_ * s_, y_ + f_ * s_ + l_ * c_, u_)
        except Exception as e:
            log.warning("[Stage2] depth ghost-veto frame failed: %s: %s", type(e).__name__, e)
            return 0
        cleared = self._lidar_layer.veto_free(pts, keep=obst)
        if self._occ is not None:
            self._occ.free_points(pts, keep=obst)
        if cleared:
            log.info("[Stage2] depth ghost-veto: cleared %d cell(s)", cleared)
        return cleared

    # [팀 신규 추가] curb_layer.hazard_xy 의 near_edge: 지도 도로 경계 MPPI_CURB_EDGE_M 안이면 True.
    @staticmethod
    def _road_edge_zone(planner, radii=None):
        """near_edge(xy[, r]) -> bool array: within r (default MPPI_CURB_EDGE_M) of the map's
        road edge. One distance transform per leg (~0.5 s); r must be one of `radii`."""
        from scipy import ndimage
        g, res = planner._grid, planner._res
        road, unk = g == 0, g < 0
        edge = road & ndimage.binary_dilation(unk)
        dist = ndimage.distance_transform_edt(~edge) * res
        zones = {r: dist <= r for r in set(radii or ()) | {MPPI_CURB_EDGE_M}}
        ox, oy = planner._ox, planner._oy

        def near_edge(xy, r=MPPI_CURB_EDGE_M):
            zone = zones[r]
            j = np.floor((xy[:, 0] - ox) / res).astype(np.int64)
            i = np.floor((xy[:, 1] - oy) / res).astype(np.int64)
            ok = (i >= 0) & (i < zone.shape[0]) & (j >= 0) & (j < zone.shape[1])
            out = np.zeros(len(xy), bool)
            out[ok] = zone[i[ok], j[ok]]
            return out
        return near_edge

    def _plan_obstacles(self, goal, exclude_m, lidar_max_age_scans=None, near_xy=None,
                        near_m=0.0, stab=None):
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
        lidar_discs = (list(self._lidar_layer.discs(lidar_max_age_scans, near_xy, near_m))
                       if self._lidar_layer is not None and self._occ is None else [])
        lidar_discs = obstacles_excluding_goal(lidar_discs, goal, exclude_m=exclude_m)
        if stab is not None:
            lidar_discs = stab.update(lidar_discs)
        memory = obstacles_excluding_goal(self._obstacles, goal, exclude_m=BLOCKAGE_RADIUS_M)
        return lidar_discs + memory

    def _plan_circles(self, goal, exclude_m):
        """Obstacles as one circle per object -- for the waypoint-circle reroute (relocate()).

        Uses `clusters()` (one circle per connected blob) instead of `discs()` (one circle
        per cell). That is what makes a narrower goal exemption possible: `_plan_obstacles`
        above has to drop *everything* within `exclude_m` of the goal, because a lidar disc
        carries no identity -- there is no way to tell "this cell is the target object" from
        "this cell is the post it leans against" without also erasing the post. A cluster
        does carry identity (it is already one physical blob), so only the cluster nearest
        the goal needs exempting -- see `nearest_only_exclude`'s own docstring for the one
        case this still cannot separate (target and landmark fused into a single blob).
        """
        circles = list(self._lidar_layer.clusters()) if self._lidar_layer is not None else []
        circles = nearest_only_exclude(circles, goal, search_m=exclude_m)
        memory = obstacles_excluding_goal(self._obstacles, goal, exclude_m=BLOCKAGE_RADIUS_M)
        return circles + memory

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

    # [팀 신규 추가 2026-10-07] MPPI_SPIN_STALL_* 위 주석 참고.
    def _spin_stall_escape(self, field, pose):
        """Back straight up MPPI_SPIN_ESCAPE_M; if the chassis does not move, crab sideways
        MPPI_SPIN_ESCAPE_M towards the side whose cost-to-go is lower."""
        c = self.client
        dur = MPPI_SPIN_ESCAPE_M / MPPI_SPIN_ESCAPE_V
        self._reverse_escape(duration_s=dur, speed=MPPI_SPIN_ESCAPE_V)
        self._halt()
        p = c.get_world_pose() or pose
        moved = math.hypot(p[0] - pose[0], p[1] - pose[1])
        if moved >= 0.1:
            log.info("[Stage2] spin stall escape: reversed %.2f m", moved)
            return
        x, y, yaw = p
        lx, ly = -math.sin(yaw), math.cos(yaw)             # left unit vector
        d = MPPI_SPIN_ESCAPE_M + 0.1
        v_left = field.value(x + d * lx, y + d * ly)
        v_right = field.value(x - d * lx, y - d * ly)
        side = 1.0 if v_left <= v_right else -1.0
        t_end = self._clock.elapsed + dur
        while c.is_running and self._clock.tick() < t_end:
            c.send_cmd_vel(linear_x=0.0, linear_y=side * MPPI_SPIN_ESCAPE_V, angular_z=0.0)
            c.sleep(STAGE2_NAV_DT)
        self._halt()
        q = c.get_world_pose() or p
        log.info("[Stage2] spin stall escape: reverse moved %.2f m -> crab %s (V left %.1f, "
                 "right %.1f) moved %.2f m", moved, "left" if side > 0 else "right", v_left,
                 v_right, math.hypot(q[0] - p[0], q[1] - p[1]))

    def _reverse_escape(self, duration_s=1.5, speed=0.2):
        """Back straight up briefly -- recovery for a chassis physically wedged in front.

        Marking blockages and replanning assume the chassis can still execute commands;
        a wedged chassis cannot (measured live: 75s of linear=0.8 with a frozen pose).
        Reversing is the one motion a front-wedged chassis can still make, and the space
        just behind is the robot's own freshly traversed track -- the best-known-clear
        direction available without any sensing. Deliberately short: unjam, stop, and let
        the normal plan loop take over from the new pose.
        """
        c = self.client
        t_end = self._clock.elapsed + duration_s
        while c.is_running and self._clock.tick() < t_end:
            c.send_cmd_vel(linear_x=-speed, angular_z=0.0)
            c.sleep(STAGE2_NAV_DT)
        c.send_cmd_vel(0.0, 0.0)

    # [팀 신규 추가] 고정 "0 명령 x 5회 + 대기"(0.25~0.5s, 접근 직후엔 두 번 겹쳐 0.75s)를
    # 대체. 0 을 계속 보내면서 실제로 멈춘 게 pose 로 확인되는 즉시 돌아간다.
    def _halt(self, max_s=0.5, min_s=0.1, still_m=0.003, still_rad=0.01):
        """Command zero until the chassis is confirmed stopped, or ``max_s`` elapses.

        The chassis driver keeps applying the last cmd_vel, so a stop must be commanded,
        and residual speed must be gone before the arm commits to a grasp point. "Stopped"
        = pose moved <= ``still_m`` and turned <= ``still_rad`` over 0.1s (~3 cm/s). Zero is
        sent for at least ``min_s`` (more than one publish, in case one is lost), and
        ``max_s`` equals the longest old fixed wait, so this is never slower than before.
        Returns True if a stop was confirmed.
        """
        c = self.client
        start = self._clock.elapsed
        ref = None   # (elapsed, pose) the next stillness comparison is made against
        while c.is_running:
            c.send_cmd_vel(0.0, 0.0)
            c.sleep(0.05)
            t = self._clock.tick() - start
            if t >= max_s:
                break
            pose = c.get_world_pose()
            if pose is None:
                continue
            if ref is None:
                ref = (t, pose)
                continue
            if t - ref[0] < 0.1 - 1e-6:
                continue
            moved = math.hypot(pose[0] - ref[1][0], pose[1] - ref[1][1])
            dyaw = abs(math.atan2(math.sin(pose[2] - ref[1][2]), math.cos(pose[2] - ref[1][2])))
            if t >= min_s and moved <= still_m and dyaw <= still_rad:
                return True
            ref = (t, pose)
        c.send_cmd_vel(0.0, 0.0)
        return False

    # [팀 신규 추가] 속도 비례 반응형 제동거리 -- REACT_T_REACT_S 위 주석 참고.
    @staticmethod
    def _react_stop_dist(speed):
        v = max(0.0, float(speed))
        return max(REACT_STOP_DIST_M,
                   ROBOT_RADIUS_M + REACT_EXTRA_MARGIN_M + v * REACT_T_REACT_S
                   + v * v / (2.0 * REACT_BRAKE_DECEL))

    # [팀 신규 추가] 종방향 가속도 제한 -- STAGE2_ACCEL_MPS2 위 주석 참고.
    @staticmethod
    def _limit_accel(linear, prev_linear, dt=STAGE2_NAV_DT):
        """Rate-limit forward speed change per nav tick (accel / decel separately)."""
        if linear > prev_linear:
            return min(linear, prev_linear + STAGE2_ACCEL_MPS2 * dt)
        return max(linear, prev_linear - STAGE2_DECEL_MPS2 * dt)

    # [스타터킷 골격 + 대폭 확장] 이 메서드가 회피 로직이 실제로 조립되는 곳.
    # 원본(starter kit)에 있던 것: A* 갱신 -> PathFollower 추종 -> follower 없으면
    # GoalSeekController 직접 폴백 -> "8초간 못 움직이면 재계획" 단일 정체 판정.
    # 팀이 새로 엮어 넣은 것(아래 while 루프 안에 순서대로): 라이다 레이어 갱신 +
    # 회전 중 스킵 게이트, depth veto 갱신, blocks_path 감지 시 waypoint_avoid.
    # relocate() 우선 시도 후 실패하면 전체 재계획, 8초 슬라이딩 윈도우 정체 판정
    # (+ 같은 지점 연속 정체 시 _reverse_escape), 반응형 브레이크(danger_ahead)와
    # 그 BRAKE_STALL_S 에스컬레이션. snap_m 파라미터화도 팀 추가.
    def _navigate_to(self, goal, deadline, tol=GOAL_TOLERANCE,
                     inflation_m=STAGE2_INFLATION_M, exclude_m=0.6, cruise=False,
                     snap_m=None, avoid_xy=None, offroad_r=DEFAULT_HINT_RADIUS_M):
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
        if LOCAL_PLANNER == "mppi":
            return self._navigate_mppi(goal, deadline, tol=tol, inflation_m=inflation_m,
                                       exclude_m=exclude_m, avoid_xy=avoid_xy,
                                       offroad_r=offroad_r)
        c = self.client
        planner = OccupancyGridPlanner()
        follower = None
        fallback_ctrl = GoalSeekController()
        # The controller's class-level reached test (0.7 m) can be looser than this leg's
        # tol (e.g. the tight pick approach at 0.45) -- shadow it per-instance so the
        # fallback actually finishes the leg inside tol instead of declaring victory 0.7 m
        # out (see the follower-done handling below for the measured failure this caused).
        fallback_ctrl.GOAL_TOLERANCE = min(fallback_ctrl.GOAL_TOLERANCE, tol)
        # NOTE: do NOT shadow the follower's controller the same way. When the goal cell is
        # blocked the path ends short of the goal, and the follower's looser 0.7 m "done"
        # is the hysteresis that keeps the fallback in charge once it takes over -- at
        # 0.45 the follower flips done/not-done every few ticks, the two controllers aim
        # in opposite directions, and the chassis spins in place until the stall detector
        # fires (measured offline, rt01: leg1 30s -> 45s).

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
        last_cmd = (0.0, 0.0)     # (linear, angular) sent last tick -- for the lidar rotation gate
        last_stall_xy = None      # where the previous stall fired -- repeat at the same spot = wedged

        while c.is_running and self._clock.tick() < deadline:
            pose = c.get_world_pose()
            if pose is None:
                c.sleep(STAGE2_NAV_DT)
                continue
            x, y, yaw = pose
            tick += 1
            if self._lidar_acc is not None:
                # slice deskew + IMU leveling history (lidar_accumulator.py)
                self._lidar_acc.record_pose(c.now_s(), pose, c.get_imu())
            # Sense before hitting anything. The stall path below only reacts after contact.
            new_cells = 0
            if tick % LIDAR_UPDATE_EVERY_TICKS == 0:
                # Rotation gate: the layer's sweep-wide single-percentile ground estimate
                # misfires per-azimuth while the chassis spins in place, so rotation is a
                # ghost generator (its own MAX_RANGE_M note; measured live 2026-09-01
                # 06:00-06:02: blocked_area exploded to 56 m2 during a replanning spin,
                # relocate failed on every side and A* went unplannable while the chassis
                # sat wedged). Skip *accumulating* sweeps while the last command was a
                # pure spin -- nothing is being approached during a spin, so the safety
                # cost is nil. The veto below only clears cells, so it always runs.
                spinning = last_cmd[0] < 0.05 and abs(last_cmd[1]) > 0.3
                if not spinning:
                    new_cells = self._update_lidar_layer(pose)
                elif self._lidar_acc is not None:
                    self._lidar_acc.take_merged(pose)   # gate holds: spin-time slices dropped
                self._update_depth_veto(pose)

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
                    # Try bending just the crossed segment(s) around the obstacle's circle
                    # first -- no grid refresh / A* / follower swap, so it lands this same
                    # tick instead of waiting a full replan cycle (see waypoint_avoid.py).
                    # margin_m mirrors this leg's own A* erosion (inflation_m) plus a touch
                    # of slack for the follower's corner-cutting P-control.
                    circles = self._plan_circles(goal, exclude_m)
                    rerouted = relocate(
                        [(x, y)] + remaining, circles,
                        margin_m=inflation_m + RELOCATE_MARGIN_M,
                        is_free=planner.is_free if planner.ready else None)
                    if rerouted is not None:
                        follower.path[follower.idx:] = rerouted[1:]
                        log.info("[Stage2] lidar: %d new cell(s) -> rerouted %d waypoint(s) "
                                 "around obstacle circle(s) (%d blocked, %.1f m2)", new_cells,
                                 len(rerouted) - 1,
                                 self._lidar_layer.stats()["blocked_cells"],
                                 self._lidar_layer.stats()["blocked_area_m2"])
                    else:
                        log.info("[Stage2] lidar: %d new cell(s) -- circle reroute blocked "
                                 "on both sides -> full replan (%d blocked, %.1f m2)",
                                 new_cells, self._lidar_layer.stats()["blocked_cells"],
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
                    # A SECOND stall at (nearly) the same spot means marking blockages and
                    # replanning is not helping -- the chassis is physically wedged, where
                    # no forward/turn command has any effect (measured live, rt02
                    # 2026-09-01 06:01: pose frozen for 75s with linear=0.8 commanded and
                    # even a w=1.2 spin frozen -- jammed on something below the lidar's
                    # ground band). Reverse first, then let the normal replan continue.
                    if (last_stall_xy is not None
                            and math.hypot(x - last_stall_xy[0], y - last_stall_xy[1]) < 0.15):
                        log.info("[Stage2] consecutive stall at the same spot -- chassis "
                                 "wedged, reversing to escape")
                        self._reverse_escape()
                    last_stall_xy = (x, y)
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
            # Stop distance scales with the speed the chassis was last told to hold (the
            # accel limit keeps that close to its real speed) -- see REACT_T_REACT_S.
            stop_dist = self._react_stop_dist(last_cmd[0])
            danger = self._lidar_layer is not None and self._lidar_layer.danger_ahead(
                pose, stop_dist, REACT_STOP_CONE_DEG,
                goal=goal, goal_exclude_m=exclude_m)
            blocked_by_brake = danger and linear > 0.0
            if blocked_by_brake:
                if not braking:
                    log.info("[Stage2] lidar: obstacle within %.2fm ahead (v=%.2f) -> braking",
                             stop_dist, last_cmd[0])
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
                # Accel limit (not on a brake -- an emergency stop is never slowed down).
                # Re-cap angular afterwards: a speed held lower than the controller asked
                # for permits less turn (cap_angular), and one held higher while slowing
                # must not exceed the lateral-accel limit (tip-over).
                limited = self._limit_accel(linear, last_cmd[0])
                if limited != linear:
                    linear = limited
                    angular = cap_angular(linear, angular)
                    if linear > 1e-3:
                        w_lat = GoalSeekController.MAX_LAT_ACCEL / linear
                        angular = max(-w_lat, min(w_lat, angular))
            braking = danger

            c.send_cmd_vel(linear_x=linear, angular_z=angular)
            last_cmd = (linear, angular)

            # Progress log (throttled to ~1s SIM time) -- for tracking pose/dist/v/w
            now = c.now_s()
            if now - last_log_t >= 1.0:
                log.info("[STAGE2] pose=(%.2f, %.2f, yaw=%.0fdeg) dist=%.2fm%s v=%.2f w=%+.2f",
                         x, y, math.degrees(yaw), dist, wp_str, linear, angular)
                last_log_t = now

            c.sleep(STAGE2_NAV_DT)

        return reached_goal

    # [팀 신규 추가] 지역 내비게이션 함수(nav_field.py) + MPPI(mppi.py) 주행 루프.
    # _navigate_to 의 감지/복구 장치(라이다 레이어, depth veto, 정체 판정 + 정체 기억 +
    # 후진 탈출, 반응형 브레이크)는 그대로 쓰고, 경로 추종(PathFollower + P 제어)과 원호
    # 우회(relocate)를 "cost-to-go 장 위의 MPPI" 하나로 바꾼다.
    def _navigate_mppi(self, goal, deadline, tol=GOAL_TOLERANCE, inflation_m=STAGE2_INFLATION_M,
                       exclude_m=0.6, avoid_xy=None, offroad_r=DEFAULT_HINT_RADIUS_M):
        c = self.client
        planner = OccupancyGridPlanner()
        tight = tol < GOAL_TOLERANCE                  # pick approach leg
        inscribed = 0.22 if tight else 0.30
        clear_m = min(float(exclude_m), MPPI_GOAL_CLEAR_M)
        # off-road (map unknown = beyond the kerb) is lethal except within offroad_r of the
        # goal / pick object -- see nav_field.py docstring
        offroad_ok = [(goal[0], goal[1], float(offroad_r))]
        if avoid_xy and math.hypot(avoid_xy[0] - goal[0], avoid_xy[1] - goal[1]) < 1e-3:
            # the goal IS the object (leg 1 stops tol short of it): a keep disc there would
            # make the goal lethal and snap it to an arbitrary side -- the tol stop already
            # keeps the chassis off the object
            avoid_xy = None
        if avoid_xy:
            offroad_ok.append((avoid_xy[0], avoid_xy[1], float(offroad_r)))
        field, ctrl = None, None
        field_dirty, built_at = True, -1e9
        hold_stage, hold_until = 0, -1e9      # stage hold -- see MPPI_STAGE_HOLD_S
        tilt_last = None                      # (x, y, tilt deg) of the last tilt-guard trigger
        last_resort = False                   # current build's stage is last-resort
        edge_zone = None                      # map road-edge test for curb_layer.hazard_xy
        hazard_all = np.zeros((0, 2))         # kerb/ledge lines from the last build (slow zone)
        slow = False                          # inside the kerb / road-edge slow zone
        disc_stab = {}                        # stage name -> DiscStabilizer
        frozen = {}                           # stage name -> disc set frozen for the active hold
        spin_t0 = None                        # (t, yaw) when the current pure spin began
        yaw_acc = 0.0                         # yaw turned since spin_t0 (unwrapped)
        yaw_prev = None
        pos_history = deque()        # (t, x, y, yaw, commanded v)
        pin_history = deque()        # same, over MPPI_PIN_WINDOW_S
        v_history = deque()          # (t, V(robot)) -- tight legs: no V progress near goal = arrived
        last_cmd = (0.0, 0.0)
        all_hit = False
        last_stall_xy = None
        braking = False
        last_log_t = 0.0
        tick = 0
        while c.is_running and self._clock.tick() < deadline:
            pose = c.get_world_pose()
            if pose is None:
                c.sleep(STAGE2_NAV_DT)
                continue
            x, y, yaw = pose
            tick += 1
            if self._lidar_acc is not None:
                self._lidar_acc.record_pose(c.now_s(), pose, c.get_imu())
            if tick % LIDAR_UPDATE_EVERY_TICKS == 0:
                spinning = last_cmd[0] < 0.05 and abs(last_cmd[1]) > 0.3
                if not spinning:
                    if self._update_lidar_layer(pose):
                        field_dirty = True
                    if self._curb is not None:
                        field_dirty = True      # kerb / flat cells change every sweep
                elif self._lidar_acc is not None:
                    self._lidar_acc.take_merged(pose)
                self._update_depth_veto(pose)

            dist = math.hypot(goal[0] - x, goal[1] - y)
            if dist < tol:
                log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                return True
            if not planner.ready:
                occ = c.get_occupancy_map()
                if occ is None:
                    c.send_cmd_vel(0.0, 0.0)
                    c.sleep(STAGE2_NAV_DT)
                    continue
                planner.update_from_msg(occ)

            now_t = c.now_s()
            if field is None or (field_dirty and now_t - built_at >= NAVFIELD_REBUILD_S):
                t0 = time.monotonic()
                # kerbs / steps are obstacles in every stage; sensed-flat off-road is passable
                step_discs, flat_xy, hazard_xy = (), None, None
                if self._curb is not None:
                    if edge_zone is None:
                        edge_zone = self._road_edge_zone(
                            planner, radii=(MPPI_SLOW_EDGE_M, MPPI_SLOW_EDGE_OFF_M))
                    hz = self._curb.hazard_xy(near_edge=edge_zone)
                    hazard_all = hz
                    self._curb_edge_zone = edge_zone
                    if len(hz):
                        keep_ = ((np.hypot(hz[:, 0] - goal[0], hz[:, 1] - goal[1]) >= MPPI_STEP_GOAL_CLEAR_M)
                                 & (np.hypot(hz[:, 0] - x, hz[:, 1] - y) >= MPPI_STEP_NEAR_M))
                        hazard_xy = hz[keep_]

                def build(discs, ok, r=inscribed):
                    return NavField(inscribed_m=r).build(
                        planner._grid, (planner._ox, planner._oy), planner._res, (x, y), goal,
                        discs=discs, goal_clear_m=clear_m,
                        keep_discs=([(avoid_xy[0], avoid_xy[1], MPPI_OBJECT_RADIUS_M)]
                                    if avoid_xy else ()),
                        free_xy=(x, y), offroad_ok=ok, step_discs=step_discs, flat_xy=flat_xy,
                        edge_band_m=0.0, hazard_xy=hazard_xy, occ_layer=self._occ)

                # goal unreachable -> escalate (see MPPI_DISC_STALE_S / MPPI_STAGE_HOLD_S)
                scans_per_s = STAGE2_NAV_RATE_HZ / LIDAR_UPDATE_EVERY_TICKS
                rt = min(inscribed, MPPI_TIGHT_INSCRIBED_M)
                stages = [  # (name, lidar max age in scans or None = all, radius, off-road ok)
                    ("normal", None, inscribed, offroad_ok),
                    ("tight", None, rt, offroad_ok),
                    ("fresh+near", MPPI_DISC_STALE_S * scans_per_s, rt, offroad_ok),
                    ("near-only", 0, rt, offroad_ok),
                    ("last-resort", 0, rt, None),
                ]
                start = hold_stage if now_t < hold_until else 0
                n_all = None
                for k in range(start, len(stages)):
                    name, max_age, r_k, ok_k = stages[k]
                    stab = disc_stab.setdefault(name, DiscStabilizer())
                    if k == hold_stage and now_t < hold_until and name in frozen:
                        discs = frozen[name]
                    elif max_age is None:
                        discs = self._plan_obstacles(goal, clear_m, stab=stab)
                    else:
                        discs = self._plan_obstacles(goal, clear_m, lidar_max_age_scans=max_age,
                                                     near_xy=(x, y), near_m=MPPI_DISC_NEAR_M,
                                                     stab=stab)
                    if n_all is None:
                        n_all = len(discs)
                    field = build(discs, ok_k, r_k)
                    reach = math.isfinite(field.value(x, y))
                    if k > 0:
                        log.warning("[Stage2] navfield: stage %s (r=%.2f, %d/%d discs%s) "
                                    "V(robot)=%.1f", name, r_k, len(discs), n_all,
                                    ", off-road open" if ok_k is None else "", field.value(x, y))
                    if reach:
                        break
                if k > 0 and (k != hold_stage or now_t >= hold_until):
                    hold_s = (MPPI_LAST_RESORT_HOLD_S
                              if stages[k][0] == "last-resort" and self._curb is not None
                              else MPPI_STAGE_HOLD_S)
                    hold_stage, hold_until = k, now_t + hold_s
                    frozen = {stages[k][0]: discs}
                    log.info("[Stage2] navfield: holding stage %s for %.0fs (%d discs frozen)",
                             stages[k][0], hold_s, len(discs))
                if ctrl is None:
                    ctrl = MPPI(field, steering_track=STEERING_TRACK)
                else:
                    ctrl.set_field(field)
                last_resort = stages[k][0] == "last-resort"
                field_dirty, built_at = False, now_t
                log.info("[Stage2] navfield %dx%d built in %.0f ms (%d discs, occ hard/soft %d/%d, "
                         "%d hazard cells, %d off-road "
                         "cells, ok r=%s) V(robot)=%.1f goal%s=(%.2f, %.2f)",
                         field.w, field.h, 1000 * (time.monotonic() - t0), len(discs),
                         field.occ_hard, field.occ_soft,
                         0 if hazard_xy is None else len(hazard_xy), int(field.offroad.sum()),
                         "/".join("%.1f" % c[2] for c in getattr(field, "offroad_ok_eff", ())) or "-",
                         field.value(x, y),
                         " SNAPPED" if field.goal_snapped else "", field.goal[0], field.goal[1])
            if field.goal_snapped and math.hypot(field.goal[0] - x, field.goal[1] - y) < tol:
                log.info("[STAGE2] reached snapped goal (%.2f, %.2f) -- original %.2fm away",
                         field.goal[0], field.goal[1], dist)
                return True

            rp = imu_roll_pitch(c.get_imu())
            tilt = math.degrees(max(abs(rp[0]), abs(rp[1]))) if rp is not None else 0.0
            repeat = (tilt_last is not None
                      and math.hypot(x - tilt_last[0], y - tilt_last[1]) < TILT_REPEAT_M
                      and tilt < tilt_last[2] + TILT_REPEAT_DEG)
            if tilt > TILT_GUARD_DEG and not repeat:
                log.warning("[Stage2] tilt guard: roll %.1f pitch %.1f deg -> stop, settle%s",
                            math.degrees(rp[0]), math.degrees(rp[1]),
                            "" if tilt_last is None else " (worse / new spot)")
                self._halt(max_s=TILT_SETTLE_S, min_s=TILT_SETTLE_S)
                rp2 = imu_roll_pitch(c.get_imu())
                tilt2 = math.degrees(max(abs(rp2[0]), abs(rp2[1]))) if rp2 is not None else 0.0
                new_spot = (tilt_last is None
                            or math.hypot(x - tilt_last[0], y - tilt_last[1]) >= TILT_REPEAT_M)
                if tilt2 > TILT_GUARD_DEG:
                    log.warning("[Stage2] tilt guard: still %.1f deg after settling -> back off "
                                "slowly", tilt2)
                    self._reverse_escape(speed=TILT_REVERSE_V)
                if new_spot:
                    ahead = (x + 0.6 * math.cos(yaw), y + 0.6 * math.sin(yaw), yaw)
                    self._remember_blockage(ahead, goal)
                tilt_last = (x, y, max(tilt, tilt2))
                field_dirty, built_at = True, -1e9
                ctrl.reset()
                last_cmd = (0.0, 0.0)
                continue
            # slow zone (MPPI_SLOW_*): near a sensed kerb/ledge line or the map's road edge
            if self._curb is not None and edge_zone is not None:
                d_hz = (float(np.min(np.hypot(hazard_all[:, 0] - x, hazard_all[:, 1] - y)))
                        if len(hazard_all) else 1e9)
                here = np.array([[x, y]])
                if slow:
                    slow = (d_hz < MPPI_SLOW_HAZARD_OFF_M
                            or bool(edge_zone(here, MPPI_SLOW_EDGE_OFF_M)[0]))
                else:
                    slow = d_hz < MPPI_SLOW_HAZARD_M or bool(edge_zone(here, MPPI_SLOW_EDGE_M)[0])
                    if slow:
                        log.info("[Stage2] slow zone: kerb/ledge %.1f m away or road edge -> "
                                 "v<=%.1f w<=%.1f", d_hz, MPPI_SLOW_V, MPPI_SLOW_W)
            v_cap = V_MAX_MPPI
            if last_resort:
                v_cap = min(v_cap, MPPI_LAST_RESORT_V)
            if slow:
                v_cap = min(v_cap, MPPI_SLOW_V)
            ctrl.v_max = v_cap
            ctrl.w_max = MPPI_SLOW_W if slow else MPPI_W_MAX
            # roll out from the last command, not odom: odom speed jitters ~0.1 m/s at a
            # standstill, which kept the decel limit from ever reaching a pure spin
            linear, angular = ctrl.step(pose, last_cmd[0], goal_tol=0.8 * tol)
            # Every rollout collides (stop included -- rolled out from the last command, a
            # stop from 1.1 m/s still travels ~0.6 m): the weighted update is noise, and in
            # r4 that noise was v=1.1 into the obstacle. Stop; next tick rolls out from rest,
            # where stopping/spinning/reversing are collision-free again.
            hit_frac = ctrl.last_info.get("collide_frac", 0.0)
            if hit_frac >= MPPI_ALL_HIT_FRAC:
                if not all_hit:
                    log.info("[Stage2] mppi: %.0f%% of rollouts collide -> stopping",
                             100 * hit_frac)
                linear, angular = 0.0, 0.0
                ctrl.reset()
            all_hit = hit_frac >= MPPI_ALL_HIT_FRAC

            if tight and dist < MPPI_ACCEPT_NEAR_M:
                v_history.append((now_t, field.value(x, y)))
                while len(v_history) >= 2 and now_t - v_history[1][0] >= STAGE2_STUCK_TIME_S:
                    v_history.popleft()
                if (now_t - v_history[0][0] >= STAGE2_STUCK_TIME_S
                        and v_history[0][1] - v_history[-1][1] < 0.1):
                    log.info("[STAGE2] no cost-to-go progress for %.0fs %.2fm from the pick "
                             "standoff -- accepting", STAGE2_STUCK_TIME_S, dist)
                    return True
            else:
                v_history.clear()
            # Two stall rules over (t, x, y, yaw, commanded v) windows:
            # - pinned (MPPI_PIN_WINDOW_S): commanding forward, not moving -> in contact with
            #   something the field does not hold. Back off at once, then mark it.
            # - stall (STAGE2_STUCK_TIME_S): same rule as _navigate_to, except that turning
            #   in place is progress (MPPI spins to face downhill) -- but only while a spin
            #   is what was commanded; a yaw twisting while pushing forward is not progress.
            stalled = None
            for hist, span in ((pin_history, MPPI_PIN_WINDOW_S),
                               (pos_history, STAGE2_STUCK_TIME_S)):
                hist.append((now_t, x, y, yaw, last_cmd[0]))
                while len(hist) >= 2 and now_t - hist[1][0] >= span:
                    hist.popleft()
            if now_t - pin_history[0][0] >= MPPI_PIN_WINDOW_S:
                moved = math.hypot(x - pin_history[0][1], y - pin_history[0][2])
                cmd_v = sum(h[4] for h in pin_history) / len(pin_history)
                if cmd_v >= MPPI_PIN_CMD_V and moved < MPPI_PIN_MOVE_M:
                    stalled = "pinned"
                    log.info("[Stage2] pinned: v_cmd=%.2f but moved %.3fm over %.1fs",
                             cmd_v, moved, now_t - pin_history[0][0])
            if stalled is None and now_t - pos_history[0][0] >= STAGE2_STUCK_TIME_S:
                moved = math.hypot(x - pos_history[0][1], y - pos_history[0][2])
                turned = abs(math.atan2(math.sin(yaw - pos_history[0][3]),
                                        math.cos(yaw - pos_history[0][3])))
                spin_cmd = (sum(abs(h[4]) for h in pos_history) / len(pos_history)
                            < MPPI_SPIN_CMD_V)
                if moved <= STAGE2_STUCK_MOVE_M and not (spin_cmd and turned >= 0.5):
                    stalled = "stall"
                    log.info("[Stage2] stall detected (%.3fm over %.1fs) -> replanning",
                             moved, now_t - pos_history[0][0])
            if stalled is not None:
                if tight and dist < MPPI_ACCEPT_NEAR_M:
                    log.info("[STAGE2] stalled %.2fm from the pick standoff -- accepting "
                             "(align + straight approach take it from here)", dist)
                    return True
                if stalled == "pinned" or (
                        last_stall_xy is not None
                        and math.hypot(x - last_stall_xy[0], y - last_stall_xy[1]) < 0.15):
                    log.info("[Stage2] chassis wedged (%s), reversing to escape", stalled)
                    self._reverse_escape()
                last_stall_xy = (x, y)
                # mark from the contact pose (before the reverse), so the disc sits on
                # what was hit; NavField(free_xy=) keeps it off the robot's new spot
                self._remember_blockage((x, y, yaw), goal)
                field_dirty, built_at = True, -1e9
                ctrl.reset()
                last_cmd = (0.0, 0.0)
                pin_history.clear()
                pos_history.clear()
                c.sleep(STAGE2_NAV_DT)
                continue

            # spin stall (MPPI_SPIN_STALL_*): a pure spin that does not turn the chassis
            if yaw_prev is not None:
                yaw_acc += math.atan2(math.sin(yaw - yaw_prev), math.cos(yaw - yaw_prev))
            yaw_prev = yaw
            if abs(last_cmd[0]) < MPPI_SPIN_CMD_V and abs(last_cmd[1]) >= MPPI_SPIN_STALL_W:
                if spin_t0 is None:
                    spin_t0, yaw_acc = now_t, 0.0
                elif (now_t - spin_t0 >= MPPI_SPIN_STALL_S
                      and abs(yaw_acc) < math.radians(MPPI_SPIN_STALL_DEG)):
                    log.warning("[Stage2] spin stall: w=%+.2f for %.1fs turned %.1f deg -> "
                                "escape", last_cmd[1], now_t - spin_t0, math.degrees(yaw_acc))
                    self._spin_stall_escape(field, pose)
                    spin_t0, yaw_prev = None, None
                    field_dirty, built_at = True, -1e9
                    ctrl.reset()
                    last_cmd = (0.0, 0.0)
                    pin_history.clear()
                    pos_history.clear()
                    c.sleep(STAGE2_NAV_DT)
                    continue
            else:
                spin_t0 = None

            # reactive brake (kept until the CBF safety filter replaces it)
            stop_dist = self._react_stop_dist(max(last_cmd[0], 0.0))
            brake_layer = self._occ if self._occ is not None else self._lidar_layer
            danger = brake_layer is not None and brake_layer.danger_ahead(
                pose, stop_dist, REACT_STOP_CONE_DEG, goal=goal, goal_exclude_m=clear_m)
            if danger and linear > 0.0:
                if not braking:
                    log.info("[Stage2] lidar: obstacle within %.2fm ahead (v=%.2f) -> braking",
                             stop_dist, last_cmd[0])
                linear = 0.0
            braking = danger
            angular = cap_angular(linear, angular)
            c.send_cmd_vel(linear_x=linear, angular_z=angular)
            last_cmd = (linear, angular)

            if now_t - last_log_t >= 1.0:
                info = ctrl.last_info
                log.info("[STAGE2] mppi pose=(%.2f, %.2f, yaw=%.0fdeg) dist=%.2fm V=%.1f "
                         "v=%.2f w=%+.2f collide=%.0f%%", x, y, math.degrees(yaw), dist,
                         field.value(x, y), linear, angular, 100 * info.get("collide_frac", 0))
                last_log_t = now_t
            c.sleep(STAGE2_NAV_DT)
        c.send_cmd_vel(0.0, 0.0)
        return False

    # -- notifications --

    # [스타터킷 골격 + 확장] 로그 출력 두 줄은 원본과 동일, JSONL 파일 기록
    # (self._append_score_log 호출)만 팀이 추가.
    def _on_score(self, score):
        if score.is_final:
            log.info("[FINAL] %s", score.scores)
        else:
            log.info("[Score] round %s: total=%s", score.round, score.total)
        self._append_score_log(score)

    # [팀 신규 추가] starter kit에 없던 메서드.
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

    # [팀 신규 추가] starter kit에 없던 메서드.
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

    # [스타터킷 원본] 아래 두 메서드 모두 로직 변경 없음.
    def _on_state_change(self, old, new):
        log.info("[STATE] %s -> %s", old, new)

    def _on_time_expired(self, which):
        log.info("[TIME] expired: %s", which)

    # -- run --

    # [팀 신규 추가] starter kit에 없던 메서드 -- tf2 리스너(world->arm_base_link,
    # 카메라/라이다 마운트 변환에 전부 필요) 셋업.
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

    # [스타터킷 골격 + 확장] connect()/run() 호출은 원본과 동일, self._setup_tf()
    # 호출 한 줄만 팀이 추가(connect() 뒤여야 rclpy.init()이 끝나 있어서 안전).
    def run(self):
        if not self.client.connect():
            raise SystemExit("Registration failed -- check the runtime and the token.")
        self._setup_tf()
        self.client.run()


# [스타터킷 원본] 로직 변경 없음.
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    DemoParticipant().run()


if __name__ == "__main__":
    main()
