"""MARC 2026 SDK demo participant -- Stage 1 grounding via taejeong/ (text parsing +
pure geometry, no scenario hardcoding), kept as a separate entry point from
participant_app.py (real_agent_vla / inline heuristics) and
participant_app_agent_vla.py (Grounding DINO + Qwen2.5-VL) so all three Stage 1
implementations can be run and tested independently without touching one another.

Stage 2 occupancy A* driving (cmd_vel) is identical to the other two entry points.
The protocol/ROS wiring (register, subscribe/publish, world_pose, occupancy) is all
handled by ``MARCClient``, so only the participant logic (VLA + nav) is connected
through handlers.

Run:
    source /opt/ros/humble/setup.bash
    export MARC_TEAM_ID=u1
    export MARC_TOKEN=<token>            # your assigned token (required; issued by the organizers)
    MARC_ENTRYPOINT=participant_app_taejeong.py ./launch.sh u1

Per-round scores are logged to ./results/scores.jsonl (joined back to the
voice_command that produced them) -- see _append_score_log. This is what the
chungmu validation run reads to check the pipeline against reliable GT.

Precondition: the simulation platform must be running first.
"""

import json
import logging
import math
import os

from marc_sdk import MARCClient

from taejeong_agent_vla import TaejeongVLAGrounding
from mock_agent_navigation import (
    OccupancyGridPlanner, PathFollower, GoalSeekController,
    load_obstacles, obstacles_excluding_goal, load_mock_data,
)
# Closed-loop placo pick from the manipulation kit (arm_pick.py): grasp_yaw wrist-roll +
# DLS servo, targeting the object position. Requires placo (installed in the demo image).
from arm_pick import run_pick_sequence

log = logging.getLogger("participant_app")

# -- Stage 2 navigation parameters (identical to participant_app.py / _agent_vla.py) --
STAGE2_NAV_TIMEOUT_S = 1800.0
STAGE2_NAV_RATE_HZ = 10.0
STAGE2_NAV_DT = 1.0 / STAGE2_NAV_RATE_HZ
STAGE2_INFLATION_M = 0.25
STAGE2_STUCK_TIME_S = 8.0
STAGE2_STUCK_MOVE_M = 0.05
GOAL_TOLERANCE = 0.7

TUMBLER_STANDOFF_M = 0.58
ARM_BASE_FWD = 0.18
GRASP_FWD_BACKOFF = -0.01

_DELIVERY_COORD = load_mock_data().get("delivery_coord") or [-55.29138, 142.09763, 16.47326]
DELIVERY_XY = (_DELIVERY_COORD[0], _DELIVERY_COORD[1])


class DemoParticipant:
    """SDK-based demo participant -- connects the taejeong VLA/nav logic through handlers."""

    def __init__(self):
        self.client = MARCClient.from_env()
        self.vla = TaejeongVLAGrounding(self.client)   # YOLO load happens here, once, before connect()
        # Per-round detail-score JSONL log (see _on_score). Round -> voice_command so each
        # score line can be joined back to the question that produced it.
        self._round_commands = {}
        self._score_log_path = os.environ.get("MARC_SCORE_LOG", "./results/scores.jsonl")
        self._collect_xy = None
        self._stage2_target_xy = None
        self._pickup_xy = None
        self._occ_logged = False
        self._obstacles = load_obstacles()
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

    def _on_mission(self, mission):
        log.info("[Stage1] round %s/%s: \"%s\"",
                 mission.round, mission.total_rounds, mission.voice_command)
        self._round_commands[mission.round] = mission.voice_command
        result = self.vla.process(mission.voice_command, self.client.list_cctv())
        self.client.submit_grounding(result)

    # -- Stage 2: submit interpretation -> drive --

    def _on_stage2_mission(self, mission):
        log.info("[Stage2] task: \"%s\" (limit %.0fs)",
                 mission.task_description, mission.time_limit)
        op = getattr(mission, "owner_position", None)
        if op and len(op) >= 2:
            self._pickup_xy = (float(op[0]), float(op[1]))
            log.info("[Stage2] owner_position(runtime-provided)=%s", self._pickup_xy)
        result = self.vla.process_stage2(mission.task_description)
        self.client.submit_stage2_grounding(result)
        tc = result.target_coord
        if tc and len(tc) >= 2:
            self._stage2_target_xy = (float(tc[0]), float(tc[1]))
        log.info("[Stage2] submitted grounding target=%s, delivery point=%s",
                 self._stage2_target_xy, self._pickup_xy or DELIVERY_XY)

    def _on_stage2_reveal(self, reveal):
        log.info("[Stage2] reveal received -- score=%s, type=%s, center=%s, r=%.1f",
                 reveal.grounding_score, reveal.target_type,
                 reveal.hint_center, reveal.hint_radius)
        if reveal.hint_center and len(reveal.hint_center) >= 2:
            self._collect_xy = (float(reveal.hint_center[0]), float(reveal.hint_center[1]))

    def _drive(self):
        c = self.client
        deadline = c.now_s() + STAGE2_NAV_TIMEOUT_S
        collect_xy = self._wait_for_collect_goal(timeout=5.0)
        pickup_xy = self._pickup_xy if self._pickup_xy is not None else DELIVERY_XY
        log.info("[Stage2] navigation start -- collect=%s deliver=%s (timeout %.0fs)",
                 collect_xy, pickup_xy, STAGE2_NAV_TIMEOUT_S)

        self._stow_arm()

        picked = False
        approach = self._standoff_goal(collect_xy, TUMBLER_STANDOFF_M)
        log.info("[Stage2] leg1 -> object %s front standoff %s",
                 tuple(round(v, 2) for v in collect_xy), tuple(round(v, 2) for v in approach))
        self._navigate_to(approach, deadline, tol=0.45, inflation_m=0.10, exclude_m=1.0)
        if not c.is_running:
            return
        self._approach_straight(collect_xy, deadline, target_fwd=0.40)
        for _ in range(5):
            c.send_cmd_vel(0.0, 0.0)
            c.sleep(0.1)
        log.info("[Stage2] standoff stop -- ready to pick")
        try:
            grasp_xy = self._object_arm_xy(collect_xy)
            log.info("[Stage2] object pick sequence (object-position grasp=%s)",
                     tuple(round(v, 3) for v in grasp_xy) if grasp_xy else None)
            held = run_pick_sequence(c, log, grasp_xy)
            c.send_cmd_vel(0.0, 0.0)
            picked = True if held is None else bool(held)
            log.info("[Stage2] pick %s (grip held=%s)", "ok" if picked else "failed", held)
        except Exception as e:
            log.warning("[Stage2] pick failed: %s", e)

        if not picked:
            self._stow_arm()

        log.info("[Stage2] leg2 -> deliver to owner zone %s", pickup_xy)
        reached = self._navigate_to(pickup_xy, deadline)

        c.stop()
        log.info("[Stage2] driving finished (delivery_reached=%s) -> task_complete", reached)
        c.task_complete()

    def _stow_arm(self):
        try:
            from arm_pick import HOME, GRIPPER_OPEN, _make_js
            self.client.send_arm_command(_make_js(HOME, GRIPPER_OPEN))
            log.info("[Stage2] arm stow -> HOME (lower CoG, tip-over prevention)")
        except Exception as e:
            log.warning("[Stage2] arm stow failed: %s", e)

    def _standoff_goal(self, dest, standoff):
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
        pose = self.client.get_world_pose()
        if pose is None:
            return None
        x, y, yaw = pose
        dx, dy = obj_xy[0] - x, obj_xy[1] - y
        cc, ss = math.cos(yaw), math.sin(yaw)
        fwd = dx * cc + dy * ss
        lat = -dx * ss + dy * cc
        return (fwd - ARM_BASE_FWD - GRASP_FWD_BACKOFF, lat)

    def _approach_straight(self, obj_xy, deadline, target_fwd=0.40, max_t=20.0):
        c = self.client
        t_end = min(deadline, c.now_s() + max_t)
        while c.is_running and c.now_s() < t_end:
            gxy = self._object_arm_xy(obj_xy)
            if gxy is None:
                break
            fwd, lat = gxy
            if fwd <= target_fwd + 0.02:
                break
            ang = max(-0.4, min(0.4, 1.5 * lat))
            lin = 0.12 if abs(lat) < 0.15 else 0.06
            c.send_cmd_vel(linear_x=lin, angular_z=ang)
            c.sleep(STAGE2_NAV_DT)
        for _ in range(5):
            c.send_cmd_vel(0.0, 0.0)
            c.sleep(0.05)
        gxy = self._object_arm_xy(obj_xy)
        log.info("[Stage2] straight approach done -- object arm coords %s",
                 tuple(round(v, 3) for v in gxy) if gxy else None)

    def _wait_for_collect_goal(self, timeout: float = 5.0):
        c = self.client
        deadline = c.now_s() + timeout
        while c.now_s() < deadline:
            if self._collect_xy is not None:
                return self._collect_xy
            c.sleep(0.1)
        log.warning("[Stage2] reveal not received (%.0fs) -- falling back to grounding target %s",
                    timeout, self._stage2_target_xy)
        return self._stage2_target_xy

    def _navigate_to(self, goal, deadline, tol=GOAL_TOLERANCE,
                     inflation_m=STAGE2_INFLATION_M, exclude_m=0.6, cruise=False):
        c = self.client
        planner = OccupancyGridPlanner()
        follower = None
        fallback_ctrl = GoalSeekController()
        leg_obstacles = obstacles_excluding_goal(self._obstacles, goal, exclude_m=exclude_m)

        last_xy = None
        last_progress_t = c.now_s()
        last_log_t = 0.0
        reached_goal = False

        while c.is_running and c.now_s() < deadline:
            pose = c.get_world_pose()
            if pose is None:
                c.sleep(STAGE2_NAV_DT)
                continue
            x, y, yaw = pose
            dist = math.hypot(goal[0] - x, goal[1] - y)
            if dist < tol:
                reached_goal = True
                log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                break

            occ = c.get_occupancy_map()
            if occ is not None:
                planner.update_from_msg(occ)
                if not self._occ_logged:
                    self._occ_logged = True
                    log.info("[Stage2] occupancy received: %dx%d res=%.3f origin=(%.1f, %.1f)",
                             occ.info.width, occ.info.height, occ.info.resolution,
                             occ.info.origin.position.x, occ.info.origin.position.y)

            need_replan = follower is None
            if last_xy is not None:
                if math.hypot(x - last_xy[0], y - last_xy[1]) > STAGE2_STUCK_MOVE_M:
                    last_progress_t = c.now_s()
                elif c.now_s() - last_progress_t > STAGE2_STUCK_TIME_S:
                    log.info("[Stage2] stall detected -> replanning")
                    need_replan = True
                    last_progress_t = c.now_s()
            last_xy = (x, y)

            if need_replan:
                if not planner.ready:
                    if not self._occ_logged:
                        log.warning("[Stage2] occupancy not received -> straight-line fallback")
                else:
                    path = planner.plan((x, y), goal, inflation_m=inflation_m,
                                        snap_m=8.0, extra_obstacles=leg_obstacles)
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

            wp_str = ""
            if follower is not None:
                linear, angular, done, info = follower.compute(pose)
                wp_str = " wp %d/%d=(%.1f, %.1f)" % (
                    info["idx"], info["total"], info["waypoint"][0], info["waypoint"][1])
                if done:
                    reached_goal = True
                    log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                    break
            else:
                linear, angular, reached = fallback_ctrl.compute(pose, goal, cruise=cruise)
                if reached:
                    reached_goal = True
                    log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                    break

            c.send_cmd_vel(linear_x=linear, angular_z=angular)

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

    def _on_state_change(self, old, new):
        log.info("[STATE] %s -> %s", old, new)

    def _on_time_expired(self, which):
        log.info("[TIME] expired: %s", which)

    # -- run --

    def run(self):
        if not self.client.connect():
            raise SystemExit("Registration failed -- check the runtime and the token.")
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
