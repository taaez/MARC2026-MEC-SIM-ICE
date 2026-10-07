"""MPPI (Model Predictive Path Integral) local controller -- CPU / numpy.

Williams et al., "Information Theoretic MPC for Model-Based Reinforcement Learning" (2017);
the same structure as the nav2 MPPI controller, sized for this robot.

매 틱:
  1. 지난 틱의 최적 명령 수열을 한 칸 밀어 warm start (U, 길이 T)
  2. K 개의 잡음 수열 U + eps 를 샘플링
  3. 각 수열을 차체 모델로 굴림. 제약은 **굴리는 동안 강제**한다 (사후 클리핑이 아니라
     샘플 자체가 제약을 만족): 종방향 가/감속, 횡가속 |v*w|, 4WS 조향 유효성 cap_angular
  4. 비용 S_k = sum(장애물 팽창 + 충돌) + 끝점의 남은 비용 V(x_T) + 명령 변화 + 후진
  5. 가중치 w_k = exp(-(S_k - min S) / lambda), U <- U + sum w_k eps_k
  6. U[0] 를 보낸다

V 는 nav_field.NavField (지역 최소점 없는 cost-to-go). 실측: K=2000, T=56 이 numpy 2코어에서
9 ms/iter -- 여기 기본값(K=1024, T=25)은 ~3 ms.
"""

from __future__ import annotations

import math

import numpy as np

DT = 0.1
HORIZON = 25                 # 2.5 s
SAMPLES = 1024
NOISE_V = 0.30               # m/s
NOISE_W = 0.70               # rad/s
LAMBDA = 0.05                # temperature, relative to the normalised cost spread
V_MAX = 1.1
V_MIN = -0.25                # a little reverse, for un-wedging
W_MAX = 1.2
ACCEL = 0.6                  # m/s^2 (same as participant_app.STAGE2_ACCEL_MPS2)
DECEL = 1.0
LAT_ACCEL = 1.0              # |v*w| cap -- tip-over (mock_agent_navigation.MAX_LAT_ACCEL)
STEER_TRACK = 0.5            # overwritten from mock_agent_navigation.STEERING_TRACK

W_COLLISION = 1e4
W_INFLATION = 4.0            # per second spent near obstacles
W_GOAL = 3.0                 # x V(x_T) (V is ~metres of weighted path)
W_PROGRESS = 0.5             # x sum V(x_t) dt -- prefers getting closer early
W_DU = 0.3                   # command smoothness
W_REVERSE = 2.0
W_HEADING = 1.5              # (1 - cos(yaw - downhill)) per second, away from the goal
W_STOP_AT_GOAL = 2.0         # speed^2 penalty once a rollout has reached the goal tolerance
NOISE_CORR = 0.85            # AR(1) time correlation of the sampling noise -- see step()
STRUCT_V = (-0.2, 0.0, 0.4, 0.8, 1.1)
SPIN_SNAP_V = 0.12           # |averaged v| below this -> pure spin (see step())
STRUCT_W = (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0)


class MPPI:
    def __init__(self, field, steering_track=None, rng=None, v_max=V_MAX):
        self.field = field
        self.track = float(steering_track or STEER_TRACK)
        self.rng = rng or np.random.default_rng()
        self.U = np.zeros((HORIZON, 2))
        self.v_max = float(v_max)
        # caller may lower both (e.g. near kerbs); rollouts obey them. w_max caps the turn rate
        # while MOVING only -- a pure spin keeps W_MAX: slow spins barely turn this chassis
        # (rt01_r5_mppi with w_max 0.6: spin commands of ~0.28 rad/s turned 1.4 rad in 40 s and
        # the robot never left the spot), and a spin at v = 0 has no lateral load to cap
        self.w_max = W_MAX
        self.last_info = {}

    def set_field(self, field):
        self.field = field

    def _rollout(self, x0, v0, U):
        """U: (K, T, 2) commanded (v, w). Returns states (K, T, 3), applied (K, T, 2)."""
        K, T, _ = U.shape
        X = np.empty((K, T, 3))
        A = np.empty((K, T, 2))
        x = np.repeat(np.asarray(x0, float)[None], K, axis=0)
        v = np.full(K, float(v0))
        for t in range(T):
            vc = np.clip(U[:, t, 0], V_MIN, self.v_max)
            v = np.clip(vc, v - DECEL * DT, v + ACCEL * DT)
            # a spin command from a crawl stops at once (|v| <= 0.15 is below one decel
            # step and a half) -- otherwise the decel limit keeps the crawl alive and the 4WS
            # turn cap strangles the spin (live rt01: stuck at 0.08 m/s)
            v = np.where((np.abs(vc) < SPIN_SNAP_V) & (np.abs(v) <= 1.5 * DECEL * DT), 0.0, v)
            w = np.clip(U[:, t, 1], -W_MAX, W_MAX)
            av = np.abs(v)
            w = np.where(av > 0.05, np.clip(w, -self.w_max, self.w_max), w)
            # lateral accel cap, then 4WS validity: |w| <= 1.8 v / track unless spinning
            wl = np.where(av > 1e-3, LAT_ACCEL / np.maximum(av, 1e-3), W_MAX)
            w = np.clip(w, -wl, wl)
            ws = np.where(av > 1e-3, 1.8 * av / self.track, W_MAX)
            w = np.where(av > 0.05, np.clip(w, -ws, ws), w)
            x[:, 2] += w * DT
            x[:, 0] += v * np.cos(x[:, 2]) * DT
            x[:, 1] += v * np.sin(x[:, 2]) * DT
            X[:, t] = x
            A[:, t, 0], A[:, t, 1] = v, w
        return X, A

    def step(self, pose, v_now, goal_tol=0.5):
        """One MPPI iteration from pose (x, y, yaw) with current speed. Returns (v, w)."""
        K, T = SAMPLES, HORIZON
        U = np.vstack([self.U[1:], self.U[-1:]])          # warm start
        # Time-correlated noise: with white noise per step the accel limit (0.06 m/s per
        # step) averages it out and every rollout stays near the nominal speed -- measured
        # offline, the robot crept at 0.1-0.3 m/s. AR(1) keeps a sample's deviation going.
        white = self.rng.normal(size=(K, T, 2)) * np.array([NOISE_V, NOISE_W])
        eps = np.empty_like(white)
        eps[:, 0] = white[:, 0]
        a = NOISE_CORR
        for t in range(1, T):
            eps[:, t] = a * eps[:, t - 1] + math.sqrt(1 - a * a) * white[:, t]
        eps[0] = 0.0                                       # keep the nominal in the set
        # Structured candidates (constant v, w) -- cruise, turn, spin, stop, reverse -- so a
        # good maneuver far from the warm start is always in the set.
        sv, sw = np.meshgrid(STRUCT_V, STRUCT_W)
        n = sv.size
        eps[1:1 + n, :, 0] = sv.ravel()[:, None] - U[None, :, 0]
        eps[1:1 + n, :, 1] = sw.ravel()[:, None] - U[None, :, 1]
        # spin in place for k steps, then drive -- the only way this 4WS chassis makes a
        # large heading change (it cannot steer tighter than 1.8 v / track while moving)
        m = 1 + n
        for k in (3, 6, 10, 15, 20):
            for wsp in (-W_MAX, W_MAX):
                for vgo in (0.4, 0.8):
                    if m >= K:
                        break
                    cmd = np.zeros((T, 2))
                    cmd[:k, 1] = wsp
                    cmd[k:, 0] = vgo
                    eps[m] = cmd - U
                    m += 1
        Us = U[None] + eps
        X, A = self._rollout(pose, v_now, Us)
        fld, infl, lethal = self.field.lookup(X[..., 0], X[..., 1])
        # Contact counts only after the rollout has been in a free cell: when the robot
        # itself sits inside an inflated obstacle (wedged, or the layer just painted one
        # around it) every rollout would otherwise "collide" at t=0, the costs would tie and
        # the update would be noise. This way leaving the obstacle is still preferred.
        free_seen = np.cumsum(~lethal, axis=1) > 0
        hit = np.cumsum(lethal & free_seen, axis=1) > 0    # once in contact, stays in contact
        finite = np.isfinite(fld)
        fmax = float(fld[finite].max()) if finite.any() else 0.0
        fld = np.where(finite, fld, fmax + 20.0)           # unreachable = worse than anything
        gx, gy = self.field.goal
        dgoal = np.hypot(X[..., 0] - gx, X[..., 1] - gy)
        # once a rollout is inside the goal tolerance it has "arrived": no more cost-to-go,
        # only a penalty for still moving (so it stops there instead of overshooting).
        arrived = np.cumsum(dgoal < goal_tol, axis=1) > 0
        fld = np.where(arrived, 0.0, fld)
        ang, ok = self.field.descent_at(X[..., 0], X[..., 1])
        head = np.where(ok & ~arrived & (dgoal > 1.0), 1.0 - np.cos(X[..., 2] - ang), 0.0)
        cost = (W_COLLISION * hit.sum(axis=1)
                + W_INFLATION * (infl * DT).sum(axis=1)
                + W_HEADING * (head * DT).sum(axis=1)
                + W_PROGRESS * (fld * DT).sum(axis=1)
                + W_GOAL * fld[:, -1]
                + W_DU * (np.diff(A, axis=1) ** 2).sum(axis=(1, 2))
                + W_REVERSE * (np.minimum(A[..., 0], 0.0) ** 2).sum(axis=1)
                + W_STOP_AT_GOAL * (arrived * A[..., 0] ** 2).sum(axis=1))
        smin = cost.min()
        spread = max(float(np.percentile(cost, 50) - smin), 1e-6)
        wts = np.exp(-(cost - smin) / (LAMBDA * spread))
        wts /= wts.sum()
        U = U + np.tensordot(wts, eps, axes=1)
        # Averaging a spin (v=0, |w| large) with a drive (v>0, small w) gives a crawl that the
        # 4WS limit |w| <= 1.8 v / track then strangles (live rt01: 0.05-0.1 m/s, 0.25 rad/s
        # for minutes next to the standoff). A command that slow is a spin: make it one.
        if abs(U[0, 0]) < SPIN_SNAP_V:
            U[0, 0] = 0.0
        self.U = U
        # what the robot will actually do with U[0] (same constraints as the rollout)
        Xn, An = self._rollout(pose, v_now, U[None])
        best = int(np.argmin(cost))
        self.last_info = {"cost_min": float(smin), "collide_frac": float(hit[:, -1].mean()),
                          "nominal_hits": bool(self.field.lookup(Xn[0, :, 0], Xn[0, :, 1])[2].any()),
                          "best_end": (float(X[best, -1, 0]), float(X[best, -1, 1]))}
        return float(An[0, 0, 0]), float(An[0, 0, 1])

    def reset(self):
        self.U[:] = 0.0
