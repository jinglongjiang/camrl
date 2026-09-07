"""V0 one-step successor-state lookahead.

Why this exists. The frozen chain scores actions as Q(s, b, a): the action is
five scalars (vx, vy, speed, goal_alignment, turn_cost) fed alongside the state
embedding, and only the ONE executed action of each transition ever receives an
MC return target -- the other 79 are shaped by a ranking hinge alone. Ranking
teaches which action outranks which; it cannot teach what the world looks like
after an action is taken. Deployment then demands a reliable comparison across
all 80. Measured consequence: offline ranking at its best (top-1 0.869,
margin +0.055) with the WORST closed-loop macro of three weights (0.078).

Here the comparison is structural instead of learned:

    a* = argmax_a [ r(s, a) + gamma * V(s'_a, b) ]

r is exact, not approximate. crowd_sim.step() computes its swept clearance from
the humans' CURRENT velocities held constant across the interval (crowd_sim.py
~L370), so reproducing that same computation reproduces the environment's own
reward bit for bit. The belief therefore does NOT enter r -- it enters the value
term, through where the humans are predicted to BE in s'_a.

That is the whole point of the rewrite: the posterior stops being a few extra
numbers in a 70-dim input row and becomes the thing that decides which future
the value function is asked about.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.intent_runtime_config import (
    FROZEN_VALUES, HUMAN_SCALAR_IDX_ROBOT_RELATIVE, NORMALIZATION_CONSTANTS,
)
from crowd_nav.bayesian_dvl import normalization as norm
from crowd_nav.bayesian_dvl.intent_policy import _ttc

DT = float(FROZEN_VALUES["dt"])
GAMMA_DEFAULT = 0.99

# Which packed-row columns depend on the ROBOT. Everything else in the 70-dim
# row -- belief vector, entropy, top-1 margin, future delta/spread, candidate
# block, temporal trends, track age, human speed and radius -- is a function of
# the human and the belief only. Measured: rebuilding the whole row for all 80
# candidates costs 6.37 s/step in the `full` arm; patching these five costs
# ~0.08 s/step for an identical result.
#
# The offsets come from intent_runtime_config, which is where the rest of the
# row layout is declared. They are NOT restated here: a hard-coded copy cannot
# notice when the row layout moves, and rewriting the wrong five columns of
# every candidate successor would still produce plausible numbers.
IDX_DX, IDX_DY, IDX_RVX, IDX_RVY, IDX_TTC = HUMAN_SCALAR_IDX_ROBOT_RELATIVE


def assert_robot_relative_columns(bank, robot, humans, mode, rng, horizon, n_samples) -> None:
    """Prove, against the real feature builder, that exactly those five columns
    move when only the robot moves.

    A declared constant still only records an intention. This builds one packed
    batch twice -- identical humans, identical belief, identical rng, a robot
    displaced -- and checks that the set of columns which changed is exactly
    HUMAN_SCALAR_IDX_ROBOT_RELATIVE. If the row layout ever shifts, this fails
    loudly at startup instead of silently corrupting every lookahead.
    """
    from crowd_nav.bayesian_dvl.intent_policy import build_intent_human_feature_batch
    seed = int(rng.integers(1 << 31))
    a_rows, a_mask = build_intent_human_feature_batch(
        bank, robot, humans, mode=mode, rng=np.random.default_rng(seed),
        horizon=horizon, n_samples=n_samples)
    moved = RobotObservation(
        px=robot.px + 0.37, py=robot.py - 0.21, vx=robot.vx + 0.19, vy=robot.vy - 0.11,
        radius=robot.radius, gx=robot.gx, gy=robot.gy, v_pref=robot.v_pref, theta=robot.theta)
    b_rows, _ = build_intent_human_feature_batch(
        bank, moved, humans, mode=mode, rng=np.random.default_rng(seed),
        horizon=horizon, n_samples=n_samples)
    if not a_mask.any():
        return                                    # no tracked humans yet; nothing to check
    diff = np.abs(a_rows[a_mask] - b_rows[a_mask]).max(axis=0)
    changed = frozenset(int(i) for i in np.nonzero(diff > 1e-7)[0])
    expected = frozenset(HUMAN_SCALAR_IDX_ROBOT_RELATIVE)
    # SUBSET, not equality. The failure that matters is a column OUTSIDE the
    # patch list moving with the robot, because patch_robot_columns would then
    # leave it stale. A column inside the list that happens not to move is
    # harmless -- it is rewritten with its correct value regardless. The first
    # version of this check demanded equality and fired on episode step 0, where
    # freshly spawned humans are near-stationary so ttc saturates at the time
    # limit for both robot positions and legitimately does not move.
    unexpected = changed - expected
    if unexpected:
        raise RuntimeError(
            "packed-row layout moved: displacing the robot also changed columns "
            f"{sorted(unexpected)}, which patch_robot_columns does not rewrite. "
            f"Expected the robot-dependent set to be within {sorted(expected)}.")


def robot_successor(robot: RobotObservation, vx: float, vy: float) -> RobotObservation:
    """Holonomic single integrator -- the same model crowd_sim uses to decide
    whether an action reaches the goal (`robot.compute_position`)."""
    return RobotObservation(
        px=robot.px + vx * DT, py=robot.py + vy * DT, vx=vx, vy=vy,
        radius=robot.radius, gx=robot.gx, gy=robot.gy, v_pref=robot.v_pref,
        theta=float(np.arctan2(vy, vx)) if (vx or vy) else robot.theta,
    )


def _point_to_segment_dist(x1, y1, x2, y2, x3, y3) -> float:
    """Distance from (x3,y3) to segment (x1,y1)-(x2,y2). Same function
    crowd_sim uses for its swept clearance."""
    px, py = x2 - x1, y2 - y1
    if px == 0 and py == 0:
        return float(np.hypot(x3 - x1, y3 - y1))
    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)
    u = min(max(u, 0.0), 1.0)
    return float(np.hypot(x1 + u * px - x3, y1 + u * py - y3))


def swept_dmin(robot: RobotObservation, humans: Sequence[HumanObservation],
               vx: float, vy: float) -> float:
    """crowd_sim's own clearance, reproduced exactly: each human's position
    RELATIVE to the robot is swept over one interval at the relative velocity
    (human velocity minus the commanded robot velocity), and the closest
    approach of that segment to the origin is taken, minus both radii."""
    dmin = float("inf")
    for h in humans:
        px, py = h.px - robot.px, h.py - robot.py
        rvx, rvy = h.vx - vx, h.vy - vy
        closest = _point_to_segment_dist(px, py, px + rvx * DT, py + rvy * DT, 0.0, 0.0) \
            - h.radius - robot.radius
        if closest < dmin:
            dmin = closest
        if closest < 0:
            break
    return dmin


def immediate_reward(robot: RobotObservation, humans: Sequence[HumanObservation],
                     vx: float, vy: float) -> Tuple[float, bool, float]:
    """(reward, terminal, dmin) for one action, matching crowd_sim.step().

    Deliberately NOT belief-conditioned: the environment's own reward holds the
    humans at constant velocity across the interval, so using their observed
    velocities here makes this term exact rather than an approximation of it.
    Introducing predicted positions would make the selection reward disagree
    with the training returns -- the exact class of two-sided contract drift
    that has already cost this project several runs.
    """
    goal = np.array([robot.gx, robot.gy], dtype=np.float64)
    here = np.array([robot.px, robot.py], dtype=np.float64)
    nxt = here + np.array([vx, vy], dtype=np.float64) * DT
    dmin = swept_dmin(robot, humans, vx, vy)

    if dmin < 0:
        return float(FROZEN_VALUES["collision_penalty"]), True, dmin
    if float(np.linalg.norm(nxt - goal)) < robot.radius:
        return float(FROZEN_VALUES["success_reward"]), True, dmin

    r = float(FROZEN_VALUES["progress_reward"]) * float(
        np.linalg.norm(here - goal) - np.linalg.norm(nxt - goal))
    r += float(FROZEN_VALUES["time_penalty"])
    if float(np.hypot(vx, vy)) < float(FROZEN_VALUES["stand_speed_threshold"]):
        r += float(FROZEN_VALUES["stand_penalty"])
    dd = float(FROZEN_VALUES["discomfort_distance"])
    if dmin < dd:
        r -= float(FROZEN_VALUES["discomfort_penalty_factor"]) * (dd - dmin) * DT
    return r, False, dmin


def predict_humans(bank, humans: Sequence[HumanObservation], mode: str,
                   rng: np.random.Generator, n_samples: int) -> List[HumanObservation]:
    """Where each human is predicted to be one step from now, UNDER THE BELIEF.

    This is the one place the ablation changes the successor state, and it uses
    the frozen arm definitions rather than a new set:

      full -- mean over posterior-SAMPLED futures (multimodal)
      mean -- the posterior-WEIGHTED mean future (unimodal)
      cv   -- constant velocity: no goal posterior at all

    So `cv` is exactly "predict the crowd without using the belief", which makes
    full-vs-cv a direct measurement of what the posterior buys, rather than an
    input-ablation whose effect the network might route around.
    """
    out: List[HumanObservation] = []
    for h in humans:
        pos = np.array([h.px, h.py], dtype=np.float64)
        vel = np.array([h.vx, h.vy], dtype=np.float64)
        try:
            tracker = bank.tracker_for(h.track_id)
        except Exception:
            out.append(h)          # not yet tracked -> nothing to condition on
            continue
        if mode == "cv":
            nxt = pos + vel * tracker.dt
        else:
            trajs = tracker.sample_futures(pos, vel, horizon=1, mode=mode,
                                           rng=rng, n_samples=n_samples)
            nxt = np.mean(np.asarray(trajs, dtype=np.float64)[:, 0, :], axis=0)
        out.append(HumanObservation(
            h.track_id, float(nxt[0]), float(nxt[1]),
            float((nxt[0] - pos[0]) / tracker.dt), float((nxt[1] - pos[1]) / tracker.dt),
            h.radius))
    return out


def patch_robot_columns(rows: np.ndarray, mask: np.ndarray,
                        robot: RobotObservation, humans: Sequence[HumanObservation]) -> np.ndarray:
    """Rewrite ONLY the robot-relative columns of an already-built packed batch.

    The belief columns were computed for these same humans and are unchanged by
    the robot's action, so recomputing them per candidate would burn 80x the
    sampling cost to produce identical numbers.
    """
    out = rows.copy()
    tl = float(FROZEN_VALUES["time_limit"])
    for i, h in enumerate(humans[: rows.shape[0]]):
        if not mask[i]:
            continue
        raw_dx, raw_dy = h.px - robot.px, h.py - robot.py
        raw_rvx, raw_rvy = h.vx - robot.vx, h.vy - robot.vy
        dx, dy = norm.normalize_position(raw_dx, raw_dy)
        rvx, rvy = norm.normalize_relative_velocity(raw_rvx, raw_rvy)
        ttc = norm.normalize_ttc(
            min(_ttc(raw_dx, raw_dy, raw_rvx, raw_rvy, h.radius + robot.radius), tl), tl)
        out[i, IDX_DX], out[i, IDX_DY] = dx, dy
        out[i, IDX_RVX], out[i, IDX_RVY] = rvx, rvy
        out[i, IDX_TTC] = ttc
    return out
