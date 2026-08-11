"""Named one-step transition + reward, equivalent to
``CrowdSim.step``/``onestep_lookahead`` (guide.md A2).

Deliberately reimplemented from named ``RobotObservation``/
``HumanObservation`` fields rather than reusing
``crowd_nav.contracts.simulate_next_frames`` (verified buggy: reads
``current_state[5]`` believing it is ``v_pref``, but that index is
``gx`` -- see ``contracts.py`` module docstring).

Collision/clearance uses the SAME formula as
``crowd_sim/envs/crowd_sim.py:step()``: the swept segment from the
robot-relative human position to where it would be one dt later,
assuming the robot moves at its NEWLY CHOSEN action velocity and each
human continues at its CURRENT (pre-step) velocity -- NOT the human's
own new action for this step. This is intentional and matches the real
environment exactly (verified by direct code reading, then by the A2
equivalence test against ``CrowdSim.onestep_lookahead``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation

EVENT_TIMEOUT = "timeout"
EVENT_COLLISION = "collision"
EVENT_SUCCESS = "reach_goal"
EVENT_NORMAL = "nothing"


@dataclass(frozen=True)
class RewardConfig:
    success_reward: float
    collision_penalty: float
    timeout_penalty: float
    progress_reward: float
    time_penalty: float
    stand_penalty: float
    stand_speed_threshold: float
    discomfort_distance: float
    discomfort_penalty_factor: float


@dataclass(frozen=True)
class StepResult:
    next_robot: RobotObservation
    next_humans: Tuple[HumanObservation, ...]
    reward: float
    terminated: bool
    truncated: bool
    event: str
    dmin: float


@dataclass(frozen=True)
class BatchStepResult:
    """Vectorized counterpart of :class:`StepResult`.

    The first axis is the candidate action and the second is the sampled
    world. The scalar ``step`` implementation remains the reference for
    equivalence tests; this container only removes Python overhead from the
    repeated candidate evaluation in BDVLPolicy.
    """

    next_robot_positions: np.ndarray  # [A, 2]
    next_human_positions: np.ndarray  # [S, H, 2]
    next_human_velocities: np.ndarray  # [S, H, 2]
    rewards: np.ndarray  # [A, S]
    terminated: np.ndarray  # [A, S]
    truncated: np.ndarray  # [A, S]
    events: np.ndarray  # [A, S], dtype=object
    dmin: np.ndarray  # [A]


def point_to_segment_dist(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    """Closest distance from point (x3,y3) to segment (x1,y1)-(x2,y2).
    Bit-for-bit port of crowd_sim/envs/utils/utils.py:point_to_segment_dist."""
    px = x2 - x1
    py = y2 - y1
    if px == 0 and py == 0:
        return float(np.hypot(x3 - x1, y3 - y1))
    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)
    u = min(1.0, max(0.0, u))
    x = x1 + u * px
    y = y1 + u * py
    return float(np.hypot(x3 - x, y3 - y))


def propagate_holonomic(px: float, py: float, vx: float, vy: float, dt: float) -> Tuple[float, float]:
    return px + vx * dt, py + vy * dt


def propagate_robot(robot: RobotObservation, action_vx: float, action_vy: float, dt: float) -> RobotObservation:
    next_px, next_py = propagate_holonomic(robot.px, robot.py, action_vx, action_vy, dt)
    return RobotObservation(
        px=next_px, py=next_py, vx=action_vx, vy=action_vy,
        radius=robot.radius, gx=robot.gx, gy=robot.gy,
        v_pref=robot.v_pref, theta=robot.theta,
    )


def propagate_human(human: HumanObservation, action_vx: float, action_vy: float, dt: float) -> HumanObservation:
    next_px, next_py = propagate_holonomic(human.px, human.py, action_vx, action_vy, dt)
    return HumanObservation(
        track_id=human.track_id, px=next_px, py=next_py,
        vx=action_vx, vy=action_vy, radius=human.radius,
    )


def _collision_and_clearance(
    robot: RobotObservation, action_vx: float, action_vy: float,
    humans: Sequence[HumanObservation], dt: float,
) -> Tuple[bool, float]:
    """Matches crowd_sim.py:step()'s collision loop exactly: relative
    velocity uses the robot's NEW action and each human's CURRENT
    (pre-step) velocity; swept over the segment [now, now+dt] in the
    robot-relative frame."""
    rvx, rvy = -action_vx, -action_vy
    dmin = float("inf")
    collision = False
    for human in humans:
        px = human.px - robot.px
        py = human.py - robot.py
        vx = human.vx + rvx
        vy = human.vy + rvy
        ex = px + vx * dt
        ey = py + vy * dt
        closest = point_to_segment_dist(px, py, ex, ey, 0.0, 0.0) - human.radius - robot.radius
        if closest < 0:
            return True, closest
        if closest < dmin:
            dmin = closest
    return collision, dmin


def step(
    robot: RobotObservation,
    humans: Sequence[HumanObservation],
    action_vx: float,
    action_vy: float,
    human_actions: Sequence[Tuple[float, float]],
    dt: float,
    time_limit: float,
    global_time: float,
    reward_config: RewardConfig,
    success_distance: Optional[float] = None,
) -> StepResult:
    """One environment step. ``success_distance`` defaults to
    ``robot.radius`` to match the REAL environment's actual success
    check (``distance_to_goal < robot.radius``) -- guide.md 7.3
    confirms there is no separate success_radius parameter read by the
    real env, and the config's own ``success_radius`` field is dead."""
    if len(human_actions) != len(humans):
        raise ValueError(f"{len(humans)} humans but {len(human_actions)} human actions")

    collision, dmin = _collision_and_clearance(robot, action_vx, action_vy, humans, dt)

    next_robot = propagate_robot(robot, action_vx, action_vy, dt)
    next_humans = tuple(
        propagate_human(human, hvx, hvy, dt)
        for human, (hvx, hvy) in zip(humans, human_actions)
    )

    threshold = robot.radius if success_distance is None else success_distance
    reaching_goal = float(np.hypot(next_robot.px - robot.gx, next_robot.py - robot.gy)) < threshold

    prev_dist = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
    curr_dist = float(np.hypot(next_robot.px - robot.gx, next_robot.py - robot.gy))
    progress = prev_dist - curr_dist

    terminated = False
    truncated = False

    if global_time >= time_limit - 1e-6:
        reward = reward_config.timeout_penalty
        truncated = True
        event = EVENT_TIMEOUT
    elif collision:
        reward = reward_config.collision_penalty
        terminated = True
        event = EVENT_COLLISION
    elif reaching_goal:
        reward = reward_config.success_reward
        terminated = True
        event = EVENT_SUCCESS
    else:
        reward = reward_config.progress_reward * progress
        reward += reward_config.time_penalty
        robot_speed = float(np.hypot(action_vx, action_vy))
        if robot_speed < reward_config.stand_speed_threshold:
            reward += reward_config.stand_penalty
        if dmin < reward_config.discomfort_distance:
            penalty = reward_config.discomfort_penalty_factor * (reward_config.discomfort_distance - dmin) * dt
            reward -= penalty
        event = EVENT_NORMAL

    return StepResult(
        next_robot=next_robot, next_humans=next_humans,
        reward=float(reward), terminated=terminated, truncated=truncated,
        event=event, dmin=max(dmin, 0.0) if not collision else dmin,
    )


def batch_step(
    robot: RobotObservation,
    humans: Sequence[HumanObservation],
    actions: np.ndarray,
    human_actions: np.ndarray,
    dt: float,
    time_limit: float,
    global_time: float,
    reward_config: RewardConfig,
    success_distance: Optional[float] = None,
) -> BatchStepResult:
    """Evaluate all action/world pairs without Python loops.

    ``actions`` has shape ``[A, 2]`` and ``human_actions`` has shape
    ``[S, H, 2]``. Collision geometry and event priority intentionally match
    ``step``: timeout, then collision, then success. The scalar ``step``
    implementation remains the reference for equivalence tests.
    """
    actions = np.asarray(actions, dtype=np.float64)
    human_actions = np.asarray(human_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"actions must have shape [A,2], got {actions.shape}")
    if human_actions.ndim != 3 or human_actions.shape[2] != 2:
        raise ValueError(f"human_actions must have shape [S,H,2], got {human_actions.shape}")
    n_actions = actions.shape[0]
    n_samples, n_humans = human_actions.shape[:2]
    if n_actions < 1 or n_samples < 1:
        raise ValueError("batch_step requires at least one action and one world sample")
    if n_humans != len(humans):
        raise ValueError(f"human_actions has {n_humans} humans but input has {len(humans)}")

    human_positions = np.asarray([[h.px, h.py] for h in humans], dtype=np.float64).reshape(n_humans, 2)
    human_velocities = np.asarray([[h.vx, h.vy] for h in humans], dtype=np.float64).reshape(n_humans, 2)
    human_radii = np.asarray([h.radius for h in humans], dtype=np.float64).reshape(n_humans)

    # Vector form of _collision_and_clearance. Retain the first negative
    # clearance for collision cases to match the scalar function's early
    # return, rather than silently changing dmin semantics.
    if n_humans:
        relative_start = human_positions - np.asarray([robot.px, robot.py], dtype=np.float64)
        relative_velocity = human_velocities[None, :, :] - actions[:, None, :]
        segment = relative_velocity * dt
        denominator = np.sum(segment * segment, axis=2)
        numerator = -np.sum(relative_start[None, :, :] * segment, axis=2)
        u = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0.0)
        u = np.clip(u, 0.0, 1.0)
        closest = relative_start[None, :, :] + u[:, :, None] * segment
        closest_distance = np.hypot(closest[:, :, 0], closest[:, :, 1])
        clearances = closest_distance - human_radii[None, :] - float(robot.radius)
        collision = np.any(clearances < 0.0, axis=1)
        first_collision = np.argmax(clearances < 0.0, axis=1)
        dmin = np.min(clearances, axis=1)
        collision_dmin = clearances[np.arange(n_actions), first_collision]
        dmin = np.where(collision, collision_dmin, dmin)
    else:
        collision = np.zeros(n_actions, dtype=bool)
        dmin = np.full(n_actions, np.inf, dtype=np.float64)

    next_robot_positions = np.asarray([robot.px, robot.py], dtype=np.float64)[None, :] + actions * dt
    next_human_positions = human_positions[None, :, :] + human_actions * dt
    next_human_velocities = human_actions.copy()

    threshold = float(robot.radius if success_distance is None else success_distance)
    goal = np.asarray([robot.gx, robot.gy], dtype=np.float64)
    next_goal_distance = np.hypot(next_robot_positions[:, 0] - goal[0], next_robot_positions[:, 1] - goal[1])
    reaching_goal = next_goal_distance < threshold
    previous_goal_distance = float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
    progress = previous_goal_distance - next_goal_distance
    timeout = bool(global_time >= time_limit - 1e-6)

    rewards = np.empty((n_actions, n_samples), dtype=np.float64)
    terminated = np.zeros((n_actions, n_samples), dtype=bool)
    truncated = np.zeros((n_actions, n_samples), dtype=bool)
    events = np.full((n_actions, n_samples), EVENT_NORMAL, dtype=object)

    if timeout:
        rewards.fill(float(reward_config.timeout_penalty))
        truncated.fill(True)
        events.fill(EVENT_TIMEOUT)
    else:
        collision_matrix = np.broadcast_to(collision[:, None], (n_actions, n_samples))
        success_matrix = np.broadcast_to(reaching_goal[:, None], (n_actions, n_samples))
        rewards[:] = float(reward_config.progress_reward) * progress[:, None]
        rewards += float(reward_config.time_penalty)
        speed = np.hypot(actions[:, 0], actions[:, 1])
        if reward_config.stand_penalty:
            rewards += np.where(speed[:, None] < reward_config.stand_speed_threshold, reward_config.stand_penalty, 0.0)
        discomfort = float(reward_config.discomfort_distance)
        discomfort_penalty = float(reward_config.discomfort_penalty_factor) * (discomfort - dmin) * float(dt)
        rewards += np.where(dmin[:, None] < discomfort, -discomfort_penalty[:, None], 0.0)
        rewards = np.where(collision_matrix, float(reward_config.collision_penalty), rewards)
        rewards = np.where(success_matrix & ~collision_matrix, float(reward_config.success_reward), rewards)
        terminated = collision_matrix | success_matrix
        events[collision_matrix] = EVENT_COLLISION
        events[success_matrix & ~collision_matrix] = EVENT_SUCCESS

    return BatchStepResult(
        next_robot_positions=next_robot_positions,
        next_human_positions=next_human_positions,
        next_human_velocities=next_human_velocities,
        rewards=rewards.astype(np.float32),
        terminated=terminated,
        truncated=truncated,
        events=events,
        dmin=np.where(collision, dmin, np.maximum(dmin, 0.0)),
    )
