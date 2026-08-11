"""Belief-INDEPENDENT geometry features (robot state, action features).

Moved out of policy.py (consolidation plan Order 4 item 3) so the NEW
goal-intent main chain (intent_features.py) can reuse them without
importing policy.py -- which pulls in belief.py/rollout.py/world_model.py
(SBK-HMM) at module level. These two functions never touched belief, an
action table, or humans' predictive moments; they are pure functions of
the robot's own state (+ the action table for
``compute_action_features_array``), so moving them changes nothing about
their behavior. Re-exported from policy.py unchanged for backward
compatibility -- every existing caller/import keeps working.
"""

from __future__ import annotations

import numpy as np

from crowd_nav.bayesian_dvl import normalization as norm
from crowd_nav.bayesian_dvl.config import NORMALIZATION_CONSTANTS
from crowd_nav.bayesian_dvl.contracts import RobotObservation


def _robot_feature_vector(robot: RobotObservation, remaining_fraction: float) -> np.ndarray:
    # guide.md 5.1: "goal-relative position, velocity, radius, v_pref";
    # R2-1 adds remaining_fraction as the 7th dim (see
    # policy.remaining_time_fraction's docstring for why). R3-2: every raw
    # value here goes through the ONE frozen normalizer (normalization.py)
    # before reaching the network -- never a hand-rescaled duplicate.
    dx, dy = norm.normalize_position(robot.gx - robot.px, robot.gy - robot.py)
    vx, vy = norm.normalize_robot_velocity(robot.vx, robot.vy)
    radius = norm.normalize_radius(robot.radius)
    v_pref = norm.normalize_speed(robot.v_pref, NORMALIZATION_CONSTANTS["robot_max_speed"])
    return np.array(
        [dx, dy, vx, vy, radius, v_pref, remaining_fraction],
        dtype=np.float32,
    )


def compute_action_features_array(robot: RobotObservation, actions: np.ndarray) -> np.ndarray:
    """Canonical ``ActionFeature`` builder (guide.md R4-1): one row of
    (norm_vx, norm_vy, norm_speed, goal_alignment, turn_cost) per
    candidate action. THE single place this is computed -- both the old
    SBK-HMM main chain (policy.py's ``_vectorized_candidate_batch``) and
    the new goal-intent main chain must call this, never a re-derived copy.

    Pure function of the robot's CURRENT (pre-action) state and the
    action table -- never depends on world samples, belief, or humans.

    ``goal_alignment``: cosine of the angle between the action's
    direction and the direction from the robot's CURRENT position to
    its goal (1 = heading straight at the goal, -1 = straight away, 0
    for a zero-speed action or a robot already at its goal).
    ``turn_cost``: 0.5*(1-cos(delta heading)) between the action's
    direction and the robot's CURRENT velocity direction (0 = no
    heading change, 1 = full reversal; 0 for a zero-speed action or a
    robot currently at a standstill, since no baseline heading exists
    to measure a turn against).
    """
    actions = np.asarray(actions, dtype=np.float64)
    norm_vx, norm_vy = norm.normalize_robot_velocity_array(actions[:, 0], actions[:, 1])
    speed = np.hypot(actions[:, 0], actions[:, 1])
    norm_speed = norm.normalize_speed_array(speed, NORMALIZATION_CONSTANTS["robot_max_speed"])
    safe_speed = np.maximum(speed, 1e-9)
    has_speed = speed > 1e-9

    goal_vec = np.array([robot.gx - robot.px, robot.gy - robot.py], dtype=np.float64)
    goal_norm = float(np.linalg.norm(goal_vec))
    if goal_norm < 1e-9:
        goal_alignment = np.zeros(actions.shape[0], dtype=np.float64)
    else:
        cos_goal = (actions[:, 0] * goal_vec[0] + actions[:, 1] * goal_vec[1]) / (safe_speed * goal_norm)
        goal_alignment = np.where(has_speed, np.clip(cos_goal, -1.0, 1.0), 0.0)

    current_speed = float(np.hypot(robot.vx, robot.vy))
    if current_speed < 1e-9:
        turn_cost = np.zeros(actions.shape[0], dtype=np.float64)
    else:
        cos_delta = (actions[:, 0] * robot.vx + actions[:, 1] * robot.vy) / (safe_speed * current_speed)
        turn_cost = np.where(has_speed, 0.5 * (1.0 - np.clip(cos_delta, -1.0, 1.0)), 0.0)

    return np.stack([norm_vx, norm_vy, norm_speed, goal_alignment, turn_cost], axis=1).astype(np.float32)
