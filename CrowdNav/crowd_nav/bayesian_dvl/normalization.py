"""Single frozen feature normalizer (guide.md R3-2).

Every raw feature dimension built anywhere in BDVL (real state at time t,
all 80 hypothetical successor states, IL demo capture, online RL capture,
smoke/latency fixtures) MUST go through these functions. They are pure,
stateless, and derive their scale purely from
``config.NORMALIZATION_CONSTANTS`` -- never from validation/formal-run
statistics, so the same normalizer is valid before a single episode has
been collected.

Every function clips its output to a bounded range: normalization here is
about SCALE, not distribution-shape; clipping (rather than raising) on an
out-of-range physical value keeps a single sensor glitch or edge-case
state from producing an unbounded feature, matching the same
"conservative bound over precision" philosophy as
``config.derive_return_bounds``.

Each quantity has exactly ONE ``_array`` core (accepts/returns numpy
arrays or scalars unchanged, no python-float coercion) plus a thin
``float()``-wrapping scalar wrapper for call sites that want a plain
python number. ``_vectorized_candidate_batch`` in policy.py (which builds
all 80 successor states' features in one batched numpy pass) calls the
SAME ``_array`` functions the scalar per-human/per-robot builders use --
this is deliberate: guide.md R3-2's acceptance requires scalar and batch
construction to be "逐位等价" (element-for-element identical), which is
only guaranteed if there is exactly one formula per quantity, not a
numpy-vectorized "equivalent" rewrite maintained separately.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from crowd_nav.bayesian_dvl.config import NORMALIZATION_CONSTANTS

_C = NORMALIZATION_CONSTANTS


def normalize_position_array(dx, dy):
    scale = _C["max_position_distance"]
    return np.clip(dx / scale, -1.0, 1.0), np.clip(dy / scale, -1.0, 1.0)


def normalize_position(dx: float, dy: float) -> Tuple[float, float]:
    nx, ny = normalize_position_array(dx, dy)
    return float(nx), float(ny)


def normalize_robot_velocity_array(vx, vy):
    scale = _C["robot_max_speed"]
    return np.clip(vx / scale, -1.0, 1.0), np.clip(vy / scale, -1.0, 1.0)


def normalize_robot_velocity(vx: float, vy: float) -> Tuple[float, float]:
    nx, ny = normalize_robot_velocity_array(vx, vy)
    return float(nx), float(ny)


def normalize_human_velocity_array(vx, vy):
    scale = _C["max_human_speed"]
    return np.clip(vx / scale, -1.0, 1.0), np.clip(vy / scale, -1.0, 1.0)


def normalize_human_velocity(vx: float, vy: float) -> Tuple[float, float]:
    nx, ny = normalize_human_velocity_array(vx, vy)
    return float(nx), float(ny)


def normalize_relative_velocity_array(vx, vy):
    scale = _C["max_relative_speed"]
    return np.clip(vx / scale, -1.0, 1.0), np.clip(vy / scale, -1.0, 1.0)


def normalize_relative_velocity(vx: float, vy: float) -> Tuple[float, float]:
    nx, ny = normalize_relative_velocity_array(vx, vy)
    return float(nx), float(ny)


def normalize_speed_array(speed, max_speed):
    return np.clip(speed / max_speed, 0.0, 1.0)


def normalize_speed(speed: float, max_speed: float) -> float:
    return float(normalize_speed_array(speed, max_speed))


def normalize_radius_array(radius):
    return np.clip(radius / _C["max_agent_radius"], 0.0, 1.0)


def normalize_radius(radius: float) -> float:
    return float(normalize_radius_array(radius))


def normalize_ttc_array(ttc, time_limit):
    # ttc is already capped at time_limit by the caller (guide.md 5.1);
    # this only rescales into [0, 1].
    return np.clip(ttc / time_limit, 0.0, 1.0)


def normalize_ttc(ttc: float, time_limit: float) -> float:
    return float(normalize_ttc_array(ttc, time_limit))


def normalize_entropy_array(entropy):
    max_entropy = float(np.log(_C["n_semantic_modes"]))
    return np.clip(entropy / max_entropy, 0.0, 1.0)


def normalize_entropy(entropy: float) -> float:
    return float(normalize_entropy_array(entropy))


def normalize_track_age_array(track_age):
    return np.clip(track_age / _C["max_track_age_steps"], 0.0, 1.0)


def normalize_track_age(track_age: float) -> float:
    return float(normalize_track_age_array(track_age))


def normalize_pred_mean_array(a_parallel, omega):
    return (
        np.clip(a_parallel / _C["max_acceleration"], -1.0, 1.0),
        np.clip(omega / _C["max_turn_rate"], -1.0, 1.0),
    )


def normalize_pred_mean(a_parallel: float, omega: float) -> Tuple[float, float]:
    a, o = normalize_pred_mean_array(a_parallel, omega)
    return float(a), float(o)


def normalize_pred_cov_upper_array(var_a, cov_a_omega, var_omega):
    a_scale = _C["max_acceleration"] ** 2
    cross_scale = _C["max_acceleration"] * _C["max_turn_rate"]
    omega_scale = _C["max_turn_rate"] ** 2
    return (
        np.clip(var_a / a_scale, 0.0, 4.0),
        np.clip(cov_a_omega / cross_scale, -4.0, 4.0),
        np.clip(var_omega / omega_scale, 0.0, 4.0),
    )


def normalize_pred_cov_upper(var_a: float, cov_a_omega: float, var_omega: float) -> Tuple[float, float, float]:
    a, c, o = normalize_pred_cov_upper_array(var_a, cov_a_omega, var_omega)
    return float(a), float(c), float(o)
