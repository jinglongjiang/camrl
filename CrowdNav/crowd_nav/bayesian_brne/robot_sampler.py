"""robot_sampler.py: candidate robot control/trajectory generation for
CrowdNav's holonomic robot (Order F3, 2026-08-03 -- guide.md section 4.3).

Produces a fixed number ``M`` of candidate robot control sequences and their
resulting position trajectories, all sharing ONE goal-directed nominal plus
GP-correlated perturbations (the SAME kernel machinery
``trajectory_sampler.sample_cv`` uses for its pedestrian CV ablation, for a
consistent, reproducible sampling convention across the whole method) --
never upstream's own module-level global RNG (``brne_adapter.load_upstream_brne``
is only used here for its noise-free ``get_Lmat_nb`` covariance construction;
the actual standard-normal draws are made with THIS call's own local ``rng``
argument, exactly the fix guide.md already required of ``sample_cv``).

Given a fixed ``rng`` seed, ``sample_robot_candidates`` is fully
deterministic and reproducible. The SAME batch of ``M`` candidates it
returns is meant to be reused by the main method AND every BRNE-family
baseline (original/CV/posterior-mean/self-only BRNE) for a given real
control step -- this function itself has no opinion about which ablation
is running; callers (Order F4's outer loop / Order F5's policy) are
responsible for calling it ONCE per real step and sharing the result,
never re-sampling per ablation.

Physical realism matches ``trajectory_sampler.py``'s own discipline: every
step clips ``delta_v`` to ``max_acceleration * dt`` and the resulting speed
to ``max_speed``, using the SAME clipping helper (imported, not
re-derived) so the two modules cannot silently drift apart on what
"physically realistic" means.

Only CrowdNav's holonomic robot is implemented this round; Gazebo's
differential-drive kinematics are deferred to Phase S4 (guide.md section 6),
after the closed-loop CrowdNav pipeline is validated end to end.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from crowd_nav.bayesian_brne.trajectory_sampler import (
    DEFAULT_MAX_ACCELERATION, DEFAULT_MAX_SPEED, _clip_delta_v_and_speed,
)

DEFAULT_V_PREF = 1.0


def sample_robot_candidates(
    robot_px: float, robot_py: float, robot_vx: float, robot_vy: float,
    goal_gx: float, goal_gy: float,
    horizon: int,
    num_candidates: int,
    dt: float,
    rng: np.random.Generator,
    v_pref: float = DEFAULT_V_PREF,
    max_speed: float = DEFAULT_MAX_SPEED,
    max_acceleration: float = DEFAULT_MAX_ACCELERATION,
    brne_root: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(actions[M,H,2], state_positions[M,H,2],
    state_velocities[M,H,2], future_positions[M,H,2])``.

    Order R1 (2026-08-03) fix: an earlier version returned only
    ``(controls, positions)`` where ``positions[m,h]`` was the position
    AFTER applying ``controls[m,h]`` -- callers then paired that
    post-action position with the SAME-INDEX action as if they were
    simultaneous ("robot_state[h]"), which is exactly backwards from the
    offline training convention (``context[t] = f(human[t], robot_state[t])``,
    ``u_r[t]`` is the action APPLIED AT ``t`` that produces the transition
    to ``t+1`` -- i.e. ``robot_state[t]`` must be the state BEFORE ``u_r[t]``
    is applied, never after). This version returns FOUR explicit,
    unambiguous arrays instead:

    - ``actions[m,h]``: candidate ``m``'s COMMANDED velocity applied
      DURING step ``h`` (this method's ``u_r[h]``).
    - ``state_positions[m,h]`` / ``state_velocities[m,h]``: the robot's
      position/velocity BEFORE ``actions[m,h]`` is applied -- i.e. exactly
      ``robot_state[h]`` in the offline training convention.
      ``state_positions[m,0] == (robot_px, robot_py)`` and
      ``state_velocities[m,0] == (robot_vx, robot_vy)`` for EVERY
      candidate ``m`` (all candidates share the same starting state by
      construction); for ``h >= 1`` they equal the PREVIOUS step's
      ``future_positions``/``actions``.
    - ``future_positions[m,h]``: the position AFTER applying
      ``actions[m,h]`` -- the array BRNE's collision cost should use (it
      needs to know where the robot will ACTUALLY be, not where it was
      before moving).

    Candidate 0 is the strict, noise-free nominal. Its heading points from
    the CURRENT robot position straight at the goal, held at
    ``min(v_pref, max_speed)``
    for the whole horizon (a simple, deliberately un-clever "aim at the
    goal" pure-pursuit nominal -- CrowdNav's short planning horizons make
    re-aiming at a FIXED goal each step, rather than a curved/replanned
    path, a reasonable nominal; Order F4's outer BRNE loop is what actually
    lets the robot react to the crowd, not this nominal by itself). Each of
    candidates 1..M-1 perturbs that nominal with GP-correlated noise
    (scaled by the nominal speed, matching ``sample_cv``'s own
    speed-scaling convention) so the candidate SET has smooth, temporally
    correlated diversity rather than independent per-step jitter, then
    clips every step to the same physical acceleration/speed limits
    ``trajectory_sampler.py`` enforces for pedestrians.
    """
    from crowd_nav.bayesian_brne.brne_adapter import load_upstream_brne

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")

    upstream = load_upstream_brne(brne_root)
    tlist = np.arange(horizon) * dt
    train_ts = np.array([tlist[0]])
    train_noise = np.array([1e-2])
    cov_lmat, _cov = upstream.get_Lmat_nb(train_ts, tlist, train_noise)

    goal_vec = np.array([goal_gx - robot_px, goal_gy - robot_py])
    goal_dist = float(np.linalg.norm(goal_vec))
    heading = goal_vec / max(goal_dist, 1e-6)
    nominal_speed = float(min(v_pref, max_speed))
    nominal_control = heading * nominal_speed  # [2], SAME every horizon step

    noisy_candidates = max(0, num_candidates - 1)
    base_x = rng.standard_normal(size=(horizon, noisy_candidates))
    base_y = rng.standard_normal(size=(horizon, noisy_candidates))
    noise_x = (cov_lmat @ base_x).T  # [M-1, H]
    noise_y = (cov_lmat @ base_y).T

    actions = np.zeros((num_candidates, horizon, 2))
    state_positions = np.zeros((num_candidates, horizon, 2))
    state_velocities = np.zeros((num_candidates, horizon, 2))
    future_positions = np.zeros((num_candidates, horizon, 2))
    noise_scale = max(nominal_speed, 1e-3)
    for m in range(num_candidates):
        p = np.array([robot_px, robot_py])
        v_prev = np.array([robot_vx, robot_vy])
        for h in range(horizon):
            # state_positions[m,h]/state_velocities[m,h] are the state
            # BEFORE this step's action -- recorded FIRST, before p/v_prev
            # are advanced below.
            state_positions[m, h] = p
            state_velocities[m, h] = v_prev

            if m == 0:
                desired_v = nominal_control
            else:
                desired_v = nominal_control + np.array([noise_x[m - 1, h], noise_y[m - 1, h]]) * noise_scale
            delta_v = desired_v - v_prev
            v_new = _clip_delta_v_and_speed(v_prev, delta_v, dt, max_speed, max_acceleration)
            actions[m, h] = v_new
            p = p + dt * v_new
            future_positions[m, h] = p
            v_prev = v_new
    return actions, state_positions, state_velocities, future_positions
