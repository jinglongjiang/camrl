"""Posterior-preserving pedestrian trajectory sampling -- and its ablations.

Implements guide.md 6.5's six sampling modes through ONE shared rollout
primitive, so an ablation can only ever change WHICH distribution feeds the
sampler, never the sampling/rollout mechanics themselves:

    full_posterior      main method: z0 ~ b_next, z_{h+1} ~ Pi[z_h], process
                         noise ~ Q[z_h] each step (guide.md 2.2).
    map_mode            z fixed at argmax(b) for the whole horizon.
    posterior_mean       *banned from the main method* -- ablation only.
                         A single-Gaussian, MOMENT-MATCHED collapse of the
                         full mixture at every step: same first moment
                         (belief-weighted mean next-velocity) AND same
                         second moment (within-mode covariance PLUS
                         between-mode covariance of the mode means around
                         that mean -- see ``_moment_matched_mixture_gaussian``).
                         This is the fair unimodal baseline: full vs. mean
                         must differ only because of HIGHER-ORDER structure
                         (genuine multimodality), never because the mean
                         ablation was secretly given an artificially narrow
                         spread.
    uniform_posterior   same rollout as full_posterior, with b replaced by
                         1/K -- tests whether the SHAPE of the belief matters
                         at all, independent of having a plausible-looking
                         multi-modal sampler.
    shuffled_posterior  same rollout as full_posterior; the caller supplies
                         a DIFFERENT track's belief, fixed for the whole
                         episode (guide.md 6.5: shuffle once per episode,
                         never re-shuffled per step) -- see
                         `shuffle_track_beliefs`.
    cv                  constant-velocity mean + the SAME Gaussian-process
                         KERNEL upstream BRNE's own demo uses, sampled with
                         THIS call's own local RNG (not upstream's global
                         one -- see ``sample_cv``), for a fair, reproducible
                         apples-to-apples CV-BRNE ablation. Independent of
                         the AR-HMM entirely (no robot/context inputs).

Order F3 (2026-08-03) migration to ``action_conditioned_arhmm.ARHMMArtifact``:
the earlier ``mode_model.ModeModelArtifact``-based version had one
explicitly-documented approximation -- during a multi-step rollout, the
robot-relative features (relative position/velocity, TTC, passing side)
could not be re-derived from the pedestrian's own future state alone, so
they were FROZEN at their value from the start of the rollout. This
version removes that approximation entirely: every rollout mode recomputes
``context[h]`` from the ACTUAL candidate robot state at every step h via
``action_conditioned_arhmm.compute_context_features``. There is also no
more separate self-kinematic ``phi`` vector to carry across steps: the
AR-HMM's ``A_k @ v_h[h]`` autoregressive term replaces what ``phi``'s
speed/delta_speed/heading_change/lateral_acceleration features did in the
old model, so a rollout step only needs the pedestrian's own
``[px,py,vx,vy]`` state, never an extra hand-maintained feature vector.

Order R1 (2026-08-03) fix -- training/rollout timing semantics: an earlier
version took ONE ``robot_velocity_sequence[H,2]`` and used it BOTH as the
robot's state velocity for ``context[h]`` AND as the model's ``u_r[h]``
action term, paired with a ``robot_position_sequence[H,2]`` that was
actually the position AFTER applying that same step's action (as returned
by ``robot_sampler``'s old two-array contract). That conflates "robot state
BEFORE this step's action" with "the action itself" -- exactly backwards
from the offline training convention (``context[t] = f(human[t],
robot_state[t])``, ``u_r[t]`` produces the transition to ``t+1``, so
``robot_state[t]`` must be the PRE-action state). This version takes THREE
separate sequences -- ``robot_state_position_sequence[H,2]``,
``robot_state_velocity_sequence[H,2]`` (both PRE-action state, matching
``robot_sampler.sample_robot_candidates``'s ``state_positions``/
``state_velocities``), and ``robot_action_sequence[H,2]`` (the actual
``u_r[h]``, matching its ``actions``) -- and never reuses one array for
two different roles. See ``arhmm_offline_online_context_equivalence`` and
``arhmm_rollout_fails_on_old_conflated_sequence_when_accel_nonzero`` in
selftest.py for the reference/adversarial tests this fix added.

Physical realism (fixes B3 from the 2026-08-03 independent audit): every
rollout clips delta_v to ``max_acceleration * dt`` and the resulting speed to
``max_speed`` each step -- an earlier version let delta_v accumulate
unbounded across a multi-step rollout, reaching a 7.55m lateral spread in
2.5 real seconds, far beyond any real pedestrian's kinematics.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact, compute_context_features

DEFAULT_MAX_SPEED = 2.0
DEFAULT_MAX_ACCELERATION = 2.0


def _clip_delta_v_and_speed(v_prev: np.ndarray, delta_v: np.ndarray, dt: float, max_speed: float, max_acceleration: float) -> np.ndarray:
    """Clip ``delta_v``'s magnitude to ``max_acceleration * dt``, then clip
    the resulting new velocity's magnitude to ``max_speed``. Returns the
    clipped ``v_new`` (not the clipped delta_v alone), since the speed cap
    must apply to the accumulated velocity, not just one step's change."""
    max_step_delta = max_acceleration * dt
    delta_norm = float(np.linalg.norm(delta_v))
    if delta_norm > max_step_delta:
        delta_v = delta_v * (max_step_delta / delta_norm)
    v_new = v_prev + delta_v
    speed = float(np.linalg.norm(v_new))
    if speed > max_speed:
        v_new = v_new * (max_speed / speed)
    return v_new


def _mode_mean_velocity(
    v_prev: np.ndarray, u_r_h: np.ndarray, context_h: np.ndarray, artifact: ARHMMArtifact, k: int,
) -> np.ndarray:
    """[2] this mode's predicted ABSOLUTE next velocity (not a delta) --
    ``A_k @ v_prev + B_k @ u_r_h + C_k @ context_h + d_k``."""
    return artifact.A[k] @ v_prev + artifact.B[k] @ u_r_h + artifact.C[k] @ context_h + artifact.d[k]


def build_rollout_step_inputs(
    pedestrian_state: np.ndarray,
    robot_state_position: np.ndarray,
    robot_state_velocity: np.ndarray,
    robot_action: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact per-step inputs shared by rollout and its reference test.

    ``pedestrian_state`` is ``[px, py, vx, vy]`` at the beginning of the
    step. The returned tuple is ``(v_current, u_robot, context)`` and mirrors
    the offline AR-HMM row definition without a second test-only formula.
    """
    x = np.asarray(pedestrian_state, dtype=np.float64)
    robot_state_position = np.asarray(robot_state_position, dtype=np.float64)
    robot_state_velocity = np.asarray(robot_state_velocity, dtype=np.float64)
    robot_action = np.asarray(robot_action, dtype=np.float64)
    if x.shape != (4,) or robot_state_position.shape != (2,) or robot_state_velocity.shape != (2,) or robot_action.shape != (2,):
        raise ValueError(
            f"rollout step shapes must be (4,), (2,), (2,), (2,), got "
            f"{x.shape}, {robot_state_position.shape}, {robot_state_velocity.shape}, {robot_action.shape}"
        )
    if not all(np.all(np.isfinite(v)) for v in (x, robot_state_position, robot_state_velocity, robot_action)):
        raise ValueError("rollout step inputs contain NaN or Inf")
    v_current = x[2:4].copy()
    context = compute_context_features(x[:2], v_current, robot_state_position, robot_state_velocity)
    return v_current, robot_action.copy(), context


def _moment_matched_mixture_gaussian(
    prior: np.ndarray, v_prev: np.ndarray, u_r_h: np.ndarray, context_h: np.ndarray, artifact: ARHMMArtifact,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the TRUE mixture distribution
    ``sum_k prior[k] * N(mean_k, Q_k)`` -- i.e. law of total variance:

        mean = sum_k prior[k] * mean_k
        cov  = sum_k prior[k] * (Q_k + (mean_k - mean)(mean_k - mean)^T)
             = within-mode covariance + between-mode covariance

    Fixes B1 from the 2026-08-03 independent audit (carried over from the
    legacy ``ModeModelArtifact``-based version): an earlier version of
    ``sample_posterior_mean`` used only the within-mode term, understating
    the fixture's true mixture variance by ~91x -- silently manufacturing
    an artificially narrow "mean" ablation that would make ``full_posterior``
    look better than a FAIR moment-matched unimodal baseline actually is.
    Returns the ABSOLUTE mean next-velocity (not a delta)."""
    mode_means = np.stack([_mode_mean_velocity(v_prev, u_r_h, context_h, artifact, k) for k in range(artifact.K)])
    mean = np.sum(prior[:, None] * mode_means, axis=0)
    centered = mode_means - mean[None, :]
    total_cov = sum(
        prior[k] * (artifact.Q[k] + np.outer(centered[k], centered[k]))
        for k in range(artifact.K)
    )
    return mean, total_cov


def _rollout_one(
    z0: int,
    fixed_z: bool,
    state0: np.ndarray,  # [px, py, vx, vy]
    artifact: ARHMMArtifact,
    robot_state_position_sequence: np.ndarray,  # [H, 2] -- PRE-action robot position at each step
    robot_state_velocity_sequence: np.ndarray,  # [H, 2] -- PRE-action robot velocity at each step
    robot_action_sequence: np.ndarray,          # [H, 2] -- u_r[h], the action APPLIED during step h
    horizon: int,
    rng: np.random.Generator,
    max_speed: float = DEFAULT_MAX_SPEED,
    max_acceleration: float = DEFAULT_MAX_ACCELERATION,
    return_modes: bool = False,
):
    """One sample trajectory: [H, 2] positions (and, if requested, the
    [H] mode path actually taken -- used by selftest to verify sampled
    transition frequencies match Pi without needing a separate
    reimplementation of the same logic). ``context[h]`` is recomputed EVERY
    step from ``robot_state_position_sequence[h]``/
    ``robot_state_velocity_sequence[h]`` (never frozen from the rollout's
    starting conditions), and ``robot_action_sequence[h]`` supplies
    ``u_r[h]`` -- Order R1: these are three SEPARATE arrays, never one
    array standing in for two different roles."""
    x = state0.copy()
    z = z0
    traj = np.zeros((horizon, 2))
    modes = np.zeros(horizon, dtype=np.int64)
    for h in range(horizon):
        modes[h] = z
        v_prev, u_r_h, context_h = build_rollout_step_inputs(
            x, robot_state_position_sequence[h], robot_state_velocity_sequence[h], robot_action_sequence[h],
        )
        mean_v = _mode_mean_velocity(v_prev, u_r_h, context_h, artifact, z)
        noise = rng.multivariate_normal(np.zeros(2), artifact.Q[z])
        delta_v = (mean_v - v_prev) + noise
        v_new = _clip_delta_v_and_speed(v_prev, delta_v, artifact.dt, max_speed, max_acceleration)
        p_new = x[:2] + artifact.dt * v_new
        x = np.array([p_new[0], p_new[1], v_new[0], v_new[1]])
        traj[h] = x[:2]
        # z0 is the mode for the first emitted transition. Only after that
        # transition has been generated may the Markov chain advance for the
        # next horizon step.
        if not fixed_z and h + 1 < horizon:
            z = int(rng.choice(artifact.K, p=artifact.Pi[z]))
    if return_modes:
        return traj, modes
    return traj


def sample_full_posterior(
    posterior, state0, artifact,
    robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence,
    horizon, num_samples, rng,
    max_speed: float = DEFAULT_MAX_SPEED, max_acceleration: float = DEFAULT_MAX_ACCELERATION,
) -> np.ndarray:
    _validate_arhmm_rollout_inputs(
        posterior, state0, artifact, robot_state_position_sequence,
        robot_state_velocity_sequence, robot_action_sequence, horizon, num_samples,
    )
    trajs = np.zeros((num_samples, horizon, 2))
    for m in range(num_samples):
        z0 = int(rng.choice(artifact.K, p=posterior))
        trajs[m] = _rollout_one(
            z0, fixed_z=False, state0=state0, artifact=artifact,
            robot_state_position_sequence=robot_state_position_sequence,
            robot_state_velocity_sequence=robot_state_velocity_sequence,
            robot_action_sequence=robot_action_sequence,
            horizon=horizon, rng=rng, max_speed=max_speed, max_acceleration=max_acceleration,
        )
    return trajs


def sample_full_posterior_with_modes(
    posterior, state0, artifact,
    robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence,
    horizon, num_samples, rng,
    max_speed: float = DEFAULT_MAX_SPEED, max_acceleration: float = DEFAULT_MAX_ACCELERATION,
):
    """Testing/diagnostic variant of ``sample_full_posterior`` that also
    returns each sample's initial mode choice ``z0`` [M] and full mode path
    [M, H] -- used by selftest to verify sampled mode frequencies match the
    given posterior and sampled transitions match Pi (guide.md 8.1 items
    8-9). Not used by the policy at runtime."""
    _validate_arhmm_rollout_inputs(
        posterior, state0, artifact, robot_state_position_sequence,
        robot_state_velocity_sequence, robot_action_sequence, horizon, num_samples,
    )
    trajs = np.zeros((num_samples, horizon, 2))
    z0_choices = np.zeros(num_samples, dtype=np.int64)
    mode_paths = np.zeros((num_samples, horizon), dtype=np.int64)
    for m in range(num_samples):
        z0 = int(rng.choice(artifact.K, p=posterior))
        z0_choices[m] = z0
        traj, modes = _rollout_one(
            z0, fixed_z=False, state0=state0, artifact=artifact,
            robot_state_position_sequence=robot_state_position_sequence,
            robot_state_velocity_sequence=robot_state_velocity_sequence,
            robot_action_sequence=robot_action_sequence,
            horizon=horizon, rng=rng, max_speed=max_speed, max_acceleration=max_acceleration,
            return_modes=True,
        )
        trajs[m] = traj
        mode_paths[m] = modes
    return trajs, z0_choices, mode_paths


def sample_map_mode(
    posterior, state0, artifact,
    robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence,
    horizon, num_samples, rng,
    max_speed: float = DEFAULT_MAX_SPEED, max_acceleration: float = DEFAULT_MAX_ACCELERATION,
) -> np.ndarray:
    _validate_arhmm_rollout_inputs(
        posterior, state0, artifact, robot_state_position_sequence,
        robot_state_velocity_sequence, robot_action_sequence, horizon, num_samples,
    )
    z0 = int(np.argmax(posterior))
    trajs = np.zeros((num_samples, horizon, 2))
    for m in range(num_samples):
        trajs[m] = _rollout_one(
            z0, fixed_z=True, state0=state0, artifact=artifact,
            robot_state_position_sequence=robot_state_position_sequence,
            robot_state_velocity_sequence=robot_state_velocity_sequence,
            robot_action_sequence=robot_action_sequence,
            horizon=horizon, rng=rng, max_speed=max_speed, max_acceleration=max_acceleration,
        )
    return trajs


def sample_uniform_posterior(
    posterior, state0, artifact,
    robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence,
    horizon, num_samples, rng,
    max_speed: float = DEFAULT_MAX_SPEED, max_acceleration: float = DEFAULT_MAX_ACCELERATION,
) -> np.ndarray:
    uniform = np.full(artifact.K, 1.0 / artifact.K)
    return sample_full_posterior(
        uniform, state0, artifact,
        robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence,
        horizon, num_samples, rng, max_speed, max_acceleration,
    )


def sample_shuffled_posterior(
    shuffled_posterior, state0, artifact,
    robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence,
    horizon, num_samples, rng,
    max_speed: float = DEFAULT_MAX_SPEED, max_acceleration: float = DEFAULT_MAX_ACCELERATION,
) -> np.ndarray:
    """Identical mechanics to ``sample_full_posterior``; the caller is
    responsible for passing a DIFFERENT track's belief (fixed for the whole
    episode -- see ``shuffle_track_beliefs``), not this function."""
    return sample_full_posterior(
        shuffled_posterior, state0, artifact,
        robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence,
        horizon, num_samples, rng, max_speed, max_acceleration,
    )


def sample_posterior_mean(
    posterior, state0, artifact,
    robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence,
    horizon, num_samples, rng,
    max_speed: float = DEFAULT_MAX_SPEED, max_acceleration: float = DEFAULT_MAX_ACCELERATION,
) -> np.ndarray:
    """BANNED from the main method (guide.md 2.2/6.5) -- ablation only.
    Every step samples from the TRUE moment-matched mixture Gaussian (mean
    AND full covariance including between-mode spread -- see
    ``_moment_matched_mixture_gaussian``), collapsing whatever multimodality
    exists into ONE Gaussian per step before BRNE sees it. The belief's own
    marginal is propagated forward through Pi each step, but the mixture is
    never actually sampled as a mixture."""
    _validate_arhmm_rollout_inputs(
        posterior, state0, artifact, robot_state_position_sequence,
        robot_state_velocity_sequence, robot_action_sequence, horizon, num_samples,
    )
    trajs = np.zeros((num_samples, horizon, 2))
    for m in range(num_samples):
        x = state0.copy()
        prior = posterior.copy()
        for h in range(horizon):
            v_prev = x[2:4].copy()
            u_r_h = robot_action_sequence[h]
            context_h = compute_context_features(
                x[:2], v_prev, robot_state_position_sequence[h], robot_state_velocity_sequence[h],
            )
            mean_v, mixture_cov = _moment_matched_mixture_gaussian(prior, v_prev, u_r_h, context_h, artifact)
            noise = rng.multivariate_normal(np.zeros(2), mixture_cov)
            delta_v = (mean_v - v_prev) + noise
            v_new = _clip_delta_v_and_speed(v_prev, delta_v, artifact.dt, max_speed, max_acceleration)
            p_new = x[:2] + artifact.dt * v_new
            x = np.array([p_new[0], p_new[1], v_new[0], v_new[1]])
            trajs[m, h] = x[:2]
            if h + 1 < horizon:
                prior = prior @ artifact.Pi
                prior = np.maximum(prior, 1e-12)
                prior /= prior.sum()
    return trajs


def sample_cv(state0, horizon, num_samples, dt, brne_root, rng) -> np.ndarray:
    """Constant-velocity mean + the SAME Gaussian-process KERNEL upstream
    BRNE's own demo uses, but sampled with THIS call's own ``rng`` argument
    -- the fair, apples-to-apples, REPRODUCIBLE CV-BRNE ablation baseline.
    Independent of the AR-HMM entirely (no robot/context inputs needed --
    this ablation deliberately does not react to the robot at all).

    Fixes B4 from the 2026-08-03 independent audit: an earlier version
    called upstream's ``mvn_sample_normal``, which internally samples from
    upstream MODULE-LEVEL global ``rng`` (``brne.py``'s own
    ``rng = np.random.default_rng(1)``), completely ignoring the ``rng``
    argument passed into this function -- two calls with "the same seed"
    produced DIFFERENT results (max trajectory diff 2.1999 on independent
    re-test), and every ablation sharing the upstream global RNG silently
    stole draws from each other's stream. This version reuses upstream's
    covariance/L-matrix construction (``get_Lmat_nb``, which has no
    randomness) but samples the standard normals itself.
    """
    from crowd_nav.bayesian_brne.brne_adapter import load_upstream_brne

    upstream = load_upstream_brne(brne_root)
    tlist = np.arange(horizon) * dt
    train_ts = np.array([tlist[0]])
    train_noise = np.array([1e-2])
    cov_lmat, _cov = upstream.get_Lmat_nb(train_ts, tlist, train_noise)

    base_x = rng.standard_normal(size=(horizon, num_samples))
    base_y = rng.standard_normal(size=(horizon, num_samples))
    x_pts = (cov_lmat @ base_x).T
    y_pts = (cov_lmat @ base_y).T

    speed = float(np.hypot(state0[2], state0[3]))
    x_mean = state0[0] + tlist * state0[2]
    y_mean = state0[1] + tlist * state0[3]
    trajs = np.zeros((num_samples, horizon, 2))
    trajs[:, :, 0] = x_mean[None, :] + x_pts * max(speed, 1e-3)
    trajs[:, :, 1] = y_mean[None, :] + y_pts * max(speed, 1e-3)
    return trajs


SAMPLING_MODES = (
    "full_posterior", "map_mode", "posterior_mean",
    "uniform_posterior", "shuffled_posterior", "cv",
)


def _validate_arhmm_rollout_inputs(
    posterior: np.ndarray,
    state0: np.ndarray,
    artifact: ARHMMArtifact,
    robot_state_position_sequence: np.ndarray,
    robot_state_velocity_sequence: np.ndarray,
    robot_action_sequence: np.ndarray,
    horizon: int,
    num_samples: int,
) -> None:
    """Fail closed on the complete AR-HMM rollout contract."""
    if artifact is None:
        raise ValueError("AR-HMM sampling requires a non-null artifact")
    horizon = int(horizon)
    num_samples = int(num_samples)
    if horizon < 1 or num_samples < 1:
        raise ValueError(f"horizon and num_samples must be positive, got {horizon}, {num_samples}")
    state0 = np.asarray(state0, dtype=np.float64)
    posterior = np.asarray(posterior, dtype=np.float64)
    if state0.shape != (4,):
        raise ValueError(f"state0 must have shape (4,), got {state0.shape}")
    if posterior.shape != (artifact.K,):
        raise ValueError(f"posterior must have shape ({artifact.K},), got {posterior.shape}")
    if not np.all(np.isfinite(state0)) or not np.all(np.isfinite(posterior)):
        raise ValueError("state0/posterior contain NaN or Inf")
    if np.any(posterior < 0.0) or not np.isclose(float(posterior.sum()), 1.0, atol=1e-8):
        raise ValueError(f"posterior must be non-negative and normalized, got {posterior}")
    for name, values in (
        ("robot_state_position_sequence", robot_state_position_sequence),
        ("robot_state_velocity_sequence", robot_state_velocity_sequence),
        ("robot_action_sequence", robot_action_sequence),
    ):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (horizon, 2):
            raise ValueError(f"{name} must have shape ({horizon}, 2), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or Inf")


def sample_pedestrian_trajectories(
    mode: str,
    *,
    posterior: Optional[np.ndarray] = None,
    state0: np.ndarray,
    artifact: Optional[ARHMMArtifact] = None,
    robot_state_position_sequence: Optional[np.ndarray] = None,
    robot_state_velocity_sequence: Optional[np.ndarray] = None,
    robot_action_sequence: Optional[np.ndarray] = None,
    horizon: int,
    num_samples: int,
    rng: np.random.Generator,
    dt: float = 0.25,
    brne_root: Optional[str] = None,
    max_speed: float = DEFAULT_MAX_SPEED,
    max_acceleration: float = DEFAULT_MAX_ACCELERATION,
) -> np.ndarray:
    if mode == "full_posterior":
        return sample_full_posterior(posterior, state0, artifact, robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence, horizon, num_samples, rng, max_speed, max_acceleration)
    if mode == "map_mode":
        return sample_map_mode(posterior, state0, artifact, robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence, horizon, num_samples, rng, max_speed, max_acceleration)
    if mode == "posterior_mean":
        return sample_posterior_mean(posterior, state0, artifact, robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence, horizon, num_samples, rng, max_speed, max_acceleration)
    if mode == "uniform_posterior":
        return sample_uniform_posterior(posterior, state0, artifact, robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence, horizon, num_samples, rng, max_speed, max_acceleration)
    if mode == "shuffled_posterior":
        return sample_shuffled_posterior(posterior, state0, artifact, robot_state_position_sequence, robot_state_velocity_sequence, robot_action_sequence, horizon, num_samples, rng, max_speed, max_acceleration)
    if mode == "cv":
        return sample_cv(state0, horizon, num_samples, dt, brne_root, rng)
    raise ValueError(f"unknown sampling mode: {mode!r}; must be one of {SAMPLING_MODES}")


def shuffle_track_beliefs(
    track_ids: list, beliefs_by_track: Dict[int, np.ndarray], rng: np.random.Generator
) -> Dict[int, np.ndarray]:
    """Fixed-per-episode permutation of track_id -> another track's belief
    (guide.md 6.5: sampled once per episode, never re-shuffled per step)."""
    ids = list(track_ids)
    permuted = ids.copy()
    rng.shuffle(permuted)
    return {tid: beliefs_by_track[other] for tid, other in zip(ids, permuted)}
