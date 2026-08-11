"""Posterior world sampler (guide.md section 3.2/A5).

Samples hypothetical next human states from a track's CURRENT belief,
shared across all 80 candidate robot actions via common random numbers
(guide.md 5.4: "所有动作共享同一组世界随机数...防止某动作只因抽到更幸运
的行人样本而胜出"). Robot action does NOT affect this sampling (guide.md
4.5: robot action is not in the human transition model in the main
version) -- it only affects downstream relative-geometry/reward via
transition.py.

Five posterior "sources" share the exact same code path here, differing
only in which mode-probability vector is used to pick a mode and where
the sampled residual comes from (guide.md A5 acceptance: "full/MAP/
moment-mean/CV/shuffled五种模式走同一代码路径，只替换posterior source"):

    full         the track's real belief vector
    map          one-hot at argmax(belief)
    moment_mean  a single "average mode": use the belief-weighted mean/
                 cov directly as one Gaussian instead of sampling a mode
    cv           always mode 0 (CV), ignoring the belief entirely
    shuffled     a real belief vector taken from a DIFFERENT, randomly
                 chosen track/step (necessity-ablation negative control)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.world_model import N_MODES, SBKHMMArtifact, sample_sign_truncated_predictive

POSTERIOR_SOURCES = ("full", "map", "moment_mean", "cv", "shuffled")


def _stable_seed(*parts: int) -> int:
    """Cross-process deterministic seed; Python's built-in hash is salted
    for strings and not spec-guaranteed stable across versions/platforms
    even for pure-integer tuples (guide.md R4-3 point 8: "不应该依赖
    未冻结的Python hash实现"). Moved here (originally in policy.py) so
    both policy.py and this module -- and now counterfactual.py -- share
    ONE seeding function; policy.py already imports from this module, so
    importing it back the other way would be circular."""
    blob = ":".join(str(int(part)) for part in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2**31 - 1)


class RolloutError(ValueError):
    pass


@dataclass(frozen=True)
class SampledHumanNextState:
    track_id: int
    sampled_mode: Optional[int]  # None for moment_mean (no single mode is drawn)
    a_parallel: float
    omega: float
    next_speed: float
    next_heading: float
    next_position: np.ndarray  # [2]


def _mode_probabilities(belief: np.ndarray, source: str, rng: np.random.Generator, shuffled_pool: Optional[Sequence[np.ndarray]]) -> np.ndarray:
    if source == "full":
        return belief
    if source == "map":
        onehot = np.zeros(N_MODES)
        onehot[int(np.argmax(belief))] = 1.0
        return onehot
    if source == "cv":
        onehot = np.zeros(N_MODES)
        onehot[0] = 1.0  # CV = mode index 0
        return onehot
    if source == "shuffled":
        if not shuffled_pool:
            raise RolloutError("source='shuffled' requires a non-empty shuffled_pool of other beliefs")
        idx = rng.integers(0, len(shuffled_pool))
        return shuffled_pool[idx]
    raise RolloutError(f"moment_mean has no discrete mode distribution; use _moment_mean_sample instead")


def _integrate_holonomic(
    old_speed: float, old_heading: float, a_parallel: float, omega: float,
    position: np.ndarray, dt: float, max_speed: float,
) -> Tuple[float, float, np.ndarray]:
    """Bit-for-bit guide.md 4.3 integration:
        speed'    = clip(speed + a_parallel*dt, 0, max_speed)
        heading'  = wrap(heading + omega*dt)
        velocity' = speed' * [cos(heading'), sin(heading')]
        position' = position + 0.5*(velocity + velocity')*dt
    (trapezoidal average of the OLD and NEW velocity, not just the new one)."""
    new_speed = float(np.clip(old_speed + a_parallel * dt, 0.0, max_speed))
    new_heading = float((old_heading + omega * dt + np.pi) % (2 * np.pi) - np.pi)
    velocity_old = old_speed * np.array([np.cos(old_heading), np.sin(old_heading)])
    velocity_new = new_speed * np.array([np.cos(new_heading), np.sin(new_heading)])
    next_position = position + 0.5 * (velocity_old + velocity_new) * dt
    return new_speed, new_heading, next_position


def sample_human_next_states(
    artifact: SBKHMMArtifact,
    track_beliefs: Dict[int, np.ndarray],
    track_positions: Dict[int, np.ndarray],
    track_speeds: Dict[int, float],
    track_headings: Dict[int, float],
    n_samples: int,
    dt: float,
    max_human_speed: float,
    source: str,
    seed: Tuple[int, int, int],  # (suite_seed, episode_seed_or_step, sample_index) style key
    shuffled_pool_by_track: Optional[Dict[int, Sequence[np.ndarray]]] = None,
) -> Dict[int, Tuple[SampledHumanNextState, ...]]:
    """Returns {track_id: tuple of n_samples SampledHumanNextState}.
    Deterministic given ``seed`` (guide.md A5: "相同seed逐字节一致，不同
    seed有差异")."""
    if source not in POSTERIOR_SOURCES:
        raise RolloutError(f"unknown posterior source {source!r}, must be one of {POSTERIOR_SOURCES}")

    # guide.md R4-3 point 8: Python's built-in hash() is not spec-guaranteed
    # stable across versions/platforms even for pure-integer tuples; use
    # the project's frozen SHA256-based seed derivation instead.
    rng = np.random.default_rng(_stable_seed(*seed))
    results: Dict[int, Tuple[SampledHumanNextState, ...]] = {}

    for track_id, belief in track_beliefs.items():
        position = track_positions[track_id]
        speed = track_speeds[track_id]
        heading = track_headings[track_id]
        samples = []

        for _ in range(n_samples):
            if source == "moment_mean":
                a_parallel = float(belief @ artifact.emission_mean[:, 0])
                omega = float(belief @ artifact.emission_mean[:, 1])
                sampled_mode = None
            else:
                shuffled_pool = (shuffled_pool_by_track or {}).get(track_id)
                mode_probs = _mode_probabilities(belief, source, rng, shuffled_pool)
                mode_probs = mode_probs / mode_probs.sum()
                sampled_mode = int(rng.choice(N_MODES, p=mode_probs))
                # R1 fix (independent audit B1 repair item 3, 2026-08-06):
                # sample from the mode's actual Student-t posterior
                # predictive with sign-truncated rejection, not a plain
                # Gaussian draw from the (mean, covariance) proxy --
                # the latter could and did admit sign-inconsistent
                # samples (e.g. a negative acceleration for ACC).
                residual = sample_sign_truncated_predictive(rng, artifact.niw(sampled_mode), sampled_mode)
                a_parallel, omega = float(residual[0]), float(residual[1])

            next_speed, next_heading, next_position = _integrate_holonomic(
                speed, heading, a_parallel, omega, position, dt, max_human_speed
            )
            samples.append(SampledHumanNextState(
                track_id=track_id, sampled_mode=sampled_mode,
                a_parallel=a_parallel, omega=omega,
                next_speed=next_speed, next_heading=next_heading,
                next_position=next_position,
            ))
        results[track_id] = tuple(samples)

    return results
