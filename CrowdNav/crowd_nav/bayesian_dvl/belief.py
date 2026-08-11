"""Online per-track exact Bayesian filtering (guide.md section 4.4, A4).

    b_t(z) proportional to  p(u_t | z, artifact) * sum_zprev Pi[zprev,z] b_{t-1}(zprev)

This is exact filtering recursion given the frozen SBK-HMM artifact
(contrast with world_model.py's fit, which is approximate EM). One
``BeliefTracker`` instance owns ALL tracks for one episode; hypothetical
("what if I observe X next") updates must go through
``clone_for_hypothetical`` and never mutate the real tracker.

R1 fix (independent audit B2, 2026-08-06): the previous version tracked
a single ``last_timestamp`` that was NOT advanced on a missing-frame
step, so a track that went missing for one step and then reappeared
got the elapsed gap counted TWICE -- once by the missing-step's own
``_predict_only`` call, and again by ``_update_one`` recomputing
``n_steps`` from the stale timestamp. Reproduced by hand: seed belief
b0, one miss, one reappearance two dt later should give ``b0 @ Pi^2``;
the old code returned ``b0 @ Pi^3``. Fixed by tracking ``filter_time``
(advanced on EVERY call, observed or missing) separately from the
observation bookkeeping (``last_position``/``last_velocity``) used only
for feature/likelihood computation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.world_model import N_MODES, MIN_SPEED_FOR_HEADING, SBKHMMArtifact


class BeliefError(ValueError):
    pass


@dataclass
class _TrackState:
    belief: np.ndarray  # [5], sums to 1
    filter_time: float  # time the belief recursion has been advanced to (every call, observed or missing)
    last_position: np.ndarray  # [2], position at the last point used for feature bookkeeping
    last_velocity: Optional[np.ndarray]  # [2] or None if no valid one-step velocity is available yet
    last_position_is_fresh: bool = True  # False right after any miss: `last_position` is
    # then stale (from before the gap) and must NOT be diffed against
    # the reappearance position -- that would silently divide a
    # multi-step displacement by a single dt. The reappearance call
    # instead just re-baselines `last_position` and waits.
    steps_since_update: int = 0  # 0 = updated this call; >0 = consecutive drops
    track_age_steps: int = 0  # number of update() calls (observed or missing) since creation


class BeliefTracker:
    """Owns per-track belief state for one episode. Not thread-safe;
    one instance per episode, discarded at episode end (guide.md
    A7 acceptance: reset_episode_stats() must clear all tracks)."""

    def __init__(self, artifact: SBKHMMArtifact, missing_timeout_steps: int = 8):
        self.artifact = artifact
        self.missing_timeout_steps = missing_timeout_steps
        self._tracks: Dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._tracks = {}

    def active_track_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._tracks.keys()))

    def belief_for(self, track_id: int) -> np.ndarray:
        if track_id not in self._tracks:
            raise BeliefError(f"unknown track_id {track_id}; call update() first")
        return self._tracks[track_id].belief.copy()

    def _expire_stale_tracks(self, seen_ids) -> None:
        stale = [
            tid for tid, state in self._tracks.items()
            if tid not in seen_ids and state.steps_since_update >= self.missing_timeout_steps
        ]
        for tid in stale:
            del self._tracks[tid]

    def update(self, observations: Dict[int, Tuple[float, np.ndarray]]) -> None:
        """``observations``: {track_id: (timestamp, position[2])}. Order
        of the dict does not affect the result -- each track's update
        depends only on its own history (A4 acceptance: list-order
        invariance)."""
        seen_ids = set(observations.keys())

        # Tracks present before but missing this call: one prediction-only
        # step each (no likelihood term -- guide.md 4.4 "丢帧只做一次
        # transition prediction，不使用旧观测likelihood"). filter_time
        # MUST advance here too, or the elapsed gap gets double-counted
        # when the track reappears (the R1 bug).
        for track_id, state in list(self._tracks.items()):
            if track_id not in seen_ids:
                state.belief = self._predict_only(state.belief)
                state.filter_time += self.artifact.dt
                state.last_velocity = None
                state.last_position_is_fresh = False
                state.steps_since_update += 1
                state.track_age_steps += 1
        self._expire_stale_tracks(seen_ids)

        for track_id, (timestamp, position) in observations.items():
            self._update_one(track_id, timestamp, np.asarray(position, dtype=np.float64))

    def _predict_only(self, belief: np.ndarray) -> np.ndarray:
        predicted = belief @ self.artifact.transition_matrix
        return predicted / predicted.sum()

    def _update_one(self, track_id: int, timestamp: float, position: np.ndarray) -> None:
        if track_id not in self._tracks:
            self._tracks[track_id] = _TrackState(
                belief=self.artifact.initial_distribution.copy(),
                filter_time=timestamp,
                last_position=position,
                last_velocity=None,
                last_position_is_fresh=True,
                steps_since_update=0,
            )
            return

        state = self._tracks[track_id]
        dt_observed = timestamp - state.filter_time
        expected_dt = self.artifact.dt
        if dt_observed <= 0:
            raise BeliefError(
                f"track {track_id}: timestamp did not advance ({state.filter_time} -> {timestamp})"
            )
        # Must be a positive integer multiple of the frozen dt (guide.md
        # 4.4: "时间戳重复、倒退或非dt整数倍间隔直接报错").
        n_steps = round(dt_observed / expected_dt)
        if n_steps < 1 or abs(dt_observed - n_steps * expected_dt) > 1e-6:
            raise BeliefError(
                f"track {track_id}: dt {dt_observed} is not a positive integer multiple of artifact dt {expected_dt}"
            )

        # filter_time already absorbed any missing-step transitions
        # (they advance it immediately in update()), so exactly n_steps
        # transitions remain to apply here -- never n_steps-1-then-1-more.
        belief = state.belief
        for _ in range(n_steps):
            belief = self._predict_only(belief)
        predicted = belief

        if not state.last_position_is_fresh:
            # A miss happened since `last_position` was recorded: it is
            # now stale (from before the gap). Diffing it against the
            # current position would divide a multi-step displacement
            # by a single dt_observed. Re-baseline only; contribute no
            # velocity/feature this call (guide.md 4.4 / R1 fix).
            posterior = predicted
            new_velocity = None
        else:
            velocity = (position - state.last_position) / dt_observed
            speed = float(np.linalg.norm(velocity))
            if state.last_velocity is not None and speed > MIN_SPEED_FOR_HEADING and float(np.linalg.norm(state.last_velocity)) > MIN_SPEED_FOR_HEADING:
                a_parallel = (speed - float(np.linalg.norm(state.last_velocity))) / dt_observed
                prev_heading = float(np.arctan2(state.last_velocity[1], state.last_velocity[0]))
                heading = float(np.arctan2(velocity[1], velocity[0]))
                dheading = heading - prev_heading
                dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
                omega = dheading / dt_observed
                feature = np.array([a_parallel, omega])
                log_emission = self.artifact.emission_log_prob(feature)
                log_emission = log_emission - log_emission.max()
                likelihood = np.exp(log_emission)
                posterior = predicted * likelihood
                total = posterior.sum()
                if total <= 0 or not np.isfinite(total):
                    posterior = predicted  # degenerate likelihood: fall back to prediction only
                else:
                    posterior = posterior / total
            else:
                # Not enough valid history yet for a feature -- prediction only.
                posterior = predicted
            new_velocity = velocity

        self._tracks[track_id] = _TrackState(
            belief=posterior,
            filter_time=timestamp,
            last_position=position,
            last_velocity=new_velocity,
            last_position_is_fresh=True,
            steps_since_update=0,
            track_age_steps=state.track_age_steps + 1,
        )

    def entropy_for(self, track_id: int) -> float:
        """Shannon entropy (nats) of the track's current belief -- part
        of the canonical human feature row (guide.md 5.1: "belief[5],
        entropy, track_age")."""
        belief = self.belief_for(track_id)
        nonzero = belief[belief > 0]
        return float(-(nonzero * np.log(nonzero)).sum())

    def track_age_for(self, track_id: int) -> int:
        if track_id not in self._tracks:
            raise BeliefError(f"unknown track_id {track_id}; call update() first")
        return self._tracks[track_id].track_age_steps

    def predictive_moments_for(self, track_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Belief-weighted one-step predictive mean[2] and symmetric
        covariance[2,2] in [a_parallel, omega] space (guide.md 5.1:
        "one-step predictive mean[2], covariance[3]" -- a 2x2 symmetric
        matrix has 3 unique entries, exposed by the caller via the
        upper triangle). This is a mixture-of-Gaussians moment
        combination over the 5 fixed semantic modes, weighted by the
        CURRENT belief -- not a single mode's own covariance."""
        belief = self.belief_for(track_id)
        emission_mean = self.artifact.emission_mean
        emission_cov = self.artifact.emission_cov
        mean = belief @ emission_mean  # [2]
        diffs = emission_mean - mean  # [5,2]
        between = diffs[:, :, None] * diffs[:, None, :]
        cov = np.sum(belief[:, None, None] * (emission_cov + between), axis=0)
        return mean, cov

    def clone_for_hypothetical(self) -> "BeliefTracker":
        """Deep copy for what-if rollouts (guide.md A4/A5: "hypothetical
        rollout使用belief副本，绝不能污染真实在线tracker"). Mutating the
        clone must never affect self."""
        clone = BeliefTracker(self.artifact, self.missing_timeout_steps)
        clone._tracks = copy.deepcopy(self._tracks)
        return clone
