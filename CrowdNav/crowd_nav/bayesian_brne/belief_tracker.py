"""BeliefBank: per-track online Bayesian posterior over the action-conditioned
sticky AR-HMM's discrete modes (Order F2, 2026-08-03 -- guide.md section 4.2).

Single public per-timestep entry point:

    class BeliefBank:
        def reset(self, episode_seed): ...
        def update(self, observations, robot_action, robot_px, robot_py, robot_vx, robot_vy, current_timestamp) -> None: ...
        def predictive_mode_distribution(self, track_id: int) -> np.ndarray: ...  # [K]

Replaces the earlier ``ModeModelArtifact``-based version: this tracker now
accepts ONLY ``action_conditioned_arhmm.ARHMMArtifact`` and its emission model
``v_h[t] ~ N(A_k v_h[t-1] + B_k u_r[t-1] + C_k context[t-1] + d_k, Q_k)`` --
the SAME model trained offline, so training and online deployment share
exactly one mode definition, never two.

``robot_action`` passed to ``update()`` MUST be the robot's ACTUALLY EXECUTED
action that produced the transition INTO the observations passed in the SAME
call (i.e. ``u_r[t-1]`` in the pseudocode below) -- NEVER a candidate/
hypothetical action still being considered by an outer planning loop (e.g.
Order F4's BRNE fixed-point iterations). Feeding a candidate action into this
function to "explain" a transition that has already happened is exactly the
bug this module's own docstring and test suite (see
``arhmm_belief_action_misalignment_degrades_recovery`` in selftest.py) are
built to catch -- the model is deliberately SENSITIVE to getting the u_r
timing right, which only means anything if the tracker actually treats the
correctly-timed action differently from an incorrectly-timed one.

Per-track update rule (guide.md 4.2):

    b_next(t) = p(z_{t+1} | observations through t)

The stored ``b_next(t-1)`` is the direct prior for the next observed
transition. After a consecutive-frame likelihood update, ``Pi`` is applied
once to produce ``b_next(t)``. This explicit next-transition convention keeps
the first rollout step from receiving an accidental extra transition.

Each track stores exactly:

    next_mode_prior[K]                 # p(z_{t+1} | observations through t)
    previous_human_velocity[2]   (None for a brand-new/just-reappeared track)
    previous_robot_action[2]
    previous_context[len(CONTEXT_FEATURE_NAMES)]
    last_timestamp
    missed_steps

A gap (a track not present in ``observations`` this call) advances that
track's next-mode prior by the Bayes transition EXACTLY ONCE per missed real timestep
(never a separate "predict" call in addition to ``update([])`` -- there is
only one entry point). The moment a track is missed, its
``previous_human_velocity``/``previous_robot_action``/``previous_context``
are cleared to ``None``; its next reappearance advances the next-mode prior
once for that real timestep, but applies NO likelihood term on that first
reappearing frame, since the gap's true duration is unknown/unmodeled and
would otherwise silently mismatch ``Q_k``, which is calibrated for exactly
one real timestep). Only the FOLLOWING frame (now a genuine consecutive
pair) resumes normal likelihood updates.

Identity handling: predictive distributions are keyed by ``track_id``, never by
"current nearest-distance order" or list position. A track missing for more
than ``max_missed_steps`` is deleted; a later reappearance under the same ID
starts from ``artifact.initial_distribution``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
    ARHMMArtifact, CONTEXT_FEATURE_NAMES, compute_context_features,
)
from crowd_nav.bayesian_brne.schemas import TrackObservation


class BeliefTimestampError(ValueError):
    """Raised by ``BeliefBank.update`` (Order R1, 2026-08-03) when a track's
    new timestamp is duplicate/backwards relative to its own last-seen
    timestamp, or the gap is not a (near-)integer multiple of
    ``artifact.dt`` -- fail closed rather than silently computing a
    likelihood calibrated for a completely different elapsed time."""


def _build_context(
    obs: TrackObservation,
    robot_px: float, robot_py: float, robot_vx: float, robot_vy: float,
) -> np.ndarray:
    """Thin wrapper around ``action_conditioned_arhmm.compute_context_features``
    (Order F3, 2026-08-03: factored out to the ONE canonical implementation,
    since this formula was previously duplicated independently here,
    in ``mode_model.extract_transitions``, and in
    ``action_conditioned_arhmm._build_sequence`` -- three copies of the same
    formula is exactly the kind of drift risk that could silently make
    online belief tracking disagree with what the model was fit on)."""
    return compute_context_features(
        np.array([obs.px, obs.py]), np.array([obs.vx, obs.vy]),
        np.array([robot_px, robot_py]), np.array([robot_vx, robot_vy]),
    )


def _emission_log_prob(
    v_next: np.ndarray, v_current: np.ndarray, u_robot: np.ndarray, context: np.ndarray,
    A_k: np.ndarray, B_k: np.ndarray, C_k: np.ndarray, d_k: np.ndarray, Q_k: np.ndarray,
) -> float:
    """log N(v_next; A_k v_current + B_k u_robot + C_k context + d_k, Q_k) --
    the same per-row emission density ``action_conditioned_arhmm.
    _emission_log_probs`` computes (vectorized there, scalar here since
    online tracking is inherently one step at a time). Deliberately does
    NOT add any covariance floor: Order F1/the U3-monotonicity investigation
    (2026-08-03) found that silently flooring the covariance used to score
    a likelihood, while the OFFLINE model that produced ``Q_k`` uses the
    raw (unfloored) value, breaks the exact correspondence between what
    training optimizes for and what gets scored online -- the same class of
    bug that broke EM's own monotonicity guarantee when unaddressed."""
    mean = A_k @ v_current + B_k @ u_robot + C_k @ context + d_k
    diff = v_next - mean
    sign, logdet = np.linalg.slogdet(Q_k)
    inv = np.linalg.inv(Q_k)
    return float(-0.5 * (diff @ inv @ diff + logdet + len(v_next) * np.log(2.0 * np.pi)))


@dataclass
class _TrackState:
    next_mode_prior: np.ndarray                  # [K], p(z_{t+1}|history through t)
    previous_human_velocity: Optional[np.ndarray]  # [2], None if next update must skip the likelihood term
    previous_robot_action: Optional[np.ndarray]    # [2]
    previous_context: Optional[np.ndarray]         # [len(CONTEXT_FEATURE_NAMES)]
    missed_steps: int
    last_timestamp: float


class BeliefBank:
    def __init__(self, artifact: ARHMMArtifact, max_missed_steps: int = 5, seed: int = 2407):
        self.artifact = artifact
        self.K = artifact.K
        self.max_missed_steps = int(max_missed_steps)
        self._rng = np.random.default_rng(seed)  # reserved; unused by the deterministic filter itself
        self._tracks: Dict[int, _TrackState] = {}
        self._current_timestamp: Optional[float] = None

    def reset(self, episode_seed: int) -> None:
        self._tracks.clear()
        self._rng = np.random.default_rng(episode_seed)
        self._current_timestamp = None

    def _predict_prior(self, posterior: np.ndarray) -> np.ndarray:
        prior = posterior @ self.artifact.Pi
        prior = np.maximum(prior, 1e-12)
        return prior / prior.sum()

    def update(
        self,
        observations: List[TrackObservation],
        robot_action: Tuple[float, float],
        robot_px: float, robot_py: float,
        robot_vx: float, robot_vy: float,
        current_timestamp: float,
    ) -> None:
        """The ONLY per-timestep entry point. ``robot_action`` MUST be the
        action ACTUALLY EXECUTED since the last call to this method (i.e.
        ``u_r[t-1]`` -- see module docstring) -- it is ALREADY known at
        call time (the environment executed it before producing the new
        observations passed in THIS SAME call), so it is used IMMEDIATELY
        here, paired with the STORED ``previous_human_velocity``/
        ``previous_context`` (from the LAST call, i.e. ``v_h[t-1]``/
        ``context[t-1]``) to explain the current ``v_now`` (``v_h[t]``) --
        it is NOT deferred/stored to be consumed one call later (an earlier
        version of this method did exactly that, introducing a spurious
        extra one-call delay that was caught by
        ``arhmm_belief_action_misalignment_degrades_recovery`` in
        selftest.py: feeding the CORRECT, already-known ``u_r[t-1]``
        produced LOWER mode-recovery confidence than deliberately feeding a
        one-step-stale action, exactly backwards from what a correct
        implementation should do). ``robot_px/py/vx/vy`` are the robot's
        CURRENT position/velocity, used to build ``context[t]`` for each
        track (stored for use explaining the NEXT observation). Pass every
        track observed THIS step in ``observations`` (an empty list if none
        were); every track this ``BeliefBank`` currently knows about that is
        NOT in that list is automatically advanced by the Bayes prior
        alone. Never call this twice for the same timestep."""
        current_timestamp = float(current_timestamp)
        if not np.isfinite(current_timestamp):
            raise BeliefTimestampError(f"current_timestamp must be finite, got {current_timestamp!r}")
        if self._current_timestamp is not None:
            global_steps = (current_timestamp - self._current_timestamp) / self.artifact.dt
            if abs(global_steps - 1.0) > 1e-6:
                raise BeliefTimestampError(
                    f"global timestamp must advance by exactly artifact.dt={self.artifact.dt}; "
                    f"previous={self._current_timestamp}, current={current_timestamp}, "
                    f"gap/dt={global_steps}"
                )
        for obs in observations:
            if abs(float(obs.timestamp) - current_timestamp) > 1e-6:
                raise BeliefTimestampError(
                    f"track_id={obs.track_id}: observation timestamp {obs.timestamp} does not match "
                    f"current_timestamp={current_timestamp}"
                )

        robot_action_now = np.asarray(robot_action, dtype=np.float64)
        if robot_action_now.shape != (2,) or not np.all(np.isfinite(robot_action_now)):
            raise BeliefTimestampError(f"robot_action must be finite shape [2], got {robot_action_now}")
        seen_ids = set()
        for obs in observations:
            seen_ids.add(obs.track_id)
            state = self._tracks.get(obs.track_id)
            v_now = np.array([obs.vx, obs.vy], dtype=np.float64)
            context_now = _build_context(obs, robot_px, robot_py, robot_vx, robot_vy)

            # Order R1: fail closed on a duplicate/backwards/non-integer-gap
            # timestamp for a track we already have state for -- silently
            # scoring the likelihood as if exactly one ``artifact.dt`` had
            # elapsed, when it actually hadn't, would misuse Q_k/A_k/B_k/C_k
            # (all calibrated for exactly one real timestep) without any
            # signal that anything was wrong.
            if state is not None:
                gap = obs.timestamp - state.last_timestamp
                if gap <= 0:
                    raise BeliefTimestampError(
                        f"track_id={obs.track_id}: new timestamp {obs.timestamp} is not strictly "
                        f"greater than last-seen timestamp {state.last_timestamp} (gap={gap})"
                    )
                dt = self.artifact.dt
                n_steps = gap / dt
                if abs(n_steps - round(n_steps)) > 1e-6 or round(n_steps) < 1:
                    raise BeliefTimestampError(
                        f"track_id={obs.track_id}: gap {gap} between timestamps {state.last_timestamp} -> "
                        f"{obs.timestamp} is not a positive integer multiple of artifact.dt={dt} "
                        f"(gap/dt={n_steps}); a missed step must be reported via absence from "
                        "``observations``, never via a skipped timestamp on the same track"
                    )
                if round(n_steps) > 1 and state.previous_human_velocity is not None:
                    # A gap of exactly N>1 real timesteps is only legitimate
                    # when THIS tracker's own bookkeeping already knows the
                    # track was missing (state.previous_human_velocity ==
                    # None, cleared by the missed-step loop below on a prior
                    # call) -- the prior was already correctly advanced once
                    # per missed step in that case. If previous_human_velocity
                    # is STILL set, this track was continuously present per
                    # this tracker's own records, so a gap >1 step can only
                    # mean the caller skipped calling ``update`` entirely for
                    # some real steps while claiming continuity, which this
                    # class cannot silently paper over.
                    raise BeliefTimestampError(
                        f"track_id={obs.track_id}: timestamp gap spans {round(n_steps)} artifact.dt steps "
                        f"({state.last_timestamp} -> {obs.timestamp}) while continuously present (never "
                        "missing) per this BeliefBank's own records -- BeliefBank.update() must be called "
                        "every real timestep; a genuinely missed sighting must be represented by the "
                        "track's ABSENCE from ``observations``, not a stretched timestamp gap"
                    )

            if state is None or state.previous_human_velocity is None:
                # Fresh start: brand-new track, OR a track reappearing after
                # a miss. No likelihood term is applied here (see module
                # docstring) -- the posterior is carried over unchanged (it
                # was already correctly advanced by the prior once per
                # missed step); we only seed previous_* for the FOLLOWING
                # frame's likelihood. previous_robot_action is stored here
                # purely for diagnostics/introspection -- it is NEVER read
                # back to compute a likelihood (see above); the likelihood
                # always uses THIS call's ``robot_action_now`` directly.
                if state is None:
                    # No past observation exists: use the fitted initial
                    # distribution, not an arbitrary uniform replacement.
                    next_mode_prior = np.asarray(
                        self.artifact.initial_distribution, dtype=np.float64
                    ).copy()
                else:
                    # A reappearance is a real current timestep but has no
                    # valid previous human state for an emission likelihood.
                    next_mode_prior = self._predict_prior(state.next_mode_prior)
                self._tracks[obs.track_id] = _TrackState(
                    next_mode_prior=next_mode_prior,
                    previous_human_velocity=v_now, previous_robot_action=robot_action_now,
                    previous_context=context_now, missed_steps=0, last_timestamp=current_timestamp,
                )
                continue

            # Genuinely consecutive real frame: normal one-step Bayes update
            # using STORED v_h[t-1]/context[t-1] and THIS CALL'S u_r[t-1]
            # (robot_action_now, used immediately, never deferred) to
            # explain v_now (= v_h[t]).
            prior = state.next_mode_prior
            log_terms = np.empty(self.K)
            for k in range(self.K):
                log_terms[k] = np.log(prior[k]) + _emission_log_prob(
                    v_now, state.previous_human_velocity, robot_action_now, state.previous_context,
                    self.artifact.A[k], self.artifact.B[k], self.artifact.C[k], self.artifact.d[k], self.artifact.Q[k],
                )
            m = log_terms.max()
            likelihood = np.exp(log_terms - m)
            filtered = likelihood / max(float(likelihood.sum()), 1e-12)
            next_mode_prior = self._predict_prior(filtered)

            self._tracks[obs.track_id] = _TrackState(
                next_mode_prior=next_mode_prior,
                previous_human_velocity=v_now, previous_robot_action=robot_action_now,
                previous_context=context_now, missed_steps=0, last_timestamp=current_timestamp,
            )

        for track_id, state in list(self._tracks.items()):
            if track_id not in seen_ids:
                state.missed_steps += 1
                if state.missed_steps > self.max_missed_steps:
                    del self._tracks[track_id]
                else:
                    state.next_mode_prior = self._predict_prior(state.next_mode_prior)
                    state.last_timestamp = current_timestamp
                    # Mark that the NEXT sighting must be treated as a fresh
                    # start, not a stale multi-step gap masquerading as one
                    # real timestep.
                    state.previous_human_velocity = None
                    state.previous_robot_action = None
                    state.previous_context = None

        self._current_timestamp = current_timestamp

    def predictive_mode_distribution(self, track_id: int) -> np.ndarray:
        state = self._tracks.get(track_id)
        if state is None:
            return np.asarray(self.artifact.initial_distribution, dtype=np.float64).copy()
        return state.next_mode_prior.copy()

    def posterior(self, track_id: int) -> np.ndarray:
        """Deprecated alias for the next-transition distribution."""
        return self.predictive_mode_distribution(track_id)

    def active_track_ids(self) -> List[int]:
        return list(self._tracks.keys())
