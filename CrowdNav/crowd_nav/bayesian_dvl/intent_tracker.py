"""Goal-conditioned Bayesian intent posterior -- BDVL final main chain,
block 1 (guide/consolidation plan v1, 2026-08-11).

This is the PREDICTIVE intent belief that the reactive kinematic SBK-HMM
tracker (belief.py) could not provide: it maintains a Bayesian posterior
over a set of PUBLIC candidate destinations, so BEFORE a pedestrian
commits (e.g. at a junction) the posterior stays genuinely multimodal
("could go to any reachable exit"), collapsing only once the observed
motion disambiguates. Validated in reset_exp/step2_goal_intent.py:
pre-fork L/R mass ~0.33 (vs the kinematic tracker's ~0.001), collapse
0-1 step after the fork.

LABEL-LEAKAGE CONTRACT (hard requirement, consolidation plan Decision 2):
this module consumes ONLY observable pedestrian positions + PUBLIC
candidate routes (derived from scene geometry/entries). The hidden true
goal (a CrowdNav human's gx/gy) MUST NEVER be passed here -- it may only
drive the simulated pedestrian and post-hoc scoring. ``update`` takes a
position and nothing else; candidates are public and fixed at
construction. The regression test ``test_intent_tracker_no_label_leakage``
asserts that two pedestrians with IDENTICAL observed motion but DIFFERENT
hidden goals produce the IDENTICAL belief.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.intent_runtime_config import (
    NORMALIZATION_CONSTANTS, TEMPORAL_HISTORY_STEPS, TEMPORAL_SUMMARY_DIM,
)


class IntentTrackerError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateGoal:
    """One PUBLIC candidate destination, described by a route of waypoints
    (e.g. [junction, exit]). Never carries a hidden-truth label."""

    name: str
    waypoints: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.waypoints) == 0:
            raise IntentTrackerError(f"candidate {self.name!r} has no waypoints")
        for i, wp in enumerate(self.waypoints):
            arr = np.asarray(wp, dtype=np.float64)
            if arr.shape != (2,):
                raise IntentTrackerError(f"candidate {self.name!r} waypoint {i} must be a 2D coordinate, got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise IntentTrackerError(f"candidate {self.name!r} waypoint {i} is not finite: {wp!r}")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(2)


class GoalIntentTracker:
    """Recursive Bayesian posterior over public candidate goals from
    observed motion. Per-candidate waypoint progress is LATCHED (advances
    only forward as the observed pedestrian passes waypoints), so the
    preferred-velocity model is geometry-general -- not tied to a
    monotonic-descent assumption."""

    # Speed estimation (paper-test post-mortem). ``speed`` used to be a
    # hard-coded 1.0 m/s applied to every pedestrian in every scene. On the
    # held-out junction crowd that constant sits OUTSIDE both real ranges
    # (ambiguous 1.15-1.45, background 1.05-1.40), and the measured
    # candidate-velocity residual on the background humans was 1.03 m/s --
    # comparable to the speeds themselves. The preferred-velocity model is
    # what the likelihood compares against, so a wrong speed corrupts every
    # candidate's likelihood by roughly the same amount and the posterior
    # stops discriminating.
    #
    # ``speed`` is now only the PUBLIC PRIOR used before any velocity has
    # been observed; from the first real displacement onward the tracker
    # runs a clipped EMA of the observed speed.
    def __init__(
        self,
        candidates: Sequence[CandidateGoal],
        dt: float,
        speed: float,
        sigma: float = None,
        persistence: float = None,
        wp_radius: float = None,
        estimate_speed: bool = None,
        speed_ema_alpha: float = None,
        speed_min: float = None,
        speed_max: float = None,
    ) -> None:
        from crowd_nav.bayesian_dvl.intent_runtime_config import TRACKER_DEFAULTS
        d = TRACKER_DEFAULTS
        sigma = d["sigma"] if sigma is None else sigma
        persistence = d["persistence"] if persistence is None else persistence
        wp_radius = d["waypoint_radius"] if wp_radius is None else wp_radius
        estimate_speed = d["estimate_speed"] if estimate_speed is None else estimate_speed
        self.speed_ema_alpha = float(d["speed_ema_alpha"] if speed_ema_alpha is None else speed_ema_alpha)
        self.speed_min = float(d["speed_min"] if speed_min is None else speed_min)
        self.speed_max = float(d["speed_max"] if speed_max is None else speed_max)
        if not (0.0 < self.speed_ema_alpha <= 1.0):
            raise IntentTrackerError(f"speed_ema_alpha must be in (0,1], got {self.speed_ema_alpha}")
        if not (0.0 < self.speed_min < self.speed_max):
            raise IntentTrackerError(f"need 0 < speed_min < speed_max, got {self.speed_min}/{self.speed_max}")
        if len(candidates) == 0:
            raise IntentTrackerError("GoalIntentTracker requires at least one candidate goal")
        names = [c.name for c in candidates]
        if len(set(names)) != len(names):
            raise IntentTrackerError(f"candidate goal names must be unique, got {names}")
        for nm, val in (("dt", dt), ("speed", speed), ("sigma", sigma), ("wp_radius", wp_radius)):
            if not (np.isfinite(val) and val > 0):
                raise IntentTrackerError(f"{nm} must be finite and positive, got {val}")
        if not (0.0 < persistence <= 1.0):
            raise IntentTrackerError(f"persistence must be in (0,1], got {persistence}")
        self.candidates = list(candidates)
        self._routes = [np.asarray(c.waypoints, dtype=np.float64) for c in self.candidates]
        self.dt = float(dt)
        self.speed = float(speed)          # PUBLIC PRIOR only (see SPEED_EMA_ALPHA above)
        self.estimate_speed = bool(estimate_speed)
        self._speed_est = float(speed)     # what the preferred-velocity model actually uses
        self.sigma = float(sigma)
        self.persistence = float(persistence)
        self.wp_radius = float(wp_radius)
        n = len(self.candidates)
        self._log_b = np.log(np.ones(n) / n)
        self._wp_idx = [0] * n
        self._last_pos = None
        # False after a MISSED frame: the stored last position is then stale
        # (a diff against it would divide a multi-frame displacement by a
        # single dt -> fake huge velocity). The next observation re-baselines
        # instead of computing a likelihood. Same stale-gap discipline as
        # belief.py's kinematic tracker.
        self._last_position_is_fresh = False

    def _preferred_velocity_at(self, ci: int, position: np.ndarray, wp_idx: int) -> np.ndarray:
        route = self._routes[ci]
        if wp_idx >= len(route):
            return np.zeros(2)
        return self._speed_est * _unit(route[wp_idx] - position)

    def _advance_waypoints(self, position: np.ndarray) -> None:
        for ci in range(len(self.candidates)):
            route = self._routes[ci]
            while self._wp_idx[ci] < len(route) and float(np.linalg.norm(route[self._wp_idx[ci]] - position)) <= self.wp_radius:
                self._wp_idx[ci] += 1

    def note_missing(self) -> None:
        """Record that a frame elapsed with NO observation of this track. The
        stored last position becomes stale, so the next ``update`` re-baselines
        (belief unchanged that frame) instead of dividing a multi-frame
        displacement by a single dt."""
        self._last_position_is_fresh = False

    def update(self, observed_position: Sequence[float]) -> None:
        """Ingest ONE observed pedestrian position (label-leakage-safe:
        this is the only mutating input, and it is observable)."""
        pos = np.asarray(observed_position, dtype=np.float64)
        if pos.shape != (2,):
            raise IntentTrackerError(f"observed_position must be shape (2,), got {pos.shape}")
        if not np.all(np.isfinite(pos)):
            raise IntentTrackerError(f"observed_position must be finite, got {observed_position!r}")
        if self._last_pos is not None and self._last_position_is_fresh:
            # normal single-step likelihood update. likelihood uses the
            # waypoint each candidate was heading to over the LAST interval
            # (current latched index, before this step's advancement)
            v = (pos - self._last_pos) / self.dt
            ll = np.array([
                -float(np.sum((v - self._preferred_velocity_at(ci, self._last_pos, self._wp_idx[ci])) ** 2)) / (2 * self.sigma ** 2)
                for ci in range(len(self.candidates))
            ])
            self._log_b = self._log_b + ll
            self._log_b -= self._log_b.max()
            b = np.exp(self._log_b)
            b /= b.sum()
            # light persistence smoothing toward uniform: goals are sticky
            # but not frozen, so a beaten-down candidate stays recoverable
            b = self.persistence * b + (1.0 - self.persistence) / len(self.candidates)
            self._log_b = np.log(b)
            if self.estimate_speed:
                # AFTER the likelihood, never before: scoring this frame
                # against a model already fitted to this frame's own
                # displacement would make every candidate fit equally well
                # and flatten the posterior.
                observed = float(np.linalg.norm(v))
                self._speed_est = float(np.clip(
                    (1.0 - self.speed_ema_alpha) * self._speed_est + self.speed_ema_alpha * observed,
                    self.speed_min, self.speed_max))
        # else: first observation OR the first frame after a gap -> re-baseline
        # only (belief unchanged); the stored position was absent/stale so no
        # valid single-step velocity exists.
        self._advance_waypoints(pos)
        self._last_pos = pos
        self._last_position_is_fresh = True

    def belief(self) -> np.ndarray:
        b = np.exp(self._log_b - self._log_b.max())
        return b / b.sum()

    def roll_candidate_future(self, ci: int, position: Sequence[float], horizon: int) -> np.ndarray:
        """Predicted future [horizon,2] if the pedestrian pursues candidate
        ``ci`` from ``position``, latching forward through its remaining
        route (starts from the tracker's current per-candidate progress)."""
        if horizon < 1:
            raise IntentTrackerError(f"horizon must be >= 1, got {horizon}")
        route = self._routes[ci]
        pos = np.asarray(position, dtype=np.float64).copy()
        idx = self._wp_idx[ci]
        out = []
        for _ in range(horizon):
            while idx < len(route) and float(np.linalg.norm(route[idx] - pos)) <= self.wp_radius:
                idx += 1
            v = np.zeros(2) if idx >= len(route) else self._speed_est * _unit(route[idx] - pos)
            pos = pos + v * self.dt
            out.append(pos.copy())
        return np.array(out)

    def sample_futures(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        horizon: int,
        mode: str,
        rng: np.random.Generator,
        n_samples: int = 100,
    ) -> List[np.ndarray]:
        """Block 2 + the ablation interface. ``mode``:
          full    -- sample candidates ~ posterior (multimodal futures)
          mean    -- belief-weighted AVERAGE of the per-candidate futures
          cv      -- ignore intent; extrapolate current velocity
          uniform -- sample candidates ~ uniform (ignores the posterior)
        """
        if mode not in ("full", "mean", "cv", "uniform"):
            raise IntentTrackerError(f"unknown mode {mode!r}, must be one of full/mean/cv/uniform")
        if horizon < 1:
            raise IntentTrackerError(f"horizon must be >= 1, got {horizon}")
        if mode in ("full", "uniform") and n_samples <= 0:
            raise IntentTrackerError(f"n_samples must be positive for mode {mode!r}, got {n_samples}")
        pos = np.asarray(position, dtype=np.float64)
        vel = np.asarray(velocity, dtype=np.float64)
        if pos.shape != (2,) or vel.shape != (2,):
            raise IntentTrackerError(f"position/velocity must be shape (2,), got {pos.shape}/{vel.shape}")
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            raise IntentTrackerError("position/velocity must be finite")
        if mode == "cv":
            return [np.array([pos + vel * self.dt * (t + 1) for t in range(horizon)])]
        if mode == "mean":
            b = self.belief()
            trajs = [self.roll_candidate_future(ci, pos, horizon) for ci in range(len(self.candidates))]
            return [sum(b[ci] * trajs[ci] for ci in range(len(self.candidates)))]
        # mode in ("full", "uniform") -- validated at the top
        p = self.belief() if mode == "full" else np.ones(len(self.candidates)) / len(self.candidates)
        return [self.roll_candidate_future(int(rng.choice(len(self.candidates), p=p)), pos, horizon) for _ in range(n_samples)]


def _normalized_entropy(p: np.ndarray, log_k: float) -> float:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    q = q / q.sum()
    return float(-(q * np.log(q)).sum() / log_k) if log_k > 0 else 0.0


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JSD(p, q) / log(2) -- in [0, 1], symmetric, and finite even when a
    candidate has zero probability (KL alone is not)."""
    a = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None); a = a / a.sum()
    b = np.clip(np.asarray(q, dtype=np.float64), 1e-12, None); b = b / b.sum()
    m = 0.5 * (a + b)
    kl = lambda x, y: float((x * np.log(x / y)).sum())
    return float(np.clip(0.5 * (kl(a, m) + kl(b, m)) / np.log(2.0), 0.0, 1.0))


class IntentBeliefBank:
    """Per-pedestrian intent trackers keyed by STABLE track_id (consolidation
    plan Order 4, item 5). Identity is bound to track_id, NEVER to list
    position -- feeding the same observations in a different dict order gives
    the same per-track beliefs. Supports episode ``reset``, track loss, and
    reappearance: a track missing for more than ``missing_timeout_steps``
    consecutive updates is expired, so a later reappearance of that id starts
    a fresh tracker; a shorter gap resumes the existing belief.

    LABEL-LEAKAGE CONTRACT: ``candidate_fn(track_id, first_observed_position)``
    receives ONLY a stable id and a PUBLIC observed entry position; it must
    derive candidates from public scene geometry, never from a Human's hidden
    gx/gy. The bank never accepts a Human object.
    """

    def __init__(
        self,
        candidate_fn: Callable[[int, np.ndarray], Sequence[CandidateGoal]],
        dt: float,
        speed: float,
        missing_timeout_steps: int = 8,
        **tracker_kwargs,
    ) -> None:
        if not callable(candidate_fn):
            raise IntentTrackerError("candidate_fn must be callable(track_id, first_position) -> candidates")
        if missing_timeout_steps < 1:
            raise IntentTrackerError(f"missing_timeout_steps must be >= 1, got {missing_timeout_steps}")
        self._candidate_fn = candidate_fn
        self._dt = float(dt)
        self._speed = float(speed)
        self._missing_timeout = int(missing_timeout_steps)
        self._tracker_kwargs = tracker_kwargs
        self._trackers: Dict[int, GoalIntentTracker] = {}
        self._missing: Dict[int, int] = {}
        self._age: Dict[int, int] = {}
        # Order 12A: the last TEMPORAL_HISTORY_STEPS public observations and
        # posteriors per track, keyed by track_id -- NEVER by list position,
        # for the same reason the trackers are: reordering the observation
        # dict must not move one pedestrian's history onto another.
        self._history: Dict[int, Deque[Tuple[np.ndarray, np.ndarray]]] = {}

    def reset(self) -> None:
        self._trackers.clear()
        self._missing.clear()
        self._age.clear()
        self._history.clear()

    def update(self, observations: Dict[int, Sequence[float]]) -> None:
        """``observations``: {track_id: observed_position[2]}. Order-invariant
        (each track updates only from its own history)."""
        seen = set(observations.keys())
        for tid in list(self._trackers.keys()):
            if tid not in seen:
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > self._missing_timeout:
                    del self._trackers[tid]
                    del self._missing[tid]
                    del self._age[tid]
                    self._history.pop(tid, None)
                else:
                    # within the timeout: keep the tracker but mark its stored
                    # position stale so reappearance re-baselines (no fake
                    # cross-gap velocity)
                    # the tracker re-baselines its position on reappearance, so
                    # a velocity spanning the gap would be fabricated; the
                    # temporal history is dropped for exactly that reason.
                    self._trackers[tid].note_missing()
                    self._history.pop(tid, None)
        for tid, pos in observations.items():
            position = np.asarray(pos, dtype=np.float64)
            if tid not in self._trackers:
                candidates = self._candidate_fn(tid, position)  # PUBLIC inputs only
                self._trackers[tid] = GoalIntentTracker(
                    candidates, dt=self._dt, speed=self._speed, **self._tracker_kwargs
                )
                self._age[tid] = 0
                self._history.pop(tid, None)
            self._trackers[tid].update(position)
            self._missing[tid] = 0
            self._age[tid] += 1
            hist = self._history.setdefault(tid, deque(maxlen=TEMPORAL_HISTORY_STEPS))
            # push AFTER the tracker update so the stored posterior is the one
            # that was current at this step; only past and present ever enter.
            hist.append((position.copy(), np.asarray(self._trackers[tid].belief(), dtype=np.float64).copy()))

    def temporal_summary_for(self, track_id: int, *, include_posterior: bool = True) -> np.ndarray:
        """The Order 12A trend summary for one track: TEMPORAL_SUMMARY_DIM
        scalars, every one bounded and finite.

        Reads ONLY the stored history, which is appended after each update --
        so a value here can never depend on data the policy would not have had
        at that step. With fewer than two observations every trend is 0 and
        only ``history_fill`` is informative, which is the honest encoding of
        "nothing has been seen yet" rather than a fabricated zero-velocity.

        ``include_posterior=False`` zeroes the three posterior trends for the
        mean/cv ablations. They are computed from the FULL posterior, so an
        ablation that saw them would be reading exactly the quantity it is
        supposed to be denied -- the public motion trends are untouched, so
        the arms still differ in the belief treatment and nothing else.
        """
        out = np.zeros(TEMPORAL_SUMMARY_DIM, dtype=np.float64)
        hist = self._history.get(track_id)
        if not hist:
            return out
        n = len(hist)
        out[4] = float(n - 1) / float(TEMPORAL_HISTORY_STEPS - 1)   # history_fill
        if n < 2:
            return out

        max_speed = float(NORMALIZATION_CONSTANTS["max_human_speed"]) or 1.0
        (p_old, b_old), (p_prev, _), (p_now, b_now) = hist[0], hist[-2], hist[-1]
        # velocity from consecutive PUBLIC positions -- the bank is never given
        # a velocity, and deriving it keeps the summary a function of what the
        # bank actually observed.
        v_now = (np.asarray(p_now) - np.asarray(p_prev)) / self._dt
        if n >= 3:
            v_old = (np.asarray(hist[1][0]) - np.asarray(p_old)) / self._dt
        else:
            v_old = v_now

        dv = (v_now - v_old) / max_speed
        out[0] = float(np.clip(dv[0], -1.0, 1.0))
        out[1] = float(np.clip(dv[1], -1.0, 1.0))
        out[2] = float(np.clip(
            (float(np.linalg.norm(v_now)) - float(np.linalg.norm(v_old))) / max_speed, -1.0, 1.0))

        # heading is undefined for a stationary pedestrian; report no turn
        # rather than the arbitrary angle of a numerical-noise velocity.
        if float(np.linalg.norm(v_now)) > 1e-6 and float(np.linalg.norm(v_old)) > 1e-6:
            d = float(np.arctan2(v_now[1], v_now[0]) - np.arctan2(v_old[1], v_old[0]))
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
            out[3] = float(np.clip(d / np.pi, -1.0, 1.0))

        if not include_posterior:
            return out

        b_old = np.asarray(b_old, dtype=np.float64)
        b_now = np.asarray(b_now, dtype=np.float64)
        if b_old.shape == b_now.shape and b_old.size > 0:
            out[5] = _jensen_shannon_divergence(b_old, b_now)
            k = int(b_now.size)
            if k > 1:
                log_k = float(np.log(k))
                out[6] = float(np.clip(
                    (_normalized_entropy(b_now, log_k) - _normalized_entropy(b_old, log_k)), -1.0, 1.0))
            top1 = int(np.argmax(b_now))
            same = sum(1 for _, b in hist if b.shape == b_now.shape and int(np.argmax(b)) == top1)
            out[7] = float(same) / float(n)
        return out

    def belief_for(self, track_id: int) -> np.ndarray:
        if track_id not in self._trackers:
            raise IntentTrackerError(f"no active track {track_id}")
        return self._trackers[track_id].belief()

    def tracker_for(self, track_id: int) -> GoalIntentTracker:
        if track_id not in self._trackers:
            raise IntentTrackerError(f"no active track {track_id}")
        return self._trackers[track_id]

    def speed_estimate_for(self, track_id: int) -> float:
        """The tracker's current speed estimate -- exposed so the candidate
        audit can measure the residual against observed motion instead of
        trusting that the model matches."""
        if track_id not in self._trackers:
            raise IntentTrackerError(f"no active track {track_id}")
        return float(self._trackers[track_id]._speed_est)

    def track_age_for(self, track_id: int) -> int:
        """Number of real `update` calls this track has received since it
        was (re)acquired (resets to 0 on reappearance after expiry, keeps
        counting through a short within-timeout gap)."""
        if track_id not in self._age:
            raise IntentTrackerError(f"no active track {track_id}")
        return self._age[track_id]

    def active_tracks(self) -> set:
        return set(self._trackers.keys())
