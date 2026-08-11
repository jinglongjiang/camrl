"""Canonical, named observation contracts for BDVL (guide.md section 5.1, A1).

This module deliberately does NOT reuse ``crowd_nav.contracts``:

- ``crowd_nav.contracts.simulate_next_frames`` reads ``current_state[5]``
  believing it is ``v_pref``; the real ``FullState`` field order is
  ``(px, py, vx, vy, radius, gx, gy, v_pref, theta)`` so index 5 is
  actually ``gx``. Verified directly against
  ``crowd_sim/envs/utils/state.py`` and ``crowd_nav/contracts.py:794``.
- ``crowd_nav.contracts``'s 34D/token path truncates to the nearest 5
  humans (``sorted_indices[:5]`` / ``distances[:5]``), so 20-human
  scenarios are blind to most of the crowd.
- ``Robot.get_obs_array()`` returns ``(px,py,gx,gy,vx,vy,radius,v_pref,
  theta)`` -- a DIFFERENT field order from ``FullState.to_array()``'s
  ``(px,py,vx,vy,radius,gx,gy,v_pref,theta)``. The two are not
  interchangeable index-for-index.

All BDVL code must go through the named dataclasses here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

MAX_HUMANS = 20


@dataclass(frozen=True)
class RobotObservation:
    """Canonical robot fields, named -- never a bare index into an array."""

    px: float
    py: float
    vx: float
    vy: float
    radius: float
    gx: float
    gy: float
    v_pref: float
    theta: float

    @classmethod
    def from_full_state(cls, full_state) -> "RobotObservation":
        """Build from ``crowd_sim.envs.utils.state.FullState`` by named
        attribute access (never ``.to_array()`` + index), so this is
        immune to any future field-order change in either class."""
        return cls(
            px=float(full_state.px), py=float(full_state.py),
            vx=float(full_state.vx), vy=float(full_state.vy),
            radius=float(full_state.radius),
            gx=float(full_state.gx), gy=float(full_state.gy),
            v_pref=float(full_state.v_pref), theta=float(full_state.theta),
        )

    def to_canonical_array(self) -> np.ndarray:
        """The one BDVL-internal order: matches ``FullState.to_array()``,
        NOT ``Robot.get_obs_array()``."""
        return np.array(
            [self.px, self.py, self.vx, self.vy, self.radius,
             self.gx, self.gy, self.v_pref, self.theta],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class HumanObservation:
    """Canonical per-human fields plus a stable track id."""

    track_id: int
    px: float
    py: float
    vx: float
    vy: float
    radius: float

    @classmethod
    def from_observable_state(cls, track_id: int, observable_state) -> "HumanObservation":
        return cls(
            track_id=int(track_id),
            px=float(observable_state.px), py=float(observable_state.py),
            vx=float(observable_state.vx), vy=float(observable_state.vy),
            radius=float(observable_state.radius),
        )

    def to_canonical_array(self) -> np.ndarray:
        return np.array([self.px, self.py, self.vx, self.vy, self.radius], dtype=np.float64)


@dataclass(frozen=True)
class CanonicalObservation:
    """Full-crowd observation: robot + up to ``MAX_HUMANS`` humans with mask.

    ``human_track_ids[i]``/``human_features[i, :]`` are only meaningful
    where ``human_mask[i]`` is True; masked rows are zero-filled and
    must never contribute to any pooled/aggregated feature.
    """

    robot: RobotObservation
    human_track_ids: Tuple[int, ...]
    human_features: np.ndarray  # [MAX_HUMANS, 5], zero-filled beyond n_humans
    human_mask: np.ndarray  # [MAX_HUMANS], bool

    def __post_init__(self) -> None:
        if self.human_features.shape != (MAX_HUMANS, 5):
            raise ValueError(f"human_features must be [{MAX_HUMANS}, 5], got {self.human_features.shape}")
        if self.human_mask.shape != (MAX_HUMANS,):
            raise ValueError(f"human_mask must be [{MAX_HUMANS}], got {self.human_mask.shape}")
        if self.human_mask.dtype != np.bool_:
            raise ValueError(f"human_mask must be bool dtype, got {self.human_mask.dtype}")
        n_valid = int(self.human_mask.sum())
        if len(self.human_track_ids) != n_valid:
            raise ValueError(
                f"human_track_ids length {len(self.human_track_ids)} != valid mask count {n_valid}"
            )
        if not np.all(self.human_features[~self.human_mask] == 0.0):
            raise ValueError("masked-out human rows must be zero-filled")

    @property
    def n_humans(self) -> int:
        return int(self.human_mask.sum())


def canonicalize(robot_full_state, human_observable_states: Sequence, track_ids: Sequence[int]) -> CanonicalObservation:
    """Build a ``CanonicalObservation`` from raw CrowdSim state objects.

    Reads the FULL human list (no top-K truncation by distance or TTC):
    guide.md 5.1 explicitly forbids truncating to the nearest 5.
    Excess humans beyond ``MAX_HUMANS`` raise, rather than silently
    dropping the tail -- CrowdNav's own formal scenarios cap at 20.
    """
    n = len(human_observable_states)
    if n != len(track_ids):
        raise ValueError(f"{n} human states but {len(track_ids)} track ids")
    if n > MAX_HUMANS:
        raise ValueError(f"{n} humans exceeds MAX_HUMANS={MAX_HUMANS}")

    robot = RobotObservation.from_full_state(robot_full_state)

    features = np.zeros((MAX_HUMANS, 5), dtype=np.float64)
    mask = np.zeros((MAX_HUMANS,), dtype=np.bool_)
    for i in range(n):
        human_obs = HumanObservation.from_observable_state(track_ids[i], human_observable_states[i])
        features[i, :] = human_obs.to_canonical_array()
        mask[i] = True

    return CanonicalObservation(
        robot=robot,
        human_track_ids=tuple(int(t) for t in track_ids[:n]),
        human_features=features,
        human_mask=mask,
    )


def canonical_observation_is_permutation_invariant_content(
    obs_a: CanonicalObservation, obs_b: CanonicalObservation
) -> bool:
    """True iff obs_b is obs_a with its valid humans permuted (same set
    of (track_id, feature) pairs, robot unchanged). Used by tests only;
    the set encoder (A6) is what must actually be permutation-invariant
    at the network level -- this helper just validates test fixtures."""
    if obs_a.robot != obs_b.robot:
        return False
    if obs_a.n_humans != obs_b.n_humans:
        return False
    pairs_a = {
        obs_a.human_track_ids[i]: tuple(obs_a.human_features[i].tolist())
        for i in range(obs_a.n_humans)
    }
    pairs_b = {
        obs_b.human_track_ids[i]: tuple(obs_b.human_features[i].tolist())
        for i in range(obs_b.n_humans)
    }
    return pairs_a == pairs_b
