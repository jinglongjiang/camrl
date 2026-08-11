"""Ranking-IL data model and expert-action-equivalence utilities
(guide.md R3-3).

Stage 1 was previously "IL" in name only: it trained a state-value
regression on ORCA-visited states' Monte Carlo returns but never
supervised WHICH action ORCA actually chose, even though
``action_index`` was stored right alongside. This module fixes that by
defining a typed sample that carries everything needed to reconstruct
the exact 80-candidate scoring batch ``_score_all_candidates`` would
build at deployment time for that same real state, plus the set of
grid actions ORCA's continuous choice is considered equivalent to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation


class RankingError(ValueError):
    pass


def derive_action_equivalence_tolerance(action_table: Sequence[Tuple[float, float]]) -> float:
    """Median nearest-neighbor Euclidean distance across the frozen
    80-action grid -- a data-derived measure of "grid spacing", used to
    decide which OTHER actions besides the single closest one should be
    treated as equivalent to ORCA's continuous action (guide.md R3-3:
    "容差只由动作网格相邻速度/角度间距推导，不根据validation表现调节").
    """
    pts = np.asarray(action_table, dtype=np.float64)
    n = len(pts)
    if n < 2:
        raise RankingError("action_table must have at least 2 actions")
    nearest = np.empty(n)
    for i in range(n):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        d[i] = np.inf
        nearest[i] = d.min()
    return float(np.median(nearest))


def nearest_action_index(vx: float, vy: float, action_table: Sequence[Tuple[float, float]]) -> int:
    """The SINGLE closest grid action to a continuous (vx, vy) choice --
    guide.md R4-2's ``executed_action_index``, deliberately NOT the
    (possibly multi-member) equivalence class ``build_action_equivalence_class``
    returns for ranking. Always exactly one winner (ties broken by
    ``np.argmin``'s first-occurrence rule, matching the tie-break used
    implicitly wherever this codebase already does distance argmin)."""
    pts = np.asarray(action_table, dtype=np.float64)
    dists = np.hypot(pts[:, 0] - vx, pts[:, 1] - vy)
    return int(np.argmin(dists))


def build_action_equivalence_class(
    orca_vx: float, orca_vy: float,
    action_table: Sequence[Tuple[float, float]],
    tolerance: float,
) -> Tuple[int, ...]:
    """Grid action indices within ``tolerance`` of the single closest
    match to ORCA's continuous ``(vx, vy)`` -- guide.md R3-3's "等价专家
    动作集合": avoids forcing the network to treat every heading grid
    line as a hard boundary when ORCA's real velocity falls between two
    adjacent grid points."""
    if tolerance < 0:
        raise RankingError(f"tolerance must be non-negative, got {tolerance}")
    pts = np.asarray(action_table, dtype=np.float64)
    dists = np.hypot(pts[:, 0] - orca_vx, pts[:, 1] - orca_vy)
    best = float(dists.min())
    indices = tuple(int(i) for i in np.where(dists <= best + tolerance)[0])
    if not indices:
        raise RankingError("action equivalence class must never be empty")
    return indices


@dataclass
class RankingDemoSample:
    """One ORCA-demonstration decision point, carrying everything
    needed to reconstruct the exact 80-candidate scoring batch
    ``_score_all_candidates`` would build at deployment time for this
    same real state (guide.md R3-3: "不得只保存已编码feature后丢失重建
    候选所需信息").

    ``belief_tracker_snapshot`` is a deep copy of
    ``BeliefTracker._tracks`` at this step -- sufficient to reconstruct
    a live ``BeliefTracker`` whose ``belief_for``/``entropy_for``/
    ``track_age_for`` are valid for this exact moment, without re-running
    the whole episode's filtering history.

    R4-2 fix (2026-08-10, guide.md "R4-1R-4 -- 冻结 R4-2 的精确数学目标"):
    ``executed_action_index``/``executed_action_features`` are NEW,
    genuinely distinct from ``expert_action_indices`` -- the latter is
    ORCA's continuous choice quantized to a TOLERANCE-WIDENED equivalence
    class (used only for the ranking loss, may contain several grid
    actions), the former is the SINGLE actual grid action nearest to
    what ORCA really executed this step (used for the MC-loss target;
    guide.md is explicit these "二者不可互相代替" -- swapping one for the
    other must not silently work). ``posterior_seed_key`` is the exact
    ``(suite_seed, episode_seed, decision_counter)`` tuple the MC-loss
    target must reuse so its world-sampling/tau seeding is bit-identical
    to what a live decision at this state would have produced for this
    same real action (see ``policy._vectorized_candidate_batch``'s
    ``action_indices`` parameter). ``source_role`` distinguishes this
    from ``replay.MCReturnSample`` records without relying on isinstance
    checks leaking into training code that should treat both uniformly.
    """

    robot: RobotObservation
    humans: Tuple[HumanObservation, ...]
    global_time: float
    belief_tracker_snapshot: Dict[int, Any]
    expert_action_indices: Tuple[int, ...]
    executed_action_index: int
    executed_action_features: np.ndarray
    posterior_seed_key: Tuple[int, int, int]
    target_return: float
    artifact_sha256: str
    episode_seed: int
    step_index: int
    outcome: str = ""
    source_role: str = "demo"
