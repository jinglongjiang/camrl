"""R4-4R belief-relevant training-data gate (guide.md "R4-4R冻结契约:
belief-relevant数据门禁", 2026-08-10).

Replaces the original R4-4 gate, which counted decisions rather than
independent episodes/events: ``outcome``/``clearance_bin`` were computed
ONCE per episode but broadcast onto every one of that episode's
per-decision records, so a single collision episode with N steps could
singlehandedly clear a ">=5 samples" threshold. It also required a
purely-reciprocal ORCA teacher to genuinely collide, which is not a
meaningful target: ORCA is specifically designed not to collide, so
"0 real collisions" is not evidence anything is broken.

The corrected design has two tiers:

1. ``EpisodeCoverageRecord`` -- exactly ONE record per episode
   (profile/outcome/clearance/non-reciprocal), REPORT ONLY, never gates.
2. ``RiskOpportunityRecord`` -- one record per DECISION whose already-
   computed 80-candidate counterfactual set genuinely contains both a
   safe and a dangerous non-trivial option. This is the real gating
   unit: guide.md's root-cause diagnosis is about whether the belief
   changes the CHOSEN action in states where it matters, not about
   whether the demonstration trajectory happened to end in collision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.config import FROZEN_VALUES
from crowd_nav.bayesian_dvl.counterfactual import CounterfactualCandidateResult, ROLLOUT_HORIZON

MIN_STRATUM_SAMPLES = 5  # guide.md's existing calibration convention (statistics.py's MIN_STRATUM_SAMPLES)

CLEARANCE_BINS = ("near_collision", "risk_opportunity", "normal")
BELIEF_ENTROPY_BINS = ("low", "medium", "high")
ACTION_TYPES = ("stop_or_slow", "straight", "turn")

# guide.md R4-4R: a candidate counts as a non-trivial "safe" option only
# if its expected 8-step progress exceeds what merely crawling at the
# stand/not-stand boundary speed for the whole horizon would produce --
# reuses the already-frozen stand_speed_threshold instead of inventing a
# new magic number (0.05 m/s * 0.25 s * 8 steps = 0.1 m).
MIN_MEANINGFUL_PROGRESS: float = float(FROZEN_VALUES["stand_speed_threshold"]) * float(FROZEN_VALUES["dt"]) * ROLLOUT_HORIZON


class DataCoverageError(ValueError):
    pass


@dataclass(frozen=True)
class EpisodeCoverageRecord:
    """One episode's outcome/clearance tags. REPORT ONLY -- guide.md
    R4-4R explicitly demotes collision/near_collision counts to
    informational, since a purely-reciprocal ORCA teacher not colliding
    is expected behavior, not missing coverage."""

    episode_seed: int
    profile: str
    outcome: str
    is_non_reciprocal_episode: bool
    clearance_bin: str

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "collision", "timeout"}:
            raise DataCoverageError(f"invalid outcome {self.outcome!r}")
        if self.clearance_bin not in CLEARANCE_BINS:
            raise DataCoverageError(f"invalid clearance_bin {self.clearance_bin!r}")


@dataclass(frozen=True)
class RiskOpportunityRecord:
    """One qualifying decision -- guide.md R4-4R's real gating unit for
    coverage/disagreement/action-diversity purposes.

    ``full_top_action``/``mean_top_action``/``cv_top_action`` are the
    top-ranked action index under posterior_source "full"/"moment_mean"/
    "cv" respectively (same seed_key, so only the posterior source
    differs), all computed over the WHOLE scene (all humans). Grading
    these choices for real decision value is NOT done here: guide.md
    R4-4R2 found that comparing a whole-scene choice against a truth
    oracle that can only see the non-reciprocal subset is a population
    mismatch that can hide a real effect. That population-aligned
    comparison lives in ``oracle_regret.py``'s ``AuditRecord`` /
    ``check_oracle_regret_gate`` instead -- this record only carries the
    coverage/disagreement/action-type statistics, which don't have that
    problem (they don't involve the oracle at all).
    """

    episode_seed: int
    decision_seed: int
    profile: str
    belief_entropy_bin: str
    action_type: str  # of full_top_action
    full_top_action: int
    mean_top_action: int
    cv_top_action: int
    agree_with_mean: bool
    agree_with_cv: bool
    internal_regret_vs_mean: float
    internal_regret_vs_cv: float

    def __post_init__(self) -> None:
        if self.belief_entropy_bin not in BELIEF_ENTROPY_BINS:
            raise DataCoverageError(f"invalid belief_entropy_bin {self.belief_entropy_bin!r}")
        if self.action_type not in ACTION_TYPES:
            raise DataCoverageError(f"invalid action_type {self.action_type!r}")


def is_risk_opportunity_state(
    candidate_results: Sequence[CounterfactualCandidateResult],
    discomfort_distance: float,
    min_progress: float = MIN_MEANINGFUL_PROGRESS,
) -> bool:
    """guide.md R4-4R: True iff the already-computed 80-candidate set
    (posterior_source="full") contains at least one SAFE non-trivial
    candidate AND at least one DANGEROUS candidate -- an ambiguous state
    where the choice of action actually matters, independent of whether
    the demonstration that actually ran through this state collided."""
    if not candidate_results:
        raise DataCoverageError("is_risk_opportunity_state requires at least one candidate result")
    has_safe = any(
        c.collision_prob == 0.0 and c.lower_tail_clearance >= discomfort_distance and c.expected_progress > min_progress
        for c in candidate_results
    )
    has_dangerous = any(c.collision_prob > 0.0 or c.lower_tail_clearance < 0.0 for c in candidate_results)
    return has_safe and has_dangerous


def classify_clearance(min_human_dist: float, discomfort_distance: float) -> str:
    """near_collision -- dmin below zero (an actual swept collision was
    recorded at some point in the episode); risk_opportunity -- within
    the discomfort band (same threshold the reward function itself
    uses); normal -- everything else. Episode-level only (report tier)."""
    if min_human_dist < 0.0:
        return "near_collision"
    if min_human_dist < discomfort_distance:
        return "risk_opportunity"
    return "normal"


def classify_belief_entropy(entropy: float, max_entropy: float, low_frac: float = 1.0 / 3.0, high_frac: float = 2.0 / 3.0) -> str:
    """Tercile split of [0, max_entropy] (max_entropy = log(N_MODES) for
    a uniform 5-mode belief) -- fixed, reproducible bin boundaries,
    independent of what a particular collection run happens to observe."""
    if max_entropy <= 0:
        raise DataCoverageError(f"max_entropy must be positive, got {max_entropy}")
    frac = entropy / max_entropy
    if frac < low_frac:
        return "low"
    if frac < high_frac:
        return "medium"
    return "high"


def classify_action_type(vx: float, vy: float, heading_reference: Tuple[float, float], turn_threshold_rad: float = 0.3) -> str:
    """stop_or_slow / straight / turn, relative to ``heading_reference``
    (typically the robot's current velocity direction at decision time)."""
    speed = float(np.hypot(vx, vy))
    if speed < 1e-6:
        return "stop_or_slow"
    ref_norm = float(np.hypot(*heading_reference))
    if ref_norm < 1e-6:
        return "straight"  # no reference direction available (e.g. at goal); treat as straight
    cos_angle = (vx * heading_reference[0] + vy * heading_reference[1]) / (speed * ref_norm)
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if speed < 0.3:
        return "stop_or_slow"
    return "turn" if angle > turn_threshold_rad else "straight"


def compute_episode_coverage_report(records: Sequence[EpisodeCoverageRecord]) -> Dict[str, object]:
    """Marginal per-episode counts -- REPORT ONLY (guide.md R4-4R)."""
    if not records:
        raise DataCoverageError("compute_episode_coverage_report requires at least one record")

    def _tally(values: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return counts

    return {
        "n_episodes": len(records),
        "profile": _tally([r.profile for r in records]),
        "outcome": _tally([r.outcome for r in records]),
        "non_reciprocal_episode": _tally(["non_reciprocal" if r.is_non_reciprocal_episode else "reciprocal" for r in records]),
        "clearance_bin": _tally([r.clearance_bin for r in records]),
    }


def check_episode_gate(report: Dict[str, object], required_profiles: Sequence[str], min_samples: int = MIN_STRATUM_SAMPLES) -> Dict[str, object]:
    """Sanity-only gate on the episode-level report: every required
    profile must have been collected at all. Deliberately does NOT check
    outcome/clearance_bin -- those are informational under R4-4R."""
    reasons: List[str] = []
    counts = report.get("profile", {})
    for profile in required_profiles:
        n = counts.get(profile, 0)
        if n < min_samples:
            reasons.append(f"profile={profile!r} has {n} episodes, need >= {min_samples}")
    return {"passed": not reasons, "reasons": reasons}


def compute_risk_opportunity_report(records: Sequence[RiskOpportunityRecord]) -> Dict[str, object]:
    if not records:
        raise DataCoverageError("compute_risk_opportunity_report requires at least one record")

    def _tally(values: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return counts

    n = len(records)
    return {
        "n_states": n,
        "n_episodes": len({r.episode_seed for r in records}),
        "profile": _tally([r.profile for r in records]),
        "action_type": _tally([r.action_type for r in records]),
        "disagreement_rate_vs_mean": sum(1 for r in records if not r.agree_with_mean) / n,
        "disagreement_rate_vs_cv": sum(1 for r in records if not r.agree_with_cv) / n,
    }


def check_risk_opportunity_gate(
    records: Sequence[RiskOpportunityRecord],
    required_profiles: Sequence[str],
    min_states: int = 100,
    min_episodes: int = 30,
    min_stratum: int = MIN_STRATUM_SAMPLES,
    min_disagreement_rate: float = 0.05,
) -> Dict[str, object]:
    """guide.md R4-4R's coverage/disagreement/action-diversity gate.
    Pure/testable like check_r2_gate -- takes already-computed records,
    makes no I/O. Does NOT judge real decision value -- that is
    ``oracle_regret.check_oracle_regret_gate``'s job (guide.md R4-4R2),
    kept separate because it needs the population-aligned AuditRecord,
    not this whole-scene RiskOpportunityRecord."""
    reasons: List[str] = []
    n = len(records)
    if n < min_states:
        reasons.append(f"n_risk_opportunity_states={n} < {min_states}")

    episode_ids = {r.episode_seed for r in records}
    if len(episode_ids) < min_episodes:
        reasons.append(f"n_risk_opportunity_episodes={len(episode_ids)} < {min_episodes}")

    for profile in required_profiles:
        cnt = sum(1 for r in records if r.profile == profile)
        if cnt < min_stratum:
            reasons.append(f"profile={profile!r} has {cnt} risk-opportunity states, need >= {min_stratum}")

    if n > 0:
        disagree_mean = sum(1 for r in records if not r.agree_with_mean) / n
        if disagree_mean < min_disagreement_rate:
            reasons.append(f"disagreement_rate_vs_mean={disagree_mean:.4f} < {min_disagreement_rate}")
        disagree_cv = sum(1 for r in records if not r.agree_with_cv) / n
        if disagree_cv < min_disagreement_rate:
            reasons.append(f"disagreement_rate_vs_cv={disagree_cv:.4f} < {min_disagreement_rate}")

    turn_episodes = {r.episode_seed for r in records if r.action_type == "turn"}
    if len(turn_episodes) < 1:
        reasons.append("action_type='turn' has 0 independent episodes among full's chosen actions, need >= 1")
    stop_episodes = {r.episode_seed for r in records if r.action_type == "stop_or_slow"}
    if len(stop_episodes) < 1:
        reasons.append("action_type='stop_or_slow' has 0 independent episodes among full's chosen actions, need >= 1")

    return {"passed": not reasons, "reasons": reasons}


def topk_overlap(indices_a: Sequence[int], indices_b: Sequence[int], k: int) -> float:
    """Fraction overlap of the top-k of two ranked action-index
    sequences -- used by the world-sample-count stability check."""
    if k <= 0:
        raise DataCoverageError(f"k must be positive, got {k}")
    set_a, set_b = set(indices_a[:k]), set(indices_b[:k])
    return len(set_a & set_b) / k


def check_world_sample_stability(overlap_fractions: Sequence[float], min_overlap: float = 0.85) -> Dict[str, object]:
    """guide.md R4-4R point 6: mean top-3 overlap between the 32- and
    128-world-sample rankings of the same risk-opportunity decisions
    must be high, or the ranking is at risk of being driven by sampling
    noise rather than genuine risk structure."""
    if not overlap_fractions:
        raise DataCoverageError("check_world_sample_stability requires at least one overlap fraction")
    mean_overlap = float(np.mean(overlap_fractions))
    reasons: List[str] = []
    if mean_overlap < min_overlap:
        reasons.append(f"mean top-3 overlap {mean_overlap:.4f} < {min_overlap}")
    return {"passed": not reasons, "reasons": reasons, "mean_overlap": mean_overlap, "n_checked": len(overlap_fractions)}
