"""R4-4R2: population-aligned oracle regret (guide.md "R4-4R2冻结契约",
2026-08-10).

R4-4R's first oracle attempt graded full/mean/cv's top-1 action -- each
chosen using the belief over ALL 5 humans -- against a truth oracle that
could only see the non-reciprocal subset (20-50% of humans, since only
their real future is action-invariant ground truth). That is an
apples-to-oranges comparison: near-zero regret could mean "belief
doesn't help" OR simply "the deciding population and the grading
population don't match", and real diagnosis on the R4-4R pilot could not
rule out the second explanation (a concrete example: full and mean chose
visibly different actions, but the non-reciprocal humans in that episode
were nowhere near either action's path, so grading against just those
humans showed identical, near-zero regret regardless of which action was
actually chosen).

R4-4R2 removes the mismatch: full/mean/cv are RE-RANKED using ONLY the
non-reciprocal subset (the same, and only, population the truth oracle
can validly grade), then compared against the oracle's own full 80-
action ranking of that SAME subset. Regret is computed via the frozen
lexicographic priority (collision, then clearance, then progress among
safety-equivalent actions, then control cost) rather than a single ad
hoc scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.counterfactual import CounterfactualCandidateResult, rank_counterfactual_candidates
from crowd_nav.bayesian_dvl.ranking import derive_action_equivalence_tolerance
from crowd_nav.bayesian_dvl.statistics import paired_difference_bootstrap


class OracleRegretError(ValueError):
    pass


def clearance_equivalence_tolerance(action_table: Sequence[Tuple[float, float]], dt: float, horizon: int) -> float:
    """guide.md R4-4R2 point 7: a physically-grounded tolerance for
    "practically equally safe", derived the SAME way guide.md R3-3
    already derives the action-equivalence tolerance (median nearest-
    neighbor grid spacing) -- not tuned against any validation outcome.
    Two adjacent grid actions can differ in position by at most
    ``spacing * dt`` per step; over the full horizon (worst case: they
    diverge every step) that bounds how much their realized clearance
    can differ due to grid discretization alone, giving a tolerance in
    clearance (meters) rather than velocity units."""
    spacing = derive_action_equivalence_tolerance(action_table)
    return float(spacing) * float(dt) * int(horizon)


def build_safety_optimal_layer(
    oracle_results: Sequence[CounterfactualCandidateResult], clearance_tolerance: float,
) -> Tuple[int, ...]:
    """guide.md R4-4R2 point 7: action indices "practically as safe as"
    the oracle's own single best action -- same collision_prob as the
    best, and clearance within ``clearance_tolerance`` of it."""
    if not oracle_results:
        raise OracleRegretError("build_safety_optimal_layer requires at least one oracle result")
    if clearance_tolerance < 0:
        raise OracleRegretError(f"clearance_tolerance must be non-negative, got {clearance_tolerance}")
    by_idx = {r.action_index: r for r in oracle_results}
    ranked = rank_counterfactual_candidates(oracle_results, clearance_tolerance)
    best = by_idx[ranked[0]]
    layer = tuple(
        r.action_index for r in oracle_results
        if r.collision_prob == best.collision_prob and r.lower_tail_clearance >= best.lower_tail_clearance - clearance_tolerance
    )
    if not layer:
        raise OracleRegretError("safety-optimal layer must never be empty (the best action always belongs to it)")
    return layer


@dataclass(frozen=True)
class MethodRegret:
    """guide.md R4-4R2 point 6/8: one method's (full/mean/cv) regret at
    one audited decision, graded against the population-aligned truth
    oracle for the SAME (non-reciprocal-only) population that method's
    own ranking used. ``efficiency_regret`` is None whenever
    ``chosen_action`` is NOT in the safety-optimal layer -- guide.md is
    explicit that an efficiency claim is only valid when safety is tied,
    never as a trade-off against more collision/clearance risk."""

    collision_regret: float
    clearance_regret: float
    efficiency_regret: Optional[float]
    normalized_rank_regret: float
    in_safety_optimal_layer: bool


def compute_method_regret(
    chosen_action: int,
    oracle_results: Sequence[CounterfactualCandidateResult],
    clearance_tolerance: float,
) -> MethodRegret:
    """guide.md R4-4R2 point 5-6: the oracle's own full lexicographic
    ranking of all 80 candidates picks the true optimum; a method's
    regret is measured against THAT ranking, not against a single other
    method's chosen action."""
    by_idx = {r.action_index: r for r in oracle_results}
    if chosen_action not in by_idx:
        raise OracleRegretError(f"chosen_action {chosen_action} is not among the oracle-evaluated candidates")
    ranked = rank_counterfactual_candidates(oracle_results, clearance_tolerance)
    best = by_idx[ranked[0]]
    chosen = by_idx[chosen_action]
    layer = build_safety_optimal_layer(oracle_results, clearance_tolerance)
    in_layer = chosen_action in layer

    efficiency_regret: Optional[float] = None
    if in_layer:
        best_progress_in_layer = max(by_idx[i].expected_progress for i in layer)
        efficiency_regret = best_progress_in_layer - chosen.expected_progress

    rank_position = ranked.index(chosen_action)
    normalized_rank_regret = rank_position / (len(ranked) - 1) if len(ranked) > 1 else 0.0

    return MethodRegret(
        collision_regret=chosen.collision_prob - best.collision_prob,
        clearance_regret=best.lower_tail_clearance - chosen.lower_tail_clearance,
        efficiency_regret=efficiency_regret,
        normalized_rank_regret=normalized_rank_regret,
        in_safety_optimal_layer=in_layer,
    )


@dataclass(frozen=True)
class AuditRecord:
    """One population-aligned audit decision -- guide.md R4-4R2's real
    gating unit, superseding R4-4R's ``RiskOpportunityRecord.oracle_regret_*``
    fields (which compared whole-scene method choices against a
    subset-only oracle)."""

    episode_seed: int
    decision_seed: int
    profile: str
    full: MethodRegret
    mean: MethodRegret
    cv: MethodRegret


def _group_by_episode(records: Sequence[AuditRecord], extractor) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for r in records:
        value = extractor(r)
        if value is None:
            continue
        out.setdefault(r.episode_seed, []).append(float(value))
    return out


def check_oracle_regret_gate(
    records: Sequence[AuditRecord],
    min_states: int = 20,
    min_episodes: int = 10,
    non_inferiority_margin: float = 1e-6,
    ci_resamples: int = 2000,
    ci_confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> Dict[str, object]:
    """guide.md R4-4R2's decision rule:

    - full's safety (collision_regret, clearance_regret) must not be
      worse than mean/CV's, on average, beyond ``non_inferiority_margin``.
    - at least ONE of vs_mean / vs_cv must show a normalized_rank_regret
      95% CI (paired, episode-blocked) lower bound > 0 for
      (other - full) -- i.e. full is significantly closer to the true
      optimum than that baseline.
    - efficiency_regret is reported only among states where BOTH methods
      land in the oracle's safety-optimal layer (informational; never a
      pass/fail criterion on its own).
    - if the significant-improvement test fails everywhere, this FAILS
      -- guide.md is explicit that "全部为0" after fixing the population
      mismatch means the full posterior changes actions without a
      demonstrated real decision benefit, and R4-4 stays FAIL.
    """
    reasons: List[str] = []
    n = len(records)
    if n < min_states:
        reasons.append(f"n_audit_states={n} < {min_states}")
    episode_ids = {r.episode_seed for r in records}
    if len(episode_ids) < min_episodes:
        reasons.append(f"n_audit_episodes={len(episode_ids)} < {min_episodes}")
    if n == 0 or not episode_ids:
        return {"passed": False, "reasons": reasons, "metrics": {}}

    metrics: Dict[str, object] = {}
    significant_improvement_found = False
    for label, other_attr in (("vs_mean", "mean"), ("vs_cv", "cv")):
        full_collision = _group_by_episode(records, lambda r: r.full.collision_regret)
        other_collision = _group_by_episode(records, lambda r: getattr(r, other_attr).collision_regret)
        mean_full_collision = float(np.mean([v for vs in full_collision.values() for v in vs]))
        mean_other_collision = float(np.mean([v for vs in other_collision.values() for v in vs]))
        if mean_full_collision > mean_other_collision + non_inferiority_margin:
            reasons.append(
                f"full collision_regret ({mean_full_collision:.6f}) worse than {other_attr} ({mean_other_collision:.6f})"
            )

        full_clearance = _group_by_episode(records, lambda r: r.full.clearance_regret)
        other_clearance = _group_by_episode(records, lambda r: getattr(r, other_attr).clearance_regret)
        mean_full_clearance = float(np.mean([v for vs in full_clearance.values() for v in vs]))
        mean_other_clearance = float(np.mean([v for vs in other_clearance.values() for v in vs]))
        if mean_full_clearance > mean_other_clearance + non_inferiority_margin:
            reasons.append(
                f"full clearance_regret ({mean_full_clearance:.6f}) worse than {other_attr} ({mean_other_clearance:.6f})"
            )

        full_rank = _group_by_episode(records, lambda r: r.full.normalized_rank_regret)
        other_rank = _group_by_episode(records, lambda r: getattr(r, other_attr).normalized_rank_regret)
        rank_boot = paired_difference_bootstrap(
            other_rank, full_rank, statistic_fn=np.mean, n_resamples=ci_resamples,
            confidence_level=ci_confidence, seed=bootstrap_seed,
        )
        entry = {
            "mean_full_collision_regret": mean_full_collision, "mean_other_collision_regret": mean_other_collision,
            "mean_full_clearance_regret": mean_full_clearance, "mean_other_clearance_regret": mean_other_clearance,
            "rank_regret_improvement": {"point": rank_boot.point, "ci_low": rank_boot.ci_low, "ci_high": rank_boot.ci_high},
        }
        if rank_boot.ci_low > 0.0:
            significant_improvement_found = True

        both_in_layer = [r for r in records if r.full.in_safety_optimal_layer and getattr(r, other_attr).in_safety_optimal_layer]
        if len(both_in_layer) >= 2:
            full_eff = _group_by_episode(both_in_layer, lambda r: r.full.efficiency_regret)
            other_eff = _group_by_episode(both_in_layer, lambda r: getattr(r, other_attr).efficiency_regret)
            if full_eff and other_eff:
                eff_boot = paired_difference_bootstrap(
                    other_eff, full_eff, statistic_fn=np.mean, n_resamples=ci_resamples,
                    confidence_level=ci_confidence, seed=bootstrap_seed,
                )
                entry["efficiency_regret_improvement"] = {
                    "point": eff_boot.point, "ci_low": eff_boot.ci_low, "ci_high": eff_boot.ci_high,
                    "n_co_safe_states": len(both_in_layer),
                }
        metrics[label] = entry

    if not significant_improvement_found:
        reasons.append(
            "neither vs_mean nor vs_cv shows a normalized_rank_regret 95% CI lower bound > 0 -- "
            "full posterior changes actions without a demonstrated real decision benefit"
        )

    return {"passed": not reasons, "reasons": reasons, "metrics": metrics}
