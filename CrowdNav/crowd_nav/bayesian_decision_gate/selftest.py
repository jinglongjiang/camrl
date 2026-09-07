#!/usr/bin/env python3
"""GPU-free, environment-free correctness checks for the Bayesian decision
gate's pure logic: candidate-action generation, ground-truth counterfactual
rollout math, tie-tolerant ranking, event_any/event_near window labeling,
bootstrap CI, transition extraction, and the Gate-A/Gate-B pre-registered
criteria. Run before any real (GPU/simulation) execution.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.bayesian_decision_gate.bootstrap import (  # noqa: E402
    block_bootstrap_ci,
    paired_block_bootstrap_diff,
    spearman_corr,
    top_k_hit_rate,
)
from crowd_nav.bayesian_decision_gate.evaluate_action_ranking import (  # noqa: E402
    _seed_blocks,
    aggregate,
    build_candidate_actions,
    decision_points,
    event_window_labels,
    ground_truth_for_step,
    rank_and_hit,
)
from crowd_nav.bayesian_decision_gate.fit_models import (  # noqa: E402
    extract_transitions_variable,
)
from crowd_nav.bayesian_decision_gate.protocol import lock_action_grid  # noqa: E402
from crowd_nav.bayesian_decision_gate.run_gate import (  # noqa: E402
    combine_gates,
    evaluate_one_gate,
    GATE_MIN_EVENT_DECISIONS,
    GATE_MIN_OPPORTUNITY_RATE,
    GATE_PRIMARY_DENSITY,
    GATE_SECONDARY_DENSITY,
    GATE_WINDOW,
    HORIZONS,
    PRIMARY_HORIZON,
)
from crowd_nav.contracts import GRID  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


def test_lock_action_grid_matches_belief_mdp_production():
    """The candidate action grid must come from env_belief_mdp.config (what
    frozen_k3 and the real policy actually use), not whatever configs/env.config
    happens to be reachable from cwd -- 80 actions (5 speeds x 16 headings),
    include_stop=False."""
    grid = lock_action_grid()
    check("lock_action_grid_n_speeds", int(grid["n_speeds"]) == 5, f"got {grid['n_speeds']}")
    check("lock_action_grid_n_headings", int(grid["n_headings"]) == 16, f"got {grid['n_headings']}")
    check("lock_action_grid_no_stop", not bool(grid.get("include_stop", False)))


def test_candidate_actions_match_grid():
    actions = build_candidate_actions()
    expected_n = int(GRID["n_speeds"]) * int(GRID["n_headings"]) + int(
        bool(GRID.get("include_stop", False))
    )
    check(
        "candidate_actions_count",
        len(actions) == expected_n,
        f"got {len(actions)}, expected {expected_n}",
    )
    check("candidate_actions_is_80", len(actions) == 80, f"got {len(actions)}")
    speeds = np.linalg.norm(actions, axis=1)
    check(
        "candidate_actions_within_v_max",
        bool(np.all(speeds <= float(GRID["v_max"]) + 1e-6)),
    )


def test_spearman_corr():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    check("spearman_perfect_positive", abs(spearman_corr(x, y) - 1.0) < 1e-9)
    check("spearman_perfect_negative", abs(spearman_corr(x, y[::-1]) - (-1.0)) < 1e-9)
    check("spearman_constant_is_nan", np.isnan(spearman_corr(x, np.ones_like(x))))
    x_ties = np.array([1.0, 2.0, 3.0, 3.0, 5.0])
    y_plain = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    corr = spearman_corr(x_ties, y_plain)
    check("spearman_ties_handled", 0.9 < corr < 1.0, f"corr={corr:.4f}")


def test_top_k_hit_rate():
    ranks = [0, 0, 1, 2, 3]
    check("top1_hit_rate", abs(top_k_hit_rate(ranks, 1) - 0.4) < 1e-9)
    check("top3_hit_rate", abs(top_k_hit_rate(ranks, 3) - 0.8) < 1e-9)
    check("top_k_empty_is_nan", np.isnan(top_k_hit_rate([], 1)))


def test_rank_and_hit_tie_tolerance():
    """The core Fix-2 correctness test: several actions tied at the true
    minimum risk. A model choosing ANY of the tied-safest actions must count
    as a Top-1 hit and zero regret -- argsort-based ranking would wrongly
    fail every choice except whichever one argsort happened to place first."""
    true_risk = np.array([0.0, 0.0, 0.0, 0.5, 0.9])
    for tied_choice in (0, 1, 2):
        rank, top1, topk, regret = rank_and_hit(true_risk, tied_choice, top_k=1, rank_tolerance=1e-3)
        check(
            f"tied_choice_{tied_choice}_is_top1_hit",
            top1 == 1.0 and rank == 0 and abs(regret) < 1e-9,
            f"rank={rank} top1={top1} regret={regret}",
        )
    rank, top1, topk, regret = rank_and_hit(true_risk, 3, top_k=1, rank_tolerance=1e-3)
    check("clearly_worse_choice_is_not_top1", top1 == 0.0 and rank == 3, f"rank={rank} top1={top1}")

    # Near-tie within tolerance must also count as tied (floating point noise).
    near_tied_risk = np.array([0.30000, 0.30000009, 0.9])
    rank_a, top1_a, _, regret_a = rank_and_hit(near_tied_risk, 1, top_k=1, rank_tolerance=1e-3)
    check("near_tied_within_tolerance_is_top1", top1_a == 1.0 and abs(regret_a) < 1e-3)


def test_block_bootstrap_ci():
    rng = np.random.default_rng(0)
    blocks = [np.full(20, 1.0), np.full(20, 1.0), np.full(20, 1.0)]
    mean_val, lo, hi = block_bootstrap_ci(blocks, replicates=500, rng=rng)
    check("bootstrap_constant_blocks_mean", abs(mean_val - 1.0) < 1e-9)
    check("bootstrap_constant_blocks_ci_tight", abs(hi - lo) < 1e-9)

    rng2 = np.random.default_rng(1)
    varied_blocks = [np.zeros(50), np.ones(50)]
    mean_val2, lo2, hi2 = block_bootstrap_ci(varied_blocks, replicates=2000, rng=rng2)
    check("bootstrap_varied_blocks_mean", abs(mean_val2 - 0.5) < 1e-9)
    check("bootstrap_varied_blocks_ci_brackets_extremes", lo2 < 0.5 < hi2 or (lo2 <= 0.0 and hi2 >= 1.0))

    empty_mean, empty_lo, empty_hi = block_bootstrap_ci([], replicates=10)
    check("bootstrap_empty_is_nan", np.isnan(empty_mean) and np.isnan(empty_lo) and np.isnan(empty_hi))


def test_paired_block_bootstrap_diff():
    rng = np.random.default_rng(0)
    a = [np.full(20, 2.0), np.full(20, 2.0)]
    b = [np.full(20, 1.0), np.full(20, 1.0)]
    mean_diff, lo, hi = paired_block_bootstrap_diff(a, b, replicates=500, rng=rng)
    check("paired_bootstrap_diff_correct", abs(mean_diff - 1.0) < 1e-9)
    check("paired_bootstrap_diff_ci_excludes_zero", lo > 0.0, f"lo={lo:.4f}")

    identical_a = [np.full(20, 3.0)]
    mean_same, lo_same, hi_same = paired_block_bootstrap_diff(identical_a, identical_a, replicates=100, rng=rng)
    check("paired_bootstrap_diff_zero_for_identical", abs(mean_same) < 1e-9)


def test_decision_points():
    check("decision_points_too_short_is_empty", decision_points(T=4, horizon=5, stride=1, max_decisions=10) == [])
    points = decision_points(T=30, horizon=5, stride=5, max_decisions=3)
    check("decision_points_respects_stride", points == [0, 5, 10])
    points_capped = decision_points(T=100, horizon=5, stride=1, max_decisions=4)
    check("decision_points_respects_cap", len(points_capped) == 4)


def test_event_window_labels_near_vs_any():
    """20-person-density regression check: a non-nominal pedestrian FAR from
    the robot must set event_any but NOT event_near -- this is exactly the
    dilution Fix-5 exists to prevent."""
    num_humans = 3
    T = 6
    width = 9 + 5 * num_humans
    obs_seq = np.zeros((T, width), dtype=np.float32)
    for t in range(T):
        obs_seq[t, :2] = [0.0, 0.0]  # robot always at origin
        # ped 0: far away (20m), ped 1: close (1m), ped 2: unused/zero
        obs_seq[t, 9:11] = [20.0, 0.0]
        obs_seq[t, 14:16] = [1.0, 0.0]

    modes_far_only = np.zeros((T, num_humans), dtype=np.int8)
    modes_far_only[3, 0] = 2  # only the FAR pedestrian has an event
    far_labels = event_window_labels(obs_seq, modes_far_only, num_humans, t=0, h_max=4, near_radius=3.0)
    check("event_any_true_for_far_pedestrian", far_labels["event_any"] is True)
    check("event_near_false_for_far_pedestrian", far_labels["event_near"] is False, str(far_labels))

    modes_near = np.zeros((T, num_humans), dtype=np.int8)
    modes_near[3, 1] = 2  # the CLOSE pedestrian has an event
    near_labels = event_window_labels(obs_seq, modes_near, num_humans, t=0, h_max=4, near_radius=3.0)
    check("event_near_true_for_close_pedestrian", near_labels["event_near"] is True, str(near_labels))

    modes_none = np.zeros((T, num_humans), dtype=np.int8)
    none_labels = event_window_labels(obs_seq, modes_none, num_humans, t=0, h_max=4, near_radius=3.0)
    check("no_event_when_all_nominal", none_labels["event_any"] is False and none_labels["event_near"] is False)


def test_ground_truth_rollout_stationary_pedestrian():
    """One stationary pedestrian directly ahead: an action moving straight at
    it must show higher true_risk than an action moving away, and 'stand
    still' must NOT be flagged as high-progress (so the near-optimal-by-
    -progress filter would correctly exclude it in a real evaluation)."""
    num_humans = 1
    T = 3
    width = 9 + 5 * num_humans
    obs_seq = np.zeros((T, width), dtype=np.float32)
    for t in range(T):
        obs_seq[t, 0:2] = [0.0, 0.0]
        obs_seq[t, 4] = 0.3
        obs_seq[t, 6:8] = [10.0, 0.0]
    for t in range(T):
        obs_seq[t, 9:11] = [1.0, 0.0]
        obs_seq[t, 13] = 0.3

    toward = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    result = ground_truth_for_step(
        obs_seq, num_humans, t=0, horizon=2, candidate_actions=toward,
        dt=0.25, safe_distance=0.2, pedestrian_aggregation="max",
    )
    check("ground_truth_returns_result", result is not None)
    if result is not None:
        check(
            "ground_truth_toward_riskier_than_away",
            result["true_risk"][0] > result["true_risk"][1],
            f"toward={result['true_risk'][0]:.4f} away={result['true_risk'][1]:.4f}",
        )
        check(
            "ground_truth_stop_has_least_progress",
            result["progress"][2] < result["progress"][0],
            f"stop_progress={result['progress'][2]:.4f} toward_progress={result['progress'][0]:.4f}",
        )
        check(
            "ground_truth_toward_has_most_progress",
            result["progress"][0] > result["progress"][1],
        )
        check("ground_truth_has_collision_field", "true_collision" in result)
        check("ground_truth_has_near_miss_field", "true_near_miss" in result)


def test_ground_truth_insufficient_horizon_returns_none():
    obs_seq = np.zeros((2, 14), dtype=np.float32)
    result = ground_truth_for_step(
        obs_seq, num_humans=1, t=1, horizon=5,
        candidate_actions=np.zeros((3, 2)), dt=0.25, safe_distance=0.2,
        pedestrian_aggregation="max",
    )
    check("ground_truth_none_when_no_future", result is None)


def test_extract_transitions_variable_generalizes_beyond_five_peds():
    """The whole point of this module vs. the legacy 34-D-only fitter: this
    must correctly extract transitions for a 20-pedestrian episode, not
    silently truncate to 5."""
    num_humans = 20
    T = 5
    width = 9 + 5 * num_humans
    obs = np.zeros((T, width), dtype=np.float32)
    act = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        obs[t, :9] = [0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 5.0, 5.0, 0.0]
        for p in range(num_humans):
            s = 9 + p * 5
            obs[t, s:s + 5] = [float(p), float(t), 0.1, 0.0, 0.3]

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "synthetic_20person.npz"
        np.savez(
            path,
            obs=np.asarray([obs], dtype=object),
            act=np.asarray([act], dtype=object),
            num_humans=np.asarray([num_humans], dtype=np.int64),
        )
        rows = extract_transitions_variable([path], dt=0.25)
        peds_seen = sorted(set(r["ped"] for r in rows))
        check(
            "extract_transitions_sees_all_20_pedestrians",
            peds_seen == list(range(num_humans)),
            f"peds_seen={peds_seen}",
        )
        check(
            "extract_transitions_expected_row_count",
            len(rows) == num_humans * (T - 1),
            f"got {len(rows)}, expected {num_humans * (T - 1)}",
        )


def _metric_block(mean, n_decisions):
    return {"mean": mean, "ci_lo": mean - 0.05, "ci_hi": mean + 0.05, "n_decisions": n_decisions, "n_seeds": 5, "per_seed_means": {}}


def _horizon_entry(
    *, n, window, top1_model, top1_cv, meaningful_disagreement_rate,
    regret_mean, regret_ci_lo, opportunity_rate, model_name,
):
    return {
        "per_model": {
            model_name: {
                window: {
                    "spearman": _metric_block(0.5, n),
                    "top1_hit": _metric_block(top1_model, n),
                },
            },
            "cv": {
                window: {
                    "spearman": _metric_block(0.45, n),
                    "top1_hit": _metric_block(top1_cv, n),
                },
            },
        },
        "cv_vs_gdbn": {
            model_name: {
                window: {
                    "meaningful_disagreement_rate": {"mean": meaningful_disagreement_rate},
                    "regret_reduction_vs_cv": {
                        "mean": regret_mean, "ci_lo": regret_ci_lo, "ci_hi": regret_ci_lo + 0.1,
                    },
                },
            },
        },
        "opportunity": {
            window: {"mean": opportunity_rate, "n_decisions": n},
        },
    }


def _synth_all_results(
    *,
    model_name: str,
    density: str = GATE_PRIMARY_DENSITY,
    window: str = GATE_WINDOW,
    n_decisions_by_horizon,
    regret_reduction_mean_by_horizon,
    regret_reduction_ci_lo_by_horizon,
    meaningful_disagreement_rate: float = 0.15,
    opportunity_rate: float = 0.20,
    top1_model: float = 0.6,
    top1_cv: float = 0.4,
    nominal_regret_ci_lo: float = -0.005,
    density_20_n: int = GATE_MIN_EVENT_DECISIONS + 50,
    density_20_regret_ci_lo: float = 0.005,
):
    all_results = {}
    for horizon in HORIZONS:
        n = n_decisions_by_horizon[horizon]
        key = f"heldout_nonstationary__{density}__h{horizon}"
        all_results[key] = _horizon_entry(
            n=n, window=window, top1_model=top1_model, top1_cv=top1_cv,
            meaningful_disagreement_rate=meaningful_disagreement_rate,
            regret_mean=regret_reduction_mean_by_horizon[horizon],
            regret_ci_lo=regret_reduction_ci_lo_by_horizon[horizon],
            opportunity_rate=opportunity_rate, model_name=model_name,
        )
        # Secondary (20-person) density is only checked at the primary
        # horizon (see _evaluate_secondary_density) -- populate it there.
        if horizon == PRIMARY_HORIZON:
            key_20 = f"heldout_nonstationary__{GATE_SECONDARY_DENSITY}__h{horizon}"
            all_results[key_20] = _horizon_entry(
                n=density_20_n, window=window, top1_model=top1_model, top1_cv=top1_cv,
                meaningful_disagreement_rate=meaningful_disagreement_rate,
                regret_mean=0.02, regret_ci_lo=density_20_regret_ci_lo,
                opportunity_rate=opportunity_rate, model_name=model_name,
            )
    nominal_key = f"nominal__{density}__h{PRIMARY_HORIZON}"
    all_results[nominal_key] = {
        "cv_vs_gdbn": {
            model_name: {
                "all": {
                    "regret_reduction_vs_cv": {"mean": 0.0, "ci_lo": nominal_regret_ci_lo, "ci_hi": 0.005},
                },
            },
        },
    }
    return all_results


_FULL_N = {1: GATE_MIN_EVENT_DECISIONS + 50, 3: GATE_MIN_EVENT_DECISIONS + 50, 5: GATE_MIN_EVENT_DECISIONS + 50}
_FULL_REGRET_MEAN = {1: 0.03, 3: 0.04, 5: 0.05}
_FULL_REGRET_CI_LO = {1: 0.005, 3: 0.008, 5: 0.01}


def test_gate_a_pass_case():
    all_results = _synth_all_results(
        model_name="frozen_k3",
        n_decisions_by_horizon=_FULL_N,
        regret_reduction_mean_by_horizon=_FULL_REGRET_MEAN,
        regret_reduction_ci_lo_by_horizon=_FULL_REGRET_CI_LO,
    )
    result = evaluate_one_gate(all_results, "frozen_k3")
    check("gate_pass_case_passes", result["status"] == "PASS", json.dumps(result["checks"]))


def test_gate_inconclusive_on_underpowered():
    all_results = _synth_all_results(
        model_name="frozen_k3",
        n_decisions_by_horizon={1: 10, 3: 10, 5: 10},
        regret_reduction_mean_by_horizon=_FULL_REGRET_MEAN,
        regret_reduction_ci_lo_by_horizon=_FULL_REGRET_CI_LO,
    )
    result = evaluate_one_gate(all_results, "frozen_k3")
    check(
        "gate_inconclusive_when_underpowered",
        result["status"] == "INCONCLUSIVE" and "insufficient_event_decisions" in result["reasons"],
        json.dumps(result["reasons"]),
    )


def test_gate_inconclusive_on_insufficient_opportunity():
    """Enough decisions were sampled, but the near-optimal action subset
    almost never had any real risk variation -- the protocol posed no real
    questions, so this must be INCONCLUSIVE, NOT a FAIL (a FAIL here would
    wrongly imply GDBN failed a real test it was never actually given)."""
    all_results = _synth_all_results(
        model_name="frozen_k3",
        n_decisions_by_horizon=_FULL_N,
        regret_reduction_mean_by_horizon=_FULL_REGRET_MEAN,
        regret_reduction_ci_lo_by_horizon=_FULL_REGRET_CI_LO,
        opportunity_rate=0.01,
    )
    result = evaluate_one_gate(all_results, "frozen_k3")
    check(
        "gate_inconclusive_on_insufficient_opportunity",
        result["status"] == "INCONCLUSIVE" and any("insufficient_opportunity" in r for r in result["reasons"]),
        json.dumps(result["reasons"]),
    )


def test_gate_fails_on_low_disagreement_with_sufficient_opportunity():
    """The key three-state distinction (Fix B): real opportunities existed
    (opportunity_rate is healthy) but the model rarely chose differently
    from CV anyway -- this is a genuine FAIL (Bayesian belief didn't change
    decisions), not an inconclusive underpowered sample."""
    all_results = _synth_all_results(
        model_name="frozen_k3",
        n_decisions_by_horizon=_FULL_N,
        regret_reduction_mean_by_horizon=_FULL_REGRET_MEAN,
        regret_reduction_ci_lo_by_horizon=_FULL_REGRET_CI_LO,
        opportunity_rate=0.25,
        meaningful_disagreement_rate=0.01,
    )
    result = evaluate_one_gate(all_results, "frozen_k3")
    check(
        "gate_fails_on_low_disagreement_despite_opportunity",
        result["status"] == "FAIL" and "no_meaningful_disagreement" in result["reasons"][0],
        json.dumps(result["reasons"]),
    )


def test_gate_fails_when_regret_ci_includes_zero():
    all_results = _synth_all_results(
        model_name="frozen_k3",
        n_decisions_by_horizon=_FULL_N,
        regret_reduction_mean_by_horizon=_FULL_REGRET_MEAN,
        regret_reduction_ci_lo_by_horizon={1: -0.01, 3: -0.005, 5: -0.002},
    )
    result = evaluate_one_gate(all_results, "frozen_k3")
    check(
        "gate_fails_when_regret_ci_includes_zero",
        result["status"] == "FAIL" and not result["checks"]["regret_reduction_significant"],
    )


def test_gate_fails_on_horizon_disagreement():
    """H=1 says GDBN is meaningfully better, H=5 says CV is better -- the
    open-loop counterfactual oracle's approximation error likely dominates,
    so the gate must NOT pass even though the primary horizon (H=5) alone
    looks significant."""
    all_results = _synth_all_results(
        model_name="frozen_k3",
        n_decisions_by_horizon=_FULL_N,
        regret_reduction_mean_by_horizon={1: -0.05, 3: 0.01, 5: 0.05},
        regret_reduction_ci_lo_by_horizon={1: -0.08, 3: -0.01, 5: 0.01},
    )
    result = evaluate_one_gate(all_results, "frozen_k3")
    check(
        "gate_fails_on_horizon_disagreement",
        result["status"] == "FAIL" and not result["checks"]["horizon_agreement"],
        json.dumps(result["checks"]),
    )


def test_gate_fails_on_nominal_regression():
    all_results = _synth_all_results(
        model_name="frozen_k3",
        n_decisions_by_horizon=_FULL_N,
        regret_reduction_mean_by_horizon=_FULL_REGRET_MEAN,
        regret_reduction_ci_lo_by_horizon=_FULL_REGRET_CI_LO,
        nominal_regret_ci_lo=-0.05,
    )
    result = evaluate_one_gate(all_results, "frozen_k3")
    check(
        "gate_fails_on_nominal_regression",
        result["status"] == "FAIL" and not result["checks"]["no_nominal_regression"],
    )


def test_gate_fails_on_20person_regression():
    """Fix 3: a clear 20-person regression must block PASS even though the
    5-person primary-density result alone looks perfect."""
    all_results = _synth_all_results(
        model_name="frozen_k3",
        n_decisions_by_horizon=_FULL_N,
        regret_reduction_mean_by_horizon=_FULL_REGRET_MEAN,
        regret_reduction_ci_lo_by_horizon=_FULL_REGRET_CI_LO,
        density_20_regret_ci_lo=-0.05,
    )
    result = evaluate_one_gate(all_results, "frozen_k3")
    check(
        "gate_fails_on_20person_regression",
        result["status"] == "FAIL" and not result["checks"]["no_20person_regression"],
        json.dumps(result.get("density_20_person")),
    )


def test_gate_inconclusive_on_20person_insufficient_data():
    """5-person alone would PASS, but 20-person has too few decisions to
    certify non-inferiority -- since the paper's final scenarios include
    high-density crowds, this must downgrade to INCONCLUSIVE, not a silent
    PASS-by-omission."""
    all_results = _synth_all_results(
        model_name="frozen_k3",
        n_decisions_by_horizon=_FULL_N,
        regret_reduction_mean_by_horizon=_FULL_REGRET_MEAN,
        regret_reduction_ci_lo_by_horizon=_FULL_REGRET_CI_LO,
        density_20_n=5,
    )
    result = evaluate_one_gate(all_results, "frozen_k3")
    check(
        "gate_inconclusive_on_20person_insufficient_data",
        result["status"] == "INCONCLUSIVE" and any("20person_insufficient_data" in r for r in result["reasons"]),
        json.dumps(result["reasons"]),
    )


def test_combine_gates_tri_state_table():
    pass_gate = {"status": "PASS", "reasons": []}
    fail_gate = {"status": "FAIL", "reasons": ["no_meaningful_disagreement"]}
    inconclusive_gate = {"status": "INCONCLUSIVE", "reasons": ["insufficient_event_decisions"]}

    both_pass = combine_gates(pass_gate, pass_gate)
    check("combine_both_pass", both_pass["status"] == "PASS" and "Route B is ALSO" in both_pass["decision"])

    a_fail_b_pass = combine_gates(fail_gate, pass_gate)
    check(
        "combine_a_fail_b_pass_suggests_route_b",
        a_fail_b_pass["status"] == "FAIL" and "seriously consider Route B" in a_fail_b_pass["decision"],
    )

    a_pass_b_fail = combine_gates(pass_gate, fail_gate)
    check(
        "combine_a_pass_b_fail_keeps_route_a",
        a_pass_b_fail["status"] == "FAIL" and "keep Route A as-is" in a_pass_b_fail["decision"],
    )

    both_fail = combine_gates(fail_gate, fail_gate)
    check(
        "combine_both_fail_rejects_route_b",
        both_fail["status"] == "FAIL" and "Reject Route B" in both_fail["decision"],
    )

    # A single INCONCLUSIVE gate must make the COMBINED result INCONCLUSIVE,
    # never silently resolve to PASS or FAIL -- this is exactly the
    # self-contradiction Fix B removes (previously: "FAIL... [INCONCLUSIVE]").
    a_inconclusive_b_pass = combine_gates(inconclusive_gate, pass_gate)
    check(
        "combine_inconclusive_propagates",
        a_inconclusive_b_pass["status"] == "INCONCLUSIVE" and "Gate-A INCONCLUSIVE" in a_inconclusive_b_pass["decision"],
        a_inconclusive_b_pass["decision"],
    )
    check(
        "combine_inconclusive_never_self_contradicts",
        "FAIL" not in a_inconclusive_b_pass["decision"].split("--")[0],
    )


def test_suite_seed_bootstrap_grouping():
    """Fix A: the bootstrap's top-level block must be the SUITE seed
    (experiment identity, e.g. 2407/3407/...), not the per-episode seed.
    Two suite seeds with three episodes each must produce n_seeds == 2, not
    6 (the episode count) and not 1."""
    def fake_episode(value: float, n_rows: int = 4):
        rows = [
            {
                "t": i, "event_any": True, "event_near": True, "opportunity": True,
                "spearman": value, "top1_hit": 1.0, "topk_hit": 1.0, "regret": 0.0,
                "true_collision": 0.0, "true_near_miss": 0.0, "min_clearance": 1.0,
                "chosen_action_idx": 0,
            }
            for i in range(n_rows)
        ]
        return {
            "per_model_rows": {"cv": rows, "frozen_k3": rows},
            "disagreements": [],
            "opportunities": [
                {"t": i, "opportunity": True, "event_any": True, "event_near": True}
                for i in range(n_rows)
            ],
        }

    episode_results = []
    episode_seeds = []
    for suite_seed in (100, 200):
        for _episode_index in range(3):
            episode_results.append(fake_episode(0.5))
            episode_seeds.append(suite_seed)

    summary = aggregate(episode_results, episode_seeds, ["cv", "frozen_k3"], replicates=50, seed=0)
    n_seeds = summary["per_model"]["frozen_k3"]["all"]["spearman"]["n_seeds"]
    check(
        "suite_seed_grouping_yields_2_not_6",
        n_seeds == 2,
        f"got n_seeds={n_seeds} (episodes=6, suite_seeds=2)",
    )

    # Direct check on the grouping primitive itself.
    blocks = _seed_blocks(episode_results, episode_seeds, lambda ep: [r["spearman"] for r in ep["per_model_rows"]["cv"]])
    check("seed_blocks_dict_has_2_keys", sorted(blocks.keys()) == [100, 200], f"got {sorted(blocks.keys())}")
    check(
        "seed_blocks_concatenates_all_episodes_per_seed",
        len(blocks[100][0]) == 12,  # 3 episodes x 4 rows
        f"got {len(blocks[100][0])}",
    )


def main():
    test_lock_action_grid_matches_belief_mdp_production()
    test_candidate_actions_match_grid()
    test_spearman_corr()
    test_top_k_hit_rate()
    test_rank_and_hit_tie_tolerance()
    test_block_bootstrap_ci()
    test_paired_block_bootstrap_diff()
    test_decision_points()
    test_event_window_labels_near_vs_any()
    test_ground_truth_rollout_stationary_pedestrian()
    test_ground_truth_insufficient_horizon_returns_none()
    test_extract_transitions_variable_generalizes_beyond_five_peds()
    test_suite_seed_bootstrap_grouping()
    test_gate_a_pass_case()
    test_gate_inconclusive_on_underpowered()
    test_gate_inconclusive_on_insufficient_opportunity()
    test_gate_fails_on_low_disagreement_with_sufficient_opportunity()
    test_gate_fails_when_regret_ci_includes_zero()
    test_gate_fails_on_horizon_disagreement()
    test_gate_fails_on_nominal_regression()
    test_gate_fails_on_20person_regression()
    test_gate_inconclusive_on_20person_insufficient_data()
    test_combine_gates_tri_state_table()

    print()
    if FAILURES:
        print(f"[SELFTEST] {len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print("[SELFTEST] all tests passed")


if __name__ == "__main__":
    main()
