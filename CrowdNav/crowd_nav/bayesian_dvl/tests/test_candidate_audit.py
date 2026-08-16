"""The pre-training audit must REFUSE, not warn.

Each test breaks one property and asserts the whole audit fails, because the
failure mode being guarded against is a pipeline that ran to completion with
a broken candidate model and produced a publishable-looking number.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from crowd_nav.bayesian_dvl.candidate_audit import (
    MAX_MEAN_COVERAGE_ERROR_M, MAX_MEAN_SPEED_RESIDUAL, MAX_PERSISTENTLY_FLAT_RATE,
    MAX_UNEXPECTED_SINGLE_CANDIDATE_RATE, MAX_WORST_COVERAGE_ERROR_M, AuditResult,
    CandidateAuditError, audit_permutation_invariance, audit_scenario, run_pretraining_audit,
)
from crowd_nav.bayesian_dvl.intent_policy import HUMAN_FEATURE_DIM_V5
from crowd_nav.bayesian_dvl.junction_scenario import (
    junction_crowd_role_of_seed,
    AMBIGUOUS_TRACK_INDEX, JUNCTION_CROWD_HELDOUT_SEEDS, JUNCTION_CROWD_TRAIN_SEEDS,
    JunctionCrowdEpisodeConfig, build_junction_crowd_episode, maybe_reveal_crowd_exit,
    public_junction_crowd_scene,
)
from crowd_nav.bayesian_dvl.model import DistributionalValueModel

ENV = Path(__file__).resolve().parents[2] / "configs" / "env_bayesian_dvl.config"


def _episodes(seeds, is_heldout):
    for seed in seeds:
        env, _robot, true_exit = build_junction_crowd_episode(
            ENV, JunctionCrowdEpisodeConfig(episode_seed=seed, role=junction_crowd_role_of_seed(seed)))
        state = {"wp": False}

        def advance(env=env, true_exit=true_exit, state=state, hd=is_heldout):
            state["wp"] = maybe_reveal_crowd_exit(
                env.humans[AMBIGUOUS_TRACK_INDEX], true_exit, state["wp"], is_heldout=hd)
        yield env, advance


def _clean(scenario="ok"):
    return AuditResult(scenario=scenario, n_episodes=5, n_observations=500,
                       mean_coverage_error_m=0.2, worst_coverage_error_m=0.4,
                       mean_speed_residual=0.2, single_candidate_rate=0.0,
                       persistently_flat_rate=0.05)


@pytest.mark.parametrize("is_heldout,seeds", [
    (False, list(JUNCTION_CROWD_TRAIN_SEEDS)[:8]),
    (True, list(JUNCTION_CROWD_HELDOUT_SEEDS)[:8]),
])
def test_the_current_junction_crowd_model_passes_the_audit(is_heldout, seeds):
    r = audit_scenario(_episodes(seeds, is_heldout), public_junction_crowd_scene(is_heldout=is_heldout),
                       scenario="junction_crowd", ambiguous_index=AMBIGUOUS_TRACK_INDEX, max_steps=30)
    assert r.passed, r.failures
    assert r.n_observations > 0 and r.n_episodes == len(seeds)


@pytest.mark.parametrize("is_heldout", [False, True])
def test_junction_audit_oracle_uses_the_public_crossing_dictionary(is_heldout):
    """Assignment regret must compare against the same public rule used by
    the tracker, including the crossing-band candidates.

    Before this check, the audit used only the two junction exits as its
    dictionary.  A lateral crosser could therefore have a negative regret:
    its assigned crossing candidate was closer to the hidden endpoint than
    the incomplete "oracle".  That made the audit metric mathematically
    incoherent even though the candidate model itself was correct.
    """
    r = audit_scenario(
        _episodes((JUNCTION_CROWD_HELDOUT_SEEDS if is_heldout else JUNCTION_CROWD_TRAIN_SEEDS)[:8], is_heldout),
        public_junction_crowd_scene(is_heldout=is_heldout),
        scenario="junction_crowd", ambiguous_index=AMBIGUOUS_TRACK_INDEX, max_steps=20,
    )
    assert r.n_regret_offenders == 0, r.failures
    assert r.worst_assignment_regret_m >= -1e-9


def test_audit_measures_something_and_does_not_pass_vacuously():
    """A vacuous audit (no observations) would pass every budget with zeros;
    the checks below only mean anything if data was actually collected."""
    r = audit_scenario(_episodes(list(JUNCTION_CROWD_TRAIN_SEEDS)[:3], False),
                       public_junction_crowd_scene(is_heldout=False),
                       scenario="junction_crowd", ambiguous_index=AMBIGUOUS_TRACK_INDEX, max_steps=20)
    assert r.n_observations >= 3 * 20 * 5
    assert r.mean_speed_residual > 0.0, "a real residual was measured, not an empty mean"
    assert r.mean_coverage_error_m > 0.0


@pytest.mark.parametrize("field,value,needle", [
    ("mean_coverage_error_m", MAX_MEAN_COVERAGE_ERROR_M + 0.1, "coverage"),
    ("worst_coverage_error_m", MAX_WORST_COVERAGE_ERROR_M + 0.1, "worst candidate-coverage"),
    ("mean_speed_residual", MAX_MEAN_SPEED_RESIDUAL + 0.1, "velocity residual"),
    ("single_candidate_rate", MAX_UNEXPECTED_SINGLE_CANDIDATE_RATE + 0.1, "ONE candidate"),
    ("persistently_flat_rate", MAX_PERSISTENTLY_FLAT_RATE + 0.1, "EXACTLY uniform"),
])
def test_each_budget_breach_refuses_training(field, value, needle, tmp_path):
    from crowd_nav.bayesian_dvl import candidate_audit as CA
    r = _clean("broken")
    setattr(r, field, value)
    # re-run the threshold logic the same way audit_scenario does
    r.failures = []
    checks = [
        (r.mean_coverage_error_m > CA.MAX_MEAN_COVERAGE_ERROR_M, "coverage"),
        (r.worst_coverage_error_m > CA.MAX_WORST_COVERAGE_ERROR_M, "worst candidate-coverage"),
        (r.mean_speed_residual > CA.MAX_MEAN_SPEED_RESIDUAL, "velocity residual"),
        (r.single_candidate_rate > CA.MAX_UNEXPECTED_SINGLE_CANDIDATE_RATE, "ONE candidate"),
        (r.persistently_flat_rate > CA.MAX_PERSISTENTLY_FLAT_RATE, "EXACTLY uniform"),
    ]
    r.failures = [msg for hit, msg in checks if hit]
    assert needle in " ".join(r.failures)
    with pytest.raises(CandidateAuditError):
        run_pretraining_audit([r], 0.0, out_path=tmp_path / "audit.json")
    import json
    payload = json.loads((tmp_path / "audit.json").read_text())
    assert payload["passed"] is False, "the report must be written even when training is refused"


def test_permutation_breach_refuses_training(tmp_path):
    with pytest.raises(CandidateAuditError, match="candidate ORDER"):
        run_pretraining_audit([_clean()], permutation_delta=1e-3, out_path=tmp_path / "a.json")


def test_a_clean_audit_passes_and_records_its_budgets(tmp_path):
    payload = run_pretraining_audit([_clean()], 0.0, out_path=tmp_path / "a.json")
    assert payload["passed"] is True
    assert payload["budgets"]["max_mean_coverage_error_m"] == MAX_MEAN_COVERAGE_ERROR_M
    assert (tmp_path / "a.json").exists()


def test_permutation_check_is_zero_for_the_pooled_encoder():
    torch.manual_seed(0)
    model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V5)
    assert audit_permutation_invariance(model) == pytest.approx(0.0, abs=1e-7)


def test_audit_module_is_not_imported_by_the_belief_chain():
    """It reads hidden goals on purpose. That is fine for an offline scorer
    and fatal for the chain that builds the posterior."""
    import crowd_nav.bayesian_dvl.candidate_audit as CA
    src = Path(CA.__file__).read_text()
    assert ".gx" in src or "gx" in src, "precondition: this module does read hidden goals"
    for module in ("intent_tracker", "scene_candidates", "intent_policy", "intent_train"):
        text = (Path(CA.__file__).parent / f"{module}.py").read_text()
        assert "candidate_audit" not in text, (
            f"{module}.py imports candidate_audit, which would pull hidden-goal access into the "
            "belief chain")
