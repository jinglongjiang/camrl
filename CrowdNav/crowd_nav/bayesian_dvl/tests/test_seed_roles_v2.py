"""Seed ROLES are the protocol wiring, so they get direct tests.

The V1 failure was not that a rule was wrong, it was that a rule existed in
one place and not another: 97601-97700 was frozen as junction selection-dev
in the seed inventory but never added to the scenario builder's allowlist, so
selection silently ran on the held-out block instead. These tests pin the
wiring end to end rather than the declarations alone.
"""
from pathlib import Path

import numpy as np
import pytest

from crowd_nav.bayesian_dvl.intent_train import (
    PAPER_MAIN_BASE_SEED, STANDARD_SELECTION_DEV_SEEDS, TEST8_AUDIT_BASE_SEED, paper_main_jobs,
)
from crowd_nav.bayesian_dvl.intent_train_cli import _assert_not_formal_seed, IntentCLIError
from crowd_nav.bayesian_dvl.junction_scenario import (
    FORMAL_ONLY_ROLES, HELDOUT_GEOMETRY_ROLES, JUNCTION_CROWD_HELDOUT_SEEDS, JUNCTION_CROWD_IL_SEEDS,
    JUNCTION_CROWD_ONLINE_SEEDS, JUNCTION_CROWD_PAPER_TEST_SEEDS, JUNCTION_CROWD_SEED_ROLES,
    JUNCTION_CROWD_SELECTION_DEV_SEEDS, JUNCTION_CROWD_TRAIN_SEEDS, JUNCTION_CROWD_VALIDATION_SEEDS,
    TRAIN_GEOMETRY_ROLES, JunctionCrowdEpisodeConfig, JunctionScenarioError,
    build_junction_crowd_episode, junction_crowd_role_of_seed,
)

ENV = Path(__file__).resolve().parents[2] / "configs" / "env_bayesian_dvl.config"

ALL_ROLES = ("mechanism_train", "mechanism_heldout", "il", "online", "validation",
             "selection_dev", "paper_test")


def test_the_seven_roles_exist_and_partition_the_seed_space():
    assert set(JUNCTION_CROWD_SEED_ROLES) == set(ALL_ROLES)
    seen = {}
    for role, seeds in JUNCTION_CROWD_SEED_ROLES.items():
        for s in seeds:
            assert s not in seen, f"seed {s} claimed by {seen.get(s)} and {role}"
            seen[s] = role


@pytest.mark.parametrize("role", ALL_ROLES)
def test_every_role_builds_a_real_episode_with_the_right_geometry(role):
    seed = JUNCTION_CROWD_SEED_ROLES[role][0]
    cfg = JunctionCrowdEpisodeConfig(episode_seed=seed, role=role)
    assert cfg.is_heldout == (role in HELDOUT_GEOMETRY_ROLES)
    assert (role in TRAIN_GEOMETRY_ROLES) != (role in HELDOUT_GEOMETRY_ROLES)
    env, robot, true_exit = build_junction_crowd_episode(ENV, cfg)
    assert len(env.humans) == 5 and true_exit in ("left", "right")


def test_a_seed_maps_to_exactly_one_role():
    for role, seeds in JUNCTION_CROWD_SEED_ROLES.items():
        assert junction_crowd_role_of_seed(seeds[0]) == role
        assert junction_crowd_role_of_seed(seeds[-1]) == role
    with pytest.raises(JunctionScenarioError):
        junction_crowd_role_of_seed(12345)


@pytest.mark.parametrize("seed_role,claimed_role", [
    ("paper_test", "il"), ("paper_test", "selection_dev"), ("selection_dev", "paper_test"),
    ("il", "paper_test"), ("online", "validation"), ("mechanism_heldout", "mechanism_train"),
])
def test_mismatched_role_and_seed_is_refused(seed_role, claimed_role):
    """The interface this replaces let any holder of the heldout flag run any
    heldout-block seed, including the paper-test block."""
    seed = JUNCTION_CROWD_SEED_ROLES[seed_role][0]
    with pytest.raises(JunctionScenarioError, match="not in the frozen"):
        JunctionCrowdEpisodeConfig(episode_seed=seed, role=claimed_role)


def test_unknown_role_is_refused():
    with pytest.raises(JunctionScenarioError, match="unknown junction_crowd seed role"):
        JunctionCrowdEpisodeConfig(episode_seed=JUNCTION_CROWD_IL_SEEDS[0], role="heldout")


def test_selection_uses_the_selection_block_and_its_first_episode_is_2_400_000():
    from crowd_nav.bayesian_dvl import intent_selection
    src = Path(intent_selection.__file__).read_text()
    assert "JUNCTION_CROWD_SELECTION_DEV_SEEDS" in src
    assert "JUNCTION_CROWD_HELDOUT_SEEDS" not in src, (
        "selection must not fall back to the mechanism-audit block; selecting a checkpoint on the "
        "episodes that certified the candidate model would couple the two")
    assert JUNCTION_CROWD_SELECTION_DEV_SEEDS[0] == 2_400_000


def test_paper_test_first_episode_is_2_500_000():
    assert JUNCTION_CROWD_PAPER_TEST_SEEDS[0] == 2_500_000
    assert len(JUNCTION_CROWD_PAPER_TEST_SEEDS) == 500


def test_selection_and_paper_blocks_are_disjoint_both_ways():
    sel, paper = set(JUNCTION_CROWD_SELECTION_DEV_SEEDS), set(JUNCTION_CROWD_PAPER_TEST_SEEDS)
    assert not sel & paper
    for s in list(sel)[:5]:
        assert junction_crowd_role_of_seed(s) == "selection_dev"
    for s in list(paper)[:5]:
        assert junction_crowd_role_of_seed(s) == "paper_test"


def test_il_and_online_never_touch_heldout_selection_or_paper():
    trainable = set(JUNCTION_CROWD_IL_SEEDS) | set(JUNCTION_CROWD_ONLINE_SEEDS)
    forbidden = (set(JUNCTION_CROWD_HELDOUT_SEEDS) | set(JUNCTION_CROWD_SELECTION_DEV_SEEDS)
                 | set(JUNCTION_CROWD_PAPER_TEST_SEEDS))
    assert not trainable & forbidden
    for s in (JUNCTION_CROWD_IL_SEEDS[0], JUNCTION_CROWD_ONLINE_SEEDS[0],
              JUNCTION_CROWD_TRAIN_SEEDS[0], JUNCTION_CROWD_VALIDATION_SEEDS[0]):
        _assert_not_formal_seed(s)          # must NOT raise


@pytest.mark.parametrize("role", sorted(FORMAL_ONLY_ROLES))
def test_training_refuses_every_formal_only_role(role):
    with pytest.raises(IntentCLIError):
        _assert_not_formal_seed(JUNCTION_CROWD_SEED_ROLES[role][0])


def test_training_refuses_standard_selection_dev():
    with pytest.raises(IntentCLIError, match="SELECTION"):
        _assert_not_formal_seed(STANDARD_SELECTION_DEV_SEEDS[0])


@pytest.mark.parametrize("base", [PAPER_MAIN_BASE_SEED, TEST8_AUDIT_BASE_SEED])
def test_training_refuses_test8_derived_identities(base):
    """Test8 identities are DERIVED, so no inventory range can catch them."""
    for scenario, seed, _hd in paper_main_jobs(episodes_per_scenario=2, base_seed=base)[:6]:
        with pytest.raises(IntentCLIError, match="Test8"):
            _assert_not_formal_seed(seed)


def test_test8_base_seed_42_is_retired():
    assert PAPER_MAIN_BASE_SEED == 30_260_816
    assert TEST8_AUDIT_BASE_SEED == 40_260_817
    assert PAPER_MAIN_BASE_SEED != TEST8_AUDIT_BASE_SEED, (
        "the candidate audit must not run on formal episode identities")
    assert not set(paper_main_jobs(episodes_per_scenario=100, base_seed=PAPER_MAIN_BASE_SEED)) & \
        set(paper_main_jobs(episodes_per_scenario=100, base_seed=TEST8_AUDIT_BASE_SEED))


def test_no_production_module_still_defaults_to_base_seed_42():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for f in sorted(root.glob("*.py")):
        text = f.read_text()
        for line in text.splitlines():
            if "base_seed" in line and "= 42" in line.replace(" ", " "):
                offenders.append(f"{f.name}: {line.strip()}")
    assert not offenders, f"base_seed 42 is retired but still appears as a default: {offenders}"


# ------------------------------------------------ Test8 audit gate revision

def test_square_scene_never_double_filters_the_opposite_half_plane():
    """The exact opposite-half-plane rule and the generic forward cone were
    BOTH applied, so a legitimate opposite-side candidate could still be cut
    by the cone -- a crosser starting near a corner has real goals more than
    107 degrees off the entry-to-centroid direction."""
    from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn, square_scene
    for width in (10.0, 14.0):
        scene = square_scene(width=width, n_rows=4)
        fn = make_candidate_fn(scene)
        half = width / 2
        for x in np.linspace(-half + 0.05, half - 0.05, 11):
            if abs(x) < 1e-9:
                continue                    # entry exactly on the axis has no side
            for y in np.linspace(-half + 0.05, half - 0.05, 11):
                cands = fn(0, np.array([x, y]))
                assert len(cands) == 4, (
                    f"entry ({x:.2f}, {y:.2f}) kept {len(cands)} candidates, expected all 4 on the "
                    "opposite side")
                for c in cands:
                    assert np.sign(c.waypoints[-1][0]) == -np.sign(x), (
                        f"entry ({x:.2f}, {y:.2f}) was given a SAME-side candidate {c.waypoints[-1]}")


def test_square_filter_never_deletes_the_dictionary_best_candidate():
    """Assignment regret, on real episodes: the public filter must not remove
    the candidate that best describes a pedestrian."""
    from crowd_nav.bayesian_dvl.intent_train import build_formal_scenario_env, paper_main_episode_seed
    from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn, square_scene
    for scenario, width in (("baseline_square", 10.0), ("large_square", 14.0)):
        scene = square_scene(width=width, n_rows=4)
        fn = make_candidate_fn(scene)
        dictionary = [np.asarray(d.position) for d in scene.destinations]
        for i in range(6):
            sd = paper_main_episode_seed(scenario, i, TEST8_AUDIT_BASE_SEED)
            env, _robot, _s, _z = build_formal_scenario_env(ENV, scenario)
            env.case_counter["test"] = sd % (2 ** 32 - 1)
            env.reset()
            for h in env.humans:
                ends = [np.asarray(c.waypoints[-1]) for c in fn(0, np.array([h.px, h.py]))]
                assigned = min(float(np.hypot(e[0] - h.gx, e[1] - h.gy)) for e in ends)
                oracle = min(float(np.hypot(d[0] - h.gx, d[1] - h.gy)) for d in dictionary)
                assert assigned - oracle <= 1e-6, (
                    f"{scenario}: filter cost this pedestrian {assigned - oracle:.3f} m of "
                    f"representability (assigned {assigned:.3f}, oracle {oracle:.3f})")


def test_circle_and_junction_candidate_behaviour_is_unchanged_by_the_square_fix():
    from crowd_nav.bayesian_dvl.scene_candidates import circle_scene, make_candidate_fn
    from crowd_nav.bayesian_dvl.junction_scenario import public_junction_crowd_scene
    fn = make_candidate_fn(circle_scene(radius=4.0, n_sectors=8))
    for t in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        assert len(fn(0, np.array([4 * np.cos(t), 4 * np.sin(t)]))) == 8
    jf = make_candidate_fn(public_junction_crowd_scene(is_heldout=True))
    assert len(jf(0, np.array([0.0, 4.0]))) == 2          # corridor -> two exits
    assert len(jf(0, np.array([2.0, 4.0]))) == 4          # outside  -> crossing band


def test_analytic_oracle_bounds_come_from_geometry_not_measurement():
    from crowd_nav.bayesian_dvl.candidate_audit import analytic_oracle_bound
    # circle: half-chord between adjacent sectors + the entry noise CrowdSim adds
    assert analytic_oracle_bound("circle", 4.0, v_pref=1.0) == pytest.approx(
        2 * 4.0 * np.sin(np.pi / 16) + 1.0 / np.sqrt(2) + 1e-6)
    # square: (w/4) in x, (w/16) in y -> sqrt(5)/8 * w
    assert analytic_oracle_bound("square", 10.0) == pytest.approx(np.sqrt(5) / 8 * 10.0 + 1e-6)
    assert analytic_oracle_bound("square", 14.0) > analytic_oracle_bound("square", 10.0)
    with pytest.raises(Exception):
        analytic_oracle_bound("hexagon", 3.0)


def test_representability_mode_requires_an_analytic_bound():
    """Leaving the bound unset would turn the check into whatever the
    measurement happened to be."""
    from crowd_nav.bayesian_dvl.candidate_audit import CandidateAuditError, audit_scenario
    from crowd_nav.bayesian_dvl.scene_candidates import circle_scene
    with pytest.raises(CandidateAuditError, match="ANALYTIC"):
        audit_scenario(iter(()), circle_scene(radius=4.0, n_sectors=8), scenario="x",
                       coverage_mode="representability")


def test_retired_audit_base_seed_is_refused():
    from crowd_nav.bayesian_dvl.intent_train import TEST8_AUDIT_BASE_SEED_RETIRED
    assert TEST8_AUDIT_BASE_SEED == 40_260_817
    assert TEST8_AUDIT_BASE_SEED_RETIRED == 20_260_816
    for base in (PAPER_MAIN_BASE_SEED, TEST8_AUDIT_BASE_SEED, TEST8_AUDIT_BASE_SEED_RETIRED):
        for _sc, seed, _hd in paper_main_jobs(episodes_per_scenario=2, base_seed=base)[:4]:
            with pytest.raises(IntentCLIError, match="Test8"):
                _assert_not_formal_seed(seed)


def test_the_three_test8_bases_derive_disjoint_identities():
    from crowd_nav.bayesian_dvl.intent_train import TEST8_AUDIT_BASE_SEED_RETIRED
    sets = [{s for _sc, s, _hd in paper_main_jobs(episodes_per_scenario=500, base_seed=b)}
            for b in (PAPER_MAIN_BASE_SEED, TEST8_AUDIT_BASE_SEED, TEST8_AUDIT_BASE_SEED_RETIRED)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not sets[i] & sets[j]


def test_the_invalid_gate_diagnostic_record_exists_and_is_not_a_pass():
    import json
    root = Path(__file__).resolve().parents[3]
    rec = root / "runs" / "v2" / "test8_candidate_audit_20260816_INVALID_GATE_DIAGNOSTIC.json"
    assert rec.exists(), "the failed 20_260_816 audit must be kept, not deleted"
    payload = json.loads(rec.read_text())
    assert payload["status"] == "INVALID_GATE_DIAGNOSTIC"
    assert payload.get("passed") is None and payload["base_seed_retired"] is True


# --------------------------------------------- diagnostic optimizer seeds

def test_diagnostic_optimizer_seeds_are_a_separate_frozen_role():
    from crowd_nav.bayesian_dvl.intent_train import DIAGNOSTIC_OPTIMIZER_SEEDS
    from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
    assert DIAGNOSTIC_OPTIMIZER_SEEDS == (98211, 98212, 98213)
    formal = set(load_intent_training_config().training_seeds)
    assert not formal & set(DIAGNOSTIC_OPTIMIZER_SEEDS), (
        "the 2x2 decides how the update is assembled; its seeds must never overlap the formal ones")


def test_diagnostic_seeds_are_disjoint_from_every_episode_block():
    from crowd_nav.bayesian_dvl.intent_train import (
        DIAGNOSTIC_OPTIMIZER_SEEDS, FORMAL_EVAL_HELDOUT_SEEDS, STANDARD_DEV_DIAGNOSTIC_SEEDS,
        STANDARD_SELECTION_DEV_SEEDS)
    diag = set(DIAGNOSTIC_OPTIMIZER_SEEDS)
    for name, block in (("junction roles", {s for v in JUNCTION_CROWD_SEED_ROLES.values() for s in v}),
                        ("formal eval", set(FORMAL_EVAL_HELDOUT_SEEDS)),
                        ("standard diagnostic", set(STANDARD_DEV_DIAGNOSTIC_SEEDS)),
                        ("standard selection", set(STANDARD_SELECTION_DEV_SEEDS))):
        assert not diag & block, f"diagnostic optimizer seeds collide with {name}"


def test_formal_plan_refuses_a_diagnostic_seed():
    from crowd_nav.bayesian_dvl.intent_train import (
        DIAGNOSTIC_OPTIMIZER_SEEDS, IntentTrainError, build_formal_plan)
    with pytest.raises(IntentTrainError, match="DIAGNOSTIC"):
        build_formal_plan([DIAGNOSTIC_OPTIMIZER_SEEDS[0]], "code", "cfg", "corpus", 5000, 10000)
    # a legitimate plan still builds
    plan = build_formal_plan([98201, 98202], "code", "cfg", "corpus", 5000, 10000)
    assert {r["seed"] for r in plan["runs"]} == {98201, 98202}


def test_selection_and_paper_refuse_a_diagnostic_seed():
    from crowd_nav.bayesian_dvl.intent_train import DIAGNOSTIC_OPTIMIZER_SEEDS
    from crowd_nav.bayesian_dvl.intent_train_cli import assert_not_diagnostic_seed
    for seed in DIAGNOSTIC_OPTIMIZER_SEEDS:
        with pytest.raises(IntentCLIError, match="not eligible"):
            assert_not_diagnostic_seed(seed, "selection")
    assert_not_diagnostic_seed(98201, "selection")      # formal seed passes


def test_diagnostic_seeds_appear_in_the_inventory():
    from crowd_nav.bayesian_dvl.intent_train_cli import seed_inventory
    from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
    inv = seed_inventory(load_intent_training_config())
    assert "diagnostic_optimizer_seeds" in inv["blocks"]
    assert inv["ok"], f"seed roles must stay mutually exclusive: {inv['overlaps']}"
