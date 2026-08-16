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
    assert TEST8_AUDIT_BASE_SEED == 20_260_816
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
