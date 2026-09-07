"""Evaluation-protocol seed blocks and episode identities.

Split out of intent_train.py. Training has no business knowing which seeds
are reserved for the paper, for checkpoint selection or for a one-time
stress report -- it only needs its own training and validation seeds. Every
block below is read by evaluation and selection, and the ONE place that
proves they are mutually exclusive is preflight.
"""

from __future__ import annotations


class EvaluationProtocolError(ValueError):
    pass


from typing import Dict, List, Tuple

from crowd_nav.bayesian_dvl.intent_runtime_config import FORMAL_SCENARIOS as _FORMAL_SCENARIOS
from crowd_nav.bayesian_dvl.junction_scenario import (  # noqa: F401
    JUNCTION_CROWD_PAPER_TEST_SEEDS,
    JUNCTION_CROWD_SELECTION_DEV_SEEDS as JUNCTION_SELECTION_DEV_SEEDS,
    JUNCTION_CROWD_STAGE_ACCEPT_SEEDS,
)


# FROZEN, disjoint from JUNCTION_TRAIN_SEEDS (96001-96200) and
# JUNCTION_HELDOUT_SEEDS (96501-96600) -- same "9Xxxx block" convention.
# Independent held-out stress seeds: never used for training, IL
# collection, or checkpoint selection, only for this one-time formal report.
FORMAL_EVAL_HELDOUT_SEEDS: Tuple[int, ...] = tuple(range(97001, 97101))  # 100


# Order 1: DEVELOPMENT-ONLY standard-scenario diagnostic seeds.
#
# The `standard` (circle-crossing) scenario was the one real blind spot of
# the lambda=380 run: development validation sampled only 10 standard
# episodes per checkpoint, which cannot separate 0.85 from 0.95, and the
# only large-sample greedy evidence that existed was for junction_crowd.
# Every large-n number quoted for `standard` came from ONLINE episodes,
# which carry epsilon-greedy exploration and therefore cannot describe the
# greedy policy at all.
#
# FROZEN and disjoint from every other block (proved in seed_inventory and
# in the config's mutual-exclusion check). DEVELOPMENT ONLY: these seeds
# diagnose a run, they must never pick the paper's weights and must never
# be used for training -- _assert_not_formal_seed rejects them.
STANDARD_DEV_DIAGNOSTIC_SEEDS: Tuple[int, ...] = tuple(range(97401, 97501))  # 100

# Order 5: CHECKPOINT-SELECTION development seeds -- deliberately SEPARATE
# from the 97401-97500 diagnostic block.
#
# 97401-97500 has already been looked at (it is what diagnosed the
# lambda=380 run), so selecting weights on it would be selecting on data
# whose answers are known. These two blocks are reserved, unseen, and exist
# for exactly one job: scoring the four pre-registered milestone candidates
# under the frozen selection rule.
#
# Never used for training (rejected by _assert_not_formal_seed), never used
# for the paper's formal/Test8 numbers.
STANDARD_SELECTION_DEV_SEEDS: Tuple[int, ...] = tuple(range(2_900_000, 2_900_100))  # 100 (V2)

# Order 14 section 7: the STAGE-ACCEPTANCE block. 30 layouts per training
# scenario, evaluated closed-loop and greedily at a stage boundary (end of
# IL, end of DAgger) to decide whether the next stage may start at all.
#
# Why a new block rather than reusing an existing one: every block above is
# either already trained on or already reserved for choosing/reporting the
# paper's weights. This one exists to answer a single question -- can the
# policy complete an episode -- and a stage gate that shares seeds with the
# training distribution answers a different question than it claims to.
#
# junction_crowd here is the NOMINAL scene (is_heldout=False), the same
# geometry training uses; the shifted held-out variant is a different
# distribution and would confound a stage gate with a generalisation test.
#
# Never trained on: _assert_not_formal_seed rejects the whole block.
STAGE_ACCEPT_CIRCLE_SEEDS: Tuple[int, ...] = tuple(range(3_000_000, 3_000_030))          # 30
STAGE_ACCEPT_SQUARE_SEEDS: Tuple[int, ...] = tuple(range(3_010_000, 3_010_030))          # 30
# junction_crowd's stage block is owned by junction_scenario.py, which is also
# where the role resolver lives. Restating the range here is what broke the
# 90-episode gate: two sources of truth, only one of them consulted.
STAGE_ACCEPT_JUNCTION_CROWD_SEEDS: Tuple[int, ...] = JUNCTION_CROWD_STAGE_ACCEPT_SEEDS
STAGE_ACCEPT_SEEDS: Tuple[int, ...] = (
    STAGE_ACCEPT_CIRCLE_SEEDS + STAGE_ACCEPT_SQUARE_SEEDS + STAGE_ACCEPT_JUNCTION_CROWD_SEEDS)
STAGE_ACCEPT_PLAN = (
    (("circle",) * len(STAGE_ACCEPT_CIRCLE_SEEDS)) +
    (("square",) * len(STAGE_ACCEPT_SQUARE_SEEDS)) +
    (("junction_crowd",) * len(STAGE_ACCEPT_JUNCTION_CROWD_SEEDS)))
# V2: junction selection-dev now lives in junction_scenario.py beside the
# other junction blocks, so a block can never again be wired into the
# inventory without also being wired into the scenario's allowlist.
from crowd_nav.bayesian_dvl.junction_scenario import (  # noqa: E402
    JUNCTION_CROWD_SELECTION_DEV_SEEDS as JUNCTION_SELECTION_DEV_SEEDS,
    JUNCTION_CROWD_PAPER_TEST_SEEDS,
)

# C4RF.5: the PAPER-MAIN protocol. To be comparable episode-for-episode
# with Mamba-VL / SARL / LSTM (which are scored through test8.py) the
# goal-intent evaluator must use test8's OWN episode identities, not a
# private seed block. Formula and defaults are taken verbatim from
# crowd_nav/tools/evaluate_bdvl_paper_main.py, which already exists to
# make exactly this comparison bit-identical.
# V2: base seed 42 is RETIRED. The V6 candidate encoding changed what a
# circle/square episode's features MEAN (candidates now carry geometry and a
# count), so the V5 Test8 numbers do not describe this system and its episode
# identities have been seen. Any external method compared against these
# numbers -- Mamba-VL, SARL, LSTM -- must be re-run on this base seed too.
PAPER_MAIN_BASE_SEED = 30_260_816
# A separate, model-INDEPENDENT base used only by audit-test8-candidates, so
# the candidate audit never touches a formal episode identity.
#
# 20_260_816 is RETIRED. Its 600-episode run is kept as
# INVALID_GATE_DIAGNOSTIC: it was gated with the junction's absolute metre
# budget, which measures discretization rather than the model on a continuous
# goal space, so its FAIL is not a statement about the candidate model and
# must never be reinterpreted as a PASS.
TEST8_AUDIT_BASE_SEED_RETIRED = 20_260_816
TEST8_AUDIT_BASE_SEED = 40_260_817
# Order 12 section 7: the development acceptance base for the domain-
# randomised, invisible-robot chain. A NEW base is required because every
# earlier development block has been spent -- 2_900_000/2_400_000 decided the
# previous full arm, and the paper blocks must stay untouched for the paper.
# Reusing a seen block would mean judging a new policy on episodes an earlier
# decision already looked at.
DOMAIN_V3_DEV_BASE_SEED = 50_260_818
DOMAIN_V3_DEV_EPISODES_PER_SCENARIO = 100


def domain_v3_dev_episode_seed(scenario_name: str, episode_index: int,
                               base_seed: int = DOMAIN_V3_DEV_BASE_SEED) -> int:
    """Same identity formula as the paper protocol, a different base."""
    return paper_main_episode_seed(scenario_name, episode_index, base_seed)


def domain_v3_dev_jobs(episodes_per_scenario: int = DOMAIN_V3_DEV_EPISODES_PER_SCENARIO,
                       base_seed: int = DOMAIN_V3_DEV_BASE_SEED):
    return [(name, domain_v3_dev_episode_seed(name, ep, base_seed), False)
            for name in PAPER_MAIN_CASE_IDS for ep in range(episodes_per_scenario)]


PAPER_MAIN_EPISODES_PER_SCENARIO = 500
# case_id follows FORMAL_SIX_SCENARIOS' insertion order, which matches
# test8.py's hardcoded list: baseline_circle=0 ... large_square=5.
PAPER_MAIN_CASE_IDS: Dict[str, int] = {name: i for i, name in enumerate(_FORMAL_SCENARIOS)}


def paper_main_episode_seed(scenario_name: str, episode_index: int,
                            base_seed: int = PAPER_MAIN_BASE_SEED) -> int:
    """``(base_seed + case_id*1_000_003 + ep) % (2**31 - 1)`` -- bit-identical
    to test8.py / evaluate_bdvl_paper_main.py."""
    if scenario_name not in PAPER_MAIN_CASE_IDS:
        raise EvaluationProtocolError(f"unknown paper-main scenario {scenario_name!r}")
    if episode_index < 0:
        raise EvaluationProtocolError(f"episode_index must be >= 0, got {episode_index}")
    case_id = PAPER_MAIN_CASE_IDS[scenario_name]
    return (base_seed + case_id * 1_000_003 + episode_index) % (2**31 - 1)


def paper_main_jobs(episodes_per_scenario: int = PAPER_MAIN_EPISODES_PER_SCENARIO,
                    base_seed: int = PAPER_MAIN_BASE_SEED) -> List[Tuple[str, int, bool]]:
    """The frozen paper-main episode identity table."""
    if episodes_per_scenario <= 0:
        raise EvaluationProtocolError(f"episodes_per_scenario must be positive, got {episodes_per_scenario}")
    return [(name, paper_main_episode_seed(name, ep, base_seed), False)
            for name in PAPER_MAIN_CASE_IDS for ep in range(episodes_per_scenario)]
