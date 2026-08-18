"""Goal-intent main chain: IL(ORCA) + RL collection and training step
(consolidation plan Order 4 items 6-8). Self-contained -- no import of the
old trainer.py/replay.py/train_bdvl.py (those transitively pull in
belief.py). Reuses ONLY iqn.py's pure quantile-Huber loss (belief-free)
for the actual loss math.

Training target: plain Monte-Carlo return regression --
``G_t = sum_k gamma^k r_{t+k}``, IQN-regressed at the EXECUTED action's
quantiles. This is intentionally simpler than the old chain's R4-2
world-aggregated executed-action target (that formula existed to
aggregate multi-sample counterfactual rollouts this new chain does not
build) -- it is this chain's OWN, from-scratch training contract, not a
compatibility shim for the old one.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from crowd_nav.bayesian_dvl.intent_runtime_config import ActionGridSpec, FROZEN_VALUES, TRACKER_DEFAULTS
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.geometry_features import _robot_feature_vector, compute_action_features_array
from crowd_nav.bayesian_dvl.iqn import expert_ranking_loss, quantile_huber_loss
from crowd_nav.bayesian_dvl.ranking import (
    build_action_equivalence_class, derive_action_equivalence_tolerance, nearest_action_index,
)
from crowd_nav.bayesian_dvl.intent_policy import (
    HUMAN_FEATURE_DIM_V5, IntentPolicyError, build_intent_human_feature_batch, remaining_time_fraction,
    score_candidates_v5,
)
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank
from crowd_nav.bayesian_dvl.model import DistributionalValueModel
from crowd_nav.bayesian_dvl.scene_candidates import circle_scene, make_candidate_fn, square_scene
from crowd_nav.bayesian_dvl.junction_scenario import (
    junction_crowd_role_of_seed,
    AMBIGUOUS_TRACK_INDEX, JunctionCrowdEpisodeConfig, JunctionEpisodeConfig, build_junction_crowd_episode,
    build_junction_episode, maybe_reveal_crowd_exit, maybe_reveal_exit, public_junction_crowd_scene,
    public_junction_scene,
)
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY


# Structured abort types. Callers branch on THESE, never on the message.
ABORT_MC_REGRESSION = "mc_regression"
ABORT_RANKING_QUALITY = "ranking_quality"
ABORT_NON_FINITE = "non_finite"
ABORT_CLIPPING = "clipping"
ABORT_WARMUP_FAILURE = "warmup_failure"
ABORT_TYPES = (ABORT_MC_REGRESSION, ABORT_RANKING_QUALITY, ABORT_NON_FINITE,
               ABORT_CLIPPING, ABORT_WARMUP_FAILURE)

RANK_CAP_RHO = 0.5
# |g'_rank| at or below this fraction of max(|g_MC|, 1) is treated as zero.
RANK_GRADIENT_ZERO_TOL = 1e-8


from crowd_sim.envs.utils.robot import Robot


class IntentTrainError(ValueError):
    pass


def compute_mc_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    if not (0.0 < gamma <= 1.0):
        raise IntentTrainError(f"invalid gamma: {gamma}")
    if len(rewards) == 0:
        raise IntentTrainError("cannot compute MC returns from an empty episode")
    T = len(rewards)
    returns = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns


@dataclass
class IntentTransition:
    robot_features: np.ndarray
    human_features: np.ndarray
    human_mask: np.ndarray
    action_index: int
    action_features: np.ndarray
    # every candidate action's features at this decision (real bug found by
    # review: training previously only regressed MC-return on the EXECUTED
    # action, with no signal telling the network the executed action was
    # better than the other 79 -- this is what expert_ranking_loss needs).
    all_action_features: np.ndarray
    remaining_fraction: float
    # --- C0.1: sample role (plan section 3.1) ---
    # ``demo``   = an ORCA teacher decision -> gets L_MC + lambda*L_rank
    # ``online`` = the agent's OWN epsilon-greedy decision -> gets L_MC ONLY
    # Real, blocking objective-function bug found by audit: an earlier
    # version applied the SAME expert ranking loss to online samples, so an
    # epsilon-RANDOM action was being forcibly trained to outrank the other
    # 79 actions. The old chain already had the correct "demo has rank,
    # online is MC-only" semantics (stage2_train_step); the V5 chain had
    # not carried it over.
    source_role: str = "demo"
    # C0.2: ORCA's continuous action quantized to a TOLERANCE-WIDENED
    # equivalence class (ranking.build_action_equivalence_class), NOT the
    # single nearest action -- adjacent grid actions can be genuinely
    # equivalent to ORCA's real velocity, and a bare margin loss against
    # the single nearest one wrongly penalizes them. MUST be empty for
    # ``online`` samples.
    expert_action_indices: Tuple[int, ...] = ()
    reward: float = 0.0
    mc_return: Optional[float] = None

    def __post_init__(self) -> None:
        # all_action_features may be None ONLY transiently, while an online
        # row is being serialised (see IntentReplay._strip_for_persist).
        if self.source_role not in ("demo", "online"):
            raise IntentTrainError(f"source_role must be 'demo' or 'online', got {self.source_role!r}")
        if self.source_role == "online" and self.expert_action_indices:
            raise IntentTrainError(
                "online samples must have an EMPTY expert_action_indices -- the agent's own "
                "(possibly epsilon-random) action is never an expert demonstration"
            )
        if self.source_role == "demo" and not self.expert_action_indices:
            raise IntentTrainError("demo samples must carry a non-empty expert_action_indices equivalence set")


@dataclass
class EpisodeCollectionResult:
    transitions: List[IntentTransition]
    outcome: str  # success | collision | timeout
    episode_return: float = 0.0
    steps: int = 0
    navigation_time: float = 0.0
    path_length: float = 0.0
    path_ratio: float = 0.0
    min_clearance: float = float("inf")
    discomfort_frequency: float = 0.0


def _make_standard_env(env_config_path: Path, n_humans: int = 5):
    import configparser
    cfg = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not cfg.read(str(env_config_path)):
        raise IntentTrainError(f"env config not found: {env_config_path}")
    cfg.set("sim", "human_num", str(n_humans))
    cfg.set("robot", "policy", "orca")
    cfg.set("sim", "train_val_sim", "circle_crossing")
    cfg.set("sim", "test_sim", "circle_crossing")
    env = CrowdSim(); env.configure(cfg)
    env.phase = "train"
    robot = Robot(cfg, "robot")
    robot_orca = ORCA(); robot_orca.configure(cfg)
    # CrowdSim.reset() gates human_num behind robot.policy.multiagent_training
    # (defaults to None on a bare ORCA()) -- without this, reset() silently
    # forces human_num=1 regardless of the config, a real bug found by
    # measurement (AttentionPool received zero gradient; traced to every
    # "standard" episode having exactly 1 human despite human_num=5).
    robot_orca.multiagent_training = True
    robot.set_policy(robot_orca); robot.visible = True; robot.time_step = FROZEN_VALUES["dt"]
    robot.env = env
    env.set_robot(robot)
    return env, robot



SCENARIOS = ("standard", "junction", "junction_crowd")


class _ScenarioEpisode:
    """ONE place that knows how to build each scenario and how to advance
    its hidden state per step (C1.2: added ``junction_crowd`` -- three
    near-identical copies of this setup had been inlined into
    collect_orca_episode / run_ablation_episode / collect_online_episode,
    which is exactly how a fourth scenario silently gets wired into two of
    three call sites)."""

    def __init__(self, env_config_path: Path, scenario: str, episode_seed: int, is_heldout: bool = False):
        if scenario not in SCENARIOS:
            raise IntentTrainError(f"unknown scenario {scenario!r}, expected one of {SCENARIOS}")
        self.scenario = scenario
        self._true_exit = None
        self._waypoint_reached = False
        self._is_heldout = is_heldout
        if scenario == "standard":
            self.env, self.robot = _make_standard_env(env_config_path)
            self.env.case_counter["train"] = episode_seed % (2**32 - 1)
            self.env.reset()
            self.scene = circle_scene(radius=float(FROZEN_VALUES.get("circle_radius", 4.0)) or 4.0, n_sectors=8)
        elif scenario == "junction":
            cfg = JunctionEpisodeConfig(episode_seed=episode_seed, is_heldout=is_heldout)
            self.env, self.robot, self._true_exit = build_junction_episode(env_config_path, cfg)
            self.scene = public_junction_scene()
        else:  # junction_crowd
            # The SEED determines the role, and the role determines the
            # geometry -- a caller cannot mislabel an episode. is_heldout is
            # kept only as a cross-check: disagreeing with the seed's own
            # block is an error, not a reinterpretation.
            role = junction_crowd_role_of_seed(episode_seed)
            cfg = JunctionCrowdEpisodeConfig(episode_seed=episode_seed, role=role)
            if bool(is_heldout) != cfg.is_heldout:
                raise IntentTrainError(
                    f"seed {episode_seed} has role {role!r} (heldout geometry={cfg.is_heldout}), but the "
                    f"caller asked for is_heldout={is_heldout}")
            is_heldout = cfg.is_heldout
            self.env, self.robot, self._true_exit = build_junction_crowd_episode(env_config_path, cfg)
            self.scene = public_junction_crowd_scene(is_heldout=is_heldout)

    def advance_hidden_state(self) -> None:
        """MUST be called once per step BEFORE the environment steps (which
        triggers each pedestrian's ORCA act()). Reveals the ambiguous
        pedestrian's hidden exit only once it physically reaches the shared
        waypoint. Without it the pedestrian stalls at the junction forever."""
        if self.scenario == "junction":
            self._waypoint_reached = maybe_reveal_exit(
                self.env.humans[0], self._true_exit, self._waypoint_reached)
        elif self.scenario == "junction_crowd":
            self._waypoint_reached = maybe_reveal_crowd_exit(
                self.env.humans[AMBIGUOUS_TRACK_INDEX], self._true_exit, self._waypoint_reached,
                is_heldout=self._is_heldout)


def collect_orca_episode(
    env_config_path: Path, scenario: str, episode_seed: int, is_heldout: bool = False, gamma: float = 0.95,
    belief_mode: str = "full", n_samples: int = 60, horizon: int = 8,
) -> EpisodeCollectionResult:
    """scenario in {"standard", "junction"}. Both build a real CrowdSim
    episode driven by ORCA (IL demonstration), record V5 features for the
    'full' ablation mode at every step (the demonstration itself always
    uses the richest belief; mean/cv/uniform are compared at EVAL time,
    consolidation plan hard requirement 4), and compute MC returns."""
    episode = _ScenarioEpisode(env_config_path, scenario, episode_seed, is_heldout=is_heldout)
    env, robot, scene = episode.env, episode.robot, episode.scene

    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=TRACKER_DEFAULTS["speed_prior"])
    rng = np.random.default_rng(episode_seed)
    max_steps = int(round(FROZEN_VALUES["time_limit"] / FROZEN_VALUES["dt"])) + 1
    action_table = np.asarray(ActionGridSpec.from_env_config(str(env_config_path)).build_action_table(), dtype=np.float64)
    # derived ONCE from the frozen grid's own spacing (never tuned on
    # validation performance) -- see ranking.derive_action_equivalence_tolerance
    equivalence_tolerance = derive_action_equivalence_tolerance(action_table)

    transitions: List[IntentTransition] = []
    outcome = None
    for step in range(max_steps):
        episode.advance_hidden_state()
        humans = [
            HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
            for i, h in enumerate(env.humans)
        ]
        bank.update({h.track_id: (h.px, h.py) for h in humans})
        robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
        remaining = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])
        # C4R.6: the arm's belief mode must apply to the IL features too --
        # an independently trained `mean` arm has to be trained on `mean`
        # inputs, not on `full` inputs it will never see at deployment.
        human_feats, human_mask = build_intent_human_feature_batch(
            bank, robot_obs, humans, mode=belief_mode, rng=rng, horizon=horizon, n_samples=n_samples)
        robot_feats = _robot_feature_vector(robot_obs, remaining)

        orca_action = env.robot.act([h.get_observable_state() for h in env.humans])
        # ``executed_action_index``: the SINGLE nearest grid action, which is
        # what the environment actually executes and what L_MC regresses.
        # ``expert_action_indices``: the tolerance-widened EQUIVALENCE CLASS
        # (C0.2, plan section 3.1) that L_rank supervises -- deliberately
        # distinct, reusing ranking.py's belief-free, grid-spacing-derived
        # tolerance rather than a hand-picked number.
        executed_idx = nearest_action_index(orca_action.vx, orca_action.vy, action_table)
        expert_indices = build_action_equivalence_class(
            orca_action.vx, orca_action.vy, action_table, equivalence_tolerance,
        )
        all_action_feats_here = compute_action_features_array(robot_obs, action_table)
        executed_action_feat = all_action_feats_here[executed_idx]
        transitions.append(IntentTransition(
            robot_features=robot_feats, human_features=human_feats, human_mask=human_mask,
            action_index=executed_idx, action_features=executed_action_feat,
            all_action_features=all_action_feats_here, remaining_fraction=remaining,
            source_role="demo", expert_action_indices=expert_indices,
        ))
        # step with the DISCRETIZED action, not the raw continuous ORCA
        # velocity (real bug found by review: stepping with the continuous
        # action while labeling the transition with the nearest grid index
        # means the recorded reward/return does not strictly correspond to
        # the labeled action -- a small but real train/label mismatch that
        # would bias MC-return targets away from what the discrete-action
        # greedy policy could actually achieve).
        gvx, gvy = action_table[executed_idx]
        _, reward, terminated, truncated, info = env.step(ActionXY(float(gvx), float(gvy)))
        event = info.get("event")
        # use CrowdSim's OWN reward directly (real bug found by review: a
        # previously recomputed simplified reward silently dropped the
        # discomfort and stand-still penalties env.step() actually applies).
        transitions[-1].reward = float(reward)
        if terminated or truncated:
            outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}.get(event, "timeout")
            break
    if outcome is None:
        outcome = "timeout"

    returns = compute_mc_returns([t.reward for t in transitions], gamma)
    for t, g in zip(transitions, returns):
        t.mc_return = g
    return EpisodeCollectionResult(transitions=transitions, outcome=outcome)


@dataclass
class IntentBatch:
    """One training batch. Carries the per-sample ``demo_mask`` and each
    demo's ``expert_action_indices`` so ``train_step`` can apply the
    ranking loss to demo samples ONLY (plan section 3.1) -- a plain tuple
    of tensors could not express "this sample has no expert"."""
    robot_feats: torch.Tensor
    human_feats: torch.Tensor
    human_mask: torch.Tensor
    action_feats: torch.Tensor
    all_action_feats: torch.Tensor
    action_indices: torch.Tensor
    mc_returns: torch.Tensor
    demo_mask: torch.Tensor                       # [B] bool
    expert_indices: Tuple[Tuple[int, ...], ...]   # per sample; () for online


# --------------------------------------------------------------------- #
# A3: ONE arm-independent raw IL corpus.
#
# The belief features are the ONLY arm-dependent part of a demo
# transition, and they are a deterministic function of (observation
# sequence, mode, episode_seed). ORCA itself never consults the belief --
# it acts on the raw env state -- so an episode can be collected ONCE
# without any belief at all, and each arm can regenerate its own features
# on load. That replaces three near-identical corpora (~3.2 GB) with one.
# --------------------------------------------------------------------- #

@dataclass
class RawStep:
    robot: RobotObservation
    humans: List[HumanObservation]
    remaining_fraction: float
    action_index: int
    expert_action_indices: Tuple[int, ...]
    reward: float
    mc_return: Optional[float] = None


@dataclass
class RawEpisode:
    scenario: str
    episode_seed: int
    is_heldout: bool
    outcome: str
    steps: List[RawStep]


def materialize_arm_transitions(
    raw: RawEpisode, mode: str, action_table: np.ndarray,
    horizon: int = 8, n_samples: int = 60,
) -> List[IntentTransition]:
    """Regenerate one arm's demo transitions from a raw episode.

    Replays the belief bank over the SAME observation sequence with an rng
    seeded from the SAME episode_seed and consumed in the SAME order, so
    ``mode="full"`` reproduces ``collect_orca_episode`` exactly (asserted
    by test_c5_shared_raw_corpus_reproduces_per_arm_transitions)."""
    if mode not in ("full", "mean", "cv", "uniform"):
        raise IntentTrainError(f"unknown belief mode {mode!r}")
    scene = _scene_for_scenario(raw.scenario, raw.is_heldout)
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=TRACKER_DEFAULTS["speed_prior"])
    rng = np.random.default_rng(raw.episode_seed)
    out: List[IntentTransition] = []
    for st in raw.steps:
        bank.update({h.track_id: (h.px, h.py) for h in st.humans})
        human_feats, human_mask = build_intent_human_feature_batch(
            bank, st.robot, st.humans, mode=mode, rng=rng, horizon=horizon, n_samples=n_samples)
        all_action_feats = compute_action_features_array(st.robot, action_table)
        out.append(IntentTransition(
            robot_features=_robot_feature_vector(st.robot, st.remaining_fraction),
            human_features=human_feats, human_mask=human_mask,
            action_index=st.action_index, action_features=all_action_feats[st.action_index],
            all_action_features=all_action_feats, remaining_fraction=st.remaining_fraction,
            source_role="demo", expert_action_indices=tuple(st.expert_action_indices),
            reward=st.reward, mc_return=st.mc_return,
        ))
    return out


def _scene_for_scenario(scenario: str, is_heldout: bool):
    if scenario == "standard":
        return circle_scene(radius=float(FROZEN_VALUES.get("circle_radius", 4.0)) or 4.0, n_sectors=8)
    if scenario == "junction":
        return public_junction_scene()
    if scenario == "junction_crowd":
        return public_junction_crowd_scene(is_heldout=is_heldout)
    raise IntentTrainError(f"unknown scenario {scenario!r}")


def collect_raw_orca_episode(
    env_config_path: Path, scenario: str, episode_seed: int,
    is_heldout: bool = False, gamma: float = 0.95,
) -> RawEpisode:
    """Collect ONE ORCA demonstration WITHOUT computing any belief
    features -- arm-independent by construction."""
    episode = _ScenarioEpisode(env_config_path, scenario, episode_seed, is_heldout=is_heldout)
    env = episode.env
    max_steps = int(round(FROZEN_VALUES["time_limit"] / FROZEN_VALUES["dt"])) + 1
    action_table = np.asarray(
        ActionGridSpec.from_env_config(str(env_config_path)).build_action_table(), dtype=np.float64)
    tol = derive_action_equivalence_tolerance(action_table)

    steps: List[RawStep] = []
    outcome = None
    for _ in range(max_steps):
        episode.advance_hidden_state()
        humans = [
            HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
            for i, h in enumerate(env.humans)
        ]
        robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
        remaining = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])
        orca_action = env.robot.act([h.get_observable_state() for h in env.humans])
        idx = nearest_action_index(orca_action.vx, orca_action.vy, action_table)
        experts = build_action_equivalence_class(orca_action.vx, orca_action.vy, action_table, tol)
        gvx, gvy = action_table[idx]
        _, reward, terminated, truncated, info = env.step(ActionXY(float(gvx), float(gvy)))
        steps.append(RawStep(robot=robot_obs, humans=humans, remaining_fraction=remaining,
                              action_index=idx, expert_action_indices=experts, reward=float(reward)))
        if terminated or truncated:
            outcome = {"reach_goal": "success", "collision": "collision",
                       "timeout": "timeout"}.get(info.get("event"), "timeout")
            break
    if outcome is None:
        outcome = "timeout"
    for st, g in zip(steps, compute_mc_returns([s.reward for s in steps], gamma)):
        st.mc_return = g
    return RawEpisode(scenario=scenario, episode_seed=episode_seed, is_heldout=is_heldout,
                       outcome=outcome, steps=steps)


def batch_to_tensors(transitions: Sequence[IntentTransition], device: str = "cpu") -> IntentBatch:
    robot_feats = torch.as_tensor(np.stack([t.robot_features for t in transitions]), dtype=torch.float32, device=device)
    human_feats = torch.as_tensor(np.stack([t.human_features for t in transitions]), dtype=torch.float32, device=device)
    human_mask = torch.as_tensor(np.stack([t.human_mask for t in transitions]), dtype=torch.bool, device=device)
    action_feats = torch.as_tensor(np.stack([t.action_features for t in transitions]), dtype=torch.float32, device=device)
    all_action_feats = torch.as_tensor(np.stack([t.all_action_features for t in transitions]), dtype=torch.float32, device=device)
    action_indices = torch.as_tensor(np.array([t.action_index for t in transitions]), dtype=torch.long, device=device)
    returns = torch.as_tensor(np.array([t.mc_return for t in transitions]), dtype=torch.float32, device=device).unsqueeze(1)
    demo_mask = torch.as_tensor(
        np.array([t.source_role == "demo" for t in transitions]), dtype=torch.bool, device=device)
    expert_indices = tuple(tuple(t.expert_action_indices) for t in transitions)
    return IntentBatch(
        robot_feats=robot_feats, human_feats=human_feats, human_mask=human_mask, action_feats=action_feats,
        all_action_feats=all_action_feats, action_indices=action_indices, mc_returns=returns,
        demo_mask=demo_mask, expert_indices=expert_indices,
    )


@dataclass
class TrainStepResult:
    loss: float
    mc_loss: float
    rank_loss: float
    # C4R.3 stability diagnostics. ``grad_norm_preclip`` is measured BEFORE
    # clipping (after clipping it would be pinned at the clip value and
    # tell you nothing). The two per-term norms are computed with separate
    # autograd.grad calls -- the TOTAL gradient norm cannot stand in for
    # "the ratio between the two loss terms", which is the quantity the
    # frozen config's ratio gate is actually about.
    grad_norm_preclip: float = 0.0
    clipped: bool = False
    n_demo: int = 0
    n_online: int = 0
    mc_grad_norm: float = 0.0
    rank_grad_norm: float = 0.0
    weighted_rank_grad_norm: float = 0.0
    gradient_ratio: float = 0.0
    ratio_measured: bool = False
    # Order 2: the lambda this step ACTUALLY used (decided before the step
    # from past gradients), and the angle between the two objectives.
    lambda_used: float = 0.0
    gradient_cosine: float = 0.0
    # --- post-Adam diagnostics (populated only when diagnose=True) -----
    # The projection makes the RAW combined gradient non-opposing to g_MC.
    # It does NOT make Adam's realised parameter change do the same: Adam
    # rescales per-parameter by accumulated second moments, so the step
    # actually taken points somewhere else. Measured at the aborted V6
    # run's final checkpoint, cos(-delta, g_MC) was 0.03-0.06 with the
    # carried optimizer state versus 0.20-0.46 with a fresh one, while
    # dot(g_MC, delta) stayed NEGATIVE in every case -- so a sign test on
    # the dot reports "healthy" for an update that has nearly stopped
    # descending MC. Both are recorded: the cosine is the efficiency
    # metric, the dot only says which way it moved. Neither replaces the
    # fixed held-out audit.
    post_adam_dot_mc: float = 0.0
    post_adam_cos_mc: float = 0.0
    post_adam_cos_rank: float = 0.0
    active_hinge_fraction: float = 0.0
    active_hinge_count: int = 0
    projected_rank_norm: float = 0.0
    combined_grad_norm: float = 0.0
    adam_step: int = 0
    adam_exp_avg_norm: float = 0.0
    adam_exp_avg_sq_norm: float = 0.0
    module_diagnostics: dict = field(default_factory=dict)
    # Populated by ``run_online_training_step``. Keeping these on the
    # existing result preserves the public ``result.loss`` contract while
    # making navigation quality observable at every online episode.
    outcome: str = ""
    episode_return: float = 0.0
    episode_steps: int = 0
    navigation_time: float = 0.0
    path_length: float = 0.0
    path_ratio: float = 0.0
    min_clearance: float = float("inf")
    discomfort_frequency: float = 0.0
    scenario: str = ""
    episode_seed: int = -1
    epsilon: float = 0.0


def train_step(
    model: DistributionalValueModel, optimizer: torch.optim.Optimizer,
    batch: IntentBatch, generator: torch.Generator, n_taus: int = 16,
    ranking_margin: float = 0.1, ranking_batch_size: Optional[int] = None,
    grad_clip_norm: Optional[float] = None, measure_gradient_ratio: bool = False,
    rho: float = RANK_CAP_RHO, diagnose: bool = False,
) -> TrainStepResult:
    """ONE gradient step. Plan section 3.1's FROZEN objective:

        demo sample:   L = L_MC + lambda_rank * L_rank(expert equivalence set)
        online sample: L = L_MC
        mixed batch:   L_rank averaged over the DEMO MASK only; strictly 0
                       (and gradient-free) when the batch has no demo sample

    ``L_mc``: IQN quantile-Huber regression of the EXECUTED action's Q
    toward the real MC return -- applied to EVERY sample regardless of role.

    ``L_rank``: DQfD-style hardest-negative margin loss
    (``iqn.expert_ranking_loss``) supervising ORCA's expert equivalence
    CLASS (``batch.expert_indices[i]``, not the single executed action) to
    outrank the remaining grid actions. Applied ONLY to ``source_role ==
    "demo"`` samples.

    C0.1 fix, a real BLOCKING objective-function bug found by audit: an
    earlier version applied this ranking loss to every sample in the
    replay batch, including ONLINE ones -- so an epsilon-RANDOM action was
    being explicitly trained to outrank the other 79 actions. That is not
    a slow-learning problem, it is the wrong objective: it teaches the
    network that whatever the agent happened to do randomly was correct.
    The old SBK-HMM chain already had the right semantics
    (``stage2_train_step``: demo gets rank, online is MC-only); the V5
    chain had simply not carried it over.

    Scored with a FIXED tau grid (like ``score_candidates_v5``, not the
    ``generator``-drawn tau used for ``L_mc``) so this term consumes no
    extra randomness and does not disturb resume bit-identity.
    ``ranking_batch_size`` caps how many DEMO samples get the expensive
    80-action pass; default ``None`` covers all of them.

    ``generator`` is REQUIRED (real bug found by measurement, not assumed):
    an earlier version drew tau from the unseeded GLOBAL torch RNG. That
    made training trajectories depend on how many OTHER random draws had
    happened earlier in the process -- reconstructing a fresh model mid-
    script (even though its weights are immediately overwritten by
    ``load_intent_checkpoint``) consumes init-time random draws and shifts
    every subsequent tau draw, so a "resumed" run silently diverged from
    the "continuous" run it should have reproduced exactly. Same bug CLASS
    as R4-3's `_stable_seed`/per-track seeding fixes elsewhere in this
    project. Fix: an explicit, CHECKPOINTABLE ``torch.Generator`` (its
    ``.get_state()`` is saved/restored alongside the checkpoint) is now the
    only source of tau randomness for ``L_mc``."""
    model.train()
    robot_feats, human_feats, human_mask = batch.robot_feats, batch.human_feats, batch.human_mask
    B = robot_feats.shape[0]
    device = robot_feats.device
    # tau is drawn on the CPU with the checkpointed CPU generator and then
    # moved: a torch.Generator is device-bound, so drawing directly on CUDA
    # would need a CUDA generator whose state is NOT interchangeable with a
    # CPU run's. Drawing on CPU keeps ONE device-independent random stream,
    # so a run resumed on a different device follows the same tau sequence
    # (and CPU/CUDA parity is checkable at all -- Order C4).
    tau = torch.rand(B, n_taus, generator=generator).to(device)
    predicted = model(robot_feats, human_feats, human_mask, batch.action_feats, tau)  # [B, n_taus]
    target = batch.mc_returns.expand(B, 1)  # [B, 1] -- single MC sample as the target distribution
    mc_loss = quantile_huber_loss(predicted, tau, target).mean()

    if ranking_batch_size is not None and ranking_batch_size <= 0:
        raise IntentTrainError(f"ranking_batch_size must be positive, got {ranking_batch_size}")

    demo_positions = [i for i in range(B) if bool(batch.demo_mask[i])]
    if ranking_batch_size is not None:
        demo_positions = demo_positions[:ranking_batch_size]

    n_actions = batch.all_action_feats.shape[1]
    fixed_tau = (torch.arange(n_taus, dtype=torch.float32, device=device) + 0.5) / n_taus

    rank_losses = []
    if demo_positions:
        for i in demo_positions:
            experts = batch.expert_indices[i]
            if not experts:
                raise IntentTrainError(f"demo sample at batch position {i} has an empty expert set")
            if len(experts) >= n_actions:
                # expert_ranking_loss rejects an expert set covering
                # everything (no negatives to rank against). A grid-spacing-
                # derived tolerance should never produce this; fail loudly
                # rather than silently degrade the objective.
                raise IntentTrainError(
                    f"demo sample at batch position {i} has an expert set covering all {n_actions} actions")

        # C2.3: ONE vectorized [n_demo * n_actions] forward instead of a
        # Python loop of n_demo separate 80-row forwards. Critically, the
        # SET ENCODER runs only n_demo times, not n_demo * n_actions times:
        # the state embedding does not depend on the candidate action, so
        # the previous formulation recomputed the most expensive part of
        # the network 80x redundantly per sample. At the formal batch size
        # (256) that is the difference between 256 and 20480 encoder passes
        # per update.
        idx = torch.as_tensor(demo_positions, dtype=torch.long, device=device)
        n_demo = len(demo_positions)
        state_emb = model.encode(robot_feats[idx], human_feats[idx], human_mask[idx])   # [n_demo, E]
        state_rep = state_emb.repeat_interleave(n_actions, dim=0)                        # [n_demo*A, E]
        action_rep = batch.all_action_feats[idx].reshape(n_demo * n_actions, -1)         # [n_demo*A, Fa]
        action_emb = model.action_encoder(action_rep)
        tau_rep = fixed_tau.unsqueeze(0).expand(n_demo * n_actions, n_taus)
        scores = model.value_network(state_rep, action_emb, tau_rep).mean(dim=1)         # [n_demo*A]
        scores = scores.view(n_demo, n_actions)
        for row, i in enumerate(demo_positions):
            rank_losses.append(expert_ranking_loss(scores[row], batch.expert_indices[i], ranking_margin))

    # How many of the scored rows still VIOLATE the margin. A hinge that is
    # satisfied contributes exactly zero gradient, so this is what says
    # whether the ranking term still has anything to teach -- as opposed to
    # its gradient norm, which the share renormalisation fixes by
    # construction and which therefore cannot answer the question.
    active_hinges = int(sum(1 for l in rank_losses if float(l.detach()) > 0.0))
    active_fraction = active_hinges / max(len(rank_losses), 1)

    if rank_losses:
        rank_loss = torch.stack(rank_losses).mean()
    else:
        # online-only batch: strictly zero AND gradient-free (not a
        # detached-but-nonzero constant, and not a tensor that would add a
        # spurious node to the graph).
        rank_loss = torch.zeros((), device=device)

    # There is no longer a single weighted scalar objective: the update is
    # assembled from gradients, not from a summed loss. `loss` is reported
    # as the UNWEIGHTED sum purely so the two terms remain comparable across
    # runs; nothing optimises it directly.
    loss = mc_loss.detach() + rank_loss.detach()

    # Order 2R: the update is built from PROJECTED, norm-normalised
    # gradients, not from a weighted sum of the two losses.
    #
    # What this replaces and why: a scalar lambda can only decide which term
    # is louder. It cannot stop the ranking supervision from pulling against
    # value regression, and an EMA-tracked lambda could not keep up with how
    # fast |g_MC| collapses during IL -- the pilot aborted on exactly that,
    # with lambda overshooting to ~102 while the fixed audit set showed MC
    # still improving. Here the conflicting component is removed outright
    # and the ranking size is set by construction every step.
    params = [p for p in model.parameters() if p.requires_grad]
    mc_grads = torch.autograd.grad(mc_loss, params, retain_graph=True, allow_unused=True)
    rank_grads = None
    if rank_loss.requires_grad:
        rank_grads = torch.autograd.grad(rank_loss, params, retain_graph=True, allow_unused=True)

    if rank_grads is None or rho <= 0.0:
        # online-only batch (or ranking disabled): pure MC, no projection
        combined = list(mc_grads)
        info = {"mc_grad_norm": _grad_l2(mc_grads), "rank_grad_norm": 0.0,
                "projected_rank_norm": 0.0, "gradient_cosine": 0.0,
                "projection_coefficient": 0.0, "rank_scale": 0.0,
                "conflict_removed": False, "rank_negligible": True}
    else:
        combined, info = combine_gradients(mc_grads, rank_grads, rho=rho)

    optimizer.zero_grad()
    for p_, g_ in zip(params, combined):
        p_.grad = None if g_ is None else g_.detach().clone()
    grad_norm = float(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5)
    if not np.isfinite(grad_norm):
        raise IntentTrainError(f"non-finite gradient norm {grad_norm}")
    clipped = False
    if grad_clip_norm is not None:
        if grad_clip_norm <= 0:
            raise IntentTrainError(f"grad_clip_norm must be positive, got {grad_clip_norm}")
        torch.nn.utils.clip_grad_norm_(params, float(grad_clip_norm))
        clipped = grad_norm > float(grad_clip_norm)
    mc_grad_norm = info["mc_grad_norm"]
    rank_grad_norm = info["rank_grad_norm"]
    weighted = info["rank_scale"] * info["projected_rank_norm"]
    ratio = weighted / max(mc_grad_norm, 1e-12)
    cosine = info["gradient_cosine"]

    # ---- post-Adam diagnostics ---------------------------------------
    # Snapshot BEFORE the step so the realised parameter change can be
    # compared against the gradient that was supposed to drive it.
    diag = {}
    pre = [p_.detach().clone() for p_ in params] if diagnose else None
    optimizer.step()
    if diagnose:
        delta = [p_.detach() - b for p_, b in zip(params, pre)]
        d_norm = sum(float((d ** 2).sum()) for d in delta) ** 0.5
        dot_mc = sum(float((g * d).sum()) for g, d in zip(mc_grads, delta) if g is not None)
        cos_mc = -dot_mc / max(d_norm * mc_grad_norm, 1e-12)
        cos_rank = 0.0
        if rank_grads is not None and rank_grad_norm > 0:
            dot_rank = sum(float((g * d).sum()) for g, d in zip(rank_grads, delta) if g is not None)
            cos_rank = -dot_rank / max(d_norm * rank_grad_norm, 1e-12)
        a_step, a_m, a_v = adam_state_norms(optimizer)
        per_module = {}
        mc_by = _grouped_norms(model, mc_grads)
        rk_by = _grouped_norms(model, rank_grads) if rank_grads is not None else {}
        dl_by = _grouped_norms(model, delta)
        dot_by = _grouped_dot(model, mc_grads, delta)
        for g in MODULE_GROUPS:
            per_module[g] = {
                "mc_grad_norm": mc_by.get(g, 0.0),
                "rank_grad_norm": rk_by.get(g, 0.0),
                "delta_norm": dl_by.get(g, 0.0),
                "dot_mc": dot_by.get(g, 0.0),
                "cos_mc": -dot_by.get(g, 0.0) / max(dl_by.get(g, 0.0) * mc_by.get(g, 0.0), 1e-12),
            }
        diag = {"post_adam_dot_mc": dot_mc, "post_adam_cos_mc": cos_mc,
                "post_adam_cos_rank": cos_rank, "adam_step": a_step,
                "adam_exp_avg_norm": a_m, "adam_exp_avg_sq_norm": a_v,
                "module_diagnostics": per_module}
    return TrainStepResult(
        loss=float(loss.item()), mc_loss=float(mc_loss.item()), rank_loss=float(rank_loss.item()),
        grad_norm_preclip=grad_norm, clipped=clipped,
        projected_rank_norm=float(info["projected_rank_norm"]),
        combined_grad_norm=float(grad_norm),
        active_hinge_count=int(active_hinges), active_hinge_fraction=float(active_fraction),
        **diag,
        n_demo=int(batch.demo_mask.sum()), n_online=int((~batch.demo_mask).sum()),
        mc_grad_norm=mc_grad_norm, rank_grad_norm=rank_grad_norm,
        weighted_rank_grad_norm=weighted, gradient_ratio=ratio, ratio_measured=measure_gradient_ratio,
        lambda_used=float(info["rank_scale"]), gradient_cosine=cosine,
    )


def _grad_l2(grads) -> float:
    return float(sum((g ** 2).sum() for g in grads if g is not None) ** 0.5)


def _grad_cosine(a, b) -> float:
    """cos(g_MC, g_rank). Norm ratio alone says which term is LOUDER; the
    angle says whether they pull together, sideways, or against each other.
    Reported only -- never used to tune anything (Order 2).

    Each norm is taken over the FULL parameter vector, treating an unused
    parameter as a zero gradient. A parameter that only ONE loss touches
    (``allow_unused=True`` gives the other loss ``None`` there) still
    contributes to its own loss's norm -- skipping the whole parameter, as
    an earlier version did, shrank both denominators to the shared support
    and inflated |cos|. With the ranking loss flowing only through demo
    positions that covers many parameters, so the error was systematic and
    biased toward spurious OBJECTIVE_CONFLICT aborts. Verified: for
    g_MC=(3,4), g_rank=(1,None) the correct cosine is 3/(5*1)=0.6; the old
    form returned 1.0."""
    dot = 0.0
    na = nb = 0.0
    for ga, gb in zip(a, b):
        if ga is not None:
            na += float((ga ** 2).sum())
        if gb is not None:
            nb += float((gb ** 2).sum())
        if ga is not None and gb is not None:
            dot += float((ga * gb).sum())
    denom = (na ** 0.5) * (nb ** 0.5)
    return float(dot / denom) if denom > 1e-30 else 0.0


class TrainingHealthMonitor:
    """Order 3R: health is judged on MEASURED EFFECT, not on gradient ratios.

    The ratio gate is gone entirely. The ranking contribution is now bounded
    above by ``rho`` * |g_MC| by construction, so a gate on that number could
    only ever fire when the projection degenerates -- it cannot detect the
    thing that actually went wrong. The previous incarnation of this class
    learned that the expensive way: it aborted a pilot for exceeding a
    gradient budget while the fixed audit set showed value regression
    improving the whole time. And the reverse also holds: at the old fixed
    2x share the ratio sat exactly where it was told to, while the held-out
    MC degraded on 3/3 seeds. Only the measured effect catches that.

    What is checked instead, every ``interval`` updates, on the FIXED audit
    set (stationary rows, fixed midpoint quantiles, no RNG):

      * audit_mc  > best_mc * mc_regression_factor      -> abort
      * audit_rank > rank_max, or top1 < top1_min,
        on ``consecutive_bad`` CONSECUTIVE checks        -> abort
      * clipping on > clip_fraction of the last
        ``clip_window`` updates                          -> abort
      * non-finite anything                              -> abort immediately

    One bad check does not abort: the audit is deterministic but the weights
    it scores are mid-optimisation. Two in a row is a trend.

    Raw gradient norms and the cosine are still recorded, as diagnostics
    only -- they no longer gate anything.
    """

    def __init__(self, interval: int = 100, mc_regression_factor: float = 1.5,
                 rank_max: float = 0.075, ranking_grace_il_passes: int = 500,
                 consecutive_bad: int = 2, clip_window: int = 500, clip_fraction: float = 0.80):
        if interval <= 0 or clip_window <= 0:
            raise IntentTrainError("interval and clip_window must be positive")
        if mc_regression_factor <= 1.0:
            raise IntentTrainError(f"mc_regression_factor must exceed 1, got {mc_regression_factor}")
        self.interval = int(interval)
        self.mc_regression_factor = float(mc_regression_factor)
        self.rank_max = float(rank_max)
        self.ranking_grace_il_passes = int(ranking_grace_il_passes)
        self.consecutive_bad = int(consecutive_bad)
        self.clip_window, self.clip_fraction = int(clip_window), float(clip_fraction)
        self.best_mc: Optional[float] = None
        self.consecutive_rank_bad = 0
        self.n_checks = 0
        self.clips: List[bool] = []
        self.diagnostics: List[dict] = []

    _FROZEN = ("interval", "mc_regression_factor", "rank_max", "ranking_grace_il_passes",
               "consecutive_bad", "clip_window", "clip_fraction")

    def observe_update(self, clipped: bool) -> Optional[str]:
        self.clips.append(bool(clipped))
        if len(self.clips) > self.clip_window:
            del self.clips[:-self.clip_window]
        if len(self.clips) >= self.clip_window:
            frac = sum(self.clips) / len(self.clips)
            if frac > self.clip_fraction:
                return (f"gradient clipping fired on {frac:.1%} of the last {self.clip_window} updates "
                        f"(> {self.clip_fraction:.0%})")
        return None

    def observe_audit(self, audit_mc: float, audit_rank: float, top1: float,
                      check_ranking: bool = True, il_pass: Optional[int] = None,
                      disable_ranking_gate: bool = False) -> Optional[Tuple[str, str]]:
        """``None`` when healthy, else ``(reason_type, message)``.

        The TYPE is what callers branch on. Three different failures used to
        arrive as three English sentences behind one shared prefix, so the
        only way to tell "the experiment answered" from "the run broke" was
        to match prose -- which is exactly what the 2x2 runner had to do, and
        exactly what a reworded message would silently break.
        """
        for name, v in (("audit_mc", audit_mc), ("audit_rank", audit_rank), ("top1", top1)):
            if not np.isfinite(v):
                return (ABORT_NON_FINITE, f"non-finite {name} = {v}")
        self.n_checks += 1
        if self.best_mc is None or audit_mc < self.best_mc:
            self.best_mc = float(audit_mc)
        if self.best_mc > 0 and audit_mc > self.best_mc * self.mc_regression_factor:
            return (ABORT_MC_REGRESSION,
                    f"value regression on the FIXED audit set reached {audit_mc / self.best_mc:.2f}x its own "
                    f"best ({audit_mc:.6f} vs {self.best_mc:.6f}), limit {self.mc_regression_factor}x")
        # The ranking-quality gate presupposes a PASSED warm-up: it exists to
        # catch joint training eroding a ranking structure that was actually
        # built. With the warm-up skipped (pilot only) there is nothing to
        # protect, and firing here would just be reporting that fact.
        if not check_ranking:
            self.consecutive_rank_bad = 0
            return None
        # PHASE-TRANSITION GRACE. When MC joins a freshly warmed-up ranker the
        # ranking metrics take a transient step backwards and then recover.
        # Measured on the same code and corpus:
        #   pilot   pass 0/100/200/300 -> 0.0287 / 0.0848 / 0.0732 / 0.0656
        #   formal  pass 0/100/200     -> 0.0292 / 0.0881 / 0.0775  -> ABORT
        # Both are the same transient; only the pilot's pass-200 sample
        # happened to land 0.0018 inside the ceiling and reset the streak,
        # while the formal run's landed 0.0025 outside it. A gate whose
        # verdict turns on 0.004 of a known transient is a false positive,
        # not a measurement. The pilot's own trajectory shows where it goes:
        # 0.0656 -> ... -> 0.0193 by pass 2000, far under the ceiling.
        #
        # So the RANKING gate is suspended for the first
        # `ranking_grace_il_passes` IL passes, with the streak starting from
        # zero afterwards. Everything else -- non-finite values and MC
        # regression -- is enforced from pass 0.
        if il_pass is not None and il_pass < self.ranking_grace_il_passes:
            self.consecutive_rank_bad = 0
            return None
        # top-1 is telemetry, not a gate: its threshold shares the leaked
        # provenance of the warm-up one. rank_loss IS the ranking objective.
        bad = audit_rank > self.rank_max
        self.consecutive_rank_bad = self.consecutive_rank_bad + 1 if bad else 0
        if self.consecutive_rank_bad >= self.consecutive_bad:
            if disable_ranking_gate:
                # DIAGNOSTIC ONLY. The 2x2's independent variable IS ranking
                # pressure, and two of its four cells train with no ranking
                # gradient at all -- so gating them on ranking quality would
                # decide the question in advance rather than measure it. The
                # metric is still computed and recorded; only the verdict is
                # suspended. Every other gate stays live.
                return None
            return (ABORT_RANKING_QUALITY,
                    f"ranking quality failed {self.consecutive_rank_bad} consecutive audits "
                    f"(rank={audit_rank:.5f} > {self.rank_max}; top1={top1:.3f} reported only)")
        return None

    def record_diagnostic(self, **kw) -> None:
        """Gradient norms / cosine / rank_scale: kept for the report, never gates."""
        self.diagnostics.append(dict(kw))
        if len(self.diagnostics) > 500:
            del self.diagnostics[:-500]

    def state_dict(self) -> dict:
        state = {"best_mc": self.best_mc, "consecutive_rank_bad": self.consecutive_rank_bad,
                 "n_checks": self.n_checks, "clips": [bool(x) for x in self.clips],
                 "diagnostics": list(self.diagnostics)}
        state.update({k: getattr(self, k) for k in self._FROZEN})
        return state

    def load_state_dict(self, state: dict) -> None:
        missing = [k for k in self._FROZEN if k not in state]
        if missing:
            raise IntentTrainError(f"health monitor state is missing frozen thresholds {missing}")
        for k in self._FROZEN:
            if float(state[k]) != float(getattr(self, k)):
                raise IntentTrainError(
                    f"health monitor config drift on {k}: checkpoint {state[k]} != {getattr(self, k)}")
        self.best_mc = None if state["best_mc"] is None else float(state["best_mc"])
        self.consecutive_rank_bad = int(state["consecutive_rank_bad"])
        self.n_checks = int(state["n_checks"])
        self.clips = [bool(x) for x in state["clips"]]
        self.diagnostics = list(state["diagnostics"])


class GradientRatioMonitor:
    """C4R.3: the frozen config's sustained-window ratio gate, made real.

    A SINGLE out-of-range batch is normal (the MC gradient can be near
    convergence), so an out-of-range ratio is a diagnostic EVENT and only
    a sustained run of them aborts -- the same convention the older chain
    already used. Kept as a separate object so its counter can be
    checkpointed with the rest of the run state."""

    def __init__(self, ratio_min: float, ratio_max: float, sustained_updates: int):
        if not (0 < ratio_min < ratio_max):
            raise IntentTrainError(f"require 0 < ratio_min < ratio_max, got {ratio_min}/{ratio_max}")
        if sustained_updates <= 0:
            raise IntentTrainError(f"sustained_updates must be positive, got {sustained_updates}")
        self.ratio_min, self.ratio_max = float(ratio_min), float(ratio_max)
        self.sustained_updates = int(sustained_updates)
        self.consecutive_out_of_range = 0
        self.n_measured = 0
        self.n_out_of_range = 0

    def observe(self, ratio: float) -> Optional[str]:
        """Returns an abort REASON once the window is exhausted, else None."""
        self.n_measured += 1
        if self.ratio_min <= ratio <= self.ratio_max:
            self.consecutive_out_of_range = 0
            return None
        self.n_out_of_range += 1
        self.consecutive_out_of_range += 1
        if self.consecutive_out_of_range >= self.sustained_updates:
            return (f"weighted-rank/MC gradient ratio stayed outside "
                    f"[{self.ratio_min}, {self.ratio_max}] for {self.consecutive_out_of_range} "
                    f"consecutive measured updates (last ratio {ratio:.4g})")
        return None

    def state_dict(self) -> dict:
        return {"consecutive_out_of_range": self.consecutive_out_of_range,
                "n_measured": self.n_measured, "n_out_of_range": self.n_out_of_range}

    def load_state_dict(self, state: dict) -> None:
        self.consecutive_out_of_range = int(state["consecutive_out_of_range"])
        self.n_measured = int(state["n_measured"])
        self.n_out_of_range = int(state["n_out_of_range"])


@dataclass
class AblationEpisodeResult:
    mode: str
    episode_seed: int
    outcome: str
    steps: int


def run_ablation_episode(
    env_config_path: Path, model: DistributionalValueModel, action_table: np.ndarray,
    scenario: str, episode_seed: int, mode: str, planner_seed: int, is_heldout: bool = False,
    n_samples: int = 60, horizon: int = 8, device: str = "cpu",
) -> AblationEpisodeResult:
    """ONE episode, driven by the model's OWN greedy action choice under
    the given belief ``mode``. Consolidation plan hard requirement 4: to
    compare full/mean/cv/uniform fairly, call this with the SAME
    (scenario, episode_seed, planner_seed, model) for every mode -- the
    environment trajectory and the model are then identical; only the
    belief representation fed to the network differs."""
    episode = _ScenarioEpisode(env_config_path, scenario, episode_seed, is_heldout=is_heldout)
    env, robot, scene = episode.env, episode.robot, episode.scene

    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=TRACKER_DEFAULTS["speed_prior"])
    planner_rng = np.random.default_rng(planner_seed)
    max_steps = int(round(FROZEN_VALUES["time_limit"] / FROZEN_VALUES["dt"])) + 1
    outcome = None
    step = 0
    for step in range(max_steps):
        episode.advance_hidden_state()
        humans = [
            HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
            for i, h in enumerate(env.humans)
        ]
        bank.update({h.track_id: (h.px, h.py) for h in humans})
        robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
        remaining = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])
        human_feats, human_mask = build_intent_human_feature_batch(
            bank, robot_obs, humans, mode=mode, rng=planner_rng, horizon=horizon, n_samples=n_samples,
        )
        results = score_candidates_v5(
            model, robot_obs, human_feats, human_mask, action_table, remaining, device=device)
        best = max(results, key=lambda r: r.q_mean)
        avx, avy = action_table[best.action_index]
        _, _reward, terminated, truncated, info = env.step(ActionXY(float(avx), float(avy)))
        if terminated or truncated:
            event = info.get("event")
            outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}.get(event, "timeout")
            break
    if outcome is None:
        outcome = "timeout"
    return AblationEpisodeResult(mode=mode, episode_seed=episode_seed, outcome=outcome, steps=step + 1)


def run_ablation_suite(
    env_config_path: Path, model: DistributionalValueModel, action_table: np.ndarray,
    scenario: str, episode_seeds: Sequence[int], is_heldout: bool = False, n_samples: int = 60,
    horizon: int = 8, device: str = "cpu",
) -> Dict[str, List[AblationEpisodeResult]]:
    """full/mean/cv/uniform, each run on the IDENTICAL set of
    (episode_seed, planner_seed) pairs -- same env trajectory, same
    checkpoint, only the belief mode differs (hard requirement 4).

    The paper-relevant comparison is full vs. mean vs. cv (these three are
    all PLAUSIBLE belief representations a real system might use, so
    differences among them are informative about the value of the full
    posterior specifically). ``uniform`` (belief replaced by a flat 1/n
    vector, i.e. "candidates exist but nothing is known") is an EXTRA,
    secondary sanity control -- it answers "does having ANY structured
    belief help at all," not "does the FULL posterior help beyond a
    cheaper summary." Per review: don't treat uniform as a fourth
    co-equal, required-isolated arm in the main story; report it as a
    supplementary check.
    """
    results: Dict[str, List[AblationEpisodeResult]] = {m: [] for m in ("full", "mean", "cv", "uniform")}
    for episode_seed in episode_seeds:
        planner_seed = 5_000_000 + episode_seed  # SAME for all 4 modes at this episode_seed
        for mode in results:
            results[mode].append(run_ablation_episode(
                env_config_path, model, action_table, scenario, episode_seed, mode, planner_seed,
                is_heldout=is_heldout, n_samples=n_samples, horizon=horizon, device=device,
            ))
    return results


# --------------------------------------------------------------------- #
# Online RL: policy-driven exploration + replay buffer (real bug found by
# review: the chain previously had only ORCA IL collection -- no online
# collection under the model's OWN policy, no exploration, no replay
# buffer, no resumable online-loop state).
# --------------------------------------------------------------------- #

def collect_online_episode(
    env_config_path: Path, model: DistributionalValueModel, action_table: np.ndarray,
    scenario: str, episode_seed: int, epsilon: float, explore_rng: np.random.Generator,
    is_heldout: bool = False, gamma: float = 0.95, n_samples: int = 60, horizon: int = 8,
    device: str = "cpu", belief_mode: str = "full",
) -> EpisodeCollectionResult:
    """Policy-driven online collection: epsilon-greedy over the model's OWN
    full-belief scoring (``score_candidates_v5``) -- with probability
    ``epsilon`` act uniformly at random over the 80-action grid, otherwise
    take the model's greedy action. Produces ``IntentTransition`` objects
    with the SAME schema as ``collect_orca_episode`` (real env reward, full
    ``all_action_features``) so online and IL transitions can share one
    replay buffer and one ``train_step``. ``explore_rng`` is the ONLY
    source of exploration randomness -- deliberately separate from the
    per-step belief-sampling rng (seeded from ``episode_seed``, matching
    ``collect_orca_episode``/``run_ablation_episode``) so resuming the
    online loop only requires persisting ``explore_rng``'s state, not
    reconstructing the whole episode's belief-sampling history."""
    if not (0.0 <= epsilon <= 1.0):
        raise IntentTrainError(f"epsilon must be in [0, 1], got {epsilon}")
    episode = _ScenarioEpisode(env_config_path, scenario, episode_seed, is_heldout=is_heldout)
    env, robot, scene = episode.env, episode.robot, episode.scene

    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=TRACKER_DEFAULTS["speed_prior"])
    belief_rng = np.random.default_rng(episode_seed)
    max_steps = int(round(FROZEN_VALUES["time_limit"] / FROZEN_VALUES["dt"])) + 1

    transitions: List[IntentTransition] = []
    start = np.asarray([robot.px, robot.py], dtype=np.float64)
    goal = np.asarray([robot.gx, robot.gy], dtype=np.float64)
    initial_goal_distance = float(np.linalg.norm(goal - start))
    previous_position = start.copy()
    path_length = 0.0
    min_clearance = float("inf")
    discomfort_dist = float(FROZEN_VALUES.get("discomfort_distance", 0.2) or 0.2)
    discomfort_steps = 0
    outcome = None
    for step in range(max_steps):
        episode.advance_hidden_state()
        humans = [
            HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
            for i, h in enumerate(env.humans)
        ]
        bank.update({h.track_id: (h.px, h.py) for h in humans})
        robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
        remaining = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])
        human_feats, human_mask = build_intent_human_feature_batch(
            bank, robot_obs, humans, mode=belief_mode, rng=belief_rng, horizon=horizon, n_samples=n_samples,
        )
        robot_feats = _robot_feature_vector(robot_obs, remaining)

        if explore_rng.random() < epsilon:
            executed_idx = int(explore_rng.integers(len(action_table)))
        else:
            results = score_candidates_v5(
                model, robot_obs, human_feats, human_mask, action_table, remaining, device=device)
            executed_idx = max(results, key=lambda r: r.q_mean).action_index

        all_action_feats_here = compute_action_features_array(robot_obs, action_table)
        executed_action_feat = all_action_feats_here[executed_idx]
        # C0.1/C0.2: the agent's OWN action -- epsilon-random or its own
        # greedy choice -- is NEVER an expert demonstration. source_role
        # "online" + an empty expert set is what makes train_step exclude
        # this sample from the ranking loss entirely.
        transitions.append(IntentTransition(
            robot_features=robot_feats, human_features=human_feats, human_mask=human_mask,
            action_index=executed_idx, action_features=executed_action_feat,
            all_action_features=all_action_feats_here, remaining_fraction=remaining,
            source_role="online", expert_action_indices=(),
        ))
        gvx, gvy = action_table[executed_idx]
        _, reward, terminated, truncated, info = env.step(ActionXY(float(gvx), float(gvy)))
        current_position = np.asarray([robot.px, robot.py], dtype=np.float64)
        path_length += float(np.linalg.norm(current_position - previous_position))
        previous_position = current_position

        # Order 1R (extended to the training side): clearance is the
        # simulator's SWEPT ``dmin`` for the interval this action covered --
        # the SAME definition intent_evaluate.py uses, so online telemetry,
        # development validation and formal evaluation are finally
        # comparable numbers.
        #
        # What this replaces: the post-step END-POINT distance between robot
        # and human centres. That misses the closest approach WITHIN the
        # interval (brush past, then separate, and it is never seen), and on
        # a collision it reports how deep the overlap happened to be at the
        # end of the step rather than the true minimum. It also silently
        # disagreed with the evaluator, which measured a PRE-action snapshot
        # -- so "online clearance" and "eval clearance" were never the same
        # quantity, and comparing them (as an earlier analysis did) is
        # meaningless.
        #
        # Telemetry only: reward, termination, MC returns, replay contents
        # and the executed action are all untouched by this block.
        if "dmin" not in info:
            raise IntentTrainError(
                "env.step() returned no 'dmin'; the swept clearance is required and there is no safe "
                f"fallback (info keys: {sorted(info)})")
        dmin = float(info["dmin"])
        if not np.isfinite(dmin) and dmin != float("inf"):
            raise IntentTrainError(f"env.step() returned a non-finite dmin {info['dmin']!r}")
        if np.isnan(dmin):
            raise IntentTrainError("env.step() returned NaN dmin")
        min_clearance = min(min_clearance, dmin)
        if dmin < discomfort_dist:
            discomfort_steps += 1

        event = info.get("event")
        transitions[-1].reward = float(reward)
        if terminated or truncated:
            outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}.get(event, "timeout")
            break
    if outcome is None:
        outcome = "timeout"

    returns = compute_mc_returns([t.reward for t in transitions], gamma)
    for t, g in zip(transitions, returns):
        t.mc_return = g
    episode_return = float(sum(t.reward for t in transitions))
    steps = len(transitions)
    return EpisodeCollectionResult(
        transitions=transitions,
        outcome=outcome,
        episode_return=episode_return,
        steps=steps,
        navigation_time=float(env.global_time),
        path_length=path_length,
        path_ratio=path_length / max(initial_goal_distance, 1e-12),
        min_clearance=min_clearance,
        discomfort_frequency=float(discomfort_steps) / max(steps, 1),
    )


class IntentReplay:
    """THE single V6 replay (Order C4R.2). Holds BOTH roles:

      * a DEMO reservoir  -- bounded, reservoir-sampled so an arbitrarily
        large IL corpus stays within a fixed memory budget while remaining
        a uniform sample of everything ever added;
      * an ONLINE ring    -- the usual fixed-capacity recent-experience ring.

    Real gap this closes (audit 2.2 point 2): the previous
    ``OnlineReplayBuffer`` stored ONLY online transitions, so once the IL
    phase ended every sampled batch was online-only and the ranking loss
    was permanently zero. That is no longer mathematically WRONG (an
    epsilon-random action is correctly never treated as an expert), but it
    silently discards the demonstration supervision the plan requires and
    invites catastrophic forgetting of the IL behaviour.

    ``sample`` therefore draws a FIXED ``demo_ratio`` share from the demo
    reservoir and the rest from the online ring, so online updates keep
    seeing expert-ranked demo samples. The loss semantics are unchanged:
    demo rows carry an expert set and get L_rank, online rows do not.

    Both sub-buffers AND the reservoir counter are in ``state_dict`` --
    without the counter, a resumed run's reservoir would accept new items
    with the wrong probability and silently bias the demo distribution.
    """

    def __init__(self, demo_capacity: int, online_capacity: int):
        if demo_capacity <= 0 or online_capacity <= 0:
            raise IntentTrainError(
                f"capacities must be positive, got demo={demo_capacity} online={online_capacity}")
        self.demo_capacity = int(demo_capacity)
        self.online_capacity = int(online_capacity)
        self._demo: List[IntentTransition] = []
        self._online: List[IntentTransition] = []
        self._online_next = 0
        self._demo_seen = 0   # total demo items EVER offered (reservoir denominator)

    # ---- sizes ----
    def __len__(self) -> int:
        return len(self._demo) + len(self._online)

    @property
    def n_demo(self) -> int:
        return len(self._demo)

    @property
    def n_online(self) -> int:
        return len(self._online)

    # ---- writes ----
    def add_demo(self, transitions: Sequence[IntentTransition], rng: np.random.Generator) -> None:
        """Reservoir sampling: after N offers each retained item is a
        uniform draw from all N. ``rng`` must be the caller's explicit,
        checkpointed generator -- never the global RNG."""
        for t in transitions:
            if t.source_role != "demo":
                raise IntentTrainError(f"add_demo got a {t.source_role!r} transition")
            self._demo_seen += 1
            if len(self._demo) < self.demo_capacity:
                self._demo.append(t)
            else:
                j = int(rng.integers(0, self._demo_seen))
                if j < self.demo_capacity:
                    self._demo[j] = t

    def add_online(self, transitions: Sequence[IntentTransition]) -> None:
        for t in transitions:
            if t.source_role != "online":
                raise IntentTrainError(f"add_online got a {t.source_role!r} transition")
            if len(self._online) < self.online_capacity:
                self._online.append(t)
            else:
                self._online[self._online_next] = t
                self._online_next = (self._online_next + 1) % self.online_capacity

    # ---- reads ----
    def sample_demo_only(self, batch_size: int, rng: np.random.Generator) -> List[IntentTransition]:
        """Demo rows only -- the ranking warm-up has no use for online rows
        (they carry no expert set, so they contribute nothing to L_rank)."""
        if batch_size <= 0:
            raise IntentTrainError(f"batch_size must be positive, got {batch_size}")
        if not self._demo:
            raise IntentTrainError("ranking warm-up needs a populated demo reservoir")
        idx = rng.integers(0, len(self._demo), size=min(batch_size, len(self._demo)))
        return [self._demo[i] for i in idx]

    def sample(self, batch_size: int, rng: np.random.Generator, demo_ratio: float) -> List[IntentTransition]:
        """Mixed batch. ``demo_ratio`` is the TARGET demo share; if one
        side is empty the other supplies the whole batch (so the IL phase
        works before any online data exists, and vice versa)."""
        if batch_size <= 0:
            raise IntentTrainError(f"batch_size must be positive, got {batch_size}")
        if not (0.0 <= demo_ratio <= 1.0):
            raise IntentTrainError(f"demo_ratio must be in [0,1], got {demo_ratio}")
        if not self._demo and not self._online:
            raise IntentTrainError("cannot sample from an empty replay")
        n_demo = int(round(batch_size * demo_ratio))
        if not self._demo:
            n_demo = 0
        elif not self._online:
            n_demo = batch_size
        n_online = batch_size - n_demo
        out: List[IntentTransition] = []
        if n_demo > 0:
            idx = rng.integers(0, len(self._demo), size=n_demo)
            out.extend(self._demo[i] for i in idx)
        if n_online > 0:
            idx = rng.integers(0, len(self._online), size=n_online)
            out.extend(self._online[i] for i in idx)
        return out

    # ---- resume ----
    ONLINE_PERSIST_DROPS_ACTION_FEATURES = True

    @staticmethod
    def _compact_humans(t: "IntentTransition"):
        """A2: keep only the ROWS THE MASK MARKS VALID.

        ``human_features`` is [MAX_HUMANS=20, 29] but every scenario runs 5
        pedestrians, so 15 of 20 rows are structural zeros -- ~58% of a
        persisted row. Store the valid rows plus the mask; the full
        [20, 29] tensor is rebuilt exactly on load (padding is zeros by
        construction, see build_intent_human_feature_batch)."""
        mask = np.asarray(t.human_mask, dtype=bool)
        return np.asarray(t.human_features, dtype=np.float32)[mask], mask

    @staticmethod
    def _expand_humans(rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
        full = np.zeros((len(mask), rows.shape[1]), dtype=np.float32)
        full[np.asarray(mask, dtype=bool)] = rows
        return full

    @staticmethod
    def _strip_for_persist(t: "IntentTransition") -> "IntentTransition":
        """Drop ``all_action_features`` from an ONLINE row before saving.

        It is 80x5 floats -- ~40% of a row -- and is provably never read
        for online samples: ``train_step`` indexes ``all_action_feats``
        ONLY at demo positions (online rows carry no expert set and get no
        ranking loss). Persisting it multiplied every checkpoint by ~1.7x
        for data that is dead weight. Restored as zeros on load so the
        batch stacking shape is unchanged.

        Uses dataclasses.replace -- it must NOT mutate the live object, or
        a still-running training loop would lose the array it is using.
        """
        import dataclasses
        rows, mask = IntentReplay._compact_humans(t)
        return dataclasses.replace(t, all_action_features=None,
                                    human_features=rows, human_mask=mask)

    def state_dict(self, include_demo: bool = True) -> dict:
        """C4RF.3: ``include_demo=False`` omits the demo corpus, which is
        IMMUTABLE once collected and is stored once as its own artifact.

        Real problem found by audit: embedding it in every checkpoint cost
        ~5.8 KB/transition measured, i.e. ~2.4 GB per full checkpoint at
        the formal budget; keeping a permanent copy every 500 episodes
        across 15 formal runs projected to ~705 GB. The demo side is
        byte-identical in all of them."""
        online = self._online
        if self.ONLINE_PERSIST_DROPS_ACTION_FEATURES:
            online = [self._strip_for_persist(t) for t in online]
        demo_rows = None
        if include_demo:
            demo_rows = [self._compact_humans(t) for t in self._demo]
        state = {
            "online": online,
            "online_action_feature_shape": (
                list(self._online[0].all_action_features.shape) if self._online else None),
            "online_next": self._online_next, "demo_seen": self._demo_seen,
            "demo_capacity": self.demo_capacity, "online_capacity": self.online_capacity,
            "demo_included": bool(include_demo),
        }
        if include_demo:
            import dataclasses
            state["demo"] = [
                dataclasses.replace(t, human_features=rows, human_mask=mask)
                for t, (rows, mask) in zip(self._demo, demo_rows)
            ]
        state["human_feature_rows_are_compacted"] = True
        return state

    def load_state_dict(self, state: dict) -> None:
        if (state["demo_capacity"] != self.demo_capacity
                or state["online_capacity"] != self.online_capacity):
            raise IntentTrainError(
                f"replay capacity mismatch: saved demo/online "
                f"{state['demo_capacity']}/{state['online_capacity']} != "
                f"{self.demo_capacity}/{self.online_capacity}")
        compacted = state.get("human_feature_rows_are_compacted", False)

        def _restore(t):
            if not compacted:
                return t
            import dataclasses
            return dataclasses.replace(
                t, human_features=self._expand_humans(t.human_features, t.human_mask))

        if state.get("demo_included", True):
            self._demo = [_restore(t) for t in state["demo"]]
        # else: the caller must have already populated the demo side from
        # the immutable corpus artifact BEFORE loading this state.
        elif not self._demo:
            raise IntentTrainError(
                "checkpoint omits the demo corpus (demo_included=False) but the replay's demo side is "
                "empty -- load the immutable IL corpus artifact first")
        online = [_restore(t) for t in state["online"]]
        shape = state.get("online_action_feature_shape")
        if shape is not None:
            import dataclasses
            zeros = np.zeros(tuple(shape), dtype=np.float32)
            online = [dataclasses.replace(t, all_action_features=zeros)
                      if t.all_action_features is None else t for t in online]
        elif any(t.all_action_features is None for t in online):
            raise IntentTrainError(
                "online rows were persisted without all_action_features but no shape was recorded")
        self._online = online
        self._online_next = int(state["online_next"])
        self._demo_seen = int(state["demo_seen"])


def run_online_training_step(
    env_config_path: Path, model: DistributionalValueModel, optimizer: torch.optim.Optimizer,
    action_table: np.ndarray, scenario: str, episode_seed: int, epsilon: float,
    buffer: IntentReplay, batch_size: int, explore_rng: np.random.Generator,
    sample_rng: np.random.Generator, tau_generator: torch.Generator, is_heldout: bool = False,
    gamma: float = 0.95, n_taus: int = 16, ranking_margin: float = 0.1,
    rho: float = RANK_CAP_RHO,
    ranking_batch_size: Optional[int] = None, n_samples: int = 60, horizon: int = 8, device: str = "cpu",
    demo_ratio: float = 0.20, grad_clip_norm: Optional[float] = None,
    measure_gradient_ratio: bool = False, updates: int = 1, belief_mode: str = "full",
) -> TrainStepResult:
    """ONE online RL iteration: collect one epsilon-greedy episode with the
    CURRENT model, push it into the ONLINE side of the replay, then take
    ``updates`` gradient steps on MIXED batches.

    C4R.2: each batch mixes a fixed ``demo_ratio`` share of DEMO samples
    with online experience, so the expert ranking supervision keeps acting
    during the online phase instead of vanishing the moment IL ends. The
    loss semantics are unchanged -- demo rows carry an expert set and get
    L_rank, online rows get L_MC only.

    Every randomness source (``explore_rng`` for action selection,
    ``sample_rng`` for replay sampling, ``tau_generator`` for the IQN
    quantile draw) is explicit and externally owned -- together with
    ``buffer.state_dict()`` and the model/optimizer/EMA state, this is
    EXACTLY what a caller must persist to resume the online loop
    bit-identically."""
    episode = collect_online_episode(
        env_config_path, model, action_table, scenario, episode_seed, epsilon, explore_rng,
        is_heldout=is_heldout, gamma=gamma, n_samples=n_samples, horizon=horizon, device=device,
        belief_mode=belief_mode,
    )
    buffer.add_online(episode.transitions)
    result = None
    for _ in range(max(1, updates)):
        batch = buffer.sample(min(batch_size, len(buffer)), sample_rng, demo_ratio=demo_ratio)
        result = train_step(
            model, optimizer, batch_to_tensors(batch, device=device), tau_generator, n_taus=n_taus,
            ranking_margin=ranking_margin, rho=rho, ranking_batch_size=ranking_batch_size,
            grad_clip_norm=grad_clip_norm, measure_gradient_ratio=measure_gradient_ratio,
        )
    return replace(
        result,
        outcome=episode.outcome,
        episode_return=episode.episode_return,
        episode_steps=episode.steps,
        navigation_time=episode.navigation_time,
        path_length=episode.path_length,
        path_ratio=episode.path_ratio,
        min_clearance=episode.min_clearance,
        discomfort_frequency=episode.discomfort_frequency,
        scenario=scenario,
        episode_seed=int(episode_seed),
        epsilon=float(epsilon),
    )


def run_il_update(
    model: DistributionalValueModel, optimizer: torch.optim.Optimizer, buffer: IntentReplay,
    batch_size: int, sample_rng: np.random.Generator, tau_generator: torch.Generator,
    n_taus: int = 16, ranking_margin: float = 0.1,
    ranking_batch_size: Optional[int] = None, device: str = "cpu",
    grad_clip_norm: Optional[float] = None, measure_gradient_ratio: bool = False,
    rho: float = RANK_CAP_RHO, diagnose: bool = False,
) -> TrainStepResult:
    """ONE IL mini-batch update (C4R.1).

    Real blocker this fixes (audit 2.2 point 1): the CLI previously moved
    EVERY demo transition to the device as ONE batch and then ran
    ``il_passes`` FULL-BATCH updates over it. At the frozen budget that is
    ~195,000 transitions (measured: ~39 transitions/episode x 5000), so a
    single update's ranking pass alone would push 195,000 x 80 = 15.6M rows
    through the encoder+IQN -- guaranteed OOM on a 24 GB card, and at the
    measured 0.61 s/update-at-256 it would extrapolate to days per update.

    Demo transitions now stay in the CPU-side replay reservoir and each
    update draws a ``batch_size`` mini-batch with an explicit, resumable
    sampler RNG. ``il_passes`` is therefore a count of fixed-size UPDATES,
    which is what it was always documented to be."""
    batch = buffer.sample(min(batch_size, len(buffer)), sample_rng, demo_ratio=1.0)
    return train_step(
        model, optimizer, batch_to_tensors(batch, device=device), tau_generator, n_taus=n_taus,
        ranking_margin=ranking_margin, rho=rho, ranking_batch_size=ranking_batch_size,
        grad_clip_norm=grad_clip_norm, measure_gradient_ratio=measure_gradient_ratio,
        diagnose=diagnose,
    )


# --------------------------------------------------------------------- #
# EMA (review point 8, real bug found by review: point 8 explicitly
# requires an EMA of the model weights -- deployment/eval should read the
# smoothed weights, not the noisy raw training weights of whatever step
# training happened to stop at).
# --------------------------------------------------------------------- #

class EMAModel:
    """Exponential moving average of a model's parameters. Deliberately
    the simplest correct thing: one ``decay`` constant, no warmup
    schedule (nothing in the consolidation plan calls for one, and adding
    it would be an unrequested architecture change)."""

    def __init__(self, model: DistributionalValueModel, decay: float):
        if not (0.0 < decay < 1.0):
            raise IntentTrainError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: DistributionalValueModel) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                shadow_v = self.shadow[k]
                if torch.is_floating_point(shadow_v):
                    shadow_v.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    shadow_v.copy_(v)  # non-float buffers (e.g. num_batches_tracked) just track raw

    def copy_to(self, model: DistributionalValueModel) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        """Restore the shadow, MOVING each tensor onto the device the
        shadow currently lives on.

        Real bug found by the C4RF.6 final-hash dry-run (CUDA resume only,
        which is the formal path): checkpoints are read with
        ``map_location="cpu"``, so a restored shadow was CPU-resident while
        the model had already been moved to CUDA. The next ``update()``
        then did ``cpu_tensor.mul_(...).add_(cuda_tensor)`` and raised
        "Expected all tensors to be on the same device". CPU-only resume
        never hit it, and the earlier CUDA test covered `train`, not
        `resume` -- this is exactly the gap a final-hash dry-run exists to
        catch."""
        if set(state.keys()) != set(self.shadow.keys()):
            raise IntentTrainError("EMA state_dict keys do not match this model's parameters/buffers")
        self.shadow = {k: v.detach().clone().to(self.shadow[k].device) for k, v in state.items()}

    def to(self, device) -> "EMAModel":
        """Move the shadow (used when the model is moved after construction)."""
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self


# --------------------------------------------------------------------- #
# Formal six-scenario + held-out stress evaluation (review point 8).
# Matches this project's existing formal six-scenario convention (see
# crowd_nav/belief_mdp/runtime.py's SIX_SCENARIOS: "shape:size:human_count",
# baseline/dense/large x circle/square) -- reimplemented independently here
# (no cross-import of belief_mdp, which is an unrelated sibling package)
# using ONLY this chain's own CrowdSim-building convention.
# --------------------------------------------------------------------- #

FORMAL_SIX_SCENARIOS: Dict[str, Tuple[str, float, int]] = {
    "baseline_circle": ("circle", 4.0, 5),
    "baseline_square": ("square", 10.0, 10),
    "dense_circle": ("circle", 4.0, 10),
    "dense_square": ("square", 10.0, 20),
    "large_circle": ("circle", 6.0, 12),
    "large_square": ("square", 14.0, 20),
}

# FROZEN, disjoint from JUNCTION_TRAIN_SEEDS (96001-96200) and
# JUNCTION_HELDOUT_SEEDS (96501-96600) -- same "9Xxxx block" convention.
# Independent held-out stress seeds: never used for training, IL
# collection, or checkpoint selection, only for this one-time formal report.
FORMAL_EVAL_HELDOUT_SEEDS: Tuple[int, ...] = tuple(range(97001, 97101))  # 100

# OPTIMIZER seeds for the 2x2 causal diagnostic. Deliberately a separate
# ROLE from the formal optimizer seeds (98201-98205): the 2x2 exists to
# decide how the update is assembled, so its runs must never be able to
# become a formal result, seed a selection, or touch a paper number. The
# launcher refuses them in a formal plan and preflight proves them disjoint
# from every other block.
DIAGNOSTIC_OPTIMIZER_SEEDS: Tuple[int, ...] = (98211, 98212, 98213)

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
PAPER_MAIN_EPISODES_PER_SCENARIO = 500
# case_id follows FORMAL_SIX_SCENARIOS' insertion order, which matches
# test8.py's hardcoded list: baseline_circle=0 ... large_square=5.
PAPER_MAIN_CASE_IDS: Dict[str, int] = {name: i for i, name in enumerate(FORMAL_SIX_SCENARIOS)}


def paper_main_episode_seed(scenario_name: str, episode_index: int,
                            base_seed: int = PAPER_MAIN_BASE_SEED) -> int:
    """``(base_seed + case_id*1_000_003 + ep) % (2**31 - 1)`` -- bit-identical
    to test8.py / evaluate_bdvl_paper_main.py."""
    if scenario_name not in PAPER_MAIN_CASE_IDS:
        raise IntentTrainError(f"unknown paper-main scenario {scenario_name!r}")
    if episode_index < 0:
        raise IntentTrainError(f"episode_index must be >= 0, got {episode_index}")
    case_id = PAPER_MAIN_CASE_IDS[scenario_name]
    return (base_seed + case_id * 1_000_003 + episode_index) % (2**31 - 1)


def paper_main_jobs(episodes_per_scenario: int = PAPER_MAIN_EPISODES_PER_SCENARIO,
                    base_seed: int = PAPER_MAIN_BASE_SEED) -> List[Tuple[str, int, bool]]:
    """The frozen paper-main episode identity table."""
    if episodes_per_scenario <= 0:
        raise IntentTrainError(f"episodes_per_scenario must be positive, got {episodes_per_scenario}")
    return [(name, paper_main_episode_seed(name, ep, base_seed), False)
            for name in FORMAL_SIX_SCENARIOS for ep in range(episodes_per_scenario)]


def build_formal_scenario_env(env_config_path: Path, scenario_name: str):
    if scenario_name not in FORMAL_SIX_SCENARIOS:
        raise IntentTrainError(f"unknown formal scenario {scenario_name!r}, expected one of {sorted(FORMAL_SIX_SCENARIOS)}")
    shape, size, human_num = FORMAL_SIX_SCENARIOS[scenario_name]
    import configparser
    cfg = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not cfg.read(str(env_config_path)):
        raise IntentTrainError(f"env config not found: {env_config_path}")
    cfg.set("sim", "test_sim", f"{shape}_crossing")
    cfg.set("sim", "human_num", str(int(human_num)))
    if shape == "square":
        cfg.set("sim", "square_width", str(float(size)))
    else:
        cfg.set("sim", "circle_radius", str(float(size)))
    cfg.set("robot", "policy", "orca")
    env = CrowdSim(); env.configure(cfg)
    env.phase = "test"
    robot = Robot(cfg, "robot")
    robot_orca = ORCA(); robot_orca.configure(cfg)
    robot_orca.multiagent_training = True  # see _make_standard_env's comment
    robot.set_policy(robot_orca); robot.visible = True; robot.time_step = FROZEN_VALUES["dt"]
    robot.env = env
    env.set_robot(robot)
    return env, robot, shape, float(size)


def run_formal_scenario_episode(
    env_config_path: Path, model: DistributionalValueModel, action_table: np.ndarray,
    scenario_name: str, episode_seed: int, n_samples: int = 60, horizon: int = 8, device: str = "cpu",
) -> AblationEpisodeResult:
    """ONE formal-eval episode: the model's OWN greedy full-belief policy
    (mode="full", matching deployment) in a generic multi-human crowd
    (not the junction scenario -- these six scenarios test general crowd
    navigation, not the goal-ambiguity mechanism specifically)."""
    env, robot, shape, size = build_formal_scenario_env(env_config_path, scenario_name)
    env.case_counter["test"] = episode_seed % (2**32 - 1)
    env.reset()
    # C1.1: the public candidate provider must match the scenario's REAL
    # geometry. Using circle_scene() for square scenarios (the previous
    # behaviour) put every candidate destination on a ring unrelated to
    # where square-crossing pedestrians actually walk.
    scene = square_scene(width=size, n_rows=4) if shape == "square" else circle_scene(radius=size, n_sectors=8)
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=TRACKER_DEFAULTS["speed_prior"])
    planner_rng = np.random.default_rng(episode_seed)
    max_steps = int(round(FROZEN_VALUES["time_limit"] / FROZEN_VALUES["dt"])) + 1
    outcome = None
    step = 0
    for step in range(max_steps):
        humans = [
            HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
            for i, h in enumerate(env.humans)
        ]
        bank.update({h.track_id: (h.px, h.py) for h in humans})
        robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
        remaining = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])
        human_feats, human_mask = build_intent_human_feature_batch(
            bank, robot_obs, humans, mode="full", rng=planner_rng, horizon=horizon, n_samples=n_samples,
        )
        results = score_candidates_v5(
            model, robot_obs, human_feats, human_mask, action_table, remaining, device=device)
        best = max(results, key=lambda r: r.q_mean)
        avx, avy = action_table[best.action_index]
        _, _reward, terminated, truncated, info = env.step(ActionXY(float(avx), float(avy)))
        if terminated or truncated:
            event = info.get("event")
            outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}.get(event, "timeout")
            break
    if outcome is None:
        outcome = "timeout"
    return AblationEpisodeResult(mode="full", episode_seed=episode_seed, outcome=outcome, steps=step + 1)


def run_formal_six_scenario_evaluation(
    env_config_path: Path, model: DistributionalValueModel, action_table: np.ndarray,
    episode_seeds: Sequence[int] = FORMAL_EVAL_HELDOUT_SEEDS, n_samples: int = 60,
    horizon: int = 8, device: str = "cpu",
) -> Dict[str, List[AblationEpisodeResult]]:
    """The formal, one-time report: every scenario in FORMAL_SIX_SCENARIOS
    run on the SAME frozen held-out seed set. Not used for training,
    checkpoint selection, or any earlier decision -- only for the final
    report (same discipline as belief_mdp's VALIDATION_SCENARIOS vs.
    SIX_SCENARIOS split)."""
    results: Dict[str, List[AblationEpisodeResult]] = {name: [] for name in FORMAL_SIX_SCENARIOS}
    for scenario_name in FORMAL_SIX_SCENARIOS:
        for episode_seed in episode_seeds:
            results[scenario_name].append(run_formal_scenario_episode(
                env_config_path, model, action_table, scenario_name, episode_seed, n_samples=n_samples,
                horizon=horizon, device=device,
            ))
    return results


def summarize_scenario_results(results: Sequence[AblationEpisodeResult]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        raise IntentTrainError("cannot summarize an empty result list")
    return {
        "n": float(n),
        "success_rate": sum(1 for r in results if r.outcome == "success") / n,
        "collision_rate": sum(1 for r in results if r.outcome == "collision") / n,
        "timeout_rate": sum(1 for r in results if r.outcome == "timeout") / n,
        "mean_steps": sum(r.steps for r in results) / n,
    }


# --------------------------------------------------------------------- #
# Order 4: a FIXED IL audit set.
# --------------------------------------------------------------------- #

IL_AUDIT_SET_SIZE = 512
IL_AUDIT_PER_SCENARIO = 256
#: Rows per forward pass when scoring the audit set. The audit is scored over
#: ALL 80 actions, so a one-shot pass is n_rows * 80 network rows: at the
#: frozen 50-episodes-per-scenario split that is 4316 * 80 = 345,280 and it
#: asked CUDA for 2.63 GiB in a single allocation (measured -- it OOM'd on a
#: shared 24 GB card). Chunking bounds the peak without touching the numbers:
#: every statistic here is a plain sum over rows, so it is accumulated as
#: sum/count and divided once at the end.
IL_AUDIT_CHUNK_ROWS = 256


def build_il_audit_set(demo_transitions: Sequence[IntentTransition],
                       scenario_of, n_per_scenario: int = IL_AUDIT_PER_SCENARIO) -> List[IntentTransition]:
    """Freeze a balanced, deterministic slice of the IL corpus.

    Why this exists: during IL the corpus is immutable, so the MC targets
    are STATIONARY -- which makes IL the one phase where a rising MC loss
    is unambiguous evidence that the ranking term is displacing value
    regression, with no "the target moved" escape. On the lambda=380 run
    that is exactly what happened: MC loss bottomed at 0.157 (pass 400) and
    finished at 0.469, 3.0x its own best, while rank loss fell.

    The training loss cannot serve as that detector, because it is measured
    on a fresh random minibatch every pass and carries sampling noise. This
    set is fixed, balanced across both scenarios, and scored with FIXED
    midpoint quantiles, so ``audit_mc_loss`` is a deterministic function of
    the weights alone -- comparable across passes, runs and machines.

    Selection is by stable position within each scenario, not by RNG: no
    training random stream is touched, so adding the audit cannot perturb
    resume bit-identity.
    """
    by_scenario: Dict[str, List[IntentTransition]] = {}
    for t in demo_transitions:
        by_scenario.setdefault(str(scenario_of(t)), []).append(t)
    if len(by_scenario) < 2:
        raise IntentTrainError(
            f"audit set needs both training scenarios, got {sorted(by_scenario)}")
    audit: List[IntentTransition] = []
    for scenario in sorted(by_scenario):
        rows = by_scenario[scenario]
        if len(rows) < n_per_scenario:
            raise IntentTrainError(
                f"scenario {scenario!r} has {len(rows)} demo rows, need {n_per_scenario} for the audit set")
        stride = len(rows) // n_per_scenario
        audit.extend(rows[i * stride] for i in range(n_per_scenario))
    return audit


def il_audit_identity(audit: Sequence[IntentTransition]) -> str:
    """Content hash of the audit set. Pinned in the checkpoint so a later
    'audit_mc_loss' can never be compared against a DIFFERENT set of rows."""
    h = hashlib.sha256()
    h.update(str(len(audit)).encode())
    for t in audit:
        h.update(np.asarray(t.robot_features, dtype=np.float32).tobytes())
        h.update(np.asarray(t.human_features, dtype=np.float32).tobytes())
        h.update(np.asarray(t.human_mask, dtype=bool).tobytes())
        h.update(np.asarray(t.action_features, dtype=np.float32).tobytes())
        h.update(str(t.action_index).encode())
        h.update(f"{float(t.mc_return):.12g}".encode())
        h.update(str(tuple(t.expert_action_indices)).encode())
    return h.hexdigest()


@torch.no_grad()
def _audit_mc_loss(model, batch: IntentBatch, n_taus: int,
                   chunk_rows: int = IL_AUDIT_CHUNK_ROWS) -> float:
    device = batch.robot_feats.device
    B = batch.robot_feats.shape[0]
    total, count = 0.0, 0
    for lo in range(0, B, chunk_rows):
        hi = min(lo + chunk_rows, B)
        n = hi - lo
        tau = ((torch.arange(n_taus, dtype=torch.float32, device=device) + 0.5) / n_taus)
        tau = tau.unsqueeze(0).expand(n, n_taus)
        predicted = model(batch.robot_feats[lo:hi], batch.human_feats[lo:hi],
                          batch.human_mask[lo:hi], batch.action_feats[lo:hi], tau)
        # quantile_huber_loss already reduces to a SCALAR (its final .mean()
        # is over the batch dim, uniformly per row), so the exact overall mean
        # is the ROW-WEIGHTED mean of the chunk means -- counting elements
        # instead would silently average the chunk means and disagree with the
        # one-shot value whenever the last chunk is short.
        per = quantile_huber_loss(predicted, tau, batch.mc_returns[lo:hi].expand(n, 1))
        total += float(per) * n
        count += n
    return total / max(count, 1)


@torch.no_grad()
def _audit_scores_chunked(model, batch: IntentBatch, n_taus: int, lo: int, hi: int):
    """Expected Q over all 80 actions for rows [lo, hi), fixed midpoint tau."""
    device = batch.robot_feats.device
    n = hi - lo
    n_actions = batch.all_action_feats.shape[1]
    fixed_tau = (torch.arange(n_taus, dtype=torch.float32, device=device) + 0.5) / n_taus
    state_emb = model.encode(batch.robot_feats[lo:hi], batch.human_feats[lo:hi],
                             batch.human_mask[lo:hi])
    state_rep = state_emb.repeat_interleave(n_actions, dim=0)
    action_emb = model.action_encoder(batch.all_action_feats[lo:hi].reshape(n * n_actions, -1))
    tau_rep = fixed_tau.unsqueeze(0).expand(n * n_actions, n_taus)
    return model.value_network(state_rep, action_emb, tau_rep).mean(dim=1).view(n, n_actions)


@torch.no_grad()
def _audit_rank_loss(model, audit: Sequence[IntentTransition], batch: IntentBatch,
                     n_taus: int, ranking_margin: float,
                     chunk_rows: int = IL_AUDIT_CHUNK_ROWS) -> float:
    total, count = 0.0, 0
    for lo in range(0, len(audit), chunk_rows):
        hi = min(lo + chunk_rows, len(audit))
        scores = _audit_scores_chunked(model, batch, n_taus, lo, hi)
        for r in range(hi - lo):
            experts = audit[lo + r].expert_action_indices
            if not experts:
                continue
            total += float(expert_ranking_loss(scores[r], experts, ranking_margin))
            count += 1
    return total / count if count else 0.0


def evaluate_il_audit(model, audit: Sequence[IntentTransition], n_taus: int = 16,
                      ranking_margin: float = 0.1, device: str = "cpu",
                      scenario_of: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Deterministic: fixed rows, fixed midpoint quantiles, no RNG. Two
    calls on the same weights MUST return identical numbers.

    ``scenario_of`` (one label per audit row) additionally reports the MC
    loss PER SCENARIO. The aggregate can hide a scenario going backwards
    while the other improves, which is exactly the kind of thing the 2x2
    has to be able to see.
    """
    was_training = bool(model.training)
    model.eval()
    try:
        batch = batch_to_tensors(audit, device=device)
        out = {
            "audit_mc_loss": _audit_mc_loss(model, batch, n_taus),
            "audit_rank_loss": _audit_rank_loss(model, audit, batch, n_taus, ranking_margin),
            "n": float(len(audit)),
        }
        if scenario_of is not None:
            if len(scenario_of) != len(audit):
                raise IntentTrainError(
                    f"scenario_of has {len(scenario_of)} labels for {len(audit)} audit rows")
            for name in sorted(set(str(x) for x in scenario_of)):
                rows = [t for t, sc in zip(audit, scenario_of) if str(sc) == name]
                sub = batch_to_tensors(rows, device=device)
                out[f"audit_mc_loss_{name}"] = _audit_mc_loss(model, sub, n_taus)
                out[f"audit_n_{name}"] = float(len(rows))
        return out
    finally:
        model.train(was_training)


# --------------------------------------------------------------------- #
# Order 5: pre-registered checkpoint selection.
# --------------------------------------------------------------------- #

CHECKPOINT_SELECTION_MIN_SR = 0.90


def select_checkpoint(candidates: Sequence[dict], min_sr: float = CHECKPOINT_SELECTION_MIN_SR) -> dict:
    """Pick ONE checkpoint by a rule fixed BEFORE any of them was scored.

    Why a rule and not judgement: on the lambda=380 run the four milestones
    sat at genuinely different operating points (junction speed 0.501 to
    0.907), so "which is best" is undefined until you say what you are
    optimising. Picking afterwards -- either "always take final" or "take
    the one that looks good" -- is selection on seen data. The first is
    also demonstrably bad there: that run's final checkpoint was the WORST
    on standard (0.83 vs 0.99, McNemar p=0.0001).

    Each candidate is a dict with ``name`` and a ``scenarios`` mapping of
    scenario -> {success_rate, collision_rate, discomfort_frequency,
    navigation_time}. Scored on the SELECTION development seeds only --
    never the diagnostic block (already seen), never the paper's seeds.

    Order:
      1. every scenario must reach ``min_sr``, else the candidate is
         ineligible (NOT a fallback to "best available");
      2. maximise the WORST scenario's SR -- a checkpoint that is excellent
         on one scenario and poor on the other is not a good policy;
      3. maximise macro-average SR;
      4. minimise the worst scenario's collision rate;
      5. minimise discomfort frequency, then navigation time;
      6. ties break to the LATER checkpoint.

    Raises when nothing qualifies: a run with no eligible checkpoint has
    failed, and must be reported as such rather than quietly yielding its
    least-bad weights.
    """
    if not candidates:
        raise IntentTrainError("no candidates supplied to select_checkpoint")
    eligible = []
    for c in candidates:
        scen = c["scenarios"]
        if not scen:
            raise IntentTrainError(f"candidate {c.get('name')!r} has no scenario results")
        if all(float(s["success_rate"]) >= min_sr for s in scen.values()):
            eligible.append(c)
    if not eligible:
        detail = {c["name"]: {k: round(float(v["success_rate"]), 3) for k, v in c["scenarios"].items()}
                  for c in candidates}
        raise IntentTrainError(
            f"RUN FAILED: no checkpoint reached SR >= {min_sr} on every development scenario. "
            f"Per-candidate SR: {detail}. This run does not yield a usable checkpoint -- it must be "
            f"reported as a failure, not resolved by relaxing the bar or taking the least-bad weights.")

    def key(c):
        scen = c["scenarios"]
        srs = [float(s["success_rate"]) for s in scen.values()]
        crs = [float(s["collision_rate"]) for s in scen.values()]
        disc = [float(s.get("discomfort_frequency", 0.0)) for s in scen.values()]
        nav = [float(s.get("navigation_time", 0.0)) for s in scen.values()]
        return (-min(srs),                       # 2
                -sum(srs) / len(srs),            # 3
                max(crs),                        # 4
                sum(disc) / len(disc),           # 5
                sum(nav) / len(nav),
                -int(c.get("order", 0)))         # 6: later wins ties
    return sorted(eligible, key=key)[0]


# --------------------------------------------------------------------- #
# Order 2R: projected, norm-normalised gradient combination.
# --------------------------------------------------------------------- #




MODULE_GROUPS = ("encoder", "action_encoder", "value_network")


def _module_of(name: str) -> str:
    for g in MODULE_GROUPS:
        if name.startswith(g + "."):
            return g
    return "other"


def _grouped_norms(model, vectors) -> dict:
    """{module: l2 norm} for a per-parameter vector list aligned to
    ``[p for p in model.parameters() if p.requires_grad]``."""
    out = {g: 0.0 for g in MODULE_GROUPS + ("other",)}
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    for n, v in zip(names, vectors):
        if v is not None:
            out[_module_of(n)] += float((v ** 2).sum())
    return {g: v ** 0.5 for g, v in out.items()}


def _grouped_dot(model, a_list, b_list) -> dict:
    out = {g: 0.0 for g in MODULE_GROUPS + ("other",)}
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    for n, a, b in zip(names, a_list, b_list):
        if a is not None and b is not None:
            out[_module_of(n)] += float((a * b).sum())
    return out


def adam_state_norms(optimizer) -> tuple:
    """(step, |exp_avg|, |exp_avg_sq|) summed over all parameters."""
    step, m_sq, v_sq = 0, 0.0, 0.0
    for st in optimizer.state.values():
        if "step" in st:
            sv = st["step"]
            step = max(step, int(sv.item()) if torch.is_tensor(sv) else int(sv))
        if "exp_avg" in st:
            m_sq += float((st["exp_avg"] ** 2).sum())
        if "exp_avg_sq" in st:
            v_sq += float((st["exp_avg_sq"] ** 2).sum())
    return step, m_sq ** 0.5, v_sq ** 0.5


def combine_gradients(mc_grads, rank_grads, rho: float = RANK_CAP_RHO,
                      epsilon: float = 1e-12):
    """g = g_MC + scale * g'_rank, where g'_rank has had any component
    OPPOSING g_MC removed and is CAPPED at rho * |g_MC| -- never amplified.

        c           = (g_rank . g_MC) / |g_MC|^2
        g'_rank     = g_rank - min(0, c) * g_MC
        rank_budget = min(|g'_rank|, rho * |g_MC|)
        scale       = rank_budget / (|g'_rank| + eps)      # always <= 1
        g           = g_MC + scale * g'_rank

    WHAT THIS REPLACES, AND WHY. The previous rule normalised the projected
    ranking gradient to EXACTLY rank_share * |g_MC| whenever it was not
    numerically zero. That is a floor as well as a ceiling: a hinge that is
    nearly satisfied, contributing a tiny gradient, was scaled back UP to a
    fixed multiple of the value gradient, so the ranking term never yielded
    as it converged. Measured at the retired share of 2.0, on three seeds
    (12-branch 2x2, IL 2000 each):

        share=2.0, Adam kept    final/best MC = 1.500 / 1.285 / 1.162
        share=2.0, Adam reset   final/best MC = 1.526 / 1.282 / 1.217
        MC-only                 final/best MC = 1.056 / 1.059 / 1.036

    The damage is not only late over-fitting: the FIXED TRAINING set degraded
    too (0.139-0.175 against MC-only's 0.102-0.109), and the best MC ever
    reached was ~0.117-0.123 against MC-only's ~0.086-0.089. Ranking pressure
    was costing roughly 30% of the achievable value fit from the start.

    Resetting Adam at the phase transition was tested in the same experiment
    and did NOT help -- worse on 2 of 3 seeds -- so the earlier "carried
    optimizer state" hypothesis is retired.

    MC-only is not the answer either: it left rank loss at 0.10-0.13, margin
    NEGATIVE and top-1 at 22-31%. So the ranking supervision has to stay; it
    just must not be able to outbid value regression.

    Consequences worth stating plainly:
      * after projection g'_rank . g_MC >= 0, so the combined RAW gradient
        keeps a non-negative MC component. This is a statement about the
        GRADIENT, not the step: Adam rescales per coordinate, so the realised
        parameter change points elsewhere and its alignment with MC descent
        can fall close to zero WITHOUT dot(g_MC, delta) changing sign. Health
        is judged on the fixed audit set, with cos(-delta, g_MC) as the
        efficiency diagnostic (test_post_adam_alignment);
      * scale <= 1 ALWAYS: a ranking gradient smaller than its budget passes
        through untouched, and only one larger than the budget is shrunk;
      * a hinge that is fully satisfied gives exactly zero gradient, and the
        update is then bitwise the pure-MC update.

    Returns ``(combined, info)``; info carries the raw norms, the cosine and
    the scale actually applied, for diagnostics only.
    """
    if not (0.0 <= rho):
        raise IntentTrainError(f"rho must be non-negative, got {rho}")
    flat_mc = [g if g is not None else None for g in mc_grads]
    flat_rank = [g if g is not None else None for g in rank_grads]
    if len(flat_mc) != len(flat_rank):
        raise IntentTrainError("gradient lists must be aligned parameter-for-parameter")

    dot = 0.0
    mc_sq = 0.0
    rank_sq = 0.0
    for a, b in zip(flat_mc, flat_rank):
        if a is not None:
            mc_sq += float((a ** 2).sum())
        if b is not None:
            rank_sq += float((b ** 2).sum())
        if a is not None and b is not None:
            dot += float((a * b).sum())
    mc_norm, rank_norm = mc_sq ** 0.5, rank_sq ** 0.5
    cosine = dot / (mc_norm * rank_norm) if mc_norm > 0 and rank_norm > 0 else 0.0

    if not all(np.isfinite(v) for v in (dot, mc_norm, rank_norm)):
        raise IntentTrainError(f"non-finite gradient statistics dot={dot} |mc|={mc_norm} |rank|={rank_norm}")

    # remove only the OPPOSING component; a cooperative or orthogonal
    # ranking gradient passes through untouched
    c = dot / (mc_sq + epsilon) if mc_sq > 0 else 0.0
    proj = min(0.0, c)
    projected = []
    proj_sq = 0.0
    for a, b in zip(flat_mc, flat_rank):
        if b is None and a is None:
            projected.append(None)
            continue
        v = (b.clone() if b is not None else torch.zeros_like(a))
        if proj != 0.0 and a is not None:
            v = v - proj * a
        projected.append(v)
        proj_sq += float((v ** 2).sum())
    proj_norm = proj_sq ** 0.5

    # Numerical guard: a projected ranking gradient that is negligible
    # RELATIVE to the MC gradient is treated as exactly zero, so the update
    # degenerates to pure MC rather than carrying floating-point noise.
    # Under the cap this is belt-and-braces -- noise below the budget now
    # passes through at its own (negligible) size instead of being scaled up
    # -- but it keeps the zero case bitwise exact.
    negligible = proj_norm <= RANK_GRADIENT_ZERO_TOL * max(mc_norm, 1.0)
    scale = 0.0
    if proj_norm > 0.0 and mc_norm > 0.0 and not negligible:
        # CAP, never amplify: the budget is an upper bound, so a ranking
        # gradient already inside it passes through at scale 1.0.
        rank_budget = min(proj_norm, rho * mc_norm)
        scale = rank_budget / (proj_norm + epsilon)
        if scale > 1.0:      # only reachable through the epsilon; clamp anyway
            scale = 1.0
    if not np.isfinite(scale):
        raise IntentTrainError(
            f"non-finite rank scale {scale} (|g_MC|={mc_norm}, |g'_rank|={proj_norm})")

    combined = []
    for a, v in zip(flat_mc, projected):
        if a is None and v is None:
            combined.append(None)
        elif a is None:
            combined.append(scale * v)
        elif v is None:
            combined.append(a.clone())
        else:
            combined.append(a + scale * v)
    for g in combined:
        if g is not None and not bool(torch.isfinite(g).all()):
            raise IntentTrainError("combined gradient contains non-finite values")
    info = {
        "mc_grad_norm": mc_norm, "rank_grad_norm": rank_norm,
        "projected_rank_norm": proj_norm, "gradient_cosine": cosine,
        "projection_coefficient": float(proj), "rank_scale": float(scale),
        "conflict_removed": bool(proj != 0.0), "rank_negligible": bool(negligible),
    }
    return combined, info


# --------------------------------------------------------------------- #
# Order 1W: ranking warm-up.
# --------------------------------------------------------------------- #

WARMUP_MAX_STEPS = 750
#: RETIRED as a gate. top1 >= 0.95 was set on an audit set drawn from the
#: TRAINING pool, i.e. on leaked rows: warm-up hit 0.951 in 500 steps that
#: way. On a genuinely held-out split the same procedure plateaus at 0.933
#: (three independent runs: 0.9328 / 0.933 / 0.937; 500 -> 750 steps moved it
#: only 0.929 -> 0.933) while rank_loss reached 0.0166 and margin +0.117.
#: A threshold calibrated on a leaked measurement is not evidence about the
#: held-out one, so it is recorded as telemetry and no longer aborts.
WARMUP_RANK_LOSS_MAX = 0.05


def expert_rank_diagnostics(model, audit: Sequence[IntentTransition], n_taus: int = 16,
                            device: str = "cpu",
                            chunk_rows: int = IL_AUDIT_CHUNK_ROWS) -> Dict[str, float]:
    """top-1 rate, expert-vs-hardest-negative margin, and score spread on a
    FIXED row set. These are the quantities that actually say whether the
    ranking objective was learned -- rank_loss alone sat at exactly the
    margin for an entire 951-pass run while looking merely 'flat'.

    Chunked over rows: scoring all 80 actions for the whole audit set at once
    is n_rows * 80 network rows (4316 * 80 = 345,280 at the frozen split,
    2.63 GiB in one allocation -- measured, it OOM'd). Every statistic is a
    plain sum over rows, so chunking changes the peak, not the numbers.
    """
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            batch = batch_to_tensors(audit, device=device)
            top1, margin_sum, range_sum, loss_sum, scored = 0, 0.0, 0.0, 0.0, 0
            n_rows = len(audit)
            for lo in range(0, n_rows, chunk_rows):
                hi = min(lo + chunk_rows, n_rows)
                scores = _audit_scores_chunked(model, batch, n_taus, lo, hi)
                range_sum += float((scores.max(dim=1).values - scores.min(dim=1).values).sum())
                for r in range(hi - lo):
                    ex = audit[lo + r].expert_action_indices
                    if not ex:
                        continue
                    srow = scores[r]
                    mask = torch.zeros_like(srow, dtype=torch.bool)
                    mask[list(ex)] = True
                    margin_sum += float(srow[mask].max() - srow[~mask].max())
                    if int(srow.argmax()) in set(ex):
                        top1 += 1
                    loss_sum += float(expert_ranking_loss(srow, ex, 0.1))
                    scored += 1
            return {
                "expert_top1_rate": top1 / max(scored, 1),
                "expert_margin_mean": margin_sum / max(scored, 1),
                "score_range_mean": range_sum / max(n_rows, 1),
                "audit_rank_loss": loss_sum / max(scored, 1),
            }
    finally:
        model.train(was_training)


def run_ranking_warmup(model, optimizer, buffer: IntentReplay, batch_size: int,
                       sample_rng: np.random.Generator, audit: Sequence[IntentTransition],
                       *, max_steps: int = WARMUP_MAX_STEPS, n_taus: int = 16,
                       ranking_margin: float = 0.1, ranking_batch_size: Optional[int] = None,
                       grad_clip_norm: Optional[float] = None, device: str = "cpu",
                       check_interval: int = 50,
                       rank_loss_max: float = WARMUP_RANK_LOSS_MAX, log=None) -> Dict[str, object]:
    """Train the RANKING objective alone until the fixed audit set says it
    was actually learned, or give up.

    Why a separate phase: measured on a fixed 128-row real batch, ranking
    alone reaches top-1 0.984 and margin +0.088 in 500 steps, while under
    joint training with an ill-scaled weight it never left the margin
    (score range 0.0017, top-1 52.7%). MC supervises only the executed
    action, so it constrains nothing about the other 79 -- the two objectives
    are not competing for the same signal, they simply need the ranking
    structure to exist BEFORE value regression starts moving the scores.

    Gate: audit_rank_loss <= rank_loss_max AND mean margin > 0 -- the two
    quantities the ranking objective is DEFINED by. top-1 is recorded but
    does not gate: its 0.95 threshold was calibrated on an audit set drawn
    from the training pool, and a number measured on leaked rows says
    nothing about a held-out set (measured: 0.951 in-pool at 500 steps vs a
    0.933 plateau held out).

    Not reaching the gate within max_steps is a HARD STOP -- entering joint
    training with an unlearned ranking term is what produced the first
    aborted pilot.
    """
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    history = []
    passed = False
    step = 0
    for step in range(max_steps + 1):
        if step % check_interval == 0:
            d = expert_rank_diagnostics(model, audit, n_taus=n_taus, device=device)
            d["step"] = step
            history.append(d)
            if log is not None:
                log(f"WARMUP[{step}/{max_steps}] rank={d['audit_rank_loss']:.5f} "
                    f"top1={d['expert_top1_rate']:.3f} margin={d['expert_margin_mean']:+.5f} "
                    f"range={d['score_range_mean']:.5f}")
            if d["audit_rank_loss"] <= rank_loss_max and d["expert_margin_mean"] > 0.0:
                passed = True
                break
        if step == max_steps:
            break
        batch_rows = buffer.sample_demo_only(batch_size, sample_rng)
        batch = batch_to_tensors(batch_rows, device=device)
        demo_positions = list(range(len(batch_rows)))
        if ranking_batch_size is not None:
            demo_positions = demo_positions[:ranking_batch_size]
        n_actions = batch.all_action_feats.shape[1]
        fixed_tau = (torch.arange(n_taus, dtype=torch.float32, device=batch.robot_feats.device) + 0.5) / n_taus
        idx = torch.as_tensor(demo_positions, dtype=torch.long, device=batch.robot_feats.device)
        n_demo = len(demo_positions)
        emb = model.encode(batch.robot_feats[idx], batch.human_feats[idx], batch.human_mask[idx])
        state_rep = emb.repeat_interleave(n_actions, dim=0)
        act = model.action_encoder(batch.all_action_feats[idx].reshape(n_demo * n_actions, -1))
        tau = fixed_tau.unsqueeze(0).expand(n_demo * n_actions, n_taus)
        scores = model.value_network(state_rep, act, tau).mean(dim=1).view(n_demo, n_actions)
        losses = [expert_ranking_loss(scores[r], batch.expert_indices[i], ranking_margin)
                  for r, i in enumerate(demo_positions) if batch.expert_indices[i]]
        if not losses:
            raise IntentTrainError("ranking warm-up drew a batch with no expert sets")
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        gn = float(sum(p.grad.norm() ** 2 for p in params if p.grad is not None) ** 0.5)
        if not np.isfinite(gn):
            raise IntentTrainError(f"non-finite warm-up gradient norm {gn}")
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, float(grad_clip_norm))
        optimizer.step()
    final = history[-1] if history else {}
    return {"passed": passed, "steps": step, "history": history, "final": final}


# --------------------------------------------------------------------- #
# Order 5W: three-arm pairing.
# --------------------------------------------------------------------- #

FORMAL_ARMS = ("full", "mean", "cv")


def build_formal_plan(seeds: Sequence[int], code_hash: str, config_hash: str,
                      corpus_identity: str, il_episodes: int, online_episodes: int) -> dict:
    """A frozen plan binding every arm to the SAME optimizer seeds.

    Why it is a file and not a convention: the comparison this project
    exists to make is full vs mean vs cv, and it is only valid if the arms
    differ in the belief treatment and nothing else. With a manual launcher
    nothing stops five seeds being run for `full` and one for a baseline --
    which would make `full` look better for a reason that has nothing to do
    with belief. The launcher refuses any (arm, seed) outside this plan.
    """
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise IntentTrainError("formal plan needs at least one seed")
    if len(set(seeds)) != len(seeds):
        raise IntentTrainError(f"formal plan seeds must be unique, got {seeds}")
    diagnostic = sorted(set(seeds) & set(DIAGNOSTIC_OPTIMIZER_SEEDS))
    if diagnostic:
        raise IntentTrainError(
            f"formal plan names DIAGNOSTIC optimizer seed(s) {diagnostic}; those runs exist to decide "
            "how the update is assembled and must never become a formal result")
    return {
        "plan_schema": "bdvl_intent_formal_plan_v1",
        "arms": list(FORMAL_ARMS),
        "seeds": list(seeds),
        "runs": [{"arm": a, "seed": s} for s in seeds for a in FORMAL_ARMS],
        "code_hash": code_hash,
        "config_hash": config_hash,
        "corpus_identity_hash": corpus_identity,
        "il_episodes": int(il_episodes),
        "online_episodes": int(online_episodes),
    }


def assert_in_formal_plan(plan: dict, arm: str, seed: int, *, code_hash: str,
                          config_hash: str, corpus_identity: str) -> None:
    """Fail closed unless this exact (arm, seed) is in the plan AND the code,
    config and corpus still match the ones the plan was frozen against."""
    if plan.get("plan_schema") != "bdvl_intent_formal_plan_v1":
        raise IntentTrainError(f"unknown formal plan schema {plan.get('plan_schema')!r}")
    for field, current in (("code_hash", code_hash), ("config_hash", config_hash),
                           ("corpus_identity_hash", corpus_identity)):
        if plan.get(field) != current:
            raise IntentTrainError(
                f"formal plan was frozen against {field} {str(plan.get(field))[:12]} but this run has "
                f"{str(current)[:12]}; the arms would not be comparable")
    if not any(r["arm"] == arm and int(r["seed"]) == int(seed) for r in plan["runs"]):
        raise IntentTrainError(
            f"(arm={arm!r}, seed={seed}) is not in the frozen formal plan. Arms must be paired on the "
            f"SAME seeds: plan has arms {plan['arms']} x seeds {plan['seeds']}.")
