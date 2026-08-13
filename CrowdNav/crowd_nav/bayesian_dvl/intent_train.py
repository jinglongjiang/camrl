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

from crowd_nav.bayesian_dvl.intent_runtime_config import ActionGridSpec, FROZEN_VALUES
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
    AMBIGUOUS_TRACK_INDEX, JunctionCrowdEpisodeConfig, JunctionEpisodeConfig, build_junction_crowd_episode,
    build_junction_episode, maybe_reveal_crowd_exit, maybe_reveal_exit, public_junction_crowd_scene,
    public_junction_scene,
)
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY
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
            cfg = JunctionCrowdEpisodeConfig(episode_seed=episode_seed, is_heldout=is_heldout)
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

    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
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
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
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
    ranking_margin: float = 0.1, lambda_rank: float = 0.5, ranking_batch_size: Optional[int] = None,
    grad_clip_norm: Optional[float] = None, measure_gradient_ratio: bool = False,
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

    if rank_losses:
        rank_loss = torch.stack(rank_losses).mean()
    else:
        # online-only batch: strictly zero AND gradient-free (not a
        # detached-but-nonzero constant, and not a tensor that would add a
        # spurious node to the graph).
        rank_loss = torch.zeros((), device=device)

    loss = mc_loss + lambda_rank * rank_loss

    # C4R.3: the frozen config declares grad_clip_norm and a rank/MC
    # gradient-ratio gate. Those were previously READ AND VALIDATED but
    # never applied -- contract drift: the config claimed a safeguard the
    # production path did not implement. Both are now real.
    params = [p for p in model.parameters() if p.requires_grad]
    mc_grad_norm = rank_grad_norm = weighted = ratio = cosine = 0.0
    if measure_gradient_ratio:
        # separate autograd passes: the ratio is BETWEEN THE TWO TERMS, and
        # the total gradient norm cannot express that.
        mc_grads = torch.autograd.grad(mc_loss, params, retain_graph=True, allow_unused=True)
        mc_grad_norm = _grad_l2(mc_grads)
        if rank_loss.requires_grad:
            rank_grads = torch.autograd.grad(rank_loss, params, retain_graph=True, allow_unused=True)
            rank_grad_norm = _grad_l2(rank_grads)
            cosine = _grad_cosine(mc_grads, rank_grads)
        weighted = abs(float(lambda_rank)) * rank_grad_norm
        # r_t uses the lambda THIS step ran with -- see AdaptiveRankBalancer
        # on why recomputing it from these same gradients would be circular.
        ratio = weighted / max(mc_grad_norm, 1e-12)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5)
    if not np.isfinite(grad_norm):
        raise IntentTrainError(f"non-finite gradient norm {grad_norm}")
    clipped = False
    if grad_clip_norm is not None:
        if grad_clip_norm <= 0:
            raise IntentTrainError(f"grad_clip_norm must be positive, got {grad_clip_norm}")
        torch.nn.utils.clip_grad_norm_(params, float(grad_clip_norm))
        clipped = grad_norm > float(grad_clip_norm)
    optimizer.step()
    return TrainStepResult(
        loss=float(loss.item()), mc_loss=float(mc_loss.item()), rank_loss=float(rank_loss.item()),
        grad_norm_preclip=grad_norm, clipped=clipped,
        n_demo=int(batch.demo_mask.sum()), n_online=int((~batch.demo_mask).sum()),
        mc_grad_norm=mc_grad_norm, rank_grad_norm=rank_grad_norm,
        weighted_rank_grad_norm=weighted, gradient_ratio=ratio, ratio_measured=measure_gradient_ratio,
        lambda_used=float(lambda_rank), gradient_cosine=cosine,
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


class AdaptiveRankBalancer:
    """Order 2: replaces the frozen ``lambda_rank = 380``.

    Why the fixed value had to go, measured on the lambda=380 run: the
    weighted ranking gradient sat at 80-350x the MC gradient (median 148),
    93.5% of diagnostic points were outside the config's own [0.05, 50]
    band, and the band drifted WORSE over training (107 -> 167 -> 146 ->
    187). 380 had been calibrated ONCE against a randomly initialised
    network and then frozen: as the value head converges ||g_MC|| shrinks
    while ||g_rank|| does not, so a constant lambda must diverge.

    LAGGED by construction, and that is the whole point. The naive fix --
    compute lambda from this batch's gradients so the ratio comes out at
    the target -- makes the health gate TAUTOLOGICAL: r_t would equal the
    target identically, and the only way it could ever fail is lambda
    saturation. Here lambda is decided from PAST gradients (an EMA), used
    for the current update, and the ratio it actually produced is recorded
    afterwards. r_t is therefore an out-of-sample measurement that can
    genuinely fall outside its band.

    target_ratio is 0.25, not 1.0: MC value regression is the objective
    this method is ABOUT; ranking is auxiliary supervision. Parity would
    still let the auxiliary term contribute as much gradient as the thing
    being learned.
    """

    def __init__(self, lambda_init: float = 1.0, target_ratio: float = 0.25, ema_beta: float = 0.9,
                 lambda_min: float = 1e-4, lambda_max: float = 128.0, max_change_factor: float = 2.0,
                 epsilon: float = 1e-12, rank_floor: float = 1e-12):
        if not (0 < lambda_min < lambda_max):
            raise IntentTrainError(f"require 0 < lambda_min < lambda_max, got {lambda_min}/{lambda_max}")
        if not (0.0 < target_ratio):
            raise IntentTrainError(f"target_ratio must be positive, got {target_ratio}")
        if not (0.0 < ema_beta < 1.0):
            raise IntentTrainError(f"ema_beta must be in (0,1), got {ema_beta}")
        if max_change_factor <= 1.0:
            raise IntentTrainError(f"max_change_factor must exceed 1, got {max_change_factor}")
        if not (lambda_min <= lambda_init <= lambda_max):
            raise IntentTrainError(f"lambda_init {lambda_init} outside [{lambda_min}, {lambda_max}]")
        self.target_ratio = float(target_ratio)
        self.ema_beta = float(ema_beta)
        self.lambda_min, self.lambda_max = float(lambda_min), float(lambda_max)
        self.max_change_factor = float(max_change_factor)
        self.epsilon = float(epsilon)
        self.rank_floor = float(rank_floor)
        self.lambda_value = float(lambda_init)
        self.ema_mc: Optional[float] = None
        self.ema_rank: Optional[float] = None
        self.n_observations = 0
        self.n_saturated = 0

    @property
    def saturated(self) -> bool:
        return (self.lambda_value <= self.lambda_min * (1 + 1e-9)
                or self.lambda_value >= self.lambda_max * (1 - 1e-9))

    def observe(self, mc_grad_norm: float, rank_grad_norm: float) -> float:
        """Fold this measurement into the EMAs and set the lambda that the
        NEXT updates will use. Returns the new lambda.

        Called AFTER the optimizer step, never before -- the lag is the
        mechanism that keeps the health gate meaningful."""
        mc, rk = float(mc_grad_norm), float(rank_grad_norm)
        if not (np.isfinite(mc) and np.isfinite(rk)):
            raise IntentTrainError(f"non-finite gradient norms mc={mc} rank={rk}")
        b = self.ema_beta
        self.ema_mc = mc if self.ema_mc is None else b * self.ema_mc + (1 - b) * mc
        self.ema_rank = rk if self.ema_rank is None else b * self.ema_rank + (1 - b) * rk
        self.n_observations += 1

        if self.ema_mc <= self.rank_floor:
            # MC gradient has effectively vanished. Raising lambda here
            # would hand the whole update to the auxiliary term -- a
            # rank-only optimizer wearing a value-learning label. Pin to the
            # floor instead.
            target = self.lambda_min
        elif self.ema_rank <= self.rank_floor:
            # The RANKING gradient has vanished -- the auxiliary objective is
            # satisfied. Dividing by ~0 would drive lambda to its ceiling and
            # then trip the saturation gate, aborting a run for being
            # HEALTHY. There is nothing left to weight up: pin to the floor.
            # (Whether ranking actually learned is judged by the Order 4
            # fixed audit set, not by this gradient magnitude.)
            target = self.lambda_min
        else:
            target = self.target_ratio * self.ema_mc / (self.ema_rank + self.epsilon)

        lo = self.lambda_value / self.max_change_factor
        hi = self.lambda_value * self.max_change_factor
        target = min(max(target, lo), hi)               # no more than x2 / /2 per observation
        self.lambda_value = min(max(target, self.lambda_min), self.lambda_max)
        if self.saturated:
            self.n_saturated += 1
        return self.lambda_value

    def state_dict(self) -> dict:
        return {
            "lambda_value": self.lambda_value, "ema_mc": self.ema_mc, "ema_rank": self.ema_rank,
            "n_observations": self.n_observations, "n_saturated": self.n_saturated,
            "target_ratio": self.target_ratio, "ema_beta": self.ema_beta,
            "lambda_min": self.lambda_min, "lambda_max": self.lambda_max,
            "max_change_factor": self.max_change_factor, "epsilon": self.epsilon,
            "rank_floor": self.rank_floor,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("target_ratio", "ema_beta", "lambda_min", "lambda_max", "max_change_factor",
                    "epsilon", "rank_floor"):
            if key in state and abs(float(state[key]) - float(getattr(self, key))) > 1e-12:
                raise IntentTrainError(
                    f"balancer config drift on {key}: checkpoint {state[key]} != current {getattr(self, key)}")
        self.lambda_value = float(state["lambda_value"])
        self.ema_mc = None if state["ema_mc"] is None else float(state["ema_mc"])
        self.ema_rank = None if state["ema_rank"] is None else float(state["ema_rank"])
        self.n_observations = int(state["n_observations"])
        self.n_saturated = int(state["n_saturated"])


class GradientHealthMonitor:
    """Order 3: replaces the ``GradientRatioMonitor`` "128 consecutive
    diagnostic points" rule, which was structurally incapable of firing.

    Why it could not fire, measured on the lambda=380 run: diagnostics run
    every 50 updates, so 128 CONSECUTIVE points meant tolerating 6400 of
    that run's 12000 updates, and any single in-range point reset the
    counter to zero. 213 of 240 points (89%) were out of range, the longest
    consecutive stretch was 32, and the run emitted no warning at all.

    A sliding-window PROPORTION cannot be reset by an occasional good
    point, which is exactly the failure mode that was missed.

    Every window, counter and statistic here is checkpointed, so a resumed
    run continues the same windows rather than starting a fresh clean slate
    (another way a gate can be silently defeated).
    """

    def __init__(self, ratio_min: float = 0.05, ratio_max: float = 0.75,
                 window: int = 20, ratio_violation_fraction: float = 0.60,
                 clip_window: int = 500, clip_fraction: float = 0.80,
                 saturation_fraction: float = 0.20,
                 cosine_threshold: float = -0.5, cosine_fraction: float = 0.80):
        if not (0 < ratio_min < ratio_max):
            raise IntentTrainError(f"require 0 < ratio_min < ratio_max, got {ratio_min}/{ratio_max}")
        if window <= 0 or clip_window <= 0:
            raise IntentTrainError("windows must be positive")
        self.ratio_min, self.ratio_max = float(ratio_min), float(ratio_max)
        self.window, self.clip_window = int(window), int(clip_window)
        self.ratio_violation_fraction = float(ratio_violation_fraction)
        self.clip_fraction = float(clip_fraction)
        self.saturation_fraction = float(saturation_fraction)
        self.cosine_threshold, self.cosine_fraction = float(cosine_threshold), float(cosine_fraction)
        self.ratios: List[float] = []
        self.saturations: List[bool] = []
        self.cosines: List[float] = []
        self.clips: List[bool] = []
        self.n_measured = 0
        self.n_out_of_range = 0

    def observe_update(self, clipped: bool) -> Optional[str]:
        """EVERY update -- clipping is measured on all of them, not only on
        the sparse diagnostic points."""
        self.clips.append(bool(clipped))
        if len(self.clips) > self.clip_window:
            self.clips = self.clips[-self.clip_window:]
        if len(self.clips) >= self.clip_window:
            frac = sum(self.clips) / len(self.clips)
            if frac > self.clip_fraction:
                return (f"gradient clipping fired on {frac:.1%} of the last {self.clip_window} updates "
                        f"(> {self.clip_fraction:.0%}): the loss scale and grad_clip_norm disagree")
        return None

    def observe_diagnostic(self, ratio: float, lambda_saturated: bool, cosine: float) -> Optional[str]:
        """Diagnostic points only. ``ratio`` MUST be the one produced by the
        lambda this update actually used -- a ratio recomputed from the
        lambda that these same gradients just implied would be tautological."""
        if not np.isfinite(ratio):
            return f"non-finite gradient ratio {ratio}"
        if not np.isfinite(cosine):
            return f"non-finite gradient cosine {cosine}"
        self.n_measured += 1
        if not (self.ratio_min <= ratio <= self.ratio_max):
            self.n_out_of_range += 1
        for buf, val in ((self.ratios, float(ratio)), (self.saturations, bool(lambda_saturated)),
                         (self.cosines, float(cosine))):
            buf.append(val)
            if len(buf) > self.window:
                del buf[:-self.window]

        if len(self.ratios) >= self.window:
            # ABORT ON THE UPPER SIDE ONLY. A ratio BELOW ratio_min means the
            # weighted ranking gradient has become small relative to MC --
            # which is what happens when the auxiliary objective is already
            # satisfied, i.e. a healthy state, not a failure. Aborting on it
            # would kill runs for succeeding. Whether ranking actually
            # learned is judged by the Order 4 fixed audit set
            # (audit_rank_loss), not by a gradient magnitude. ratio_min is
            # still recorded (n_out_of_range) for reporting.
            high = sum(1 for r in self.ratios if r > self.ratio_max) / len(self.ratios)
            if high > self.ratio_violation_fraction:
                return (f"weighted-rank/MC gradient ratio exceeded {self.ratio_max} on {high:.0%} of the "
                        f"last {self.window} diagnostics (> {self.ratio_violation_fraction:.0%}): the "
                        f"auxiliary ranking term is dominating value regression")
            sat = sum(self.saturations) / len(self.saturations)
            if sat > self.saturation_fraction:
                return (f"lambda sat at a bound on {sat:.0%} of the last {self.window} diagnostics "
                        f"(> {self.saturation_fraction:.0%}): the balancer cannot reach its target")
            conflict = sum(1 for c in self.cosines if c < self.cosine_threshold) / len(self.cosines)
            if conflict > self.cosine_fraction:
                return (f"OBJECTIVE_CONFLICT: cos(g_MC, g_rank) < {self.cosine_threshold} on "
                        f"{conflict:.0%} of the last {self.window} diagnostics")
        return None

    #: EVERY threshold, not just the obvious four. A resume that silently
    #: relaxed, say, clip_fraction would produce a "healthy" run under a
    #: gate the author never agreed to.
    _FROZEN_THRESHOLDS = (
        "ratio_min", "ratio_max", "window", "clip_window", "ratio_violation_fraction",
        "clip_fraction", "saturation_fraction", "cosine_threshold", "cosine_fraction",
    )

    def state_dict(self) -> dict:
        state = {
            "ratios": list(self.ratios), "saturations": [bool(x) for x in self.saturations],
            "cosines": list(self.cosines), "clips": [bool(x) for x in self.clips],
            "n_measured": self.n_measured, "n_out_of_range": self.n_out_of_range,
        }
        state.update({k: getattr(self, k) for k in self._FROZEN_THRESHOLDS})
        return state

    def load_state_dict(self, state: dict) -> None:
        missing = [k for k in self._FROZEN_THRESHOLDS if k not in state]
        if missing:
            raise IntentTrainError(f"health monitor state is missing frozen thresholds {missing}")
        for key in self._FROZEN_THRESHOLDS:
            if key in state and float(state[key]) != float(getattr(self, key)):
                raise IntentTrainError(
                    f"health monitor config drift on {key}: checkpoint {state[key]} != {getattr(self, key)}")
        self.ratios = [float(x) for x in state["ratios"]]
        self.saturations = [bool(x) for x in state["saturations"]]
        self.cosines = [float(x) for x in state["cosines"]]
        self.clips = [bool(x) for x in state["clips"]]
        self.n_measured = int(state["n_measured"])
        self.n_out_of_range = int(state["n_out_of_range"])


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

    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
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

    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
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
    gamma: float = 0.95, n_taus: int = 16, ranking_margin: float = 0.1, lambda_rank: float = 0.5,
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
            ranking_margin=ranking_margin, lambda_rank=lambda_rank, ranking_batch_size=ranking_batch_size,
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
    n_taus: int = 16, ranking_margin: float = 0.1, lambda_rank: float = 0.5,
    ranking_batch_size: Optional[int] = None, device: str = "cpu",
    grad_clip_norm: Optional[float] = None, measure_gradient_ratio: bool = False,
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
        ranking_margin=ranking_margin, lambda_rank=lambda_rank, ranking_batch_size=ranking_batch_size,
        grad_clip_norm=grad_clip_norm, measure_gradient_ratio=measure_gradient_ratio,
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
STANDARD_SELECTION_DEV_SEEDS: Tuple[int, ...] = tuple(range(97501, 97601))  # 100
JUNCTION_SELECTION_DEV_SEEDS: Tuple[int, ...] = tuple(range(97601, 97701))  # 100

# C4RF.5: the PAPER-MAIN protocol. To be comparable episode-for-episode
# with Mamba-VL / SARL / LSTM (which are scored through test8.py) the
# goal-intent evaluator must use test8's OWN episode identities, not a
# private seed block. Formula and defaults are taken verbatim from
# crowd_nav/tools/evaluate_bdvl_paper_main.py, which already exists to
# make exactly this comparison bit-identical.
PAPER_MAIN_BASE_SEED = 42
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
    bank = IntentBeliefBank(make_candidate_fn(scene), dt=FROZEN_VALUES["dt"], speed=1.0)
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
def _audit_mc_loss(model, batch: IntentBatch, n_taus: int) -> float:
    device = batch.robot_feats.device
    B = batch.robot_feats.shape[0]
    tau = ((torch.arange(n_taus, dtype=torch.float32, device=device) + 0.5) / n_taus)
    tau = tau.unsqueeze(0).expand(B, n_taus)
    predicted = model(batch.robot_feats, batch.human_feats, batch.human_mask, batch.action_feats, tau)
    return float(quantile_huber_loss(predicted, tau, batch.mc_returns.expand(B, 1)).mean())


@torch.no_grad()
def _audit_rank_loss(model, audit: Sequence[IntentTransition], batch: IntentBatch,
                     n_taus: int, ranking_margin: float) -> float:
    device = batch.robot_feats.device
    n = len(audit)
    n_actions = batch.all_action_feats.shape[1]
    fixed_tau = (torch.arange(n_taus, dtype=torch.float32, device=device) + 0.5) / n_taus
    state_emb = model.encode(batch.robot_feats, batch.human_feats, batch.human_mask)
    state_rep = state_emb.repeat_interleave(n_actions, dim=0)
    action_rep = batch.all_action_feats.reshape(n * n_actions, -1)
    action_emb = model.action_encoder(action_rep)
    tau_rep = fixed_tau.unsqueeze(0).expand(n * n_actions, n_taus)
    scores = model.value_network(state_rep, action_emb, tau_rep).mean(dim=1).view(n, n_actions)
    losses = [expert_ranking_loss(scores[i], audit[i].expert_action_indices, ranking_margin)
              for i in range(n) if audit[i].expert_action_indices]
    return float(torch.stack(losses).mean()) if losses else 0.0


def evaluate_il_audit(model, audit: Sequence[IntentTransition], n_taus: int = 16,
                      ranking_margin: float = 0.1, device: str = "cpu") -> Dict[str, float]:
    """Deterministic: fixed rows, fixed midpoint quantiles, no RNG. Two
    calls on the same weights MUST return identical numbers."""
    was_training = bool(model.training)
    model.eval()
    try:
        batch = batch_to_tensors(audit, device=device)
        return {
            "audit_mc_loss": _audit_mc_loss(model, batch, n_taus),
            "audit_rank_loss": _audit_rank_loss(model, audit, batch, n_taus, ranking_margin),
            "n": float(len(audit)),
        }
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
