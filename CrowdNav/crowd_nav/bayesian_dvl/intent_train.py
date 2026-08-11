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

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from crowd_nav.bayesian_dvl.config import ActionGridSpec, FROZEN_VALUES
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
    mc_grad_norm = rank_grad_norm = weighted = ratio = 0.0
    if measure_gradient_ratio:
        # separate autograd passes: the ratio is BETWEEN THE TWO TERMS, and
        # the total gradient norm cannot express that.
        mc_grad_norm = _grad_l2(torch.autograd.grad(mc_loss, params, retain_graph=True, allow_unused=True))
        if rank_loss.requires_grad:
            rank_grad_norm = _grad_l2(torch.autograd.grad(rank_loss, params, retain_graph=True, allow_unused=True))
        weighted = abs(float(lambda_rank)) * rank_grad_norm
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
    )


def _grad_l2(grads) -> float:
    return float(sum((g ** 2).sum() for g in grads if g is not None) ** 0.5)


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
    return EpisodeCollectionResult(transitions=transitions, outcome=outcome)


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
        return dataclasses.replace(t, all_action_features=None)

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
        state = {
            "online": online,
            "online_action_feature_shape": (
                list(self._online[0].all_action_features.shape) if self._online else None),
            "online_next": self._online_next, "demo_seen": self._demo_seen,
            "demo_capacity": self.demo_capacity, "online_capacity": self.online_capacity,
            "demo_included": bool(include_demo),
        }
        if include_demo:
            state["demo"] = list(self._demo)
        return state

    def load_state_dict(self, state: dict) -> None:
        if (state["demo_capacity"] != self.demo_capacity
                or state["online_capacity"] != self.online_capacity):
            raise IntentTrainError(
                f"replay capacity mismatch: saved demo/online "
                f"{state['demo_capacity']}/{state['online_capacity']} != "
                f"{self.demo_capacity}/{self.online_capacity}")
        if state.get("demo_included", True):
            self._demo = list(state["demo"])
        # else: the caller must have already populated the demo side from
        # the immutable corpus artifact BEFORE loading this state.
        elif not self._demo:
            raise IntentTrainError(
                "checkpoint omits the demo corpus (demo_included=False) but the replay's demo side is "
                "empty -- load the immutable IL corpus artifact first")
        online = list(state["online"])
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
    return result


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
