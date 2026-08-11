"""IL + online RL training primitives (guide.md 6.1/6.2/8.1, A8).

Only the self-contained, unit-testable algorithmic core lives here:
n-step return accumulation, target-network soft update, and a guarded
training step that aborts on NaN/Inf or artifact/config drift rather
than silently continuing (guide.md A8 acceptance). Full episode
collection against a live CrowdSim instance is orchestrated by
``crowd_nav/tools/train_bdvl.py`` (A10+), not implemented here.

R2-3 fix (2026-08-07): ``compute_n_step_returns``/``build_n_step_transitions``/
``train_step`` (below) are the v1 3-step bootstrapped-TD path. They are
kept ONLY for the v1 historical regression tests in selftest.py -- the
independent diagnosis of BDVL's 66/200 common-timeout episodes found
that a 3-step bootstrap only lets the timeout penalty influence the
final 3 steps directly, propagating further back solely through a
slow-moving target network, while decide() simultaneously scores 80
candidate successor states the replay buffer rarely visited (classic
off-policy value overestimation). ``build_mc_return_samples``/
``mc_train_step`` (bottom of this file) are the R2 replacement: full
undiscounted-to-terminal Monte Carlo returns, computed only once a
complete episode is available, with NO bootstrap and NO target network
at all. The R2 CLI (``train_bdvl.py``) must call only the MC path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np

from crowd_nav.bayesian_dvl import normalization as norm
from crowd_nav.bayesian_dvl.belief import BeliefTracker
from crowd_nav.bayesian_dvl.config import NORMALIZATION_CONSTANTS
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork, expert_ranking_loss, quantile_huber_loss
from crowd_nav.bayesian_dvl.policy import (
    ReplayIntegrityError, _score_all_candidates, compute_executed_action_quantile_target,
    validate_replay_sample_integrity,
)
from crowd_nav.bayesian_dvl.ranking import RankingDemoSample
from crowd_nav.bayesian_dvl.replay import MCReturnSample, Transition
from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, SetEncoder


def _partial_action_features_from_table(action_table: np.ndarray, action_indices: Sequence[int]) -> np.ndarray:
    """R4-1 MECHANICAL interim builder (2026-08-10), NOT the full
    ``policy.compute_action_features_array``: MC-loss training samples
    (``Transition``/``MCReturnSample``) only store pre-flattened
    ``robot_features`` arrays, not a live ``RobotObservation``, so
    ``goal_alignment``/``turn_cost`` (which need the robot's current
    goal/heading) cannot be reconstructed here -- filled with 0.0.
    ``vx``/``vy``/``speed`` ARE recoverable from ``action_index`` alone
    and are computed identically to the real builder. guide.md R4-2
    ("Replay 明确保存 action_index/action_features") is where this gets
    replaced by a genuinely stored, full-fidelity per-sample
    ActionFeature -- do not treat this as the final MC-loss design.
    """
    rows = action_table[np.asarray(action_indices, dtype=np.int64)]
    norm_vx, norm_vy = norm.normalize_robot_velocity_array(rows[:, 0], rows[:, 1])
    speed = np.hypot(rows[:, 0], rows[:, 1])
    norm_speed = norm.normalize_speed_array(speed, NORMALIZATION_CONSTANTS["robot_max_speed"])
    zeros = np.zeros(rows.shape[0], dtype=np.float64)
    return np.stack([norm_vx, norm_vy, norm_speed, zeros, zeros], axis=1).astype(np.float32)


class TrainerError(RuntimeError):
    pass


def soft_update_target(online: nn.Module, target: nn.Module, tau: float) -> None:
    """target = tau*online + (1-tau)*target, in place. guide.md 11's
    frozen ``target_tau=0.005``."""
    if not (0.0 < tau <= 1.0):
        raise TrainerError(f"tau must be in (0,1], got {tau}")
    with torch.no_grad():
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)


def hard_copy_to_target(online: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


def compute_n_step_returns(
    rewards: Sequence[float], dones: Sequence[bool], gamma: float, n_step: int,
) -> List[Tuple[float, int]]:
    """For each starting index t in a single episode's reward sequence,
    return (n_step_return, actual_horizon) where actual_horizon <
    n_step if the episode terminates before n_step lookahead steps
    (the terminal reward is then the last term, with no further
    bootstrap needed beyond it -- guide.md 6.2: "同时保留terminal
    collision/success/timeout transition" must not be lost).
    n_step_return = sum_{k=0}^{h-1} gamma^k * r_{t+k} (bootstrap term
    added separately by the caller using the state h steps ahead)."""
    T = len(rewards)
    if len(dones) != T:
        raise TrainerError("rewards and dones must be the same length")
    results = []
    for t in range(T):
        acc = 0.0
        horizon = 0
        for k in range(n_step):
            if t + k >= T:
                break
            acc += (gamma ** k) * rewards[t + k]
            horizon = k + 1
            if dones[t + k]:
                break
        results.append((acc, horizon))
    return results


def build_n_step_transitions(
    states: Sequence[Mapping[str, object]],
    actions: Sequence[int],
    rewards: Sequence[float],
    dones: Sequence[bool],
    gamma: float,
    n_step: int,
    artifact_sha256: str,
    episode_seed: int,
) -> List[Transition]:
    """Build typed n-step transitions from one episode.

    ``states`` must contain T+1 snapshots: the first T are the states at
    which actions were executed and the final snapshot is the successor
    needed by a non-terminal n-step transition. A terminal flag is the
    OR over the actual n-step window, never merely ``dones[t]``.
    """
    T = len(rewards)
    if len(actions) != T or len(dones) != T or len(states) != T + 1:
        raise TrainerError("episode transition lengths must be actions/rewards/dones=T and states=T+1")
    if not (0.0 < gamma <= 1.0) or n_step <= 0:
        raise TrainerError("invalid gamma or n_step")
    returns = compute_n_step_returns(rewards, dones, gamma, n_step)
    output: List[Transition] = []
    for t, ((return_n, horizon), _) in enumerate(zip(returns, dones)):
        end = min(T, t + horizon)
        terminal = any(bool(flag) for flag in dones[t:end])
        next_index = t + horizon
        if next_index >= len(states):
            raise TrainerError(f"missing successor state for t={t}, horizon={horizon}")
        state = states[t]
        next_state = states[next_index]
        output.append(Transition(
            robot_features=np.asarray(state["robot_features"], dtype=np.float32),
            human_features=np.asarray(state["human_features"], dtype=np.float32),
            human_mask=np.asarray(state["human_mask"], dtype=bool),
            belief=np.asarray(state["belief"], dtype=np.float32),
            action_index=int(actions[t]),
            reward=float(return_n),
            done=terminal,
            next_robot_features=np.asarray(next_state["robot_features"], dtype=np.float32),
            next_human_features=np.asarray(next_state["human_features"], dtype=np.float32),
            next_human_mask=np.asarray(next_state["human_mask"], dtype=bool),
            next_belief=np.asarray(next_state["belief"], dtype=np.float32),
            artifact_sha256=artifact_sha256,
            gamma_pow_n=float(gamma ** horizon),
            episode_seed=int(episode_seed),
            step_index=t,
        ))
    return output


@dataclass
class TrainStepResult:
    loss: float
    n_demo: int
    n_online: int
    aborted: bool
    abort_reason: str = ""
    mc_grad_norm: float = 0.0
    rank_grad_norm: float = 0.0
    weighted_rank_grad_norm: float = 0.0
    gradient_ratio: float = 0.0


def check_for_nan_inf(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if not torch.isfinite(t).all():
            raise TrainerError("non-finite value encountered (NaN/Inf) -- aborting per guide.md A8")


def _gradient_l2(grads) -> float:
    """Return one comparable L2 norm across the shared encoder/IQN params."""
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(torch.sum(grad.detach() * grad.detach()).cpu())
    return float(np.sqrt(total))


def train_step(
    encoder: SetEncoder,
    value_network: IQNValueNetwork,
    action_encoder: ActionEncoder,
    target_encoder: SetEncoder,
    target_value_network: IQNValueNetwork,
    target_action_encoder: ActionEncoder,
    action_table: np.ndarray,
    action_indices: Sequence[int],
    robot_features: torch.Tensor,
    human_features: torch.Tensor,
    human_mask: torch.Tensor,
    rewards: torch.Tensor,  # [B] n-step accumulated reward
    dones: torch.Tensor,  # [B] bool
    next_robot_features: torch.Tensor,
    next_human_features: torch.Tensor,
    next_human_mask: torch.Tensor,
    gamma_pow_n: torch.Tensor,  # [B] gamma**horizon, per-sample (horizon may vary near episode end)
    n_train_quantiles: int,
    n_target_samples: int,
    optimizer: torch.optim.Optimizer,
) -> TrainStepResult:
    """One distributional-TD gradient step:
        y = r_nstep + gamma^h * (1-done) * Z_target(s', tau')
    guide.md 6.2. Aborts cleanly (no optimizer.step()) on non-finite
    values anywhere in the pipeline.

    R4-1 MECHANICAL fix (2026-08-10): this is the RETIRED v1 3-step
    bootstrap path (see module docstring -- kept only for v1 historical
    regression tests, the real R2+ path is ``mc_train_step``/
    ``stage1_train_step``/``stage2_train_step``). ``IQNValueNetwork`` now
    requires an explicit action embedding; a proper Q-learning target
    would need ``max_a' Q(s',b',a')`` (re-scoring all 80 actions at the
    successor state), which this retired path never did even under V(s')
    and is out of scope for a mechanical signature fix. As a deliberate
    simplification confined to this historical-regression-only function,
    the SAME executed ``action_indices`` are reused for both the current
    state's prediction and the next state's target (i.e. "what if the
    same action were repeated"), via the same crude
    ``_partial_action_features_from_table`` builder ``mc_train_step``
    uses -- this is NOT a claim that repeating the action is the correct
    bootstrap target, only that the n-step-return/NaN-abort arithmetic
    this function's tests actually check does not depend on which action
    embedding is fed in.
    """
    try:
        check_for_nan_inf(robot_features, human_features, rewards.float(), next_robot_features, next_human_features)
    except TrainerError as exc:
        return TrainStepResult(loss=float("nan"), n_demo=0, n_online=0, aborted=True, abort_reason=str(exc))

    batch_size = robot_features.shape[0]
    # R1 fix (independent audit B3, 2026-08-06): tau (and every other
    # per-batch tensor built inside this function) MUST be created on
    # the same device as the networks -- reproduced independently: a
    # CUDA train_step call crashed immediately because torch.rand()
    # with no device= defaults to CPU while target_encoder/
    # target_value_network live on CUDA. Derive the device from the
    # network's own parameters (not from an input tensor, which the
    # caller might hand in on the wrong device too) and move every
    # tensor built or received here onto it explicitly.
    device = next(target_encoder.parameters()).device
    robot_features = robot_features.to(device)
    human_features = human_features.to(device)
    human_mask = human_mask.to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)
    next_robot_features = next_robot_features.to(device)
    next_human_features = next_human_features.to(device)
    next_human_mask = next_human_mask.to(device)
    gamma_pow_n = gamma_pow_n.to(device)
    action_features_t = torch.tensor(
        _partial_action_features_from_table(action_table, action_indices), dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        next_state_embedding = target_encoder(next_robot_features, next_human_features, next_human_mask)
        next_action_embedding = target_action_encoder(action_features_t)
        target_tau = torch.rand(batch_size, n_target_samples, device=device)
        target_quantiles = target_value_network(next_state_embedding, next_action_embedding, target_tau)
        not_done = (~dones).float().unsqueeze(1)
        target_samples = rewards.unsqueeze(1) + gamma_pow_n.unsqueeze(1) * not_done * target_quantiles

    if not torch.isfinite(target_samples).all():
        return TrainStepResult(loss=float("nan"), n_demo=0, n_online=0, aborted=True, abort_reason="non-finite TD target")

    state_embedding = encoder(robot_features, human_features, human_mask)
    action_embedding = action_encoder(action_features_t)
    train_tau = torch.rand(batch_size, n_train_quantiles, device=device)
    predicted_quantiles = value_network(state_embedding, action_embedding, train_tau)

    loss = quantile_huber_loss(predicted_quantiles, train_tau, target_samples)
    if not torch.isfinite(loss):
        return TrainStepResult(loss=float("nan"), n_demo=0, n_online=0, aborted=True, abort_reason="non-finite loss")

    optimizer.zero_grad()
    loss.backward()
    for param in list(encoder.parameters()) + list(value_network.parameters()):
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return TrainStepResult(loss=float(loss.detach()), n_demo=0, n_online=0, aborted=True, abort_reason="non-finite gradient")
    optimizer.step()

    return TrainStepResult(loss=float(loss.detach()), n_demo=0, n_online=0, aborted=False)


def compute_mc_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    """``G_t = sum_k gamma^k * r_{t+k}`` for every t in ONE complete
    episode, computed backward in a single pass. The one place this
    formula is written -- ``build_mc_return_samples`` (below) and
    ``train_bdvl.py``'s ranking-IL sample builder both call this rather
    than each re-deriving the backward accumulation."""
    T = len(rewards)
    if not (0.0 < gamma <= 1.0):
        raise TrainerError(f"invalid gamma: {gamma}")
    if T == 0:
        raise TrainerError("cannot compute MC returns from an empty episode")
    returns = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns


def build_mc_return_samples(
    states: Sequence[Mapping[str, object]],
    actions: Sequence[int],
    rewards: Sequence[float],
    gamma: float,
    artifact_sha256: str,
    suite_seed: int,
    episode_seed: int,
    outcome: str = "",
) -> List[MCReturnSample]:
    """Full-trajectory Monte Carlo return for every step of ONE COMPLETE
    episode (guide.md R2-3): ``G_t = sum_k gamma^k * r_{t+k}`` from t to
    the episode's actual end, computed backward in a single pass.

    Unlike ``compute_n_step_returns``, this never truncates at a fixed
    horizon and needs no bootstrap term or successor state: the
    timeout/collision/success reward at the final step is directly,
    fully visible (after gamma-discounting) from every earlier state in
    the same episode. ``states``/``actions``/``rewards`` must all have
    the SAME length T (one entry per action taken) -- no extra
    successor snapshot is needed, unlike the n-step path.

    R4-2 fix (2026-08-10): ``states[t]`` is now the raw dict
    ``train_bdvl._capture_state`` builds (``robot``/``humans``/
    ``belief_tracker_snapshot``/``global_time``/``executed_action_features``/
    ``posterior_seed_key``), not a pre-flattened feature array -- see
    ``MCReturnSample``'s own docstring for why.

    R4-2R-2 fix (2026-08-10, guide.md "R4-2R-2 -- 保存真实posterior
    seed"): ``posterior_seed_key`` is COPIED VERBATIM from
    ``state["posterior_seed_key"]`` -- the REAL
    ``BDVLPolicy.next_seed_key()``/``last_seed_key`` value captured live
    at collection time (see ``train_bdvl._collect_online_episode``), no
    longer reconstructed here as ``(suite_seed, episode_seed, t+1)``,
    which silently drifted from the real decision counter once any
    exploration step was interleaved with greedy decisions. ``suite_seed``
    is kept as a parameter purely to fail closed if a stored key's own
    suite-seed component doesn't match what the caller believes it
    collected under -- a real cross-check, not a construction input.
    """
    T = len(rewards)
    if len(actions) != T or len(states) != T:
        raise TrainerError("build_mc_return_samples requires states/actions/rewards of equal length T (one per step)")
    returns = compute_mc_returns(rewards, gamma)

    output: List[MCReturnSample] = []
    for t in range(T):
        state = states[t]
        seed_key = tuple(int(x) for x in state["posterior_seed_key"])
        if seed_key[0] != int(suite_seed) or seed_key[1] != int(episode_seed):
            raise TrainerError(
                f"step {t}: stored posterior_seed_key {seed_key} does not match "
                f"suite_seed={suite_seed}/episode_seed={episode_seed}"
            )
        output.append(MCReturnSample(
            robot=state["robot"],
            humans=tuple(state["humans"]),
            global_time=float(state["global_time"]),
            belief_tracker_snapshot=state["belief_tracker_snapshot"],
            executed_action_index=int(actions[t]),
            executed_action_features=np.asarray(state["executed_action_features"], dtype=np.float32),
            posterior_seed_key=seed_key,
            target_return=float(returns[t]),
            artifact_sha256=artifact_sha256,
            episode_seed=int(episode_seed),
            step_index=t,
            outcome=outcome,
            source_role="online",
        ))
    return output


def assert_returns_within_bounds(returns: Sequence[float], v_min: float, v_max: float, tolerance: float = 1e-6) -> None:
    """guide.md R2-2: "训练target超界立即报错" -- a target outside the
    analytically-derived achievable range means the bound derivation or
    one of the reward/episode-length assumptions it rests on has
    drifted, and must be investigated. Never silently clip: clipping
    would hide exactly the kind of bug this check exists to catch."""
    if not returns:
        return
    arr = np.asarray(returns, dtype=np.float64)
    if arr.min() < v_min - tolerance or arr.max() > v_max + tolerance:
        raise TrainerError(
            f"MC return target out of derived bounds [{v_min}, {v_max}]: "
            f"observed min={arr.min()}, max={arr.max()}"
        )


def mc_train_step(
    encoder: SetEncoder,
    value_network: IQNValueNetwork,
    action_encoder: ActionEncoder,
    action_table: np.ndarray,
    action_indices: Sequence[int],
    robot_features: torch.Tensor,
    human_features: torch.Tensor,
    human_mask: torch.Tensor,
    target_returns: torch.Tensor,  # [B] full realized Monte Carlo return, one scalar target sample per row
    n_train_quantiles: int,
    optimizer: torch.optim.Optimizer,
) -> TrainStepResult:
    """One distributional regression step against REALIZED full-episode
    Monte Carlo returns (guide.md R2-3). No target network, no
    bootstrap, no next-state forward pass: ``target_returns`` are
    already complete numbers from ``build_mc_return_samples``, so the
    quantile Huber loss regresses the CURRENT state's predicted quantile
    function directly onto a single realized-return sample per row
    (K2=1), exactly as the original IL update did for demonstrations.

    NOT the real train_bdvl.py path (that is ``stage1_train_step``/
    ``stage2_train_step``, which fold this same MC regression together
    with the expert-ranking loss) -- kept for its own selftest coverage.
    R4-1 MECHANICAL fix (2026-08-10): see
    ``_partial_action_features_from_table``'s docstring -- ``action_index``
    IS stored on every ``MCReturnSample``, but not the goal/heading
    context needed for a full ``ActionFeature``, so only the crude
    (vx, vy, speed) subset is used here pending R4-2's replay schema.
    """
    try:
        check_for_nan_inf(robot_features, human_features, target_returns.float())
    except TrainerError as exc:
        return TrainStepResult(loss=float("nan"), n_demo=0, n_online=0, aborted=True, abort_reason=str(exc))

    device = next(encoder.parameters()).device
    robot_features = robot_features.to(device)
    human_features = human_features.to(device)
    human_mask = human_mask.to(device)
    target_returns = target_returns.to(device)
    action_features_t = torch.tensor(
        _partial_action_features_from_table(action_table, action_indices), dtype=torch.float32, device=device,
    )

    try:
        assert_returns_within_bounds(target_returns.detach().cpu().numpy().tolist(), value_network.v_min, value_network.v_max)
    except TrainerError as exc:
        return TrainStepResult(loss=float("nan"), n_demo=0, n_online=0, aborted=True, abort_reason=str(exc))

    batch_size = robot_features.shape[0]
    state_embedding = encoder(robot_features, human_features, human_mask)
    action_embedding = action_encoder(action_features_t)
    train_tau = torch.rand(batch_size, n_train_quantiles, device=device)
    predicted_quantiles = value_network(state_embedding, action_embedding, train_tau)

    loss = quantile_huber_loss(predicted_quantiles, train_tau, target_returns.unsqueeze(1))
    if not torch.isfinite(loss):
        return TrainStepResult(loss=float("nan"), n_demo=0, n_online=0, aborted=True, abort_reason="non-finite loss")

    optimizer.zero_grad()
    loss.backward()
    for param in list(encoder.parameters()) + list(value_network.parameters()) + list(action_encoder.parameters()):
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return TrainStepResult(loss=float(loss.detach()), n_demo=0, n_online=0, aborted=True, abort_reason="non-finite gradient")
    optimizer.step()

    return TrainStepResult(loss=float(loss.detach()), n_demo=0, n_online=0, aborted=False)


@dataclass
class Stage1TrainStepResult:
    loss: float
    mc_loss: float
    rank_loss: float
    aborted: bool
    abort_reason: str = ""
    mc_grad_norm: float = 0.0
    rank_grad_norm: float = 0.0
    weighted_rank_grad_norm: float = 0.0
    gradient_ratio: float = 0.0


def stage1_train_step(
    encoder: SetEncoder,
    value_network: IQNValueNetwork,
    action_encoder: ActionEncoder,
    artifact,
    action_table: Sequence[Tuple[float, float]],
    reward_config,
    dt: float,
    time_limit: float,
    max_human_speed: float,
    n_world_samples: int,
    n_iqn_quantiles: int,
    posterior_source: str,
    device,
    gamma: float,
    demo_batch: Sequence[RankingDemoSample],
    ranking_margin: float,
    lambda_rank: float,
    n_train_quantiles: int,
    optimizer: torch.optim.Optimizer,
    ranking_batch_size: Optional[int] = None,
) -> Stage1TrainStepResult:
    """R3-3/R4-2's real Stage 1 update:
        L_stage1 = L_executed_MC + lambda_rank * L_expert_rank

    R4-2 fix (2026-08-10, guide.md "R4-1R-4 -- 冻结 R4-2 的精确数学目标"):
    ``L_executed_MC`` no longer regresses a bare encoder(state) forward
    pass onto ``target_return`` -- it reuses the SAME candidate builder
    deployment uses (``_score_all_candidates``, restricted to the one
    ``sample.executed_action_index`` via a length-1 action table), then
    ``policy.compute_executed_action_quantile_target`` aggregates over
    the shared posterior worlds BEFORE comparing to the single realized
    ``G_t`` (never treats a hypothetical world as though it independently
    realized the one true future). This means every demo sample now gets
    a real (cheap, n_actions=1) world-sampling pass for L_executed_MC, in
    addition to the existing (capped, n_actions=80) pass for L_expert_rank
    -- a real, intentional compute increase, not an oversight.

    ``L_expert_rank`` is computed via ``_score_all_candidates`` -- THE
    SAME production scoring path ``BDVLPolicy.decide()`` uses -- so the
    ranking supervision is applied to the actual deployment decision
    surface, not an approximate stand-in. ``executed_action_index``
    (MC-loss) and ``expert_action_indices`` (ranking) are deliberately
    separate fields on ``RankingDemoSample`` and must never be derived
    from each other (guide.md R4-2).

    R3R-2 fix (2026-08-07): ``_score_all_candidates`` re-scores 80
    actions x ``n_world_samples`` posterior draws PER demo sample --
    running it for the whole demo share of a 256-item batch was never
    validated for compute/memory (guide.md finding C5). ``ranking_batch_
    size`` (default: ``config.FROZEN_VALUES["ranking_batch_size"]``=16)
    caps how many of ``demo_batch`` get the expensive 80-action ranking
    pass; L_executed_MC still covers the FULL demo_batch (n_actions=1
    per sample, much cheaper than the ranking pass).

    Every ``RankingDemoSample`` carries its own belief-tracker snapshot
    (guide.md R3-3: reconstructing a live ``BeliefTracker`` from it is
    cheap and exact, unlike re-running the whole episode's filtering
    history).
    """
    if not demo_batch:
        raise TrainerError("stage1_train_step requires at least one demo sample")
    if ranking_batch_size is None:
        ranking_batch_size = len(demo_batch)
    if ranking_batch_size <= 0:
        raise TrainerError(f"ranking_batch_size must be positive, got {ranking_batch_size}")

    artifact_sha256 = artifact.content_sha256()
    try:
        for sample in demo_batch:
            validate_replay_sample_integrity(sample, action_table, artifact_sha256, expected_source_role="demo")
    except ReplayIntegrityError as exc:
        return Stage1TrainStepResult(loss=float("nan"), mc_loss=float("nan"), rank_loss=float("nan"), aborted=True, abort_reason=str(exc))

    target_returns: List[float] = []
    z_bar_rows: List[torch.Tensor] = []
    tau_rows: List[torch.Tensor] = []
    rank_losses: List[torch.Tensor] = []

    for index, sample in enumerate(demo_batch):
        tracker = BeliefTracker(artifact)
        tracker._tracks = copy.deepcopy(sample.belief_tracker_snapshot)
        target_returns.append(float(sample.target_return))

        _, _, rf, hf, mk, af, rw, nt, _, _ = _score_all_candidates(
            tracker=tracker, robot=sample.robot, humans=sample.humans, global_time=sample.global_time,
            artifact=artifact, action_table=(action_table[sample.executed_action_index],),
            action_indices=(sample.executed_action_index,),
            reward_config=reward_config, dt=dt, time_limit=time_limit, max_human_speed=max_human_speed,
            n_world_samples=n_world_samples, n_iqn_quantiles=n_train_quantiles, tau_upper=1.0,
            posterior_source=posterior_source, set_encoder=encoder, value_network=value_network,
            action_encoder=action_encoder, device=device, seed_key=sample.posterior_seed_key, gamma=gamma,
        )
        z_bar, tau_used = compute_executed_action_quantile_target(
            encoder, value_network, action_encoder, rf, hf, mk, af, rw, nt,
            n_quantiles=n_train_quantiles, tau_upper=1.0, seed_key=sample.posterior_seed_key,
            executed_action_index=sample.executed_action_index, gamma=gamma, device=device,
        )
        z_bar_rows.append(z_bar)
        tau_rows.append(tau_used)

        if index >= ranking_batch_size:
            continue  # L_executed_MC still covers this sample above; only L_expert_rank is capped

        # IL has no live decision_counter (ORCA never calls decide());
        # step_index+1 plays the same role -- a strictly increasing,
        # reproducible per-decision counter within the episode. Reuses
        # sample.posterior_seed_key rather than re-deriving it, since
        # that field IS exactly this value (stored at collection time).
        action_scores, *_ = _score_all_candidates(
            tracker=tracker, robot=sample.robot, humans=sample.humans, global_time=sample.global_time,
            artifact=artifact, action_table=action_table, reward_config=reward_config,
            dt=dt, time_limit=time_limit, max_human_speed=max_human_speed,
            n_world_samples=n_world_samples, n_iqn_quantiles=n_iqn_quantiles,
            tau_upper=1.0,  # ranking supervises the full return distribution's mean, not a CVaR tail
            posterior_source=posterior_source,
            set_encoder=encoder, value_network=value_network, action_encoder=action_encoder, device=device,
            seed_key=sample.posterior_seed_key, gamma=gamma,
        )
        rank_losses.append(expert_ranking_loss(action_scores, sample.expert_action_indices, ranking_margin))

    try:
        assert_returns_within_bounds(target_returns, value_network.v_min, value_network.v_max)
    except TrainerError as exc:
        return Stage1TrainStepResult(loss=float("nan"), mc_loss=float("nan"), rank_loss=float("nan"), aborted=True, abort_reason=str(exc))

    target_returns_t = torch.tensor(target_returns, dtype=torch.float32, device=device)
    z_bar_batch = torch.stack(z_bar_rows)  # [B, K]
    tau_batch = torch.stack(tau_rows)  # [B, K]

    try:
        check_for_nan_inf(z_bar_batch, target_returns_t)
    except TrainerError as exc:
        return Stage1TrainStepResult(loss=float("nan"), mc_loss=float("nan"), rank_loss=float("nan"), aborted=True, abort_reason=str(exc))

    mc_loss = quantile_huber_loss(z_bar_batch, tau_batch, target_returns_t.unsqueeze(1))

    rank_loss = torch.stack(rank_losses).mean()
    total_loss = mc_loss + lambda_rank * rank_loss

    if not torch.isfinite(total_loss):
        return Stage1TrainStepResult(loss=float("nan"), mc_loss=float("nan"), rank_loss=float("nan"), aborted=True, abort_reason="non-finite loss")

    optimizer.zero_grad()
    params = list(encoder.parameters()) + list(value_network.parameters()) + list(action_encoder.parameters())
    mc_grads = torch.autograd.grad(mc_loss, params, retain_graph=True, allow_unused=True)
    rank_grads = (
        torch.autograd.grad(rank_loss, params, retain_graph=True, allow_unused=True)
        if rank_loss.requires_grad else [None] * len(params)
    )
    mc_grad_norm = _gradient_l2(mc_grads)
    rank_grad_norm = _gradient_l2(rank_grads)
    weighted_rank_grad_norm = abs(float(lambda_rank)) * rank_grad_norm
    gradient_ratio = weighted_rank_grad_norm / max(mc_grad_norm, 1e-12)
    total_loss.backward()
    for param in params:
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return Stage1TrainStepResult(
                loss=float(total_loss.detach()), mc_loss=float(mc_loss.detach()), rank_loss=float(rank_loss.detach()),
                aborted=True, abort_reason="non-finite gradient",
            )
    optimizer.step()

    return Stage1TrainStepResult(
        loss=float(total_loss.detach()), mc_loss=float(mc_loss.detach()), rank_loss=float(rank_loss.detach()),
        aborted=False, mc_grad_norm=mc_grad_norm, rank_grad_norm=rank_grad_norm,
        weighted_rank_grad_norm=weighted_rank_grad_norm, gradient_ratio=gradient_ratio,
    )


@dataclass
class Stage2TrainStepResult:
    loss: float
    mc_loss: float
    rank_loss: float
    grad_norm: float
    n_demo: int
    n_online: int
    aborted: bool
    abort_reason: str = ""
    mc_grad_norm: float = 0.0
    rank_grad_norm: float = 0.0
    weighted_rank_grad_norm: float = 0.0
    gradient_ratio: float = 0.0


def stage2_train_step(
    encoder: SetEncoder,
    value_network: IQNValueNetwork,
    action_encoder: ActionEncoder,
    artifact,
    action_table: Sequence[Tuple[float, float]],
    reward_config,
    dt: float,
    time_limit: float,
    max_human_speed: float,
    n_world_samples: int,
    n_iqn_quantiles: int,
    posterior_source: str,
    device,
    gamma: float,
    demo_batch: Sequence[RankingDemoSample],
    online_batch: Sequence[MCReturnSample],
    ranking_margin: float,
    lambda_rank: float,
    n_train_quantiles: int,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
    ranking_batch_size: Optional[int] = None,
) -> Stage2TrainStepResult:
    """R3-4/R4-2's online-RL update: demo-drawn samples (from the fixed
    20% reservoir share) get BOTH L_executed_MC and L_expert_rank so
    action knowledge from ORCA is never overwritten/forgotten purely by
    online exploration; online-drawn samples (``MCReturnSample``, no
    stored action-equivalence set) get L_executed_MC only. One combined
    optimizer step per call, with a fixed global gradient-norm clip
    (guide.md R3-4 point 3) -- the clip is applied AFTER computing (and
    returning) the pre-clip norm, so callers can log/diagnose runaway
    gradients rather than have them silently capped without a trace.

    R4-2 fix (2026-08-10, guide.md "R4-1R-4"): L_executed_MC for BOTH
    demo and online rows now goes through the SAME
    ``_score_all_candidates`` (restricted to the row's own
    ``executed_action_index``) + ``compute_executed_action_quantile_target``
    path ``stage1_train_step`` uses -- see that function's docstring for
    the full rationale. ``RankingDemoSample`` and ``MCReturnSample`` both
    carry the fields this needs (``robot``/``humans``/
    ``belief_tracker_snapshot``/``global_time``/``executed_action_index``/
    ``posterior_seed_key``) since R4-2, so both are handled by the same
    loop body below.

    R3R-2 fix: ``ranking_batch_size`` caps how many of ``demo_batch``
    get the expensive 80-action ``_score_all_candidates`` ranking pass
    (see ``stage1_train_step``'s docstring for the same fix);
    L_executed_MC still covers every demo AND online sample.
    """
    if not demo_batch and not online_batch:
        raise TrainerError("stage2_train_step requires at least one demo or online sample")
    if ranking_batch_size is None:
        ranking_batch_size = len(demo_batch)
    if ranking_batch_size < 0:
        raise TrainerError(f"ranking_batch_size must be non-negative, got {ranking_batch_size}")

    artifact_sha256 = artifact.content_sha256()
    try:
        for sample in demo_batch:
            validate_replay_sample_integrity(sample, action_table, artifact_sha256, expected_source_role="demo")
        for sample in online_batch:
            validate_replay_sample_integrity(sample, action_table, artifact_sha256, expected_source_role="online")
    except ReplayIntegrityError as exc:
        return Stage2TrainStepResult(
            loss=float("nan"), mc_loss=float("nan"), rank_loss=float("nan"), grad_norm=float("nan"),
            n_demo=len(demo_batch), n_online=len(online_batch), aborted=True, abort_reason=str(exc),
        )

    target_returns: List[float] = []
    z_bar_rows: List[torch.Tensor] = []
    tau_rows: List[torch.Tensor] = []
    rank_losses: List[torch.Tensor] = []

    def _executed_mc_target(tracker, sample) -> None:
        _, _, rf, hf, mk, af, rw, nt, _, _ = _score_all_candidates(
            tracker=tracker, robot=sample.robot, humans=sample.humans, global_time=sample.global_time,
            artifact=artifact, action_table=(action_table[sample.executed_action_index],),
            action_indices=(sample.executed_action_index,),
            reward_config=reward_config, dt=dt, time_limit=time_limit, max_human_speed=max_human_speed,
            n_world_samples=n_world_samples, n_iqn_quantiles=n_train_quantiles, tau_upper=1.0,
            posterior_source=posterior_source, set_encoder=encoder, value_network=value_network,
            action_encoder=action_encoder, device=device, seed_key=sample.posterior_seed_key, gamma=gamma,
        )
        z_bar, tau_used = compute_executed_action_quantile_target(
            encoder, value_network, action_encoder, rf, hf, mk, af, rw, nt,
            n_quantiles=n_train_quantiles, tau_upper=1.0, seed_key=sample.posterior_seed_key,
            executed_action_index=sample.executed_action_index, gamma=gamma, device=device,
        )
        z_bar_rows.append(z_bar)
        tau_rows.append(tau_used)
        target_returns.append(float(sample.target_return))

    for index, sample in enumerate(demo_batch):
        tracker = BeliefTracker(artifact)
        tracker._tracks = copy.deepcopy(sample.belief_tracker_snapshot)
        _executed_mc_target(tracker, sample)

        if index >= ranking_batch_size:
            continue

        action_scores, *_ = _score_all_candidates(
            tracker=tracker, robot=sample.robot, humans=sample.humans, global_time=sample.global_time,
            artifact=artifact, action_table=action_table, reward_config=reward_config,
            dt=dt, time_limit=time_limit, max_human_speed=max_human_speed,
            n_world_samples=n_world_samples, n_iqn_quantiles=n_iqn_quantiles,
            tau_upper=1.0, posterior_source=posterior_source,
            set_encoder=encoder, value_network=value_network, action_encoder=action_encoder, device=device,
            seed_key=sample.posterior_seed_key, gamma=gamma,
        )
        rank_losses.append(expert_ranking_loss(action_scores, sample.expert_action_indices, ranking_margin))

    for sample in online_batch:
        tracker = BeliefTracker(artifact)
        tracker._tracks = copy.deepcopy(sample.belief_tracker_snapshot)
        _executed_mc_target(tracker, sample)

    try:
        assert_returns_within_bounds(target_returns, value_network.v_min, value_network.v_max)
    except TrainerError as exc:
        return Stage2TrainStepResult(
            loss=float("nan"), mc_loss=float("nan"), rank_loss=float("nan"), grad_norm=float("nan"),
            n_demo=len(demo_batch), n_online=len(online_batch), aborted=True, abort_reason=str(exc),
        )

    target_returns_t = torch.tensor(target_returns, dtype=torch.float32, device=device)
    z_bar_batch = torch.stack(z_bar_rows)  # [B, K]
    tau_batch = torch.stack(tau_rows)  # [B, K]

    try:
        check_for_nan_inf(z_bar_batch, target_returns_t)
    except TrainerError as exc:
        return Stage2TrainStepResult(
            loss=float("nan"), mc_loss=float("nan"), rank_loss=float("nan"), grad_norm=float("nan"),
            n_demo=len(demo_batch), n_online=len(online_batch), aborted=True, abort_reason=str(exc),
        )

    mc_loss = quantile_huber_loss(z_bar_batch, tau_batch, target_returns_t.unsqueeze(1))

    if rank_losses:
        rank_loss = torch.stack(rank_losses).mean()
    else:
        rank_loss = torch.zeros((), device=device)
    total_loss = mc_loss + lambda_rank * rank_loss

    if not torch.isfinite(total_loss):
        return Stage2TrainStepResult(
            loss=float("nan"), mc_loss=float("nan"), rank_loss=float("nan"), grad_norm=float("nan"),
            n_demo=len(demo_batch), n_online=len(online_batch), aborted=True, abort_reason="non-finite loss",
        )

    optimizer.zero_grad()
    params = list(encoder.parameters()) + list(value_network.parameters()) + list(action_encoder.parameters())
    mc_grads = torch.autograd.grad(mc_loss, params, retain_graph=True, allow_unused=True)
    rank_grads = (
        torch.autograd.grad(rank_loss, params, retain_graph=True, allow_unused=True)
        if rank_loss.requires_grad else [None] * len(params)
    )
    mc_grad_norm = _gradient_l2(mc_grads)
    rank_grad_norm = _gradient_l2(rank_grads)
    weighted_rank_grad_norm = abs(float(lambda_rank)) * rank_grad_norm
    gradient_ratio = weighted_rank_grad_norm / max(mc_grad_norm, 1e-12)
    total_loss.backward()
    for param in params:
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return Stage2TrainStepResult(
                loss=float(total_loss.detach()), mc_loss=float(mc_loss.detach()), rank_loss=float(rank_loss.detach()),
                grad_norm=float("nan"), n_demo=len(demo_batch), n_online=len(online_batch),
                aborted=True, abort_reason="non-finite gradient",
            )
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm))
    optimizer.step()

    return Stage2TrainStepResult(
        loss=float(total_loss.detach()), mc_loss=float(mc_loss.detach()), rank_loss=float(rank_loss.detach()),
        grad_norm=grad_norm, n_demo=len(demo_batch), n_online=len(online_batch), aborted=False,
        mc_grad_norm=mc_grad_norm, rank_grad_norm=rank_grad_norm,
        weighted_rank_grad_norm=weighted_rank_grad_norm, gradient_ratio=gradient_ratio,
    )
