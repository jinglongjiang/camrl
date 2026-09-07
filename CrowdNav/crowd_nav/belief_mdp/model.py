"""Belief-conditioned twin-head action-value network and replay storage.

Revision note (fixes applied after the first smoke test found real bugs):

- Twin heads Q_R (task) / Q_C (risk cost), each with its own Bellman target,
  combined only at decision time as ``Q_score = Q_R - beta * Q_C``. The
  previous single-head ``Q_task - beta*Risk`` design let Q_task's TD update
  (driven only by task reward) freely learn ``+beta*Risk`` on top of the true
  task value, canceling the penalty in the combined output -- sweeping beta
  then had no reliable effect on behaviour. Q_C now has its own TD target
  built from an actual risk-reward signal (see train.py), so it accumulates
  real discounted future risk instead of being silently absorbed.
- Candidate features now include real action kinematics (velocity, speed,
  goal-alignment, progress, turn cost) in addition to the six Bayesian
  rollout features. Previously Q depended on the action only through the
  Bayesian risk features, so `state_only` (which intentionally broadcasts
  the same risk to every action) made all 80 Q-values identical and argmax
  always picked action 0. Kinematic features now let every ablation mode
  still distinguish actions by their actual motion, independent of whether
  risk is action-conditioned.
- Mamba's context still only reaches the network through a bounded, belief-
  gated correction (see ``refine_belief``) -- but this is honestly a fusion
  path with real gradient flow into Q, not a claim that Mamba is fully
  excluded from the decision. Call it belief-conditioned fusion, not a
  single belief-only channel.

Second revision note (fixes applied after the first real smoke run):

- ``Q_C`` now approximates the discounted probability of eventual
  collision, not an accumulated sum of per-step Bayesian or near-miss risk.
  Its training target in train.py is ``1.0`` only at a terminal collision
  step and ``0.0`` everywhere else -- so it stays roughly in ``[0, 1]``
  instead of accumulating to the tens under ``gamma=0.99``, which would
  otherwise let ``beta`` dominate ``Q_R`` almost regardless of its value.
  ``dmin`` remains available for evaluation metrics and could support a
  separate near-miss auxiliary loss later, but it does not enter ``Q_C``'s
  Bellman target.
- ``q_c_head``'s final linear layer is now zero-initialized with bias
  ``-4.600166`` so training starts at ``softplus(-4.600166) ~= 0.01``
  (roughly the base collision rate) instead of ``softplus(0) ~= 0.693``.

Third revision note (full TaskEncoder/RiskEncoder separation):

Every prior version routed a single ``refine_belief`` fusion (belief +
gated Mamba context) into a shared ``joint`` tensor consumed by *both*
``q_r_head`` and ``q_c_head``. That meant Mamba's context could reach Q_C
through the gate, and belief/risk features could reach Q_R through the
joint tensor -- exactly the cross-talk "Mamba never enters Q_C, Q_C only
sees belief/risk" was supposed to rule out. Fixed by two fully separate
encoder stacks with no shared layer and no concatenation of both input
families into one tensor:

- TaskEncoder: ``context`` (Mamba) + the candidate tensor's kinematic
  columns only -> ``Q_R``. Never touches ``belief`` or the risk columns.
- RiskEncoder: ``belief`` (GDBN) + the candidate tensor's risk columns
  only -> ``Q_C``. Never touches ``context`` or the kinematic columns.

The candidate tensor's layout is unchanged (6 risk features from
``runtime.RISK_FEATURE_NAMES`` followed by 6 kinematic features from
``runtime.KINEMATIC_FEATURE_NAMES``); ``n_risk_features`` just tells this
network where to split it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn


@dataclass
class DecisionFeatures:
    """Features for one decision state."""

    context: np.ndarray
    belief: np.ndarray
    candidates: np.ndarray


class BeliefMDPQNetwork(nn.Module):
    """Q_score(b, a) = Q_R(b, a) - beta * Q_C(b, a), combined at decision time only.

    ``beta`` is a fixed, externally swept hyperparameter (0, 0.25, 0.5, 1.0),
    not a learned Lagrange multiplier -- see README for why. Both Q_R and Q_C
    are learned; Q_C is not a raw pass-through of the GDBN rollout feature
    (an earlier version did this and made beta's effect purely cosmetic).

    TaskEncoder (``context`` + kinematic candidate columns -> Q_R) and
    RiskEncoder (``belief`` + risk candidate columns -> Q_C) are two fully
    separate stacks with no shared layer and no joint tensor -- see the
    module's third revision note for why an earlier version's shared fusion
    could not support the claim "Mamba never reaches Q_C, Q_C only sees
    belief/risk."
    """

    def __init__(
        self,
        context_dim: int,
        belief_dim: int,
        candidate_dim: int,
        n_risk_features: int = 6,
        hidden_dim: int = 96,
        beta: float = 0.5,
    ):
        super().__init__()
        self.beta = float(beta)
        self.n_risk_features = int(n_risk_features)
        n_kinematic_features = candidate_dim - self.n_risk_features
        if n_kinematic_features <= 0:
            raise ValueError(
                f"candidate_dim={candidate_dim} must exceed n_risk_features={n_risk_features}"
            )

        # ---- TaskEncoder: Mamba context + action kinematics -> Q_R. ----
        # Never receives belief or risk features.
        self.task_context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.task_kinematic_encoder = nn.Sequential(
            nn.Linear(n_kinematic_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.q_r_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ---- RiskEncoder: GDBN belief + per-action posterior risk -> Q_C. ----
        # Never receives Mamba context or kinematic features.
        self.risk_belief_encoder = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.risk_action_encoder = nn.Sequential(
            nn.Linear(self.n_risk_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.q_c_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Q_C approximates a discounted eventual-collision probability
        )
        # Softplus(0) = ln(2) ~= 0.693 -- far too large an initial estimate
        # for a quantity that should start near the base collision rate
        # (~2-4%). Zero the last layer and set its bias so the network
        # starts at softplus(-4.600166) ~= 0.01.
        last_linear = self.q_c_head[2]
        nn.init.zeros_(last_linear.weight)
        nn.init.constant_(last_linear.bias, -4.600166)

    def forward(
        self,
        context: torch.Tensor,
        belief: torch.Tensor,
        candidates: torch.Tensor,
        return_components: bool = False,
    ):
        action_count = candidates.shape[1]
        risk_features = candidates[..., : self.n_risk_features]
        kinematic_features = candidates[..., self.n_risk_features :]

        # Q_R: Mamba context + kinematics only. `belief`/`risk_features`
        # never enter this computation.
        context_h = self.task_context_encoder(context)[:, None, :].expand(-1, action_count, -1)
        kinematic_h = self.task_kinematic_encoder(kinematic_features)
        q_r = self.q_r_head(torch.cat((context_h, kinematic_h), dim=-1)).squeeze(-1)

        # Q_C: belief + per-action risk only. `context`/`kinematic_features`
        # never enter this computation.
        belief_h = self.risk_belief_encoder(belief)[:, None, :].expand(-1, action_count, -1)
        risk_h = self.risk_action_encoder(risk_features)
        q_c = self.q_c_head(torch.cat((belief_h, risk_h), dim=-1)).squeeze(-1)

        q_score = q_r - self.beta * q_c

        if return_components:
            return q_score, {"q_r": q_r, "q_c": q_c}
        return q_score


class ReplayBuffer:
    """Fixed-size replay buffer of precomputed belief-MDP features.

    Stores both the one-step TD targets' raw ingredients (``task_reward``,
    ``risk_cost``) and each transition's Monte-Carlo return under the policy
    that generated it (``mc_task_return``, ``mc_collision_return``, computed
    by a reverse pass over the whole episode once it ends). The MC return
    gives every state in a trajectory that eventually collides a nonzero
    collision-return target, not just the single terminal transition where
    ``risk_cost`` itself is 1 -- ``risk_cost``-based TD learning alone has to
    propagate that signal backward one Bellman step at a time, which is slow
    and, combined with a shrinking replay fraction of colliding episodes as
    the policy improves, can leave most of a trajectory under-supervised.

    ``permanent=True`` makes ``add`` a no-op once ``capacity`` transitions
    have been stored, instead of overwriting the oldest ones -- used for the
    teacher demonstration buffer, which must never be evicted by online
    experience (see ``sample_stratified``).
    """

    def __init__(
        self,
        capacity: int,
        context_dim: int,
        belief_dim: int,
        action_count: int,
        candidate_dim: int,
        permanent: bool = False,
        expert_set_size: int = 3,
    ):
        self.capacity = int(capacity)
        self.permanent = bool(permanent)
        self.context = np.zeros((capacity, context_dim), dtype=np.float16)
        self.belief = np.zeros((capacity, belief_dim), dtype=np.float16)
        self.candidates = np.zeros(
            (capacity, action_count, candidate_dim), dtype=np.float16
        )
        self.next_context = np.zeros_like(self.context)
        self.next_belief = np.zeros_like(self.belief)
        self.next_candidates = np.zeros_like(self.candidates)
        self.action = np.zeros(capacity, dtype=np.int16)
        self.expert_action = np.zeros(capacity, dtype=np.int16)
        self.expert_action_set = np.zeros((capacity, expert_set_size), dtype=np.int16)
        # Full per-action teacher value-lookahead score (Round 11) -- lets
        # any later analysis compute a continuous "regret" (teacher's best
        # score minus its score for whatever action was actually chosen,
        # normalized by the teacher's own score range for that state)
        # instead of only the hard top-K membership expert_action_set
        # already captures. expert_action/expert_action_set remain (derived
        # from this same array at collection time) as cheap precomputed
        # conveniences for the existing margin loss.
        self.teacher_scores = np.zeros((capacity, action_count), dtype=np.float32)
        self.task_reward = np.zeros(capacity, dtype=np.float32)
        self.risk_cost = np.zeros(capacity, dtype=np.float32)
        self.mc_task_return = np.zeros(capacity, dtype=np.float32)
        self.mc_collision_return = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        # has_expert_label: this transition has a valid teacher label to
        # imitate (true for Stage 1 demo collection and DAgger-round data,
        # false for pure Stage 2 online experience) -- gates margin loss.
        # is_expert_action: the action actually *executed* in this
        # transition came from the teacher (true) vs. the policy itself
        # (false) -- bookkeeping only; TD/MC targets always supervise
        # whichever action was actually taken and its real outcome,
        # regardless of this flag.
        self.has_expert_label = np.zeros(capacity, dtype=np.float32)
        self.is_expert_action = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    @property
    def full(self) -> bool:
        return self.size >= self.capacity

    def clear(self):
        """Reset to empty without reallocating the underlying arrays -- used
        to start a fresh DAgger round's ``recent_dagger`` buffer after its
        contents have already been moved into ``older_dagger`` via
        ``export_all``/``import_batch``."""
        self.position = 0
        self.size = 0

    def add(
        self,
        state: DecisionFeatures,
        action: int,
        expert_action: int,
        expert_action_set: np.ndarray,
        task_reward: float,
        risk_cost: float,
        mc_task_return: float,
        mc_collision_return: float,
        next_state: DecisionFeatures,
        done: bool,
        has_expert_label: bool,
        is_expert_action: bool,
        teacher_scores: np.ndarray,
    ):
        if self.permanent and self.full:
            return
        index = self.position
        self.context[index] = state.context
        self.belief[index] = state.belief
        self.candidates[index] = state.candidates
        self.next_context[index] = next_state.context
        self.next_belief[index] = next_state.belief
        self.next_candidates[index] = next_state.candidates
        self.action[index] = int(action)
        self.expert_action[index] = int(expert_action)
        self.expert_action_set[index] = np.asarray(expert_action_set, dtype=np.int16)
        self.teacher_scores[index] = np.asarray(teacher_scores, dtype=np.float32)
        self.task_reward[index] = float(task_reward)
        self.risk_cost[index] = float(risk_cost)
        self.mc_task_return[index] = float(mc_task_return)
        self.mc_collision_return[index] = float(mc_collision_return)
        self.done[index] = float(done)
        self.has_expert_label[index] = float(has_expert_label)
        self.is_expert_action[index] = float(is_expert_action)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _fetch(self, indices: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(array[indices], dtype=dtype, device=device)

        return {
            "context": tensor(self.context),
            "belief": tensor(self.belief),
            "candidates": tensor(self.candidates),
            "next_context": tensor(self.next_context),
            "next_belief": tensor(self.next_belief),
            "next_candidates": tensor(self.next_candidates),
            "action": tensor(self.action, torch.long),
            "expert_action": tensor(self.expert_action, torch.long),
            "expert_action_set": tensor(self.expert_action_set, torch.long),
            "teacher_scores": tensor(self.teacher_scores),
            "task_reward": tensor(self.task_reward),
            "risk_cost": tensor(self.risk_cost),
            "mc_task_return": tensor(self.mc_task_return),
            "mc_collision_return": tensor(self.mc_collision_return),
            "done": tensor(self.done),
            "has_expert_label": tensor(self.has_expert_label),
            "is_expert_action": tensor(self.is_expert_action),
        }

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Random sampling *with* replacement -- used for training batches,
        where repeated draws across updates are expected and desired."""
        indices = rng.integers(0, self.size, size=int(batch_size))
        return self._fetch(indices, device)

    def deterministic_chunks(self, chunk_size: int, device: torch.device):
        """Yield every stored transition exactly once, in a fixed order, no
        replacement -- for validation, where re-sampling the same
        transition twice (or skipping others) would bias the estimate, and
        where nothing about the ordering should depend on the training RNG.
        """
        for start in range(0, self.size, chunk_size):
            indices = np.arange(start, min(start + chunk_size, self.size))
            yield self._fetch(indices, device)

    _RAW_FIELDS = (
        "context", "belief", "candidates", "next_context", "next_belief",
        "next_candidates", "action", "expert_action", "expert_action_set",
        "teacher_scores", "task_reward", "risk_cost", "mc_task_return",
        "mc_collision_return", "done", "has_expert_label", "is_expert_action",
    )

    def export_all(self) -> Dict[str, np.ndarray]:
        """All currently stored transitions' raw fields, in storage order --
        paired with ``import_batch`` to move a whole batch of transitions
        from one buffer into another without going through per-transition
        ``DecisionFeatures`` reconstruction (used by the stratified DAgger
        buffers: a round's ``recent_dagger`` contents move into
        ``older_dagger`` once the next round starts)."""
        return {name: getattr(self, name)[: self.size] for name in self._RAW_FIELDS}

    def import_batch(self, batch: Dict[str, np.ndarray]):
        """Bulk-append raw transitions from ``export_all``, respecting this
        buffer's own ``permanent``/circular-overwrite semantics exactly like
        repeated ``add`` calls would, just vectorized."""
        n = int(len(batch["action"]))
        if n == 0:
            return
        if self.permanent:
            n = min(n, max(0, self.capacity - self.size))
            if n == 0:
                return
        positions = (self.position + np.arange(n)) % self.capacity
        for name in self._RAW_FIELDS:
            getattr(self, name)[positions] = batch[name][:n]
        self.position = int((self.position + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)


def sample_stratified(
    buffers: Dict[str, "ReplayBuffer"],
    ratios: Dict[str, float],
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Sample a batch from N replay buffers at fixed ratios.

    A buffer that is currently empty (e.g. ``online_rl`` before Stage 2, or
    ``older_dagger`` before a second DAgger round has happened) contributes
    zero samples; its share of ``batch_size`` is redistributed
    proportionally among the remaining non-empty buffers, rather than
    crashing on an empty buffer or silently training on fewer than
    ``batch_size`` transitions whenever some strata haven't appeared yet.
    """
    non_empty = [name for name, buf in buffers.items() if buf.size > 0]
    if not non_empty:
        raise ValueError("sample_stratified: every buffer is empty")
    total_ratio = sum(ratios[name] for name in non_empty)
    counts = {}
    allocated = 0
    for name in non_empty[:-1]:
        count = int(round(batch_size * ratios[name] / total_ratio))
        counts[name] = count
        allocated += count
    counts[non_empty[-1]] = max(0, batch_size - allocated)

    batches = {
        name: buffers[name].sample(count, rng, device)
        for name, count in counts.items()
        if count > 0
    }
    keys = next(iter(batches.values())).keys()
    return {key: torch.cat([batches[name][key] for name in batches], dim=0) for key in keys}
