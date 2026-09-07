"""Trainable belief-space action-value model and replay storage."""

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


class BeliefSpaceQNetwork(nn.Module):
    """Bayesian action scorer with a bounded Mamba context residual.

    Candidate values are produced by the belief and action-risk towers. The
    frozen Mamba context can refine each score, but its residual is bounded and
    gated by the Bayesian belief. No pretrained Mamba Q/value output is used.
    """

    def __init__(
        self,
        context_dim: int,
        belief_dim: int,
        candidate_dim: int,
        hidden_dim: int = 96,
        context_residual_cap: float = 0.15,
    ):
        super().__init__()
        self.context_residual_cap = float(context_residual_cap)
        self.belief_encoder = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.bayesian_value = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.context_residual = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        context: torch.Tensor,
        belief: torch.Tensor,
        candidates: torch.Tensor,
        return_components: bool = False,
    ):
        belief_h = self.belief_encoder(belief)
        candidate_h = self.candidate_encoder(candidates)
        action_count = candidates.shape[1]
        belief_actions = belief_h[:, None, :].expand(-1, action_count, -1)

        bayesian_input = torch.cat((belief_actions, candidate_h), dim=-1)
        bayesian_q = self.bayesian_value(bayesian_input).squeeze(-1)

        context_h = self.context_encoder(context)
        gate = self.context_gate(belief)
        gated_context = (context_h * gate)[:, None, :].expand(
            -1, action_count, -1
        )
        residual_input = torch.cat(
            (belief_actions, candidate_h, gated_context),
            dim=-1,
        )
        residual = self.context_residual(residual_input).squeeze(-1)
        residual = self.context_residual_cap * residual
        q_values = bayesian_q + residual
        if return_components:
            return q_values, {
                "bayesian_q": bayesian_q,
                "context_residual": residual,
                "context_gate": gate,
            }
        return q_values


class ReplayBuffer:
    """Fixed-size replay buffer storing already computed belief features."""

    def __init__(
        self,
        capacity: int,
        context_dim: int,
        belief_dim: int,
        action_count: int,
        candidate_dim: int,
    ):
        self.capacity = int(capacity)
        self.context = np.zeros(
            (capacity, context_dim), dtype=np.float16
        )
        self.belief = np.zeros(
            (capacity, belief_dim), dtype=np.float16
        )
        self.candidates = np.zeros(
            (capacity, action_count, candidate_dim), dtype=np.float16
        )
        self.next_context = np.zeros_like(self.context)
        self.next_belief = np.zeros_like(self.belief)
        self.next_candidates = np.zeros_like(self.candidates)
        self.action = np.zeros(capacity, dtype=np.int16)
        self.expert_action = np.zeros(capacity, dtype=np.int16)
        self.teacher_scores = np.zeros(
            (capacity, action_count), dtype=np.float16
        )
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.demo = np.zeros(capacity, dtype=np.float32)
        self.collision_credit = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: DecisionFeatures,
        action: int,
        expert_action: int,
        teacher_scores: np.ndarray,
        reward: float,
        next_state: DecisionFeatures,
        done: bool,
        demo: bool,
        collision_credit: float = 0.0,
    ):
        index = self.position
        self.context[index] = state.context
        self.belief[index] = state.belief
        self.candidates[index] = state.candidates
        self.next_context[index] = next_state.context
        self.next_belief[index] = next_state.belief
        self.next_candidates[index] = next_state.candidates
        self.action[index] = int(action)
        self.expert_action[index] = int(expert_action)
        self.teacher_scores[index] = teacher_scores
        self.reward[index] = float(reward)
        self.done[index] = float(done)
        self.demo[index] = float(demo)
        self.collision_credit[index] = float(collision_credit)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        indices = rng.integers(0, self.size, size=int(batch_size))

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(
                array[indices],
                dtype=dtype,
                device=device,
            )

        return {
            "context": tensor(self.context),
            "belief": tensor(self.belief),
            "candidates": tensor(self.candidates),
            "next_context": tensor(self.next_context),
            "next_belief": tensor(self.next_belief),
            "next_candidates": tensor(self.next_candidates),
            "action": tensor(self.action, torch.long),
            "expert_action": tensor(self.expert_action, torch.long),
            "teacher_scores": tensor(self.teacher_scores),
            "reward": tensor(self.reward),
            "done": tensor(self.done),
            "demo": tensor(self.demo),
            "collision_credit": tensor(self.collision_credit),
        }
