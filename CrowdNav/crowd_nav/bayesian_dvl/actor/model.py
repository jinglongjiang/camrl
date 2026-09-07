"""Belief-conditioned policy actor: 80-way action classification.

The whole of layer 4 is replaced, not patched. Deleted with it: the scalar value
head, MC return regression, r + gamma*V, successor-state construction, value
ranking, Bellman targets, IQN and its tau grid.

What the evidence said. Every anomaly measured over three days shares one shape
-- a value function is fitted, then a very sensitive argmax converts small value
errors into actions. Three seeds reached indistinguishable training MSE
(0.104-0.118) and produced closed-loop macro 0.556 / 0.300 / 0.356; the same
checkpoint drifted 0.300 -> 0.511 on 500 further MSE updates with the loss flat;
the score gap between candidates was ~0.03 yet the chosen heading reversed; more
data, expert ranking and learner-state ranking each failed to fix it. The
argmax-over-values amplifier is removed here rather than damped.

The 80-action grid stays. It is not what failed -- it is an audited, frozen
control discretisation, and keeping it avoids introducing a second variable.
Crucially, an argmax over POLICY LOGITS is not the same operation as an argmax
over 80 counterfactual value estimates: the network is trained directly on the
quantity it is asked for.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from crowd_nav.bayesian_dvl.intent_runtime_config import HUMAN_FEATURE_DIM_V7
from crowd_nav.bayesian_dvl.set_encoder import ROBOT_FEATURE_DIM, SetEncoder


class PolicyActor(nn.Module):
    """(robot features, belief-conditioned crowd rows, mask) -> 80 action logits."""

    def __init__(self, n_actions: int, robot_feature_dim: int = ROBOT_FEATURE_DIM,
                 human_feature_dim: int = HUMAN_FEATURE_DIM_V7,
                 embedding_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.n_actions = int(n_actions)
        self.encoder = SetEncoder(robot_feature_dim=robot_feature_dim,
                                  human_feature_dim=human_feature_dim,
                                  embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions),
        )

    def forward(self, robot_features: torch.Tensor, human_features: torch.Tensor,
                human_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(robot_features, human_features, human_mask))


def soft_targets(expert_indices, n_actions: int, device) -> torch.Tensor:
    """Uniform probability over the given index set, zero elsewhere.

    Two label widths are supplied by the caller, and this function does not
    choose between them:

    - the ORCA equivalence class, every grid action within tolerance 0.1867 of
      ORCA's continuous velocity (mean 8.27 actions of 80, median 6). This says
      only "ORCA's velocity fell between grid points and the neighbours around
      it are indistinguishable at grid resolution". It is grid-adjacency
      around ONE ORCA action -- it does NOT express going left or right around
      a pedestrian, which are far apart on the grid and never in one class.
    - a single index, ORCA's nearest grid action alone.

    The wide class calls ~10% of the grid correct in a single state, so a high
    top-1 hit rate against it is compatible with executing a member of the
    class that collides. The narrow label removes that slack at the cost of
    treating grid boundaries as hard.
    """
    t = torch.zeros(len(expert_indices), n_actions, device=device)
    for i, ex in enumerate(expert_indices):
        t[i, list(ex)] = 1.0 / len(ex)
    return t


def policy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against the soft target (equivalently KL up to the target's
    own entropy, which is constant in the parameters)."""
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
