"""V0 scalar state-value network: V(s, b).

What was removed and why.

IQN is gone. It learned a return DISTRIBUTION whose only consumer was
`max(results, key=lambda r: r.q_mean)` -- the mean, taken right back out. The
cost of that unused generality was a tau embedding, a quantile Huber loss, two
separate quantile counts, and a quantile-grid contract; and the single largest
real defect found in this project came straight out of it: the ranking loss and
the audit scored actions at 16 fixed midpoints while deployment scored them at
32, two disjoint point sets, moving the argmax on 678 of 768 audit rows. A
distributional head earns its complexity once posterior uncertainty is actually
consumed by a risk-sensitive rule. Until then it is a liability, so it comes
back only when there is a claim that needs it.

The ActionEncoder is gone from the scoring path. The action is no longer an
input to be interpreted; it is applied to the world by `lookahead.py` and the
network is asked about the state that results.

What is kept: the permutation-invariant SetEncoder over the belief-conditioned
crowd, unchanged, because nothing in the diagnosis implicated it.

The learned quantity is now well defined for the first time. The frozen chain
regressed MC returns from ORCA demonstrations AND from the student's own
rollouts into one action-value head, with no Bellman operator to reconcile two
different policies' returns. Here the target is the return of ORCA from a
state, so the network estimates V^{pi_ORCA}(s, b) and the lookahead is a
one-step improvement over it -- one policy, one quantity, no mixing.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from crowd_nav.bayesian_dvl.intent_runtime_config import (
    FROZEN_VALUES, HUMAN_FEATURE_DIM_V7, derive_return_bounds,
)
from crowd_nav.bayesian_dvl.set_encoder import ROBOT_FEATURE_DIM, SetEncoder


class ScalarValueModel(nn.Module):
    """(robot features, belief-conditioned crowd rows, mask) -> V, bounded to
    the reachable return range so a fresh network cannot propose values the
    reward function could never produce."""

    def __init__(self, robot_feature_dim: int = ROBOT_FEATURE_DIM,
                 human_feature_dim: int = HUMAN_FEATURE_DIM_V7,
                 embedding_dim: int = 128, hidden_dim: int = 128,
                 v_min: float = None, v_max: float = None):
        super().__init__()
        lo, hi = derive_return_bounds(FROZEN_VALUES)
        self.v_min = lo if v_min is None else float(v_min)
        self.v_max = hi if v_max is None else float(v_max)
        self.encoder = SetEncoder(robot_feature_dim=robot_feature_dim,
                                  human_feature_dim=human_feature_dim,
                                  embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, robot_features: torch.Tensor, human_features: torch.Tensor,
                human_mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(robot_features, human_features, human_mask)
        raw = self.head(z).squeeze(-1)
        return self.v_min + (self.v_max - self.v_min) * torch.sigmoid(raw)
