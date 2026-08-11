"""DistributionalValueModel: composes SetEncoder + IQNValueNetwork into
ONE ``nn.Module`` (guide.md 8.1's ``model.py``).

Before this file, ``BDVLPolicy``/``trainer.train_step`` each carried the
encoder and value network as two separate objects moved/saved/loaded in
lockstep by convention (e.g. ``save_composed_checkpoint`` in policy.py).
This composition makes that pairing an actual invariant of the type
system: one module, one ``.to(device)`` call, one state dict, one
parameter list for the optimizer -- reduces the chance of a future
caller updating/moving one half without the other.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crowd_nav.bayesian_dvl.intent_runtime_config import ACTION_FEATURE_DIM, FROZEN_VALUES, derive_return_bounds
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork
from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, HUMAN_FEATURE_DIM, ROBOT_FEATURE_DIM, SetEncoder


class DistributionalValueModel(nn.Module):
    """Q(s, b, a; tau): robot/human/action features + mask -> quantile
    value estimates (guide.md R4-1: action must be an explicit input,
    not inferred from the successor state alone)."""

    def __init__(
        self,
        robot_feature_dim: int = ROBOT_FEATURE_DIM,
        human_feature_dim: int = HUMAN_FEATURE_DIM,
        action_feature_dim: int = ACTION_FEATURE_DIM,
        embedding_dim: int = 128,
        action_embedding_dim: int = 16,
        n_cosines: int = 64,
        iqn_hidden_dim: int = 128,
        v_min: float = None,
        v_max: float = None,
    ):
        super().__init__()
        derived_min, derived_max = derive_return_bounds(FROZEN_VALUES)
        v_min = derived_min if v_min is None else float(v_min)
        v_max = derived_max if v_max is None else float(v_max)
        self.encoder = SetEncoder(
            robot_feature_dim=robot_feature_dim, human_feature_dim=human_feature_dim,
            embedding_dim=embedding_dim,
        )
        self.action_encoder = ActionEncoder(action_feature_dim=action_feature_dim, embed_dim=action_embedding_dim)
        self.value_network = IQNValueNetwork(
            state_embedding_dim=embedding_dim, action_embedding_dim=action_embedding_dim,
            n_cosines=n_cosines, hidden_dim=iqn_hidden_dim, v_min=v_min, v_max=v_max,
        )

    def forward(
        self, robot_features: torch.Tensor, human_features: torch.Tensor,
        human_mask: torch.Tensor, action_features: torch.Tensor, tau: torch.Tensor,
    ) -> torch.Tensor:
        state_embedding = self.encoder(robot_features, human_features, human_mask)
        action_embedding = self.action_encoder(action_features)
        return self.value_network(state_embedding, action_embedding, tau)

    def encode(self, robot_features: torch.Tensor, human_features: torch.Tensor, human_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(robot_features, human_features, human_mask)
