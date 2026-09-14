"""Masked permutation-invariant set encoder for SB3 continuous policies."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SetEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=96):
        super().__init__(observation_space, features_dim)
        self.human = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.robot = nn.Sequential(nn.Linear(7, 32), nn.ReLU())
        self.query = nn.Linear(32, 64)
        self.fuse = nn.Sequential(nn.Linear(160, features_dim), nn.ReLU())

    def forward(self, observations):
        mask = observations['mask'] > .5
        encoded = self.human(observations['humans'])
        count = mask.sum(1, keepdim=True)
        robot = self.robot(observations['robot'])
        logits = (encoded * self.query(robot)[:, None]).sum(-1) / 8.
        weights = logits.masked_fill(~mask, -1e9).softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        average = (encoded * weights[..., None]).sum(1)
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        return self.fuse(torch.cat([robot, average, maximum], dim=1))
