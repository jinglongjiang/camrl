"""Masked permutation-invariant set encoder for SB3 continuous policies."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SetEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=96):
        super().__init__(observation_space, features_dim)
        self.human = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.robot = nn.Sequential(nn.Linear(9, 32), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(161, features_dim), nn.ReLU())

    def forward(self, observations):
        mask = observations['mask'] > .5
        encoded = self.human(observations['humans'])
        count = mask.sum(1, keepdim=True)
        average = (encoded * mask[..., None]).sum(1) / count.clamp_min(1)
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        return self.fuse(torch.cat([self.robot(observations['robot']), average, maximum,
                                    torch.log1p(count.float())], dim=1))
