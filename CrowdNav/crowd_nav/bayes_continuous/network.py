"""Masked permutation-invariant set encoder for SB3 continuous policies."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class PhysicalActionMean(nn.Module):
    """TD3 normalized deterministic mean -> physical Gaussian mean for PPO."""
    def __init__(self, low, high):
        super().__init__()
        self.register_buffer('center', torch.as_tensor((high+low)/2))
        self.register_buffer('scale', torch.as_tensor((high-low)/2))

    def forward(self, values):
        return self.center + self.scale * torch.tanh(values)


class DaggerGaussianPolicy(MultiInputActorCriticPolicy):
    """Gaussian PPO with a bounded mean, not a squashed Gaussian distribution.

    Actions retain SB3's ordinary physical-space sampling, clipping and log-prob.
    The mean parameterization preserves the existing deterministic actor exactly.
    """
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Sequential(self.action_net,
            PhysicalActionMean(self.action_space.low, self.action_space.high))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                               **self.optimizer_kwargs)


class SetEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=96):
        super().__init__(observation_space, features_dim)
        self.human = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.robot = nn.Sequential(nn.Linear(observation_space['robot'].shape[0], 32), nn.ReLU())
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
