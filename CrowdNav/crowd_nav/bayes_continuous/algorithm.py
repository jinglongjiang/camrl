"""Explicit online TD3/BC objectives with separately learned collision cost.

Extends SB3's MIT-licensed TD3 update. A soft cost penalty is not a safety guarantee.
"""
import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.utils import polyak_update, obs_as_tensor


class CostReplay(DictReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise ValueError('Cost replay currently requires one environment')
        self.costs = np.zeros((self.buffer_size, 1), np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        self.costs[self.pos, 0] = float(infos[0]['collision_cost'])
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        sample = super()._get_samples(batch_inds, env)
        return SimpleNamespace(**sample._asdict(), costs=self.to_torch(self.costs[batch_inds]))


class BayesSetTD3(TD3):
    def _setup_model(self):
        super()._setup_model()
        self.cost_critic = copy.deepcopy(self.critic)
        self.cost_target = copy.deepcopy(self.critic_target)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=3e-4)
        self.actor_enabled = False
        self.bc_receipt = getattr(self, 'bc_receipt', None)
        self.warmup_updates = getattr(self, 'warmup_updates', 0)
        self.critic_validation = getattr(self, 'critic_validation', {'passed': False})
        self.loss_history = getattr(self, 'loss_history', [])
        self.demo_observations = None
        self.demo_actions = None
        self.cost_weight = getattr(self, 'cost_weight', 2.)
        self.bc_weight = getattr(self, 'bc_weight', .1)

    def _get_torch_save_params(self):
        names, variables = super()._get_torch_save_params()
        return names + ['cost_critic', 'cost_target', 'cost_optimizer'], variables

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['demo_observations', 'demo_actions']

    def enable_actor(self, receipt):
        if (receipt['episodes'] < 100 or receipt['success_rate'] < .9 or
                receipt['collision_rate'] > .02 or self.warmup_updates < 1000 or
                not self.critic_validation['passed']):
            raise RuntimeError('BC closed-loop or critic warm-up gate failed')
        self.bc_receipt = dict(receipt)
        self.actor_enabled = True

    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size)
            self._n_updates += 1
            with torch.no_grad():
                noise = torch.randn_like(data.actions)*self.target_policy_noise
                next_action = (self.actor_target(data.next_observations) +
                               noise.clamp(-self.target_noise_clip, self.target_noise_clip)).clamp(-1, 1)
                next_q = torch.cat(self.critic_target(data.next_observations, next_action), 1).min(1, keepdim=True).values
                target = data.rewards + (1-data.dones)*self.gamma*next_q
                next_cost = torch.cat(self.cost_target(data.next_observations, next_action), 1).max(1, keepdim=True).values
                cost_target = data.costs + (1-data.dones)*next_cost.clamp(0, 1)
            reward_loss = sum(F.smooth_l1_loss(q, target) for q in self.critic(data.observations, data.actions))
            cost_loss = sum(F.smooth_l1_loss(q, cost_target) for q in self.cost_critic(data.observations, data.actions))
            for optimizer, loss in [(self.critic.optimizer, reward_loss), (self.cost_optimizer, cost_loss)]:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            row = {'reward_td': float(reward_loss.detach()), 'cost_td': float(cost_loss.detach())}
            if not self.actor_enabled:
                self.warmup_updates += 1
            elif self._n_updates % self.policy_delay == 0:
                if self.demo_observations is None:
                    raise RuntimeError('Permanent BC data required')
                actions = self.actor(data.observations)
                qr = -self.critic.q1_forward(data.observations, actions).mean()
                qc = self.cost_critic.q1_forward(data.observations, actions).mean()*self.cost_weight
                indices = np.random.randint(len(self.demo_actions), size=batch_size)
                obs = obs_as_tensor({k:v[indices] for k,v in self.demo_observations.items()}, self.device)
                bc = self.bc_weight*F.mse_loss(self.actor(obs),
                    torch.as_tensor(self.demo_actions[indices], device=self.device))
                # Log component gradient norms without merging their interpretation.
                for name, loss in [('actor_reward', qr), ('actor_cost', qc), ('actor_bc', bc)]:
                    grads = torch.autograd.grad(loss, self.actor.parameters(), retain_graph=True, allow_unused=True)
                    row[name] = float(loss.detach())
                    row[name+'_grad_norm'] = float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))
                self.actor.optimizer.zero_grad()
                (qr+qc+bc).backward()
                self.actor.optimizer.step()
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.cost_critic.parameters(), self.cost_target.parameters(), self.tau)
            if not np.isfinite(list(row.values())).all():
                raise FloatingPointError('Nonfinite optimization loss')
            self.loss_history.append(row)
