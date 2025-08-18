import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
import logging
from collections import deque
import random

from models.state_encoder import StateEncoder
from models.value_net import ValueNet

class CAMRLAgent:
    """
    CAMRL Agent（Value-based）：采样多个动作，选择 value 最大的动作输出，标准 DQN/DDQN + target 实现。
    支持 IL（行为克隆）预训练，高效 Tensor 化、软更新、PyTorch DataLoader 加速。
    """
    def __init__(self, obs_dim, hidden_dim, cfg):
        self.device = torch.device(cfg['device'])
        self.T = cfg['T']
        self.n_action_samples = cfg.get('n_action_samples', 32)
        self.action_dim = cfg['action_dim']
        
        # 优化：预分配GPU内存
        self.state_buffer = torch.zeros((self.n_action_samples, self.T + 1, obs_dim), 
                                       device=self.device, dtype=torch.float32)
        self.action_buffer = torch.zeros((self.n_action_samples, self.action_dim), 
                                        device=self.device, dtype=torch.float32)

        self.encoder = StateEncoder(obs_dim, hidden_dim).to(self.device)
        self.value_net = ValueNet(hidden_dim, **cfg['mamba']).to(self.device)
        self.target_net = deepcopy(self.value_net)

        self.policy_head = torch.nn.Linear(hidden_dim, self.action_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_head.parameters(), lr=float(cfg['lr']))
        self.optimizer = torch.optim.Adam(
            list(self.value_net.parameters()) + list(self.encoder.parameters()),
            lr=float(cfg['lr'])
        )

        # 优化：使用deque提高内存效率
        self.replay_buffer = deque(maxlen=cfg.get('buffer_size', 10000))
        self.gamma = float(cfg['gamma'])
        
        # 优化：添加梯度裁剪
        self.grad_clip = cfg.get('grad_clip', 1.0)
        
        # 优化：缓存状态编码
        self.state_cache = {}

    def select_action(self, state_seq, noise_scale=0.3):
        # 评估模式
        self.encoder.eval()
        self.policy_head.eval()
        self.value_net.eval()

        # 优化：减少CPU-GPU转换
        if isinstance(state_seq, np.ndarray):
            state_seq_tensor = torch.from_numpy(state_seq).unsqueeze(0).to(self.device)
        else:
            state_seq_tensor = state_seq.unsqueeze(0) if state_seq.dim() == 2 else state_seq
            
        with torch.no_grad():
            h_seq = self.encoder(state_seq_tensor)
            h_last = h_seq
            expert_action = self.policy_head(h_last).cpu().numpy().flatten()

        # 优化：批量生成动作样本
        action_samples = np.clip(
            expert_action + np.random.normal(0, noise_scale, (self.n_action_samples, self.action_dim)),
            -1.0, 1.0
        ).astype(np.float32)

        # 优化：使用预分配的buffer
        self.state_buffer.copy_(torch.from_numpy(state_seq).unsqueeze(0).expand(self.n_action_samples, -1, -1))
        self.state_buffer[:, -1, 2:2+self.action_dim] = torch.from_numpy(action_samples).to(self.device)

        with torch.no_grad():
            h = self.encoder(self.state_buffer)
            v = self.value_net(h.unsqueeze(1)).squeeze(-1)

        idx = torch.argmax(v).item()
        return action_samples[idx]

    def train_step(self, batch):
        # RL训练：训练encoder和value_net
        self.encoder.train()
        self.value_net.train()

        s_seq, _, rewards, next_seq, dones = zip(*batch)
        
        # 优化：批量处理，减少转换开销
        s_seq = torch.stack([torch.from_numpy(s) if isinstance(s, np.ndarray) else s for s in s_seq]).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_seq = torch.stack([torch.from_numpy(s) if isinstance(s, np.ndarray) else s for s in next_seq]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            h_next = self.encoder(next_seq)
            v_next = self.target_net(h_next.unsqueeze(1)).squeeze(-1)
            target = rewards + self.gamma * (1 - dones) * v_next

        h = self.encoder(s_seq)
        v = self.value_net(h.unsqueeze(1)).squeeze(-1)
        loss = F.mse_loss(v, target)

        self.optimizer.zero_grad()
        loss.backward()
        
        # 优化：添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
        
        self.optimizer.step()
        return loss.item()

    def imitation_update(self, demo_obs, demo_actions, epochs=10, batch_size=64):
        """
        只训练encoder和policy_head，不训练value_net，兼容Mamba shape
        """
        self.encoder.train()
        self.policy_head.train()

        # 优化：使用GPU数据，减少转换
        if isinstance(demo_obs, np.ndarray):
            demo_obs = torch.from_numpy(demo_obs).float().to(self.device)
        if isinstance(demo_actions, np.ndarray):
            demo_actions = torch.from_numpy(demo_actions).float().to(self.device)

        dataset = TensorDataset(demo_obs, demo_actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_obs, batch_actions in dataloader:
                self.policy_optimizer.zero_grad()
                
                h = self.encoder(batch_obs)
                pred_actions = self.policy_head(h)
                loss = F.mse_loss(pred_actions, batch_actions)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.policy_head.parameters(), self.grad_clip)
                self.policy_optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                logging.info(f"IL Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")

    def soft_update_target(self, tau=0.01):
        """软更新目标网络"""
        for target_param, local_param in zip(self.target_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def add_experience(self, state_seq, action, reward, next_seq, done):
        """添加经验到replay buffer"""
        self.replay_buffer.append((state_seq, action, reward, next_seq, done))

    def sample_batch(self, batch_size):
        """从replay buffer采样"""
        if len(self.replay_buffer) < batch_size:
            return None
        return random.sample(self.replay_buffer, batch_size)

