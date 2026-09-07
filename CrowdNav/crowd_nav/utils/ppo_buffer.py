# -*- coding: utf-8 -*-
"""
Replay Buffer - Off-Policy Data Collection (Supports DoubleQ Discrete Actions)
"""

import numpy as np
import torch
from collections import deque
import logging


# ========== Replay Buffer (Off-Policy, 支持离散动作) ==========

class ReplayBufferIQL:
    """Universal Replay Buffer (Off-Policy)

    Features:
    - Stores transitions (s, a, r, s', done)
    - Supports discrete action indices (action_indices) for DoubleQ
    - Retains BC/IL data long-term (no clearing)
    - Supports random sampling
    - Can mix IL+RL data
    - Stores Monte Carlo Returns (G_t) for stable regression target
    """
    def __init__(self, capacity=200000, seq_len=12, obs_shape=(8, 13),
                 n_step: int = 1, gamma: float = 0.99,
                 use_per: bool = False, per_alpha: float = 0.6, per_beta: float = 0.4, per_eps: float = 1e-6,
                 occlusion_mode: str = 'off', belief_features: str = 'full'):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_shape = obs_shape
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.use_per = bool(use_per)
        self.per_alpha = float(per_alpha)
        self.per_beta = float(per_beta)
        self.per_eps = float(per_eps)
        self.occlusion_mode = str(occlusion_mode)
        self.belief_features = str(belief_features)
        self.belief_contract = {
            'episodes': 0, 'entity_rows': 0, 'hidden_rows': 0,
            'nontrivial_confidence_rows': 0,
        }

        # Circular buffer for transitions
        self.ptr = 0
        self.size = 0

        # Pre-allocate arrays
        self.states = np.zeros((capacity, seq_len, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.action_indices = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, seq_len, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # Monte Carlo Returns
        self.returns = np.zeros(capacity, dtype=np.float32)
        
        # PER
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

        logging.info(f"[IQL-REPLAY] Initialized: capacity={capacity}, seq_len={seq_len}, with MC returns")

    def store(self, state, action, reward, next_state, done, action_idx=None, priority: float = None, mc_return: float = 0.0):
        """Store a single transition

        Args:
            state: [T, entity_rows+3, 13] tokens
            action: [2] continuous action (kept for debug)
            reward: scalar
            next_state: [T, entity_rows+3, 13] tokens
            done: bool
            action_idx: int, discrete action index [0, 79]
            mc_return: float, full trajectory return from this step
        """
        # Ensure state is [seq_len, *obs_shape]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.ndim != 3 or tuple(state.shape[1:]) != tuple(self.obs_shape):
            raise RuntimeError(
                f"replay state contract {state.shape}; expected [T,{self.obs_shape}]")
        if (next_state.ndim != 3 or
                tuple(next_state.shape[1:]) != tuple(self.obs_shape)):
            raise RuntimeError(
                f"replay next_state contract {next_state.shape}; "
                f"expected [T,{self.obs_shape}]")

        # Truncate or pad to seq_len
        if state.shape[0] > self.seq_len:
            state = state[-self.seq_len:]
        elif state.shape[0] < self.seq_len:
            pad_len = self.seq_len - state.shape[0]
            state = np.concatenate([np.repeat(state[:1], pad_len, axis=0), state], axis=0)

        if next_state.shape[0] > self.seq_len:
            next_state = next_state[-self.seq_len:]
        elif next_state.shape[0] < self.seq_len:
            pad_len = self.seq_len - next_state.shape[0]
            next_state = np.concatenate([np.repeat(next_state[:1], pad_len, axis=0), next_state], axis=0)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action[:2]  # Only take [vx, vy]
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.returns[self.ptr] = float(mc_return)

        # Store discrete action index
        if action_idx is not None:
            self.action_indices[self.ptr] = action_idx
        # PER priority
        if self.use_per:
            if priority is None:
                priority = self.max_priority
            priority = float(priority)
            self.priorities[self.ptr] = priority
            if priority > self.max_priority:
                self.max_priority = priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def store_episode(self, episode_data):
        """Split episode data into transitions and store

        Args:
            episode_data: dict with keys:
                - states: [T, 34] or tokens [T, 6, 13]
                - actions_continuous: [T, 2]
                - rewards: [T]
                - dones: [T]
        """
        from crowd_nav.contracts import joint34_to_tokens

        states_raw = episode_data['states']
        actions = episode_data.get('actions_continuous', episode_data.get('actions', []))
        rewards = episode_data['rewards']
        dones = episode_data.get('dones', np.zeros(len(states_raw), dtype=bool))

        T = len(states_raw)
        if T <= 1:
            return  # Need at least two steps to form a transition

        # Convert states to tokens
        if 'tokens' in episode_data and episode_data['tokens'] is not None:
            states_tokens = episode_data['tokens']
        else:
            # An occlusion episode already carries native policy tokens.
            # tokens whose columns 9-12 hold the belief. Round-tripping those
            # through the 34-D vector would silently discard exactly the four
            # columns this whole experiment is about, so a state that is
            # already tokenised is stored as-is.
            _sr = np.asarray(states_raw, dtype=np.float32)
            if (_sr.ndim == 3 and
                    tuple(_sr.shape[1:]) == tuple(self.obs_shape)):
                states_tokens = _sr
            else:
                states_tokens = joint34_to_tokens(np.array(states_raw, dtype=np.float32))

        if isinstance(states_tokens, torch.Tensor):
            states_tokens = states_tokens.detach().cpu().numpy()
        states_tokens = np.asarray(states_tokens, dtype=np.float32)

        # Ensure [T, 6, 13]
        if states_tokens.ndim == 4 and states_tokens.shape[1] == 1:
            states_tokens = np.squeeze(states_tokens, axis=1)
        self._validate_belief_contract(states_tokens)

        # Only use externally provided discrete action indices (forbid reverse engineering from continuous)
        actions_arr = np.array(actions, dtype=np.float32)[:, :2]
        action_indices_arr = episode_data.get('action_indices', None)
        if action_indices_arr is None:
            logging.warning("[REPLAY] Missing action_indices; skip store_episode to avoid mismatched action grid.")
            return
        if any(a is None for a in action_indices_arr):
            logging.warning("[REPLAY] action_indices contains None; skip store_episode.")
            return
        action_indices_arr = np.array(action_indices_arr, dtype=np.int64)
        if action_indices_arr.shape[0] != T:
            logging.warning(f"[REPLAY] action_indices length mismatch: {action_indices_arr.shape[0]} vs T={T}. Skip store.")
            return

        # Pre-calculate Monte Carlo returns (G_t)
        # G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} ...
        mc_returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            # SARL uses 0.99 gamma usually
            G = rewards[t] + self.gamma * G
            mc_returns[t] = G

        # Split into transitions. Keep the final transition: it carries the
        # terminal success/collision/timeout reward used by discrete Q-learning.
        for t in range(T):
            n = self.n_step
            end = min(t + n, T - 1)
            # n-step return (still kept for reference, but MC return is preferred for SARL regression)
            Rn = 0.0
            done_n = False
            for k in range(n):
                idx = t + k
                if idx >= T:
                    break
                Rn += (self.gamma ** k) * float(rewards[idx])
                if bool(dones[idx]):
                    done_n = True
                    end = min(idx + 1, T - 1)
                    break

            # Build history window
            if t + 1 >= self.seq_len:
                state_t = states_tokens[t + 1 - self.seq_len:t + 1]
            else:
                pad_len = self.seq_len - (t + 1)
                state_t = np.concatenate([
                    np.repeat(states_tokens[:1], pad_len, axis=0),
                    states_tokens[:t+1]
                ], axis=0)

            # Window for s_{t+n}
            if end + 1 >= self.seq_len:
                next_state_t = states_tokens[end + 1 - self.seq_len:end + 1]
            else:
                pad_len = self.seq_len - (end + 1)
                next_state_t = np.concatenate([
                    np.repeat(states_tokens[:1], pad_len, axis=0),
                    states_tokens[:end+1]
                ], axis=0)

            # Store with action_idx and MC return
            self.store(
                state=state_t,
                action=actions_arr[t],
                reward=Rn,
                next_state=next_state_t,
                done=done_n,
                action_idx=int(action_indices_arr[t]),
                mc_return=mc_returns[t]
            )

    def _validate_belief_contract(self, tokens):
        if self.occlusion_mode == 'off':
            return
        if (tokens.ndim != 3 or
                tuple(tokens.shape[1:]) != tuple(self.obs_shape)):
            raise RuntimeError(
                f"occlusion replay received {tokens.shape}; "
                f"expected [T,{self.obs_shape}]")
        humans = tokens[:, 3:, :]
        entity = (humans[:, :, 11] + humans[:, :, 12]) > 0.5
        hidden = humans[:, :, 12] > 0.5
        self.belief_contract['episodes'] += 1
        self.belief_contract['entity_rows'] += int(entity.sum())
        self.belief_contract['hidden_rows'] += int(hidden.sum())
        if not np.any(entity):
            raise RuntimeError("occlusion replay contains no labelled entity rows")
        if self.belief_features == 'fixed_confidence':
            if not np.allclose(humans[:, :, 9][entity], 1.0, atol=1e-7):
                raise RuntimeError("fixed_confidence replay has non-unit p_exist")
            if not np.allclose(humans[:, :, 10][entity], 0.0, atol=1e-7):
                raise RuntimeError("fixed_confidence replay has nonzero uncertainty")
        elif self.belief_features == 'full':
            nontrivial = hidden & (
                (np.abs(humans[:, :, 9] - 1.0) > 1e-7) |
                (np.abs(humans[:, :, 10]) > 1e-7))
            n_nontrivial = int(nontrivial.sum())
            self.belief_contract['nontrivial_confidence_rows'] += n_nontrivial
            if np.any(hidden) and n_nontrivial == 0 and self.occlusion_mode == 'bayes':
                raise RuntimeError(
                    "full Bayes episode has hidden rows but no posterior confidence values")
        else:
            raise RuntimeError(
                f"unknown belief_features contract {self.belief_features!r}")
        if self.belief_contract['episodes'] == 1:
            logging.info(
                "[BELIEF-CONTRACT] mode=%s features=%s shape=%s "
                "entity_rows=%d hidden_rows=%d nontrivial_confidence_rows=%d",
                self.occlusion_mode, self.belief_features, tuple(tokens.shape),
                self.belief_contract['entity_rows'],
                self.belief_contract['hidden_rows'],
                self.belief_contract['nontrivial_confidence_rows'],
            )

    def push_episode(self, episode_data=None, **kwargs):
        """Accept both the training dict form and Explorer keyword form."""
        if episode_data is not None and kwargs:
            raise TypeError("push_episode accepts either one dict or keywords, not both")
        if episode_data is None:
            episode_data = kwargs
        if not isinstance(episode_data, dict):
            raise TypeError("push_episode requires an episode dictionary")
        return self.store_episode(episode_data)

    def sample(self, batch_size, device='cpu'):
        """随机采样batch

        Returns:
            dict with keys:
                - states: [B, T, 6, 13]
                - actions: [B, 2]
                - rewards: [B]
                - next_states: [B, T, 6, 13]
                - dones: [B]
                - returns: [B] (MC Return)
        """
        if self.use_per:
            prios = self.priorities[:self.size] + self.per_eps
            probs = prios ** self.per_alpha
            probs = probs / probs.sum()
            indices = np.random.choice(self.size, batch_size, p=probs)
            weights = (self.size * probs[indices]) ** (-self.per_beta)
            weights = weights / weights.max()
            weights_t = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)
            weights_t = None

        batch = {
            'states': torch.tensor(self.states[indices], dtype=torch.float32, device=device),
            'actions': torch.tensor(self.actions[indices], dtype=torch.float32, device=device),
            'action_indices': torch.tensor(self.action_indices[indices], dtype=torch.long, device=device),
            'rewards': torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            'next_states': torch.tensor(self.next_states[indices], dtype=torch.float32, device=device),
            'dones': torch.tensor(self.dones[indices], dtype=torch.float32, device=device),
            'returns': torch.tensor(self.returns[indices], dtype=torch.float32, device=device),  # ✅ 新增：MC Return
            'indices': indices
        }
        if weights_t is not None:
            batch['weights'] = weights_t

        return batch

    def update_priorities(self, indices, priorities):
        if not self.use_per:
            return
        if indices is None:
            return
        for idx, prio in zip(indices, priorities):
            p = float(prio)
            if p <= 0:
                p = self.per_eps
            self.priorities[int(idx)] = p
            if p > self.max_priority:
                self.max_priority = p

    def set_per_beta(self, beta: float):
        self.per_beta = float(beta)

    def __len__(self):
        return self.size

    def clear(self):
        """清空buffer（IQL通常不需要，但提供接口）"""
        self.ptr = 0
        self.size = 0


class ReplayBufferSAC(ReplayBufferIQL):
    """
    SAC专用Replay Buffer（与IQL实现一致，但独立命名和日志，避免SAC路径出现IQL标签）
    """
    def __init__(self, capacity=200000, seq_len=12, obs_shape=(8, 13)):
        super().__init__(capacity=capacity, seq_len=seq_len, obs_shape=obs_shape)
        logging.info(f"[SAC-REPLAY] Initialized: capacity={capacity}, seq_len={seq_len}")


class ReplayBufferDoubleQ(ReplayBufferIQL):
    """
    DoubleQ专用Replay Buffer（与IQL实现一致，但独立命名和日志）
    """
    def __init__(self, capacity=200000, seq_len=12, obs_shape=(8, 13)):
        super().__init__(capacity=capacity, seq_len=seq_len, obs_shape=obs_shape)
        logging.info(f"[DOUBLEQ-REPLAY] Initialized: capacity={capacity}, seq_len={seq_len}")
