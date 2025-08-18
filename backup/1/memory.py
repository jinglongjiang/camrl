import random
import numpy as np
from torch.utils.data import Dataset


class ReplayMemory(Dataset):
    """
    RL 回放：存 (state, action, reward, next_state, done)
    sample(B) -> (states, actions, rewards, next_states, dones)  # 全部 numpy
    """
    def __init__(self, capacity: int, seed: int | None = None):
        self.capacity = int(capacity)
        self.memory: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self.position = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # 仅支持五元组
    def push(self, state, action, reward, next_state, done: bool):
        item = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) >= self.capacity

    def clear(self):
        self.memory.clear()
        self.position = 0

    def sample(self, batch_size: int):
        """均匀采样并堆叠为 numpy。若样本不足则按可用数量采样。"""
        n = len(self.memory)
        if n == 0:
            raise ValueError("ReplayMemory is empty")
        B = min(batch_size, n)
        batch = random.sample(self.memory, B)
        s  = np.stack([b[0] for b in batch]).astype(np.float32)
        a  = np.stack([b[1] for b in batch]).astype(np.float32)
        r  = np.array ([b[2] for b in batch], dtype=np.float32).reshape(B, 1)
        s2 = np.stack([b[3] for b in batch]).astype(np.float32)
        d  = np.array ([b[4] for b in batch], dtype=np.bool_)
        return s, a, r, s2, d

    # 兼容 Dataset（很少用到）
    def __getitem__(self, idx: int):
        return self.memory[idx]

    def compute_mean_std(self):
        if len(self.memory) == 0:
            return None, None
        states = np.stack([t[0] for t in self.memory]).astype(np.float32)
        obs_mean = states.mean(axis=0)
        obs_std  = states.std(axis=0) + 1e-8
        return obs_mean, obs_std


class ExpertReplayMemory(Dataset):
    """
    专家标签缓存：存 (state, expert_action)
    sample(B) -> (states, expert_actions)  # 全部 numpy
    """
    def __init__(self, capacity: int, seed: int | None = None):
        self.capacity = int(capacity)
        self.memory: list[tuple[np.ndarray, np.ndarray]] = []
        self.position = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state, expert_action):
        item = (
            np.asarray(state, dtype=np.float32),
            np.asarray(expert_action, dtype=np.float32),
        )
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) >= self.capacity

    def clear(self):
        self.memory.clear()
        self.position = 0

    def sample(self, batch_size: int):
        n = len(self.memory)
        if n == 0:
            raise ValueError("ExpertReplayMemory is empty")
        B = min(batch_size, n)
        batch = random.sample(self.memory, B)
        s  = np.stack([b[0] for b in batch]).astype(np.float32)
        ae = np.stack([b[1] for b in batch]).astype(np.float32)
        return s, ae

    # 兼容 Dataset
    def __getitem__(self, idx: int):
        return self.memory[idx]

