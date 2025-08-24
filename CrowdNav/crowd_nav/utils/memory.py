# crowd_nav/utils/memory.py
# -*- coding: utf-8 -*-
import random
import numpy as np
from torch.utils.data import Dataset

# --------- 小工具：将 ActionXY/ActionRot 变成 numpy 向量 ----------
def _action_to_np(action):
    # 允许 ActionXY / ActionRot 或 numpy/list/tuple
    if hasattr(action, 'vx') and hasattr(action, 'vy'):      # ActionXY
        return np.asarray([action.vx, action.vy], dtype=np.float32)
    if hasattr(action, 'v') and hasattr(action, 'r'):        # ActionRot
        return np.asarray([float(action.v), float(action.r)], dtype=np.float32)
    return np.asarray(action, dtype=np.float32)

def _maybe_warn_if_not_normalized(a_np: np.ndarray, once_flag: dict, key: str, thresh: float = 1.05):
    """
    RL 回放应是 [-1,1]；若检测到明显示度>1，提示一次（不打断训练）。
    """
    if once_flag.get(key, False):
        return
    if np.any(np.abs(a_np) > thresh):
        once_flag[key] = True
        print(f"[ReplayMemory] WARNING: actions look >1 in magnitude (max={np.abs(a_np).max():.3f}). "
              f"RL buffer is expected to hold tanh-normalized actions in [-1,1]. "
              f"Please check your writer (collect loop).")

class ReplayMemory(Dataset):
    """
    RL 回放：存 (state, action, reward, next_state, done)
    sample(B) -> (states, actions, rewards, next_states, dones)  # 全部 numpy
    约定：actions 为 tanh 范围 [-1,1]。如误塞入物理速度，会打印一次警告。
    """
    def __init__(self, capacity: int, seed: int | None = None):
        self.capacity = int(capacity)
        self.memory: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self.position = 0
        self._warn_flags = {}  # 只提示一次
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # 仅支持五元组
    def push(self, state, action, reward, next_state, done: bool):
        a_np = _action_to_np(action).astype(np.float32)
        _maybe_warn_if_not_normalized(a_np, self._warn_flags, "rl_action_norm_check")
        item = (
            np.asarray(state, dtype=np.float32),
            a_np,
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    # 批量塞入（来自并发采样器时更高效）
    def push_many(self, items):
        """
        items: 可迭代的 (state, action, reward, next_state, done)
        """
        for s, a, r, s2, d in items:
            self.push(s, a, r, s2, d)

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) >= self.capacity

    def clear(self):
        self.memory.clear()
        self.position = 0
        self._warn_flags.clear()

    # 统一打包（保证维度）
    @staticmethod
    def _pack(batch):
        B = len(batch)
        s  = np.stack([b[0] for b in batch]).astype(np.float32)              # (B, Ds)
        a  = np.stack([b[1] for b in batch]).astype(np.float32)              # (B, 2)  in [-1,1]
        r  = np.array ([b[2] for b in batch], dtype=np.float32).reshape(B,1) # (B,1)
        s2 = np.stack([b[3] for b in batch]).astype(np.float32)              # (B, Ds)
        d  = np.array ([b[4] for b in batch], dtype=np.bool_)                # (B,)
        return s, a, r, s2, d

    def sample(self, batch_size: int):
        """均匀采样并堆叠为 numpy。若样本不足则按可用数量采样。"""
        n = len(self.memory)
        if n == 0:
            raise ValueError("ReplayMemory is empty")
        B = min(batch_size, n)
        batch = random.sample(self.memory, B)
        return self._pack(batch)

    def sample_mixed(self, batch_size: int, priority_frac: float = 0.2):
        """
        混合采样：前 20%（可调）的高回报轨迹 + 随机样本，其余与 sample() 保持同形状。
        不依赖外部权重，不改变 self.memory 顺序。
        """
        n = len(self.memory)
        if n == 0:
            raise ValueError("ReplayMemory is empty")
        B = min(batch_size, n)
        Bp = int(max(0, min(priority_frac, 0.9)) * B)  # 保护 0~0.9
        Br = B - Bp

        # 构造奖励数组
        rewards = np.fromiter((itm[2] for itm in self.memory), dtype=np.float32, count=n)

        # 选 top-Bp 下标
        if Bp > 0 and n >= 50:
            if Bp < n:
                top_idx = np.argpartition(rewards, -Bp)[-Bp:]
            else:
                top_idx = np.arange(n)
            pri_idx = np.random.choice(top_idx, size=Bp, replace=False)
        else:
            pri_idx = np.empty((0,), dtype=np.int64)

        # 其余随机
        pool = np.setdiff1d(np.arange(n), pri_idx, assume_unique=False)
        if Br > 0:
            rnd_idx = np.random.choice(pool, size=Br, replace=False) if len(pool) >= Br else pool
        else:
            rnd_idx = np.empty((0,), dtype=np.int64)

        idx = np.concatenate([pri_idx, rnd_idx])
        np.random.shuffle(idx)
        batch = [self.memory[i] for i in idx]
        return self._pack(batch)

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
    专家标签缓存：存 (state, expert_action_phys)
    sample(B) -> (states, expert_actions_phys)  # 全部 numpy
    约定：expert_action 为**物理速度**（m/s）。Trainer 会用 v_pref*action_scale 归一化到 [-1,1]。
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
            _action_to_np(expert_action).astype(np.float32),  # 允许 ActionXY/array
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
        ae = np.stack([b[1] for b in batch]).astype(np.float32)  # 物理速度
        return s, ae

    # 兼容 Dataset
    def __getitem__(self, idx: int):
        return self.memory[idx]
