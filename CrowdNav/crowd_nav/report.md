没问题。下面给你一份**完整、干净、论文级**的 `SequenceReplayMemory`（单文件可直接替换 `memory.py`）。它只做三件事：**环形存储整条 episode → 按规则“无放回”抽取固定长度子序列 → 返回用于 n-step/1-step 的批次张量和有效掩码**。
同时把你关心的 1~4 点全部落地（尾窗扩大&末步降权、成功样本保底、timeout 判定统一、无放回采样）。

---

## ✅ 你能得到什么（接口保持克制、便于消融）

* `push_episode(...)`：一次写入一整条轨迹（states、rewards、dones、timeouts）。
* `set_sampling_params(...)`：一处设置所有采样旋钮（min_success_frac、max_timeout_frac、tail_window、tail_last_prob 等）。
* `sample(batch_size, seq_len, device)`：返回 **[B, T, …]** 的张量字典：
  `states, next_states, rewards, dones, timeouts, mask`，及 `mix_info`（本批次成功/超时占比等，便于打印）。
* 其余模块（trainer/policy/env）**不用改签名**；训练端按你现在的方式，在 `optimize_step` 里直接构造 **n-step 目标**即可。

---

## memory.py（完整实现）

```python
# -*- coding: utf-8 -*-
"""
SequenceReplayMemory: 环形序列回放（干净版）
- 环形存整条 episode
- 按“成功保底 / 超时上限 / 尾窗采样 / 无放回”规则抽取固定长度子序列
- 返回 [B, T, ...] 张量 + 有效掩码 mask，训练端据此构造 1-step 或 n-step TD

兼容点：
- set_sampling_params(...) 与你原来的命名一致
- sample(...) 仍从回放内部完成“子序列起点”的选择；训练端不需要知道尾窗细节
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
import torch


class SequenceReplayMemory:
    """
    论文级“干净”实现，不做批次后处理/重排/改写，只在采样入口做三段式控制与尾窗选择。

    分类语义（默认，可被 set_classify_threshold 改）：
      - success: episode 内存在 reward >= success_threshold（通常 1.0）
      - collision: episode 内存在 reward <= collision_threshold（通常 -0.25）
      - timeout: episode 终止 且 累计回报 <= timeout_threshold（通常等于 timeout_penalty）
      - running: 其余

    重要：训练端若将 timeout 视为“终止不引导”（bootstrap_on_timeout=False），
          则 n-step 目标构造时遇到 timeout 要停止引导（不接 V）。
    """

    # -------------------------- 初始化 & 配置 -------------------------- #
    def __init__(
        self,
        capacity_episodes: int,
        timeout_penalty: float = -0.5,   # 与环境/奖励保持一致
        success_threshold: float = 0.9,  # 识别“成功”的阈值
        collision_threshold: float = -0.25,  # 识别“碰撞”的阈值
        seed: Optional[int] = None,
    ):
        self.capacity = int(capacity_episodes)
        self.rng = random.Random(seed)
        self.episodes: List[Dict[str, np.ndarray]] = []
        self.pos = 0

        # 分类阈值（可被 set_classify_threshold 改）
        self.timeout_threshold = float(timeout_penalty) + 1e-8
        self.success_threshold = float(success_threshold)
        self.collision_threshold = float(collision_threshold)

        # 采样参数（可被 set_sampling_params 改）
        self.min_success_frac = 0.20   # 1) 成功样本保底（默认 20%）
        self.max_timeout_frac = 0.20   # 2) timeout 上限（默认 20%）
        self.il_bias = 1.0             # 若有 IL/teacher 标记，这里可>1 暂不使用时=1
        self.tail_window = 8           # 3) 尾窗覆盖（默认 8 步）
        self.tail_last_prob = 0.20     # 4) 末步偏置（默认 0.2）

    def set_classify_threshold(
        self,
        timeout_penalty: Optional[float] = None,
        success_threshold: Optional[float] = None,
        collision_threshold: Optional[float] = None,
    ):
        if timeout_penalty is not None:
            self.timeout_threshold = float(timeout_penalty) + 1e-8
        if success_threshold is not None:
            self.success_threshold = float(success_threshold)
        if collision_threshold is not None:
            self.collision_threshold = float(collision_threshold)

    def set_sampling_params(
        self,
        min_success_frac: Optional[float] = None,
        max_timeout_frac: Optional[float] = None,
        il_bias: Optional[float] = None,
        tail_window: Optional[int] = None,
        tail_last_prob: Optional[float] = None,
    ):
        if min_success_frac is not None:
            self.min_success_frac = float(min_success_frac)
        if max_timeout_frac is not None:
            self.max_timeout_frac = float(max_timeout_frac)
        if il_bias is not None:
            self.il_bias = float(il_bias)
        if tail_window is not None:
            self.tail_window = int(tail_window)
        if tail_last_prob is not None:
            self.tail_last_prob = float(tail_last_prob)

    # -------------------------- 写入 -------------------------- #
    def push_episode(
        self,
        states: np.ndarray,      # [T, *obs]
        rewards: np.ndarray,     # [T]
        dones: np.ndarray,       # [T] bool
        timeouts: np.ndarray,    # [T] bool（每步的 timeout 标记；没有就传 np.zeros(T)）
        meta: Optional[Dict] = None,  # 可选：例如 {"is_il": True}
    ):
        """
        以“整条 episode”为单位写入（干净、简单）。
        - 自动兼容变长 T；采样时再做等长子序列切片
        """
        T = int(len(rewards))
        assert states.shape[0] == T and dones.shape[0] == T and timeouts.shape[0] == T

        ep = dict(
            states=np.asarray(states),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_),
            timeouts=np.asarray(timeouts, dtype=np.bool_),
            length=T,
            meta=meta or {},
        )

        if len(self.episodes) < self.capacity:
            self.episodes.append(ep)
        else:
            self.episodes[self.pos] = ep
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self):
        return len(self.episodes)

    # -------------------------- 分类辅助 -------------------------- #
    def _classify_episode(self, ep: Dict) -> str:
        r = ep["rewards"]
        d = ep["dones"]
        # 成功：存在 ≥ success_threshold 的终止奖励（或任意时刻）
        if np.any(r >= self.success_threshold):
            return "success"
        # 碰撞：存在 ≤ collision_threshold 的奖励
        if np.any(r <= self.collision_threshold):
            # 有些环境把碰撞也置 done=True（与否不影响学习侧）
            return "collision"
        # 超时：终止且累计回报足够低
        if bool(d[-1]) and float(r.sum()) <= self.timeout_threshold:
            return "timeout"
        return "running"

    # -------------------------- 采样主体 -------------------------- #
    def _pick(self, pool: List[int], k: int) -> List[int]:
        """尽量无放回采样，数量不足时退回有放回，保证可用性。"""
        if len(pool) >= k:
            return random.sample(pool, k)
        if len(pool) == 0:
            return []
        return random.choices(pool, k=k)

    def _choose_start(self, T: int, seq_len: int, terminal: bool) -> int:
        """选择子序列起点：终止序列优先从尾窗中选；running 随机选。"""
        if T <= seq_len:
            return 0
        if terminal:
            # 尾窗起点范围的上界（含）
            tail_start_max = max(0, T - seq_len)
            left = max(0, T - self.tail_window - seq_len)
            right = tail_start_max
            if left > right:
                left = max(0, right - 1)
            if self.rng.random() < self.tail_last_prob:
                # 末步偏置：尽量让最后几步更常见，但不会只盯最后一格
                return right
            return self.rng.randint(left, right)
        # 非终止：均匀选择
        return self.rng.randint(0, T - seq_len)

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor | Dict]:
        """
        返回：
          states       [B, T, *obs]
          next_states  [B, T, *obs]
          rewards      [B, T]
          dones        [B, T]  (bool)
          timeouts     [B, T]  (bool)
          mask         [B, T]  (1/0，补齐位为 0)
          mix_info     dict（本批次成功/超时/碰撞/运行中占比，便于打印）
        """
        assert len(self.episodes) > 0, "Empty replay memory."

        # 1) 先按 episode 粒度做分类索引
        idx_success, idx_timeout, idx_collision, idx_running = [], [], [], []
        for i, ep in enumerate(self.episodes):
            c = self._classify_episode(ep)
            if c == "success":   idx_success.append(i)
            elif c == "timeout": idx_timeout.append(i)
            elif c == "collision": idx_collision.append(i)
            else:                idx_running.append(i)

        # 2) 计算本批次各类配额（成功保底 / timeout 上限）
        b = int(batch_size)
        succ_need = min(int(round(self.min_success_frac * b)), b)
        tout_cap  = min(int(round(self.max_timeout_frac * b)), b)

        # 先抽成功 / timeout，再用其余填满
        succ_idx = self._pick(idx_success, succ_need)
        tout_idx = self._pick(idx_timeout, min(tout_cap, b - len(succ_idx)))

        remain = b - len(succ_idx) - len(tout_idx)
        # 其余从 collision + running 混合里取
        misc_pool = idx_collision + idx_running
        misc_idx = self._pick(misc_pool, remain)

        chosen_episodes = succ_idx + tout_idx + misc_idx
        self.rng.shuffle(chosen_episodes)

        # 3) 逐条 episode 切出固定长度子序列，必要时做 padding
        # 先探测 obs 形状
        any_ep = self.episodes[chosen_episodes[0]]
        obs_shape = tuple(any_ep["states"].shape[1:])

        states  = np.zeros((b, seq_len) + obs_shape, dtype=any_ep["states"].dtype)
        nstates = np.zeros((b, seq_len) + obs_shape, dtype=any_ep["states"].dtype)
        rewards = np.zeros((b, seq_len), dtype=np.float32)
        dones   = np.zeros((b, seq_len), dtype=np.bool_)
        timeouts= np.zeros((b, seq_len), dtype=np.bool_)
        mask    = np.zeros((b, seq_len), dtype=np.float32)

        succ_cnt = len(succ_idx)
        tout_cnt = len(tout_idx)
        coll_cnt = sum(1 for i in misc_idx if i in idx_collision)

        for bi, epi in enumerate(chosen_episodes):
            ep = self.episodes[epi]
            T  = ep["length"]
            terminal = bool(ep["dones"][-1])  # 终止（成功/碰撞/超时）

            if T <= 0:
                continue

            start = self._choose_start(T, seq_len, terminal)
            end   = min(start + seq_len, T)
            K     = end - start  # 真正有效步数

            # 填充有效段
            states[bi, :K]   = ep["states"][start:end]
            rewards[bi, :K]  = ep["rewards"][start:end]
            dones[bi, :K]    = ep["dones"][start:end]
            timeouts[bi, :K] = ep["timeouts"][start:end]
            mask[bi, :K]     = 1.0

            # next_states：右移一格；末尾对齐 padding（被 mask=0 掩掉）
            if K >= 2:
                nstates[bi, :K-1] = ep["states"][start+1:end]
                # 最后一个有效 next_state：若 end<T，则可用真实下一帧；否则复制末帧（反正 mask 会屏蔽）
                if end < T:
                    nstates[bi, K-1] = ep["states"][end]
                else:
                    nstates[bi, K-1] = ep["states"][end-1]
            elif K == 1:
                nstates[bi, 0] = ep["states"][start]  # 占位

        # 4) 打包为 torch 张量 + mix 信息
        out = dict(
            states      = torch.as_tensor(states),
            next_states = torch.as_tensor(nstates),
            rewards     = torch.as_tensor(rewards),
            dones       = torch.as_tensor(dones),
            timeouts    = torch.as_tensor(timeouts),
            mask        = torch.as_tensor(mask),
            mix_info    = dict(
                batch=b,
                succ=succ_cnt,
                tout=tout_cnt,
                coll=coll_cnt,
                run = b - succ_cnt - tout_cnt - coll_cnt
            )
        )
        # 交给调用端 to(device)
        for k in ("states","next_states","rewards","dones","timeouts","mask"):
            out[k] = out[k].to(device, non_blocking=True)
        return out
```

---

## 需要你在外部“适配”的两处（非常小）

1. **训练端（optimize_step）构造 TD 目标**

   * 现在 `sample(...)` 已经给了 `[B,T,...] + mask`，你在 `optimize_step` 里按 n-step 公式滚动即可（遇到 `dones=True` 或 `timeouts=True 且 bootstrap_on_timeout=False` 就截断不引导）。
   * 这一步你之前已经有草稿，只要把 **mask** 用上，把 **timeouts 的 bootstrap 语义**统一即可。

2. **配置保持“一处生效”**

   * 把回放旋钮集中在 `[buffer]` / 或你指定的单一 section，启动时打印**真生效值**，防止重复键覆盖。

---

## 与你旧版 memory 的**关键差异（逐条对应 1~4）**

1. **成功尾段“可见度”**

   * 旧：`tail_window=4` 且 `tail_last_prob=0.5`，几乎只会抽到末 1~2 步 → 时序信用传不回去。
   * 新：默认 `tail_window=8`、`tail_last_prob=0.2`，覆盖倒数 8 步的更长推进段，同时不过度黏末步。

2. **正样本密度（成功保底）**

   * 旧：`min_success_frac=0.05`（每批仅 5% 成功）。
   * 新：默认 `min_success_frac=0.20`（20% 保底），timeout 上限 `max_timeout_frac=0.20`。这两者都可在 `set_sampling_params` 一处调。

3. **timeout 判定统一**

   * 旧：由 `timeout_penalty` 与 `timeout_threshold` 推断，若外部奖励或缩放与之不一致会错类。
   * 新：仍用阈值，但提供 `set_classify_threshold(timeout_penalty=你的真实惩罚)` 一处对齐（建议把**环境里真实的 -0.5/-0.6**传进来，避免错判）。

4. **批内多样性**

   * 旧：多个位置用 `random.choices`（有放回）→ 同一轨迹重复率高。
   * 新：优先 `random.sample`（无放回），不足才退回 `choices`，提升有效信息量，不改变算法行为。

**此外**：

* 采样仍返回**子序列窗口**（不是整个 episode），训练端可以直接 n-step；
* 没有“批次后重排/改写/限幅”之类的工业化步骤，保持**干净可复现**；
* `mix_info` 给你打印健康度（succ/tout/coll/run），方便像以前一样在日志里核对“采样是否按预期生效”。

---

## 建议的默认参数（你放在 config 里对应到 `set_sampling_params`）

```ini
# [buffer]
min_success_frac = 0.20
max_timeout_frac = 0.20
il_bias          = 1.0     # 暂不用 IL 倾斜时就设 1.0
tail_window      = 8
tail_last_prob   = 0.20

# [train]（与回放无关，放这里提醒你统一）
bootstrap_on_timeout = false   # n-step 构造时遇到 timeout 不引导
n_step = 3
```

---

## 接入小贴士（不“动坏”别处）

* 你现有地方若调用了 `memory.sample(batch, seq_len, device)` 并假定返回 dict（含张量），**不需要改签名**。
* 你原代码里若对“成功/timeout 统计”有打印，改为使用本函数返回的 `mix_info` 即可。
* 若你之前是“先抽单步，再在训练端拼序列”，直接替换成这版“回放抽子序列”即可（训练端更简单）。

---

> 这份 `SequenceReplayMemory` 就是“只把该做的事做到位”的实现：**统一 timeout 语义 → 成功尾段更可见 → 正样本密度到位 → 批内多样性更高**。配合你训练端的 n-step TD，按我们前面说的验收口径，`succ@50` 应能跨过 0.65 并继续爬升；若不达标，我们再仅通过这些参数做消融，而不用再改代码。


