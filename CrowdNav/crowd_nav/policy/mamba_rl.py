# -*- coding: utf-8 -*-
"""
MambaRL - Enhanced Version with Improved Spatial Encoding & Temporal Modeling

Key improvements:
1. Spatial Encoder: uses full token information + relational encoding + Attention
2. Motion augmentation: explicitly encodes velocity and acceleration
3. Longer temporal window: supports T=24+

"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import logging

# ========== 官方Mamba SSM导入 ==========
# 使用 Mamba-1 (兼容性最好，Docker内已验证可用)
# Mamba-2 需要 triton 库，暂不切换
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    MAMBA_SSM_AVAILABLE = True
    logging.info("[MAMBA-SSM] ✓ Successfully imported Mamba-1 (mamba_simple)")
except ImportError as e:
    MAMBA_SSM_AVAILABLE = False
    logging.error(f"[MAMBA-SSM] ✗ Failed to import mamba_ssm: {e}")
    logging.error("[MAMBA-SSM] Please install: pip install mamba-ssm")
    Mamba = None


class EnhancedSpatialEncoder(nn.Module):
    """
    增强空间编码器 v4

    特征：
    1. 完整13维token
    2. 关系特征：rel_x/y/dist + rel_vx/vy/speed + closing_speed + ttc_inv (8维)
    3. MultiheadAttention聚合人类特征
    4. 2路融合（robot + human含relation）
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # 机器人特征编码器（完整13维）
        self.robot_encoder = nn.Sequential(
            nn.Linear(13, d_model // 4),
            nn.ReLU(),
            nn.LayerNorm(d_model // 4)
        )

        # 人类特征编码器：13维token + 8维关系特征 = 21维
        self.human_encoder = nn.Sequential(
            nn.Linear(21, d_model // 4),
            nn.ReLU(),
            nn.LayerNorm(d_model // 4)
        )

        # 人类特征注意力聚合
        self.human_attention = nn.MultiheadAttention(
            embed_dim=d_model // 4,
            num_heads=4,
            batch_first=True
        )

        # Dropout正则化
        self.dropout_fusion = nn.Dropout(p=0.1)

        # 融合层：2路输入（robot + human含relation）
        self.fusion = nn.Sequential(
            nn.Linear(2 * (d_model // 4), d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, joint_state):
        B, T = joint_state.shape[:2]

        # 1. 机器人特征
        robot_tokens = joint_state[:, :, 0, :]  # [B, T, 13]
        robot_feat = self.robot_encoder(robot_tokens)  # [B, T, d/4]

        # 2. Entity features. Legacy inputs contain five rows; belief_v3
        # contains fifteen. The feature width and network parameters are
        # unchanged, so this dimension can remain a set dimension.
        humans_all = joint_state[:, :, 3:, :]  # [B, T, N, 13]
        N = humans_all.shape[2]
        if N < 1:
            raise ValueError("policy tokens contain no entity rows")
        entity_mask = (humans_all[..., 11] + humans_all[..., 12]) > 0.5
        distances = humans_all[..., 2]
        # Preserve the paper-1 path bit-for-bit. New wide contracts label every
        # real row and must keep zero padding out of sorting and attention.
        wide_contract = N != 5
        sort_distances = (distances.masked_fill(~entity_mask, float('inf'))
                          if wide_contract else distances)
        _, sorted_indices = torch.sort(sort_distances, dim=-1, descending=False)
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, 13)
        humans_tokens = torch.gather(humans_all, dim=2, index=sorted_indices_expanded)
        sorted_mask = torch.gather(entity_mask, dim=2, index=sorted_indices)

        # 3. 关系特征（8维：含closing_speed + ttc_inv）
        robot_px = robot_tokens[..., 0:1]
        robot_py = robot_tokens[..., 1:2]
        robot_vx = robot_tokens[..., 2:3]
        robot_vy = robot_tokens[..., 3:4]
        h_px = humans_tokens[..., 0]
        h_py = humans_tokens[..., 1]
        h_vx = humans_tokens[..., 3]
        h_vy = humans_tokens[..., 4]

        rel_x = h_px - robot_px
        rel_y = h_py - robot_py
        rel_dist = torch.sqrt(rel_x**2 + rel_y**2 + 1e-6)
        rel_vx = h_vx - robot_vx
        rel_vy = h_vy - robot_vy
        rel_speed = torch.sqrt(rel_vx**2 + rel_vy**2 + 1e-6)
        closing_speed = -(rel_x * rel_vx + rel_y * rel_vy) / (rel_dist + 1e-6)
        ttc_inv = torch.where(closing_speed > 0, closing_speed / (rel_dist + 1e-6), torch.zeros_like(closing_speed))

        rel_feat = torch.stack([rel_x, rel_y, rel_dist, rel_vx, rel_vy, rel_speed, closing_speed, ttc_inv], dim=-1)
        human_with_relation = torch.cat([humans_tokens, rel_feat], dim=-1)

        # 4. 编码 + Attention聚合
        hr_flat = human_with_relation.reshape(B * T * N, 21)
        hr_feats = self.human_encoder(hr_flat)
        human_feats_stack = hr_feats.reshape(B, T, N, -1)
        B, T, N, D_h = human_feats_stack.shape
        human_feats_flat = human_feats_stack.view(B * T, N, D_h)
        if wide_contract:
            flat_mask = sorted_mask.reshape(B * T, N)
            # MultiheadAttention cannot accept a row whose every key is
            # masked. Temporarily expose one zero row, then zero the aggregate.
            any_entity = flat_mask.any(dim=1)
            safe_mask = flat_mask.clone()
            safe_mask[~any_entity, 0] = True
            attn_out, _ = self.human_attention(
                query=human_feats_flat, key=human_feats_flat,
                value=human_feats_flat, key_padding_mask=~safe_mask)
            attn_out = attn_out.masked_fill(~safe_mask.unsqueeze(-1), float('-inf'))
            human_global = attn_out.max(dim=1)[0]
            human_global = torch.where(
                any_entity.unsqueeze(-1), human_global,
                torch.zeros_like(human_global))
        else:
            attn_out, _ = self.human_attention(
                query=human_feats_flat, key=human_feats_flat,
                value=human_feats_flat)
            human_global = attn_out.max(dim=1)[0]
        human_global = human_global.view(B, T, D_h)

        # 5. 融合
        all_features = torch.cat([robot_feat, human_global], dim=-1)
        all_features = self.dropout_fusion(all_features)
        output = self.fusion(all_features)
        return output


class MambaTemporalEncoder(nn.Module):
    """
    时序编码器：使用官方mamba_ssm实现
    """
    def __init__(self, d_model: int = 256, n_layers: int = 2, **mamba_kwargs):
        super().__init__()
        self.backbone_name = 'mamba'
        self.n_layers = n_layers

        if not MAMBA_SSM_AVAILABLE or Mamba is None:
            raise RuntimeError(
                "[MAMBA-SSM] ✗ mamba_ssm库不可用！\n"
                "请安装：pip install mamba-ssm\n"
                "不支持fallback - 仅使用纯Mamba实现"
            )

        self.backend = nn.ModuleList([
            nn.Sequential(
                Mamba(d_model=d_model, **mamba_kwargs),
                nn.LayerNorm(d_model)
            )
            for _ in range(n_layers)
        ])
        self.use_fast_path = self.backend[0][0].use_fast_path if len(self.backend) > 0 else False

    def forward(self, x):
        """
        x: [B, T, d_model]
        Returns: [B, T, d_model]
        """
        for block in self.backend:
            x = x + block(x)  # Residual connection
        return x


class GRUTemporalEncoder(nn.Module):
    """GRU时序编码器：保留用于对照实验，不作为当前默认主干。"""
    def __init__(self, d_model: int = 256, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.backbone_name = 'gru'
        self.n_layers = n_layers
        gru_dropout = dropout if n_layers > 1 else 0.0
        self.backend = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y, _ = self.backend(x)
        return self.norm(y)


class MLPTemporalEncoder(nn.Module):
    """Per-frame backbone with no hidden state or temporal communication."""
    def __init__(self, d_model: int = 256, dropout: float = 0.0):
        super().__init__()
        self.backbone_name = 'mlp'
        self.backend = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.backend(x))


def build_temporal_encoder(
    backbone: str,
    d_model: int,
    n_layers: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dropout: float = 0.0,
):
    backbone = str(backbone).strip().lower()
    if backbone == 'mamba':
        return MambaTemporalEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    if backbone == 'gru':
        return GRUTemporalEncoder(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
    if backbone == 'mlp':
        return MLPTemporalEncoder(d_model=d_model, dropout=dropout)
    raise ValueError(
        f"Unsupported temporal_backbone='{backbone}'. "
        "Expected 'mamba', 'gru', or 'mlp'.")


class MambaRLPolicy(nn.Module):
    """MambaRL策略网络：DoubleQ离散动作架构

    架构：
    - Spatial Encoder: 空间特征提取（robot + humans + relations）
    - Temporal Encoder: Mamba时序建模
    - Q-head: 80维离散动作Q值输出（5 speeds × 16 headings）

    支持两种训练模式：
    - BC模式：单步预测，跳过时序组件（快速）
    - RL模式：完整时序建模（高性能）

    """
    def __init__(self, config=None, device='cpu'):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device

        if not (config and hasattr(config, 'getint') and hasattr(config, 'getfloat')):
            raise ValueError("MambaRLPolicy requires a config with getint/getfloat (no fallback defaults).")

        # Network config (strictly from config)
        d_model = config.getint('mamba', 'd_model')
        n_layers = config.getint('mamba', 'n_layers')
        d_state = config.getint('mamba', 'd_state')
        d_conv = config.getint('mamba', 'd_conv')
        expand = config.getint('mamba', 'expand')
        temporal_backbone = config.get('mamba', 'temporal_backbone', fallback='mamba').strip().lower()
        temporal_dropout = config.getfloat('mamba', 'dropout', fallback=0.0)
        self.temporal_backbone = temporal_backbone
        self.v_max = config.getfloat('robot', 'v_pref')
        self.seq_len = config.getint('buffer', 'seq_len')
        if config.has_option('env', 'time_step'):
            self.env_dt = config.getfloat('env', 'time_step')
        elif config.has_option('action_space', 'time_step'):
            self.env_dt = config.getfloat('action_space', 'time_step')
        else:
            raise ValueError("Missing time_step in config (expected [env] time_step or [action_space] time_step).")

        # Build network: Spatial -> Temporal -> Heads
        self.spatial_encoder = EnhancedSpatialEncoder(d_model)
        self.temporal_encoder = build_temporal_encoder(
            temporal_backbone,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=temporal_dropout,
        )

        # ========== Discrete Action DoubleQ Architecture ==========
        # Removed continuous actor (action_mean, action_log_std)
        # Use 80-dim Q-head (5 speeds x 16 headings = 80 actions)

        # Read action space size from config (strict)
        n_speeds = config.getint('policy', 'n_speeds')
        n_headings = config.getint('policy', 'n_headings')
        include_stop = config.getboolean('policy', 'include_stop', fallback=False)
        self.include_stop = include_stop

        self.n_actions = n_speeds * n_headings + (1 if include_stop else 0)

        # Q-head: Output Q-value for each discrete action
        self.q_head = nn.Linear(d_model, self.n_actions)

        # Value-head: Output single scalar value (SARL-style)
        self.value_head = nn.Linear(d_model, 1)

        logging.info(f"[DISCRETE-Q] Q-head initialized: {self.n_actions} actions ({n_speeds} speeds × {n_headings} headings, stop={include_stop})")
        logging.info(f"[VALUE-HEAD] Value head initialized (SARL-style, 1-dim output)")

        # 🔥 Auxiliary Head: Predict future positions of 3 nearest neighbors for next 4 steps (self-supervised)
        # Input: temporal features [B, T, d_model]
        # Output: [B, T, 3, 4, 2] - 3 neighbors x 4 future steps x (px, py)
        self.future_pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3 * 4 * 2)  
        )
        logging.info(f"[AUXILIARY-HEAD] Future position predictor initialized: 3 neighbors × 4 future steps")

        self._phase = 'train'

        # Environment interface compatibility: Single agent training
        self.multiagent_training = False

        # SARL-style attributes (strict from config)
        self.gamma = config.getfloat('train', 'gamma')
        if config.has_option('action_space', 'time_step'):
            self.time_step = config.getfloat('action_space', 'time_step')
        elif config.has_option('env', 'time_step'):
            self.time_step = config.getfloat('env', 'time_step')
        else:
            raise ValueError("Missing time_step in config (expected [action_space] time_step or [env] time_step).")
        self.success_radius = config.getfloat('robot', 'success_radius')

        # Reward参数（统一参数名，与env.config一致）
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.progress_reward = config.getfloat('reward', 'progress_reward', fallback=0.0)
        self.time_penalty = config.getfloat('reward', 'time_penalty', fallback=0.0)
        self.stand_penalty = config.getfloat('reward', 'stand_penalty', fallback=0.0)

        # ε从config读取（无默认值）
        if config.has_section('sarl') and config.has_option('sarl', 'epsilon_start'):
            self.epsilon = config.getfloat('sarl', 'epsilon_start')
        elif config.has_section('doubleq') and config.has_option('doubleq', 'epsilon_start'):
            self.epsilon = config.getfloat('doubleq', 'epsilon_start')
        else:
            raise ValueError("Missing epsilon_start in config ([sarl] or [doubleq])")
        self.action_space = None  # 动作空间（延迟初始化）
        self.last_state = None  # 用于训练时存储state
        self.use_sarl_predict = False  # 是否使用SARL-style predict（默认False，使用DoubleQ）
        self._q_networks_loaded = False  # 是否加载过Q网络（离散搜索兼容）

        # Lookahead ablation mode (test-time decomposition):
        # - full:        r + gamma * V(s')
        # - reward_only: r
        # - value_only:  gamma * V(s')
        mode = os.getenv("MAMBA_LOOKAHEAD_ABLATION", "full").strip().lower()
        if mode not in ("full", "reward_only", "value_only"):
            logging.warning(f"[ABLATION] Invalid MAMBA_LOOKAHEAD_ABLATION='{mode}', fallback to 'full'")
            mode = "full"
        self.lookahead_ablation_mode = mode
        logging.info(f"[ABLATION] Lookahead scoring mode: {self.lookahead_ablation_mode}")

        # ==== Test-time safety tweaks (eval-only) ====
        # 默认值为0 -> 不启用；只在 test/val/eval 阶段生效
        if config.has_section('eval_protocol'):
            self.test_min_clearance = config.getfloat('eval_protocol', 'safety_margin', fallback=
                                                     config.getfloat('eval_protocol', 'discomfort_threshold', fallback=0.0))
            self.test_risk_lambda = config.getfloat('eval_protocol', 'risk_lambda', fallback=0.0)
            self.test_action_smoothing = config.getfloat('eval_protocol', 'action_smoothing', fallback=0.0)
        else:
            self.test_min_clearance = 0.0
            self.test_risk_lambda = 0.0
            self.test_action_smoothing = 0.0
        self._last_action = None  # for test-time smoothing
        self._last_action_index = None

        # Training mode: 'bc' (fast single-step prediction, skip temporal) or 'rl' (full temporal modeling)
        self.training_mode = 'rl'  # Default RL mode

        # History buffer: maintain seq_len frames to match BC training input distribution
        from collections import deque
        self._history = deque(maxlen=self.seq_len)  
        self._zero_token = np.zeros((8, 13), dtype=np.float32)

        # Move all modules to target device
        self.to(self.device)

        # SSOT verification log
        logging.info("=" * 80)
        logging.info(f"[TEMPORAL-BACKBONE] {temporal_backbone.upper()} backbone configuration:")
        logging.info(f"  d_model={d_model}, n_layers={n_layers}, d_state={d_state}")
        logging.info(f"  d_conv={d_conv}, expand={expand}")
        logging.info(f"[SEQ-FINAL] Sequence Length Configuration (SSOT):")
        logging.info(f"  seq_len={self.seq_len} (from train.config:[buffer])")
        logging.info(f"  T={self.seq_len} (should match policy.config:[temporal].T)")
        logging.info(f"[ARCH-FINAL] Architecture: EnhancedSpatial(3-way:robot+human+relation) -> {temporal_backbone.upper()} -> DualHead")
        logging.info(f"[DEVICE] {self.device}")
        logging.info("=" * 80)

    def set_env_dt(self, dt: float):
        self.env_dt = dt
        logging.info(f"[MAMBA-RL] Set env_dt={dt}")

    def set_phase(self, phase: str):
        self._phase = phase

    def set_training_mode(self, mode: str):
        """
        设置训练模式

        Args:
            mode: 'bc' (快速单步预测，跳过时序) 或 'rl' (完整时序建模)
        """
        assert mode in ['bc', 'rl'], f"Invalid training mode: {mode}, must be 'bc' or 'rl'"
        self.training_mode = mode
        logging.info(f"[TRAINING-MODE] Switched to '{mode}' mode")

    def reset_episode_stats(self):
        """清空历史缓存，在每个 episode 开头调用（Explorer 自动调用）"""
        self._history.clear()
        self._last_action = None
        self._last_action_index = None

    @staticmethod
    def _clip_unit_ball(action_tensor: torch.Tensor) -> torch.Tensor:
        """将归一化动作裁剪到单位球，保证log_prob与实际执行动作一致"""
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        norm = action_tensor.norm(dim=-1, keepdim=True)
        safe_norm = torch.clamp(norm, min=1e-6)
        scale = torch.clamp(1.0 / safe_norm, max=1.0)  # norm>1时缩放到1
        clipped = action_tensor * scale

        return clipped.squeeze(0) if squeeze_back else clipped

    def forward_value(self, joint_tokens, mask=None):
        """Critic前向传播：总是使用完整时序编码"""
        # 处理输入维度：确保是 [B, T, 8, 13]
        if joint_tokens.dim() == 3:
            # [B, 8, 13] - 单步，添加时序维度
            joint_tokens = joint_tokens.unsqueeze(1)  # [B, 1, 8, 13]
        elif joint_tokens.dim() == 4:
            # [B, T, 8, 13] - 已是正确格式
            pass
        else:
            raise ValueError(f"Expected [B, 8, 13] or [B, T, 8, 13], got {joint_tokens.shape}")

        # 空间编码 → 时序建模 → 价值预测
        spatial_features = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        temporal_features = self.temporal_encoder(spatial_features)  # [B, T, d_model]

        # 取最后一帧特征预测价值
        last_features = temporal_features[:, -1, :]  # [B, d_model]
        values = self.value_head(last_features).squeeze(-1)  # [B]

        return values

    def forward_q(self, joint_tokens, return_sequence=False):
        """DoubleQ前向传播：输出离散动作的Q值

        Args:
            joint_tokens: [B, T, 8, 13] or [B, 8, 13]
            return_sequence: 是否返回整个序列（训练时True，推理时False）

        Returns:
            如果return_sequence=True（训练）:
                q_values: [B, T, 80] - 每个时间步的80个离散动作的Q值
            如果return_sequence=False（推理）:
                q_values: [B, 80] - 最后一帧的80个离散动作的Q值
        """
        # 确保输入在正确的设备上
        if joint_tokens.device != next(self.parameters()).device:
            joint_tokens = joint_tokens.to(next(self.parameters()).device)

        # 处理输入维度
        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)  # [B, 1, 8, 13]
        elif joint_tokens.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected [B, 8, 13] or [B, T, 8, 13], got {joint_tokens.shape}")

        # 完整时序建模（当前GRU版本同样依赖完整序列）
        spatial_features = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        temporal_features = self.temporal_encoder(spatial_features)  # [B, T, d_model]

        if return_sequence:
            # 训练：返回整个序列的Q值 [B, T, 80]
            q_values = self.q_head(temporal_features)  # [B, T, 80]
        else:
            # 推理：只返回最后一帧的Q值 [B, 80]
            last_features = temporal_features[:, -1, :]  # [B, d_model]
            q_values = self.q_head(last_features)  # [B, 80]

        return q_values

    def act(self, state):
        """推理接口：使用argmax选择最优离散动作

        Args:
            state: JointState或状态数组

        Returns:
            Tuple[ActionXY, int]: (最优动作, 动作索引)
        """
        # 🔥 修复：SARL模式分流到predict_sarl_style
        if self.use_sarl_predict:
            action = self.predict_sarl_style(state)
            return action, self._last_action_index

        from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized, discrete_index_to_action
        from crowd_sim.envs.utils.action import ActionXY

        # 1. 把当前 state 转成 6×13 token
        if hasattr(state, 'to_array'):
            state_34 = state.to_array()
        elif isinstance(state, (list, tuple)):
            # 处理列表/元组：可能是多个ObservableState
            if len(state) > 0 and hasattr(state[0], 'to_array'):
                # 展平所有ObservableState
                state_34 = np.concatenate([s.to_array() if hasattr(s, 'to_array') else np.array(s, dtype=np.float32) for s in state])
            else:
                state_34 = np.array(state, dtype=np.float32).flatten()
        else:
            try:
                state_34 = np.array(state, dtype=np.float32).flatten()
            except (TypeError, ValueError) as e:
                raise TypeError(f"Cannot convert state to array: type={type(state)}, value={state}") from e

        # 🔥 修复：验证维度并padding到34D（处理动态人类数量/维度不匹配）
        state_34 = state_34.flatten()  # 确保1D
        if state_34.size < 34:
            # 状态不足34维：padding到34D
            pad = np.zeros(34 - state_34.size, dtype=np.float32)
            state_34 = np.concatenate([state_34, pad])
        elif state_34.size > 34:
            # 状态超过34维：截断到34D
            state_34 = state_34[:34]

        state_token = _batch_joint34_to_tokens_vectorized(state_34.reshape(1, -1))[0]

        # 2. 压入历史缓存
        self._history.append(state_token.copy())

        # 3. 组织成 [seq_len, 8, 13] 的序列（🔥 修复：与BC训练保持一致，用第一帧填充而非zero_token）
        history_list = list(self._history)
        if len(history_list) < self.seq_len:
            # 用当前第一帧重复填充（匹配BC训练时的pad_frame逻辑）
            pad_frame = history_list[0]
            pad_needed = self.seq_len - len(history_list)
            history_list = [pad_frame.copy() for _ in range(pad_needed)] + history_list

        history_list = history_list[-self.seq_len:]
        history_tensor = np.array(history_list, dtype=np.float32)

        # NumPy 2.x兼容性处理
        try:
            history_tensor = torch.from_numpy(history_tensor).float().to(self.device)
        except RuntimeError:
            # NumPy 2.x + 旧版PyTorch兼容方案
            history_tensor = torch.tensor(history_tensor, dtype=torch.float32, device=self.device)

        history_tensor = history_tensor.unsqueeze(0)  # [1, seq_len, 8, 13]

        # 4. 前向传播获取Q值
        self.eval()
        with torch.no_grad():
            # forward_q输出Q值 [1, 80]
            q_values = self.forward_q(history_tensor, return_sequence=False)  # [1, 80]

            # ✅ DoubleQ核心：ε-greedy探索
            if self.epsilon > 0 and np.random.random() < self.epsilon:
                # 随机探索
                action_idx = np.random.randint(0, q_values.shape[-1])
            else:
                # Argmax选择最优动作
                action_idx = q_values.argmax(dim=-1).item()  # int in [0, 79]

            # 转换为(vx, vy)
            vx, vy = discrete_index_to_action(action_idx)

        # ✅ 修复：返回 (action, idx) 元组
        return ActionXY(vx, vy), action_idx

    def predict(self, state, deterministic: bool = True, return_idx: bool = False, **kwargs):
        """兼容接口：根据use_sarl_predict选择预测方式

        Args:
            state: JointState或状态数组
            deterministic: True时强制ε=0（评估用），False时使用当前ε
            return_idx: 是否同时返回离散动作索引
        """
        # Order 17 item 14: an occlusion arm may only run on value-lookahead.
        # Direct-Q was measured at 18.5% success against 87.9% for lookahead,
        # so letting an arm fall through to it would confound the experiment
        # with a known-broken interface. Refuse rather than run.
        if getattr(state, 'policy_entities', None) is not None and not self.use_sarl_predict:
            raise RuntimeError(
                "occlusion arms require value-lookahead; set use_sarl_predict=True "
                "(direct-Q is refused for occlusion experiments)")
        # 如果启用SARL-style predict，使用one-step lookahead
        if self.use_sarl_predict:
            action = self.predict_sarl_style(state)
            if return_idx:
                return action, self._last_action_index
            return action

        # 否则使用原有的DoubleQ方式（argmax Q值）
        eps_prev = self.epsilon
        if deterministic:
            self.epsilon = 0.0
        try:
            action, action_idx = self.act(state)
        finally:
            self.epsilon = eps_prev

        if return_idx:
            return action, action_idx
        return action

    def set_epsilon(self, epsilon: float):
        """设置ε-greedy探索率（由trainer调用）

        Args:
            epsilon: 探索率 [0, 1]
        """
        self.epsilon = epsilon

    # ============ SARL-style Methods ============

    def build_action_space(self, v_pref):
        """构建离散动作表（与GRID同源）

        - 支持 exponential/even 采样
        - 支持 include_stop=True 时自动包含 stop 动作
        """
        from crowd_sim.envs.utils.action import ActionXY
        from crowd_nav.contracts import discrete_index_to_action, GRID, grid_action_dim

        # 从GRID读取动作数量
        n_actions = grid_action_dim(GRID)

        action_space = []
        for idx in range(n_actions):
            vx, vy = discrete_index_to_action(idx)
            action_space.append(ActionXY(vx, vy))

        self.action_space = action_space
        logging.info(f"[ACTION-SPACE] Built {len(action_space)} actions (GRID {GRID['n_speeds']}×{GRID['n_headings']}, stop={GRID.get('include_stop', False)})")

    def continuous_to_discrete_index(self, action):
        """将连续动作转换为最近的离散动作索引（用于IL prefill）

        Args:
            action: ActionXY对象或(vx, vy)元组

        Returns:
            int: 离散动作索引
        """
        from crowd_sim.envs.utils.action import ActionXY

        # 确保action_space已构建
        if self.action_space is None:
            self.build_action_space(1.0)  # 使用默认v_pref

        # 提取vx, vy
        if isinstance(action, ActionXY):
            vx, vy = action.vx, action.vy
        elif isinstance(action, (tuple, list)) and len(action) >= 2:
            vx, vy = action[0], action[1]
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        # 找到最近的离散动作
        min_dist = float('inf')
        best_idx = 0
        for idx, discrete_action in enumerate(self.action_space):
            dist = (discrete_action.vx - vx)**2 + (discrete_action.vy - vy)**2
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        return best_idx

    def propagate(self, state, action):
        """状态传播（SARL-style）"""
        from crowd_sim.envs.utils.state import ObservableState, FullState

        if isinstance(state, ObservableState):
            # 人类状态传播
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            return ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # 机器人状态传播
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            return FullState(next_px, next_py, action.vx, action.vy, state.radius,
                            state.gx, state.gy, state.v_pref, state.theta)
        else:
            raise ValueError(f"Unknown state type: {type(state)}")

    def compute_reward(self, nav, humans, prev_nav=None, action=None):
        """计算即时reward（与env.step精简版一致）"""
        # 碰撞检测
        dmin = float('inf')
        collision = False
        for human in humans:
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            dmin = min(dmin, dist)

        # 到达目标检测
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < self.success_radius

        # reward计算
        if collision:
            return float(self.collision_penalty)
        elif reaching_goal:
            return float(self.success_reward)
        else:
            reward = 0.0
            # progress reward
            if prev_nav is not None:
                prev_dist = np.linalg.norm((prev_nav.px - prev_nav.gx, prev_nav.py - prev_nav.gy))
                curr_dist = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy))
                progress = prev_dist - curr_dist
                reward += self.progress_reward * progress
            # time penalty
            reward += self.time_penalty
            # stand penalty
            if action is not None:
                robot_speed = np.linalg.norm([action.vx, action.vy])
                if robot_speed < 0.05:
                    reward += self.stand_penalty
            # discomfort penalty (线性，与env一致)
            if dmin < self.discomfort_dist:
                penalty = self.discomfort_penalty_factor * (self.discomfort_dist - dmin) * self.time_step
                reward -= penalty
            return reward

    def reach_destination(self, state):
        """检查是否到达目标（🔥 使用success_radius）"""
        self_state = state.self_state if hasattr(state, 'self_state') else state
        return np.linalg.norm((self_state.px - self_state.gx, self_state.py - self_state.gy)) < self.success_radius

    def _build_joint_state_34(self, robot_state, human_states):
        """构建34维joint state"""
        # robot: 9维 [px, py, vx, vy, radius, gx, gy, v_pref, theta]
        robot_arr = np.array([
            robot_state.px, robot_state.py, robot_state.vx, robot_state.vy,
            robot_state.radius, robot_state.gx, robot_state.gy,
            robot_state.v_pref, robot_state.theta
        ], dtype=np.float32)

        # humans: 5×5维 [px, py, vx, vy, radius] × 5
        human_arr = []
        for h in human_states[:5]:  # 最多5个人
            human_arr.extend([h.px, h.py, h.vx, h.vy, h.radius])

        # padding到25维
        while len(human_arr) < 25:
            human_arr.append(0.0)

        return np.concatenate([robot_arr, human_arr[:25]])

    # ------------------------------------------------------------------
    # Order 17 item 12: ONE entry point for turning a state into policy
    # tokens. act(), transform() and predict_sarl_style() all go through it so
    # an arm cannot leak in through a forgotten code path.
    # ------------------------------------------------------------------
    def _policy_entities(self, state):
        """Entity dicts the policy is allowed to condition on.

        With occlusion off this is every human with certainty 1, which
        reproduces the legacy semantics exactly.
        """
        ents = getattr(state, 'policy_entities', None)
        if ents is not None:
            return ents
        return [{"px": h.px, "py": h.py, "vx": h.vx, "vy": h.vy,
                 "radius": h.radius, "p_exist": 1.0, "uncertainty": 0.0,
                 "visible": 1.0, "hidden": 0.0} for h in state.human_states]

    def _state_to_policy_tokens(self, state):
        """Policy tokens for the current state. Falls back to the legacy
        converter when there is no occlusion payload, so the untouched path
        stays bit-identical."""
        from crowd_nav.contracts import (_batch_joint34_to_tokens_vectorized,
                                         entities_to_tokens)
        if getattr(state, 'policy_entities', None) is None:
            s34 = self._build_joint_state_34(state.self_state, state.human_states)
            return _batch_joint34_to_tokens_vectorized(s34.reshape(1, -1))[0]
        tokens = state.to_policy_tokens()
        if tokens is None:
            raise RuntimeError("occlusion state lost its policy-token payload")
        return tokens

    def _successor_tokens(self, state, next_self, entities):
        """Successor tokens under a candidate action. Visible pedestrians and
        believed occupancy modes are propagated over the SAME time step; a
        believed mode keeps its existence probability and uncertainty, so the
        value network sees how much of the successor is guesswork."""
        from crowd_nav.contracts import entities_to_tokens
        dt = self.time_step
        nxt = [{"px": e["px"] + e["vx"] * dt, "py": e["py"] + e["vy"] * dt,
                "vx": e["vx"], "vy": e["vy"], "radius": e["radius"],
                "p_exist": e.get("p_exist", 1.0),
                "uncertainty": e.get("uncertainty", 0.0),
                "visible": e.get("visible", 1.0),
                "hidden": e.get("hidden", 0.0)} for e in entities]
        # The legacy path propagates the current first five humans and keeps
        # their order for every candidate action. Do not reselect at t+1.
        return entities_to_tokens(
            next_self,
            nxt,
            preselected=True,
            belief_features=getattr(state, 'belief_features', 'full'),
            token_contract=getattr(state, 'token_contract', 'legacy_top5'),
            visible_slots=getattr(state, 'visible_slots', 5),
            hidden_slots=getattr(state, 'hidden_slots', 10),
        ), nxt

    def predict_sarl_style(self, state):
        """SARL-style: 枚举动作 + one-step lookahead + batch化加速"""
        from crowd_sim.envs.utils.action import ActionXY
        from crowd_sim.envs.utils.state import ObservableState as _ObservableState
        from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized

        # 1. 到达目标检查
        if self.reach_destination(state):
            return ActionXY(0, 0)

        # 2. 构建动作空间
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        # 3. ε-greedy
        if self._phase == 'train' and np.random.random() < self.epsilon:
            random_idx = int(np.random.choice(len(self.action_space)))
            self._last_action_index = random_idx
            return self.action_space[random_idx]

        # 4. 🔥 修复：先把当前state token加入历史（Mamba需要当前帧）
        # Order 17 item 12: the single entry point. Identical to the previous
        # two lines when there is no occlusion payload.
        current_token = self._state_to_policy_tokens(state)
        _pol_ents = self._policy_entities(state)
        _occ_on = getattr(state, 'policy_entities', None) is not None
        _token_ents = _pol_ents
        if _occ_on:
            from crowd_nav.contracts import select_entities_by_contract
            _token_ents = select_entities_by_contract(
                state.self_state, _pol_ents,
                token_contract=getattr(state, 'token_contract', 'legacy_top5'),
                visible_slots=getattr(state, 'visible_slots', 5),
                hidden_slots=getattr(state, 'hidden_slots', 10))

        # 构建基础历史（包含当前帧）
        base_hist = list(self._history) + [current_token]

        # 5. 🔥 Batch化lookahead（关键性能优化）
        n_actions = len(self.action_space)
        next_states_batch = []  # 存储所有next_state的tokens
        rewards_batch = []      # 存储所有即时reward
        dmins_batch = []        # 存储最小安全距离（用于测试期安全过滤）

        for action in self.action_space:
            # 5.1 Propagate
            next_self = self.propagate(state.self_state, action)
            # The immediate reward and the clearance test must stay on VISIBLE
            # geometry only: a believed pedestrian is not an observed one, and
            # letting it fire the collision penalty would be a risk shield by
            # the back door. Hidden risk reaches the decision only through the
            # value network, via columns 9-12 of the successor token.
            if _occ_on:
                _vis = [e for e in _pol_ents if e.get("visible", 1.0) > 0.5]
                next_humans = [self.propagate(
                    _ObservableState(e["px"], e["py"], e["vx"], e["vy"], e["radius"]),
                    ActionXY(e["vx"], e["vy"])) for e in _vis]
            else:
                next_humans = [self.propagate(h, ActionXY(h.vx, h.vy)) for h in state.human_states]

            # 5.2 计算即时reward（🔥 传入action用于stand_penalty）
            reward = self.compute_reward(next_self, next_humans, prev_nav=state.self_state, action=action)
            rewards_batch.append(reward)

            # 5.2.1 计算最小安全距离（测试期安全过滤）
            dmin = float('inf')
            for h in next_humans:
                dist = np.hypot(next_self.px - h.px, next_self.py - h.py) - next_self.radius - h.radius
                if dist < dmin:
                    dmin = dist
            dmins_batch.append(dmin)

            # 5.3 构建next_state token
            if _occ_on:
                next_token, _ = self._successor_tokens(state, next_self, _token_ents)
            else:
                next_state_34 = self._build_joint_state_34(next_self, next_humans)
                next_token = _batch_joint34_to_tokens_vectorized(next_state_34.reshape(1, -1))[0]

            # 5.4 组织序列：base_hist（含当前帧）+ next_token
            temp_hist = base_hist + [next_token]
            temp_hist = temp_hist[-self.seq_len:]
            if len(temp_hist) < self.seq_len:
                temp_hist = [temp_hist[0]] * (self.seq_len - len(temp_hist)) + temp_hist

            next_states_batch.append(np.array(temp_hist))

        # 6. 🔥 Batch forward（一次推理所有动作）
        next_states_tensor = torch.from_numpy(
            np.array(next_states_batch)).float().to(self.device)
        with torch.no_grad():
            next_values = self.forward_value(next_states_tensor)  # [80]

        # 7. 计算总value（支持test-time分解消融）
        rewards_tensor = torch.tensor(rewards_batch, device=self.device)
        # 使用与训练一致的折扣因子，避免 value 标度不一致
        gamma_bar = self.gamma
        if self.lookahead_ablation_mode == "reward_only":
            total_values = rewards_tensor  # [80]
        elif self.lookahead_ablation_mode == "value_only":
            total_values = gamma_bar * next_values  # [80]
        else:
            total_values = rewards_tensor + gamma_bar * next_values  # [80]

        # 7.1 测试期安全过滤（不影响训练）
        if self._phase in ('test', 'val', 'eval') and (self.test_min_clearance > 0.0 or self.test_risk_lambda > 0.0):
            dmins_tensor = torch.tensor(dmins_batch, device=self.device)
            if self.test_min_clearance > 0.0:
                safe_mask = dmins_tensor >= self.test_min_clearance
                if safe_mask.any():
                    total_values = total_values.masked_fill(~safe_mask, -1e9)
            if self.test_risk_lambda > 0.0:
                risk_margin = self.test_min_clearance if self.test_min_clearance > 0.0 else self.discomfort_dist
                total_values = total_values - self.test_risk_lambda * torch.clamp(risk_margin - dmins_tensor, min=0.0)

        # 🔥 Tie-break: discourage STOP when values are nearly equal
        if total_values.numel() > 0:
            total_values[0] -= 1e-3

        # 8. 选择最优动作
        best_idx = total_values.argmax().item()
        best_action = self.action_space[best_idx]
        # Preserve the exact grid index selected by value lookahead. Online
        # replay must not reverse-engineer it from the continuous action.
        self._last_action_index = int(best_idx)

        # 9. 更新历史（添加当前帧到_history）
        self._history.append(current_token)

        # 10. 测试期动作平滑（减少抖动）
        if self._phase in ('test', 'val', 'eval') and self.test_action_smoothing > 0.0:
            if self._last_action is not None:
                alpha = float(self.test_action_smoothing)
                best_action = ActionXY(
                    alpha * self._last_action.vx + (1.0 - alpha) * best_action.vx,
                    alpha * self._last_action.vy + (1.0 - alpha) * best_action.vy
                )
            self._last_action = best_action

        return best_action

    # ============ 方案1：离散动作搜索 + 密度门控 ============

    def load_q_networks(self, checkpoint_path: str):
        """加载Q网络权重用于离散动作搜索"""
        import os
        if not os.path.exists(checkpoint_path):
            logging.warning(f"[Q-NET] Checkpoint not found: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 检查是否有Q网络权重（支持两种key格式）
            q1_key = 'q_net1' if 'q_net1' in checkpoint else 'q_net1_state'
            q2_key = 'q_net2' if 'q_net2' in checkpoint else 'q_net2_state'
            if q1_key not in checkpoint or q2_key not in checkpoint:
                logging.warning(f"[Q-NET] Checkpoint missing Q network weights (tried {q1_key}, {q2_key})")
                return False

            # 从checkpoint自动推断Q网络架构
            # 检测n_layers：数backend.X.0的数量
            q1_state = checkpoint[q1_key]
            n_layers = sum(1 for k in q1_state.keys() if k.startswith('temporal_encoder.backend.') and '.0.A_log' in k)
            n_layers = max(n_layers, 2)  # 至少2层

            # 检测d_state：从A_log的shape推断
            for k, v in q1_state.items():
                if 'A_log' in k:
                    d_state = v.shape[1]  # A_log shape is [d_inner, d_state]
                    break
            else:
                d_state = 16

            d_model = 256
            d_conv = 4
            expand = 2
            logging.info(f"[Q-NET] Auto-detected: n_layers={n_layers}, d_state={d_state}")

            # 创建Q网络
            self.q_net1 = SACQNetwork(d_model, n_layers, d_state, d_conv, expand).to(self.device)
            self.q_net2 = SACQNetwork(d_model, n_layers, d_state, d_conv, expand).to(self.device)

            # 加载权重（使用检测到的key）
            self.q_net1.load_state_dict(checkpoint[q1_key])
            self.q_net2.load_state_dict(checkpoint[q2_key])

            self.q_net1.eval()
            self.q_net2.eval()

            self._q_networks_loaded = True
            logging.info(f"[Q-NET] Successfully loaded Q networks from {checkpoint_path}")
            return True

        except Exception as e:
            logging.error(f"[Q-NET] Failed to load Q networks: {e}")
            self._q_networks_loaded = False
            return False

    def compute_scene_danger(self, state) -> Tuple[int, float]:
        """计算场景危险度，用于门控决策

        Returns:
            (human_count, min_ttc): 人类数量和最小TTC
        """
        # 提取状态
        if hasattr(state, 'to_array'):
            state_arr = state.to_array()
        else:
            state_arr = np.array(state, dtype=np.float32).flatten()

        # 解析机器人状态
        robot_px, robot_py = state_arr[0], state_arr[1]
        robot_vx, robot_vy = state_arr[2], state_arr[3]
        robot_radius = state_arr[4] if state_arr.size > 4 else 0.0

        # 解析人类状态 (从第9个元素开始，每5个一组)
        human_data = state_arr[9:]
        n_humans = len(human_data) // 5

        if n_humans == 0:
            return 0, float('inf')

        min_ttc = float('inf')
        for i in range(n_humans):
            idx = i * 5
            hx, hy = human_data[idx], human_data[idx + 1]
            hvx, hvy = human_data[idx + 2], human_data[idx + 3]
            hr = human_data[idx + 4] if (idx + 4) < len(human_data) else 0.0

            rel_x = hx - robot_px
            rel_y = hy - robot_py
            rel_vx = hvx - robot_vx
            rel_vy = hvy - robot_vy
            ttc = self._ttc_rel(rel_x, rel_y, rel_vx, rel_vy, robot_radius + hr)
            if ttc < min_ttc:
                min_ttc = ttc

        return n_humans, min_ttc

    def act_discrete(self, state, n_actions: Optional[int] = None) -> 'ActionXY':
        """使用离散动作搜索选择最佳动作（SARL风格）

        对n_actions个离散动作进行Q值评估，选择Q值最高的动作

        Args:
            state: JointState或状态数组
            n_actions: 离散动作数量（默认使用GRID配置）

        Returns:
            ActionXY: 最佳动作
        """
        from crowd_nav.contracts import discrete_index_to_action, _batch_joint34_to_tokens_vectorized, GRID, grid_action_dim
        from crowd_sim.envs.utils.action import ActionXY

        if not self._q_networks_loaded:
            logging.warning("[DISCRETE-ACT] Q networks not loaded, falling back to continuous")
            return self.predict(state)

        # 1. 状态转换为tokens
        if hasattr(state, 'to_array'):
            state_34 = state.to_array()
        else:
            state_34 = np.array(state, dtype=np.float32).flatten()

        if state_34.size < 34:
            pad = np.zeros(34 - state_34.size, dtype=np.float32)
            state_34 = np.concatenate([state_34, pad])
        elif state_34.size > 34:
            state_34 = state_34[:34]

        state_token = _batch_joint34_to_tokens_vectorized(state_34.reshape(1, -1))[0]

        # 2. 构建历史序列
        self._history.append(state_token.copy())
        history_list = list(self._history)
        if len(history_list) < self.seq_len:
            pad_frame = history_list[0]
            pad_needed = self.seq_len - len(history_list)
            history_list = [pad_frame.copy() for _ in range(pad_needed)] + history_list
        history_list = history_list[-self.seq_len:]

        # 转换为tensor [1, T, 8, 13]
        history_tensor = np.array(history_list, dtype=np.float32)
        try:
            history_tensor = torch.from_numpy(history_tensor).float().to(self.device)
        except RuntimeError:
            history_tensor = torch.tensor(history_tensor, dtype=torch.float32, device=self.device)
        history_tensor = history_tensor.unsqueeze(0)  # [1, T, 8, 13]

        # 3. 生成离散动作候选
        from crowd_nav.contracts import grid_action_dim
        total_actions = grid_action_dim(GRID)

        actions_list = []
        for a_idx in range(total_actions):
            vx, vy = discrete_index_to_action(a_idx)
            actions_list.append([vx / self.v_max, vy / self.v_max])  # 归一化

        actions_tensor = torch.tensor(actions_list, dtype=torch.float32, device=self.device)  # [N, 2]

        # 4. 批量评估Q值
        with torch.no_grad():
            # 扩展state到batch维度 [N, T, 8, 13]
            state_batch = history_tensor.expand(total_actions, -1, -1, -1)

            # 计算Q值 (取Q1和Q2的最小值，保守估计)
            q1 = self.q_net1(state_batch, actions_tensor)  # [N]
            q2 = self.q_net2(state_batch, actions_tensor)  # [N]
            q_values = torch.min(q1, q2)  # [N]

            # 选择Q值最大的动作
            best_idx = q_values.argmax().item()

        # 5. 返回最佳动作
        best_vx, best_vy = discrete_index_to_action(best_idx)
        return ActionXY(best_vx, best_vy)

    def predict_with_gating(self, state, human_threshold: int = 8, ttc_threshold: float = 2.0) -> 'ActionXY':
        """带门控的预测：根据场景危险度选择连续或离散动作

        门控规则：
        - 人少(<=human_threshold) 且 安全(min_ttc > ttc_threshold)：用连续动作（平滑）
        - 人多 或 危险：用离散搜索（安全）

        Args:
            state: JointState
            human_threshold: 人数阈值，超过则启用离散搜索
            ttc_threshold: TTC阈值，低于则启用离散搜索

        Returns:
            ActionXY: 选择的动作
        """
        # 计算场景危险度
        n_humans, min_ttc = self.compute_scene_danger(state)

        # 门控决策
        use_discrete = (n_humans > human_threshold) or (min_ttc < ttc_threshold)

        if use_discrete and self._q_networks_loaded:
            return self.act_discrete(state)
        else:
            return self.predict(state)

    def to(self, *args, **kwargs):
        """覆盖 to() 方法以跟踪设备"""
        # 调用父类的 to() 方法（会自动移动所有子模块）
        super().to(*args, **kwargs)
        # 提取设备参数并保存到 self.device
        if args and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])
        elif 'device' in kwargs:
            self.device = torch.device(kwargs['device'])
        return self


# ========== IQL Networks (Value & Q) ==========

class IQLValueNetwork(nn.Module):
    """IQL Value Network: V(s)

    使用与Policy相同的backbone（空间+时序编码），独立的value头

    """
    def __init__(self, d_model=256, n_layers=2, d_state=16, d_conv=4, expand=2,
                 temporal_backbone='mamba', dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.temporal_backbone = str(temporal_backbone).strip().lower()

        # 共享的backbone
        self.spatial_encoder = EnhancedSpatialEncoder(d_model)
        self.temporal_encoder = build_temporal_encoder(
            self.temporal_backbone,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Value head
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, joint_tokens, return_sequence=False):
        """
        Args:
            joint_tokens: [B, T, 8, 13]
            return_sequence: 是否返回整个序列

        Returns:
            value: [B] or [B, T]
        """
        spatial_feat = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        temporal_feat = self.temporal_encoder(spatial_feat)  # [B, T, d_model]

        if return_sequence:
            return self.value_head(temporal_feat).squeeze(-1)  # [B, T]
        else:
            last_feat = temporal_feat[:, -1, :]  # [B, d_model]
            return self.value_head(last_feat).squeeze(-1)  # [B]


class IQLQNetwork(nn.Module):
    """IQL Q Network: Q(s, a)

    支持两种模式：
    1. 连续动作模式（forward）：输入state+action，输出单个Q值
    2. 离散动作模式（forward_q）：输入state，输出所有80个动作的Q值

    """
    def __init__(self, d_model=256, n_layers=2, d_state=16, d_conv=4, expand=2,
                 action_dim=2, n_actions=80, temporal_backbone='mamba', dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.temporal_backbone = str(temporal_backbone).strip().lower()

        # 共享的backbone
        self.spatial_encoder = EnhancedSpatialEncoder(d_model)
        self.temporal_encoder = build_temporal_encoder(
            self.temporal_backbone,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # 连续动作Q head: state features + action → Q value
        self.q_head_continuous = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # 离散动作Q head: state features → Q values for all actions
        self.q_head_discrete = nn.Linear(d_model, n_actions)

    def forward(self, joint_tokens, actions):
        """连续动作模式：Q(s, a)

        Args:
            joint_tokens: [B, T, 8, 13]
            actions: [B, 2] continuous actions

        Returns:
            q_value: [B]
        """
        spatial_feat = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        temporal_feat = self.temporal_encoder(spatial_feat)  # [B, T, d_model]
        last_feat = temporal_feat[:, -1, :]  # [B, d_model]

        # 拼接state features和action
        qa = torch.cat([last_feat, actions], dim=-1)  # [B, d_model + 2]
        q_value = self.q_head_continuous(qa).squeeze(-1)  # [B]

        return q_value

    def forward_q(self, joint_tokens, return_sequence=False):
        """离散动作模式：输出所有80个动作的Q值

        Args:
            joint_tokens: [B, T, 8, 13]
            return_sequence: 是否返回整个序列

        Returns:
            如果return_sequence=True:
                q_values: [B, T, 80]
            如果return_sequence=False:
                q_values: [B, 80]
        """
        spatial_feat = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        temporal_feat = self.temporal_encoder(spatial_feat)  # [B, T, d_model]

        if return_sequence:
            # 返回整个序列的Q值
            q_values = self.q_head_discrete(temporal_feat)  # [B, T, 80]
        else:
            # 只返回最后一帧的Q值
            last_feat = temporal_feat[:, -1, :]  # [B, d_model]
            q_values = self.q_head_discrete(last_feat)  # [B, 80]

        return q_values

    # ============ SARL Compatibility Methods ============

    def get_model(self):
        """返回value network（SARL Trainer需要）"""
        return self

    def transform(self, state):
        """将JointState转换为tensor（SARL Explorer需要）"""
        from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized

        # 构建34维state
        state_34 = self._build_joint_state_34(state.self_state, state.human_states)
        # 转换为tokens
        state_token = _batch_joint34_to_tokens_vectorized(state_34.reshape(1, -1))[0]
        # 组织序列
        temp_history = list(self._history) + [state_token]
        temp_history = temp_history[-self.seq_len:]
        if len(temp_history) < self.seq_len:
            pad_frame = temp_history[0]
            temp_history = [pad_frame] * (self.seq_len - len(temp_history)) + temp_history

        return torch.from_numpy(np.array(temp_history)).float().to(self.device)

    def set_device(self, device):
        """设置设备（SARL兼容）"""
        self.device = device
        self.to(device)

    def configure(self, config):
        """配置参数（SARL兼容）"""
        if not hasattr(config, 'getfloat'):
            raise ValueError("configure() requires a config with getfloat (no fallback defaults).")
        # Strictly read from config
        self.gamma = config.getfloat('train', 'gamma')
        if config.has_option('action_space', 'time_step'):
            self.time_step = config.getfloat('action_space', 'time_step')
        elif config.has_option('env', 'time_step'):
            self.time_step = config.getfloat('env', 'time_step')
        else:
            raise ValueError("Missing time_step in config (expected [action_space] time_step or [env] time_step).")
        self.success_radius = config.getfloat('robot', 'success_radius')


class SACQNetwork(IQLQNetwork):
    """
    SAC专用Q网络（结构与IQLQNetwork一致，作为清晰别名，避免SAC路径出现IQL标签）
    """
    pass


# ========== 兼容别名 ==========
# 保持向后兼容，旧代码可能import MambaRL
MambaRL = MambaRLPolicy


# ---------------------------------------------------------------------------
# Order 17 item 13: explicit warm-start from a pre-occlusion checkpoint.
#
# Token columns 9-12 of a human row were always zero before this work, so the
# corresponding input weights of human_encoder[0] (a Linear(21, d/4), where
# inputs 0-12 are the token and 13-20 the relation features) never received a
# gradient and still hold their random initialisation. Loading such a
# checkpoint and then feeding real values into those columns would perturb the
# policy immediately, driven by noise. Zeroing exactly those four input columns
# makes training step 0 bit-identical to the pre-occlusion policy, after which
# the network learns to use them.
# ---------------------------------------------------------------------------
def warm_start_from_legacy(model, state_dict, occlusion_cols=(9, 10, 11, 12),
                           strict=False, verbose=True):
    import torch as _torch
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
          for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    enc = None
    for name, mod in model.named_modules():
        if name.endswith("human_encoder") and isinstance(mod, _torch.nn.Sequential):
            enc = mod[0]
            break
    if enc is None or not isinstance(enc, _torch.nn.Linear):
        raise RuntimeError("human_encoder[0] Linear not found; refusing a silent warm-start")
    if enc.in_features != 21:
        raise RuntimeError(f"expected Linear(21,*), got in_features={enc.in_features}")
    with _torch.no_grad():
        enc.weight[:, list(occlusion_cols)] = 0.0
    if verbose:
        print(f"[WARM-START] loaded legacy weights; zeroed human_encoder input "
              f"columns {list(occlusion_cols)}; missing={len(missing)} "
              f"unexpected={len(unexpected)}")
    return {"missing": list(missing), "unexpected": list(unexpected)}


def occlusion_checkpoint_meta(env_config, mode, backbone):
    """Provenance written into every checkpoint so a resume cannot silently mix
    an occlusion arm with a different one."""
    token_contract = env_config.get(
        'occlusion', 'token_contract', fallback='legacy_top5').strip().lower()
    schema_version = 3 if token_contract == 'belief_v3' else 2
    token_schema = ('v3_belief_15_entities' if schema_version == 3
                    else 'v2_belief_features')
    return {"occlusion_mode": mode, "token_schema": token_schema,
            "belief_features": env_config.get(
                'occlusion', 'belief_features', fallback='full').strip().lower(),
            "token_contract": token_contract,
            "visible_slots": env_config.getint(
                'occlusion', 'visible_slots', fallback=5),
            "hidden_slots": env_config.getint(
                'occlusion', 'hidden_slots', fallback=10),
            "max_entities": env_config.getint(
                'occlusion', 'max_entities', fallback=5),
            "grid_resolution": env_config.getfloat('occlusion', 'grid_resolution', fallback=0.25),
            "grid_extent": env_config.getfloat('occlusion', 'grid_extent', fallback=5.0),
            "fov_radius": env_config.getfloat('occlusion', 'fov_radius', fallback=5.0),
            "backbone": backbone, "schema_version": schema_version}


def assert_checkpoint_compatible(meta, mode, backbone, expected_meta=None):
    """Fail closed on resume. A mismatched arm is a silently wrong experiment,
    which is worse than a crash."""
    if not meta:
        raise ValueError("checkpoint carries no occlusion metadata; refusing resume")
    for k, want in (("occlusion_mode", mode), ("backbone", backbone)):
        if meta.get(k) != want:
            raise ValueError(f"checkpoint {k}={meta.get(k)!r} but this run is {want!r}")
    if expected_meta is not None:
        identity_keys = ["token_schema", "schema_version"]
        if int(expected_meta.get("schema_version", -1)) >= 3:
            identity_keys.extend([
                "token_contract", "visible_slots", "hidden_slots",
                "max_entities"])
        for key in identity_keys:
            if meta.get(key) != expected_meta.get(key):
                raise ValueError(
                    f"checkpoint {key}={meta.get(key)!r} but this run is "
                    f"{expected_meta.get(key)!r}")
        if meta.get("belief_features") != expected_meta.get("belief_features"):
            raise ValueError(
                f"checkpoint belief_features={meta.get('belief_features')!r} "
                f"but this run is {expected_meta.get('belief_features')!r}")
        for key in ("grid_resolution", "grid_extent", "fov_radius"):
            got = float(meta.get(key, float("nan")))
            want = float(expected_meta[key])
            if not math.isclose(got, want, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"checkpoint {key}={got!r} but this run is {want!r}")
    else:
        if meta.get("token_schema") != "v2_belief_features":
            raise ValueError(f"token schema mismatch: {meta.get('token_schema')!r}")
        if int(meta.get("schema_version", -1)) != 2:
            raise ValueError(f"checkpoint schema mismatch: {meta.get('schema_version')!r}")
