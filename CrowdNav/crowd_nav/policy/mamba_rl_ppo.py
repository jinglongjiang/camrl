# -*- coding: utf-8 -*-
"""
MambaRL - Enhanced Version with Improved Spatial Encoding & Temporal Modeling

核心改进：
1. 空间编码器：使用完整token信息 + 关系编码 + Attention
2. 运动信息增强：显式编码速度和加速度
3. 更长时序窗口：支持T=12-16
"""

import math
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
    增强空间编码器 v3：todo.md精简优化版

    核心改进（基于todo.md优化方案）：
    1. 使用完整13维token（不丢弃信息）
    2. 显式编码robot-human关系（相对位置/速度/TTC）
    3. 【优化】删除context_encoder（goal/motion已在robot_tokens中）
    4. 【优化】human分支改mean pooling（3人场景attention收益极小）
    5. 【优化】3路融合（robot+human+relation），删除冗余context分支
    """
    def __init__(self, d_model: int = 160):
        super().__init__()
        self.d_model = d_model

        # 机器人特征编码器（完整13维）
        self.robot_encoder = nn.Sequential(
            nn.Linear(13, d_model // 4),
            nn.ReLU(),
            nn.LayerNorm(d_model // 4)
        )

        # 每个人类的特征编码器（独立编码，保留细节）
        self.human_encoder = nn.Sequential(
            nn.Linear(13, d_model // 4),
            nn.ReLU(),
            nn.LayerNorm(d_model // 4)
        )

        # 关系特征编码器（robot-human关系）
        self.relation_encoder = nn.Sequential(
            nn.Linear(8, d_model // 4),
            nn.ReLU()
        )

        # 人类特征注意力聚合（防止Dense环境性能退化）
        self.human_attention = nn.MultiheadAttention(
            embed_dim=d_model // 4,  # 对应 human_encoder 的输出维度
            num_heads=4,             # 4头注意力，捕捉不同类型的危险模式
            batch_first=True
        )

        # Dropout正则化（todo.md优化：只保留fusion前，删除attn后的过度正则化）
        self.dropout_fusion = nn.Dropout(p=0.1)  # fusion前

        # 融合所有特征（todo.md优化：3路输入，删除context分支）
        self.fusion = nn.Sequential(
            nn.Linear(3 * (d_model // 4), d_model),  # 3路：robot + human + relation
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, joint_state):
        """
        joint_state: [B, T, 6, 13] tokens
        Returns: [B, T, d_model]
        """
        B, T = joint_state.shape[:2]

        # ====== 1. 提取机器人特征（row 0，完整13维） ======
        robot_tokens = joint_state[:, :, 0, :]  # [B, T, 13]
        robot_feat = self.robot_encoder(robot_tokens)  # [B, T, d/4]

        # ====== 2. 提取人类特征（row 3-5，3个最近的人） ======
        # todo.md优化：Top-K兜底，防止上游contracts.py改变排序逻辑
        humans_all = joint_state[:, :, 3:6, :]  # [B, T, 3, 13]

        # 从row 3-5的第2列提取距离（contracts.py中已存储rel_dist）
        distances = humans_all[..., 2]  # [B, T, 3]

        # 按距离排序（ascending），确保始终是最近的3个
        _, sorted_indices = torch.sort(distances, dim=-1, descending=False)  # [B, T, 3]

        # 使用sorted_indices重新排列humans_tokens
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, 13)  # [B, T, 3, 13]
        humans_tokens = torch.gather(humans_all, dim=2, index=sorted_indices_expanded)  # [B, T, 3, 13]

        # 向量化编码所有人类（去掉for循环）
        # Reshape: [B, T, 3, 13] → [B*T*3, 13]
        h_flat = humans_tokens.reshape(B * T * 3, 13)
        h_feats = self.human_encoder(h_flat)  # [B*T*3, d/4] 一次前向
        human_feats_stack = h_feats.reshape(B, T, 3, -1)  # [B, T, 3, d/4]

        # ====== 3. 人类特征聚合（Attention机制，防止Dense环境性能退化） ======
        B, T, N, D_h = human_feats_stack.shape  # N=3 (最近的3个人)
        human_feats_flat = human_feats_stack.view(B * T, N, D_h)  # [B*T, 3, d/4]

        # Multi-head self-attention：捕捉人类之间的交互模式
        attn_out, _ = self.human_attention(
            query=human_feats_flat,
            key=human_feats_flat,
            value=human_feats_flat
        )  # [B*T, 3, d/4]

        # Max pooling：关注最危险的人类特征（比mean更敏感）
        human_global = attn_out.max(dim=1)[0]  # [B*T, d/4]
        human_global = human_global.view(B, T, D_h)  # [B, T, d/4]

        # ====== 4. 计算关系特征（向量化，去掉for循环） ======
        # Robot特征广播: [B, T] → [B, T, 1] for broadcasting
        robot_px = robot_tokens[..., 0:1]  # [B, T, 1]
        robot_py = robot_tokens[..., 1:2]  # [B, T, 1]
        robot_vx = robot_tokens[..., 2:3]  # [B, T, 1]
        robot_vy = robot_tokens[..., 3:4]  # [B, T, 1]

        # 人类特征: [B, T, 3]
        h_px = humans_tokens[..., 0]  # [B, T, 3]
        h_py = humans_tokens[..., 1]  # [B, T, 3]
        h_vx = humans_tokens[..., 3]  # [B, T, 3]
        h_vy = humans_tokens[..., 4]  # [B, T, 3]

        # 广播计算相对位置（自动扩展维度）
        rel_x = h_px - robot_px  # [B, T, 3]
        rel_y = h_py - robot_py  # [B, T, 3]
        rel_dist = torch.sqrt(rel_x**2 + rel_y**2 + 1e-6)

        # 广播计算相对速度
        rel_vx = h_vx - robot_vx  # [B, T, 3]
        rel_vy = h_vy - robot_vy  # [B, T, 3]
        rel_speed = torch.sqrt(rel_vx**2 + rel_vy**2 + 1e-6)

        # TTC批量计算
        closing_speed = -(rel_x * rel_vx + rel_y * rel_vy) / (rel_dist + 1e-6)
        closing_speed = torch.clamp(closing_speed, min=0.0, max=10.0)
        ttc = torch.where(
            closing_speed > 0.01,
            rel_dist / (closing_speed + 1e-6),
            torch.full_like(closing_speed, 100.0)
        )
        ttc = torch.clamp(ttc, min=0.0, max=100.0)

        # 组合关系特征 [B, T, 3, 8]
        rel_feat = torch.stack([
            rel_x, rel_y, rel_dist,
            rel_vx, rel_vy, rel_speed,
            closing_speed, 1.0 / (ttc + 1.0)
        ], dim=-1)

        # 向量化编码: [B, T, 3, 8] → [B*T*3, 8]
        rel_flat = rel_feat.reshape(B * T * 3, 8)
        rel_encoded = self.relation_encoder(rel_flat)  # [B*T*3, d/4]
        rel_encoded = rel_encoded.reshape(B, T, 3, -1)  # [B, T, 3, d/4]

        # 平均所有人类的关系特征
        relation_global = rel_encoded.mean(dim=2)  # [B, T, d/4]

        # ====== 5. 融合3路特征（todo.md优化：删除context，只保留核心3路） ======
        all_features = torch.cat([
            robot_feat,      # [B, T, d/4] 机器人自身（已含goal/motion信息）
            human_global,    # [B, T, d/4] 人类聚合（mean pooling）
            relation_global  # [B, T, d/4] 关系特征（TTC/relative等）
        ], dim=-1)  # [B, T, 3*d/4]

        # Dropout before fusion (防止过拟合)
        all_features = self.dropout_fusion(all_features)

        output = self.fusion(all_features)
        return output


class MambaTemporalEncoder(nn.Module):
    """
    时序编码器：使用官方mamba_ssm实现
    仅支持Mamba，不提供fallback
    """
    def __init__(self, d_model: int = 160, n_layers: int = 2, **mamba_kwargs):
        super().__init__()

        if not MAMBA_SSM_AVAILABLE or Mamba is None:
            raise RuntimeError(
                "[MAMBA-SSM] ✗ mamba_ssm库不可用！\n"
                "请安装：pip install mamba-ssm\n"
                "不支持fallback - 仅使用纯Mamba实现"
            )

        # 使用官方Mamba模块
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


class MambaRL(nn.Module):
    """增强版MambaRL：改进空间编码 + 运动增强 + 更长时序

    支持两种训练模式：
    - BC模式：单步预测，跳过时序组件（快速）
    - RL模式：完整时序建模（高性能）
    """
    def __init__(self, config=None, device='cpu'):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.env_dt = 0.25
        self.trainable = True
        self.multiagent_training = False

        # 网络配置
        if config and hasattr(config, 'getint'):
            d_model = config.getint('mamba', 'd_model', fallback=160)
            n_layers = config.getint('mamba', 'n_layers', fallback=2)  # 优化：SSM层数减少到2
            d_state = config.getint('mamba', 'd_state', fallback=16)
            d_conv = config.getint('mamba', 'd_conv', fallback=4)
            expand = config.getint('mamba', 'expand', fallback=2)
            self.v_max = config.getfloat('robot', 'v_pref', fallback=1.0)
            self.seq_len = config.getint('buffer', 'seq_len', fallback=1)
        else:
            d_model, n_layers, d_state, d_conv, expand = 160, 2, 16, 4, 2
            self.v_max = 1.0
            self.seq_len = 1

        # 构建网络 - 精简版本：Spatial → Temporal → Heads
        self.spatial_encoder = EnhancedSpatialEncoder(d_model)
        self.temporal_encoder = MambaTemporalEncoder(
            d_model, n_layers, d_state=d_state, d_conv=d_conv, expand=expand
        )

        # 双头：连续控制（Gaussian policy）+ 价值
        # PPO requires stochastic policy: output mean and log_std
        self.value_head = nn.Linear(d_model, 1)

        # Gaussian policy: output mean (2D) and log_std (2D)
        self.action_mean = nn.Linear(d_model, 2)  # (vx, vy)
        self.action_log_std = nn.Parameter(torch.full((2,), -1.5))  # std≈0.22（降低探索噪声，提升BC成功率）

        self._phase = 'train'

        # 训练模式：'bc' (跳过时序，快速) 或 'rl' (完整时序建模)
        self.training_mode = 'rl'  # 默认RL模式

        # 历史缓存机制：维护 seq_len 帧的历史，以匹配 BC 训练时的输入分布
        from collections import deque
        self._history = deque(maxlen=self.seq_len)  # 自动丢弃超过 seq_len 的老帧
        self._zero_token = np.zeros((6, 13), dtype=np.float32)  # 零填充用的模板

        # 移动所有模块到目标设备
        self.to(self.device)

        # SSOT验证日志（todo.md要求：明确打印最终配置，避免口径漂移）
        logging.info("=" * 80)
        logging.info("[MAMBA-FINAL] 最终Mamba配置 (SSOT from policy.config):")
        logging.info(f"  d_model={d_model}, n_layers={n_layers}, d_state={d_state}")
        logging.info(f"  d_conv={d_conv}, expand={expand}")
        logging.info(f"[SEQ-FINAL] 序列长度配置 (SSOT):")
        logging.info(f"  seq_len={self.seq_len} (来自train.config:[buffer])")
        logging.info(f"  T={self.seq_len} (应与policy.config:[temporal].T一致)")
        logging.info("[ARCH-FINAL] 架构: EnhancedSpatial(3路:robot+human+relation) → MambaSSM → DualHead")
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
        # 处理输入维度：确保是 [B, T, 6, 13]
        if joint_tokens.dim() == 3:
            # [B, 6, 13] - 单步，添加时序维度
            joint_tokens = joint_tokens.unsqueeze(1)  # [B, 1, 6, 13]
        elif joint_tokens.dim() == 4:
            # [B, T, 6, 13] - 已是正确格式
            pass
        else:
            raise ValueError(f"Expected [B, 6, 13] or [B, T, 6, 13], got {joint_tokens.shape}")

        # 空间编码 → 时序建模 → 价值预测
        spatial_features = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        temporal_features = self.temporal_encoder(spatial_features)  # [B, T, d_model]

        # 取最后一帧特征预测价值
        last_features = temporal_features[:, -1, :]  # [B, d_model]
        values = self.value_head(last_features).squeeze(-1)  # [B]

        return values

    def forward_both(self, joint_tokens, mask=None, return_sequence=False):
        """同时输出动作分布参数和价值（PPO需要）

        Args:
            joint_tokens: [B, T, 6, 13] or [B, 6, 13]
            mask: [B, T] - 可选，标记有效时间步
            return_sequence: 是否返回整个序列（PPO训练需要True，predict需要False）

        Returns:
            如果return_sequence=True（PPO训练）:
                action_mean: [B, T, 2] - 每个时间步的动作均值
                action_log_std: [2] - 全局标准差
                values: [B, T] - 每个时间步的价值
            如果return_sequence=False（predict推理）:
                action_mean: [B, 2] - 最后一帧的动作均值
                action_log_std: [2] - 全局标准差
                values: [B] - 最后一帧的价值
        """
        # 确保输入在正确的设备上
        if joint_tokens.device != next(self.parameters()).device:
            joint_tokens = joint_tokens.to(next(self.parameters()).device)

        # 处理输入维度
        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)  # [B, 1, 6, 13]
        elif joint_tokens.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected [B, 6, 13] or [B, T, 6, 13], got {joint_tokens.shape}")

        if self.training_mode == 'bc':
            # BC模式：单步预测，跳过时序
            if joint_tokens.shape[1] > 1:
                joint_tokens_bc = joint_tokens[:, -1:, :, :]
            else:
                joint_tokens_bc = joint_tokens

            spatial_features = self.spatial_encoder(joint_tokens_bc)  # [B, 1, d_model]
            last_features = spatial_features[:, 0, :]  # [B, d_model]

            action_mean = self.action_mean(last_features)  # [B, 2]
            values = self.value_head(last_features).squeeze(-1)  # [B]

        else:
            # RL模式：完整时序建模
            spatial_features = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
            temporal_features = self.temporal_encoder(spatial_features)  # [B, T, d_model]

            if return_sequence:
                # PPO训练：返回整个序列 [B, T, ...]
                action_mean = self.action_mean(temporal_features)  # [B, T, 2]
                values = self.value_head(temporal_features).squeeze(-1)  # [B, T]
            else:
                # Predict推理：只返回最后一帧 [B, ...]
                last_features = temporal_features[:, -1, :]  # [B, d_model]
                action_mean = self.action_mean(last_features)  # [B, 2]
                values = self.value_head(last_features).squeeze(-1)  # [B]

        # [FIX] 约束输出防止数值爆炸
        action_mean = torch.tanh(action_mean)  # 限制到 [-1, 1]
        action_log_std = torch.clamp(self.action_log_std, -2.3, 0.7)  # std ∈ [0.1, 2.0]

        return action_mean, action_log_std, values

    def predict(self, state):
        """推理接口：从Gaussian policy采样动作（PPO标准做法）

        PPO使用stochastic policy，训练和推理都从分布采样：
        - 训练时：采样动作用于探索
        - 推理时：可选择mean（确定性）或采样（随机性）
        """
        from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized, _to_tensor
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
                state_34 = np.array(state, dtype=np.float32)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Cannot convert state to array: type={type(state)}, value={state}") from e

        state_token = _batch_joint34_to_tokens_vectorized(state_34.reshape(1, -1))[0]

        # 2. 压入历史缓存
        self._history.append(state_token.copy())

        # 3. 组织成 [seq_len, 6, 13] 的序列
        history_list = list(self._history)
        while len(history_list) < self.seq_len:
            if history_list:
                history_list.insert(0, history_list[0].copy())
            else:
                history_list.insert(0, self._zero_token.copy())

        history_list = history_list[-self.seq_len:]
        history_tensor = np.array(history_list, dtype=np.float32)
        history_tensor = torch.tensor(history_tensor, dtype=torch.float32, device=self.device)
        history_tensor = history_tensor.unsqueeze(0)  # [1, seq_len, 6, 13]

        # 4. 前向传播获取动作分布
        self.eval()
        with torch.no_grad():
            # [修复] 保持rl模式，使用完整12帧历史进行时序建模
            # 删除了orig_mode和self.training_mode='bc'的强制切换
            action_mean, action_log_std, _ = self.forward_both(history_tensor)

            action_mean = action_mean.squeeze(0)  # [2]
            action_std = torch.exp(action_log_std)  # [2]

            # PPO推理：训练时采样，测试时用mean
            if self._phase == 'train':
                # 训练阶段：从分布采样（探索）
                dist = torch.distributions.Normal(action_mean, action_std)
                action_sample = dist.sample()
            else:
                # 测试阶段：使用mean（确定性）
                action_sample = action_mean

            # 裁剪到单位球，确保与真实执行动作一致
            action_safe = self._clip_unit_ball(action_sample)
            vx_normalized = float(action_safe[0].cpu().item())
            vy_normalized = float(action_safe[1].cpu().item())

            # [修复C] 还原到真实速度空间
            # BC训练时动作归一化到[-1, 1]，推理时必须乘回v_max
            vx = vx_normalized * self.v_max
            vy = vy_normalized * self.v_max

            # Clamp到合理范围（双重保险）
            speed = (vx**2 + vy**2) ** 0.5
            if speed > self.v_max:
                vx = vx / speed * self.v_max
                vy = vy / speed * self.v_max

            # [DIAGNOSTIC] 缓存动作用于调试（每100步打印一次）
            if not hasattr(self, '_action_log_count'):
                self._action_log_count = 0
            self._action_log_count += 1
            if self._action_log_count <= 5 or self._action_log_count % 100 == 0:
                import logging
                logging.info(f"[PREDICT-DIAG] step={self._action_log_count} phase={self._phase} vx_norm={vx_normalized:.4f} vy_norm={vy_normalized:.4f} vx={vx:.4f} vy={vy:.4f} speed={speed:.4f}")

        return ActionXY(vx, vy)

    def predict_with_value(self, state):
        """PPO专用：推理时返回(action, log_prob, value)用于rollout收集

        Returns:
            action: ActionXY对象
            log_prob: float - 动作的对数概率
            value: float - 状态价值估计
        """
        from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
        from crowd_sim.envs.utils.action import ActionXY

        # 1. 转换state为token
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
                state_34 = np.array(state, dtype=np.float32)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Cannot convert state to array: type={type(state)}, value={state}") from e

        state_token = _batch_joint34_to_tokens_vectorized(state_34.reshape(1, -1))[0]

        # 2. 压入历史缓存
        self._history.append(state_token.copy())

        # 3. 组织成序列
        history_list = list(self._history)
        while len(history_list) < self.seq_len:
            if history_list:
                history_list.insert(0, history_list[0].copy())
            else:
                history_list.insert(0, self._zero_token.copy())

        history_list = history_list[-self.seq_len:]
        history_tensor = np.array(history_list, dtype=np.float32)
        history_tensor = torch.tensor(history_tensor, dtype=torch.float32, device=self.device)
        history_tensor = history_tensor.unsqueeze(0)  # [1, seq_len, 6, 13]

        # 4. 前向传播获取动作分布和价值
        self.eval()
        with torch.no_grad():
            action_mean, action_log_std, value = self.forward_both(history_tensor)

            action_mean = action_mean.squeeze(0)  # [2]
            action_std = torch.exp(action_log_std)  # [2]
            value = value.squeeze(0)  # scalar

            dist = torch.distributions.Normal(action_mean, action_std)

            # 训练阶段采样，评估/IL阶段用均值（减少随机扰动，提高评测稳定性）
            if getattr(self, "_phase", "train") == "train":
                action_sample = dist.sample()
            else:
                action_sample = action_mean

            # 裁剪到单位球，log_prob基于实际执行动作，避免ratio失真
            action_safe = self._clip_unit_ball(action_sample)
            log_prob = dist.log_prob(action_safe).sum()  # 对2D动作求和

            vx_normalized = float(action_safe[0].cpu().item())
            vy_normalized = float(action_safe[1].cpu().item())

            # 还原到真实速度空间
            vx = vx_normalized * self.v_max
            vy = vy_normalized * self.v_max

            # Clamp到合理范围
            speed = (vx**2 + vy**2) ** 0.5
            if speed > self.v_max:
                vx = vx / speed * self.v_max
                vy = vy / speed * self.v_max

        return ActionXY(vx, vy), float(log_prob.cpu().item()), float(value.cpu().item())

    def evaluate_actions(self, states_tokens, actions_continuous):
        """PPO专用：重新计算给定state-action对的log_prob和value

        Args:
            states_tokens: [B, T, 6, 13] - token化的状态序列
            actions_continuous: [B, T, 2] - 连续动作(vx, vy)已归一化到[-1,1]

        Returns:
            log_probs: [B, T] - 每个时间步的动作对数概率
            values: [B, T] - 每个时间步的状态价值
        """
        # 前向传播获取分布参数和价值
        action_mean, action_log_std, values = self.forward_both(states_tokens, return_sequence=True)

        # action_mean: [B, T, 2], action_log_std: [2], values: [B, T]
        action_std = torch.exp(action_log_std)  # [2]

        # 计算log_prob
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions_continuous).sum(dim=-1)  # [B, T, 2] -> [B, T]

        return log_probs, values

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
