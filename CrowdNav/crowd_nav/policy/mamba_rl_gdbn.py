# -*- coding: utf-8 -*-
"""
MambaRL - Enhanced Version with Improved Spatial Encoding & Temporal Modeling

Key improvements:
1. Spatial Encoder: uses full token information + relational encoding + Attention
2. Motion augmentation: explicitly encodes velocity and acceleration
3. Longer temporal window: supports T=24+

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import logging

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    MAMBA_SSM_AVAILABLE = True
    logging.info("[MAMBA-SSM]  Successfully imported Mamba-1 (mamba_simple)")
except ImportError as e:
    MAMBA_SSM_AVAILABLE = False
    logging.error(f"[MAMBA-SSM]  Failed to import mamba_ssm: {e}")
    logging.error("[MAMBA-SSM] Please install: pip install mamba-ssm")
    Mamba = None

class EnhancedSpatialEncoder(nn.Module):
    """
 v4

 1. 13token
 2. rel_x/y/dist + rel_vx/vy/speed + closing_speed + ttc_inv (8)
 3. MultiheadAttention
 4. 2robot + humanrelation
    """
    def __init__(
        self,
        d_model: int = 256,
        token_dim: int = 23,
        base_token_dim: int = 13,
        relation_dim: int = 8,
        human_start: int = 3,
        num_humans: int = 5,
        human_order: str = 'preserve',
        relation_frame: str = 'token',
    ):
        super().__init__()
        self.d_model = d_model
        self.token_dim = int(token_dim)
        self.base_token_dim = int(base_token_dim)
        self.relation_dim = int(relation_dim)
        self.human_start = int(human_start)
        self.num_humans = int(num_humans)
        self.human_end = self.human_start + self.num_humans
        self.human_order = str(human_order or 'preserve').strip().lower()
        self.relation_frame = str(relation_frame or 'token').strip().lower()

        # Belief columns are zero-init so IL phase (zero-padded belief) is unaffected.
        self.robot_encoder = nn.Sequential(
            nn.Linear(self.token_dim, d_model // 4),
            nn.ReLU(),
            nn.LayerNorm(d_model // 4)
        )

        self.human_encoder = nn.Sequential(
            nn.Linear(self.token_dim + self.relation_dim, d_model // 4),
            nn.ReLU(),
            nn.LayerNorm(d_model // 4)
        )

        # Match the old base fan-in scale, then zero-init belief columns.
        with torch.no_grad():
            human_base_dim = self.base_token_dim + self.relation_dim
            self.robot_encoder[0].weight.zero_()
            self.human_encoder[0].weight.zero_()
            nn.init.kaiming_uniform_(self.robot_encoder[0].weight[:, :self.base_token_dim], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.human_encoder[0].weight[:, :self.base_token_dim], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.human_encoder[0].weight[:, self.token_dim:self.token_dim + self.relation_dim], a=math.sqrt(5))
            self.robot_encoder[0].bias.uniform_(-1.0 / math.sqrt(self.base_token_dim), 1.0 / math.sqrt(self.base_token_dim))
            self.human_encoder[0].bias.uniform_(-1.0 / math.sqrt(human_base_dim), 1.0 / math.sqrt(human_base_dim))

        self.human_attention = nn.MultiheadAttention(
            embed_dim=d_model // 4,
            num_heads=4,
            batch_first=True
        )

        self.dropout_fusion = nn.Dropout(p=0.1)

        self.fusion = nn.Sequential(
            nn.Linear(2 * (d_model // 4), d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, joint_state):
        B, T = joint_state.shape[:2]

        robot_tokens = joint_state[:, :, 0, :]
        robot_feat = self.robot_encoder(robot_tokens)  # [B, T, d/4]

        humans_all = joint_state[:, :, self.human_start:self.human_end, :]
        tok_dim = humans_all.shape[-1]
        if self.human_order in ('distance', 'nearest', 'legacy_distance'):
            distances = torch.where(
                (humans_all.abs().sum(dim=-1) > 0),
                humans_all[..., 2],
                torch.full_like(humans_all[..., 2], 1e9),
            )
            _, sorted_indices = torch.sort(distances, dim=-1, descending=False)
            sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, tok_dim)
            humans_tokens = torch.gather(humans_all, dim=2, index=sorted_indices_expanded)
        elif self.human_order in ('ttc', 'threat'):
            threat = torch.where(
                (humans_all.abs().sum(dim=-1) > 0),
                -humans_all[..., 7],
                torch.full_like(humans_all[..., 7], 1e9),
            )
            _, sorted_indices = torch.sort(threat, dim=-1, descending=False)
            sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, tok_dim)
            humans_tokens = torch.gather(humans_all, dim=2, index=sorted_indices_expanded)
        else:
            humans_tokens = humans_all

        robot_px = robot_tokens[..., 0:1]
        robot_py = robot_tokens[..., 1:2]
        robot_vx = robot_tokens[..., 2:3]
        robot_vy = robot_tokens[..., 3:4]

        h_vx = humans_tokens[..., 3]
        h_vy = humans_tokens[..., 4]
        if self.relation_frame in ('token', 'relative', 'path_y'):
            rel_x = humans_tokens[..., 0]
            rel_y = humans_tokens[..., 1]
            rel_dist = torch.clamp(humans_tokens[..., 2], min=0.0)
        else:
            h_px = humans_tokens[..., 0]
            h_py = humans_tokens[..., 1]
            rel_x = h_px - robot_px
            rel_y = h_py - robot_py
            rel_dist = torch.sqrt(rel_x**2 + rel_y**2 + 1e-6)
        rel_vx = h_vx - robot_vx
        rel_vy = h_vy - robot_vy
        rel_speed = torch.sqrt(rel_vx**2 + rel_vy**2 + 1e-6)
        closing_speed = -(rel_x * rel_vx + rel_y * rel_vy) / (rel_dist + 1e-6)
        ttc_inv = torch.where(closing_speed > 0, closing_speed / (rel_dist + 1e-6), torch.zeros_like(closing_speed))

        rel_feat = torch.stack([rel_x, rel_y, rel_dist, rel_vx, rel_vy, rel_speed, closing_speed, ttc_inv], dim=-1)
        human_with_relation = torch.cat([humans_tokens, rel_feat], dim=-1)  # [B, T, 5, 31]

        hr_flat = human_with_relation.reshape(B * T * self.num_humans, human_with_relation.shape[-1])
        hr_feats = self.human_encoder(hr_flat)
        human_feats_stack = hr_feats.reshape(B, T, self.num_humans, -1)
        B, T, N, D_h = human_feats_stack.shape
        human_feats_flat = human_feats_stack.view(B * T, N, D_h)
        attn_out, _ = self.human_attention(query=human_feats_flat, key=human_feats_flat, value=human_feats_flat)
        human_global = attn_out.max(dim=1)[0]
        human_global = human_global.view(B, T, D_h)

        all_features = torch.cat([robot_feat, human_global], dim=-1)
        all_features = self.dropout_fusion(all_features)
        output = self.fusion(all_features)
        return output

class MambaTemporalEncoder(nn.Module):
    """
 mamba_ssm
    """
    def __init__(self, d_model: int = 256, n_layers: int = 2, **mamba_kwargs):
        super().__init__()

        if not MAMBA_SSM_AVAILABLE or Mamba is None:
            raise RuntimeError(
                "[MAMBA-SSM]  mamba_ssm\n"
                "pip install mamba-ssm\n"
                "fallback - Mamba"
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
    """GRU"""
    def __init__(self, d_model: int = 256, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
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

class BayesianCoreScorer:
    """Bayesian decision core; Mamba supplies bounded residual action/value priors."""

    def __init__(self, config):
        self.enabled = config.getboolean('bayesian_core', 'enable', fallback=True)
        self.rollout_horizon = config.getint('bayesian_core', 'rollout_horizon', fallback=1)
        self.train_rollout_horizon = config.getint('bayesian_core', 'train_rollout_horizon', fallback=self.rollout_horizon)
        self.safe_distance = config.getfloat('bayesian_core', 'safe_distance', fallback=0.6)
        self.risk_weight = config.getfloat('bayesian_core', 'risk_weight', fallback=2.0)
        self.uncertainty_weight = config.getfloat('bayesian_core', 'uncertainty_weight', fallback=0.25)
        self.klda_weight = config.getfloat('bayesian_core', 'klda_weight', fallback=0.5)
        self.epistemic_weight = config.getfloat('bayesian_core', 'epistemic_weight', fallback=0.05)
        self.reward_weight = config.getfloat('bayesian_core', 'reward_weight', fallback=1.0)
        self.action_residual_weight = config.getfloat('bayesian_core', 'action_residual_weight', fallback=0.50)
        self.action_prior_temperature = config.getfloat('bayesian_core', 'action_prior_temperature', fallback=1.0)
        self.value_bootstrap_weight = config.getfloat('bayesian_core', 'value_bootstrap_weight', fallback=0.10)
        self.cvar_alpha = config.getfloat('bayesian_core', 'cvar_alpha', fallback=0.80)
        self.require_action_model = config.getboolean('bayesian_core', 'require_action_model', fallback=True)
        self.require_action_indices = config.getboolean('bayesian_core', 'require_action_indices', fallback=True)

    def validate(self, gdbn_module):
        if not self.enabled:
            return
        if gdbn_module is None or not getattr(gdbn_module, 'is_fitted', False):
            raise RuntimeError("[BAYES-CORE] Enabled but no fitted GDBN world model is attached.")
        if not hasattr(gdbn_module, 'predict_action_rollout'):
            raise RuntimeError("[BAYES-CORE] GDBN world model has no action rollout API.")
        if self.require_action_model and not getattr(gdbn_module, 'action_fitted', False):
            raise RuntimeError("[BAYES-CORE] Enabled but GDBN action model is not fitted.")

    def efe(self, risk, entropy, klda, epistemic):
        return (
            self.risk_weight * risk
            + self.uncertainty_weight * entropy
            + self.klda_weight * klda
            - self.epistemic_weight * epistemic
        )

    def score(self, rewards, risk, entropy, klda, epistemic, action_residual=None,
              residual_value=None, gamma=0.99, residual_sigma=None, beta_uncertainty=0.0):
        """Unified Bayesian action scorer.

        score(a) = reward - EFE(a) + action_residual_weight * tanh(q_mamba(a) / T)

        The Mamba Q-head contributes a bounded residual correction in [-1, 1].
        Value bootstrap (residual_value) is kept as optional but defaults to disabled
        via config value_bootstrap_weight=0.0 to prevent value-head poisoning.
        """
        expected_free_energy = self.efe(risk, entropy, klda, epistemic)
        scores = self.reward_weight * rewards - expected_free_energy
        if action_residual is not None and self.action_residual_weight != 0.0:
            temperature = max(float(self.action_prior_temperature), 1e-6)
            mamba_residual = torch.tanh(action_residual / temperature)
            scores = scores + self.action_residual_weight * mamba_residual
        if residual_value is not None and self.value_bootstrap_weight != 0.0:
            scores = scores + self.value_bootstrap_weight * float(gamma) * torch.tanh(residual_value / 2.0)
        return scores, expected_free_energy

    def rollout_candidates(self, gdbn_module, current_34, action_space, dt, robot_radius, initial_belief_vec=None):
        self.validate(gdbn_module)
        if hasattr(gdbn_module, 'predict_action_rollout_batch'):
            try:
                actions_xy = np.asarray([(a.vx, a.vy) for a in action_space], dtype=np.float64)
                states_34d = np.repeat(np.asarray(current_34, dtype=np.float64).reshape(1, -1), len(action_space), axis=0)
                belief_batch = None
                if initial_belief_vec is not None:
                    belief_batch = np.repeat(
                        np.asarray(initial_belief_vec, dtype=np.float64).reshape(1, *np.asarray(initial_belief_vec).shape),
                        len(action_space),
                        axis=0,
                    )
                rb = gdbn_module.predict_action_rollout_batch(
                    states_34d,
                    actions_xy,
                    belief_vecs=belief_batch,
                    horizon=self.rollout_horizon,
                    dt=dt,
                    safe_distance=self.safe_distance,
                    cvar_alpha=self.cvar_alpha,
                )
                belief_vecs = [initial_belief_vec] * len(action_space)
                return (
                    np.asarray(rb['risk'], dtype=np.float64).tolist(),
                    np.asarray(rb.get('tail_risk', rb['risk']), dtype=np.float64).tolist(),
                    np.asarray(rb['entropy'], dtype=np.float64).tolist(),
                    np.asarray(rb['klda'], dtype=np.float64).tolist(),
                    np.asarray(rb['epistemic_value'], dtype=np.float64).tolist(),
                    belief_vecs,
                )
            except Exception:
                pass

        risk, tail_risk, entropy, klda, epistemic, belief_vecs = [], [], [], [], [], []
        for action in action_space:
            rollout = gdbn_module.predict_action_rollout(
                current_34,
                (action.vx, action.vy),
                horizon=self.rollout_horizon,
                dt=dt,
                robot_radius=robot_radius,
                safe_distance=self.safe_distance,
            )
            risk.append(float(rollout.get('risk', 0.0)))
            tail_risk.append(float(rollout.get('tail_risk', rollout.get('risk', 0.0))))
            entropy.append(float(rollout.get('entropy', 0.0)))
            klda.append(float(rollout.get('klda', 0.0)))
            epistemic.append(float(rollout.get('epistemic_value', 0.0)))
            belief_vecs.append(rollout.get('belief_vec', None))
        return risk, tail_risk, entropy, klda, epistemic, belief_vecs


class MambaRLPolicy(nn.Module):
    def __init__(self, config=None, device='cpu'):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self._blas_limit_ctx = None
        self._blas_limit_active = False

        if not (config and hasattr(config, 'getint') and hasattr(config, 'getfloat')):
            raise ValueError("MambaRLPolicy requires a config with getint/getfloat (no fallback defaults).")

        # Network config (strictly from config)
        d_model = config.getint('mamba', 'd_model')
        n_layers = config.getint('mamba', 'n_layers')
        d_state = config.getint('mamba', 'd_state')
        d_conv = config.getint('mamba', 'd_conv')
        expand = config.getint('mamba', 'expand')
        self.v_max = config.getfloat('robot', 'v_pref')
        self.seq_len = config.getint(
            'buffer', 'seq_len',
            fallback=config.getint('temporal', 'T', fallback=24)
        )
        self.belief_enabled = config.getboolean('belief', 'enable', fallback=True)
        self.disable_belief_tokens = False
        self.base_token_dim = config.getint('belief', 'base_token_dim', fallback=13)
        self.belief_dim = config.getint('belief', 'belief_dim', fallback=1)
        self.token_dim = config.getint('belief', 'token_dim', fallback=self.base_token_dim + self.belief_dim)
        self.num_entities = config.getint('belief', 'num_entities', fallback=8)
        self.human_start = config.getint('belief', 'human_start', fallback=3)
        self.num_humans = config.getint('belief', 'num_humans', fallback=5)
        self.human_end = self.human_start + self.num_humans
        if config.has_section('human_selection'):
            self.human_selection_mode = config.get('human_selection', 'mode', fallback='threat').strip().lower()
            self.human_selection_horizon = config.getint('human_selection', 'horizon', fallback=5)
            self.human_selection_ttc_weight = config.getfloat('human_selection', 'ttc_weight', fallback=2.0)
            self.human_selection_clearance_weight = config.getfloat('human_selection', 'clearance_weight', fallback=1.0)
            self.human_selection_distance_weight = config.getfloat('human_selection', 'distance_weight', fallback=0.15)
            self.human_selection_closing_weight = config.getfloat('human_selection', 'closing_weight', fallback=0.25)
            self.human_selection_k_nearest = config.getint('human_selection', 'k_nearest', fallback=3)
            self.human_selection_k_threat = config.getint(
                'human_selection', 'k_threat',
                fallback=max(0, self.num_humans - self.human_selection_k_nearest)
            )
            self.human_selection_encoder_order = config.get('human_selection', 'encoder_order', fallback='preserve').strip().lower()
            self.human_relation_frame = config.get('human_selection', 'relation_frame', fallback='token').strip().lower()
        else:
            self.human_selection_mode = 'nearest'
            self.human_selection_horizon = 5
            self.human_selection_ttc_weight = 2.0
            self.human_selection_clearance_weight = 1.0
            self.human_selection_distance_weight = 0.15
            self.human_selection_closing_weight = 0.25
            self.human_selection_k_nearest = 3
            self.human_selection_k_threat = max(0, self.num_humans - self.human_selection_k_nearest)
            self.human_selection_encoder_order = 'distance'
            self.human_relation_frame = 'legacy'
        self._human_selection_logged = False
        self.belief_start = self.base_token_dim
        self.belief_end = min(self.token_dim, self.base_token_dim + self.belief_dim)
        self.bayesian_core = BayesianCoreScorer(config)
        self.bayesian_core_enable = self.bayesian_core.enabled
        self.bayesian_rollout_horizon = self.bayesian_core.rollout_horizon
        self.bayesian_train_rollout_horizon = self.bayesian_core.train_rollout_horizon
        self.bayesian_risk_weight = self.bayesian_core.risk_weight
        self.bayesian_uncertainty_weight = self.bayesian_core.uncertainty_weight
        self.bayesian_klda_weight = self.bayesian_core.klda_weight
        self.bayesian_action_residual_weight = self.bayesian_core.action_residual_weight
        self.bayesian_value_bootstrap_weight = self.bayesian_core.value_bootstrap_weight
        self.bayesian_reward_weight = self.bayesian_core.reward_weight
        self.bayesian_safe_distance = self.bayesian_core.safe_distance
        self.bayesian_epistemic_weight = self.bayesian_core.epistemic_weight
        self.bayesian_require_action_model = self.bayesian_core.require_action_model
        self.bayesian_require_action_indices = self.bayesian_core.require_action_indices
        self._bayes_core_logged = False
        self._q_only_logged = False
        if config.has_option('env', 'time_step'):
            self.env_dt = config.getfloat('env', 'time_step')
        elif config.has_option('action_space', 'time_step'):
            self.env_dt = config.getfloat('action_space', 'time_step')
        else:
            raise ValueError("Missing time_step in config (expected [env] time_step or [action_space] time_step).")

        # Build network: Spatial -> Temporal -> Heads
        self.spatial_encoder = EnhancedSpatialEncoder(
            d_model=d_model,
            token_dim=self.token_dim,
            base_token_dim=self.base_token_dim,
            relation_dim=8,
            human_start=self.human_start,
            num_humans=self.num_humans,
            human_order=self.human_selection_encoder_order,
            relation_frame=self.human_relation_frame,
        )
        self.temporal_encoder = MambaTemporalEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Discrete head is retained for IL/diagnostics; BayesianCoreScorer owns action selection.

        # Read action space size from config (strict)
        n_speeds = config.getint('policy', 'n_speeds')
        n_headings = config.getint('policy', 'n_headings')
        include_stop = config.getboolean('policy', 'include_stop', fallback=False)
        self.include_stop = include_stop

        self.n_actions = n_speeds * n_headings + (1 if include_stop else 0)

        # Actor/proposal head: outputs behavior-prior logits for each discrete action.
        # RL critics live in q1_head/q2_head so TD targets do not overwrite the
        # action proposal distribution learned by IL.
        self.q_head = nn.Linear(d_model, self.n_actions)
        self.q1_head = nn.Linear(d_model, self.n_actions)
        self.q2_head = nn.Linear(d_model, self.n_actions)
        self.bayes_reranker_head = nn.Linear(d_model, self.n_actions)
        nn.init.zeros_(self.bayes_reranker_head.weight)
        nn.init.zeros_(self.bayes_reranker_head.bias)

        # Value-head: scalar bootstrap value used as a residual auxiliary.
        self.value_head = nn.Linear(d_model, 1)

        self.sigma_head = nn.Linear(d_model, 1)
        nn.init.constant_(self.sigma_head.bias, -2.0)  # softplus(-2) ≈ 0.13
        nn.init.normal_(self.sigma_head.weight, std=0.01)  # allow sigma to learn

        self.gdbn_module = None  # type: Optional[object]

        self.beta_uncertainty = config.getfloat('eval_protocol', 'beta_uncertainty', fallback=0.0)

        logging.info(f"[DISCRETE-Q] Q-head: {self.n_actions} actions ({n_speeds}×{n_headings}, stop={include_stop})")
        logging.info(f"[VALUE-HEAD] mean μ_V")
        logging.info(f"[SIGMA-HEAD] heteroscedastic σ_V")

        self._phase = 'train'

        # Environment interface compatibility: Single agent training
        self.multiagent_training = False

        # RL/task attributes (strict from config)
        self.gamma = config.getfloat(
            'train', 'gamma',
            fallback=config.getfloat('trainer', 'gamma', fallback=0.99)
        )
        if config.has_option('action_space', 'time_step'):
            self.time_step = config.getfloat('action_space', 'time_step')
        elif config.has_option('env', 'time_step'):
            self.time_step = config.getfloat('env', 'time_step')
        else:
            raise ValueError("Missing time_step in config (expected [action_space] time_step or [env] time_step).")
        self.success_radius = config.getfloat('robot', 'success_radius')

        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.progress_reward = config.getfloat('reward', 'progress_reward', fallback=0.0)
        self.time_penalty = config.getfloat('reward', 'time_penalty', fallback=0.0)
        self.stand_penalty = config.getfloat('reward', 'stand_penalty', fallback=0.0)

        if config.has_section('bayesian_rl') and config.has_option('bayesian_rl', 'epsilon_start'):
            self.epsilon = config.getfloat('bayesian_rl', 'epsilon_start')
        elif config.has_section('train') and config.has_option('train', 'epsilon_start'):
            self.epsilon = config.getfloat('train', 'epsilon_start')
        elif config.has_section('sarl') and config.has_option('sarl', 'epsilon_start'):
            self.epsilon = config.getfloat('sarl', 'epsilon_start')
        elif config.has_section('doubleq') and config.has_option('doubleq', 'epsilon_start'):
            self.epsilon = config.getfloat('doubleq', 'epsilon_start')
        else:
            raise ValueError("Missing epsilon_start in config ([bayesian_rl], [sarl], or [doubleq])")
        self.action_space = None
        self.last_state = None
        self.use_bayesian_planner = bool(self.bayesian_core_enable)

        # ==== Test-time safety tweaks (eval-only) ====
        if config.has_section('eval_protocol'):
            self.test_min_clearance = config.getfloat('eval_protocol', 'safety_margin', fallback=
                                                     config.getfloat('eval_protocol', 'discomfort_threshold', fallback=0.0))
            self.test_risk_lambda = config.getfloat('eval_protocol', 'risk_lambda', fallback=0.0)
            self.test_action_smoothing = config.getfloat('eval_protocol', 'action_smoothing', fallback=0.0)
            self.q_only_rerank_topk = config.getint('eval_protocol', 'q_only_rerank_topk', fallback=0)
            self.q_only_progress_weight = config.getfloat('eval_protocol', 'q_only_progress_weight', fallback=0.0)
            self.q_only_speed_weight = config.getfloat('eval_protocol', 'q_only_speed_weight', fallback=0.0)
            self.q_only_clearance_margin = config.getfloat('eval_protocol', 'q_only_clearance_margin', fallback=0.0)
            self.q_only_clearance_penalty = config.getfloat('eval_protocol', 'q_only_clearance_penalty', fallback=0.0)
            self.q_only_stop_penalty = config.getfloat('eval_protocol', 'q_only_stop_penalty', fallback=0.0)
            self.utility_far_goal_radius = config.getfloat('eval_protocol', 'utility_far_goal_radius', fallback=2.0)
            self.utility_far_topk = config.getint('eval_protocol', 'utility_far_topk', fallback=self.q_only_rerank_topk)
            self.utility_progress_floor = config.getfloat('eval_protocol', 'utility_progress_floor', fallback=0.04)
            self.utility_far_progress_weight = config.getfloat('eval_protocol', 'utility_far_progress_weight', fallback=0.0)
            self.utility_far_speed_weight = config.getfloat('eval_protocol', 'utility_far_speed_weight', fallback=0.0)
            self.utility_far_min_clearance = config.getfloat('eval_protocol', 'utility_far_min_clearance', fallback=0.20)
            self.bayesian_rerank_topk = config.getint('eval_protocol', 'bayesian_rerank_topk', fallback=0)
            self.bayesian_progress_weight = config.getfloat('eval_protocol', 'bayesian_progress_weight', fallback=0.0)
            self.bayesian_speed_weight = config.getfloat('eval_protocol', 'bayesian_speed_weight', fallback=0.0)
            self.bayesian_clearance_margin = config.getfloat('eval_protocol', 'bayesian_clearance_margin', fallback=0.0)
            self.bayesian_clearance_penalty = config.getfloat('eval_protocol', 'bayesian_clearance_penalty', fallback=0.0)
            self.bayesian_stop_penalty = config.getfloat('eval_protocol', 'bayesian_stop_penalty', fallback=0.0)
            self.bayesian_q_veto_mode = config.getboolean('eval_protocol', 'bayesian_q_veto_mode', fallback=True)
            self.bayesian_train_hard_veto = config.getboolean(
                'offline_to_online', 'train_hard_veto',
                fallback=config.getboolean('eval_protocol', 'bayesian_train_hard_veto', fallback=True),
            )
            self.bayesian_policy_mode = config.get('eval_protocol', 'bayesian_policy_mode', fallback='safety_layer').strip().lower()
            self.bayesian_follow_q_only_shape = config.getboolean('eval_protocol', 'bayesian_follow_q_only_shape', fallback=True)
            self.bayesian_rl_reranker_enable = config.getboolean(
                'offline_to_online', 'use_topk_reranker',
                fallback=config.getboolean('eval_protocol', 'bayesian_rl_reranker_enable', fallback=False),
            )
            self.bayesian_rl_reranker_topk = config.getint(
                'offline_to_online', 'reranker_topk',
                fallback=config.getint('eval_protocol', 'bayesian_rl_reranker_topk', fallback=self.bayesian_rerank_topk),
            )
            self.bayesian_rl_reranker_scale = config.getfloat(
                'offline_to_online', 'reranker_residual_scale',
                fallback=config.getfloat('eval_protocol', 'bayesian_rl_reranker_scale', fallback=1.0),
            )
            self.bayesian_value_lookahead_enable = config.getboolean(
                'offline_to_online', 'use_value_lookahead',
                fallback=config.getboolean('eval_protocol', 'bayesian_value_lookahead_enable', fallback=False),
            )
            self.bayesian_value_lookahead_risk_weight = config.getfloat(
                'offline_to_online', 'value_lookahead_risk_weight',
                fallback=config.getfloat('eval_protocol', 'bayesian_value_lookahead_risk_weight', fallback=0.0),
            )
            self.bayesian_q_prior_belief_mode = config.get(
                'eval_protocol', 'bayesian_q_prior_belief_mode', fallback='zero'
            ).strip().lower()
            self.bayesian_risk_veto_threshold = config.getfloat('eval_protocol', 'bayesian_risk_veto_threshold', fallback=0.0)
            self.bayesian_train_risk_veto_threshold = config.getfloat(
                'offline_to_online', 'train_risk_veto_threshold',
                fallback=self.bayesian_risk_veto_threshold,
            )
            self.bayesian_override_min_clearance = config.getfloat(
                'eval_protocol', 'bayesian_override_min_clearance',
                fallback=max(self.test_min_clearance, self.bayesian_clearance_margin)
            )
            self.extra_human_veto_clearance = config.getfloat(
                'eval_protocol', 'extra_human_veto_clearance', fallback=0.05
            )
            self.extra_human_veto_horizon = config.getint(
                'eval_protocol', 'extra_human_veto_horizon', fallback=3
            )
            self.extra_human_penalty_margin = config.getfloat(
                'eval_protocol', 'extra_human_penalty_margin', fallback=0.25
            )
            self.extra_human_penalty_weight = config.getfloat(
                'eval_protocol', 'extra_human_penalty_weight', fallback=4.0
            )
        else:
            self.test_min_clearance = 0.0
            self.test_risk_lambda = 0.0
            self.test_action_smoothing = 0.0
            self.q_only_rerank_topk = 0
            self.q_only_progress_weight = 0.0
            self.q_only_speed_weight = 0.0
            self.q_only_clearance_margin = 0.0
            self.q_only_clearance_penalty = 0.0
            self.q_only_stop_penalty = 0.0
            self.utility_far_goal_radius = 2.0
            self.utility_far_topk = 0
            self.utility_progress_floor = 0.04
            self.utility_far_progress_weight = 0.0
            self.utility_far_speed_weight = 0.0
            self.utility_far_min_clearance = 0.20
            self.bayesian_rerank_topk = 0
            self.bayesian_progress_weight = 0.0
            self.bayesian_speed_weight = 0.0
            self.bayesian_clearance_margin = 0.0
            self.bayesian_clearance_penalty = 0.0
            self.bayesian_stop_penalty = 0.0
            self.bayesian_q_veto_mode = True
            self.bayesian_policy_mode = 'safety_layer'
            self.bayesian_follow_q_only_shape = True
            self.bayesian_rl_reranker_enable = False
            self.bayesian_rl_reranker_topk = 0
            self.bayesian_rl_reranker_scale = 1.0
            self.bayesian_value_lookahead_enable = False
            self.bayesian_value_lookahead_risk_weight = 0.0
            self.bayesian_q_prior_belief_mode = 'zero'
            self.bayesian_risk_veto_threshold = 0.0
            self.bayesian_train_risk_veto_threshold = 0.0
            self.bayesian_override_min_clearance = 0.0
            self.extra_human_veto_clearance = 0.05
            self.extra_human_veto_horizon = 3
            self.extra_human_penalty_margin = 0.25
            self.extra_human_penalty_weight = 4.0
        self._last_action = None  # for test-time smoothing
        self._q_only_shape_logged = False
        self._bayesian_shape_logged = False
        self._bayesian_fallback_logged = False
        self._bayesian_prior_logged = False
        self._bayesian_q_veto_logged = False
        self._value_lookahead_logged = False
        self._bayesian_safety_stats = {'calls': 0, 'vetoed': 0, 'fallbacks': 0}
        self._reset_bayesian_risk_episode_stats()

        # Training mode: 'bc' (fast single-step prediction, skip temporal) or 'rl' (full temporal modeling)
        self.training_mode = 'rl'  # Default RL mode

        from collections import deque
        self._history = deque(maxlen=self.seq_len)
        self._zero_token = np.zeros((self.num_entities, self.token_dim), dtype=np.float32)

        # Move all modules to target device
        self.to(self.device)

        # SSOT verification log
        logging.info("=" * 80)
        logging.info("[MAMBA-BACKBONE] Mamba backbone configuration:")
        logging.info(f"  d_model={d_model}, n_layers={n_layers}, d_state={d_state}")
        logging.info(f"  d_conv={d_conv}, expand={expand}")
        logging.info(f"[SEQ-FINAL] Sequence Length Configuration (SSOT):")
        logging.info(f"  seq_len={self.seq_len} (from train.config:[buffer])")
        logging.info(f"  T={self.seq_len} (should match policy.config:[temporal].T)")
        logging.info(
            f"  belief_token={self.belief_enabled} token_dim={self.token_dim} "
            f"base_dim={self.base_token_dim} belief_dim={self.belief_dim}"
        )
        logging.info(
            f"  bayesian_core={self.bayesian_core_enable} horizon={self.bayesian_rollout_horizon} "
            f"risk_w={self.bayesian_risk_weight} action_residual_w={self.bayesian_action_residual_weight} "
            f"value_bootstrap_w={self.bayesian_value_bootstrap_weight}"
        )
        logging.info("[ARCH-FINAL] Architecture: EnhancedSpatial(3-way:robot+human+relation) -> Mamba -> DualHead")
        logging.info(f"[DEVICE] {self.device}")
        logging.info("=" * 80)

    def initialize_rl_critics_from_actor(self):
        """Warm-start twin RL critics from the IL actor logits for old checkpoints."""
        with torch.no_grad():
            for critic in (self.q1_head, self.q2_head):
                critic.weight.copy_(self.q_head.weight)
                critic.bias.copy_(self.q_head.bias)
        logging.info("[OFF2ON] initialized q1/q2 critics from IL q_head logits")

    def set_env_dt(self, dt: float):
        self.env_dt = dt
        logging.info(f"[BAYES-RL] Set env_dt={dt}")

    def set_phase(self, phase: str):
        self._phase = phase

    def set_training_mode(self, mode: str):
        """

        Args:
 mode: 'bc' () 'rl' ()
        """
        assert mode in ['bc', 'rl'], f"Invalid training mode: {mode}, must be 'bc' or 'rl'"
        self.training_mode = mode
        logging.info(f"[TRAINING-MODE] Switched to '{mode}' mode")

    def _make_token_frame(self, base_tokens, belief_vec=None) -> np.ndarray:
        """Create one configured token frame from base kinematic tokens plus optional belief."""
        frame = np.zeros((self.num_entities, self.token_dim), dtype=np.float32)
        base_tokens = np.asarray(base_tokens, dtype=np.float32)
        rows = min(self.num_entities, base_tokens.shape[0])
        cols = min(self.base_token_dim, base_tokens.shape[1], self.token_dim)
        frame[:rows, :cols] = base_tokens[:rows, :cols]

        if self.belief_enabled and belief_vec is not None and self.belief_end > self.belief_start:
            belief_vec = np.asarray(belief_vec, dtype=np.float32)
            b_rows = min(self.num_humans, belief_vec.shape[0])
            b_cols = min(self.belief_end - self.belief_start, belief_vec.shape[1])
            frame[
                self.human_start:self.human_start + b_rows,
                self.belief_start:self.belief_start + b_cols
            ] = belief_vec[:b_rows, :b_cols]
        return frame

    def _zero_belief_frame(self, frame) -> np.ndarray:
        """Return a token frame with belief columns cleared for BC-compatible Q priors."""
        zero_frame = np.array(frame, dtype=np.float32, copy=True)
        if self.belief_end > self.belief_start:
            zero_frame[:, self.belief_start:self.belief_end] = 0.0
        return zero_frame

    def reset_episode_stats(self):
        """ episode Explorer """
        self._history.clear()
        self._last_action = None
        self._reset_bayesian_risk_episode_stats()
        if self.gdbn_module is not None:
            self.gdbn_module.reset(n_peds=self.num_humans)

    def _reset_bayesian_risk_episode_stats(self):
        self._bayesian_risk_episode_stats = {
            'selected': [],
            'candidate_min': [],
            'candidate_mean': [],
            'candidate_max': [],
            'selected_clearance': [],
            'candidate_clearance_min': [],
            'calls': 0,
            'vetoed': 0,
            'fallbacks': 0,
            'candidate_actions': 0,
            'risk_gt_005': 0,
            'risk_gt_010': 0,
            'risk_gt_020': 0,
            'risk_gt_100': 0,
            'risk_gt_200': 0,
            'risk_gt_350': 0,
        }

    def pop_bayesian_risk_episode_stats(self):
        """Return and reset per-episode Bayesian risk diagnostics."""
        stats = getattr(self, '_bayesian_risk_episode_stats', None)
        if stats is None:
            self._reset_bayesian_risk_episode_stats()
            stats = self._bayesian_risk_episode_stats
        out = {}
        for key, value in stats.items():
            out[key] = list(value) if isinstance(value, list) else value
        self._reset_bayesian_risk_episode_stats()
        return out

    def _capture_candidate_diagnostics(self, state, scores, best_idx, use_bayes, result=None, dmins_batch=None):
        """Cache a compact top-k action-score breakdown for external test diagnostics."""
        if not bool(getattr(self, 'candidate_diagnostics_enable', False)):
            self._last_candidate_diagnostics = None
            return
        try:
            n = int(scores.numel())
            topk = max(1, min(int(getattr(self, 'candidate_diagnostics_topk', 8) or 8), n))
            finite_mask = torch.isfinite(scores) & (scores > -1e8)
            rank_scores = scores.clone()
            if finite_mask.any():
                rank_scores = rank_scores.masked_fill(~finite_mask, -1e9)
            top_indices = torch.topk(rank_scores, k=topk).indices.detach().cpu().tolist()

            action_residual = result.get('action_residual') if isinstance(result, dict) else None
            value_next = result.get('value_next') if isinstance(result, dict) else None
            risk_t = result.get('risk') if isinstance(result, dict) else None
            efe_t = result.get('efe') if isinstance(result, dict) else None
            entropy_t = result.get('entropy') if isinstance(result, dict) else None
            klda_t = result.get('klda') if isinstance(result, dict) else None
            curr_goal_dist = float(np.hypot(
                state.self_state.px - state.self_state.gx,
                state.self_state.py - state.self_state.gy,
            ))
            safe_clearance = float(getattr(self, 'candidate_diag_safe_clearance', 0.20) or 0.20)
            thresholds_raw = getattr(self, 'candidate_diag_thresholds', (0.22, 0.20, 0.15))
            try:
                clearance_thresholds = [float(x) for x in thresholds_raw]
            except Exception:
                clearance_thresholds = [0.22, 0.20, 0.15]

            def _tensor_val(t, idx):
                if isinstance(t, torch.Tensor) and t.numel() > idx:
                    return float(t[idx].detach().cpu().item())
                return None

            candidates = []
            for idx in top_indices:
                action = self.action_space[int(idx)]
                next_self = self.propagate(state.self_state, action)
                next_goal_dist = float(np.hypot(
                    next_self.px - next_self.gx,
                    next_self.py - next_self.gy,
                ))
                clearance = None
                if dmins_batch is not None and int(idx) < len(dmins_batch):
                    clearance = float(dmins_batch[int(idx)])
                speed = float(np.hypot(action.vx, action.vy))
                progress = curr_goal_dist - next_goal_dist
                candidates.append({
                    'idx': int(idx),
                    'vx': float(action.vx),
                    'vy': float(action.vy),
                    'speed': speed,
                    'progress': float(progress),
                    'clearance': clearance,
                    'safe_by_clearance': bool(clearance is not None and clearance >= safe_clearance),
                    'final_score': _tensor_val(scores, int(idx)),
                    'q_score': _tensor_val(action_residual, int(idx)),
                    'value_next': _tensor_val(value_next, int(idx)),
                    'risk': _tensor_val(risk_t, int(idx)),
                    'efe': _tensor_val(efe_t, int(idx)),
                    'entropy': _tensor_val(entropy_t, int(idx)),
                    'klda': _tensor_val(klda_t, int(idx)),
                })

            global_clearance = {}
            if dmins_batch is not None and len(dmins_batch) == n:
                clearance_arr = np.asarray(dmins_batch, dtype=np.float64)
                finite_clear = np.isfinite(clearance_arr)
                if finite_clear.any():
                    safest_idx = int(np.nanargmax(clearance_arr))
                    global_clearance = {
                        'max_clearance': float(clearance_arr[safest_idx]),
                        'safest_idx': safest_idx,
                        'safest_vx': float(self.action_space[safest_idx].vx),
                        'safest_vy': float(self.action_space[safest_idx].vy),
                        'safest_in_recorded_topk': bool(safest_idx in top_indices),
                        'selected_clearance': float(clearance_arr[int(best_idx)]),
                    }
                    for thr in clearance_thresholds:
                        safe_indices = np.where(clearance_arr >= float(thr))[0]
                        label = f"safe_ge_{float(thr):.2f}"
                        global_clearance[label] = {
                            'any': bool(safe_indices.size > 0),
                            'count': int(safe_indices.size),
                            'best_idx': int(safe_indices[np.argmax(clearance_arr[safe_indices])]) if safe_indices.size else None,
                            'selected_safe': bool(clearance_arr[int(best_idx)] >= float(thr)),
                        }

            try:
                selected_humans = self._selected_human_indices(state.self_state, state.human_states)
            except Exception:
                selected_humans = []

            self._last_candidate_diagnostics = {
                'use_bayes': bool(use_bayes),
                'selected_idx': int(best_idx),
                'selected_human_indices': [int(i) for i in selected_humans],
                'n_visible_humans': int(len(state.human_states or [])),
                'safe_clearance_threshold': safe_clearance,
                'global_clearance': global_clearance,
                'topk': candidates,
            }
        except Exception as exc:
            self._last_candidate_diagnostics = {'error': str(exc)}

    def encode_last_features(self, joint_tokens):
        """Return the last temporal feature used by value/q/reranker heads."""
        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)  # [B, 1, 8, 23]
        elif joint_tokens.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected [B, {self.num_entities}, {self.token_dim}] or [B, T, {self.num_entities}, {self.token_dim}], got {joint_tokens.shape}")

        spatial_features = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        temporal_features = self.temporal_encoder(spatial_features)  # [B, T, d_model]
        return temporal_features[:, -1, :]  # [B, d_model]

    def forward_value(self, joint_tokens, mask=None):
        """Critic"""
        last_features = self.encode_last_features(joint_tokens)
        values = self.value_head(last_features).squeeze(-1)  # [B]

        return values

    def forward_sigma(self, joint_tokens) -> 'torch.Tensor':
        """ σ_V(s) [B] forward_value backbone"""
        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)
        spatial_features = self.spatial_encoder(joint_tokens)
        temporal_features = self.temporal_encoder(spatial_features)
        last_features = temporal_features[:, -1, :]
        sigma = F.softplus(self.sigma_head(last_features)).squeeze(-1) + 1e-4  # [B]
        return sigma

    def forward_value_with_uncertainty(self, joint_tokens):
        """ (μ_V, σ_V)backbone"""
        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)
        spatial_features = self.spatial_encoder(joint_tokens)
        temporal_features = self.temporal_encoder(spatial_features)
        last_features = temporal_features[:, -1, :]
        mu    = self.value_head(last_features).squeeze(-1)
        sigma = F.softplus(self.sigma_head(last_features)).squeeze(-1) + 1e-4
        return mu, sigma

    def forward_q(self, joint_tokens, return_sequence=False):
        if joint_tokens.device != next(self.parameters()).device:
            joint_tokens = joint_tokens.to(next(self.parameters()).device)

        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)  # [B, 1, 8, 23]
        elif joint_tokens.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected [B, {self.num_entities}, {self.token_dim}] or [B, T, {self.num_entities}, {self.token_dim}], got {joint_tokens.shape}")

        spatial_features = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        temporal_features = self.temporal_encoder(spatial_features)  # [B, T, d_model]

        if return_sequence:
            q_values = self.q_head(temporal_features)  # [B, T, 80]
        else:
            last_features = temporal_features[:, -1, :]  # [B, d_model]
            q_values = self.q_head(last_features)  # [B, 80]

        return q_values

    def forward_reranker(self, joint_tokens, return_sequence=False):
        """Residual top-k reranker. Zero initialization preserves IL proposal."""
        if joint_tokens.device != next(self.parameters()).device:
            joint_tokens = joint_tokens.to(next(self.parameters()).device)

        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)
        elif joint_tokens.dim() == 4:
            pass
        else:
            raise ValueError(
                f"Expected [B, {self.num_entities}, {self.token_dim}] or "
                f"[B, T, {self.num_entities}, {self.token_dim}], got {joint_tokens.shape}"
            )

        spatial_features = self.spatial_encoder(joint_tokens)
        temporal_features = self.temporal_encoder(spatial_features)
        if return_sequence:
            return self.bayes_reranker_head(temporal_features)
        last_features = temporal_features[:, -1, :]
        return self.bayes_reranker_head(last_features)

    def forward_critic(self, joint_tokens, return_sequence=False):
        """Twin discrete critics Q1/Q2 for offline-to-online RL."""
        if joint_tokens.device != next(self.parameters()).device:
            joint_tokens = joint_tokens.to(next(self.parameters()).device)

        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)
        elif joint_tokens.dim() == 4:
            pass
        else:
            raise ValueError(
                f"Expected [B, {self.num_entities}, {self.token_dim}] or "
                f"[B, T, {self.num_entities}, {self.token_dim}], got {joint_tokens.shape}"
            )

        spatial_features = self.spatial_encoder(joint_tokens)
        temporal_features = self.temporal_encoder(spatial_features)
        if return_sequence:
            return self.q1_head(temporal_features), self.q2_head(temporal_features)
        last_features = temporal_features[:, -1, :]
        return self.q1_head(last_features), self.q2_head(last_features)

    def act(self, state):
        return self.select_action(state)

    def predict(self, state, deterministic: bool = True, return_idx: bool = False, **kwargs):
        action, best_idx = self.select_action(state)
        if return_idx:
            return action, best_idx
        return action

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    # ============ Bayesian-Core Planning Methods ============

    def build_action_space(self, v_pref):
        from crowd_sim.envs.utils.action import ActionXY
        from crowd_nav.contracts_gdbn import discrete_index_to_action, GRID, grid_action_dim

        n_actions = grid_action_dim(GRID)

        action_space = []
        for idx in range(n_actions):
            vx, vy = discrete_index_to_action(idx)
            action_space.append(ActionXY(vx, vy))

        self.action_space = action_space
        logging.info(f"[ACTION-SPACE] Built {len(action_space)} actions (GRID {GRID['n_speeds']}×{GRID['n_headings']}, stop={GRID.get('include_stop', False)})")

    def continuous_to_discrete_index(self, action):
        from crowd_sim.envs.utils.action import ActionXY

        if self.action_space is None:
            self.build_action_space(1.0)

        if isinstance(action, ActionXY):
            vx, vy = action.vx, action.vy
        elif isinstance(action, (tuple, list)) and len(action) >= 2:
            vx, vy = action[0], action[1]
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        min_dist = float('inf')
        best_idx = 0
        for idx, discrete_action in enumerate(self.action_space):
            dist = (discrete_action.vx - vx)**2 + (discrete_action.vy - vy)**2
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        return best_idx

    def propagate(self, state, action):
        """Map a continuous action to the configured discrete action grid."""
        from crowd_sim.envs.utils.state import ObservableState, FullState

        if isinstance(state, ObservableState):
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            return ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            return FullState(next_px, next_py, action.vx, action.vy, state.radius,
                            state.gx, state.gy, state.v_pref, state.theta)
        else:
            raise ValueError(f"Unknown state type: {type(state)}")

    def compute_reward(self, nav, humans, prev_nav=None, action=None):
        """rewardenv.step"""
        dmin = float('inf')
        collision = False
        for human in humans:
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            dmin = min(dmin, dist)

        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < self.success_radius

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
            if dmin < self.discomfort_dist:
                penalty = self.discomfort_penalty_factor * (self.discomfort_dist - dmin) * self.time_step
                reward -= penalty
            return reward

    def reach_destination(self, state):
        """ success_radius"""
        self_state = state.self_state if hasattr(state, 'self_state') else state
        return np.linalg.norm((self_state.px - self_state.gx, self_state.py - self_state.gy)) < self.success_radius

    def _build_joint_state_34(self, robot_state, human_states):
        """[9 + num_humans*5] joint state. Name kept for compatibility."""
        from crowd_nav.contracts_gdbn import build_selected_joint_state_array

        selected_state, selected_indices = build_selected_joint_state_array(
            robot_state,
            human_states or [],
            self.num_humans,
            mode=getattr(self, 'human_selection_mode', 'threat'),
            dt=float(getattr(self, 'time_step', 0.25)),
            horizon=int(getattr(self, 'human_selection_horizon', 5)),
            ttc_weight=float(getattr(self, 'human_selection_ttc_weight', 2.0)),
            clearance_weight=float(getattr(self, 'human_selection_clearance_weight', 1.0)),
            distance_weight=float(getattr(self, 'human_selection_distance_weight', 0.15)),
            closing_weight=float(getattr(self, 'human_selection_closing_weight', 0.25)),
            k_nearest=int(getattr(self, 'human_selection_k_nearest', 3)),
            k_threat=int(getattr(self, 'human_selection_k_threat', max(0, int(self.num_humans) - 3))),
            return_indices=True,
        )
        if not self._human_selection_logged:
            logging.info(
                "[HUMAN-SELECTION] mode=%s k=%d hybrid=%d+%d horizon=%d encoder_order=%s relation_frame=%s visible=%d selected=%s",
                getattr(self, 'human_selection_mode', 'threat'), int(self.num_humans),
                int(getattr(self, 'human_selection_k_nearest', 3)),
                int(getattr(self, 'human_selection_k_threat', max(0, int(self.num_humans) - 3))),
                int(getattr(self, 'human_selection_horizon', 5)),
                getattr(self, 'human_selection_encoder_order', 'preserve'),
                getattr(self, 'human_relation_frame', 'token'),
                len(human_states or []), selected_indices,
            )
            self._human_selection_logged = True
        return selected_state

    def _selected_human_indices(self, robot_state, human_states):
        from crowd_nav.contracts_gdbn import select_human_indices
        return select_human_indices(
            robot_state,
            human_states or [],
            self.num_humans,
            mode=getattr(self, 'human_selection_mode', 'threat'),
            dt=float(getattr(self, 'time_step', 0.25)),
            horizon=int(getattr(self, 'human_selection_horizon', 5)),
            ttc_weight=float(getattr(self, 'human_selection_ttc_weight', 2.0)),
            clearance_weight=float(getattr(self, 'human_selection_clearance_weight', 1.0)),
            distance_weight=float(getattr(self, 'human_selection_distance_weight', 0.15)),
            closing_weight=float(getattr(self, 'human_selection_closing_weight', 0.25)),
            k_nearest=int(getattr(self, 'human_selection_k_nearest', 3)),
            k_threat=int(getattr(self, 'human_selection_k_threat', max(0, int(self.num_humans) - 3))),
        )

    def score_actions(self, state):
        """Compute Bayesian score for all candidate actions.

        Updates GDBN belief, runs Mamba forward pass, queries BayesianCoreScorer.
        Returns a dict of tensors/data needed by select_action().
        """
        # GDBN uses np.linalg internally. Limit BLAS once per policy instance instead
        # of entering/exiting threadpoolctl on every simulation step.
        if not self._blas_limit_active:
            try:
                from threadpoolctl import threadpool_limits as _tpl
                self._blas_limit_ctx = _tpl(limits=1, user_api='blas')
                self._blas_limit_ctx.__enter__()
            except Exception:
                self._blas_limit_ctx = None
            self._blas_limit_active = True
        return self._score_actions_inner(state)

    def _score_actions_inner(self, state):
        from crowd_sim.envs.utils.action import ActionXY
        from crowd_nav.contracts_gdbn import _batch_joint34_to_tokens_vectorized

        current_34 = self._build_joint_state_34(state.self_state, state.human_states)
        current_token_13 = _batch_joint34_to_tokens_vectorized(current_34.reshape(1, -1))[0]

        skip_gdbn_compute = bool(getattr(self, 'skip_gdbn_compute', False))
        belief_vec = None
        if self.belief_enabled and self.gdbn_module is not None and not skip_gdbn_compute:
            try:
                self.gdbn_module.update(current_34)
                if hasattr(self.gdbn_module, 'get_per_ped_belief_vec'):
                    belief_vec = self.gdbn_module.get_per_ped_belief_vec()
            except Exception:
                belief_vec = None
        network_belief_vec = None if self.disable_belief_tokens else belief_vec
        current_token = self._make_token_frame(current_token_13, network_belief_vec)
        current_token_zero_belief = self._make_token_frame(current_token_13, belief_vec=None)
        base_hist = list(self._history) + [current_token]
        q_prior_belief_mode = str(getattr(self, 'bayesian_q_prior_belief_mode', 'zero')).strip().lower()
        if q_prior_belief_mode in ('real', 'live', 'gdbn', 'belief'):
            prior_hist = base_hist
            q_prior_label = 'real-belief'
        else:
            prior_hist = [self._zero_belief_frame(frame) for frame in self._history] + [current_token_zero_belief]
            q_prior_label = 'zero-belief'

        need_gdbn_rollout = bool(
            self.bayesian_core_enable and
            self.gdbn_module is not None and
            not skip_gdbn_compute
        )
        if need_gdbn_rollout:
            self.bayesian_core.validate(self.gdbn_module)

        need_value_next = bool(getattr(self, 'bayesian_value_lookahead_enable', False))
        try:
            need_value_next = need_value_next or float(getattr(self.bayesian_core, 'value_bootstrap_weight', 0.0)) != 0.0
        except Exception:
            pass
        need_value_next = need_value_next or float(getattr(self, 'beta_uncertainty', 0.0) or 0.0) > 0.0
        next_humans = [self.propagate(h, ActionXY(h.vx, h.vy)) for h in state.human_states]
        candidate_next_tokens_13 = []
        rewards_batch = []
        dmins_batch = []
        for action in self.action_space:
            next_self = self.propagate(state.self_state, action)
            rewards_batch.append(self.compute_reward(next_self, next_humans, prev_nav=state.self_state, action=action))
            dmin = min(
                np.hypot(next_self.px - h.px, next_self.py - h.py) - next_self.radius - h.radius
                for h in next_humans
            ) if next_humans else float('inf')
            dmins_batch.append(dmin)
            if need_value_next:
                next_state_34 = self._build_joint_state_34(next_self, next_humans)
                candidate_next_tokens_13.append(_batch_joint34_to_tokens_vectorized(next_state_34.reshape(1, -1))[0])

        if need_gdbn_rollout:
            bayes_risk_batch, bayes_tail_risk_batch, bayes_entropy_batch, bayes_klda_batch, bayes_epistemic_batch, belief_candidates = (
                self.bayesian_core.rollout_candidates(
                    self.gdbn_module, current_34, self.action_space,
                    dt=self.time_step, robot_radius=getattr(state.self_state, 'radius', 0.3),
                    initial_belief_vec=belief_vec,
                )
            )
        else:
            n = len(self.action_space)
            belief_candidates = [belief_vec] * n
            bayes_risk_batch = [0.0] * n
            bayes_tail_risk_batch = [0.0] * n
            bayes_entropy_batch = [0.0] * n
            bayes_klda_batch = [0.0] * n
            bayes_epistemic_batch = [0.0] * n

        next_mu = None
        next_sigma = None
        next_features = None
        next_states_tensor = None
        if need_value_next:
            next_states_batch = []
            for next_token_13, next_belief_vec in zip(candidate_next_tokens_13, belief_candidates):
                if self.disable_belief_tokens:
                    next_belief_vec = None
                elif next_belief_vec is None:
                    next_belief_vec = belief_vec
                next_token = self._make_token_frame(next_token_13, next_belief_vec)
                temp_hist = (base_hist + [next_token])[-self.seq_len:]
                if len(temp_hist) < self.seq_len:
                    temp_hist = [temp_hist[0]] * (self.seq_len - len(temp_hist)) + temp_hist
                next_states_batch.append(np.array(temp_hist))

        with torch.no_grad():
            current_hist = prior_hist[-self.seq_len:]
            if len(current_hist) < self.seq_len:
                current_hist = [current_hist[0]] * (self.seq_len - len(current_hist)) + current_hist
            current_states_tensor = torch.from_numpy(np.array(current_hist)).float().unsqueeze(0).to(self.device)
            action_residual = self.forward_q(current_states_tensor, return_sequence=False).squeeze(0)
            if bool(getattr(self, 'bayesian_rl_reranker_enable', False)):
                rerank_residual = self.forward_reranker(current_states_tensor, return_sequence=False).squeeze(0)
            else:
                rerank_residual = None
            if not self._bayesian_prior_logged and self._phase in ('test', 'val', 'eval'):
                logging.info("[BAYES-PRIOR] using %s q-prior inside Bayesian scorer", q_prior_label)
                self._bayesian_prior_logged = True
            if need_value_next:
                next_states_tensor = torch.from_numpy(np.array(next_states_batch)).float().to(self.device)
                next_features = self.encode_last_features(next_states_tensor)
                next_mu = self.value_head(next_features).squeeze(-1)
                if self.beta_uncertainty > 0.0:
                    next_sigma = F.softplus(self.sigma_head(next_features)).squeeze(-1) + 1e-4

        rewards_t = torch.tensor(rewards_batch, device=self.device)
        risk_t = torch.tensor(bayes_risk_batch, dtype=torch.float32, device=self.device)
        tail_risk_t = torch.tensor(
            bayes_tail_risk_batch, dtype=torch.float32, device=self.device
        )
        entropy_t = torch.tensor(bayes_entropy_batch, dtype=torch.float32, device=self.device)
        klda_t = torch.tensor(bayes_klda_batch, dtype=torch.float32, device=self.device)
        epistemic_t = torch.tensor(bayes_epistemic_batch, dtype=torch.float32, device=self.device)

        scores, efe = self.bayesian_core.score(
            rewards_t, risk_t, entropy_t, klda_t, epistemic_t,
            action_residual=action_residual,
            residual_value=next_mu,
            gamma=self.gamma,
            residual_sigma=next_sigma,
            beta_uncertainty=self.beta_uncertainty,
        )
        return {
            'scores': scores,
            'efe': efe,
            'risk': risk_t,
            'tail_risk': tail_risk_t,
            'entropy': entropy_t,
            'klda': klda_t,
            'dmins': dmins_batch,
            'current_token': current_token,
            'action_residual': action_residual,
            'rerank_residual': rerank_residual,
            'value_next': next_mu,
            'next_features': next_features,
        }

    def _bayesian_proposal_scores(self, state, scores, action_residual=None, rerank_residual=None):
        """Return the action proposal distribution before Bayesian safety veto."""
        if action_residual is None or action_residual.numel() != scores.numel():
            return scores.clone(), 'bayes_score', None

        mode = str(getattr(self, 'bayesian_policy_mode', 'safety_layer')).lower()
        if mode in ('legacy_score', 'efe_score', 'efe'):
            return scores.clone(), 'legacy_efe', int(torch.argmax(scores).item())

        if bool(getattr(self, 'bayesian_follow_q_only_shape', True)):
            proposal = self._shape_q_only_scores(state, action_residual, force=True).clone()
            proposal_mode = 'q_only_shape'
        else:
            proposal = action_residual.clone()
            proposal_mode = 'raw_q'

        if (
            bool(getattr(self, 'bayesian_rl_reranker_enable', False)) and
            isinstance(rerank_residual, torch.Tensor) and
            rerank_residual.numel() == proposal.numel()
        ):
            n = int(proposal.numel())
            topk = int(getattr(self, 'bayesian_rl_reranker_topk', 0))
            if topk <= 0:
                topk = int(getattr(self, 'bayesian_rerank_topk', 0))
            topk = max(1, min(topk, n))
            candidate_mask = torch.zeros(n, dtype=torch.bool, device=proposal.device)
            candidate_mask[torch.topk(proposal, k=topk).indices] = True
            reranked = proposal + float(getattr(self, 'bayesian_rl_reranker_scale', 1.0)) * rerank_residual
            reranked = reranked.masked_fill(~candidate_mask, -1e9)
            return reranked, f'{proposal_mode}+rl_rerank', int(torch.argmax(reranked).item())

        return proposal, proposal_mode, int(torch.argmax(proposal).item())

    def _bayesian_safety_layer_scores(self, state, proposal_scores, action_residual=None, dmins_batch=None, risk=None):
        """Apply Bayesian hard-veto on top of BC/Mamba proposal scores."""
        if self.action_space is None or proposal_scores.numel() == 0:
            return proposal_scores

        shaped = proposal_scores.clone()
        n = int(shaped.numel())
        ranking_scores = proposal_scores if proposal_scores.numel() == n else shaped
        protected_idx = int(torch.argmax(ranking_scores).item())

        topk = int(self.bayesian_rerank_topk)
        candidate_mask = None
        if 0 < topk < n:
            candidate_mask = torch.zeros(n, dtype=torch.bool, device=shaped.device)
            candidate_mask[torch.topk(ranking_scores, k=topk).indices] = True
            shaped = shaped.masked_fill(~candidate_mask, -1e9)

        clearances = []
        for idx, action in enumerate(self.action_space):
            next_self = self.propagate(state.self_state, action)
            if dmins_batch is not None and idx < len(dmins_batch):
                clearance = float(dmins_batch[idx])
            elif state.human_states:
                clearance = min(
                    float(np.hypot(next_self.px - h.px, next_self.py - h.py) - next_self.radius - h.radius)
                    for h in state.human_states
                )
            else:
                clearance = float('inf')
            clearances.append(clearance)

        # Multi-step kinematic veto for extra humans beyond the policy token schema.
        # GDBN risk covers the configured token humans; this covers any remaining humans.
        # For any humans beyond slot-5, we project constant-velocity trajectories over the
        # same horizon and veto actions that would bring the robot within the clearance
        # threshold at any projected step.
        extra_kinematic_veto = None
        extra_min_clearances = None
        extra_humans_beyond5 = []
        covered_humans = int(getattr(self, 'num_humans', 5))
        if state.human_states and len(state.human_states) > covered_humans:
            try:
                selected = set(self._selected_human_indices(state.self_state, state.human_states))
                extra_humans_beyond5 = [
                    h for i, h in enumerate(state.human_states)
                    if i not in selected
                ]
            except Exception:
                sorted_by_dist = sorted(
                    state.human_states,
                    key=lambda h: (h.px - state.self_state.px) ** 2 + (h.py - state.self_state.py) ** 2,
                )
                extra_humans_beyond5 = sorted_by_dist[covered_humans:]

        if extra_humans_beyond5:
            extra_clr_thr = float(getattr(self, 'extra_human_veto_clearance', 0.05))
            extra_hor = int(getattr(self, 'extra_human_veto_horizon', 3))
            if extra_clr_thr > 0.0 and extra_hor > 0:
                robot_r = float(getattr(state.self_state, 'radius', 0.3))
                steps = np.arange(1, extra_hor + 1, dtype=np.float32) * self.time_step  # (H,)

                eh_data = np.array(
                    [[h.px, h.py, h.vx, h.vy, h.radius] for h in extra_humans_beyond5],
                    dtype=np.float32,
                )                                                           # (n_extra, 5)
                eh_px = eh_data[:, 0:1] + eh_data[:, 2:3] * steps         # (n_extra, H)
                eh_py = eh_data[:, 1:2] + eh_data[:, 3:4] * steps         # (n_extra, H)
                coll_r = (robot_r + eh_data[:, 4])[:, None]                # (n_extra, 1)
                rpx0, rpy0 = state.self_state.px, state.self_state.py

                extra_kinematic_veto = []
                extra_min_clearances = []
                for action in self.action_space:
                    r_px = rpx0 + action.vx * steps                        # (H,)
                    r_py = rpy0 + action.vy * steps                        # (H,)
                    dists = np.hypot(r_px - eh_px, r_py - eh_py) - coll_r  # (n_extra, H)
                    min_extra_clearance = float(np.min(dists))
                    extra_min_clearances.append(min_extra_clearance)
                    extra_kinematic_veto.append(min_extra_clearance < extra_clr_thr)

        # Cache for value_lookahead to apply extra-human penalty after V(s') replacement.
        self._last_extra_min_clearances = extra_min_clearances
        self._last_extra_human_count = len(extra_humans_beyond5)

        veto_mask = torch.zeros(n, dtype=torch.bool, device=shaped.device)
        min_clearance = float(self.bayesian_override_min_clearance)
        risk_threshold = float(self.bayesian_risk_veto_threshold)
        if self._phase == 'train' and bool(getattr(self, 'bayesian_train_hard_veto', True)):
            risk_threshold = float(getattr(self, 'bayesian_train_risk_veto_threshold', risk_threshold))
        risk_t = risk if isinstance(risk, torch.Tensor) else None
        for idx, clearance in enumerate(clearances):
            unsafe_clearance = np.isfinite(clearance) and clearance < min_clearance
            unsafe_risk = (
                risk_t is not None and
                risk_threshold > 0.0 and
                float(risk_t[idx].item()) > risk_threshold
            )
            unsafe_extra = (
                extra_kinematic_veto is not None and
                idx < len(extra_kinematic_veto) and
                extra_kinematic_veto[idx]
            )
            if unsafe_clearance or unsafe_risk or unsafe_extra:
                veto_mask[idx] = True

        full_veto_mask = veto_mask.clone()
        if candidate_mask is not None:
            veto_mask = full_veto_mask & candidate_mask
        scope_mask = candidate_mask
        if scope_mask is None:
            scope_mask = torch.ones(n, dtype=torch.bool, device=shaped.device)
        kept_mask = ~veto_mask
        if candidate_mask is not None:
            kept_mask = kept_mask & candidate_mask
        if kept_mask.any():
            shaped = shaped.masked_fill(~kept_mask, -1e9)
            if extra_min_clearances is not None and len(extra_min_clearances) == n:
                penalty_margin = float(getattr(self, 'extra_human_penalty_margin', 0.25))
                penalty_weight = float(getattr(self, 'extra_human_penalty_weight', 4.0))
                if penalty_margin > 0.0 and penalty_weight > 0.0:
                    extra_clear_t = torch.tensor(
                        extra_min_clearances,
                        dtype=shaped.dtype,
                        device=shaped.device,
                    )
                    extra_penalty = penalty_weight * torch.clamp(
                        penalty_margin - extra_clear_t,
                        min=0.0,
                    )
                    shaped = shaped - extra_penalty.masked_fill(~kept_mask, 0.0)
        else:
            # If every proposal-scope candidate is vetoed, keep the least unsafe
            # candidate inside the current proposal scope. This preserves the
            # accepted champion selector instead of expanding to the full grid.
            fallback_scope = scope_mask
            fallback_cost = torch.full_like(shaped, float('inf'))
            fallback_clearances = list(clearances)
            if extra_min_clearances is not None and len(extra_min_clearances) == len(fallback_clearances):
                fallback_clearances = [
                    min(c, ec) if np.isfinite(c) else ec
                    for c, ec in zip(fallback_clearances, extra_min_clearances)
                ]
            clearance_cost = torch.tensor(
                [(-c if np.isfinite(c) else -1e6) for c in fallback_clearances],
                dtype=shaped.dtype,
                device=shaped.device,
            )
            fallback_cost[fallback_scope] = clearance_cost[fallback_scope]
            if risk_t is not None and risk_t.numel() == n:
                fallback_cost[fallback_scope] = fallback_cost[fallback_scope] + 0.05 * risk_t[fallback_scope]
            best_fallback = int(torch.argmin(fallback_cost).item())
            kept_mask = torch.zeros(n, dtype=torch.bool, device=shaped.device)
            kept_mask[best_fallback] = True
            shaped = shaped.masked_fill(~kept_mask, -1e9)

        best_idx = int(torch.argmax(shaped).item())
        fallback_applied = not kept_mask[protected_idx].item() and best_idx != protected_idx

        self._bayesian_safety_stats['calls'] += 1
        self._bayesian_safety_stats['vetoed'] += int(veto_mask.sum().item())
        if fallback_applied:
            self._bayesian_safety_stats['fallbacks'] += 1
        ep_stats = getattr(self, '_bayesian_risk_episode_stats', None)
        if ep_stats is not None:
            ep_stats['calls'] += 1
            ep_stats['vetoed'] += int(veto_mask.sum().item())
            ep_stats['fallbacks'] += int(fallback_applied)
            if risk_t is not None:
                if candidate_mask is not None:
                    risk_scope = risk_t[candidate_mask]
                else:
                    risk_scope = risk_t
                ep_stats['candidate_actions'] += int(risk_scope.numel())
                ep_stats['risk_gt_005'] += int((risk_scope > 0.005).sum().item())
                ep_stats['risk_gt_010'] += int((risk_scope > 0.010).sum().item())
                ep_stats['risk_gt_020'] += int((risk_scope > 0.020).sum().item())
                ep_stats['risk_gt_100'] += int((risk_scope > 0.100).sum().item())
                ep_stats['risk_gt_200'] += int((risk_scope > 0.200).sum().item())
                ep_stats['risk_gt_350'] += int((risk_scope > 0.350).sum().item())

        if not self._bayesian_q_veto_logged:
            finite_clearances = [c for c in clearances if np.isfinite(c)]
            cmin = min(finite_clearances) if finite_clearances else float('inf')
            cmean = float(np.mean(finite_clearances)) if finite_clearances else float('inf')
            q_min = float(action_residual.min().item()) if action_residual is not None and action_residual.numel() == n else float('nan')
            q_max = float(action_residual.max().item()) if action_residual is not None and action_residual.numel() == n else float('nan')
            extra_count = len(extra_humans_beyond5)
            extra_veto_count = int(sum(extra_kinematic_veto)) if extra_kinematic_veto is not None else 0
            logging.info(
                "[BAYES-SAFETY] phase=%s mode=%s topk=%d proposal_idx=%d best_idx=%d "
                "kept=%d vetoed=%d fallbacks=%d min_clear=%.3f risk_thr=%.5f q_min/max=%.3f/%.3f "
                "clearance_min/mean=%.3f/%.3f extra_humans=%d extra_veto=%d extra_penalty=%.2f@%.2f",
                self._phase, getattr(self, 'bayesian_policy_mode', 'safety_layer'), topk,
                protected_idx, best_idx, int(kept_mask.sum().item()),
                int(veto_mask.sum().item()), int(self._bayesian_safety_stats['fallbacks']),
                min_clearance, risk_threshold, q_min, q_max, cmin, cmean,
                extra_count, extra_veto_count,
                float(getattr(self, 'extra_human_penalty_weight', 4.0)),
                float(getattr(self, 'extra_human_penalty_margin', 0.25)),
            )
            self._bayesian_q_veto_logged = True
        return shaped

    def _shape_bayesian_scores(self, state, scores, action_residual=None, dmins_batch=None, risk=None, rerank_residual=None):
        """Route Bayesian action selection through a proposal policy plus safety layer."""
        eval_phase = self._phase in ('test', 'val', 'eval')
        train_phase = self._phase == 'train'
        train_proposal = train_phase and self.bayesian_q_veto_mode
        train_hard_veto = train_proposal and bool(getattr(self, 'bayesian_train_hard_veto', True))
        if not (eval_phase or train_proposal):
            return scores
        active = (
            self.bayesian_q_veto_mode or
            self.bayesian_rerank_topk > 0 or
            self.bayesian_progress_weight != 0.0 or
            self.bayesian_speed_weight != 0.0 or
            self.bayesian_clearance_penalty != 0.0 or
            self.bayesian_stop_penalty != 0.0
        )
        if not active or self.action_space is None or scores.numel() == 0:
            return scores

        proposal_scores, proposal_mode, _ = self._bayesian_proposal_scores(
            state, scores, action_residual, rerank_residual=rerank_residual
        )
        if self.bayesian_q_veto_mode and (eval_phase or train_hard_veto):
            shaped = self._bayesian_safety_layer_scores(
                state, proposal_scores,
                action_residual=action_residual,
                dmins_batch=dmins_batch,
                risk=risk,
            )
        else:
            shaped = proposal_scores

        if not self._bayesian_shape_logged:
            logging.info(
                "[BAYES-SHAPE] mode=%s proposal=%s topk=%d qveto=%s train_hard_veto=%s progress_w=%.3f speed_w=%.3f "
                "clearance_margin=%.3f clearance_penalty=%.3f stop_penalty=%.3f",
                getattr(self, 'bayesian_policy_mode', 'safety_layer'), proposal_mode,
                int(self.bayesian_rerank_topk), bool(self.bayesian_q_veto_mode),
                bool(getattr(self, 'bayesian_train_hard_veto', True)),
                self.bayesian_progress_weight, self.bayesian_speed_weight,
                self.bayesian_clearance_margin, self.bayesian_clearance_penalty,
                self.bayesian_stop_penalty,
            )
            self._bayesian_shape_logged = True
        return shaped

    def _shape_q_only_scores(self, state, scores, force=False):
        """Light rerank inside raw q-head top-k to reduce timeout without ignoring BC."""
        from crowd_sim.envs.utils.action import ActionXY

        if not force and self._phase not in ('test', 'val', 'eval'):
            return scores
        active = (
            self.q_only_rerank_topk > 0 or
            self.q_only_progress_weight != 0.0 or
            self.q_only_speed_weight != 0.0 or
            self.q_only_clearance_penalty != 0.0 or
            self.q_only_stop_penalty != 0.0
        )
        if not active or self.action_space is None or scores.numel() == 0:
            return scores

        shaped = scores.clone()
        n = int(scores.numel())
        curr_goal_dist = float(np.hypot(
            state.self_state.px - state.self_state.gx,
            state.self_state.py - state.self_state.gy
        ))
        topk = int(self.q_only_rerank_topk)
        if curr_goal_dist >= float(self.utility_far_goal_radius):
            topk = max(topk, int(self.utility_far_topk))
        candidate_mask = None
        if 0 < topk < n:
            candidate_mask = torch.zeros(n, dtype=torch.bool, device=scores.device)
            candidate_mask[torch.topk(scores, k=topk).indices] = True
        bonuses = []
        clearances = []
        for action in self.action_space:
            next_self = self.propagate(state.self_state, action)
            next_goal_dist = float(np.hypot(next_self.px - next_self.gx, next_self.py - next_self.gy))
            progress = curr_goal_dist - next_goal_dist
            speed = float(np.hypot(action.vx, action.vy))
            if state.human_states:
                next_humans = [
                    self.propagate(h, ActionXY(h.vx, h.vy))
                    for h in state.human_states
                ]
                clearance = min(
                    float(np.hypot(next_self.px - h.px, next_self.py - h.py) - next_self.radius - h.radius)
                    for h in next_humans
                )
            else:
                clearance = float('inf')
            clearances.append(clearance)
            clearance_penalty = self.q_only_clearance_penalty * max(
                float(self.q_only_clearance_margin) - clearance, 0.0
            )
            stop_penalty = self.q_only_stop_penalty if speed < 0.05 else 0.0
            far_goal_bonus = 0.0
            if (
                curr_goal_dist >= float(self.utility_far_goal_radius)
                and clearance >= float(self.utility_far_min_clearance)
            ):
                positive_progress = max(progress - float(self.utility_progress_floor), 0.0)
                far_goal_bonus = (
                    float(self.utility_far_progress_weight) * positive_progress
                    + float(self.utility_far_speed_weight) * speed
                )
            bonus = (
                self.q_only_progress_weight * progress +
                self.q_only_speed_weight * speed -
                clearance_penalty -
                stop_penalty +
                far_goal_bonus
            )
            bonuses.append(bonus)

        bonus_t = torch.tensor(bonuses, dtype=scores.dtype, device=scores.device)
        shaped = shaped + bonus_t
        if candidate_mask is not None:
            shaped = shaped.masked_fill(~candidate_mask, -1e9)

        if not self._q_only_shape_logged:
            finite_clearances = [c for c in clearances if np.isfinite(c)]
            cmin = min(finite_clearances) if finite_clearances else float('inf')
            cmean = float(np.mean(finite_clearances)) if finite_clearances else float('inf')
            logging.info(
                "[BC-Q-SHAPE] topk=%d progress_w=%.3f speed_w=%.3f clearance_margin=%.3f "
                "clearance_penalty=%.3f stop_penalty=%.3f far_r=%.3f far_topk=%d far_pw=%.3f far_sw=%.3f "
                "clearance_min/mean=%.3f/%.3f",
                topk, self.q_only_progress_weight, self.q_only_speed_weight,
                self.q_only_clearance_margin, self.q_only_clearance_penalty,
                self.q_only_stop_penalty, self.utility_far_goal_radius,
                int(self.utility_far_topk),
                self.utility_far_progress_weight, self.utility_far_speed_weight,
                cmin, cmean
            )
            self._q_only_shape_logged = True
        return shaped

    def select_action(self, state):
        """Sole action-selection entry point.

        Handles early termination, epsilon-greedy exploration, calls score_actions(),
        applies test-time safety constraints, and returns (action, action_idx).
        """
        from crowd_sim.envs.utils.action import ActionXY

        if self.reach_destination(state):
            return ActionXY(0, 0), 0

        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        if self._phase == 'train' and np.random.random() < self.epsilon:
            rand_idx = int(np.random.choice(len(self.action_space)))
            return self.action_space[rand_idx], rand_idx

        use_bayes = bool(self.bayesian_core_enable and getattr(self, 'use_bayesian_planner', True))

        if use_bayes:
            result = self.score_actions(state)
            scores = result['scores']
            current_token = result['current_token']
            dmins_batch = result['dmins']
            scores = self._shape_bayesian_scores(
                state, scores,
                action_residual=result.get('action_residual'),
                dmins_batch=dmins_batch,
                risk=result.get('risk'),
                rerank_residual=result.get('rerank_residual'),
            )
            if bool(getattr(self, 'bayesian_value_lookahead_enable', False)):
                value_next = result.get('value_next')
                if isinstance(value_next, torch.Tensor) and value_next.numel() == scores.numel():
                    allowed_mask = torch.isfinite(scores) & (scores > -1e8)
                    lookahead_scores = value_next.to(scores.device).reshape_as(scores).clone()
                    # GDBN risk penalty
                    risk_t = result.get('risk')
                    risk_w = float(getattr(self, 'bayesian_value_lookahead_risk_weight', 0.0))
                    if risk_w != 0.0 and isinstance(risk_t, torch.Tensor) and risk_t.numel() == scores.numel():
                        lookahead_scores = lookahead_scores - risk_w * risk_t.to(scores.device).reshape_as(scores)
                    # Extra-human kinematic soft penalty (carried from safety layer cache)
                    _pen_margin = float(getattr(self, 'extra_human_penalty_margin', 0.25))
                    _pen_weight = float(getattr(self, 'extra_human_penalty_weight', 4.0))
                    _extra_clrs = getattr(self, '_last_extra_min_clearances', None)
                    if _extra_clrs is not None and len(_extra_clrs) == scores.numel():
                        if _pen_margin > 0.0 and _pen_weight > 0.0:
                            _ec_t = torch.tensor(_extra_clrs, dtype=lookahead_scores.dtype,
                                                 device=lookahead_scores.device)
                            _ep = _pen_weight * torch.clamp(_pen_margin - _ec_t, min=0.0)
                            lookahead_scores = lookahead_scores - _ep.masked_fill(~allowed_mask, 0.0)
                    scores = lookahead_scores.masked_fill(~allowed_mask, -1e9)
                    if not self._value_lookahead_logged:
                        _extra_count = int(getattr(self, '_last_extra_human_count', 0) or 0)
                        logging.info(
                            "[VALUE-LOOKAHEAD] active: candidates=%d risk_w=%.3f clr_pen=%.2f@%.2f extra_humans=%d pen_margin=%.2f",
                            int(allowed_mask.sum().item()), risk_w, _pen_weight, _pen_margin,
                            _extra_count,
                            _pen_margin,
                        )
                        self._value_lookahead_logged = True

            if self._phase in ('test', 'val', 'eval') and (self.test_min_clearance > 0.0 or self.test_risk_lambda > 0.0):
                dmins_tensor = torch.tensor(dmins_batch, device=self.device)
                if self.test_min_clearance > 0.0:
                    safe_mask = dmins_tensor >= self.test_min_clearance
                    if safe_mask.any():
                        scores = scores.masked_fill(~safe_mask, -1e9)
                if self.test_risk_lambda > 0.0:
                    risk_margin = self.test_min_clearance if self.test_min_clearance > 0.0 else self.discomfort_dist
                    scores = scores - self.test_risk_lambda * torch.clamp(risk_margin - dmins_tensor, min=0.0)
        else:
            from crowd_nav.contracts_gdbn import _batch_joint34_to_tokens_vectorized

            current_34 = self._build_joint_state_34(state.self_state, state.human_states)
            current_token_13 = _batch_joint34_to_tokens_vectorized(current_34.reshape(1, -1))[0]
            # BC was trained with zero belief tokens, so q-only evaluation must use
            # the same representation instead of injecting online GDBN beliefs.
            current_token = self._make_token_frame(current_token_13, belief_vec=None)
            hist = (list(self._history) + [current_token])[-self.seq_len:]
            if len(hist) < self.seq_len:
                hist = [hist[0]] * (self.seq_len - len(hist)) + hist
            states_tensor = torch.from_numpy(np.asarray(hist, dtype=np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores = self.forward_q(states_tensor, return_sequence=False).squeeze(0)

            if not self._q_only_logged:
                logging.info("[BC-Q] q-only action scorer active: bayesian_planner=False zero_belief=True")
                self._q_only_logged = True
            scores = self._shape_q_only_scores(state, scores)

        # Slight bias against stop (index 0) to prefer movement when scores are tied
        if scores.numel() > 0:
            scores[0] -= 1e-3

        best_idx = int(scores.argmax().item())
        best_action = self.action_space[best_idx]
        self._capture_candidate_diagnostics(
            state,
            scores,
            best_idx,
            use_bayes,
            result=result if use_bayes else None,
            dmins_batch=dmins_batch if use_bayes else None,
        )

        if use_bayes:
            try:
                risk_t = result.get('risk')
                ep_stats = getattr(self, '_bayesian_risk_episode_stats', None)
                if isinstance(risk_t, torch.Tensor) and ep_stats is not None and risk_t.numel() == scores.numel():
                    finite_mask = torch.isfinite(scores) & (scores > -1e8)
                    candidate_risk = risk_t[finite_mask] if finite_mask.any() else risk_t
                    ep_stats['selected'].append(float(risk_t[best_idx].detach().item()))
                    ep_stats['candidate_min'].append(float(candidate_risk.min().detach().item()))
                    ep_stats['candidate_mean'].append(float(candidate_risk.mean().detach().item()))
                    ep_stats['candidate_max'].append(float(candidate_risk.max().detach().item()))
                    if dmins_batch and best_idx < len(dmins_batch):
                        ep_stats['selected_clearance'].append(float(dmins_batch[best_idx]))
                    finite_clearances = [float(c) for c in (dmins_batch or []) if np.isfinite(c)]
                    if finite_clearances:
                        ep_stats['candidate_clearance_min'].append(min(finite_clearances))
            except Exception:
                pass

        if use_bayes and not self._bayes_core_logged:
            logging.info(
                "[BAYES-CORE] scorer active: "
                f"risk={float(result['risk'].min()):.4f}/{float(result['risk'].mean()):.4f}/{float(result['risk'].max()):.4f} "
                f"efe_mean={float(result['efe'].mean()):.4f} "
                f"entropy_mean={float(result['entropy'].mean()):.4f} "
                f"klda_mean={float(result['klda'].mean()):.4f} "
                f"residual_w={self.bayesian_action_residual_weight:.3f} "
                f"bootstrap_w={self.bayesian_value_bootstrap_weight:.3f}"
            )
            self._bayes_core_logged = True

        self._history.append(current_token)

        if use_bayes and self._phase in ('test', 'val', 'eval') and self.test_action_smoothing > 0.0:
            if self._last_action is not None:
                alpha = float(self.test_action_smoothing)
                best_action = ActionXY(
                    alpha * self._last_action.vx + (1.0 - alpha) * best_action.vx,
                    alpha * self._last_action.vy + (1.0 - alpha) * best_action.vy,
                )
            self._last_action = best_action

        return best_action, best_idx

    def to(self, *args, **kwargs):
        """ to() """
        super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])
        elif 'device' in kwargs:
            self.device = torch.device(kwargs['device'])
        return self

MambaRL = MambaRLPolicy
