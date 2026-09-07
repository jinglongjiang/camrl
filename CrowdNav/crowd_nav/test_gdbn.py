#!/usr/bin/env python3
"""
Mamba Policy Testing - Run 7 (Optimized)
Strategy:
1. No Patch (Dirty Data) - Proven to give best success (44%).
2. Config: T=10, dt=0.1 (Match default.yaml exactly).
3. v_max=1.3 (Slightly conservative vs 1.5 to reduce collision).
4. Pre-Sort Humans: Fixes blindness in >5 human scenarios (Critical for square_crossing).
"""
import logging
import argparse
import configparser
import os
import re
import json
import torch
import numpy as np
import random
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from crowd_nav.policy.policy_factory import policy_factory as _base_policy_factory
from crowd_nav.policy.mamba_rl_gdbn import MambaRLPolicy as GDBNMambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState, ObservableState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.contracts_gdbn import GRID, discrete_index_to_action, init_grid_from_cfg, joint34_to_tokens

policy_factory = dict(_base_policy_factory)
policy_factory.update({
    'mamba': GDBNMambaRLPolicy,
    'mamba_rl': GDBNMambaRLPolicy,
    'mamba_vl': GDBNMambaRLPolicy,
})

try:
    from crowd_nav.gdbn import GDBNIntegration
    _GDBN_AVAILABLE = True
except Exception as exc:
    GDBNIntegration = None
    _GDBN_AVAILABLE = False
    print(f"[WARNING] GDBN unavailable in test.py: {exc}")

try:
    from crowd_nav.risk_models import ConstantVelocityRiskModel
except Exception as exc:
    ConstantVelocityRiskModel = None
    print(f"[WARNING] CV risk model unavailable in test_gdbn.py: {exc}")

# ========== SARL Value Network for Discrete Action Search ==========
import torch.nn as nn

def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _raw_policy(policy):
    return getattr(policy, '_orig_mod', policy)

def _set_policy_attr(policy, name, value):
    raw = _raw_policy(policy)
    setattr(raw, name, value)
    if raw is not policy:
        setattr(policy, name, value)

def _get_policy_attr(policy, name, default=None):
    return getattr(_raw_policy(policy), name, getattr(policy, name, default))

def _parse_csv_list(value, cast=str):
    if value is None:
        return []
    return [cast(item.strip()) for item in str(value).split(',') if item.strip()]

def _make_policy_human_observations(robot, sorted_humans, test_args, rng):
    """Build policy-only human observations, optionally with test-time sensor noise.

    The environment state is not mutated; collision/reward/metrics still use the
    true simulator state. This keeps noise tests as observation-noise ablations.
    """
    states = [h.get_observable_state() for h in sorted_humans]
    if test_args is None:
        return states

    dropout = float(getattr(test_args, 'obs_dropout_prob', 0.0) or 0.0)
    noise_std = float(getattr(test_args, 'obs_noise_std', 0.0) or 0.0)
    if dropout <= 0.0 and noise_std <= 0.0:
        return states

    if dropout > 0.0 and states:
        keep = rng.random(len(states)) >= dropout
        states = [s for s, flag in zip(states, keep) if bool(flag)]

    if noise_std > 0.0 and states:
        noisy_states = []
        for s in states:
            dx, dy = rng.normal(0.0, noise_std, size=2)
            noisy_states.append(
                ObservableState(
                    float(s.px + dx),
                    float(s.py + dy),
                    float(s.vx),
                    float(s.vy),
                    float(s.radius),
                )
            )
        states = noisy_states

    return states

def _resolve_weights_path(model_dir, weights):
    aliases = {
        'k4_safety': 'k4_safety_policy.pth',
        'k4_safety_policy': 'k4_safety_policy.pth',
        'final': 'rl_model_final.pth',
        'rl_final': 'rl_model_final.pth',
        'rl_model_final': 'rl_model_final.pth',
        'ep10000': 'rl_model_ep10000.pth',
        'rl_model_ep10000': 'rl_model_ep10000.pth',
        'il': 'il_policy.pth',
        'il_policy': 'il_policy.pth',
    }
    if weights == 'latest':
        files = [f for f in os.listdir(model_dir) if re.match(r'rl_model_ep\d+\.pth', f)]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
            return os.path.join(model_dir, files[-1])
        return os.path.join(model_dir, 'rl_model_final.pth')
    if weights in aliases:
        return os.path.join(model_dir, aliases[weights])
    if os.path.isabs(weights):
        return weights
    candidate = os.path.join(model_dir, weights)
    if not os.path.exists(candidate) and not weights.endswith('.pth'):
        candidate_pth = candidate + '.pth'
        if os.path.exists(candidate_pth):
            return candidate_pth
    return candidate

def _install_gdbn(policy, cfg, model_dir, K, params_dir, tag):
    if not _GDBN_AVAILABLE:
        print(f"[WARNING] {tag}: GDBN unavailable, cannot install Bayesian safety module")
        return False
    if not params_dir:
        return False
    params_dir = os.path.expanduser(str(params_dir))
    if not os.path.isabs(params_dir) and not os.path.isdir(params_dir):
        params_dir = os.path.join(model_dir, params_dir)
    if not os.path.isdir(params_dir):
        print(f"[WARNING] {tag}: GDBN params not found: {params_dir}")
        return False
    try:
        max_peds = cfg.getint('belief', 'num_humans', fallback=cfg.getint('sim', 'human_num', fallback=5))
        klda_clip = cfg.getfloat('belief', 'klda_clip', fallback=5.0)
        n_particles = cfg.getint('gdbn', 'n_particles', fallback=50)
        gdbn = GDBNIntegration(
            K=int(K),
            n_particles=n_particles,
            params_dir=params_dir,
            max_peds=max_peds,
            klda_norm_clip=klda_clip,
        )
        _set_policy_attr(policy, 'gdbn_module', gdbn)
        print(f"[INFO] {tag}: installed GDBN K={gdbn.K} params={params_dir}")
        return True
    except Exception as exc:
        print(f"[WARNING] {tag}: failed to install GDBN: {exc}")
        return False

def _apply_checkpoint_runtime(policy, checkpoint, cfg, model_dir, weights_name, args):
    meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
    if isinstance(meta, dict):
        if meta.get('q_prior_belief_mode'):
            mode = str(meta.get('q_prior_belief_mode')).strip().lower()
            _set_policy_attr(policy, 'bayesian_q_prior_belief_mode', mode)
            print(f"[INFO] checkpoint meta: q_prior_belief_mode={mode}")
        if 'use_topk_reranker' in meta:
            use_reranker = bool(meta.get('use_topk_reranker'))
            reranker_topk = int(meta.get('reranker_topk', 0) or 0)
            reranker_scale = float(meta.get('reranker_residual_scale', 1.0) or 1.0)
            _set_policy_attr(policy, 'bayesian_rl_reranker_enable', use_reranker)
            _set_policy_attr(policy, 'bayesian_rl_reranker_topk', reranker_topk)
            _set_policy_attr(policy, 'bayesian_rl_reranker_scale', reranker_scale)
            print(f"[INFO] checkpoint meta: topk_reranker={use_reranker} topk={reranker_topk} scale={reranker_scale:.3f}")
        if 'use_value_lookahead' in meta:
            use_lookahead = bool(meta.get('use_value_lookahead'))
            meta_risk_w = float(meta.get('value_lookahead_risk_weight', 0.0) or 0.0)
            # Config overrides meta for risk_w so we can tune without retraining.
            cfg_risk_w = float(cfg.getfloat(
                'eval_protocol', 'bayesian_value_lookahead_risk_weight', fallback=0.0))
            risk_w = cfg_risk_w if cfg_risk_w != 0.0 else meta_risk_w
            _set_policy_attr(policy, 'bayesian_value_lookahead_enable', use_lookahead)
            _set_policy_attr(policy, 'bayesian_value_lookahead_risk_weight', risk_w)
            print(f"[INFO] checkpoint meta: value_lookahead={use_lookahead} risk_w={risk_w:.3f} (meta={meta_risk_w:.3f} cfg={cfg_risk_w:.3f})")
        if (
            args.risk_model == 'gdbn'
            and not args.gdbn_params
            and not args.disable_gdbn_compute
            and meta.get('gdbn_params_dir')
            and meta.get('gdbn_K')
        ):
            if _install_gdbn(policy, cfg, model_dir, int(meta.get('gdbn_K')),
                             meta.get('gdbn_params_dir'), 'checkpoint meta'):
                return

    wants_k4 = str(weights_name).lower() in ('k4_safety', 'k4_safety_policy') or 'k4' in os.path.basename(str(weights_name)).lower()
    if args.risk_model == 'gdbn' and not args.gdbn_params and not args.disable_gdbn_compute and wants_k4:
        _install_gdbn(policy, cfg, model_dir, 4, 'gdbn_params_k4', 'k4 fallback')

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class SARLValueNetwork(nn.Module):
    """SARL's Value Network - exact copy for compatibility"""
    def __init__(self, input_dim=13, self_state_dim=6, mlp1_dims=[150, 100], mlp2_dims=[100, 50],
                 mlp3_dims=[150, 100, 100, 1], attention_dims=[100, 100, 1], with_global_state=True):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        mlp3_input_dim = mlp2_dims[-1] + self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)

    def forward(self, state):
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        if self.with_global_state:
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value

def rotate_state(self_state, human_state):
    """Rotate state to robot-centric coordinates (SARL standard)

    Input: self_state (FullState), human_state (ObservableState)
    Output: 13-dim rotated state matching SARL's rotate() output

    SARL rotate() expects 14-dim input:
    [px, py, vx, vy, radius, gx, gy, v_pref, theta, px1, py1, vx1, vy1, radius1]
      0   1   2   3    4      5   6     7      8     9   10   11   12     13

    SARL rotate() outputs 13-dim:
    [dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum]
    """
    # Goal direction
    dx = self_state.gx - self_state.px
    dy = self_state.gy - self_state.py
    dg = np.sqrt(dx**2 + dy**2)
    rot = np.arctan2(dy, dx)

    # Rotated self velocity
    vx = self_state.vx * np.cos(rot) + self_state.vy * np.sin(rot)
    vy = self_state.vy * np.cos(rot) - self_state.vx * np.sin(rot)

    # Rotated human position (relative to robot)
    px1 = (human_state.px - self_state.px) * np.cos(rot) + (human_state.py - self_state.py) * np.sin(rot)
    py1 = (human_state.py - self_state.py) * np.cos(rot) - (human_state.px - self_state.px) * np.sin(rot)

    # Rotated human velocity
    vx1 = human_state.vx * np.cos(rot) + human_state.vy * np.sin(rot)
    vy1 = human_state.vy * np.cos(rot) - human_state.vx * np.sin(rot)

    # Distance to human
    da = np.sqrt((self_state.px - human_state.px)**2 + (self_state.py - human_state.py)**2)

    # Radius sum
    radius_sum = self_state.radius + human_state.radius

    # theta is 0 for holonomic
    theta = 0.0

    return [dg, self_state.v_pref, theta, self_state.radius, vx, vy,
            px1, py1, vx1, vy1, human_state.radius, da, radius_sum]

def compute_scene_complexity(state):
    robot = state.self_state
    human_states = state.human_states

    n_humans = len(human_states)

    if n_humans == 0:
        return 0, float('inf')

    min_ttc = float('inf')
    for h in human_states:
        rel_x = h.px - robot.px
        rel_y = h.py - robot.py
        rel_vx = h.vx - robot.vx
        rel_vy = h.vy - robot.vy
        dist = np.sqrt(rel_x**2 + rel_y**2 + 1e-6)
        closing = -(rel_x * rel_vx + rel_y * rel_vy) / (dist + 1e-6)
        ttc = dist / (closing + 1e-6) if closing > 0.1 else dist * 10
        min_ttc = min(min_ttc, ttc)

    return n_humans, min_ttc

def sarl_discrete_search(state, sarl_value_net, device, time_step=0.25, gamma=0.9, v_pref=1.0):
    """Use SARL Value Network to evaluate discrete actions and return best action"""
    from crowd_sim.envs.utils.state import ObservableState, FullState

    # Build action space (SARL style: 5 speeds x 16 rotations = 80 actions)
    speeds = [(np.exp((i + 1) / 5) - 1) / (np.e - 1) * v_pref for i in range(5)]
    rotations = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    action_space = [ActionXY(0, 0)]  # Include stop action
    for rotation in rotations:
        for speed in speeds:
            action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))

    self_state = state.self_state
    human_states = state.human_states

    max_value = float('-inf')
    best_action = action_space[0]

    for action in action_space:
        # 1. Propagate robot state
        next_px = self_state.px + action.vx * time_step
        next_py = self_state.py + action.vy * time_step
        next_self = FullState(next_px, next_py, action.vx, action.vy, self_state.radius,
                              self_state.gx, self_state.gy, self_state.v_pref, self_state.theta)

        # 2. Propagate human states (assume constant velocity)
        next_humans = []
        for h in human_states:
            nh_px = h.px + h.vx * time_step
            nh_py = h.py + h.vy * time_step
            next_humans.append(ObservableState(nh_px, nh_py, h.vx, h.vy, h.radius))

        # 3. Compute immediate reward
        dmin = float('inf')
        collision = False
        for h in next_humans:
            dist = np.sqrt((next_self.px - h.px)**2 + (next_self.py - h.py)**2) - next_self.radius - h.radius
            if dist < 0:
                collision = True
                break
            dmin = min(dmin, dist)

        reaching_goal = np.sqrt((next_self.px - next_self.gx)**2 + (next_self.py - next_self.gy)**2) < next_self.radius

        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1.0
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * time_step
        else:
            reward = 0.0

        # 4. Build rotated input for SARL network
        if len(next_humans) > 0:
            rotated_states = []
            for h in next_humans:
                rotated = rotate_state(next_self, h)
                rotated_states.append(rotated)

            # Pad to at least 1 human
            batch_input = torch.tensor(rotated_states, dtype=torch.float32).unsqueeze(0).to(device)

            # 5. Get value from SARL network
            with torch.no_grad():
                next_value = sarl_value_net(batch_input).item()
        else:
            next_value = 0.0

        # 6. Compute total value
        value = reward + (gamma ** (time_step * v_pref)) * next_value

        if value > max_value:
            max_value = value
            best_action = action

    return best_action

# NO Monkey Patch

def test_episode(env, robot, policy, test_case_idx, robot_start=None, robot_goal=None, sarl_value_net=None, device=None, case_desc=None):
    if hasattr(policy, 'reset_episode_stats'):
        policy.reset_episode_stats()
    # reset progress history for test-time heuristics
    if hasattr(policy, "_dist_hist"):
        policy._dist_hist = []
    else:
        setattr(policy, "_dist_hist", [])

    mamba_count = 0
    sarl_count = 0

    reset_result = env.reset(options={'test_case': int(test_case_idx)})
    if isinstance(reset_result, tuple):
        ob = reset_result[0]
    else:
        ob = reset_result

    if robot_start is not None:
        robot.px, robot.py = robot_start
        robot.vx, robot.vy = 0.0, 0.0
    if robot_goal is not None:
        robot.gx, robot.gy = robot_goal

    done = False
    steps = 0
    robot_positions = [robot.get_position()]
    min_dists = []
    candidate_records = []
    obs_rng = np.random.default_rng(int(test_case_idx) + 7919)

    max_steps = 500

    while not done and steps < max_steps:
        # 1. Get full state
        robot_state = robot.get_full_state()

        humans = env.humans
        ttc_list = []
        for h in humans:
            rel_x = h.px - robot.px
            rel_y = h.py - robot.py
            rel_vx = h.vx - robot.vx
            rel_vy = h.vy - robot.vy
            dist = np.sqrt(rel_x**2 + rel_y**2 + 1e-6)
            closing = -(rel_x * rel_vx + rel_y * rel_vy) / (dist + 1e-6)
            ttc = dist / (closing + 1e-6) if closing > 0.1 else dist * 10
            ttc_list.append((ttc, h))
        min_ttc = min([x[0] for x in ttc_list]) if ttc_list else float('inf')
        sorted_humans = [h for _, h in sorted(ttc_list, key=lambda x: x[0])]

        test_args = getattr(policy, '_test_args', None)
        # 3. Construct policy input JointState. Optional observation noise is
        # policy-only: env.step/collision/metrics still use true env.humans.
        human_states = _make_policy_human_observations(robot, sorted_humans, test_args, obs_rng)
        state = JointState(robot_state, human_states)
        used_mamba = False

        if sarl_value_net is not None and policy is not None:
            n_humans, min_ttc = compute_scene_complexity(state)

            HUMAN_THRESHOLD = 7
            TTC_THRESHOLD = 2.5

            if n_humans > HUMAN_THRESHOLD or min_ttc < TTC_THRESHOLD:
                action = sarl_discrete_search(state, sarl_value_net, device)
                sarl_count += 1
            else:
                action = policy.predict(state)
                mamba_count += 1
                used_mamba = True
            # ==================================
        elif sarl_value_net is not None:
            action = sarl_discrete_search(state, sarl_value_net, device)
            sarl_count += 1
        elif test_args and test_args.gating and hasattr(policy, 'predict_with_gating'):
            action = policy.predict_with_gating(
                state,
                human_threshold=test_args.human_threshold,
                ttc_threshold=test_args.ttc_threshold
            )
            used_mamba = True
        elif test_args and test_args.discrete_search and hasattr(policy, 'act_discrete'):
            action = policy.act_discrete(state)
            used_mamba = True
        else:
            action = policy.predict(state)
            used_mamba = True
        # --------------------------------------------------------------

        if used_mamba and test_args and test_args.candidate_diag and policy is not None:
            raw_policy = _raw_policy(policy)
            diag = getattr(raw_policy, '_last_candidate_diagnostics', None)
            if isinstance(diag, dict):
                rec = dict(diag)
                rec.update({
                    'step': int(steps),
                    'case': case_desc,
                    'robot_px': float(robot.px),
                    'robot_py': float(robot.py),
                    'goal_x': float(robot.gx),
                    'goal_y': float(robot.gy),
                    'goal_dist': float(np.linalg.norm([robot.gx - robot.px, robot.gy - robot.py])),
                    'min_ttc': float(min_ttc),
                })
                candidate_records.append(rec)

        # Optional: goal-bias blend for hard scenarios (test-time heuristic)
        if used_mamba and test_args and test_args.mamba_bias and hasattr(action, 'vx'):
            alpha = test_args.bias_alpha
            stall_alpha = test_args.bias_stall_alpha
            min_clear_req = test_args.bias_min_clear
            min_ttc_req = test_args.bias_min_ttc
            speed_floor = test_args.bias_speed_floor

            if case_desc in test_args.bias_scenarios:
                # compute min clearance
                min_clear = float('inf')
                for h in humans:
                    d = np.linalg.norm(np.array([h.px - robot.px, h.py - robot.py])) - robot.radius - h.radius
                    if d < min_clear:
                        min_clear = d

                # track progress (for stall detection)
                dist_goal = np.linalg.norm(np.array([robot.gx - robot.px, robot.gy - robot.py]))
                policy._dist_hist.append(dist_goal)
                if len(policy._dist_hist) > test_args.bias_patience:
                    policy._dist_hist = policy._dist_hist[-test_args.bias_patience:]
                progress = policy._dist_hist[0] - policy._dist_hist[-1] if len(policy._dist_hist) >= 2 else 0.0
                late_stage = steps >= int(max_steps * test_args.bias_late_ratio)
                stalled = progress < test_args.bias_progress_eps and late_stage

                # late-stage relaxation (optional): allow a bit more risk to avoid timeout
                late_frac = 0.0
                if late_stage:
                    late_frac = (steps / max_steps - test_args.bias_late_ratio) / max(1e-6, (1.0 - test_args.bias_late_ratio))
                    late_frac = float(np.clip(late_frac, 0.0, 1.0))
                min_clear_eff = min_clear_req * (1.0 - test_args.bias_late_relax * late_frac)
                min_ttc_eff = min_ttc_req * (1.0 - test_args.bias_late_relax * late_frac)
                speed_floor_eff = speed_floor * (1.0 + test_args.bias_late_speed_boost * late_frac)
                speed_floor_eff = float(min(1.0, speed_floor_eff))

                if min_clear > min_clear_eff and min_ttc > min_ttc_eff:
                    dx = robot.gx - robot.px
                    dy = robot.gy - robot.py
                    norm = np.linalg.norm([dx, dy]) + 1e-6
                    goal_vx = robot.v_pref * dx / norm
                    goal_vy = robot.v_pref * dy / norm
                    alpha_use = stall_alpha if stalled else alpha

                    # deadline boost (late-stage ramp, only if very safe)
                    if test_args.bias_deadline_alpha > 0:
                        if late_stage and min_clear > test_args.bias_deadline_min_clear and min_ttc > test_args.bias_deadline_min_ttc:
                            alpha_use = max(alpha_use, test_args.bias_deadline_alpha * late_frac)

                    vx = (1 - alpha_use) * action.vx + alpha_use * goal_vx
                    vy = (1 - alpha_use) * action.vy + alpha_use * goal_vy

                    # speed floor to reduce dithering
                    speed = np.linalg.norm([vx, vy]) + 1e-6
                    if speed < speed_floor_eff * robot.v_pref:
                        vx = vx / speed * robot.v_pref * speed_floor_eff
                        vy = vy / speed * robot.v_pref * speed_floor_eff
                    elif speed > robot.v_pref:
                        vx = vx / speed * robot.v_pref
                        vy = vy / speed * robot.v_pref
                    # optional: avoid stop action if safe (late-stage or stalled)
                    if test_args.bias_no_stop:
                        stop_eps = test_args.bias_stop_eps * robot.v_pref
                        if np.linalg.norm([vx, vy]) < stop_eps:
                            vx = goal_vx
                            vy = goal_vy
                    action = ActionXY(vx, vy)

        # Optional: Mamba timeout rescue (safe, late-stage only)
        if test_args and test_args.mamba_rescue:
            # progress history
            if not hasattr(policy, "_dist_hist"):
                policy._dist_hist = []
            dist_goal = np.linalg.norm(np.array([robot.gx - robot.px, robot.gy - robot.py]))
            policy._dist_hist.append(dist_goal)
            if len(policy._dist_hist) > test_args.rescue_patience:
                policy._dist_hist = policy._dist_hist[-test_args.rescue_patience:]
            # check progress
            progress = policy._dist_hist[0] - policy._dist_hist[-1] if len(policy._dist_hist) >= 2 else 0.0
            late_stage = steps >= int(max_steps * test_args.rescue_late_ratio)
            if late_stage and progress < test_args.rescue_progress_eps and min_ttc > test_args.rescue_min_ttc:
                # try goal-directed action if predicted clearance is safe
                dx = robot.gx - robot.px
                dy = robot.gy - robot.py
                norm = np.linalg.norm([dx, dy]) + 1e-6
                speed = robot.v_pref * test_args.rescue_speed_scale
                cand = ActionXY(speed * dx / norm, speed * dy / norm)
                # predict clearance after 1 step (humans constant velocity)
                min_pred_clear = float('inf')
                next_rx = robot.px + cand.vx * env.time_step
                next_ry = robot.py + cand.vy * env.time_step
                for h in humans:
                    nx = h.px + h.vx * env.time_step
                    ny = h.py + h.vy * env.time_step
                    d = np.linalg.norm(np.array([nx - next_rx, ny - next_ry])) - robot.radius - h.radius
                    if d < min_pred_clear:
                        min_pred_clear = d
                if min_pred_clear > test_args.rescue_min_clear:
                    action = cand

        step_result = env.step(action)
        if len(step_result) == 5:
            ob, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            ob, reward, done, info = step_result
        robot_positions.append(robot.get_position())

        min_dist = float('inf')
        for human in env.humans:
            dist = np.linalg.norm(np.array([robot.px, robot.py]) - np.array([human.px, human.py]))
            dist -= (robot.radius + human.radius)
            min_dist = min(min_dist, dist)
        min_dists.append(min_dist)
        steps += 1

    outcome = 'timeout'
    if isinstance(info, dict):
        event = info.get('event', '').lower()
        if any(x in event for x in ['reach_goal', 'success', 'reachgoal']):
            outcome = 'success'
        elif any(x in event for x in ['collision']):
            outcome = 'collision'
    else:
        event_name = info.__class__.__name__
        if event_name == 'ReachGoal':
            outcome = 'success'
        elif event_name == 'Collision':
            outcome = 'collision'

    if test_args and test_args.candidate_diag and policy is not None:
        should_write = bool(test_args.candidate_diag_all or outcome != 'success')
        diag_path = getattr(_raw_policy(policy), '_candidate_diag_path', None)
        if should_write and diag_path and candidate_records:
            payload = {
                'episode_seed': int(test_case_idx),
                'case': case_desc,
                'outcome': outcome,
                'steps': int(steps),
                'records': candidate_records,
            }
            with open(diag_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=True) + '\n')

    return outcome, steps, robot_positions, min_dists, mamba_count, sarl_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='mamba', help='Policy name')
    parser.add_argument('--model_dir', type=str, default='runs/mamba_vl', help='Model directory')
    parser.add_argument('--weights', type=str, default='k4_safety', help='Checkpoint alias/path: k4_safety, rl_model_final, ep10000, il_policy, latest, or filename')
    parser.add_argument('--env_config', type=str, default='configs/env_gdbn.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy_gdbn.config')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--episodes', type=int, default=500, help='Episodes per scenario')
    parser.add_argument('--test_case', type=int, default=None, help='Run only specified test case')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=None, help='Override seq_len/T (e.g. 12 for GRU, 24 for Mamba T24)')
    parser.add_argument('--no_progress', action='store_true', help='Disable tqdm progress bars')
    parser.add_argument('--progress_interval', type=int, default=10, help='Print compact progress every N episodes when --no_progress is set')
    parser.add_argument('--human_nums', type=str, default='5', help='Comma-separated human counts, e.g. 5 or 5,7,10')
    parser.add_argument('--test_sims', type=str, default='circle_crossing', help='Comma-separated sims: circle_crossing,square_crossing')
    parser.add_argument('--circle_radius', type=float, default=4.0)
    parser.add_argument('--square_width', type=float, default=10.0)
    parser.add_argument('--time_limit', type=float, default=35.0, help='Evaluation time limit in seconds')
    parser.add_argument('--eval_mode', choices=['bayesian', 'q_only'], default='bayesian')
    parser.add_argument(
        '--ablation_profile',
        choices=['none', 'no_vl', 'no_gdbn_module', 'vl_only', 'nearest', 'veto025', 'veto045', 'qonly'],
        default='none',
        help='Named eval-only ablation bundle; explicit low-level flags remain available'
    )
    parser.add_argument('--q_prior_belief_mode', choices=['checkpoint', 'zero', 'real'], default='checkpoint')
    parser.add_argument('--gdbn_params', type=str, default=None, help='Override GDBN params dir, e.g. runs/mamba_vl/gdbn_params_k4')
    parser.add_argument('--gdbn_K', type=int, default=4)
    parser.add_argument('--risk_model', choices=['gdbn', 'cv'], default='gdbn',
                        help='Risk predictor used by the fixed governance interface')
    parser.add_argument('--disable_belief_tokens', action='store_true',
                        help='Zero belief-token features while retaining the selected risk model')
    parser.add_argument('--disable_value_lookahead', action='store_true',
                        help='Ablation: disable V(s_prime)-based candidate rescoring after checkpoint load')
    parser.add_argument('--value_lookahead_risk_weight', type=float, default=None,
                        help='Ablation: override value-lookahead risk weight after checkpoint load')
    parser.add_argument('--disable_gdbn_veto', action='store_true',
                        help='Ablation: disable Bayesian/GDBN clearance and risk veto thresholds')
    parser.add_argument('--disable_gdbn_compute', action='store_true',
                        help='Ablation: detach GDBN and skip Bayesian world-model update/rollout compute')
    parser.add_argument('--disable_extra_veto', action='store_true',
                        help='Ablation: disable extra-human kinematic veto and penalty')
    parser.add_argument('--human_selection_mode', choices=['nearest', 'threat', 'hybrid'], default=None,
                        help='Ablation: override [human_selection] mode before policy init')
    parser.add_argument('--k_nearest', type=int, default=None,
                        help='Ablation: override [human_selection] k_nearest before policy init')
    parser.add_argument('--k_threat', type=int, default=None,
                        help='Ablation: override [human_selection] k_threat before policy init')
    parser.add_argument('--risk_veto_threshold', type=float, default=None,
                        help='Ablation: override Bayesian/GDBN risk veto threshold after checkpoint load')
    parser.add_argument('--override_min_clearance', type=float, default=None,
                        help='Ablation: override Bayesian hard min-clearance threshold after checkpoint load')
    parser.add_argument('--legacy_cases', action='store_true', help='Use the old mixed 5/10/20-human test-case list')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--discrete_search', action='store_true', help='Enable discrete action search (SARL-style)')
    parser.add_argument('--gating', action='store_true', help='Enable density gating (auto switch between continuous/discrete)')
    parser.add_argument('--human_threshold', type=int, default=8, help='Human count threshold for gating')
    parser.add_argument('--ttc_threshold', type=float, default=2.0, help='TTC threshold for gating')
    parser.add_argument('--sarl_value', type=str, default=None, help='Path to SARL rl_model.pth for value-based discrete search')
    # Test-time bias (goal-directed blend for hard scenarios)
    parser.add_argument('--mamba_bias', action='store_true', help='Enable goal-bias blend for hard scenarios')
    parser.add_argument('--bias_alpha', type=float, default=0.25, help='Blend weight toward goal')
    parser.add_argument('--bias_min_clear', type=float, default=0.20, help='Min clearance to apply bias')
    parser.add_argument('--bias_min_ttc', type=float, default=1.5, help='Min TTC to apply bias')
    parser.add_argument('--bias_speed_floor', type=float, default=0.6, help='Min speed ratio when bias active')
    parser.add_argument('--bias_patience', type=int, default=8, help='Steps to measure progress')
    parser.add_argument('--bias_progress_eps', type=float, default=0.05, help='Min progress within window')
    parser.add_argument('--bias_late_ratio', type=float, default=0.5, help='Start stall check after ratio of max steps')
    parser.add_argument('--bias_stall_alpha', type=float, default=0.5, help='Stronger blend when stalled')
    parser.add_argument('--bias_deadline_alpha', type=float, default=0.0, help='Extra late-stage alpha ramp')
    parser.add_argument('--bias_deadline_min_clear', type=float, default=0.30, help='Min clearance for deadline boost')
    parser.add_argument('--bias_deadline_min_ttc', type=float, default=2.0, help='Min TTC for deadline boost')
    parser.add_argument('--bias_late_relax', type=float, default=0.0, help='Late-stage relaxation factor for clearance/TTC')
    parser.add_argument('--bias_late_speed_boost', type=float, default=0.0, help='Late-stage boost to speed floor (ratio)')
    parser.add_argument('--bias_no_stop', action='store_true', help='Avoid stop action when safe (goal-directed override)')
    parser.add_argument('--bias_stop_eps', type=float, default=0.05, help='Speed ratio treated as stop')
    parser.add_argument('--bias_scenarios', type=str, default='dense_square,large_circle,large_square',
                        help='Comma-separated scenario names to apply bias')
    # Mamba timeout rescue (late-stage, safe override)
    parser.add_argument('--mamba_rescue', action='store_true', help='Enable late-stage timeout rescue')
    parser.add_argument('--rescue_patience', type=int, default=8, help='Steps to measure progress')
    parser.add_argument('--rescue_progress_eps', type=float, default=0.05, help='Min progress within window')
    parser.add_argument('--rescue_min_clear', type=float, default=0.25, help='Min clearance to allow rescue')
    parser.add_argument('--rescue_min_ttc', type=float, default=2.0, help='Min TTC to allow rescue')
    parser.add_argument('--rescue_late_ratio', type=float, default=0.6, help='Start rescue after ratio of max steps')
    parser.add_argument('--rescue_speed_scale', type=float, default=1.0, help='Speed scale for rescue action')
    # Candidate failure analysis: no policy changes, only JSONL diagnostics.
    parser.add_argument('--candidate_diag', action='store_true',
                        help='Write per-step top-k candidate diagnostics for failed episodes')
    parser.add_argument('--candidate_diag_out', type=str, default='runs/mamba_vl/candidate_diag',
                        help='Directory for candidate diagnostic JSONL files')
    parser.add_argument('--candidate_diag_topk', type=int, default=8,
                        help='Number of ranked candidate actions to record per step')
    parser.add_argument('--candidate_diag_all', action='store_true',
                        help='Also record successful episodes; default records collision/timeout only')
    parser.add_argument('--candidate_safe_clearance', type=float, default=0.20,
                        help='Clearance threshold used to tag candidate actions as safe in diagnostics')
    parser.add_argument('--candidate_diag_thresholds', type=str, default='0.22,0.20,0.15',
                        help='Comma-separated clearance thresholds for global action-space safety summaries')
    parser.add_argument('--obs_noise_std', type=float, default=0.0,
                        help='Policy-observation Gaussian noise std in meters for human px/py; simulator truth is unchanged')
    parser.add_argument('--obs_dropout_prob', type=float, default=0.0,
                        help='Policy-observation probability of dropping each human; simulator truth is unchanged')
    args = parser.parse_args()
    if args.ablation_profile != 'none':
        if args.ablation_profile == 'no_vl':
            args.disable_value_lookahead = True
        elif args.ablation_profile == 'no_gdbn_module':
            args.disable_gdbn_veto = True
            args.disable_gdbn_compute = True
        elif args.ablation_profile == 'vl_only':
            args.disable_gdbn_veto = True
            args.disable_gdbn_compute = True
            args.disable_extra_veto = True
            args.value_lookahead_risk_weight = 0.0
        elif args.ablation_profile == 'nearest':
            args.human_selection_mode = 'nearest'
            args.k_nearest = 8
            args.k_threat = 0
        elif args.ablation_profile == 'veto025':
            args.risk_veto_threshold = 0.25
        elif args.ablation_profile == 'veto045':
            args.risk_veto_threshold = 0.45
        elif args.ablation_profile == 'qonly':
            args.eval_mode = 'q_only'
    # normalize scenario list
    args.bias_scenarios = [s.strip() for s in args.bias_scenarios.split(',') if s.strip()]

    logging.basicConfig(level=logging.INFO)
    print("[INFO] ========================================")
    print("[INFO] TEST_GDBN.PY - Controlled Bayesian Evaluation")
    print("[INFO] ========================================")
    print("[INFO] Config: T=24, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)")
    print("[INFO] Feature: Path-Y human selection (contracts_gdbn.py)")
    if args.discrete_search:
        print("[INFO] Feature: Discrete action search ENABLED (SARL-style)")
    if args.gating:
        print(f"[INFO] Feature: Density gating ENABLED (human>{args.human_threshold} or ttc<{args.ttc_threshold}s -> discrete)")
    print(f"[INFO] Checkpoint: {args.weights}")
    print(f"[INFO] Eval mode: {args.eval_mode}")
    print(f"[INFO] Protocol: time_limit={args.time_limit:g}s seed={args.seed}")
    if args.obs_noise_std > 0 or args.obs_dropout_prob > 0:
        print(
            f"[INFO] Observation perturbation: noise_std={args.obs_noise_std:.3f}m "
            f"dropout_prob={args.obs_dropout_prob:.3f} (policy input only)"
        )

    # Load SARL Value Network if specified
    sarl_value_net = None
    if args.sarl_value:
        print(f"[INFO] Loading SARL Value Network from {args.sarl_value}")
        sarl_value_net = SARLValueNetwork()
        sarl_weights = torch.load(args.sarl_value, map_location='cpu')
        sarl_value_net.load_state_dict(sarl_weights)
        sarl_value_net.eval()
        print("[INFO] SARL Value Network loaded successfully")
        print("[INFO] ========================================")
        print("[INFO] HYBRID MODE: Mamba-SARL Adaptive Gating")
        print("[INFO] - Simple scenes (≤7 humans, TTC>2.5s): Mamba (fast)")
        print("[INFO] - Complex scenes (>7 humans or TTC<2.5s): SARL (accurate)")
        print("[INFO] ========================================")

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    if sarl_value_net:
        sarl_value_net = sarl_value_net.to(device)

    if args.legacy_cases:
        test_cases = [
            {'desc': 'baseline_circle', 'sim': 'circle_crossing', 'human_num': 5,  'circle_radius': 4.0},
            {'desc': 'baseline_square', 'sim': 'square_crossing', 'human_num': 10, 'square_width': 10.0},
            {'desc': 'dense_circle',    'sim': 'circle_crossing', 'human_num': 10, 'circle_radius': 4.0},
            {'desc': 'dense_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 10.0},
            {'desc': 'large_circle',    'sim': 'circle_crossing', 'human_num': 12, 'circle_radius': 6.0},
            {'desc': 'large_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 14.0},
        ]
    else:
        test_cases = []
        for human_num in _parse_csv_list(args.human_nums, int):
            for sim in _parse_csv_list(args.test_sims, str):
                case = {'desc': f'Nh{human_num}_{sim}', 'sim': sim, 'human_num': human_num}
                if sim == 'circle_crossing':
                    case['circle_radius'] = args.circle_radius
                if sim == 'square_crossing':
                    case['square_width'] = args.square_width
                test_cases.append(case)

    for _case_idx, _case in enumerate(test_cases):
        _case['_case_idx'] = _case_idx

    if args.test_case is not None:
        if 0 <= args.test_case < len(test_cases):
            test_cases = [test_cases[args.test_case]]

    print(f"Using device: {device}")

    # Initialize global random seed for reproducibility
    base_seed = int(args.seed)
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32 - 1))
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    for case_idx, case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"Test Case [{case_idx}]: {case['desc']} | {case['human_num']} humans")
        print(f"{'='*70}")

        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config)
        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config)
        time_limit = int(float(args.time_limit))

        # [FIX 1] Config Injection - MATCH TRAINING CONFIG
        if not policy_config.has_section('buffer'): policy_config.add_section('buffer')
        _seq_len = str(args.seq_len) if args.seq_len is not None else policy_config.get('buffer', 'seq_len', fallback=policy_config.get('temporal', 'T', fallback='24'))
        policy_config.set('buffer', 'seq_len', _seq_len)
        if not policy_config.has_section('temporal'): policy_config.add_section('temporal')
        policy_config.set('temporal', 'T', _seq_len)

        if not policy_config.has_section('robot'): policy_config.add_section('robot')
        policy_config.set('robot', 'v_pref', '1.0')  # Match training
        policy_config.set('robot', 'success_radius', '0.25')
        if not policy_config.has_section('eval_protocol'): policy_config.add_section('eval_protocol')
        policy_config.set('eval_protocol', 'time_limit', str(int(time_limit)))
        policy_config.set('eval_protocol', 'time_step', '0.25')
        policy_config.set('eval_protocol', 'success_radius', '0.25')

        # [FIX 2] Env Injection - MATCH TRAINING CONFIG
        if not env_config.has_section('env'): env_config.add_section('env')
        env_config.set('env', 'time_step', '0.25')  # Match training: dt=0.25
        env_config.set('env', 'time_limit', str(int(time_limit)))
        if not env_config.has_section('robot'): env_config.add_section('robot')
        env_config.set('robot', 'success_radius', '0.25')

        # [FIX FINAL] MERGE EVERYTHING from env_config into policy_config
        # This mimics train.py behavior and solves all missing section errors.
        for section in env_config.sections():
            if not policy_config.has_section(section):
                policy_config.add_section(section)
            for key, value in env_config.items(section):
                # Only set if not already present (preserve policy specific overrides)
                if not policy_config.has_option(section, key):
                    policy_config.set(section, key, value)

        # [FIX SUPPLEMENT] Ensure manual injections are still present if not in env.config
        if not policy_config.has_section('train'): policy_config.add_section('train')
        if not policy_config.has_option('train', 'gamma'): policy_config.set('train', 'gamma', '0.99')

        if not policy_config.has_section('env'): policy_config.add_section('env')
        if not policy_config.has_option('env', 'time_step'): policy_config.set('env', 'time_step', '0.25')

        # [FIX 6] Minimal config injection for Mamba
        # Mamba only needs epsilon_start from [sarl] section (for compatibility)
        if not policy_config.has_section('sarl'):
            policy_config.add_section('sarl')
        policy_config.set('sarl', 'epsilon_start', '0.0')

        selector_overrides = []
        if args.human_selection_mode is not None or args.k_nearest is not None or args.k_threat is not None:
            if not policy_config.has_section('human_selection'):
                policy_config.add_section('human_selection')
            if args.human_selection_mode is not None:
                policy_config.set('human_selection', 'mode', args.human_selection_mode)
                selector_overrides.append(f"mode={args.human_selection_mode}")
            if args.k_nearest is not None:
                policy_config.set('human_selection', 'k_nearest', str(int(args.k_nearest)))
                selector_overrides.append(f"k_nearest={int(args.k_nearest)}")
            if args.k_threat is not None:
                policy_config.set('human_selection', 'k_threat', str(int(args.k_threat)))
                selector_overrides.append(f"k_threat={int(args.k_threat)}")
            print("[INFO] Ablation: human_selection " + " ".join(selector_overrides))

        if not env_config.has_section('robot'): env_config.add_section('robot')
        env_config.set('robot', 'v_pref', '1.0')

        for section in policy_config.sections():
            if not env_config.has_section(section): env_config.add_section(section)
            for key, value in policy_config.items(section): env_config.set(section, key, value)

        if not env_config.has_section('sim'): env_config.add_section('sim')
        env_config.set('sim', 'test_sim', case['sim'])
        env_config.set('sim', 'human_num', str(case['human_num']))
        if 'circle_radius' in case: env_config.set('sim', 'circle_radius', str(case['circle_radius']))
        if 'square_width' in case: env_config.set('sim', 'square_width', str(case['square_width']))

        weights_path = _resolve_weights_path(args.model_dir, args.weights)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        checkpoint = safe_torch_load(weights_path, map_location=device)
        state_dict = checkpoint.get('policy_state', checkpoint.get('model_state_dict', checkpoint.get('value_state', checkpoint.get('model', checkpoint))))
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        if 'action_mean.weight' in state_dict and 'q_head.weight' not in state_dict:
            raise RuntimeError(
                "Incompatible checkpoint: this file contains the old continuous-action "
                "Mamba policy (action_mean) but the current test path expects the "
                "discrete q_head policy. Use a discrete checkpoint such as "
                "k4_safety_policy.pth / rl_model_final.pth from the current run, "
                "or run the old test code for this legacy checkpoint."
            )

        # Build the policy with the token width used by the checkpoint.
        # Older accepted checkpoints were trained with 13-dim kinematic tokens;
        # newer belief-token experiments use 23 dims. strict=False does not
        # ignore shape mismatches, so the config must be aligned before init.
        robot_w = state_dict.get('spatial_encoder.robot_encoder.0.weight')
        human_w = state_dict.get('spatial_encoder.human_encoder.0.weight')
        if robot_w is not None and getattr(robot_w, 'ndim', 0) == 2:
            ck_token_dim = int(robot_w.shape[1])
            if human_w is not None and getattr(human_w, 'ndim', 0) == 2:
                ck_human_token_dim = int(human_w.shape[1]) - 8
                if ck_human_token_dim > 0 and ck_human_token_dim != ck_token_dim:
                    print(
                        "[WARNING] checkpoint token dims disagree: "
                        f"robot={ck_token_dim}, human={ck_human_token_dim}; using robot dim"
                    )
            if not policy_config.has_section('belief'):
                policy_config.add_section('belief')
            base_dim = min(13, ck_token_dim)
            belief_dim = max(0, ck_token_dim - base_dim)
            policy_config.set('belief', 'base_token_dim', str(base_dim))
            policy_config.set('belief', 'belief_dim', str(belief_dim))
            policy_config.set('belief', 'token_dim', str(ck_token_dim))
            policy_config.set('belief', 'enable', 'true' if belief_dim > 0 else 'false')
            print(
                "[INFO] checkpoint architecture: "
                f"token_dim={ck_token_dim}, belief_dim={belief_dim}"
            )

        q_head_w = state_dict.get('q_head.weight')
        if q_head_w is not None and getattr(q_head_w, 'ndim', 0) == 2:
            ck_actions = int(q_head_w.shape[0])
            if not policy_config.has_section('policy'):
                policy_config.add_section('policy')
            n_speeds = policy_config.getint('policy', 'n_speeds', fallback=5)
            n_headings = policy_config.getint('policy', 'n_headings', fallback=16)
            grid_no_stop = int(n_speeds) * int(n_headings)
            if ck_actions in (grid_no_stop, grid_no_stop + 1):
                include_stop = ck_actions == grid_no_stop + 1
                policy_config.set('policy', 'include_stop', 'true' if include_stop else 'false')
                if not env_config.has_section('policy'):
                    env_config.add_section('policy')
                env_config.set('policy', 'include_stop', 'true' if include_stop else 'false')
                print(
                    "[INFO] checkpoint action grid: "
                    f"actions={ck_actions}, include_stop={include_stop}"
                )
                init_grid_from_cfg(policy_config)
            else:
                print(
                    "[WARNING] checkpoint action count does not match 5x16(+stop): "
                    f"actions={ck_actions}"
                )

        # ========================================
        # MAMBA POLICY LOADING (test.py specific)
        # ========================================
        try:
            print(f"[INFO] Loading Mamba policy: {args.policy}")
            policy_class = policy_factory[args.policy]
            policy = policy_class(policy_config)
        except Exception as e:
            print(f"[WARNING] Failed to init with config, trying without: {e}")
            policy = policy_class()
            if hasattr(policy, 'configure'):
                policy.configure(policy_config)

        # [FIX CRITICAL] Ensure device and phase are set
        if hasattr(policy, 'set_device'):
            policy.set_device(device)
        if hasattr(policy, 'set_phase'):
            policy.set_phase('test')

        # Auto-detect architecture from checkpoint and rebuild policy if needed
        ck_is_gru = any('weight_ih_l' in k for k in state_dict.keys())
        model_is_gru = any('weight_ih_l' in k for k, _ in policy.named_parameters())
        if ck_is_gru != model_is_gru:
            from crowd_nav.policy.mamba_rl_gdbn import GRUTemporalEncoder, MambaTemporalEncoder
            if ck_is_gru:
                policy.temporal_encoder = GRUTemporalEncoder(
                    d_model=policy.temporal_encoder.d_model if hasattr(policy.temporal_encoder, 'd_model') else 256,
                    n_layers=4, dropout=0.0
                ).to(device)
                print("[INFO] Auto-switched temporal encoder: Mamba -> GRU")
            else:
                d_model = next(iter(state_dict.values())).shape[-1] if state_dict else 256
                policy.temporal_encoder = MambaTemporalEncoder(
                    d_model=256, n_layers=4, d_state=64, d_conv=4, expand=2
                ).to(device)
                print("[INFO] Auto-switched temporal encoder: GRU -> Mamba")

        if hasattr(policy, 'load_state_dict'):
            result = policy.load_state_dict(state_dict, strict=False)
            missing = getattr(result, 'missing_keys', [])
            unexpected = getattr(result, 'unexpected_keys', [])
            print(f"[INFO] Loaded checkpoint: {weights_path} missing={len(missing)} unexpected={len(unexpected)}")
        elif hasattr(policy, 'model'):
            policy.model.load_state_dict(state_dict, strict=False)

        _apply_checkpoint_runtime(policy, checkpoint, policy_config, args.model_dir, args.weights, args)
        if args.q_prior_belief_mode != 'checkpoint':
            _set_policy_attr(policy, 'bayesian_q_prior_belief_mode', args.q_prior_belief_mode)
            print(f"[INFO] Override q_prior_belief_mode={args.q_prior_belief_mode}")

        if args.eval_mode == 'bayesian':
            _set_policy_attr(policy, 'use_bayesian_planner', True)
            if args.risk_model == 'cv':
                if ConstantVelocityRiskModel is None:
                    raise RuntimeError("CV risk model could not be imported")
                cv_model = ConstantVelocityRiskModel(
                    max_peds=policy_config.getint('belief', 'num_humans', fallback=8),
                    belief_dim=policy_config.getint('belief', 'belief_dim', fallback=10),
                    modeled_peds=5,
                )
                _set_policy_attr(policy, 'gdbn_module', cv_model)
                print("[INFO] Runtime risk model: constant velocity")
            elif args.gdbn_params:
                installed = _install_gdbn(
                    policy, policy_config, args.model_dir,
                    args.gdbn_K, args.gdbn_params, 'CLI override',
                )
                if not installed:
                    raise RuntimeError(
                        f"Failed to install requested GDBN K={args.gdbn_K}: {args.gdbn_params}"
                    )
            elif (
                not args.disable_gdbn_compute
                and _get_policy_attr(policy, 'gdbn_module', None) is None
            ):
                fallback_K = policy_config.getint('gdbn', 'K', fallback=4)
                fallback_params = policy_config.get('gdbn', 'params_dir', fallback='gdbn_params_k4')
                if not fallback_params or str(fallback_params).strip() in ('', 'None', 'none'):
                    fallback_params = 'gdbn_params_k4'
                _install_gdbn(policy, policy_config, args.model_dir, fallback_K, fallback_params, 'bayesian fallback')
        elif args.eval_mode == 'q_only':
            _set_policy_attr(policy, 'use_bayesian_planner', False)

        if args.disable_belief_tokens:
            _set_policy_attr(policy, 'disable_belief_tokens', True)
            _set_policy_attr(policy, 'bayesian_q_prior_belief_mode', 'zero')
            print("[INFO] Ablation: belief-token features disabled")

        if args.disable_value_lookahead:
            _set_policy_attr(policy, 'bayesian_value_lookahead_enable', False)
            _set_policy_attr(policy, 'bayesian_value_lookahead_risk_weight', 0.0)
            print("[INFO] Ablation: disabled value_lookahead")
        elif args.value_lookahead_risk_weight is not None:
            _set_policy_attr(policy, 'bayesian_value_lookahead_risk_weight', float(args.value_lookahead_risk_weight))
            print(f"[INFO] Ablation: value_lookahead_risk_weight={float(args.value_lookahead_risk_weight):.3f}")

        if args.risk_veto_threshold is not None:
            _set_policy_attr(policy, 'bayesian_risk_veto_threshold', float(args.risk_veto_threshold))
            print(f"[INFO] Ablation: risk_veto_threshold={float(args.risk_veto_threshold):.3f}")

        if args.override_min_clearance is not None:
            _set_policy_attr(policy, 'bayesian_override_min_clearance', float(args.override_min_clearance))
            print(f"[INFO] Ablation: override_min_clearance={float(args.override_min_clearance):.3f}")

        if args.disable_gdbn_veto:
            _set_policy_attr(policy, 'bayesian_risk_veto_threshold', 0.0)
            _set_policy_attr(policy, 'bayesian_override_min_clearance', 0.0)
            print("[INFO] Ablation: disabled Bayesian/GDBN clearance+risk veto")

        if args.disable_gdbn_compute:
            _set_policy_attr(policy, 'skip_gdbn_compute', True)
            _set_policy_attr(policy, 'gdbn_module', None)
            _set_policy_attr(policy, 'belief_enabled', False)
            _set_policy_attr(policy, 'bayesian_q_prior_belief_mode', 'zero')
            print("[INFO] Ablation: disabled GDBN compute/update/rollout")

        if args.disable_extra_veto:
            _set_policy_attr(policy, 'extra_human_veto_clearance', -1e9)
            _set_policy_attr(policy, 'extra_human_penalty_weight', 0.0)
            print("[INFO] Ablation: disabled extra-human veto+penalty")

        print(
            "[INFO] Runtime: planner=%s q_prior=%s risk_model=%s K=%s belief_tokens=%s reranker=%s/top%d" % (
                _get_policy_attr(policy, 'use_bayesian_planner', None),
                _get_policy_attr(policy, 'bayesian_q_prior_belief_mode', None),
                args.risk_model,
                getattr(_get_policy_attr(policy, 'gdbn_module', None), 'K', None),
                not bool(_get_policy_attr(policy, 'disable_belief_tokens', False)),
                _get_policy_attr(policy, 'bayesian_rl_reranker_enable', False),
                int(_get_policy_attr(policy, 'bayesian_rl_reranker_topk', 0) or 0),
            )
        )
        print(
            "[INFO] Runtime: value_lookahead=%s risk_w=%.3f" % (
                _get_policy_attr(policy, 'bayesian_value_lookahead_enable', False),
                float(_get_policy_attr(policy, 'bayesian_value_lookahead_risk_weight', 0.0) or 0.0),
            )
        )

        if args.candidate_diag:
            os.makedirs(args.candidate_diag_out, exist_ok=True)
            safe_case = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(case['desc']))
            safe_weights = re.sub(r'[^A-Za-z0-9_.-]+', '_', os.path.basename(str(args.weights)))
            diag_path = os.path.join(
                args.candidate_diag_out,
                f"{safe_case}_{safe_weights}_{int(time.time())}.jsonl",
            )
            _set_policy_attr(policy, 'candidate_diagnostics_enable', True)
            _set_policy_attr(policy, 'candidate_diagnostics_topk', int(args.candidate_diag_topk))
            _set_policy_attr(policy, 'candidate_diag_safe_clearance', float(args.candidate_safe_clearance))
            _set_policy_attr(policy, 'candidate_diag_thresholds', _parse_csv_list(args.candidate_diag_thresholds, float))
            setattr(_raw_policy(policy), '_candidate_diag_path', diag_path)
            print(f"[INFO] Candidate diagnostics: {diag_path}")

        if hasattr(policy, 'to'): policy.to(device)
        if hasattr(policy, 'eval'): policy.eval()
        elif hasattr(policy, 'model'): policy.model.eval()

        print(f"[DEBUG] Policy Class: {type(policy)}")

        q_networks_loaded = False
        if args.discrete_search or args.gating:
            if hasattr(policy, 'load_q_networks'):
                q_ckpt_candidates = [
                    os.path.join(args.model_dir, 'value_pretrain.pth'),
                    weights_path,
                ]
                for q_path in q_ckpt_candidates:
                    if os.path.exists(q_path):
                        q_networks_loaded = policy.load_q_networks(q_path)
                        if q_networks_loaded:
                            print(f"[INFO] Q networks loaded from {q_path}")
                            break
                if not q_networks_loaded:
                    print(f"[WARNING] Q networks not found, using continuous only")

        env = CrowdSim()
        env.configure(env_config)
        env.phase = 'test'
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        robot.env = env
        env.set_robot(robot)

        if hasattr(policy, 'set_env'): policy.set_env(env)
        if hasattr(policy, 'set_phase'): policy.set_phase('test')
        if hasattr(policy, 'set_env_dt'): policy.set_env_dt(0.25)  # Match training dt

        policy._test_args = args
        policy._q_networks_loaded = q_networks_loaded

        success_count = 0
        collision_count = 0
        timeout_count = 0
        total_mamba_steps = 0
        total_sarl_steps = 0
        success_times = []
        min_dist_sum = 0.0
        min_dist_steps = 0
        discomfort_events = 0
        time_step = env_config.getfloat('env', 'time_step', fallback=0.25)
        discomfort_dist = env_config.getfloat('reward', 'discomfort_dist', fallback=0.2)

        pbar = tqdm(range(args.episodes), ncols=100, disable=args.no_progress)
        for ep in pbar:
            episode_seed = base_seed * 1000000 + int(case.get('_case_idx', case_idx)) * 100000 + ep

            outcome, steps, robot_positions, min_dists, mamba_count, sarl_count = test_episode(
                env, robot, policy, episode_seed,
                robot_start=case.get('robot_start'),
                robot_goal=case.get('robot_goal'),
                sarl_value_net=sarl_value_net,
                device=device,
                case_desc=case['desc'],
            )

            if outcome == 'success':
                success_count += 1
                success_times.append(steps * time_step)
            elif outcome == 'collision': collision_count += 1
            else: timeout_count += 1

            total_mamba_steps += mamba_count
            total_sarl_steps += sarl_count

            if min_dists:
                min_dist_sum += float(np.sum(min_dists))
                min_dist_steps += len(min_dists)
                discomfort_events += sum(1 for d in min_dists if d < discomfort_dist)

            pbar.set_postfix({'S': f'{success_count}/{ep+1}', 'C': f'{collision_count}/{ep+1}'})
            if args.no_progress and args.progress_interval > 0 and (
                ep == 0 or (ep + 1) % args.progress_interval == 0 or (ep + 1) == args.episodes
            ):
                print(
                    f"[PROGRESS] {case['desc']} {ep+1}/{args.episodes} "
                    f"S={success_count} C={collision_count} T={timeout_count}",
                    flush=True,
                )

        print(f"\nResults for {case['desc']}:")
        print(f"  SUCCESS:   {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
        print(f"  COLLISION: {collision_count}/{args.episodes} ({100*collision_count/args.episodes:.1f}%)")
        print(f"  TIMEOUT:   {timeout_count}/{args.episodes} ({100*timeout_count/args.episodes:.1f}%)")
        avg_time = float(np.mean(success_times)) if success_times else 0.0
        avg_disc_freq = discomfort_events / args.episodes
        avg_disc_dist = (min_dist_sum / min_dist_steps) if min_dist_steps else float('inf')
        print(f"  TIME TAKEN (s): {avg_time:.2f}" if success_times else "  TIME TAKEN (s): n/a")
        print(f"  DISC. FREQ: {avg_disc_freq:.2f}")
        print(f"  DISC. DIST (m): {avg_disc_dist:.2f}")

        if sarl_value_net is not None and policy is not None:
            total_steps = total_mamba_steps + total_sarl_steps
            if total_steps > 0:
                mamba_pct = 100 * total_mamba_steps / total_steps
                sarl_pct = 100 * total_sarl_steps / total_steps
                print(f"  HYBRID:    Mamba {mamba_pct:.1f}% | SARL {sarl_pct:.1f}% (of {total_steps} steps)")

if __name__ == '__main__':
    main()
