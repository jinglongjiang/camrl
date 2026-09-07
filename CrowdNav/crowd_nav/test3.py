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
import sys
import re
import torch
import numpy as np
import random
import time
from tqdm import tqdm
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.contracts import GRID, discrete_index_to_action, joint34_to_tokens

# ========== SARL Value Network for Discrete Action Search ==========
import torch.nn as nn

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
    """计算场景复杂度指标

    Returns:
        n_humans: 人类数量
        min_ttc: 最小TTC（Time-To-Collision）
    """
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_episode(env, robot, policy, test_case_idx, robot_start=None, robot_goal=None, sarl_value_net=None, device=None):
    if hasattr(policy, 'reset_episode_stats'):
        policy.reset_episode_stats()

    # 统计混合策略使用情况
    mamba_count = 0
    sarl_count = 0

    reset_result = env.reset(seed=test_case_idx)
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

    max_steps = 500

    while not done and steps < max_steps:
        # --- Manual Act with TTC-Sorting (与训练时contracts.py一致) ---
        # 1. Get full state
        robot_state = robot.get_full_state()

        # 2. Get sorted humans by TTC (与contracts.py第435行一致)
        humans = env.humans
        ttc_list = []
        for h in humans:
            rel_x = h.px - robot.px
            rel_y = h.py - robot.py
            rel_vx = h.vx - robot.vx
            rel_vy = h.vy - robot.vy
            dist = np.sqrt(rel_x**2 + rel_y**2 + 1e-6)
            closing = -(rel_x * rel_vx + rel_y * rel_vy) / (dist + 1e-6)
            # 与contracts.py完全一致：静止人用距离*10作为TTC
            ttc = dist / (closing + 1e-6) if closing > 0.1 else dist * 10
            ttc_list.append((ttc, h))
        # 按TTC升序排序（最危险的排前面）
        sorted_humans = [h for _, h in sorted(ttc_list, key=lambda x: x[0])]

        # 3. Construct JointState
        human_states = [h.get_observable_state() for h in sorted_humans]
        state = JointState(robot_state, human_states)

        # 4. Predict (混合策略：Mamba-SARL Hybrid with Adaptive Gating)
        test_args = getattr(policy, '_test_args', None)

        if sarl_value_net is not None and policy is not None:
            # ========== 混合门控策略 ==========
            # 计算场景复杂度
            n_humans, min_ttc = compute_scene_complexity(state)

            # 门控阈值（可调参数）
            HUMAN_THRESHOLD = 7      # 人数超过7个认为是复杂场景
            TTC_THRESHOLD = 2.5      # TTC小于2.5s认为是危险场景

            # 决策：复杂/危险场景用SARL（慢但准），简单场景用Mamba（快）
            if n_humans > HUMAN_THRESHOLD or min_ttc < TTC_THRESHOLD:
                # 复杂场景：使用SARL Value网络做离散搜索
                action = sarl_discrete_search(state, sarl_value_net, device)
                sarl_count += 1
            else:
                # 简单场景：使用Mamba快速预测
                action = policy.predict(state)
                mamba_count += 1
            # ==================================
        elif sarl_value_net is not None:
            # 纯SARL模式（无Mamba）
            action = sarl_discrete_search(state, sarl_value_net, device)
            sarl_count += 1
        elif test_args and test_args.gating and hasattr(policy, 'predict_with_gating'):
            # Mamba内置门控模式
            action = policy.predict_with_gating(
                state,
                human_threshold=test_args.human_threshold,
                ttc_threshold=test_args.ttc_threshold
            )
        elif test_args and test_args.discrete_search and hasattr(policy, 'act_discrete'):
            # Mamba离散搜索模式
            action = policy.act_discrete(state)
        else:
            # 默认连续动作（纯Mamba）
            action = policy.predict(state)
        # --------------------------------------------------------------

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

    return outcome, steps, robot_positions, min_dists, mamba_count, sarl_count

def compute_metrics(outcome, steps, robot_positions, min_dists, start_pos, goal_pos, time_step):
    metrics = {}
    metrics['time'] = steps * time_step
    if len(robot_positions) > 1:
        path_length = sum(np.linalg.norm(np.array(robot_positions[i+1]) - np.array(robot_positions[i]))
                         for i in range(len(robot_positions)-1))
        optimal_length = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))
        metrics['path_efficiency'] = optimal_length / max(path_length, 1e-6) if path_length > 0 else 0.0
    else:
        metrics['path_efficiency'] = 0.0
    metrics['min_separation'] = np.min(min_dists) if min_dists else float('inf')
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='lstm', help='Policy name (only lstm supported)')
    parser.add_argument('--model_dir', type=str, default='runs/mamba_vl', help='Model directory')
    parser.add_argument('--weights', type=str, default='rl_model_lstm.pth', help='Checkpoint name or "latest"')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--episodes', type=int, default=500, help='Episodes per scenario')
    parser.add_argument('--test_case', type=int, default=None, help='Run only specified test case')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--visualize', action='store_true')
    # 方案1：离散动作搜索选项
    parser.add_argument('--discrete_search', action='store_true', help='Enable discrete action search (SARL-style)')
    parser.add_argument('--gating', action='store_true', help='Enable density gating (auto switch between continuous/discrete)')
    parser.add_argument('--human_threshold', type=int, default=8, help='Human count threshold for gating')
    parser.add_argument('--ttc_threshold', type=float, default=2.0, help='TTC threshold for gating')
    # 方案2：使用SARL Value网络做离散搜索
    parser.add_argument('--sarl_value', type=str, default=None, help='Path to SARL rl_model.pth for value-based discrete search')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print("[INFO] ========================================")
    print("[INFO] TEST3.PY - LSTM Baseline Testing")
    print("[INFO] ========================================")
    print("[INFO] Config: T=12, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)")
    print("[INFO] Feature: TTC-Sorting enabled (与contracts.py训练时一致)")
    if args.discrete_search:
        print("[INFO] Feature: Discrete action search ENABLED (SARL-style)")
    if args.gating:
        print(f"[INFO] Feature: Density gating ENABLED (human>{args.human_threshold} or ttc<{args.ttc_threshold}s -> discrete)")

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

    test_cases = [
        {'desc': 'baseline_circle', 'sim': 'circle_crossing', 'human_num': 5,  'circle_radius': 4.0},
        {'desc': 'baseline_square', 'sim': 'square_crossing', 'human_num': 10, 'square_width': 10.0},
        {'desc': 'dense_circle',    'sim': 'circle_crossing', 'human_num': 10, 'circle_radius': 4.0},
        {'desc': 'dense_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 10.0},
        {'desc': 'large_circle',    'sim': 'circle_crossing', 'human_num': 12, 'circle_radius': 6.0},
        {'desc': 'large_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 14.0},
    ]

    if args.test_case is not None:
        if 0 <= args.test_case < len(test_cases):
            test_cases = [test_cases[args.test_case]]

    print(f"Using device: {device}")

    base_seed = int(time.time() * 1000) % (2**31)
    random.seed(base_seed)
    np.random.seed(base_seed)

    for case_idx, case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"Test Case [{case_idx}]: {case['desc']} | {case['human_num']} humans")
        print(f"{'='*70}")

        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config)
        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config)

        # [FIX 1] Config Injection - MATCH TRAINING CONFIG
        if not policy_config.has_section('buffer'): policy_config.add_section('buffer')
        policy_config.set('buffer', 'seq_len', '12')  # Match training: T=12 (from train.config)

        if not policy_config.has_section('robot'): policy_config.add_section('robot')
        policy_config.set('robot', 'v_pref', '1.0')   # Match training

        # [FIX 2] Env Injection - MATCH TRAINING CONFIG
        if not env_config.has_section('env'): env_config.add_section('env')
        env_config.set('env', 'time_step', '0.25')  # Match training: dt=0.25
        env_config.set('env', 'time_limit', '25')

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

        # [FIX 6] Inject [sarl] section for epsilon_start (Required by MambaRLPolicy __init__)
        if not policy_config.has_section('sarl'): policy_config.add_section('sarl')
        policy_config.set('sarl', 'epsilon_start', '0.0')
        # Also need these for SARL configure()
        if not policy_config.has_option('sarl', 'mlp1_dims'): policy_config.set('sarl', 'mlp1_dims', '150, 100')
        if not policy_config.has_option('sarl', 'mlp2_dims'): policy_config.set('sarl', 'mlp2_dims', '100, 50')
        if not policy_config.has_option('sarl', 'mlp3_dims'): policy_config.set('sarl', 'mlp3_dims', '150, 100, 100, 1')
        if not policy_config.has_option('sarl', 'attention_dims'): policy_config.set('sarl', 'attention_dims', '100, 100, 1')
        if not policy_config.has_option('sarl', 'with_om'): policy_config.set('sarl', 'with_om', 'false')
        if not policy_config.has_option('sarl', 'with_global_state'): policy_config.set('sarl', 'with_global_state', 'true')
        if not policy_config.has_option('sarl', 'multiagent_training'): policy_config.set('sarl', 'multiagent_training', 'false')

        # [FIX 7] Inject [rl] section for SARL (cadrl.py expects [rl] gamma)
        if not policy_config.has_section('rl'): policy_config.add_section('rl')
        policy_config.set('rl', 'gamma', '0.99')

        # [FIX 8] Inject [action_space] sampling for SARL
        if not policy_config.has_section('action_space'): policy_config.add_section('action_space')
        if not policy_config.has_option('action_space', 'sampling'): policy_config.set('action_space', 'sampling', 'exponential')
        if not policy_config.has_option('action_space', 'speed_samples'): policy_config.set('action_space', 'speed_samples', '5')
        if not policy_config.has_option('action_space', 'rotation_samples'): policy_config.set('action_space', 'rotation_samples', '16')
        if not policy_config.has_option('action_space', 'kinematics'): policy_config.set('action_space', 'kinematics', 'holonomic')
        if not policy_config.has_option('action_space', 'query_env'): policy_config.set('action_space', 'query_env', 'true')
        # CADRL expects time_step in [action_space] too (duplicate from env is safer)
        if not policy_config.has_option('action_space', 'time_step'): policy_config.set('action_space', 'time_step', '0.25')

        # [FIX 9] Inject [om] section for CADRL/SARL (Occupancy Map params)
        # Even if with_om=false, set_common_parameters reads these.
        if not policy_config.has_section('om'): policy_config.add_section('om')
        if not policy_config.has_option('om', 'cell_num'): policy_config.set('om', 'cell_num', '4')
        if not policy_config.has_option('om', 'cell_size'): policy_config.set('om', 'cell_size', '1.0')
        if not policy_config.has_option('om', 'om_channel_size'): policy_config.set('om', 'om_channel_size', '3')

        # [FIX 10] Inject [cadrl] section for CADRL policy
        if not policy_config.has_section('cadrl'): policy_config.add_section('cadrl')
        if not policy_config.has_option('cadrl', 'mlp_dims'): policy_config.set('cadrl', 'mlp_dims', '150, 100, 100, 1')
        if not policy_config.has_option('cadrl', 'with_om'): policy_config.set('cadrl', 'with_om', 'false')
        if not policy_config.has_option('cadrl', 'with_global_state'): policy_config.set('cadrl', 'with_global_state', 'false')
        if not policy_config.has_option('cadrl', 'multiagent_training'): policy_config.set('cadrl', 'multiagent_training', 'false')

        # [FIX 11] Inject [lstm] and [lstm_rl] sections for LSTM policy
        if not policy_config.has_section('lstm'): policy_config.add_section('lstm')
        if not policy_config.has_option('lstm', 'with_om'): policy_config.set('lstm', 'with_om', 'false')
        if not policy_config.has_option('lstm', 'with_global_state'): policy_config.set('lstm', 'with_global_state', 'false')

        if not policy_config.has_section('lstm_rl'): policy_config.add_section('lstm_rl')
        if not policy_config.has_option('lstm_rl', 'mlp2_dims'): policy_config.set('lstm_rl', 'mlp2_dims', '150, 100, 100, 1')
        if not policy_config.has_option('lstm_rl', 'mlp1_dims'): policy_config.set('lstm_rl', 'mlp1_dims', '150, 100')
        if not policy_config.has_option('lstm_rl', 'global_state_dim'): policy_config.set('lstm_rl', 'global_state_dim', '50')
        if not policy_config.has_option('lstm_rl', 'with_om'): policy_config.set('lstm_rl', 'with_om', 'false')
        if not policy_config.has_option('lstm_rl', 'with_global_state'): policy_config.set('lstm_rl', 'with_global_state', 'false')
        if not policy_config.has_option('lstm_rl', 'with_interaction_module'): policy_config.set('lstm_rl', 'with_interaction_module', 'false')
        if not policy_config.has_option('lstm_rl', 'multiagent_training'): policy_config.set('lstm_rl', 'multiagent_training', 'false')

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

        # ========================================
        # LSTM POLICY LOADING (test3.py specific)
        # ========================================
        try:
            print(f"[INFO] Loading LSTM policy")
            from crowd_nav.policy.lstm_rl import LstmRL
            policy = LstmRL()

            # Configure policy
            if hasattr(policy, 'configure'):
                policy.configure(policy_config)

        except Exception as e:
            print(f"[ERROR] LSTM initialization failed: {e}")
            raise

        # [FIX CRITICAL] Ensure device and phase are set
        if hasattr(policy, 'set_device'):
            policy.set_device(device)
        if hasattr(policy, 'set_phase'):
            policy.set_phase('test')

        # [FIX CRITICAL] Force SARL mode if supported
        if hasattr(policy, 'use_sarl_predict'):
            policy.use_sarl_predict = True
            print("[INFO] Forced SARL-style prediction (use_sarl_predict=True)")

        if args.weights == 'latest':
            files = [f for f in os.listdir(args.model_dir) if re.match(r'rl_model_ep\d+\.pth', f)]
            if files:
                files.sort(key=lambda x: os.path.getmtime(os.path.join(args.model_dir, x)))
                weights_path = os.path.join(args.model_dir, files[-1])
            else:
                weights_path = os.path.join(args.model_dir, 'rl_model.pth')
        else:
            weights_path = os.path.join(args.model_dir, args.weights)

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('policy_state', checkpoint.get('model_state_dict', checkpoint.get('value_state', checkpoint.get('model', checkpoint))))
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        if hasattr(policy, 'load_state_dict'): policy.load_state_dict(state_dict)
        elif hasattr(policy, 'model'): policy.model.load_state_dict(state_dict)

        # 强制移动到设备（修复LSTM hidden state设备不匹配）
        if hasattr(policy, 'to'):
            policy.to(device)
        if hasattr(policy, 'model'):
            policy.model.to(device)

        if hasattr(policy, 'eval'): policy.eval()
        elif hasattr(policy, 'model'): policy.model.eval()

        print(f"[DEBUG] Policy Class: {type(policy)}")

        # [FIX LSTM] Monkey-patch forward to fix device issue
        if hasattr(policy, 'model') and hasattr(policy.model, 'lstm'):
            original_forward = policy.model.forward
            def fixed_forward(state):
                size = state.shape
                self_state = state[:, 0, :policy.model.self_state_dim]
                h0 = torch.zeros(1, size[0], policy.model.lstm_hidden_dim, device=state.device)
                c0 = torch.zeros(1, size[0], policy.model.lstm_hidden_dim, device=state.device)
                output, (hn, cn) = policy.model.lstm(state, (h0, c0))
                hn = hn.squeeze(0)
                joint_state = torch.cat([self_state, hn], dim=1)
                value = policy.model.mlp(joint_state)
                return value
            policy.model.forward = fixed_forward

        # 方案1：尝试加载Q网络（用于离散动作搜索）
        q_networks_loaded = False
        if args.discrete_search or args.gating:
            if hasattr(policy, 'load_q_networks'):
                # 优先尝试 value_pretrain.pth（Q网络专用checkpoint）
                q_ckpt_candidates = [
                    os.path.join(args.model_dir, 'value_pretrain.pth'),
                    weights_path,  # 回退到policy checkpoint
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

        # 保存测试配置到policy（用于门控决策）
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

        pbar = tqdm(range(args.episodes), ncols=100)
        for ep in pbar:
            episode_seed = random.randint(0, 2**31 - 1)

            outcome, steps, robot_positions, min_dists, mamba_count, sarl_count = test_episode(
                env, robot, policy, episode_seed,
                robot_start=case.get('robot_start'),
                robot_goal=case.get('robot_goal'),
                sarl_value_net=sarl_value_net,
                device=device
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

        # 打印混合策略统计
        if sarl_value_net is not None and policy is not None:
            total_steps = total_mamba_steps + total_sarl_steps
            if total_steps > 0:
                mamba_pct = 100 * total_mamba_steps / total_steps
                sarl_pct = 100 * total_sarl_steps / total_steps
                print(f"  HYBRID:    Mamba {mamba_pct:.1f}% | SARL {sarl_pct:.1f}% (of {total_steps} steps)")

if __name__ == '__main__':
    main()
