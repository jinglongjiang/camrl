# -*- coding: utf-8 -*-
"""
P2/P3 Contracts - SSOT (Single Source of Truth) for Actions, States, and Physics

Core functions:
1. Action Grid: Reads from config [policy], with fuse assertion (v_max >= 0.8*v_pref)
2. Legacy Interface: ensure_tensor / DEFAULT_DTYPE / validate_state_shape (compatible with other imports)
3. State Conversion: tokens_to_34d, joint34_to_tokens (including derived features like ttc_inv/in_soc)
4. Physics Stepping: simulate_next_frames([B,34], A*2, dt) -> [B,A,34] (humans constant vel, robot kinematic limits)
5. Vectorized Look-ahead: batched_lookahead_Q([B,T,6,13], value_network, dt, gamma, ...) -> [B,A] (r_sa=0, reward handled by env)
"""

import numpy as np
import torch
from typing import Union, Tuple, List, Optional, Callable
import logging
import configparser
import os

# CUDA tensor护栏
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def _to_numpy(x):
    """将任何张量安全转换为numpy数组"""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _to_tensor(x, dtype=None):
    """将numpy数组安全转换为torch tensor"""
    if dtype is None:
        dtype = torch.float32
    if _HAS_TORCH:
        if isinstance(x, torch.Tensor):
            return x.to(dtype)
        return torch.tensor(x, dtype=dtype)
    return x

# 默认数据类型
DEFAULT_DTYPE = torch.float32

# 从配置读取动作网格参数
def _load_grid_from_config():
    """从配置文件加载动作网格参数"""
    # 默认参数
    default_grid = {
        'n_speeds': 6,
        'n_headings': 16,
        'v_min': 0.0,
        'v_max': 1.0,
        'sampling': 'even',
        'include_stop': False,
        'stop_eps': 0.05
    }

    # 尝试从配置文件读取
    config_paths = [
        'configs/env.config',
        'crowd_nav/configs/env.config',
        '../configs/env.config'
    ]

    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                config = configparser.ConfigParser()
                config.read(config_path)

                if 'policy' in config:
                    policy_section = config['policy']
                    grid = {
                        'n_speeds': policy_section.getint('n_speeds', default_grid['n_speeds']),
                        'n_headings': policy_section.getint('n_headings', default_grid['n_headings']),
                        'v_min': policy_section.getfloat('v_min', default_grid['v_min']),
                        'v_max': policy_section.getfloat('v_max', default_grid['v_max']),
                        'sampling': policy_section.get('sampling', default_grid['sampling']),
                        'include_stop': policy_section.getboolean('include_stop', default_grid['include_stop']),
                        'stop_eps': policy_section.getfloat('stop_eps', default_grid['stop_eps'])
                    }

                    # 动作表熔断断言
                    v_pref = 1.0  # 机器人首选速度
                    vmax = grid['v_max']
                    assert vmax >= 0.8 * v_pref, f"[ACTIONS] vmax({vmax}) too small vs v_pref({v_pref})"

                    logging.info(f"[CONTRACTS] Loaded grid from {config_path}: {grid}")
                    return grid

            except Exception as e:
                logging.warning(f"[CONTRACTS] Failed to load config from {config_path}: {e}")
                continue

    logging.info(f"[CONTRACTS] Using default grid: {default_grid}")
    return default_grid

# 全局动作网格配置
GRID = _load_grid_from_config()


def init_grid_from_cfg(cfg):
    """
    从运行时cfg显式初始化全局GRID（确保训练时IL/RL使用同一套网格）

    Args:
        cfg: ConfigParser对象（train.py传入的运行时配置）
    """
    global GRID

    # 从cfg读取，允许fallback到当前GRID值
    n_speeds  = cfg.getint('policy', 'n_speeds',  fallback=GRID.get('n_speeds', 5))
    n_heads   = cfg.getint('policy', 'n_headings', fallback=GRID.get('n_headings', 16))
    v_min     = cfg.getfloat('policy', 'v_min',    fallback=GRID.get('v_min', 0.3))
    v_max     = cfg.getfloat('policy', 'v_max',    fallback=GRID.get('v_max', 1.0))
    sampling  = cfg.get('policy', 'sampling',      fallback=GRID.get('sampling', 'even'))
    include_stop = cfg.getboolean('policy', 'include_stop', fallback=GRID.get('include_stop', False))
    stop_eps = cfg.getfloat('policy', 'stop_eps', fallback=GRID.get('stop_eps', 0.05))

    GRID.update(dict(
        n_speeds=n_speeds,
        n_headings=n_heads,
        v_min=v_min,
        v_max=v_max,
        sampling=sampling,
        include_stop=include_stop,
        stop_eps=stop_eps
    ))

    # 保险栅：确保动作表合法
    v_pref = 1.0
    assert v_max >= 0.8 * v_pref, f"[ACTIONS] vmax({v_max}) too small vs v_pref({v_pref})"

    logging.info(f"[CONTRACTS] GRID initialized from runtime cfg: {GRID}")
    return GRID


def discrete_index_to_action(action_idx: int, **grid_params) -> Tuple[float, float]:
    """
    离散动作索引 → (vx, vy) 动作向量

    Args:
        action_idx: 动作索引 [0, n_speeds * n_headings)
        **grid_params: 网格参数（如果不提供则使用全局GRID）

    Returns:
        Tuple[float, float]: (vx, vy) 动作向量
    """
    if grid_params:
        grid = grid_params
    else:
        grid = GRID

    n_speeds = grid['n_speeds']
    n_headings = grid['n_headings']
    v_min = grid['v_min']
    v_max = grid['v_max']
    sampling = str(grid.get('sampling', 'even')).strip().lower()
    include_stop = bool(grid.get('include_stop', False))

    # 检查索引范围
    total_actions = n_speeds * n_headings + (1 if include_stop else 0)
    if not (0 <= action_idx < total_actions):
        raise ValueError(f"Action index {action_idx} out of range [0, {total_actions})")

    if include_stop:
        if action_idx == 0:
            return 0.0, 0.0
        action_idx = action_idx - 1

    # 解析速度和朝向索引
    speed_idx = action_idx % n_speeds
    heading_idx = action_idx // n_speeds

    # todo.md致命修复：统一动作网格映射，消灭硬编码后门
    # 之前n_speeds==5时硬编码[0.0, 0.25, 0.5, 0.75, 1.0]，导致：
    # - 即使v_min=0.3，仍会输出speed=0.0的动作→机器人停止不动→超时暴涨
    # - IL量化用v_min=0.3，RL执行用v_min=0.0→分布错配
    # 现在统一使用v_min/v_max均匀插值（与action_to_discrete_index口径一致）

    if n_speeds == 1:
        speed = v_max
    else:
        if sampling == 'exponential':
            # SARL-style exponential sampling
            speed = (np.exp((speed_idx + 1) / n_speeds) - 1) / (np.e - 1) * v_max
        else:
            # Even sampling (default)
            speed = v_min + (v_max - v_min) * speed_idx / (n_speeds - 1)

    # 计算朝向角度
    if n_headings == 1:
        angle = 0.0
    else:
        angle = 2 * np.pi * heading_idx / n_headings

    # 转换为笛卡尔坐标
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)

    # debug.md P4修复：允许零速动作（真停/慢速档）
    action_speed = np.sqrt(vx*vx + vy*vy)
    # 移除零速断言，允许停止和慢速动作以防超时

    return float(vx), float(vy)


def action_to_discrete_index(vx: float, vy: float, grid=None, **grid_params) -> int:
    """
    (vx, vy) 动作向量 → 离散动作索引

    Args:
        vx, vy: 动作向量分量
        grid: 网格参数字典（新接口）
        **grid_params: 网格参数（旧接口）

    Returns:
        int: 最接近的动作索引
    """
    if grid is not None:
        grid_config = grid
    elif grid_params:
        grid_config = grid_params
    else:
        grid_config = GRID

    n_speeds = grid_config['n_speeds']
    n_headings = grid_config['n_headings']
    v_min = grid_config['v_min']
    v_max = grid_config['v_max']
    sampling = str(grid_config.get('sampling', 'even')).strip().lower()
    include_stop = bool(grid_config.get('include_stop', False))
    stop_eps = float(grid_config.get('stop_eps', 0.05))

    # 计算速度和角度
    speed = np.sqrt(vx**2 + vy**2)
    angle = np.arctan2(vy, vx)
    if angle < 0:
        angle += 2 * np.pi

    # Stop action (if enabled)
    if include_stop and speed <= stop_eps:
        return 0

    # 量化速度
    if n_speeds == 1:
        speed_idx = 0
    else:
        if sampling == 'exponential':
            speeds = [(np.exp((i + 1) / n_speeds) - 1) / (np.e - 1) * v_max for i in range(n_speeds)]
            speed_idx = int(np.argmin([abs(speed - s) for s in speeds]))
        else:
            normalized_speed = (speed - v_min) / (v_max - v_min) if v_max > v_min else 0.0
            normalized_speed = np.clip(normalized_speed, 0, 1)
            speed_idx = int(round(normalized_speed * (n_speeds - 1)))

    # 量化朝向
    if n_headings == 1:
        heading_idx = 0
    else:
        normalized_angle = angle / (2 * np.pi)
        heading_idx = int(round(normalized_angle * n_headings)) % n_headings

    # 计算索引
    action_idx = heading_idx * n_speeds + speed_idx
    if include_stop:
        action_idx += 1
    return action_idx


def grid_action_dim(grid=None) -> int:
    """Return total number of discrete actions for a grid."""
    g = GRID if grid is None else grid
    base = int(g.get('n_speeds', 0)) * int(g.get('n_headings', 0))
    if bool(g.get('include_stop', False)):
        return base + 1
    return base


def ensure_tensor(data, dtype=None, device=None):
    """
    确保数据是PyTorch张量

    Args:
        data: 输入数据
        dtype: 目标数据类型
        device: 目标设备

    Returns:
        torch.Tensor: 转换后的张量
    """
    if dtype is None:
        dtype = DEFAULT_DTYPE

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)

    # 转换数据类型
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)

    # 移动到设备
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def validate_state_shape(state, expected_shape):
    """
    验证状态形状

    Args:
        state: 状态数据
        expected_shape: 期望形状

    Raises:
        ValueError: 如果形状不匹配
    """
    if hasattr(state, 'shape'):
        actual_shape = state.shape
    elif hasattr(state, '__len__'):
        actual_shape = (len(state),)
    else:
        raise ValueError(f"Cannot determine shape of state: {type(state)}")

    if isinstance(expected_shape, int):
        expected_shape = (expected_shape,)

    if actual_shape != expected_shape:
        raise ValueError(f"State shape mismatch: got {actual_shape}, expected {expected_shape}")


def joint34_to_tokens(state_34d):
    """
    34D联合状态 → 6×13 tokens格式

    Args:
        state_34d: 34维联合状态或批次 [px,py,vx,vy,radius,v_pref,gx,gy,theta,sin_theta,cos_theta] + 5×[px,py,vx,vy,radius]
                   支持形状: [34] 或 [B, 34]
                   也支持JointState对象（自动调用to_array()）

    Returns:
        np.ndarray: [8, 13] tokens 或 [B, 8, 13] batch tokens
    """
    # 根源修复：统一处理JointState对象
    # 如果输入是JointState（或任何有to_array方法的对象），先转换
    if hasattr(state_34d, 'to_array') and callable(getattr(state_34d, 'to_array')):
        state_34d = state_34d.to_array()

    # 如果是list of JointState，批量转换
    elif isinstance(state_34d, (list, tuple)) and len(state_34d) > 0:
        if hasattr(state_34d[0], 'to_array') and callable(getattr(state_34d[0], 'to_array')):
            state_34d = np.array([s.to_array() for s in state_34d], dtype=np.float32)

    state_34d = _to_numpy(state_34d)
    state_34d = np.array(state_34d, dtype=np.float32)

    # 检查是否为批次数据
    if state_34d.ndim == 2:
        # 批次处理 [B, 34] - 使用向量化版本
        B, state_dim = state_34d.shape

        # 支持课程学习的动态维度：19D/24D/29D/34D → 34D
        if state_dim != 34:
            if state_dim < 34:
                # 填充到34维
                state_34d = np.pad(state_34d, ((0, 0), (0, 34 - state_dim)), 'constant', constant_values=0.0)
            else:
                # 截断到34维
                state_34d = state_34d[:, :34]

        # 使用向量化批量转换（10-50x加速）
        batch_array = _batch_joint34_to_tokens_vectorized(state_34d)  # [B, 8, 13]
        batch_array = np.expand_dims(batch_array, axis=1)  # [B, 1, 8, 13] - 添加时间维度
        return ensure_tensor(batch_array)  # 转换为torch张量

    else:
        # 单个状态处理 [34]
        # 支持课程学习的动态维度：19D/24D/29D/34D → 34D
        if len(state_34d) != 34:
            if len(state_34d) < 34:
                # 填充到34维
                state_34d = np.pad(state_34d, (0, 34 - len(state_34d)), 'constant', constant_values=0.0)
            else:
                # 截断到34维
                state_34d = state_34d[:34]

        return _single_joint34_to_tokens(state_34d)


def _batch_joint34_to_tokens_vectorized(state_batch):
    """
    批量向量化转换：[B, 34] → [B, 8, 13]
    完全使用numpy向量化操作，避免Python循环，10-50x加速
    """
    B = state_batch.shape[0]
    tokens_batch = np.zeros((B, 8, 13), dtype=np.float32)

    # 提取机器人状态 [B, 9]
    robot_px = state_batch[:, 0]
    robot_py = state_batch[:, 1]
    robot_vx = state_batch[:, 2]
    robot_vy = state_batch[:, 3]
    robot_radius = state_batch[:, 4]
    robot_v_pref = state_batch[:, 5]
    robot_gx = state_batch[:, 6]
    robot_gy = state_batch[:, 7]
    robot_theta = state_batch[:, 8]

    # Clamp v_pref to valid range [0, 3.0]
    robot_v_pref = np.clip(robot_v_pref, 0.0, 3.0)

    # 第一行：机器人基础状态 [B, 13]
    tokens_batch[:, 0, :9] = state_batch[:, :9]
    tokens_batch[:, 0, 9] = np.sqrt((robot_gx - robot_px)**2 + (robot_gy - robot_py)**2)  # goal_dist
    tokens_batch[:, 0, 10] = np.sqrt(robot_vx**2 + robot_vy**2)  # speed
    tokens_batch[:, 0, 11] = robot_v_pref
    tokens_batch[:, 0, 12] = 1.0  # robot flag

    # 第二行：相对目标信息
    goal_dx = robot_gx - robot_px
    goal_dy = robot_gy - robot_py
    goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)
    goal_angle = np.arctan2(goal_dy, goal_dx)

    tokens_batch[:, 1, 0] = goal_dx
    tokens_batch[:, 1, 1] = goal_dy
    tokens_batch[:, 1, 2] = goal_dist
    tokens_batch[:, 1, 3] = goal_angle
    tokens_batch[:, 1, 4] = np.cos(goal_angle)
    tokens_batch[:, 1, 5] = np.sin(goal_angle)

    # 第三行：机器人运动信息
    robot_speed = np.sqrt(robot_vx**2 + robot_vy**2)
    motion_angle = np.arctan2(robot_vy, robot_vx)
    motion_mask = robot_speed > 1e-6

    tokens_batch[:, 2, 0] = np.where(motion_mask, robot_speed, 0.0)
    tokens_batch[:, 2, 1] = np.where(motion_mask, motion_angle, 0.0)
    tokens_batch[:, 2, 2] = np.where(motion_mask, np.cos(motion_angle), 0.0)
    tokens_batch[:, 2, 3] = np.where(motion_mask, np.sin(motion_angle), 0.0)

    # 解析人类状态 [B, 5, 5]
    humans_batch = state_batch[:, 9:34].reshape(B, 5, 5)

    # 处理每个样本的人类（需要排序，无法完全向量化，但优化了计算）
    for b in range(B):
        humans = humans_batch[b]  # [5, 5]

        # 过滤有效人类（非零）
        valid_mask = np.any(humans != 0, axis=1)
        valid_humans = humans[valid_mask]

        if len(valid_humans) == 0:
            continue

        # 计算TTC并排序
        hx = valid_humans[:, 0]
        hy = valid_humans[:, 1]
        hvx = valid_humans[:, 2]
        hvy = valid_humans[:, 3]
        rel_x = hx - robot_px[b]
        rel_y = hy - robot_py[b]
        rel_vx = hvx - robot_vx[b]
        rel_vy = hvy - robot_vy[b]
        distances = np.sqrt(rel_x**2 + rel_y**2 + 1e-6)
        closing = -(rel_x * rel_vx + rel_y * rel_vy) / (distances + 1e-6)
        # 修复：静止的人按距离排序，不会被忽略
        ttc = np.where(closing > 0.1, distances / (closing + 1e-6), distances * 10)
        sorted_indices = np.argsort(ttc)

        # 取前5个最危险的人类
        for i, idx in enumerate(sorted_indices[:5]):
            row = 3 + i
            human = valid_humans[idx]
            dist = distances[idx]

            hx, hy, hvx, hvy, hradius = human
            rel_x = hx - robot_px[b]
            rel_y = hy - robot_py[b]

            tokens_batch[b, row, 0] = rel_x
            tokens_batch[b, row, 1] = rel_y
            tokens_batch[b, row, 2] = dist
            tokens_batch[b, row, 3] = hvx
            tokens_batch[b, row, 4] = hvy
            tokens_batch[b, row, 5] = np.sqrt(hvx**2 + hvy**2)
            tokens_batch[b, row, 6] = hradius

            # TTC计算
            if dist > 1e-6:
                rel_vx = hvx - robot_vx[b]
                rel_vy = hvy - robot_vy[b]
                closing_speed = -(rel_x * rel_vx + rel_y * rel_vy) / dist
                if closing_speed > 1e-6:
                    ttc = dist / closing_speed
                    ttc_inv = 1.0 / max(ttc, 0.1)
                else:
                    ttc_inv = 0.0
            else:
                ttc_inv = 10.0

            tokens_batch[b, row, 7] = ttc_inv
            tokens_batch[b, row, 8] = 1.0 if dist < 2.0 else 0.0  # social zone

    return tokens_batch


def _single_joint34_to_tokens(state_34d):
    """单个34D状态 → 6×13 tokens的具体实现"""
    # 解析机器人状态（前11维）
    robot_state = state_34d[:11]
    px, py, vx, vy, radius, v_pref, gx, gy, theta = robot_state[:9]

    # 防呆检查（允许0值padding，但不允许异常值）
    if not (0.0 <= v_pref <= 3.0):
        import logging
        logging.warning(f"[CONTRACTS] v_pref out of range: {v_pref}, clamping to [0, 3.0]")
        v_pref = max(0.0, min(3.0, v_pref))

    # 解析人类状态（5×5=25维）
    humans_state = state_34d[9:34].reshape(5, 5)  # [human_id, [px,py,vx,vy,radius]]

    # 构建8×13 tokens
    tokens = np.zeros((8, 13), dtype=np.float32)

    # 第一行：机器人基础状态
    tokens[0, :9] = robot_state[:9]
    tokens[0, 9] = np.sqrt((gx - px)**2 + (gy - py)**2)  # 到目标距离
    tokens[0, 10] = np.sqrt(vx**2 + vy**2)  # 当前速度
    tokens[0, 11] = v_pref  # 首选速度
    tokens[0, 12] = 1.0  # 机器人标识

    # 第二行：相对目标信息
    goal_dx = gx - px
    goal_dy = gy - py
    goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)
    goal_angle = np.arctan2(goal_dy, goal_dx)

    tokens[1, 0] = goal_dx
    tokens[1, 1] = goal_dy
    tokens[1, 2] = goal_dist
    tokens[1, 3] = goal_angle
    tokens[1, 4] = np.cos(goal_angle)
    tokens[1, 5] = np.sin(goal_angle)

    # 第三行：机器人运动信息
    speed = np.sqrt(vx**2 + vy**2)
    if speed > 1e-6:
        motion_angle = np.arctan2(vy, vx)
        tokens[2, 0] = speed
        tokens[2, 1] = motion_angle
        tokens[2, 2] = np.cos(motion_angle)
        tokens[2, 3] = np.sin(motion_angle)

    # 第四到八行：人类信息（取前5个最近的人类）
    if len(humans_state) > 0:
        # 计算到机器人的距离
        distances = []
        for i, human in enumerate(humans_state):
            if np.any(human != 0):  # 有效人类
                hx, hy = human[0], human[1]
                dist = np.sqrt((px - hx)**2 + (py - hy)**2)
                distances.append((dist, i, human))

        # 按距离排序，取前5个
        distances.sort(key=lambda x: x[0])

        for row_idx, (dist, human_idx, human) in enumerate(distances[:5]):
            row = 3 + row_idx
            hx, hy, hvx, hvy, hradius = human

            # 相对位置
            rel_x = hx - px
            rel_y = hy - py
            rel_dist = np.sqrt(rel_x**2 + rel_y**2)

            tokens[row, 0] = rel_x
            tokens[row, 1] = rel_y
            tokens[row, 2] = rel_dist
            tokens[row, 3] = hvx
            tokens[row, 4] = hvy
            tokens[row, 5] = np.sqrt(hvx**2 + hvy**2)  # 人类速度
            tokens[row, 6] = hradius

            # TTC计算
            if rel_dist > 1e-6:
                rel_vx = hvx - vx
                rel_vy = hvy - vy
                closing_speed = -(rel_x * rel_vx + rel_y * rel_vy) / rel_dist
                if closing_speed > 1e-6:
                    ttc = rel_dist / closing_speed
                    ttc_inv = 1.0 / max(ttc, 0.1)  # TTC倒数，避免除零
                else:
                    ttc_inv = 0.0
            else:
                ttc_inv = 10.0  # 很近的情况

            tokens[row, 7] = ttc_inv

            # 社交区域指示
            in_social = 1.0 if rel_dist < 2.0 else 0.0
            tokens[row, 8] = in_social

    return tokens


def tokens_to_34d(tokens):
    """
    6×13 tokens → 34D联合状态

    Args:
        tokens: [8, 13] tokens格式状态

    Returns:
        np.ndarray: 34维联合状态
    """
    tokens = _to_numpy(tokens)
    tokens = np.array(tokens, dtype=np.float32)
    if tokens.shape != (8, 13):
        raise ValueError(f"Expected [8, 13] tokens, got {tokens.shape}")

    # 重构34D状态
    state_34d = np.zeros(34, dtype=np.float32)

    # 机器人状态（前9维）
    state_34d[:9] = tokens[0, :9]

    # 人类状态重构（简化版，从相对信息推测绝对位置）
    robot_px, robot_py = tokens[0, 0], tokens[0, 1]

    human_start_idx = 9
    for i in range(3):  # 前3个人类
        row = 3 + i
        if np.any(tokens[row] != 0):  # 有效人类
            rel_x = tokens[row, 0]
            rel_y = tokens[row, 1]
            hvx = tokens[row, 3]
            hvy = tokens[row, 4]
            hradius = tokens[row, 6]

            # 绝对位置
            hx = robot_px + rel_x
            hy = robot_py + rel_y

            # 填入34D状态
            human_idx = human_start_idx + i * 5
            if human_idx + 4 < 34:
                state_34d[human_idx:human_idx+5] = [hx, hy, hvx, hvy, hradius]

    return state_34d


def to_btnd(states_list, sequence_length=8):
    """
    训练入口"唯一 4D 断言" - 转换状态列表为 [B, T, 8, 13] 格式

    Args:
        states_list: 状态列表
        sequence_length: 序列长度

    Returns:
        torch.Tensor: [B, T, 8, 13] 格式张量
    """
    if not states_list:
        raise ValueError("Empty states list")

    # 如果输入已经是4D张量，直接验证并返回
    if isinstance(states_list, torch.Tensor) and states_list.ndim == 4:
        B, T, H, W = states_list.shape
        assert H == 8 and W == 13, f"Expected [B, T, 8, 13], got {states_list.shape}"
        return states_list

    # 转换为tokens格式
    batch_tokens = []

    for batch_item in states_list:
        if isinstance(batch_item, list):
            # 序列数据
            tokens_seq = []
            for state in batch_item:
                if hasattr(state, 'to_array'):
                    state_34d = state.to_array()
                else:
                    state_34d = np.array(state, dtype=np.float32)

                # 确保34维
                if len(state_34d) != 34:
                    if len(state_34d) < 34:
                        state_34d = np.pad(state_34d, (0, 34 - len(state_34d)), 'constant')
                    else:
                        state_34d = state_34d[:34]

                tokens = joint34_to_tokens(state_34d)
                tokens_seq.append(tokens)

            # 填充到序列长度
            while len(tokens_seq) < sequence_length:
                tokens_seq.append(tokens_seq[-1])  # 重复最后一个

            if len(tokens_seq) > sequence_length:
                tokens_seq = tokens_seq[-sequence_length:]  # 取最后sequence_length个

            batch_tokens.append(np.stack(tokens_seq, axis=0))
        else:
            # 单个状态，扩展为序列
            if hasattr(batch_item, 'to_array'):
                state_34d = batch_item.to_array()
            else:
                state_34d = np.array(batch_item, dtype=np.float32)

            # 确保34维
            if len(state_34d) != 34:
                if len(state_34d) < 34:
                    state_34d = np.pad(state_34d, (0, 34 - len(state_34d)), 'constant')
                else:
                    state_34d = state_34d[:34]

            tokens = joint34_to_tokens(state_34d)
            # 重复为序列
            tokens_seq = [tokens] * sequence_length
            batch_tokens.append(np.stack(tokens_seq, axis=0))

    # 转换为张量
    batch_array = np.stack(batch_tokens, axis=0)  # [B, T, 8, 13]
    return ensure_tensor(batch_array)


def assert_4d(tensor):
    """4D张量断言"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")
    B, T, H, W = tensor.shape
    if H != 8 or W != 13:
        raise ValueError(f"Expected [B, T, 8, 13] shape, got {tensor.shape}")


def pick_last(tensor):
    """从[B, T, ...]中提取最后时刻 → [B, ...]"""
    if tensor.ndim < 2:
        raise ValueError(f"Need at least 2D tensor, got {tensor.ndim}D")
    return tensor[:, -1]


def simulate_next_frames(states_34d, actions, dt):
    """
    物理步进：模拟下一帧状态

    Args:
        states_34d: [B, 34] 当前状态
        actions: [B, A, 2] 所有动作 或 [B, 2] 单个动作
        dt: 时间步长

    Returns:
        torch.Tensor: [B, A, 34] 下一状态 或 [B, 34] 单个下一状态
    """
    states_34d = ensure_tensor(states_34d)
    actions = ensure_tensor(actions)

    if states_34d.ndim != 2 or states_34d.shape[1] != 34:
        raise ValueError(f"Expected [B, 34] states, got {states_34d.shape}")

    B = states_34d.shape[0]

    if actions.ndim == 2:
        # 单个动作 [B, 2]
        A = 1
        actions = actions.unsqueeze(1)  # [B, 1, 2]
    elif actions.ndim == 3:
        # 多个动作 [B, A, 2]
        A = actions.shape[1]
    else:
        raise ValueError(f"Expected actions shape [B, 2] or [B, A, 2], got {actions.shape}")

    # 扩展状态到所有动作
    next_states = states_34d.unsqueeze(1).repeat(1, A, 1)  # [B, A, 34]

    for b in range(B):
        for a in range(A):
            current_state = states_34d[b].clone()  # [34]
            action = actions[b, a]  # [2]

            # 解析状态
            px, py, vx, vy = current_state[:4]
            v_pref = current_state[5]

            # 应用动作（限速）
            new_vx, new_vy = action[0], action[1]
            action_speed = torch.sqrt(new_vx**2 + new_vy**2)

            if action_speed > v_pref:
                scale = v_pref / action_speed
                new_vx *= scale
                new_vy *= scale

            # 更新位置和速度
            new_px = px + new_vx * dt
            new_py = py + new_vy * dt

            next_state = current_state.clone()
            next_state[0] = new_px
            next_state[1] = new_py
            next_state[2] = new_vx
            next_state[3] = new_vy

            # 更新人类状态（简单的常速运动）
            for h in range(5):
                h_start = 9 + h * 5
                if h_start + 4 <= 34:  # 确保不越界
                    # 安全地提取人类状态
                    human_slice = next_state[h_start:h_start+4]
                    if human_slice.numel() == 4:  # 确保有4个元素
                        hx, hy, hvx, hvy = human_slice
                        if torch.abs(hx) > 1e-6 or torch.abs(hy) > 1e-6:  # 有效人类
                            next_state[h_start] = hx + hvx * dt
                            next_state[h_start + 1] = hy + hvy * dt

            next_states[b, a] = next_state

    if A == 1:
        return next_states.squeeze(1)  # [B, 34]
    else:
        return next_states  # [B, A, 34]


def batched_lookahead_Q(joint_tokens, value_network, dt, gamma=0.95, reward_fn=None):
    """
    向量化look-ahead：计算所有动作的Q值

    Args:
        joint_tokens: [B, T, 8, 13] 当前状态序列
        value_network: 价值网络，输入[B, T, 8, 13] → 输出[B]
        dt: 时间步长
        gamma: 折扣因子
        reward_fn: 奖励函数（可选，默认r_sa=0）

    Returns:
        torch.Tensor: [B, A] Q值
    """
    assert_4d(joint_tokens)
    B, T = joint_tokens.shape[:2]

    # 获取当前状态（最后时刻）
    current_tokens = pick_last(joint_tokens)  # [B, 8, 13]
    current_states_34d = torch.stack([
        ensure_tensor(tokens_to_34d(current_tokens[b].cpu().numpy()))
        for b in range(B)
    ], dim=0)  # [B, 34]

    # 生成所有动作
    total_actions = GRID['n_speeds'] * GRID['n_headings']
    all_actions = []
    for a_idx in range(total_actions):
        vx, vy = discrete_index_to_action(a_idx)
        all_actions.append([vx, vy])

    all_actions = ensure_tensor(all_actions, device=joint_tokens.device)  # [A, 2]
    all_actions = all_actions.unsqueeze(0).repeat(B, 1, 1)  # [B, A, 2]

    # 物理步进
    next_states_34d = simulate_next_frames(current_states_34d, all_actions, dt)  # [B, A, 34]

    # 转换为tokens并计算价值
    q_values = torch.zeros(B, total_actions, device=joint_tokens.device)

    for b in range(B):
        for a in range(total_actions):
            # 即时奖励（默认为0，奖励由环境负责）
            if reward_fn is not None:
                reward = reward_fn(current_states_34d[b], all_actions[b, a])
            else:
                reward = 0.0

            # 下一状态价值
            next_state_34d = next_states_34d[b, a].cpu().numpy()
            next_tokens = joint34_to_tokens(next_state_34d)

            # 构建下一状态序列（简化：重复当前序列并替换最后帧）
            next_sequence = joint_tokens[b:b+1].clone()  # [1, T, 8, 13]
            next_sequence[0, -1] = ensure_tensor(next_tokens, device=joint_tokens.device)

            next_value = value_network(next_sequence).item()

            # Q值计算
            q_values[b, a] = reward + gamma * next_value

    return q_values


def immediate_reward(event, d_curr, rc, dt, extras=None):
    """
    统一即时奖励函数 - MSR进度奖励的唯一入口
    debug.md B项修复：统一并限幅进度奖励(MSR)

    Args:
        event: 事件类型 ('success', 'collision', 'timeout', 'normal')
        d_curr: 当前到目标距离
        rc: 舒适距离
        dt: 时间步长
        extras: 额外信息字典 {d_prev, v_pref, lambda_p, use_msr}

    Returns:
        float: 即时奖励值
    """
    r = 0.0

    # 基础奖励结构（SARL标准）
    if event == 'success':
        r = 1.0
    elif event == 'collision':
        r = -0.25
    elif event == 'timeout':
        r = -0.5
    elif event == 'normal':
        # 舒适区线性奖励（可选）
        if d_curr < rc:
            r = (d_curr - rc) * 0.1  # 轻微不适惩罚

    # debug.md单点修复：彻底禁用MSR进度奖励，对齐SARL invisible模式
    if extras is not None and extras.get('use_msr', False):  # 默认禁用MSR
        d_prev = extras.get('d_prev', None)
        v_pref = float(extras.get('v_pref', 1.0) or 1.0)
        lambda_p = float(extras.get('lambda_p', 0.3))  # debug.md E修复：提升λ=0.1→0.3

        if d_prev is not None and d_curr is not None:
            # debug.md B1: 每步可走的最大距离
            max_step = v_pref * dt          # 例如 1.0 * 0.25 = 0.25m
            raw_prog = (d_prev - d_curr) / max_step
            prog = float(max(-1.0, min(1.0, raw_prog)))  # debug.md B1: clip 到 [-1,1]
            msr = lambda_p * prog           # debug.md B1: 建议 lambda_p = 0.3 起步
            r += msr

            # debug.md B1: 每步时间惩罚（time cost）
            r += -0.002   # 每步 -0.002，对 100 步就是 -0.2，不会盖过 +1 成功，但能惩罚发呆

            # debug: 统计MSR生效情况（节流日志）
            if hasattr(immediate_reward, '_msr_count'):
                immediate_reward._msr_count += 1
            else:
                immediate_reward._msr_count = 1

            # debug.md E项：日志节流 - 每500次记录一次改为每1000次
            if immediate_reward._msr_count % 1000 == 0:
                import logging
                logging.info(f"[MSR-FIXED] prog_raw={raw_prog:.3f}, prog_clipped={prog:.3f}, msr={msr:.6f}, λ={lambda_p}")

    return r


# 兼容性别名
ensure_btnd = to_btnd

# 导出主要接口
__all__ = [
    'GRID', 'DEFAULT_DTYPE',
    'discrete_index_to_action', 'action_to_discrete_index',
    'ensure_tensor', 'validate_state_shape',
    'joint34_to_tokens', 'tokens_to_34d',
    'to_btnd', 'ensure_btnd', 'assert_4d', 'pick_last',
    'simulate_next_frames', 'batched_lookahead_Q', 'immediate_reward',
    'select_entities_for_tokens', 'select_belief_entities_for_tokens',
    'select_entities_by_contract', 'entities_to_tokens',
    'token_shape_for_contract',
]


# ---------------------------------------------------------------------------
# Order 17 item 9: occlusion-aware token construction.
#
# The legacy converter above is left untouched: with occlusion off it is still
# the only path, so the pre-occlusion behaviour stays bit-identical. This
# function is used only when an occlusion arm is active.
#
# Human rows keep columns 0-8 with exactly the paper-1 semantics. The four
# columns that were always zero become:
#     9   existence probability      1.0 for a seen pedestrian, the posterior
#                                    mass for a believed one
#     10  normalised uncertainty     0.0 when seen
#     11  visible flag
#     12  hidden-belief flag
# The encoder already concatenates all 13 columns (mamba_rl.py builds a 21-D
# per-human feature from token(13) + relation(8)), so nothing in the network
# changes shape. Because those four inputs were always zero, their weights in a
# pre-occlusion checkpoint never received gradient and sit at their random
# initialisation -- which is why warm-start must zero them explicitly.
# ---------------------------------------------------------------------------
def select_entities_for_tokens(robot_state, entities, max_humans=5):
    """Select entities with the exact ordering used by the legacy 34-D path."""
    import numpy as _np
    px, py = float(robot_state.px), float(robot_state.py)
    vx, vy = float(robot_state.vx), float(robot_state.vy)
    ranked = []
    for order, entity in enumerate(entities or []):
        rx = float(entity["px"]) - px
        ry = float(entity["py"]) - py
        distance = _np.sqrt(rx ** 2 + ry ** 2 + 1e-6)
        rvx = float(entity["vx"]) - vx
        rvy = float(entity["vy"]) - vy
        closing = -(rx * rvx + ry * rvy) / (distance + 1e-6)
        ttc = (distance / (closing + 1e-6)
               if closing > 0.1 else distance * 10.0)
        ranked.append((ttc, order, entity))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:max_humans]]


BELIEF_FEATURE_MODES = ("full", "fixed_confidence")

LEGACY_TOKEN_CONTRACT = "legacy_top5"
BELIEF_TOKEN_CONTRACT = "belief_v3"
TOKEN_CONTRACTS = (LEGACY_TOKEN_CONTRACT, BELIEF_TOKEN_CONTRACT)


def token_shape_for_contract(token_contract=LEGACY_TOKEN_CONTRACT,
                             visible_slots=5, hidden_slots=10):
    """Return the complete token shape, including the three robot rows."""
    if token_contract == LEGACY_TOKEN_CONTRACT:
        return (8, 13)
    if token_contract != BELIEF_TOKEN_CONTRACT:
        raise ValueError(
            f"token_contract={token_contract!r}; expected one of {TOKEN_CONTRACTS}")
    visible_slots, hidden_slots = int(visible_slots), int(hidden_slots)
    if visible_slots < 0 or hidden_slots < 0 or visible_slots + hidden_slots < 1:
        raise ValueError("belief_v3 requires at least one non-negative entity slot")
    return (3 + visible_slots + hidden_slots, 13)


def _entity_clearance(robot_state, entity):
    import numpy as _np
    return (_np.hypot(float(entity["px"]) - float(robot_state.px),
                      float(entity["py"]) - float(robot_state.py)) -
            float(getattr(robot_state, "radius", 0.0)) -
            float(entity.get("radius", 0.0)))


def select_belief_entities_for_tokens(robot_state, entities,
                                      visible_slots=5, hidden_slots=10):
    """The independently validated belief-v3 selection contract.

    Visible entities retain the legacy TTC ordering. Hidden modes are ranked by
    posterior existence probability, then clearance, and the unused capacity of
    either group is backfilled from the other. This is the production form of
    ``cap_10__slots_15__visible5_hidden_confidence``.
    """
    visible, hidden = [], []
    for order, entity in enumerate(entities or []):
        item = (order, entity)
        if float(entity.get("hidden", 0.0)) > 0.5:
            hidden.append(item)
        else:
            visible.append(item)

    visible_ranked = select_entities_for_tokens(
        robot_state, [entity for _, entity in visible],
        max_humans=len(visible))
    hidden.sort(key=lambda item: (
        -float(item[1].get("p_exist", 1.0)),
        _entity_clearance(robot_state, item[1]), item[0]))
    hidden_ranked = [entity for _, entity in hidden]

    visible_slots, hidden_slots = int(visible_slots), int(hidden_slots)
    total_slots = visible_slots + hidden_slots
    visible_take = min(len(visible_ranked), visible_slots)
    hidden_take = min(
        len(hidden_ranked), hidden_slots + (visible_slots - visible_take))
    selected = visible_ranked[:visible_take] + hidden_ranked[:hidden_take]
    if len(selected) < total_slots:
        selected.extend(
            visible_ranked[visible_take:visible_take + total_slots - len(selected)])
    return selected[:total_slots]


def select_entities_by_contract(robot_state, entities,
                                token_contract=LEGACY_TOKEN_CONTRACT,
                                visible_slots=5, hidden_slots=10):
    if token_contract == LEGACY_TOKEN_CONTRACT:
        return select_entities_for_tokens(robot_state, entities, max_humans=5)
    if token_contract == BELIEF_TOKEN_CONTRACT:
        return select_belief_entities_for_tokens(
            robot_state, entities, visible_slots, hidden_slots)
    raise ValueError(
        f"token_contract={token_contract!r}; expected one of {TOKEN_CONTRACTS}")


def entities_to_tokens(robot_state, entities, max_humans=5, preselected=False,
                       belief_features="full",
                       token_contract=LEGACY_TOKEN_CONTRACT,
                       visible_slots=5, hidden_slots=10):
    """Build a token block from visible pedestrians and/or belief modes.

    ``legacy_top5`` executes the historical implementation exactly.
    ``belief_v3`` uses the validated 15-entity contract while preserving each
    row's legacy numeric semantics in columns 0-8.
    """
    import numpy as _np
    if belief_features not in BELIEF_FEATURE_MODES:
        raise ValueError(
            f"belief_features={belief_features!r}; expected one of "
            f"{BELIEF_FEATURE_MODES}")
    if token_contract not in TOKEN_CONTRACTS:
        raise ValueError(
            f"token_contract={token_contract!r}; expected one of {TOKEN_CONTRACTS}")
    if token_contract == LEGACY_TOKEN_CONTRACT:
        selected = (list(entities)[:max_humans] if preselected else
                    select_entities_for_tokens(robot_state, entities, max_humans))
    else:
        selected = (list(entities)[:visible_slots + hidden_slots]
                    if preselected else select_belief_entities_for_tokens(
                        robot_state, entities, visible_slots, hidden_slots))

    # Build the same legacy 34-D array and call the same converter. This is
    # intentional: the pretrained model depends on historical field-order and
    # floating-point quirks in that path, including robot-derived columns.
    robot_arr = _np.array([
        robot_state.px, robot_state.py, robot_state.vx, robot_state.vy,
        robot_state.radius, robot_state.gx, robot_state.gy,
        robot_state.v_pref, robot_state.theta,
    ], dtype=_np.float32)
    if token_contract == LEGACY_TOKEN_CONTRACT:
        human_arr = []
        for entity in selected:
            human_arr.extend([
                entity["px"], entity["py"], entity["vx"], entity["vy"],
                entity["radius"],
            ])
        while len(human_arr) < 5 * 5:
            human_arr.append(0.0)
        state_34 = _np.concatenate([robot_arr, human_arr[:5 * 5]])
        t = _batch_joint34_to_tokens_vectorized(state_34.reshape(1, -1))[0]
    else:
        shape = token_shape_for_contract(
            token_contract, visible_slots, hidden_slots)
        empty_34 = _np.concatenate(
            [robot_arr, _np.zeros(25, dtype=_np.float32)])
        legacy_empty = _batch_joint34_to_tokens_vectorized(
            empty_34.reshape(1, -1))[0]
        t = _np.zeros(shape, dtype=legacy_empty.dtype)
        t[:3] = legacy_empty[:3]
        # Reuse the legacy converter one entity at a time. This avoids a second
        # implementation of its historical floating-point and TTC quirks.
        for row, entity in enumerate(selected, start=3):
            human = _np.array([
                entity["px"], entity["py"], entity["vx"], entity["vy"],
                entity["radius"],
            ], dtype=_np.float32)
            one_34 = _np.concatenate(
                [robot_arr, human, _np.zeros(20, dtype=_np.float32)])
            t[row] = _batch_joint34_to_tokens_vectorized(
                one_34.reshape(1, -1))[0, 3]

    # Only the four historically unused human columns are new.
    for row, e in enumerate(selected, start=3):
        # The paired control keeps exactly the same entity positions and
        # visible/hidden status, while replacing posterior confidence with a
        # deterministic point estimate. Thus only columns 9-10 differ.
        if belief_features == "fixed_confidence":
            t[row, 9] = 1.0
            t[row, 10] = 0.0
        else:
            t[row, 9] = float(e.get("p_exist", 1.0))
            t[row, 10] = float(e.get("uncertainty", 0.0))
        t[row, 11] = float(e.get("visible", 1.0))
        t[row, 12] = float(e.get("hidden", 0.0))
    return t
