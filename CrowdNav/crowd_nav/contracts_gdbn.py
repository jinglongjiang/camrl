# -*- coding: utf-8 -*-
"""
P2/P3 Contracts - SSOT (Single Source of Truth) for Actions, States, and Physics

Core functions:
1. Action Grid: Reads from config [policy], with fuse assertion (v_max >= 0.8*v_pref)
2. Legacy Interface: ensure_tensor / DEFAULT_DTYPE / validate_state_shape (compatible with other imports)
3. State Conversion: tokens_to_34d, joint34_to_tokens (including derived features like ttc_inv/in_soc)
4. Physics Stepping: simulate_next_frames([B,9+N*5], A*2, dt) -> [B,A,9+N*5] (humans constant vel, robot kinematic limits)
5. Vectorized Look-ahead: batched_lookahead_Q([B,T,6,13], value_network, dt, gamma, ...) -> [B,A] (r_sa=0, reward handled by env)
"""

import numpy as np
import torch
from typing import Union, Tuple, List, Optional, Callable
import logging
import configparser
import os

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def _to_numpy(x):
    """numpy"""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _to_tensor(x, dtype=None):
    """numpytorch tensor"""
    if dtype is None:
        dtype = torch.float32
    if _HAS_TORCH:
        if isinstance(x, torch.Tensor):
            return x.to(dtype)
        return torch.tensor(x, dtype=dtype)
    return x

DEFAULT_DTYPE = torch.float32

def _state_obj_to_array(state, expected_dim: int) -> np.ndarray:
    if hasattr(state, 'to_array') and callable(getattr(state, 'to_array')):
        arr = state.to_array()
    else:
        arr = np.asarray(state, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] < expected_dim:
        arr = np.pad(arr, (0, expected_dim - arr.shape[0]), mode='constant')
    return arr[:expected_dim]

def _human_selection_metrics(
    robot_arr: np.ndarray,
    humans_arr: np.ndarray,
    dt: float = 0.25,
    horizon: int = 5,
) -> dict:
    """Compute per-human geometry used by dynamic top-K selection."""
    humans_arr = np.asarray(humans_arr, dtype=np.float32)
    if humans_arr.size == 0:
        return {}
    robot_arr = np.asarray(robot_arr, dtype=np.float32).reshape(-1)
    rel = humans_arr[:, 0:2] - robot_arr[None, 0:2]
    rel_v = humans_arr[:, 2:4] - robot_arr[None, 2:4]
    robot_r = float(robot_arr[4]) if robot_arr.shape[0] > 4 and robot_arr[4] > 0 else 0.3
    human_r = np.where(humans_arr[:, 4] > 0, humans_arr[:, 4], 0.3)
    coll_r = robot_r + human_r

    dist = np.sqrt(np.sum(rel * rel, axis=1) + 1e-6)
    clearance = dist - coll_r
    closing = -np.sum(rel * rel_v, axis=1) / (dist + 1e-6)
    ttc = np.where(closing > 0.05, np.maximum(clearance, 0.0) / (closing + 1e-6), np.inf)

    hor = max(0, int(horizon))
    if hor > 0:
        steps = (np.arange(1, hor + 1, dtype=np.float32) * float(dt)).reshape(1, hor, 1)
        rel_future = rel[:, None, :] + rel_v[:, None, :] * steps
        future_dist = np.sqrt(np.sum(rel_future * rel_future, axis=2) + 1e-6)
        min_future_clearance = np.min(future_dist - coll_r[:, None], axis=1)
    else:
        min_future_clearance = clearance

    valid = np.any(humans_arr != 0, axis=1)
    return {
        'valid': valid,
        'dist': dist,
        'clearance': clearance,
        'closing': closing,
        'ttc': ttc,
        'min_future_clearance': min_future_clearance,
    }

def select_human_indices(
    robot_state,
    human_states,
    k: int,
    mode: str = 'threat',
    dt: float = 0.25,
    horizon: int = 5,
    ttc_weight: float = 2.0,
    clearance_weight: float = 1.0,
    distance_weight: float = 0.15,
    closing_weight: float = 0.25,
    k_nearest: Optional[int] = None,
    k_threat: Optional[int] = None,
) -> List[int]:
    """Select top-K humans from the full visible crowd.

    Modes:
      - nearest/distance: legacy nearest-K selection.
      - ttc: earliest time-to-collision, distance tie-break.
      - threat/dynamic: weighted TTC + projected clearance + distance score.
      - hybrid: nearest quota first, then highest-threat humans from the rest.
    """
    n = len(human_states or [])
    k = max(0, int(k))
    if n == 0 or k == 0:
        return []

    robot_arr = _state_obj_to_array(robot_state, 9)
    humans_arr = np.stack([_state_obj_to_array(h, 5) for h in human_states], axis=0)
    metrics = _human_selection_metrics(robot_arr, humans_arr, dt=dt, horizon=horizon)
    valid = metrics['valid']
    idx = np.arange(n)
    mode = str(mode or 'nearest').strip().lower()

    dist = np.where(valid, metrics['dist'], np.inf)
    if mode in ('nearest', 'distance', 'dist'):
        order = np.lexsort((idx, dist))
    elif mode in ('ttc', 'ttc_then_dist', 'time_to_collision'):
        ttc = np.where(valid, metrics['ttc'], np.inf)
        order = np.lexsort((idx, dist, ttc))
    elif mode in ('threat', 'dynamic', 'risk', 'risk_ttc', 'hybrid'):
        ttc_term = float(ttc_weight) / (np.clip(metrics['ttc'], 0.05, 20.0) + 0.25)
        clearance_term = float(clearance_weight) / (np.clip(metrics['min_future_clearance'], 0.02, 8.0) + 0.35)
        distance_term = float(distance_weight) / (dist + 0.25)
        closing_term = float(closing_weight) * np.maximum(metrics['closing'], 0.0)
        score = np.where(valid, ttc_term + clearance_term + distance_term + closing_term, -np.inf)
        if mode == 'hybrid':
            nearest_quota = int(k_nearest) if k_nearest is not None else min(3, k)
            nearest_quota = max(0, min(k, nearest_quota))
            threat_quota = int(k_threat) if k_threat is not None else max(0, k - nearest_quota)
            threat_quota = max(0, min(k - nearest_quota, threat_quota))
            if nearest_quota + threat_quota < k:
                threat_quota = k - nearest_quota

            nearest_order = np.lexsort((idx, dist))
            threat_order = np.lexsort((idx, dist, -score))
            selected = []
            used = set()
            for i in nearest_order:
                ii = int(i)
                if bool(valid[ii]) and ii not in used:
                    selected.append(ii)
                    used.add(ii)
                    if len(selected) >= nearest_quota:
                        break
            threat_added = 0
            for i in threat_order:
                ii = int(i)
                if bool(valid[ii]) and ii not in used:
                    selected.append(ii)
                    used.add(ii)
                    threat_added += 1
                    if threat_added >= threat_quota or len(selected) >= k:
                        break
            if len(selected) < k:
                for i in nearest_order:
                    ii = int(i)
                    if bool(valid[ii]) and ii not in used:
                        selected.append(ii)
                        used.add(ii)
                        if len(selected) >= k:
                            break
            return selected[:min(k, len(selected))]
        order = np.lexsort((idx, dist, -score))
    else:
        logging.warning("[HUMAN-SELECTION] unknown mode=%s, falling back to nearest", mode)
        order = np.lexsort((idx, dist))

    selected = [int(i) for i in order if bool(valid[int(i)])]
    return selected[:min(k, len(selected))]

def build_selected_joint_state_array(
    robot_state,
    human_states,
    k: int,
    mode: str = 'threat',
    dt: float = 0.25,
    horizon: int = 5,
    ttc_weight: float = 2.0,
    clearance_weight: float = 1.0,
    distance_weight: float = 0.15,
    closing_weight: float = 0.25,
    k_nearest: Optional[int] = None,
    k_threat: Optional[int] = None,
    return_indices: bool = False,
):
    """Build [robot9 + selected_humans*5] using the shared Path-Y selector."""
    robot_arr = _state_obj_to_array(robot_state, 9)
    selected = select_human_indices(
        robot_state, human_states, k,
        mode=mode, dt=dt, horizon=horizon,
        ttc_weight=ttc_weight,
        clearance_weight=clearance_weight,
        distance_weight=distance_weight,
        closing_weight=closing_weight,
        k_nearest=k_nearest,
        k_threat=k_threat,
    )
    human_arr = []
    for idx in selected:
        human_arr.extend(_state_obj_to_array(human_states[idx], 5).tolist())
    target_len = max(0, int(k)) * 5
    if len(human_arr) < target_len:
        human_arr.extend([0.0] * (target_len - len(human_arr)))
    out = np.concatenate([robot_arr, np.asarray(human_arr[:target_len], dtype=np.float32)], axis=0)
    if return_indices:
        return out.astype(np.float32), selected
    return out.astype(np.float32)

def select_state_array_humans(
    state_arr,
    k: int,
    mode: str = 'threat',
    dt: float = 0.25,
    horizon: int = 5,
    ttc_weight: float = 2.0,
    clearance_weight: float = 1.0,
    distance_weight: float = 0.15,
    closing_weight: float = 0.25,
    k_nearest: Optional[int] = None,
    k_threat: Optional[int] = None,
) -> np.ndarray:
    """Apply the shared selector to an existing [9+N*5] array."""
    arr = np.asarray(state_arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] <= 9:
        return np.pad(arr, (0, max(0, 9 + int(k) * 5 - arr.shape[0])), mode='constant')
    n_h = max(0, (arr.shape[0] - 9) // 5)
    humans = arr[9:9 + n_h * 5].reshape(n_h, 5)
    selected = select_human_indices(
        arr[:9], [humans[i] for i in range(n_h)], k,
        mode=mode, dt=dt, horizon=horizon,
        ttc_weight=ttc_weight,
        clearance_weight=clearance_weight,
        distance_weight=distance_weight,
        closing_weight=closing_weight,
        k_nearest=k_nearest,
        k_threat=k_threat,
    )
    human_arr = []
    for idx in selected:
        human_arr.extend(humans[idx].tolist())
    target_len = max(0, int(k)) * 5
    if len(human_arr) < target_len:
        human_arr.extend([0.0] * (target_len - len(human_arr)))
    return np.concatenate([arr[:9], np.asarray(human_arr[:target_len], dtype=np.float32)], axis=0).astype(np.float32)

def _load_grid_from_config():
    """"""
    default_grid = {
        'n_speeds': 6,
        'n_headings': 16,
        'v_min': 0.0,
        'v_max': 1.0,
        'sampling': 'even',
        'include_stop': False,
        'stop_eps': 0.05
    }

    config_paths = [
        'configs/env_gdbn.config',
        'crowd_nav/configs/env_gdbn.config',
        '../configs/env_gdbn.config'
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

                    v_pref = 1.0
                    vmax = grid['v_max']
                    assert vmax >= 0.8 * v_pref, f"[ACTIONS] vmax({vmax}) too small vs v_pref({v_pref})"

                    logging.info(f"[CONTRACTS] Loaded grid from {config_path}: {grid}")
                    return grid

            except Exception as e:
                logging.warning(f"[CONTRACTS] Failed to load config from {config_path}: {e}")
                continue

    logging.info(f"[CONTRACTS] Using default grid: {default_grid}")
    return default_grid

GRID = _load_grid_from_config()

def init_grid_from_cfg(cfg):
    """
 cfgGRIDIL/RL

    Args:
 cfg: ConfigParsertrain.py
    """
    global GRID

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

    v_pref = 1.0
    assert v_max >= 0.8 * v_pref, f"[ACTIONS] vmax({v_max}) too small vs v_pref({v_pref})"

    logging.info(f"[CONTRACTS] GRID initialized from runtime cfg: {GRID}")
    return GRID

def discrete_index_to_action(action_idx: int, **grid_params) -> Tuple[float, float]:
    """
 (vx, vy)

    Args:
 action_idx: [0, n_speeds * n_headings)
 **grid_params: GRID

    Returns:
 Tuple[float, float]: (vx, vy)
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

    total_actions = n_speeds * n_headings + (1 if include_stop else 0)
    if not (0 <= action_idx < total_actions):
        raise ValueError(f"Action index {action_idx} out of range [0, {total_actions})")

    if include_stop:
        if action_idx == 0:
            return 0.0, 0.0
        action_idx = action_idx - 1

    speed_idx = action_idx % n_speeds
    heading_idx = action_idx // n_speeds

    if n_speeds == 1:
        speed = v_max
    else:
        if sampling == 'exponential':
            # SARL-style exponential sampling
            speed = (np.exp((speed_idx + 1) / n_speeds) - 1) / (np.e - 1) * v_max
        else:
            # Even sampling (default)
            speed = v_min + (v_max - v_min) * speed_idx / (n_speeds - 1)

    if n_headings == 1:
        angle = 0.0
    else:
        angle = 2 * np.pi * heading_idx / n_headings

    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)

    action_speed = np.sqrt(vx*vx + vy*vy)

    return float(vx), float(vy)

def action_to_discrete_index(vx: float, vy: float, grid=None, **grid_params) -> int:
    """
 (vx, vy)

    Args:
 vx, vy:
 grid:
 **grid_params:

    Returns:
 int:
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

    speed = np.sqrt(vx**2 + vy**2)
    angle = np.arctan2(vy, vx)
    if angle < 0:
        angle += 2 * np.pi

    # Stop action (if enabled)
    if include_stop and speed <= stop_eps:
        return 0

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

    if n_headings == 1:
        heading_idx = 0
    else:
        normalized_angle = angle / (2 * np.pi)
        heading_idx = int(round(normalized_angle * n_headings)) % n_headings

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
 PyTorch

    Args:
 data:
 dtype:
 device:

    Returns:
 torch.Tensor:
    """
    if dtype is None:
        dtype = DEFAULT_DTYPE

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)

    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)

    if device is not None:
        tensor = tensor.to(device)

    return tensor

def validate_state_shape(state, expected_shape):
    """

    Args:
 state:
 expected_shape:

    Raises:
 ValueError:
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
 34D 6×13 tokens

    Args:
 state_34d: 34 [px,py,vx,vy,radius,gx,gy,v_pref,theta] + 5×[px,py,vx,vy,radius]
 : [34] [B, 34]
 JointStateto_array()

    Returns:
 np.ndarray: [8, 13] tokens [B, 8, 13] batch tokens
    """
    if hasattr(state_34d, 'to_array') and callable(getattr(state_34d, 'to_array')):
        state_34d = state_34d.to_array()

    elif isinstance(state_34d, (list, tuple)) and len(state_34d) > 0:
        if hasattr(state_34d[0], 'to_array') and callable(getattr(state_34d[0], 'to_array')):
            state_34d = np.array([s.to_array() for s in state_34d], dtype=np.float32)

    state_34d = _to_numpy(state_34d)
    state_34d = np.array(state_34d, dtype=np.float32)

    if state_34d.ndim == 2:
        B, state_dim = state_34d.shape

        if state_dim != 34:
            if state_dim < 34:
                state_34d = np.pad(state_34d, ((0, 0), (0, 34 - state_dim)), 'constant', constant_values=0.0)
            else:
                state_34d = state_34d[:, :34]

        batch_array = _batch_joint34_to_tokens_vectorized(state_34d)  # [B, 8, 13]
        batch_array = np.expand_dims(batch_array, axis=1)
        return ensure_tensor(batch_array)

    else:
        if len(state_34d) != 34:
            if len(state_34d) < 34:
                state_34d = np.pad(state_34d, (0, 34 - len(state_34d)), 'constant', constant_values=0.0)
            else:
                state_34d = state_34d[:34]

        return _single_joint34_to_tokens(state_34d)

def _batch_joint34_to_tokens_vectorized(state_batch):
    """[B, 9+N*5] → [B, 3+N, 13].  Auto-detects N humans from state dim (default N=5 → 34D)."""
    import torch as _torch
    if isinstance(state_batch, _torch.Tensor):
        return _batch_joint34_to_tokens_torch(state_batch)

    B = state_batch.shape[0]
    state_dim = state_batch.shape[1]
    n_h = max(1, (state_dim - 9) // 5)   # infer number of humans from state width
    n_entities = 3 + n_h
    tokens_batch = np.zeros((B, n_entities, 13), dtype=np.float32)

    robot_px = state_batch[:, 0]
    robot_py = state_batch[:, 1]
    robot_vx = state_batch[:, 2]
    robot_vy = state_batch[:, 3]
    robot_gx = state_batch[:, 5]
    robot_gy = state_batch[:, 6]
    robot_v_pref = np.clip(state_batch[:, 7], 0.0, 3.0)

    # ── Slot 0: robot ────────────────────────────────────────────
    tokens_batch[:, 0, :9] = state_batch[:, :9]
    tokens_batch[:, 0, 9]  = np.sqrt((robot_gx - robot_px)**2 + (robot_gy - robot_py)**2)
    tokens_batch[:, 0, 10] = np.sqrt(robot_vx**2 + robot_vy**2)
    tokens_batch[:, 0, 11] = robot_v_pref
    tokens_batch[:, 0, 12] = 1.0

    # ── Slot 1: goal ─────────────────────────────────────────────
    gdx = robot_gx - robot_px
    gdy = robot_gy - robot_py
    gdist = np.sqrt(gdx**2 + gdy**2)
    gangle = np.arctan2(gdy, gdx)
    tokens_batch[:, 1, 0] = gdx
    tokens_batch[:, 1, 1] = gdy
    tokens_batch[:, 1, 2] = gdist
    tokens_batch[:, 1, 3] = gangle
    tokens_batch[:, 1, 4] = np.cos(gangle)
    tokens_batch[:, 1, 5] = np.sin(gangle)

    # ── Slot 2: motion ───────────────────────────────────────────
    rspeed = np.sqrt(robot_vx**2 + robot_vy**2)
    mangle = np.arctan2(robot_vy, robot_vx)
    mmask  = rspeed > 1e-6
    tokens_batch[:, 2, 0] = np.where(mmask, rspeed, 0.0)
    tokens_batch[:, 2, 1] = np.where(mmask, mangle, 0.0)
    tokens_batch[:, 2, 2] = np.where(mmask, np.cos(mangle), 0.0)
    tokens_batch[:, 2, 3] = np.where(mmask, np.sin(mangle), 0.0)

    # ── Slots 3-(2+N): humans, TTC-sorted ────────────────────────
    H = state_batch[:, 9:9 + n_h * 5].reshape(B, n_h, 5)   # [B, N humans, 5 dims]
    hx_a  = H[:, :, 0];  hy_a  = H[:, :, 1]
    hvx_a = H[:, :, 2];  hvy_a = H[:, :, 3]
    hr_a  = H[:, :, 4]

    valid = np.any(H != 0, axis=2)               # [B, N]

    rx_a = hx_a  - robot_px[:, None]             # [B, N]
    ry_a = hy_a  - robot_py[:, None]
    rvx_a = hvx_a - robot_vx[:, None]
    rvy_a = hvy_a - robot_vy[:, None]

    dist_a = np.sqrt(rx_a**2 + ry_a**2 + 1e-6)  # [B, N]
    closing_a = -(rx_a * rvx_a + ry_a * rvy_a) / (dist_a + 1e-6)
    ttc_a = np.where(closing_a > 0.1, dist_a / (closing_a + 1e-6), dist_a * 10.0)
    ttc_sort = np.where(valid, ttc_a, np.inf)
    order = np.argsort(ttc_sort, axis=1)          # [B, N]

    bi = np.arange(B)
    for slot in range(n_h):
        row = 3 + slot
        h = order[:, slot]          # [B] selected human index
        ok = valid[bi, h]           # [B] validity mask

        rx = rx_a[bi, h];  ry = ry_a[bi, h]
        d  = dist_a[bi, h]
        vxh = hvx_a[bi, h];  vyh = hvy_a[bi, h]
        rh  = hr_a[bi, h]
        rvx = rvx_a[bi, h];  rvy = rvy_a[bi, h]

        # TTC inverse (matches original per-sample logic exactly)
        cs = -(rx * rvx + ry * rvy) / np.maximum(d, 1e-10)
        ttci = np.where(
            d > 1e-6,
            np.where(cs > 1e-6, 1.0 / np.maximum(d / np.maximum(cs, 1e-10), 0.1), 0.0),
            10.0,
        )

        tokens_batch[:, row, 0] = np.where(ok, rx,  0.0)
        tokens_batch[:, row, 1] = np.where(ok, ry,  0.0)
        tokens_batch[:, row, 2] = np.where(ok, d,   0.0)
        tokens_batch[:, row, 3] = np.where(ok, vxh, 0.0)
        tokens_batch[:, row, 4] = np.where(ok, vyh, 0.0)
        tokens_batch[:, row, 5] = np.where(ok, np.sqrt(vxh**2 + vyh**2), 0.0)
        tokens_batch[:, row, 6] = np.where(ok, rh,  0.0)
        tokens_batch[:, row, 7] = np.where(ok, ttci, 0.0)
        tokens_batch[:, row, 8] = np.where(ok & (d < 2.0), 1.0, 0.0)

    return tokens_batch


def _batch_joint34_to_tokens_torch(state_batch: 'torch.Tensor') -> 'torch.Tensor':
    """GPU-accelerated joint_state → tokens. Auto-detects N humans from state dim."""
    import torch
    B = state_batch.shape[0]
    state_dim = state_batch.shape[1]
    n_h = max(1, (state_dim - 9) // 5)
    n_entities = 3 + n_h
    dev = state_batch.device
    tokens = torch.zeros((B, n_entities, 13), dtype=torch.float32, device=dev)

    robot_px = state_batch[:, 0];  robot_py = state_batch[:, 1]
    robot_vx = state_batch[:, 2];  robot_vy = state_batch[:, 3]
    robot_gx = state_batch[:, 5];  robot_gy = state_batch[:, 6]
    robot_v_pref = state_batch[:, 7].clamp(0.0, 3.0)

    tokens[:, 0, :9] = state_batch[:, :9]
    tokens[:, 0, 9]  = ((robot_gx - robot_px)**2 + (robot_gy - robot_py)**2).sqrt()
    tokens[:, 0, 10] = (robot_vx**2 + robot_vy**2).sqrt()
    tokens[:, 0, 11] = robot_v_pref
    tokens[:, 0, 12] = 1.0

    gdx = robot_gx - robot_px;  gdy = robot_gy - robot_py
    gdist = (gdx**2 + gdy**2).sqrt()
    gangle = torch.atan2(gdy, gdx)
    tokens[:, 1, 0] = gdx;  tokens[:, 1, 1] = gdy
    tokens[:, 1, 2] = gdist;  tokens[:, 1, 3] = gangle
    tokens[:, 1, 4] = gangle.cos();  tokens[:, 1, 5] = gangle.sin()

    rspeed = (robot_vx**2 + robot_vy**2).sqrt()
    mangle = torch.atan2(robot_vy, robot_vx)
    mmask = rspeed > 1e-6
    tokens[:, 2, 0] = torch.where(mmask, rspeed,        torch.zeros_like(rspeed))
    tokens[:, 2, 1] = torch.where(mmask, mangle,        torch.zeros_like(mangle))
    tokens[:, 2, 2] = torch.where(mmask, mangle.cos(),  torch.zeros_like(mangle))
    tokens[:, 2, 3] = torch.where(mmask, mangle.sin(),  torch.zeros_like(mangle))

    H = state_batch[:, 9:9 + n_h * 5].view(B, n_h, 5)
    hx_a  = H[:, :, 0];  hy_a  = H[:, :, 1]
    hvx_a = H[:, :, 2];  hvy_a = H[:, :, 3]
    hr_a  = H[:, :, 4]

    valid = (H != 0).any(dim=2)                          # [B, N]

    rx_a  = hx_a  - robot_px.unsqueeze(1)
    ry_a  = hy_a  - robot_py.unsqueeze(1)
    rvx_a = hvx_a - robot_vx.unsqueeze(1)
    rvy_a = hvy_a - robot_vy.unsqueeze(1)

    dist_a = (rx_a**2 + ry_a**2 + 1e-6).sqrt()
    clos_a = -(rx_a * rvx_a + ry_a * rvy_a) / (dist_a + 1e-6)
    ttc_a  = torch.where(clos_a > 0.1, dist_a / (clos_a + 1e-6), dist_a * 10.0)
    ttc_sort = torch.where(valid, ttc_a, torch.full_like(ttc_a, 1e9))
    order = ttc_sort.argsort(dim=1)                       # [B, N]

    bi = torch.arange(B, device=dev)
    for slot in range(n_h):
        row = 3 + slot
        h   = order[:, slot]
        ok  = valid[bi, h]
        rx  = rx_a[bi, h];   ry  = ry_a[bi, h]
        d   = dist_a[bi, h]
        vxh = hvx_a[bi, h];  vyh = hvy_a[bi, h]
        rh  = hr_a[bi, h]
        rvx = rvx_a[bi, h];  rvy = rvy_a[bi, h]

        cs   = -(rx * rvx + ry * rvy) / d.clamp(min=1e-10)
        ttc_slot = torch.where(cs > 1e-6, d / cs.clamp(min=1e-10), torch.full_like(d, 1e9))
        ttci = torch.where(
            d > 1e-6,
            torch.where(cs > 1e-6, 1.0 / ttc_slot.clamp(min=0.1), torch.zeros_like(d)),
            torch.full_like(d, 10.0),
        )
        z = torch.zeros_like(rx)
        tokens[:, row, 0] = torch.where(ok, rx,  z)
        tokens[:, row, 1] = torch.where(ok, ry,  z)
        tokens[:, row, 2] = torch.where(ok, d,   z)
        tokens[:, row, 3] = torch.where(ok, vxh, z)
        tokens[:, row, 4] = torch.where(ok, vyh, z)
        tokens[:, row, 5] = torch.where(ok, (vxh**2 + vyh**2).sqrt(), z)
        tokens[:, row, 6] = torch.where(ok, rh,  z)
        tokens[:, row, 7] = torch.where(ok, ttci, z)
        tokens[:, row, 8] = torch.where(ok & (d < 2.0), torch.ones_like(d), z)

    return tokens

def _single_joint34_to_tokens(state_34d):
    """[9+N*5] → [3+N, 13] tokens. Auto-detects N humans from state dim."""
    state_dim = len(state_34d)
    n_h = max(1, (state_dim - 9) // 5)
    robot_state = state_34d[:9]
    px, py, vx, vy, radius, gx, gy, v_pref, theta = robot_state[:9]

    if not (0.0 <= v_pref <= 3.0):
        import logging
        logging.warning(f"[CONTRACTS] v_pref out of range: {v_pref}, clamping to [0, 3.0]")
        v_pref = max(0.0, min(3.0, v_pref))

    humans_state = state_34d[9:9 + n_h * 5].reshape(n_h, 5)  # [N, [px,py,vx,vy,radius]]

    tokens = np.zeros((3 + n_h, 13), dtype=np.float32)

    tokens[0, :9] = robot_state[:9]
    tokens[0, 9] = np.sqrt((gx - px)**2 + (gy - py)**2)
    tokens[0, 10] = np.sqrt(vx**2 + vy**2)
    tokens[0, 11] = v_pref
    tokens[0, 12] = 1.0

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

    speed = np.sqrt(vx**2 + vy**2)
    if speed > 1e-6:
        motion_angle = np.arctan2(vy, vx)
        tokens[2, 0] = speed
        tokens[2, 1] = motion_angle
        tokens[2, 2] = np.cos(motion_angle)
        tokens[2, 3] = np.sin(motion_angle)

    if len(humans_state) > 0:
        selected = select_human_indices(
            robot_state, [humans_state[i] for i in range(len(humans_state))],
            n_h, mode='ttc', dt=0.25, horizon=1,
        )

        for row_idx, human_idx in enumerate(selected[:n_h]):
            row = 3 + row_idx
            human = humans_state[human_idx]
            hx, hy, hvx, hvy, hradius = human

            rel_x = hx - px
            rel_y = hy - py
            rel_dist = np.sqrt(rel_x**2 + rel_y**2)

            tokens[row, 0] = rel_x
            tokens[row, 1] = rel_y
            tokens[row, 2] = rel_dist
            tokens[row, 3] = hvx
            tokens[row, 4] = hvy
            tokens[row, 5] = np.sqrt(hvx**2 + hvy**2)
            tokens[row, 6] = hradius

            if rel_dist > 1e-6:
                rel_vx = hvx - vx
                rel_vy = hvy - vy
                closing_speed = -(rel_x * rel_vx + rel_y * rel_vy) / rel_dist
                if closing_speed > 1e-6:
                    ttc = rel_dist / closing_speed
                    ttc_inv = 1.0 / max(ttc, 0.1)
                else:
                    ttc_inv = 0.0
            else:
                ttc_inv = 10.0

            tokens[row, 7] = ttc_inv

            in_social = 1.0 if rel_dist < 2.0 else 0.0
            tokens[row, 8] = in_social

    return tokens

def tokens_to_34d(tokens):
    """[3+N, 13] tokens -> [9+N*5] state. Name kept for compatibility."""
    tokens = _to_numpy(tokens)
    tokens = np.array(tokens, dtype=np.float32)
    if tokens.ndim == 3:
        return np.stack([tokens_to_34d(t) for t in tokens], axis=0)
    if tokens.ndim != 2 or tokens.shape[0] < 4 or tokens.shape[1] < 13:
        raise ValueError(f"Expected [3+N, 13] tokens, got {tokens.shape}")

    n_humans = tokens.shape[0] - 3
    state_34d = np.zeros(9 + n_humans * 5, dtype=np.float32)

    state_34d[:9] = tokens[0, :9]

    robot_px, robot_py = tokens[0, 0], tokens[0, 1]

    human_start_idx = 9
    for i in range(n_humans):
        row = 3 + i
        if np.any(tokens[row] != 0):
            rel_x = tokens[row, 0]
            rel_y = tokens[row, 1]
            hvx = tokens[row, 3]
            hvy = tokens[row, 4]
            hradius = tokens[row, 6]

            hx = robot_px + rel_x
            hy = robot_py + rel_y

            human_idx = human_start_idx + i * 5
            if human_idx + 4 < state_34d.shape[0]:
                state_34d[human_idx:human_idx+5] = [hx, hy, hvx, hvy, hradius]

    return state_34d

def to_btnd(states_list, sequence_length=8):
    """
 " 4D " - [B, T, 8, 13]

    Args:
 states_list:
 sequence_length:

    Returns:
 torch.Tensor: [B, T, 8, 13]
    """
    if not states_list:
        raise ValueError("Empty states list")

    if isinstance(states_list, torch.Tensor) and states_list.ndim == 4:
        B, T, H, W = states_list.shape
        assert H == 8 and W == 13, f"Expected [B, T, 8, 13], got {states_list.shape}"
        return states_list

    batch_tokens = []

    for batch_item in states_list:
        if isinstance(batch_item, list):
            tokens_seq = []
            for state in batch_item:
                if hasattr(state, 'to_array'):
                    state_34d = state.to_array()
                else:
                    state_34d = np.array(state, dtype=np.float32)

                if len(state_34d) != 34:
                    if len(state_34d) < 34:
                        state_34d = np.pad(state_34d, (0, 34 - len(state_34d)), 'constant')
                    else:
                        state_34d = state_34d[:34]

                tokens = joint34_to_tokens(state_34d)
                tokens_seq.append(tokens)

            while len(tokens_seq) < sequence_length:
                tokens_seq.append(tokens_seq[-1])

            if len(tokens_seq) > sequence_length:
                tokens_seq = tokens_seq[-sequence_length:]

            batch_tokens.append(np.stack(tokens_seq, axis=0))
        else:
            if hasattr(batch_item, 'to_array'):
                state_34d = batch_item.to_array()
            else:
                state_34d = np.array(batch_item, dtype=np.float32)

            if len(state_34d) != 34:
                if len(state_34d) < 34:
                    state_34d = np.pad(state_34d, (0, 34 - len(state_34d)), 'constant')
                else:
                    state_34d = state_34d[:34]

            tokens = joint34_to_tokens(state_34d)
            tokens_seq = [tokens] * sequence_length
            batch_tokens.append(np.stack(tokens_seq, axis=0))

    batch_array = np.stack(batch_tokens, axis=0)  # [B, T, 8, 13]
    return ensure_tensor(batch_array)

def assert_4d(tensor):
    """4D"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")
    B, T, H, W = tensor.shape
    if H != 8 or W != 13:
        raise ValueError(f"Expected [B, T, 8, 13] shape, got {tensor.shape}")

def pick_last(tensor):
    """[B, T, ...]  [B, ...]"""
    if tensor.ndim < 2:
        raise ValueError(f"Need at least 2D tensor, got {tensor.ndim}D")
    return tensor[:, -1]

def simulate_next_frames(states_34d, actions, dt):
    """

    Args:
 states_34d: [B, 9+N*5]
 actions: [B, A, 2] [B, 2]
 dt:

    Returns:
 torch.Tensor: [B, A, 9+N*5] [B, 9+N*5]
    """
    states_34d = ensure_tensor(states_34d)
    actions = ensure_tensor(actions)

    if states_34d.ndim != 2 or states_34d.shape[1] < 14 or (states_34d.shape[1] - 9) % 5 != 0:
        raise ValueError(f"Expected [B, 9+N*5] states, got {states_34d.shape}")

    B = states_34d.shape[0]
    state_dim = states_34d.shape[1]
    n_humans = (state_dim - 9) // 5

    if actions.ndim == 2:
        A = 1
        actions = actions.unsqueeze(1)  # [B, 1, 2]
    elif actions.ndim == 3:
        A = actions.shape[1]
    else:
        raise ValueError(f"Expected actions shape [B, 2] or [B, A, 2], got {actions.shape}")

    next_states = states_34d.unsqueeze(1).repeat(1, A, 1)  # [B, A, D]

    for b in range(B):
        for a in range(A):
            current_state = states_34d[b].clone()  # [D]
            action = actions[b, a]  # [2]

            px, py, vx, vy = current_state[:4]
            v_pref = current_state[7]

            new_vx, new_vy = action[0], action[1]
            action_speed = torch.sqrt(new_vx**2 + new_vy**2)

            if action_speed > v_pref:
                scale = v_pref / action_speed
                new_vx *= scale
                new_vy *= scale

            new_px = px + new_vx * dt
            new_py = py + new_vy * dt

            next_state = current_state.clone()
            next_state[0] = new_px
            next_state[1] = new_py
            next_state[2] = new_vx
            next_state[3] = new_vy

            for h in range(n_humans):
                h_start = 9 + h * 5
                if h_start + 3 < state_dim:
                    human_slice = next_state[h_start:h_start+4]
                    if human_slice.numel() == 4:
                        hx, hy, hvx, hvy = human_slice
                        if torch.abs(hx) > 1e-6 or torch.abs(hy) > 1e-6:
                            next_state[h_start] = hx + hvx * dt
                            next_state[h_start + 1] = hy + hvy * dt

            next_states[b, a] = next_state

    if A == 1:
        return next_states.squeeze(1)  # [B, D]
    else:
        return next_states  # [B, A, D]

def batched_lookahead_Q(joint_tokens, value_network, dt, gamma=0.95, reward_fn=None):
    """
 look-aheadQ

    Args:
 joint_tokens: [B, T, 8, 13]
 value_network: [B, T, 8, 13] [B]
 dt:
 gamma:
 reward_fn: r_sa=0

    Returns:
 torch.Tensor: [B, A] Q
    """
    assert_4d(joint_tokens)
    B, T = joint_tokens.shape[:2]

    current_tokens = pick_last(joint_tokens)  # [B, 8, 13]
    current_states_34d = torch.stack([
        ensure_tensor(tokens_to_34d(current_tokens[b].cpu().numpy()))
        for b in range(B)
    ], dim=0)  # [B, 34]

    total_actions = GRID['n_speeds'] * GRID['n_headings']
    all_actions = []
    for a_idx in range(total_actions):
        vx, vy = discrete_index_to_action(a_idx)
        all_actions.append([vx, vy])

    all_actions = ensure_tensor(all_actions, device=joint_tokens.device)  # [A, 2]
    all_actions = all_actions.unsqueeze(0).repeat(B, 1, 1)  # [B, A, 2]

    next_states_34d = simulate_next_frames(current_states_34d, all_actions, dt)  # [B, A, 9+N*5]

    q_values = torch.zeros(B, total_actions, device=joint_tokens.device)

    for b in range(B):
        for a in range(total_actions):
            if reward_fn is not None:
                reward = reward_fn(current_states_34d[b], all_actions[b, a])
            else:
                reward = 0.0

            next_state_34d = next_states_34d[b, a].cpu().numpy()
            next_tokens = joint34_to_tokens(next_state_34d)

            next_sequence = joint_tokens[b:b+1].clone()  # [1, T, 8, 13]
            next_sequence[0, -1] = ensure_tensor(next_tokens, device=joint_tokens.device)

            next_value = value_network(next_sequence).item()

            q_values[b, a] = reward + gamma * next_value

    return q_values

def immediate_reward(event, d_curr, rc, dt, extras=None):
    """
 - MSR
 debug.md B(MSR)

    Args:
 event: ('success', 'collision', 'timeout', 'normal')
 d_curr:
 rc:
 dt:
 extras: {d_prev, v_pref, lambda_p, use_msr}

    Returns:
 float:
    """
    r = 0.0

    if event == 'success':
        r = 1.0
    elif event == 'collision':
        r = -0.25
    elif event == 'timeout':
        r = -0.5
    elif event == 'normal':
        if d_curr < rc:
            r = (d_curr - rc) * 0.1

    if extras is not None and extras.get('use_msr', False):
        d_prev = extras.get('d_prev', None)
        v_pref = float(extras.get('v_pref', 1.0) or 1.0)
        lambda_p = float(extras.get('lambda_p', 0.3))

        if d_prev is not None and d_curr is not None:
            max_step = v_pref * dt
            raw_prog = (d_prev - d_curr) / max_step
            prog = float(max(-1.0, min(1.0, raw_prog)))
            msr = lambda_p * prog
            r += msr

            r += -0.002

            if hasattr(immediate_reward, '_msr_count'):
                immediate_reward._msr_count += 1
            else:
                immediate_reward._msr_count = 1

            if immediate_reward._msr_count % 1000 == 0:
                import logging
                logging.info(f"[MSR-FIXED] prog_raw={raw_prog:.3f}, prog_clipped={prog:.3f}, msr={msr:.6f}, λ={lambda_p}")

    return r

ensure_btnd = to_btnd

__all__ = [
    'GRID', 'DEFAULT_DTYPE',
    'discrete_index_to_action', 'action_to_discrete_index',
    'ensure_tensor', 'validate_state_shape',
    'joint34_to_tokens', 'tokens_to_34d',
    'select_human_indices', 'build_selected_joint_state_array', 'select_state_array_humans',
    'to_btnd', 'ensure_btnd', 'assert_4d', 'pick_last',
    'simulate_next_frames', 'batched_lookahead_Q', 'immediate_reward'
]
