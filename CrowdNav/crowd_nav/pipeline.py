# ============================================================
# Training Pipeline - IL/PPO Logic
# ============================================================
"""
训练流程模块 - 包含 IL 和 PPO 的核心训练逻辑

职责分离：
- pipeline.py: 训练算法实现（IL采集、BC训练、PPO rollouts & 更新）
- train.py: 入口调度、配置加载、环境搭建、日志绘图
"""
from __future__ import annotations
import os
import logging
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ============================================================
# IL Pipeline - ORCA Teacher Demonstrations & BC Pretraining
# ============================================================

def load_offline_il_dataset(train_cfg):
    """
    加载离线IL数据集（chunk合并、max_il_prefill截断、返回轨迹列表）
    Args:
        train_cfg: Training configuration object
    Returns:
        success_trajs_all: List of ((states, actions, rewards), meta) tuples
    """
    offline_dataset_path = train_cfg.offline_il_dataset if hasattr(train_cfg, 'offline_il_dataset') else None
    if offline_dataset_path is None:
        for version in ['v5.0', 'v4.0', 'v3.1', 'v3.0', 'v2.1', 'v2.0', 'v1.0']:
            candidates = [
                f'data/il_dataset_diverse_{version}.pth',
                f'../data/il_dataset_diverse_{version}.pth',
                os.path.join(os.path.dirname(__file__), f'../data/il_dataset_diverse_{version}.pth'),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    offline_dataset_path = os.path.abspath(candidate)
                    logging.info(f"[IL-CACHE] 找到数据集候选: {offline_dataset_path}")
                    break
            if offline_dataset_path:
                break
        if offline_dataset_path is None:
            offline_dataset_path = 'data/il_dataset_diverse_v3.1.pth'
    success_trajs_all = []
    if not os.path.exists(offline_dataset_path):
        logging.info(f"[IL-CACHE] 未发现离线数据集 ({offline_dataset_path})")
        return success_trajs_all
    try:
        logging.info(f"[IL-CACHE] 发现离线数据集缓存: {offline_dataset_path}")
        dataset = torch.load(offline_dataset_path, map_location='cpu', weights_only=False)
        if isinstance(dataset, dict) and 'chunk_files' in dataset:
            logging.info(f"[IL-CACHE] 检测到chunked格式数据集 (version={dataset.get('version', 'unknown')})")
            chunk_files = dataset['chunk_files']
            logging.info(f"[IL-CACHE] 发现 {len(chunk_files)} 个chunk文件")
            metadata_dir = os.path.dirname(os.path.abspath(offline_dataset_path))
            max_load_limit = train_cfg.max_il_prefill if hasattr(train_cfg, 'max_il_prefill') else 50000
            logging.info(f"[IL-CACHE] 最大加载限制: {max_load_limit} episodes (max_il_prefill)")
            merged_cache_path = offline_dataset_path.replace('.pth', f'_merged_{max_load_limit}.pth')
            use_merged_cache = False
            if os.path.exists(merged_cache_path):
                merged_mtime = os.path.getmtime(merged_cache_path)
                metadata_mtime = os.path.getmtime(offline_dataset_path)
                if merged_mtime >= metadata_mtime:
                    logging.info(f"[IL-CACHE] 发现合并缓存: {os.path.basename(merged_cache_path)}")
                    logging.info(f"[IL-CACHE] 直接加载缓存，跳过chunk合并（节省1分钟）")
                    try:
                        merged_data = torch.load(merged_cache_path, map_location='cpu', weights_only=False)
                        cached_limit = merged_data.get('max_il_prefill', 0)
                        if cached_limit == max_load_limit:
                            cached_trajs = merged_data['trajectories']
                            config_info = merged_data.get('config', {})
                            stats_info = merged_data.get('stats', {})
                            use_merged_cache = True
                            logging.info(f"[IL-CACHE] ✓ 从缓存加载 {len(cached_trajs)} 条轨迹")
                        else:
                            logging.info(f"[IL-CACHE] 缓存限制不匹配 (缓存:{cached_limit} vs 当前:{max_load_limit})，重新合并")
                            use_merged_cache = False
                    except Exception as e:
                        logging.warning(f"[IL-CACHE] 缓存加载失败: {e}，回退到chunk合并")
                        use_merged_cache = False
                else:
                    logging.info(f"[IL-CACHE] 缓存已过期（元数据更新），重新合并chunk")
            if not use_merged_cache:
                cached_trajs = []
                for i, chunk_path in enumerate(chunk_files):
                    if len(cached_trajs) >= max_load_limit:
                        logging.info(f"[IL-CACHE] 已达到max_il_prefill={max_load_limit}，跳过剩余{len(chunk_files)-i}个chunk")
                        break
                    if not os.path.isabs(chunk_path):
                        chunk_path = os.path.join(metadata_dir, os.path.basename(chunk_path))
                    logging.info(f"[IL-CACHE] 加载 chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_path)}")
                    try:
                        chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)
                        chunk_trajs = chunk_data.get('trajectories', [])
                        if len(cached_trajs) + len(chunk_trajs) > max_load_limit:
                            remaining = max_load_limit - len(cached_trajs)
                            cached_trajs.extend(chunk_trajs[:remaining])
                            logging.info(f"[IL-CACHE]   ✓ chunk{i} 部分加载 {remaining}/{len(chunk_trajs)} 条轨迹 (累计: {len(cached_trajs)})")
                            logging.info(f"[IL-CACHE]   已达到max_il_prefill={max_load_limit}，停止加载")
                            break
                        else:
                            cached_trajs.extend(chunk_trajs)
                            logging.info(f"[IL-CACHE]   ✓ chunk{i} 加载 {len(chunk_trajs)} 条轨迹 (累计: {len(cached_trajs)})")
                    except Exception as e:
                        logging.warning(f"[IL-CACHE]   ✗ chunk{i} 加载失败: {e}")
                        logging.warning(f"[IL-CACHE]   路径: {chunk_path}")
                config_info = dataset.get('config', {})
                stats_info = dataset.get('stats', {})
                logging.info(f"[IL-CACHE] ✓ 合并完成，总轨迹数: {len(cached_trajs)}")
                try:
                    logging.info(f"[IL-CACHE] 保存合并缓存到: {os.path.basename(merged_cache_path)}")
                    merged_data = {
                        'trajectories': cached_trajs,
                        'config': config_info,
                        'stats': stats_info,
                        'version': dataset.get('version', 'unknown'),
                        'merged_from_chunks': len(chunk_files),
                        'max_il_prefill': max_load_limit,
                    }
                    torch.save(merged_data, merged_cache_path, pickle_protocol=4, _use_new_zipfile_serialization=False)
                    cache_size_mb = os.path.getsize(merged_cache_path) / 1024 / 1024
                    logging.info(f"[IL-CACHE] ✓ 缓存已保存 ({cache_size_mb:.1f}MB)，下次加载将直接使用")
                except Exception as e:
                    logging.warning(f"[IL-CACHE] 缓存保存失败（不影响训练）: {e}")
        elif isinstance(dataset, dict) and 'trajectories' in dataset:
            cached_trajs = dataset['trajectories']
            config_info = dataset.get('config', {})
            stats_info = dataset.get('stats', {})
        else:
            logging.warning(f"[IL-CACHE] 数据集格式不正确")
            return success_trajs_all
        if cached_trajs:
            logging.info(f"[IL-CACHE] 数据集版本: {dataset.get('version', 'unknown')}")
            logging.info(f"[IL-CACHE] 生成时间: {dataset.get('timestamp', 'unknown')}")
            logging.info(f"[IL-CACHE] Seeds: {config_info.get('seeds', [])}")
            logging.info(f"[IL-CACHE] ORCA configs: {list(config_info.get('orca_configs', {}).keys())}")
            logging.info(f"[IL-CACHE] 总轨迹数: {len(cached_trajs)}")
            logging.info(f"[IL-CACHE] 原始成功率: {stats_info.get('success_rate', 0.0):.2%}")
            max_il_prefill = train_cfg.max_il_prefill if hasattr(train_cfg, 'max_il_prefill') else len(cached_trajs)
            cached_trajs = cached_trajs[:max_il_prefill]
            logging.info(f"[IL-CACHE] 限制加载: {len(cached_trajs)} episodes (max_il_prefill={max_il_prefill})")
            for traj in cached_trajs:
                states = traj['states']
                actions = traj['actions']
                rewards = traj.get('rewards', [0.0] * len(states))
                meta = traj.get('meta', {})
                traj_tuple = ((states, actions, rewards), meta)
                success_trajs_all.append(traj_tuple)
            logging.info(f"[IL-CACHE] ✓ 成功加载 {len(success_trajs_all)} 条轨迹")
    except Exception as e:
        logging.warning(f"[IL-CACHE] 加载失败: {e}")
        success_trajs_all = []
    return success_trajs_all


def _build_il_dataset(trajs, gamma_value, seq_len, joint34_to_tokens_func, pad_state_func,
                      limit=None, cache_dir='data/il_cache_tokens', enable_cache=True):
    """
    Build IL dataset with state windows, actions, and MC returns (with caching)
    Args:
        trajs: List of trajectories
        gamma_value: Discount factor
        seq_len: Sequence length
        joint34_to_tokens_func: Function to convert states to tokens
        pad_state_func: Function to pad states to 34D
        limit: Limit number of trajectories to process (None = all)
        cache_dir: Directory for caching tokenized datasets
        enable_cache: Enable caching mechanism
    Returns:
        dict: {'states_tokens': Tensor[N, seq_len, 6, 13], 'actions': Tensor[N, 2], 'returns': Tensor[N]}
    """
    seq_len_local = max(1, int(seq_len))
    if limit is not None and limit > 0:
        trajs = trajs[:limit]
    if enable_cache:
        os.makedirs(cache_dir, exist_ok=True)
        gamma_int = int(gamma_value * 100)
        cache_key = f"{len(trajs)}_{seq_len_local}_g{gamma_int}.pt"
        cache_path = os.path.join(cache_dir, cache_key)
        if os.path.exists(cache_path):
            try:
                logging.info(f"[IL-DATASET-CACHE] ✓ Loading cached dataset: {cache_key}")
                cached_data = torch.load(cache_path, map_location='cpu', weights_only=False)
                if 'states_tokens' in cached_data and 'actions' in cached_data and 'returns' in cached_data:
                    logging.info(f"[IL-DATASET-CACHE] ✓ Cache hit! {cached_data['states_tokens'].size(0)} samples loaded")
                    return cached_data
                else:
                    logging.warning(f"[IL-DATASET-CACHE] Cache file corrupted, rebuilding...")
            except Exception as e:
                logging.warning(f"[IL-DATASET-CACHE] Failed to load cache: {e}, rebuilding...")
    state_windows = []
    teacher_actions = []
    returns = []
    n_clipped_actions = 0
    n_total_actions = 0
    for idx, (traj, _info) in enumerate(trajs):
        if limit is not None and idx >= limit:
            break
        states, actions, rewards = traj
        T = len(states)
        if T == 0:
            continue
        states_34 = [pad_state_func(s) for s in states]
        mc = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma_value * G
            mc.insert(0, G)
        for t in range(T):
            window = states_34[max(0, t - seq_len_local + 1): t + 1]
            if not window:
                continue
            if len(window) < seq_len_local:
                pad_needed = seq_len_local - len(window)
                pad_frame = window[0]
                window = [pad_frame.copy() for _ in range(pad_needed)] + [w.copy() for w in window]
            else:
                window = [w.copy() for w in window[-seq_len_local:]]
            state_windows.append(np.stack(window, axis=0).astype(np.float32))

            action_seq = []
            action_indices = list(range(max(0, t - seq_len_local + 1), t + 1))
            if len(action_indices) < seq_len_local:
                first_idx = action_indices[0]
                action_indices = [first_idx] * (seq_len_local - len(action_indices)) + action_indices
            for ai in action_indices:
                action_t = actions[ai]
                if hasattr(action_t, 'vx') and hasattr(action_t, 'vy'):
                    action_vx, action_vy = float(action_t.vx), float(action_t.vy)
                else:
                    action_arr = np.array(action_t, dtype=np.float32)[:2]
                    action_vx, action_vy = float(action_arr[0]), float(action_arr[1])
                action_speed = np.sqrt(action_vx**2 + action_vy**2)
                v_max = 1.0
                n_total_actions += 1
                if action_speed > v_max:
                    n_clipped_actions += 1
                    action_vx = action_vx / action_speed * v_max
                    action_vy = action_vy / action_speed * v_max
                action_seq.append([action_vx, action_vy])
            teacher_actions.append(np.array(action_seq, dtype=np.float32))
            returns.append(mc[t])
    if not state_windows:
        raise RuntimeError("[IL-BC] No data available for BC training")
    state_windows = np.asarray(state_windows, dtype=np.float32)
    teacher_actions = np.asarray(teacher_actions, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)
    clip_rate = n_clipped_actions / n_total_actions * 100 if n_total_actions > 0 else 0
    logging.info(f"[IL-ACTION-CLIP] Clipped {n_clipped_actions}/{n_total_actions} actions ({clip_rate:.2f}%) to v_max=1.0")
    flat_states = state_windows.reshape(-1, state_windows.shape[-1])
    total_states = flat_states.shape[0]
    batch_size_token = 100000
    if total_states > batch_size_token:
        logging.info(f"[IL-TOKEN-CONVERT] Converting {total_states} states to tokens in batches (batch_size={batch_size_token})...")
        token_batches = []
        with tqdm(total=total_states, desc="[TOKEN-CONVERT]", ncols=100, unit="states") as pbar:
            for start_idx in range(0, total_states, batch_size_token):
                end_idx = min(start_idx + batch_size_token, total_states)
                batch_states = flat_states[start_idx:end_idx]
                batch_tokens = joint34_to_tokens_func(batch_states)
                token_batches.append(batch_tokens)
                pbar.update(end_idx - start_idx)
        if isinstance(token_batches[0], torch.Tensor):
            tokens = torch.cat(token_batches, dim=0)
        else:
            tokens = np.concatenate(token_batches, axis=0)
        logging.info(f"[IL-TOKEN-CONVERT] ✓ Token conversion complete ({total_states} states)")
    else:
        logging.info(f"[IL-TOKEN-CONVERT] Converting {total_states} states to tokens...")
        tokens = joint34_to_tokens_func(flat_states)
        logging.info(f"[IL-TOKEN-CONVERT] ✓ Token conversion complete")
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.view(state_windows.shape[0], seq_len_local, 1, 6, 13).squeeze(2).contiguous()
    else:
        tokens = torch.from_numpy(tokens).view(state_windows.shape[0], seq_len_local, 1, 6, 13).squeeze(2).contiguous()
    logging.info(f"[IL-DATASET] BC dataset: {state_windows.shape[0]} sequences (seq_len={seq_len_local}, with MC returns)")
    dataset = {
        'states_tokens': tokens.float(),
        'actions': torch.from_numpy(teacher_actions).float(),
        'returns': torch.from_numpy(returns).float()
    }
    if enable_cache:
        try:
            logging.info(f"[IL-DATASET-CACHE] Saving cache to: {cache_key}")
            torch.save(dataset, cache_path, pickle_protocol=4)
            cache_size_mb = os.path.getsize(cache_path) / 1024 / 1024
            logging.info(f"[IL-DATASET-CACHE] ✓ Cache saved ({cache_size_mb:.1f}MB)")
        except Exception as e:
            logging.warning(f"[IL-DATASET-CACHE] Failed to save cache (not critical): {e}")
    return dataset


def run_policy_bc_training(policy, device, seq_len: int, il_epochs: int, il_batch_size: int,
                           il_ckpt_path: str, explorer, gamma: float = 0.97,
                           success_trajs: Optional[List] = None, config=None,
                           joint34_to_tokens_func=None, pad_state_func=None, event_token_func=None):
    """Run BC training on policy using IL trajectories"""
    ac_net = policy
    bc_lr = getattr(config, 'il_learning_rate', None) or config.learning_rate
    optim = torch.optim.Adam(ac_net.parameters(), lr=bc_lr)
    logging.info(f"[IL-BC] Training all parameters ({sum(p.numel() for p in ac_net.parameters())} params)")
    ac_net.train()
    ac_net.set_training_mode('bc')
    logging.info(f"[IL-BC-MODE] Using BC mode (full temporal modeling with Mamba, deterministic output)")
    logging.info("=" * 80)
    logging.info(f"[IL-BC] BC训练开始（Actor预训练）")
    logging.info(f"[IL-BC] 目标：让Policy学会基本导航动作，达到50-60%成功率")
    logging.info("=" * 80)
    logging.info(f"[IL-BC] Starting BC training (Action Imitation), lr={bc_lr}, gamma={gamma}")
    dataset_trajs = success_trajs if success_trajs is not None else getattr(explorer, "_last_trajectories", [])
    if not dataset_trajs:
        logging.warning("[IL-BC] No trajectories available, skipping BC training")
        return
    limit = getattr(config.il_pipeline, "bc_samples", getattr(config, "bc_training_samples", len(dataset_trajs)))
    original_count = len(dataset_trajs)
    dataset_trajs = dataset_trajs[:limit]
    data_source = "离线数据集" if success_trajs is not None else "在线采集"
    logging.info(f"[IL-BC] 数据来源: {data_source}")
    logging.info(f"[IL-BC] 轨迹数量: {len(dataset_trajs)}/{original_count} 条成功案例 (限制: bc_samples={limit})")
    dataset = _build_il_dataset(dataset_trajs, gamma, seq_len, joint34_to_tokens_func, pad_state_func, enable_cache=False)
    states_tokens = dataset['states_tokens'].to(device, non_blocking=True)
    teacher_actions_full = dataset['actions'].to(device, non_blocking=True)
    dataset_size = states_tokens.size(0)
    logging.info(f"[IL-BC] 准备数据集: 样本数={dataset_size}, batch_size={il_batch_size}, batches={dataset_size // il_batch_size}")
    pbar = tqdm(range(1, il_epochs + 1), desc="[IL-BC]", ncols=120, leave=True)
    for ep in pbar:
        if dataset_size < il_batch_size:
            logging.warning(f"[IL-BC] Dataset too small (N={dataset_size}), reducing batch size to {dataset_size}")
        perm = torch.randperm(dataset_size, device=device)
        total_bc_loss = torch.tensor(0.0, device=device)
        n_batches = 0
        for start in range(0, dataset_size, il_batch_size):
            idx = perm[start:start + il_batch_size]
            if idx.numel() < 8:
                continue
            states_batch = states_tokens[idx].contiguous()
            teacher_actions = teacher_actions_full[idx].contiguous()
            if states_batch.device != device:
                states_batch = states_batch.to(device, non_blocking=True)
            if teacher_actions.device != device:
                teacher_actions = teacher_actions.to(device, non_blocking=True)

            continuous_preds, _ = ac_net.forward_both(states_batch, return_all_timesteps=True)
            if isinstance(continuous_preds, tuple):
                continuous_preds = continuous_preds[0]

            if n_batches == 0 and ep == 1:
                logging.info(f"[IL-BC-DEBUG] states_batch: {states_batch.shape}")
                logging.info(f"[IL-BC-DEBUG] continuous_preds: {continuous_preds.shape}")
                logging.info(f"[IL-BC-DEBUG] teacher_actions: {teacher_actions.shape}")

            bc_loss = F.mse_loss(continuous_preds, teacher_actions)
            optim.zero_grad(set_to_none=True)
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(ac_net.parameters(), 1.0)
            optim.step()
            total_bc_loss += bc_loss.detach()
            n_batches += 1
        avg_bc_loss = (total_bc_loss / max(1, n_batches)).item()
        pbar.set_postfix({'bc': f'{avg_bc_loss:.4f}'})
        if ep % max(1, il_epochs // 10) == 0 or ep == il_epochs:
            logging.info(f"[IL-BC] epoch={ep}/{il_epochs} bc_loss={avg_bc_loss:.4f}")
    pbar.close()
    logging.info(f"[IL-BC] Evaluating BC-trained model...")
    try:
        ac_net.eval()
        policy.set_phase('eval')
        stats = explorer.run_k_episodes(
            100,
            'val',
            update_memory=False,
            show_tqdm=False,
            return_stats=True,
            imitation_learning=False,
            force_joint_state_policy=False
        )
        if isinstance(stats, dict):
            success_rate = stats.get('success_rate', 0.0)
            avg_return = stats.get('total_reward', stats.get('avg_return', 0.0))
            collision_rate = stats.get('collision_rate', 0.0)
            timeout_rate = stats.get('timeout_rate', 0.0)
            logging.info(
                f"[IL-BC-EVAL] 100 episodes | success={success_rate:.1%} | "
                f"collision={collision_rate:.1%} | timeout={timeout_rate:.1%} | reward={avg_return:.4f}"
            )
        else:
            logging.info(f"[IL-BC-EVAL] Evaluation completed (stats not available)")
    except Exception as e:
        logging.warning(f"[IL-BC-EVAL] Evaluation failed: {e}")
    finally:
        ac_net.train()
        policy.set_phase('train')
    ac_net.set_training_mode('rl')

    logging.info(f"[BC→PPO] BC training finished, switched to RL mode for PPO training")

    torch.save({
        "value": ac_net.state_dict(),
        "meta": {"stage": "bc", "epochs": il_epochs}
    }, il_ckpt_path)
    logging.info(f"[IL-BC] Saved BC checkpoint to {il_ckpt_path}")


def run_il_phase(cfg, env, explorer, policy, device, outdir, train_cfg,
                 convert_state_func, convert_actions_func, meta_from_event_func,
                 joint34_to_tokens_func, pad_state_func, event_token_func, log_policy_tag_func):
    """
    IL 阶段 - 收集 ORCA teacher 示范并进行 BC 预训练

    Args:
        cfg: configparser.RawConfigParser
        env: 环境
        explorer: Explorer 对象
        policy: 策略网络
        device: 设备
        outdir: 输出目录
        train_cfg: Training config object
        convert_state_func: State conversion function
        convert_actions_func: Actions conversion function
        meta_from_event_func: Meta info construction function
        joint34_to_tokens_func: State to tokens conversion
        pad_state_func: State padding function
        event_token_func: Event token extraction function
        log_policy_tag_func: Policy tag logging function
    """
    original_time_limit = cfg.getfloat('env', 'time_limit')
    original_success_radius = cfg.getfloat('env', 'success_radius')
    original_human_num = cfg.getint('env', 'human_num')
    original_neighbor_dist = cfg.getfloat('orca', 'neighbor_dist')
    original_max_neighbors = cfg.getint('orca', 'max_neighbors')
    original_time_horizon = cfg.getfloat('orca', 'time_horizon')
    original_time_horizon_obst = cfg.getfloat('orca', 'time_horizon_obst')
    original_safety_space = cfg.getfloat('orca', 'safety_space')

    teacher_neighbor_dist = cfg.getfloat('imitation_learning', 'teacher_neighbor_dist', fallback=original_neighbor_dist)
    teacher_max_neighbors = cfg.getint('imitation_learning', 'teacher_max_neighbors', fallback=original_max_neighbors)
    teacher_time_horizon = cfg.getfloat('imitation_learning', 'teacher_time_horizon', fallback=original_time_horizon)
    teacher_time_horizon_obst = cfg.getfloat('imitation_learning', 'teacher_time_horizon_obst', fallback=original_time_horizon_obst)
    teacher_safety_space = cfg.getfloat('imitation_learning', 'teacher_safety_space', fallback=original_safety_space)
    success_target = cfg.getint('imitation_learning', 'success_target', fallback=2000)
    max_total_episodes = cfg.getint('imitation_learning', 'max_prefill_episodes', fallback=9000)
    batch_collect_episodes = cfg.getint('imitation_learning', 'prefill_batch_episodes', fallback=64)
    patience_batches = cfg.getint('imitation_learning', 'prefill_patience_batches', fallback=10)
    logging.info(
        f"[IL-TEACHER] Applying ORCA teacher profile: neighbor_dist={teacher_neighbor_dist}, max_neighbors={teacher_max_neighbors}, "
        f"time_horizon={teacher_time_horizon}, time_horizon_obst={teacher_time_horizon_obst}, safety_space={teacher_safety_space}"
    )
    env.config.set('orca', 'neighbor_dist', str(teacher_neighbor_dist))
    env.config.set('orca', 'max_neighbors', str(teacher_max_neighbors))
    env.config.set('orca', 'time_horizon', str(teacher_time_horizon))
    env.config.set('orca', 'time_horizon_obst', str(teacher_time_horizon_obst))
    env.config.set('orca', 'safety_space', str(teacher_safety_space))
    env.configure(env.config)
    target_buffer = getattr(explorer, 'memory', None)
    if target_buffer is None and hasattr(explorer, 'replay_buffer'):
        target_buffer = explorer.replay_buffer
    if target_buffer is None:
        logging.warning("[IL] Explorer has no replay buffer reference; IL data will stay in-flight only")
    initial_buf_size = len(target_buffer) if target_buffer else 0
    logging.info(f"[IL-START] Buffer size before IL: {initial_buf_size}")
    success_trajs_all = load_offline_il_dataset(train_cfg)
    loaded_from_cache = len(success_trajs_all) > 0
    success_ratio = 1.0 if loaded_from_cache else 0.0
    total_sampled_episodes = len(success_trajs_all) if loaded_from_cache else 0
    if not loaded_from_cache:
        store_only = ('success',)
        consecutive_no_gain = 0
        log_policy_tag_func('il', 'IL-ORCA-TEACHER')
        while total_sampled_episodes < max_total_episodes and len(success_trajs_all) < success_target:
            batch = min(batch_collect_episodes, max_total_episodes - total_sampled_episodes)
            explorer.run_k_episodes(batch, 'il', update_memory=True, show_tqdm=False,
                                    imitation_learning=True, force_joint_state_policy=True,
                                    store_on=store_only)
            new_success = list(getattr(explorer, '_last_trajectories', []) or [])
            if new_success:
                success_trajs_all.extend(new_success)
                consecutive_no_gain = 0
            else:
                consecutive_no_gain += 1
            total_sampled_episodes += batch
            success_ratio = len(success_trajs_all) / max(1, total_sampled_episodes)
            logging.info(
                f"[IL-COLLECT] episodes={total_sampled_episodes}/{max_total_episodes} | "
                f"success={len(success_trajs_all)}/{success_target} ({success_ratio:.2%})"
            )
            if consecutive_no_gain >= patience_batches:
                logging.warning("[IL-COLLECT] No new successes for several batches, continuing but check teacher settings")
                consecutive_no_gain = 0
    max_il_prefill = train_cfg.max_il_prefill
    if len(success_trajs_all) > max_il_prefill:
        logging.info(f"[IL-COLLECT] 限制收集数量: {len(success_trajs_all)} → {max_il_prefill}")
        success_trajs_all = success_trajs_all[:max_il_prefill]
    collected_success_count = len(success_trajs_all)
    collected_attempts = total_sampled_episodes
    il_buffer_added = 0
    if collected_success_count == 0:
        logging.error("[IL-COLLECT] ORCA teacher failed to produce any successful episodes. Aborting IL stage.")
        success_ratio = 0.0
    else:
        success_ratio = collected_success_count / max(1, collected_attempts)
        logging.info(
            f"[IL-COLLECT] Final success set: {collected_success_count} trajectories from {collected_attempts} attempts"
            f" (success_ratio={success_ratio:.2%})"
        )
        explorer._last_trajectories = success_trajs_all
        if target_buffer is not None:
            buffer_errors = 0
            for traj_data, meta in success_trajs_all:
                try:
                    states, actions, rewards = traj_data
                    states_arr = convert_state_func(states)
                    actions_arr = convert_actions_func(actions)
                    rewards_arr = np.asarray(rewards, dtype=np.float32)
                    T = len(states_arr)
                    if T == 0:
                        continue
                    dones = np.zeros(T, dtype=bool)
                    dones[-1] = True
                    timeouts = np.zeros(T, dtype=bool)
                    event_token = (meta.get("event", "") or meta.get('event', '') or '').lower()
                    if 'timeout' in event_token:
                        timeouts[-1] = True
                    meta = meta_from_event_func(meta, event_token, T)
                    target_buffer.push_episode(
                        states=states_arr,
                        rewards=rewards_arr,
                        dones=dones,
                        timeouts=timeouts,
                        meta=meta,
                        actions_continuous=actions_arr
                    )
                    il_buffer_added += 1
                except Exception as e:
                    buffer_errors += 1
                    logging.warning(f"[IL-BUFFER] Failed to push offline trajectory: {e}")
            if buffer_errors:
                logging.warning(f"[IL-BUFFER] Skipped {buffer_errors} offline trajectories due to push errors")
            logging.info(f"[IL-BUFFER] Prefilled replay buffer with {il_buffer_added} IL episodes")
    final_buf_size = len(target_buffer) if target_buffer else 0
    il_data_added = final_buf_size - initial_buf_size
    logging.info(f"[IL-BUFFER] Buffer after IL: {final_buf_size} (added {il_data_added})")
    if target_buffer and hasattr(target_buffer, 'sequences'):
        pool_timeout = sum(1 for ep in target_buffer.sequences if ep.get('meta', {}).get('flag_timeout', 0) == 1)
        pool_total = len(target_buffer.sequences)
        logging.info(f"[IL-BUFFER] timeout episodes in pool: {pool_timeout}/{pool_total}")
        il_source_count = sum(1 for ep in target_buffer.sequences if ep.get('meta', {}).get('source') == 'IL')
        logging.info(f"[IL-VERIFY] Episodes with source='IL': {il_source_count}/{pool_total}")
        if il_source_count < pool_total:
            logging.warning(f"[IL-VERIFY] ⚠️ Only {il_source_count}/{pool_total} episodes have source='IL'!")
            for i in range(min(5, pool_total)):
                meta = target_buffer.sequences[i].get('meta', {})
                source = meta.get('source', '<NONE>')
                event = meta.get('event', '<NONE>')[:20]
                logging.warning(f"[IL-VERIFY]   [{i}] source='{source}' event='{event}'")
    env.config.set('orca', 'neighbor_dist', str(original_neighbor_dist))
    env.config.set('orca', 'max_neighbors', str(original_max_neighbors))
    env.config.set('orca', 'time_horizon', str(original_time_horizon))
    env.config.set('orca', 'time_horizon_obst', str(original_time_horizon_obst))
    env.config.set('orca', 'safety_space', str(original_safety_space))
    env.configure(env.config)
    logging.info("[IL-TEACHER] Restored ORCA parameters for RL phase")
    if collected_success_count == 0:
        return {'success_rate': 0.0, 'transitions': max(0, il_data_added)}
    il_ckpt = os.path.join(outdir, 'il_policy.pth')
    il_epochs = cfg.getint('il_pipeline', 'bc_epochs', fallback=50)
    il_batch_size = cfg.getint('il_pipeline', 'bc_batch_size', fallback=256)
    seq_len = cfg.getint('buffer', 'seq_len', fallback=4)
    gamma = cfg.getfloat('train', 'gamma', fallback=0.99)
    use_bc = cfg.getboolean('il_pipeline', 'use_bc', fallback=True)
    if not use_bc:
        logging.info("=" * 80)
        logging.info(f"[IL-PIPELINE] Skipped BC training (use_bc=False)")
        logging.info("=" * 80)
    else:
        bc_checkpoint_exists = os.path.exists(il_ckpt)
        bc_force_retrain = cfg.getboolean('il_pipeline', 'bc_force_retrain', fallback=False)

        if bc_checkpoint_exists and not bc_force_retrain:
            logging.info("=" * 80)
            logging.info(f"[IL-BC] ✓ Found existing BC checkpoint: {il_ckpt}")
            logging.info(f"[IL-BC] bc_force_retrain=False, loading checkpoint to skip {il_epochs} epochs of training")
            logging.info("=" * 80)
            try:
                checkpoint = torch.load(il_ckpt, map_location=device, weights_only=False)
                state_dict_to_load = None
                if isinstance(checkpoint, dict):
                    if 'value' in checkpoint:
                        state_dict_to_load = checkpoint['value']
                    elif 'policy' in checkpoint:
                        state_dict_to_load = checkpoint['policy']
                    else:
                        state_dict_to_load = checkpoint
                else:
                    state_dict_to_load = checkpoint
                try:
                    policy.load_state_dict(state_dict_to_load, strict=True)
                    logging.info(f"[IL-BC] ✓ Loaded all parameters (strict mode)")
                except RuntimeError as e:
                    if "Missing key" in str(e) or "Unexpected key" in str(e):
                        logging.warning(f"[IL-BC] ⚠ Model structure mismatch, trying partial load (strict=False)")
                        logging.warning(f"[IL-BC] Error details: {str(e)[:200]}...")
                        result = policy.load_state_dict(state_dict_to_load, strict=False)
                        if result.missing_keys:
                            logging.warning(f"[IL-BC] Missing {len(result.missing_keys)} parameters (will use random init)")
                        if result.unexpected_keys:
                            logging.warning(f"[IL-BC] Ignored {len(result.unexpected_keys)} unexpected parameters")
                        logging.info(f"[IL-BC] ⚠ Partial load completed (may need retraining)")
                    else:
                        raise
                logging.info(f"[IL-BC] ✓ Successfully loaded BC weights from {il_ckpt}")
                logging.info("[IL-BC] Evaluating loaded BC model...")
                policy.eval()
                explorer.run_k_episodes(100, 'val', update_memory=False, show_tqdm=False,
                                        imitation_learning=False, force_joint_state_policy=False)
                _SUCCESS_TOKENS = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
                mamba_succ = sum(1 for _, info in explorer._last_trajectories
                                if event_token_func(info) in _SUCCESS_TOKENS)
                mamba_succ_rate = mamba_succ / max(1, len(explorer._last_trajectories))
                logging.info(f"[IL-BC] Loaded BC model success rate: {mamba_succ_rate:.1%} ({mamba_succ}/{len(explorer._last_trajectories)} episodes)")

                logging.info("=" * 80)
                logging.info(f"[IL-BC] ✓ Skipping BC training (checkpoint loaded successfully)")
                logging.info("=" * 80)
                policy.train()
            except Exception as e:
                logging.error(f"[IL-BC] ✗ Failed to load checkpoint: {e}")
                logging.info(f"[IL-BC] Will train from scratch due to loading error")
                bc_checkpoint_exists = False

        if not bc_checkpoint_exists or bc_force_retrain:
            logging.info("=" * 80)
            if bc_force_retrain and bc_checkpoint_exists:
                logging.info(f"[IL-BC] ⚠️ bc_force_retrain=True, ignoring existing checkpoint at {il_ckpt}")
            elif not bc_checkpoint_exists:
                logging.info(f"[IL-BC] ✗ No BC checkpoint found at {il_ckpt}")
            logging.info(f"[IL-BC] Starting {il_epochs} epochs of BC training from scratch")
            logging.info("=" * 80)
            run_policy_bc_training(policy, device, seq_len, il_epochs, il_batch_size, il_ckpt, explorer, gamma,
                                  success_trajs_all, train_cfg, joint34_to_tokens_func, pad_state_func, event_token_func)
            logging.info("[IL-BC] Evaluating newly trained BC model...")
            policy.eval()
            explorer.run_k_episodes(100, 'val', update_memory=False, show_tqdm=False,
                                    imitation_learning=False, force_joint_state_policy=False)
            _SUCCESS_TOKENS = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
            mamba_succ = sum(1 for _, info in explorer._last_trajectories
                            if event_token_func(info) in _SUCCESS_TOKENS)
            mamba_succ_rate = mamba_succ / max(1, len(explorer._last_trajectories))
            logging.info(f"[IL-BC] Newly trained BC model success rate: {mamba_succ_rate:.1%} ({mamba_succ}/{len(explorer._last_trajectories)} episodes)")
            policy.train()


# ============================================================
# PPO Training Pipeline
# ============================================================

def ppo_optimize_step(batch, policy, optimizer, cfg, device):
    """
    PPO单次优化步骤 - 使用PPO clip loss

    Args:
        batch: dict包含states, actions, log_probs_old, advantages, returns
        policy: MambaRL policy network
        optimizer: 统一的optimizer（包含encoder + actor + value）
        cfg: configparser.RawConfigParser
        device: torch device

    Returns:
        dict: 包含各项loss和统计信息
    """
    states = batch['states']
    actions = batch['actions']
    log_probs_old = batch['log_probs_old']
    advantages = batch['advantages']
    returns = batch['returns']

    clip_epsilon = cfg.getfloat('train', 'clip_epsilon', fallback=0.2)
    value_loss_coef = cfg.getfloat('train', 'value_loss_coef', fallback=0.5)
    entropy_coef = cfg.getfloat('train', 'entropy_coef', fallback=0.01)

    states_seq = states.unsqueeze(1)

    action_resampled, log_prob_new, features = policy.sample_action(states_seq, deterministic=False)
    features = features.squeeze(1) if features.dim() > 2 else features

    values_pred = policy.compute_value(features)

    ratio = torch.exp(log_prob_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values_pred, returns)

    entropy = -log_prob_new.mean()

    total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    total_loss.backward()

    max_grad_norm = cfg.getfloat('train', 'max_grad_norm', fallback=0.5)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)

    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'ratio_mean': ratio.mean().item(),
        'ratio_max': ratio.max().item(),
        'ratio_min': ratio.min().item(),
        'advantages_mean': advantages.mean().item(),
        'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    }


def run_ppo_rl_phase(cfg, env, policy, optimizer, explorer, device,
                     train_episodes, batch_size, save_every, args,
                     train_step_counter, start_episode=1):
    """
    PPO RL训练主循环

    Args:
        cfg: configparser.RawConfigParser
        env: 环境
        policy: 策略网络
        optimizer: 优化器
        explorer: Explorer对象
        device: 设备
        train_episodes: 总训练episode数
        batch_size: 批量大小
        save_every: 保存频率
        args: 命令行参数
        train_step_counter: 训练步数计数器
        start_episode: 起始episode

    Returns:
        float: 最佳成功率
    """
    logging.info("[PPO-RL] PPO training phase started")
    best_success = 0.0

    for ep in range(start_episode, train_episodes + 1):
        explorer.run_k_episodes(1, 'train', update_memory=True, show_tqdm=False)

        if ep % 10 == 0:
            logging.info(f"[PPO-RL] Episode {ep}/{train_episodes}")

        if ep % save_every == 0:
            save_path = os.path.join(args.outdir, f'policy_ep{ep}.pth')
            torch.save(policy.state_dict(), save_path)
            logging.info(f"[PPO-RL] Saved checkpoint to {save_path}")

    return best_success
