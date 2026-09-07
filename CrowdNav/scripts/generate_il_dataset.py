#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线IL数据集生成脚本 v5.0 (5人环境+极端环境泛化版本)

改进重点（v4.1 → v5.0）：
- ✅ 保持5人环境（human_num=5固定，baseline实验要求）
- ✅ 扩展密度范围：R=[2, 3, 4, 5, 6]（更极端的密度变化）
- ✅ 扩展速度范围：V=[0.6, 0.8, 1.0, 1.2, 1.4]（更大的速度差异）
- ✅ Seeds=20个，保持场景多样性
- ✅ 修复OOM：分块保存，避免torch.save()内存溢出

设计理念（学习规则而非pattern）：
1. 极端密度变化：
   - R=2.0 → density=0.398（极度拥挤，测试近距离避障规则）
   - R=3.0 → density=0.177（拥挤环境）
   - R=4.0 → density=0.099（正常，baseline）
   - R=5.0 → density=0.064（宽松）
   - R=6.0 → density=0.044（稀疏，测试远距离规划）
   → BC学到"广谱密度下的导航规则"

2. 极端速度变化：
   - v=0.6 → 极慢速（老人/儿童）
   - v=0.8 → 慢速
   - v=1.0 → 正常（baseline）
   - v=1.2 → 快速
   - v=1.4 → 极快速（跑步）
   → BC学到"应对极端速度的策略"

3. 组合多样性：
   - 5个半径 × 5个速度 = 25种环境配置
   - 20个seeds × 25配置 = 500组
   - 每组采集100集 → 总计50,000集（继续扩大）

输出：data/il_dataset_diverse_v5.0.pth
"""
import os
import sys
import argparse
import logging
import time
import configparser
from typing import List, Tuple, Dict, Any
import numpy as np
import torch

# 添加crowd_nav到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crowd_sim.envs import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.explorer import Explorer
from crowd_nav.contracts import init_grid_from_cfg, action_to_discrete_index, discrete_index_to_action, GRID

# ==================== 固定配置 v5.0 ====================

# 人数：固定5人（baseline实验要求）
HUMAN_NUM = 5

# ========== 完整配置（50,000集） ==========
# Seeds：扩展到20个（增加场景多样性）
SEEDS = [
    1000,          # 1K
    5000,          # 5K
    10000,         # 10K
    50000,         # 50K
    100000,        # 100K
    500000,        # 500K
    1000000,       # 1M
    5000000,       # 5M
    10000000,      # 10M
    50000000,      # 50M
    100000100,     # 100M
    200000000,     # 200M
    500000000,     # 500M
    800000000,     # 800M
    1000000000,    # 1B
    1200000000,    # 1.2B
    1500000000,    # 1.5B
    1800000000,    # 1.8B
    2000000000,    # 2B
    2100000000     # 2.1B
]

# 密度配置：circle_radius变化（5人固定，通过半径调整密度）
RADIUS_CONFIGS = [
    2.0,  # 极高密度 0.398 人/m² - 极度拥挤（测试极限避障）
    3.0,  # 高密度   0.177 人/m² - 拥挤
    4.0,  # 正常密度 0.099 人/m² - baseline
    5.0,  # 中密度   0.064 人/m² - 宽松
    6.0,  # 低密度   0.044 人/m² - 很宽松（测试远程规划）
]

# 速度配置：human_v_pref变化（行人速度）
HUMAN_V_PREF_CONFIGS = [
    0.6,  # 极慢速（老人/儿童场景）
    0.8,  # 慢速行人
    1.0,  # 正常速度 - baseline
    1.2,  # 快速行人
    1.4,  # 极快速（跑步场景）
]

# ORCA配置：保持单一balanced风格
ORCA_CONFIGS = {
    'balanced': {
        'neighbor_dist': 6.0,
        'max_neighbors': 15,
        'time_horizon': 4.5,
        'time_horizon_obst': 4.5,
        'safety_space': 0.10,
    }
}

# 采集策略
EPISODES_PER_CONFIG = 100  # 每组配置采集100集
MAX_ATTEMPTS_PER_CONFIG = 999999

# 总量计算
# 测试: 1 × 1 × 1 × 100 = 100集
# 完整: 20 seeds × 5 radii × 5 speeds × 1 orca × 100 episodes = 50,000 episodes
TOTAL_CONFIGS = len(SEEDS) * len(RADIUS_CONFIGS) * len(HUMAN_V_PREF_CONFIGS) * len(ORCA_CONFIGS)
TOTAL_EPISODES = TOTAL_CONFIGS * EPISODES_PER_CONFIG

# ==================== 日志配置 ====================
def setup_logging(log_file=None):
    """配置日志：同时输出到控制台和文件"""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M:%S',
        force=True,
        handlers=handlers
    )

# ==================== 配置加载 ====================
def load_config(path: str) -> configparser.RawConfigParser:
    """加载配置文件（支持多个config文件合并）"""
    cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)

    if path.endswith('env.config') or path.endswith('config.txt'):
        config_dir = os.path.dirname(path) if os.path.dirname(path) else './configs'
        config_files = [
            os.path.join(config_dir, 'env.config'),
            os.path.join(config_dir, 'policy.config'),
            os.path.join(config_dir, 'train.config')
        ]
        for config_file in config_files:
            if os.path.exists(config_file):
                cfg.read(config_file, encoding='utf-8')
                logging.info(f"Loaded config: {config_file}")
    else:
        cfg.read(path, encoding='utf-8')
        logging.info(f"Loaded config: {path}")

    return cfg

# ==================== 辅助函数 ====================
def _extract_event_token(info: Any) -> str:
    """提取episode结束事件标记"""
    try:
        if info is None:
            return ""
        if isinstance(info, dict):
            val = info.get("event") or info.get("Event") or info.get("status")
            if val is not None:
                return str(val).replace(" ", "_").lower().strip()
        for attr in ("event", "name", "value"):
            if hasattr(info, attr):
                try:
                    return str(getattr(info, attr)).replace(" ", "_").lower().strip()
                except:
                    pass
        s = str(info)
        if "." in s:
            s = s.split(".")[-1]
        return s.replace(" ", "_").lower().strip()
    except:
        return ""

def _is_success(info: Any) -> bool:
    """判断是否成功"""
    token = _extract_event_token(info)
    success_tokens = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
    return token in success_tokens

def collect_episodes_for_config(
    env,
    robot,
    explorer,
    seed: int,
    radius: float,
    human_v_pref: float,
    orca_name: str,
    orca_params: Dict,
    target_success: int,
    max_attempts: int
) -> Tuple[List, int, int]:
    """
    为指定(seed, radius, speed, ORCA)组合采集成功轨迹

    返回:
        (success_trajectories, total_attempts, success_count)
    """
    density = HUMAN_NUM / (np.pi * radius**2)

    msg = f"[COLLECT] seed={seed}, R={radius:.1f}, v={human_v_pref:.1f}, density={density:.3f}, ORCA={orca_name}"
    print(msg, flush=True)
    logging.info(msg)

    # 应用环境参数
    env.config.set('sim', 'human_num', str(HUMAN_NUM))
    env.config.set('sim', 'circle_radius', str(radius))
    # 设置人类速度（如果config支持）
    if env.config.has_section('humans'):
        env.config.set('humans', 'v_pref', str(human_v_pref))

    # 应用ORCA参数
    env.config.set('orca', 'neighbor_dist', str(orca_params['neighbor_dist']))
    env.config.set('orca', 'max_neighbors', str(orca_params['max_neighbors']))
    env.config.set('orca', 'time_horizon', str(orca_params['time_horizon']))
    env.config.set('orca', 'time_horizon_obst', str(orca_params['time_horizon_obst']))
    env.config.set('orca', 'safety_space', str(orca_params['safety_space']))

    env.configure(env.config)

    # 设置case_counter作为起始seed（必须在configure之后）
    if hasattr(env, 'case_counter'):
        env.case_counter['train'] = seed
        logging.info(f"[COLLECT] Set case_counter['train'] = {seed}")

    # 创建ORCA policy（和explorer.run_k_episodes一样的方式）
    from crowd_nav.policy.policy_factory import policy_factory
    from crowd_sim.envs.utils.state import JointState
    from crowd_sim.envs.utils.action import ActionXY

    orca_policy = policy_factory['orca']()
    if hasattr(orca_policy, 'set_env'):
        orca_policy.set_env(env)
    if hasattr(orca_policy, 'time_step'):
        orca_policy.time_step = getattr(env, 'time_step', 0.25)
    if hasattr(orca_policy, 'configure'):
        orca_policy.configure(env.config)

    success_trajs = []
    attempts = 0

    while len(success_trajs) < target_success and attempts < max_attempts:
        # 重置环境
        ob = env.reset(phase='il')

        states = []
        actions = []
        action_indices = []
        rewards = []
        done = False
        episode_reward = 0.0

        while not done:
            # 使用ORCA policy生成动作（和explorer一样）
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            orca_action = orca_policy.predict(state)
            vx = float(orca_action[0])
            vy = float(orca_action[1] if len(orca_action) > 1 else 0.0)
            action_idx = int(action_to_discrete_index(vx, vy))
            vx_d, vy_d = discrete_index_to_action(action_idx)
            action = ActionXY(vx_d, vy_d)

            # 记录状态和动作（修复：存储JointState而不是raw observation）
            states.append(state)  # 修复：从ob改为state，与online collection保持一致
            actions.append(action)
            # 记录离散动作索引（与训练时GRID一致）
            action_indices.append(action_idx)

            # 环境step
            ob, reward, done, info = env.step(action)
            rewards.append(reward)
            episode_reward += reward

        attempts += 1

        # 检查是否成功
        if _is_success(info):
            # 构造轨迹元数据
            meta = {
                'seed': seed,
                'human_num': HUMAN_NUM,
                'circle_radius': radius,
                'human_v_pref': human_v_pref,
                'density': density,
                'orca_config': orca_name,
                'total_reward': episode_reward,
                'steps': len(states),
                'event': _extract_event_token(info),
                'flag_success': 1,
                'flag_collision': 0,
                'flag_timeout': 0,
                'action_mode': 'discrete',
                'action_indices': action_indices,
            }

            # 保存轨迹 (states, actions, rewards, meta)
            trajectory = {
                'states': states,
                'actions': actions,
                'action_indices': action_indices,
                'rewards': rewards,
                'meta': meta
            }
            success_trajs.append(trajectory)

            # 每10集显示一次进度
            if len(success_trajs) % 10 == 0 or len(success_trajs) == target_success:
                msg = (
                    f"[COLLECT] R={radius:.1f}, v={human_v_pref:.1f}, seed={seed}: "
                    f"{len(success_trajs)}/{target_success} success, "
                    f"attempts={attempts}, "
                    f"success_rate={len(success_trajs)/attempts:.1%}"
                )
                print(msg, flush=True)
                logging.info(msg)

    success_rate = len(success_trajs) / max(1, attempts)
    logging.info(
        f"[COLLECT] DONE R={radius:.1f}, v={human_v_pref:.1f}, seed={seed}: "
        f"{len(success_trajs)} successes from {attempts} attempts "
        f"({success_rate:.2%})"
    )

    return success_trajs, attempts, len(success_trajs)

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description='Generate offline IL dataset v5.0')
    parser.add_argument('--env-config', type=str, default='crowd_nav/configs/env.config')
    parser.add_argument('--output', type=str, default='data/il_dataset_diverse_v5.0.pth')
    parser.add_argument('--log', type=str, default=None, help='Log file path')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    args = parser.parse_args()

    if args.log is None:
        args.log = args.output.replace('.pth', '.log')

    setup_logging(args.log)

    banner = "=" * 80
    print(banner, flush=True)
    print("离线IL数据集生成 [VERSION: 4.1-DYNAMIC-DENSITY-SPEED]", flush=True)
    print(f"5人环境 + 动态密度/速度 + 扩展Seeds", flush=True)
    print(banner, flush=True)
    print(f"\n人数: {HUMAN_NUM}人（固定）", flush=True)
    print(f"Seeds: {len(SEEDS)}个", flush=True)
    print(f"半径配置: {RADIUS_CONFIGS}", flush=True)
    print(f"速度配置: {HUMAN_V_PREF_CONFIGS}", flush=True)
    print(f"ORCA: {list(ORCA_CONFIGS.keys())}", flush=True)
    print(f"\n配置组合: {len(SEEDS)} seeds × {len(RADIUS_CONFIGS)} radii × {len(HUMAN_V_PREF_CONFIGS)} speeds = {TOTAL_CONFIGS}", flush=True)
    print(f"每组episodes: {EPISODES_PER_CONFIG}", flush=True)
    print(f"预期总episodes: {TOTAL_EPISODES}", flush=True)
    print(f"\nOutput: {args.output}", flush=True)
    print(f"Log: {args.log}", flush=True)
    print(banner, flush=True)

    # 打印密度预览
    print(f"\n密度预览（5人固定）:", flush=True)
    for r in RADIUS_CONFIGS:
        d = HUMAN_NUM / (np.pi * r**2)
        print(f"  R={r:.1f} → density={d:.3f} 人/m²", flush=True)
    print(banner, flush=True)
    print(flush=True)

    logging.info("=" * 80)
    logging.info("离线IL数据集生成 v4.1")
    logging.info("=" * 80)
    logging.info(f"Human: {HUMAN_NUM} (固定)")
    logging.info(f"Seeds: {len(SEEDS)}")
    logging.info(f"Radii: {RADIUS_CONFIGS}")
    logging.info(f"Speeds: {HUMAN_V_PREF_CONFIGS}")
    logging.info(f"Episodes per config: {EPISODES_PER_CONFIG}")
    logging.info(f"Total combinations: {len(SEEDS)} × {len(ORCA_CONFIGS)} = {len(SEEDS) * len(ORCA_CONFIGS)}")
    logging.info(f"Expected total episodes: {len(SEEDS) * len(ORCA_CONFIGS) * EPISODES_PER_CONFIG}")
    logging.info(f"Output file: {args.output}")
    logging.info("=" * 80)

    # 设备
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 加载配置（合并env.config, policy.config, train.config）
    config = load_config(args.env_config)

    # 仅从 env.config 的 [policy] 初始化动作网格（SSOT）
    env_cfg_raw = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    env_cfg_raw.read(args.env_config, encoding='utf-8')
    if not env_cfg_raw.has_section('policy'):
        logging.error(f"[GRID] Missing [policy] section in {args.env_config}")
        sys.exit(1)
    init_grid_from_cfg(env_cfg_raw)
    logging.info(f"[GRID] Initialized from env.config: {GRID}")

    # 创建环境
    env = CrowdSim()
    env.configure(config)

    # 创建Robot（使用ORCA policy生成示范）
    robot = Robot(config, 'robot')
    env.set_robot(robot)

    # 设置Robot使用ORCA policy
    robot.set_policy(robot.policy)

    # 创建Explorer（不需要memory，我们手动收集）
    explorer = Explorer(env, robot, device, memory=None)

    # ========== 分块保存配置（避免OOM）==========
    CHUNK_SIZE = 5000  # 每5000集保存一次
    chunk_files = []
    current_chunk = []
    current_chunk_id = 0

    # 收集所有轨迹
    all_trajectories = []  # 暂时保留用于统计，最后清空
    total_attempts = 0
    total_successes = 0
    config_stats = []

    start_time = time.time()
    completed_configs = 0

    def save_chunk(chunk_data, chunk_id, output_base):
        """保存一个chunk到磁盘"""
        chunk_path = output_base.replace('.pth', f'_chunk{chunk_id}.pth')
        print(f"\n{'='*80}", flush=True)
        print(f"[CHUNK-{chunk_id}] 保存chunk {chunk_id} ({len(chunk_data)} episodes)...", flush=True)
        logging.info(f"[CHUNK-{chunk_id}] 保存chunk {chunk_id} ({len(chunk_data)} episodes)")

        import gc
        gc.collect()

        try:
            torch.save({'trajectories': chunk_data}, chunk_path,
                      pickle_protocol=4, _use_new_zipfile_serialization=False)
            file_size = os.path.getsize(chunk_path) / 1024 / 1024
            print(f"[CHUNK-{chunk_id}] ✓ 已保存: {chunk_path} ({file_size:.1f}MB)", flush=True)
            logging.info(f"[CHUNK-{chunk_id}] ✓ 已保存: {file_size:.1f}MB")

            # 立即释放chunk数据内存
            del chunk_data
            gc.collect()

            return chunk_path
        except Exception as e:
            print(f"[CHUNK-{chunk_id}] ✗ 保存失败: {e}", flush=True)
            logging.error(f"[CHUNK-{chunk_id}] 保存失败: {e}")
            raise

    # 遍历所有配置组合
    for seed in SEEDS:
        for radius in RADIUS_CONFIGS:
            for human_v_pref in HUMAN_V_PREF_CONFIGS:
                for orca_name, orca_params in ORCA_CONFIGS.items():
                    config_start = time.time()

                    trajs, attempts, successes = collect_episodes_for_config(
                        env=env,
                        robot=robot,
                        explorer=explorer,
                        seed=seed,
                        radius=radius,
                        human_v_pref=human_v_pref,
                        orca_name=orca_name,
                        orca_params=orca_params,
                        target_success=EPISODES_PER_CONFIG,
                        max_attempts=MAX_ATTEMPTS_PER_CONFIG
                    )

                    current_chunk.extend(trajs)
                    all_trajectories.extend(trajs)  # 仅用于最终统计
                    total_attempts += attempts
                    total_successes += successes
                    completed_configs += 1

                    # ========== 检查是否需要保存chunk ==========
                    if len(current_chunk) >= CHUNK_SIZE:
                        chunk_path = save_chunk(current_chunk, current_chunk_id, args.output)
                        chunk_files.append(chunk_path)
                        current_chunk = []  # 清空当前chunk
                        current_chunk_id += 1

                    # 记录配置统计
                    config_success_rate = successes / max(1, attempts)
                    density = HUMAN_NUM / (np.pi * radius**2)

                    config_stats.append({
                        'seed': seed,
                        'radius': radius,
                        'human_v_pref': human_v_pref,
                        'density': density,
                        'orca': orca_name,
                        'successes': successes,
                        'attempts': attempts,
                        'success_rate': config_success_rate,
                        'time': time.time() - config_start
                    })

                    config_time = time.time() - config_start

                    # 判断成功率等级
                    if config_success_rate >= 0.90:
                        rating = "✓"
                    elif config_success_rate >= 0.80:
                        rating = "○"
                    else:
                        rating = "✗"

                    msg = (
                        f"[PROGRESS] {rating} | seed={seed}, R={radius:.1f}, v={human_v_pref:.1f}, d={density:.3f} | "
                        f"{successes}/{attempts} ({config_success_rate:.1%}) | "
                        f"time={config_time:.1f}s | ({completed_configs}/{TOTAL_CONFIGS})"
                    )
                    print(msg, flush=True)
                    logging.info(msg)

                    if completed_configs % 10 == 0:
                        overall_rate = total_successes/max(1, total_attempts)
                        msg = f"[OVERALL] {total_successes}/{TOTAL_EPISODES} episodes, rate={overall_rate:.1%}"
                        print(msg, flush=True)
                        logging.info(msg)

    # ========== 保存最后一个chunk（如果有剩余）==========
    if len(current_chunk) > 0:
        chunk_path = save_chunk(current_chunk, current_chunk_id, args.output)
        chunk_files.append(chunk_path)
        current_chunk = []

    total_time = time.time() - start_time
    overall_success_rate = total_successes / max(1, total_attempts)

    # ========== 完整统计报告 ==========
    print("\n" + "=" * 80, flush=True)
    print("数据集生成完成！", flush=True)
    print("=" * 80, flush=True)

    # 总体统计
    print(f"\n【总体统计】", flush=True)
    print(f"  总采集时间: {total_time:.1f}s ({total_time/60:.1f}min)", flush=True)
    print(f"  总尝试次数: {total_attempts}", flush=True)
    print(f"  成功轨迹数: {total_successes} / {TOTAL_EPISODES}", flush=True)
    print(f"  整体成功率: {overall_success_rate:.2%}", flush=True)

    if overall_success_rate >= 0.85:
        rating = "✓ 优秀"
    elif overall_success_rate >= 0.75:
        rating = "○ 良好"
    else:
        rating = "✗ 警告"
    print(f"  质量评价: {rating}", flush=True)

    # 按半径统计
    print(f"\n【按半径统计（密度变化）】", flush=True)
    radius_stats = {}
    for stat in config_stats:
        r = stat['radius']
        if r not in radius_stats:
            radius_stats[r] = {'successes': 0, 'attempts': 0}
        radius_stats[r]['successes'] += stat['successes']
        radius_stats[r]['attempts'] += stat['attempts']

    for r in RADIUS_CONFIGS:
        s = radius_stats[r]
        rate = s['successes'] / max(1, s['attempts'])
        density = HUMAN_NUM / (np.pi * r**2)
        print(f"  R={r:.1f} (d={density:.3f}): {s['successes']:5d}/{s['attempts']:5d} ({rate:.1%})", flush=True)

    # 按速度统计
    print(f"\n【按速度统计】", flush=True)
    speed_stats = {}
    for stat in config_stats:
        v = stat['human_v_pref']
        if v not in speed_stats:
            speed_stats[v] = {'successes': 0, 'attempts': 0}
        speed_stats[v]['successes'] += stat['successes']
        speed_stats[v]['attempts'] += stat['attempts']

    for v in HUMAN_V_PREF_CONFIGS:
        s = speed_stats[v]
        rate = s['successes'] / max(1, s['attempts'])
        print(f"  v={v:.1f}: {s['successes']:5d}/{s['attempts']:5d} ({rate:.1%})", flush=True)

    # 最差和最佳配置
    config_stats_sorted = sorted(config_stats, key=lambda x: x['success_rate'])
    worst = config_stats_sorted[0]
    best = config_stats_sorted[-1]

    print(f"\n【最差配置】", flush=True)
    print(f"  seed={worst['seed']}, R={worst['radius']:.1f}, v={worst['human_v_pref']:.1f}: {worst['successes']}/{worst['attempts']} ({worst['success_rate']:.1%})", flush=True)

    print(f"\n【最佳配置】", flush=True)
    print(f"  seed={best['seed']}, R={best['radius']:.1f}, v={best['human_v_pref']:.1f}: {best['successes']}/{best['attempts']} ({best['success_rate']:.1%})", flush=True)

    print("\n" + "=" * 80, flush=True)

    # 同时写入日志
    logging.info("=" * 80)
    logging.info("数据集生成完成！")
    logging.info("=" * 80)
    logging.info(f"总采集时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    logging.info(f"总尝试次数: {total_attempts}")
    logging.info(f"成功轨迹数: {total_successes}")
    logging.info(f"整体成功率: {overall_success_rate:.2%}")
    logging.info(f"质量评价: {rating}")

    # ========== 保存元数据文件（不包含trajectories，避免OOM）==========
    print("\n" + "=" * 80, flush=True)
    print("保存元数据文件...", flush=True)
    logging.info("保存元数据文件...")

    # 统计信息（从all_trajectories计算，然后释放）
    num_trajs = len(all_trajectories)
    avg_traj_size = sum(len(t['states']) for t in all_trajectories[:min(100, num_trajs)]) / min(100, num_trajs) if num_trajs > 0 else 0

    # 元数据（不包含轨迹数据，只有索引和统计）
    metadata = {
        'chunk_files': chunk_files,  # chunk文件列表
        'num_chunks': len(chunk_files),
        'total_trajectories': num_trajs,
        'config': {
            'human_num': HUMAN_NUM,
            'seeds': SEEDS,
            'radius_configs': RADIUS_CONFIGS,
            'human_v_pref_configs': HUMAN_V_PREF_CONFIGS,
            'orca_configs': ORCA_CONFIGS,
            'episodes_per_config': EPISODES_PER_CONFIG,
            'action_grid': GRID,
            'action_mode': 'discrete',
        },
        'stats': {
            'total_attempts': total_attempts,
            'total_successes': total_successes,
            'success_rate': overall_success_rate,
            'collection_time': total_time,
            'avg_traj_length': avg_traj_size,
            'config_stats': config_stats,
        },
        'version': '5.0_chunked',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 立即释放all_trajectories以节省内存
    del all_trajectories
    import gc
    gc.collect()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 保存元数据（很小，不会OOM）
    try:
        print(f"  保存元数据...", flush=True)
        torch.save(metadata, args.output, pickle_protocol=4, _use_new_zipfile_serialization=False)

        file_size = os.path.getsize(args.output) / 1024 / 1024
        print(f"✓ 元数据已保存: {args.output} ({file_size:.1f} MB)", flush=True)
        logging.info(f"✓ 元数据已保存: {file_size:.1f} MB")

        # 计算总文件大小
        total_size = file_size
        for chunk_file in chunk_files:
            total_size += os.path.getsize(chunk_file) / 1024 / 1024

        print(f"\n数据集文件汇总:", flush=True)
        print(f"  元数据文件: {args.output} ({file_size:.1f} MB)", flush=True)
        print(f"  Chunk文件数: {len(chunk_files)}", flush=True)
        for i, cf in enumerate(chunk_files):
            cf_size = os.path.getsize(cf) / 1024 / 1024
            print(f"    chunk{i}: {os.path.basename(cf)} ({cf_size:.1f} MB)", flush=True)
        print(f"  总大小: {total_size:.1f} MB", flush=True)
        print(f"  总轨迹数: {num_trajs}", flush=True)
        logging.info(f"数据集总大小: {total_size:.1f} MB, {num_trajs}条轨迹")

    except Exception as e:
        print(f"✗ 元数据保存失败: {e}", flush=True)
        logging.error(f"元数据保存失败: {e}")
        raise

    msg = f"日志文件: {args.log}"
    print(msg, flush=True)
    logging.info(msg)

    print("=" * 80, flush=True)
    logging.info("=" * 80)

    # 使用说明
    print("\n【使用说明】", flush=True)
    print(f"训练时会自动检测并加载此数据集，无需修改代码。", flush=True)
    print(f"如果数据集质量不满意，可以调整ORCA参数重新生成。", flush=True)
    print(flush=True)

if __name__ == '__main__':
    main()
