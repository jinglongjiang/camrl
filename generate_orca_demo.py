#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_orca_demo.py
生成 GDBN 训练数据（npz，34D 序列）。

覆盖范围与 IL 数据集对齐，避免 Mamba 和 GDBN 训练分布偏差：
  - 3 种圆半径（R=2,4,6）×  2 种行人速度（v=0.8,1.2）= 6 配置
  - 每配置 500 集 × 多策略行人（ORCA 50%/SFM 30%/CV 15%/RW 5%）
  - 机器人始终用 ORCA（保证轨迹质量）
  - 最近 5 人取 34D，向后兼容 GDBN

默认总量：3000 集（约 100K 步），可通过 --eps_per_cfg 调整。
"""

import argparse
import configparser
import copy
import numpy as np
from tqdm import tqdm
import gym

# 多密度 × 多速度配置（与 IL 数据集对齐）
_RADIUS_CONFIGS  = [2.0, 4.0, 6.0]   # 高/正常/低密度
_VPREF_CONFIGS   = [0.8, 1.2]         # 慢/快行人

_HUMAN_POL_NAMES   = ['orca', 'social_force', 'cv', 'random_walk']
_HUMAN_POL_WEIGHTS = [0.50,   0.30,           0.15, 0.05]


def load_cfg(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=(';', '#'),
        strict=False
    )
    with open(path, 'r', encoding='utf-8') as f:
        cfg.read_file(f)
    return cfg


def _assign_human_policies(env, policy_factory, env_cfg):
    for human in env.humans:
        choice = np.random.choice(_HUMAN_POL_NAMES, p=_HUMAN_POL_WEIGHTS)
        pol = policy_factory[choice]()
        pol.time_step = env.time_step
        if hasattr(pol, 'configure'):
            pol.configure(env_cfg)
        human.set_policy(pol)


def _collect_episode(env, policy_factory, env_cfg, seed: int):
    """收集一条完整 episode，返回 (obs_seq, act_seq)。"""
    np.random.seed(seed)

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        pass  # gymnasium 返回 (obs, info)，直接丢弃

    _assign_human_policies(env, policy_factory, env_cfg)

    ep_obs, ep_act = [], []
    done = False

    while not done:
        obs_list = [h.get_observable_state() for h in env.humans]
        action = env.robot.act(obs_list)

        # 34D: robot(9) + 最近5人(5×5)
        robot_arr = env.robot.get_full_state().to_array()
        humans_sorted = sorted(
            env.humans,
            key=lambda h: (h.px - env.robot.px)**2 + (h.py - env.robot.py)**2
        )[:5]
        pad = max(0, 5 - len(humans_sorted))
        h_arrays  = [h.get_observable_state().to_array() for h in humans_sorted]
        h_arrays += [np.zeros(5, dtype=np.float32)] * pad
        obs_34 = np.concatenate([robot_arr, np.concatenate(h_arrays)]).astype(np.float32)
        ep_obs.append(obs_34)

        if hasattr(action, 'vx'):
            ep_act.append(np.array([action.vx, action.vy], dtype=np.float32))
        else:
            ep_act.append(np.zeros(2, dtype=np.float32))

        step_result = env.step(action)
        if len(step_result) == 5:
            _, _, terminated, truncated, _ = step_result
            done = bool(terminated) or bool(truncated)
        else:
            _, _, done, _ = step_result

    return np.stack(ep_obs), np.stack(ep_act)


def main():
    parser = argparse.ArgumentParser('Generate GDBN Demo Data (multi-density/speed)')
    parser.add_argument('--env_config',    type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy',        type=str, default='orca')
    parser.add_argument('--eps_per_cfg',   type=int, default=500,
                        help='Episodes per (radius, v_pref) config (default 500 → 3000 total)')
    parser.add_argument('--output',        type=str, default='orca_demos_seq.npz')
    parser.add_argument('--seed',          type=int, default=42)
    args = parser.parse_args()

    from crowd_sim.envs.policy.policy_factory import policy_factory
    from crowd_sim.envs.utils.robot import Robot

    env_cfg_orig    = load_cfg(args.env_config)
    policy_cfg      = load_cfg(args.policy_config)

    total_configs   = len(_RADIUS_CONFIGS) * len(_VPREF_CONFIGS)
    total_eps       = total_configs * args.eps_per_cfg
    print(f"[INFO] Configs: {len(_RADIUS_CONFIGS)} radii × {len(_VPREF_CONFIGS)} speeds = {total_configs}")
    print(f"[INFO] Episodes: {args.eps_per_cfg}/cfg × {total_configs} = {total_eps} total")

    obs_seqs, act_seqs = [], []
    global_ep = 0

    for radius in _RADIUS_CONFIGS:
        for vpref in _VPREF_CONFIGS:
            # 为每个配置克隆 config 并修改参数
            env_cfg = copy.deepcopy(env_cfg_orig)
            env_cfg.set('sim',    'circle_radius', str(radius))
            env_cfg.set('humans', 'v_pref',        str(vpref))

            env = gym.make('CrowdSim-v0')
            env.configure(env_cfg)
            env.phase = 'train'

            robot = Robot(env_cfg, 'robot')
            expert = policy_factory[args.policy]()
            expert.configure(policy_cfg)
            robot.set_policy(expert)
            env.set_robot(robot)

            desc = f"R={radius:.1f} v={vpref:.1f}"
            pbar = tqdm(range(args.eps_per_cfg), desc=desc, leave=False)
            collected = 0
            attempt   = 0

            while collected < args.eps_per_cfg:
                seed_ep = args.seed + global_ep * 1000 + attempt
                attempt += 1
                try:
                    obs_seq, act_seq = _collect_episode(env, policy_factory, env_cfg, seed_ep)
                    obs_seqs.append(obs_seq)
                    act_seqs.append(act_seq)
                    collected += 1
                    global_ep += 1
                    pbar.update(1)
                except Exception as e:
                    pass  # 极少数初始化失败的 episode 直接跳过
            pbar.close()
            print(f"  ✓ R={radius:.1f}, v={vpref:.1f}: {collected} eps collected")

    np.savez(
        args.output,
        obs=np.array(obs_seqs, dtype=object),
        act=np.array(act_seqs, dtype=object),
    )
    print(f"\n✔ Saved {len(obs_seqs)} episodes → {args.output}")
    print(f"  Configs: R={_RADIUS_CONFIGS}  v={_VPREF_CONFIGS}")
    print(f"  Human policies: {dict(zip(_HUMAN_POL_NAMES, _HUMAN_POL_WEIGHTS))}")


if __name__ == '__main__':
    main()
