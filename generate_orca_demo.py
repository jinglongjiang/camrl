#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_orca_demo.py
用 CrowdSim + ORCA 生成 IL 轨迹（npz）
"""

import argparse
import configparser
import numpy as np
from tqdm import trange
import gym

def load_cfg(path: str) -> configparser.ConfigParser:
    # 关键：支持行内注释“; …”和“# …”
    cfg = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=(';', '#'),
        strict=False
    )
    with open(path, 'r', encoding='utf-8') as f:
        cfg.read_file(f)
    return cfg

def main():
    parser = argparse.ArgumentParser('Generate ORCA Demo Data')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca', help='expert policy name')
    parser.add_argument('--num_eps', type=int, default=1000)
    parser.add_argument('--output', type=str, default='orca_demos_seq.npz')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    # 延迟导入，避免循环依赖
    from crowd_sim.envs.policy.policy_factory import policy_factory
    from crowd_sim.envs.utils.robot import Robot

    # 读取配置（支持行内注释）
    env_config = load_cfg(args.env_config)
    policy_config = load_cfg(args.policy_config)

    # 创建环境并配置
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    # 有些版本用 env.phase 控制随机/评估，这里保守设置为 train
    try:
        setattr(env, 'phase', 'train')
    except Exception:
        pass

    # 构造 ORCA 教师
    policy = policy_factory[args.policy]()
    policy.configure(policy_config)

    # 绑定机器人
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    # 打印关键信息
    try:
        ts = env.time_step
    except Exception:
        ts = None
    print(f"[INFO] Env ready: human_num={env.human_num}, time_step={ts}, policy={policy.name}")

    rng = np.random.RandomState(args.seed) if args.seed is not None else np.random

    obs_seqs, act_seqs = [], []

    for ep in trange(args.num_eps, desc="Generating ORCA Demos"):
        # 每个 episode 的随机种子
        if args.seed is not None:
            np.random.seed(args.seed + ep)

        # 如果你的第二版 ORCA里实现了 randomize_teacher()，这里调用一下
        if hasattr(policy, "randomize_teacher"):
            try:
                policy.randomize_teacher()
            except Exception:
                pass

        # 重置环境
        reset_out = env.reset()
        # 兼容 gymnasium(reset)->(obs, info)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            _obs0, _info0 = reset_out

        done = False
        ep_obs, ep_act = [], []

        while not done:
            # 组装观测：robot full + humans observable
            obs_list = [h.get_observable_state() for h in env.humans]
            action = env.robot.act(obs_list)

            robot_state = env.robot.get_full_state().to_array()
            human_states = np.concatenate([h.get_observable_state().to_array() for h in env.humans]) \
                           if len(env.humans) > 0 else np.zeros(0, dtype=np.float32)
            obs_all = np.concatenate([robot_state, human_states]).astype(np.float32)
            ep_obs.append(obs_all)

            if hasattr(action, 'vx'):
                act_all = np.array([action.vx, action.vy], dtype=np.float32)
            else:
                # 保险：ActionRot 或其他类型
                act_all = np.array(getattr(action, 'to_tuple', lambda: (0.0, 0.0))(), dtype=np.float32)
            ep_act.append(act_all)

            step_result = env.step(action)
            # 兼容老 gym 和 gymnasium
            if isinstance(step_result, tuple) and len(step_result) == 4:
                _, _, done, _ = step_result
            elif isinstance(step_result, tuple) and len(step_result) == 5:
                _, _, terminated, truncated, _ = step_result
                done = bool(terminated) or bool(truncated)
            else:
                raise RuntimeError(f"Unexpected env.step() output length: {len(step_result)}")

        obs_seqs.append(np.stack(ep_obs, axis=0))
        act_seqs.append(np.stack(ep_act, axis=0))

    # 保存为变长序列（object）
    np.savez(args.output,
             obs=np.array(obs_seqs, dtype=object),
             act=np.array(act_seqs, dtype=object))
    print(f"\n✔ Saved {len(obs_seqs)} episodes → {args.output}")

if __name__ == '__main__':
    main()
