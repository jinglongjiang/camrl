#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速种子测试脚本 - 不需要完整训练，直接测试BC模型在不同seed下的表现

原理：
1. 加载已有的BC checkpoint (il_policy.pth)
2. 用不同的环境随机种子测试BC模型
3. 找到表现最稳定、成功率最高的环境种子
4. 这个种子将用于后续RL训练

优势：
- 速度快：每个seed只需要跑100个episodes（<1分钟）
- 不需要训练：直接用现有BC模型
- 能找到"对BC友好"的环境seed，RL阶段会继承这个优势

用法：
    python quick_seed_test.py --checkpoint runs/mamba_vl/il_policy.pth
    python quick_seed_test.py --checkpoint runs/mamba_vl/il_policy.pth --num_seeds 20
"""

import os
import sys
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def test_seed_with_bc_model(seed, policy, env, num_test_episodes=100):
    """
    用固定的BC模型测试指定环境种子

    Args:
        seed: 环境随机种子
        policy: BC policy model
        env: CrowdSim environment
        num_test_episodes: 测试episodes数量

    Returns:
        dict: 性能指标
    """
    # 设置环境种子
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_rewards = []
    episode_times = []

    # 设置policy为eval模式（确定性）
    policy.model.eval()

    with torch.no_grad():
        for ep in range(num_test_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            steps = 0

            while not (done or truncated):
                # BC模型确定性输出
                action = policy.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                if steps >= 100:  # 防止无限循环
                    truncated = True

            # 统计结果
            event = info.get('event', 'unknown').lower() if isinstance(info, dict) else 'unknown'

            if 'success' in event or 'reach' in event:
                success_count += 1
            elif 'collision' in event:
                collision_count += 1
            elif 'timeout' in event:
                timeout_count += 1
            else:
                timeout_count += 1  # 未知事件归为timeout

            total_rewards.append(episode_reward)
            episode_times.append(steps * env.time_step)

    success_rate = success_count / num_test_episodes
    collision_rate = collision_count / num_test_episodes
    timeout_rate = timeout_count / num_test_episodes
    avg_reward = np.mean(total_rewards)
    avg_time = np.mean(episode_times)

    return {
        'seed': seed,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'avg_reward': avg_reward,
        'avg_time': avg_time,
        'success_count': success_count,
        'collision_count': collision_count,
        'timeout_count': timeout_count
    }


def load_bc_model(checkpoint_path, device):
    """加载BC checkpoint"""
    from crowd_nav.policy.policy_factory import policy_factory
    from crowd_sim.envs import CrowdSim
    from crowd_sim.envs.utils.robot import Robot
    from configparser import ConfigParser

    logging.info(f"Loading BC checkpoint from: {checkpoint_path}")

    # 加载配置
    config_dir = Path(checkpoint_path).parent.parent / 'configs'
    if not config_dir.exists():
        config_dir = Path('./configs')

    env_config = config_dir / 'env.config'
    policy_config = config_dir / 'policy.config'

    if not env_config.exists() or not policy_config.exists():
        raise FileNotFoundError(f"Config files not found in {config_dir}")

    # 创建环境
    env_cfg = ConfigParser()
    env_cfg.read(env_config, encoding='utf-8')

    env = CrowdSim()
    env.configure(env_cfg)

    # 创建robot和policy
    robot = Robot(env_cfg, 'robot')

    policy_cfg = ConfigParser()
    policy_cfg.read(policy_config, encoding='utf-8')

    # 从policy.config中获取实际的policy名称（而不是env.config中的"none"）
    policy_name = policy_cfg.get('policy', 'name', fallback='mamba_rl')
    logging.info(f"Using policy: {policy_name}")

    policy = policy_factory[policy_name](env_cfg, policy_cfg)
    policy.set_device(device)
    robot.set_policy(policy)
    env.set_robot(robot)

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 兼容多种checkpoint格式
    if isinstance(checkpoint, dict):
        if 'policy' in checkpoint:
            state_dict = checkpoint['policy']
        elif 'value' in checkpoint:
            state_dict = checkpoint['value']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 加载模型
    try:
        policy.model.load_state_dict(state_dict, strict=True)
        logging.info("✅ BC model loaded successfully (strict mode)")
    except RuntimeError as e:
        logging.warning(f"⚠️  Strict loading failed, trying non-strict mode")
        result = policy.model.load_state_dict(state_dict, strict=False)
        logging.info(f"✅ BC model loaded (non-strict): missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    policy.model.eval()

    return policy, env


def main():
    parser = argparse.ArgumentParser(description='Quick seed testing with BC model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to BC checkpoint (il_policy.pth)')
    parser.add_argument('--num_seeds', type=int, default=16,
                       help='Number of seeds to test (default: 16)')
    parser.add_argument('--test_episodes', type=int, default=100,
                       help='Episodes per seed (default: 100)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='./seed_test_results.json',
                       help='Output JSON file')

    args = parser.parse_args()

    device = torch.device(args.device)

    logging.info("=" * 80)
    logging.info("🔍 Quick Seed Testing with BC Model")
    logging.info("=" * 80)
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Testing {args.num_seeds} seeds × {args.test_episodes} episodes each")
    logging.info(f"Device: {device}")
    logging.info("=" * 80)

    # 加载BC模型
    policy, env = load_bc_model(args.checkpoint, device)

    # 生成候选种子
    candidate_seeds = [
        42, 43, 44, 45,  # 基础种子附近
        100, 200, 300, 400,  # 100倍间隔
        1000, 2000, 3000, 4000,  # 1000倍间隔
        123, 456, 777, 1234, 2023, 2024, 3407, 8888  # 常用幸运数字
    ]

    # 截取或补充到num_seeds
    if len(candidate_seeds) < args.num_seeds:
        np.random.seed(42)
        additional = list(np.random.randint(1, 10000, size=args.num_seeds - len(candidate_seeds)))
        candidate_seeds.extend(additional)

    candidate_seeds = candidate_seeds[:args.num_seeds]

    logging.info(f"\nTesting seeds: {candidate_seeds}\n")

    # 测试每个seed
    results = []

    for i, seed in enumerate(candidate_seeds, 1):
        logging.info(f"[{i}/{args.num_seeds}] Testing seed {seed}...")

        result = test_seed_with_bc_model(seed, policy, env, args.test_episodes)
        results.append(result)

        logging.info(f"  Success: {result['success_rate']:.1%} ({result['success_count']}/{args.test_episodes}), "
                    f"Collision: {result['collision_rate']:.1%}, "
                    f"Timeout: {result['timeout_rate']:.1%}, "
                    f"Avg Reward: {result['avg_reward']:.2f}")

    # 按成功率排序
    results.sort(key=lambda x: x['success_rate'], reverse=True)

    # 打印结果表格
    print("\n" + "=" * 80)
    print("📊 Seed Test Results (sorted by success rate)")
    print("=" * 80)
    print("\nRank | Seed  | Success | Collision | Timeout | Avg Reward | Avg Time")
    print("-" * 80)

    for rank, result in enumerate(results, 1):
        print(f"{rank:4d} | {result['seed']:5d} | "
              f"{result['success_rate']:7.1%} | "
              f"{result['collision_rate']:9.1%} | "
              f"{result['timeout_rate']:7.1%} | "
              f"{result['avg_reward']:10.2f} | "
              f"{result['avg_time']:8.2f}s")

    # 保存结果
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"\n✅ Results saved to: {args.output}")

    # 推荐最佳seed
    best_result = results[0]
    print("\n" + "=" * 80)
    print("🏆 RECOMMENDED BEST SEED")
    print("=" * 80)
    print(f"Seed: {best_result['seed']}")
    print(f"Success Rate: {best_result['success_rate']:.1%} ({best_result['success_count']}/{args.test_episodes})")
    print(f"Collision Rate: {best_result['collision_rate']:.1%}")
    print(f"Timeout Rate: {best_result['timeout_rate']:.1%}")
    print(f"Avg Reward: {best_result['avg_reward']:.2f}")
    print(f"Avg Time: {best_result['avg_time']:.2f}s")
    print("=" * 80)

    # 统计分析
    all_success_rates = [r['success_rate'] for r in results]
    print("\n📈 Statistical Summary:")
    print(f"  Mean success rate: {np.mean(all_success_rates):.1%}")
    print(f"  Std deviation:     {np.std(all_success_rates):.1%}")
    print(f"  Min success rate:  {np.min(all_success_rates):.1%}")
    print(f"  Max success rate:  {np.max(all_success_rates):.1%}")
    print(f"  Range:             {np.max(all_success_rates) - np.min(all_success_rates):.1%}")

    # 使用建议
    print("\n" + "=" * 80)
    print("💡 How to use the best seed for training:")
    print("=" * 80)
    print(f"1. Set environment variable:")
    print(f"   export PYTHONHASHSEED=0")
    print(f"   export CUBLAS_WORKSPACE_CONFIG=:4096:8")
    print(f"")
    print(f"2. Run training with best seed:")
    print(f"   python train.py --seed {best_result['seed']} --gpu \\")
    print(f"     --outdir runs/mamba_vl_seed{best_result['seed']}")
    print(f"")
    print(f"3. Expected BC baseline: {best_result['success_rate']:.1%}")
    print(f"   (RL should improve beyond this)")
    print("=" * 80)

    # Top 3推荐
    print("\n🥇🥈🥉 Top 3 Seeds:")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"  {i+1}. Seed {r['seed']:5d}: {r['success_rate']:.1%} success, {r['avg_reward']:.2f} reward")


if __name__ == '__main__':
    main()
