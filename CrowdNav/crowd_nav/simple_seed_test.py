#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超简单种子测试 - 用ORCA baseline测试环境在不同seed下的难度

原理：
- ORCA是确定性算法，不需要训练
- 在不同seed下，环境难度不同（人群分布、起点终点）
- ORCA表现好的seed → 环境相对简单 → RL训练也会更容易收敛

用法：
    python simple_seed_test.py --num_seeds 16
"""

import os
import sys
import numpy as np
import argparse
import logging
import json
from datetime import datetime
from configparser import ConfigParser

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crowd_sim.envs import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def test_seed_with_orca(seed, env, robot, num_episodes=100):
    """
    用ORCA测试指定seed下的环境难度

    Returns:
        dict: 性能指标
    """
    # 设置主进程种子
    np.random.seed(seed)

    success = 0
    collision = 0
    timeout = 0
    rewards = []
    times = []

    # 静默测试（不显示进度条，加快速度）
    for ep in range(num_episodes):
        # env.reset()会在内部使用test_case参数作为seed
        test_case_seed = seed + ep  # 每个episode使用不同但确定的seed
        ob = env.reset(phase='test', test_case=test_case_seed)
        done = False
        ep_reward = 0
        steps = 0
        info = None

        while not done and steps < 100:
            action = robot.act(ob)
            step_result = env.step(action)

            # 兼容不同版本的step返回值
            if len(step_result) == 4:
                ob, reward, done, info = step_result
            elif len(step_result) == 5:
                ob, reward, done, truncated, info = step_result
                if truncated:
                    done = True
            else:
                ob, reward, done = step_result[0], step_result[1], step_result[2]
                info = step_result[3] if len(step_result) > 3 else None
            ep_reward += reward
            steps += 1

        # 统计
        if isinstance(info, dict):
            event = info.get('event', '').lower()
        else:
            event = str(info).lower()

        if 'success' in event or 'reach' in event:
            success += 1
        elif 'collision' in event:
            collision += 1
        else:
            timeout += 1

        rewards.append(ep_reward)
        times.append(steps * env.time_step)

    return {
        'seed': seed,
        'success_rate': success / num_episodes,
        'collision_rate': collision / num_episodes,
        'timeout_rate': timeout / num_episodes,
        'avg_reward': float(np.mean(rewards)),
        'avg_time': float(np.mean(times)),
        'success_count': success,
        'total_episodes': num_episodes
    }


def generate_stratified_seed(tested_seeds, strata_index):
    """
    统计学分层采样 - 生成下一个测试种子

    将种子空间分为多个层级（strata），每层随机采样
    """
    # 分层策略：将0-10000分为10个层级
    num_strata = 10
    strata_size = 10000 // num_strata

    # 当前层级
    current_stratum = strata_index % num_strata

    # 在当前层级内随机采样
    strata_start = current_stratum * strata_size
    strata_end = strata_start + strata_size

    # 避免重复
    max_attempts = 100
    for _ in range(max_attempts):
        # 使用当前时间+层级作为随机种子，确保每次不同
        import time
        np.random.seed(int(time.time() * 1000) % 100000 + strata_index)

        seed = np.random.randint(strata_start, strata_end)

        if seed not in tested_seeds:
            return seed

    # 如果当前层级都测试过了，随机生成
    for _ in range(max_attempts):
        seed = np.random.randint(1, 10000)
        if seed not in tested_seeds:
            return seed

    # 最后手段：递增查找
    for seed in range(1, 10000):
        if seed not in tested_seeds:
            return seed

    return None  # 所有种子都测试过了（理论上不会发生）


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_success_rate', type=float, default=0.90,
                       help='Target success rate to stop testing (default: 0.90)')
    parser.add_argument('--test_episodes', type=int, default=50,
                       help='Episodes per seed (default: 50, reduced for speed)')
    parser.add_argument('--max_seeds', type=int, default=200,
                       help='Maximum seeds to test before giving up (default: 200)')
    parser.add_argument('--num_seeds_required', type=int, default=8,
                       help='Number of seeds meeting target to collect (default: 8 for 8 workers)')
    parser.add_argument('--config_dir', type=str, default='./configs')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🎯 目标驱动种子搜索 (ORCA Baseline)")
    print("=" * 80)
    print(f"🎯 目标成功率: {args.target_success_rate:.0%}")
    print(f"🎯 需要收集种子数: {args.num_seeds_required} 个")
    print(f"📊 每个种子测试: {args.test_episodes} episodes")
    print(f"🔢 最大测试种子数: {args.max_seeds}")
    print(f"📈 采样策略: 统计学分层采样（10层，每层1000个种子）")
    print("=" * 80)
    print()

    # 加载配置
    env_cfg = ConfigParser()
    env_cfg.read(os.path.join(args.config_dir, 'env.config'), encoding='utf-8')

    # 创建环境和ORCA policy
    env = CrowdSim()
    env.configure(env_cfg)

    robot = Robot(env_cfg, 'robot')
    orca_policy = policy_factory['orca']()
    robot.set_policy(orca_policy)
    env.set_robot(robot)

    # 目标驱动的搜索循环
    results = []
    tested_seeds = set()
    found_seeds = []  # 收集所有达标种子
    best_result = None

    print(f"🔍 开始搜索，将持续测试直到找到 {args.num_seeds_required} 个 ≥{args.target_success_rate:.0%} 成功率的种子...\n")

    for i in range(args.max_seeds):
        # 使用分层采样生成下一个种子
        seed = generate_stratified_seed(tested_seeds, i)

        if seed is None:
            print(f"\n⚠️  已测试完所有可能的种子，只找到{len(found_seeds)}个")
            break

        tested_seeds.add(seed)

        # 显示当前进度
        print(f"[{i+1}/{args.max_seeds}] 测试 Seed {seed:5d} ", end='', flush=True)

        # 测试当前种子
        result = test_seed_with_orca(seed, env, robot, args.test_episodes)
        results.append(result)

        # 实时显示结果
        print(f"→ 成功率: {result['success_rate']:5.1%} ({result['success_count']}/{args.test_episodes})", end='')

        # 检查是否达到目标
        if result['success_rate'] >= args.target_success_rate:
            found_seeds.append(result)
            print(f"  ✅ 达标！({len(found_seeds)}/{args.num_seeds_required})")

            # 检查是否收集够了
            if len(found_seeds) >= args.num_seeds_required:
                print(f"\n🎉 成功收集到 {args.num_seeds_required} 个达标种子！")
                break
        else:
            # 显示距离目标还差多少
            gap = args.target_success_rate - result['success_rate']
            print(f"  ❌ 差 {gap:.1%}")

        # 更新当前最佳
        if best_result is None or result['success_rate'] > best_result['success_rate']:
            best_result = result

    print("\n" + "=" * 80)

    # 搜索总结
    print("📊 搜索总结")
    print("=" * 80)

    if len(found_seeds) >= args.num_seeds_required:
        print(f"✅ 成功收集到 {len(found_seeds)} 个达标种子！")
        print(f"   测试了 {len(results)} 个种子")
        print(f"   目标成功率: {args.target_success_rate:.0%}")
    elif len(found_seeds) > 0:
        print(f"⚠️  只找到 {len(found_seeds)} 个达标种子 (目标: {args.num_seeds_required} 个)")
        print(f"   测试了 {len(results)} 个种子")
        print(f"   距离目标还差: {args.num_seeds_required - len(found_seeds)} 个")
    else:
        print(f"⚠️  未找到达到 {args.target_success_rate:.0%} 目标的种子")
        print(f"   测试了 {len(results)} 个种子")
        if best_result:
            print(f"   最佳种子: {best_result['seed']}")
            print(f"   最高成功率: {best_result['success_rate']:.1%}")

    print("=" * 80)

    # 排序并显示详细结果
    results.sort(key=lambda x: x['success_rate'], reverse=True)

    print("\n📊 所有测试结果 (按成功率排序)")
    print("=" * 80)
    print("\nRank | Seed  | Success | Collision | Timeout | Avg Reward")
    print("-" * 70)

    for rank, r in enumerate(results, 1):
        marker = " 🎯" if r['success_rate'] >= args.target_success_rate else ""
        print(f"{rank:4d} | {r['seed']:5d} | {r['success_rate']:7.1%} | "
              f"{r['collision_rate']:9.1%} | {r['timeout_rate']:7.1%} | "
              f"{r['avg_reward']:10.2f}{marker}")

    # 保存结果
    output_file = f'seed_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    seed_list_file = f'seed_list_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

    with open(output_file, 'w') as f:
        json.dump({
            'target_success_rate': args.target_success_rate,
            'num_seeds_required': args.num_seeds_required,
            'found_seeds_count': len(found_seeds),
            'seeds_tested': len(results),
            'found_seeds': found_seeds,
            'all_results': results,
            'args': vars(args)
        }, f, indent=2)

    # 推荐
    print("\n" + "=" * 80)
    print("🏆 找到的达标种子列表")
    print("=" * 80)

    if len(found_seeds) >= args.num_seeds_required:
        print(f"\n✅ 成功找到 {len(found_seeds)} 个 ≥{args.target_success_rate:.0%} 的种子：\n")

        # 保存种子列表到文件
        with open(seed_list_file, 'w') as f:
            f.write("# 达标种子列表（供训练使用）\n")
            f.write(f"# 目标成功率: {args.target_success_rate:.0%}\n")
            f.write(f"# 测试episodes: {args.test_episodes}\n\n")

            for i, r in enumerate(found_seeds, 1):
                print(f"{i}. Seed {r['seed']:5d}: {r['success_rate']:5.1%} "
                      f"(碰撞{r['collision_rate']:5.1%}, 奖励{r['avg_reward']:6.2f})")
                f.write(f"{r['seed']}\n")

        print(f"\n种子列表已保存: {seed_list_file}")

        # 提取种子数组用于训练
        seed_array = [r['seed'] for r in found_seeds[:args.num_seeds_required]]
        print(f"\n📋 用于训练的种子数组：")
        print(f"{seed_array}")

    elif len(found_seeds) > 0:
        print(f"\n⚠️  只找到 {len(found_seeds)} 个达标种子：\n")
        for i, r in enumerate(found_seeds, 1):
            print(f"{i}. Seed {r['seed']:5d}: {r['success_rate']:5.1%}")
        print(f"\n建议：增加 --max_seeds 或降低 --target_success_rate")
    else:
        print(f"\n⚠️  未找到达标种子")
        if best_result:
            print(f"最佳种子: {best_result['seed']} ({best_result['success_rate']:.1%})")

    print("=" * 80)

    # 统计
    sr = [r['success_rate'] for r in results]
    print(f"\n📈 统计信息:")
    print(f"  测试种子数: {len(results)}")
    print(f"  平均成功率: {np.mean(sr):.1%} ± {np.std(sr):.1%}")
    print(f"  成功率范围: {np.min(sr):.1%} ~ {np.max(sr):.1%}")
    print(f"  达标种子数: {sum(1 for r in results if r['success_rate'] >= args.target_success_rate)}")

    print(f"\n✅ 结果已保存: {output_file}")

    if len(found_seeds) >= args.num_seeds_required:
        seed_array = [r['seed'] for r in found_seeds[:args.num_seeds_required]]
        print(f"\n💡 使用这些种子训练（需要修改代码支持种子列表）:")
        print(f"   种子数组: {seed_array}")
        print(f"\n   或使用第一个种子（传统方式）:")
        print(f"   python train.py --seed {found_seeds[0]['seed']} --gpu")
    elif best_result:
        print(f"\n💡 使用最佳种子训练:")
        print(f"   python train.py --seed {best_result['seed']} --gpu")


if __name__ == '__main__':
    main()
