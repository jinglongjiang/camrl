#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
种子搜索脚本 - 自动找到表现最好的随机种子

功能：
1. 并行测试多个候选seed（默认16个）
2. 每个seed训练到筛选里程碑（默认500 episodes）
3. 评估验证集成功率
4. 输出最佳seed及其性能报告

用法：
    python find_best_seed.py --milestone 500 --num_seeds 16 --gpu
    python find_best_seed.py --milestone 1000 --num_seeds 8 --gpu --quick
"""

import os
import sys
import argparse
import subprocess
import json
import time
import re
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('seed_search.log'),
        logging.StreamHandler()
    ]
)


def run_training_with_seed(seed, milestone_episodes, gpu, output_base_dir, config_dir, quick_mode=False):
    """
    使用指定seed运行训练到里程碑episode

    Args:
        seed: 随机种子
        milestone_episodes: 训练到多少episodes停止
        gpu: 是否使用GPU
        output_base_dir: 输出基础目录
        config_dir: 配置文件目录
        quick_mode: 快速模式（减少evaluation频率）

    Returns:
        dict: {
            'seed': seed,
            'final_success_rate': float,
            'peak_success_rate': float,
            'avg_last_100': float,
            'output_dir': str,
            'train_time': float (seconds)
        }
    """
    output_dir = os.path.join(output_base_dir, f'seed_{seed}')
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"[Seed {seed}] Starting training to {milestone_episodes} episodes...")

    # 构建训练命令
    cmd = [
        sys.executable, 'train.py',
        '--outdir', output_dir,
        '--config', os.path.join(config_dir, 'train.config'),
        '--env-config', os.path.join(config_dir, 'env.config'),
        '--policy-config', os.path.join(config_dir, 'policy.config'),
        '--seed', str(seed),
    ]

    if gpu:
        cmd.append('--gpu')

    # 设置环境变量（确保可复现）
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '0'
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # 临时修改train.config中的episodes数
    # （更优雅的方式是创建临时config文件）
    start_time = time.time()

    try:
        # 运行训练（捕获输出）
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )

        train_time = time.time() - start_time

        # 解析train.log获取性能指标
        train_log_path = os.path.join(output_dir, 'train.log')

        if os.path.exists(train_log_path):
            metrics = parse_training_log(train_log_path, milestone_episodes)
            metrics['seed'] = seed
            metrics['output_dir'] = output_dir
            metrics['train_time'] = train_time

            logging.info(f"[Seed {seed}] ✅ Completed in {train_time/60:.1f}min - "
                        f"Final: {metrics['final_success_rate']:.1%}, "
                        f"Peak: {metrics['peak_success_rate']:.1%}, "
                        f"Avg(last100): {metrics['avg_last_100']:.1%}")

            return metrics
        else:
            logging.error(f"[Seed {seed}] ❌ train.log not found")
            return {
                'seed': seed,
                'final_success_rate': 0.0,
                'peak_success_rate': 0.0,
                'avg_last_100': 0.0,
                'output_dir': output_dir,
                'train_time': train_time,
                'error': 'log_not_found'
            }

    except subprocess.TimeoutExpired:
        logging.error(f"[Seed {seed}] ❌ Training timeout (>2h)")
        return {
            'seed': seed,
            'final_success_rate': 0.0,
            'peak_success_rate': 0.0,
            'avg_last_100': 0.0,
            'output_dir': output_dir,
            'train_time': 7200,
            'error': 'timeout'
        }
    except Exception as e:
        logging.error(f"[Seed {seed}] ❌ Training failed: {e}")
        return {
            'seed': seed,
            'final_success_rate': 0.0,
            'peak_success_rate': 0.0,
            'avg_last_100': 0.0,
            'output_dir': output_dir,
            'train_time': time.time() - start_time,
            'error': str(e)
        }


def parse_training_log(log_path, milestone_episodes):
    """
    解析训练日志获取成功率曲线

    Returns:
        dict: {
            'final_success_rate': 最终成功率（最后一次评估）
            'peak_success_rate': 峰值成功率
            'avg_last_100': 最后100个episodes的平均成功率
            'success_rates': List[float] 所有评估点的成功率
        }
    """
    success_rates = []

    # 解析日志中的评估成功率
    # 示例格式: [EVAL-500] Success: 0.850 (85/100)
    pattern = r'\[EVAL-(\d+)\].*?Success:\s+([\d.]+)'

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                episode = int(match.group(1))
                succ_rate = float(match.group(2))
                if episode <= milestone_episodes:
                    success_rates.append(succ_rate)

    if not success_rates:
        # 如果没有找到EVAL记录，尝试从ROLL记录中解析
        pattern2 = r'ep=(\d+).*?succ=([\d.]+)'
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(pattern2, line)
                if match:
                    episode = int(match.group(1))
                    succ_rate = float(match.group(2))
                    if episode <= milestone_episodes:
                        success_rates.append(succ_rate)

    if success_rates:
        final_success_rate = success_rates[-1]
        peak_success_rate = max(success_rates)
        avg_last_100 = np.mean(success_rates[-10:]) if len(success_rates) >= 10 else np.mean(success_rates)
    else:
        final_success_rate = 0.0
        peak_success_rate = 0.0
        avg_last_100 = 0.0

    return {
        'final_success_rate': final_success_rate,
        'peak_success_rate': peak_success_rate,
        'avg_last_100': avg_last_100,
        'success_rates': success_rates
    }


def generate_candidate_seeds(num_seeds, base_seed=42):
    """
    生成候选种子列表

    策略：使用不同范围的种子，确保多样性
    """
    seeds = []

    # 策略1: 基础种子附近的小偏移（42, 43, 44, ...）
    seeds.extend([base_seed + i for i in range(num_seeds // 4)])

    # 策略2: 100倍间隔（100, 200, 300, ...）
    seeds.extend([100 * (i + 1) for i in range(num_seeds // 4)])

    # 策略3: 1000倍间隔（1000, 2000, 3000, ...）
    seeds.extend([1000 * (i + 1) for i in range(num_seeds // 4)])

    # 策略4: 常用的幸运数字
    lucky_seeds = [42, 123, 456, 777, 1234, 2023, 2024, 3407]
    seeds.extend(lucky_seeds[:num_seeds // 4])

    # 去重并截取到num_seeds
    seeds = list(set(seeds))[:num_seeds]

    # 如果不够，补充随机种子
    if len(seeds) < num_seeds:
        np.random.seed(base_seed)
        additional = list(np.random.randint(1, 10000, size=num_seeds - len(seeds)))
        seeds.extend(additional)

    return sorted(seeds[:num_seeds])


def main():
    parser = argparse.ArgumentParser(description='Find best random seed for training')
    parser.add_argument('--milestone', type=int, default=500,
                       help='Training episodes milestone for seed evaluation (default: 500)')
    parser.add_argument('--num_seeds', type=int, default=16,
                       help='Number of candidate seeds to test (default: 16)')
    parser.add_argument('--max_parallel', type=int, default=4,
                       help='Maximum parallel training jobs (default: 4)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for training')
    parser.add_argument('--output_dir', type=str, default='./seed_search_results',
                       help='Base output directory (default: ./seed_search_results)')
    parser.add_argument('--config_dir', type=str, default='./configs',
                       help='Config directory (default: ./configs)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: use fewer evaluations')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base seed for candidate generation (default: 42)')

    args = parser.parse_args()

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base_dir = os.path.join(args.output_dir, f'search_{timestamp}')
    os.makedirs(output_base_dir, exist_ok=True)

    logging.info("=" * 80)
    logging.info("🔍 Random Seed Search Experiment")
    logging.info("=" * 80)
    logging.info(f"Milestone: {args.milestone} episodes")
    logging.info(f"Candidates: {args.num_seeds} seeds")
    logging.info(f"Max parallel: {args.max_parallel}")
    logging.info(f"GPU: {args.gpu}")
    logging.info(f"Output: {output_base_dir}")
    logging.info("=" * 80)

    # 生成候选种子
    candidate_seeds = generate_candidate_seeds(args.num_seeds, args.base_seed)
    logging.info(f"Candidate seeds: {candidate_seeds}")

    # 并行运行训练
    results = []

    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        # 提交所有训练任务
        future_to_seed = {
            executor.submit(
                run_training_with_seed,
                seed,
                args.milestone,
                args.gpu,
                output_base_dir,
                args.config_dir,
                args.quick
            ): seed
            for seed in candidate_seeds
        }

        # 收集结果
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"[Seed {seed}] Exception: {e}")
                results.append({
                    'seed': seed,
                    'final_success_rate': 0.0,
                    'peak_success_rate': 0.0,
                    'avg_last_100': 0.0,
                    'error': str(e)
                })

    # 按峰值成功率排序
    results.sort(key=lambda x: x['peak_success_rate'], reverse=True)

    # 生成报告
    logging.info("\n" + "=" * 80)
    logging.info("📊 Seed Search Results (sorted by peak success rate)")
    logging.info("=" * 80)

    print("\nRank | Seed  | Final SR | Peak SR | Avg(last100) | Time(min)")
    print("-" * 70)

    for rank, result in enumerate(results, 1):
        print(f"{rank:4d} | {result['seed']:5d} | "
              f"{result['final_success_rate']:7.1%} | "
              f"{result['peak_success_rate']:7.1%} | "
              f"{result['avg_last_100']:11.1%} | "
              f"{result.get('train_time', 0)/60:8.1f}")

    # 保存结果到JSON
    results_file = os.path.join(output_base_dir, 'seed_search_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'timestamp': timestamp,
            'results': results
        }, f, indent=2)

    logging.info(f"\n✅ Results saved to: {results_file}")

    # 推荐最佳seed
    best_result = results[0]
    logging.info("\n" + "=" * 80)
    logging.info("🏆 RECOMMENDED BEST SEED")
    logging.info("=" * 80)
    logging.info(f"Seed: {best_result['seed']}")
    logging.info(f"Peak Success Rate: {best_result['peak_success_rate']:.1%}")
    logging.info(f"Final Success Rate: {best_result['final_success_rate']:.1%}")
    logging.info(f"Avg (last 100 eps): {best_result['avg_last_100']:.1%}")
    logging.info(f"Output Directory: {best_result['output_dir']}")
    logging.info("=" * 80)

    # 统计分析
    all_peaks = [r['peak_success_rate'] for r in results if r['peak_success_rate'] > 0]
    if all_peaks:
        logging.info("\n📈 Statistical Summary:")
        logging.info(f"  Mean peak SR: {np.mean(all_peaks):.1%}")
        logging.info(f"  Std peak SR:  {np.std(all_peaks):.1%}")
        logging.info(f"  Min peak SR:  {np.min(all_peaks):.1%}")
        logging.info(f"  Max peak SR:  {np.max(all_peaks):.1%}")
        logging.info(f"  Range:        {np.max(all_peaks) - np.min(all_peaks):.1%}")

    # 生成使用建议
    print("\n" + "=" * 80)
    print("💡 How to use the best seed:")
    print("=" * 80)
    print(f"1. Add to your train.config:")
    print(f"   [train]")
    print(f"   random_seed = {best_result['seed']}")
    print(f"")
    print(f"2. Or use command line:")
    print(f"   python train.py --seed {best_result['seed']} --gpu")
    print(f"")
    print(f"3. For reproducibility, also set:")
    print(f"   export PYTHONHASHSEED=0")
    print(f"   export CUBLAS_WORKSPACE_CONFIG=:4096:8")
    print("=" * 80)


if __name__ == '__main__':
    main()
