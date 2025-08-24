#!/usr/bin/env python3
"""
调试版测试脚本 - 验证训练结果的真实性
主要功能：
1. 详细记录每个episode的执行过程
2. 验证成功判断的准确性
3. 分析奖励计算
4. 生成详细的调试信息
"""

import os
import re
import torch
import numpy as np
import logging
import argparse
import configparser
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import crowd_sim
import gym
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.explorer import Explorer

def load_weights(policy, model_dir, weights_file, device):
    """加载模型权重"""
    weights_full = weights_file if os.path.isabs(weights_file) else os.path.join(model_dir, weights_file)
    if not os.path.isfile(weights_full):
        raise FileNotFoundError(f"Model weights not found: {weights_full}")
    
    if hasattr(policy, 'load_model'):
        policy.load_model(weights_full)
    else:
        state = torch.load(weights_full, map_location=device)
        policy.load_state_dict(state)
    
    logging.info(f"Weights loaded: {os.path.basename(weights_full)}")
    return weights_full

def debug_episode(env, policy, episode_idx, max_steps=400):
    """详细调试单个episode"""
    obs, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    rewards = []
    positions = []
    actions = []
    min_distances = []
    
    # 记录初始状态
    robot_pos = [env.robot.px, env.robot.py]
    goal_pos = env.robot.get_goal_position()
    initial_dist = np.linalg.norm(np.array(robot_pos) - np.array(goal_pos))
    
    logging.info(f"Episode {episode_idx}: Start at {robot_pos}, Goal at {goal_pos}, Distance: {initial_dist:.3f}")
    
    while not done and steps < max_steps:
        # 记录当前状态
        robot_pos = [env.robot.px, env.robot.py]
        positions.append(robot_pos)
        
        # 计算到目标距离
        curr_dist = np.linalg.norm(np.array(robot_pos) - np.array(goal_pos))
        
        # 计算到最近人类距离
        min_dist = float('inf')
        for human in env.humans:
            dist = np.linalg.norm(np.array(robot_pos) - np.array([human.px, human.py]))
            min_dist = min(min_dist, dist)
        min_distances.append(min_dist)
        
        # 预测动作
        action = policy.predict(obs)
        actions.append([action.vx, action.vy])
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        
        rewards.append(reward)
        total_reward += reward
        steps += 1
        
        # 每50步记录一次状态
        if steps % 50 == 0:
            logging.info(f"Step {steps}: pos={robot_pos}, dist_to_goal={curr_dist:.3f}, "
                        f"min_dist_to_human={min_dist:.3f}, reward={reward:.3f}, "
                        f"action=[{action.vx:.3f}, {action.vy:.3f}]")
    
    # 分析结果
    final_pos = [env.robot.px, env.robot.py]
    final_dist = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
    
    # 判断是否真的成功
    success_radius = env.robot.radius
    is_success = final_dist < success_radius
    
    # 统计信息
    avg_reward = np.mean(rewards) if rewards else 0.0
    min_human_dist = min(min_distances) if min_distances else float('inf')
    avg_human_dist = np.mean(min_distances) if min_distances else 0.0
    
    logging.info(f"Episode {episode_idx} Results:")
    logging.info(f"  Steps: {steps}")
    logging.info(f"  Final position: {final_pos}")
    logging.info(f"  Distance to goal: {final_dist:.3f} (radius: {success_radius})")
    logging.info(f"  Success: {is_success}")
    logging.info(f"  Total reward: {total_reward:.3f}")
    logging.info(f"  Average reward per step: {avg_reward:.3f}")
    logging.info(f"  Min distance to human: {min_human_dist:.3f}")
    logging.info(f"  Average distance to human: {avg_human_dist:.3f}")
    logging.info(f"  Event: {info.get('event', 'unknown')}")
    logging.info(f"  Result: {info.get('result', 'unknown')}")
    
    # 绘制轨迹
    if len(positions) > 1:
        positions = np.array(positions)
        plt.figure(figsize=(10, 8))
        
        # 绘制机器人轨迹
        plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Robot Path')
        plt.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
        plt.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
        
        # 绘制目标
        plt.plot(goal_pos[0], goal_pos[1], 'k*', markersize=15, label='Goal')
        
        # 绘制人类
        for i, human in enumerate(env.humans):
            plt.plot(human.px, human.py, 'ko', markersize=8, label=f'Human {i}' if i == 0 else "")
        
        # 绘制成功半径
        circle = plt.Circle(goal_pos, success_radius, color='red', fill=False, linestyle='--', label='Success Radius')
        plt.gca().add_patch(circle)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Episode {episode_idx} - {"SUCCESS" if is_success else "FAILED"}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # 保存图片
        plt.savefig(f'debug_episode_{episode_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'success': is_success,
        'steps': steps,
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'final_distance': final_dist,
        'min_human_distance': min_human_dist,
        'event': info.get('event', 'unknown'),
        'result': info.get('result', 'unknown')
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='mamba_rl')
    parser.add_argument('--model_dir', type=str, default='data/output')
    parser.add_argument('--env_config', type=str, default='configs/env_fixed.config')
    parser.add_argument('--policy_config', type=str, default='data/output/policy.config')
    parser.add_argument('--weights', type=str, default='rl_model_ep1000.pth')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s')

    # 加载配置
    env_config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    env_config.read(args.env_config)
    
    policy_config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    policy_config.read(args.policy_config)

    # 创建策略
    policy = policy_factory[args.policy](policy_config)
    policy.configure(policy_config)
    policy.set_device(device)

    # 加载权重
    load_weights(policy, args.model_dir, args.weights, device)

    # 创建环境
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    # 设置测试模式
    if hasattr(policy, 'set_phase'):
        policy.set_phase('test')
    if hasattr(policy, 'set_epsilon'):
        policy.set_epsilon(0.0)

    # 调试多个episode
    results = []
    for i in range(args.episodes):
        logging.info(f"\n{'='*50}")
        logging.info(f"DEBUGGING EPISODE {i+1}/{args.episodes}")
        logging.info(f"{'='*50}")
        
        result = debug_episode(env, policy, i+1)
        results.append(result)
        
        logging.info(f"Episode {i+1} Summary: Success={result['success']}, "
                    f"Reward={result['total_reward']:.3f}, Steps={result['steps']}")

    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    success_rate = success_count / len(results)
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    logging.info(f"\n{'='*50}")
    logging.info(f"FINAL DEBUG RESULTS")
    logging.info(f"{'='*50}")
    logging.info(f"Total episodes: {len(results)}")
    logging.info(f"Success count: {success_count}")
    logging.info(f"Success rate: {success_rate:.3f}")
    logging.info(f"Average reward: {avg_reward:.3f}")
    logging.info(f"Average steps: {avg_steps:.1f}")
    
    # 分析奖励分布
    rewards = [r['total_reward'] for r in results]
    positive_rewards = [r for r in rewards if r > 0]
    negative_rewards = [r for r in rewards if r <= 0]
    
    logging.info(f"Positive rewards: {len(positive_rewards)}/{len(rewards)}")
    logging.info(f"Negative rewards: {len(negative_rewards)}/{len(rewards)}")
    if positive_rewards:
        logging.info(f"Average positive reward: {np.mean(positive_rewards):.3f}")
    if negative_rewards:
        logging.info(f"Average negative reward: {np.mean(negative_rewards):.3f}")

if __name__ == '__main__':
    main()
