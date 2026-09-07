#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门的BC (Behavioral Cloning) 训练脚本

功能：
1. 从IL数据集（ORCA专家轨迹）训练初始策略
2. 保存BC权重（policy + Q网络）
3. 供RL训练直接加载，跳过IL阶段

使用方法：
  python train_bc.py --policy mamba_rl --dataset data/il_dataset_diverse_v4.0.pth --output data/bc_mamba_v4.0.pth --gpu

设计理念：
- 模块化：BC训练独立于RL训练
- 灵活性：可以用不同数据集训练多个BC baseline
- 可复现：固定随机种子，确保结果一致
"""

import os
import sys
import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
import configparser


def setup_logging(log_file=None):
    """配置日志"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )


def load_il_dataset(dataset_path, device):
    """加载IL数据集并转换为训练格式"""
    print(f"\n{'='*80}")
    print(f"加载IL数据集: {dataset_path}")
    print(f"{'='*80}\n")

    dataset = torch.load(dataset_path, map_location='cpu')
    trajectories = dataset['trajectories']

    print(f"数据集信息:")
    print(f"  版本: {dataset.get('version', 'unknown')}")
    print(f"  轨迹数: {len(trajectories)}")

    if 'config' in dataset:
        cfg = dataset['config']
        print(f"  人数: {cfg.get('human_num', 'unknown')}")
        print(f"  半径配置: {cfg.get('radius_configs', 'unknown')}")
        print(f"  速度配置: {cfg.get('human_v_pref_configs', 'unknown')}")

    if 'stats' in dataset:
        stats = dataset['stats']
        print(f"  成功率: {stats.get('success_rate', 0):.2%}")

    # 提取数据
    all_states = []
    all_actions = []
    all_rewards = []
    all_returns = []

    for traj in trajectories:
        states = traj['states']      # [T, state_dim]
        actions = traj['actions']    # [T, 2]
        rewards = traj['rewards']    # [T]
        mc_returns = traj['mc_returns']  # [T]

        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_returns.append(mc_returns)

    # 转换为tensor
    states_tensor = torch.cat(all_states, dim=0).float().to(device)
    actions_tensor = torch.cat(all_actions, dim=0).float().to(device)
    returns_tensor = torch.cat(all_returns, dim=0).float().unsqueeze(1).to(device)

    print(f"\n训练数据:")
    print(f"  States: {states_tensor.shape}")
    print(f"  Actions: {actions_tensor.shape}")
    print(f"  Returns: {returns_tensor.shape}")
    print(f"  内存占用: {(states_tensor.numel() + actions_tensor.numel() + returns_tensor.numel()) * 4 / 1024**2:.1f} MB")

    return states_tensor, actions_tensor, returns_tensor


def train_bc(policy, states, actions, returns, args):
    """BC训练主循环"""
    print(f"\n{'='*80}")
    print(f"开始BC训练")
    print(f"{'='*80}\n")

    # 创建数据加载器
    dataset = TensorDataset(states, actions, returns)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # 优化器
    policy_optimizer = optim.Adam(
        list(policy.spatial_encoder.parameters()) +
        list(policy.temporal_encoder.parameters()) +
        list(policy.actor_mean.parameters()) +
        list(policy.actor_log_std.parameters()),
        lr=args.policy_lr
    )

    q_optimizer = optim.Adam(
        list(policy.q1_head.parameters()) +
        list(policy.q2_head.parameters()),
        lr=args.q_lr
    )

    print(f"训练配置:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Policy LR: {args.policy_lr}")
    print(f"  Q LR: {args.q_lr}")
    print(f"  BC-Q coef: {args.bc_q_coef}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Device: {states.device}")

    # 训练循环
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_bc_loss = 0.0
        epoch_q_loss = 0.0
        num_batches = 0

        for batch_idx, (batch_states, batch_actions, batch_returns) in enumerate(dataloader):
            batch_size = batch_states.shape[0]

            # 序列化状态（为Mamba准备）
            # states: [B, state_dim] -> [B, seq_len, state_dim]
            if args.seq_len > 1:
                # 重复当前状态seq_len次（简化版，真实场景应该用历史）
                states_seq = batch_states.unsqueeze(1).repeat(1, args.seq_len, 1)
            else:
                states_seq = batch_states.unsqueeze(1)

            # 前向传播
            policy_optimizer.zero_grad()
            q_optimizer.zero_grad()

            # Actor loss (BC loss)
            sp_feat = policy.spatial_encoder(states_seq)
            temp_feat = policy.temporal_encoder(sp_feat)
            state_features = temp_feat[:, -1, :]  # [B, hidden_dim]

            pred_actions = policy.actor_mean(state_features)
            bc_loss = F.mse_loss(pred_actions, batch_actions)

            # Q loss (用MC returns监督)
            q1_pred, q2_pred = policy.compute_q(state_features, batch_actions)
            q_loss = F.mse_loss(q1_pred, batch_returns) + F.mse_loss(q2_pred, batch_returns)

            # 总损失
            total_loss = bc_loss + args.bc_q_coef * q_loss

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

            policy_optimizer.step()
            q_optimizer.step()

            epoch_bc_loss += bc_loss.item()
            epoch_q_loss += q_loss.item()
            num_batches += 1

        # Epoch统计
        avg_bc_loss = epoch_bc_loss / num_batches
        avg_q_loss = epoch_q_loss / num_batches
        avg_total_loss = avg_bc_loss + args.bc_q_coef * avg_q_loss
        epoch_time = time.time() - epoch_start

        # 日志
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"BC: {avg_bc_loss:.4f} | "
                  f"Q: {avg_q_loss:.4f} | "
                  f"Total: {avg_total_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            logging.info(f"Epoch {epoch+1}/{args.epochs}: BC={avg_bc_loss:.4f}, Q={avg_q_loss:.4f}, Total={avg_total_loss:.4f}")

        # 保存最佳模型
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_epoch = epoch + 1

    print(f"\n训练完成!")
    print(f"  最佳Epoch: {best_epoch}")
    print(f"  最佳Loss: {best_loss:.4f}")

    return best_loss


def save_bc_weights(policy, output_path, args, best_loss, metadata=None):
    """保存BC权重"""
    print(f"\n{'='*80}")
    print(f"保存BC权重: {output_path}")
    print(f"{'='*80}\n")

    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'training_config': {
            'policy_lr': args.policy_lr,
            'q_lr': args.q_lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'seq_len': args.seq_len,
            'bc_q_coef': args.bc_q_coef,
        },
        'best_loss': best_loss,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_path': args.dataset,
    }

    if metadata:
        checkpoint['dataset_metadata'] = metadata

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        torch.save(checkpoint, output_path)
        file_size = os.path.getsize(output_path) / 1024**2
        print(f"✓ 权重已保存")
        print(f"  文件大小: {file_size:.1f} MB")
        print(f"  最佳Loss: {best_loss:.4f}")
        logging.info(f"BC权重已保存到: {output_path}, 大小: {file_size:.1f}MB")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        logging.error(f"保存BC权重失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='BC Training Script')

    # 基础配置
    parser.add_argument('--policy', type=str, default='mamba_rl', help='Policy name')
    parser.add_argument('--dataset', type=str, required=True, help='IL dataset path')
    parser.add_argument('--output', type=str, required=True, help='Output BC weights path')
    parser.add_argument('--env-config', type=str, default='crowd_nav/configs/env.config')
    parser.add_argument('--policy-config', type=str, default='crowd_nav/configs/policy.config')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log', type=str, default=None, help='Log file')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--policy-lr', type=float, default=1e-4, help='Policy learning rate')
    parser.add_argument('--q-lr', type=float, default=1e-4, help='Q network learning rate')
    parser.add_argument('--bc-q-coef', type=float, default=0.1, help='Q loss coefficient')
    parser.add_argument('--seq-len', type=int, default=12, help='Sequence length for Mamba')
    parser.add_argument('--log-interval', type=int, default=5, help='Log every N epochs')

    args = parser.parse_args()

    # 日志
    if args.log is None:
        args.log = args.output.replace('.pth', '.log')
    setup_logging(args.log)

    # 设备
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"BC训练配置")
    print(f"{'='*80}")
    print(f"Policy: {args.policy}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 加载配置
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)

    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)

    # 创建policy（参考train.py的build_policy函数）
    print(f"\n{'='*80}")
    print(f"创建策略网络: {args.policy}")
    print(f"{'='*80}\n")

    # 直接用构造函数创建policy（MambaRL接受policy_config和device参数）
    Policy = policy_factory[args.policy]
    policy = Policy(policy_config, device=device).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 加载IL数据集
    states, actions, returns = load_il_dataset(args.dataset, device)

    # 训练BC
    best_loss = train_bc(policy, states, actions, returns, args)

    # 保存权重
    dataset_info = torch.load(args.dataset, map_location='cpu')
    metadata = {
        'version': dataset_info.get('version', 'unknown'),
        'num_trajectories': len(dataset_info.get('trajectories', [])),
        'success_rate': dataset_info.get('stats', {}).get('success_rate', 0),
    }

    save_bc_weights(policy, args.output, args, best_loss, metadata)

    print(f"\n{'='*80}")
    print(f"BC训练完成！")
    print(f"{'='*80}")
    print(f"使用方法（在RL训练中）:")
    print(f"  python train.py --policy {args.policy} --load-bc {args.output} --skip-il --gpu")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
