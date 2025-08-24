#!/usr/bin/env python3
"""
超快速训练脚本 - 6小时内达到85%+成功率
关键优化：
1. 超大batch训练 (8192)
2. 高质量IL预训练 (500 episodes, 8000 epochs)
3. 激进的学习率和探索策略
4. 12并行workers采样
5. 仅400个训练episodes
"""

import sys, os, shutil, logging, argparse, configparser, random

# Add parent directory to path to import crowd_sim
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np, torch
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 确保CrowdSim环境被注册
import crowd_sim
try:
    import gymnasium as gym
except ImportError:
    import gym

from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.memory import ReplayMemory, ExpertReplayMemory
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.parallel_sampler import ParallelSampler

# ====== 高性能设置 ======
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

ROBOT_VPREF_IDX = 7
DEFAULT_SEED = 42

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # 启用benchmark加速
    torch.backends.cudnn.deterministic = False  # 牺牲确定性换取速度

def format_time(seconds):
    """格式化时间显示"""
    return str(timedelta(seconds=int(seconds)))

class FastTrainer:
    """快速训练控制器"""
    def __init__(self, config_dict):
        self.config = config_dict
        self.start_time = datetime.now()
        self.episode_times = []
        
    def estimate_completion(self, current_ep, total_ep):
        """预估完成时间"""
        if not self.episode_times:
            return "N/A"
        avg_time = np.mean(self.episode_times[-10:])  # 最近10个episode的平均时间
        remaining_eps = total_ep - current_ep
        est_seconds = remaining_eps * avg_time
        return format_time(est_seconds)
    
    def log_progress(self, episode, total, success_rate, collision_rate):
        """打印进度信息"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = self.estimate_completion(episode, total)
        progress = episode / total * 100
        
        logging.info(
            f"Episode {episode}/{total} ({progress:.1f}%) | "
            f"Success: {success_rate:.2%} | Collision: {collision_rate:.2%} | "
            f"Elapsed: {format_time(elapsed)} | ETA: {eta}"
        )

def enhanced_bc_pretrain(policy, exp_buf, device, epochs=8000, batch_size=8192, lr=0.0005):
    """增强的BC预训练，确保高初始成功率"""
    if len(exp_buf) == 0:
        return 0.0
    
    actor = policy.actor
    actor.train()
    
    # 使用Adam + 学习率衰减
    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
    
    losses = []
    best_loss = float('inf')
    patience = 0
    max_patience = 500
    
    bar = tqdm(range(epochs), desc="Enhanced BC Training", dynamic_ncols=True)
    for epoch in bar:
        # 大batch训练
        s, a_star = exp_buf.sample(min(batch_size, len(exp_buf)))
        v_pref = np.clip(s[:, ROBOT_VPREF_IDX:ROBOT_VPREF_IDX+1], 1e-6, None)
        a_norm = np.clip(a_star / (v_pref * getattr(policy, 'action_scale', 1.0)), -1.0, 1.0)
        
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
        if hasattr(policy, "_norm_obs"):
            s_t = policy._norm_obs(s_t)
        if hasattr(policy, "_encode"):
            h = policy._encode(s_t)
            mu, _ = actor(h)
        else:
            mu, _ = actor(s_t)
        
        a_mu = torch.tanh(mu)
        target = torch.as_tensor(a_norm, dtype=torch.float32, device=device)
        
        # Huber loss for robustness
        loss = torch.nn.functional.huber_loss(a_mu, target, reduction='mean', delta=0.5)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        loss_val = float(loss.item())
        losses.append(loss_val)
        
        # Early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience and epoch > 1000:
            logging.info(f"BC Early stopping at epoch {epoch}")
            break
        
        if epoch % 100 == 0:
            bar.set_postfix({
                "loss": f"{np.mean(losses[-100:]):.4f}",
                "best": f"{best_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })
    
    return float(np.mean(losses[-100:])) if losses else 0.0

def main():
    parser = argparse.ArgumentParser('Fast training for CrowdNav')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='mamba_rl')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/output_fast')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    
    set_global_seed(args.seed)
    
    # 创建输出目录
    if os.path.exists(args.output_dir) and not args.resume:
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 复制配置文件
    for config_file in [args.env_config, args.policy_config, args.train_config]:
        if os.path.exists(config_file):
            shutil.copy(config_file, args.output_dir)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, 'output.log')
    mode = 'a' if args.resume else 'w'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode=mode),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info(f'Starting FAST training on {device}')
    
    # 加载配置
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy = policy_factory[args.policy](policy_config)
    policy.configure(policy_config)
    policy.set_device(device)
    
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    
    # 训练参数
    train_episodes = train_config.getint('train', 'train_episodes')
    
    # 初始化训练器
    trainer = Trainer(
        policy, device,
        gamma=train_config.getfloat('trainer', 'gamma'),
        tau=train_config.getfloat('trainer', 'tau'),
        lr_actor=train_config.getfloat('trainer', 'lr_actor'),
        lr_critic=train_config.getfloat('trainer', 'lr_critic'),
        lr_alpha=train_config.getfloat('trainer', 'lr_alpha'),
        batch_size=train_config.getint('trainer', 'batch_size'),
        target_entropy=train_config.getfloat('trainer', 'target_entropy'),
        use_amp=train_config.getboolean('trainer', 'use_amp'),
        grad_clip=train_config.getfloat('trainer', 'grad_clip')
    )
    
    # 初始化内存
    capacity = train_config.getint('train', 'capacity')
    rl_memory = ReplayMemory(capacity)
    exp_memory = ExpertReplayMemory(capacity)
    explorer = Explorer(env, robot, device, rl_memory, policy.gamma, target_policy=policy)
    
    # 快速训练控制器
    fast_trainer = FastTrainer({'train_episodes': train_episodes})
    
    # ========== 阶段1: 高质量IL训练 ==========
    il_episodes = train_config.getint('imitation_learning', 'il_episodes')
    if il_episodes > 0:
        logging.info(f"Phase 1: Collecting {il_episodes} expert demonstrations")
        
        il_policy_name = train_config.get('imitation_learning', 'il_policy')
        expert_policy = policy_factory[il_policy_name](policy_config)
        expert_policy.configure(policy_config)
        expert_policy.set_device(device)
        expert_policy.safety_space = train_config.getfloat('imitation_learning', 'safety_space')
        
        robot.set_policy(expert_policy)
        
        # 收集专家数据
        success_count = 0
        for i in tqdm(range(il_episodes), desc="Expert demos"):
            ob, _ = env.reset()  # Unpack tuple (observation, info)
            done = False
            episode_reward = 0
            
            while not done:
                prev_ob = ob  # Store previous observation
                action = robot.act(ob)
                ob, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Store expert demonstration in memory
                exp_memory.push(prev_ob, action)
                
                if isinstance(info, dict) and info.get('event') == 'success':
                    success_count += 1
        
        expert_success_rate = success_count / il_episodes
        logging.info(f"Expert success rate: {expert_success_rate:.2%}")
        
        # BC预训练
        il_epochs = train_config.getint('imitation_learning', 'il_epochs')
        il_lr = train_config.getfloat('imitation_learning', 'il_learning_rate')
        
        bc_loss = enhanced_bc_pretrain(
            policy, exp_memory, device,
            epochs=il_epochs,
            batch_size=train_config.getint('trainer', 'batch_size'),
            lr=il_lr
        )
        logging.info(f"BC training complete. Final loss: {bc_loss:.6f}")
        
        robot.set_policy(policy)
    
    # ========== 阶段2: 快速RL训练 ==========
    logging.info(f"Phase 2: Fast RL training for {train_episodes} episodes")
    
    # 并行采样器
    if train_config.getboolean('vectorize', 'enable'):
        sampler = ParallelSampler(
            env, robot, policy, device,
            num_workers=train_config.getint('vectorize', 'num_workers'),
            episodes_per_worker=train_config.getint('vectorize', 'episodes_per_worker'),
            worker_device=train_config.get('vectorize', 'worker_device')
        )
    
    # 训练循环
    success_rates = []
    best_success_rate = 0
    
    for episode in range(train_episodes):
        episode_start = datetime.now()
        
        # 采样
        if train_config.getboolean('vectorize', 'enable'):
            episodes_data = sampler.sample_episodes()
            # 将数据加入replay buffer
            for ep_data in episodes_data:
                for transition in ep_data:
                    rl_memory.push(transition)
        
        # 训练
        if len(rl_memory) > train_config.getint('trainer', 'batch_size'):
            train_batches = train_config.getint('train', 'train_batches')
            for _ in range(train_batches):
                trainer.update(rl_memory, exp_memory)
        
        # 评估
        if episode % train_config.getint('train', 'evaluation_interval') == 0:
            success_count = 0
            collision_count = 0
            eval_episodes = 50
            
            for _ in range(eval_episodes):
                ob, _ = env.reset()  # Unpack tuple
                done = False
                while not done:
                    action = robot.act(ob)
                    ob, _, done, info = env.step(action)
                    if isinstance(info, dict):
                        if info.get('event') == 'success':
                            success_count += 1
                        elif info.get('event') == 'collision':
                            collision_count += 1
            
            success_rate = success_count / eval_episodes
            collision_rate = collision_count / eval_episodes
            success_rates.append(success_rate)
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                # 保存最佳模型
                torch.save(policy.state_dict(), 
                          os.path.join(args.output_dir, f'best_model_ep{episode}.pth'))
            
            # 记录时间
            episode_time = (datetime.now() - episode_start).total_seconds()
            fast_trainer.episode_times.append(episode_time)
            
            # 打印进度
            fast_trainer.log_progress(episode, train_episodes, success_rate, collision_rate)
            
            # 早停条件：达到85%成功率
            if success_rate >= 0.85:
                logging.info(f"🎉 Target success rate achieved: {success_rate:.2%}")
                break
    
    # 最终测试
    logging.info("Final testing on 300 episodes...")
    test_success = 0
    for _ in tqdm(range(300), desc="Final test"):
        ob, _ = env.reset()  # Unpack tuple
        done = False
        while not done:
            action = robot.act(ob)
            ob, _, done, info = env.step(action)
            if isinstance(info, dict) and info.get('event') == 'success':
                test_success += 1
    
    final_success_rate = test_success / 300
    total_time = (datetime.now() - fast_trainer.start_time).total_seconds()
    
    logging.info("="*50)
    logging.info(f"Training Complete!")
    logging.info(f"Final Success Rate: {final_success_rate:.2%}")
    logging.info(f"Best Success Rate: {best_success_rate:.2%}")
    logging.info(f"Total Training Time: {format_time(total_time)}")
    logging.info(f"Target: 85% success in 6 hours")
    logging.info(f"Achieved: {final_success_rate:.2%} in {format_time(total_time)}")
    logging.info("="*50)

if __name__ == '__main__':
    main()