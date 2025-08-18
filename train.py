import yaml
import torch
import numpy as np
import os
import random
import logging
from tqdm import trange
import time
from collections import deque

from env.crowd_env import CrowdNavEnv
from models.camrl_agent import CAMRLAgent

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_all_weights(agent, path_prefix):
    torch.save(agent.value_net.state_dict(), f"{path_prefix}_value_net.pth")
    torch.save(agent.encoder.state_dict(), f"{path_prefix}_encoder.pth")
    torch.save(agent.policy_head.state_dict(), f"{path_prefix}_policy_head.pth")

def main():
    cfg = yaml.safe_load(open('config/default.yaml'))
    set_seed(cfg.get('seed', 42))

    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    log_file = "logs/train_debug.log"
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    mamba_cfg = cfg['mamba']
    env = CrowdNavEnv(cfg)
    agent = CAMRLAgent(cfg['obs_dim'], mamba_cfg['state_dim'], cfg)

    # ========= 行为克隆预训练 =========
    il_demo_path = cfg.get("il_demo_path", "orca_demos_flat.npz")
    do_il = os.path.exists(il_demo_path)
    if do_il:
        print(f"[INFO] 加载IL演示数据: {il_demo_path}")
        try:
            demo = np.load(il_demo_path, allow_pickle=True)
            if isinstance(demo, np.lib.npyio.NpzFile):
                all_obs = demo['obs']
                all_act = demo['act']
            else:
                # 兼容旧npy格式
                demo_data = demo
                all_obs = np.concatenate([d["obs"] for d in demo_data], axis=0)
                all_act = np.concatenate([d["act"] for d in demo_data], axis=0)
        except Exception as e:
            print(f"[ERROR] 加载IL数据失败: {e}")
            return

        assert all_obs.shape[0] == all_act.shape[0], "IL数据obs/act维度不一致"
        print(f"[INFO] Loaded IL demos: obs shape={all_obs.shape}, act shape={all_act.shape}")
        logging.info(f"IL数据obs维度: {all_obs.shape}, act维度: {all_act.shape}")

        il_epochs = cfg.get('il_epochs', 30)
        il_batch_size = cfg.get('il_batch_size', 128)
        
        print("开始IL预训练...")
        start_time = time.time()
        agent.imitation_update(all_obs, all_act, epochs=il_epochs, batch_size=il_batch_size)
        il_time = time.time() - start_time
        print(f"IL预训练完成，耗时: {il_time:.2f}秒")
        logging.info(f"IL imitation pretrain finished in {il_time:.2f}s")
        save_all_weights(agent, "weights/camrl_il")

        # 如只想做IL，可直接return
        if cfg.get('il_only', False):
            print("配置已设置 il_only=True，训练结束。")
            return

    # ========= RL训练主循环 =========
    num_episodes = cfg.get('episodes', 100)
    batch_size = cfg.get('batch_size', 32)
    target_update = cfg.get('target_update', 10)
    max_steps_per_episode = cfg.get('max_steps_per_episode', 500)
    print_per = 20
    
    # 优化：添加训练统计
    episode_stats = {"reach_goal": 0, "collision": 0, "timeout": 0, "other": 0}
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    # 优化：预分配环境
    print("开始RL训练...")
    start_time = time.time()

    for ep in trange(num_episodes, desc="Training Episodes", ncols=90):
        episode_start = time.time()
        state_seq = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        last_event = "other"

        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state_seq)
            next_seq, reward, done, info = env.step(action)
            reward = np.clip(reward, -10, 10)
            
            # 优化：使用新的经验添加方法
            agent.add_experience(state_seq, action, reward, next_seq, done)
            state_seq = next_seq
            episode_reward += reward
            steps += 1

        # 优化：批量训练
        if len(agent.replay_buffer) >= batch_size:
            batch = agent.sample_batch(batch_size)
            if batch is not None:
                loss = agent.train_step(batch)
                
                # 定期更新目标网络
                if ep % target_update == 0:
                    agent.soft_update_target()

        # 统计信息
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if 'event' in info:
            last_event = info['event']
            episode_stats[last_event] += 1
        
        episode_time = time.time() - episode_start
        
        # 定期打印统计信息
        if (ep + 1) % print_per == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            success_rate = episode_stats["reach_goal"] / (ep + 1) * 100
            collision_rate = episode_stats["collision"] / (ep + 1) * 100
            
            print(f"Episode {ep+1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}, Avg Length: {avg_length:.1f}")
            print(f"  Success Rate: {success_rate:.1f}%, Collision Rate: {collision_rate:.1f}%")
            print(f"  Episode Time: {episode_time:.3f}s")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
            print("-" * 50)
            
            logging.info(f"Episode {ep+1}: reward={avg_reward:.3f}, length={avg_length:.1f}, "
                        f"success={success_rate:.1f}%, collision={collision_rate:.1f}%")

    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time:.2f}秒")
    print(f"平均每episode耗时: {total_time/num_episodes:.3f}秒")
    
    # 最终统计
    final_success_rate = episode_stats["reach_goal"] / num_episodes * 100
    final_collision_rate = episode_stats["collision"] / num_episodes * 100
    final_timeout_rate = episode_stats["timeout"] / num_episodes * 100
    
    print(f"\n最终结果:")
    print(f"  成功率: {final_success_rate:.1f}%")
    print(f"  碰撞率: {final_collision_rate:.1f}%")
    print(f"  超时率: {final_timeout_rate:.1f}%")
    
    logging.info(f"Training finished in {total_time:.2f}s")
    logging.info(f"Final success rate: {final_success_rate:.1f}%")
    
    save_all_weights(agent, "weights/camrl_final")

if __name__ == "__main__":
    main()

