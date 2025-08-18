import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import git
import numpy as np
from tqdm import tqdm
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory

from crowd_sim.envs.utils.state import JointState

# ----------------- 工具函数 -----------------
def to_joint_state(obs):
    if isinstance(obs, JointState):
        return obs
    elif hasattr(obs, "self_state"):
        return obs
    elif isinstance(obs, np.ndarray):
        return JointState.from_array(obs)
    elif isinstance(obs, (list, tuple)):
        return JointState.from_array(np.array(obs))
    else:
        raise ValueError(f"未知的状态类型: {type(obs)}")

def safe_predict(policy, obs, env=None):
    """推理阶段也做 obs 归一化，避免 IL→RL 分布漂移。"""
    try:
        js = to_joint_state(obs)
        # === 推理阶段的归一化 ===
        if hasattr(policy, "obs_mean") and hasattr(policy, "obs_std") and policy.obs_mean is not None:
            arr = js.to_array()
            m, s = policy.obs_mean, policy.obs_std
            if isinstance(m, np.ndarray) and isinstance(s, np.ndarray) and arr.shape == m.shape:
                arr = (arr - m) / (s + 1e-6)
                js = JointState.from_array(arr)
        return policy.predict(js)
    except Exception as e:
        print(f"[safe_predict] policy.predict失败:{e}, obs type:{type(obs)}")
        if env is not None and hasattr(env, "get_full_state"):
            try:
                return policy.predict(env.get_full_state())
            except Exception as e2:
                print(f"[safe_predict] policy.predict(env.get_full_state())二次失败:{e2}")
                raise
        raise

def debug_shape(tag, arr):
    try:
        if isinstance(arr, list):
            arr = np.array(arr)
        if hasattr(arr, 'shape'):
            print(f"[DEBUG] {tag} shape: {arr.shape}")
        else:
            print(f"[DEBUG] {tag} type: {type(arr)}")
    except Exception as e:
        print(f"[DEBUG] {tag} shape打印出错: {e}")

# DAgger 概率计划
def get_expert_prob(ep, exp_full=500, exp_anneal=2000, min_prob=0.2):
    if ep < exp_full:
        return 1.0
    elif ep < exp_anneal:
        return 1.0 - (ep - exp_full) / (exp_anneal - exp_full) * (1.0 - min_prob)
    else:
        return min_prob

# 兼容不同 ReplayMemory 实现，迭代出 transition
def iter_memory_transitions(memory):
    for attr in ["buffer", "memory", "storage", "deque"]:
        if hasattr(memory, attr):
            container = getattr(memory, attr)
            return list(container)
    if hasattr(memory, "__iter__"):
        try:
            return list(iter(memory))
        except Exception:
            pass
    for k, v in memory.__dict__.items():
        if isinstance(v, (list, tuple, collections.deque)) and len(v) > 0:
            return list(v)
    raise RuntimeError("无法从 ReplayMemory 中找到可迭代的存储容器。")

# 从 transition 里提取“可转为数组的观测”
def extract_obs_from_transition(tr):
    candidates = []
    if isinstance(tr, (list, tuple)):
        candidates = list(tr)
    elif hasattr(tr, "_fields"):  # namedtuple
        candidates = [getattr(tr, f) for f in tr._fields]
    else:
        return None
    for part in candidates:
        if hasattr(part, "to_array"):
            return part.to_array()
        if isinstance(part, np.ndarray) and part.ndim == 1:
            return part
    return None

def compute_and_save_obs_stats(memory, out_path):
    transitions = iter_memory_transitions(memory)
    obs_list = []
    for tr in transitions:
        arr = extract_obs_from_transition(tr)
        if arr is not None:
            obs_list.append(np.asarray(arr))
    if len(obs_list) == 0:
        raise RuntimeError("在 ReplayMemory 中没有提取到可用观测，无法计算 obs_mean/std。")
    obs_mat = np.stack(obs_list, axis=0)
    mean = obs_mat.mean(axis=0)
    std = obs_mat.std(axis=0) + 1e-6
    np.savez(out_path, mean=mean, std=std)
    return mean, std

def moving_average(x, w):
    if len(x) == 0:
        return np.array([])
    w = max(1, min(w, len(x)))
    return np.convolve(np.array(x, dtype=float), np.ones(w)/w, mode='valid')

# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    plot_interval = 50

    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)

    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')
    obs_stats_file = os.path.join(args.output_dir, 'il_obs_stats.npz')

    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # Git hash
    try:
        repo = git.Repo(search_parent_directories=True)
        logging.info('Current git head hash code: %s', repo.head.object.hexsha)
    except Exception as e:
        logging.warning('Git repo not found or hash fetch failed: %s', e)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # ----------- policy -----------
    policy_config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    policy_config.read(args.policy_config)
    print("[DEBUG] policy_config keys:", policy_config.sections())
    policy = policy_factory[args.policy](policy_config)

    if not policy.trainable:
        parser.error('Policy has to be trainable')
    if args.policy_config is None:
        parser.error('Policy config has to be specified for a trainable network')
    policy.configure(policy_config)
    policy.set_device(device)

    # ----------- env -----------
    env_config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    env_config.read(args.env_config)
    print("[DEBUG] env_config keys:", env_config.sections())
    if 'sim' in env_config.sections() and 'humans' in env_config.sections():
        print("[DEBUG] sim config:", dict(env_config['sim']))
        print("[DEBUG] humans config:", dict(env_config['humans']))
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # ----------- train params -----------
    if args.train_config is None:
        parser.error('Train config has to be specified for a trainable network')
    train_config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    train_config.read(args.train_config)
    print("[DEBUG] train_config keys:", train_config.sections())
    if 'trainer' in train_config.sections():
        print("[DEBUG] trainer config:", dict(train_config['trainer']))
    rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
    train_batches = train_config.getint('train', 'train_batches')
    train_episodes = train_config.getint('train', 'train_episodes')
    sample_episodes = train_config.getint('train', 'sample_episodes')
    target_update_interval = train_config.getint('train', 'target_update_interval')
    evaluation_interval = train_config.getint('train', 'evaluation_interval')
    capacity = train_config.getint('train', 'capacity')
    epsilon_start = train_config.getfloat('train', 'epsilon_start')
    epsilon_end = train_config.getfloat('train', 'epsilon_end')
    epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
    checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

    # 可选：从配置读取 DAgger 计划（没有则回退默认）
    dagger_exp_full   = train_config.getint('dagger', 'exp_full',   fallback=500)
    dagger_exp_anneal = train_config.getint('dagger', 'exp_anneal', fallback=2000)
    dagger_min_prob   = train_config.getfloat('dagger', 'min_prob', fallback=0.2)

    # ----------- trainer, explorer -----------
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.getint('trainer', 'batch_size')
    trainer = Trainer(model, memory, device, batch_size)
    explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)

    # expert policy for DAgger
    il_policy_name = train_config.get('imitation_learning', 'il_policy')
    expert_policy = policy_factory[il_policy_name](policy_config)
    expert_policy.configure(policy_config)
    expert_policy.set_device(device)
    expert_policy.multiagent_training = policy.multiagent_training
    expert_policy.safety_space = train_config.getfloat('imitation_learning', 'safety_space')

    # ----------- imitation learning -----------
    il_just_ran = False
    if args.resume:
        if not os.path.exists(rl_weight_file):
            logging.error('RL weights does not exist')
        model.load_state_dict(torch.load(rl_weight_file, map_location=device))
        rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
        logging.info('Load reinforcement learning trained weights. Resume training')
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file, map_location=device))
        logging.info('Load imitation learning trained weights.')
        try:
            if os.path.exists(obs_stats_file):
                stats = np.load(obs_stats_file)
                policy.obs_mean = stats['mean']
                policy.obs_std = stats['std']
                debug_shape("IL obs_mean", policy.obs_mean)
                debug_shape("IL obs_std", policy.obs_std)
                logging.info('Load IL obs_mean/obs_std for normalization.')
            else:
                logging.warning('IL obs stats file not found, proceeding without normalization.')
        except Exception as e:
            logging.warning(f"Failed to load IL obs_mean/obs_std: {e}")
    else:
        il_episodes = train_config.getint('imitation_learning', 'il_episodes')
        il_epochs = train_config.getint('imitation_learning', 'il_epochs')
        il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
        trainer.set_learning_rate(il_learning_rate)

        # EXPERT 采样
        il_policy = expert_policy
        robot.set_policy(il_policy)
        expert_name = il_policy.__class__.__name__.lower()
        force_joint_state_policy = any(key in expert_name for key in ['orca', 'cadrl', 'sarl', 'multi_human_rl'])

        print("==========[DEBUG] Imitation learning stage start ==========")
        explorer.run_k_episodes(
            il_episodes,
            'train',
            update_memory=True,
            imitation_learning=True,
            force_joint_state_policy=force_joint_state_policy,
            return_stats=False
        )
        print("==========[DEBUG] Imitation learning stage end ==========")

        trainer.optimize_epoch(il_epochs)

        # 计算/保存 obs 统计
        try:
            mean, std = compute_and_save_obs_stats(memory, obs_stats_file)
            policy.obs_mean, policy.obs_std = mean, std
            debug_shape("IL obs_mean", mean)
            debug_shape("IL obs_std", std)
            logging.info('IL obs_mean/obs_std computed and saved for normalization.')
        except Exception as e:
            logging.warning(f"Failed to compute/save IL obs_mean/obs_std: {e}")

        # 保存并立刻再加载一遍兜底，保证磁盘与内存一致
        torch.save(model.state_dict(), il_weight_file)
        model.load_state_dict(torch.load(il_weight_file, map_location=device))
        logging.info('Finish imitation learning. Weights saved & reloaded.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
        il_just_ran = True

    explorer.update_target_model(model)

    # ----------- RL主循环 -----------
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_learning_rate(rl_learning_rate)

    # RL 开始前兜底：确保从 IL 权重+统计启动
    if os.path.exists(il_weight_file):
        try:
            model.load_state_dict(torch.load(il_weight_file, map_location=device))
            logging.info("Loaded IL weights into policy for RL init (safety reload).")
        except Exception as e:
            logging.warning("Safety reload IL weights failed: %s", e)
    if os.path.exists(obs_stats_file):
        try:
            stats = np.load(obs_stats_file)
            policy.obs_mean = stats['mean']
            policy.obs_std = stats['std']
            logging.info("Loaded IL obs stats into policy for RL init.")
        except Exception as e:
            logging.warning("Load IL obs stats failed: %s", e)

    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0, return_stats=False)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    # ---- 统计历史（用于画图）----
    train_succ_hist, train_coll_hist, train_timeout_hist = [], [], []
    train_reward_hist = []

    logging.info("Sanity: RL init from IL weights: %s; obs_stats exists: %s",
                 os.path.exists(il_weight_file), os.path.exists(obs_stats_file))
    if hasattr(policy, "obs_mean") and hasattr(policy, "obs_std"):
        try:
            debug_shape("policy.obs_mean", policy.obs_mean)
            debug_shape("policy.obs_std", policy.obs_std)
        except Exception:
            pass

    for episode in tqdm(range(train_episodes), desc="Training progress", unit="ep"):
        # epsilon
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # DAgger 概率（可由配置覆盖）
        p_expert = get_expert_prob(episode,
                                   exp_full=dagger_exp_full,
                                   exp_anneal=dagger_exp_anneal,
                                   min_prob=dagger_min_prob)

        # 评估（固定 RL 策略）
        if episode % evaluation_interval == 0:
            val_stats = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode, return_stats=True)
            logging.info("VAL ep=%d | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
                         episode, val_stats["success_rate"], val_stats["collision_rate"],
                         val_stats["timeout_rate"], val_stats["nav_time"], val_stats["total_reward"])

        # === 真正影响 replay 的 DAgger：按“集”混合专家与 RL ===
        use_expert_for_this_batch = (np.random.rand() < p_expert)
        orig_policy = robot.policy
        try:
            if use_expert_for_this_batch:
                robot.set_policy(expert_policy)
                logging.info(f"[DAgger] episode={episode} 用专家采样，p_expert={p_expert:.2f}")
            else:
                robot.set_policy(policy)
                logging.info(f"[DAgger] episode={episode} 用RL采样，p_expert={p_expert:.2f}")

            train_stats = explorer.run_k_episodes(sample_episodes, 'train', update_memory=True,
                                                  episode=episode, return_stats=True)
        finally:
            robot.set_policy(orig_policy)

        # 收集真实训练统计
        train_succ_hist.append(train_stats["success_rate"])
        train_coll_hist.append(train_stats["collision_rate"])
        train_timeout_hist.append(train_stats["timeout_rate"])
        train_reward_hist.append(train_stats["total_reward"])

        # 训练
        trainer.optimize_batch(train_batches)

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), rl_weight_file)

        # 画图（移动平均）
        if episode % plot_interval == 0 and episode > 0:
            W = min(plot_interval, len(train_reward_hist))
            reward_ma = moving_average(train_reward_hist, W)
            succ_ma   = moving_average(train_succ_hist, W)
            coll_ma   = moving_average(train_coll_hist, W)
            tout_ma   = moving_average(train_timeout_hist, W)

            plt.figure(figsize=(12,6))
            # 上：train 平均 reward（移动平均）
            plt.subplot(211)
            plt.plot(train_reward_hist, label='Train Avg Reward')
            if len(reward_ma) > 0:
                plt.axhline(np.mean(train_reward_hist[-W:]), ls='--', label=f'mean({W})')
            plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend()

            # 下：三条率曲线
            plt.subplot(212)
            x0 = np.arange(W-1, W-1+len(succ_ma))
            if len(succ_ma): plt.plot(x0, succ_ma, label=f'Success (w={W})')
            if len(coll_ma): plt.plot(x0, coll_ma, label=f'Collision (w={W})')
            if len(tout_ma): plt.plot(x0, tout_ma, label=f'Timeout (w={W})')
            plt.xlabel('Episode'); plt.ylabel('Rate'); plt.legend()
            plt.tight_layout()
            fig_name = os.path.join(args.output_dir, f"train_curves_ep{episode}.png")
            plt.savefig(fig_name); plt.close()
            logging.info(f"[Plot] saved to {fig_name}")

        tqdm.write("Episode: {:d}/{:d}, epsilon: {:.4f}".format(episode + 1, train_episodes, epsilon))

    test_stats = explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, return_stats=True)
    logging.info("TEST | succ={:.2f} coll={:.2f} timeout={:.2f} nav={:.2f} reward={:.2f}",
                 test_stats["success_rate"], test_stats["collision_rate"],
                 test_stats["timeout_rate"], test_stats["nav_time"], test_stats["total_reward"])
    logging.info("Training complete.")

if __name__ == '__main__':
    main()
