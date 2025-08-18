# train.py  — minimal-diff：保留原结构 + 逐步DAgger + SAC(双Q) + BC预训练 + 进度条
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
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer            # SAC 训练器
from crowd_nav.utils.memory import ReplayMemory, ExpertReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import JointState

# ---------- 常量 ----------
ROBOT_VPREF_IDX = 6   # 若你状态里 v_pref 位置不同，请改这里
DEFAULT_SEED = 42

# ----------------- 工具函数（保留） -----------------
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
    try:
        js = to_joint_state(obs)
        if hasattr(policy, "obs_mean") and hasattr(policy, "obs_var") and policy.obs_mean is not None:
            arr = js.to_array()
            m = np.array(policy.obs_mean.detach().cpu()) if torch.is_tensor(policy.obs_mean) else policy.obs_mean
            v = np.array(policy.obs_var.detach().cpu())  if torch.is_tensor(policy.obs_var)  else policy.obs_var
            if isinstance(m, np.ndarray) and isinstance(v, np.ndarray) and arr.shape == m.shape:
                arr = (arr - m) / (np.sqrt(v) + 1e-6)
                js = JointState.from_array(arr)
        return policy.predict(js)
    except Exception as e:
        print(f"[safe_predict] policy.predict失败:{e}, obs type:{type(obs)}")
        raise

def moving_average(x, w):
    if len(x) == 0:
        return np.array([])
    w = max(1, min(w, len(x)))
    return np.convolve(np.array(x, dtype=float), np.ones(w)/w, mode='valid')

# ----------------- 随机种子 -----------------
def set_global_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # 若需完全确定性可开启，但会慢：torch.use_deterministic_algorithms(True)

# ----------------- DAgger 概率 -----------------
def get_expert_prob(ep, exp_full=100, exp_anneal=600, min_prob=0.2):
    if ep < exp_full:
        return 1.0
    t = min(1.0, (ep - exp_full) / max(1, exp_anneal - exp_full))
    return 1.0 * (1 - t) + min_prob * t

# ----------------- 仅用于 IL 的 Actor-BC 预训练 -----------------
def bc_pretrain_actor(policy, exp_buf, device, iters=2000, batch_size=256, lr=3e-4):
    if len(exp_buf) == 0:
        return 0.0
    actor = policy.actor
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    losses = []
    for _ in range(iters):
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
        loss = ((a_mu - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt.step()
        losses.append(float(loss.item()))
        if hasattr(policy, "update_obs_rms"):
            with torch.no_grad():
                policy.update_obs_rms(s)
    return float(np.mean(losses)) if losses else 0.0

# ----------------- 检查点保存 -----------------
def save_checkpoint(output_dir, policy, trainer, episode, extra=None):
    os.makedirs(output_dir, exist_ok=True)
    # 1) 策略权重
    model_path = os.path.join(output_dir, f'rl_model_ep{episode}.pth')
    if hasattr(policy, "save_model"):
        policy.save_model(model_path)
    else:
        torch.save(policy.get_state_dict() if hasattr(policy, "get_state_dict") else policy.state_dict(), model_path)
    # 2) 训练器/优化器状态（若可用）
    try:
        t_state = trainer.state_dict() if hasattr(trainer, "state_dict") else None
    except Exception:
        t_state = None
    ckpt = {
        "episode": int(episode),
        "trainer": t_state,
        "rng": {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if isinstance(extra, dict):
        ckpt.update(extra)
    torch.save(ckpt, os.path.join(output_dir, f'checkpoint_ep{episode}.pth'))
    # 3) 观测归一化统计
    try:
        obs_npz = {}
        for k in ("obs_mean", "obs_var", "obs_std"):
            if hasattr(policy, k) and getattr(policy, k) is not None:
                v = getattr(policy, k)
                if torch.is_tensor(v):
                    v = v.detach().cpu().numpy()
                obs_npz[k] = np.asarray(v)
        if obs_npz:
            np.savez(os.path.join(output_dir, f'obs_stats_ep{episode}.npz'), **obs_npz)
    except Exception as e:
        logging.warning("Saving obs stats failed: %s", e)

# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='mamba_rl')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # 固定随机种子
    set_global_seed(args.seed, deterministic=True)

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
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)

    log_file = os.path.join(args.output_dir, 'output.log')
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

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
    policy = policy_factory[args.policy](policy_config)
    if not policy.trainable:
        raise SystemExit('Policy has to be trainable')
    policy.configure(policy_config)
    policy.set_device(device)

    # ----------- env -----------
    env_config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0'); env.configure(env_config)
    robot = Robot(env_config, 'robot'); env.set_robot(robot)

    # ----------- train params -----------
    train_config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    train_config.read(args.train_config)

    train_episodes = train_config.getint('train', 'train_episodes')
    sample_episodes = train_config.getint('train', 'sample_episodes')
    evaluation_interval = train_config.getint('train', 'evaluation_interval')
    checkpoint_interval = train_config.getint('train', 'checkpoint_interval')
    capacity = train_config.getint('train', 'capacity')

    epsilon_start = train_config.getfloat('train', 'epsilon_start', fallback=0.2)
    epsilon_end   = train_config.getfloat('train', 'epsilon_end',   fallback=0.05)
    epsilon_decay = train_config.getfloat('train', 'epsilon_decay', fallback=1000)

    # DAgger 调度
    dagger_exp_full   = train_config.getint('dagger', 'exp_full',   fallback=1500)
    dagger_exp_anneal = train_config.getint('dagger', 'exp_anneal', fallback=3000)
    dagger_min_prob   = train_config.getfloat('dagger', 'min_prob', fallback=0.3)

    # SAC 超参
    gamma     = train_config.getfloat('trainer', 'gamma',     fallback=0.99)
    tau       = train_config.getfloat('trainer', 'tau',       fallback=0.005)
    lr_actor  = train_config.getfloat('trainer', 'lr_actor',  fallback=3e-4)
    lr_critic = train_config.getfloat('trainer', 'lr_critic', fallback=3e-4)
    lr_alpha  = train_config.getfloat('trainer', 'lr_alpha',  fallback=3e-4)
    batch_size= train_config.getint  ('trainer', 'batch_size', fallback=256)
    bc_ratio  = train_config.getfloat('trainer', 'bc_ratio',  fallback=1.0)
    lambda_bc = train_config.getfloat('trainer', 'lambda_bc', fallback=1.0)
    train_batches = train_config.getint('train', 'train_batches', fallback=1)

    # ----------- buffers / explorer / expert -----------
    rl_buf = ReplayMemory(capacity)
    exp_buf = ExpertReplayMemory(capacity)
    explorer = Explorer(env, robot, device, rl_buf, policy.gamma, target_policy=policy)

    il_policy_name = train_config.get('imitation_learning', 'il_policy', fallback='orca')
    expert_policy = policy_factory[il_policy_name](policy_config)
    expert_policy.configure(policy_config)
    expert_policy.set_device(device)
    expert_policy.multiagent_training = policy.multiagent_training
    expert_policy.safety_space = train_config.getfloat('imitation_learning', 'safety_space', fallback=0.0)

    # ----------- Trainer (SAC + 双Q + BC正则) -----------
    trainer = Trainer(policy, device,
                      gamma=gamma, tau=tau,
                      lr_actor=lr_actor, lr_critic=lr_critic, lr_alpha=lr_alpha,
                      batch_size=batch_size, bc_ratio=bc_ratio, lambda_bc=lambda_bc)

    # ----------- 可选：预训练（行为克隆）-----------
    il_episodes = train_config.getint('imitation_learning', 'il_episodes', fallback=0)
    il_epochs   = train_config.getint('imitation_learning', 'il_epochs',   fallback=0)
    il_lr       = train_config.getfloat('imitation_learning', 'il_learning_rate', fallback=3e-4)
    if il_episodes > 0:
        robot.set_policy(expert_policy)

        use_new = hasattr(explorer, 'collect_with_labels')
        succ = coll = tout = 0
        succ_times = []
        ep_rewards = []
        ep_steps = []

        if use_new:
            bar = tqdm(range(il_episodes), desc="IL sampling (expert labels)", dynamic_ncols=True, unit="ep")
            for i in bar:
                st = explorer.collect_with_labels(
                    env, policy, expert_policy, rl_buf, exp_buf,
                    eps=0.0, return_stats=True, show_step_bar=False
                )
                if st:
                    if st['event'] == 'success':
                        succ += 1
                        if st.get('success_time') is not None:
                            succ_times.append(st['success_time'])
                    elif st['event'] == 'collision':
                        coll += 1
                    else:
                        tout += 1
                    ep_rewards.append(st['total_reward'])
                    ep_steps.append(st['steps'])

                n = i + 1
                bar.set_postfix({
                    "succ": f"{succ/max(1,n):.2f}",
                    "coll": f"{coll/max(1,n):.2f}",
                    "timeout": f"{tout/max(1,n):.2f}"
                })

                if (i + 1) % 50 == 0 or (i + 1) == il_episodes:
                    nav_time = (np.mean(succ_times) if len(succ_times) else env.time_limit)
                    logging.info(
                        "IL   has success rate: %.2f, collision rate: %.2f, timeout rate: %.2f, nav time: %.2f, total reward: %.4f",
                        succ/max(1,n), coll/max(1,n), tout/max(1,n),
                        nav_time, (np.mean(ep_rewards) if ep_rewards else 0.0)
                    )
            # 完成后再汇总
            if il_episodes > 0:
                nav_time = (np.mean(succ_times) if len(succ_times) else env.time_limit)
                logging.info(
                    "IL   done | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.4f",
                    succ/il_episodes, coll/il_episodes, tout/il_episodes,
                    nav_time, (np.mean(ep_rewards) if ep_rewards else 0.0)
                )
        else:
            # 回退：整集运行接口，并打开 Explorer 的进度条
            stats = explorer.run_k_episodes(
                il_episodes, 'train', update_memory=True,
                imitation_learning=True, return_stats=True, show_tqdm=True
            )
            logging.info(
                "IL   has success rate: %.2f, collision rate: %.2f, timeout rate: %.2f, nav time: %.2f, total reward: %.4f",
                stats["success_rate"], stats["collision_rate"], stats["timeout_rate"],
                stats["nav_time"], stats["total_reward"]
            )

        robot.set_policy(policy)

        # BC 预训练
        bc_loss = bc_pretrain_actor(policy, exp_buf, device, iters=max(1, il_epochs),
                                    batch_size=batch_size, lr=il_lr)
        logging.info("BC pretrain done. avg_loss=%.6f, exp_buf=%d", bc_loss, len(exp_buf))

    # ----------- RL 主循环 -----------
    policy.set_env(env) if hasattr(policy, "set_env") else None
    robot.set_policy(policy)
    try:
        robot.print_info()
    except Exception:
        pass

    train_succ_hist, train_coll_hist, train_timeout_hist, train_reward_hist = [], [], [], []

    for episode in tqdm(range(train_episodes), desc="Training progress", unit="ep", dynamic_ncols=True):
        # epsilon
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, epsilon_decay))
        else:
            epsilon = epsilon_end
        if hasattr(policy, "set_epsilon"):
            policy.set_epsilon(epsilon)

        # DAgger 概率
        p_expert = get_expert_prob(episode, dagger_exp_full, dagger_exp_anneal, dagger_min_prob)

        # 评估
        if episode % evaluation_interval == 0:
            val_stats = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode, return_stats=True, show_tqdm=True)
            logging.info("VAL  ep=%d | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
                         episode, val_stats["success_rate"], val_stats["collision_rate"],
                         val_stats["timeout_rate"], val_stats["nav_time"], val_stats["total_reward"])

        # ===== 采样：逐步打标签（带进度条与统计）=====
        use_new = hasattr(explorer, 'collect_with_labels')
        if use_new:
            succ = coll = tout = 0
            ep_rewards = []
            for _ in tqdm(range(sample_episodes), desc="Collect (DAgger)", dynamic_ncols=True, leave=False, unit="ep"):
                st = explorer.collect_with_labels(
                    env, policy, expert_policy, rl_buf, exp_buf,
                    eps=epsilon, return_stats=True, show_step_bar=False
                )
                if st:
                    if st['event'] == 'success':   succ += 1
                    elif st['event'] == 'collision': coll += 1
                    else: tout += 1
                    ep_rewards.append(st['total_reward'])

            logging.info("TRAIN(collect) | succ=%.2f coll=%.2f timeout=%.2f reward=%.2f",
                         succ/max(1,sample_episodes), coll/max(1,sample_episodes),
                         tout/max(1,sample_episodes),
                         (np.mean(ep_rewards) if ep_rewards else 0.0))

            # 小概率整集专家（稳定早期策略）
            if np.random.rand() < p_expert:
                orig = robot.policy
                robot.set_policy(expert_policy)
                explorer.run_k_episodes(1, 'train', update_memory=True,
                                        episode=episode, return_stats=False, show_tqdm=True)
                robot.set_policy(orig)
        else:
            # 回退：整集专家/整集RL混采
            if np.random.rand() < p_expert:
                orig = robot.policy; robot.set_policy(expert_policy)
                explorer.run_k_episodes(sample_episodes, 'train', update_memory=True,
                                        episode=episode, return_stats=False, show_tqdm=True)
                robot.set_policy(orig)
            else:
                explorer.run_k_episodes(sample_episodes, 'train', update_memory=True,
                                        episode=episode, return_stats=False, show_tqdm=True)

        # ===== 训练 SAC（含 BC 正则）=====
        meter = trainer.optimize_batch(rl_buf, exp_buf, updates=train_batches)

        # 统计（不入库）
        tr = explorer.run_k_episodes(1, 'train', update_memory=False, episode=episode, return_stats=True, show_tqdm=False)
        train_succ_hist.append(tr["success_rate"])
        train_coll_hist.append(tr["collision_rate"])
        train_timeout_hist.append(tr["timeout_rate"])
        train_reward_hist.append(tr["total_reward"])
        logging.info("TRAIN has success rate: %.2f, collision rate: %.2f, timeout rate: %.2f, nav time: %.2f, total reward: %.4f",
                     tr["success_rate"], tr["collision_rate"], tr["timeout_rate"], tr["nav_time"], tr["total_reward"])

        # 保存
        if episode != 0 and episode % checkpoint_interval == 0:
            save_checkpoint(args.output_dir, policy, trainer, episode, extra={"meter": meter})
            logging.info("Checkpoint saved at ep=%d", episode)

        # 画图
        if episode % plot_interval == 0 and episode > 0:
            W = min(plot_interval, len(train_reward_hist))
            reward_ma = moving_average(train_reward_hist, W)
            succ_ma   = moving_average(train_succ_hist, W)
            coll_ma   = moving_average(train_coll_hist, W)
            tout_ma   = moving_average(train_timeout_hist, W)
            plt.figure(figsize=(12,6))
            plt.subplot(211); plt.plot(train_reward_hist, label='Train Avg Reward')
            if len(reward_ma) > 0: plt.axhline(np.mean(train_reward_hist[-W:]), ls='--', label=f'mean({W})')
            plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend()
            plt.subplot(212)
            x0 = np.arange(W-1, W-1+len(succ_ma))
            if len(succ_ma): plt.plot(x0, succ_ma, label=f'Success (w={W})')
            if len(coll_ma): plt.plot(x0, coll_ma, label=f'Collision (w={W})')
            if len(tout_ma): plt.plot(x0, tout_ma, label=f'Timeout (w={W})')
            plt.xlabel('Episode'); plt.ylabel('Rate'); plt.legend()
            plt.tight_layout()
            fig_name = os.path.join(args.output_dir, f"train_curves_ep{episode}.png")
            plt.savefig(fig_name); plt.close()
            logging.info("Plot saved: %s", fig_name)

        tqdm.write("Episode: {:d}/{:d}, epsilon: {:.4f}".format(episode + 1, train_episodes, epsilon))

    # 测试
    test_stats = explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, return_stats=True, show_tqdm=True)
    logging.info("TEST | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
                 test_stats["success_rate"], test_stats["collision_rate"],
                 test_stats["timeout_rate"], test_stats["nav_time"], test_stats["total_reward"])
    logging.info("Training complete.")

if __name__ == '__main__':
    main()

