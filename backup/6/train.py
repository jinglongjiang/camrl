#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, shutil, json, logging, argparse, configparser, random
import gym, git, numpy as np, torch
from tqdm.auto import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.memory import ReplayMemory, ExpertReplayMemory
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.explorer import Explorer
from crowd_sim.envs.utils.state import JointState

# ====== 高性能设置（Ampere+ 显著提速；安全）======
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

# 并发采样器
from crowd_nav.utils.parallel_sampler import ParallelSampler

ROBOT_VPREF_IDX = 7
DEFAULT_SEED = 42

# ----------------- 工具 -----------------
def set_global_seed(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic

def moving_average(x, w):
    if len(x) == 0:
        return np.array([])
    w = max(1, min(w, len(x)))
    return np.convolve(np.array(x, dtype=float), np.ones(w)/w, mode='valid')

class _quiet_logs:
    """临时把 root logger 降到 WARNING，用于静音 Explorer 的频繁 info 输出"""
    def __init__(self, level=logging.WARNING):
        self.level = level
        self.prev = None
    def __enter__(self):
        root = logging.getLogger()
        self.prev = root.level
        root.setLevel(self.level)
    def __exit__(self, exc_type, exc, tb):
        logging.getLogger().setLevel(self.prev if self.prev is not None else logging.INFO)

# ----------------- IL：BC 预训练 -----------------
def bc_pretrain_actor(policy, exp_buf, device, iters=2000, batch_size=256, lr=3e-4):
    if len(exp_buf) == 0:
        return 0.0
    actor = policy.actor
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    losses = []
    bar = tqdm(range(iters), desc="BC pretrain (Actor on demos)", dynamic_ncols=True, leave=False)
    for _ in bar:
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
        if len(losses) >= 10:
            bar.set_postfix({"loss(ema)": f"{np.mean(losses[-10:]):.4f}"})
    return float(np.mean(losses)) if losses else 0.0

# ----------------- 主程序 -----------------
def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='mamba')  # 你的工厂键为 'mamba'
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic=True)

    # 输出目录 & 日志
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

    # git
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
    sample_episodes = train_config.getint('train', 'sample_episodes', fallback=12)
    evaluation_interval = train_config.getint('train', 'evaluation_interval', fallback=50)
    checkpoint_interval = train_config.getint('train', 'checkpoint_interval', fallback=250)
    capacity = train_config.getint('train', 'capacity', fallback=200000)

    epsilon_start = train_config.getfloat('train', 'epsilon_start', fallback=0.1)
    epsilon_end   = train_config.getfloat('train', 'epsilon_end',   fallback=0.05)
    epsilon_decay = train_config.getfloat('train', 'epsilon_decay', fallback=1000)

    # 轻量评估参数
    val_subset        = train_config.getint('train', 'val_subset',        fallback=50)
    eval_full_every   = train_config.getint('train', 'eval_full_every',   fallback=200)
    train_stat_stride = train_config.getint('train', 'train_stat_stride', fallback=5)

    # SAC 超参
    gamma     = train_config.getfloat('trainer', 'gamma',     fallback=0.99)
    tau       = train_config.getfloat('trainer', 'tau',       fallback=0.005)
    lr_actor  = train_config.getfloat('trainer', 'lr_actor',  fallback=3e-4)
    lr_critic = train_config.getfloat('trainer', 'lr_critic', fallback=3e-4)
    lr_alpha  = train_config.getfloat('trainer', 'lr_alpha',  fallback=3e-4)
    batch_size= train_config.getint  ('trainer', 'batch_size', fallback=256)
    target_entropy = train_config.getfloat('trainer', 'target_entropy', fallback=-2.0)
    use_amp        = train_config.getboolean('trainer', 'use_amp',       fallback=True)
    grad_clip      = train_config.getfloat ('trainer', 'grad_clip',      fallback=1.0)

    # ----------- Trainer（仅一次） -----------
    trainer = Trainer(policy, device,
                      gamma=gamma, tau=tau,
                      lr_actor=lr_actor, lr_critic=lr_critic, lr_alpha=lr_alpha,
                      batch_size=batch_size,
                      target_entropy=target_entropy,
                      use_amp=use_amp,
                      grad_clip=grad_clip,
                      v_pref_idx=ROBOT_VPREF_IDX)

    # ----------- buffers / explorer / expert -----------
    rl_buf = ReplayMemory(capacity)
    exp_buf = ExpertReplayMemory(capacity)
    explorer = Explorer(env, robot, device, rl_buf, policy.gamma, target_policy=policy)

    il_policy_name = train_config.get('imitation_learning', 'il_policy', fallback='orca')
    expert_policy = policy_factory[il_policy_name](policy_config)
    expert_policy.configure(policy_config)
    expert_policy.set_device(device)
    expert_policy.multiagent_training = getattr(policy, 'multiagent_training', False)
    expert_policy.safety_space = train_config.getfloat('imitation_learning', 'safety_space', fallback=0.0)

    # ----------- 并发采样配置 -----------
    vec_enable     = train_config.getboolean('vectorize', 'enable',           fallback=True)
    vec_workers    = train_config.getint('vectorize', 'num_workers',          fallback=4)
    vec_eps_per    = train_config.getint('vectorize', 'episodes_per_worker',  fallback=3)
    broadcast_k    = train_config.getint('vectorize', 'broadcast_interval',   fallback=5)
    worker_device  = train_config.get('vectorize', 'worker_device',
                                      fallback=('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'))

    sampler = None
    if vec_enable:
        sampler = ParallelSampler(
            num_workers=vec_workers,
            env_config_path=args.env_config,
            policy_config_path=args.policy_config,
            policy_name_student=args.policy,
            policy_name_expert=il_policy_name,
            device_str=worker_device,          # <<< 关键：worker 放到 CUDA 做前向（mamba_ssm 需要）
            base_seed=args.seed * 1000
        )
        sampler.broadcast_policy(policy)

    # ----------- IL: 专家开车 + BC（线下一次；无 DAgger） ----------- 
    il_episodes = train_config.getint('imitation_learning', 'il_episodes', fallback=0)
    il_epochs   = train_config.getint('imitation_learning', 'il_epochs',   fallback=0)
    il_lr       = train_config.getfloat('imitation_learning', 'il_learning_rate', fallback=3e-4)
    if il_episodes > 0:
        # 为确保 ExpertReplayMemory 标签完整性，这里保持单进程收集
        robot.set_policy(expert_policy)

        succ = coll = tout = 0
        succ_times, ep_rewards = [], []
        bar = tqdm(range(il_episodes), desc="IL collect (expert drives)", dynamic_ncols=True, unit="ep")
        for i in bar:
            st = explorer.collect_with_labels(
                env, expert_policy, expert_policy, rl_buf, exp_buf,
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
            n = i + 1
            bar.set_postfix({
                "succ": f"{succ/max(1,n):.2f}",
                "coll": f"{coll/max(1,n):.2f}",
                "timeout": f"{tout/max(1,n):.2f}",
                "exp_buf": f"{len(exp_buf)}"
            })

        nav_time = (np.mean(succ_times) if len(succ_times) else env.time_limit)
        logging.info("IL   done | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.4f | demos=%d",
                     succ/max(1,il_episodes), coll/max(1,il_episodes), tout/max(1,il_episodes),
                     nav_time, (np.mean(ep_rewards) if ep_rewards else 0.0), len(exp_buf))

        # 仅 Actor 的 BC 预训练
        if il_epochs > 0:
            bc_loss = bc_pretrain_actor(policy, exp_buf, device, iters=max(1, il_epochs),
                                        batch_size=batch_size, lr=il_lr)
            logging.info("BC pretrain done. avg_loss=%.6f, exp_buf=%d", bc_loss, len(exp_buf))

        # 清空 RL 回放池（不把专家轨迹混入 RL 学习）
        rl_buf = ReplayMemory(capacity)
        explorer.memory = rl_buf

        # 还回学生策略
        robot.set_policy(policy)

    # ----------- RL 主循环（SACfD：固定 demos 正则；无 DAgger） -----------
    policy.set_env(env) if hasattr(policy, "set_env") else None
    robot.set_policy(policy)
    try:
        robot.print_info()
    except Exception:
        pass

    train_succ_hist, train_coll_hist, train_timeout_hist, train_reward_hist = [], [], [], []
    best_val = None
    plot_interval = 50

    for episode in tqdm(range(train_episodes), desc="Training progress", unit="ep", dynamic_ncols=True):
        # epsilon
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, epsilon_decay))
        else:
            epsilon = epsilon_end
        if hasattr(policy, "set_epsilon"):
            policy.set_epsilon(epsilon)

        # 定期广播学生策略到并发 worker
        if sampler and (episode % max(1, broadcast_k) == 0):
            sampler.broadcast_policy(policy)

        # 评估：子集 + 定期全量
        if episode % evaluation_interval == 0:
            do_full = (episode % eval_full_every == 0)
            val_k = env.case_size['val'] if do_full else min(val_subset, env.case_size['val'])
            val_stats = explorer.run_k_episodes(val_k, 'val', episode=episode,
                                                return_stats=True, show_tqdm=True)
            logging.info("VAL  ep=%d | [%s] succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
                         episode, "FULL" if do_full else f"SUB({val_k})",
                         val_stats["success_rate"], val_stats["collision_rate"],
                         val_stats["timeout_rate"], val_stats["nav_time"], val_stats["total_reward"])
            if (best_val is None or
                val_stats["success_rate"] > best_val.get("success_rate", -1) or
                (val_stats["success_rate"] == best_val.get("success_rate", -1) and
                 val_stats["collision_rate"] < best_val.get("collision_rate", 1e9))):
                best_val = dict(val_stats); best_val["episode"] = int(episode)

        # ===== 采样（并发优先；否则单进程一次性采样）=====
        if sampler:
            try:
                tr, traj = sampler.collect(
                    episodes_per_worker=vec_eps_per,
                    episode_idx=episode,
                    use_expert=False
                )
            except RuntimeError as e:
                logging.error("Parallel sampler error: %s", e)
                raise
            # 写入主进程回放池
            for (args_, kwargs_) in traj:
                rl_buf.push(*args_, **kwargs_)
            logging.info("TRAIN(interact, parallel) | succ=%.2f coll=%.2f timeout=%.2f reward=%.2f | rl_buf=%d",
                         tr.get("success_rate", 0.0),
                         tr.get("collision_rate", 0.0),
                         tr.get("timeout_rate", 0.0),
                         tr.get("total_reward", 0.0),
                         len(rl_buf))
        else:
            with _quiet_logs():  # 静音 Explorer 内部的 info 打印
                tr = explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode,
                                             return_stats=True, show_tqdm=True)
            logging.info("TRAIN(interact) | succ=%.2f coll=%.2f timeout=%.2f reward=%.2f | rl_buf=%d",
                         tr.get("success_rate", 0.0), tr.get("collision_rate", 0.0),
                         tr.get("timeout_rate", 0.0), tr.get("total_reward", 0.0),
                         len(rl_buf))

        # ===== Optimize（SACfD：仅用固定 demos 作正则）=====
        # 线性调度：p_demo 0.5→0.1；lambda_bc 1.0→0（到 60% 进度）
        t = episode / max(1, train_episodes - 1)
        p_demo = float(np.interp(t, [0.0, 1.0], [0.5, 0.1]))
        lambda_bc = float(np.interp(min(t, 0.6), [0.0, 0.6], [1.0, 0.0]))
        meter = trainer.optimize_batch(
            rl_buf, exp_buf, updates=train_config.getint('train', 'train_batches', fallback=128),
            p_demo=p_demo, lambda_bc=lambda_bc, use_q_filter=True
        )

        # 快速统计（本轮交互聚合统计）
        train_succ_hist.append(tr.get("success_rate", 0.0))
        train_coll_hist.append(tr.get("collision_rate", 0.0))
        train_timeout_hist.append(tr.get("timeout_rate", 0.0))
        train_reward_hist.append(tr.get("total_reward", 0.0))
        logging.info("TRAIN ep=%d | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.4f | p_demo=%.2f λ_bc=%.2f | meter=%s",
                     episode, tr.get("success_rate", 0.0), tr.get("collision_rate", 0.0),
                     tr.get("timeout_rate", 0.0), tr.get("nav_time", 0.0),
                     tr.get("total_reward", 0.0), p_demo, lambda_bc,
                     {k: round(v, 4) for k, v in (meter or {}).items()})

        # 保存
        if episode != 0 and episode % checkpoint_interval == 0:
            model_path = os.path.join(args.output_dir, f'rl_model_ep{episode}.pth')
            if hasattr(policy, "save_model"):
                policy.save_model(model_path)
            elif hasattr(policy, "get_state_dict"):
                torch.save(policy.get_state_dict(), model_path)
            else:
                torch.save(policy.state_dict(), model_path)
            logging.info("Checkpoint saved at ep=%d -> %s", episode, model_path)

        # 画图（每 50 ep）
        plot_interval = 50
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

    # ----------- TEST -----------
    test_stats = explorer.run_k_episodes(env.case_size['test'], 'test', episode=train_episodes,
                                         return_stats=True, show_tqdm=True)
    logging.info("TEST | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
                 test_stats["success_rate"], test_stats["collision_rate"],
                 test_stats["timeout_rate"], test_stats["nav_time"], test_stats["total_reward"])

    # ----------- FINAL 汇总 -----------
    def _mean_last(arr, n=100):
        return float(np.mean(arr[-n:])) if len(arr) else float('nan')

    W = 100
    train_summary = {
        "window": W,
        "success_rate": _mean_last(train_succ_hist,   W),
        "collision_rate": _mean_last(train_coll_hist, W),
        "timeout_rate": _mean_last(train_timeout_hist,W),
        "total_reward": _mean_last(train_reward_hist, W),
    }

    best_val_local = locals().get("best_val", None)
    if best_val_local is not None:
        logging.info(
            "FINAL | Train(last%02d): succ=%.2f coll=%.2f timeout=%.2f reward=%.2f | "
            "BestVAL(ep=%d): succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f | "
            "TEST: succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
            W,
            train_summary["success_rate"], train_summary["collision_rate"],
            train_summary["timeout_rate"], train_summary["total_reward"],
            best_val_local["episode"], best_val_local["success_rate"], best_val_local["collision_rate"],
            best_val_local["timeout_rate"], best_val_local["nav_time"], best_val_local["total_reward"],
            test_stats["success_rate"], test_stats["collision_rate"],
            test_stats["timeout_rate"], test_stats["nav_time"], test_stats["total_reward"]
        )
    else:
        logging.info(
            "FINAL | Train(last%02d): succ=%.2f coll=%.2f timeout=%.2f reward=%.2f | "
            "TEST: succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
            W,
            train_summary["success_rate"], train_summary["collision_rate"],
            train_summary["timeout_rate"], train_summary["total_reward"],
            test_stats["success_rate"], test_stats["collision_rate"],
            test_stats["timeout_rate"], test_stats["nav_time"], test_stats["total_reward"]
        )

    with open(os.path.join(args.output_dir, "final_summary.json"), "w") as f:
        json.dump({
            "train_last_window": train_summary,
            "best_val": best_val_local,
            "test": test_stats
        }, f, indent=2)

    # 回收并发资源
    if sampler is not None:
        sampler.close()

    logging.info("Training complete.")

if __name__ == '__main__':
    main()

