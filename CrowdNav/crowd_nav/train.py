#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, shutil, logging, argparse, configparser, random
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

# ====== 统一口径：v_pref 索引（必须与环境/Trainer/推理一致）======
ROBOT_VPREF_IDX = 7
DEFAULT_SEED = 42

# ----------------- 工具 -----------------
def set_global_seed(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic

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

# ----------------- 统一动作缩放（单一入口，BC/训练/推理都用） -----------------
def phys_to_norm_actions(a_phys_np: np.ndarray, s_np: np.ndarray, action_scale: float) -> np.ndarray:
    """物理动作 -> [-1,1]。s_np 为观测（含 v_pref），必须和 ROBOT_VPREF_IDX 一致。"""
    v_pref = np.clip(s_np[:, ROBOT_VPREF_IDX:ROBOT_VPREF_IDX+1], 1e-6, None)
    return np.clip(a_phys_np / (v_pref * action_scale), -1.0, 1.0)

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
        a_norm = phys_to_norm_actions(a_star, s, getattr(policy, 'action_scale', 1.0))  # 统一口径

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
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()

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
    parser.add_argument('--policy', type=str, default='mamba')  # policy_factory 键
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

    # 取出环境里的终局奖励（用于“终局曲线”）
    succ_bonus      = env_config.getfloat('reward', 'success_reward',      fallback=10.0)
    coll_penalty    = env_config.getfloat('reward', 'collision_penalty',   fallback=-3.0)
    timeout_penalty = env_config.getfloat('reward', 'timeout_penalty',     fallback=0.0)

    # ----------- train params -----------
    train_config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    train_config.read(args.train_config)

    train_episodes     = train_config.getint('train', 'train_episodes')
    sample_episodes    = train_config.getint('train', 'sample_episodes', fallback=12)
    evaluation_interval= train_config.getint('train', 'evaluation_interval', fallback=150)
    checkpoint_interval= train_config.getint('train', 'checkpoint_interval', fallback=500)
    capacity           = train_config.getint('train', 'capacity', fallback=100000)  # 100k，训练更稳

    epsilon_start = train_config.getfloat('train', 'epsilon_start', fallback=0.08)
    epsilon_end   = train_config.getfloat('train', 'epsilon_end',   fallback=0.03)
    epsilon_decay = train_config.getfloat('train', 'epsilon_decay', fallback=1500)

    # 轻量评估参数
    val_subset        = train_config.getint('train', 'val_subset',        fallback=50)
    eval_full_every   = train_config.getint('train', 'eval_full_every',   fallback=300)
    train_stat_stride = train_config.getint('train', 'train_stat_stride', fallback=5)

    # SAC 超参
    gamma     = train_config.getfloat('trainer', 'gamma',     fallback=0.95)
    tau       = train_config.getfloat('trainer', 'tau',       fallback=0.0075)   # 略强的 Polyak
    lr_actor  = train_config.getfloat('trainer', 'lr_actor',  fallback=3e-4)
    lr_critic = train_config.getfloat('trainer', 'lr_critic', fallback=3.6e-4)   # 1.2x actor
    lr_alpha  = train_config.getfloat('trainer', 'lr_alpha',  fallback=2.4e-4)   # 0.8x actor
    batch_size= train_config.getint  ('trainer', 'batch_size', fallback=1024)
    target_entropy = train_config.getfloat('trainer', 'target_entropy', fallback=-1.8)  # 会被日程覆盖
    use_amp        = train_config.getboolean('trainer', 'use_amp',       fallback=True)
    grad_clip      = train_config.getfloat ('trainer', 'grad_clip',      fallback=1.0)
    awbc_beta      = train_config.getfloat ('trainer', 'awbc_beta',      fallback=2.5)

    # ----------- Trainer（仅一次） -----------
    trainer = Trainer(policy, device,
                      gamma=gamma, tau=tau,
                      lr_actor=lr_actor, lr_critic=lr_critic, lr_alpha=lr_alpha,
                      batch_size=batch_size,
                      target_entropy=target_entropy,
                      use_amp=use_amp,
                      grad_clip=grad_clip,
                      v_pref_idx=ROBOT_VPREF_IDX,
                      awbc_beta=awbc_beta)

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
            device_str=worker_device,          # worker 在 CUDA 前向（mamba_ssm 需要）
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

        # —— 仅日志：专家数据质控（不过度筛掉慢样本，保多样性） ——
        if len(succ_times) > 0 and len(exp_buf) > 0:
            median_time = float(np.median(succ_times))
            fast_threshold = median_time * 0.8
            logging.info("Enhanced expert quality control: median=%.2f, fast_threshold=%.2f", 
                        median_time, fast_threshold)

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

        # —— 轻量 BC SanITY（不加噪，快速验证口径一致性） ——
        with _quiet_logs():
            val_k = min(30, env.case_size['val'])
            bc_val = explorer.run_k_episodes(val_k, 'val', episode=0, return_stats=True, show_tqdm=False)
        logging.info("BC SANITY | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
                     bc_val["success_rate"], bc_val["collision_rate"],
                     bc_val["timeout_rate"], bc_val["nav_time"], bc_val["total_reward"])

    # ----------- RL 主循环（SACfD：固定 demos 正则；无 DAgger） -----------
    policy.set_env(env) if hasattr(policy, "set_env") else None
    robot.set_policy(policy)
    try:
        robot.print_info()
    except Exception:
        pass

    # 历史记录（含“终局期望回报”）
    train_succ_hist, train_coll_hist, train_timeout_hist = [], [], []
    train_reward_hist = []          # shaped 总回报
    train_term_hist   = []          # 终局期望回报：succ*R + coll*P + timeout*P_t

    plot_interval = 50
    best_val = None

    for episode in tqdm(range(train_episodes), desc="Training progress", unit="ep", dynamic_ncols=True):
        # epsilon（若策略支持）
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, epsilon_decay))
        else:
            epsilon = epsilon_end
        if hasattr(policy, "set_epsilon"):
            policy.set_epsilon(epsilon)

        # ---- 目标熵日程：-2.2 → -2.8，覆盖前 80% ----
        # 降低探索强度，让策略更确定性，减少随机冒险
        ent_t = episode / max(1, int(0.8 * train_episodes))
        target_H_hi, target_H_lo = -2.2, -2.8  # 更低的目标熵，减少探索
        trainer.target_entropy = float(target_H_hi + (target_H_lo - target_H_hi) * min(1.0, ent_t))

        # ---- 学习率日程：10% warmup → 60% 保持 → 30% 线性衰减到 0.4x ----
        lr_progress = episode / max(1, train_episodes)
        if lr_progress <= 0.1:      # 前10% warmup（0.5x → 1.0x）
            lr_mult = 0.5 + 0.5 * (lr_progress / 0.1)
        elif lr_progress <= 0.6:    # 10%-60% 高学习率探索
            lr_mult = 1.0
        else:                       # 后40% 逐步衰减到 0.4x
            lr_mult = 1.0 - 0.6 * ((lr_progress - 0.6) / 0.4)

        current_lr_actor  = lr_actor  * lr_mult
        current_lr_critic = lr_critic * lr_mult
        current_lr_alpha  = lr_alpha  * lr_mult

        # 更新优化器学习率
        for param_group in trainer.opt_actor.param_groups:
            param_group['lr'] = current_lr_actor
        for param_group in trainer.opt_q1.param_groups:
            param_group['lr'] = current_lr_critic
        for param_group in trainer.opt_q2.param_groups:
            param_group['lr'] = current_lr_critic
        for param_group in trainer.opt_alpha.param_groups:
            param_group['lr'] = current_lr_alpha

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
            tr, traj = sampler.collect(
                episodes_per_worker=vec_eps_per,
                episode_idx=episode,
                use_expert=False
            )
            # 写入主进程回放池（优先用批量接口以省 Python 循环）
            if hasattr(rl_buf, "push_many"):
                rl_buf.push_many([args_ for (args_, _) in traj])
            else:
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

        # ===== Optimize（SACfD：固定 demos 正则；无 DAgger）=====
        # 线性进度刻度
        t = episode / max(1, train_episodes - 1)

        # —— 更持久的 BC 日程：p_demo 0.50→0.35；λ_bc 0.95→0.70→0.40 ——
        # 大幅增强BC正则，让策略更保守，减少冒进
        p_demo    = float(np.interp(t, [0.0, 1.0],        [0.50, 0.35]))  # 保持更高的专家数据比例
        lambda_bc = float(np.interp(t, [0.0, 0.6, 1.0],   [0.95, 0.70, 0.40]))  # BC权重衰减更慢

        # —— 固定 updates（速度稳定）：早期 40，之后恒定 base；成功率>0.6 降到 0.85× ——
        updates_base = train_config.getint('train', 'train_batches', fallback=120)
        if len(rl_buf) < 10_000:
            updates = 40
        else:
            updates = updates_base
            # 简单自适应：近 50 集成功率 > 0.6 降低少量 SGD 步以稳住策略
            if len(train_succ_hist) >= 50 and np.mean(train_succ_hist[-50:]) > 0.6:
                updates = max(102, int(updates_base * 0.85))

        meter = trainer.optimize_batch(
            rl_buf, exp_buf, updates=updates,
            p_demo=p_demo, lambda_bc=lambda_bc, use_q_filter=True
        )

        # —— 终局期望回报（只看成功/碰撞/超时的终局项）——
        term_ret = (
            tr.get("success_rate", 0.0)  * succ_bonus +
            tr.get("collision_rate", 0.0)* coll_penalty +
            tr.get("timeout_rate", 0.0)  * timeout_penalty
        )

        # 快速统计（本轮交互聚合统计）
        train_succ_hist.append(tr.get("success_rate", 0.0))
        train_coll_hist.append(tr.get("collision_rate", 0.0))
        train_timeout_hist.append(tr.get("timeout_rate", 0.0))
        train_reward_hist.append(tr.get("total_reward", 0.0))  # shaped
        train_term_hist.append(term_ret)                       # terminal-only
        logging.info("TRAIN ep=%d | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.4f | p_demo=%.2f λ_bc=%.2f upd=%d | meter=%s",
                     episode, tr.get("success_rate", 0.0), tr.get("collision_rate", 0.0),
                     tr.get("timeout_rate", 0.0), tr.get("nav_time", 0.0),
                     tr.get("total_reward", 0.0), p_demo, lambda_bc, updates,
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

        # 画图（每 50 ep）：三联图 + 双滑窗（100 / 500）
        if episode % plot_interval == 0 and episode > 0:
            Ws = 100
            Wl = 500

            def ma(arr, w):
                import numpy as _np
                if len(arr) == 0: return _np.array([])
                w = max(1, min(w, len(arr)))
                return _np.convolve(_np.array(arr, dtype=float), _np.ones(w)/w, mode='valid')

            # 1) Shaped Return（总回报）
            R_ma_s = ma(train_reward_hist, Ws)
            R_ma_l = ma(train_reward_hist, Wl)
            x_s_R = np.arange(Ws-1, Ws-1+len(R_ma_s))
            x_l_R = np.arange(Wl-1, Wl-1+len(R_ma_l))

            # 2) 终局回报（只由 success/collision/timeout 的终局项构成）
            T_ma_s = ma(train_term_hist, Ws)
            T_ma_l = ma(train_term_hist, Wl)
            x_s_T = np.arange(Ws-1, Ws-1+len(T_ma_s))
            x_l_T = np.arange(Wl-1, Wl-1+len(T_ma_l))

            # 3) 成功/碰撞/超时（短窗 100）
            S_ma = ma(train_succ_hist, Ws)
            C_ma = ma(train_coll_hist, Ws)
            O_ma = ma(train_timeout_hist, Ws)
            x_SCO = np.arange(Ws-1, Ws-1+len(S_ma))

            plt.figure(figsize=(12, 10))

            # --- Panel 1: Shaped Return（MA100/MA500）---
            plt.subplot(311)
            if len(R_ma_s): plt.plot(x_s_R, R_ma_s, label=f'Shaped Return (MA{Ws})')
            if len(R_ma_l): plt.plot(x_l_R, R_ma_l, label=f'Shaped Return (MA{Wl})')
            plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend(); plt.grid(alpha=0.2)

            # --- Panel 2: Terminal-only Return（MA100/MA500）---
            plt.subplot(312)
            if len(T_ma_s): plt.plot(x_s_T, T_ma_s, label=f'Terminal Return (MA{Ws})')
            if len(T_ma_l): plt.plot(x_l_T, T_ma_l, label=f'Terminal Return (MA{Wl})')
            plt.xlabel('Episode'); plt.ylabel('Terminal Reward'); plt.legend(); plt.grid(alpha=0.2)

            # --- Panel 3: Rates（MA100）---
            plt.subplot(313)
            if len(S_ma): plt.plot(x_SCO, S_ma, label=f'Success (MA{Ws})')
            if len(C_ma): plt.plot(x_SCO, C_ma, label=f'Collision (MA{Ws})')
            if len(O_ma): plt.plot(x_SCO, O_ma, label=f'Timeout (MA{Ws})')
            plt.xlabel('Episode'); plt.ylabel('Rate'); plt.legend(); plt.grid(alpha=0.2)

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
        "terminal_reward": _mean_last(train_term_hist, W),
    }

    best_val_local = locals().get("best_val", None)
    if best_val_local is not None:
        logging.info(
            "FINAL | Train(last%02d): succ=%.2f coll=%.2f timeout=%.2f reward=%.2f term=%.2f | "
            "BestVAL(ep=%d): succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f | "
            "TEST: succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
            W,
            train_summary["success_rate"], train_summary["collision_rate"],
            train_summary["timeout_rate"], train_summary["total_reward"], train_summary["terminal_reward"],
            best_val_local["episode"], best_val_local["success_rate"], best_val_local["collision_rate"],
            best_val_local["timeout_rate"], best_val_local["nav_time"], best_val_local["total_reward"],
            test_stats["success_rate"], test_stats["collision_rate"],
            test_stats["timeout_rate"], test_stats["nav_time"], test_stats["total_reward"]
        )
    else:
        logging.info(
            "FINAL | Train(last%02d): succ=%.2f coll=%.2f timeout=%.2f reward=%.2f term=%.2f | "
            "TEST: succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
            W,
            train_summary["success_rate"], train_summary["collision_rate"],
            train_summary["timeout_rate"], train_summary["total_reward"], train_summary["terminal_reward"],
            test_stats["success_rate"], test_stats["collision_rate"],
            test_stats["timeout_rate"], test_stats["nav_time"], test_stats["total_reward"]
        )

    # 回收并发资源
    if sampler is not None:
        sampler.close()

    logging.info("Training complete.")

if __name__ == '__main__':
    main()
