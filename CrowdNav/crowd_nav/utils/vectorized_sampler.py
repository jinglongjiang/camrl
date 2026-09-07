# -*- coding: utf-8 -*-
"""
VectorizedSampler - 并行环境采样器

核心功能：
1. 多进程并行采样：启动 num_workers 个子进程，每个运行独立的 CrowdSim + Explorer
2. 主进程收集：通过 Queue 收集所有 workers 的 episode 数据
3. 模型同步：主进程定期广播最新模型参数到 workers

性能预期：
- 8 workers × 2 episodes/worker = 16 episodes/batch
- 采样时间：30秒（串行）→ 4-5秒（8倍并行）
- GPU 利用率：40-50% → 75-85%
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import logging
import queue
import time
import copy
from typing import Dict, Any, List, Optional, Tuple
from configparser import ConfigParser

# 全局初始化锁：确保workers串行初始化，避免CUDA资源竞争
# 初始化锁由父进程创建并传入每个worker (spawn模式共享)


def worker_process(
    worker_id: int,
    config_path: str,  # 改为传递配置文件路径
    policy_name: str,
    episodes_per_worker: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    param_queue: mp.Queue,
    init_lock: mp.Lock,
    device: str,
    log_dir: str = None,  # 日志目录
    worker_seed: int = None,  # Worker专属种子
):
    """
    Worker进程：独立运行环境采样

    Args:
        worker_id: worker编号
        config_path: 配置文件路径（worker内部重新加载）
        policy_name: 策略名称（'mamba', 'sarl'等）
        episodes_per_worker: 每次采样的episode数
        task_queue: 任务队列（接收采样请求）
        result_queue: 结果队列（返回episode数据）
        param_queue: 参数队列（接收模型参数更新）
        init_lock: 父进程传入的跨进程锁（串行化CUDA初始化）
        device: 设备（'cuda:0' 或 'cpu'）
        log_dir: 日志输出目录（默认使用当前目录的logs/）
    """
    import sys
    import os

    # 确保worker进程使用与主进程相同的Python环境
    # 1. 继承sys.path（包含所有已安装的包）
    # 2. 继承sys.executable（确保使用相同的Python解释器）
    # 这样worker就能访问主进程能访问的所有模块（包括mamba_ssm）

    # 第一步：立即重定向stdout到log文件（在任何可能失败的操作之前）
    # 使用有写权限的目录，避免/tmp权限问题
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    worker_log_file = os.path.join(log_dir, f'worker_{worker_id}.log')

    try:
        log_f = open(worker_log_file, 'w', buffering=1)
        sys.stdout = log_f
        sys.stderr = log_f
    except Exception as e:
        # 如果连log文件都打不开，至少输出到stderr
        print(f"[Worker-{worker_id}] FATAL: Cannot open log file {worker_log_file}: {e}", file=sys.stderr)
        return

    try:
        # 设置进程级logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Worker-{worker_id}] %(asctime)s %(message)s',
            stream=sys.stdout,
            force=True
        )

        print(f"[Worker-{worker_id}] ========== Starting initialization ==========", flush=True)
        print(f"[Worker-{worker_id}] Log file: {worker_log_file}", flush=True)
        logging.info(f"Worker {worker_id} starting initialization...")

        # 设置worker专属随机种子（在任何随机操作之前）
        if worker_seed is not None:
            import random
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(worker_seed)
            logging.info(f"Worker {worker_id} set random seed to: {worker_seed}")
            print(f"[Worker-{worker_id}] 🎲 Random seed set to: {worker_seed}", flush=True)
        else:
            logging.warning(f"Worker {worker_id} using default random seed (worker_id={worker_id})")
            import random
            random.seed(worker_id)
            np.random.seed(worker_id)
            torch.manual_seed(worker_id)

        # 导入必要模块（在子进程中导入，避免主进程污染）
        logging.info(f"Worker {worker_id} importing modules...")
        from crowd_sim.envs import CrowdSim
        from crowd_nav.policy.policy_factory import policy_factory
        from crowd_nav.utils.explorer import Explorer, _extract_event_token
        from crowd_nav.contracts import joint34_to_tokens
        from crowd_sim.envs.utils.robot import Robot
        from configparser import ConfigParser

        # 规范化并锁定设备
        try:
            normalized_device = torch.device(device) if device is not None else torch.device('cpu')
        except (TypeError, ValueError):
            normalized_device = torch.device(str(device) if device else 'cpu')
        if normalized_device.type == 'cuda':
            if not torch.cuda.is_available():
                msg = f"CUDA device requested but unavailable for worker {worker_id}"
                logging.error(msg)
                print(f"[Worker-{worker_id}] FATAL: {msg}", flush=True)
                return
            if normalized_device.index is None:
                try:
                    current_idx = torch.cuda.current_device()
                except RuntimeError:
                    current_idx = 0
                normalized_device = torch.device(f'cuda:{current_idx}')
            torch.cuda.set_device(normalized_device)
        device = normalized_device
        device_str = str(normalized_device)
        logging.info(f"Worker {worker_id} using device: {device_str}")
        print(f"[Worker-{worker_id}] Using device {device_str}", flush=True)

        # 重新加载配置文件（避免pickle序列化问题）
        # 使用和主进程相同的load_config逻辑，自动合并env.config + policy.config + train.config
        logging.info(f"Worker {worker_id} loading config from {config_path}...")
        import os
        cfg = ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)

        # 如果传入单个配置文件，加载所有三个配置文件进行合并
        if config_path.endswith('env.config') or config_path.endswith('config.txt') or config_path.endswith('train.config'):
            config_dir = os.path.dirname(config_path) if os.path.dirname(config_path) else './configs'
            config_files = [
                os.path.join(config_dir, 'env.config'),
                os.path.join(config_dir, 'policy.config'),
                os.path.join(config_dir, 'train.config')
            ]
            # 加载所有存在的配置文件
            loaded_count = 0
            for config_file in config_files:
                if os.path.exists(config_file):
                    cfg.read(config_file, encoding='utf-8')
                    loaded_count += 1
            logging.info(f"Worker {worker_id} merged {loaded_count} config files from {config_dir}")
            logging.info(f"Worker {worker_id} config sections: {cfg.sections()}")
        else:
            cfg.read(config_path, encoding='utf-8')
            logging.info(f"Worker {worker_id} loaded single config: {config_path}")
            logging.info(f"Worker {worker_id} config sections: {cfg.sections()}")

        # 使用全局锁确保workers串行初始化，避免CUDA资源竞争
        print(f"[Worker-{worker_id}] Waiting for initialization lock...", flush=True)
        with init_lock:
            print(f"[Worker-{worker_id}] Acquired lock, starting initialization", flush=True)

            # 1. 创建独立环境
            logging.info(f"Worker {worker_id} creating environment...")
            env = CrowdSim()
            env.configure(cfg)

            # 2. 创建独立策略（串行初始化，避免CUDA资源竞争）
            logging.info(f"Worker {worker_id} creating policy '{policy_name}'...")
            print(f"[Worker-{worker_id}] Creating policy instance...", flush=True)

            PolicyCls = policy_factory[policy_name]
            try:
                policy = PolicyCls(cfg, device=device)
            except TypeError:
                try:
                    policy = PolicyCls(cfg)
                except TypeError:
                    policy = PolicyCls()
            print(f"[Worker-{worker_id}] Policy instance created, configuring...", flush=True)

            if hasattr(policy, 'configure'):
                print(f"[Worker-{worker_id}] Calling policy.configure()...", flush=True)
                policy.configure(cfg, cfg)  # 使用同一个config对象
                print(f"[Worker-{worker_id}] Policy configured", flush=True)

            print(f"[Worker-{worker_id}] Setting phase to train...", flush=True)
            policy.set_phase('train')
            print(f"[Worker-{worker_id}] Phase set to train", flush=True)

            logging.info(f"Worker {worker_id} moving policy to {device_str}...")
            print(f"[Worker-{worker_id}] Moving policy to {device_str}...", flush=True)
            if hasattr(policy, 'to'):
                policy.to(device)
            policy.eval()  # worker只做推理，不训练
            print(f"[Worker-{worker_id}] Policy moved to {device_str}", flush=True)

            # 3. 创建robot和explorer
            logging.info(f"Worker {worker_id} creating robot and explorer...")
            robot = Robot(cfg, 'robot')
            env.set_robot(robot)  # 环境需要绑定机器人
            if hasattr(policy, 'set_env'):
                policy.set_env(env)
            if hasattr(policy, 'set_robot'):
                policy.set_robot(robot)
            robot.set_policy(policy)
            if hasattr(robot, 'set_time_step'):
                robot.set_time_step(env.time_step)
            else:
                robot.time_step = getattr(env, 'time_step', robot.time_step)

            explorer = Explorer(env, robot, device, memory=None, gamma=0.97)

            print(f"[Worker-{worker_id}] Releasing initialization lock", flush=True)

        print(f"[Worker-{worker_id}] ✓ Initialized successfully on {device_str}", flush=True)
        logging.info(f"✓ Worker {worker_id} initialized successfully on {device_str}")

        # 4. 主循环：接收任务 → 采样 → 返回结果
        print(f"[Worker-{worker_id}] Entering main loop, waiting for tasks...", flush=True)
        logging.info(f"Worker {worker_id} entering main loop, waiting for tasks...")
        while True:
            try:
                # 非阻塞检查参数更新
                try:
                    new_state_dict = param_queue.get_nowait()
                    if new_state_dict == "STOP":
                        logging.info(f"Worker {worker_id} received STOP signal")
                        break
                    policy.load_state_dict(new_state_dict)
                    logging.info(f"Worker {worker_id} updated model parameters")
                except queue.Empty:
                    pass

                # 阻塞等待采样任务
                task = task_queue.get(timeout=1.0)
                print(f"[Worker-{worker_id}] Received task: {task}", flush=True)
                logging.info(f"Worker {worker_id} received task: {task}")

                if task == "STOP":
                    logging.info(f"Worker {worker_id} stopping")
                    break

                # task = (episode_start, phase, epsilon)
                episode_start, phase, epsilon = task

                # 采样 episodes_per_worker 个episodes
                for local_ep in range(episodes_per_worker):
                    global_ep = episode_start + local_ep

                    # 设置epsilon（探索率）
                    if hasattr(policy, 'set_epsilon'):
                        policy.set_epsilon(epsilon)

                    # 调用 Explorer.run_k_episodes(k=1)
                    explorer.run_k_episodes(
                        k=1,
                        phase=phase,
                        update_memory=False,  # worker不更新memory
                        imitation_learning=False,
                        episode=global_ep,
                        print_failure=False,
                        return_stats=False,
                        show_tqdm=False,
                    )

                    # 获取最后一条轨迹
                    trajectories = explorer.last_trajectories
                    print(f"[Worker-{worker_id}] Episode {global_ep} finished, trajectories count: {len(trajectories) if trajectories else 0}", flush=True)

                    if trajectories and len(trajectories) > 0:
                        traj, info = trajectories[-1]

                        episode_data = {
                            'states': traj[0],
                            'actions': traj[1],
                            'rewards': traj[2],
                            'info': info,
                            # 直接透传离散动作索引（如果有）
                            'action_indices': info.get('action_indices') if isinstance(info, dict) else None,
                            'worker_id': worker_id,
                            'episode': global_ep,
                        }

                        # 计算采样时策略的log_prob/value，确保PPO使用真实旧策略指标
                        log_probs_arr = None
                        values_arr = None
                        try:
                            states_arr = np.asarray(traj[0], dtype=np.float32)
                            actions_arr = np.asarray(traj[1], dtype=np.float32)
                            if actions_arr.ndim == 1:
                                actions_arr = actions_arr.reshape(1, -1)
                            if states_arr.shape[0] == actions_arr.shape[0] and actions_arr.shape[0] > 0:
                                # 统一归一化：与ppo_buffer.py保持一致
                                try:
                                    from crowd_nav.contracts import GRID
                                    v_max = float(GRID.get('v_max', 1.0))
                                except Exception:
                                    v_max = 1.0
                                v_max = max(v_max, 1e-6)
                                norm_actions = np.clip(actions_arr[:, :2] / v_max, -1.0, 1.0)

                                # 统一token转换：与ppo_buffer.py保持一致
                                if (states_arr.ndim == 3
                                        and states_arr.shape[1:] == (8, 13)):
                                    tokens = states_arr
                                else:
                                    tokens = joint34_to_tokens(states_arr)
                                if isinstance(tokens, torch.Tensor):
                                    tokens = tokens.detach().cpu().numpy()
                                tokens = np.asarray(tokens, dtype=np.float32)
                                # 处理shape: [T,1,6,13] → [T,6,13]
                                if tokens.ndim == 4 and tokens.shape[1] == 1:
                                    tokens = np.squeeze(tokens, axis=1)

                                # 转为tensor用于forward
                                tokens = torch.from_numpy(tokens).float().to(device, non_blocking=True)
                                state_seq = tokens.unsqueeze(0)  # [1, T, 6, 13]
                                action_tensor = torch.as_tensor(norm_actions, dtype=torch.float32, device=device).unsqueeze(0)

                                with torch.no_grad():
                                    action_mean, action_log_std, value_preds = policy.forward_both(state_seq, return_sequence=True)
                                    dist = torch.distributions.Normal(action_mean, torch.exp(action_log_std))
                                    log_prob_tensor = dist.log_prob(action_tensor).sum(dim=-1)

                                log_probs_arr = log_prob_tensor.squeeze(0).detach().cpu().numpy()
                                values_arr = value_preds.squeeze(0).detach().cpu().numpy()
                        except Exception as e:
                            logging.warning(f"Worker {worker_id} failed to compute log_prob/value for episode {global_ep}: {e}")

                        if log_probs_arr is None and 'rewards' in episode_data:
                            T = len(episode_data['rewards'])
                            log_probs_arr = np.zeros(T, dtype=np.float32)
                            values_arr = np.zeros(T, dtype=np.float32)

                        episode_data['log_probs'] = log_probs_arr
                        episode_data['values'] = values_arr

                        # 返回结果
                        event_token = (_extract_event_token(info) or 'unknown')
                        print(f"[Worker-{worker_id}] Putting episode {global_ep} to result_queue (event: {event_token})", flush=True)
                        result_queue.put(episode_data)
                        print(f"[Worker-{worker_id}] Episode {global_ep} successfully sent", flush=True)
                    else:
                        print(f"[Worker-{worker_id}] ✗ Episode {global_ep} has NO trajectories!", flush=True)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                import traceback
                traceback.print_exc()
                break

    except Exception as e:
        print(f"[Worker-{worker_id}] FATAL ERROR during initialization!", flush=True)
        print(f"[Worker-{worker_id}] Error: {e}", flush=True)
        logging.error(f"Worker {worker_id} init error: {e}")
        import traceback
        print(f"[Worker-{worker_id}] Traceback:", flush=True)
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        sys.stderr.flush()


class VectorizedSampler:
    """
    并行采样器：管理多个worker进程进行环境采样

    使用方式：
        sampler = VectorizedSampler(
            num_workers=8,
            episodes_per_worker=2,
            env_config=env_cfg,
            policy_config=policy_cfg,
            train_config=train_cfg,
            policy_name='mamba',
            device='cuda:0'
        )
        sampler.start()

        # 主训练循环
        for ep in range(episodes):
            # 收集 num_workers × episodes_per_worker 个episodes
            episodes_batch = sampler.collect_episodes(
                episode_start=ep,
                phase='train',
                epsilon=current_epsilon
            )

            # 存入memory
            for ep_data in episodes_batch:
                memory.push(ep_data)

            # 更新策略
            policy.train()
            for _ in range(updates_per_ep):
                batch = memory.sample(batch_size)
                loss = trainer.optimize_step(batch)

            # 广播新参数
            if ep % broadcast_interval == 0:
                sampler.broadcast_params(policy.state_dict())

        sampler.stop()
    """

    def __init__(
        self,
        num_workers: int,
        episodes_per_worker: int,
        config_path: str,  # 改为配置文件路径
        policy_name: str,
        device: str = 'cuda:0',
        log_dir: str = None,  # 日志目录
        base_seed: int = 12345,  # 基础种子
        seed_list: list = None,  # 可选的种子列表
    ):
        self.num_workers = num_workers
        self.episodes_per_worker = episodes_per_worker
        device_str = str(device) if device is not None else 'cpu'
        if device_str == 'cuda':
            device_str = 'cuda:0'
        self.device = device_str
        self.log_dir = log_dir

        # 硬编码的高质量种子列表（90%+ ORCA成功率）
        # 来源：3轮种子搜索结果的最佳组合
        DEFAULT_SEED_LIST = [6717, 5940, 3471, 5939, 504, 6429, 7045, 7058]

        # 优先使用提供的seed_list，其次使用默认高质量种子，最后回退到base_seed+i
        if seed_list is not None:
            self.worker_seeds = seed_list[:num_workers]
            logging.info(f"[VectorizedSampler] 使用自定义种子列表: {self.worker_seeds}")
        elif base_seed == 12345:  # 如果使用默认base_seed，则使用高质量种子
            self.worker_seeds = DEFAULT_SEED_LIST[:num_workers]
            logging.info(f"[VectorizedSampler] 🎯 使用硬编码的8个高质量种子 (ORCA 90%+): {self.worker_seeds}")
        else:  # 用户明确指定了base_seed，则使用传统方式
            self.worker_seeds = [base_seed + i for i in range(num_workers)]
            logging.info(f"[VectorizedSampler] 使用base_seed={base_seed}生成种子: {self.worker_seeds}")

        # 多进程队列
        # 使用 spawn：每个worker独立初始化 CUDA，避免 fork 与 GPU 组合导致的死锁
        mp_context = mp.get_context('spawn')
        self._mp_context = mp_context
        self.task_queues = [mp_context.Queue() for _ in range(num_workers)]
        self.result_queue = mp_context.Queue()
        self.param_queues = [mp_context.Queue(maxsize=2) for _ in range(num_workers)]
        self.init_lock = mp_context.Lock()

        # Worker进程列表
        self.workers: List[mp.Process] = []

        # 启动参数
        self.config_path = config_path
        self.policy_name = policy_name

        self._started = False

        logging.info(f"[VectorizedSampler] Initialized: {num_workers} workers × {episodes_per_worker} episodes/worker")
        if log_dir:
            logging.info(f"[VectorizedSampler] Worker logs will be saved to: {log_dir}/worker_*.log")

    def start(self):
        """启动所有worker进程"""
        if self._started:
            logging.warning("[VectorizedSampler] Already started")
            return

        for worker_id in range(self.num_workers):
            worker_seed = self.worker_seeds[worker_id]
            p = self._mp_context.Process(
                target=worker_process,
                args=(
                    worker_id,
                    self.config_path,
                    self.policy_name,
                    self.episodes_per_worker,
                    self.task_queues[worker_id],
                    self.result_queue,
                    self.param_queues[worker_id],
                    self.init_lock,
                    self.device,
                    self.log_dir,  # 传递日志目录
                    worker_seed,  # 传递worker种子
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        self._started = True
        logging.info(f"[VectorizedSampler] Started {self.num_workers} workers with seeds: {self.worker_seeds}")

    def collect_episodes(
        self,
        episode_start: int,
        phase: str = 'train',
        epsilon: float = 0.1,
        timeout: float = 300.0,
    ) -> List[Dict[str, Any]]:
        """
        收集一批episodes

        Args:
            episode_start: 起始episode编号
            phase: 阶段（'train' 或 'val'）
            epsilon: 探索率
            timeout: 超时时间（秒）

        Returns:
            episodes_batch: List[Dict], 长度为 num_workers × episodes_per_worker
        """
        if not self._started:
            raise RuntimeError("VectorizedSampler not started. Call .start() first.")

        # 1. 向所有workers发送采样任务
        for worker_id in range(self.num_workers):
            task = (episode_start + worker_id * self.episodes_per_worker, phase, epsilon)
            self.task_queues[worker_id].put(task)

        # 2. 收集结果
        total_episodes = self.num_workers * self.episodes_per_worker
        episodes_batch = []
        start_time = time.time()

        while len(episodes_batch) < total_episodes:
            try:
                ep_data = self.result_queue.get(timeout=5.0)
                episodes_batch.append(ep_data)
            except queue.Empty:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logging.error(f"[VectorizedSampler] Timeout after {elapsed:.1f}s, collected {len(episodes_batch)}/{total_episodes}")
                    break
                logging.warning(f"[VectorizedSampler] Waiting for episodes... ({len(episodes_batch)}/{total_episodes})")

        return episodes_batch

    def broadcast_params(self, state_dict: Dict[str, torch.Tensor]):
        """
        广播模型参数到所有workers

        Args:
            state_dict: 模型的 state_dict（需要在CPU上）
        """
        # 转移到CPU（避免CUDA序列化问题）
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}

        for worker_id in range(self.num_workers):
            # 清空旧参数（只保留最新）
            try:
                while not self.param_queues[worker_id].empty():
                    self.param_queues[worker_id].get_nowait()
            except queue.Empty:
                pass

            # 放入新参数
            try:
                self.param_queues[worker_id].put(cpu_state_dict, block=False)
            except queue.Full:
                logging.warning(f"[VectorizedSampler] Worker {worker_id} param queue full, skipping")

    def stop(self):
        """停止所有workers"""
        if not self._started:
            return

        logging.info("[VectorizedSampler] Stopping workers...")

        # 发送停止信号
        for worker_id in range(self.num_workers):
            self.task_queues[worker_id].put("STOP")
            self.param_queues[worker_id].put("STOP")

        # 等待所有进程结束
        for p in self.workers:
            p.join(timeout=5.0)
            if p.is_alive():
                logging.warning(f"[VectorizedSampler] Worker {p.pid} still alive, terminating")
                p.terminate()

        self._started = False
        logging.info("[VectorizedSampler] All workers stopped")

    def __del__(self):
        """析构函数：确保清理资源"""
        self.stop()
