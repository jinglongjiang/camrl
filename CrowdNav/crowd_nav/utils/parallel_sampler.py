# crowd_nav/utils/parallel_sampler.py
# -*- coding: utf-8 -*-
"""
并发采样器（修正版）：
- worker 禁止打印以 [DEBUG] 开头的行，避免多进程重复刷屏
- 支持两种权重接口：state_dict()/load_state_dict() 或 get_state_dict()/set_state_dict()
- worker 设备可设为 cuda:0（mamba_ssm 需要在 CUDA 上前向）
- collect() 内部 try/except，把错误通过管道返回，避免 EOFError
- ★ 关键修复：广播前强制把权重转成 float32；worker 端加载后再 cast 到 float32，避免 Float vs Half 报错
"""

import os
import configparser
import multiprocessing as mp
import torch


def _init_fast_math():
    import torch as _torch
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        _torch.backends.cuda.matmul.allow_tf32 = True
        _torch.backends.cudnn.allow_tf32 = True
        _torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    _torch.set_num_threads(1)


def _silence_debug_prints_in_worker():
    """把 worker 里的 `[DEBUG] ...` 输出静音（主进程不受影响）。"""
    import builtins
    _orig_print = builtins.print

    def _filtered_print(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("[DEBUG]"):
            return
        return _orig_print(*args, **kwargs)

    builtins.print = _filtered_print


def _to_cpu_state_dict(sd):
    """将任意权重字典中的 tensor 迁到 CPU，并强制转为 float32；非张量原样返回。"""
    out = {}
    for k, v in (sd.items() if hasattr(sd, "items") else []):
        if torch.is_tensor(v):
            t = v.detach().to('cpu')
            if t.is_floating_point() and t.dtype != torch.float32:
                t = t.float()  # ★ 强制 float32，避免 Half/Fp16 混入
            out[k] = t
        else:
            out[k] = v
    return out


def _cast_policy_fp32(policy, device):
    """将策略的关键子模块 cast 成 float32（worker 端加载权重后调用）。"""
    for name in ('encoder', 'actor', 'q1', 'q2'):
        if hasattr(policy, name):
            getattr(policy, name).to(device=device, dtype=torch.float32)


def _worker_entry(conn, seed, env_config_path, policy_name_student, policy_name_expert,
                  policy_config_path, device_str="cpu"):
    """
    Worker 子进程：
      - 本地创建 env/robot/explorer
      - set_policy / collect / close
      - 用 ProxyReplayMemory 收集条目打包回传
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    _init_fast_math()
    _silence_debug_prints_in_worker()   # 关键：静音重复 [DEBUG] 打印

    import numpy as np
    import torch, gym, random
    from crowd_sim.envs.utils.robot import Robot
    from crowd_nav.policy.policy_factory import policy_factory
    from crowd_nav.utils.explorer import Explorer

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)       # 关键：worker 只做推理
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device(device_str)   # 可设 cuda:0

    # ===== 环境 =====
    env_cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
    env_cfg.read(env_config_path)
    env = gym.make('CrowdSim-v0')
    env.configure(env_cfg)
    robot = Robot(env_cfg, 'robot')
    env.set_robot(robot)

    # ===== 策略 =====
    pol_cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
    pol_cfg.read(policy_config_path)

    student = policy_factory[policy_name_student](pol_cfg); student.configure(pol_cfg); student.set_device(device)
    expert  = policy_factory[policy_name_expert](pol_cfg);  expert.configure(pol_cfg);  expert.set_device(device)

    # ★ 强制 FP32，以防框架/权重默认为半精度
    _cast_policy_fp32(student, device)
    _cast_policy_fp32(expert, device)

    robot.set_policy(student)

    # 轻量代理回放池
    class ProxyReplayMemory:
        def __init__(self): self.data = []
        def push(self, *args, **kwargs):
            self.data.append((args, kwargs))
        def __len__(self): return len(self.data)
        def pop_all(self):
            d = self.data; self.data = []; return d

    mem = ProxyReplayMemory()
    explorer = Explorer(env, robot, device, mem, getattr(student, 'gamma', 0.99), target_policy=student)

    while True:
        msg = conn.recv()
        mtype = msg.get("type", "")

        if mtype == "close":
            conn.close()
            break

        elif mtype == "set_policy":
            sd = msg.get("state_dict", {})
            loaded = False
            # a) 标准
            if hasattr(student, "load_state_dict"):
                try:
                    student.load_state_dict(sd); loaded = True
                except Exception:
                    loaded = False
            # b) 自定义
            if (not loaded) and hasattr(student, "set_state_dict"):
                try:
                    student.set_state_dict(sd); loaded = True
                except Exception:
                    loaded = False
            # c) 其他自定义
            if (not loaded) and hasattr(student, "load_model_from_state_dict"):
                try:
                    student.load_model_from_state_dict(sd); loaded = True
                except Exception:
                    loaded = False
            # d) 兜底：逐子模块
            if (not loaded) and hasattr(sd, "items"):
                try:
                    for k, v in sd.items():
                        if hasattr(student, k):
                            sub = getattr(student, k)
                            if hasattr(sub, "load_state_dict") and isinstance(v, dict):
                                sub.load_state_dict(v)
                    loaded = True
                except Exception:
                    pass

            # ★ 加载完成后再转 FP32，避免半精度权重混入
            _cast_policy_fp32(student, device)

        elif mtype == "collect":
            k = int(msg.get("episodes", 1))
            use_expert = bool(msg.get("use_expert", False))
            robot.set_policy(expert if use_expert else student)

            try:
                stats = explorer.run_k_episodes(
                    k, 'train',
                    update_memory=True,
                    episode=msg.get("episode", 0),
                    return_stats=True,
                    show_tqdm=False
                )
                traj = mem.pop_all()
                conn.send({"stats": stats, "traj": traj})
            except Exception as e:
                # 关键：把错误发回主进程，而不是让进程直接挂掉
                conn.send({"error": repr(e), "stats": None, "traj": []})

        else:
            conn.send({"error": f"unknown command: {mtype}"})


class ParallelSampler:
    """
    多进程并发采样：
      - broadcast_policy(policy): 广播学生策略权重
      - collect(...): 并发收集轨迹 + 聚合统计
      - close(): 清理
    """
    def __init__(self, num_workers, env_config_path, policy_config_path,
                 policy_name_student="mamba", policy_name_expert="orca",
                 device_str="cpu", base_seed=12345):
        ctx = mp.get_context("spawn")
        self.ctx = ctx
        self.procs, self.conns = [], []
        for i in range(num_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_worker_entry,
                args=(child, base_seed + i, env_config_path,
                      policy_name_student, policy_name_expert,
                      policy_config_path, device_str)
            )
            p.daemon = True
            p.start()
            child.close()
            self.procs.append(p)
            self.conns.append(parent)

    def broadcast_policy(self, policy):
        """导出权重并广播到所有 worker（兼容多种导出接口；统一转 float32）"""
        sd = None
        if hasattr(policy, "state_dict"):
            try:
                sd = _to_cpu_state_dict(policy.state_dict())
            except Exception:
                sd = None
        if sd is None and hasattr(policy, "get_state_dict"):
            try:
                sd = _to_cpu_state_dict(policy.get_state_dict())
            except Exception:
                sd = None
        if sd is None:
            raise TypeError("Cannot export policy weights: neither state_dict() nor get_state_dict() is available.")
        for c in self.conns:
            c.send({"type": "set_policy", "state_dict": sd})

    def collect(self, episodes_per_worker=1, episode_idx=0, use_expert=False, timeout=None):
        for c in self.conns:
            c.send({
                "type": "collect",
                "episodes": int(episodes_per_worker),
                "episode": int(episode_idx),
                "use_expert": bool(use_expert)
            })

        results = []
        for c in self.conns:
            if timeout is None:
                res = c.recv()
            else:
                res = c.recv() if c.poll(timeout) else {"stats": None, "traj": []}
            results.append(res)

        # 如果有任何 worker 报错，直接抛出，提示哪个进程出错
        err_msgs = [r.get("error") for r in results if r.get("error")]
        if err_msgs:
            raise RuntimeError(f"ParallelSampler worker error(s): {err_msgs}")

        # 聚合统计
        keys = ["success_rate", "collision_rate", "timeout_rate", "total_reward", "nav_time"]
        agg = {k: 0.0 for k in keys}; cnt = 0
        for r in results:
            st = r.get("stats") or {}
            if st:
                for k in keys:
                    if k in st: agg[k] += float(st[k])
                cnt += 1
        if cnt > 0:
            for k in keys: agg[k] /= cnt

        # 合并轨迹
        traj = []
        for r in results:
            traj.extend(r.get("traj", []))

        return agg, traj

    def close(self):
        for c in self.conns:
            try: c.send({"type": "close"})
            except Exception: pass
        for p in self.procs:
            try: p.join(timeout=1.0)
            except Exception: pass
