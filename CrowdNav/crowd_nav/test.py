#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CrowdNav test & render (raw-action, no transform)

关键点：
- 不对策略动作做任何坐标变换（不 flip_y、不 local→world 旋转）
- 渲染与评测都使用“策略原样动作”，保证口径一致
- 若仅想让画面视觉朝上，可用 --invert_y 只调整绘图坐标（不影响控制）
- 支持导出 CSV/JSON/PNG、随机抽 case、图像视频导出、首步调试
"""

import os, re, math, argparse, logging, random, csv, json
from typing import Optional, Tuple, List

import numpy as np
import torch
import gym

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.explorer import Explorer
try:
    from crowd_sim.envs.utils.action import ActionXY
except Exception:
    ActionXY = None


# --------------------- utils: model I/O ---------------------
def _find_latest(model_dir: str) -> Optional[str]:
    if not os.path.isdir(model_dir):
        return None
    cand = []
    for n in os.listdir(model_dir):
        m = re.match(r"rl_model_ep(\d+)\.pth$", n)
        if m:
            cand.append((int(m.group(1)), os.path.join(model_dir, n)))
    cand.sort()
    if cand:
        return cand[-1][1]
    alt = os.path.join(model_dir, "rl_model.pth")
    return alt if os.path.isfile(alt) else None


def load_weights(policy, model_dir: str, weights: str, device: torch.device) -> str:
    path = _find_latest(model_dir) if weights == "latest" else (
        weights if os.path.isabs(weights) else os.path.join(model_dir, weights)
    )
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path or "(no path)")
    if hasattr(policy, "load_model"):
        policy.load_model(path); way = "policy.load_model"
    else:
        sd = torch.load(path, map_location=device)
        tgt = policy.get_model() if hasattr(policy, "get_model") else policy
        tgt.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
        way = "state_dict"
    logging.info("Loaded weights via %s: %s", way, os.path.basename(path))
    return path


# --------------------- utils: drawing ---------------------
def _extract_bounds(env) -> Tuple[float, float, float, float]:
    w = getattr(env, "square_width", None)
    r = getattr(env, "circle_radius", None)
    if w is not None:
        half = float(w) / 2.0; return (-half, half, -half, half)
    if r is not None:
        half = float(r) * 1.3; return (-half, half, -half, half)
    xs = [getattr(env.robot, "px", 0.0), getattr(env.robot, "gx", 0.0)]
    ys = [getattr(env.robot, "py", 0.0), getattr(env.robot, "gy", 0.0)]
    for h in getattr(env, "humans", []) or []:
        xs.append(getattr(h, "px", 0.0)); ys.append(getattr(h, "py", 0.0))
    return (min(xs)-1, max(xs)+1, min(ys)-1, max(ys)+1)


def _draw_goal(ax, gx: float, gy: float):
    ax.scatter([gx], [gy], marker="*", s=180, linewidths=1.5, edgecolors="k", zorder=5)


def _draw_robot(ax, x: float, y: float, r: float, vx: float, vy: float):
    c = plt.Circle((x, y), r, fill=True, alpha=0.85, zorder=6)
    ax.add_patch(c)
    if abs(vx) > 1e-8 or abs(vy) > 1e-8:
        hx = x + r*1.4*(vx / max(1e-8, math.hypot(vx, vy)))
        hy = y + r*1.4*(vy / max(1e-8, math.hypot(vx, vy)))
        ax.plot([x, hx], [y, hy], linewidth=2.0, zorder=7)


def _draw_humans(ax, humans):
    for h in humans:
        c = plt.Circle((h.px, h.py), h.radius, fill=True, alpha=0.75, zorder=4)
        c.set_edgecolor("k"); c.set_linewidth(0.8); ax.add_patch(c)


def _append_frame(fig, writer):
    fig.canvas.draw()
    try:
        frame = np.asarray(fig.canvas.buffer_rgba())
    except Exception:
        # 兼容旧版
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        argb = argb.reshape(h, w, 4)
        frame = argb[:, :, [1, 2, 3, 0]]
    writer.append_data(frame)


def _min_dist(robot, humans) -> float:
    rx, ry, rr = robot.px, robot.py, robot.radius
    dmin = float("inf")
    for h in humans or []:
        d = math.hypot(h.px - rx, h.py - ry) - h.radius - rr
        dmin = min(dmin, d)
    return dmin


# --------------------- rollout & render ---------------------
def rollout_and_render_one(env, policy, case_idx: Optional[int], out_path: Optional[str],
                           fps: int = 30, dpi: int = 128, dump_dir: Optional[str] = None,
                           export_png: bool = False, invert_y: bool = False) -> dict:
    # reset（尽量用 options 指定 case）
    try:
        reset_ret = env.reset(options={"test_case": int(case_idx)}) if case_idx is not None else env.reset()
    except TypeError:
        reset_ret = env.reset()
    obs = reset_ret[0] if isinstance(reset_ret, tuple) and len(reset_ret) == 2 else reset_ret

    # 视频 writer
    writer = None
    if out_path and HAVE_IMAGEIO:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)

    # 画布
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    xmin, xmax, ymin, ymax = _extract_bounds(env)
    ax.set_aspect("equal", adjustable="box"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    if invert_y: ax.invert_yaxis()
    ax.grid(alpha=0.15)

    goal = (env.robot.gx, env.robot.gy)
    dt = float(getattr(env, "time_step", 0.25))
    t = 0.0; done = False; info = {}

    case_tag = f"case{case_idx:03d}" if case_idx is not None else "caseNA"
    robot_rows, human_rows = [], []

    # 使用原始策略动作（与Explorer评测保持一致）
    action = env.robot.act(obs)
    
    # 提取速度用于显示
    try:
        vx, vy = float(action.vx), float(action.vy)
    except Exception:
        vx, vy = float(action[0]), float(action[1])

    while not done:
        # 绘制
        ax.clear()
        ax.set_aspect("equal", adjustable="box"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        if invert_y: ax.invert_yaxis()
        ax.grid(alpha=0.15)
        _draw_goal(ax, goal[0], goal[1])
        _draw_humans(ax, env.humans or [])
        _draw_robot(ax, env.robot.px, env.robot.py, env.robot.radius, vx, vy)

        d_goal = math.hypot(goal[0]-env.robot.px, goal[1]-env.robot.py)
        d_min = _min_dist(env.robot, env.humans or [])
        ax.set_title("CrowdNav Evaluation (goal • avoid • reach)")
        ax.text(0.02, 0.98, f"t={t:.2f}s   dist_to_goal={d_goal:.2f}   min_dist={d_min:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.3", alpha=0.9))
        if writer: _append_frame(fig, writer)

        # 记录
        robot_rows.append([t, float(env.robot.px), float(env.robot.py),
                           float(vx), float(vy),
                           float(goal[0]), float(goal[1]),
                           float(d_goal), float(d_min)])
        if env.humans:
            for hid, h in enumerate(env.humans):
                human_rows.append([t, hid, float(h.px), float(h.py), float(h.radius)])

        # step （使用原始动作，与Explorer保持一致）
        step_ret = env.step(action)
        if len(step_ret) == 5:
            obs, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        elif len(step_ret) == 4:
            obs, reward, done, info = step_ret
        else:
            raise RuntimeError("Unexpected env.step return signature")
        t += dt
        if done: break

        # 下一步动作（使用原始策略输出）
        action = env.robot.act(obs)
        try:
            vx, vy = float(action.vx), float(action.vy)
        except Exception:
            vx, vy = float(action[0]), float(action[1])

    # 结束页
    ev = str((info or {}).get("event","")).lower()
    label = "SUCCESS" if ev in ("success","reach_goal") else ("COLLISION" if ev=="collision" else "TIMEOUT")
    ax.text(0.98, 0.02, f"[{label}]",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.3", alpha=0.95))
    if writer:
        _append_frame(fig, writer); writer.close()

    # 导出
    meta = {
        "case": case_idx, "event": ev, "label": label, "duration_sec": t,
        "time_step": float(getattr(env, "time_step", 0.25)),
        "human_num": len(env.humans) if getattr(env, "humans", None) else 0,
        "square_width": getattr(env, "square_width", None),
        "circle_radius": getattr(env, "circle_radius", None),
        "transform": "raw_action"  # 明确标记：使用原始动作，与Explorer一致
    }
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, f"{case_tag}_traj.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t","rx","ry","vx","vy","gx","gy","dist_goal","min_sep"]); w.writerows(robot_rows)
        with open(os.path.join(dump_dir, f"{case_tag}_humans.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t","human_id","px","py","radius"]); w.writerows(human_rows)
        with open(os.path.join(dump_dir, f"{case_tag}_meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        fig.savefig(os.path.join(dump_dir, f"{case_tag}_plot.png"), dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    return info


# --------------------- eval helpers ---------------------
def _test_case_success(env, policy, case_idx: int) -> bool:
    """测试单个case是否成功，快速版本（不渲染）"""
    try:
        reset_ret = env.reset(options={"test_case": int(case_idx)}) if case_idx is not None else env.reset()
    except TypeError:
        reset_ret = env.reset()
    obs = reset_ret[0] if isinstance(reset_ret, tuple) and len(reset_ret) == 2 else reset_ret
    
    done = False
    max_steps = int(60.0 / float(getattr(env, "time_step", 0.25)))  # 60秒超时
    steps = 0
    
    while not done and steps < max_steps:
        action = env.robot.act(obs)
        step_ret = env.step(action)
        if len(step_ret) == 5:
            obs, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        elif len(step_ret) == 4:
            obs, reward, done, info = step_ret
        else:
            raise RuntimeError("Unexpected env.step return signature")
        steps += 1
    
    event = str((info or {}).get("event", "")).lower()
    return event in ("reachgoal", "reach_goal", "success")


def _get_total_cases(env, split: str, default_eps: int) -> int:
    cs = getattr(env, "case_size", None)
    if isinstance(cs, dict):
        return int(cs.get(split, default_eps))
    if isinstance(cs, int):
        return cs
    return default_eps


# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    # policy/env
    ap.add_argument("--model_dir", type=str, default="data/output")
    ap.add_argument("--env_config", type=str, default="data/output/env.config")
    ap.add_argument("--policy_config", type=str, default="data/output/policy.config")
    ap.add_argument("--policy", type=str, default="mamba")
    ap.add_argument("--weights", type=str, default="latest")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--stochastic", action="store_true")
    # eval
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--split", type=str, default="test")
    # render & export
    ap.add_argument("--save_video", type=str, default="")
    ap.add_argument("--video_case", type=int, default=-1)
    ap.add_argument("--cases", type=str, default="")
    ap.add_argument("--random_n", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=128)
    ap.add_argument("--dump_dir", type=str, default="")
    ap.add_argument("--export_png", action="store_true")
    ap.add_argument("--invert_y", action="store_true", help="仅影响绘图坐标，不改变控制动作")
    ap.add_argument("--find_success", action="store_true", help="寻找第一个成功案例来渲染")
    # debug
    ap.add_argument("--dbg_first_steps", type=int, default=0)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(levelname)s: %(message)s")

    # device
    device = torch.device("cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    logging.info("Using device: %s", device)

    # configs
    import configparser
    env_cfg = configparser.RawConfigParser(inline_comment_prefixes=("#", ";"))
    pol_cfg = configparser.RawConfigParser(inline_comment_prefixes=("#", ";"))
    if os.path.isfile(args.env_config): env_cfg.read(args.env_config)
    if os.path.isfile(args.policy_config): pol_cfg.read(args.policy_config)

    # policy
    ctor = policy_factory[args.policy]
    policy = ctor(pol_cfg)
    if hasattr(policy, "configure"):  policy.configure(pol_cfg)
    if hasattr(policy, "set_device"): policy.set_device(device)
    load_weights(policy, args.model_dir, args.weights, device)

    # env
    env = gym.make("CrowdSim-v0")
    if hasattr(env, "configure"): env.configure(env_cfg)
    robot = Robot(env_cfg, "robot"); robot.set_policy(policy); env.set_robot(robot)

    # runtime flags
    if hasattr(policy, "set_env"): policy.set_env(env)
    if hasattr(policy, "set_phase"): policy.set_phase("train" if args.stochastic else "test")
    if hasattr(policy, "env_stochastic"): policy.env_stochastic = bool(args.stochastic)
    if hasattr(policy, "set_epsilon"): policy.set_epsilon(float(args.epsilon))
    if hasattr(policy, "robot_v_pref"):
        policy.robot_v_pref = robot.v_pref
        logging.info("Fixed robot_v_pref: %.3f", policy.robot_v_pref)

    # 单案例首步调试（修正动作）
    if args.dbg_first_steps > 0:
        logging.info("=== SINGLE CASE DEBUG MODE ===")
        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) and len(reset_ret) == 2 else reset_ret
        
        for step in range(args.dbg_first_steps):
            # 处理观察格式问题
            if isinstance(obs, tuple) and len(obs) > 0:
                obs_for_policy = obs[0]  # 使用第一个元素
            else:
                obs_for_policy = obs
            
            action = env.robot.act(obs_for_policy)
            try:
                vx, vy = float(action.vx), float(action.vy)
            except Exception:
                vx, vy = float(action[0]), float(action[1])
            
            out = env.step(action)
            logging.info("Step %d: raw_action=(%.3f,%.3f)", step, vx, vy)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = out
            logging.info("Step %d: action=(%.3f,%.3f) reward=%.3f", step, vx, vy, float(reward))
            if done:
                logging.info("Episode done: %s", (info or {}).get("event","unknown"))
                break
        logging.info("=== END DEBUG MODE ===")
        return

    # 官方评测（与渲染同口径：使用原始动作）
    logging.info("Running Explorer evaluation (official) ...")
    explorer = Explorer(env, robot, device=device, memory=None, gamma=0.99, target_policy=policy)
    stats = explorer.run_k_episodes(args.episodes, args.split, update_memory=False, return_stats=True, show_tqdm=True)
    logging.info("TEST  has success rate: %.2f, collision rate: %.2f, timeout rate: %.2f, nav time: %.2f, total reward: %.4f",
                 stats.get("success_rate", 0.0),
                 stats.get("collision_rate", 0.0),
                 stats.get("timeout_rate", 0.0),
                 stats.get("nav_time", 0.0),
                 stats.get("total_reward", 0.0))

    # 渲染
    if args.save_video and HAVE_IMAGEIO:
        total_cases = _get_total_cases(env, args.split, args.episodes)
        if args.cases.strip():
            case_list = [int(tok.strip()) for tok in args.cases.split(",") if tok.strip()]
        elif args.random_n > 0:
            ids = list(range(total_cases)); random.seed()
            case_list = sorted(random.sample(ids, k=min(args.random_n, total_cases)))
        elif args.video_case >= 0:
            case_list = [int(args.video_case)]
        elif args.find_success:
            # 寻找第一个成功案例
            logging.info("正在寻找成功案例...")
            case_list = []
            for cid in range(min(100, total_cases)):  # 最多测试100个案例
                if _test_case_success(env, policy, cid):
                    logging.info("找到成功案例: case %d", cid)
                    case_list = [cid]
                    break
                if (cid + 1) % 10 == 0:
                    logging.info("已测试 %d 个案例...", cid + 1)
            if not case_list:
                logging.warning("前100个案例中未找到成功案例，使用随机案例")
                ids = list(range(total_cases)); random.seed()
                case_list = [random.choice(ids)]
        else:
            ids = list(range(total_cases)); random.seed()
            case_list = [random.choice(ids)]

        for cid in case_list:
            out_path = args.save_video
            if "%" in out_path:
                try: out_path = out_path % (cid,)
                except TypeError: out_path = out_path.replace("%03d", f"{cid:03d}")
            logging.info("Rendering video for case=%d -> %s", cid, out_path)
            info = rollout_and_render_one(env, policy, cid, out_path,
                                          fps=args.fps, dpi=args.dpi,
                                          dump_dir=(args.dump_dir or None),
                                          export_png=bool(args.export_png),
                                          invert_y=bool(args.invert_y))
            logging.info("CASE %d done. event=%s", cid, str((info or {}).get("event","")))
    elif args.save_video and not HAVE_IMAGEIO:
        logging.warning("imageio not found; pip install -U imageio imageio-ffmpeg for MP4 export")

    logging.info("Done.")


if __name__ == "__main__":
    main()
