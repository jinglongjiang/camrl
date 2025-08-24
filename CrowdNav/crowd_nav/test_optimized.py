# test_optimized.py — 优化版测试脚本：支持快速视频生成和多视频批量生成

import os
import re
import torch
import numpy as np
import logging
import argparse
import configparser
import shutil
import random
from datetime import datetime

# --- no display backend & ffmpeg autodetect ---
import matplotlib
matplotlib.use('Agg')
FFMPEG_BIN = shutil.which('ffmpeg')
if FFMPEG_BIN:
    matplotlib.rcParams['animation.ffmpeg_path'] = FFMPEG_BIN

# 确保CrowdSim环境被注册
import crowd_sim
import gym

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.explorer import Explorer


def load_obs_stats_if_any(policy, stats_path, device):
    """Try loading obs stats from npz (optional; weights already contain stats)."""
    if not stats_path or not os.path.isfile(stats_path):
        logging.info("Obs stats npz not found (this is OK; stats are in weights).")
        return False
    try:
        data = np.load(stats_path)
        m = data.get('obs_mean', None)
        v = data.get('obs_var', None)
        s = data.get('obs_std', None)
        if m is not None:
            policy.obs_mean = torch.as_tensor(m, dtype=torch.float32, device=device)
        if v is not None:
            policy.obs_var = torch.as_tensor(v, dtype=torch.float32, device=device)
        if s is not None:  # optional
            policy.obs_std = torch.as_tensor(s, dtype=torch.float32, device=device)
        logging.info("Loaded extra obs stats: %s", os.path.basename(stats_path))
        return True
    except Exception as e:
        logging.warning("Load obs stats npz failed: %s", e)
        return False


def derive_stats_path_from_weights(model_dir, weights):
    m = re.search(r'_ep(\d+)\.pth$', os.path.basename(weights))
    if not m:
        return None
    ep = m.group(1)
    cand = os.path.join(model_dir, f'obs_stats_ep{ep}.npz')
    return cand if os.path.isfile(cand) else None


def robust_load_weights(policy, model_dir, weights_file, device):
    """Prefer policy.load_model(); fallback to raw state_dict."""
    weights_full = weights_file if os.path.isabs(weights_file) else os.path.join(model_dir, weights_file)
    if not os.path.isfile(weights_full):
        alt = os.path.join(model_dir, 'rl_model.pth')
        if os.path.isfile(alt):
            weights_full = alt
        else:
            raise FileNotFoundError(f"Model weights not found: {weights_full}")

    if hasattr(policy, 'load_model') and callable(policy.load_model):
        try:
            policy.load_model(weights_full)
            logging.info("Weights loaded via policy.load_model: %s", os.path.basename(weights_full))
            return weights_full
        except Exception as e:
            logging.warning("policy.load_model failed (%s). Try raw state_dict.", e)

    state = torch.load(weights_full, map_location=device)
    model_obj = policy.get_model() if hasattr(policy, 'get_model') and callable(policy.get_model) else policy
    try:
        model_obj.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'model' in state:
            model_obj.load_state_dict(state['model'])
        else:
            raise
    logging.info("Weights loaded via state_dict: %s", os.path.basename(weights_full))
    return weights_full


def rollout_one_case(env, policy, test_case_idx=None, time_limit=None):
    """Run one episode so that env.states are filled for video rendering."""
    if test_case_idx is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(options={'test_case': int(test_case_idx)})
    done = False
    steps = 0
    total_reward = 0
    
    while not done:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += reward
        steps += 1
        if time_limit is not None and steps * env.time_step >= time_limit:
            break
    
    # 添加更多统计信息
    info['total_reward'] = total_reward
    info['steps'] = steps
    info['time'] = steps * env.time_step
    
    return info


def setup_policy_and_env(policy_name, model_dir, weights_path, env_cfg_path, policy_cfg_path, device, human_num=5):
    """Setup policy and environment - 复用代码避免重复初始化"""
    # 1) read configs
    env_cfg = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    env_cfg.read(env_cfg_path)
    pol_cfg = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    pol_cfg.read(policy_cfg_path)

    # 2) clamp test human_num to trained K
    trained_k = None
    if pol_cfg.has_section('mamba') and pol_cfg.has_option('mamba', 'human_num'):
        trained_k = pol_cfg.getint('mamba', 'human_num')
    trained_k = trained_k or 5
    hn_use = min(human_num, trained_k)

    if not env_cfg.has_section('sim'):
        env_cfg.add_section('sim')
    env_cfg.set('sim', 'test_sim', 'circle_crossing')
    env_cfg.set('sim', 'human_num', str(hn_use))
    env_cfg.set('sim', 'circle_radius', str(4.0))

    # 3) build policy
    if policy_name not in policy_factory:
        raise ValueError(f"Unknown policy: {policy_name}")
    policy_ctor = policy_factory[policy_name]
    policy = policy_ctor(pol_cfg) if callable(policy_ctor) else policy_ctor
    if hasattr(policy, 'configure'):
        policy.configure(pol_cfg)
    if hasattr(policy, 'set_device'):
        policy.set_device(device)

    # 4) load weights
    if getattr(policy, 'trainable', True):
        weights_full = robust_load_weights(policy, model_dir, weights_path, device)
        stats_path = derive_stats_path_from_weights(model_dir, weights_full) or os.path.join(model_dir, 'obs_stats.npz')
        load_obs_stats_if_any(policy, stats_path, device)

    # 5) env
    env = gym.make('CrowdSim-v0')
    env.configure(env_cfg)
    robot = Robot(env_cfg, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    # 6) test-mode policy (deterministic)
    if hasattr(policy, 'set_env'):
        policy.set_env(env)
    if hasattr(policy, 'set_phase'):
        policy.set_phase('test')
    if hasattr(policy, 'set_epsilon'):
        policy.set_epsilon(0.0)
    if hasattr(policy, 'env_stochastic'):
        policy.env_stochastic = False

    return policy, env, hn_use


def generate_multiple_videos(policy, env, num_videos=3, output_dir="data/output", base_name="demo", 
                           random_cases=True, specific_cases=None, max_case=300):
    """生成多个随机视频"""
    video_paths = []
    video_infos = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确定要生成的case列表
    if specific_cases:
        cases = specific_cases[:num_videos]  # 限制数量
    elif random_cases:
        cases = random.sample(range(max_case), min(num_videos, max_case))
    else:
        cases = list(range(num_videos))
    
    logging.info(f"Generating {len(cases)} videos for cases: {cases}")
    
    for i, case_idx in enumerate(cases):
        try:
            # 运行单个case
            info = rollout_one_case(env, policy, test_case_idx=case_idx, time_limit=getattr(env, 'time_limit', None))
            
            # 生成视频文件名
            result_tag = info.get('result', 'unknown')
            video_name = f"{base_name}_{timestamp}_case{case_idx:03d}_{result_tag}.mp4"
            video_path = os.path.join(output_dir, video_name)
            
            # 渲染视频
            try:
                env.render(mode='video', output_file=video_path)
                video_paths.append(video_path)
                
                # 记录详细信息
                info_detail = {
                    'case': case_idx,
                    'result': result_tag,
                    'reward': info.get('total_reward', 0),
                    'time': info.get('time', 0),
                    'steps': info.get('steps', 0),
                    'video_path': video_path
                }
                video_infos.append(info_detail)
                
                logging.info(f"Video {i+1}/{len(cases)}: Case {case_idx} ({result_tag}) -> {video_name}")
                logging.info(f"  Reward: {info.get('total_reward', 0):.2f}, Time: {info.get('time', 0):.1f}s")
                
            except Exception as e:
                logging.error(f"Failed to save video for case {case_idx}: {e}")
                # 尝试GIF fallback
                try:
                    gif_path = os.path.splitext(video_path)[0] + ".gif"
                    env.render(mode='video', output_file=gif_path)
                    video_paths.append(gif_path)
                    logging.info(f"Saved GIF fallback: {gif_path}")
                except:
                    logging.error(f"Both MP4 and GIF failed for case {case_idx}")
                    
        except Exception as e:
            logging.error(f"Error processing case {case_idx}: {e}")
    
    return video_paths, video_infos


def quick_evaluation(policy, env, device, num_samples=50):
    """快速评估：只运行少量episodes获得性能指标"""
    logging.info(f"Running quick evaluation with {num_samples} episodes...")
    
    explorer = Explorer(env, env.robot, device=device, memory=None, gamma=0.99, target_policy=policy)
    stats = explorer.run_k_episodes(num_samples, 'test', update_memory=False, return_stats=True, show_tqdm=True)
    
    logging.info(
        "QUICK EVAL | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f (%d episodes)",
        stats["success_rate"], stats["collision_rate"], stats["timeout_rate"],
        stats["nav_time"], stats["total_reward"], num_samples
    )
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Optimized test script with multi-video support')
    parser.add_argument('--policy', type=str, default='mamba', help='mamba/orca/...')
    parser.add_argument('--model_dir', type=str, default='data/output')
    parser.add_argument('--env_config', type=str, default='data/output/env.config')
    parser.add_argument('--policy_config', type=str, default='data/output/policy.config')
    parser.add_argument('--weights', type=str, default='rl_model_ep1000.pth')
    parser.add_argument('--device', type=str, default='auto', help='"auto" | "cpu" | "cuda:0" ...')
    
    # 模式选择
    parser.add_argument('--mode', type=str, choices=['full_eval', 'video_only', 'quick_eval'], 
                       default='video_only', help='full_eval: complete evaluation; video_only: only generate videos; quick_eval: fast evaluation + videos')
    
    # 视频生成选项
    parser.add_argument('--num_videos', type=int, default=3, help='Number of videos to generate')
    parser.add_argument('--output_dir', type=str, default='data/output', help='Output directory for videos')
    parser.add_argument('--video_base_name', type=str, default='demo', help='Base name for video files')
    parser.add_argument('--random_cases', action='store_true', default=True, help='Use random test cases')
    parser.add_argument('--specific_cases', type=int, nargs='+', help='Specific test case numbers (e.g., --specific_cases 66 123 200)')
    parser.add_argument('--max_case_range', type=int, default=1000, help='Maximum case number for random selection')
    
    # 评估选项
    parser.add_argument('--episodes', type=int, default=300, help='Number of episodes for full evaluation')
    parser.add_argument('--quick_eval_samples', type=int, default=50, help='Number of samples for quick evaluation')
    
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s')
    logging.info(f"Mode: {args.mode}, Device: {device}")

    # 设置策略和环境
    policy, env, human_num = setup_policy_and_env(
        args.policy, args.model_dir, args.weights, 
        args.env_config, args.policy_config, device
    )

    logging.info(f'Human number: {human_num}')
    logging.info(f'Environment: {getattr(env, "test_sim", "circle_crossing")}, Time limit: {getattr(env, "time_limit", "?")}')

    # 根据模式执行不同操作
    if args.mode == 'video_only':
        # 仅生成视频模式
        logging.info("=== VIDEO GENERATION MODE ===")
        video_paths, video_infos = generate_multiple_videos(
            policy, env, 
            num_videos=args.num_videos,
            output_dir=args.output_dir,
            base_name=args.video_base_name,
            random_cases=args.random_cases,
            specific_cases=args.specific_cases,
            max_case=args.max_case_range
        )
        
        logging.info(f"Generated {len(video_paths)} videos:")
        for info in video_infos:
            logging.info(f"  Case {info['case']:3d}: {info['result']:>9s} | Reward: {info['reward']:6.2f} | Time: {info['time']:5.1f}s | {os.path.basename(info['video_path'])}")
    
    elif args.mode == 'quick_eval':
        # 快速评估+视频模式
        logging.info("=== QUICK EVALUATION + VIDEO MODE ===")
        stats = quick_evaluation(policy, env, device, args.quick_eval_samples)
        
        video_paths, video_infos = generate_multiple_videos(
            policy, env, 
            num_videos=args.num_videos,
            output_dir=args.output_dir,
            base_name=args.video_base_name,
            random_cases=args.random_cases,
            specific_cases=args.specific_cases,
            max_case=args.max_case_range
        )
        
        logging.info(f"Generated {len(video_paths)} videos after quick evaluation")
    
    elif args.mode == 'full_eval':
        # 完整评估模式
        logging.info("=== FULL EVALUATION MODE ===")
        explorer = Explorer(env, env.robot, device=device, memory=None, gamma=0.99, target_policy=policy)
        stats = explorer.run_k_episodes(args.episodes, 'test', update_memory=False, return_stats=True, show_tqdm=True)
        
        logging.info(
            "FULL EVAL | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f (%d episodes)",
            stats["success_rate"], stats["collision_rate"], stats["timeout_rate"],
            stats["nav_time"], stats["total_reward"], args.episodes
        )
        
        # 可选择性生成视频
        if args.num_videos > 0:
            video_paths, video_infos = generate_multiple_videos(
                policy, env, 
                num_videos=args.num_videos,
                output_dir=args.output_dir,
                base_name=args.video_base_name,
                random_cases=args.random_cases,
                specific_cases=args.specific_cases,
                max_case=args.episodes
            )
            logging.info(f"Additionally generated {len(video_paths)} videos")

    logging.info("=== TEST COMPLETED ===")


if __name__ == '__main__':
    main()