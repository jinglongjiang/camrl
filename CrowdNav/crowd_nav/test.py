# test.py — eval with train-matched configs, deterministic policy, safe weight/stat loading

import os
import re
import gym
import torch
import numpy as np
import logging
import argparse
import configparser
import shutil

# --- no display backend & ffmpeg autodetect ---
import matplotlib
matplotlib.use('Agg')
FFMPEG_BIN = shutil.which('ffmpeg')
if FFMPEG_BIN:
    matplotlib.rcParams['animation.ffmpeg_path'] = FFMPEG_BIN

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
            policy.load_model(weights_full)  # our MambaRL also restores obs_mean/var/count
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
    while not done:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        if time_limit is not None and steps * env.time_step >= time_limit:
            break
    return info


def eval_one_scenario(
    policy_name, model_dir, weights_path, env_cfg_path, policy_cfg_path,
    device, episodes, sim_name, human_num_case, circle_radius=None, square_width=None,
    save_video=None, video_case=None
):
    # 1) read configs (use the same copies saved by training)
    env_cfg = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    env_cfg.read(env_cfg_path)
    pol_cfg = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    pol_cfg.read(policy_cfg_path)

    # 2) clamp test human_num to trained K
    trained_k = None
    if pol_cfg.has_section('mamba') and pol_cfg.has_option('mamba', 'human_num'):
        trained_k = pol_cfg.getint('mamba', 'human_num')
    trained_k = trained_k or 5
    hn_case = int(human_num_case)
    if hn_case != trained_k:
        logging.warning(
            "Scenario requests %d humans but model was trained with %d. "
            "Clamping to %d for evaluation.", hn_case, trained_k, trained_k
        )
    hn_use = min(hn_case, trained_k)

    if not env_cfg.has_section('sim'):
        env_cfg.add_section('sim')
    env_cfg.set('sim', 'test_sim', sim_name)
    env_cfg.set('sim', 'human_num', str(hn_use))
    if circle_radius is not None:
        env_cfg.set('sim', 'circle_radius', str(circle_radius))
    if square_width is not None:
        env_cfg.set('sim', 'square_width', str(square_width))

    # 3) build policy
    if policy_name not in policy_factory:
        raise ValueError(f"Unknown policy: {policy_name}")
    policy_ctor = policy_factory[policy_name]
    policy = policy_ctor(pol_cfg) if callable(policy_ctor) else policy_ctor
    if hasattr(policy, 'configure'):
        policy.configure(pol_cfg)
    if hasattr(policy, 'set_device'):
        policy.set_device(device)

    # 4) load weights (+ stats)
    weights_full = None
    if getattr(policy, 'trainable', True):
        weights_full = robust_load_weights(policy, model_dir, weights_path, device)
        stats_path = derive_stats_path_from_weights(model_dir, weights_full) or os.path.join(model_dir, 'obs_stats.npz')
        loaded_npz = load_obs_stats_if_any(policy, stats_path, device)
        # quick check: stats should be on device
        try:
            logging.info("obs_count=%s mean_norm=%.3f (npz_loaded=%s)",
                         getattr(policy, 'obs_count', None),
                         float(getattr(policy, 'obs_mean', torch.zeros(1)).norm().item()),
                         loaded_npz)
        except Exception:
            pass

    # 5) env
    env = gym.make('CrowdSim-v0')
    env.configure(env_cfg)
    robot = Robot(env_cfg, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    # ---- consistency logs
    kin = getattr(getattr(env, 'action_space', env), 'kinematics', '?')
    succ_r = getattr(env, 'success_radius', None)
    tlimit = getattr(env, 'time_limit', None)
    logging.info("Env kinematics=%s, success_radius=%s, time_limit=%s", kin, succ_r, tlimit)
    logging.info('human number: %d', hn_use)
    logging.info("Randomize human attrs: %s", bool(getattr(env, 'randomize_attributes', False)))
    logging.info('Train sim: %s, Test sim: %s', getattr(env, 'train_val_sim', '?'), getattr(env, 'test_sim', '?'))
    logging.info('Square width: %s, circle radius: %s', getattr(env, 'square_width', '?'), getattr(env, 'circle_radius', '?'))

    # 6) test-mode policy (deterministic)
    if hasattr(policy, 'set_env'):
        policy.set_env(env)
    if hasattr(policy, 'set_phase'):
        policy.set_phase('test')
    if hasattr(policy, 'set_epsilon'):
        policy.set_epsilon(0.0)         # no extra noise at test
    if hasattr(policy, 'env_stochastic'):
        policy.env_stochastic = False   # use mean action

    # 7) evaluate
    explorer = Explorer(env, robot, device=device, memory=None, gamma=0.99, target_policy=policy)
    stats = explorer.run_k_episodes(episodes, 'test', update_memory=False, return_stats=True, show_tqdm=True)
    logging.info(
        "TEST | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
        stats["success_rate"], stats["collision_rate"], stats["timeout_rate"],
        stats["nav_time"], stats["total_reward"]
    )

    # 8) optional video (mp4 -> gif fallback)
    video_out_path = None
    if save_video:
        case_idx = int(video_case) if video_case is not None else 0
        rollout_one_case(env, policy, test_case_idx=case_idx, time_limit=getattr(env, 'time_limit', None))
        try:
            env.render(mode='video', output_file=save_video)
            video_out_path = save_video
        except Exception as e:
            logging.error("Saving video failed: %s", e)
            try:
                gif_path = os.path.splitext(save_video)[0] + ".gif"
                env.render(mode='video', output_file=gif_path)
                video_out_path = gif_path
                logging.info("Saved GIF fallback to %s", gif_path)
            except Exception as e2:
                logging.error("GIF fallback also failed: %s", e2)
                video_out_path = None

    return stats, video_out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='mamba', help='mamba/orca/...')
    parser.add_argument('--model_dir', type=str, default='data/output')
    # ↓ 默认直接用训练输出目录里的那两份配置（与训练完全一致）
    parser.add_argument('--env_config', type=str, default='data/output/env.config')
    parser.add_argument('--policy_config', type=str, default='data/output/policy.config')
    parser.add_argument('--weights', type=str, default='rl_model_ep1000.pth')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--device', type=str, default='auto', help='"auto" | "cpu" | "cuda:0" ...')
    parser.add_argument('--save_video', type=str, default=None)
    parser.add_argument('--video_case', type=int, default=None)
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s')

    scenarios = [
        dict(desc='baseline_circle', sim='circle_crossing', human_num=5,  circle_radius=4.0),
        # 如需更快评测，可以先只跑 baseline_circle；其他场景按需开启
        # dict(desc='dense_circle',    sim='circle_crossing', human_num=10, circle_radius=4.0),
        # dict(desc='large_circle',    sim='circle_crossing', human_num=12, circle_radius=6.0),
    ]

    for sc in scenarios:
        logging.info("=== Scenario: %s ===", sc['desc'])
        stats, video_path = eval_one_scenario(
            policy_name=args.policy,
            model_dir=args.model_dir,
            weights_path=args.weights,
            env_cfg_path=args.env_config,
            policy_cfg_path=args.policy_config,
            device=device,
            episodes=args.episodes,
            sim_name=sc['sim'],
            human_num_case=sc['human_num'],
            circle_radius=sc.get('circle_radius'),
            square_width=sc.get('square_width'),
            save_video=args.save_video if sc['desc'] == 'baseline_circle' else None,
            video_case=args.video_case
        )
        logging.info(
            "RESULT | sim=%s humans(case)=%d | succ=%.2f coll=%.2f timeout=%.2f nav=%.2f reward=%.2f",
            sc['sim'], sc['human_num'],
            stats["success_rate"], stats["collision_rate"], stats["timeout_rate"],
            stats["nav_time"], stats["total_reward"]
        )
        if sc['desc'] == 'baseline_circle' and video_path:
            logging.info("Video saved to: %s", video_path)


if __name__ == '__main__':
    main()
