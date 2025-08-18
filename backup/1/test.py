import logging
import argparse
import configparser
import os
import torch
import gym
import numpy as np
import random

from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot

# ==== Global seed for full reproducibility ====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ABS_POLICY_CONFIG = "/home/abc/crowdnav_ws/src/CrowdNav/crowd_nav/configs/policy.config"
ABS_ENV_CONFIG = "/home/abc/crowdnav_ws/src/CrowdNav/crowd_nav/configs/env.config"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='orca', help='Policy: orca/sarl/cadrl')
    parser.add_argument('--model_dir', type=str, default=None, help='RL model directory (for RL policy only)')
    parser.add_argument('--env_config', type=str, default=ABS_ENV_CONFIG)
    parser.add_argument('--policy_config', type=str, default=ABS_POLICY_CONFIG)
    parser.add_argument('--episodes', type=int, default=10000, help='Episodes per scenario')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--test_case', type=int, default=None, help='Run only specified test_case index')
    args = parser.parse_args()

    device = torch.device(args.device if args.device != 'cpu' else (
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # Scenarios: Circle and Square
    test_cases = [
        {'desc': 'baseline_circle', 'sim': 'circle_crossing', 'human_num': 5,  'circle_radius': 4.0,  'robot_start': (0, -4.0),  'robot_goal': (0, 4.0)},
        {'desc': 'dense_circle',    'sim': 'circle_crossing', 'human_num': 10, 'circle_radius': 4.0,  'robot_start': (0, -4.0),  'robot_goal': (0, 4.0)},
        {'desc': 'large_circle',    'sim': 'circle_crossing', 'human_num': 12, 'circle_radius': 6.0,  'robot_start': (0, -6.0),  'robot_goal': (0, 6.0)},
        {'desc': 'baseline_square', 'sim': 'square_crossing', 'human_num': 10, 'square_width': 10.0,  'robot_start': (0, -5.0),  'robot_goal': (0, 5.0)},
        {'desc': 'dense_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 10.0,  'robot_start': (0, -5.0),  'robot_goal': (0, 5.0)},
        {'desc': 'large_square',    'sim': 'square_crossing', 'human_num': 20, 'square_width': 14.0,  'robot_start': (0, -7.0),  'robot_goal': (0, 7.0)},
    ]

    results_summary = []

    for case in test_cases:
        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config, encoding='utf-8')
        if not env_config.has_section('sim'):
            env_config.add_section('sim')
        env_config.set('sim', 'test_sim', case['sim'])
        env_config.set('sim', 'human_num', str(case['human_num']))
        if 'circle_radius' in case:
            env_config.set('sim', 'circle_radius', str(case['circle_radius']))
        if 'square_width' in case:
            env_config.set('sim', 'square_width', str(case['square_width']))

        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config, encoding='utf-8')
        if not policy_config.sections():
            print(f"!!! policy_config not loaded: {args.policy_config}")
            raise FileNotFoundError("policy_config not found.")

        policy = policy_factory[args.policy]()
        policy.configure(policy_config)
        if hasattr(policy, "trainable") and policy.trainable:
            if not args.model_dir:
                raise ValueError("RL policy requires --model_dir")
            model_weights = os.path.join(args.model_dir, "rl_model.pth")
            state_dict = torch.load(model_weights, map_location=device)
            policy.get_model().load_state_dict(state_dict)

        env = gym.make('CrowdSim-v0')
        env.configure(env_config)
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        env.set_robot(robot)
        explorer = Explorer(env, robot, device=device, gamma=0.9)

        policy.set_phase('test')
        policy.set_device(device)
        policy.set_env(env)

        print(f"\n======= Evaluate: {case['desc']} | {args.policy} | {case['human_num']} agents =======")
        success_cnt, collision_cnt, timeout_cnt, total_time = 0, 0, 0, 0.0
        efficiencies, min_separations, mean_separations = [], [], []

        ep_range = range(args.episodes) if args.test_case is None else [args.test_case]
        for ep_i in ep_range:
            ob = env.reset(phase='test', test_case=ep_i)
            robot.set_position(case['robot_start'])
            robot.gx, robot.gy = case['robot_goal']

            done = False
            steps = 0
            robot_positions = [robot.get_position()]
            min_dists = []

            while not done:
                action = robot.act(ob)
                ob, reward, done, info = env.step(action)
                robot_positions.append(robot.get_position())
                # 计算与最近人的距离
                robot_pos = np.array(robot.get_position())
                dists = [np.linalg.norm(robot_pos - np.array(h.get_position())) for h in env.humans]
                min_dists.append(min(dists))
                steps += 1

            # 路径效率比统计
            actual_length = sum(
                np.linalg.norm(np.array(robot_positions[i+1]) - np.array(robot_positions[i]))
                for i in range(len(robot_positions)-1)
            )
            shortest = np.linalg.norm(np.array(case['robot_goal']) - np.array(case['robot_start']))
            efficiency = actual_length / shortest if shortest > 1e-6 else float('nan')
            efficiencies.append(efficiency)
            # min separation
            min_separations.append(min(min_dists))
            mean_separations.append(np.mean(min_dists))

            if info.__class__.__name__.lower().startswith('reach'):
                success_cnt += 1
            elif info.__class__.__name__.lower().startswith('collision'):
                collision_cnt += 1
            elif info.__class__.__name__.lower().startswith('timeout'):
                timeout_cnt += 1
            total_time += steps * env.time_step

        n = len(ep_range)
        avg_time = (total_time / success_cnt) if success_cnt > 0 else float('nan')
        avg_eff = np.nanmean(efficiencies)
        avg_min_sep = np.nanmean(min_separations)
        avg_mean_sep = np.nanmean(mean_separations)
        results_summary.append({
            'scene': case['desc'],
            'policy': args.policy,
            'num_humans': case['human_num'],
            'success_rate': success_cnt / n if n else 0,
            'collision_rate': collision_cnt / n if n else 0,
            'timeout_rate': timeout_cnt / n if n else 0,
            'avg_time': avg_time,
            'path_efficiency': avg_eff,
            'min_sep': avg_min_sep,
            'mean_sep': avg_mean_sep,
        })

    print("\n==== Summary of All Experiments ====")
    print("{:<18} {:<8} {:<6} {:>10} {:>12} {:>10} {:>12} {:>14} {:>12}".format(
        "Scenario", "Policy", "Num", "SuccessRate", "CollisionRate", "TimeoutRate",
        "AvgTime(s)", "PathEff", "MinDist"
    ))
    for r in results_summary:
        avg_time_str = "-" if np.isnan(r['avg_time']) else f"{r['avg_time']:12.2f}"
        avg_eff_str = "-" if np.isnan(r['path_efficiency']) else f"{r['path_efficiency']:14.3f}"
        min_sep_str = "-" if np.isnan(r['min_sep']) else f"{r['min_sep']:12.2f}"
        print("{:<18} {:<8} {:<6d} {:>10.3f} {:>12.3f} {:>10.3f} {} {} {}".format(
            r['scene'], r['policy'], r['num_humans'],
            r['success_rate'], r['collision_rate'], r['timeout_rate'],
            avg_time_str, avg_eff_str, min_sep_str
        ))

    # -------- Optionally: Save summary as .csv or .txt here --------
    # with open('summary.csv', 'w') as f:
    #     ...

if __name__ == '__main__':
    main()

