#!/usr/bin/env python3
"""
Local AttnGraph smoke/evaluation entry.

This keeps AttnGraph testing separate from test.py so the Mamba evaluation path
is not affected. It uses the local CrowdNav 6-scenario protocol and the
AttnGraphPolicy adapter.
"""
from __future__ import annotations

import argparse
import configparser
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from crowd_nav.policy.attngraph_policy import (
    AttnGraphPolicy,
    DEFAULT_GST_MODEL_DIR,
    DEFAULT_MODEL_DIR,
)
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_configs(args, case):
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)

    if not env_config.has_section("env"):
        env_config.add_section("env")
    env_config.set("env", "time_step", "0.25")
    env_config.set("env", "time_limit", str(int(args.time_limit)))

    if not env_config.has_section("robot"):
        env_config.add_section("robot")
    env_config.set("robot", "v_pref", "1.0")

    for section in policy_config.sections():
        if not env_config.has_section(section):
            env_config.add_section(section)
        for key, value in policy_config.items(section):
            env_config.set(section, key, value)

    if not env_config.has_section("sim"):
        env_config.add_section("sim")
    env_config.set("sim", "test_sim", case["sim"])
    env_config.set("sim", "human_num", str(case["human_num"]))
    if "circle_radius" in case:
        env_config.set("sim", "circle_radius", str(case["circle_radius"]))
    if "square_width" in case:
        env_config.set("sim", "square_width", str(case["square_width"]))


    return env_config, policy_config


def compute_min_clearance(env, robot):
    min_dist = float("inf")
    for human in env.humans:
        dist = np.linalg.norm(np.array([robot.px, robot.py]) - np.array([human.px, human.py]))
        dist -= (robot.radius + human.radius)
        min_dist = min(min_dist, float(dist))
    return min_dist


def run_episode(env, robot, policy, seed):
    if hasattr(policy, "reset_episode_stats"):
        policy.reset_episode_stats()

    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        _ = reset_result[0]

    done = False
    steps = 0
    info = None
    max_steps = 500
    min_dists = []

    while not done and steps < max_steps:
        robot_state = robot.get_full_state()
        human_states = [human.get_observable_state() for human in env.humans]
        state = JointState(robot_state, human_states)
        action = policy.predict(state)
        step_result = env.step(action)
        if len(step_result) == 5:
            _, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            _, _, done, info = step_result
        min_dists.append(compute_min_clearance(env, robot))
        steps += 1

    outcome = "timeout"
    if isinstance(info, dict):
        event = str(info.get("event", "")).lower()
        if any(x in event for x in ["reach_goal", "success", "reachgoal"]):
            outcome = "success"
        elif "collision" in event:
            outcome = "collision"
    elif info is not None:
        event_name = info.__class__.__name__
        if event_name == "ReachGoal":
            outcome = "success"
        elif event_name == "Collision":
            outcome = "collision"
    return outcome, steps, min_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gst_model_dir", type=str, default=DEFAULT_GST_MODEL_DIR)
    parser.add_argument("--weights", type=str, default="checkpoints/41665.pt")
    parser.add_argument("--prediction", choices=["gst", "cv"], default="gst")
    parser.add_argument("--env_config", type=str, default="configs/env.config")
    parser.add_argument("--policy_config", type=str, default="configs/policy.config")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--test_case", type=int, default=None)
    parser.add_argument("--time_limit", type=float, default=25.0)
    parser.add_argument(
        "--attngraph_native_circle",
        action="store_true",
        help="Run a native-scale AttnGraph diagnostic case: circle, 20 humans, radius 6*sqrt(2), 50s.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)

    print("[INFO] TEST8 - AttnGraph local adapter")
    print(f"[INFO] device={device}")
    print(f"[INFO] model_dir={args.model_dir}")
    print(f"[INFO] weights={args.weights}")
    print(f"[INFO] prediction input={args.prediction}")
    if args.prediction == "gst":
        print(f"[INFO] gst_model_dir={args.gst_model_dir}")

    weights_path = args.weights
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(args.model_dir, args.weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)

    policy = AttnGraphPolicy(
        model_dir=args.model_dir,
        gst_model_dir=args.gst_model_dir,
        prediction=args.prediction,
        device=device,
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    policy.load_state_dict(state_dict)
    policy.set_phase("test")
    policy.eval()

    test_cases = [
        {"case_id": 0, "desc": "baseline_circle", "sim": "circle_crossing", "human_num": 5, "circle_radius": 4.0},
        {"case_id": 1, "desc": "baseline_square", "sim": "square_crossing", "human_num": 10, "square_width": 10.0},
        {"case_id": 2, "desc": "dense_circle", "sim": "circle_crossing", "human_num": 10, "circle_radius": 4.0},
        {"case_id": 3, "desc": "dense_square", "sim": "square_crossing", "human_num": 20, "square_width": 10.0},
        {"case_id": 4, "desc": "large_circle", "sim": "circle_crossing", "human_num": 12, "circle_radius": 6.0},
        {"case_id": 5, "desc": "large_square", "sim": "square_crossing", "human_num": 20, "square_width": 14.0},
    ]
    if args.attngraph_native_circle:
        args.time_limit = 50.0
        test_cases = [{
            "case_id": 0,
            "desc": "attngraph_native_circle",
            "sim": "circle_crossing",
            "human_num": 20,
            "circle_radius": float(6 * np.sqrt(2)),
        }]
    if args.test_case is not None:
        test_cases = [test_cases[args.test_case]]

    all_success = 0
    all_collision = 0
    all_timeout = 0
    all_episodes = 0
    all_times = []
    all_min_dist_sum = 0.0
    all_min_dist_steps = 0
    all_discomfort_events = 0

    for case in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Test Case [{case['case_id']}]: {case['desc']} | {case['human_num']} humans")
        print(f"{'=' * 70}")

        env_config, policy_config = build_configs(args, case)
        policy.configure(policy_config)

        env = CrowdSim()
        env.configure(env_config)
        env.phase = "test"
        robot = Robot(env_config, "robot")
        robot.set_policy(policy)
        robot.env = env
        env.set_robot(robot)

        success = collision = timeout = 0
        times = []
        min_dist_sum = 0.0
        min_dist_steps = 0
        discomfort_events = 0
        discomfort_dist = env_config.getfloat("reward", "discomfort_dist", fallback=0.2)
        pbar = tqdm(range(args.episodes), ncols=100)
        for ep in pbar:
            episode_seed = (args.seed + case["case_id"] * 1_000_003 + ep) % (2**31 - 1)
            outcome, steps, min_dists = run_episode(env, robot, policy, episode_seed)
            if outcome == "success":
                success += 1
                times.append(steps * env.time_step)
            elif outcome == "collision":
                collision += 1
            else:
                timeout += 1
            if min_dists:
                min_dist_sum += float(np.sum(min_dists))
                min_dist_steps += len(min_dists)
                discomfort_events += sum(1 for d in min_dists if d < discomfort_dist)
            pbar.set_postfix(S=f"{success}/{ep + 1}", C=f"{collision}/{ep + 1}")

        all_success += success
        all_collision += collision
        all_timeout += timeout
        all_episodes += args.episodes
        all_times.extend(times)
        all_min_dist_sum += min_dist_sum
        all_min_dist_steps += min_dist_steps
        all_discomfort_events += discomfort_events
        avg_disc_freq = discomfort_events / args.episodes
        avg_disc_dist = (min_dist_sum / min_dist_steps) if min_dist_steps else float("inf")
        print(f"\nResults for {case['desc']}:")
        print(f"  SUCCESS:   {success}/{args.episodes} ({success / args.episodes * 100:.1f}%)")
        print(f"  COLLISION: {collision}/{args.episodes} ({collision / args.episodes * 100:.1f}%)")
        print(f"  TIMEOUT:   {timeout}/{args.episodes} ({timeout / args.episodes * 100:.1f}%)")
        if times:
            print(f"  TIME TAKEN (s): {np.mean(times):.2f}")
        else:
            print("  TIME TAKEN (s): n/a")
        print(f"  DISC. FREQ: {avg_disc_freq:.2f}")
        print(f"  DISC. DIST (m): {avg_disc_dist:.2f}")

    print(f"\n{'=' * 70}")
    print("Aggregate:")
    print(f"  SUCCESS:   {all_success}/{all_episodes} ({all_success / all_episodes * 100:.1f}%)")
    print(f"  COLLISION: {all_collision}/{all_episodes} ({all_collision / all_episodes * 100:.1f}%)")
    print(f"  TIMEOUT:   {all_timeout}/{all_episodes} ({all_timeout / all_episodes * 100:.1f}%)")
    if all_times:
        print(f"  TIME TAKEN (s): {np.mean(all_times):.2f}")
    else:
        print("  TIME TAKEN (s): n/a")
    avg_disc_freq = all_discomfort_events / all_episodes
    avg_disc_dist = (all_min_dist_sum / all_min_dist_steps) if all_min_dist_steps else float("inf")
    print(f"  DISC. FREQ: {avg_disc_freq:.2f}")
    print(f"  DISC. DIST (m): {avg_disc_dist:.2f}")


if __name__ == "__main__":
    main()
