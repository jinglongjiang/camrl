#!/usr/bin/env python3
"""Generate simulator-derived action preferences for difficult crowd states."""

from __future__ import annotations

import argparse
from collections import deque
import configparser
import copy
import json
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.contracts import (
    _batch_joint34_to_tokens_vectorized,
    action_to_discrete_index,
    init_grid_from_cfg,
)
from crowd_nav.policy.bayesian_fullcrowd_risk_value import (
    BayesianFullCrowdRiskValuePolicy,
)
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_nav.tools.collect_bayesian_risk_trajectories import (
    SCENARIOS,
    extract_state_dict,
    merge_config,
    min_clearance,
    outcome_from_info,
    torch_load,
)
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_bayesian_config(args, scenario: Dict) -> configparser.RawConfigParser:
    config = configparser.RawConfigParser()
    loaded = config.read(
        [args.env_config, args.bayesian_policy_config, args.train_config]
    )
    if len(loaded) != 3:
        raise FileNotFoundError(
            f"Expected three Bayesian config files, loaded={loaded}"
        )
    if not config.has_section("sim"):
        config.add_section("sim")
    config.set("sim", "test_sim", scenario["sim"])
    config.set("sim", "human_num", str(scenario["humans"]))
    for key in ("circle_radius", "square_width"):
        if key in scenario:
            config.set("sim", key, str(scenario[key]))
    if not config.has_section("env"):
        config.add_section("env")
    config.set("env", "time_step", str(args.time_step))
    config.set("env", "time_limit", str(int(round(args.time_limit))))
    init_grid_from_cfg(config)
    return config


def make_world(args, scenario: Dict, device: torch.device):
    env_config, policy_config = merge_config(args, scenario)
    policy = MambaRLPolicy(config=policy_config, device=device)
    policy.load_state_dict(extract_state_dict(torch_load(args.source)))
    policy.use_sarl_predict = True
    policy.set_phase("test")
    policy.eval()

    env = CrowdSim()
    env.configure(env_config)
    env.phase = "test"
    robot = Robot(env_config, "robot")
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)
    policy.set_env_dt(args.time_step)

    bayesian = BayesianFullCrowdRiskValuePolicy(
        config=build_bayesian_config(args, scenario),
        device=device,
    )
    bayesian.load_state_dict(
        extract_state_dict(torch_load(args.bayesian_checkpoint)),
        strict=True,
    )
    bayesian.set_phase("test")
    bayesian.eval()
    return env, robot, policy, bayesian


def presence_from_states(states: np.ndarray, num_humans: int) -> np.ndarray:
    output = np.zeros((len(states), num_humans), dtype=np.bool_)
    for pedestrian_index in range(num_humans):
        start = 9 + 5 * pedestrian_index
        pedestrian = states[:, start:start + 5]
        output[:, pedestrian_index] = ~np.all(
            np.isclose(pedestrian, 0.0),
            axis=1,
        )
    return output


def state_objects(state_34: np.ndarray):
    robot = FullState(*np.asarray(state_34[:9], dtype=np.float32).tolist())
    humans = []
    for pedestrian_index in range(5):
        start = 9 + 5 * pedestrian_index
        pedestrian = np.asarray(
            state_34[start:start + 5],
            dtype=np.float32,
        )
        if pedestrian.size == 5 and not np.allclose(pedestrian, 0.0):
            humans.append(ObservableState(*pedestrian.tolist()))
    return robot, humans


def candidate_sequences(policy, history_tokens, state_34):
    if policy.action_space is None:
        policy.build_action_space(1.0)
    robot, humans = state_objects(state_34)
    sequences = []
    rewards = []
    actions = []
    for action in policy.action_space:
        next_robot = policy.propagate(robot, action)
        next_humans = [
            policy.propagate(human, ActionXY(human.vx, human.vy))
            for human in humans
        ]
        rewards.append(
            policy.compute_reward(
                next_robot,
                next_humans,
                prev_nav=robot,
                action=action,
            )
        )
        actions.append([float(action.vx), float(action.vy)])
        next_state = policy._build_joint_state_34(next_robot, next_humans)
        next_token = _batch_joint34_to_tokens_vectorized(
            next_state.reshape(1, -1)
        )[0]
        sequences.append(
            torch.cat(
                (
                    history_tokens[1:],
                    torch.as_tensor(next_token).unsqueeze(0),
                ),
                dim=0,
            )
        )
    return (
        torch.stack(sequences).float(),
        torch.tensor(rewards, dtype=torch.float32),
        np.asarray(actions, dtype=np.float64),
    )


def joint_state(robot, humans) -> JointState:
    return JointState(
        robot.get_full_state(),
        [human.get_observable_state() for human in humans],
    )


def event_flags(info) -> Tuple[bool, bool]:
    event = ""
    if isinstance(info, dict):
        event = str(info.get("event", "")).lower()
    elif info is not None:
        event = info.__class__.__name__.lower()
    return "collision" in event, (
        "reachgoal" in event or "reach_goal" in event or "success" in event
    )


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        _, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        _, reward, done, info = result
    return float(reward), bool(done), info


def agent_state(agent) -> Dict:
    return {
        key: copy.deepcopy(getattr(agent, key))
        for key in (
            "px",
            "py",
            "gx",
            "gy",
            "vx",
            "vy",
            "theta",
            "radius",
            "v_pref",
        )
    }


def restore_agent(agent, state: Dict) -> None:
    for key, value in state.items():
        setattr(agent, key, copy.deepcopy(value))


def human_policy_state(human) -> Dict:
    policy = human.policy
    return {
        "last_pref_vel": copy.deepcopy(
            getattr(policy, "_last_pref_vel", None)
        ),
    }


def restore_human_policy(human, state: Dict) -> None:
    policy = human.policy
    if getattr(policy, "sim", None) is not None:
        policy.sim = None
    if hasattr(policy, "_last_pref_vel"):
        policy._last_pref_vel = copy.deepcopy(state["last_pref_vel"])


def capture_runtime(env, policy) -> Dict:
    return {
        "robot": agent_state(env.robot),
        "humans": [agent_state(human) for human in env.humans],
        "human_policies": [
            human_policy_state(human) for human in env.humans
        ],
        "global_time": float(env.global_time),
        "human_times": copy.deepcopy(env.human_times),
        "no_prog_steps": int(getattr(env, "_no_prog_steps", 0)),
        "states_len": len(env.states),
        "action_values_len": (
            len(env.action_values)
            if isinstance(env.action_values, list)
            else None
        ),
        "attention_len": (
            len(env.attention_weights)
            if isinstance(env.attention_weights, list)
            else None
        ),
        "history": [np.asarray(item).copy() for item in policy._history],
        "last_action": copy.deepcopy(getattr(policy, "_last_action", None)),
        "numpy_state": np.random.get_state(),
    }


def restore_runtime(env, policy, snapshot: Dict) -> None:
    restore_agent(env.robot, snapshot["robot"])
    if len(env.humans) != len(snapshot["humans"]):
        raise RuntimeError("Human count changed during counterfactual rollout")
    for human, state, policy_state in zip(
        env.humans,
        snapshot["humans"],
        snapshot["human_policies"],
    ):
        restore_agent(human, state)
        restore_human_policy(human, policy_state)
    env.global_time = snapshot["global_time"]
    env.human_times = copy.deepcopy(snapshot["human_times"])
    env._no_prog_steps = snapshot["no_prog_steps"]
    del env.states[snapshot["states_len"]:]
    if snapshot["action_values_len"] is not None:
        del env.action_values[snapshot["action_values_len"]:]
    if snapshot["attention_len"] is not None:
        del env.attention_weights[snapshot["attention_len"]:]
    policy._history = deque(
        [item.copy() for item in snapshot["history"]],
        maxlen=policy.seq_len,
    )
    policy._last_action = copy.deepcopy(snapshot["last_action"])
    np.random.set_state(snapshot["numpy_state"])


def padded_history(policy) -> torch.Tensor:
    history = [np.asarray(item).copy() for item in policy._history]
    if not history:
        raise RuntimeError("Policy history is empty after action selection")
    if len(history) < policy.seq_len:
        history = [history[0].copy()] * (policy.seq_len - len(history)) + history
    return torch.from_numpy(np.asarray(history[-policy.seq_len:])).float()


def trace_episode(
    env,
    robot,
    policy,
    case_index: int,
    seed: int,
) -> Tuple[List[Dict], str]:
    policy.reset_episode_stats()
    env.reset(seed=seed, options={"test_case": case_index})
    trace: List[Dict] = []
    done = False
    info = None
    max_steps = int(np.ceil(env.time_limit / env.time_step)) + 1
    while not done and len(trace) < max_steps:
        state = joint_state(robot, env.humans)
        state_34 = policy._build_joint_state_34(
            state.self_state,
            state.human_states,
        ).astype(np.float32)
        current_clearance = float(min_clearance(robot, env.humans))
        action = policy.predict(state)
        _, done, info = step_env(env, action)
        trace.append(
            {
                "step": len(trace),
                "state_34": state_34,
                "clearance": min(
                    current_clearance,
                    float(min_clearance(robot, env.humans)),
                ),
            }
        )
    return trace, outcome_from_info(info)


def selected_steps(
    trace: Sequence[Dict],
    outcome: str,
    states_per_episode: int,
) -> List[int]:
    if not trace:
        return []
    ranked = sorted(trace, key=lambda item: item["clearance"])
    selected = {
        int(item["step"])
        for item in ranked[:states_per_episode]
    }
    if outcome in {"collision", "timeout"}:
        selected.update(
            range(
                max(0, len(trace) - states_per_episode),
                len(trace),
            )
        )
    ordered = sorted(
        selected,
        key=lambda step: (trace[step]["clearance"], -step),
    )
    return sorted(ordered[: max(states_per_episode, 1)])


def candidate_pool(
    teacher_scores: np.ndarray,
    selected_action,
    candidate_count: int,
) -> np.ndarray:
    total = int(len(teacher_scores))
    top_count = max(1, candidate_count // 2)
    indexes: List[int] = np.argsort(-teacher_scores)[:top_count].tolist()
    indexes.extend(
        np.linspace(
            0,
            total - 1,
            num=max(1, candidate_count - top_count),
            dtype=np.int64,
        ).tolist()
    )
    try:
        indexes.append(
            int(
                action_to_discrete_index(
                    float(selected_action.vx),
                    float(selected_action.vy),
                )
            )
        )
    except Exception:
        pass
    unique = []
    seen = set()
    for index in indexes + np.argsort(-teacher_scores).tolist():
        index = int(index)
        if index not in seen:
            seen.add(index)
            unique.append(index)
        if len(unique) >= min(candidate_count, total):
            break
    return np.asarray(unique, dtype=np.int64)


def branch_label(
    env,
    robot,
    policy,
    snapshot: Dict,
    first_action: ActionXY,
    horizon: int,
    stall_progress: float,
) -> Dict:
    restore_runtime(env, policy, snapshot)
    start_distance = float(
        np.hypot(robot.gx - robot.px, robot.gy - robot.py)
    )
    minimum_clearance = float(min_clearance(robot, env.humans))
    total_reward = 0.0
    collision = False
    success = False
    done = False
    steps = 0
    action = first_action
    for step in range(horizon):
        reward, done, info = step_env(env, action)
        total_reward += reward
        steps += 1
        minimum_clearance = min(
            minimum_clearance,
            float(min_clearance(robot, env.humans)),
            float(info.get("dmin", float("inf")))
            if isinstance(info, dict)
            else float("inf"),
        )
        collided, reached = event_flags(info)
        collision = collision or collided
        success = success or reached
        if done:
            break
        state = joint_state(robot, env.humans)
        action = policy.predict(state)
    end_distance = float(np.hypot(robot.gx - robot.px, robot.gy - robot.py))
    progress = start_distance - end_distance
    return {
        "collision": bool(collision),
        "success": bool(success),
        "min_clearance": float(minimum_clearance),
        "progress": float(progress),
        "stalled": bool(not success and progress < stall_progress),
        "reward": float(total_reward),
        "steps": int(steps),
    }


def preference_pairs(
    labels: Sequence[Dict],
    teacher_top_index: int,
    clearance_gap: float,
    progress_gap: float,
    max_pairs: int,
    rng: np.random.Generator,
) -> Dict:
    pairs = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            a, b = labels[left], labels[right]
            preferred = rejected = None
            weight = 1.0
            reason = ""
            if a["collision"] != b["collision"]:
                preferred, rejected = (
                    (right, left) if a["collision"] else (left, right)
                )
                weight, reason = 3.0, "collision"
            elif a["success"] != b["success"]:
                preferred, rejected = (
                    (left, right) if a["success"] else (right, left)
                )
                weight, reason = 2.0, "success"
            elif a["stalled"] != b["stalled"]:
                preferred, rejected = (
                    (right, left) if a["stalled"] else (left, right)
                )
                weight, reason = 1.5, "stall"
            else:
                clearance_delta = (
                    a["min_clearance"] - b["min_clearance"]
                )
                progress_delta = a["progress"] - b["progress"]
                if (
                    abs(clearance_delta) >= clearance_gap
                    and (
                        (clearance_delta > 0 and progress_delta >= -progress_gap)
                        or (
                            clearance_delta < 0
                            and progress_delta <= progress_gap
                        )
                    )
                ):
                    preferred, rejected = (
                        (left, right)
                        if clearance_delta > 0
                        else (right, left)
                    )
                    weight, reason = 1.0, "clearance"
                elif (
                    abs(progress_delta) >= progress_gap
                    and min(a["min_clearance"], b["min_clearance"]) >= 0.05
                ):
                    preferred, rejected = (
                        (left, right)
                        if progress_delta > 0
                        else (right, left)
                    )
                    weight, reason = 0.75, "progress"
            if preferred is not None:
                if rejected == teacher_top_index:
                    if reason == "collision":
                        weight *= 4.0
                    elif reason in {"stall", "success"}:
                        weight *= 2.5
                    else:
                        weight *= 1.5
                pairs.append((preferred, rejected, weight, reason))
    if len(pairs) > max_pairs:
        collision_pairs = [pair for pair in pairs if pair[3] == "collision"]
        other_pairs = [pair for pair in pairs if pair[3] != "collision"]
        keep_other = max(0, max_pairs - len(collision_pairs))
        if len(other_pairs) > keep_other:
            chosen = rng.choice(
                len(other_pairs),
                size=keep_other,
                replace=False,
            )
            other_pairs = [other_pairs[int(index)] for index in chosen]
        pairs = collision_pairs[:max_pairs] + other_pairs
    return {
        "preferred": torch.tensor(
            [pair[0] for pair in pairs],
            dtype=torch.long,
        ),
        "rejected": torch.tensor(
            [pair[1] for pair in pairs],
            dtype=torch.long,
        ),
        "weights": torch.tensor(
            [pair[2] for pair in pairs],
            dtype=torch.float32,
        ),
        "reasons": [pair[3] for pair in pairs],
    }


def replay_and_label(
    args,
    env,
    robot,
    policy,
    bayesian,
    trace: Sequence[Dict],
    target_steps: Iterable[int],
    scenario: str,
    outcome: str,
    case_index: int,
    seed: int,
    rng: np.random.Generator,
) -> List[Dict]:
    target_steps = set(int(step) for step in target_steps)
    policy.reset_episode_stats()
    bayesian.reset_episode_stats()
    env.reset(seed=seed, options={"test_case": case_index})
    samples = []
    done = False
    step = 0
    while not done and step < len(trace):
        state = joint_state(robot, env.humans)
        state_34 = policy._build_joint_state_34(
            state.self_state,
            state.human_states,
        ).astype(np.float32)
        if not np.allclose(state_34, trace[step]["state_34"], atol=1e-5):
            raise RuntimeError(
                f"Replay diverged in {scenario} case={case_index} step={step}"
            )
        belief_state = bayesian.build_belief_state(
            state.self_state,
            state.human_states,
        )
        bayesian.belief_filter.update(belief_state)
        selected_action = policy.predict(state)
        if step in target_steps:
            history = padded_history(policy)
            sequences, rewards, actions = candidate_sequences(
                policy,
                history,
                state_34,
            )
            with torch.no_grad():
                teacher_values = policy.forward_value(
                    sequences.to(policy.device)
                ).reshape(-1)
                teacher_scores = (
                    rewards.to(policy.device)
                    + policy.gamma * teacher_values
                ).cpu().numpy()
            indexes = candidate_pool(
                teacher_scores,
                selected_action,
                args.candidate_count,
            )
            selected_actions = actions[indexes]
            belief = np.asarray(
                bayesian.belief_filter.get_per_ped_belief_vec(),
                dtype=np.float32,
            )
            belief_batch = np.repeat(
                belief.reshape(1, *belief.shape),
                len(indexes),
                axis=0,
            )
            state_batch = np.repeat(
                belief_state.reshape(1, -1),
                len(indexes),
                axis=0,
            )
            action_features = bayesian.compute_action_features_batch(
                state_batch,
                selected_actions,
                belief_batch,
            )
            presence = presence_from_states(
                belief_state.reshape(1, -1),
                bayesian.belief_num_humans,
            )[0]
            snapshot = capture_runtime(env, policy)
            labels = [
                branch_label(
                    env,
                    robot,
                    policy,
                    snapshot,
                    ActionXY(float(action[0]), float(action[1])),
                    horizon=args.horizon,
                    stall_progress=args.stall_progress,
                )
                for action in selected_actions
            ]
            restore_runtime(env, policy, snapshot)
            pairs = preference_pairs(
                labels,
                teacher_top_index=int(
                    np.argmax(teacher_scores[indexes])
                ),
                clearance_gap=args.clearance_gap,
                progress_gap=args.progress_gap,
                max_pairs=args.max_pairs,
                rng=rng,
            )
            if len(pairs["preferred"]) > 0:
                samples.append(
                    {
                        "scenario": scenario,
                        "episode_outcome": outcome,
                        "case_index": int(case_index),
                        "step": int(step),
                        "source_clearance": float(trace[step]["clearance"]),
                        "sequences": sequences[indexes].cpu(),
                        "rewards": rewards[indexes].cpu(),
                        "teacher_scores": torch.from_numpy(
                            teacher_scores[indexes].copy()
                        ).float(),
                        "belief": torch.from_numpy(belief.copy()),
                        "presence": torch.from_numpy(presence.copy()),
                        "action_features": torch.from_numpy(
                            action_features.copy()
                        ),
                        "actions": torch.from_numpy(
                            selected_actions.astype(np.float32, copy=True)
                        ),
                        "labels": labels,
                        "pairs": pairs,
                    }
                )
        _, done, _ = step_env(env, selected_action)
        step += 1
    return samples


def summarize(samples: Sequence[Dict]) -> Dict:
    reasons: Dict[str, int] = {}
    outcomes: Dict[str, int] = {}
    scenarios: Dict[str, int] = {}
    candidates = 0
    collisions = 0
    stalls = 0
    for sample in samples:
        scenarios[sample["scenario"]] = scenarios.get(sample["scenario"], 0) + 1
        outcome = sample["episode_outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        candidates += len(sample["labels"])
        collisions += sum(int(label["collision"]) for label in sample["labels"])
        stalls += sum(int(label["stalled"]) for label in sample["labels"])
        for reason in sample["pairs"]["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "states": len(samples),
        "candidates": candidates,
        "candidate_collisions": collisions,
        "candidate_stalls": stalls,
        "pair_reasons": reasons,
        "episode_outcomes": outcomes,
        "scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="runs/mamba_vl/rl_model_ep10000_T24.pth",
    )
    parser.add_argument(
        "--bayesian_checkpoint",
        default=(
            "runs/bayesian_distributional/"
            "model_fullcrowd_directrisk_lambda1_clean.pth"
        ),
    )
    parser.add_argument("--env_config", default="configs/env.config")
    parser.add_argument("--policy_config", default="configs/policy.config")
    parser.add_argument(
        "--bayesian_policy_config",
        default="configs/policy_bayesian_distributional.config",
    )
    parser.add_argument("--train_config", default="configs/train.config")
    parser.add_argument(
        "--scenarios",
        default="dense_square,large_square",
    )
    parser.add_argument("--episodes_per_scenario", type=int, default=12)
    parser.add_argument("--case_offset", type=int, default=200)
    parser.add_argument("--states_per_episode", type=int, default=3)
    parser.add_argument("--candidate_count", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stall_progress", type=float, default=0.05)
    parser.add_argument("--clearance_gap", type=float, default=0.08)
    parser.add_argument("--progress_gap", type=float, default=0.08)
    parser.add_argument("--max_pairs", type=int, default=96)
    parser.add_argument("--time_limit", type=float, default=25.0)
    parser.add_argument("--time_step", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--output",
        default=(
            "runs/bayesian_distributional/"
            "counterfactual_targeted.pt"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "runs/bayesian_distributional/"
            "counterfactual_targeted_report.json"
        ),
    )
    args = parser.parse_args()

    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    if args.candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    )
    requested = {
        name.strip() for name in args.scenarios.split(",") if name.strip()
    }
    scenarios = [item for item in SCENARIOS if item["name"] in requested]
    missing = requested - {item["name"] for item in scenarios}
    if missing:
        raise ValueError(f"Unknown scenarios: {sorted(missing)}")

    rng = np.random.default_rng(args.seed)
    samples: List[Dict] = []
    progress = tqdm(
        total=len(scenarios) * args.episodes_per_scenario,
        desc="counterfactual",
        unit="ep",
    )
    for scenario_index, scenario in enumerate(scenarios):
        env, robot, policy, bayesian = make_world(args, scenario, device)
        for episode in range(args.episodes_per_scenario):
            case_index = args.case_offset + episode
            episode_seed = (
                args.seed
                + scenario_index * 1_000_003
                + episode
            ) % (2**31 - 1)
            trace, outcome = trace_episode(
                env,
                robot,
                policy,
                case_index=case_index,
                seed=episode_seed,
            )
            targets = selected_steps(
                trace,
                outcome,
                args.states_per_episode,
            )
            samples.extend(
                replay_and_label(
                    args,
                    env,
                    robot,
                    policy,
                    bayesian,
                    trace,
                    targets,
                    scenario["name"],
                    outcome,
                    case_index,
                    episode_seed,
                    rng,
                )
            )
            progress.update(1)
            progress.set_postfix(
                states=len(samples),
                outcome=outcome[:1].upper(),
            )
    progress.close()

    summary = summarize(samples)
    payload = {
        "version": 1,
        "samples": samples,
        "meta": {
            "source_checkpoint": str(Path(args.source).resolve()),
            "bayesian_checkpoint": str(
                Path(args.bayesian_checkpoint).resolve()
            ),
            "seed": args.seed,
            "case_offset": args.case_offset,
            "scenarios": [item["name"] for item in scenarios],
            "episodes_per_scenario": args.episodes_per_scenario,
            "states_per_episode": args.states_per_episode,
            "candidate_count": args.candidate_count,
            "horizon": args.horizon,
            "label_source": "CrowdSim ORCA counterfactual rollout",
            "summary": summary,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload["meta"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[SAVE] dataset={output}")
    print(f"[SAVE] report={report}")
    print("[SUMMARY] " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
