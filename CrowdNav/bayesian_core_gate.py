#!/usr/bin/env python3
"""Fast, falsifiable gate for replacing Mamba memory with a Bayesian state.

The script does not modify or train the navigation policy.  It collects frozen
Mamba rollouts, probes what its temporal state represents, and checks whether a
causal per-person posterior changes decisions for the better over the complete
80-action grid.
"""

from __future__ import annotations

import argparse
import configparser
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln
from scipy.stats import f as f_dist


ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState


HORIZONS = (1, 2, 4, 8)
DT = 0.25
EXPECTED_CHECKPOINT_SHA256 = (
    "0c0a4c41efab63d6786b52f7b62b6ef087270e6b6f211a64fb175531fd428906"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_configs(env_path: Path, policy_path: Path, human_num: int):
    env = configparser.RawConfigParser()
    policy = configparser.RawConfigParser()
    if not env.read(env_path) or not policy.read(policy_path):
        raise FileNotFoundError("failed to read environment or policy config")

    if not env.has_section("occlusion"):
        env.add_section("occlusion")
    env.set("occlusion", "enabled", "false")
    env.set("occlusion", "mode", "off")
    env.set("env", "time_step", str(DT))
    env.set("env", "time_limit", "25")
    env.set("env", "test_size", "1000")
    env.set("sim", "test_sim", "circle_crossing")
    env.set("sim", "human_num", str(human_num))
    env.set("sim", "circle_radius", "4.0")
    env.set("robot", "v_pref", "1.0")

    for section in env.sections():
        if not policy.has_section(section):
            policy.add_section(section)
        for key, value in env.items(section):
            if not policy.has_option(section, key):
                policy.set(section, key, value)
    if not policy.has_section("buffer"):
        policy.add_section("buffer")
    policy.set("buffer", "seq_len", "24")
    policy.set("temporal", "T", "24")
    if not policy.has_section("train"):
        policy.add_section("train")
    policy.set("train", "gamma", policy.get("train", "gamma", fallback="0.99"))
    if not policy.has_section("sarl"):
        policy.add_section("sarl")
    policy.set("sarl", "epsilon_start", "0.0")

    for section in policy.sections():
        if not env.has_section(section):
            env.add_section(section)
        for key, value in policy.items(section):
            env.set(section, key, value)
    env.set("sim", "human_num", str(human_num))
    env.set("sim", "test_sim", "circle_crossing")
    env.set("occlusion", "enabled", "false")
    env.set("occlusion", "mode", "off")
    return env, policy


def load_policy(args, device):
    _, policy_config = merge_configs(
        Path(args.env_config), Path(args.policy_config), human_num=5
    )
    policy = policy_factory["mamba"](policy_config)
    checkpoint_path = Path(args.checkpoint).resolve()
    actual_hash = file_sha256(checkpoint_path)
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint hash mismatch: {actual_hash}; expected "
            f"{EXPECTED_CHECKPOINT_SHA256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get(
        "policy_state",
        checkpoint.get(
            "model_state_dict",
            checkpoint.get("value_state", checkpoint.get("model", checkpoint)),
        ),
    )
    state_dict = {
        key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }
    incompat = policy.load_state_dict(state_dict, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"checkpoint incompatibility: {incompat}")
    policy.to(device)
    policy.device = device
    policy.set_phase("test")
    policy.set_env_dt(DT)
    policy.use_sarl_predict = True
    policy.eval()
    return policy, actual_hash


def make_env(args, policy, human_num: int):
    env_config, _ = merge_configs(
        Path(args.env_config), Path(args.policy_config), human_num=human_num
    )
    env = CrowdSim()
    env.configure(env_config)
    env.phase = "test"
    robot = Robot(env_config, "robot")
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)
    return env, robot


def ttc_sorted_humans(env, robot):
    ranked = []
    for idx, human in enumerate(env.humans):
        rel_x = human.px - robot.px
        rel_y = human.py - robot.py
        rel_vx = human.vx - robot.vx
        rel_vy = human.vy - robot.vy
        distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6)
        closing = -(rel_x * rel_vx + rel_y * rel_vy) / (distance + 1e-6)
        ttc = distance / (closing + 1e-6) if closing > 0.1 else distance * 10.0
        ranked.append((ttc, idx, human))
    ranked.sort(key=lambda row: row[0])
    return ranked


@torch.no_grad()
def current_hidden(policy, state):
    token = policy._state_to_policy_tokens(state)
    history = list(policy._history) + [token]
    if len(history) < policy.seq_len:
        history = [history[0]] * (policy.seq_len - len(history)) + history
    history = history[-policy.seq_len :]
    tensor = torch.as_tensor(
        np.asarray(history), dtype=torch.float32, device=policy.device
    ).unsqueeze(0)
    spatial = policy.spatial_encoder(tensor)
    temporal = policy.temporal_encoder(spatial)
    hidden = temporal[0, -1]
    direct_value = policy.value_head(hidden).reshape(())
    public_value = policy.forward_value(tensor).reshape(())
    if not torch.allclose(direct_value, public_value, atol=1e-6, rtol=1e-6):
        raise RuntimeError("hidden extraction does not reproduce forward_value")
    return hidden.detach().cpu().numpy().astype(np.float32), token


def snapshot(robot, humans, selected, hidden, token):
    robot_row = np.asarray(
        [
            robot.px,
            robot.py,
            robot.vx,
            robot.vy,
            robot.radius,
            robot.gx,
            robot.gy,
            robot.v_pref,
            robot.theta,
        ],
        dtype=np.float32,
    )
    human_rows = np.asarray(
        [
            [
                human.px,
                human.py,
                human.vx,
                human.vy,
                human.radius,
                human.gx,
                human.gy,
                human.v_pref,
            ]
            for human in humans
        ],
        dtype=np.float32,
    )
    return {
        "robot": robot_row,
        "humans": human_rows,
        "selected": np.asarray(selected, dtype=np.int16),
        "hidden": hidden,
        "token": np.asarray(token, dtype=np.float32),
    }


def infer_outcome(info) -> str:
    text = str(info).lower()
    if isinstance(info, dict):
        text += " " + str(info.get("event", "")).lower()
    if "reach" in text or "success" in text:
        return "success"
    if "collision" in text:
        return "collision"
    return "timeout"


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def collect(args, policy, checkpoint_hash):
    cache_path = Path(args.cache)
    payload = None
    if cache_path.exists() and not args.fresh:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        expected = {
            "checkpoint_sha256": checkpoint_hash,
            "episodes_per_density": args.episodes_per_density,
            "max_steps": args.max_steps,
            "densities": list(args.densities),
            "dt": DT,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"cache metadata mismatch for {key}: "
                    f"{metadata.get(key)!r} != {value!r}; use --fresh"
                )
        episodes = payload["episodes"]
    else:
        episodes = []
        payload = {
            "metadata": {
                "checkpoint_sha256": checkpoint_hash,
                "episodes_per_density": args.episodes_per_density,
                "max_steps": args.max_steps,
                "densities": list(args.densities),
                "dt": DT,
                "collector_schema": 1,
            },
            "episodes": episodes,
        }

    completed = {(ep["density"], ep["episode_index"]) for ep in episodes}
    started = time.time()
    for density in args.densities:
        env, robot = make_env(args, policy, density)
        policy.set_env(env) if hasattr(policy, "set_env") else None
        for episode_index in range(args.episodes_per_density):
            if (density, episode_index) in completed:
                continue
            seed = args.seed_base + episode_index
            seed_everything(seed + density * 10000)
            policy.reset_episode_stats()
            reset_result = env.reset(
                seed=seed,
                options={"test_case": episode_index % max(1, env.case_size["test"])},
            )
            _ = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            steps = []
            done = False
            info = {}
            while not done and len(steps) < args.max_steps:
                ranked = ttc_sorted_humans(env, robot)
                state = JointState(
                    robot.get_full_state(),
                    [human.get_observable_state() for _, _, human in ranked],
                )
                hidden, token = current_hidden(policy, state)
                row = snapshot(
                    robot,
                    env.humans,
                    [idx for _, idx, _ in ranked[:5]],
                    hidden,
                    token,
                )
                action = policy.predict(state, deterministic=True)
                row["action"] = np.asarray([action.vx, action.vy], dtype=np.float32)
                row["action_index"] = int(policy._last_action_index)
                expected_history = min(len(steps) + 1, policy.seq_len)
                if len(policy._history) != expected_history:
                    raise RuntimeError(
                        f"policy history advanced incorrectly: {len(policy._history)} "
                        f"!= {expected_history}"
                    )
                if not 0 <= row["action_index"] < len(policy.action_space):
                    raise RuntimeError("policy returned an invalid action index")
                steps.append(row)
                result = env.step(action)
                if len(result) == 5:
                    _, _, terminated, truncated, info = result
                    done = bool(terminated or truncated)
                else:
                    _, _, done, info = result
            episode = {
                "episode_id": f"n{density}_e{episode_index}",
                "density": density,
                "episode_index": episode_index,
                "seed": seed,
                "outcome": infer_outcome(info),
                "steps": steps,
            }
            episodes.append(episode)
            completed.add((density, episode_index))
            if len(episodes) % 5 == 0:
                atomic_torch_save(payload, cache_path)
            print(
                f"[COLLECT] {len(episodes)}/{len(args.densities) * args.episodes_per_density} "
                f"{episode['episode_id']} steps={len(steps)} outcome={episode['outcome']} "
                f"elapsed={(time.time() - started) / 60:.1f}m",
                flush=True,
            )
    episodes.sort(key=lambda ep: (ep["density"], ep["episode_index"]))
    atomic_torch_save(payload, cache_path)
    return payload


def split_name(episode, episodes_per_density):
    density = episode["density"]
    index = episode["episode_index"]
    if density == 20:
        return "ood"
    train_end = max(1, int(episodes_per_density * 2 / 3))
    dev_end = max(train_end + 1, int(episodes_per_density * 5 / 6))
    if index < train_end:
        return "train"
    if index < dev_end:
        return "dev"
    return "id_test"


def raw_features(step, human_index: int, rank: int, density: int):
    robot = step["robot"]
    human = step["humans"][human_index]
    rel = human[:2] - robot[:2]
    rel_v = human[2:4] - robot[2:4]
    distance = float(np.linalg.norm(rel) + 1e-6)
    closing = float(-np.dot(rel, rel_v) / distance)
    goal = robot[5:7] - robot[:2]
    goal /= float(np.linalg.norm(goal) + 1e-6)
    return np.asarray(
        [
            rel[0],
            rel[1],
            human[2],
            human[3],
            robot[2],
            robot[3],
            distance,
            closing,
            goal[0],
            goal[1],
            np.linalg.norm(human[2:4]),
            rank / 4.0,
        ],
        dtype=np.float32,
    )


def same_goal(step_a, step_b, human_index: int) -> bool:
    return bool(
        np.allclose(
            step_a["humans"][human_index, 5:7],
            step_b["humans"][human_index, 5:7],
            atol=1e-6,
            rtol=0.0,
        )
    )


def build_samples(payload, episodes_per_density):
    samples = []
    for episode in payload["episodes"]:
        steps = episode["steps"]
        split = split_name(episode, episodes_per_density)
        for time_index in range(max(0, len(steps) - max(HORIZONS))):
            current = steps[time_index]
            for rank, human_index in enumerate(current["selected"]):
                human_index = int(human_index)
                if any(
                    not same_goal(current, steps[time_index + horizon], human_index)
                    for horizon in HORIZONS
                ):
                    continue
                current_position = current["humans"][human_index, :2]
                targets = np.stack(
                    [
                        steps[time_index + horizon]["humans"][human_index, :2]
                        - current_position
                        for horizon in HORIZONS
                    ]
                ).astype(np.float32)
                if not np.isfinite(targets).all():
                    continue
                samples.append(
                    {
                        "episode_id": episode["episode_id"],
                        "density": episode["density"],
                        "split": split,
                        "time": time_index,
                        "human": human_index,
                        "raw": raw_features(
                            current, human_index, rank, episode["density"]
                        ),
                        "hidden": current["hidden"],
                        "target": targets,
                        "velocity": current["humans"][human_index, 2:4].copy(),
                    }
                )
    if not samples:
        raise RuntimeError("collector produced no valid future samples")
    return samples


class GaussianProbe(torch.nn.Module):
    def __init__(self, input_dim: int, nonlinear: bool):
        super().__init__()
        if nonlinear:
            self.mean_net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, len(HORIZONS) * 2),
            )
            self.log_variance_net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, len(HORIZONS) * 2),
            )
        else:
            self.mean_net = torch.nn.Linear(input_dim, len(HORIZONS) * 2)
            self.log_variance_net = torch.nn.Linear(
                input_dim, len(HORIZONS) * 2
            )

    def forward(self, inputs):
        mean = self.mean_net(inputs).reshape(-1, len(HORIZONS), 2)
        log_variance = self.log_variance_net(inputs).reshape(
            -1, len(HORIZONS), 2
        )
        return mean, log_variance.clamp(-8.0, 5.0)


def sample_arrays(samples, split, use_hidden):
    selected = [sample for sample in samples if sample["split"] == split]
    raw = np.stack([sample["raw"] for sample in selected])
    if use_hidden:
        hidden = np.stack([sample["hidden"] for sample in selected])
        inputs = np.concatenate([raw, hidden], axis=1)
    else:
        inputs = raw
    targets = np.stack([sample["target"] for sample in selected])
    return selected, inputs.astype(np.float32), targets.astype(np.float32)


def gaussian_metrics(mean, variance, target):
    error = target - mean
    variance = np.maximum(variance, 1e-8)
    d2 = np.sum(error * error / variance, axis=-1)
    nll = 0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + error * error / variance, axis=-1
    )
    result = {
        "n": int(target.shape[0]),
        "ade": float(np.mean(np.linalg.norm(error, axis=-1))),
        "nll": float(np.mean(nll)),
        "coverage50": float(np.mean(d2 <= 1.38629436112)),
        "coverage90": float(np.mean(d2 <= 4.60517018599)),
        "sharpness": float(np.mean(np.sqrt(variance[..., 0] * variance[..., 1]))),
    }
    result["calibration_error"] = abs(result["coverage50"] - 0.5) + abs(
        result["coverage90"] - 0.9
    )
    result["by_horizon"] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        one = gaussian_metrics_single(
            mean[:, horizon_index], variance[:, horizon_index], target[:, horizon_index]
        )
        result["by_horizon"][str(horizon)] = one
    return result


def gaussian_metrics_single(mean, variance, target):
    error = target - mean
    d2 = np.sum(error * error / np.maximum(variance, 1e-8), axis=-1)
    nll = 0.5 * np.sum(
        np.log(2.0 * np.pi * np.maximum(variance, 1e-8))
        + error * error / np.maximum(variance, 1e-8),
        axis=-1,
    )
    return {
        "ade": float(np.mean(np.linalg.norm(error, axis=-1))),
        "nll": float(np.mean(nll)),
        "coverage50": float(np.mean(d2 <= 1.38629436112)),
        "coverage90": float(np.mean(d2 <= 4.60517018599)),
        "sharpness": float(np.mean(np.sqrt(variance[:, 0] * variance[:, 1]))),
    }


def train_probe(samples, use_hidden, nonlinear, device, seed):
    train_rows, train_x, train_y = sample_arrays(samples, "train", use_hidden)
    dev_rows, dev_x, dev_y = sample_arrays(samples, "dev", use_hidden)
    _ = train_rows, dev_rows
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True) + 1e-6
    y_mean = train_y.mean(axis=0, keepdims=True)
    y_std = train_y.std(axis=0, keepdims=True) + 1e-4
    train_x = (train_x - x_mean) / x_std
    dev_x = (dev_x - x_mean) / x_std
    train_y_std = (train_y - y_mean) / y_std
    dev_y_std = (dev_y - y_mean) / y_std

    seed_everything(seed)
    model = GaussianProbe(train_x.shape[1], nonlinear).to(device)
    train_x_t = torch.from_numpy(train_x).to(device)
    train_y_t = torch.from_numpy(train_y_std).to(device)
    dev_x_t = torch.from_numpy(dev_x).to(device)
    dev_y_t = torch.from_numpy(dev_y_std).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Fit the conditional mean first. Joint NLL training can hide a poor mean
    # behind an inflated variance, which is not evidence of decodability.
    optimizer = torch.optim.AdamW(
        model.mean_net.parameters(), lr=2e-3, weight_decay=1e-4
    )
    best_mean_loss = float("inf")
    best_mean_state = None
    stale = 0
    for _epoch in range(120):
        model.train()
        permutation = torch.randperm(len(train_x_t), generator=generator)
        for start in range(0, len(permutation), 2048):
            index = permutation[start : start + 2048].to(device)
            mean = model.mean_net(train_x_t[index]).reshape(
                -1, len(HORIZONS), 2
            )
            loss = torch.mean((train_y_t[index] - mean) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mean_net.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_mean = model.mean_net(dev_x_t).reshape(-1, len(HORIZONS), 2)
            dev_loss = float(torch.mean((dev_y_t - dev_mean) ** 2).cpu())
        if dev_loss < best_mean_loss - 1e-6:
            best_mean_loss = dev_loss
            best_mean_state = copy.deepcopy(model.mean_net.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 15:
                break
    if best_mean_state is None:
        raise RuntimeError("mean probe never produced a finite checkpoint")
    model.mean_net.load_state_dict(best_mean_state)
    for parameter in model.mean_net.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.log_variance_net.parameters(), lr=2e-3, weight_decay=1e-4
    )
    best_loss = float("inf")
    best_state = None
    stale = 0
    for _epoch in range(100):
        model.train()
        permutation = torch.randperm(len(train_x_t), generator=generator)
        for start in range(0, len(permutation), 2048):
            index = permutation[start : start + 2048].to(device)
            mean, log_variance = model(train_x_t[index])
            variance = torch.exp(log_variance)
            error = train_y_t[index] - mean
            loss = 0.5 * (log_variance + error * error / variance).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.log_variance_net.parameters(), 5.0
            )
            optimizer.step()
        model.eval()
        with torch.no_grad():
            mean, log_variance = model(dev_x_t)
            variance = torch.exp(log_variance)
            error = dev_y_t - mean
            dev_loss = float(
                (0.5 * (log_variance + error * error / variance).sum(dim=-1))
                .mean()
                .cpu()
            )
        if dev_loss < best_loss - 1e-5:
            best_loss = dev_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 12:
                break
    if best_state is None:
        raise RuntimeError("probe training never produced a finite checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "best_dev_mean_mse": best_mean_loss,
        "best_dev_loss": best_loss,
    }


@torch.no_grad()
def evaluate_probe(model, stats, samples, split, use_hidden, device):
    _, inputs, target = sample_arrays(samples, split, use_hidden)
    normalized = (inputs - stats["x_mean"]) / stats["x_std"]
    mean_std, log_variance_std = model(torch.from_numpy(normalized).to(device))
    mean_std = mean_std.cpu().numpy()
    variance_std = np.exp(log_variance_std.cpu().numpy())
    mean = stats["y_mean"] + stats["y_std"] * mean_std
    variance = stats["y_std"] ** 2 * variance_std
    return gaussian_metrics(mean, variance, target)


def fit_static_covariances(samples):
    train = [sample for sample in samples if sample["split"] == "train"]
    covariance = []
    for horizon_index, horizon in enumerate(HORIZONS):
        residual = np.stack(
            [
                sample["target"][horizon_index]
                - sample["velocity"] * (horizon * DT)
                for sample in train
            ]
        )
        cov = np.cov(residual.T) + np.eye(2) * 1e-5
        covariance.append(cov)
    return np.stack(covariance)


def posterior_states(payload, covariance_one, rho, a0=3.0, b0=3.0):
    inverse = np.linalg.inv(covariance_one)
    states = {}
    for episode in payload["episodes"]:
        steps = episode["steps"]
        if not steps:
            continue
        human_count = len(steps[0]["humans"])
        shape = np.full(human_count, a0, dtype=np.float64)
        rate = np.full(human_count, b0, dtype=np.float64)
        for time_index, step in enumerate(steps):
            if time_index > 0:
                previous = steps[time_index - 1]
                for human_index in range(human_count):
                    if not same_goal(previous, step, human_index):
                        shape[human_index] = a0
                        rate[human_index] = b0
                        continue
                    residual = (
                        step["humans"][human_index, :2]
                        - previous["humans"][human_index, :2]
                        - previous["humans"][human_index, 2:4] * DT
                    )
                    d2 = float(residual @ inverse @ residual)
                    shape[human_index] = rho * shape[human_index] + (1.0 - rho) * a0 + 1.0
                    rate[human_index] = rho * rate[human_index] + (1.0 - rho) * b0 + 0.5 * d2
            for human_index in range(human_count):
                states[(episode["episode_id"], time_index, human_index)] = (
                    float(shape[human_index]),
                    float(rate[human_index]),
                )
    return states


def student_t_metrics(samples, split, covariance, posterior):
    rows = [sample for sample in samples if sample["split"] == split]
    nll_values = []
    d2_values = []
    coverage50 = []
    coverage90 = []
    sharpness = []
    ade = []
    by_horizon = defaultdict(list)
    for sample in rows:
        shape, rate = posterior[
            (sample["episode_id"], sample["time"], sample["human"])
        ]
        degrees = 2.0 * shape
        for horizon_index, horizon in enumerate(HORIZONS):
            mean = sample["velocity"] * (horizon * DT)
            error = sample["target"][horizon_index] - mean
            scale = covariance[horizon_index] * (rate / shape)
            inverse = np.linalg.inv(scale)
            q = float(error @ inverse @ error)
            log_det = float(np.linalg.slogdet(scale)[1])
            log_pdf = (
                gammaln((degrees + 2.0) / 2.0)
                - gammaln(degrees / 2.0)
                - 0.5 * (2.0 * math.log(degrees * math.pi) + log_det)
                - 0.5 * (degrees + 2.0) * math.log1p(q / degrees)
            )
            threshold50 = 2.0 * f_dist.ppf(0.5, 2, degrees)
            threshold90 = 2.0 * f_dist.ppf(0.9, 2, degrees)
            record = {
                "nll": -float(log_pdf),
                "d2": q,
                "coverage50": float(q <= threshold50),
                "coverage90": float(q <= threshold90),
                "sharpness": float(math.sqrt(max(np.linalg.det(scale), 1e-12))),
                "ade": float(np.linalg.norm(error)),
            }
            for key, value in record.items():
                by_horizon[horizon].append((key, value))
            nll_values.append(record["nll"])
            d2_values.append(q)
            coverage50.append(record["coverage50"])
            coverage90.append(record["coverage90"])
            sharpness.append(record["sharpness"])
            ade.append(record["ade"])
    result = {
        "n": len(rows),
        "ade": float(np.mean(ade)),
        "nll": float(np.mean(nll_values)),
        "coverage50": float(np.mean(coverage50)),
        "coverage90": float(np.mean(coverage90)),
        "sharpness": float(np.mean(sharpness)),
    }
    result["calibration_error"] = abs(result["coverage50"] - 0.5) + abs(
        result["coverage90"] - 0.9
    )
    result["by_horizon"] = {}
    for horizon, values in by_horizon.items():
        grouped = defaultdict(list)
        for key, value in values:
            grouped[key].append(value)
        result["by_horizon"][str(horizon)] = {
            key: float(np.mean(item)) for key, item in grouped.items() if key != "d2"
        }
    return result


def static_cv_metrics(samples, split, covariance):
    rows = [sample for sample in samples if sample["split"] == split]
    target = np.stack([sample["target"] for sample in rows])
    velocity = np.stack([sample["velocity"] for sample in rows])
    mean = np.stack(
        [velocity * (horizon * DT) for horizon in HORIZONS], axis=1
    )
    variance = np.broadcast_to(
        np.diagonal(covariance, axis1=1, axis2=2)[None], mean.shape
    ).copy()
    return gaussian_metrics(mean, variance, target)


def tune_bayes(samples, payload, covariance):
    candidates = (0.80, 0.90, 0.95, 0.98)
    dev_scores = {}
    posteriors = {}
    for rho in candidates:
        posterior = posterior_states(payload, covariance[0], rho)
        posteriors[rho] = posterior
        dev_scores[rho] = student_t_metrics(
            samples, "dev", covariance, posterior
        )["nll"]
    best_rho = min(candidates, key=lambda value: dev_scores[value])
    return best_rho, dev_scores, posteriors[best_rho]


def episode_bootstrap(rows, value_key, seed=2309, repeats=4000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode_id"]].append(float(row[value_key]))
    episode_values = np.asarray(
        [np.mean(values) for values in groups.values()], dtype=np.float64
    )
    if len(episode_values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(episode_values), size=(repeats, len(episode_values)))
    boot = episode_values[indices].mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def action_utility(robot, human_positions, actions, radii):
    horizon_steps = np.arange(1, max(HORIZONS) + 1, dtype=np.float32)
    robot_positions = (
        robot[None, None, :2]
        + actions[:, None, :] * (horizon_steps[None, :, None] * DT)
    )
    delta = robot_positions[:, :, None, :] - human_positions[None, :, :, :]
    clearance = np.linalg.norm(delta, axis=-1) - robot[4] - radii[None, None, :]
    min_clearance = clearance.min(axis=(1, 2))
    collision = (clearance < 0.0).any(axis=(1, 2)).astype(np.float32)
    discomfort = np.maximum(0.2 - clearance.min(axis=2), 0.0).sum(axis=1)
    start_distance = float(np.linalg.norm(robot[5:7] - robot[:2]))
    end_distance = np.linalg.norm(
        robot[5:7][None] - robot_positions[:, -1, :], axis=1
    )
    progress = start_distance - end_distance
    utility = progress - 5.0 * collision - 0.25 * discomfort
    return utility, min_clearance, collision, progress


def sampled_action_utility(robot, samples, actions, radii):
    horizon_steps = np.arange(1, max(HORIZONS) + 1, dtype=np.float32)
    robot_positions = (
        robot[None, None, :2]
        + actions[:, None, :] * (horizon_steps[None, :, None] * DT)
    )
    delta = (
        robot_positions[None, :, :, None, :]
        - samples[:, None, :, :, :]
    )
    clearance = (
        np.linalg.norm(delta, axis=-1)
        - robot[4]
        - radii[None, None, None, :]
    )
    collision_probability = (clearance < 0.0).any(axis=(2, 3)).mean(axis=0)
    discomfort = np.maximum(0.2 - clearance.min(axis=3), 0.0).sum(axis=2).mean(axis=0)
    start_distance = float(np.linalg.norm(robot[5:7] - robot[:2]))
    end_positions = robot_positions[:, -1, :]
    progress = start_distance - np.linalg.norm(
        robot[5:7][None] - end_positions, axis=1
    )
    utility = progress - 5.0 * collision_probability - 0.25 * discomfort
    return utility, collision_probability


def posterior_samples(step, episode_id, time_index, covariance, posterior, rng, count):
    humans = step["humans"]
    output = np.empty(
        (count, max(HORIZONS), len(humans), 2), dtype=np.float32
    )
    for human_index, human in enumerate(humans):
        shape, rate = posterior[(episode_id, time_index, human_index)]
        degrees = 2.0 * shape
        chi_square = rng.chisquare(degrees, size=(count, len(HORIZONS), 1))
        normal = rng.normal(size=(count, len(HORIZONS), 2))
        for horizon_index, horizon in enumerate(HORIZONS):
            scale = covariance[horizon_index] * (rate / shape)
            cholesky = np.linalg.cholesky(scale + np.eye(2) * 1e-9)
            draw = normal[:, horizon_index] @ cholesky.T
            draw *= np.sqrt(degrees / chi_square[:, horizon_index])
            mean = human[:2] + human[2:4] * (horizon * DT)
            position = mean + draw
            previous_horizon = 0 if horizon_index == 0 else HORIZONS[horizon_index - 1]
            for step_index in range(previous_horizon, horizon):
                alpha = (step_index + 1 - previous_horizon) / (horizon - previous_horizon)
                if horizon_index == 0:
                    start = human[:2]
                else:
                    start = output[:, previous_horizon - 1, human_index]
                output[:, step_index, human_index] = (
                    (1.0 - alpha) * start + alpha * position
                )
    return output


def static_gaussian_samples(step, covariance, rng, count):
    humans = step["humans"]
    output = np.empty(
        (count, max(HORIZONS), len(humans), 2), dtype=np.float32
    )
    for human_index, human in enumerate(humans):
        for horizon_index, horizon in enumerate(HORIZONS):
            cholesky = np.linalg.cholesky(
                covariance[horizon_index] + np.eye(2) * 1e-9
            )
            position = (
                human[:2]
                + human[2:4] * (horizon * DT)
                + rng.normal(size=(count, 2)) @ cholesky.T
            )
            previous_horizon = 0 if horizon_index == 0 else HORIZONS[horizon_index - 1]
            for step_index in range(previous_horizon, horizon):
                alpha = (step_index + 1 - previous_horizon) / (horizon - previous_horizon)
                start = (
                    human[:2]
                    if horizon_index == 0
                    else output[:, previous_horizon - 1, human_index]
                )
                output[:, step_index, human_index] = (
                    (1.0 - alpha) * start + alpha * position
                )
    return output


def decision_gate(
    payload,
    policy,
    covariance,
    posterior,
    episodes_per_density,
    decision_seed,
    decision_samples,
):
    if policy.action_space is None:
        policy.build_action_space(1.0)
    actions = np.asarray(
        [[action.vx, action.vy] for action in policy.action_space], dtype=np.float32
    )
    rng = np.random.default_rng(decision_seed)
    rows = []
    for episode in payload["episodes"]:
        split = split_name(episode, episodes_per_density)
        if split not in {"id_test", "ood"}:
            continue
        steps = episode["steps"]
        for time_index in range(0, len(steps) - max(HORIZONS), 2):
            current = steps[time_index]
            if any(
                not same_goal(current, steps[time_index + max(HORIZONS)], human_index)
                for human_index in range(len(current["humans"]))
            ):
                continue
            humans = current["humans"]
            radii = humans[:, 4]
            mean_positions = np.stack(
                [
                    humans[:, :2] + humans[:, 2:4] * (step * DT)
                    for step in range(1, max(HORIZONS) + 1)
                ]
            )
            truth_positions = np.stack(
                [steps[time_index + step]["humans"][:, :2] for step in range(1, max(HORIZONS) + 1)]
            )
            mean_utility, mean_clearance, _, progress = action_utility(
                current["robot"], mean_positions, actions, radii
            )
            truth_utility, truth_clearance, truth_collision, _ = action_utility(
                current["robot"], truth_positions, actions, radii
            )
            draws = posterior_samples(
                current,
                episode["episode_id"],
                time_index,
                covariance,
                posterior,
                rng,
                count=decision_samples,
            )
            full_utility, _ = sampled_action_utility(
                current["robot"], draws, actions, radii
            )
            static_draws = static_gaussian_samples(
                current, covariance, rng, count=decision_samples
            )
            static_utility, _ = sampled_action_utility(
                current["robot"], static_draws, actions, radii
            )
            mean_index = int(np.argmax(mean_utility))
            full_index = int(np.argmax(full_utility))
            static_index = int(np.argmax(static_utility))
            oracle_index = int(np.argmax(truth_utility))
            high_risk = bool(mean_clearance[mean_index] < 0.5)
            rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "split": split,
                    "high_risk": high_risk,
                    "mean_index": mean_index,
                    "full_index": full_index,
                    "static_index": static_index,
                    "oracle_index": oracle_index,
                    "full_action_delta": float(
                        np.linalg.norm(actions[full_index] - actions[mean_index])
                    ),
                    "oracle_action_delta": float(
                        np.linalg.norm(actions[oracle_index] - actions[mean_index])
                    ),
                    "static_action_delta": float(
                        np.linalg.norm(actions[static_index] - actions[mean_index])
                    ),
                    "bayes_static_action_delta": float(
                        np.linalg.norm(actions[full_index] - actions[static_index])
                    ),
                    "full_true_gain": float(
                        truth_utility[full_index] - truth_utility[mean_index]
                    ),
                    "oracle_true_gain": float(
                        truth_utility[oracle_index] - truth_utility[mean_index]
                    ),
                    "static_true_gain": float(
                        truth_utility[static_index] - truth_utility[mean_index]
                    ),
                    "bayes_static_true_gain": float(
                        truth_utility[full_index] - truth_utility[static_index]
                    ),
                    "full_collision_delta": float(
                        truth_collision[full_index] - truth_collision[mean_index]
                    ),
                    "oracle_collision_delta": float(
                        truth_collision[oracle_index] - truth_collision[mean_index]
                    ),
                    "static_collision_delta": float(
                        truth_collision[static_index] - truth_collision[mean_index]
                    ),
                    "bayes_static_collision_delta": float(
                        truth_collision[full_index] - truth_collision[static_index]
                    ),
                    "full_progress_delta": float(
                        progress[full_index] - progress[mean_index]
                    ),
                    "oracle_progress_delta": float(
                        progress[oracle_index] - progress[mean_index]
                    ),
                    "static_progress_delta": float(
                        progress[static_index] - progress[mean_index]
                    ),
                    "bayes_static_progress_delta": float(
                        progress[full_index] - progress[static_index]
                    ),
                    "full_speed_delta": float(
                        np.linalg.norm(actions[full_index])
                        - np.linalg.norm(actions[mean_index])
                    ),
                    "oracle_speed_delta": float(
                        np.linalg.norm(actions[oracle_index])
                        - np.linalg.norm(actions[mean_index])
                    ),
                    "static_speed_delta": float(
                        np.linalg.norm(actions[static_index])
                        - np.linalg.norm(actions[mean_index])
                    ),
                    "mean_true_clearance": float(truth_clearance[mean_index]),
                    "full_true_clearance": float(truth_clearance[full_index]),
                    "oracle_true_clearance": float(truth_clearance[oracle_index]),
                    "static_true_clearance": float(truth_clearance[static_index]),
                }
            )

    def summarize(selected_rows):
        if not selected_rows:
            return {"n": 0}
        result = {"n": len(selected_rows)}
        for prefix in ("static", "full", "oracle"):
            result[f"{prefix}_meaningful_divergence"] = float(
                np.mean([row[f"{prefix}_action_delta"] > 0.1 for row in selected_rows])
            )
            result[f"{prefix}_true_gain"] = float(
                np.mean([row[f"{prefix}_true_gain"] for row in selected_rows])
            )
            result[f"{prefix}_true_gain_ci95"] = episode_bootstrap(
                selected_rows, f"{prefix}_true_gain"
            )
            result[f"{prefix}_collision_delta"] = float(
                np.mean([row[f"{prefix}_collision_delta"] for row in selected_rows])
            )
            result[f"{prefix}_progress_delta"] = float(
                np.mean([row[f"{prefix}_progress_delta"] for row in selected_rows])
            )
            result[f"{prefix}_speed_delta"] = float(
                np.mean([row[f"{prefix}_speed_delta"] for row in selected_rows])
            )
            result[f"{prefix}_clearance_gain"] = float(
                np.mean(
                    [
                        row[f"{prefix}_true_clearance"] - row["mean_true_clearance"]
                        for row in selected_rows
                    ]
                )
            )
        result["bayes_static_meaningful_divergence"] = float(
            np.mean([row["bayes_static_action_delta"] > 0.1 for row in selected_rows])
        )
        for metric in (
            "true_gain", "collision_delta", "progress_delta"
        ):
            key = f"bayes_static_{metric}"
            result[key] = float(np.mean([row[key] for row in selected_rows]))
        result["bayes_static_true_gain_ci95"] = episode_bootstrap(
            selected_rows, "bayes_static_true_gain"
        )
        return result

    return {
        "all": summarize(rows),
        "high_risk": summarize([row for row in rows if row["high_risk"]]),
        "id_test": summarize([row for row in rows if row["split"] == "id_test"]),
        "ood": summarize([row for row in rows if row["split"] == "ood"]),
    }


def analyze(args, payload, policy, device, checkpoint_hash):
    started = time.time()
    samples = build_samples(payload, args.episodes_per_density)
    split_counts = defaultdict(int)
    for sample in samples:
        split_counts[sample["split"]] += 1
    probes = {}
    definitions = (
        ("raw_linear", False, False),
        ("raw_mlp", False, True),
        ("hidden_linear", True, False),
        ("hidden_mlp", True, True),
    )
    for offset, (name, use_hidden, nonlinear) in enumerate(definitions):
        model, stats = train_probe(
            samples, use_hidden, nonlinear, device, args.seed_base + 500 + offset
        )
        probes[name] = {
            split: evaluate_probe(model, stats, samples, split, use_hidden, device)
            for split in ("dev", "id_test", "ood")
        }
        print(
            f"[PROBE] {name} OOD ADE={probes[name]['ood']['ade']:.4f} "
            f"NLL={probes[name]['ood']['nll']:.4f} "
            f"Cov90={probes[name]['ood']['coverage90']:.3f}",
            flush=True,
        )

    covariance = fit_static_covariances(samples)
    best_rho, rho_scores, posterior = tune_bayes(samples, payload, covariance)
    baselines = {
        "static_cv": {
            split: static_cv_metrics(samples, split, covariance)
            for split in ("dev", "id_test", "ood")
        },
        "bayes_student_t": {
            split: student_t_metrics(samples, split, covariance, posterior)
            for split in ("dev", "id_test", "ood")
        },
        "best_rho": best_rho,
        "rho_dev_nll": {str(key): value for key, value in rho_scores.items()},
    }
    decisions = decision_gate(
        payload,
        policy,
        covariance,
        posterior,
        args.episodes_per_density,
        args.decision_seed,
        args.decision_samples,
    )

    # The pre-registered primary question is linear decodability.  Nonlinear
    # probes are diagnostics only: under a held-out density they can extrapolate
    # arbitrarily and must not decide the gate.
    hidden_ood = probes["hidden_linear"]["ood"]
    raw_ood = probes["raw_linear"]["ood"]
    bayes_ood = baselines["bayes_student_t"]["ood"]
    representation_gap = bool(
        bayes_ood["nll"] + 0.05 < hidden_ood["nll"]
        or bayes_ood["calibration_error"] + 0.03 < hidden_ood["calibration_error"]
    )
    hidden_has_temporal_value = bool(
        hidden_ood["nll"] + 0.05 < raw_ood["nll"]
        or hidden_ood["ade"] + 0.01 < raw_ood["ade"]
    )
    high = decisions["high_risk"]
    oracle_ceiling = bool(
        high.get("n", 0) >= 50
        and high.get("oracle_meaningful_divergence", 0.0) >= 0.10
        and high.get("oracle_true_gain_ci95", [-1.0])[0] > 0.0
    )
    distribution_decision_value = bool(
        high.get("n", 0) >= 50
        and high.get("full_meaningful_divergence", 0.0) >= 0.10
        and high.get("full_true_gain_ci95", [-1.0])[0] > 0.0
        and high.get("full_collision_delta", 1.0) <= 0.0
        and high.get("full_progress_delta", -1.0) >= -0.10
        and high.get("full_speed_delta", -1.0) >= -0.20
    )
    bayes_specific_value = bool(
        high.get("n", 0) >= 50
        and high.get("bayes_static_meaningful_divergence", 0.0) >= 0.10
        and high.get("bayes_static_true_gain_ci95", [-1.0])[0] > 0.0
        and high.get("bayes_static_collision_delta", 1.0) <= 0.0
        and high.get("bayes_static_progress_delta", -1.0) >= -0.10
    )
    verdict = {
        "representation_gap_for_explicit_bayes": representation_gap,
        "mamba_hidden_has_temporal_value": hidden_has_temporal_value,
        "full_action_oracle_ceiling": oracle_ceiling,
        "bayesian_distribution_improves_actions": distribution_decision_value,
        "online_bayes_beats_static_distribution": bayes_specific_value,
        "overall_gate": (
            "PASS"
            if representation_gap
            and oracle_ceiling
            and distribution_decision_value
            and bayes_specific_value
            else "STOP"
        ),
        "rule": (
            "PASS requires an uncertainty gap in frozen Mamba plus >=10% meaningful "
            "high-risk action divergence and episode-bootstrap-positive realized utility "
            "for oracle and causal Bayesian choices, no progress collapse, and a "
            "positive Bayesian gain over the same static-distribution interface."
        ),
    }
    result = {
        "metadata": {
            "checkpoint_sha256": checkpoint_hash,
            "episodes": len(payload["episodes"]),
            "split_samples": dict(split_counts),
            "horizons": list(HORIZONS),
            "dt": DT,
            "decision_seed": args.decision_seed,
            "decision_samples": args.decision_samples,
            "elapsed_minutes": (time.time() - started) / 60.0,
        },
        "probes": probes,
        "baselines": baselines,
        "decision_gate": decisions,
        "verdict": verdict,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"[RESULT] {output_path.resolve()}", flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "crowd_nav/runs/mamba_vl/rl_model_ep9000_t24.pth"),
    )
    parser.add_argument(
        "--env-config", default=str(ROOT / "crowd_nav/configs/env.config")
    )
    parser.add_argument(
        "--policy-config", default=str(ROOT / "crowd_nav/configs/policy.config")
    )
    parser.add_argument(
        "--cache",
        default=str(ROOT / "crowd_nav/runs/bayesian_core_gate/rollouts.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "crowd_nav/runs/bayesian_core_gate/result.json"),
    )
    parser.add_argument("--densities", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--episodes-per-density", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=27100)
    parser.add_argument("--decision-seed", type=int, default=4417)
    parser.add_argument("--decision-samples", type=int, default=128)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.collect_only and args.analyze_only:
        raise ValueError("--collect-only and --analyze-only are mutually exclusive")
    if args.episodes_per_density < 12:
        raise ValueError("at least 12 episodes per density are required for episode splits")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed_base)
    policy, checkpoint_hash = load_policy(args, device)
    if args.analyze_only:
        payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    else:
        payload = collect(args, policy, checkpoint_hash)
    if not args.collect_only:
        analyze(args, payload, policy, device, checkpoint_hash)


if __name__ == "__main__":
    main()
