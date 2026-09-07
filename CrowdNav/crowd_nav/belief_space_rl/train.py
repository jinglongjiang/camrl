#!/usr/bin/env python3
"""Train and evaluate the minimal Bayesian-first belief-space RL pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.belief_space_rl.model import (  # noqa: E402
    BeliefSpaceQNetwork,
    ReplayBuffer,
)
from crowd_nav.belief_space_rl.runtime import (  # noqa: E402
    BeliefFeatureEngine,
    PilotNavigationEnvironment,
    build_frozen_mamba,
    merged_policy_config,
    nearest_action_index,
)
from crowd_sim.envs.utils.action import ActionXY  # noqa: E402


SCENARIOS = ("circle_crossing", "square_crossing")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linear_schedule(start: float, end: float, step: int, total: int) -> float:
    fraction = min(max(step / max(total, 1), 0.0), 1.0)
    return float(start + fraction * (end - start))


def outcome_name(raw: str) -> str:
    value = raw.lower()
    if "reach" in value or "success" in value:
        return "success"
    if "collision" in value:
        return "collision"
    return "timeout"


def tensor_features(features, device):
    return (
        torch.as_tensor(
            features.context, dtype=torch.float32, device=device
        ).unsqueeze(0),
        torch.as_tensor(
            features.belief, dtype=torch.float32, device=device
        ).unsqueeze(0),
        torch.as_tensor(
            features.candidates, dtype=torch.float32, device=device
        ).unsqueeze(0),
    )


def select_action(network, features, device, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, len(features.candidates)))
    with torch.no_grad():
        q_values = network(*tensor_features(features, device))
    return int(q_values.argmax(dim=-1).item())


def optimize(
    network,
    target,
    optimizer,
    replay,
    batch_size,
    gamma,
    args,
    rng,
    device,
):
    batch = replay.sample(batch_size, rng, device)
    q_values = network(
        batch["context"],
        batch["belief"],
        batch["candidates"],
    )
    chosen_q = q_values.gather(1, batch["action"][:, None]).squeeze(1)
    with torch.no_grad():
        next_online = network(
            batch["next_context"],
            batch["next_belief"],
            batch["next_candidates"],
        )
        next_action = next_online.argmax(dim=-1)
        next_target = target(
            batch["next_context"],
            batch["next_belief"],
            batch["next_candidates"],
        ).gather(1, next_action[:, None]).squeeze(1)
        td_target = (
            batch["reward"]
            + float(gamma) * (1.0 - batch["done"]) * next_target
        )
    td_loss = F.smooth_l1_loss(chosen_q, td_target)

    teacher_scores = batch["teacher_scores"]
    teacher_valid = teacher_scores > -1000.0
    teacher_count = teacher_valid.sum(dim=-1, keepdim=True).clamp_min(1)
    teacher_mean = (
        teacher_scores.masked_fill(~teacher_valid, 0.0).sum(
            dim=-1,
            keepdim=True,
        )
        / teacher_count
    )
    teacher_variance = (
        (teacher_scores - teacher_mean)
        .square()
        .masked_fill(~teacher_valid, 0.0)
        .sum(dim=-1, keepdim=True)
        / teacher_count
    )
    teacher_scores = (
        (teacher_scores - teacher_mean)
        / teacher_variance.sqrt().clamp_min(1e-4)
    ).masked_fill(~teacher_valid, -20.0)
    teacher_probabilities = F.softmax(
        teacher_scores / float(args.teacher_temperature),
        dim=-1,
    )
    student_log_probabilities = F.log_softmax(
        q_values / float(args.student_temperature),
        dim=-1,
    )
    per_sample_distill = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=-1)

    tail_risk = batch["candidates"][:, :, 1].clamp(0.0, 1.0)
    progress = batch["candidates"][:, :, 6]
    teacher_action = batch["expert_action"]
    teacher_risk = tail_risk.gather(
        1, teacher_action[:, None]
    ).squeeze(1)
    teacher_progress = progress.gather(
        1, teacher_action[:, None]
    ).squeeze(1)
    low_risk_anchor = (
        1.0
        - 0.75
        * (teacher_risk / max(float(args.risk_gate), 1e-6))
        .clamp(0.0, 1.0)
    )
    distill_loss = (
        per_sample_distill * low_risk_anchor
    ).sum() / low_risk_anchor.sum().clamp_min(1.0)
    top_action_loss = (
        F.cross_entropy(
            q_values,
            teacher_action,
            reduction="none",
        )
        * low_risk_anchor
    ).sum() / low_risk_anchor.sum().clamp_min(1.0)

    teacher_q = q_values.gather(
        1, teacher_action[:, None]
    ).squeeze(1)
    safe_mask = (
        (tail_risk <= teacher_risk[:, None] - float(args.risk_gap))
        & (
            progress
            >= teacher_progress[:, None] - float(args.progress_tolerance)
        )
    )
    safe_cost = (
        tail_risk
        + 0.25
        * (teacher_progress[:, None] - progress).clamp_min(0.0)
    ).masked_fill(~safe_mask, float("inf"))
    safe_action = safe_cost.argmin(dim=-1)
    safe_q = q_values.gather(
        1, safe_action[:, None]
    ).squeeze(1)
    risk_active = (
        safe_mask.any(dim=-1)
        & (teacher_risk >= float(args.risk_gate))
    )
    if risk_active.any():
        risk_margin_loss = F.relu(
            float(args.risk_margin) - (safe_q - teacher_q)
        )[risk_active].mean()
    else:
        risk_margin_loss = q_values.sum() * 0.0

    selected_progress = progress.gather(
        1, batch["action"][:, None]
    ).squeeze(1)
    action_ids = torch.arange(
        q_values.shape[1],
        device=q_values.device,
    )[None, :]
    collision_safe_mask = (
        progress
        >= selected_progress[:, None] - float(args.progress_tolerance)
    ) & (action_ids != batch["action"][:, None])
    collision_safe_cost = tail_risk.masked_fill(
        ~collision_safe_mask,
        float("inf"),
    )
    collision_safe_action = collision_safe_cost.argmin(dim=-1)
    collision_safe_q = q_values.gather(
        1, collision_safe_action[:, None]
    ).squeeze(1)
    collision_active = (
        (batch["collision_credit"] > 0.0)
        & collision_safe_mask.any(dim=-1)
    )
    if collision_active.any():
        collision_terms = F.relu(
            float(args.collision_margin)
            - (collision_safe_q - chosen_q)
        )
        collision_weights = batch["collision_credit"]
        collision_loss = (
            collision_terms[collision_active]
            * collision_weights[collision_active]
        ).sum() / collision_weights[collision_active].sum().clamp_min(1e-6)
    else:
        collision_loss = q_values.sum() * 0.0

    loss = (
        float(args.td_weight) * td_loss
        + float(args.distill_weight) * distill_loss
        + float(args.top_action_weight) * top_action_loss
        + float(args.risk_margin_weight) * risk_margin_loss
        + float(args.collision_margin_weight) * collision_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "td_loss": float(td_loss.detach()),
        "distill_loss": float(distill_loss.detach()),
        "top_action_loss": float(top_action_loss.detach()),
        "risk_margin_loss": float(risk_margin_loss.detach()),
        "collision_loss": float(collision_loss.detach()),
        "risk_active_rate": float(risk_active.float().mean().detach()),
        "collision_credit_rate": float(
            collision_active.float().mean().detach()
        ),
        "q_mean": float(chosen_q.detach().mean()),
    }


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, source_param in zip(
            target.parameters(), source.parameters()
        ):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def evaluate(
    network,
    mamba,
    args,
    device,
    profile,
    episodes,
    seed_offset,
):
    network.eval()
    outcomes = {"success": 0, "collision": 0, "timeout": 0}
    episode_outcomes = []
    steps = []
    risk_selected = []
    residual_abs = []
    for episode in range(int(episodes)):
        scenario = SCENARIOS[episode % len(SCENARIOS)]
        environment = PilotNavigationEnvironment(args.env_config, scenario)
        engine = BeliefFeatureEngine(
            mamba,
            args.gdbn_params,
            device,
            particles=args.particles,
            risk_samples=args.risk_samples,
            risk_horizon=args.risk_horizon,
            seed=args.seed + seed_offset + episode,
        )
        state = environment.reset(
            seed=args.seed + seed_offset + episode * 101,
            profile=profile,
            test_case=(seed_offset + episode) % 9000,
        )
        engine.reset()
        features = engine.encode(state)
        done = False
        episode_steps = 0
        final_outcome = "timeout"
        while not done and episode_steps < 102:
            with torch.no_grad():
                q_values, components = network(
                    *tensor_features(features, device),
                    return_components=True,
                )
            action_index = int(q_values.argmax(dim=-1).item())
            risk_selected.append(
                float(features.candidates[action_index, 0])
            )
            residual_abs.append(
                float(
                    components["context_residual"][0, action_index]
                    .abs()
                    .item()
                )
            )
            vx, vy = engine.actions[action_index]
            result = environment.step(ActionXY(float(vx), float(vy)))
            done = result.done
            final_outcome = outcome_name(result.outcome) if done else "timeout"
            episode_steps += 1
            if not done:
                features = engine.encode(result.state_34)
        outcomes[final_outcome] += 1
        episode_outcomes.append(final_outcome)
        steps.append(episode_steps)
    total = max(sum(outcomes.values()), 1)
    return {
        "profile": profile,
        "episodes": total,
        "SR": outcomes["success"] / total,
        "CR": outcomes["collision"] / total,
        "TR": outcomes["timeout"] / total,
        "mean_steps": float(np.mean(steps)),
        "selected_expected_risk": float(np.mean(risk_selected)),
        "mean_abs_context_residual": float(np.mean(residual_abs)),
        "episode_outcomes": episode_outcomes,
    }


def evaluate_mamba_baseline(
    mamba,
    args,
    profile,
    episodes,
    seed_offset,
):
    """Evaluate the frozen value-lookahead policy on the same protocol."""
    mamba.eval()
    mamba.set_phase("test")
    mamba.use_sarl_predict = True
    outcomes = {"success": 0, "collision": 0, "timeout": 0}
    episode_outcomes = []
    steps = []
    for episode in range(int(episodes)):
        scenario = SCENARIOS[episode % len(SCENARIOS)]
        environment = PilotNavigationEnvironment(args.env_config, scenario)
        environment.reset(
            seed=args.seed + seed_offset + episode * 101,
            profile=profile,
            test_case=(seed_offset + episode) % 9000,
        )
        mamba.reset_episode_stats()
        done = False
        episode_steps = 0
        final_outcome = "timeout"
        while not done and episode_steps < 102:
            with torch.no_grad():
                action = mamba.predict(environment.joint_state())
            result = environment.step(action)
            done = result.done
            if done:
                final_outcome = outcome_name(result.outcome)
            episode_steps += 1
        outcomes[final_outcome] += 1
        episode_outcomes.append(final_outcome)
        steps.append(episode_steps)
    total = max(sum(outcomes.values()), 1)
    return {
        "profile": profile,
        "episodes": total,
        "SR": outcomes["success"] / total,
        "CR": outcomes["collision"] / total,
        "TR": outcomes["timeout"] / total,
        "mean_steps": float(np.mean(steps)),
        "episode_outcomes": episode_outcomes,
    }


def save_checkpoint(path, network, target, optimizer, args, metrics):
    torch.save(
        {
            "network": network.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "architecture": {
                "decision_core": "Bayesian belief/action-risk towers",
                "mamba_role": (
                    "frozen history encoder and training-only "
                    "value-lookahead teacher"
                ),
                "algorithm": (
                    "risk-gated lookahead distillation, simulator collision "
                    "credit, and conservative Double-DQN"
                ),
            },
        },
        path,
    )


def write_artifacts(output: Path, metrics_path: Path, baseline: dict):
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_records = [
        record for record in records
        if record.get("type") == "train_episode"
    ]
    evaluations = [
        record for record in records
        if record.get("type") == "evaluation"
    ]
    best = {}
    best_path = output / "best_model.pth"
    if best_path.exists():
        try:
            checkpoint = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(best_path, map_location="cpu")
        best = checkpoint.get("metrics", {})
    summary = {
        "frozen_mamba_baseline": baseline,
        "best_belief_space_policy": best,
        "train_episodes": len(train_records),
        "evaluations": len(evaluations),
        "acceptance_rule": {
            "nominal_SR_drop_max": 0.05,
            "stress_SR_drop_max": 0.05,
            "stress_CR_must_not_increase": True,
            "stress_requires_SR_or_CR_improvement": True,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(10, 7))
        episodes = [record["episode"] for record in train_records]
        rewards = [record["reward"] for record in train_records]
        losses = [record.get("loss", np.nan) for record in train_records]
        axes[0, 0].plot(episodes, rewards, linewidth=1.0)
        axes[0, 0].set_title("Episode reward")
        axes[0, 1].plot(episodes, losses, linewidth=1.0)
        axes[0, 1].set_title("Training loss")
        eval_episode = [record["episode"] for record in evaluations]
        for key, color in (("nominal", "tab:blue"), ("nonstationary", "tab:red")):
            axes[1, 0].plot(
                eval_episode,
                [record[key]["SR"] for record in evaluations],
                marker="o",
                label=key,
                color=color,
            )
            axes[1, 1].plot(
                eval_episode,
                [record[key]["CR"] for record in evaluations],
                marker="o",
                label=key,
                color=color,
            )
        axes[1, 0].set_title("Evaluation success rate")
        axes[1, 1].set_title("Evaluation collision rate")
        for axis in axes.flat:
            axis.grid(alpha=0.25)
            axis.set_xlabel("Episode")
        axes[1, 0].legend()
        axes[1, 1].legend()
        figure.tight_layout()
        figure.savefig(output / "training_curves.png", dpi=160)
        plt.close(figure)
    except (ImportError, RuntimeError) as error:
        (output / "plot_error.txt").write_text(
            f"{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="runs/belief_space_rl_teacher_guided_final_chance",
    )
    parser.add_argument("--policy_config", default="configs/policy.config")
    parser.add_argument("--base_env_config", default="configs/env.config")
    parser.add_argument("--env_config", default="configs/env_gdbn.config")
    parser.add_argument(
        "--base_checkpoint",
        default="runs/mamba_vl/rl_model_ep10000_T24.pth",
    )
    parser.add_argument(
        "--gdbn_params",
        default=(
            "runs/bayesian_belief_pilot_20260730_full/"
            "gate_protocol_aware/selected_gdbn"
        ),
    )
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--demo_episodes", type=int, default=300)
    parser.add_argument("--demo_updates", type=int, default=3000)
    parser.add_argument("--rl_episodes", type=int, default=2400)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--baseline_eval_episodes", type=int, default=30)
    parser.add_argument("--eval_profile", default="decision_stress")
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_capacity", type=int, default=50000)
    parser.add_argument("--warmup_transitions", type=int, default=3000)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--td_weight", type=float, default=1.0)
    parser.add_argument("--distill_weight", type=float, default=0.8)
    parser.add_argument("--top_action_weight", type=float, default=0.4)
    parser.add_argument("--risk_margin_weight", type=float, default=0.6)
    parser.add_argument("--collision_margin_weight", type=float, default=1.2)
    parser.add_argument("--teacher_temperature", type=float, default=0.7)
    parser.add_argument("--student_temperature", type=float, default=1.0)
    parser.add_argument("--risk_gate", type=float, default=0.20)
    parser.add_argument("--risk_gap", type=float, default=0.10)
    parser.add_argument("--risk_margin", type=float, default=0.05)
    parser.add_argument("--collision_margin", type=float, default=0.10)
    parser.add_argument("--progress_tolerance", type=float, default=0.20)
    parser.add_argument("--risk_reward_weight", type=float, default=0.02)
    parser.add_argument("--collision_credit_steps", type=int, default=8)
    parser.add_argument("--particles", type=int, default=50)
    parser.add_argument("--risk_samples", type=int, default=24)
    parser.add_argument("--risk_horizon", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.demo_episodes = 3
        args.demo_updates = 20
        args.rl_episodes = 3
        args.eval_episodes = 2
        args.baseline_eval_episodes = 2
        args.eval_interval = 3
        args.batch_size = 8
        args.replay_capacity = 400
        args.warmup_transitions = 16
        args.risk_samples = 8
        args.particles = 12

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    config = merged_policy_config(
        args.policy_config,
        args.base_env_config,
    )
    mamba = build_frozen_mamba(
        config,
        args.base_checkpoint,
        device,
    )
    probe = BeliefFeatureEngine(
        mamba,
        args.gdbn_params,
        device,
        particles=args.particles,
        risk_samples=args.risk_samples,
        risk_horizon=args.risk_horizon,
        seed=args.seed,
    )
    network = BeliefSpaceQNetwork(
        context_dim=256,
        belief_dim=probe.belief_dim,
        candidate_dim=probe.candidate_dim,
    ).to(device)
    target = deepcopy(network).to(device)
    target.eval()
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )
    replay = ReplayBuffer(
        args.replay_capacity,
        context_dim=256,
        belief_dim=probe.belief_dim,
        action_count=len(probe.actions),
        candidate_dim=probe.candidate_dim,
    )
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "device": str(device),
        "args": vars(args),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "gdbn_files_sha256": {
            name: sha256(str(Path(args.gdbn_params) / name))
            for name in ("gng.npz", "gdbn.npz", "action_model.npz")
        },
        "architecture": (
            "Bayesian-first action-value towers trained by frozen Mamba "
            "lookahead distillation, calibrated action risk, simulator "
            "collision credit, and conservative Double-DQN"
        ),
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics_path = output / "metrics.jsonl"
    start_time = time.time()
    updates = 0
    latest_losses = {}

    print(
        f"[BELIEF-RL] device={device} actions={len(probe.actions)} "
        f"belief_dim={probe.belief_dim} candidate_dim={probe.candidate_dim}",
        flush=True,
    )
    print("[BELIEF-RL] Baseline: frozen Mamba value-lookahead", flush=True)
    baseline = {
        "nominal": evaluate_mamba_baseline(
            mamba,
            args,
            "nominal",
            args.baseline_eval_episodes,
            810000,
        ),
        "nonstationary": evaluate_mamba_baseline(
            mamba,
            args,
            args.eval_profile,
            args.baseline_eval_episodes,
            910000,
        ),
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"type": "frozen_mamba_baseline", **baseline}) + "\n"
        )
    print(
        "[BASELINE] "
        f"nominal={baseline['nominal']['SR']:.1%}/"
        f"{baseline['nominal']['CR']:.1%}/"
        f"{baseline['nominal']['TR']:.1%} "
        f"nonstat={baseline['nonstationary']['SR']:.1%}/"
        f"{baseline['nonstationary']['CR']:.1%}/"
        f"{baseline['nonstationary']['TR']:.1%}",
        flush=True,
    )
    print("[BELIEF-RL] Stage 1/3: demonstration replay warmup", flush=True)
    total_episodes = args.demo_episodes + args.rl_episodes
    for episode in range(total_episodes):
        is_demo_stage = episode < args.demo_episodes
        rl_index = max(episode - args.demo_episodes, 0)
        scenario = SCENARIOS[episode % len(SCENARIOS)]
        profile = (
            "nominal"
            if episode % 2 == 0
            else "train_risk"
        )
        environment = PilotNavigationEnvironment(args.env_config, scenario)
        engine = BeliefFeatureEngine(
            mamba,
            args.gdbn_params,
            device,
            particles=args.particles,
            risk_samples=args.risk_samples,
            risk_horizon=args.risk_horizon,
            seed=args.seed + episode,
        )
        state = environment.reset(
            seed=args.seed + episode * 101,
            profile=profile,
            test_case=(args.seed * 13 + episode) % 9000,
        )
        mamba.reset_episode_stats()
        mamba.set_phase("test")
        mamba.use_sarl_predict = True
        engine.reset()
        features = engine.encode(state)
        done = False
        final_outcome = "timeout"
        reward_sum = 0.0
        steps = 0
        teacher_agreements = 0
        teacher_last_action = None
        transitions = []
        network.train()
        while not done and steps < 102:
            teacher_scores = engine.teacher_scores(state)
            raw_teacher_index = int(np.argmax(teacher_scores))
            raw_teacher_velocity = engine.actions[raw_teacher_index]
            deployed_teacher_action = ActionXY(
                float(raw_teacher_velocity[0]),
                float(raw_teacher_velocity[1]),
            )
            if (
                teacher_last_action is not None
                and float(mamba.test_action_smoothing) > 0.0
            ):
                alpha = float(mamba.test_action_smoothing)
                deployed_teacher_action = ActionXY(
                    alpha * teacher_last_action.vx
                    + (1.0 - alpha) * deployed_teacher_action.vx,
                    alpha * teacher_last_action.vy
                    + (1.0 - alpha) * deployed_teacher_action.vy,
                )
            teacher_last_action = deployed_teacher_action
            deployed_teacher_index = nearest_action_index(
                deployed_teacher_action,
                engine.actions,
            )
            teacher_agreements += int(
                raw_teacher_index == deployed_teacher_index
            )
            valid_teacher_scores = teacher_scores[teacher_scores > -1000.0]
            teacher_scale = (
                float(valid_teacher_scores.std())
                if len(valid_teacher_scores) > 1
                else 0.01
            )
            teacher_scores[deployed_teacher_index] = (
                float(valid_teacher_scores.max())
                + max(teacher_scale, 0.01)
            )
            expert_index = deployed_teacher_index
            epsilon = linear_schedule(
                0.15, 0.02, rl_index, args.rl_episodes
            )
            teacher_mix = linear_schedule(
                0.70, 0.10, rl_index, args.rl_episodes
            )
            if is_demo_stage or rng.random() < teacher_mix:
                action_index = expert_index
                demo = True
            else:
                action_index = select_action(
                    network, features, device, epsilon, rng
                )
                demo = False
            vx, vy = engine.actions[action_index]
            result = environment.step(ActionXY(float(vx), float(vy)))
            next_features = engine.encode(result.state_34)
            selected_tail_risk = float(
                features.candidates[action_index, 1]
            )
            shaped_reward = (
                float(result.reward)
                - float(args.risk_reward_weight) * selected_tail_risk
            )
            transitions.append(
                {
                    "state": features,
                    "action": action_index,
                    "expert_action": expert_index,
                    "teacher_scores": teacher_scores,
                    "reward": shaped_reward,
                    "next_state": next_features,
                    "done": result.done,
                    "demo": demo,
                }
            )
            state = result.state_34
            features = next_features
            done = result.done
            reward_sum += result.reward
            steps += 1
            if done:
                final_outcome = outcome_name(result.outcome)

        for transition_index, transition in enumerate(transitions):
            collision_credit = 0.0
            if final_outcome == "collision":
                distance_to_collision = (
                    len(transitions) - 1 - transition_index
                )
                if distance_to_collision < args.collision_credit_steps:
                    collision_credit = 1.0 - (
                        distance_to_collision
                        / max(float(args.collision_credit_steps), 1.0)
                    )
            replay.add(
                transition["state"],
                transition["action"],
                transition["expert_action"],
                transition["teacher_scores"],
                transition["reward"],
                transition["next_state"],
                transition["done"],
                transition["demo"],
                collision_credit=collision_credit,
            )

        if (
            not is_demo_stage
            and replay.size >= args.warmup_transitions
        ):
            for _ in range(
                max(1, len(transitions) * args.gradient_steps)
            ):
                latest_losses = optimize(
                    network,
                    target,
                    optimizer,
                    replay,
                    args.batch_size,
                    args.gamma,
                    args,
                    rng,
                    device,
                )
                soft_update(target, network, args.tau)
                updates += 1

        record = {
            "type": "train_episode",
            "episode": episode + 1,
            "stage": "demo" if is_demo_stage else "rl",
            "profile": profile,
            "scenario": scenario,
            "outcome": final_outcome,
            "reward": reward_sum,
            "steps": steps,
            "teacher_raw_vs_smoothed_agreement": (
                teacher_agreements / max(steps, 1)
            ),
            "collision_credited_transitions": (
                min(len(transitions), args.collision_credit_steps)
                if final_outcome == "collision"
                else 0
            ),
            "replay_size": replay.size,
            "updates": updates,
            **latest_losses,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"[TRAIN] {episode + 1}/{total_episodes} "
                f"stage={record['stage']} outcome={final_outcome} "
                f"reward={reward_sum:.3f} replay={replay.size} "
                f"loss={latest_losses.get('loss', float('nan')):.4f}",
                flush=True,
            )

        if episode + 1 == args.demo_episodes:
            print(
                f"[BELIEF-RL] Stage 2/3: {args.demo_updates} "
                "demonstration-regularized Double-DQN updates",
                flush=True,
            )
            network.train()
            for update_index in range(args.demo_updates):
                latest_losses = optimize(
                    network,
                    target,
                    optimizer,
                    replay,
                    min(args.batch_size, replay.size),
                    args.gamma,
                    args,
                    rng,
                    device,
                )
                soft_update(target, network, args.tau)
                updates += 1
                if (
                    update_index + 1
                ) % max(1, args.demo_updates // 4) == 0:
                    print(
                        f"[DEMO-UPDATE] {update_index + 1}/"
                        f"{args.demo_updates} loss="
                        f"{latest_losses['loss']:.4f}",
                        flush=True,
                    )
            print(
                "[BELIEF-RL] Stage 3/3: online Double-DQN refinement",
                flush=True,
            )

        should_evaluate = (
            episode + 1 == args.demo_episodes
            or (
                not is_demo_stage
                and (
                    rl_index + 1
                ) % args.eval_interval == 0
            )
            or episode + 1 == total_episodes
        )
        if should_evaluate:
            nominal = evaluate(
                network,
                mamba,
                args,
                device,
                "nominal",
                args.eval_episodes,
                810000,
            )
            nonstationary = evaluate(
                network,
                mamba,
                args,
                device,
                args.eval_profile,
                args.eval_episodes,
                910000,
            )
            evaluation = {
                "type": "evaluation",
                "episode": episode + 1,
                "nominal": nominal,
                "nonstationary": nonstationary,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(evaluation) + "\n")
            score = (
                0.5 * (nominal["SR"] + nonstationary["SR"])
                - (nominal["CR"] + nonstationary["CR"])
                - 0.25 * (nominal["TR"] + nonstationary["TR"])
            )
            eligible = (
                nominal["SR"] >= baseline["nominal"]["SR"] - 0.05
                and nonstationary["SR"]
                >= baseline["nonstationary"]["SR"] - 0.05
                and nonstationary["CR"]
                <= baseline["nonstationary"]["CR"]
                and (
                    nonstationary["SR"]
                    > baseline["nonstationary"]["SR"]
                    or nonstationary["CR"]
                    < baseline["nonstationary"]["CR"]
                )
            )
            print(
                f"[EVAL] episode={episode + 1} score={score:.3f} "
                f"eligible={eligible} "
                f"nominal={nominal['SR']:.1%}/{nominal['CR']:.1%}/"
                f"{nominal['TR']:.1%} nonstat={nonstationary['SR']:.1%}/"
                f"{nonstationary['CR']:.1%}/{nonstationary['TR']:.1%}",
                flush=True,
            )
            best_path = output / "best_model.pth"
            previous_score = -float("inf")
            if best_path.exists():
                try:
                    previous = torch.load(
                        best_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    previous_score = float(
                        previous.get("metrics", {}).get("score", -float("inf"))
                    )
                except (TypeError, RuntimeError, EOFError):
                    previous_score = -float("inf")
            if score > previous_score:
                save_checkpoint(
                    best_path,
                    network,
                    target,
                    optimizer,
                    args,
                    {
                        "score": score,
                        "eligible": eligible,
                        "episode": episode + 1,
                        "nominal": nominal,
                        "nonstationary": nonstationary,
                    },
                )
            if eligible:
                eligible_path = output / "best_eligible_model.pth"
                eligible_score = -float("inf")
                if eligible_path.exists():
                    try:
                        previous = torch.load(
                            eligible_path,
                            map_location="cpu",
                            weights_only=False,
                        )
                        eligible_score = float(
                            previous.get("metrics", {}).get(
                                "score",
                                -float("inf"),
                            )
                        )
                    except (TypeError, RuntimeError, EOFError):
                        eligible_score = -float("inf")
                if score > eligible_score:
                    save_checkpoint(
                        eligible_path,
                        network,
                        target,
                        optimizer,
                        args,
                        {
                            "score": score,
                            "eligible": True,
                            "episode": episode + 1,
                            "nominal": nominal,
                            "nonstationary": nonstationary,
                        },
                    )

    final_metrics = {
        "elapsed_seconds": time.time() - start_time,
        "episodes": total_episodes,
        "updates": updates,
        "replay_size": replay.size,
    }
    save_checkpoint(
        output / "final_model.pth",
        network,
        target,
        optimizer,
        args,
        final_metrics,
    )
    write_artifacts(output, metrics_path, baseline)
    manifest.update(
        {
            "status": "completed",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_mamba_baseline": baseline,
            **final_metrics,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[BELIEF-RL] complete: {output}", flush=True)


if __name__ == "__main__":
    main()
