#!/usr/bin/env python3
"""BDVL IL + online distributional-RL trainer.

The trainer is deliberately the only place that assembles live episodes
into typed replay transitions.  It uses the frozen BDVL environment config,
keeps real action/belief identities, builds mathematically correct n-step
targets, and stores a complete restartable checkpoint.
"""

from __future__ import annotations

import argparse
import configparser
import copy
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: E402
from crowd_sim.envs.utils.action import ActionXY  # noqa: E402
from crowd_sim.envs.utils.robot import Robot  # noqa: E402
from crowd_sim.envs.policy.orca import ORCA  # noqa: E402

from crowd_nav.bayesian_dvl.config import ActionGridSpec, BDVL_PRODUCTION_SOURCES, FROZEN_VALUES, R4_2_REPLAY_CONTRACT_COMPLETE, TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC, load_and_validate_registry, derive_return_bounds, FEATURE_SCHEMA_V4, REWARD_SCHEMA_V2 as REWARD_SCHEMA  # noqa: E402
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact  # noqa: E402
from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, HUMAN_FEATURE_DIM, SetEncoder  # noqa: E402
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork  # noqa: E402
from crowd_nav.bayesian_dvl.transition import RewardConfig  # noqa: E402
from crowd_nav.bayesian_dvl.policy import BDVLPolicy, BayesianDVLPolicy, remaining_time_fraction, save_composed_checkpoint  # noqa: E402
from crowd_nav.bayesian_dvl.replay import DemoOnlineReplay, MCReturnSample  # noqa: E402
from crowd_nav.bayesian_dvl.trainer import build_mc_return_samples, compute_mc_returns  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest, sha256_of_file  # noqa: E402
from crowd_nav.bayesian_pilot.protocol import BehaviorScheduler, InterventionORCA, PROFILES  # noqa: E402

# R2-3 fix (2026-08-07): this CLI is the R2 (Monte Carlo return) trainer.
# The v1 3-step bootstrapped-TD path (trainer.build_n_step_transitions/
# train_step, target networks, soft_update_target) is kept ONLY for
# selftest.py's historical regression coverage and is deliberately never
# imported here -- guide.md R2-3: "旧train_step()和build_n_step_transitions()
# 仅保留为v1历史回归，不得由R2 CLI调用".


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def _sha256_obj(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_training_values(path: Path) -> Dict[str, object]:
    parser = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not parser.read(str(path)) or not parser.has_section("training"):
        raise SystemExit(f"training config missing [training]: {path}")
    train = parser["training"]
    iqn = parser["iqn"]
    values: Dict[str, object] = {
        "gamma": train.getfloat("gamma"),
        "replay_capacity": train.getint("replay_capacity"),
        "demo_sample_ratio": train.getfloat("demo_sample_ratio"),
        "batch_size": train.getint("batch_size"),
        "learning_rate": train.getfloat("learning_rate"),
        "epsilon_start": train.getfloat("epsilon_start"),
        "epsilon_end": train.getfloat("epsilon_end"),
        "updates_per_episode": train.getint("updates_per_episode"),
        "il_episodes": train.getint("il_episodes"),
        "iqn_train_quantiles": iqn.getint("iqn_train_quantiles"),
    }
    expected = {
        "gamma": FROZEN_VALUES["gamma"], "replay_capacity": FROZEN_VALUES["replay_capacity"],
        "demo_sample_ratio": FROZEN_VALUES["demo_sample_ratio"], "batch_size": FROZEN_VALUES["batch_size"],
        "learning_rate": FROZEN_VALUES["learning_rate"], "iqn_train_quantiles": FROZEN_VALUES["iqn_train_quantiles"],
        "epsilon_start": FROZEN_VALUES["epsilon_start"], "epsilon_end": FROZEN_VALUES["epsilon_end"],
        "updates_per_episode": FROZEN_VALUES["updates_per_episode"], "il_episodes": FROZEN_VALUES["il_episodes"],
    }
    for key, expected_value in expected.items():
        if values[key] != expected_value:
            raise SystemExit(f"training config drift for {key}: {values[key]!r} != frozen {expected_value!r}")
    return values


def _validate_and_record_gradient_diagnostic(
    result, stage: str, episode: int, gradient_diagnostics=None, monitor_state=None
) -> None:
    """Validate ranking/MC gradient diagnostics without rejecting inactive ranking terms.

    A ranking loss can be positive while its current batch has zero gradient
    (for example, the active margin constraint has already been satisfied or
    the differentiable score path is locally flat).  Zero is therefore a
    valid *inactive* diagnostic, not evidence of numerical failure.  A
    non-zero ranking gradient must still stay inside the frozen audit range.
    """
    fields = {
        "mc_grad_norm": float(result.mc_grad_norm),
        "rank_grad_norm": float(result.rank_grad_norm),
        "weighted_rank_grad_norm": float(result.weighted_rank_grad_norm),
        "gradient_ratio": float(result.gradient_ratio),
    }
    if not all(np.isfinite(value) for value in fields.values()):
        raise RuntimeError(f"{stage} gradient diagnostic is non-finite: {fields}")

    activity_epsilon = 1e-12
    ranking_active = (
        fields["rank_grad_norm"] > activity_epsilon
        or fields["weighted_rank_grad_norm"] > activity_epsilon
    )
    in_frozen_range = True
    outside_streak = 0
    if ranking_active:
        lower = float(FROZEN_VALUES["rank_gradient_ratio_min"])
        upper = float(FROZEN_VALUES["rank_gradient_ratio_max"])
        in_frozen_range = lower <= fields["gradient_ratio"] <= upper
        if monitor_state is not None:
            if in_frozen_range:
                monitor_state["outside_streak"] = 0
            else:
                monitor_state["outside_streak"] = int(monitor_state.get("outside_streak", 0)) + 1
                sustained = int(FROZEN_VALUES["gradient_ratio_sustained_updates"])
                if monitor_state["outside_streak"] >= sustained:
                    raise RuntimeError(
                        f"{stage} ranking/MC gradient ratio persistently outside frozen range: "
                        f"{fields['gradient_ratio']} for {sustained} consecutive active updates "
                        f"(expected [{lower}, {upper}])"
                    )
            outside_streak = int(monitor_state.get("outside_streak", 0))
        elif not in_frozen_range:
            # Direct callers without a monitor are diagnostic-only.  Runtime
            # training always supplies monitor_state, so this path remains
            # useful for unit tests without silently weakening the live gate.
            outside_streak = 1
    elif monitor_state is not None:
        monitor_state["outside_streak"] = 0

    if gradient_diagnostics is not None:
        gradient_diagnostics.append({
            "stage": stage,
            "episode": int(episode),
            **fields,
            "ranking_active": bool(ranking_active),
            "ratio_in_frozen_range": bool(in_frozen_range),
            "outside_streak": outside_streak,
        })


def _make_env(env_config_path: Path, human_num: int):
    if env_config_path.name != "env_bayesian_dvl.config":
        raise ValueError("BDVL training must use env_bayesian_dvl.config")
    env_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not env_config.read(str(env_config_path)):
        raise SystemExit(f"env config not found: {env_config_path}")
    env_config.set("sim", "human_num", str(human_num))
    env_config.set("robot", "policy", "orca")
    env = CrowdSim()
    env.configure(env_config)
    env.phase = "train"
    return env, env_config


def _make_reward_config() -> RewardConfig:
    return RewardConfig(
        success_reward=FROZEN_VALUES["success_reward"], collision_penalty=FROZEN_VALUES["collision_penalty"],
        timeout_penalty=FROZEN_VALUES["timeout_penalty"], progress_reward=FROZEN_VALUES["progress_reward"],
        time_penalty=FROZEN_VALUES["time_penalty"], stand_penalty=FROZEN_VALUES["stand_penalty"],
        stand_speed_threshold=FROZEN_VALUES["stand_speed_threshold"],
        discomfort_distance=FROZEN_VALUES["discomfort_distance"],
        discomfort_penalty_factor=FROZEN_VALUES["discomfort_penalty_factor"],
    )


def _install_human_profile(env, env_config, profile: str, seed: int) -> BehaviorScheduler:
    if profile not in PROFILES:
        raise ValueError(f"unknown training profile {profile!r}")
    for human in env.humans:
        policy = InterventionORCA(env_config)
        policy.time_step = env.time_step
        human.set_policy(policy)
    scheduler = BehaviorScheduler(PROFILES[profile], seed=seed)
    scheduler.reset(len(env.humans))
    return scheduler


def _capture_state(tracker, robot, humans, global_time: float, action_index: int, action_table, posterior_seed_key) -> dict:
    """R4-2 fix (2026-08-10, guide.md "R4-1R-4"): previously flattened
    into robot_features/human_features/human_mask arrays immediately
    (a lossy, pre-encoded snapshot) -- Q(s,b,a)'s MC-loss target must be
    built from the SAME candidate builder deployment uses
    (``policy._vectorized_candidate_batch``), which needs the RAW
    ``robot``/``humans``/belief tracker snapshot, not a pre-flattened
    array. Keeping the raw pieces here means ``build_mc_return_samples``
    never has to (and cannot) reconstruct what was thrown away.

    R4-2R-2 fix (2026-08-10, guide.md "R4-2R-2 -- 保存真实posterior
    seed"): ``posterior_seed_key`` is now a REQUIRED caller-supplied
    value (the real ``BDVLPolicy.last_seed_key`` for a greedy step, or a
    fresh ``next_seed_key()`` tick for an exploration step -- see
    ``_collect_online_episode``), never reconstructed here or later from
    ``step_index``.
    """
    import copy as _copy
    from crowd_nav.bayesian_dvl.policy import compute_action_features_array
    action_table_array = np.asarray(action_table, dtype=np.float64)
    return {
        "robot": robot,
        "humans": tuple(humans),
        "belief_tracker_snapshot": _copy.deepcopy(tracker._tracks),
        "global_time": float(global_time),
        "executed_action_features": compute_action_features_array(robot, action_table_array)[action_index],
        "posterior_seed_key": tuple(int(x) for x in posterior_seed_key),
    }


def _episode_setup(env_config_path: Path, human_num: int, episode_seed: int, profile: str):
    env, env_config = _make_env(env_config_path, human_num)
    placeholder = Robot(env_config, "robot")
    placeholder_orca = ORCA()
    placeholder_orca.configure(env_config)
    placeholder.set_policy(placeholder_orca)
    placeholder.visible = True
    placeholder.env = env
    env.set_robot(placeholder)
    env.case_counter["train"] = int(episode_seed) % (2**32 - 1)
    env.reset()
    scheduler = _install_human_profile(env, env_config, profile, episode_seed)
    return env, env_config, scheduler


def _max_environment_steps() -> int:
    """Return the exact number of ``env.step`` calls to reach timeout.

    CrowdSim checks timeout at the beginning of ``step``.  With a 35 s
    horizon and 0.25 s timestep, the timeout event is therefore returned by
    call 141, not call 140.  Training and evaluation must share this contract
    so MC targets contain the real terminal event rather than a fabricated
    replacement reward.
    """
    return int(round(float(FROZEN_VALUES["time_limit"]) / float(FROZEN_VALUES["dt"]))) + 1


def _collect_orca_episode(env_config_path: Path, action_table, artifact, episode_seed: int, profile: str, action_tolerance: float, max_steps: int | None = None):
    """Real ORCA demonstration collection for R3-3's ranking-IL Stage 1.

    Unlike the R2 version, this captures what's needed to reconstruct
    the exact deployment-time scoring batch for every step: the raw
    (unquantized) ORCA action (for the expert-equivalence class), and a
    deep-copied belief-tracker snapshot (for ``RankingDemoSample``).
    """
    from crowd_nav.bayesian_dvl.belief import BeliefTracker
    from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
    from crowd_nav.bayesian_dvl.policy import compute_action_features_array
    from crowd_nav.bayesian_dvl.ranking import RankingDemoSample, build_action_equivalence_class, nearest_action_index

    env, env_config, scheduler = _episode_setup(env_config_path, 5, episode_seed, profile)
    robot = env.robot
    orca = ORCA()
    orca.configure(env_config)
    robot.set_policy(orca)
    robot.visible = True
    robot.env = env
    env.set_robot(robot)
    tracker = BeliefTracker(artifact)
    artifact_hash = artifact.content_sha256()
    action_table_array = np.asarray(action_table, dtype=np.float64)

    step_records = []  # (robot_obs, humans, global_time, tracker_snapshot, expert_indices, executed_index, executed_feats)
    rewards: List[float] = []
    global_time = 0.0
    event = None
    for step_index in range(max_steps if max_steps is not None else _max_environment_steps()):
        humans = [HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius)) for i, h in enumerate(env.humans)]
        robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
        tracker.update({h.track_id: (global_time, np.array([h.px, h.py])) for h in humans})
        tracker_snapshot = copy.deepcopy(tracker._tracks)
        action = env.robot.act([h.get_observable_state() for h in env.humans])
        # guide.md R4-2: expert_indices (ranking, an equivalence CLASS) and
        # executed_index (MC-loss, the SINGLE nearest grid action to what
        # ORCA really did) are deliberately computed independently -- never
        # derive one from the other.
        expert_indices = build_action_equivalence_class(float(action.vx), float(action.vy), action_table, action_tolerance)
        executed_index = nearest_action_index(float(action.vx), float(action.vy), action_table)
        executed_feats = compute_action_features_array(robot_obs, action_table_array)[executed_index]
        step_records.append((robot_obs, tuple(humans), global_time, tracker_snapshot, expert_indices, executed_index, executed_feats))
        if profile != "nominal":
            scheduler.advance([h.policy for h in env.humans])
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        done = bool(terminated or truncated)
        global_time += FROZEN_VALUES["dt"]
        if done:
            event = info.get("event") if isinstance(info, dict) else None
            break
    if not rewards:
        raise RuntimeError("ORCA episode produced no transitions")
    if event not in {"reach_goal", "collision", "timeout"}:
        raise RuntimeError(
            "ORCA episode did not return a valid CrowdSim terminal event "
            f"after {_max_environment_steps()} steps"
        )
    outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}[event]
    returns = compute_mc_returns(rewards, float(FROZEN_VALUES["gamma"]))

    samples = [
        RankingDemoSample(
            robot=robot_obs, humans=humans, global_time=global_time_t,
            belief_tracker_snapshot=tracker_snapshot, expert_action_indices=expert_indices,
            executed_action_index=executed_index, executed_action_features=executed_feats,
            # IL has no live decision_counter (ORCA never calls decide());
            # step_index+1 plays the same role, matching the identical
            # convention stage1_train_step/stage2_train_step already use
            # when building seed_key for the ranking pass over this same
            # sample (see their "IL has no live decision_counter" comment).
            posterior_seed_key=(0, episode_seed, t + 1),
            target_return=returns[t], artifact_sha256=artifact_hash,
            episode_seed=episode_seed, step_index=t, outcome=outcome, source_role="demo",
        )
        for t, (robot_obs, humans, global_time_t, tracker_snapshot, expert_indices, executed_index, executed_feats) in enumerate(step_records)
    ]
    return samples, outcome


def stage1_il_pretrain(encoder, value_network, action_encoder, artifact, action_table, env_config_path, replay, n_episodes, gamma, optimizer, device, profiles, episode_seed_base, batch_size, gradient_diagnostics=None):
    from crowd_nav.bayesian_dvl.ranking import derive_action_equivalence_tolerance
    from crowd_nav.bayesian_dvl.trainer import stage1_train_step

    action_tolerance = derive_action_equivalence_tolerance(action_table)
    ranking_margin = float(FROZEN_VALUES["ranking_margin"])
    lambda_rank = float(FROZEN_VALUES["lambda_rank"])
    losses, mc_losses, rank_losses = [], [], []
    gradient_monitor = {"outside_streak": 0}
    outcome_counts = {"success": 0, "collision": 0, "timeout": 0}
    reward_config = _make_reward_config()
    for episode_index in range(n_episodes):
        episode_seed = int(episode_seed_base + episode_index)
        profile = profiles[episode_index % len(profiles)]
        samples, outcome = _collect_orca_episode(env_config_path, action_table, artifact, episode_seed, profile, action_tolerance)
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        for sample in samples:
            replay.demo.add(sample)
        batch, _, _ = replay.sample_batch(batch_size, random.Random(episode_seed))
        result = stage1_train_step(
            encoder, value_network, action_encoder, artifact, action_table, reward_config,
            dt=float(FROZEN_VALUES["dt"]), time_limit=float(FROZEN_VALUES["time_limit"]),
            max_human_speed=float(FROZEN_VALUES["max_human_speed"]),
            n_world_samples=int(FROZEN_VALUES["world_samples_train"]),
            n_iqn_quantiles=int(FROZEN_VALUES["iqn_train_quantiles"]),
            posterior_source="full", device=device, gamma=gamma,
            demo_batch=batch, ranking_margin=ranking_margin, lambda_rank=lambda_rank,
            n_train_quantiles=int(FROZEN_VALUES["iqn_train_quantiles"]), optimizer=optimizer,
            ranking_batch_size=int(FROZEN_VALUES["ranking_batch_size"]),
        )
        if result.aborted:
            raise RuntimeError(f"IL update aborted at episode {episode_index}: {result.abort_reason}")
        _validate_and_record_gradient_diagnostic(
            result, "il", episode_index + 1, gradient_diagnostics, gradient_monitor
        )
        losses.append(float(result.loss))
        mc_losses.append(float(result.mc_loss))
        rank_losses.append(float(result.rank_loss))
        if (episode_index + 1) % 500 == 0 or episode_index == n_episodes - 1:
            print(f"IL_PROGRESS episode={episode_index + 1} outcomes={outcome_counts} "
                  f"mean_loss={np.mean(losses[-500:]):.4f} mean_mc={np.mean(mc_losses[-500:]):.4f} mean_rank={np.mean(rank_losses[-500:]):.4f}")
    print(f"IL_STAGE_DONE n_episodes={n_episodes} outcomes={outcome_counts}")
    return losses


def _collect_online_episode(env_config_path, artifact, action_table, seed, episode_index, profile, encoder, value_network, action_encoder, epsilon):
    from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
    episode_seed = seed * 100000 + episode_index
    env, env_config, scheduler = _episode_setup(env_config_path, 5, episode_seed, profile)
    bdvl = BDVLPolicy(
        artifact=artifact, set_encoder=encoder, value_network=value_network, action_encoder=action_encoder, action_table=action_table,
        reward_config=_make_reward_config(), dt=FROZEN_VALUES["dt"], time_limit=FROZEN_VALUES["time_limit"],
        max_human_speed=FROZEN_VALUES["max_human_speed"], cvar_alpha=FROZEN_VALUES["cvar_alpha"],
        n_world_samples=FROZEN_VALUES["world_samples_train"], n_iqn_quantiles=FROZEN_VALUES["iqn_train_quantiles"],
        risk_neutral=True,
    )
    adapter = BayesianDVLPolicy(bdvl_policy=bdvl, suite_seed=seed)
    adapter.time_step = FROZEN_VALUES["dt"]
    adapter.reset_episode_stats(
        suite_seed=seed,
        episode_seed=episode_seed,
    )
    robot = env.robot
    robot.set_policy(adapter)
    robot.visible = True
    robot.time_step = FROZEN_VALUES["dt"]
    robot.env = env
    env.set_robot(robot)
    states, actions, rewards = [], [], []
    event = None
    rng = np.random.default_rng(seed * 1000003 + episode_index)
    for _ in range(_max_environment_steps()):
        humans = [HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius)) for i, h in enumerate(env.humans)]
        robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
        if profile != "nominal":
            scheduler.advance([h.policy for h in env.humans])
        # Read the adapter's own clock BEFORE either branch advances it, so
        # the captured remaining_fraction matches the time the decision
        # (exploratory or greedy) was actually conditioned on.
        decision_time = adapter._global_time
        if float(rng.random()) < epsilon:
            action_index = int(rng.integers(len(action_table)))
            action = ActionXY(*action_table[action_index])
            adapter.bdvl_policy.belief_tracker.update({h.track_id: (decision_time, np.array([h.px, h.py])) for h in humans})
            adapter._global_time += FROZEN_VALUES["dt"]
            # R4-2R-2 fix (2026-08-10, guide.md "R4-2R-2"): an exploration
            # step never calls decide() (no candidate scoring happens),
            # but it must still consume exactly one decision-counter tick
            # AFTER the tracker update above, so every real environment
            # step -- greedy or exploratory -- gets its own unique,
            # reproducible seed_key and none are silently skipped/reused.
            posterior_seed_key = adapter.bdvl_policy.next_seed_key(seed, episode_seed)
        else:
            action = env.robot.act([h.get_observable_state() for h in env.humans])
            action_index = int(adapter.last_action_index)
            # The real seed_key decide() just used for THIS step's actual
            # scoring -- not reconstructed from step_index afterward.
            posterior_seed_key = adapter.bdvl_policy.last_seed_key
        states.append(_capture_state(adapter.bdvl_policy.belief_tracker, robot_obs, humans, decision_time, action_index, action_table, posterior_seed_key))
        _, reward, terminated, truncated, info = env.step(action)
        actions.append(action_index)
        rewards.append(float(reward))
        done = bool(terminated or truncated)
        if done:
            event = info.get("event") if isinstance(info, dict) else None
            break
    if not rewards:
        raise RuntimeError("online episode produced no transitions")
    if event not in {"reach_goal", "collision", "timeout"}:
        raise RuntimeError(
            "online episode did not return a valid CrowdSim terminal event "
            f"after {_max_environment_steps()} steps"
        )
    outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}.get(event, "timeout")
    return states, actions, rewards, outcome


def _save_training_checkpoint(path: Path, encoder, value_network, action_encoder, optimizer, replay, state: dict, ema_encoder=None, ema_value_network=None, ema_action_encoder=None) -> None:
    payload = {
        # R4-2R-1 fix (2026-08-10, guide.md "R4-2R-1"): format_version 6
        # adds training_contract_schema -- R4-2 changed the Replay data
        # model and MC-loss math without any accompanying checkpoint
        # version bump at the time, which would have let a v5 (R4-1-loss)
        # resume checkpoint keep being silently treated as R4-2-compatible
        # indefinitely. A v5 checkpoint's replay_state also predates
        # replay.REPLAY_SCHEMA_V2_R4_2 and would fail that check anyway,
        # but failing on format_version first gives a clearer diagnosis.
        "format_version": 6,
        "training_contract_schema": TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC,
        "encoder_state_dict": encoder.state_dict(), "value_network_state_dict": value_network.state_dict(),
        "action_encoder_state_dict": action_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "replay_state": replay.state_dict(), "trainer_state": state,
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "action_grid_hash": state["action_grid_hash"], "registry_content_sha256": state["registry_content_sha256"],
        "artifact_sha256": state["artifact_sha256"], "world_train_data_sha256": state["world_train_data_sha256"],
        "training_config_sha256": state["training_config_sha256"],
        "feature_schema": FEATURE_SCHEMA_V4, "reward_schema": REWARD_SCHEMA,
        "return_bounds": [float(value_network.v_min), float(value_network.v_max)],
        "ema_initialized": bool(state.get("ema_initialized", False)),
        "ema_encoder_state_dict": ema_encoder.state_dict() if ema_encoder is not None else None,
        "ema_value_network_state_dict": ema_value_network.state_dict() if ema_value_network is not None else None,
        "ema_action_encoder_state_dict": ema_action_encoder.state_dict() if ema_action_encoder is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _save_selector_checkpoint(path: Path, encoder, value_network, action_encoder, registry: dict, artifact_sha256: str, training_config_sha256: str) -> None:
    """Save a small policy-only candidate for validation checkpoint selection.

    Full resume checkpoints contain the replay buffer and are intentionally
    large.  Keeping every one of them made a formal run exhaust the 4090
    host's root filesystem.  Selector candidates need only the deployed
    encoder/value/action weights plus provenance; the single
    ``resume_latest.pth`` written beside them retains exact-resume state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_composed_checkpoint(
        encoder,
        value_network,
        str(temporary),
        registry["action_grid_hash"],
        registry["content_sha256"],
        action_encoder=action_encoder,
        artifact_sha256=artifact_sha256,
        training_config_sha256=training_config_sha256,
    )
    os.replace(temporary, path)


def _load_training_checkpoint(path: Path, encoder, value_network, action_encoder, optimizer, replay, device: str, ema_encoder=None, ema_value_network=None, ema_action_encoder=None) -> dict:
    # This is an explicitly trusted, repository-owned full-resume artifact;
    # it intentionally contains MCReturnSample objects and RNG/replay state.
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 6:
        # R4-2R-1 fix (2026-08-10, guide.md "R4-2R-1"): a v5 (R4-1-loss)
        # checkpoint's replay_state also predates
        # replay.REPLAY_SCHEMA_V2_R4_2 and its weights were fit under the
        # retired placeholder MC-loss target -- resuming from one would
        # silently continue training under a training_contract_schema
        # this code no longer even records, let alone validates.
        raise RuntimeError("checkpoint is not a v6 (R4-2, executed-action-MC-aware) full BDVL resume checkpoint")
    required = ("encoder_state_dict", "value_network_state_dict", "action_encoder_state_dict",
                "optimizer_state_dict", "replay_state",
                "trainer_state", "python_rng_state", "numpy_rng_state", "torch_rng_state",
                "action_grid_hash", "registry_content_sha256", "artifact_sha256",
                "world_train_data_sha256", "training_config_sha256", "feature_schema", "reward_schema",
                "training_contract_schema", "return_bounds", "ema_initialized", "ema_encoder_state_dict",
                "ema_value_network_state_dict", "ema_action_encoder_state_dict")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"resume checkpoint missing required fields: {missing}")
    if payload["feature_schema"] != FEATURE_SCHEMA_V4 or payload["reward_schema"] != REWARD_SCHEMA:
        raise RuntimeError("resume checkpoint schema drift")
    if payload["training_contract_schema"] != TRAINING_CONTRACT_V1_EXECUTED_ACTION_MC:
        raise RuntimeError("resume checkpoint training_contract_schema drift")
    stored_min, stored_max = float(payload["return_bounds"][0]), float(payload["return_bounds"][1])
    if abs(stored_min - value_network.v_min) > 1e-9 or abs(stored_max - value_network.v_max) > 1e-9:
        raise RuntimeError("resume checkpoint return bounds do not match the current value network")
    if payload["world_train_data_sha256"] != payload["trainer_state"].get("world_train_data_sha256"):
        raise RuntimeError("resume checkpoint world-train data hash mismatch")
    encoder.load_state_dict(payload["encoder_state_dict"]); value_network.load_state_dict(payload["value_network_state_dict"])
    action_encoder.load_state_dict(payload["action_encoder_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"]); replay.load_state_dict(payload["replay_state"])
    random.setstate(payload["python_rng_state"]); np.random.set_state(payload["numpy_rng_state"]); torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if payload["ema_initialized"]:
        if ema_encoder is None or ema_value_network is None or ema_action_encoder is None:
            raise RuntimeError("resume checkpoint has EMA state but no EMA modules were supplied to load into")
        ema_encoder.load_state_dict(payload["ema_encoder_state_dict"])
        ema_value_network.load_state_dict(payload["ema_value_network_state_dict"])
        ema_action_encoder.load_state_dict(payload["ema_action_encoder_state_dict"])
    trainer_state = dict(payload["trainer_state"])
    trainer_state["ema_initialized"] = bool(payload["ema_initialized"])
    return trainer_state


def stage2_online_rl(encoder, value_network, action_encoder, artifact, action_table, env_config_path, replay, optimizer, device, seed, n_episodes, start_episode, total_rl_episodes, gamma, epsilon_start, epsilon_end, batch_size, updates_per_episode, profiles, checkpoint_dir, registry, training_config_sha256, ema_encoder=None, ema_value_network=None, ema_action_encoder=None, ema_initialized=True, gradient_diagnostics=None):
    from crowd_nav.bayesian_dvl.ranking import RankingDemoSample
    from crowd_nav.bayesian_dvl.replay import MCReturnSample
    from crowd_nav.bayesian_dvl.trainer import stage2_train_step

    losses = []
    gradient_monitor = {"outside_streak": 0}
    outcome_counts = {"success": 0, "collision": 0, "timeout": 0}
    artifact_hash = artifact.content_sha256()
    reward_config = _make_reward_config()
    ranking_margin = float(FROZEN_VALUES["ranking_margin"])
    lambda_rank = float(FROZEN_VALUES["lambda_rank"])
    grad_clip_norm = float(FROZEN_VALUES["grad_clip_norm"])
    ema_decay = float(FROZEN_VALUES["ema_decay"])
    for local_index in range(n_episodes):
        episode_index = start_episode + local_index
        # R3R-3 fix: epsilon must be a function of the ABSOLUTE episode
        # index against the TOTAL frozen budget, never re-derived from
        # this invocation's own (possibly-partial, post-resume) episode
        # count -- otherwise a resumed run's epsilon schedule diverges
        # from an uninterrupted one covering the same total_rl_episodes.
        fraction = episode_index / max(1, total_rl_episodes - 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * min(1.0, fraction)
        # Use the absolute episode identity, not the local segment index;
        # otherwise a resumed run restarts the profile schedule at nominal
        # and diverges from an uninterrupted run.
        profile = profiles[episode_index % len(profiles)]
        # R2-3: the whole episode is collected BEFORE anything enters
        # replay -- MC returns need the complete trajectory, so there is
        # no more "push one transition, train, repeat" within an episode.
        states, actions, rewards, outcome = _collect_online_episode(env_config_path, artifact, action_table, seed, episode_index, profile, encoder, value_network, action_encoder, epsilon)
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        for sample in build_mc_return_samples(states, actions, rewards, gamma, artifact_hash, seed, seed * 100000 + episode_index, outcome=outcome):
            replay.online.add(sample)
        if len(replay.online) >= 1:
            for _ in range(updates_per_episode):
                # R3-4: demo-drawn items are RankingDemoSample (get both
                # L_MC and L_rank); online-drawn items are MCReturnSample
                # (L_MC only). sample_batch's own n_demo count is what
                # tells us where the split falls in the concatenated list.
                batch, n_demo, n_online = replay.sample_batch(batch_size, random.Random(seed * 100000 + episode_index))
                demo_batch = batch[:n_demo]
                online_batch = batch[n_demo:]
                assert all(isinstance(s, RankingDemoSample) for s in demo_batch)
                assert all(isinstance(s, MCReturnSample) for s in online_batch)
                result = stage2_train_step(
                    encoder, value_network, action_encoder, artifact, action_table, reward_config,
                    dt=float(FROZEN_VALUES["dt"]), time_limit=float(FROZEN_VALUES["time_limit"]),
                    max_human_speed=float(FROZEN_VALUES["max_human_speed"]),
                    n_world_samples=int(FROZEN_VALUES["world_samples_train"]),
                    n_iqn_quantiles=int(FROZEN_VALUES["iqn_train_quantiles"]),
                    posterior_source="full", device=device, gamma=gamma,
                    demo_batch=demo_batch, online_batch=online_batch,
                    ranking_margin=ranking_margin, lambda_rank=lambda_rank,
                    n_train_quantiles=int(FROZEN_VALUES["iqn_train_quantiles"]), optimizer=optimizer,
                    grad_clip_norm=grad_clip_norm,
                    ranking_batch_size=int(FROZEN_VALUES["ranking_batch_size"]),
                )
                if result.aborted:
                    raise RuntimeError(result.abort_reason)
                _validate_and_record_gradient_diagnostic(
                    result, "rl", episode_index + 1, gradient_diagnostics, gradient_monitor
                )
                losses.append(float(result.loss))
                if ema_encoder is not None and ema_value_network is not None:
                    _ema_update(ema_encoder, encoder, ema_decay)
                    _ema_update(ema_value_network, value_network, ema_decay)
                    if ema_action_encoder is not None:
                        _ema_update(ema_action_encoder, action_encoder, ema_decay)
        if (episode_index + 1) % 500 == 0 or local_index == n_episodes - 1:
            n = sum(outcome_counts.values())
            print(f"RL_PROGRESS episode={episode_index + 1} outcomes={outcome_counts} sr={outcome_counts['success']/n:.3f} cr={outcome_counts['collision']/n:.3f} tr={outcome_counts['timeout']/n:.3f}")
        if checkpoint_dir is not None and ((episode_index + 1) % 500 == 0 or local_index == n_episodes - 1):
            # Keep exactly one full replay/RNG-bearing checkpoint for resume;
            # all historical candidates are policy-only and therefore small.
            _save_training_checkpoint(
                checkpoint_dir / "resume_latest.pth", encoder, value_network, action_encoder, optimizer, replay,
                {"seed": seed, "episode": episode_index + 1, "action_grid_hash": registry["action_grid_hash"], "registry_content_sha256": registry["content_sha256"], "artifact_sha256": artifact_hash, "world_train_data_sha256": artifact.train_data_sha256, "training_config_sha256": training_config_sha256, "ema_initialized": ema_initialized},
                ema_encoder=ema_encoder, ema_value_network=ema_value_network, ema_action_encoder=ema_action_encoder,
            )
            _save_selector_checkpoint(
                checkpoint_dir / f"checkpoint_ep{episode_index + 1}.pth",
                encoder,
                value_network,
                action_encoder,
                registry,
                artifact_hash,
                training_config_sha256,
            )
            if ema_encoder is not None and ema_value_network is not None:
                _save_selector_checkpoint(
                    checkpoint_dir / f"checkpoint_ep{episode_index + 1}_ema.pth",
                    ema_encoder,
                    ema_value_network,
                    ema_action_encoder,
                    registry,
                    artifact_hash,
                    training_config_sha256,
                )
    return losses


def _ema_update(ema_module, online_module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, online_param in zip(ema_module.parameters(), online_module.parameters()):
            ema_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default="crowd_nav/configs/env_bayesian_dvl.config")
    parser.add_argument("--train-config", default="crowd_nav/configs/train_bayesian_dvl.config")
    # R4-1R-2a fix (2026-08-10, guide.md "R4-1R-2a"): train_bdvl.py used to
    # call build_frozen_registry() and construct a registry dict IN MEMORY
    # every run, never loading the same on-disk file select/evaluate/queue
    # read -- in practice the two happened to produce identical content
    # (same FROZEN_VALUES + same env config), but that was an IMPLICIT
    # consistency assumption, not an enforced contract: nothing would have
    # caught it if the two ever silently diverged (e.g. env config edited
    # without regenerating the registry file). Loading the same explicit
    # file every stage reads closes that gap for real.
    parser.add_argument("--registry", default="crowd_nav/configs/bayesian_dvl_registry_r4.json")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--il-episodes", type=int, default=FROZEN_VALUES["il_episodes"])
    parser.add_argument("--rl-episodes", type=int, default=FROZEN_VALUES["rl_min_episodes"])
    parser.add_argument("--stop-after-rl-episodes", type=int, default=None,
                        help="diagnostic checkpoint boundary; total target remains --rl-episodes")
    parser.add_argument("--seed", type=int, default=93001)
    parser.add_argument("--profiles", nargs="+", choices=["nominal", "train_nonstationary"], default=["nominal", "train_nonstationary"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--updates-per-episode", type=int, default=None)
    parser.add_argument("--allow-short-run", action="store_true", help="diagnostic-only budget below the frozen formal minimum")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # R4-1R-3 fix (2026-08-10, guide.md "R4-1R -- 在 R4-2 完成前禁止误启动训练"):
    # see config.R4_2_REPLAY_CONTRACT_COMPLETE's own comment for why. This
    # check has no override flag by design -- --allow-short-run controls a
    # DIFFERENT, unrelated thing (diagnostic-only episode-count budgets)
    # and must not double as a bypass for an incomplete supervision
    # target. The only way to unblock this CLI is to actually land R4-2
    # and flip the flag in config.py.
    if not R4_2_REPLAY_CONTRACT_COMPLETE:
        raise SystemExit(
            "train_bdvl.py is fail-closed pending guide.md R4-2 (Replay 与监督目标): "
            "stage1_train_step/stage2_train_step's MC-loss target still uses placeholder "
            "action features (trainer._partial_action_features_from_table / "
            "expert_action_indices[0]-as-executed-action), not a genuinely recorded executed "
            "action. Training now would bake a wrong supervision target into every resulting "
            "checkpoint. Flip config.R4_2_REPLAY_CONTRACT_COMPLETE only once R4-2's replay "
            "schema and MC-loss rewrite are actually implemented and tested -- there is no CLI "
            "override for this check."
        )

    env_config_path = _resolve_path(args.env_config)
    train_config_path = _resolve_path(args.train_config)
    training = _load_training_values(train_config_path)
    batch_size = int(args.batch_size or training["batch_size"])
    if args.batch_size is not None and args.batch_size != training["batch_size"]:
        raise SystemExit("batch-size override is not allowed; use a separate frozen training config")
    updates_per_episode = int(args.updates_per_episode or training["updates_per_episode"])
    if updates_per_episode <= 0:
        raise SystemExit("updates-per-episode must be positive")
    if (args.il_episodes < int(FROZEN_VALUES["il_episodes"]) or args.rl_episodes < int(FROZEN_VALUES["rl_min_episodes"])) and not args.allow_short_run:
        raise SystemExit("short IL/RL budgets are diagnostic-only; pass --allow-short-run explicitly")
    if args.stop_after_rl_episodes is not None and not (0 <= args.stop_after_rl_episodes <= args.rl_episodes):
        raise SystemExit("stop-after-rl-episodes must be within [0, rl-episodes]")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but CUDA is unavailable")
    _seed_everything(args.seed)
    action_spec = ActionGridSpec.from_env_config(str(env_config_path))
    action_table = action_spec.build_action_table()
    artifact = SBKHMMArtifact.load(str(_resolve_path(args.artifact_path)))
    registry = load_and_validate_registry(str(_resolve_path(args.registry)), env_config_path=str(env_config_path))
    config_hash = sha256_of_file(str(train_config_path))

    v_min, v_max = derive_return_bounds(FROZEN_VALUES)
    encoder = SetEncoder().to(device)
    action_encoder = ActionEncoder().to(device)
    value_network = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim, v_min=v_min, v_max=v_max).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(value_network.parameters()) + list(action_encoder.parameters()), lr=training["learning_rate"])
    # R3R-3 fix (2026-08-07): EMA must NOT be constructed here -- doing
    # so copies random-init weights into the EMA shadow before Stage 1
    # has run at all, so early Stage 2 EMA checkpoints stay contaminated
    # with pre-IL noise no matter how much online training follows.
    # Construction is deferred to immediately after Stage 1 (a HARD
    # copy of the just-trained online weights, not an EMA-decayed one)
    # or restored from a resume checkpoint below.
    ema_encoder = SetEncoder().to(device)
    ema_action_encoder = ActionEncoder().to(device)
    ema_value_network = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=ema_action_encoder.embed_dim, v_min=v_min, v_max=v_max).to(device)
    for p in list(ema_encoder.parameters()) + list(ema_value_network.parameters()) + list(ema_action_encoder.parameters()):
        p.requires_grad_(False)
    ema_initialized = False
    # R2-3: online capacity is fixed at 50,000 (a recent-policy window),
    # independent of the frozen demo reservoir capacity -- guide.md R2-3:
    # "online replay固定容量为50,000 transitions，只保留最近策略窗口".
    replay = DemoOnlineReplay(int(training["replay_capacity"]), 50000, float(training["demo_sample_ratio"]))
    gradient_diagnostics = []
    start_episode = 0
    if args.resume:
        resume_state = _load_training_checkpoint(_resolve_path(args.resume), encoder, value_network, action_encoder, optimizer, replay, device, ema_encoder=ema_encoder, ema_value_network=ema_value_network, ema_action_encoder=ema_action_encoder)
        # R3R-0 fix (2026-08-07): registry_content_sha256 was recorded in
        # every checkpoint's trainer_state but never actually checked on
        # resume -- a config.py edit between the original run and a resume
        # (frozen values, action grid, OR normalization_constants, which
        # is now folded into the same registry hash) would silently keep
        # training with drifted definitions instead of failing closed.
        if (resume_state["seed"] != args.seed or
                resume_state["artifact_sha256"] != artifact.content_sha256() or
                resume_state["world_train_data_sha256"] != artifact.train_data_sha256 or
                resume_state["training_config_sha256"] != config_hash or
                resume_state["registry_content_sha256"] != registry["content_sha256"]):
            raise SystemExit("resume checkpoint seed/artifact/training-config/registry drift")
        start_episode = int(resume_state["episode"])
        ema_initialized = bool(resume_state.get("ema_initialized", False))

    checkpoint_dir = _resolve_path(args.checkpoint_dir) if args.checkpoint_dir else None
    output_path = _resolve_path(args.output)
    try:
        if start_episode == 0 and args.il_episodes > 0:
            il_losses = stage1_il_pretrain(encoder, value_network, action_encoder, artifact, action_table, env_config_path, replay, args.il_episodes, training["gamma"], optimizer, device, args.profiles, 92001, batch_size, gradient_diagnostics=gradient_diagnostics)
        else:
            il_losses = []
        if not ema_initialized:
            # R3R-3: hard-copy (not EMA-blend) the just-trained Stage 1
            # weights as the EMA shadow's starting point -- guide.md R3R-3
            # point 1. Only happens once per training run (fresh or first
            # resume after Stage 1), guarded by ema_initialized so a
            # resumed Stage 2 run never re-clobbers accumulated EMA state.
            ema_encoder.load_state_dict(encoder.state_dict())
            ema_value_network.load_state_dict(value_network.state_dict())
            ema_action_encoder.load_state_dict(action_encoder.state_dict())
            ema_initialized = True
        # R3R-3 fix: --rl-episodes is the TOTAL target episode count, not
        # "how many more to run this invocation" -- a resumed run must only
        # execute the remainder, and epsilon must be computed against the
        # SAME total budget a non-resumed run would have used, or a resumed
        # trajectory permanently diverges from an uninterrupted one.
        total_rl_episodes = args.rl_episodes
        target_this_invocation = args.stop_after_rl_episodes if args.stop_after_rl_episodes is not None else total_rl_episodes
        if start_episode > target_this_invocation:
            raise SystemExit(f"resume episode {start_episode} is past this invocation target {target_this_invocation}")
        remaining_rl_episodes = max(0, target_this_invocation - start_episode)
        rl_losses = stage2_online_rl(encoder, value_network, action_encoder, artifact, action_table, env_config_path, replay, optimizer, device, args.seed, remaining_rl_episodes, start_episode, total_rl_episodes, training["gamma"], training["epsilon_start"], training["epsilon_end"], batch_size, updates_per_episode, args.profiles, checkpoint_dir, registry, config_hash, ema_encoder=ema_encoder, ema_value_network=ema_value_network, ema_action_encoder=ema_action_encoder, ema_initialized=ema_initialized, gradient_diagnostics=gradient_diagnostics)
    except RuntimeError as exc:
        # Preserve the diagnostic window even when a hard numerical or
        # sustained-gradient gate aborts before a checkpoint/manifest exists.
        atomic_write_json(str(output_path) + ".abort.json", {
            "schema": "bdvl_training_abort_v1",
            "status": "aborted",
            "error": str(exc),
            "seed": int(args.seed),
            "device": device,
            "il_episodes_target": int(args.il_episodes),
            "rl_episodes_target": int(args.rl_episodes),
            "gradient_diagnostics_tail": gradient_diagnostics[-256:],
            "registry_content_sha256": registry["content_sha256"],
            "action_grid_hash": registry["action_grid_hash"],
            "artifact_sha256": artifact.content_sha256(),
            "training_config_sha256": config_hash,
        })
        raise
    # R3R-3 fix: args.rl_episodes is the TOTAL target episode count, not
    # "episodes run this invocation" -- adding it to start_episode here
    # double-counts on any resumed run (e.g. resume at 1000/2000 would
    # write episode=3000). The actual count of episodes executed so far
    # is start_episode + remaining_rl_episodes, which equals
    # total_rl_episodes in the normal (non-clamped) case.
    final_state = {"seed": args.seed, "episode": start_episode + remaining_rl_episodes, "action_grid_hash": registry["action_grid_hash"], "registry_content_sha256": registry["content_sha256"], "artifact_sha256": artifact.content_sha256(), "world_train_data_sha256": artifact.train_data_sha256, "training_config_sha256": config_hash, "ema_initialized": bool(ema_initialized)}
    _save_training_checkpoint(
        output_path, encoder, value_network, action_encoder, optimizer, replay, final_state,
        ema_encoder=ema_encoder, ema_value_network=ema_value_network, ema_action_encoder=ema_action_encoder,
    )
    ema_output_path = output_path.with_name(output_path.stem + "_ema" + output_path.suffix)
    _save_selector_checkpoint(ema_output_path, ema_encoder, ema_value_network, ema_action_encoder, registry, artifact.content_sha256(), config_hash)
    if checkpoint_dir is not None:
        resume_latest = checkpoint_dir / "resume_latest.pth"
        if resume_latest.resolve() != output_path.resolve() and resume_latest.exists():
            resume_latest.unlink()
    manifest = build_run_manifest(repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv), source_files=BDVL_PRODUCTION_SOURCES, extra={"seed": args.seed, "device": device, "allow_short_run": args.allow_short_run, "updates_per_episode": updates_per_episode, "il_losses": il_losses, "rl_losses": rl_losses, "gradient_diagnostics": gradient_diagnostics, "demo_size": len(replay.demo), "online_size": len(replay.online), "artifact_sha256": artifact.content_sha256(), "world_train_data_sha256": artifact.train_data_sha256, "training_config_sha256": config_hash,
        # Engineering-gap fix (2026-08-08): the checkpoint itself always
        # carried registry_content_sha256/action_grid_hash (see
        # _save_training_checkpoint), but the run's own top-level manifest
        # never recorded them -- anyone auditing just the manifest (not
        # opening the checkpoint) had no way to confirm which frozen
        # registry this specific run used.
        "registry_content_sha256": registry["content_sha256"], "action_grid_hash": registry["action_grid_hash"],
        "feature_schema": FEATURE_SCHEMA_V4, "reward_schema": REWARD_SCHEMA, "return_bounds": [v_min, v_max], "ranking_batch_size": int(FROZEN_VALUES["ranking_batch_size"]), "lambda_rank": float(FROZEN_VALUES["lambda_rank"]), "ranking_margin": float(FROZEN_VALUES["ranking_margin"]), "grad_clip_norm": float(FROZEN_VALUES["grad_clip_norm"]), "ema_decay": float(FROZEN_VALUES["ema_decay"]), "rank_gradient_ratio_min": float(FROZEN_VALUES["rank_gradient_ratio_min"]), "rank_gradient_ratio_max": float(FROZEN_VALUES["rank_gradient_ratio_max"])})
    atomic_write_json(str(output_path) + ".manifest.json", manifest)
    print(f"TRAIN_BDVL_DONE checkpoint={output_path} device={device} demo={len(replay.demo)} online={len(replay.online)}")


if __name__ == "__main__":
    main()
