#!/usr/bin/env python3
"""Fail-closed BDVL validation/formal evaluator.

Smoke mode is the only mode that may use a synthetic artifact or random
weights. Every non-smoke run requires a production artifact, a composed
checkpoint, the frozen registry, and an explicit profile/scenario protocol.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path

from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: E402
from crowd_sim.envs.utils.robot import Robot  # noqa: E402
from crowd_nav.bayesian_pilot.protocol import BehaviorScheduler, InterventionORCA, PROFILES  # noqa: E402
from crowd_nav.bayesian_dvl.config import ActionGridSpec, BDVL_PRODUCTION_SOURCES, FROZEN_VALUES, FORMAL_SCENARIOS, derive_return_bounds, load_and_validate_registry  # noqa: E402
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact, Track, fit_sbk_hmm, promote_to_production  # noqa: E402
from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, SetEncoder  # noqa: E402
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork  # noqa: E402
from crowd_nav.bayesian_dvl.transition import RewardConfig  # noqa: E402
from crowd_nav.bayesian_dvl.policy import BDVLPolicy, BayesianDVLPolicy, load_composed_checkpoint  # noqa: E402
from crowd_nav.bayesian_dvl.statistics import CalibrationAccumulator, EpisodeRecord, validate_cvar_promotion_report  # noqa: E402
from crowd_nav.bayesian_dvl.evaluate import deterministic_records_sha256, write_episode_records_csv  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest, sha256_of_file  # noqa: E402


def _smoke_artifact() -> SBKHMMArtifact:
    dt = FROZEN_VALUES["dt"]

    def track(speed0, accel, omega):
        pos = [np.array([0.0, 0.0])]
        speed, heading = speed0, 0.0
        for _ in range(30):
            speed = max(speed + accel * dt, 0.0)
            heading += omega * dt
            vel = speed * np.array([np.cos(heading), np.sin(heading)])
            pos.append(pos[-1] + vel * dt)
        return Track(positions=np.array(pos), dt=dt)

    tracks = [track(1.0, 0.0, 0.0), track(0.3, 0.8, 0.0), track(1.5, -0.8, 0.0), track(1.0, 0.0, 1.0), track(1.0, 0.0, -1.0)]
    return promote_to_production(fit_sbk_hmm(tracks, train_data_sha256="smoke_fixture", max_iterations=15))


def _make_policy(
    action_table, artifact, checkpoint_path, registry, smoke, suite_seed,
    phase, device, *, risk_neutral=False, collect_calibration=False,
):
    v_min, v_max = derive_return_bounds(registry["frozen_values"])
    encoder = SetEncoder()
    action_encoder = ActionEncoder()
    net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim, v_min=v_min, v_max=v_max)
    if checkpoint_path:
        manifest = load_composed_checkpoint(
            checkpoint_path, encoder, net, action_encoder,
            expected_artifact_sha256=artifact.content_sha256(),
            expected_registry_content_sha256=registry["content_sha256"],
            expected_action_grid_hash=registry["action_grid_hash"],
            require_formal_provenance=not smoke,
        )
        if manifest["action_grid_hash"] != registry["action_grid_hash"] or manifest["registry_content_sha256"] != registry["content_sha256"]:
            raise SystemExit("checkpoint registry/action-grid hash mismatch")
    elif not smoke:
        raise SystemExit("formal evaluation requires --checkpoint")
    encoder.to(device).eval(); net.to(device).eval(); action_encoder.to(device).eval()
    frozen = registry["frozen_values"]
    reward_config = RewardConfig(
        success_reward=frozen["success_reward"], collision_penalty=frozen["collision_penalty"], timeout_penalty=frozen["timeout_penalty"],
        progress_reward=frozen["progress_reward"], time_penalty=frozen["time_penalty"], stand_penalty=frozen["stand_penalty"],
        stand_speed_threshold=frozen["stand_speed_threshold"], discomfort_distance=frozen["discomfort_distance"],
        discomfort_penalty_factor=frozen["discomfort_penalty_factor"],
    )
    budgets = {"train": (frozen["world_samples_train"], frozen["iqn_train_quantiles"]), "validation": (frozen["world_samples_validation"], frozen["iqn_quantiles_validation"]), "formal": (frozen["world_samples_formal"], frozen["iqn_quantiles_formal"])}
    worlds, quantiles = budgets[phase]
    bdvl = BDVLPolicy(artifact=artifact, set_encoder=encoder, value_network=net, action_encoder=action_encoder, action_table=action_table, reward_config=reward_config,
                      dt=frozen["dt"], time_limit=frozen["time_limit"], max_human_speed=frozen["max_human_speed"], cvar_alpha=frozen["cvar_alpha"],
                      n_world_samples=worlds, n_iqn_quantiles=quantiles,
                      risk_neutral=bool(risk_neutral), collect_calibration=collect_calibration)
    adapter = BayesianDVLPolicy(bdvl_policy=bdvl, suite_seed=suite_seed)
    adapter.set_device(device)
    adapter.time_step = frozen["dt"]
    return adapter


def _make_env(env_config_path: Path, human_num: int, scenario_key: str):
    if env_config_path.name != "env_bayesian_dvl.config":
        raise SystemExit("BDVL evaluation must use env_bayesian_dvl.config")
    env_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not env_config.read(str(env_config_path)):
        raise SystemExit(f"env config not found: {env_config_path}")
    scenario = FORMAL_SCENARIOS[scenario_key]
    env_config.set("sim", "human_num", str(human_num))
    env_config.set("robot", "policy", "orca")
    env_config.set("sim", "train_val_sim", "circle_crossing" if scenario["layout"] == "circle" else "square_crossing")
    env_config.set("sim", "test_sim", "circle_crossing" if scenario["layout"] == "circle" else "square_crossing")
    if scenario["layout"] == "circle":
        env_config.set("sim", "circle_radius", str(scenario["radius"]))
    else:
        env_config.set("sim", "square_width", str(scenario["width"]))
    env_config.set("env", "time_limit", str(int(FROZEN_VALUES["time_limit"])))
    env_config.set("reward", "progress_reward", str(FROZEN_VALUES["progress_reward"]))
    env_config.set("reward", "time_penalty", str(FROZEN_VALUES["time_penalty"]))
    env = CrowdSim()
    env.configure(env_config)
    env.phase = "test"
    return env, env_config


def run_episode(
    policy, env, suite_seed: int, episode_index: int, method: str, scenario: str,
    profile: str, dt: float, calibration_rows: list[dict] | None = None,
    episode_seed: int | None = None,
) -> EpisodeRecord:
    # 2026-08-08 fix: cross-method paper comparison tables need a
    # DIFFERENT deterministic episode_seed formula (matching test8.py's
    # verified-working scheme) than BDVL's own validation/formal
    # protocol uses -- passing episode_seed explicitly overrides the
    # default without touching any existing validation/formal call site.
    if episode_seed is None:
        episode_seed = suite_seed * 100000 + episode_index
    env.case_counter["test"] = episode_seed % (2**32 - 1)
    policy.reset_episode_stats(suite_seed=suite_seed, episode_seed=episode_seed)
    env.reset()
    for human in env.humans:
        human_policy = InterventionORCA(env.config)
        human_policy.time_step = env.time_step
        human.set_policy(human_policy)
    scheduler = BehaviorScheduler(PROFILES[profile], seed=episode_seed)
    scheduler.reset(len(env.humans))
    min_clearance = float("inf")
    path_length = 0.0
    latencies_ms = []
    last_pos = np.array([env.robot.px, env.robot.py])
    goal = np.array([env.robot.gx, env.robot.gy])
    initial_goal_distance = float(np.linalg.norm(last_pos - goal))
    max_goal_distance = initial_goal_distance
    # R2-5 fix (2026-08-07): "SR/CR/TR alone" cannot distinguish a robot
    # that is slowly converging from one that is actively walking away
    # from the goal (the independent diagnosis's actual failure mode) --
    # log per-decision action/goal alignment and the derived value bound
    # so every future validation run carries this evidence automatically.
    alignments: List[float] = []
    max_action_score = float("-inf")
    min_action_score = float("inf")
    score_bound_violations = 0
    bdvl_policy = getattr(policy, "bdvl_policy", None)
    v_max_bound = bdvl_policy.value_network.v_max if bdvl_policy is not None else None
    v_min_bound = bdvl_policy.value_network.v_min if bdvl_policy is not None else None
    outcome = None
    steps = 0
    step_rewards = []
    predicted_rows = []
    predicted_quantile_rows = []
    for _ in range(200):
        if profile != "nominal":
            scheduler.advance([h.policy for h in env.humans])
        human_obs = [h.get_observable_state() for h in env.humans]
        current_pos = np.array([env.robot.px, env.robot.py])
        start = time.perf_counter()
        action = env.robot.act(human_obs)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        decision = getattr(policy, "last_decision", None)
        bdvl_runtime = getattr(policy, "bdvl_policy", None)
        if calibration_rows is not None and bdvl_runtime is not None:
            predicted = bdvl_runtime.last_predicted_return_samples
            fixed_quantiles = bdvl_runtime.last_predicted_quantile_values
            if predicted is None or len(predicted) == 0 or fixed_quantiles is None:
                raise RuntimeError("BDVL policy did not expose predictive return samples for calibration")
            predicted_rows.append(np.asarray(predicted, dtype=np.float64).copy())
            # Preserve the fixed ordered IQN output separately from the
            # posterior-predictive sample row; these are not interchangeable.
            predicted_quantile_rows.append(np.asarray(fixed_quantiles, dtype=np.float64).copy())
        if decision is not None:
            action_vec = np.array(decision.chosen_action)
            to_goal = goal - current_pos
            denom = np.linalg.norm(to_goal) * np.linalg.norm(action_vec)
            if denom > 1e-9:
                alignments.append(float(np.dot(action_vec, to_goal) / denom))
            decision_max_score = max(s.cvar for s in decision.all_scores)
            decision_min_score = min(s.cvar for s in decision.all_scores)
            max_action_score = max(max_action_score, decision_max_score)
            min_action_score = min(min_action_score, decision_min_score)
            if (
                v_max_bound is not None
                and v_min_bound is not None
                and (decision_max_score > v_max_bound + 1e-3 or decision_min_score < v_min_bound - 1e-3)
            ):
                score_bound_violations += 1
        _, reward, terminated, truncated, info = env.step(action)
        step_rewards.append(float(reward))
        steps += 1
        for human in env.humans:
            center_clearance = float(np.hypot(human.px - env.robot.px, human.py - env.robot.py)) - human.radius - env.robot.radius
            min_clearance = min(min_clearance, center_clearance)
        if isinstance(info, dict) and info.get("dmin") is not None:
            min_clearance = min(min_clearance, float(info["dmin"]))
        new_pos = np.array([env.robot.px, env.robot.py])
        path_length += float(np.linalg.norm(new_pos - last_pos))
        last_pos = new_pos
        max_goal_distance = max(max_goal_distance, float(np.linalg.norm(new_pos - goal)))
        if terminated or truncated:
            event = info.get("event") if isinstance(info, dict) else None
            if event not in {"reach_goal", "collision", "timeout"}:
                raise RuntimeError(f"unknown CrowdSim termination event: {event!r}")
            outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}[event]
            break
    if outcome is None:
        raise RuntimeError("episode exceeded evaluator hard step bound without a termination event")
    if calibration_rows is not None:
        if len(predicted_rows) != len(step_rewards) or len(predicted_quantile_rows) != len(step_rewards):
            raise RuntimeError(
                f"calibration prediction/reward length mismatch: {len(predicted_rows)}/{len(predicted_quantile_rows)} != {len(step_rewards)}"
            )
        realized_returns = [0.0] * len(step_rewards)
        running = 0.0
        gamma = float(FROZEN_VALUES["gamma"])
        for index in reversed(range(len(step_rewards))):
            running = step_rewards[index] + gamma * running
            realized_returns[index] = running
        for step_index, (predicted, fixed_quantiles, realized) in enumerate(zip(predicted_rows, predicted_quantile_rows, realized_returns)):
            row = {
                "scenario": scenario, "profile": profile, "suite_seed": int(suite_seed),
                "episode_seed": int(episode_seed), "step_index": int(step_index),
                "predicted_samples": predicted.tolist(), "realized_return": float(realized),
                "predicted_quantiles": fixed_quantiles.tolist(),
                # R3R-5 fix: guide.md's calibration stratification is by
                # profile x outcome; outcome was not previously recorded
                # on the calibration row at all.
                "outcome": outcome,
            }
            if hasattr(calibration_rows, "add"):
                calibration_rows.add(row)
            else:
                calibration_rows.append(row)
    return EpisodeRecord(method=method, scenario=scenario, profile=profile, suite_seed=suite_seed, episode_seed=episode_seed,
                         outcome=outcome, steps=steps, elapsed_time=steps * dt,
                         min_clearance=min_clearance if np.isfinite(min_clearance) else 0.0,
                         path_length=path_length, mean_decision_latency_ms=float(np.mean(latencies_ms)),
                         initial_goal_distance=initial_goal_distance,
                         final_goal_distance=float(np.linalg.norm(last_pos - goal)),
                         max_goal_distance=max_goal_distance,
                         mean_action_alignment=float(np.mean(alignments)) if alignments else 0.0,
                         negative_alignment_fraction=float(np.mean([a < 0 for a in alignments])) if alignments else 0.0,
                         max_action_score=float(max_action_score) if np.isfinite(max_action_score) else 0.0,
                         min_action_score=float(min_action_score) if np.isfinite(min_action_score) else 0.0,
                         score_bound_violations=score_bound_violations)


def _assert_complete(records, scenarios, profiles, seeds, episodes_per_seed):
    expected = {(s, p, seed, seed * 100000 + i) for s in scenarios for p in profiles for seed in seeds for i in range(episodes_per_seed)}
    actual = {(r.scenario, r.profile, r.suite_seed, r.episode_seed) for r in records}
    if len(actual) != len(records) or actual != expected:
        raise RuntimeError(f"formal episode identity mismatch missing={len(expected - actual)} extra={len(actual - expected)} duplicates={len(records)-len(actual)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default="crowd_nav/configs/env_bayesian_dvl.config")
    parser.add_argument("--registry", default="crowd_nav/configs/bayesian_dvl_registry_r4.json")
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--method", default="bayesian_dvl")
    parser.add_argument("--phase", choices=["validation", "formal"], default="formal")
    parser.add_argument("--suite-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--episodes-per-seed", type=int, default=None)
    parser.add_argument("--profiles", nargs="+", choices=["nominal", "train_nonstationary", "heldout_nonstationary"], default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--risk-neutral",
        action="store_true",
        help="deprecated alias for --decision-rule risk_neutral",
    )
    parser.add_argument(
        "--decision-rule", choices=["risk_neutral", "cvar"], default="risk_neutral",
        help="value-distribution decision rule; R2 defaults to risk_neutral",
    )
    parser.add_argument(
        "--cvar-promotion-gate-report", default=None,
        help="path to a hash-bound CVaR promotion report; "
             "required to use --decision-rule cvar in a formal-phase run",
    )
    parser.add_argument("--cvar-validation-result", default=None, help="validation result JSON bound into a CVaR promotion report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.smoke:
        if args.phase != "formal":
            raise SystemExit("smoke is only for engineering phase")
    elif not args.checkpoint or not args.artifact_path:
        raise SystemExit("non-smoke evaluation requires both --checkpoint and --artifact-path")
    # R3R-5 fix (2026-08-07, point 4): guide.md "主工程规则继续risk-neutral；
    # CVaR只有完整promotion gate通过才允许晋升" -- risk_neutral is already
    # the CLI default everywhere, but nothing previously stopped a formal
    # run from passing --decision-rule cvar without ever having cleared
    # check_cvar_promotion_gate. Validation-phase cvar runs remain
    # unrestricted (that's how the gate's own input evidence gets
    # produced in the first place).
    if args.decision_rule == "cvar" and args.phase == "formal" and not args.smoke:
        if not args.cvar_promotion_gate_report:
            raise SystemExit(
                "formal-phase --decision-rule cvar requires --cvar-promotion-gate-report "
                "pointing to a check_cvar_promotion_gate() result with passed=true"
            )
        if not args.cvar_validation_result:
            raise SystemExit("formal CVaR evaluation also requires --cvar-validation-result")
        gate_report = json.loads(_resolve_path(args.cvar_promotion_gate_report).read_text())
        registry_preview = load_and_validate_registry(str(_resolve_path(args.registry)), env_config_path=str(_resolve_path(args.env_config)))
        artifact_preview = SBKHMMArtifact.load(str(_resolve_path(args.artifact_path)), expect_tier="production")
        try:
            validate_cvar_promotion_report(
                gate_report,
                registry_sha256=registry_preview["content_sha256"],
                artifact_sha256=artifact_preview.content_hash(),
                checkpoint_sha256=sha256_of_file(str(_resolve_path(args.checkpoint))),
                validation_result_sha256=sha256_of_file(str(_resolve_path(args.cvar_validation_result))),
            )
        except Exception as exc:
            raise SystemExit(f"CVaR promotion gate rejected: {exc}") from exc
    if args.device == "auto":
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    else:
        device = args.device
    env_config_path = _resolve_path(args.env_config)
    registry_path = _resolve_path(args.registry)
    registry = load_and_validate_registry(str(registry_path), env_config_path=str(env_config_path))
    action_table = ActionGridSpec.from_env_config(str(env_config_path)).build_action_table()
    if args.smoke:
        artifact = _smoke_artifact()
        seeds = args.suite_seeds or [95001]
        episodes = args.episodes_per_seed or 1
        profiles = args.profiles or ["nominal"]
        scenarios = (args.scenarios or list(FORMAL_SCENARIOS.keys()))[:1]
        phase = "validation"
    else:
        artifact = SBKHMMArtifact.load(str(_resolve_path(args.artifact_path)), expect_tier="production")
        seeds = args.suite_seeds or list(registry["seed_roles"]["formal_test_suite_seeds"] if args.phase == "formal" else registry["seed_roles"]["checkpoint_validation_seeds"])
        episodes = args.episodes_per_seed or (100 if args.phase == "formal" else 100)
        profiles = args.profiles or (["nominal", "heldout_nonstationary"] if args.phase == "formal" else ["nominal", "train_nonstationary"])
        scenarios = args.scenarios or (["baseline_circle"] if args.phase == "validation" else list(FORMAL_SCENARIOS.keys()))
        phase = args.phase
        expected_profiles = {"validation": {"nominal", "train_nonstationary"}, "formal": {"nominal", "heldout_nonstationary"}}[phase]
        if set(profiles) != expected_profiles:
            raise SystemExit(f"{phase} evaluation requires exactly profiles={sorted(expected_profiles)}")
        if phase == "formal" and set(scenarios) != set(FORMAL_SCENARIOS):
            raise SystemExit("formal evaluation requires all six frozen scenarios")
        if phase == "formal" and len(seeds) != 10:
            raise SystemExit("formal evaluation requires the frozen 10 suite seeds unless an explicit audit mode is used")
    all_records = []
    calibration_rows = CalibrationAccumulator() if not args.smoke else None
    risk_neutral = args.risk_neutral or args.decision_rule == "risk_neutral"
    method_name = (
        "bayesian_dvl_risk_neutral"
        if risk_neutral and args.method == "bayesian_dvl"
        else args.method
    )
    # Local-only visibility improvement (2026-08-08, not synced to the
    # currently-running 4090 formal eval): evaluate_bdvl.py previously held
    # every episode in memory and wrote nothing until ALL scenarios
    # finished, so a long formal run gave zero mid-run signal and any
    # crash lost every already-computed episode. output_path is now
    # resolved before the loop so each episode can be appended to a JSONL
    # progress file immediately (same durable-append pattern as
    # select_bdvl_checkpoint.py's _append_progress), and a running
    # per-(seed,profile) tally is printed -- finer-grained than the
    # previous once-per-scenario-only print. The final CSV/manifest
    # write at the end of main() is unchanged and remains authoritative.
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(output_path) + ".progress.jsonl")
    if progress_path.exists():
        raise SystemExit(f"refusing to append to existing evaluator progress file: {progress_path}")

    def _append_episode_progress(record) -> None:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "scenario": record.scenario, "profile": record.profile,
                "suite_seed": record.suite_seed, "episode_seed": record.episode_seed,
                "outcome": record.outcome, "steps": record.steps,
                "n_records_so_far": len(all_records),
            }, sort_keys=True) + "\n")
            handle.flush()
            import os as _os
            _os.fsync(handle.fileno())

    # Live in-place progress bar (2026-08-08, local-only). Separate from
    # the JSONL/EVAL_PROGRESS lines above, which stay newline-terminated
    # so a redirected log file is still readable. Purely cosmetic terminal
    # feedback; does not affect what gets written to disk.
    #
    # Redraw is THROTTLED to at most once per _PROGRESS_BAR_MIN_INTERVAL
    # seconds (2026-08-08 fix: redrawing on every single episode produced
    # one `\r`-prefixed write per episode; some docker/conda pty stacks
    # do not collapse those into a true single in-place line and instead
    # render every write as its own segment -- with hundreds/thousands of
    # episodes that reproduces the exact "wall of text" the bar was meant
    # to prevent, independent of whether `\r` itself is honored). Always
    # draws on the very first and very last episode so 0% and 100% are
    # never skipped by the throttle.
    total_episodes = len(scenarios) * len(seeds) * len(profiles) * episodes
    eval_start_time = time.time()
    _PROGRESS_BAR_MIN_INTERVAL = 2.0
    _last_draw_time = [0.0]

    def _draw_progress_bar(scenario_key, suite_seed, profile, force: bool = False) -> None:
        completed = len(all_records)
        now = time.time()
        if not force and completed < total_episodes and (now - _last_draw_time[0]) < _PROGRESS_BAR_MIN_INTERVAL:
            return
        _last_draw_time[0] = now
        fraction = completed / total_episodes if total_episodes else 0.0
        bar_width = 30
        filled = int(bar_width * fraction)
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = now - eval_start_time
        eta = (elapsed / completed * (total_episodes - completed)) if completed > 0 else 0.0
        sys.stdout.write(
            f"\r[{bar}] {fraction * 100:5.1f}% ({completed}/{total_episodes}) "
            f"{scenario_key} seed={suite_seed} profile={profile} "
            f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m   "
        )
        sys.stdout.flush()

    for scenario_key in scenarios:
        if scenario_key not in FORMAL_SCENARIOS:
            raise SystemExit(f"unknown formal scenario {scenario_key}")
        env, env_config_for_robot = _make_env(env_config_path, FORMAL_SCENARIOS[scenario_key]["humans"], scenario_key)
        for suite_seed in seeds:
            for profile in profiles:
                policy = _make_policy(
                    action_table, artifact,
                    str(_resolve_path(args.checkpoint)) if args.checkpoint else None,
                    registry, args.smoke, suite_seed, phase, device,
                    risk_neutral=risk_neutral,
                    collect_calibration=calibration_rows is not None,
                )
                robot = Robot(env_config_for_robot, "robot")
                robot.set_policy(policy); robot.visible = True; robot.time_step = FROZEN_VALUES["dt"]; policy.time_step = FROZEN_VALUES["dt"]; robot.env = env; env.set_robot(robot)
                seed_profile_outcomes = {"success": 0, "collision": 0, "timeout": 0}
                for episode_index in range(episodes):
                    record = run_episode(
                        policy, env, suite_seed, episode_index, method_name,
                        scenario_key, profile, FROZEN_VALUES["dt"], calibration_rows,
                    )
                    all_records.append(record)
                    seed_profile_outcomes[record.outcome] += 1
                    _append_episode_progress(record)
                    _draw_progress_bar(
                        scenario_key, suite_seed, profile,
                        force=(len(all_records) in (1, total_episodes)),
                    )
                n = sum(seed_profile_outcomes.values())
                sr = seed_profile_outcomes["success"] / n
                print(
                    f"\nEVAL_PROGRESS {scenario_key} seed={suite_seed} profile={profile} "
                    f"episodes={n} sr={sr:.3f} outcomes={seed_profile_outcomes} "
                    f"total_so_far={len(all_records)}", flush=True,
                )
        print(f"{scenario_key}: {len(seeds) * len(profiles) * episodes} episodes done")
    _assert_complete(all_records, scenarios, profiles, seeds, episodes)
    write_episode_records_csv(all_records, str(output_path) + ".csv")
    calibration_metrics = calibration_rows.summary() if calibration_rows is not None else {"n_decisions": 0, "status": "NOT_AVAILABLE"}
    # R3R-5/R3RF-4: iterate the complete expected profile x outcome grid;
    # missing strata remain explicit INSUFFICIENT records.
    stratified_metrics = calibration_rows.stratified_summary(profiles) if calibration_rows is not None else {}
    calibration_payload = {
        "decision_rule": "risk_neutral" if risk_neutral else "cvar",
        "metrics": calibration_metrics,
        "stratified_metrics": stratified_metrics,
        "audit_rows": calibration_rows.audit_rows if calibration_rows is not None else [],
        "audit_row_limit": calibration_rows.audit_limit if calibration_rows is not None else 0,
    }
    calibration_path = Path(str(output_path) + ".calibration.json")
    atomic_write_json(str(calibration_path), calibration_payload)
    manifest = build_run_manifest(repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv), source_files=BDVL_PRODUCTION_SOURCES, extra={"mode": "smoke" if args.smoke else phase, "method": method_name, "decision_rule": "risk_neutral_iqn_mean" if risk_neutral else "lower_tail_iqn_cvar", "risk_neutral": bool(risk_neutral), "n_episodes": len(all_records), "scenarios": scenarios, "profiles": profiles, "suite_seeds": seeds, "episodes_per_seed": episodes, "registry_sha256": sha256_of_file(str(registry_path)), "env_config_sha256": sha256_of_file(str(env_config_path)), "artifact_sha256": artifact.content_sha256(), "checkpoint_sha256": sha256_of_file(str(_resolve_path(args.checkpoint))) if args.checkpoint else None, "csv_sha256": sha256_of_file(str(output_path) + ".csv"), "calibration_sha256": sha256_of_file(str(calibration_path)), "calibration_metrics": calibration_metrics, "deterministic_records_sha256": deterministic_records_sha256(all_records), "latency_is_wall_clock": True})
    atomic_write_json(str(output_path) + ".manifest.json", manifest)
    # Local-only fix (2026-08-08): the run finished without ever printing
    # the OVERALL aggregate SR/CR/TR across every scenario/seed/profile --
    # only the per-(seed,profile) EVAL_PROGRESS lines existed, so the very
    # last one (whichever scenario/seed/profile happened to run last) was
    # easy to mistake for the overall result. Compute and print the true
    # aggregate here, over all_records exactly as written to the CSV.
    _outcome_counts = {"success": 0, "collision": 0, "timeout": 0}
    for _r in all_records:
        _outcome_counts[_r.outcome] = _outcome_counts.get(_r.outcome, 0) + 1
    _n = len(all_records)
    print(
        f"EVALUATE_BDVL_OVERALL n_episodes={_n} "
        f"sr={_outcome_counts['success'] / _n:.4f} "
        f"cr={_outcome_counts['collision'] / _n:.4f} "
        f"tr={_outcome_counts['timeout'] / _n:.4f} "
        f"outcomes={_outcome_counts}"
    )
    print(f"EVALUATE_BDVL_DONE n_episodes={len(all_records)} mode={manifest['mode']}")


if __name__ == "__main__":
    main()
