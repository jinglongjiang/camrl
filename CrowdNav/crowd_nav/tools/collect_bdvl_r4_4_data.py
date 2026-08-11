#!/usr/bin/env python3
"""R4-4R2 training-data collection with counterfactual labels + a
population-aligned belief-relevant gate (guide.md "R4-4R2冻结契约",
2026-08-10).

History: R4-4's original gate required a purely-reciprocal ORCA teacher
to genuinely collide -- not meaningful, since ORCA is specifically
designed not to -- and broadcast episode-level outcome/clearance tags
onto every one of that episode's per-decision records, letting one
collision episode trivially clear a >=5 threshold. R4-4R fixed both,
replacing it with an episode-level (informational) tier and a
risk-opportunity-level (real) tier, plus a first oracle-regret check
comparing full/mean/cv's WHOLE-SCENE top-1 choice against a truth
oracle built from the REAL recorded future of the episode's
non-reciprocal humans. Real R4-4R pilot data showed that oracle regret
was consistently ~0 -- diagnosis found this was a POPULATION MISMATCH,
not evidence belief is useless: full/mean/cv chose using all 5 humans,
but the oracle could only grade against the non-reciprocal subset (20-
50% of humans), so a real difference in the whole-scene choice could be
invisible to a subset-only grader whenever the subset wasn't near
either candidate's path.

R4-4R2 fixes this: for the population the truth oracle CAN validly
grade (the non-reciprocal subset, since only their future is action-
invariant), full/mean/cv are RE-RANKED using ONLY that subset, then
compared against the oracle's own full ranking of the SAME subset (see
``oracle_regret.py``). The original whole-scene ``RiskOpportunityRecord``/
``check_risk_opportunity_gate`` (coverage, disagreement rate, action-type
diversity, world-sample stability) are unaffected by this mismatch and
remain as-is.

Scope, per guide.md's explicit R4-4/R4-4R/R4-4R2 freeze:

1. Only 5-person baseline_circle -- never the six formal test scenarios.
2. A training-only non-reciprocal profile lives in
   ``crowd_nav.bayesian_pilot.protocol`` with a fraction range disjoint
   from ``heldout_non_reciprocal``'s.
3. Collects a NATURAL mix of normal / risk-opportunity / near-collision
   / failure states -- never an adversarially-filtered "only hard
   samples" set.
4. Verifies the registry's ``action_grid_hash`` against a FRESH
   recomputation from the env config BEFORE writing any counterfactual
   record.
5. Outputs an episode-level report (informational), a risk-opportunity-
   level report (coverage/disagreement/diversity), and a population-
   aligned oracle-regret audit report (real decision-value gate) -- see
   ``data_coverage.py`` and ``oracle_regret.py``.
6. Collects data and reports gates ONLY -- does not start any training
   (guide.md: "R4-4只完成数据采集与覆盖门禁，不启动正式训练").
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: E402
from crowd_sim.envs.utils.robot import Robot  # noqa: E402
from crowd_sim.envs.policy.orca import ORCA  # noqa: E402

from crowd_nav.bayesian_dvl.config import (  # noqa: E402
    ActionGridSpec, BDVL_PRODUCTION_SOURCES, FORMAL_SCENARIOS, FROZEN_VALUES, load_and_validate_registry,
)
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact  # noqa: E402
from crowd_nav.bayesian_dvl.belief import BeliefTracker  # noqa: E402
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation  # noqa: E402
from crowd_nav.bayesian_dvl.transition import RewardConfig  # noqa: E402
from crowd_nav.bayesian_dvl.counterfactual import (  # noqa: E402
    COUNTERFACTUAL_CONTRACT_V1, CounterfactualCandidateResult, CounterfactualRecord, HumanMultiStepTrajectory,
    ROLLOUT_HORIZON, build_counterfactual_record, evaluate_counterfactual_candidates,
)
from crowd_nav.bayesian_dvl.data_coverage import (  # noqa: E402
    EpisodeCoverageRecord, RiskOpportunityRecord, check_episode_gate, check_risk_opportunity_gate,
    check_world_sample_stability, classify_action_type, classify_belief_entropy, classify_clearance,
    compute_episode_coverage_report, compute_risk_opportunity_report, is_risk_opportunity_state, topk_overlap,
)
from crowd_nav.bayesian_dvl.oracle_regret import (  # noqa: E402
    AuditRecord, check_oracle_regret_gate, clearance_equivalence_tolerance, compute_method_regret,
)
from crowd_nav.bayesian_pilot.protocol import (  # noqa: E402
    BehaviorScheduler, InterventionORCA, NON_RECIPROCAL_PROFILES, PROFILES, assign_non_reciprocal_flags,
)
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest, sha256_of_file  # noqa: E402

TRAIN_PROFILES = ("nominal", "train_nonstationary", "train_non_reciprocal")
N_MODES_FOR_ENTROPY = 5  # world_model.N_MODES; avoided importing to keep this tool decoupled from artifact internals


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def _make_env(env_config_path: Path, human_num: int):
    if env_config_path.name != "env_bayesian_dvl.config":
        raise SystemExit("R4-4 collection must use env_bayesian_dvl.config")
    env_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    if not env_config.read(str(env_config_path)):
        raise SystemExit(f"env config not found: {env_config_path}")
    scenario = FORMAL_SCENARIOS["baseline_circle"]
    if scenario["layout"] != "circle" or scenario["humans"] != human_num:
        raise SystemExit("guide.md R4-4 point 1: collection must use the 5-person baseline_circle scenario exactly")
    env_config.set("sim", "human_num", str(human_num))
    env_config.set("robot", "policy", "orca")
    env_config.set("sim", "train_val_sim", "circle_crossing")
    env_config.set("sim", "test_sim", "circle_crossing")
    env_config.set("sim", "circle_radius", str(scenario["radius"]))
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


def _max_environment_steps() -> int:
    return int(round(float(FROZEN_VALUES["time_limit"]) / float(FROZEN_VALUES["dt"]))) + 1


def _build_oracle_candidate_results(
    step_index: int,
    n_steps: int,
    non_reciprocal_ids: List[int],
    real_velocity_trace: List[List[Tuple[float, float]]],
    robot_obs: RobotObservation,
    subset_humans: List[HumanObservation],
    action_table,
    reward_config: RewardConfig,
    global_time: float,
) -> Optional[Tuple[CounterfactualCandidateResult, ...]]:
    """guide.md R4-4R (fix carried into R4-4R2): replay the REAL recorded
    future of ``non_reciprocal_ids`` against all 80 candidates -- a
    non-circular oracle, since those humans' own ORCA never looks at the
    robot, so their real future is identical no matter which candidate
    the robot would have taken. Returns None when fewer than
    ROLLOUT_HORIZON real steps remain.

    ``real_velocity_trace[k][tid]`` is the velocity captured at the TOP
    of loop iteration k (i.e. AFTER iteration k-1's ``env.step()`` set
    it) -- crowd_sim's holonomic ``Agent.step()`` sets ``self.vx, self.vy
    = action.vx, action.vy`` at the SAME time it applies that action to
    move the agent, so the velocity that actually drove the position
    transition FROM step k TO step k+1 is recorded at index k+1, not k.
    Using index k directly (unshifted) would silently replay each
    human's PREVIOUS interval's motion one step early.
    """
    if not non_reciprocal_ids:
        return None
    if step_index + ROLLOUT_HORIZON >= n_steps:
        return None

    human_worlds_oracle = {
        tid: (HumanMultiStepTrajectory(
            track_id=tid, modes=(None,) * ROLLOUT_HORIZON,
            step_velocities=tuple(real_velocity_trace[step_index + t + 1][tid] for t in range(ROLLOUT_HORIZON)),
        ),)
        for tid in non_reciprocal_ids
    }
    return evaluate_counterfactual_candidates(
        robot=robot_obs, humans=subset_humans, human_worlds=human_worlds_oracle, n_samples=1,
        action_table=action_table, reward_config=reward_config, dt=FROZEN_VALUES["dt"],
        time_limit=FROZEN_VALUES["time_limit"], global_time=global_time, cvar_alpha=1.0, horizon=ROLLOUT_HORIZON,
    )


def collect_episode(
    env_config_path: Path, artifact: SBKHMMArtifact, action_table, registry, episode_seed: int,
    profile_name: str, reward_config: RewardConfig, n_world_samples: int, clearance_tolerance: float,
) -> Tuple[EpisodeCoverageRecord, List[RiskOpportunityRecord], List[CounterfactualRecord], List[dict], List[AuditRecord], Dict[str, int], str]:
    if profile_name not in TRAIN_PROFILES:
        raise ValueError(f"unknown R4-4 training profile {profile_name!r} -- must be one of {TRAIN_PROFILES}")
    env, env_config = _make_env(env_config_path, human_num=5)
    robot = Robot(env_config, "robot")
    orca = ORCA(); orca.configure(env_config)
    robot.set_policy(orca); robot.visible = True; robot.time_step = FROZEN_VALUES["dt"]
    robot.env = env
    env.set_robot(robot)
    env.case_counter["train"] = episode_seed % (2**32 - 1)
    env.reset()

    is_non_reciprocal_profile = profile_name == "train_non_reciprocal"
    # guide.md R4-4R2 root-cause fix (2026-08-10): train_non_reciprocal
    # used to force intervention_profile_name to "nominal"
    # (event_rate=0.0), so the ONLY humans the oracle can validly grade
    # (non-reciprocal ones, since only their future is action-invariant
    # ground truth) never had a single genuine scripted mode-switch
    # event -- the belief tracker had nothing real to anticipate for
    # that population, which alone is sufficient to explain a null/
    # negative oracle-regret result regardless of belief quality.
    # Non-reciprocity and nonstationary intervention are orthogonal
    # (InterventionORCA.predict() applies them independently: drop the
    # robot from the neighbor list, THEN apply the scheduler's mode
    # override), so non-reciprocal humans now ALSO run genuine
    # train_nonstationary-rate scripted events instead of none at all.
    intervention_profile_name = "train_nonstationary" if is_non_reciprocal_profile else profile_name
    scheduler = BehaviorScheduler(PROFILES[intervention_profile_name], seed=episode_seed)
    scheduler.reset(len(env.humans))

    human_policies = []
    for h in env.humans:
        hp = InterventionORCA(env_config); hp.time_step = env.time_step
        h.set_policy(hp)
        human_policies.append(hp)

    if is_non_reciprocal_profile:
        reciprocity_flags = assign_non_reciprocal_flags(
            len(env.humans), NON_RECIPROCAL_PROFILES["train_non_reciprocal"], seed=episode_seed,
        )
        for hp, flag in zip(human_policies, reciprocity_flags):
            hp.set_reciprocity(flag)
    else:
        reciprocity_flags = tuple(False for _ in env.humans)
    is_non_reciprocal_episode = any(reciprocity_flags)
    non_reciprocal_ids = [i for i, flag in enumerate(reciprocity_flags) if flag]

    tracker = BeliefTracker(artifact)
    global_time = 0.0
    step_states: List[dict] = []
    real_velocity_trace: List[List[Tuple[float, float]]] = []
    counterfactual_records: List[CounterfactualRecord] = []
    min_human_dist_episode = float("inf")
    event = None
    max_entropy = float(np.log(N_MODES_FOR_ENTROPY))

    for step_index in range(_max_environment_steps()):
        humans = [
            HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
            for i, h in enumerate(env.humans)
        ]
        robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
        tracker.update({h.track_id: (global_time, np.array([h.px, h.py])) for h in humans})
        scheduler.advance([h.policy for h in env.humans])

        min_dist_step = min(
            (float(np.hypot(h.px - env.robot.px, h.py - env.robot.py)) - h.radius - env.robot.radius for h in env.humans),
            default=float("inf"),
        )

        action = env.robot.act([h.get_observable_state() for h in env.humans])

        track_beliefs = {h.track_id: tracker.belief_for(h.track_id) for h in humans}
        track_positions = {h.track_id: np.array([h.px, h.py]) for h in humans}
        track_speeds = {h.track_id: float(np.hypot(h.vx, h.vy)) for h in humans}
        track_headings = {h.track_id: float(np.arctan2(h.vy, h.vx)) for h in humans}
        cf_record_full = build_counterfactual_record(
            suite_seed=0, episode_seed=episode_seed, decision_seed=step_index + 1, artifact=artifact,
            action_grid_hash=registry["action_grid_hash"], robot=robot_obs, humans=humans,
            track_beliefs=track_beliefs, track_positions=track_positions, track_speeds=track_speeds,
            track_headings=track_headings, action_table=action_table, reward_config=reward_config,
            dt=FROZEN_VALUES["dt"], time_limit=FROZEN_VALUES["time_limit"], global_time=global_time,
            max_human_speed=FROZEN_VALUES["max_human_speed"], n_world_samples=n_world_samples,
            cvar_alpha=FROZEN_VALUES["cvar_alpha"], clearance_tolerance=clearance_tolerance, posterior_source="full",
        )
        counterfactual_records.append(cf_record_full)

        step_entropy = max((tracker.entropy_for(h.track_id) for h in humans), default=0.0)
        is_risk = is_risk_opportunity_state(cf_record_full.candidate_results, FROZEN_VALUES["discomfort_distance"])

        step_states.append({
            "step_index": step_index, "robot_obs": robot_obs, "humans": humans,
            "track_beliefs": track_beliefs, "track_positions": track_positions,
            "track_speeds": track_speeds, "track_headings": track_headings, "global_time": global_time,
            "cf_record_full": cf_record_full, "is_risk": is_risk,
            "belief_entropy_bin": classify_belief_entropy(step_entropy, max_entropy),
        })
        real_velocity_trace.append([(float(h.vx), float(h.vy)) for h in humans])

        min_human_dist_episode = min(min_human_dist_episode, min_dist_step)
        _, _reward, terminated, truncated, info = env.step(action)
        global_time += FROZEN_VALUES["dt"]
        if terminated or truncated:
            event = info.get("event")
            break

    if event not in {"reach_goal", "collision", "timeout"}:
        raise RuntimeError(f"episode did not return a valid terminal event after {_max_environment_steps()} steps")
    outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}[event]
    clearance_bin = classify_clearance(min_human_dist_episode, FROZEN_VALUES["discomfort_distance"])
    episode_record = EpisodeCoverageRecord(
        episode_seed=episode_seed, profile=profile_name, outcome=outcome,
        is_non_reciprocal_episode=is_non_reciprocal_episode, clearance_bin=clearance_bin,
    )

    n_steps = len(step_states)
    risk_records: List[RiskOpportunityRecord] = []
    risk_contexts: List[dict] = []  # parallel to risk_records; carries what the stability re-check needs
    for state in step_states:
        if not state["is_risk"]:
            continue
        decision_seed = state["step_index"] + 1
        full_top = state["cf_record_full"].ranked_action_indices[0]
        avx, avy = action_table[full_top]
        action_type = classify_action_type(float(avx), float(avy), (state["robot_obs"].vx, state["robot_obs"].vy))

        mean_record = build_counterfactual_record(
            suite_seed=0, episode_seed=episode_seed, decision_seed=decision_seed, artifact=artifact,
            action_grid_hash=registry["action_grid_hash"], robot=state["robot_obs"], humans=state["humans"],
            track_beliefs=state["track_beliefs"], track_positions=state["track_positions"],
            track_speeds=state["track_speeds"], track_headings=state["track_headings"], action_table=action_table,
            reward_config=reward_config, dt=FROZEN_VALUES["dt"], time_limit=FROZEN_VALUES["time_limit"],
            global_time=state["global_time"], max_human_speed=FROZEN_VALUES["max_human_speed"],
            n_world_samples=n_world_samples, cvar_alpha=FROZEN_VALUES["cvar_alpha"],
            clearance_tolerance=clearance_tolerance, posterior_source="moment_mean",
        )
        cv_record = build_counterfactual_record(
            suite_seed=0, episode_seed=episode_seed, decision_seed=decision_seed, artifact=artifact,
            action_grid_hash=registry["action_grid_hash"], robot=state["robot_obs"], humans=state["humans"],
            track_beliefs=state["track_beliefs"], track_positions=state["track_positions"],
            track_speeds=state["track_speeds"], track_headings=state["track_headings"], action_table=action_table,
            reward_config=reward_config, dt=FROZEN_VALUES["dt"], time_limit=FROZEN_VALUES["time_limit"],
            global_time=state["global_time"], max_human_speed=FROZEN_VALUES["max_human_speed"],
            n_world_samples=n_world_samples, cvar_alpha=FROZEN_VALUES["cvar_alpha"],
            clearance_tolerance=clearance_tolerance, posterior_source="cv",
        )
        mean_top = mean_record.ranked_action_indices[0]
        cv_top = cv_record.ranked_action_indices[0]
        cand_full = {c.action_index: c for c in state["cf_record_full"].candidate_results}
        internal_regret_vs_mean = cand_full[mean_top].collision_prob - cand_full[full_top].collision_prob
        internal_regret_vs_cv = cand_full[cv_top].collision_prob - cand_full[full_top].collision_prob

        risk_records.append(RiskOpportunityRecord(
            episode_seed=episode_seed, decision_seed=decision_seed, profile=profile_name,
            belief_entropy_bin=state["belief_entropy_bin"], action_type=action_type,
            full_top_action=full_top, mean_top_action=mean_top, cv_top_action=cv_top,
            agree_with_mean=(full_top == mean_top), agree_with_cv=(full_top == cv_top),
            internal_regret_vs_mean=internal_regret_vs_mean, internal_regret_vs_cv=internal_regret_vs_cv,
        ))
        risk_contexts.append(state)

    # guide.md R4-4R2: population-aligned audit, independent of the
    # whole-scene risk flag above -- full/mean/cv are RE-RANKED using
    # ONLY the non-reciprocal subset, then graded against the oracle's
    # own ranking of that SAME subset. Filtering funnel counted
    # explicitly per guide.md R4-4R2 point 9.
    funnel = {"n_decisions": n_steps, "n_nonreciprocal_eligible": 0, "n_subset_risk_opportunity": 0, "n_audit_eligible": 0}
    audit_records: List[AuditRecord] = []
    if is_non_reciprocal_profile and non_reciprocal_ids:
        funnel["n_nonreciprocal_eligible"] = n_steps
        for state in step_states:
            subset_beliefs = {tid: state["track_beliefs"][tid] for tid in non_reciprocal_ids}
            subset_positions = {tid: state["track_positions"][tid] for tid in non_reciprocal_ids}
            subset_speeds = {tid: state["track_speeds"][tid] for tid in non_reciprocal_ids}
            subset_headings = {tid: state["track_headings"][tid] for tid in non_reciprocal_ids}
            subset_humans = [h for h in state["humans"] if h.track_id in non_reciprocal_ids]
            decision_seed = state["step_index"] + 1

            subset_records = {}
            for source in ("full", "moment_mean", "cv"):
                subset_records[source] = build_counterfactual_record(
                    suite_seed=0, episode_seed=episode_seed, decision_seed=decision_seed, artifact=artifact,
                    action_grid_hash=registry["action_grid_hash"], robot=state["robot_obs"], humans=subset_humans,
                    track_beliefs=subset_beliefs, track_positions=subset_positions, track_speeds=subset_speeds,
                    track_headings=subset_headings, action_table=action_table, reward_config=reward_config,
                    dt=FROZEN_VALUES["dt"], time_limit=FROZEN_VALUES["time_limit"], global_time=state["global_time"],
                    max_human_speed=FROZEN_VALUES["max_human_speed"], n_world_samples=n_world_samples,
                    cvar_alpha=FROZEN_VALUES["cvar_alpha"], clearance_tolerance=clearance_tolerance,
                    posterior_source=source,
                )
            subset_is_risk = is_risk_opportunity_state(
                subset_records["full"].candidate_results, FROZEN_VALUES["discomfort_distance"],
            )
            if not subset_is_risk:
                continue
            funnel["n_subset_risk_opportunity"] += 1

            oracle_results = _build_oracle_candidate_results(
                state["step_index"], n_steps, non_reciprocal_ids, real_velocity_trace, state["robot_obs"],
                subset_humans, action_table, reward_config, state["global_time"],
            )
            if oracle_results is None:
                continue
            funnel["n_audit_eligible"] += 1

            audit_records.append(AuditRecord(
                episode_seed=episode_seed, decision_seed=decision_seed, profile=profile_name,
                full=compute_method_regret(subset_records["full"].ranked_action_indices[0], oracle_results, clearance_tolerance),
                mean=compute_method_regret(subset_records["moment_mean"].ranked_action_indices[0], oracle_results, clearance_tolerance),
                cv=compute_method_regret(subset_records["cv"].ranked_action_indices[0], oracle_results, clearance_tolerance),
            ))

    return episode_record, risk_records, counterfactual_records, risk_contexts, audit_records, funnel, outcome


def _run_stability_check(
    risk_contexts: List[dict], episode_seeds: List[int], artifact, action_table, registry, reward_config,
    clearance_tolerance: float, max_checks: int = 20,
) -> Dict[str, object]:
    """guide.md R4-4R point 6: rerun a subsample of risk-opportunity
    decisions with n_world_samples=32 vs 128, compare top-3 full-
    posterior overlap. Reuses the SAME seed_key for both runs -- the
    per-track RNG is seeded independently of n_samples (guide.md R4-3R),
    so the 128-sample run extends the SAME draw stream the 32-sample run
    used, making this a genuine "does the estimate stabilize as budget
    grows" check, not two independently-reseeded runs."""
    n = min(max_checks, len(risk_contexts))
    if n == 0:
        return {"passed": False, "reasons": ["no risk-opportunity states available for stability check"], "n_checked": 0}
    step_indices = np.linspace(0, len(risk_contexts) - 1, num=n, dtype=int)
    overlaps = []
    for idx in step_indices:
        ctx = risk_contexts[int(idx)]
        episode_seed = episode_seeds[int(idx)]
        decision_seed = ctx["step_index"] + 1
        common = dict(
            suite_seed=0, episode_seed=episode_seed, decision_seed=decision_seed, artifact=artifact,
            action_grid_hash=registry["action_grid_hash"], robot=ctx["robot_obs"], humans=ctx["humans"],
            track_beliefs=ctx["track_beliefs"], track_positions=ctx["track_positions"],
            track_speeds=ctx["track_speeds"], track_headings=ctx["track_headings"], action_table=action_table,
            reward_config=reward_config, dt=FROZEN_VALUES["dt"], time_limit=FROZEN_VALUES["time_limit"],
            global_time=ctx["global_time"], max_human_speed=FROZEN_VALUES["max_human_speed"],
            cvar_alpha=FROZEN_VALUES["cvar_alpha"], clearance_tolerance=clearance_tolerance, posterior_source="full",
        )
        rec_32 = build_counterfactual_record(n_world_samples=32, **common)
        rec_128 = build_counterfactual_record(n_world_samples=128, **common)
        overlaps.append(topk_overlap(rec_32.ranked_action_indices, rec_128.ranked_action_indices, k=3))
    return check_world_sample_stability(overlaps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-config", default="crowd_nav/configs/env_bayesian_dvl.config")
    parser.add_argument("--registry", default="crowd_nav/configs/bayesian_dvl_registry_r4.json")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--n-episodes", type=int, default=60)
    parser.add_argument("--episode-seed-base", type=int, default=700001)
    parser.add_argument("--n-world-samples", type=int, default=None, help="defaults to registry's world_samples_validation")
    parser.add_argument("--stability-checks", type=int, default=20, help="number of risk-opportunity states re-checked at n=32 vs n=128")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    env_config_path = _resolve_path(args.env_config)
    registry_path = _resolve_path(args.registry)
    registry = load_and_validate_registry(str(registry_path), env_config_path=str(env_config_path))

    grid = ActionGridSpec.from_env_config(str(env_config_path))
    action_table = grid.build_action_table()
    fresh_hash = grid.table_hash()
    if fresh_hash != registry["action_grid_hash"]:
        raise SystemExit(
            f"action_grid_hash mismatch: env config gives {fresh_hash!r}, registry says "
            f"{registry['action_grid_hash']!r} -- refusing to collect data under a mismatched action grid"
        )

    artifact = SBKHMMArtifact.load(str(_resolve_path(args.artifact_path)), expect_tier="production")
    reward_config = _make_reward_config()
    n_world_samples = args.n_world_samples or registry["frozen_values"]["world_samples_validation"]
    clearance_tolerance = clearance_equivalence_tolerance(action_table, FROZEN_VALUES["dt"], ROLLOUT_HORIZON)

    episode_records: List[EpisodeCoverageRecord] = []
    all_risk_records: List[RiskOpportunityRecord] = []
    all_risk_contexts: List[dict] = []
    all_risk_episode_seeds: List[int] = []
    all_counterfactual_records: List[CounterfactualRecord] = []
    all_audit_records: List[AuditRecord] = []
    funnel_total = {"n_decisions": 0, "n_nonreciprocal_eligible": 0, "n_subset_risk_opportunity": 0, "n_audit_eligible": 0}
    episode_manifest_rows = []
    for i in range(args.n_episodes):
        episode_seed = args.episode_seed_base + i
        profile_name = TRAIN_PROFILES[i % len(TRAIN_PROFILES)]
        episode_record, risk_records, cf_records, risk_contexts, audit_records, funnel, outcome = collect_episode(
            env_config_path, artifact, action_table, registry, episode_seed, profile_name,
            reward_config, n_world_samples, clearance_tolerance,
        )
        episode_records.append(episode_record)
        all_risk_records.extend(risk_records)
        all_risk_contexts.extend(risk_contexts)
        all_risk_episode_seeds.extend([episode_seed] * len(risk_contexts))
        all_counterfactual_records.extend(cf_records)
        all_audit_records.extend(audit_records)
        for k in funnel_total:
            funnel_total[k] += funnel[k]
        episode_manifest_rows.append({
            "episode_seed": episode_seed, "profile": profile_name, "outcome": outcome,
            "n_decisions": len(cf_records), "n_risk_opportunity_states": len(risk_records),
            "n_audit_states": len(audit_records),
        })
        print(f"R4_4_EPISODE_DONE seed={episode_seed} profile={profile_name} outcome={outcome} "
              f"n_decisions={len(cf_records)} n_risk_states={len(risk_records)} n_audit_states={len(audit_records)} "
              f"total_risk_so_far={len(all_risk_records)} total_audit_so_far={len(all_audit_records)}", flush=True)

    episode_report = compute_episode_coverage_report(episode_records)
    episode_gate = check_episode_gate(episode_report, required_profiles=TRAIN_PROFILES)

    if all_risk_records:
        risk_report = compute_risk_opportunity_report(all_risk_records)
        risk_gate = check_risk_opportunity_gate(all_risk_records, required_profiles=TRAIN_PROFILES)
    else:
        risk_report = {"n_states": 0}
        risk_gate = {"passed": False, "reasons": ["no risk-opportunity states were observed at all"]}

    stability_result = _run_stability_check(
        all_risk_contexts, all_risk_episode_seeds, artifact, action_table, registry, reward_config,
        clearance_tolerance, max_checks=args.stability_checks,
    )

    oracle_gate = check_oracle_regret_gate(all_audit_records)

    overall_passed = bool(episode_gate["passed"] and risk_gate["passed"] and stability_result["passed"] and oracle_gate["passed"])

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_path = Path(str(output_path) + ".counterfactual_records.jsonl")
    if records_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {records_path}")
    with records_path.open("w") as fh:
        for rec in all_counterfactual_records:
            fh.write(json.dumps({
                "contract_version": rec.contract_version, "suite_seed": rec.suite_seed,
                "episode_seed": rec.episode_seed, "decision_seed": rec.decision_seed,
                "artifact_sha256": rec.artifact_sha256, "action_grid_hash": rec.action_grid_hash,
                "posterior_source": rec.posterior_source, "horizon": rec.horizon,
                "n_world_samples": rec.n_world_samples, "cvar_alpha": rec.cvar_alpha,
                "ranked_action_indices": list(rec.ranked_action_indices),
                "top_action_result": {
                    "action_index": rec.candidate_results[rec.ranked_action_indices[0]].action_index,
                    "collision_prob": rec.candidate_results[rec.ranked_action_indices[0]].collision_prob,
                    "lower_tail_clearance": rec.candidate_results[rec.ranked_action_indices[0]].lower_tail_clearance,
                    "expected_progress": rec.candidate_results[rec.ranked_action_indices[0]].expected_progress,
                    "control_cost": rec.candidate_results[rec.ranked_action_indices[0]].control_cost,
                },
            }, sort_keys=True) + "\n")

    audit_records_path = Path(str(output_path) + ".audit_records.jsonl")
    with audit_records_path.open("w") as fh:
        for rec in all_audit_records:
            def _mr(m):
                return {
                    "collision_regret": m.collision_regret, "clearance_regret": m.clearance_regret,
                    "efficiency_regret": m.efficiency_regret, "normalized_rank_regret": m.normalized_rank_regret,
                    "in_safety_optimal_layer": m.in_safety_optimal_layer,
                }
            fh.write(json.dumps({
                "episode_seed": rec.episode_seed, "decision_seed": rec.decision_seed, "profile": rec.profile,
                "full": _mr(rec.full), "mean": _mr(rec.mean), "cv": _mr(rec.cv),
            }, sort_keys=True) + "\n")

    summary = {
        "protocol": "r4_4r2_population_aligned_oracle_regret_audit",
        "n_episodes": args.n_episodes, "n_decisions": len(all_counterfactual_records),
        "profiles": list(TRAIN_PROFILES), "episode_manifest": episode_manifest_rows,
        "episode_report": episode_report, "episode_gate": episode_gate,
        "risk_opportunity_report": risk_report, "risk_opportunity_gate": risk_gate,
        "stability_check": stability_result,
        "audit_filter_funnel": funnel_total, "oracle_regret_gate": oracle_gate,
        "clearance_equivalence_tolerance": clearance_tolerance,
        "overall_passed": overall_passed,
        "counterfactual_contract_version": COUNTERFACTUAL_CONTRACT_V1,
        "action_grid_hash": registry["action_grid_hash"], "registry_content_sha256": registry["content_sha256"],
        "artifact_sha256": artifact.content_sha256(), "n_world_samples": n_world_samples,
        "records_path": str(records_path), "audit_records_path": str(audit_records_path),
    }
    atomic_write_json(str(output_path) + ".summary.json", summary)
    manifest = build_run_manifest(
        repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv), source_files=BDVL_PRODUCTION_SOURCES, extra=summary,
    )
    atomic_write_json(str(output_path) + ".manifest.json", manifest)

    print(f"\nR4_4R_EPISODE_REPORT {json.dumps(episode_report, sort_keys=True)}")
    print(f"R4_4R_EPISODE_GATE passed={episode_gate['passed']} reasons={episode_gate['reasons']}")
    print(f"R4_4R_RISK_OPPORTUNITY_REPORT {json.dumps(risk_report, sort_keys=True)}")
    print(f"R4_4R_RISK_OPPORTUNITY_GATE passed={risk_gate['passed']} reasons={risk_gate['reasons']}")
    print(f"R4_4R_STABILITY_CHECK {json.dumps(stability_result, sort_keys=True)}")
    print(f"R4_4R2_AUDIT_FUNNEL {json.dumps(funnel_total, sort_keys=True)}")
    print(f"R4_4R2_ORACLE_REGRET_GATE passed={oracle_gate['passed']} reasons={oracle_gate['reasons']}")
    print(f"R4_4R2_ORACLE_REGRET_METRICS {json.dumps(oracle_gate.get('metrics', {}), sort_keys=True)}")
    print(f"R4_4R_OVERALL_GATE passed={overall_passed}")
    print(f"R4_4_COLLECT_DONE n_episodes={args.n_episodes} n_decisions={len(all_counterfactual_records)} "
          f"summary={output_path}.summary.json")


if __name__ == "__main__":
    main()
