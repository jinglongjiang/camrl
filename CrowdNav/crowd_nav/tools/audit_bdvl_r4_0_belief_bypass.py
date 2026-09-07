#!/usr/bin/env python3
"""R4-0 (guide.md "R4 -- Belief-Bypass Remediation Plan"): root-cause
closure audit, analysis only. No training, no core-code changes.

Runs real dense_square / heldout_nonstationary episodes with the R3
ep500 checkpoint, capturing the belief-tracker snapshot at every real
decision. For each captured decision, records immediate_reward,
continuation_value and total_score (per action, all 80) under four
conditions:

  full               -- real belief, real world sampler, real reward, real encoder input
  belief_neutral     -- belief forced uniform on a cloned tracker; this changes the
                         WORLD SAMPLER (which draws from that belief) and, through the
                         resulting sampled positions, the one-step reward too -- "what if
                         the belief itself were flat"
  encoder_human_mask -- ONLY the Set Encoder's human_feats/mask are zeroed; robot_feats,
                         rewards and not_terminal are reused bit-identical from `full`
                         (same tau too, since tau_seeds depend only on seed_key/action/
                         sample indices -- see _vectorized_candidate_batch) -- isolates
                         whether the encoder path specifically is sensitive to humans
  full_no_human      -- sampler, reward and encoder all have humans removed entirely
                         (upper bound: "how much could removing humans altogether explain")

Also computes batch Jacobian norms d(score)/d(human_feats) and
d(score)/d(robot_feats) at the real chosen action, for every captured
state (diagnostic only per guide.md R4-0 point 4, not a standalone
pass/fail criterion by itself).

guide.md's R4-0 pass condition: full->belief_neutral and
full->encoder_human_mask must show a SMALL continuation-value/ranking
change on the risk-opportunity subset (decisions where a human is
within discomfort_distance), AND the one-step reward must still explain
most of the observed avoidance-behavior variance. If that does NOT
hold, do not proceed to R4-1 -- re-diagnose instead.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

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

from crowd_sim.envs.utils.robot import Robot  # noqa: E402

from crowd_nav.bayesian_dvl.config import (  # noqa: E402
    ActionGridSpec, BDVL_PRODUCTION_SOURCES, FORMAL_SCENARIOS, FROZEN_VALUES, load_and_validate_registry,
)
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact  # noqa: E402
from crowd_nav.bayesian_dvl.policy import (  # noqa: E402
    _score_all_candidates, score_candidate_batch, _stable_seed, _stateless_tau_cpu,
)
from crowd_nav.bayesian_dvl.belief import BeliefTracker  # noqa: E402
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation  # noqa: E402
from crowd_nav.bayesian_pilot.protocol import BehaviorScheduler, InterventionORCA, PROFILES  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest, sha256_of_file  # noqa: E402

from crowd_nav.tools.evaluate_bdvl import _make_env, _make_policy  # noqa: E402

BASE = PACKAGE_ROOT / "runs/bayesian_dvl/remote_4090_20260808_fix2"
N_EPISODES = 15
SUITE_SEEDS = [95001, 95002, 95003, 95004, 95005]
EPISODES_PER_SEED = 3
SCENARIO_KEY = "dense_square"
PROFILE = "heldout_nonstationary"
OUT_DIR = PACKAGE_ROOT / "runs/bayesian_dvl/r4_0_audit"


def decompose(action_scores, rewards_t, quantiles, not_terminal_t, n_actions, n_samples, gamma):
    immediate_reward = rewards_t.reshape(n_actions, n_samples).mean(dim=1)
    bootstrap = quantiles.mean(dim=1) * not_terminal_t.to(quantiles.dtype)
    continuation_value = bootstrap.reshape(n_actions, n_samples).mean(dim=1)
    recomposed = immediate_reward + gamma * continuation_value
    if not torch.allclose(recomposed, action_scores, atol=1e-4):
        raise RuntimeError("decomposition does not match action_scores -- formula drifted from score_candidate_batch")
    return immediate_reward, continuation_value


def deterministic_tau(seed_key, n_actions, n_samples, n_quantiles, tau_upper):
    suite_seed, episode_seed, decision_counter = seed_key
    tau_seeds = np.empty((n_actions, n_samples), dtype=np.int64)
    for action_idx in range(n_actions):
        for sample_idx in range(n_samples):
            tau_seeds[action_idx, sample_idx] = _stable_seed(suite_seed, episode_seed, decision_counter, action_idx, sample_idx)
    return _stateless_tau_cpu(tau_seeds.reshape(-1), n_quantiles, tau_upper)


def main() -> None:
    env_config_path = PACKAGE_ROOT / "crowd_nav/configs/env_bayesian_dvl.config"
    registry_path = PACKAGE_ROOT / "crowd_nav/configs/bayesian_dvl_registry.json"
    registry = load_and_validate_registry(str(registry_path))
    action_table = ActionGridSpec.from_env_config(str(env_config_path)).build_action_table()
    artifact_path = BASE / "artifact/production_artifact.json"
    artifact = SBKHMMArtifact.load(str(artifact_path), expect_tier="production")
    checkpoint_path = BASE / "selection/selected_model.pth"
    frozen = registry["frozen_values"]
    gamma = float(frozen["gamma"])
    n_world_samples = frozen["world_samples_validation"]
    n_iqn_quantiles = frozen["iqn_quantiles_validation"]
    human_num = FORMAL_SCENARIOS[SCENARIO_KEY]["humans"]

    records = []
    outcomes = {"success": 0, "collision": 0, "timeout": 0}

    for suite_seed in SUITE_SEEDS:
        env, env_config_for_robot = _make_env(env_config_path, human_num, SCENARIO_KEY)
        adapter = _make_policy(
            action_table, artifact, str(checkpoint_path), registry, False, suite_seed,
            "validation", "cpu", risk_neutral=True, collect_calibration=False,
        )
        robot = Robot(env_config_for_robot, "robot")
        robot.set_policy(adapter); robot.visible = True; robot.time_step = FROZEN_VALUES["dt"]; robot.env = env
        env.set_robot(robot)
        encoder = adapter.bdvl_policy.set_encoder
        net = adapter.bdvl_policy.value_network
        reward_config = adapter.bdvl_policy.reward_config

        for ep_idx in range(EPISODES_PER_SEED):
            episode_seed = suite_seed * 100000 + ep_idx
            adapter.reset_episode_stats(suite_seed=suite_seed, episode_seed=episode_seed)
            env.case_counter["test"] = episode_seed % (2**32 - 1)
            env.reset()
            scheduler = BehaviorScheduler(PROFILES[PROFILE], seed=episode_seed)
            scheduler.reset(len(env.humans))
            for h in env.humans:
                hp = InterventionORCA(env_config_for_robot); hp.time_step = env.time_step; h.set_policy(hp)

            event = None
            step = 0
            for step in range(200):
                scheduler.advance([h.policy for h in env.humans])
                humans = [
                    HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
                    for i, h in enumerate(env.humans)
                ]
                robot_obs = RobotObservation.from_full_state(env.robot.get_full_state())
                global_time = adapter._global_time
                min_h = min(
                    float(np.hypot(h.px - env.robot.px, h.py - env.robot.py)) - h.radius - env.robot.radius
                    for h in env.humans
                )

                action = env.robot.act([h.get_observable_state() for h in env.humans])
                chosen_idx = int(adapter.last_action_index)
                tracker_snapshot = copy.deepcopy(adapter.bdvl_policy.belief_tracker._tracks)
                seed_key = (suite_seed, episode_seed, adapter.bdvl_policy._decision_counter)

                def rebuild_tracker(uniform=False):
                    t = BeliefTracker(artifact)
                    t._tracks = copy.deepcopy(tracker_snapshot)
                    if uniform:
                        for st in t._tracks.values():
                            st.belief = np.ones_like(st.belief) / st.belief.shape[0]
                    return t

                common = dict(
                    artifact=artifact, action_table=action_table, reward_config=reward_config,
                    dt=frozen["dt"], time_limit=frozen["time_limit"], max_human_speed=frozen["max_human_speed"],
                    n_world_samples=n_world_samples, n_iqn_quantiles=n_iqn_quantiles, tau_upper=1.0,
                    posterior_source="full", set_encoder=encoder, value_network=net, device="cpu",
                    seed_key=seed_key, gamma=gamma,
                )

                entry = {
                    "suite_seed": suite_seed, "episode_seed": episode_seed, "step": step,
                    "min_human_dist": float(min_h), "chosen_idx": chosen_idx,
                }

                with torch.no_grad():
                    scores_full, quant_full, rf, hf, mk, rw, nt, na, ns = _score_all_candidates(
                        tracker=rebuild_tracker(), robot=robot_obs, humans=humans, global_time=global_time, **common,
                    )
                    ir_full, cv_full = decompose(scores_full, rw, quant_full, nt, na, ns, gamma)
                    entry["full_argmax"] = int(torch.argmax(scores_full).item())
                    entry["full_ir"] = ir_full.tolist(); entry["full_cv"] = cv_full.tolist(); entry["full_total"] = scores_full.tolist()

                    scores_bn, quant_bn, _, _, _, rw_bn, nt_bn, _, _ = _score_all_candidates(
                        tracker=rebuild_tracker(uniform=True), robot=robot_obs, humans=humans, global_time=global_time, **common,
                    )
                    ir_bn, cv_bn = decompose(scores_bn, rw_bn, quant_bn, nt_bn, na, ns, gamma)
                    entry["belief_neutral_argmax"] = int(torch.argmax(scores_bn).item())
                    entry["belief_neutral_ir"] = ir_bn.tolist(); entry["belief_neutral_cv"] = cv_bn.tolist(); entry["belief_neutral_total"] = scores_bn.tolist()

                    # encoder_human_mask: bit-identical tau/rewards/not_terminal as `full`
                    # (tau_seeds depend only on seed_key+action+sample index, never on
                    # humans/beliefs -- see _vectorized_candidate_batch), only the tensors
                    # reaching the Set Encoder are masked.
                    tau_full = deterministic_tau(seed_key, na, ns, n_iqn_quantiles, 1.0)
                    hf_masked = torch.zeros_like(hf); mk_masked = torch.zeros_like(mk)
                    scores_mask, quant_mask = score_candidate_batch(encoder, net, rf, hf_masked, mk_masked, rw, nt, tau_full, na, ns, gamma)
                    ir_mask, cv_mask = decompose(scores_mask, rw, quant_mask, nt, na, ns, gamma)
                    entry["encoder_mask_argmax"] = int(torch.argmax(scores_mask).item())
                    entry["encoder_mask_ir"] = ir_mask.tolist(); entry["encoder_mask_cv"] = cv_mask.tolist(); entry["encoder_mask_total"] = scores_mask.tolist()

                    # sanity: encoder_mask must reuse the SAME immediate_reward as full
                    # (only the encoder input differs) -- if this ever drifts, the
                    # isolation claim above is void.
                    if not torch.allclose(ir_mask, ir_full, atol=1e-6):
                        raise RuntimeError("encoder_human_mask condition leaked into immediate_reward -- isolation broken")

                    if len(humans) > 0:
                        scores_nh, quant_nh, _, _, _, rw_nh, nt_nh, _, _ = _score_all_candidates(
                            tracker=rebuild_tracker(), robot=robot_obs, humans=[], global_time=global_time, **common,
                        )
                        ir_nh, cv_nh = decompose(scores_nh, rw_nh, quant_nh, nt_nh, na, ns, gamma)
                        entry["no_human_argmax"] = int(torch.argmax(scores_nh).item())
                        entry["no_human_ir"] = ir_nh.tolist(); entry["no_human_cv"] = cv_nh.tolist(); entry["no_human_total"] = scores_nh.tolist()

                # Jacobian norms (diagnostic only, guide.md R4-0 point 4): use the SAME
                # deterministic tau as `full` so this is not confounded by fresh randomness.
                rf_g = rf.clone().requires_grad_(True)
                hf_g = hf.clone().requires_grad_(True)
                scores_g, _ = score_candidate_batch(encoder, net, rf_g, hf_g, mk, rw, nt, tau_full, na, ns, gamma)
                scores_g[entry["full_argmax"]].backward()
                entry["jacobian_norm_human_feats"] = float(hf_g.grad.norm().item())
                entry["jacobian_norm_robot_feats"] = float(rf_g.grad.norm().item())

                records.append(entry)

                _, _reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    event = info.get("event")
                    break
            outcome = {"reach_goal": "success", "collision": "collision", "timeout": "timeout"}.get(event, "timeout")
            outcomes[outcome] += 1
            print(f"episode suite_seed={suite_seed} idx={ep_idx}: outcome={outcome} steps={step + 1}", flush=True)

    # ---- Aggregate report ----
    risk_opportunity = [r for r in records if r["min_human_dist"] < frozen["discomfort_distance"]]
    print()
    print(f"total decisions: {len(records)}, risk-opportunity subset (min_human_dist < {frozen['discomfort_distance']}): {len(risk_opportunity)}")

    def agg(subset, cond_key, label):
        n = len(subset)
        if n == 0:
            print(f"{label}: no states in subset")
            return None
        argmax_key = f"{cond_key}_argmax"
        cv_key = f"{cond_key}_cv"
        disagreements = sum(1 for r in subset if r[argmax_key] != r["full_argmax"])
        argmax_diff = disagreements / n
        cv_deltas = [abs(np.array(r["full_cv"])[r["full_argmax"]] - np.array(r[cv_key])[r["full_argmax"]]) for r in subset]
        cv_delta_mean = float(np.mean(cv_deltas))
        total_deltas = [abs(np.array(r["full_total"])[r["full_argmax"]] - np.array(r[f"{cond_key}_total"])[r["full_argmax"]]) for r in subset]
        total_delta_mean = float(np.mean(total_deltas))
        print(
            f"{label}: argmax disagreement={argmax_diff:.3f} ({disagreements}/{n}), "
            f"mean|continuation_value delta| at full-argmax action={cv_delta_mean:.4f}, "
            f"mean|total_score delta| at full-argmax action={total_delta_mean:.4f}"
        )
        return {"n": n, "argmax_disagreement": argmax_diff, "mean_abs_cv_delta_at_full_argmax": cv_delta_mean, "mean_abs_total_delta_at_full_argmax": total_delta_mean}

    print("\n--- Full decision set ---")
    all_bn = agg(records, "belief_neutral", "full vs belief_neutral")
    all_em = agg(records, "encoder_mask", "full vs encoder_human_mask")
    all_nh = agg([r for r in records if "no_human_argmax" in r], "no_human", "full vs full_no_human")

    print("\n--- Risk-opportunity subset (min_human_dist < discomfort_distance) ---")
    ro_bn = agg(risk_opportunity, "belief_neutral", "full vs belief_neutral")
    ro_em = agg(risk_opportunity, "encoder_mask", "full vs encoder_human_mask")
    ro_nh = agg([r for r in risk_opportunity if "no_human_argmax" in r], "no_human", "full vs full_no_human")

    # One-step-reward-explains-avoidance check: on the risk-opportunity subset, how much
    # of the full-vs-belief_neutral / full-vs-encoder_mask TOTAL SCORE delta is attributable
    # to the immediate_reward term alone (which is belief-independent by construction, per
    # transition.py's discomfort/collision penalty) versus the continuation_value term.
    def reward_share(subset, cond_key):
        if not subset:
            return None
        ir_key = f"{cond_key}_ir"; cv_key = f"{cond_key}_cv"
        ir_deltas = [abs(np.array(r["full_ir"])[r["full_argmax"]] - np.array(r[ir_key])[r["full_argmax"]]) for r in subset]
        cv_deltas = [abs(np.array(r["full_cv"])[r["full_argmax"]] - np.array(r[cv_key])[r["full_argmax"]]) * gamma for r in subset]
        ir_sum = float(np.sum(ir_deltas)); cv_sum = float(np.sum(cv_deltas))
        share = ir_sum / (ir_sum + cv_sum) if (ir_sum + cv_sum) > 0 else float("nan")
        return share

    ro_bn_reward_share = reward_share(risk_opportunity, "belief_neutral")
    ro_em_reward_share = reward_share(risk_opportunity, "encoder_mask")
    print(f"\nOn risk-opportunity subset, immediate_reward's share of |full-vs-condition delta|:")
    print(f"  belief_neutral: {ro_bn_reward_share:.3f}" if ro_bn_reward_share is not None else "  belief_neutral: n/a")
    print(f"  encoder_mask:   {ro_em_reward_share:.3f}" if ro_em_reward_share is not None else "  encoder_mask: n/a")

    hf_jac = [r["jacobian_norm_human_feats"] for r in records]
    rf_jac = [r["jacobian_norm_robot_feats"] for r in records]
    print(f"\nJacobian norms (diagnostic only): d(score)/d(human_feats) mean={np.mean(hf_jac):.5f} median={np.median(hf_jac):.5f}")
    print(f"                                   d(score)/d(robot_feats) mean={np.mean(rf_jac):.5f} median={np.median(rf_jac):.5f}")

    # ---- Pass/fail per guide.md R4-0 ----
    SMALL_ARGMAX_DISAGREEMENT = 0.10  # ad hoc but documented threshold; a genuinely belief-driven
    # continuation value should show LARGE disagreement here, not small -- guide.md's pass
    # condition for R4-0 is that these deltas are SMALL, which is itself the FAILURE mode we
    # are trying to detect (confirms the bypass). This script reports the numbers; the human
    # judgment call ("is this small enough to call it a bypass, given reward already explains
    # most of the behavior") is made explicitly below, not silently baked into a single number.
    reward_dominates = (
        ro_bn_reward_share is not None and ro_bn_reward_share > 0.5
        and ro_em_reward_share is not None and ro_em_reward_share > 0.5
    )
    belief_sensitivity_small = (
        ro_bn is not None and ro_bn["argmax_disagreement"] < SMALL_ARGMAX_DISAGREEMENT
        and ro_em is not None and ro_em["argmax_disagreement"] < SMALL_ARGMAX_DISAGREEMENT
    )
    r4_0_pass = bool(reward_dominates and belief_sensitivity_small)
    print(f"\nR4_0_ROOT_CAUSE_CONFIRMED={r4_0_pass} (belief_sensitivity_small={belief_sensitivity_small}, reward_dominates_delta={reward_dominates})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records_path = OUT_DIR / "records.jsonl"
    if records_path.exists():
        raise SystemExit(f"refusing to overwrite existing R4-0 audit records: {records_path}")
    with records_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    summary = {
        "n_episodes": len(SUITE_SEEDS) * EPISODES_PER_SEED, "outcomes": outcomes, "n_decisions": len(records),
        "n_risk_opportunity": len(risk_opportunity),
        "full_vs_belief_neutral_all": all_bn, "full_vs_encoder_mask_all": all_em, "full_vs_full_no_human_all": all_nh,
        "full_vs_belief_neutral_risk_opportunity": ro_bn, "full_vs_encoder_mask_risk_opportunity": ro_em,
        "full_vs_full_no_human_risk_opportunity": ro_nh,
        "risk_opportunity_reward_share_belief_neutral": ro_bn_reward_share,
        "risk_opportunity_reward_share_encoder_mask": ro_em_reward_share,
        "jacobian_human_feats_mean": float(np.mean(hf_jac)), "jacobian_robot_feats_mean": float(np.mean(rf_jac)),
        "r4_0_root_cause_confirmed": r4_0_pass,
        "checkpoint_sha256": sha256_of_file(str(checkpoint_path)), "artifact_sha256": artifact.content_sha256(),
        "registry_content_sha256": registry["content_sha256"], "scenario": SCENARIO_KEY, "profile": PROFILE,
        "n_world_samples": n_world_samples, "n_iqn_quantiles": n_iqn_quantiles,
    }
    atomic_write_json(str(OUT_DIR / "summary.json"), summary)
    manifest = build_run_manifest(repo_root=str(PACKAGE_ROOT), command=" ".join(sys.argv), source_files=BDVL_PRODUCTION_SOURCES, extra=summary)
    atomic_write_json(str(OUT_DIR / "summary.manifest.json"), manifest)
    print(f"\nR4_0_AUDIT_DONE records={records_path} summary={OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
