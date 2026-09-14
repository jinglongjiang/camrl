#!/usr/bin/env python3
"""Formal six-scenario paired evaluation for a trained belief-MDP checkpoint.

Per project convention (see Bayesian_log/algorithm_experiment_log.md), no
statistical small-sample gate precedes this: only a cheap `--smoke` pass
(a handful of episodes, correctness only, not a decision) should run before
committing to this full protocol.

Fixes from the second-round review, found before any full training was
trusted:

- The evaluation profile is `heldout_nonstationary` (genuinely unseen
  duration/turn/slow-scale ranges), not the old "decision_stress" (which
  copied heldout_nonstationary's ranges verbatim and only changed the event
  rate -- a higher-intensity replay of a distribution training already saw,
  not a held-out one, despite the name).
- The environment config's time_limit now matches the paper's own formal
  35s protocol (was silently 25s, capping episodes at 100 internal steps
  regardless of --max_steps).
- Every per-episode outcome is written to `--episode_records_output` as CSV
  (scenario, profile, seed, test_case, outcome, steps, dmin). The previous
  version only ever compared each checkpoint against the frozen Mamba
  baseline; proving `action_conditioned > cv` (or `state_only`, `corrupted`,
  `no_belief`) requires the two belief-MDP checkpoints' outcomes on
  identical episodes, which these records make possible via
  `compare_checkpoints.py`.
- The bootstrap's resampling unit is now the seed (not (scenario, profile,
  seed)): each replicate resamples seeds with replacement, and for a
  resampled seed keeps all six scenarios' episodes together, preserving
  whatever cross-scenario correlation a given seed carries. Nominal and
  heldout_nonstationary get separate CIs rather than one pooled number that
  could hide a profile-specific effect.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.belief_mdp.hashing import sha256_dir, sha256_file  # noqa: E402
from crowd_nav.belief_mdp.model import BeliefMDPQNetwork  # noqa: E402
from crowd_nav.belief_mdp.runtime import (  # noqa: E402
    BeliefMDPFeatureEngine,
    DEFAULT_K1_GDBN_PARAMS,
    FullCrowdNavigationEnvironment,
    SIX_SCENARIOS,
    sort_humans_by_ttc,
)
from crowd_nav.belief_space_rl.runtime import (  # noqa: E402
    build_frozen_mamba,
    merged_policy_config,
)
from crowd_sim.envs.utils.action import ActionXY  # noqa: E402
from crowd_sim.envs.utils.state import JointState  # noqa: E402


EVAL_PROFILES = ("nominal", "heldout_nonstationary")
RECORD_FIELDS = (
    "checkpoint", "method", "scenario", "profile", "seed", "test_case",
    "outcome", "steps", "min_dmin",
)

# Every one of these must match what the checkpoint was actually trained
# with, or evaluation is silently testing a different configuration than
# the one that produced the checkpoint. CLI default is None everywhere;
# None means "use whatever the checkpoint's own args recorded."
PARAMS_FROM_CHECKPOINT = (
    "belief_mode", "gdbn_params", "k1_gdbn_params", "gdbn_K",
    "risk_horizon", "safe_distance", "cvar_alpha", "pedestrian_aggregation",
    "particles", "policy_config", "base_env_config", "env_config", "base_checkpoint",
)
# Path-valued params also get a content-hash check: a config or GDBN params
# directory can be silently edited in place without its path string
# changing, which a plain value-equality check on the path would miss.
HASH_CHECKED_PATHS = {
    "base_checkpoint": ("file", "base_checkpoint_sha256"),
    "gdbn_params": ("dir", "gdbn_params_sha256"),
    "k1_gdbn_params": ("dir", "k1_gdbn_params_sha256"),
    "policy_config": ("file", "policy_config_sha256"),
    "base_env_config": ("file", "base_env_config_sha256"),
    "env_config": ("file", "env_config_sha256"),
}


def resolve_params(args, saved_args: dict, allow_override: bool) -> dict:
    """Resolve each of PARAMS_FROM_CHECKPOINT from CLI-or-checkpoint, and
    refuse to silently evaluate a different configuration than the one the
    checkpoint was trained under."""
    resolved = {}
    from crowd_nav.belief_mdp.protocol import FEATURE_CONTRACT
    if saved_args.get("feature_contract") != FEATURE_CONTRACT:
        if not allow_override:
            raise SystemExit("Checkpoint lacks the repaired density/nonreactive contract; use its historical code")
        print("[EVAL] exploratory override: old weights under NEW feature semantics", flush=True)
    if getattr(args, "num_humans", None) is not None:
        raise SystemExit("Evaluation uses each scenario's actual population; omit --num_humans")
    resolved["train_num_humans"] = saved_args.get("train_num_humans", saved_args.get("num_humans"))
    for name in PARAMS_FROM_CHECKPOINT:
        cli_value = getattr(args, name)
        saved_value = saved_args.get(name)
        if cli_value is None:
            resolved[name] = saved_value
            continue
        if saved_value is not None and cli_value != saved_value:
            message = (
                f"--{name}={cli_value!r} does not match the checkpoint's "
                f"training-time value {saved_value!r}."
            )
            if not allow_override:
                raise SystemExit(
                    f"[EVAL] refusing to run: {message} Pass "
                    f"--allow_feature_override to evaluate with a "
                    f"deliberately different {name}."
                )
            print(f"[EVAL] WARNING (allowed by --allow_feature_override): {message}", flush=True)
        resolved[name] = cli_value
    return resolved


def verify_artifact_hashes(resolved: dict, saved_hashes: dict, allow_override: bool):
    """Recompute content hashes of every path-valued resolved parameter and
    compare against what was recorded at training time -- catches a file
    silently edited in place, which a path-string comparison would miss."""
    for name, (kind, hash_key) in HASH_CHECKED_PATHS.items():
        path = resolved.get(name)
        if not path:
            continue
        saved_hash = saved_hashes.get(hash_key)
        if saved_hash is None:
            print(f"[EVAL] WARNING: checkpoint has no recorded {hash_key}; skipping hash check for {name}.", flush=True)
            continue
        current_hash = sha256_file(path) if kind == "file" else sha256_dir(path)
        if current_hash != saved_hash:
            message = (
                f"{name}={path!r} content hash {current_hash} does not match "
                f"the checkpoint's training-time hash {saved_hash} -- this "
                f"file/directory changed since training."
            )
            if not allow_override:
                raise SystemExit(
                    f"[EVAL] refusing to run: {message} Pass "
                    f"--allow_feature_override to evaluate against changed "
                    f"content anyway."
                )
            print(f"[EVAL] WARNING (allowed by --allow_feature_override): {message}", flush=True)


def build_network_from_checkpoint(checkpoint_blob: dict, probe, device):
    saved_args = checkpoint_blob.get("args", {})
    network = BeliefMDPQNetwork(
        context_dim=256,
        belief_dim=probe.belief_dim,
        candidate_dim=probe.candidate_dim,
        beta=float(saved_args.get("beta", 0.5)),
    ).to(device)
    network.load_state_dict(checkpoint_blob["network"])
    network.eval()
    return network


def run_belief_mdp_episode(network, engine, environment, seed, profile, test_case, device, max_steps):
    robot, humans = environment.reset(seed=seed, profile=profile, test_case=test_case)
    engine.reset()
    features = engine.encode(robot, humans)
    done = False
    steps = 0
    outcome = "timeout"
    min_dmin = float("inf")
    while not done and steps < max_steps:
        context = torch.as_tensor(features.context, dtype=torch.float32, device=device).unsqueeze(0)
        belief = torch.as_tensor(features.belief, dtype=torch.float32, device=device).unsqueeze(0)
        candidates = torch.as_tensor(features.candidates, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_score = network(context, belief, candidates)
        action_index = int(q_score.argmax(dim=-1).item())
        vx, vy = engine.actions[action_index]
        result = environment.step(ActionXY(float(vx), float(vy)))
        min_dmin = min(min_dmin, result.dmin)
        done = result.done
        steps += 1
        if done:
            outcome = result.outcome
        else:
            features = engine.encode(environment.robot, list(environment.env.humans))
    return outcome, steps, min_dmin


def run_mamba_baseline_episode(mamba, environment, seed, profile, test_case, max_steps):
    """Round 12 fix: builds its own TTC-sorted ``JointState`` instead of
    ``environment.joint_state()`` (raw simulation order). The candidate
    belief-MDP policy always feeds Mamba TTC-sorted humans (``runtime.
    sort_humans_by_ttc``, matching the production ``test.py`` path); the
    frozen baseline was feeding it the environment's arbitrary internal
    order instead. A 30-episode spot check found no resulting action
    difference, but the input-construction code path must still match the
    candidate's exactly for the formal comparison to be a fair one -- a
    scenario where sorting mattered could otherwise silently bias the
    result without ever showing up as a code bug elsewhere.
    """
    environment.reset(seed=seed, profile=profile, test_case=test_case)
    mamba.reset_episode_stats()
    mamba.set_phase("test")
    mamba.use_sarl_predict = True
    done = False
    steps = 0
    outcome = "timeout"
    min_dmin = float("inf")
    while not done and steps < max_steps:
        sorted_humans = sort_humans_by_ttc(environment.robot, list(environment.env.humans))
        state = JointState(
            environment.robot.get_full_state(),
            [human.get_observable_state() for human in sorted_humans],
        )
        with torch.no_grad():
            action = mamba.predict(state)
        result = environment.step(action)
        min_dmin = min(min_dmin, result.dmin)
        done = result.done
        steps += 1
        if done:
            outcome = result.outcome
    return outcome, steps, min_dmin


def block_bootstrap_ci(seed_blocks: list, replicates: int, rng: np.random.Generator):
    """Resample whole seeds (each seed's pooled episodes across all six
    scenarios) with replacement -- not individual episodes, and not
    (scenario, seed) pairs, which would let the same seed be split across
    independently-resampled scenario blocks and discard whatever
    cross-scenario correlation that seed carries.
    """
    n_seeds = len(seed_blocks)
    if n_seeds == 0:
        return 0.0, 0.0, 0.0
    observed = np.concatenate(seed_blocks)
    means = np.empty(replicates, dtype=np.float64)
    seed_indices = rng.integers(0, n_seeds, size=(replicates, n_seeds))
    for replicate in range(replicates):
        resampled = [seed_blocks[i] for i in seed_indices[replicate]]
        means[replicate] = np.concatenate(resampled).mean()
    return (
        float(observed.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def evaluate_profile(
    network, mamba, make_engine, args, device, profile, seeds, episodes_per_seed, max_steps,
    record_writer,
):
    """Evaluate every scenario for one profile, seed as the outer loop so
    each seed's records span all six scenarios (needed for seed-level
    bootstrap blocks and for the per-episode CSV comparison)."""
    candidate_outcomes = []
    baseline_outcomes = []
    seed_diff_blocks = {"success": [], "collision": [], "timeout": []}
    per_scenario_rates = {
        scenario: {"belief_mdp": [], "frozen_mamba": []} for scenario in SIX_SCENARIOS
    }

    for seed_index in range(seeds):
        seed = args.eval_seed_base + seed_index * 1_000_000
        seed_candidate = []
        seed_baseline = []
        for scenario in SIX_SCENARIOS:
            for episode_index in range(episodes_per_seed):
                test_case = (seed_index * episodes_per_seed + episode_index) % 9000

                engine = make_engine(SIX_SCENARIOS[scenario][2])
                environment = FullCrowdNavigationEnvironment(args.env_config, scenario, robot_visible=False)
                outcome, steps, min_dmin = run_belief_mdp_episode(
                    network, engine, environment, seed, profile, test_case, device, max_steps
                )
                candidate_outcomes.append(outcome)
                seed_candidate.append(outcome)
                per_scenario_rates[scenario]["belief_mdp"].append(outcome)
                if record_writer is not None:
                    record_writer.writerow({
                        "checkpoint": args.checkpoint, "method": "belief_mdp",
                        "scenario": scenario, "profile": profile, "seed": seed,
                        "test_case": test_case, "outcome": outcome, "steps": steps,
                        "min_dmin": min_dmin,
                    })

                baseline_environment = FullCrowdNavigationEnvironment(
                    args.env_config, scenario, robot_visible=False
                )
                outcome_b, steps_b, min_dmin_b = run_mamba_baseline_episode(
                    mamba, baseline_environment, seed, profile, test_case, max_steps
                )
                baseline_outcomes.append(outcome_b)
                seed_baseline.append(outcome_b)
                per_scenario_rates[scenario]["frozen_mamba"].append(outcome_b)
                if record_writer is not None:
                    record_writer.writerow({
                        "checkpoint": args.checkpoint, "method": "frozen_mamba",
                        "scenario": scenario, "profile": profile, "seed": seed,
                        "test_case": test_case, "outcome": outcome_b, "steps": steps_b,
                        "min_dmin": min_dmin_b,
                    })

        for metric in ("success", "collision", "timeout"):
            candidate_hits = np.array([1.0 if o == metric else 0.0 for o in seed_candidate])
            baseline_hits = np.array([1.0 if o == metric else 0.0 for o in seed_baseline])
            seed_diff_blocks[metric].append(candidate_hits - baseline_hits)

    def rates(outcomes):
        total = max(len(outcomes), 1)
        return {
            "SR": outcomes.count("success") / total,
            "CR": outcomes.count("collision") / total,
            "TR": outcomes.count("timeout") / total,
            "episodes": total,
        }

    per_scenario_summary = {
        scenario: {
            "belief_mdp": rates(v["belief_mdp"]),
            "frozen_mamba": rates(v["frozen_mamba"]),
        }
        for scenario, v in per_scenario_rates.items()
    }
    return (
        rates(candidate_outcomes),
        rates(baseline_outcomes),
        per_scenario_summary,
        seed_diff_blocks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    # All of these default to None: None means "use the checkpoint's own
    # training-time value." An explicit, different value is a hard error
    # unless --allow_feature_override is also passed -- see
    # resolve_params/verify_artifact_hashes.
    parser.add_argument("--policy_config", default=None)
    parser.add_argument("--base_env_config", default=None)
    parser.add_argument("--env_config", default=None)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--gdbn_params", default=None)
    parser.add_argument("--k1_gdbn_params", default=None)
    parser.add_argument(
        "--belief_mode",
        default=None,
        choices=(
            None, "recursive", "frame_only", "action_conditioned", "cv", "state_only", "corrupted", "no_belief",
            "k1_belief",
        ),
    )
    parser.add_argument("--gdbn_K", type=int, default=None)
    parser.add_argument("--num_humans", type=int, default=None)
    parser.add_argument("--risk_horizon", type=int, default=None)
    parser.add_argument("--safe_distance", type=float, default=None)
    parser.add_argument("--cvar_alpha", type=float, default=None)
    parser.add_argument("--pedestrian_aggregation", default=None)
    parser.add_argument(
        "--allow_feature_override",
        action="store_true",
        help="Permit CLI values / changed file content that disagree with "
        "the checkpoint's own training-time record. Without this flag, any "
        "disagreement is a hard error.",
    )
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--eval_seed_base", type=int, default=5_000_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--episodes_per_seed", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--bootstrap_replicates", type=int, default=100_000)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--episode_records_output",
        default=None,
        help="CSV path to append per-episode records for cross-checkpoint comparison.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Correctness-only pass (a few episodes); not a go/no-go decision.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.episodes_per_seed = 2
        args.particles = 12
        # A real checkpoint was trained with more particles (typically 50);
        # smoke deliberately wants fewer for speed, which would otherwise be
        # rejected by the particle-count lock below. Smoke is documented
        # everywhere as "correctness only, not a decision" -- the formal,
        # non-smoke path still enforces every PARAMS_FROM_CHECKPOINT value
        # strictly.
        args.allow_feature_override = True

    # Load the checkpoint before building anything GDBN/Mamba-related, so
    # every downstream object is constructed from validated parameters.
    checkpoint_blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint_blob.get("args", {})
    saved_hashes = checkpoint_blob.get("artifact_hashes", {})
    if not saved_hashes:
        print(
            "[EVAL] WARNING: checkpoint has no artifact_hashes recorded "
            "(trained before this check existed) -- hash verification skipped.",
            flush=True,
        )

    # Round 12: warn (never block -- an old checkpoint must still be
    # evaluable with newer eval code) if train.py/model.py/runtime.py/
    # evaluate.py have changed since this checkpoint was produced. Without
    # this, a source fix landing between training and evaluation (e.g. the
    # teacher_scores() frame bug) would silently look like the same
    # version, since neither the config nor GDBN-params hashes would catch it.
    for name, key in (
        ("train.py", "train_py_sha256"), ("model.py", "model_py_sha256"),
        ("runtime.py", "runtime_py_sha256"), ("evaluate.py", "evaluate_py_sha256"),
    ):
        saved_source_hash = saved_hashes.get(key)
        if saved_source_hash is None:
            continue
        current_source_hash = sha256_file(str(THIS_DIR / name))
        if current_source_hash != saved_source_hash:
            print(
                f"[EVAL] WARNING: {name} has changed since this checkpoint was trained "
                f"(trained hash {saved_source_hash[:12]}..., current hash {current_source_hash[:12]}...) "
                "-- results reflect the checkpoint's weights under the *current* code, not "
                "necessarily the code that produced it.",
                flush=True,
            )

    resolved = resolve_params(args, saved_args, args.allow_feature_override)
    print(f"[EVAL] resolved params: {resolved}", flush=True)
    verify_artifact_hashes(resolved, saved_hashes, args.allow_feature_override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = merged_policy_config(resolved["policy_config"], resolved["base_env_config"])
    mamba = build_frozen_mamba(config, resolved["base_checkpoint"], device)
    belief_mode = resolved["belief_mode"]

    def make_engine(eval_num_humans=5):
        return BeliefMDPFeatureEngine(
            mamba,
            resolved["gdbn_params"],
            device,
            belief_mode=belief_mode,
            K=resolved["gdbn_K"],
            n_particles=resolved["particles"],
            num_humans=eval_num_humans,
            risk_horizon=resolved["risk_horizon"],
            safe_distance=resolved["safe_distance"],
            cvar_alpha=resolved["cvar_alpha"],
            pedestrian_aggregation=resolved["pedestrian_aggregation"],
            seed=args.seed,
            k1_gdbn_params=resolved["k1_gdbn_params"] or DEFAULT_K1_GDBN_PARAMS,
        )

    # args.env_config is used below by evaluate_profile/FullCrowdNavigationEnvironment;
    # point it at the resolved, validated value.
    args.env_config = resolved["env_config"]

    probe = make_engine()
    network = build_network_from_checkpoint(checkpoint_blob, probe, device)
    print(
        f"[EVAL] checkpoint={args.checkpoint} belief_mode={belief_mode} "
        f"beta={saved_args.get('beta')} smoke={args.smoke}",
        flush=True,
    )

    record_file = None
    record_writer = None
    if args.episode_records_output:
        record_path = Path(args.episode_records_output)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not record_path.exists()
        record_file = record_path.open("a", newline="", encoding="utf-8")
        record_writer = csv.DictWriter(record_file, fieldnames=RECORD_FIELDS)
        if is_new:
            record_writer.writeheader()

    rng = np.random.default_rng(args.seed + 91)
    profile_summaries = {}
    try:
        for profile in EVAL_PROFILES:
            candidate_rates, baseline_rates, per_scenario, seed_diff_blocks = evaluate_profile(
                network, mamba, make_engine, args, device, profile,
                args.seeds, args.episodes_per_seed, args.max_steps, record_writer,
            )
            bootstrap = {}
            for metric in ("success", "collision", "timeout"):
                mean_diff, low, high = block_bootstrap_ci(
                    seed_diff_blocks[metric], args.bootstrap_replicates, rng
                )
                bootstrap[metric] = {"mean_diff": mean_diff, "ci_low": low, "ci_high": high}
            profile_summaries[profile] = {
                "pooled_belief_mdp": candidate_rates,
                "pooled_frozen_mamba": baseline_rates,
                "scenarios": per_scenario,
                "paired_seed_block_bootstrap_diff": bootstrap,
            }
            print(
                f"[EVAL] profile={profile} belief_mdp SR={candidate_rates['SR']:.1%} "
                f"CR={candidate_rates['CR']:.1%} | frozen SR={baseline_rates['SR']:.1%} "
                f"CR={baseline_rates['CR']:.1%} | SR diff CI="
                f"[{bootstrap['success']['ci_low']:+.3f}, {bootstrap['success']['ci_high']:+.3f}]",
                flush=True,
            )
    finally:
        if record_file is not None:
            record_file.close()

    summary = {
        "checkpoint": args.checkpoint,
        "belief_mode": belief_mode,
        "beta": saved_args.get("beta"),
        "smoke": args.smoke,
        "profiles": profile_summaries,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
