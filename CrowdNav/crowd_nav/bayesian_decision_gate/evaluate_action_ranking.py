#!/usr/bin/env python3
"""Decision-level (counterfactual action-ranking) evaluation.

This is the core of the gate: NOT "is GDBN's one-step prediction better than
CV's" (that is ``evaluate_prediction.py``, and per the pre-registration a
pass there does NOT count) but "given the SAME 80 candidate robot actions the
real policy considers, can GDBN rank them by true counterfactual risk better
than CV, especially during stop/slow/turn stress events."

Ground truth for "true risk of candidate action a at time t" is rolled out
from the REAL RECORDED pedestrian future (``obs[t+1 : t+1+horizon]`` from the
dataset collected by ``collect_dataset.py``) combined with a hypothetical
constant-velocity ROBOT rollout under ``a`` -- never from any model's own
prediction. This is an OPEN-LOOP COUNTERFACTUAL ORACLE, not a closed-loop
ground truth: it assumes the recorded pedestrians' real trajectory would not
change had the robot taken a different action over the short horizon, which
is false in general (ORCA pedestrians react to the robot's position). Two
mitigations, both required before trusting a result:
  1. Report sensitivity across multiple horizons (``--horizons``, default
     "1,3,5"). Only trust a direction that agrees across all of them.
  2. If Gate-B passes here, the next step (NOT implemented in this file) is
     to re-verify on a sample of high-risk states using a real simulator
     branch rollout (actually re-running ORCA with the candidate action) --
     this offline gate is a cheap filter, not the final word.

Models are compared only after first restricting to the near-optimal-by-
-progress action subset (default: top 20% of the 80 candidates by how much
closer to the goal they get the robot over the horizon) -- otherwise "always
prefer standing still" trivially wins on raw risk and the comparison is
meaningless.

Two further corrections vs. the first version of this file:
  - Top-1/top-k hit and disagreement are computed with an explicit tolerance
    against the true-risk VALUE, not by argsort-ranking (which arbitrarily
    orders tied/near-tied "equally safe" actions and would wrongly fail a
    model that picked a different but equally-safe action).
  - The "event window" a decision falls into distinguishes `event_near`
    (an intervention-affected pedestrian is actually close to the robot --
    the only condition informative for a risk-ranking gate) from `event_any`
    (any pedestrian anywhere has a non-nominal mode, reported only -- at
    20-person density this is dominated by pedestrians nowhere near the
    robot and would dilute the gate with irrelevant "stress").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.bayesian_decision_gate.bootstrap import (  # noqa: E402
    block_bootstrap_ci,
    paired_block_bootstrap_diff,
    spearman_corr,
)
from crowd_nav.bayesian_decision_gate.protocol import lock_action_grid  # noqa: E402
from crowd_nav.contracts import GRID, discrete_index_to_action  # noqa: E402
from crowd_nav.gdbn import GDBNIntegration  # noqa: E402
from crowd_nav.risk_models import ConstantVelocityRiskModel  # noqa: E402

ROBOT_RADIUS_DEFAULT = 0.3
HUMAN_RADIUS_DEFAULT = 0.3

RANK_TOLERANCE_DEFAULT = 1e-3
MEANINGFUL_DIFF_TOLERANCE_DEFAULT = 0.02
OPPORTUNITY_THRESHOLD_DEFAULT = 0.05
NEAR_MISS_CLEARANCE_DEFAULT = 0.10
EVENT_NEAR_RADIUS_DEFAULT = 3.0


def build_candidate_actions() -> np.ndarray:
    n_actions = int(GRID["n_speeds"]) * int(GRID["n_headings"]) + int(
        bool(GRID.get("include_stop", False))
    )
    return np.asarray(
        [discrete_index_to_action(i) for i in range(n_actions)], dtype=np.float64
    )


def build_models(max_peds: int, frozen_k3_params: str, refit_dir: str, refit_k: int, seed: int) -> Dict:
    return {
        "cv": ConstantVelocityRiskModel(max_peds=max_peds, modeled_peds=max_peds),
        "frozen_k3": GDBNIntegration(
            K=3, params_dir=frozen_k3_params, max_peds=max_peds, random_seed=seed
        ),
        f"refit_k{refit_k}": GDBNIntegration(
            K=refit_k, params_dir=refit_dir, max_peds=max_peds, random_seed=seed
        ),
    }


def ground_truth_for_step(
    obs_seq: np.ndarray,
    num_humans: int,
    t: int,
    horizon: int,
    candidate_actions: np.ndarray,
    dt: float,
    safe_distance: float,
    pedestrian_aggregation: str,
) -> Optional[Dict]:
    T = len(obs_seq)
    h_max = min(horizon, T - 1 - t)
    if h_max < 1:
        return None

    robot0 = obs_seq[t, :9].astype(np.float64)
    robot_pos0 = robot0[:2]
    robot_radius = float(robot0[4]) if robot0[4] > 0 else ROBOT_RADIUS_DEFAULT
    goal = robot0[6:8]
    dist_to_goal0 = float(np.linalg.norm(goal - robot_pos0))

    ped_block0 = obs_seq[t, 9:9 + 5 * num_humans].reshape(num_humans, 5)
    valid0 = ~np.all(ped_block0 == 0.0, axis=1)
    ped_radius = np.where(ped_block0[:, 4] > 0, ped_block0[:, 4], HUMAN_RADIUS_DEFAULT)

    n_actions = len(candidate_actions)
    risk_max = np.zeros(n_actions, dtype=np.float64)
    min_clear = np.full(n_actions, np.inf, dtype=np.float64)

    for h in range(1, h_max + 1):
        future_block = obs_seq[t + h, 9:9 + 5 * num_humans].reshape(num_humans, 5)
        future_pos = future_block[:, :2].astype(np.float64)
        elapsed = h * dt
        robot_future = robot_pos0[None, :] + candidate_actions * elapsed
        diff = future_pos[None, :, :] - robot_future[:, None, :]
        dist = np.linalg.norm(diff, axis=-1)
        clearance = dist - (robot_radius + ped_radius)[None, :]
        clearance_masked = np.where(valid0[None, :], clearance, np.inf)
        risk = GDBNIntegration._clearance_risk(clearance_masked, safe_distance)
        risk = np.where(valid0[None, :], risk, 0.0)

        if pedestrian_aggregation == "max":
            step_agg = np.where(valid0.any(), risk.max(axis=1), 0.0)
        else:
            denom = max(int(valid0.sum()), 1)
            step_agg = (risk * valid0[None, :]).sum(axis=1) / denom

        risk_max = np.maximum(risk_max, step_agg)
        min_clear = np.minimum(min_clear, clearance_masked.min(axis=1))

    elapsed_final = h_max * dt
    robot_future_final = robot_pos0 + candidate_actions * elapsed_final
    dist_to_goal_final = np.linalg.norm(goal[None, :] - robot_future_final, axis=1)
    progress = dist_to_goal0 - dist_to_goal_final

    return {
        "true_risk": risk_max,
        "true_min_clearance": min_clear,
        "true_collision": min_clear <= 0.0,
        "true_near_miss": min_clear <= NEAR_MISS_CLEARANCE_DEFAULT,
        "progress": progress,
        "h_used": h_max,
    }


def predicted_risk_for_step(
    model,
    state_t: np.ndarray,
    belief_vec: Optional[np.ndarray],
    candidate_actions: np.ndarray,
    horizon: int,
    dt: float,
    safe_distance: float,
    cvar_alpha: float,
    pedestrian_aggregation: str,
) -> np.ndarray:
    n_actions = len(candidate_actions)
    states = np.repeat(state_t.reshape(1, -1), n_actions, axis=0)
    beliefs = None
    if belief_vec is not None:
        beliefs = np.repeat(belief_vec.reshape(1, *belief_vec.shape), n_actions, axis=0)
    rollout = model.predict_action_rollout_batch(
        states,
        candidate_actions,
        belief_vecs=beliefs,
        horizon=horizon,
        dt=dt,
        safe_distance=safe_distance,
        cvar_alpha=cvar_alpha,
        pedestrian_aggregation=pedestrian_aggregation,
    )
    return np.clip(np.asarray(rollout["risk"], dtype=np.float64), 0.0, 1.0)


def rank_and_hit(
    true_risk_subset: np.ndarray,
    chosen_local_idx: int,
    top_k: int,
    rank_tolerance: float = RANK_TOLERANCE_DEFAULT,
) -> Tuple[int, float, float, float]:
    """Tolerance-based rank/hit/regret of the chosen action within a subset.

    ``chosen_rank`` counts subset actions STRICTLY safer than the chosen one
    beyond ``rank_tolerance`` -- NOT an argsort position. Argsort forces a
    total order even among actions with essentially identical true risk
    (common when the robot is far from every pedestrian), which would
    wrongly count a model's equally-safe-but-different choice as a Top-1
    failure. Returns (chosen_rank, top1_hit, topk_hit, regret).
    """
    true_min_risk = float(true_risk_subset.min())
    chosen_risk = float(true_risk_subset[chosen_local_idx])
    chosen_rank = int(np.sum(true_risk_subset < chosen_risk - rank_tolerance))
    top1_hit = float(chosen_risk <= true_min_risk + rank_tolerance)
    topk_hit = float(chosen_rank < top_k)
    regret = float(chosen_risk - true_min_risk)
    return chosen_rank, top1_hit, topk_hit, regret


def decision_points(T: int, horizon: int, stride: int, max_decisions: int) -> List[int]:
    last_valid = T - 1 - horizon
    if last_valid < 0:
        return []
    points = list(range(0, last_valid + 1, max(1, stride)))
    return points[:max_decisions]


def event_window_labels(
    obs_seq: np.ndarray,
    modes_seq: np.ndarray,
    num_humans: int,
    t: int,
    h_max: int,
    near_radius: float,
) -> Dict[str, bool]:
    """``event_any``: any pedestrian has a non-nominal intervention mode
    somewhere in [t, t+h_max]. ``event_near``: additionally, that pedestrian
    is within ``near_radius`` of the robot at the step the event is active --
    the only condition that can plausibly make the robot's action choice
    matter. At 20-person density ``event_any`` is dominated by irrelevant
    far-away pedestrians and must not be used as the gating window."""
    window = modes_seq[t: t + h_max + 1]
    event_any = bool(np.any(window != 0))
    if not event_any:
        return {"event_any": False, "event_near": False}

    event_near = False
    for step_offset in range(window.shape[0]):
        step_modes = window[step_offset]
        active = np.where(step_modes != 0)[0]
        if len(active) == 0:
            continue
        t_step = t + step_offset
        robot_pos = obs_seq[t_step, :2].astype(np.float64)
        ped_block = obs_seq[t_step, 9:9 + 5 * num_humans].reshape(num_humans, 5)
        for ped_idx in active:
            if ped_idx >= num_humans:
                continue
            ped_pos = ped_block[ped_idx, :2].astype(np.float64)
            if np.linalg.norm(ped_pos - robot_pos) <= near_radius:
                event_near = True
                break
        if event_near:
            break
    return {"event_any": event_any, "event_near": event_near}


def evaluate_episode(
    model_names: List[str],
    models: Dict,
    candidate_actions: np.ndarray,
    obs_seq: np.ndarray,
    act_seq: np.ndarray,
    modes_seq: np.ndarray,
    num_humans: int,
    horizon: int,
    stride: int,
    max_decisions: int,
    dt: float,
    safe_distance: float,
    cvar_alpha: float,
    pedestrian_aggregation: str,
    progress_percentile: float,
    top_k: int,
    rank_tolerance: float = RANK_TOLERANCE_DEFAULT,
    meaningful_diff_tolerance: float = MEANINGFUL_DIFF_TOLERANCE_DEFAULT,
    opportunity_threshold: float = OPPORTUNITY_THRESHOLD_DEFAULT,
    near_radius: float = EVENT_NEAR_RADIUS_DEFAULT,
) -> Dict:
    T = len(obs_seq)
    for name in model_names:
        models[name].reset(n_peds=num_humans)

    points = decision_points(T, horizon, stride, max_decisions)
    point_set = set(points)

    per_model_rows: Dict[str, List[Dict]] = {name: [] for name in model_names}
    disagreements: List[Dict] = []
    opportunities: List[Dict] = []

    for t in range(T):
        state_t = obs_seq[t]
        belief_vecs = {}
        for name in model_names:
            models[name].update(state_t)
            belief_vecs[name] = models[name].get_per_ped_belief_vec()

        if t not in point_set:
            continue

        truth = ground_truth_for_step(
            obs_seq, num_humans, t, horizon, candidate_actions, dt, safe_distance,
            pedestrian_aggregation,
        )
        if truth is None:
            continue

        progress = truth["progress"]
        threshold = np.percentile(progress, progress_percentile)
        near_optimal = progress >= threshold
        if near_optimal.sum() < 2:
            continue
        idx = np.where(near_optimal)[0]

        true_risk_subset = truth["true_risk"][idx]
        true_min_risk = float(true_risk_subset.min())
        true_max_risk = float(true_risk_subset.max())
        opportunity = bool((true_max_risk - true_min_risk) > opportunity_threshold)
        windows = event_window_labels(obs_seq, modes_seq, num_humans, t, truth["h_used"], near_radius)
        opportunities.append({"t": t, "opportunity": opportunity, **windows})

        chosen_local: Dict[str, int] = {}
        for name in model_names:
            pred_risk = predicted_risk_for_step(
                models[name], state_t, belief_vecs[name], candidate_actions,
                horizon, dt, safe_distance, cvar_alpha, pedestrian_aggregation,
            )
            pred_subset = pred_risk[idx]
            corr = spearman_corr(true_risk_subset, pred_subset)
            chosen_local_idx = int(np.argmin(pred_subset))
            chosen_local[name] = chosen_local_idx
            _, top1_hit, topk_hit, regret = rank_and_hit(
                true_risk_subset, chosen_local_idx, top_k, rank_tolerance
            )
            chosen_global_idx = int(idx[chosen_local_idx])
            per_model_rows[name].append(
                {
                    "t": t,
                    "event_any": windows["event_any"],
                    "event_near": windows["event_near"],
                    "opportunity": opportunity,
                    "spearman": corr,
                    "top1_hit": top1_hit,
                    "topk_hit": topk_hit,
                    "regret": regret,
                    "true_collision": float(truth["true_collision"][chosen_global_idx]),
                    "true_near_miss": float(truth["true_near_miss"][chosen_global_idx]),
                    "min_clearance": float(truth["true_min_clearance"][chosen_global_idx]),
                    "chosen_action_idx": chosen_global_idx,
                }
            )

        # CV vs. every GDBN-family model: who actually chose the safer action.
        if "cv" in chosen_local:
            cv_choice = chosen_local["cv"]
            cv_true_risk = float(true_risk_subset[cv_choice])
            for name in model_names:
                if name == "cv":
                    continue
                other_choice = chosen_local[name]
                other_true_risk = float(true_risk_subset[other_choice])
                risk_gap = abs(other_true_risk - cv_true_risk)
                disagreements.append(
                    {
                        "t": t,
                        "event_any": windows["event_any"],
                        "event_near": windows["event_near"],
                        "opportunity": opportunity,
                        "model": name,
                        "disagree": bool(cv_choice != other_choice),
                        "meaningful_disagree": bool(
                            cv_choice != other_choice and risk_gap > meaningful_diff_tolerance
                        ),
                        "cv_true_risk": cv_true_risk,
                        "other_true_risk": other_true_risk,
                        "other_safer": bool(other_true_risk < cv_true_risk - rank_tolerance),
                        "cv_safer": bool(cv_true_risk < other_true_risk - rank_tolerance),
                    }
                )

    return {
        "per_model_rows": per_model_rows,
        "disagreements": disagreements,
        "opportunities": opportunities,
    }


def _seed_blocks(
    episode_results: List[Dict],
    seeds: List[int],
    extractor,
) -> Dict[int, List[np.ndarray]]:
    """Group per-decision values by SEED (not episode) -- decisions within
    the same seed's rollout are correlated, so the bootstrap's top-level
    resampling unit must be the seed, matching belief_mdp/evaluate.py's
    convention. Multiple episodes share a seed (one seed produces many
    episodes per collect_dataset.py file); all of a seed's episodes are
    concatenated into ONE block."""
    by_seed: Dict[int, List[np.ndarray]] = {}
    for episode, seed in zip(episode_results, seeds):
        values = extractor(episode)
        if values is None or len(values) == 0:
            continue
        by_seed.setdefault(seed, []).append(np.asarray(values, dtype=np.float64))
    return {seed: [np.concatenate(chunks)] for seed, chunks in by_seed.items() if chunks}


def aggregate(
    episode_results: List[Dict],
    episode_seeds: List[int],
    model_names: List[str],
    replicates: int,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    windows = ("event_any", "event_near", "normal", "all")
    metrics = ("spearman", "top1_hit", "topk_hit", "regret", "true_collision", "true_near_miss", "min_clearance")
    summary: Dict = {"per_model": {}, "cv_vs_gdbn": {}, "opportunity": {}}

    def window_filter(rows, window):
        if window == "all":
            return rows
        if window == "normal":
            return [r for r in rows if not r["event_near"]]
        return [r for r in rows if r[window]]

    for window in windows:
        def opp_extractor(episode, window=window):
            rows = window_filter(episode["opportunities"], window)
            return [float(r["opportunity"]) for r in rows]

        blocks = _seed_blocks(episode_results, episode_seeds, opp_extractor)
        mean_val, lo, hi = block_bootstrap_ci(
            [v[0] for v in blocks.values()], replicates=replicates, rng=rng
        )
        per_seed_means = {str(s): float(np.mean(v[0])) for s, v in blocks.items()}
        summary["opportunity"][window] = {
            "mean": mean_val, "ci_lo": lo, "ci_hi": hi,
            "n_decisions": int(sum(len(v[0]) for v in blocks.values())),
            "n_seeds": len(blocks), "per_seed_means": per_seed_means,
        }

    for name in model_names:
        summary["per_model"][name] = {}
        for window in windows:
            block_metric_arrays = {}
            for metric in metrics:
                def extractor(episode, name=name, window=window, metric=metric):
                    rows = window_filter(episode["per_model_rows"][name], window)
                    return [
                        r[metric] for r in rows
                        if not (isinstance(r[metric], float) and np.isnan(r[metric]))
                    ]

                seed_blocks = _seed_blocks(episode_results, episode_seeds, extractor)
                block_list = [v[0] for v in seed_blocks.values()]
                mean_val, lo, hi = block_bootstrap_ci(block_list, replicates=replicates, rng=rng)
                n_decisions = int(sum(len(b) for b in block_list))
                per_seed_means = {str(s): float(np.mean(v[0])) for s, v in seed_blocks.items()}
                block_metric_arrays[metric] = {
                    "mean": mean_val, "ci_lo": lo, "ci_hi": hi, "n_decisions": n_decisions,
                    "n_seeds": len(seed_blocks), "per_seed_means": per_seed_means,
                }
            summary["per_model"][name][window] = block_metric_arrays

    for name in model_names:
        if name == "cv":
            continue
        summary["cv_vs_gdbn"][name] = {}
        for window in windows:
            def disagree_extractor(episode, name=name, window=window):
                items = [d for d in episode["disagreements"] if d["model"] == name]
                items = window_filter(items, window)
                return [float(d["disagree"]) for d in items]

            def meaningful_disagree_extractor(episode, name=name, window=window):
                items = [d for d in episode["disagreements"] if d["model"] == name]
                items = window_filter(items, window)
                return [float(d["meaningful_disagree"]) for d in items]

            def win_extractor(episode, name=name, window=window):
                items = [d for d in episode["disagreements"] if d["model"] == name]
                items = window_filter(items, window)
                items = [d for d in items if d["meaningful_disagree"]]
                return [float(d["other_safer"]) for d in items]

            def cv_regret_extractor(episode, window=window):
                rows = window_filter(episode["per_model_rows"]["cv"], window)
                return [r["regret"] for r in rows]

            def other_regret_extractor(episode, name=name, window=window):
                rows = window_filter(episode["per_model_rows"][name], window)
                return [r["regret"] for r in rows]

            disagree_blocks = _seed_blocks(episode_results, episode_seeds, disagree_extractor)
            meaningful_blocks = _seed_blocks(episode_results, episode_seeds, meaningful_disagree_extractor)
            win_blocks = _seed_blocks(episode_results, episode_seeds, win_extractor)
            cv_regret_blocks = _seed_blocks(episode_results, episode_seeds, cv_regret_extractor)
            other_regret_blocks = _seed_blocks(episode_results, episode_seeds, other_regret_extractor)

            common_seeds = sorted(set(cv_regret_blocks) & set(other_regret_blocks))
            cv_list = [cv_regret_blocks[s][0] for s in common_seeds]
            other_list = [other_regret_blocks[s][0] for s in common_seeds]

            disagree_mean, disagree_lo, disagree_hi = block_bootstrap_ci(
                [v[0] for v in disagree_blocks.values()], replicates=replicates, rng=rng
            )
            meaningful_mean, meaningful_lo, meaningful_hi = block_bootstrap_ci(
                [v[0] for v in meaningful_blocks.values()], replicates=replicates, rng=rng
            )
            win_mean, win_lo, win_hi = block_bootstrap_ci(
                [v[0] for v in win_blocks.values()], replicates=replicates, rng=rng
            )
            regret_diff_mean, regret_diff_lo, regret_diff_hi = paired_block_bootstrap_diff(
                cv_list, other_list, replicates=replicates, rng=rng
            )
            summary["cv_vs_gdbn"][name][window] = {
                "disagreement_rate": {"mean": disagree_mean, "ci_lo": disagree_lo, "ci_hi": disagree_hi},
                "meaningful_disagreement_rate": {
                    "mean": meaningful_mean, "ci_lo": meaningful_lo, "ci_hi": meaningful_hi,
                    "note": "excludes disagreements where both choices are essentially equally safe",
                },
                f"{name}_win_rate_given_meaningful_disagreement": {
                    "mean": win_mean, "ci_lo": win_lo, "ci_hi": win_hi,
                },
                "regret_reduction_vs_cv": {
                    "mean": regret_diff_mean, "ci_lo": regret_diff_lo, "ci_hi": regret_diff_hi,
                    "note": "positive = model has LOWER regret than cv (better)",
                    "n_seeds": len(common_seeds),
                },
            }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="runs/bayesian_decision_gate/data")
    parser.add_argument("--models_dir", default="runs/bayesian_decision_gate/models")
    parser.add_argument("--frozen_k3_params", default="runs/bayesian_distributional/gdbn_params_cv_residual_k3")
    parser.add_argument("--splits", default="nominal,heldout_nonstationary")
    parser.add_argument("--densities", default="5person,20person")
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max_decisions_per_episode", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--safe_distance", type=float, default=0.20)
    parser.add_argument("--cvar_alpha", type=float, default=0.80)
    parser.add_argument("--pedestrian_aggregation", default="max", choices=("max", "mean"))
    parser.add_argument("--progress_percentile", type=float, default=80.0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--rank_tolerance", type=float, default=RANK_TOLERANCE_DEFAULT)
    parser.add_argument("--meaningful_diff_tolerance", type=float, default=MEANINGFUL_DIFF_TOLERANCE_DEFAULT)
    parser.add_argument("--opportunity_threshold", type=float, default=OPPORTUNITY_THRESHOLD_DEFAULT)
    parser.add_argument("--event_near_radius", type=float, default=EVENT_NEAR_RADIUS_DEFAULT)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--env_config", default=None)
    parser.add_argument("--output_dir", default="runs/bayesian_decision_gate/action_ranking")
    args = parser.parse_args()

    lock_action_grid(args.env_config)

    data_dir = Path(args.data_dir).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_summary = json.loads((models_dir / "fit_models_summary.json").read_text(encoding="utf-8"))
    refit_k = int(fit_summary["selected_K"])
    refit_dir = str(models_dir / f"refit_k{refit_k}")

    candidate_actions = build_candidate_actions()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    densities = [d.strip() for d in args.densities.split(",") if d.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    all_results = {}
    for horizon in horizons:
        for density in densities:
            max_peds = {"5person": 5, "20person": 20}[density]
            models = build_models(max_peds, args.frozen_k3_params, refit_dir, refit_k, args.seed)
            model_names = list(models.keys())

            for split in splits:
                files = sorted(data_dir.glob(f"{split}_{density}_seed*.npz"))
                if not files:
                    print(f"[RANK] WARNING: no files for split={split} density={density}, skipping")
                    continue
                episode_results = []
                episode_seeds = []
                for path in files:
                    data = np.load(path, allow_pickle=True)
                    # Bootstrap MUST group by suite_seed (the 2407/3407/...
                    # experiment-level identity), never episode_seed (unique
                    # per episode) -- see collect_dataset.py's module
                    # docstring for why conflating the two silently collapses
                    # "seed-block bootstrap" into per-episode resampling.
                    for obs_seq, act_seq, modes_seq, num_humans, suite_seed_value in zip(
                        data["obs"], data["act"], data["modes"], data["num_humans"], data["suite_seed"]
                    ):
                        result = evaluate_episode(
                            model_names, models, candidate_actions,
                            obs_seq, act_seq, modes_seq, int(num_humans),
                            horizon, args.stride, args.max_decisions_per_episode,
                            args.dt, args.safe_distance, args.cvar_alpha,
                            args.pedestrian_aggregation, args.progress_percentile, args.top_k,
                            rank_tolerance=args.rank_tolerance,
                            meaningful_diff_tolerance=args.meaningful_diff_tolerance,
                            opportunity_threshold=args.opportunity_threshold,
                            near_radius=args.event_near_radius,
                        )
                        episode_results.append(result)
                        episode_seeds.append(int(suite_seed_value))
                print(f"[RANK] h={horizon} {split}/{density}: {len(episode_results)} episodes processed")
                summary = aggregate(episode_results, episode_seeds, model_names, args.replicates, args.seed)
                key = f"{split}__{density}__h{horizon}"
                all_results[key] = summary
                (output_dir / f"{key}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (output_dir / "action_ranking_all.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print(f"[RANK] wrote {output_dir / 'action_ranking_all.json'}")


if __name__ == "__main__":
    main()
