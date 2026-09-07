#!/usr/bin/env python3
"""Analyze the two pre-registered occlusion-information gates.

Gate 1 asks whether the oracle's paired navigation rescues concentrate in
episodes that expose never-seen hidden pedestrians. Gate 2 asks whether four
frames of observable pedestrian motion add predictive information after
controlling visible count, occlusion geometry, and current kinematics.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPISODE_KEYS = ["scenario", "scenario_index", "episode", "seed"]
BASE_FEATURES = [
    "n_visible",
    "occluded_grid_fraction",
    "robot_distance_m",
    "nearest_visible_clearance_m",
    "visible_neighbors_2m",
    "rel_x",
    "rel_y",
    "vx",
    "vy",
    "speed",
]
MOTION_FEATURES = [
    "delta_vx",
    "delta_vy",
    "speed_delta",
    "heading_change_rad",
    "acceleration_mps2",
    "window_delta_vx",
    "window_delta_vy",
    "window_speed_delta",
    "window_heading_change_rad",
    "velocity_std_4",
]
LABEL = "never_seen_hidden_within_2m"


def _load_episode_pair(bayes_path, oracle_path):
    bayes = pd.read_csv(bayes_path)
    oracle = pd.read_csv(oracle_path)
    for name, frame in (("bayes", bayes), ("oracle", oracle)):
        if frame.duplicated(EPISODE_KEYS).any():
            raise ValueError(f"{name} contains duplicate episode keys")
    pair = bayes.merge(
        oracle[EPISODE_KEYS + ["outcome"]],
        on=EPISODE_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_bayes", "_oracle"),
    )
    if len(pair) != len(bayes) or len(pair) != len(oracle):
        raise ValueError(
            f"episode pairing incomplete: bayes={len(bayes)} "
            f"oracle={len(oracle)} paired={len(pair)}")
    return pair


def analyze_gate1(bayes_path, oracle_path):
    pair = _load_episode_pair(bayes_path, oracle_path)
    pair["bayes_failure"] = pair.outcome_bayes != "success"
    pair["oracle_rescue"] = (
        pair.bayes_failure & (pair.outcome_oracle == "success"))
    pair["oracle_harm"] = (
        (pair.outcome_bayes == "success")
        & (pair.outcome_oracle != "success"))

    strata = {}
    for exposure in ("had_unseen_hidden", "near_unseen_hidden_person_steps"):
        exposed = (pair[exposure] > 0).astype(int)
        failures = pair[pair.bayes_failure].copy()
        failure_exposed = (failures[exposure] > 0).astype(int)
        table = np.array([
            [int(((failure_exposed == 1) & failures.oracle_rescue).sum()),
             int(((failure_exposed == 1) & ~failures.oracle_rescue).sum())],
            [int(((failure_exposed == 0) & failures.oracle_rescue).sum()),
             int(((failure_exposed == 0) & ~failures.oracle_rescue).sum())],
        ])
        odds, pvalue = fisher_exact(table)
        strata[exposure] = {
            "episodes_exposed": int(exposed.sum()),
            "rescue_table_among_bayes_failures": table.tolist(),
            "rescue_rate_exposed": (
                float(table[0, 0] / table[0].sum()) if table[0].sum() else None),
            "rescue_rate_unexposed": (
                float(table[1, 0] / table[1].sum()) if table[1].sum() else None),
            "fisher_odds_ratio": float(odds),
            "fisher_p": float(pvalue),
        }

    rescues = pair[pair.oracle_rescue]
    failures = pair[pair.bayes_failure]
    continuous = {}
    for metric in (
            "unseen_hidden_step_fraction",
            "near_unseen_hidden_person_steps",
            "min_unseen_hidden_clearance_m"):
        rescued = failures.loc[failures.oracle_rescue, metric].dropna().to_numpy()
        not_rescued = failures.loc[~failures.oracle_rescue, metric].dropna().to_numpy()
        if len(rescued) and len(not_rescued):
            _, pvalue = mannwhitneyu(
                rescued, not_rescued, alternative="two-sided")
        else:
            pvalue = np.nan
        continuous[metric] = {
            "rescued_mean": float(np.mean(rescued)) if len(rescued) else None,
            "not_rescued_mean": (
                float(np.mean(not_rescued)) if len(not_rescued) else None),
            "mannwhitney_p": float(pvalue),
        }
    return {
        "episodes": len(pair),
        "bayes_sr": float((pair.outcome_bayes == "success").mean()),
        "oracle_sr": float((pair.outcome_oracle == "success").mean()),
        "oracle_rescues": int(pair.oracle_rescue.sum()),
        "oracle_harms": int(pair.oracle_harm.sum()),
        "rescue_fraction_with_any_never_seen": (
            float((rescues.had_unseen_hidden > 0).mean()) if len(rescues) else None),
        "rescue_fraction_with_near_never_seen": (
            float((rescues.near_unseen_hidden_person_steps > 0).mean())
            if len(rescues) else None),
        "strata": strata,
        "continuous_exposure_among_bayes_failures": continuous,
    }


def _episode_group(frame):
    return (frame.scenario.astype(str) + ":" + frame.seed.astype(str))


def _recompute_contiguous_motion(frame):
    """Rebuild temporal features from consecutive observable rows only.

    A pedestrian that disappears behind an occluder and later reappears must
    not have the two observations treated as adjacent frames.
    """
    frame = frame.sort_values(
        ["scenario", "seed", "human_id", "step"]).copy()
    keys = ["scenario", "seed", "human_id"]
    grouped = frame.groupby(keys, sort=False, group_keys=False)
    previous_step = grouped.step.shift(1)
    consecutive = frame.step.sub(previous_step).eq(1)

    # Consecutive-run length, capped at the four frames consumed below.
    breaks = (~consecutive).groupby(
        [frame[k] for k in keys], sort=False).cumsum()
    run_length = frame.groupby(
        [frame[k] for k in keys] + [breaks], sort=False).cumcount() + 1
    frame["history_len"] = np.minimum(run_length.to_numpy(), 4)

    prev_vx = grouped.vx.shift(1)
    prev_vy = grouped.vy.shift(1)
    prev_speed = grouped.speed.shift(1)
    frame["delta_vx"] = np.where(consecutive, frame.vx - prev_vx, 0.0)
    frame["delta_vy"] = np.where(consecutive, frame.vy - prev_vy, 0.0)
    frame["speed_delta"] = np.where(
        consecutive, frame.speed - prev_speed, 0.0)
    frame["acceleration_mps2"] = (
        np.hypot(frame.delta_vx, frame.delta_vy) / 0.25)

    prev_heading = np.arctan2(prev_vy, prev_vx)
    heading = np.arctan2(frame.vy, frame.vx)
    one_delta = np.arctan2(
        np.sin(heading - prev_heading), np.cos(heading - prev_heading))
    frame["heading_change_rad"] = np.where(consecutive, one_delta, 0.0)

    old_vx = grouped.vx.shift(3)
    old_vy = grouped.vy.shift(3)
    old_speed = grouped.speed.shift(3)
    full_window = frame.history_len >= 4
    frame["window_delta_vx"] = np.where(
        full_window, frame.vx - old_vx, 0.0)
    frame["window_delta_vy"] = np.where(
        full_window, frame.vy - old_vy, 0.0)
    frame["window_speed_delta"] = np.where(
        full_window, frame.speed - old_speed, 0.0)
    old_heading = np.arctan2(old_vy, old_vx)
    window_delta = np.arctan2(
        np.sin(heading - old_heading), np.cos(heading - old_heading))
    frame["window_heading_change_rad"] = np.where(
        full_window, window_delta, 0.0)

    vx_std = grouped.vx.rolling(4, min_periods=4).std(ddof=0).reset_index(
        level=keys, drop=True)
    vy_std = grouped.vy.rolling(4, min_periods=4).std(ddof=0).reset_index(
        level=keys, drop=True)
    frame["velocity_std_4"] = np.where(
        full_window, 0.5 * (vx_std + vy_std), 0.0)

    reset_window = grouped.goal_reset_recent.rolling(
        4, min_periods=1).max().reset_index(level=keys, drop=True)
    frame["goal_reset_window"] = reset_window.fillna(0).astype(int)
    return frame


def _fixed_episode_split(frame):
    groups = np.array(sorted(_episode_group(frame).unique()))
    rng = np.random.RandomState(1729)
    rng.shuffle(groups)
    cut = max(1, int(round(0.70 * len(groups))))
    train_groups = set(groups[:cut])
    group = _episode_group(frame)
    return group.isin(train_groups), ~group.isin(train_groups), group


def _make_model(kind, features):
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), features),
    ])
    if kind == "logistic":
        model = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=1729)
    elif kind == "forest":
        # Fixed nonlinear diagnostic, not a tuned research model.
        model = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=2,
            random_state=1729)
        prep = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), features),
        ])
    else:
        raise ValueError(kind)
    return Pipeline([("prep", prep), ("model", model)])


def _scores(y, probability):
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
    }


def _episode_bootstrap(y, base_p, augmented_p, groups, n_boot=1000):
    rng = np.random.RandomState(1733)
    unique = np.array(sorted(np.unique(groups)))
    deltas = {"auroc": [], "auprc": []}
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        yy = y[indices]
        if len(np.unique(yy)) < 2:
            continue
        for metric, fn in (("auroc", roc_auc_score),
                           ("auprc", average_precision_score)):
            deltas[metric].append(
                float(fn(yy, augmented_p[indices]) - fn(yy, base_p[indices])))
    return {
        metric: {
            "mean": float(np.mean(values)),
            "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])],
        }
        for metric, values in deltas.items()
    }


def analyze_gate2(step_path):
    frame = pd.read_csv(step_path)
    required = set(BASE_FEATURES + MOTION_FEATURES + [
        LABEL, "goal_reset_recent", "history_len", "scenario", "seed"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"step CSV misses columns: {missing}")
    before = len(frame)
    frame = _recompute_contiguous_motion(frame)
    frame = frame[
        (frame.goal_reset_window == 0) & (frame.history_len >= 4)
    ].copy()
    if frame[LABEL].nunique() != 2:
        raise ValueError("filtered diagnostic rows do not contain both classes")
    train_mask, test_mask, groups = _fixed_episode_split(frame)
    y_train = frame.loc[train_mask, LABEL].to_numpy(dtype=int)
    y_test = frame.loc[test_mask, LABEL].to_numpy(dtype=int)
    test_groups = groups[test_mask].to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("episode split lost a label class")

    results = {}
    for kind in ("logistic", "forest"):
        probabilities = {}
        results[kind] = {}
        for name, features in (
                ("controlled_baseline", BASE_FEATURES),
                ("plus_motion_history", BASE_FEATURES + MOTION_FEATURES)):
            model = _make_model(kind, features)
            model.fit(frame.loc[train_mask, features], y_train)
            probability = model.predict_proba(
                frame.loc[test_mask, features])[:, 1]
            probabilities[name] = probability
            results[kind][name] = _scores(y_test, probability)
        results[kind]["motion_delta"] = {
            metric: (results[kind]["plus_motion_history"][metric]
                     - results[kind]["controlled_baseline"][metric])
            for metric in ("auroc", "auprc")
        }
        results[kind]["episode_bootstrap_delta"] = _episode_bootstrap(
            y_test,
            probabilities["controlled_baseline"],
            probabilities["plus_motion_history"],
            test_groups,
        )

    # PASS is intentionally fixed before seeing the complete evaluation:
    # at least +0.03 AUPRC and a positive episode-bootstrap lower bound in
    # either a linear or a fixed nonlinear diagnostic.
    passed_by = []
    for kind in ("logistic", "forest"):
        delta = results[kind]["motion_delta"]["auprc"]
        lower = results[kind]["episode_bootstrap_delta"]["auprc"]["ci95"][0]
        if delta >= 0.03 and lower > 0.0:
            passed_by.append(kind)
    return {
        "rows_raw": before,
        "rows_after_causal_filters": len(frame),
        "episodes": int(groups.nunique()),
        "test_episodes": int(pd.Series(test_groups).nunique()),
        "positive_prevalence": float(frame[LABEL].mean()),
        "features_control_total_population": False,
        "features_include_visible_count": True,
        "true_goal_used_as_feature": False,
        "gate_pass": bool(passed_by),
        "passed_by": passed_by,
        "models": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bayes-episodes", required=True)
    parser.add_argument("--oracle-episodes", required=True)
    parser.add_argument("--bayes-steps", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "gate1_oracle_value_concentration": analyze_gate1(
            args.bayes_episodes, args.oracle_episodes),
        "gate2_observable_motion_information": analyze_gate2(args.bayes_steps),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
