#!/usr/bin/env python3
"""Evaluate the frozen D4 decision-oracle gates from persisted CSV outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _bootstrap(values: np.ndarray, seeds: list[int], rows: list[dict], *, seed: int, replicates: int) -> dict:
    if values.size == 0:
        return {"point": None, "ci_low": None, "ci_high": None, "n": 0}
    represented = sorted(set(int(row["suite_seed"]) for row in rows))
    per_seed = []
    for suite_seed in represented:
        per_seed.append(float(np.mean([
            float(row["_value"]) for row in rows if int(row["suite_seed"]) == suite_seed
        ])))
    per_seed = np.asarray(per_seed, dtype=float)
    rng = np.random.default_rng(seed)
    if per_seed.size == 1:
        boot = np.repeat(per_seed.mean(), replicates)
    else:
        sample = rng.integers(0, per_seed.size, size=(replicates, per_seed.size))
        boot = per_seed[sample].mean(axis=1)
    return {
        "point": float(per_seed.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n": int(values.size),
        "represented_suite_seeds": represented,
    }


def _metric(rows: list[dict], expression, *, seed: int, replicates: int) -> dict:
    prepared = []
    for row in rows:
        copy = dict(row)
        copy["_value"] = float(expression(row))
        prepared.append(copy)
    return _bootstrap(
        np.asarray([float(row["_value"]) for row in prepared]),
        sorted(set(int(row["suite_seed"]) for row in rows)), prepared,
        seed=seed, replicates=replicates,
    )


def _summary(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="crowd_nav/configs/decision_oracle_registry.json")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text())
    input_dir = Path(args.input).resolve()
    state_rows = _read(input_dir / "state_records.csv")
    oracle_rows = _read(input_dir / "oracle_episode_records.csv")
    scope = [row for row in state_rows if row["split"] == "test_heldout_interactive" and int(row["fsm_conflict"]) == 1]
    min_states = int(registry["gates"]["min_states"])
    replicates = int(registry["data_protocol"]["bootstrap_replicates"])
    bootstrap_seed = int(registry["data_protocol"]["bootstrap_seed"])
    details = {"scope": "test_heldout_interactive/fsm_conflict", "states": len(scope)}
    if len(scope) < min_states:
        result = {"status": "INCONCLUSIVE", "reason": "insufficient_decision_states", "details": details}
        Path(args.output).resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2)); return

    summaries = [json.loads(row["candidate_summary_json"]) for row in scope]
    structured_safe = np.asarray([float(item["structured_candidates"]["safe_candidate_exists"]) for item in summaries])
    dense_safe = np.asarray([float(item["dense_oracle_candidates"]["safe_candidate_exists"]) for item in summaries])
    legacy_safe = np.asarray([float(item["legacy_candidates"]["safe_candidate_exists"]) for item in summaries])
    def ci(values, label_offset=0):
        prepared = []
        for row, value in zip(scope, values):
            copy = dict(row); copy["_value"] = float(value); prepared.append(copy)
        return _bootstrap(values, sorted(set(int(row["suite_seed"]) for row in scope)), prepared,
                          seed=bootstrap_seed + label_offset, replicates=replicates)

    candidate_gate = {
        "structured_safe_exists": ci(structured_safe, 1),
        "legacy_safe_exists": ci(legacy_safe, 2),
        "dense_safe_exists": ci(dense_safe, 3),
    }
    gates = registry["gates"]
    structured_pass = (
        candidate_gate["structured_safe_exists"]["point"] >= float(gates["candidate_safe_exists_point"])
        and candidate_gate["structured_safe_exists"]["ci_low"] >= float(gates["candidate_safe_exists_ci_low"])
    )
    legacy_point = candidate_gate["legacy_safe_exists"]["point"]
    coverage_delta = candidate_gate["structured_safe_exists"]["point"] - legacy_point
    sampler_not_required = legacy_point >= float(gates["candidate_safe_exists_point"])
    A = {
        "status": "PASS" if structured_pass and (sampler_not_required or coverage_delta > 0.0) else "FAIL",
        "candidate_metrics": candidate_gate,
        "structured_relative_legacy_point": coverage_delta,
        "sampler_not_required": sampler_not_required,
    }

    heldout_oracle = [row for row in oracle_rows if row["split"] == "test_heldout_interactive"]
    oracle_sr = float(np.mean([row["outcome"] == "success" for row in heldout_oracle])) if heldout_oracle else 0.0
    oracle_cr = float(np.mean([row["outcome"] == "collision" for row in heldout_oracle])) if heldout_oracle else 1.0
    B = {
        "status": "PASS" if oracle_sr >= float(gates["oracle_sr_min"]) and oracle_cr <= float(gates["oracle_cr_max"]) else "FAIL",
        "episodes": len(heldout_oracle), "success_rate": oracle_sr, "collision_rate": oracle_cr,
    }

    def improvement(field_base: str, field_k4: str, offset: int) -> dict:
        rows = []
        for row in scope:
            copy = dict(row); copy["_value"] = float(row[field_base]) - float(row[field_k4]); rows.append(copy)
        return _bootstrap(np.asarray([float(row["_value"]) for row in rows]),
                          sorted(set(int(row["suite_seed"]) for row in scope)), rows,
                          seed=bootstrap_seed + offset, replicates=replicates)

    c4_k1 = improvement("k1_full_clearance_regret", "k4_full_clearance_regret", 11)
    c4_mean = improvement("k4_mean_clearance_regret", "k4_full_clearance_regret", 12)
    safe_k1 = improvement("k4_full_safe_hit", "k1_full_safe_hit", 13)
    safe_mean = improvement("k4_full_safe_hit", "k4_mean_safe_hit", 14)
    harm_k1 = improvement("k4_full_collision_regret", "k1_full_collision_regret", 15)
    harm_mean = improvement("k4_full_collision_regret", "k4_mean_collision_regret", 16)
    C = {
        "k4_vs_k1_clearance_regret_improvement": c4_k1,
        "k4_vs_mean_clearance_regret_improvement": c4_mean,
        "k4_vs_k1_safe_hit_improvement": safe_k1,
        "k4_vs_mean_safe_hit_improvement": safe_mean,
        "k4_vs_k1_collision_regret_improvement": harm_k1,
        "k4_vs_mean_collision_regret_improvement": harm_mean,
    }
    C["status"] = "PASS" if (
        c4_k1["ci_low"] > 0.0 and c4_mean["ci_low"] > 0.0
        and safe_k1["point"] >= float(gates["k4_safe_hit_improvement_percentage_points"]) / 100.0
        and safe_k1["ci_low"] > 0.0
        and safe_mean["ci_low"] > 0.0
        and harm_k1["ci_high"] <= float(gates["k4_collision_regret_harm_percentage_points"]) / 100.0
        and harm_mean["ci_high"] <= float(gates["k4_collision_regret_harm_percentage_points"]) / 100.0
    ) else "FAIL"

    weighted_rows = [row for row in scope if row.get("structured_safe_candidate_exists", "1")]
    weighted_fraction = float(np.mean([float(row["weighted_collision_when_safe_exists"]) for row in weighted_rows])) if weighted_rows else 0.0
    D = {
        "status": "PASS" if weighted_fraction >= float(gates["weighted_action_collision_min_percentage_points"]) / 100.0 else "FAIL",
        "weighted_collision_when_structured_safe_exists": weighted_fraction,
        "states": len(weighted_rows),
    }
    # The audit records the existence flag inside candidate_summary_json; make
    # the D calculation explicit instead of trusting a missing CSV shortcut.
    weighted_values = []
    for row, summary in zip(scope, summaries):
        if summary["structured_candidates"]["safe_candidate_exists"]:
            weighted_values.append(float(row["weighted_collision_when_safe_exists"]))
    D["weighted_collision_when_structured_safe_exists"] = float(np.mean(weighted_values)) if weighted_values else 0.0
    D["states"] = len(weighted_values)
    D["status"] = "PASS" if D["weighted_collision_when_structured_safe_exists"] >= float(gates["weighted_action_collision_min_percentage_points"]) / 100.0 else "FAIL"

    overall = "PASS" if all(item["status"] == "PASS" for item in (A, B, C, D)) else "FAIL"
    result = {
        "status": overall,
        "eligible_for_discrete_controller": overall == "PASS",
        "registry": _summary(registry_path),
        "input": str(input_dir),
        "gates": {"A_candidate_feasibility": A, "B_oracle_upper_bound": B, "C_bayesian_ranking": C, "D_weighted_average_pathology": D},
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if overall != "PASS":
        raise SystemExit(20)


if __name__ == "__main__":
    main()
