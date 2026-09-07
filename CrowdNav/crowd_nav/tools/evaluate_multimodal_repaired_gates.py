#!/usr/bin/env python3
"""Fail-closed evaluator for the MM-S1 repaired evaluation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _health_gate(root: Path, registry: dict) -> dict:
    spec = registry["health_gate"]
    summary_path = root / spec["audit_path"]
    collisions_path = summary_path.with_name("collision_cases.json")
    if not summary_path.is_file() or not collisions_path.is_file():
        return {"status": "INCOMPLETE", "reason": "health_audit_outputs_missing"}
    summary = json.loads(summary_path.read_text())
    collisions = json.loads(collisions_path.read_text())
    checks = {}
    checks["pilot_reproduction"] = (
        not spec["require_exact_pilot_reproduction"]
        or summary["reference_pilot_reproduction"]["status"] == "PASS"
    )
    reciprocal = summary["results"]["reciprocal_orca"]
    checks["reciprocal_per_split"] = all(
        metrics["success_rate"] >= spec["min_reciprocal_orca_success_rate_per_split"]
        and metrics["collision_rate"] <= spec["max_reciprocal_orca_collision_rate_per_split"]
        for metrics in reciprocal.values()
    )
    checks["fallbacks"] = all(
        metrics["fallback_count"] <= spec["max_controller_fallbacks"]
        for variant in summary["results"].values() for metrics in variant.values()
    )
    geometry_ok = True
    for collision in collisions:
        before = collision["before"]
        after = collision["after"]
        geometry_ok &= abs(
            (after["center_distance"] - after["radius_sum"]) - after["clearance"]
        ) <= 1e-10
        geometry_ok &= after["clearance"] < 0.0
        geometry_ok &= before["clearance"] >= 0.0
    checks["collision_geometry"] = (
        not spec["require_collision_geometry_consistency"] or geometry_ok
    )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "collision_cases": len(collisions),
        "summary_sha256": _sha256(summary_path),
        "collision_cases_sha256": _sha256(collisions_path),
    }


def _action_gate(root: Path, registry: dict, action_summary_path: Path) -> dict:
    spec = registry["action_sensitivity"]
    if not action_summary_path.is_file():
        return {"status": "INCOMPLETE", "reason": "action_sensitivity_summary_missing"}
    summary = json.loads(action_summary_path.read_text())
    if summary.get("responsibility_mode") != spec["responsibility_mode"]:
        return {
            "status": "FAIL",
            "reason": "action_sensitivity_responsibility_mode_mismatch",
            "expected": spec["responsibility_mode"],
            "actual": summary.get("responsibility_mode"),
        }
    scope = summary["scopes"].get(spec["gate_scope"])
    if scope is None:
        return {"status": "FAIL", "reason": "registered_gate_scope_missing"}
    subset = scope.get(spec["gate_subset"])
    if subset is None:
        return {"status": "FAIL", "reason": "registered_gate_subset_missing"}
    if int(subset["states"]) < int(spec["minimum_gate_states"]):
        return {
            "status": "INCONCLUSIVE",
            "reason": "insufficient_gate_states",
            "states": int(subset["states"]),
            "minimum_gate_states": int(spec["minimum_gate_states"]),
        }
    checks = {}
    details = {}
    fraction_key = f"fraction_gt_{spec['meaningful_action_l2']}"
    for pair in spec["required_pairs"]:
        metrics = subset[pair]
        mean_ci_low = float(metrics["mean_l2"]["ci_low"])
        fraction_ci_low = float(metrics[fraction_key]["ci_low"])
        pair_checks = {
            "mean_l2_ci_low": mean_ci_low >= spec["minimum_mean_l2_ci_low"],
            "meaningful_fraction_ci_low": (
                fraction_ci_low >= spec["minimum_meaningful_fraction_ci_low"]
            ),
            "all_suite_seeds_represented": (
                not metrics["mean_l2"]["missing_suite_seeds"]
                and not metrics[fraction_key]["missing_suite_seeds"]
            ),
        }
        checks[pair] = all(pair_checks.values())
        details[pair] = {
            "checks": pair_checks,
            "mean_l2_ci_low": mean_ci_low,
            "meaningful_fraction_ci_low": fraction_ci_low,
        }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "states": int(subset["states"]),
        "scope": spec["gate_scope"],
        "subset": spec["gate_subset"],
        "pair_pass": checks,
        "details": details,
        "summary_sha256": _sha256(action_summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="crowd_nav/configs/multimodal_repaired_eval_registry.json",
    )
    parser.add_argument(
        "--action-summary",
        default=(
            "runs/bayesian_brne/mm_s1_multimodal_brne_20260805/diagnostics/"
            "action_sensitivity_robot_only_formal/summary.json"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text())
    for name in ("selected", "k1"):
        path = root / registry["frozen_artifacts"][f"{name}_path"]
        expected = registry["frozen_artifacts"][f"{name}_sha256"]
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen {name} artifact missing or hash mismatch: {path}")

    health = _health_gate(root, registry)
    action = _action_gate(root, registry, Path(args.action_summary).resolve())
    eligible = health["status"] == "PASS" and action["status"] == "PASS"
    result = {
        "status": "PASS" if eligible else (
            "INCOMPLETE" if "INCOMPLETE" in (health["status"], action["status"])
            else "INCONCLUSIVE" if "INCONCLUSIVE" in (health["status"], action["status"])
            else "FAIL"
        ),
        "eligible_for_repaired_pilot": eligible,
        "health_gate": health,
        "action_sensitivity_gate": action,
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not eligible:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
