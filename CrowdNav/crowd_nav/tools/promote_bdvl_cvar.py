#!/usr/bin/env python3
"""Create a hash-bound BDVL CVaR promotion report.

The evaluator accepts only reports produced by this contract. A hand-written
``{"passed": true}`` is intentionally insufficient.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_nav.bayesian_dvl.config import load_and_validate_registry  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, sha256_of_file  # noqa: E402
from crowd_nav.bayesian_dvl.statistics import (  # noqa: E402
    CALIBRATION_TAUS,
    CVaR_PROMOTION_SCHEMA,
    check_cvar_promotion_gate,
)
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-result", required=True, help="JSON containing risk-neutral/CVaR metrics and calibration metrics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    registry_path = Path(args.registry) if Path(args.registry).is_absolute() else PACKAGE_ROOT / args.registry
    artifact_path = Path(args.artifact) if Path(args.artifact).is_absolute() else PACKAGE_ROOT / args.artifact
    checkpoint_path = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else PACKAGE_ROOT / args.checkpoint
    validation_path = Path(args.validation_result) if Path(args.validation_result).is_absolute() else PACKAGE_ROOT / args.validation_result
    registry = load_and_validate_registry(str(registry_path))
    artifact = SBKHMMArtifact.load(str(artifact_path), expect_tier="production")
    validation = json.loads(validation_path.read_text())
    metrics = validation.get("metrics", validation)
    calibration = metrics.get("calibration", validation.get("calibration_metrics", {}))
    coverage_errors = {float(k): float(v) for k, v in calibration.get("coverage_errors", {}).items()}
    gate_metrics = {
        "quantile_crossing_status": calibration.get("quantile_crossing_status"),
        "quantile_crossing_rate": calibration.get("quantile_crossing_rate"),
    }
    if gate_metrics["quantile_crossing_status"] != "OK":
        raise SystemExit("validation result has no fixed-IQN quantile crossing measurement")
    gate = check_cvar_promotion_gate(
        score_bound_violations=int(metrics.get("score_bound_violations", 0)),
        quantile_crossing_rate=float(calibration["quantile_crossing_rate"]),
        coverage_errors=coverage_errors,
        cvar_timeout_rate=float(metrics["cvar_timeout_rate"]),
        risk_neutral_timeout_rate=float(metrics["risk_neutral_timeout_rate"]),
        cvar_collision_rate=float(metrics["cvar_collision_rate"]),
        risk_neutral_collision_rate=float(metrics["risk_neutral_collision_rate"]),
    )
    if set(coverage_errors) != set(CALIBRATION_TAUS):
        raise SystemExit("validation result does not cover the complete frozen tau grid")
    payload = {
        "schema": CVaR_PROMOTION_SCHEMA,
        "passed": bool(gate["passed"]),
        "reasons": gate["reasons"],
        "registry_sha256": registry["content_sha256"],
        "artifact_sha256": artifact.content_hash(),
        "checkpoint_sha256": sha256_of_file(str(checkpoint_path)),
        "validation_result_sha256": sha256_of_file(str(validation_path)),
        "quantile_crossing_source": "fixed_iqn_tau_outputs",
        "gate_metrics": {**gate_metrics, **calibration},
        "coverage_errors": coverage_errors,
    }
    output_path = Path(args.output) if Path(args.output).is_absolute() else PACKAGE_ROOT / args.output
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing promotion report: {output_path}")
    atomic_write_json(str(output_path), payload)
    print(f"PROMOTE_BDVL_CVAR {'PASS' if payload['passed'] else 'FAIL'} report={output_path}")


if __name__ == "__main__":
    main()
