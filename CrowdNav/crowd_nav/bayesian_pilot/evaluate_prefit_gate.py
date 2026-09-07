#!/usr/bin/env python3
"""Evaluate a fixed, pre-fitted GDBN against matched constant velocity."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
TOOLS_DIR = CROWD_NAV_DIR / "tools"
REPO_ROOT = CROWD_NAV_DIR.parent
for path in (str(REPO_ROOT), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fit_behavior_gdbn_modes import _cv_covariance  # noqa: E402
from crowd_nav.bayesian_pilot.evaluate_prediction_gate import (  # noqa: E402
    evaluate_dataset,
    finite,
    load_rows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prefit_model(params_dir: Path):
    gdbn_path = params_dir / "gdbn.npz"
    action_path = params_dir / "action_model.npz"
    gdbn = np.load(gdbn_path, allow_pickle=False)
    action = np.load(action_path, allow_pickle=False)
    k = int(np.asarray(gdbn["K"]).item())
    action_k = int(np.asarray(action["K"]).item())
    if k != action_k:
        raise ValueError(f"K mismatch: gdbn={k}, action_model={action_k}")

    a_list = [np.asarray(gdbn[f"A_{index}"], dtype=np.float64) for index in range(k)]
    q_list = [np.asarray(gdbn[f"Q_{index}"], dtype=np.float64) for index in range(k)]
    b_list = [np.asarray(action[f"B_{index}"], dtype=np.float64) for index in range(k)]
    c_list = [np.asarray(action[f"C_{index}"], dtype=np.float64) for index in range(k)]
    pi = np.asarray(gdbn["Pi"], dtype=np.float64)
    return k, (None, a_list, b_list, q_list, c_list, pi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--params_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--risk_radius", type=float, default=0.9)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    params_dir = Path(args.params_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_data = [
        data_dir / "train.npz",
        data_dir / "test_nominal.npz",
        data_dir / "test_nonstationary.npz",
    ]
    required_params = [
        params_dir / "gng.npz",
        params_dir / "gdbn.npz",
        params_dir / "action_model.npz",
    ]
    for path in required_data + required_params:
        if not path.is_file():
            raise FileNotFoundError(path)

    k, model = load_prefit_model(params_dir)
    train_rows = load_rows(required_data[0])
    cv_covariance = _cv_covariance(train_rows, 0.25)
    nominal = evaluate_dataset(
        load_rows(required_data[1]),
        model,
        cv_covariance,
        args.seed,
        args.risk_radius,
    )
    nonstationary = evaluate_dataset(
        load_rows(required_data[2]),
        model,
        cv_covariance,
        args.seed + 1,
        args.risk_radius,
    )

    nominal_prediction = nominal["prediction"]
    ns_prediction = nonstationary["prediction"]
    ns_risk = nonstationary["risk"]
    checks = {
        "fixed_model_is_K3": k == 3,
        "nonstationary_nll_delta_ge_0_25": (
            ns_prediction["cv_nll"] - ns_prediction["gdbn_nll"] >= 0.25
        ),
        "nonstationary_brier_improvement_ge_10pct": (
            ns_risk["gdbn_brier"] <= 0.90 * ns_risk["cv_brier"]
        ),
        "nonstationary_risk_auc_noninferior_0_01": (
            finite(ns_risk["gdbn_auc"])
            and finite(ns_risk["cv_auc"])
            and ns_risk["gdbn_auc"] >= ns_risk["cv_auc"] - 0.01
        ),
        "nonstationary_unsafe_events_ge_50": ns_risk["unsafe_events"] >= 50,
        "nominal_nll_not_worse_than_cv": (
            nominal_prediction["gdbn_nll"] <= nominal_prediction["cv_nll"]
        ),
    }
    passed = all(checks.values())
    script_path = Path(__file__).resolve()
    summary = {
        "evaluation": "fixed_paper_k3_vs_matched_cv",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "K": k,
        "params_dir": str(params_dir),
        "data_dir": str(data_dir),
        "nominal": nominal,
        "nonstationary": nonstationary,
        "risk_definition": {
            "center_distance": float(args.risk_radius),
            "combined_radii": 0.6,
            "caution_clearance": float(args.risk_radius - 0.6),
        },
        "gate_checks": checks,
        "passed": passed,
        "decision": (
            "Proceed to a preregistered paired navigation pilot."
            if passed
            else "Stop; the paper K3 predictor did not pass the offline gate."
        ),
        "sha256": {
            "script": sha256(script_path),
            "data": {path.name: sha256(path) for path in required_data},
            "params": {path.name: sha256(path) for path in required_params},
        },
    }
    json_path = output_dir / "prefit_k3_gate.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Fixed Paper K3 Prediction Gate",
        "",
        f"- Gate passed: **{passed}**",
        f"- Decision: {summary['decision']}",
        f"- Parameters: `{params_dir}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if value else ' '}] `{name}`"
        for name, value in checks.items()
    )
    lines.extend(
        [
            "",
            "## Held-out Nonstationary",
            "",
            f"- GDBN/CV NLL: {ns_prediction['gdbn_nll']:.6f} / "
            f"{ns_prediction['cv_nll']:.6f}",
            f"- GDBN/CV H4 FDE: {ns_prediction['gdbn_h4_fde']:.6f} / "
            f"{ns_prediction['cv_h4_fde']:.6f}",
            f"- GDBN/CV AUC: {ns_risk['gdbn_auc']:.6f} / "
            f"{ns_risk['cv_auc']:.6f}",
            f"- GDBN/CV Brier: {ns_risk['gdbn_brier']:.6f} / "
            f"{ns_risk['cv_brier']:.6f}",
            f"- GDBN/CV ECE: {ns_risk['gdbn_ece']:.6f} / "
            f"{ns_risk['cv_ece']:.6f}",
            f"- Unsafe events: {ns_risk['unsafe_events']}",
        ]
    )
    md_path = output_dir / "prefit_k3_gate.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[PREFIT-K3-GATE] wrote {json_path}")
    print(f"[PREFIT-K3-GATE] passed={passed}")


if __name__ == "__main__":
    main()
