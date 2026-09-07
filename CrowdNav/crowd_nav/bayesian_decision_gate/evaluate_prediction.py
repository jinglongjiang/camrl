#!/usr/bin/env python3
"""Prediction-level metrics (ADE/FDE, NLL, Brier, ECE, AUC) for the gate.

Per the pre-registration, passing THIS evaluation alone does not count as
passing the gate -- see ``evaluate_action_ranking.py`` for the decision-level
test that actually decides the outcome. This script exists so the gate report
can show both numbers side by side (a model can look good here and still
fail the action-ranking test, which is itself informative).

Reuses ``bayesian_pilot/evaluate_prediction_gate.py``'s ``_evaluate`` (NLL/
ADE/FDE) and ``risk_calibration``/``rank_auc``/``ece`` (AUC/Brier/ECE)
functions verbatim -- they operate on per-transition dynamics matrices
(A/B/Q/Pi), not on a GDBNIntegration object, so they are reused unmodified
for both the frozen K3 model and the freshly refit model.

The CV baseline's residual covariance is estimated ONCE from
train_nonstationary(+validation_nonstationary) transitions (pooled across
densities) and FROZEN before touching nominal/heldout -- a previous version
of this script re-estimated it separately on each test split, which is
test-set leakage into CV's own predictive distribution (CV would effectively
get to "peek" at the exact noise level of the split it's being scored on).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
BAYESIAN_PILOT_DIR = THIS_DIR.parent / "bayesian_pilot"
TOOLS_DIR = THIS_DIR.parent / "tools"
for path in (str(REPO_ROOT), str(BAYESIAN_PILOT_DIR), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluate_prediction_gate import evaluate_dataset  # noqa: E402
from fit_behavior_gdbn_modes import _cv_covariance  # noqa: E402
from crowd_nav.gdbn import GDBNIntegration  # noqa: E402
from crowd_nav.bayesian_decision_gate.fit_models import (  # noqa: E402
    extract_transitions_variable,
)


def model_tuple_from_gdbn(model: GDBNIntegration):
    return (None, model.gdbn.A, model.B_action, model.gdbn.Q, None, model.gdbn.Pi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="runs/bayesian_decision_gate/data")
    parser.add_argument("--models_dir", default="runs/bayesian_decision_gate/models")
    parser.add_argument("--frozen_k3_params", default="runs/bayesian_distributional/gdbn_params_cv_residual_k3")
    parser.add_argument("--splits", default="nominal,heldout_nonstationary")
    parser.add_argument("--densities", default="5person,20person")
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--risk_radius", type=float, default=0.9)
    parser.add_argument("--output_dir", default="runs/bayesian_decision_gate/prediction")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_summary = json.loads((models_dir / "fit_models_summary.json").read_text(encoding="utf-8"))
    refit_k = int(fit_summary["selected_K"])
    refit_dir = str(models_dir / f"refit_k{refit_k}")

    frozen = GDBNIntegration(K=3, params_dir=args.frozen_k3_params, max_peds=20, random_seed=args.seed)
    refit = GDBNIntegration(K=refit_k, params_dir=refit_dir, max_peds=20, random_seed=args.seed)
    model_tuples = {
        "frozen_k3": model_tuple_from_gdbn(frozen),
        f"refit_k{refit_k}": model_tuple_from_gdbn(refit),
    }

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    densities = [d.strip() for d in args.densities.split(",") if d.strip()]

    # Freeze CV's residual covariance from TRAIN(+VALIDATION) only, pooled
    # across densities -- never re-estimated on nominal/heldout. This is the
    # single covariance used for every split's CV NLL/Brier/ECE below.
    fit_files = sorted(data_dir.glob("train_nonstationary_*.npz")) + sorted(
        data_dir.glob("validation_nonstationary_*.npz")
    )
    if not fit_files:
        raise SystemExit(f"[PRED] no train_nonstationary_*/validation_nonstationary_*.npz under {data_dir}")
    fit_rows = extract_transitions_variable(fit_files, args.dt)
    frozen_cv_cov = _cv_covariance(fit_rows, args.dt)
    print(f"[PRED] froze CV covariance from {len(fit_rows)} train+validation transitions")

    results: Dict = {
        "cv_covariance_frozen_from": [str(p) for p in fit_files],
        "cv_covariance_n_transitions": len(fit_rows),
        "splits": {},
    }
    for split in splits:
        for density in densities:
            files = sorted(data_dir.glob(f"{split}_{density}_seed*.npz"))
            if not files:
                print(f"[PRED] WARNING: no files for split={split} density={density}, skipping")
                continue
            rows = extract_transitions_variable(files, args.dt)
            key = f"{split}__{density}"
            results["splits"][key] = {"n_transitions": len(rows), "models": {}}
            for name, model_tuple in model_tuples.items():
                results["splits"][key]["models"][name] = evaluate_dataset(
                    rows, model_tuple, frozen_cv_cov, args.seed, args.risk_radius
                )
            print(f"[PRED] {key}: {len(rows)} transitions evaluated for {list(model_tuples.keys())}")

    (output_dir / "prediction_all.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[PRED] wrote {output_dir / 'prediction_all.json'}")


if __name__ == "__main__":
    main()
