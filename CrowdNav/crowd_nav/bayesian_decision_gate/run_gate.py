#!/usr/bin/env python3
"""Orchestrate the Bayesian decision gate end-to-end.

Two modes:
  --smoke  : tiny episode counts, checks the pipeline runs and produces
             well-formed output. NEVER a go/no-go signal -- statistical
             power at smoke scale is zero by construction.
  --formal : the real experiment (multi-seed, full episode counts, its own
             unique --run_id subdirectory so stale files from a previous run
             can never be glob'd in by accident). Records code/config/model/
             data SHA256 hashes before running, then applies the
             pre-registered pass/fail criteria (see ``evaluate_one_gate``
             below and README.md) to decide the outcome. The criteria are
             fixed BEFORE looking at results and must not be tuned afterward.

Pipeline: collect_dataset.py (all splits x densities x seeds) ->
          fit_models.py (fit on train_nonstationary, select K on
          validation_nonstationary) -> evaluate_prediction.py (reported,
          non-gating) -> evaluate_action_ranking.py (gating, multi-horizon) ->
          evaluate_one_gate x2 + combine_gates (TWO independent gates -- see below).

Gate-A vs Gate-B
----------------
The gate does NOT collapse to one PASS/FAIL. It answers two separate
questions and combines them:
  Gate-A: frozen_k3 (the model ALREADY deployed in belief_mdp/, i.e. Route A
          as it exists today) vs. CV.
  Gate-B: refit_k{K} (freshly fit on this experiment's own data, the best
          GDBN could plausibly do) vs. CV.
A frozen-K3 loss does not by itself mean "Bayesian belief is a dead end" --
it may just mean the CURRENT fit is bad. A refit-K loss means the belief
framework itself has no decision-level signal to extract, which is a much
stronger (and different) conclusion.

Each gate itself is a proper THREE-state result -- PASS / FAIL / INCONCLUSIVE,
never a bare bool -- see ``evaluate_one_gate``'s docstring for the exact
priority order (insufficient data and insufficient real decision-opportunities
are INCONCLUSIVE; sufficient opportunity with no meaningful disagreement is a
genuine FAIL; a clear 20-person-density regression also forces FAIL even if
5-person looks like a PASS). ``combine_gates`` turns the two gates' statuses
into one plain-language decision -- see README.md for the full table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

# Per-SEED, per-DENSITY episode counts. Codex's original review caught that
# using the FULL intended totals (400/100/300/200) at this per-seed-per-
# density granularity, times 5 seeds times 2 densities, silently inflated the
# real experiment to 10x the planned ~2000-episode budget. These counts are
# chosen so that summed over the default 5 seeds, EACH density gets the
# originally intended 400/100/300/200 totals.
SPLIT_EPISODES_FORMAL = {
    "train_nonstationary": 80,
    "validation_nonstationary": 20,
    "heldout_nonstationary": 60,
    "nominal": 40,
}
SPLIT_EPISODES_SMOKE = {
    "train_nonstationary": 6,
    "validation_nonstationary": 4,
    "heldout_nonstationary": 4,
    "nominal": 4,
}
# Extra heldout episodes collected per top-up round when event decisions are
# insufficient (see --topup_heldout below) -- NEVER a full rerun.
TOPUP_HELDOUT_EPISODES_PER_SEED = 30

DENSITIES = ("5person", "20person")
HORIZONS = (1, 3, 5)
PRIMARY_HORIZON = 5

# Pre-registered pass/fail criteria, applied identically to Gate-A and
# Gate-B. Evaluated on heldout_nonstationary's EVENT_NEAR window (only
# decisions where an intervention-affected pedestrian is actually close to
# the robot -- see evaluate_action_ranking.py's event_window_labels), 5-person
# density as the primary gate (matches the production scenario); 20-person is
# reported as a secondary generalization check and does not block by itself.
# NLL/Brier/AUC improvements alone (see evaluate_prediction.py's output) do
# NOT satisfy any of these -- only evaluate_action_ranking.py's decision-level
# metrics do.
GATE_PRIMARY_DENSITY = "5person"
GATE_SECONDARY_DENSITY = "20person"
GATE_WINDOW = "event_near"
GATE_MIN_EVENT_DECISIONS = 200
GATE_MIN_OPPORTUNITY_RATE = 0.05  # fraction of near-optimal decisions where true risk actually varies
GATE_MIN_MEANINGFUL_DISAGREEMENT_RATE = 0.05
GATE_TOP1_MUST_BEAT_CV = True
GATE_NOMINAL_REGRESSION_CI_SLACK = -0.01  # nominal regret_reduction_vs_cv CI LOWER BOUND must be >= this
GATE_SECONDARY_DENSITY_CI_SLACK = -0.01  # 20-person regret_reduction_vs_cv CI LOWER BOUND must be >= this


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        print(f"[GATE] WARNING: provenance path does not exist, hash will be empty: {path}")
        return ""
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            digest.update(str(file_path.relative_to(path)).encode("utf-8"))
            digest.update(sha256_file(file_path).encode("utf-8"))
    return digest.hexdigest()


def run(cmd: List[str]) -> None:
    print(f"[GATE] running: {' '.join(cmd)}")
    start = time.time()
    # cwd=crowd_nav/ so this experiment's own "runs/..." relative-path
    # defaults line up with every other script in this project; PYTHONPATH
    # gets REPO_ROOT (crowd_nav/'s parent) added so `-m crowd_nav.xxx` still
    # resolves the package from that cwd.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT / "crowd_nav"), env=env)
    elapsed = time.time() - start
    print(f"[GATE] finished in {elapsed:.1f}s (returncode={result.returncode})")
    if result.returncode != 0:
        raise SystemExit(f"[GATE] command failed: {' '.join(cmd)}")


def _resolve_crowd_nav_relative(path_str: str) -> Path:
    """``args.frozen_k3_params`` (and similar) are relative paths meant to be
    interpreted relative to crowd_nav/ (that's the cwd every subprocess in
    ``run()`` uses) -- but hashing happens in THIS process, whose cwd is
    whatever directory the user invoked run_gate.py from. Without this,
    ``sha256_dir`` silently returns "" for a nonexistent path instead of
    erroring, which is exactly the kind of provenance gap this function
    exists to prevent."""
    path = Path(path_str)
    return path if path.is_absolute() else (THIS_DIR.parent / path)


def record_hashes(args, output_dir: Path, refit_dir: Optional[Path], data_dir: Path) -> Dict:
    crowd_nav_dir = THIS_DIR.parent
    hashes = {
        "code_sha256": {
            name: sha256_file(THIS_DIR / name)
            for name in (
                "protocol.py", "collect_dataset.py", "fit_models.py",
                "evaluate_prediction.py", "evaluate_action_ranking.py",
                "bootstrap.py", "run_gate.py",
            )
        },
        "dependency_sha256": {
            "contracts.py": sha256_file(crowd_nav_dir / "contracts.py"),
            "gdbn.py": sha256_file(crowd_nav_dir / "gdbn.py"),
            "risk_models.py": sha256_file(crowd_nav_dir / "risk_models.py"),
            "bayesian_pilot_protocol.py": sha256_file(crowd_nav_dir / "bayesian_pilot" / "protocol.py"),
            "env_belief_mdp_config": sha256_file(crowd_nav_dir / "configs" / "env_belief_mdp.config"),
        },
        "frozen_k3_params_sha256": sha256_dir(_resolve_crowd_nav_relative(args.frozen_k3_params)),
        "refit_params_sha256": sha256_dir(refit_dir) if refit_dir else None,
        "collected_data_sha256": sha256_dir(data_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": args.run_id,
        "seeds": args.seeds,
        "mode": "formal" if args.formal else "smoke",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "provenance.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return hashes


def collect_all(python_exe: str, data_dir: Path, episode_counts: Dict[str, int], seeds: List[int]):
    for split, episodes in episode_counts.items():
        for density in DENSITIES:
            for seed in seeds:
                run([
                    python_exe, "-m", "crowd_nav.bayesian_decision_gate.collect_dataset",
                    "--split", split, "--density", density,
                    "--episodes", str(episodes), "--seed", str(seed),
                    "--output_dir", str(data_dir),
                ])


def topup_heldout(python_exe: str, data_dir: Path, seeds: List[int], extra_episodes: int):
    """Collect ADDITIONAL heldout_nonstationary episodes only (new --seed
    offset so files/episode_seeds never collide with the first round) --
    never rerun train/validation/fit. Used only when the first formal pass
    reports insufficient event decisions. Passes --suite_seed explicitly so
    the top-up episodes are still bootstrap-grouped under the ORIGINAL
    experiment seed, not a spurious 6th one."""
    for density in DENSITIES:
        for seed in seeds:
            run([
                python_exe, "-m", "crowd_nav.bayesian_decision_gate.collect_dataset",
                "--split", "heldout_nonstationary", "--density", density,
                "--episodes", str(extra_episodes), "--seed", str(seed + 500_000),
                "--suite_seed", str(seed),
                "--output_dir", str(data_dir),
            ])


def _horizon_key(split: str, density: str, horizon: int) -> str:
    return f"{split}__{density}__h{horizon}"


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _horizon_metrics(all_results: Dict, model_name: str, density: str, window: str, horizon: int) -> Optional[Dict]:
    key = _horizon_key("heldout_nonstationary", density, horizon)
    if key not in all_results:
        return None
    heldout = all_results[key]
    if model_name not in heldout["per_model"]:
        return None
    event = heldout["per_model"][model_name][window]
    event_cv = heldout["per_model"]["cv"][window]
    cv_vs = heldout["cv_vs_gdbn"].get(model_name, {}).get(window, {})
    regret_reduction = cv_vs.get("regret_reduction_vs_cv", {})
    opportunity = heldout.get("opportunity", {}).get(window, {})
    return {
        "n_decisions": event["spearman"]["n_decisions"],
        "opportunity_rate": opportunity.get("mean"),
        "meaningful_disagreement_rate": cv_vs.get("meaningful_disagreement_rate", {}).get("mean"),
        "regret_reduction_mean": regret_reduction.get("mean"),
        "regret_reduction_ci_lo": regret_reduction.get("ci_lo"),
        "regret_reduction_ci_hi": regret_reduction.get("ci_hi"),
        "top1_hit_model": event["top1_hit"]["mean"],
        "top1_hit_cv": event_cv["top1_hit"]["mean"],
        "spearman_model": event["spearman"]["mean"],
        "spearman_cv": event_cv["spearman"]["mean"],
    }


def _evaluate_secondary_density(
    all_results: Dict, model_name: str, window: str, primary_horizon: int,
) -> Dict:
    """20-person non-inferiority check (Fix 3): the paper's final scenarios
    include high-density crowds, so a model that only wins at 5-person and
    quietly regresses at 20-person cannot PASS. Returns status in
    {"OK", "FAIL", "INCONCLUSIVE"} -- INCONCLUSIVE (not OK) when 20-person
    data is too sparse to certify non-inferiority, since silently defaulting
    to "OK" would let an under-tested density pass by omission."""
    metrics = _horizon_metrics(all_results, model_name, GATE_SECONDARY_DENSITY, window, primary_horizon)
    if metrics is None or metrics["n_decisions"] < GATE_MIN_EVENT_DECISIONS:
        return {"status": "INCONCLUSIVE", "reason": "insufficient_20person_event_decisions", "metrics": metrics}
    ci_lo = metrics["regret_reduction_ci_lo"]
    if ci_lo is None or ci_lo < GATE_SECONDARY_DENSITY_CI_SLACK:
        return {"status": "FAIL", "reason": "20person_regression", "metrics": metrics}
    return {"status": "OK", "reason": None, "metrics": metrics}


def evaluate_one_gate(
    all_results: Dict,
    model_name: str,
    density: str = GATE_PRIMARY_DENSITY,
    window: str = GATE_WINDOW,
    horizons: List[int] = list(HORIZONS),
    primary_horizon: int = PRIMARY_HORIZON,
) -> Dict:
    """Evaluate the pre-registered criteria for ONE model vs. CV (Gate-A uses
    model_name='frozen_k3', Gate-B uses model_name='refit_k{K}').

    Returns a proper THREE-state ``status`` (PASS / FAIL / INCONCLUSIVE), not
    a bool -- a previous version returned only ``passed`` and appended
    "INCONCLUSIVE" text after the fact, which produced self-contradictory
    reports (a "FAIL" whose own explanation said the sample was too small to
    conclude anything). The state is decided in this fixed priority order,
    matching the reviewed spec exactly:
      1. missing/insufficient event_near decisions       -> INCONCLUSIVE
      2. insufficient opportunity_rate (protocol posed
         too few real high/low-risk choices)             -> INCONCLUSIVE
      3. sufficient opportunity but GDBN/CV rarely
         meaningfully disagree                            -> FAIL
         (the belief branch had chances to matter and didn't take them)
      4. data adequate on every count, but regret/top1/
         spearman/horizon-agreement/nominal/20-person
         checks do not all hold                            -> FAIL
      5. everything holds                                   -> PASS
    """
    per_horizon: Dict[int, Dict] = {}
    for horizon in horizons:
        metrics = _horizon_metrics(all_results, model_name, density, window, horizon)
        if metrics is not None:
            per_horizon[horizon] = metrics

    base = {"model": model_name, "density": density, "window": window, "primary_horizon": primary_horizon, "per_horizon": per_horizon}

    if primary_horizon not in per_horizon:
        return {**base, "status": "INCONCLUSIVE", "reasons": [f"missing heldout_nonstationary/{density}/h{primary_horizon} for {model_name}"], "checks": {}}

    primary = per_horizon[primary_horizon]
    checks: Dict[str, bool] = {}

    checks["sufficient_event_decisions"] = bool(primary["n_decisions"] >= GATE_MIN_EVENT_DECISIONS)
    if not checks["sufficient_event_decisions"]:
        return {**base, "status": "INCONCLUSIVE", "reasons": ["insufficient_event_decisions"], "checks": checks}

    checks["sufficient_opportunity"] = bool((primary["opportunity_rate"] or 0.0) >= GATE_MIN_OPPORTUNITY_RATE)
    if not checks["sufficient_opportunity"]:
        return {**base, "status": "INCONCLUSIVE", "reasons": ["insufficient_opportunity -- protocol did not pose enough real (risk-varying) decisions"], "checks": checks}

    checks["sufficient_disagreement"] = bool(
        (primary["meaningful_disagreement_rate"] or 0.0) >= GATE_MIN_MEANINGFUL_DISAGREEMENT_RATE
    )
    if not checks["sufficient_disagreement"]:
        return {**base, "status": "FAIL", "reasons": ["no_meaningful_disagreement -- real decision opportunities existed but the model rarely chose differently from CV"], "checks": checks}

    # From here on, data is adequate and real decisions existed: any failure
    # below is a genuine FAIL, not an underpowered/inconclusive sample.
    checks["regret_reduction_significant"] = bool(
        primary["regret_reduction_ci_lo"] is not None and primary["regret_reduction_ci_lo"] > 0.0
    )
    checks["top1_beats_cv_point_estimate"] = bool(
        not GATE_TOP1_MUST_BEAT_CV or primary["top1_hit_model"] > primary["top1_hit_cv"]
    )
    checks["spearman_noninferior"] = bool(primary["spearman_model"] >= primary["spearman_cv"] - 0.02)

    # Horizon agreement: the regret-reduction MEAN must not flip sign across
    # horizons that have enough data to be informative -- if H=1 says GDBN is
    # better and H=5 says CV is better, the open-loop counterfactual oracle's
    # approximation error dominates the signal and neither can be trusted.
    informative = [
        h for h in horizons
        if h in per_horizon and per_horizon[h]["n_decisions"] >= GATE_MIN_EVENT_DECISIONS
        and per_horizon[h]["regret_reduction_mean"] is not None
    ]
    signs = [_sign(per_horizon[h]["regret_reduction_mean"]) for h in informative]
    checks["horizon_agreement"] = bool(len(informative) < 2 or len(set(signs)) <= 1)

    nominal_key = _horizon_key("nominal", density, primary_horizon)
    nominal_ok = True
    if nominal_key in all_results:
        nominal_cv_vs = all_results[nominal_key]["cv_vs_gdbn"].get(model_name, {}).get("all", {})
        nominal_regret = nominal_cv_vs.get("regret_reduction_vs_cv", {})
        nominal_ok = bool((nominal_regret.get("ci_lo", 0.0) or 0.0) >= GATE_NOMINAL_REGRESSION_CI_SLACK)
    checks["no_nominal_regression"] = nominal_ok

    secondary = _evaluate_secondary_density(all_results, model_name, window, primary_horizon)
    checks["no_20person_regression"] = secondary["status"] != "FAIL"

    reasons = [name for name, ok in checks.items() if not ok]
    if not all(checks.values()):
        return {**base, "status": "FAIL", "reasons": reasons, "checks": checks, "density_20_person": secondary}

    if secondary["status"] == "INCONCLUSIVE":
        return {
            **base, "status": "INCONCLUSIVE",
            "reasons": ["20person_insufficient_data -- 5-person result looks like a PASS but the paper's high-density scenarios are not yet verified non-inferior"],
            "checks": checks, "density_20_person": secondary,
        }

    return {**base, "status": "PASS", "reasons": [], "checks": checks, "density_20_person": secondary}


def combine_gates(gate_a: Dict, gate_b: Dict) -> Dict:
    status_a = gate_a.get("status", "INCONCLUSIVE")
    status_b = gate_b.get("status", "INCONCLUSIVE")

    if "INCONCLUSIVE" in (status_a, status_b):
        parts = []
        for gate, label, status in ((gate_a, "Gate-A", status_a), (gate_b, "Gate-B", status_b)):
            if status == "INCONCLUSIVE":
                parts.append(f"{label} INCONCLUSIVE ({'; '.join(gate.get('reasons', [])) or 'insufficient data'})")
            else:
                parts.append(f"{label} {status}")
        decision = (
            "INCONCLUSIVE -- " + ", ".join(parts) + ". Do not treat this as a final PASS or FAIL; "
            "collect more data (see --topup_heldout) or expand the stress protocol before drawing "
            "any Route A/B conclusion."
        )
        return {"gate_a": gate_a, "gate_b": gate_b, "status": "INCONCLUSIVE", "decision": decision}

    a, b = status_a == "PASS", status_b == "PASS"
    if a and b:
        decision = (
            "Gate-A PASS, Gate-B PASS -> Continue Route A citing this gate as decisive "
            "decision-level evidence; Route B is ALSO scientifically justified if the "
            "team prefers the simpler architecture."
        )
    elif (not a) and b:
        decision = (
            "Gate-A FAIL, Gate-B PASS -> the CURRENTLY DEPLOYED frozen K3 model is not "
            "earning its complexity, but the Bayesian belief framework itself has real "
            "decision-level signal. Recommendation: refit/retrain Route A's GDBN branch "
            "on this experiment's protocol, OR seriously consider Route B (pure belief + "
            "MPC) -- do NOT conclude 'Bayesian belief is a dead end' from Gate-A alone."
        )
    elif a and (not b):
        decision = (
            "Gate-A PASS, Gate-B FAIL -> keep Route A as-is (the frozen K3 model works); "
            "do NOT switch to Route B -- refitting on fresh data made things WORSE or "
            "no-better, suggesting the frozen model's advantage may be fragile/lucky "
            "rather than a robust property of the belief framework. Investigate before "
            "trusting Gate-A's result long-term."
        )
    else:
        decision = (
            "Gate-A FAIL, Gate-B FAIL -> no GDBN variant (deployed or freshly refit) shows "
            "decision-level advantage over CV. Reject Route B as currently specified. Keep "
            "Route A's GDBN branch only if justified as a generic calibration/robustness "
            "component, not a decision-quality improvement. Do not invest further time in "
            "the GDBN direction without a materially different protocol or model."
        )

    return {"gate_a": gate_a, "gate_b": gate_b, "status": "PASS" if (a and b) else "FAIL", "decision": decision}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--topup_heldout", action="store_true", help="Only collect extra heldout episodes on top of an existing --run_id and re-evaluate; does not rerun collection/fit for other splits.")
    parser.add_argument("--seeds", default="2407,3407,4407,5407,6407")
    parser.add_argument("--run_id", default=None, help="Unique subdirectory for this run's outputs (default: timestamp). Required to avoid glob'ing stale seed files from a previous run.")
    parser.add_argument("--base_dir", default="runs/bayesian_decision_gate")
    parser.add_argument("--frozen_k3_params", default="runs/bayesian_distributional/gdbn_params_cv_residual_k3")
    parser.add_argument("--python_exe", default=sys.executable)
    args = parser.parse_args()

    if args.smoke == args.formal and not args.topup_heldout:
        raise SystemExit("[GATE] pass exactly one of --smoke or --formal")
    if args.topup_heldout and not args.run_id:
        raise SystemExit("[GATE] --topup_heldout requires --run_id of the run to extend")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.base_dir).expanduser().resolve() / run_id
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    prediction_dir = base_dir / "prediction"
    action_ranking_dir = base_dir / "action_ranking"
    report_dir = base_dir / "report"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    episode_counts = SPLIT_EPISODES_FORMAL if args.formal else SPLIT_EPISODES_SMOKE
    if not args.formal and not args.topup_heldout:
        seeds = seeds[:1]

    print(f"[GATE] run_id={run_id} mode={'formal' if args.formal else ('topup' if args.topup_heldout else 'smoke')} seeds={seeds}")

    if args.topup_heldout:
        if not data_dir.exists():
            raise SystemExit(f"[GATE] --run_id {run_id} has no existing data_dir at {data_dir}")
        topup_heldout(args.python_exe, data_dir, seeds, TOPUP_HELDOUT_EPISODES_PER_SEED)
    else:
        collect_all(args.python_exe, data_dir, episode_counts, seeds)
        run([
            args.python_exe, "-m", "crowd_nav.bayesian_decision_gate.fit_models",
            "--data_dir", str(data_dir), "--output_dir", str(models_dir),
            "--frozen_k3_params", args.frozen_k3_params,
        ])
        run([
            args.python_exe, "-m", "crowd_nav.bayesian_decision_gate.evaluate_prediction",
            "--data_dir", str(data_dir), "--models_dir", str(models_dir),
            "--frozen_k3_params", args.frozen_k3_params, "--output_dir", str(prediction_dir),
        ])

    fit_summary_path = models_dir / "fit_models_summary.json"
    refit_dir = None
    if fit_summary_path.exists():
        fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
        refit_dir = models_dir / f"refit_k{int(fit_summary['selected_K'])}"

    run([
        args.python_exe, "-m", "crowd_nav.bayesian_decision_gate.evaluate_action_ranking",
        "--data_dir", str(data_dir), "--models_dir", str(models_dir),
        "--frozen_k3_params", args.frozen_k3_params, "--output_dir", str(action_ranking_dir),
        "--horizons", ",".join(str(h) for h in HORIZONS),
    ])

    hashes = record_hashes(args, report_dir, refit_dir, data_dir)

    if not args.formal and not args.topup_heldout:
        print("[GATE] smoke run complete -- structural correctness only, NOT a go/no-go signal.")
        return

    all_results = json.loads((action_ranking_dir / "action_ranking_all.json").read_text(encoding="utf-8"))
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    refit_k = int(fit_summary["selected_K"])

    gate_a = evaluate_one_gate(all_results, "frozen_k3")
    gate_b = evaluate_one_gate(all_results, f"refit_k{refit_k}")
    combined = combine_gates(gate_a, gate_b)
    combined["provenance"] = hashes
    combined["run_id"] = run_id

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "gate_result.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))
    print(f"[GATE] Gate-A status={gate_a['status']}  Gate-B status={gate_b['status']}")
    print(f"[GATE] decision: {combined['decision']}")


if __name__ == "__main__":
    main()
