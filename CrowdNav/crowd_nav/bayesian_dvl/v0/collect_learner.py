"""Collect learner states, verify the compact record, and run the go/no-go.

The go/no-go decides whether the training change is worth making AT ALL. If the
student's greedy action already falls inside ORCA's equivalence class on the
states it actually reaches, then a ranking term there has nothing to correct --
exactly what was measured on expert states, where the term sat at 1e-8..1e-5 and
1000 updates of it changed nothing it was responsible for. Finding that again
here would refute the learner-state hypothesis, and the honest move is to report
it and stop rather than train anyway.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
from crowd_nav.bayesian_dvl.intent_runtime_config import HUMAN_FEATURE_DIM_V7
from crowd_nav.bayesian_dvl.v0.learner_states import collect, rebuild_rows
from crowd_nav.bayesian_dvl.v0.scoring import CandidateInputs, score_from_inputs
from crowd_nav.bayesian_dvl.v0.value_model import ScalarValueModel

REPO = Path(__file__).resolve().parents[3]
ENVCFG = REPO / "crowd_nav/configs/env_bayesian_dvl.config"
CFGPATH = REPO / "crowd_nav/configs/train_intent_bdvl.config"
FULL5000 = Path("/root/bdvl_diagnostics/v0/full5000")


def load_models(device):
    ms = {}
    for s in (98204, 98205, 98206):
        p = FULL5000 / f"full5000_s{s}/v0.pth"
        if not p.exists():
            continue
        m = ScalarValueModel().to(device)
        m.load_state_dict(torch.load(str(p), map_location=device)["model_state_dict"])
        m.eval(); ms[f"s{s}"] = m
    return ms


def verify_rebuild(rec) -> float:
    """The compact record must reproduce what build_candidate_inputs returned."""
    rows = rebuild_rows(rec)
    base = np.asarray(rec["base_rows"])
    # every candidate shares the non-robot columns with the stored base block
    from crowd_nav.bayesian_dvl.intent_runtime_config import HUMAN_SCALAR_IDX_ROBOT_RELATIVE
    other = [c for c in range(HUMAN_FEATURE_DIM_V7) if c not in HUMAN_SCALAR_IDX_ROBOT_RELATIVE]
    return float(np.abs(rows[:, :, other] - base[None, :, other]).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=10000)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--max-steps", type=int, default=141)
    ap.add_argument("--out", type=Path, default=Path("/root/bdvl_diagnostics/v0/learner"))
    a = ap.parse_args()

    cfg = load_intent_training_config(str(CFGPATH))
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = load_models(dev)
    print(f"checkpoints: {sorted(models)}   device={dev}   target={a.n_states} states")
    if not models:
        raise SystemExit("no FULL-5000 checkpoints found")

    recs = collect(cfg, ENVCFG, models, a.n_states, a.mode, dev, a.max_steps)
    d = verify_rebuild(recs[0])
    print(f"compact-record rebuild check: max|diff| on non-robot columns = {d:.3e}")
    if d > 1e-6:
        raise SystemExit("compact record does not reproduce build_candidate_inputs; stop")

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"records": recs, "mode": a.mode,
                "future_horizon": int(cfg.future_horizon),
                "future_n_samples": int(cfg.future_n_samples)}, str(a.out / "learner_states.pth"))
    print(f"saved -> {a.out/'learner_states.pth'}")

    # ---------------- go / no-go, no training -------------------------------
    print("\n" + "=" * 76)
    print("GO/NO-GO: is the SCORE ordering already correct on learner states?")
    print("=" * 76)
    report = {}
    for name, m in sorted(models.items()):
        sub = [r for r in recs if r["checkpoint"] == name]
        hits, viol, deltas = 0, 0, []
        for r in sub:
            ci = CandidateInputs(rebuild_rows(r), np.asarray(r["mask"]),
                                 np.asarray(r["robot_feats"]), np.asarray(r["rewards"], dtype=np.float64),
                                 np.asarray(r["terminal"]))
            with torch.no_grad():
                s, _, _ = score_from_inputs(m, ci, cfg.gamma, dev)
            s = s.cpu().numpy(); em = np.asarray(r["expert_mask"])
            hits += bool(em[int(np.argmax(s))])
            d = float(s[~em].max() - s[em].max())
            deltas.append(d); viol += (d > 0)
        n = max(len(sub), 1)
        report[name] = dict(n=len(sub), hit_rate=hits / n, violation_rate=viol / n,
                            delta_mean=float(np.mean(deltas)), delta_p90=float(np.percentile(deltas, 90)))
        print(f"  {name}  n={len(sub):>5}  greedy in ORCA class {100*hits/n:5.1f}%   "
              f"violation (delta>0) {100*viol/n:5.1f}%   "
              f"delta mean {np.mean(deltas):+.4f}  p90 {np.percentile(deltas,90):+.4f}")
    (a.out / "gonogo.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    v = np.mean([r["violation_rate"] for r in report.values()])
    print(f"\nmean violation rate across checkpoints: {100*v:.1f}%")
    print("VERDICT:", "GO -- learner states DO carry ranking violations to correct"
          if v > 0.15 else
          "NO-GO -- ordering is already correct here; the learner-state hypothesis is refuted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
