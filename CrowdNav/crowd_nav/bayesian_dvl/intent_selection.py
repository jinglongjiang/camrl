"""Pre-registered milestone selection -- an INDEPENDENT evaluation path.

Deliberately a separate module, not part of the training chain. ``code_sha256()``
hashes a fixed list of 14 main-chain files and this file is not one of them, so
adding it cannot change the training code hash. That matters twice over:

  * checkpoints already written stay loadable (nothing here alters the schema,
    the training contract, the action grid or the scene registry);
  * mean/cv must run on byte-identical training code to the full arm, and they
    still do -- this module only reads checkpoints.

It records its own ``selection_evaluator_sha256`` so a selection result can be
tied to the evaluator that produced it, separately from the training hash.

Why the whole exercise exists: on the n=20 development set the last four
checks of the full run read 0.70 / 0.90 / 0.95 with overlapping confidence
intervals, and the previous (lambda=380) run showed 1.00 on that same n=20
set at ep5000-7000 while a paired n=100 evaluation later measured `standard`
falling to 0.83. Twenty episodes cannot rank these milestones; 100 paired
episodes per scenario can.

Pairing is strict: every checkpoint is scored on the SAME episode seeds, and
the initial-state hashes are compared across checkpoints so a mismatch is an
error rather than a silent confound.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from crowd_nav.bayesian_dvl.intent_config import IntentTrainingConfig, load_intent_training_config
from crowd_nav.bayesian_dvl.intent_evaluate import run_persistent_evaluation
from crowd_nav.bayesian_dvl.intent_policy import HUMAN_FEATURE_DIM_V6, load_intent_checkpoint
from crowd_nav.bayesian_dvl.intent_runtime_config import ActionGridSpec
from crowd_nav.bayesian_dvl.intent_train import (
    STANDARD_SELECTION_DEV_SEEDS, select_checkpoint,
)
from crowd_nav.bayesian_dvl.junction_scenario import JUNCTION_CROWD_SELECTION_DEV_SEEDS
from crowd_nav.bayesian_dvl.model import DistributionalValueModel


class SelectionError(ValueError):
    pass


MILESTONES = (2500, 5000, 7500, 10000)


def selection_evaluator_sha256() -> str:
    """Identity of the EVALUATOR, kept separate from the training code hash."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("intent_selection.py", "intent_evaluate.py"):
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256((root / name).read_bytes()).digest())
    return digest.hexdigest()


def _summarise(rows: Sequence[dict]) -> Dict[str, float]:
    n = len(rows)
    f = lambda k: [float(r[k]) for r in rows]
    clear = sorted(f("min_clearance"))
    return {
        "n": n,
        "success_rate": sum(r["outcome"] == "success" for r in rows) / n,
        "collision_rate": sum(r["outcome"] == "collision" for r in rows) / n,
        "timeout_rate": sum(r["outcome"] == "timeout" for r in rows) / n,
        "mean_speed": float(np.mean(f("mean_speed"))),
        "navigation_time": float(np.mean(f("elapsed_time"))),
        "mean_path_ratio": float(np.mean(f("path_ratio"))),
        "mean_min_clearance": float(np.mean(clear)),
        "clearance_p5": float(clear[max(0, int(0.05 * n) - 1)]),
        "negative_clearance_rate": sum(1 for x in clear if x < 0) / n,
        "discomfort_frequency": float(np.mean(f("discomfort_frequency"))),
    }


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(a_success: Sequence[bool], b_success: Sequence[bool]) -> Dict[str, float]:
    """Exact two-sided McNemar on PAIRED episodes."""
    if len(a_success) != len(b_success):
        raise SelectionError("paired test needs equal-length outcome vectors")
    b01 = sum(1 for x, y in zip(a_success, b_success) if x and not y)
    b10 = sum(1 for x, y in zip(a_success, b_success) if y and not x)
    n = b01 + b10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    p = 2.0 * sum(math.comb(n, k) for k in range(0, min(b01, b10) + 1)) / (2 ** n)
    return {"a_only": b01, "b_only": b10, "p_value": min(p, 1.0)}


def write_reclassification_manifest(out_dir: Path) -> Path:
    """Record the seed reclassification BEFORE any junction result is seen.

    Written first, deliberately: a protocol change decided after looking at
    the numbers it affects is not a protocol change, it is a choice.
    """
    import datetime
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "seed_reclassification.json"
    payload = {
        "written_before_seeing_junction_results": True,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "block": "96901-97000",
        "old_role": "paper_heldout_junction_stress",
        "new_role": "checkpoint_selection_dev",
        "reason": (
            "97601-97700 was frozen as junction selection-dev in the seed inventory but never added "
            "to JunctionCrowdEpisodeConfig's own allowlist. Adding it requires editing "
            "junction_scenario.py, one of the 14 files code_sha256() hashes; the full arm is already "
            "trained under the current hash and mean/cv must run on byte-identical training code, so "
            "that edit would destroy the three-arm comparison. 96901-97000 is already accepted by the "
            "scenario builder, has 100 seeds and never entered training."),
        "consequences": [
            "96901-97000 must NEVER again be used for a paper final performance number.",
            "The paper's final evaluation continues to use the independent Test8 protocol "
            "(base seed 42) and is unaffected.",
            "These seeds run the HELD-OUT junction variant (shifted speeds, wider fork), which is "
            "harder than the junction_crowd distribution trained on; the SR >= 0.90 bar is applied "
            "to that harder variant.",
            "NOT enforced in code: eval-stress still reads this block, and adding a guard would mean "
            "editing intent_train_cli.py, which is also hashed. This manifest is the record.",
        ],
        "unchanged": ["junction_scenario.py", "training code", "formal plan", "corpus", "checkpoints"],
        "selection_evaluator_sha256": selection_evaluator_sha256(),
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"seed reclassification recorded BEFORE evaluation -> {path}", flush=True)
    return path


def evaluate_milestones(run_dir: Path, env_config: Path, cfg: IntentTrainingConfig,
                        out_dir: Path, device: str = "cpu",
                        episodes: int = 100) -> dict:
    """Score every milestone on BOTH scenarios with strictly paired episodes."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = ActionGridSpec.from_env_config(str(env_config))
    action_table = np.asarray(grid.build_action_table(), dtype=np.float64)

    # V2: junction selection-dev is its OWN frozen block, 2_400_000-2_400_099,
    # wired into JUNCTION_CROWD_SEED_ROLES so the scenario builder accepts it
    # by role. The V1 workaround -- reclassifying the held-out block
    # 96901-97000 into selection-dev because the real selection block had
    # never been added to the scenario's allowlist -- is gone with V1, and
    # the mechanism-audit block (2_010_000-2_010_099) must NOT be reused for
    # selection: selecting a checkpoint on the same episodes that certified
    # the candidate model would couple the two.
    jobs = {
        "standard": [("standard", s, False) for s in list(STANDARD_SELECTION_DEV_SEEDS)[:episodes]],
        "junction_crowd": [("junction_crowd", s, True)
                           for s in list(JUNCTION_CROWD_SELECTION_DEV_SEEDS)[:episodes]],
    }
    results, per_episode, identities = {}, {}, {}
    for order, ep in enumerate(MILESTONES, start=1):
        ck = run_dir / f"milestone_ep{ep:06d}.pth"
        if not ck.exists():
            raise SelectionError(f"missing milestone checkpoint {ck}")
        model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
        # NOTE: no code-hash check here -- load_intent_checkpoint validates the
        # schema, feature schema, training contract, action grid and scene
        # registry, which is exactly what must match for a fair comparison.
        load_intent_checkpoint(str(ck), model)
        model.to(device).eval()
        scen_summary = {}
        for scenario, job in jobs.items():
            # ONE directory per (milestone, scenario). Both scenarios used to
            # share `ep{N}/selection_dev/episodes.csv`, so resume appended the
            # junction rows to the standard ones and the file held 200 rows for
            # 100 jobs. The row-count check below caught it rather than letting
            # two scenarios be averaged together.
            csv_path = run_persistent_evaluation(
                env_config, model, action_table, out_dir / f"ep{ep:06d}_{scenario}",
                "selection_dev", job,
                belief_mode="full", n_samples=cfg.future_n_samples, horizon=cfg.future_horizon,
                # resume by episode identity: completed episodes are reused,
                # so the standard rows already scored are NOT re-run
                device=device, resume=True)
            rows = list(csv.DictReader(open(csv_path)))
            if len(rows) != len(job):
                raise SelectionError(f"{ck.name}/{scenario}: {len(rows)} rows for {len(job)} jobs")
            scen_summary[scenario] = _summarise(rows)
            per_episode[(ep, scenario)] = [r["outcome"] == "success" for r in rows]
            identities.setdefault(scenario, []).append(
                (ep, [r["episode_seed"] for r in rows], [r["initial_state_hash"] for r in rows]))
            print(f"  ep{ep:06d} {scenario:>14}: SR={scen_summary[scenario]['success_rate']:.3f} "
                  f"CR={scen_summary[scenario]['collision_rate']:.3f} "
                  f"speed={scen_summary[scenario]['mean_speed']:.3f} "
                  f"clr={scen_summary[scenario]['mean_min_clearance']:.4f} "
                  f"neg={scen_summary[scenario]['negative_clearance_rate']:.2f}", flush=True)
        results[ep] = {"name": f"ep{ep:06d}", "order": order, "scenarios": scen_summary}

    # strict pairing: identical episode seeds AND identical initial states
    for scenario, entries in identities.items():
        base_ep, base_seeds, base_hashes = entries[0]
        for ep, seeds, hashes in entries[1:]:
            if seeds != base_seeds:
                raise SelectionError(f"{scenario}: ep{ep} used different episode seeds than ep{base_ep}")
            if hashes != base_hashes:
                raise SelectionError(
                    f"{scenario}: ep{ep} initial-state hashes differ from ep{base_ep}; "
                    f"the comparison would be confounded")
    print("  strict pairing verified: identical episode seeds AND initial-state hashes", flush=True)
    return {"results": results, "per_episode": per_episode}


def report(evaluated: dict, out_dir: Path, run_dir: Path, min_sr: float = 0.90) -> dict:
    results, per_episode = evaluated["results"], evaluated["per_episode"]
    print("\n=== per-milestone summary ===", flush=True)
    for ep, r in results.items():
        for scenario, s in r["scenarios"].items():
            lo, hi = _wilson(round(s["success_rate"] * s["n"]), s["n"])
            print(f"  ep{ep:06d} {scenario:>14}  SR={s['success_rate']:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"CR={s['collision_rate']:.3f}  TR={s['timeout_rate']:.3f}  "
                  f"speed={s['mean_speed']:.3f}  time={s['navigation_time']:.2f}  "
                  f"path={s['mean_path_ratio']:.3f}  clr={s['mean_min_clearance']:.4f}  "
                  f"p5={s['clearance_p5']:+.4f}  neg={s['negative_clearance_rate']:.2f}  "
                  f"disc={s['discomfort_frequency']:.3f}", flush=True)

    print("\n=== paired McNemar (same episodes, same initial states) ===", flush=True)
    tests = {}
    eps = list(results)
    for scenario in ("standard", "junction_crowd"):
        for i in range(len(eps)):
            for j in range(i + 1, len(eps)):
                a, b = eps[i], eps[j]
                t = mcnemar(per_episode[(a, scenario)], per_episode[(b, scenario)])
                tests[f"{scenario}:ep{a}_vs_ep{b}"] = t
                if t["p_value"] < 0.05:
                    print(f"  {scenario:>14} ep{a} vs ep{b}: {a}-only={t['a_only']} "
                          f"{b}-only={t['b_only']}  p={t['p_value']:.4f}  SIGNIFICANT", flush=True)
    if not any(t["p_value"] < 0.05 for t in tests.values()):
        print("  no pair differs significantly", flush=True)

    print(f"\n=== frozen selection rule (both scenarios SR >= {min_sr}) ===", flush=True)
    for ep, r in results.items():
        srs = {k: v["success_rate"] for k, v in r["scenarios"].items()}
        ok = all(v >= min_sr for v in srs.values())
        print(f"  ep{ep:06d}: {srs}  ->  {'ELIGIBLE' if ok else 'eliminated'}", flush=True)
    chosen = None
    try:
        chosen = select_checkpoint(list(results.values()), min_sr=min_sr)
        print(f"\nSELECTED: {chosen['name']}", flush=True)
    except Exception as exc:
        print(f"\nRUN FAILED: {exc}", flush=True)

    payload = {
        "run_dir": str(run_dir),
        "selection_evaluator_sha256": selection_evaluator_sha256(),
        "min_sr": min_sr,
        "milestones": {str(k): v for k, v in results.items()},
        "mcnemar": tests,
        "selected": chosen["name"] if chosen else None,
    }
    (out_dir / "selection_result.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_dir / 'selection_result.json'}", flush=True)
    return payload


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="pre-registered milestone selection (independent evaluator)")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--env-config", type=Path,
                   default=Path("crowd_nav/configs/env_bayesian_dvl.config"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--episodes", type=int, default=100)
    args = p.parse_args(argv)
    cfg = load_intent_training_config(args.config) if args.config else load_intent_training_config()
    out = args.out_dir or (Path(args.run_dir) / "selection_dev")
    print(f"selection evaluator {selection_evaluator_sha256()[:16]}  "
          f"({args.episodes} paired episodes x 2 scenarios x {len(MILESTONES)} milestones)", flush=True)
    write_reclassification_manifest(out)
    ev = evaluate_milestones(Path(args.run_dir), args.env_config, cfg, out,
                             device=args.device, episodes=args.episodes)
    report(ev, out, Path(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
