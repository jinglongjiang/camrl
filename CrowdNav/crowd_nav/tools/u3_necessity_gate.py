#!/usr/bin/env python3
"""Upgrade U3: necessity gate for the action-conditioned sticky Bayesian
switching AR-HMM (2026-08-03, run with the user's explicit approval despite
``audit_action_identifiability.py`` already showing a small -- though
CI-supported -- incremental predictive-power signal from the robot's
action).

This is the test that decides whether "the robot's action is a necessary
part of the model" is a defensible claim, as opposed to "the model just has
more free parameters." Three variants of the SAME model class/config/K are
fit on TRAIN and scored (held-out) on the SAME real VALIDATION data:

- ``self_only``: ``u_robot`` zeroed everywhere (TRAIN AND validation) -- so
  ``B_k`` can only ever be estimated as ~0 (its design column is constant
  zero, see ``m_step``'s docstring on the MNIW update); this is the
  "restricted" nested model.
- ``action_conditioned``: the real, unmodified model -- the "full" model.
- ``action_shuffled``: TRAIN's ``robot_actions`` are swapped between
  episodes via a derangement (same technique as
  ``order9s_shuffle_test.py``'s ``shuffle_robot_pairing``, applied to
  ``robot_actions`` specifically rather than the whole ``robot`` array,
  since only the action term needs to be broken here -- ``v_current``/
  ``context`` stay TRUE). VALIDATION is always the real, unshuffled data:
  the question is whether a model trained on a scrambled action-outcome
  correspondence predicts the REAL task as well as a properly-trained one.

Procedure:
1. Pre-registered K selection: fit ``action_conditioned`` for
   K=1..k_max on TRAIN, evaluate held-out per-sequence log-likelihood on
   real VALIDATION, and pick the SMALLEST K such that no larger K shows a
   suite-seed-bootstrap-CI-supported further improvement (same
   one-SE/minimal-sufficient-K discipline as
   ``order9s_k_plateau_audit.py`` -- explicitly NOT naive argmax).
2. At that single K*, fit ``self_only`` and ``action_shuffled`` (same
   config/priors/em settings).
3. Suite-seed block bootstrap (resample precomputed per-sequence
   log-likelihoods grouped by validation suite seed, same discipline as
   ``fit_bayesian_brne.fast_bootstrap_nll_ci``) on:
   - action_conditioned vs self_only (must have CI lower bound > 0)
   - action_conditioned vs action_shuffled (must ALSO have CI lower bound > 0
     -- this is "shuffle destroys the gain")
4. GO only if BOTH hold; otherwise NO-GO. This script computes and reports
   the result honestly regardless of which way it comes out.
"""

from __future__ import annotations

import glob
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from crowd_nav.bayesian_brne import data_io
from crowd_nav.bayesian_brne.action_conditioned_arhmm import (
    ARHMMConfig,
    ARHMMSequence,
    e_step,
    extract_sequences,
    fit,
)

DATA_DIR = "runs/bayesian_brne/data_formal"
SCENARIO = "baseline_circle"
TRAIN_SEEDS = [11, 12, 13, 14, 15]
VALIDATION_SEEDS = [21, 22, 23, 24, 25]
CONTROLLERS = ["goal_directed", "orca", "original_brne", "scripted_probe"]
K_CANDIDATES = (1, 2, 3, 4, 5, 6)
DT = 0.25
N_BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2407
SHUFFLE_SEED = 2407
OUT_PATH = "runs/bayesian_brne/models/u3_necessity_gate.json"
SOURCE_FILES = [
    "crowd_nav/tools/u3_necessity_gate.py",
    "crowd_nav/bayesian_brne/action_conditioned_arhmm.py",
    "crowd_nav/bayesian_brne/data_io.py",
]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _load_split_episodes(split: str, suite_seeds: List[int]) -> Tuple[List[dict], List[int]]:
    """Load every formal episode for ``split`` across ALL 4 controllers
    combined (the robot's action policy needs to vary for a
    "does the action matter" test to have any power at all -- a single
    controller alone gives ``u_robot`` far less variation). Returns
    ``(episodes, suite_seed_per_episode)`` in the SAME order, so a
    sequence's ``track_key[0]`` (its index into ``episodes``) can be
    mapped back to a suite seed."""
    paths = sorted(glob.glob(f"{DATA_DIR}/{split}/{SCENARIO}/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episodes found under {DATA_DIR}/{split}/{SCENARIO}/")
    episodes = []
    suite_seeds_out = []
    seen_seeds = set()
    for p in paths:
        ep = data_io.load_episode(p)
        if ep["profile_name"] != "formal":
            raise ValueError(f"{p}: profile_name={ep['profile_name']!r}, expected 'formal'")
        if ep["split"] != split:
            raise ValueError(f"{p}: split={ep['split']!r}, expected {split!r}")
        if ep["controller_type"] not in CONTROLLERS:
            raise ValueError(f"{p}: unexpected controller_type={ep['controller_type']!r}")
        seen_seeds.add(int(ep["suite_seed"]))
        episodes.append({
            "humans": ep["humans"], "human_track_ids": ep["human_track_ids"],
            "robot": ep["robot"], "valid_mask": ep["valid_mask"],
            "robot_actions": np.array(ep["robot_actions"], dtype=np.float64, copy=True),
        })
        suite_seeds_out.append(int(ep["suite_seed"]))
    if seen_seeds != set(suite_seeds):
        raise ValueError(f"{split}: expected suite seeds {sorted(suite_seeds)}, found {sorted(seen_seeds)}")
    return episodes, suite_seeds_out


def _make_self_only(episodes: List[dict]) -> List[dict]:
    """Zero every episode's ``robot_actions`` -- B_k's design column becomes
    constant zero, so the MNIW M-step estimates B_k as ~0 regardless of its
    prior (see action_conditioned_arhmm.m_step's docstring); this is
    mathematically the "restricted" (no-action) nested model, produced with
    NO changes to the model code itself, only to its input data."""
    out = []
    for ep in episodes:
        ep2 = dict(ep)
        ep2["robot_actions"] = np.zeros_like(ep["robot_actions"])
        out.append(ep2)
    return out


def _make_action_shuffled(episodes: List[dict], seed: int) -> List[dict]:
    """Derangement-shuffle ``robot_actions`` ACROSS episodes (same technique
    as ``order9s_shuffle_test.py``'s ``shuffle_robot_pairing``, applied only
    to the action field rather than the whole ``robot`` array -- here we
    specifically want to break the u_robot<->outcome correspondence while
    leaving v_current/context, both derived from the TRUE robot/human
    trajectories, untouched). Only ever applied to TRAIN."""
    rng = np.random.default_rng(seed)
    n = len(episodes)
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    out = []
    for i in range(n):
        ep2 = dict(episodes[i])
        ep2["robot_actions"] = episodes[perm[i]]["robot_actions"]
        out.append(ep2)
    return out


def _held_out_sequence_lls(
    sequences: List[ARHMMSequence], artifact, suite_seed_per_episode: List[int]
) -> Dict[int, List[Tuple[float, int]]]:
    """Runs e_step on ``sequences`` under ``artifact`` and groups the
    resulting per-sequence log-likelihoods (paired with each sequence's own
    row count, for correct nats/row normalization) by validation suite
    seed -- the precomputed unit the block bootstrap resamples."""
    e_result = e_step(sequences, artifact)
    lls = e_result["sequence_log_likelihoods"]
    by_seed: Dict[int, List[Tuple[float, int]]] = {}
    for seq, ll in zip(sequences, lls):
        seed = suite_seed_per_episode[seq.track_key[0]]
        n_rows = seq.v_current.shape[0]
        by_seed.setdefault(seed, []).append((ll, n_rows))
    return by_seed


def _aggregate_nll_per_row(by_seed: Dict[int, List[Tuple[float, int]]], seeds: List[int]) -> float:
    total_ll = sum(ll for s in seeds for ll, _n in by_seed[s])
    total_rows = sum(n for s in seeds for _ll, n in by_seed[s])
    return -total_ll / total_rows if total_rows > 0 else float("nan")


def _block_bootstrap_nll_ci(
    by_seed_a: Dict[int, List[Tuple[float, int]]],
    by_seed_b: Dict[int, List[Tuple[float, int]]],
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Suite-seed block bootstrap comparing two models' held-out NLL/row on
    the SAME validation suite seeds -- ``improvement = nll_b - nll_a``
    (positive means A is better/lower-NLL than B). Resamples precomputed
    per-sequence (ll, n_rows) pairs grouped by seed; no refitting, matching
    ``fit_bayesian_brne.fast_bootstrap_nll_ci``'s discipline."""
    seeds = sorted(set(by_seed_a.keys()) & set(by_seed_b.keys()))
    rng = np.random.default_rng(seed)
    nll_a = np.zeros(n_resamples)
    nll_b = np.zeros(n_resamples)
    improvement = np.zeros(n_resamples)
    for i in range(n_resamples):
        picked = rng.choice(seeds, size=len(seeds), replace=True)
        a = _aggregate_nll_per_row(by_seed_a, list(picked))
        b = _aggregate_nll_per_row(by_seed_b, list(picked))
        nll_a[i] = a
        nll_b[i] = b
        improvement[i] = b - a
    point_a = _aggregate_nll_per_row(by_seed_a, seeds)
    point_b = _aggregate_nll_per_row(by_seed_b, seeds)
    return {
        "point_nll_a": point_a,
        "point_nll_b": point_b,
        "point_improvement_b_minus_a": point_b - point_a,
        "improvement_p2_5": float(np.percentile(improvement, 2.5)),
        "improvement_p50": float(np.percentile(improvement, 50)),
        "improvement_p97_5": float(np.percentile(improvement, 97.5)),
        "ci_supports_a_better_than_b": bool(np.percentile(improvement, 2.5) > 0),
        "n_suite_seeds": len(seeds),
        "n_resamples": n_resamples,
    }


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    print("[u3] loading formal episodes (all 4 controllers combined) ...")
    train_eps, train_seeds = _load_split_episodes("train", TRAIN_SEEDS)
    val_eps, val_seeds = _load_split_episodes("validation", VALIDATION_SEEDS)
    print(f"[u3] {len(train_eps)} train episodes, {len(val_eps)} validation episodes")

    val_sequences = extract_sequences(val_eps, dt=DT)
    print(f"[u3] {len(val_sequences)} validation sequences / {sum(s.v_current.shape[0] for s in val_sequences)} rows")

    action_conditioned_train_eps = train_eps  # real, unmodified
    self_only_train_eps = _make_self_only(train_eps)
    action_shuffled_train_eps = _make_action_shuffled(train_eps, seed=SHUFFLE_SEED)

    ac_train_sequences = extract_sequences(action_conditioned_train_eps, dt=DT)
    print(f"[u3] {len(ac_train_sequences)} action_conditioned train sequences "
          f"/ {sum(s.v_current.shape[0] for s in ac_train_sequences)} rows")

    # --- Step 1: pre-registered K selection on action_conditioned only ---
    print(f"[u3] K selection sweep K={K_CANDIDATES} on action_conditioned variant ...")
    config_sweep = ARHMMConfig(k_candidates=K_CANDIDATES, seed=2407)
    _best_artifact_naive, fit_results = fit(ac_train_sequences, val_sequences, config_sweep, dt=DT)

    per_k_report = {}
    lls_by_seed_by_k: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}
    for K in K_CANDIDATES:
        artifact_k = fit_results[K]["artifact"]
        by_seed = _held_out_sequence_lls(val_sequences, artifact_k, val_seeds)
        lls_by_seed_by_k[K] = by_seed
        nll = _aggregate_nll_per_row(by_seed, sorted(by_seed.keys()))
        per_k_report[K] = {
            "n_train_iters": len(fit_results[K]["train_objective_history"]),
            "held_out_nll_per_row": nll,
        }
        print(f"[u3]   K={K}: held_out_nll_per_row={nll:.6f} (n_iters={per_k_report[K]['n_train_iters']})")

    eligible_ks = sorted(K_CANDIDATES)
    selected_k = eligible_ks[-1]  # fallback if NLL keeps improving to the boundary
    plateau_hit_boundary = True
    for i, k in enumerate(eligible_ks):
        larger_ks = eligible_ks[i + 1:]
        if not larger_ks:
            selected_k = k
            plateau_hit_boundary = True
            break
        no_further_gain = True
        for bigger_k in larger_ks:
            comp = _block_bootstrap_nll_ci(lls_by_seed_by_k[bigger_k], lls_by_seed_by_k[k])
            if comp["ci_supports_a_better_than_b"]:
                no_further_gain = False
                break
        if no_further_gain:
            selected_k = k
            plateau_hit_boundary = False
            break

    print(f"[u3] pre-registered one-SE/minimal-sufficient K selection: K*={selected_k} "
          f"(hit_search_boundary={plateau_hit_boundary})")

    # --- Step 2: fit self_only and action_shuffled at K=K* ---
    self_only_train_sequences = extract_sequences(self_only_train_eps, dt=DT)
    action_shuffled_train_sequences = extract_sequences(action_shuffled_train_eps, dt=DT)

    config_k = ARHMMConfig(k_candidates=(selected_k,), seed=2407)
    print(f"[u3] fitting self_only at K={selected_k} ...")
    self_only_artifact, self_only_results = fit(self_only_train_sequences, val_sequences, config_k, dt=DT)
    print(f"[u3] fitting action_shuffled at K={selected_k} ...")
    action_shuffled_artifact, action_shuffled_results = fit(action_shuffled_train_sequences, val_sequences, config_k, dt=DT)

    action_conditioned_artifact = fit_results[selected_k]["artifact"]

    by_seed_ac = lls_by_seed_by_k[selected_k]
    by_seed_self = _held_out_sequence_lls(val_sequences, self_only_artifact, val_seeds)
    by_seed_shuffled = _held_out_sequence_lls(val_sequences, action_shuffled_artifact, val_seeds)

    # --- Step 3: suite-seed block bootstrap paired comparisons ---
    ac_vs_self = _block_bootstrap_nll_ci(by_seed_ac, by_seed_self)
    ac_vs_shuffled = _block_bootstrap_nll_ci(by_seed_ac, by_seed_shuffled)
    self_vs_shuffled = _block_bootstrap_nll_ci(by_seed_self, by_seed_shuffled)

    go = ac_vs_self["ci_supports_a_better_than_b"] and ac_vs_shuffled["ci_supports_a_better_than_b"]
    verdict = "GO" if go else "NO_GO"

    print()
    print("=" * 70)
    print(f"U3 NECESSITY GATE at K*={selected_k}")
    print("=" * 70)
    print(f"action_conditioned vs self_only:      point_improvement={ac_vs_self['point_improvement_b_minus_a']:+.6f} nats/row  "
          f"CI=[{ac_vs_self['improvement_p2_5']:.6f},{ac_vs_self['improvement_p97_5']:.6f}]  "
          f"ci_supports_real_improvement={ac_vs_self['ci_supports_a_better_than_b']}")
    print(f"action_conditioned vs action_shuffled: point_improvement={ac_vs_shuffled['point_improvement_b_minus_a']:+.6f} nats/row  "
          f"CI=[{ac_vs_shuffled['improvement_p2_5']:.6f},{ac_vs_shuffled['improvement_p97_5']:.6f}]  "
          f"ci_supports_real_improvement={ac_vs_shuffled['ci_supports_a_better_than_b']}")
    print(f"self_only vs action_shuffled (diagnostic): point_improvement={self_vs_shuffled['point_improvement_b_minus_a']:+.6f} nats/row  "
          f"CI=[{self_vs_shuffled['improvement_p2_5']:.6f},{self_vs_shuffled['improvement_p97_5']:.6f}]")
    print(f"\nVERDICT: {verdict} (GO requires BOTH action_conditioned-vs-self_only AND "
          f"action_conditioned-vs-action_shuffled CI lower bounds > 0)")

    manifest_path = f"{DATA_DIR}/manifest_post_collection.json"
    with open(manifest_path) as f:
        data_manifest_hash = json.load(f)["aggregate_sha256"]

    report = {
        "gate": "U3_necessity_gate",
        "date": "2026-08-03",
        "train_seeds": TRAIN_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "controllers": CONTROLLERS,
        "k_candidates_swept": list(K_CANDIDATES),
        "per_k_held_out_nll_per_row": per_k_report,
        "selected_k": selected_k,
        "k_selection_hit_search_boundary": plateau_hit_boundary,
        "k_selection_rule": "smallest K such that no larger candidate K shows a suite-seed-bootstrap-CI-supported further NLL improvement (pre-registered, not naive argmax)",
        "comparisons": {
            "action_conditioned_vs_self_only": ac_vs_self,
            "action_conditioned_vs_action_shuffled": ac_vs_shuffled,
            "self_only_vs_action_shuffled_diagnostic": self_vs_shuffled,
        },
        "verdict": verdict,
        "verdict_rule": "GO iff action_conditioned beats self_only AND beats action_shuffled, both with suite-seed bootstrap CI lower bound > 0",
        "provenance": {
            "source_sha256": {p: _sha256_file(p) for p in SOURCE_FILES},
            "data_manifest_aggregate_sha256": data_manifest_hash,
            "data_manifest_path": manifest_path,
            "git_head": _git_head(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "shuffle_seed": SHUFFLE_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_bootstrap_resamples": N_BOOTSTRAP_RESAMPLES,
        },
    }
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
