"""Paired validation/formal evaluator (guide.md 7.4/9 A9).

A thin, deterministic harness around a caller-supplied ``episode_fn``
(the actual CrowdSim + BDVLPolicy rollout loop, wired up in A10) so
this module's own determinism/fail-closed contracts are testable
without a live environment. No test-time safety tweaks, artifact
overrides, or ad-hoc alpha changes are accepted as parameters here on
purpose -- there is simply no knob for them (guide.md A9 acceptance:
"evaluator不允许test-time safety tweak、override artifact或临时改alpha").
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Sequence

from crowd_nav.bayesian_dvl.statistics import EpisodeRecord, StatisticsError

VALIDATION_ROLES = frozenset({
    "world-validation", "checkpoint-validation",
})
NON_VALIDATION_ROLES_FOR_CHECKPOINT_SELECTION = frozenset({
    "world-train", "IL-train", "RL-train", "formal-test",
})


class EvaluatorError(ValueError):
    pass


def assert_role_allowed_for_checkpoint_selection(data_role: str) -> None:
    """guide.md A9: "checkpoint选择只读取validation角色数据". Call this at
    the top of any checkpoint-selection code path with the caller's
    intended data role; raises if it is not a validation role."""
    if data_role not in VALIDATION_ROLES:
        raise EvaluatorError(
            f"checkpoint selection must only read validation-role data, got {data_role!r} "
            f"(allowed: {sorted(VALIDATION_ROLES)})"
        )


EpisodeFn = Callable[[str, str, str, int, int], EpisodeRecord]  # (method, scenario, profile, suite_seed, episode_seed) -> EpisodeRecord


def run_paired_evaluation(
    method: str,
    scenario: str,
    profile: str,
    suite_seeds: Sequence[int],
    episodes_per_seed: int,
    episode_fn: EpisodeFn,
) -> List[EpisodeRecord]:
    records = []
    for suite_seed in suite_seeds:
        for episode_index in range(episodes_per_seed):
            episode_seed = suite_seed * 100000 + episode_index
            record = episode_fn(method, scenario, profile, suite_seed, episode_seed)
            if (record.method, record.scenario, record.profile, record.suite_seed, record.episode_seed) != (
                method, scenario, profile, suite_seed, episode_seed
            ):
                raise EvaluatorError(
                    f"episode_fn returned a record with mismatched identity: "
                    f"expected ({method},{scenario},{profile},{suite_seed},{episode_seed}), "
                    f"got ({record.method},{record.scenario},{record.profile},{record.suite_seed},{record.episode_seed})"
                )
            records.append(record)
    return records


CSV_FIELDS = (
    "method", "scenario", "profile", "suite_seed", "episode_seed", "outcome", "steps",
    "elapsed_time", "min_clearance", "path_length", "mean_decision_latency_ms",
    "initial_goal_distance", "final_goal_distance", "max_goal_distance",
    "mean_action_alignment", "negative_alignment_fraction",
    "max_action_score", "min_action_score", "score_bound_violations",
)


def write_episode_records_csv(records: Sequence[EpisodeRecord], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def read_episode_records_csv(path: str) -> List[EpisodeRecord]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            records.append(EpisodeRecord(
                method=row["method"], scenario=row["scenario"], profile=row["profile"],
                suite_seed=int(row["suite_seed"]), episode_seed=int(row["episode_seed"]),
                outcome=row["outcome"], steps=int(row["steps"]),
                elapsed_time=float(row["elapsed_time"]), min_clearance=float(row["min_clearance"]),
                path_length=float(row["path_length"]), mean_decision_latency_ms=float(row["mean_decision_latency_ms"]),
                initial_goal_distance=float(row.get("initial_goal_distance") or 0.0),
                final_goal_distance=float(row.get("final_goal_distance") or 0.0),
                max_goal_distance=float(row.get("max_goal_distance") or 0.0),
                mean_action_alignment=float(row.get("mean_action_alignment") or 0.0),
                negative_alignment_fraction=float(row.get("negative_alignment_fraction") or 0.0),
                max_action_score=float(row.get("max_action_score") or 0.0),
                min_action_score=float(row.get("min_action_score") or 0.0),
                score_bound_violations=int(row.get("score_bound_violations") or 0),
            ))
        return records


def deterministic_records_sha256(records: Sequence[EpisodeRecord]) -> str:
    """Hash trajectory/outcome records while excluding wall-clock latency.

    Decision latency is measured with ``perf_counter`` and is therefore not
    expected to be byte-identical across processes.  The trajectory and
    outcome fields are deterministic under a fixed seed and are the fields
    that must be used for reproducibility checks.
    """
    rows = []
    for record in records:
        row = asdict(record)
        row.pop("mean_decision_latency_ms", None)
        rows.append(row)
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assert_no_duplicate_or_missing(records: Sequence[EpisodeRecord], expected_suite_seeds: Sequence[int], episodes_per_seed: int) -> None:
    seen = set()
    for r in records:
        key = (r.suite_seed, r.episode_seed)
        if key in seen:
            raise EvaluatorError(f"duplicate episode identity: {key}")
        seen.add(key)
    expected = {
        (seed, seed * 100000 + i)
        for seed in expected_suite_seeds
        for i in range(episodes_per_seed)
    }
    missing = expected - seen
    if missing:
        raise EvaluatorError(f"missing {len(missing)} expected episodes, e.g. {sorted(missing)[:5]}")
