"""SM-BRNE data coverage report and pilot/formal collection gate (guide.md
Order 7/8).

Single responsibility: read a directory of episode .npz files (written by
crowd_nav.bayesian_brne.collect_dataset) and report the coverage metrics
guide.md Order 7 requires -- initial overlap, per-controller episode
counts, per-behavior episode/transition counts, profile parameter min/max,
TTC<2s / near-collision / turning / deceleration transition counts,
controller fallback counts, and train/validation seed/episode/hash
intersection. ``check_gate`` turns the report into the explicit STOP
conditions Order 7 names; it does not fit any model.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from crowd_nav.bayesian_brne import data_io
from crowd_nav.bayesian_brne.mode_model import extract_transitions

NEAR_COLLISION_MARGIN = 0.05  # meters of clearance below combined radius+INITIAL_CLEARANCE_MARGIN counted as "near-collision"
TURN_THRESHOLD_RAD_PER_S = 0.5
DECEL_THRESHOLD_M_PER_S2 = 0.3


def _min_pairwise_clearance(robot_row: np.ndarray, humans_row: np.ndarray, robot_radius: float, human_radii: np.ndarray) -> float:
    """Minimum (center_distance - combined_radius) over robot-human and
    human-human pairs at one timestep. Negative means physical overlap."""
    positions = [robot_row[:2]] + [humans_row[i, :2] for i in range(humans_row.shape[0])]
    radii = [robot_radius] + list(human_radii)
    min_clear = float("inf")
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            dist = float(np.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]))
            min_clear = min(min_clear, dist - (radii[a] + radii[b]))
    return min_clear


@dataclass
class CoverageReport:
    n_episodes: int = 0
    controller_counts: Dict[str, int] = field(default_factory=dict)
    behavior_episode_counts: Dict[str, int] = field(default_factory=dict)
    behavior_transition_counts: Dict[str, int] = field(default_factory=dict)
    profile_param_min: Dict[str, float] = field(default_factory=dict)
    profile_param_max: Dict[str, float] = field(default_factory=dict)
    initial_overlap_count: int = 0
    initial_min_clearance_p5: float = float("nan")
    near_collision_transition_count: int = 0
    ttc_below_2s_transition_count: int = 0
    turning_transition_count: int = 0
    decelerating_transition_count: int = 0
    total_transitions: int = 0
    controller_fallback_count: int = 0
    controller_fallback_by_type: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    identity_keys_by_split: Dict[str, List[Tuple]] = field(default_factory=dict)
    scenario_counts: Dict[str, int] = field(default_factory=dict)


def build_coverage_report(episode_paths: List[str]) -> CoverageReport:
    report = CoverageReport()
    behavior_types_seen_per_episode: List[set] = []
    all_min_clearances: List[float] = []

    for path in episode_paths:
        ep = data_io.load_episode(path)
        report.n_episodes += 1
        report.controller_counts[ep["controller_type"]] = report.controller_counts.get(ep["controller_type"], 0) + 1
        report.split_counts[ep["split"]] = report.split_counts.get(ep["split"], 0) + 1
        report.scenario_counts[ep["scenario"]] = report.scenario_counts.get(ep["scenario"], 0) + 1

        key = (ep["suite_seed"], ep["episode_seed"], ep["initial_state_hash"])
        report.identity_keys_by_split.setdefault(ep["split"], []).append(key)

        behaviors_here = set(ep["behavior_type_map"].values())
        behavior_types_seen_per_episode.append(behaviors_here)
        for bt in behaviors_here:
            report.behavior_episode_counts[bt] = report.behavior_episode_counts.get(bt, 0) + 1

        for k, v in ep["profile_params"].items():
            if v is None:
                continue
            report.profile_param_min[k] = min(report.profile_param_min.get(k, v), v)
            report.profile_param_max[k] = max(report.profile_param_max.get(k, v), v)

        robot = ep["robot"]
        humans = ep["humans"]
        human_radii = humans[0, :, 4] if humans.shape[1] > 0 else np.zeros(0)
        min_clear_t0 = _min_pairwise_clearance(robot[0], humans[0], robot[0, 4], human_radii)
        all_min_clearances.append(min_clear_t0)
        if min_clear_t0 < -1e-9:
            report.initial_overlap_count += 1

        for e in ep["events"]:
            if e.get("type") == "controller_fallback":
                report.controller_fallback_count += 1
                c = e.get("controller", "unknown")
                report.controller_fallback_by_type[c] = report.controller_fallback_by_type.get(c, 0) + 1

        rows = extract_transitions([{
            "humans": humans, "human_track_ids": ep["human_track_ids"],
            "robot": robot, "valid_mask": ep["valid_mask"],
        }], dt=ep["dt"])
        report.total_transitions += len(rows)
        track_to_behavior = ep["behavior_type_map"]
        for row in rows:
            _, track_id = row.track_key
            bt = track_to_behavior.get(str(track_id), "unknown")
            report.behavior_transition_counts[bt] = report.behavior_transition_counts.get(bt, 0) + 1
            ttc = row.phi[8]
            heading_change = row.phi[2]
            delta_speed = row.phi[1]
            if ttc < 2.0:
                report.ttc_below_2s_transition_count += 1
            if abs(heading_change) > TURN_THRESHOLD_RAD_PER_S:
                report.turning_transition_count += 1
            if delta_speed < -DECEL_THRESHOLD_M_PER_S2:
                report.decelerating_transition_count += 1
            rel_dist = float(np.hypot(row.phi[4], row.phi[5]))
            if rel_dist - 0.6 < NEAR_COLLISION_MARGIN:
                report.near_collision_transition_count += 1

    if all_min_clearances:
        report.initial_min_clearance_p5 = float(np.percentile(all_min_clearances, 5))
    return report


@dataclass
class GateResult:
    passed: bool
    reasons: List[str]


def check_gate(report: CoverageReport, min_episodes_per_controller: int = 1) -> GateResult:
    """Order 7's explicit STOP conditions. Returns PASS only if every one
    holds; every violation is listed, not just the first."""
    reasons: List[str] = []

    if report.initial_overlap_count > 0:
        reasons.append(f"initial overlap in {report.initial_overlap_count}/{report.n_episodes} episodes (must be 0)")

    if report.controller_counts:
        counts = list(report.controller_counts.values())
        if len(set(counts)) != 1:
            reasons.append(f"controller counts not equal: {report.controller_counts}")
        if min(counts) < min_episodes_per_controller:
            reasons.append(f"some controller has < {min_episodes_per_controller} episodes: {report.controller_counts}")
    else:
        reasons.append("no controller data found")

    train_keys = set(report.identity_keys_by_split.get("train", []))
    val_keys = set(report.identity_keys_by_split.get("validation", []))
    intersection = train_keys & val_keys
    if intersection:
        reasons.append(f"train/validation (suite_seed,episode_seed,initial_state_hash) intersection is NOT empty: {len(intersection)} shared keys")

    if report.n_episodes == 0:
        reasons.append("no episodes found")

    return GateResult(passed=(len(reasons) == 0), reasons=reasons)


def print_report(report: CoverageReport, gate: GateResult) -> None:
    print(f"n_episodes: {report.n_episodes}")
    print(f"split_counts: {report.split_counts}")
    print(f"scenario_counts: {report.scenario_counts}")
    print(f"controller_counts: {report.controller_counts}")
    print(f"controller_fallback_count: {report.controller_fallback_count} ({report.controller_fallback_by_type})")
    print(f"initial_overlap_count: {report.initial_overlap_count}")
    print(f"initial_min_clearance_p5: {report.initial_min_clearance_p5:.4f} m")
    print(f"behavior_episode_counts: {report.behavior_episode_counts}")
    print(f"behavior_transition_counts: {report.behavior_transition_counts}")
    print(f"total_transitions: {report.total_transitions}")
    print(f"ttc_below_2s_transition_count: {report.ttc_below_2s_transition_count}")
    print(f"near_collision_transition_count: {report.near_collision_transition_count}")
    print(f"turning_transition_count (|heading_change|>{TURN_THRESHOLD_RAD_PER_S}): {report.turning_transition_count}")
    print(f"decelerating_transition_count (delta_speed<-{DECEL_THRESHOLD_M_PER_S2}): {report.decelerating_transition_count}")
    print("profile_param_min:", report.profile_param_min)
    print("profile_param_max:", report.profile_param_max)
    print()
    print(f"GATE: {'PASS' if gate.passed else 'FAIL'}")
    for r in gate.reasons:
        print(f"  - {r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="root dir containing split/scenario/*.npz")
    parser.add_argument("--min-episodes-per-controller", type=int, default=1)
    args = parser.parse_args()

    paths = sorted(glob.glob(f"{args.data_dir}/**/*.npz", recursive=True))
    report = build_coverage_report(paths)
    gate = check_gate(report, min_episodes_per_controller=args.min_episodes_per_controller)
    print_report(report, gate)


if __name__ == "__main__":
    main()
