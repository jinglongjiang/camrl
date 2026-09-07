"""Focused tests for the CR-S1 causal-response pipeline."""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np

from crowd_nav.bayesian_brne.causal_pair_data import (
    collect_pair_episode, load_pair_episode, save_pair_episode, validate_pair_episode,
)
from crowd_nav.bayesian_brne.causal_response_arhmm import (
    CausalFitConfig, CausalResponseArtifact, CausalSequence, fit, response_features,
    response_metrics, score_sequences,
)


def _synthetic_sequences(*, response: bool, seed: int = 7, n_sequences: int = 10,
                         length: int = 24):
    rng = np.random.default_rng(seed)
    sequences = []
    true_d = np.zeros((2, 2, 8))
    true_d[0, :, :2] = np.array([[0.45, 0.0], [0.0, -0.25]])
    true_d[1, :, :2] = np.array([[-0.35, 0.0], [0.0, 0.40]])
    if not response:
        true_d[:] = 0.0
    transition = np.array([[0.92, 0.08], [0.10, 0.90]])
    for seq_index in range(n_sequences):
        velocity = np.zeros((length, 2))
        context = np.zeros((length, 6))
        robot_velocity = np.zeros((length, 2))
        action_a = rng.uniform(-0.7, 0.7, size=(length, 2))
        action_b = rng.uniform(-0.7, 0.7, size=(length, 2))
        ref = np.zeros((length, 2))
        out_a = np.zeros((length, 2))
        out_b = np.zeros((length, 2))
        state = rng.normal(0, 0.2, 2)
        mode = seq_index % 2
        for t in range(length):
            velocity[t] = state
            context[t] = np.array([0.7, 0.2, -0.3, 0.1, 1.2, 0.4]) + rng.normal(0, 0.03, 6)
            base = 0.82 * state + np.array([0.03, -0.01])
            ref[t] = base + rng.normal(0, 0.015, 2)
            psi_a = response_features(context[t], robot_velocity[t], action_a[t], 0.25)
            psi_b = response_features(context[t], robot_velocity[t], action_b[t], 0.25)
            out_a[t] = base + true_d[mode] @ psi_a + rng.normal(0, 0.015, 2)
            out_b[t] = base + true_d[mode] @ psi_b + rng.normal(0, 0.015, 2)
            state = ref[t]
            mode = int(rng.choice(2, p=transition[mode]))
        sequence = CausalSequence(
            velocity, context, robot_velocity, action_a, action_b, ref, out_a, out_b,
            suite_seed=100 + seq_index // 2, episode_seed=1000 + seq_index, track_id=seq_index,
        )
        sequence.validate()
        sequences.append(sequence)
    return sequences


def run_tests(brne_root: str) -> None:
    assertions = 0

    interactive = collect_pair_episode(
        suite_seed=91, episode_index=0, split="train", scenario="baseline_circle",
        controller_type="goal_directed", horizon_steps=4, dt=0.25, brne_root=brne_root,
    )
    validate_pair_episode(interactive)
    assert np.array_equal(interactive["action_ref"], interactive["robot_velocity"])
    assertions += 2

    nominal = collect_pair_episode(
        suite_seed=92, episode_index=0, split="test_nominal", scenario="baseline_circle",
        controller_type="goal_directed", horizon_steps=4, dt=0.25, brne_root=brne_root,
    )
    validate_pair_episode(nominal, nominal=True)
    assert np.array_equal(nominal["next_velocity_ref"], nominal["next_velocity_a"])
    assert np.array_equal(nominal["next_velocity_ref"], nominal["next_velocity_b"])
    assertions += 3

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "episode.npz"
        save_pair_episode(path, nominal)
        loaded = load_pair_episode(path, nominal=True)
        assert np.array_equal(loaded["next_velocity_ref"], nominal["next_velocity_ref"])
        assertions += 1

    config = CausalFitConfig(em_max_iters=15, restarts=1, response_ridge=0.02, seed=13)
    responsive = _synthetic_sequences(response=True)
    artifact = fit(responsive, 2, config)
    history = artifact.model_card["log_likelihood_history"]
    assert np.all(np.diff(history) >= -1e-7)
    metrics = response_metrics(responsive, artifact)
    assert metrics["relative_mse_improvement"] > 0.50
    assertions += 2

    swapped = []
    for sequence in responsive:
        value = copy.deepcopy(sequence)
        value.action_a, value.action_b = sequence.action_b.copy(), sequence.action_a.copy()
        value.next_velocity_a, value.next_velocity_b = (
            sequence.next_velocity_b.copy(), sequence.next_velocity_a.copy()
        )
        swapped.append(value)
    assert abs(score_sequences(responsive, artifact)["mean_nll"] -
               score_sequences(swapped, artifact)["mean_nll"]) < 1e-10
    assertions += 1

    no_response = _synthetic_sequences(response=False, seed=19)
    zero_artifact = fit(no_response, 1, config)
    zero_metrics = response_metrics(no_response, zero_artifact)
    assert np.linalg.norm(zero_artifact.D) < 0.15
    assert zero_metrics["predicted_response_mean"] < 0.03
    assertions += 2

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artifact.npz"
        artifact.save(path)
        loaded = CausalResponseArtifact.load(path)
        assert loaded._numeric_hash() == artifact._numeric_hash()
        damaged = copy.deepcopy(loaded)
        damaged.D[0, 0, 0] += 1.0
        assert damaged._numeric_hash() != loaded.model_card["numeric_sha256"]
        assertions += 2

    print(f"CR-S1 selftest PASS: {assertions} assertions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brne-root", default="/home/abc/temp/brne")
    args = parser.parse_args()
    run_tests(args.brne_root)


if __name__ == "__main__":
    main()
