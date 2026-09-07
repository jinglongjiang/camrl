"""Action-intervention AR-HMM for causal pedestrian response.

Unlike the retired observational action model, this model is fitted only on
same-state triplets: a zero-acceleration reference branch and two randomized
robot-action branches. The reference branch anchors shared pedestrian
dynamics; ``D @ response_features`` can therefore represent only the change
caused by the robot action.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans

from crowd_nav.bayesian_brne.action_conditioned_arhmm import CONTEXT_FEATURE_NAMES


ARTIFACT_SCHEMA_VERSION = 1
RESPONSE_FEATURE_NAMES = (
    "ax_proximity", "ay_proximity",
    "ax_ttc", "ay_ttc",
    "ax_closing", "ay_closing",
    "ax_passing", "ay_passing",
)
BASE_DIM = 2 + len(CONTEXT_FEATURE_NAMES) + 1
RESPONSE_DIM = len(RESPONSE_FEATURE_NAMES)
DESIGN_DIM = BASE_DIM + RESPONSE_DIM


class CausalModelError(RuntimeError):
    pass


class CausalArtifactError(ValueError):
    pass


@dataclass
class CausalSequence:
    v_current: np.ndarray
    context: np.ndarray
    robot_velocity: np.ndarray
    action_a: np.ndarray
    action_b: np.ndarray
    next_velocity_ref: np.ndarray
    next_velocity_a: np.ndarray
    next_velocity_b: np.ndarray
    suite_seed: int
    episode_seed: int
    track_id: int

    def validate(self) -> None:
        length = self.v_current.shape[0]
        shapes = {
            "v_current": (length, 2),
            "context": (length, len(CONTEXT_FEATURE_NAMES)),
            "robot_velocity": (length, 2),
            "action_a": (length, 2),
            "action_b": (length, 2),
            "next_velocity_ref": (length, 2),
            "next_velocity_a": (length, 2),
            "next_velocity_b": (length, 2),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise CausalModelError(f"invalid {name}: expected {shape}, got {value.shape}")


@dataclass
class CausalFitConfig:
    dt: float = 0.25
    em_max_iters: int = 30
    em_tol: float = 1e-4
    restarts: int = 2
    sticky_kappa: float = 4.0
    dirichlet_alpha: float = 1.1
    base_ridge: float = 1e-5
    response_ridge: float = 0.05
    covariance_floor: float = 1e-4
    seed: int = 2407

    def validate(self) -> None:
        if self.dt <= 0 or self.em_max_iters < 1 or self.restarts < 1:
            raise CausalModelError("invalid fit iteration/time configuration")
        if self.dirichlet_alpha <= 1.0 or self.sticky_kappa < 0:
            raise CausalModelError("Dirichlet MAP parameters are invalid")
        if min(self.base_ridge, self.response_ridge, self.covariance_floor) <= 0:
            raise CausalModelError("ridge and covariance floor must be positive")


@dataclass
class CausalResponseArtifact:
    K: int
    A: np.ndarray
    C: np.ndarray
    D: np.ndarray
    d: np.ndarray
    Q: np.ndarray
    Pi: np.ndarray
    initial_distribution: np.ndarray
    dt: float
    model_card: dict = field(default_factory=dict)

    def validate(self, *, require_production: bool = False) -> None:
        expected = {
            "A": (self.K, 2, 2),
            "C": (self.K, 2, len(CONTEXT_FEATURE_NAMES)),
            "D": (self.K, 2, RESPONSE_DIM),
            "d": (self.K, 2),
            "Q": (self.K, 2, 2),
            "Pi": (self.K, self.K),
            "initial_distribution": (self.K,),
        }
        if self.K < 1 or not np.isfinite(self.dt) or self.dt <= 0:
            raise CausalArtifactError("invalid K/dt")
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise CausalArtifactError(f"invalid artifact field {name}: {value.shape}")
        if not np.allclose(self.Pi.sum(axis=1), 1.0, atol=1e-8):
            raise CausalArtifactError("Pi rows do not sum to one")
        if not np.isclose(self.initial_distribution.sum(), 1.0, atol=1e-8):
            raise CausalArtifactError("initial distribution does not sum to one")
        for covariance in self.Q:
            if np.min(np.linalg.eigvalsh(covariance)) <= 0:
                raise CausalArtifactError("Q is not positive definite")
        if self.model_card.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise CausalArtifactError("artifact schema mismatch")
        if self.model_card.get("response_feature_names") != list(RESPONSE_FEATURE_NAMES):
            raise CausalArtifactError("response feature schema mismatch")
        if require_production and self.model_card.get("tier") != "production":
            raise CausalArtifactError("artifact is not production tier")
        if require_production and not self.model_card.get("fit_converged", False):
            raise CausalArtifactError("unconverged artifact cannot be production tier")
        if require_production and self.model_card.get("scientific_gate") != "GO":
            raise CausalArtifactError("artifact has no frozen scientific GO decision")

    def base_mean(self, mode: int, velocity: np.ndarray, context: np.ndarray) -> np.ndarray:
        return self.A[mode] @ velocity + self.C[mode] @ context + self.d[mode]

    def mean(
        self,
        mode: int,
        velocity: np.ndarray,
        context: np.ndarray,
        robot_velocity: np.ndarray,
        robot_action: np.ndarray,
        *,
        response_enabled: bool = True,
    ) -> np.ndarray:
        value = self.base_mean(mode, velocity, context)
        if response_enabled:
            value = value + self.D[mode] @ response_features(
                context, robot_velocity, robot_action, self.dt,
            )
        return value

    def zero_response(self) -> "CausalResponseArtifact":
        result = copy.deepcopy(self)
        result.D = np.zeros_like(result.D)
        result.model_card = dict(result.model_card)
        result.model_card["ablation"] = "shared_artifact_D_zero"
        result.model_card["tier"] = "engineering_only"
        return result

    def _numeric_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray([self.K, self.dt], dtype=np.float64).tobytes())
        for value in (self.A, self.C, self.D, self.d, self.Q, self.Pi, self.initial_distribution):
            digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def save(self, path: Path, *, tier: str = "engineering_only") -> None:
        if tier not in ("engineering_only", "production"):
            raise CausalArtifactError(f"invalid tier {tier}")
        self.model_card = dict(self.model_card)
        self.model_card.update({
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "response_feature_names": list(RESPONSE_FEATURE_NAMES),
            "tier": tier,
        })
        self.validate(require_production=tier == "production")
        self.model_card["numeric_sha256"] = self._numeric_hash()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            K=np.array(self.K), A=self.A, C=self.C, D=self.D, d=self.d, Q=self.Q,
            Pi=self.Pi, initial_distribution=self.initial_distribution,
            dt=np.array(self.dt), model_card_json=np.array(json.dumps(self.model_card, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path, *, require_production: bool = False) -> "CausalResponseArtifact":
        with np.load(path, allow_pickle=False) as data:
            artifact = cls(
                K=int(data["K"]), A=data["A"], C=data["C"], D=data["D"],
                d=data["d"], Q=data["Q"], Pi=data["Pi"],
                initial_distribution=data["initial_distribution"], dt=float(data["dt"]),
                model_card=json.loads(str(data["model_card_json"])),
            )
        artifact.validate(require_production=require_production)
        if artifact.model_card.get("numeric_sha256") != artifact._numeric_hash():
            raise CausalArtifactError("artifact numeric hash mismatch")
        return artifact


def response_features(
    context: np.ndarray,
    robot_velocity: np.ndarray,
    robot_action: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Observable action-response basis, zero for the reference action."""
    context = np.asarray(context, dtype=np.float64)
    robot_velocity = np.asarray(robot_velocity, dtype=np.float64)
    robot_action = np.asarray(robot_action, dtype=np.float64)
    if context.shape[-1] != len(CONTEXT_FEATURE_NAMES) or dt <= 0:
        raise CausalModelError("invalid response feature input")
    acceleration = (robot_action - robot_velocity) / dt
    distance = np.linalg.norm(context[..., :2], axis=-1)
    proximity = np.exp(-distance / 2.0)
    ttc_weight = np.exp(-np.clip(context[..., 4], 0.0, 20.0) / 3.0)
    closing = np.clip(-np.sum(context[..., :2] * context[..., 2:4], axis=-1) /
                      np.maximum(distance, 1e-6), 0.0, 2.0) / 2.0
    passing = np.tanh(context[..., 5])
    gains = np.stack([proximity, ttc_weight, closing, passing], axis=-1)
    return (gains[..., :, None] * acceleration[..., None, :]).reshape(context.shape[:-1] + (RESPONSE_DIM,))


def sequences_from_episodes(episodes: Sequence[Dict[str, object]]) -> List[CausalSequence]:
    sequences: List[CausalSequence] = []
    for episode in episodes:
        track_ids = np.asarray(episode["track_ids"])
        for column in range(track_ids.shape[1]):
            if not np.all(track_ids[:, column] == track_ids[0, column]):
                raise CausalModelError("track identity changed within an episode")
            sequence = CausalSequence(
                v_current=np.asarray(episode["v_current"])[:, column],
                context=np.asarray(episode["context"])[:, column],
                robot_velocity=np.asarray(episode["robot_velocity"]),
                action_a=np.asarray(episode["action_a"]),
                action_b=np.asarray(episode["action_b"]),
                next_velocity_ref=np.asarray(episode["next_velocity_ref"])[:, column],
                next_velocity_a=np.asarray(episode["next_velocity_a"])[:, column],
                next_velocity_b=np.asarray(episode["next_velocity_b"])[:, column],
                suite_seed=int(episode["suite_seed"]), episode_seed=int(episode["episode_seed"]),
                track_id=int(track_ids[0, column]),
            )
            sequence.validate()
            sequences.append(sequence)
    if not sequences:
        raise CausalModelError("no causal sequences")
    return sequences


def _base_design(sequence: CausalSequence) -> np.ndarray:
    return np.concatenate([
        sequence.v_current, sequence.context, np.ones((sequence.v_current.shape[0], 1))
    ], axis=1)


def _designs(sequence: CausalSequence, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _base_design(sequence)
    zeros = np.zeros((base.shape[0], RESPONSE_DIM))
    psi_a = response_features(sequence.context, sequence.robot_velocity, sequence.action_a, dt)
    psi_b = response_features(sequence.context, sequence.robot_velocity, sequence.action_b, dt)
    return (
        np.concatenate([base, zeros], axis=1),
        np.concatenate([base, psi_a], axis=1),
        np.concatenate([base, psi_b], axis=1),
    )


def _log_gaussian(values: np.ndarray, means: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    residual = values - means
    solved = np.linalg.solve(chol, residual.T).T
    return -0.5 * (
        2 * np.log(2.0 * np.pi) + 2.0 * np.log(np.diag(chol)).sum()
        + np.sum(solved * solved, axis=1)
    )


def emission_log_probabilities(sequence: CausalSequence, artifact: CausalResponseArtifact) -> np.ndarray:
    designs = _designs(sequence, artifact.dt)
    targets = (sequence.next_velocity_ref, sequence.next_velocity_a, sequence.next_velocity_b)
    output = np.zeros((sequence.v_current.shape[0], artifact.K))
    for k in range(artifact.K):
        weights = np.concatenate([
            artifact.A[k], artifact.C[k], artifact.d[k, :, None], artifact.D[k]
        ], axis=1)
        for design, target in zip(designs, targets):
            output[:, k] += _log_gaussian(target, design @ weights.T, artifact.Q[k])
    return output


def _forward_backward(sequence: CausalSequence, artifact: CausalResponseArtifact):
    log_b = emission_log_probabilities(sequence, artifact)
    length, K = log_b.shape
    log_pi = np.log(np.maximum(artifact.initial_distribution, 1e-300))
    log_transition = np.log(np.maximum(artifact.Pi, 1e-300))
    alpha = np.zeros((length, K))
    alpha[0] = log_pi + log_b[0]
    for t in range(1, length):
        alpha[t] = log_b[t] + logsumexp(alpha[t - 1, :, None] + log_transition, axis=0)
    beta = np.zeros((length, K))
    for t in range(length - 2, -1, -1):
        beta[t] = logsumexp(log_transition + log_b[t + 1, None, :] + beta[t + 1, None, :], axis=1)
    ll = float(logsumexp(alpha[-1]))
    gamma = np.exp(np.clip(alpha + beta - ll, -700, 0))
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    xi = np.zeros((max(length - 1, 0), K, K))
    for t in range(length - 1):
        value = np.exp(np.clip(
            alpha[t, :, None] + log_transition + log_b[t + 1, None, :] + beta[t + 1, None, :] - ll,
            -700, 0,
        ))
        xi[t] = value / max(float(value.sum()), 1e-300)
    return ll, gamma, xi


def expectation(sequences: Sequence[CausalSequence], artifact: CausalResponseArtifact) -> dict:
    values = [_forward_backward(sequence, artifact) for sequence in sequences]
    return {
        "log_likelihood": float(sum(item[0] for item in values)),
        "gammas": [item[1] for item in values],
        "xis": [item[2] for item in values],
        "sequence_log_likelihoods": [item[0] for item in values],
    }


def _unpack_weights(weights: np.ndarray, K: int, Q: np.ndarray, Pi: np.ndarray,
                    initial: np.ndarray, dt: float, model_card: dict) -> CausalResponseArtifact:
    return CausalResponseArtifact(
        K=K,
        A=weights[:, :, :2],
        C=weights[:, :, 2:2 + len(CONTEXT_FEATURE_NAMES)],
        d=weights[:, :, BASE_DIM - 1],
        D=weights[:, :, BASE_DIM:],
        Q=Q, Pi=Pi, initial_distribution=initial, dt=dt, model_card=model_card,
    )


def maximization(sequences: Sequence[CausalSequence], e_result: dict, K: int,
                 config: CausalFitConfig) -> CausalResponseArtifact:
    sxx = np.zeros((K, DESIGN_DIM, DESIGN_DIM))
    sxy = np.zeros((K, DESIGN_DIM, 2))
    neff = np.zeros(K)
    transition_counts = np.zeros((K, K))
    initial_counts = np.zeros(K)
    cached = []
    for sequence, gamma, xi in zip(sequences, e_result["gammas"], e_result["xis"]):
        designs = _designs(sequence, config.dt)
        targets = (sequence.next_velocity_ref, sequence.next_velocity_a, sequence.next_velocity_b)
        cached.append((designs, targets, gamma))
        for k in range(K):
            weight = gamma[:, k]
            for design, target in zip(designs, targets):
                sxx[k] += (design * weight[:, None]).T @ design
                sxy[k] += (design * weight[:, None]).T @ target
            neff[k] += 3.0 * weight.sum()
        initial_counts += gamma[0]
        if len(xi):
            transition_counts += xi.sum(axis=0)

    ridge = np.concatenate([
        np.full(BASE_DIM, config.base_ridge), np.full(RESPONSE_DIM, config.response_ridge)
    ])
    weights = np.zeros((K, 2, DESIGN_DIM))
    for k in range(K):
        weights[k] = np.linalg.solve(sxx[k] + np.diag(ridge), sxy[k]).T

    covariance = np.zeros((K, 2, 2))
    for k in range(K):
        sse = np.zeros((2, 2))
        for designs, targets, gamma in cached:
            weight = gamma[:, k]
            for design, target in zip(designs, targets):
                residual = target - design @ weights[k].T
                sse += (residual * weight[:, None]).T @ residual
        covariance[k] = sse / max(neff[k], 1.0) + config.covariance_floor * np.eye(2)
        covariance[k] = 0.5 * (covariance[k] + covariance[k].T)

    alpha = config.dirichlet_alpha + transition_counts
    alpha += config.sticky_kappa * np.eye(K)
    transition = np.maximum(alpha - 1.0, 1e-12)
    transition /= transition.sum(axis=1, keepdims=True)
    initial_alpha = config.dirichlet_alpha + initial_counts
    initial = np.maximum(initial_alpha - 1.0, 1e-12)
    initial /= initial.sum()
    return _unpack_weights(weights, K, covariance, transition, initial, config.dt, {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "response_feature_names": list(RESPONSE_FEATURE_NAMES),
        "tier": "engineering_only",
    })


def _initialize(sequences: Sequence[CausalSequence], K: int, config: CausalFitConfig,
                seed: int) -> CausalResponseArtifact:
    features = []
    sequence_lengths = []
    for sequence in sequences:
        contrast = sequence.next_velocity_a - sequence.next_velocity_b
        features.append(np.concatenate([contrast, sequence.context[:, :4]], axis=1))
        sequence_lengths.append(len(contrast))
    values = np.concatenate(features)
    if K == 1:
        labels = np.zeros(len(values), dtype=np.int64)
    else:
        labels = MiniBatchKMeans(
            n_clusters=K, random_state=seed, n_init=3, batch_size=min(4096, len(values)),
        ).fit_predict(values)
    gammas = []
    offset = 0
    for length in sequence_lengths:
        gamma = np.eye(K)[labels[offset:offset + length]]
        gammas.append(gamma)
        offset += length
    xis = []
    for gamma in gammas:
        xi = np.einsum("ti,tj->tij", gamma[:-1], gamma[1:]) if len(gamma) > 1 else np.zeros((0, K, K))
        xis.append(xi)
    return maximization(sequences, {"gammas": gammas, "xis": xis}, K, config)


def _blend(old: CausalResponseArtifact, new: CausalResponseArtifact, fraction: float) -> CausalResponseArtifact:
    result = copy.deepcopy(old)
    for name in ("A", "C", "D", "d", "Q", "Pi", "initial_distribution"):
        value = (1.0 - fraction) * getattr(old, name) + fraction * getattr(new, name)
        if name == "Pi":
            value /= value.sum(axis=1, keepdims=True)
        elif name == "initial_distribution":
            value /= value.sum()
        setattr(result, name, value)
    return result


def fit(sequences: Sequence[CausalSequence], K: int, config: CausalFitConfig) -> CausalResponseArtifact:
    config.validate()
    for sequence in sequences:
        sequence.validate()
    best = None
    best_ll = -np.inf
    for restart in range(config.restarts):
        artifact = _initialize(sequences, K, config, config.seed + 1009 * restart + K)
        history = []
        for _ in range(config.em_max_iters):
            current = expectation(sequences, artifact)
            current_ll = float(current["log_likelihood"])
            candidate = maximization(sequences, current, K, config)
            candidate_ll = float(expectation(sequences, candidate)["log_likelihood"])
            fraction = 1.0
            while candidate_ll + 1e-8 < current_ll and fraction > 1e-4:
                fraction *= 0.5
                candidate = _blend(artifact, candidate, fraction)
                candidate_ll = float(expectation(sequences, candidate)["log_likelihood"])
            if candidate_ll + 1e-8 < current_ll:
                raise CausalModelError("EM failed monotonic line search")
            history.append(candidate_ll)
            artifact = candidate
            if len(history) > 1 and history[-1] - history[-2] < config.em_tol:
                break
        artifact.model_card.update({
            "fit_converged": len(history) < config.em_max_iters,
            "em_iterations": len(history),
            "log_likelihood_history": history,
            "fit_config": vars(config),
        })
        if history[-1] > best_ll:
            best, best_ll = artifact, history[-1]
    assert best is not None
    best.validate()
    return best


def score_sequences(sequences: Sequence[CausalSequence], artifact: CausalResponseArtifact) -> dict:
    result = expectation(sequences, artifact)
    rows = sum(3 * len(sequence.v_current) for sequence in sequences)
    return {
        "total_log_likelihood": float(result["log_likelihood"]),
        "mean_nll": float(-result["log_likelihood"] / rows),
        "rows": int(rows),
        "sequence_log_likelihoods": result["sequence_log_likelihoods"],
    }


def response_metrics(sequences: Sequence[CausalSequence], artifact: CausalResponseArtifact) -> dict:
    squared_error_full = []
    squared_error_zero = []
    predicted_magnitude = []
    suite_seeds = []
    for sequence in sequences:
        psi_a = response_features(sequence.context, sequence.robot_velocity, sequence.action_a, artifact.dt)
        psi_b = response_features(sequence.context, sequence.robot_velocity, sequence.action_b, artifact.dt)
        # Strict one-step-ahead prediction: the mode distribution at t may
        # use only evidence from transitions before t. A smoothed gamma
        # would leak the current/future target into this diagnostic.
        predictive = artifact.initial_distribution.copy()
        predicted_a = np.zeros_like(sequence.next_velocity_a)
        predicted_b = np.zeros_like(sequence.next_velocity_b)
        emissions = emission_log_probabilities(sequence, artifact)
        for t in range(len(sequence.v_current)):
            predicted_a[t] = np.einsum("k,kij,j->i", predictive, artifact.D, psi_a[t])
            predicted_b[t] = np.einsum("k,kij,j->i", predictive, artifact.D, psi_b[t])
            log_posterior = np.log(np.maximum(predictive, 1e-300)) + emissions[t]
            posterior = np.exp(log_posterior - np.max(log_posterior))
            posterior /= posterior.sum()
            predictive = posterior @ artifact.Pi
        observed_a = sequence.next_velocity_a - sequence.next_velocity_ref
        observed_b = sequence.next_velocity_b - sequence.next_velocity_ref
        squared_error_full.extend(np.sum((observed_a - predicted_a) ** 2, axis=1))
        squared_error_full.extend(np.sum((observed_b - predicted_b) ** 2, axis=1))
        squared_error_zero.extend(np.sum(observed_a ** 2, axis=1))
        squared_error_zero.extend(np.sum(observed_b ** 2, axis=1))
        predicted_magnitude.extend(np.linalg.norm(predicted_a, axis=1))
        predicted_magnitude.extend(np.linalg.norm(predicted_b, axis=1))
        suite_seeds.extend([sequence.suite_seed] * (2 * len(observed_a)))
    full = np.asarray(squared_error_full)
    zero = np.asarray(squared_error_zero)
    return {
        "mse_full": float(full.mean()),
        "mse_zero": float(zero.mean()),
        "relative_mse_improvement": float((zero.mean() - full.mean()) / max(zero.mean(), 1e-12)),
        "predicted_response_mean": float(np.mean(predicted_magnitude)),
        "per_row_full": full,
        "per_row_zero": zero,
        "per_row_predicted_magnitude": np.asarray(predicted_magnitude),
        "suite_seeds": np.asarray(suite_seeds),
    }
