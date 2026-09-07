"""Action-conditioned sticky Bayesian switching autoregressive model
(Upgrade U1, 2026-08-03 core-model replacement for mode_model.py's
K-means-then-per-cluster-regression approach).

See ``legacy_kmeans_mode/README.md`` for why that method was retired: five
independent checks (Order 9S.1/2/3/3b/4) showed its selected K behaves like
"number of piecewise-linear regression regions over smooth, continuous
kinematic variation," not a genuine, action-dependent latent interaction
mode -- it recovered similarly strong "structure" on near-unimodal ORCA
data, survived having its robot-pairing shuffled, and transferred (with an
even LARGER apparent gain) to a totally unrelated generative mechanism.

Model (per pedestrian track, over that track's own consecutive time steps):

    z_t | z_{t-1} ~ Categorical(Pi[z_{t-1}, :])
    v_{t+1} | z_t=k ~ N(A_k @ v_t + B_k @ u_R,t + C_k @ c_t + d_k, Q_k)

- ``z_t``: discrete latent mode, ``z_t in {0, ..., K-1}``.
- ``v_t``: the pedestrian's own observed 2D velocity.
- ``u_R,t``: the ROBOT's ACTUALLY EXECUTED action at time t -- this MUST be
  ``episode['robot_actions'][t]``, never derived from ``robot[:, 2:4]``
  even though the two are verified bit-identical one step apart
  (``selftest.py``'s ``upgrade_u0_*_robot_actions_is_executed_action``,
  which independently confirmed ``robot[t+1, 2:4] == robot_actions[t]``
  exactly, for every controller type). ``B_k @ u_R,t`` is the ONLY channel
  through which robot action enters the predicted response -- it must never
  be replaced or backfilled by an ordinary self-motion feature standing in
  for it, or the whole point of "action-conditioned" is lost silently.
- ``c_t``: deployment-observable relative-geometry/TTC context, a STRICT
  SUBSET of ``mode_model.FEATURE_NAMES`` containing ONLY the robot-relative
  entries (``CONTEXT_FEATURE_NAMES`` below) -- self-kinematic features
  (speed, delta_speed, heading_change, lateral_acceleration) are
  deliberately EXCLUDED from ``c_t`` because they already live in the
  autoregressive ``A_k @ v_t`` term; duplicating them into ``c_t`` would
  let the model reconstruct self-motion-only clustering through the back
  door -- exactly the feature redundancy Order 9S.3b measured in the
  legacy 10-dim ``phi`` (self-only and robot-only each recovered 86-94% of
  the "improvement" alone).
- ``Pi``: a ``[K, K]`` row-stochastic transition matrix under a STICKY
  Dirichlet prior (extra concentration added to the diagonal) to suppress
  spurious rapid mode-switching -- the legacy model estimated Pi post-hoc
  from hard K-means labels with no such regularization at all.
- ``(A_k, B_k, C_k, d_k, Q_k)``: per-mode dynamics under a genuine
  conjugate MATRIX-NORMAL/INVERSE-WISHART (MNIW) prior (2026-08-03 second
  audit round: an earlier version used an independent isotropic Gaussian
  prior on ``vec(W_k)`` DECOUPLED from ``Q_k``, which does not correspond
  to any single coherent objective once Q_k is non-isotropic -- the
  correct joint prior is ``W_k | Q_k ~ MatrixNormal(0, Q_k, V0)`` with
  ``V0 = shrinkage_scale * I`` (row-covariance shared with the noise
  covariance ``Q_k``, column-covariance ``V0``), ``Q_k ~
  InverseWishart(Psi0, nu0)``). Under this TRUE MNIW conjugacy, the
  posterior mean/mode of ``W_k`` is *coincidentally* still the same
  Q-agnostic ridge formula (``Sxy^T @ inv(Sxx + V0^{-1})`` -- a
  well-known property of matrix-normal regression: Q cancels out of the
  W-conditional's mean), but the posterior scale matrix for ``Q_k`` picks
  up an EXTRA prior-quadratic correction term (``+ W_k @ V0^{-1} @
  W_k^T``) that the earlier, decoupled-prior version omitted -- see
  ``m_step``'s docstring for the exact formula. This regularizes K the
  same way regardless: it stops K from being able to freely increase just
  to chase residual variance -- the legacy model's ridge regression had no
  such K-dependent regularization, which is consistent with the
  never-plateauing NLL curve Order 9S.1 found.

Fit via sequence-level, log-space forward-backward (Baum-Welch-style) EM.
K-means may be used ONLY to initialize the first E-step's responsibilities,
never to assign final labels (guide.md Upgrade U1). The SAME fitted model
must do exact Bayesian filtering online (replacing belief_tracker.py's role
for this model), so training and deployment share exactly one mode
definition -- never two.

STATUS (Upgrade U2, 2026-08-03): forward-backward E-step, M-step, and
sticky/shrinkage-prior updates are implemented and pass the selftest suite
(forward/backward normalization, log-likelihood monotonicity, synthetic
identifiability -- recovered B_k/Pi/z on a genuinely action-dependent
2-mode synthetic dataset at ~99% z-recovery accuracy -- K=1 degeneracy,
artifact roundtrip). This is still ONLY Upgrade U2 (numerical correctness
of the fitting machinery); Upgrade U3's necessity gate (does
action-conditioning beat a same-K/same-parameter-budget self-only model,
and does shuffling the action destroy exactly that gain) has NOT been run
yet, so no claim that this model has discovered real interaction structure
is licensed until U3 passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

STEP = "upgrade_u2"
IMPLEMENTED = True

# Deployment-observable relative-geometry context -- a STRICT SUBSET of
# mode_model.FEATURE_NAMES. Self-kinematic entries (speed, delta_speed,
# heading_change, lateral_acceleration) are excluded on purpose; see the
# module docstring's ``c_t`` paragraph above.
CONTEXT_FEATURE_NAMES = ("relative_px", "relative_py", "relative_vx", "relative_vy", "ttc_clipped", "passing_side")


class ARHMMConfigError(ValueError):
    pass


class EMMonotonicityError(RuntimeError):
    """Raised by ``fit`` when the penalized MAP-EM objective decreases
    beyond floating-point tolerance -- with the corrected conjugate MNIW
    M-step, this is a real guarantee, so a violation means a genuine bug,
    never expected/ignorable behavior."""


class ArtifactIntegrityError(ValueError):
    """Raised by ``ARHMMArtifact.save``/``.load`` (Order F1, 2026-08-03) when
    a tier is invalid, a production-tier save is attempted on an unconverged
    fit, or a loaded file's schema_version/content hash does not match what
    was recorded at save time. Fail CLOSED in all these cases -- silently
    loading a stale-schema or hand-edited/corrupted artifact, or silently
    letting an unconverged K=5/6-style fit masquerade as a production
    artifact, is exactly the kind of thing this error exists to prevent."""


ARTIFACT_SCHEMA_VERSION = 2
VALID_ARTIFACT_TIERS = ("engineering_only", "production")
REQUIRED_ARTIFACT_PROVENANCE_FIELDS = (
    "feature_schema",
    "time_semantics",
    "training_config",
    "data_aggregate_sha256",
    "source_sha256",
    "human_physical_bounds",
)


_EM_MONOTONICITY_TOLERANCE = 1e-5  # after the 2026-08-03 third-audit-round Q-denominator fix, 20 seeds x anisotropic-Q all pass at 1e-6 -- 1e-5 leaves headroom for float64 noise while still catching real math bugs (which produced ~1e-3-scale, many-iteration systematic drift before the fix)


@dataclass(frozen=True)
class ARHMMConfig:
    """Every tunable number the EM fit uses -- no magic numbers in the
    E-step/M-step implementation (same discipline as config.py's
    BayesianModelConfig)."""

    k_candidates: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    sticky_kappa: float = 10.0            # extra Dirichlet concentration added to Pi's diagonal (sticky prior)
    dirichlet_alpha: float = 2.0          # base Dirichlet concentration for Pi's off-diagonal entries (must be >1 so the posterior MODE -- not mean -- stays non-negative/well-defined, see m_step)
    shrinkage_scale: float = 1.0          # isotropic Gaussian shrinkage-prior variance for vec([A_k|B_k|C_k|d_k]) (decoupled from Q_k -- see module docstring)
    inverse_wishart_dof: float = 6.0      # inverse-Wishart prior degrees of freedom for Q_k (Q_k is 2x2, so this must exceed d-1=1 to be a PROPER distribution, per 2026-08-03 audit item 4 -- not just >0)
    inverse_wishart_scale: float = 1e-2   # inverse-Wishart prior scale (times identity) for Q_k
    em_max_iters: int = 100
    em_tol: float = 1e-4                  # stop EM when log-likelihood improvement falls below this
    seed: int = 2407

    def __post_init__(self) -> None:
        if not self.k_candidates or any(int(k) < 1 for k in self.k_candidates):
            raise ARHMMConfigError(f"k_candidates must be positive integers, got {self.k_candidates}")
        if len(set(self.k_candidates)) != len(self.k_candidates):
            raise ARHMMConfigError(f"k_candidates must not contain duplicates, got {self.k_candidates}")
        if self.sticky_kappa < 0.0:
            raise ARHMMConfigError(f"sticky_kappa must be >= 0, got {self.sticky_kappa}")
        if self.dirichlet_alpha <= 1.0:
            # 2026-08-03 second audit round: Pi/initial_distribution are now
            # Dirichlet posterior MODE estimates (not mean), which requires
            # every alpha_prior entry > 1 to stay non-negative -- an
            # earlier version only required >0 (a MEAN-estimator
            # constraint), silently permitting an ill-defined MODE.
            raise ARHMMConfigError(f"dirichlet_alpha must be > 1 (required for the Dirichlet MODE estimator used in m_step), got {self.dirichlet_alpha}")
        if self.shrinkage_scale <= 0.0:
            raise ARHMMConfigError(f"shrinkage_scale must be > 0, got {self.shrinkage_scale}")
        if self.inverse_wishart_dof <= 1.0:
            # d=2 (Q_k is always 2x2 in this model); an inverse-Wishart(nu,Psi)
            # prior is only a PROPER distribution for nu > d-1 = 1, regardless
            # of whether the point estimate used is the mode or the mean
            # (2026-08-03 audit item 4 -- an earlier version only required >0,
            # which silently permitted an improper prior).
            raise ARHMMConfigError(f"inverse_wishart_dof must be > 1 (d-1 for d=2) to be a proper IW prior, got {self.inverse_wishart_dof}")
        if self.inverse_wishart_scale <= 0.0:
            raise ARHMMConfigError(f"inverse_wishart_scale must be > 0, got {self.inverse_wishart_scale}")
        if self.em_max_iters < 1:
            raise ARHMMConfigError(f"em_max_iters must be >= 1, got {self.em_max_iters}")
        if self.em_tol <= 0.0:
            raise ARHMMConfigError(f"em_tol must be > 0, got {self.em_tol}")


@dataclass
class ARHMMArtifact:
    """The fitted model. Unlike mode_model.ModeModelArtifact, ``Pi`` and
    ``initial_distribution`` are genuine EM-converged posteriors (under the
    sticky/Dirichlet prior), never a post-hoc count from hard labels."""

    K: int
    A: List[np.ndarray]                # [K] each [2, 2]
    B: List[np.ndarray]                # [K] each [2, 2] -- the action-conditioning term
    C: List[np.ndarray]                # [K] each [2, len(CONTEXT_FEATURE_NAMES)]
    d: List[np.ndarray]                # [K] each [2]
    Q: List[np.ndarray]                # [K] each [2, 2]
    Pi: np.ndarray                     # [K, K]
    initial_distribution: np.ndarray   # [K]
    dt: float
    model_card: dict = field(default_factory=dict)

    def _default_provenance(self) -> dict:
        """Return explicit engineering placeholders for old in-memory fits.

        A placeholder is allowed only for ``engineering_only`` artifacts. It
        is still recorded and hashed, so it cannot silently become a paper
        artifact later. Production saves require real data/source hashes.
        """
        return {
            "feature_schema": {
                "name": "ARHMM_CONTEXT_FEATURES",
                "version": 1,
                "names": list(CONTEXT_FEATURE_NAMES),
            },
            "time_semantics": {
                "dt": float(self.dt),
                "state": "robot/human state at t before action",
                "action": "u_robot[t] applied during transition t->t+1",
                "target": "v_next[t] at t+1",
            },
            "training_config": {"status": "not_recorded"},
            "data_aggregate_sha256": "UNSPECIFIED",
            "source_sha256": {"code": "UNSPECIFIED"},
            "human_physical_bounds": {"max_speed": 2.0, "max_acceleration": 2.0},
        }

    def _provenance(self) -> dict:
        provenance = self._default_provenance()
        supplied = self.model_card.get("artifact_provenance", {})
        if supplied is not None:
            if not isinstance(supplied, dict):
                raise ArtifactIntegrityError("artifact_provenance must be a JSON object")
            provenance.update(supplied)
        missing = [k for k in REQUIRED_ARTIFACT_PROVENANCE_FIELDS if k not in provenance]
        if missing:
            raise ArtifactIntegrityError(f"artifact_provenance missing fields: {missing}")
        return provenance

    @staticmethod
    def _validate_provenance(provenance: dict, dt: float) -> None:
        import json

        if not isinstance(provenance, dict):
            raise ArtifactIntegrityError("artifact_provenance must be a JSON object")
        missing = [k for k in REQUIRED_ARTIFACT_PROVENANCE_FIELDS if k not in provenance]
        if missing:
            raise ArtifactIntegrityError(f"artifact_provenance missing fields: {missing}")
        feature_schema = provenance["feature_schema"]
        if not isinstance(feature_schema, dict) or feature_schema.get("names") != list(CONTEXT_FEATURE_NAMES):
            raise ArtifactIntegrityError(
                "feature_schema.names does not match the deployed CONTEXT_FEATURE_NAMES"
            )
        time_semantics = provenance["time_semantics"]
        if not isinstance(time_semantics, dict) or abs(float(time_semantics.get("dt", float("nan"))) - dt) > 1e-12:
            raise ArtifactIntegrityError("artifact time_semantics.dt does not match artifact.dt")
        if not isinstance(provenance["training_config"], dict):
            raise ArtifactIntegrityError("training_config must be a JSON object")
        data_hash = provenance["data_aggregate_sha256"]
        if not isinstance(data_hash, str) or not data_hash:
            raise ArtifactIntegrityError("data_aggregate_sha256 must be a non-empty string")
        source_hash = provenance["source_sha256"]
        if not isinstance(source_hash, (str, dict)):
            raise ArtifactIntegrityError("source_sha256 must be a string or JSON object")
        bounds = provenance["human_physical_bounds"]
        if not isinstance(bounds, dict):
            raise ArtifactIntegrityError("human_physical_bounds must be a JSON object")
        for key in ("max_speed", "max_acceleration"):
            if key not in bounds or not np.isfinite(float(bounds[key])) or float(bounds[key]) <= 0.0:
                raise ArtifactIntegrityError(f"human_physical_bounds.{key} must be finite and >0")
        # Verify the metadata is composed only of deterministic JSON values.
        try:
            json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(f"artifact_provenance is not canonical JSON: {exc}") from exc

    def _canonical_array_bytes(self) -> bytes:
        """Deterministic byte serialization of every numeric array, in a
        FIXED order, used for the content hash -- this order must never
        change without bumping ``ARTIFACT_SCHEMA_VERSION`` (Order F1,
        2026-08-03)."""
        parts = [
            np.asarray(self.K, dtype=np.int64).tobytes(),
            np.ascontiguousarray(self.Pi, dtype=np.float64).tobytes(),
            np.ascontiguousarray(self.initial_distribution, dtype=np.float64).tobytes(),
            np.asarray(self.dt, dtype=np.float64).tobytes(),
        ]
        for k in range(self.K):
            for arr in (self.A[k], self.B[k], self.C[k], self.d[k], self.Q[k]):
                parts.append(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        import json
        provenance_json = json.dumps(
            self._provenance(), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")
        parts.append(provenance_json)
        return b"".join(parts)

    def content_sha256(self) -> str:
        import hashlib
        return hashlib.sha256(self._canonical_array_bytes()).hexdigest()

    def save(self, output_dir: str, tier: str, convergence: dict) -> None:
        """Order F1 (2026-08-03) production format: ``tier`` must be
        ``'engineering_only'`` or ``'production'``; ``convergence`` must be
        a dict with at least a ``'converged': bool`` key (as returned by
        ``fit()``'s per-K result, see ``fit``'s docstring). Saving with
        ``tier='production'`` on an unconverged fit is REFUSED -- this is
        an enforced constraint, not just a documentation note, since a
        K=5/6-style fit that only hit ``em_max_iters`` without meeting
        ``em_tol`` must never silently be treated as a paper-grade result.
        Embeds ``schema_version``, ``tier``, ``convergence``, and a content
        hash (covering every numeric array) into ``model_card.json`` so
        ``load()`` can fail closed on schema drift or corruption."""
        import json
        import os
        from pathlib import Path

        if tier not in VALID_ARTIFACT_TIERS:
            raise ArtifactIntegrityError(f"tier must be one of {VALID_ARTIFACT_TIERS}, got {tier!r}")
        if tier == "production" and not convergence.get("converged", False):
            raise ArtifactIntegrityError(
                f"refusing to save an UNCONVERGED fit as tier='production' (convergence={convergence!r}) -- "
                "use tier='engineering_only' for interface wiring; a non-converged fit must never be "
                "presented as a production/paper-grade artifact."
            )

        provenance = self._provenance()
        self._validate_provenance(provenance, self.dt)
        if tier == "production":
            if provenance["data_aggregate_sha256"] == "UNSPECIFIED":
                raise ArtifactIntegrityError("production artifact requires data_aggregate_sha256")
            source_hash = provenance["source_sha256"]
            if source_hash == "UNSPECIFIED" or (
                isinstance(source_hash, dict) and any(v == "UNSPECIFIED" for v in source_hash.values())
            ):
                raise ArtifactIntegrityError("production artifact requires source_sha256")
            if provenance["training_config"].get("status") == "not_recorded":
                raise ArtifactIntegrityError("production artifact requires training_config provenance")

        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            if not output_dir.is_dir() or any(output_dir.iterdir()):
                raise ArtifactIntegrityError(f"refusing to overwrite non-empty artifact directory {output_dir}")
            output_dir.rmdir()
        import shutil
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
        payload = {
            "K": np.asarray(self.K),
            "Pi": self.Pi,
            "initial_distribution": self.initial_distribution,
            "dt": np.asarray(self.dt),
        }
        for k in range(self.K):
            payload[f"A_{k}"] = self.A[k]
            payload[f"B_{k}"] = self.B[k]
            payload[f"C_{k}"] = self.C[k]
            payload[f"d_{k}"] = self.d[k]
            payload[f"Q_{k}"] = self.Q[k]
        try:
            np.savez(temp_dir / "arhmm.npz", **payload)

            model_card_full = dict(self.model_card)
            model_card_full["artifact_provenance"] = provenance
            model_card_full.update({
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "tier": tier,
                "convergence": convergence,
            })
            self.model_card = model_card_full
            model_card_full["content_sha256"] = self.content_sha256()
            (temp_dir / "model_card.json").write_text(
                json.dumps(model_card_full, indent=2, sort_keys=True, ensure_ascii=True, default=str),
                encoding="utf-8",
            )
            # The destination did not exist while files were written; the
            # completed directory becomes visible in one rename.
            os.replace(temp_dir, output_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    @classmethod
    def load(cls, output_dir: str, expect_tier: str = None) -> "ARHMMArtifact":
        """Fails CLOSED (raises ``ArtifactIntegrityError``) if: the model
        card is missing, ``schema_version`` does not match
        ``ARTIFACT_SCHEMA_VERSION``, the recomputed content hash does not
        match what was recorded at save time (corruption or hand-editing),
        or (when ``expect_tier`` is given) the artifact's own recorded tier
        does not match what the CALLER expected -- e.g. a production
        pipeline explicitly passing ``expect_tier='production'`` must never
        silently load an ``engineering_only`` artifact."""
        import json
        from pathlib import Path

        output_dir = Path(output_dir)
        data = np.load(output_dir / "arhmm.npz")
        K = int(data["K"])
        card_path = output_dir / "model_card.json"
        if not card_path.exists():
            raise ArtifactIntegrityError(f"{card_path} missing -- cannot verify schema_version/content hash")
        model_card = json.loads(card_path.read_text(encoding="utf-8"))

        schema_version = model_card.get("schema_version")
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactIntegrityError(
                f"schema_version mismatch at {output_dir}: file has {schema_version!r}, "
                f"code expects {ARTIFACT_SCHEMA_VERSION!r}"
            )
        provenance = model_card.get("artifact_provenance")
        cls._validate_provenance(provenance, float(data["dt"]))

        artifact = cls(
            K=K,
            A=[data[f"A_{k}"] for k in range(K)],
            B=[data[f"B_{k}"] for k in range(K)],
            C=[data[f"C_{k}"] for k in range(K)],
            d=[data[f"d_{k}"] for k in range(K)],
            Q=[data[f"Q_{k}"] for k in range(K)],
            Pi=data["Pi"],
            initial_distribution=data["initial_distribution"],
            dt=float(data["dt"]),
            model_card=model_card,
        )
        recorded_hash = model_card.get("content_sha256")
        recomputed_hash = artifact.content_sha256()
        if recomputed_hash != recorded_hash:
            raise ArtifactIntegrityError(
                f"content hash mismatch at {output_dir}: recorded {recorded_hash!r}, "
                f"recomputed {recomputed_hash!r} -- file may be corrupted or hand-edited"
            )
        if model_card.get("tier") == "production":
            if provenance["data_aggregate_sha256"] == "UNSPECIFIED":
                raise ArtifactIntegrityError("production artifact has unspecified data_aggregate_sha256")
            if provenance["source_sha256"] == "UNSPECIFIED":
                raise ArtifactIntegrityError("production artifact has unspecified source_sha256")
        if expect_tier is not None and model_card.get("tier") != expect_tier:
            raise ArtifactIntegrityError(
                f"expected tier={expect_tier!r} at {output_dir}, found tier={model_card.get('tier')!r}"
            )
        return artifact


@dataclass
class ARHMMSequence:
    """One pedestrian track's full, time-ordered SUPERVISED-TRANSITION
    sequence -- the EM algorithm's unit of computation. Unlike
    ``mode_model.TransitionRow`` (a single transition, suited to the
    legacy per-transition K-means+regression fit), forward-backward needs
    the WHOLE sequence per track to compute a valid posterior over
    ``z_{0:L-1}``.

    2026-08-03 audit fix: an earlier version stored ``v``/``u_robot``/``c``
    all as ``[T, ...]`` from a ``T``-length raw track, relying on every
    reader remembering "only indices 0..T-2 are usable, robot_actions[T-1]
    is dangling with no recorded next-velocity label" -- an implicit,
    easy-to-violate slicing convention. Every array here is now already
    trimmed to exactly ``L = T-1`` SUPERVISED rows, with an explicit
    ``(v_current, u_robot, context) -> v_next`` regression target for each
    row -- there is no dangling last action and no off-by-one to get
    wrong."""

    track_key: Tuple[int, int]
    v_current: np.ndarray  # [L, 2] this track's own velocity at the START of each supervised transition
    v_next: np.ndarray     # [L, 2] this track's own velocity one dt later -- the regression TARGET, from the SAME track's next consecutive frame
    u_robot: np.ndarray    # [L, 2] robot's EXECUTED action at the transition's start step (== episode['robot_actions'][t], never derived from robot[:,2:4])
    context: np.ndarray    # [L, len(CONTEXT_FEATURE_NAMES)] relative-geometry context at the transition's start step (robot-relative only, no self-kinematics)

    def __post_init__(self) -> None:
        L = self.v_current.shape[0]
        if not (self.v_next.shape[0] == self.u_robot.shape[0] == self.context.shape[0] == L):
            raise ValueError(
                f"ARHMMSequence arrays must all have the same length L, got "
                f"v_current={self.v_current.shape[0]}, v_next={self.v_next.shape[0]}, "
                f"u_robot={self.u_robot.shape[0]}, context={self.context.shape[0]}"
            )


def extract_sequences(episodes: List[dict], dt: float) -> List[ARHMMSequence]:
    """Analogous to ``mode_model.extract_transitions``, but returns whole
    per-track SEQUENCES (not flattened single-step transitions) and reads
    ``episode['robot_actions']`` directly for ``u_robot`` rather than
    deriving it from ``robot[:, 2:4]`` (even though Upgrade U0 verified
    those are bit-identical one step apart -- using the field whose name
    and contract explicitly mean "action" removes any ambiguity for
    someone reading this code later). Only maximal runs of STRICTLY
    CONSECUTIVE valid frames become one ``ARHMMSequence`` -- a gap (a
    missing/invalid frame) splits a track into two separate sequences
    rather than bridging over it (same D3 discipline as
    ``mode_model.extract_transitions``: never treat a skipped frame's
    delta as if it were a single-``dt`` step)."""
    sequences: List[ARHMMSequence] = []
    for episode_index, episode in enumerate(episodes):
        humans = np.asarray(episode["humans"], dtype=np.float64)
        track_ids = np.asarray(episode["human_track_ids"])
        robot = np.asarray(episode["robot"], dtype=np.float64)
        robot_actions = np.asarray(episode["robot_actions"], dtype=np.float64)
        valid_mask = np.asarray(episode["valid_mask"], dtype=bool)
        T, N = track_ids.shape[0], track_ids.shape[1]

        by_track: Dict[int, List[int]] = {}
        for t in range(T):
            for n in range(N):
                if valid_mask[t, n]:
                    by_track.setdefault(int(track_ids[t, n]), []).append(t)

        for track_id, times in by_track.items():
            slot_at_t = {t: int(np.where(track_ids[t] == track_id)[0][0]) for t in times}
            run: List[int] = [times[0]]
            for prev_t, t in zip(times, times[1:]):
                if t - prev_t == 1:
                    run.append(t)
                else:
                    if len(run) >= 2:
                        sequences.append(_build_sequence(episode_index, track_id, run, humans, robot, robot_actions, slot_at_t))
                    run = [t]
            if len(run) >= 2:
                sequences.append(_build_sequence(episode_index, track_id, run, humans, robot, robot_actions, slot_at_t))
    return sequences


def compute_context_features(
    human_position: np.ndarray, human_velocity: np.ndarray,
    robot_position_now: np.ndarray, robot_velocity_now: np.ndarray,
) -> np.ndarray:
    """[len(CONTEXT_FEATURE_NAMES)] relative-geometry/TTC context -- the ONE
    canonical implementation of this formula (Order F3, 2026-08-03:
    previously duplicated independently in ``_build_sequence`` below,
    ``mode_model.extract_transitions``, and ``belief_tracker._build_context``
    -- three copies of the same formula is exactly the kind of drift risk
    that could silently make online scoring/rollout disagree with what the
    model was fit on). All four inputs are plain ``[2]`` numpy arrays
    (position/velocity), not tied to any particular episode-array or
    dataclass layout, so both offline fitting (via a thin per-row wrapper)
    and online belief tracking / trajectory rollout can share this single
    function."""
    from crowd_nav.bayesian_brne.mode_model import TTC_CLIP

    rel_pos = np.asarray(human_position) - np.asarray(robot_position_now)
    rel_vel = np.asarray(human_velocity) - np.asarray(robot_velocity_now)
    dist = float(np.hypot(rel_pos[0], rel_pos[1]) + 1e-6)
    closing = float(-np.dot(rel_pos, rel_vel) / dist)
    ttc = dist / closing if closing > 0.1 else TTC_CLIP
    ttc_clipped = float(np.clip(ttc, 0.0, TTC_CLIP))
    passing_side = float((rel_pos[0] * rel_vel[1] - rel_pos[1] * rel_vel[0]) / dist)
    return np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], ttc_clipped, passing_side], dtype=np.float64)


def _build_sequence(episode_index, track_id, run, humans, robot, robot_actions, slot_at_t) -> ARHMMSequence:
    """``run`` is a maximal list of STRICTLY CONSECUTIVE raw frame indices
    (length >= 2). Builds ``L = len(run) - 1`` supervised rows: row ``i``'s
    target ``v_next[i]`` comes from ``run[i+1]`` (the SAME track's next
    consecutive frame -- guaranteed by ``run``'s consecutiveness, never a
    frame from a different track or a skipped-over gap), and its
    ``u_robot[i]``/``context[i]`` come from ``run[i]`` (the transition's
    start step)."""
    from crowd_nav.bayesian_brne.schemas import robot_position, robot_velocity

    n_raw = len(run)
    raw_v = np.zeros((n_raw, 2))
    raw_u = np.zeros((n_raw, 2))
    raw_c = np.zeros((n_raw, len(CONTEXT_FEATURE_NAMES)))
    for i, t in enumerate(run):
        slot = slot_at_t[t]
        px, py, vx, vy = humans[t, slot, :4]
        raw_v[i] = [vx, vy]
        raw_u[i] = robot_actions[t]
        r = robot[t]
        raw_c[i] = compute_context_features(
            np.array([px, py]), np.array([vx, vy]), robot_position(r), robot_velocity(r),
        )

    return ARHMMSequence(
        track_key=(episode_index, track_id),
        v_current=raw_v[:-1], v_next=raw_v[1:],
        u_robot=raw_u[:-1], context=raw_c[:-1],
    )


def _design_dims() -> Tuple[int, int, int, int]:
    d_v, d_u, d_c = 2, 2, len(CONTEXT_FEATURE_NAMES)
    return d_v, d_u, d_c, d_v + d_u + d_c + 1  # +1 for the constant/offset term


def _emission_log_probs(sequence: ARHMMSequence, artifact: ARHMMArtifact) -> np.ndarray:
    """[L, K] log N(v_next[t]; A_k v_current[t] + B_k u_robot[t] + C_k context[t] + d_k, Q_k).

    2026-08-03 (U3 performance pass): vectorized over t. The original
    implementation called ``mode_model._logpdf_gaussian`` (its own
    ``slogdet`` + ``inv`` + Python-level dot products) once per (t, k) pair
    -- profiling a real-data fit showed this dominated e_step's cost (a
    390,000-row/2000-sequence fit took ~700s wall time, ~100x more than
    m_step). Since ``mean`` is an affine function of the PER-ROW inputs but
    ``Q_k`` is the SAME covariance for every row of a given mode, the whole
    [L] axis can be computed as one batched Gaussian log-density -- this is
    the same arithmetic ``_logpdf_gaussian`` does per row
    (``-0.5*(diff^T Q^-1 diff + logdet + d*log(2pi))``), just with ``diff``
    as an ``[L,2]`` array and the quadratic form computed via ``einsum``
    instead of a Python loop. Verified bit-for-bit equivalent (max abs diff
    ~1e-13, floating-point summation-order noise only) against the original
    per-(t,k) loop by ``arhmm_emission_log_probs_matches_looped_reference``
    in selftest.py.

    2026-08-03 (U3 monotonicity fix): does NOT add ``mode_model.
    _logpdf_gaussian``'s ``+1e-6*I`` floor -- that floor was copied over
    when this function was first written, but is wrong for THIS model.
    ``Q_k`` here is guaranteed strictly positive-definite by construction
    (``m_step``'s ``Psi_n = psi0 + sse + ridge*W@W^T`` is ``psi0`` --
    already PD since ``ARHMMConfig`` requires ``inverse_wishart_scale>0``
    -- plus two positive-semi-definite terms), so no floor is needed for
    numerical safety. Worse, an ALWAYS-ON floor silently changes what
    likelihood is being evaluated: once a mode's true ``Q_k`` legitimately
    shrinks toward ``~1e-6`` (real, observed on real formal data at K=2 --
    one mode's occupancy narrows over EM iterations, sharpening its own
    residual covariance, a known GMM-style pathology), ``Q_k+1e-6*I``
    stops being a negligible perturbation of ``Q_k`` -- so ``e_step``'s
    ``ll`` (computed under the FLOORED covariance) and ``log_prior_density``
    /``m_step`` (which target the RAW ``Q_k``, no floor) silently refer to
    two DIFFERENT models, breaking the exact E-step/M-step correspondence
    the EM monotonicity theorem depends on. This is what caused a genuine
    ``EMMonotonicityError`` on a real U3 K=2 run at iteration 16 (objective
    dropped from 1021079.60 to 1020804.61, reproducibly, even after fixing
    an unrelated Psi_n derivation bug found along the way, ruling that bug
    out as the cause). See guide.md for the full diagnostic trail and
    whether removing this floor actually resolved the real-data run."""
    L = sequence.v_current.shape[0]
    K = artifact.K
    log_b = np.zeros((L, K))
    for k in range(K):
        mean = (
            sequence.v_current @ artifact.A[k].T
            + sequence.u_robot @ artifact.B[k].T
            + sequence.context @ artifact.C[k].T
            + artifact.d[k][None, :]
        )  # [L, 2]
        diff = sequence.v_next - mean  # [L, 2]
        cov = artifact.Q[k]
        _sign, logdet = np.linalg.slogdet(cov)
        inv = np.linalg.inv(cov)
        quad = np.einsum("ij,jk,ik->i", diff, inv, diff)
        log_b[:, k] = -0.5 * (quad + logdet + 2 * np.log(2.0 * np.pi))
    return log_b


def _forward_backward_full(sequence: ARHMMSequence, artifact: ARHMMArtifact):
    from scipy.special import logsumexp

    log_b = _emission_log_probs(sequence, artifact)
    T_minus_1, K = log_b.shape
    log_pi0 = np.log(np.maximum(artifact.initial_distribution, 1e-300))
    log_Pi = np.log(np.maximum(artifact.Pi, 1e-300))

    log_alpha = np.zeros((T_minus_1, K))
    log_alpha[0] = log_pi0 + log_b[0]
    for t in range(1, T_minus_1):
        prev = log_alpha[t - 1][:, None] + log_Pi  # prev[j,k] = log_alpha[t-1,j] + log_Pi[j,k]
        log_alpha[t] = log_b[t] + logsumexp(prev, axis=0)

    log_beta = np.zeros((T_minus_1, K))
    log_beta[-1] = 0.0
    for t in range(T_minus_1 - 2, -1, -1):
        nxt = log_Pi + (log_b[t + 1] + log_beta[t + 1])[None, :]  # nxt[k,j] = log_Pi[k,j] + log_b[t+1,j] + log_beta[t+1,j]
        log_beta[t] = logsumexp(nxt, axis=1)

    seq_ll = float(logsumexp(log_alpha[-1]))
    return log_alpha, log_beta, seq_ll, log_b, log_Pi


def forward_backward(sequence: ARHMMSequence, artifact: ARHMMArtifact) -> Tuple[np.ndarray, np.ndarray, float]:
    """Log-space forward-backward for ONE sequence under the CURRENT
    artifact. Returns ``(log_alpha[L, K], log_beta[L, K],
    sequence_log_likelihood)`` where ``L = sequence.v_current.shape[0]`` is
    the number of already-supervised transitions in this sequence.
    Numerically stable throughout via ``scipy.special.logsumexp``
    (never plain products of probabilities); ``exp(log_alpha[t]+log_beta[t]
    -sequence_log_likelihood)`` is a valid (non-negative, sums-to-1)
    distribution over ``z_t`` at every ``t`` -- verified by
    ``arhmm_gamma_sums_to_one`` in selftest.py."""
    log_alpha, log_beta, seq_ll, _log_b, _log_Pi = _forward_backward_full(sequence, artifact)
    return log_alpha, log_beta, seq_ll


def e_step(sequences: List[ARHMMSequence], artifact: ARHMMArtifact) -> Dict[str, object]:
    """Runs forward-backward and returns per-sequence responsibilities
    (``gammas``, single-time marginals) and pairwise responsibilities
    (``xis``, consecutive-pair marginals -- the expected transition counts
    the M-step's sticky-Dirichlet Pi update needs), the total data
    log-likelihood under the CURRENT artifact (used by ``fit`` for the EM
    convergence check), and ``sequence_log_likelihoods`` (each sequence's
    OWN log-likelihood, same order as the input ``sequences`` -- added for
    U3's suite-seed block bootstrap, which needs per-sequence values to
    group by suite seed rather than only the pooled total).

    2026-08-03 (U3 performance pass): rewritten to batch the forward-backward
    recursion ACROSS all sequences that share the same length L, instead of
    looping over sequences one at a time and calling ``scipy.special.
    logsumexp`` once per (sequence, timestep) pair. Real formal data has
    every non-gapped track at the SAME length (39 supervised transitions per
    40-frame episode), so this turns ~2000 length-39 per-sequence recursions
    into ONE batched recursion of 39 vectorized steps -- profiling showed
    this loop (not the M-step, already vectorized separately) was the
    dominant cost of a real-data fit (~700s for 390,000 rows/K=1,2).
    Batching by EXACT length (no padding, no masking) means every batched
    step operates on fully-valid data for every sequence in that batch --
    there is no boundary/edge case to get subtly wrong, unlike a padded
    variable-length batch would need. Mathematically each batched step is
    the SAME per-sequence recursion computed in parallel across the batch
    axis (no cross-sequence terms), so results must be bit-identical (up to
    floating-point summation order) to running ``forward_backward`` on each
    sequence individually -- verified by
    ``arhmm_batched_e_step_matches_looped_reference`` in selftest.py, which
    cross-checks against ``forward_backward`` (kept unchanged as the
    reference single-sequence implementation) on deliberately
    MIXED-length synthetic sequences."""
    from scipy.special import logsumexp

    K = artifact.K
    log_pi0 = np.log(np.maximum(artifact.initial_distribution, 1e-300))
    log_Pi = np.log(np.maximum(artifact.Pi, 1e-300))

    by_length: Dict[int, List[int]] = {}
    for idx, seq in enumerate(sequences):
        by_length.setdefault(seq.v_current.shape[0], []).append(idx)

    gammas: List[np.ndarray] = [None] * len(sequences)  # type: ignore[list-item]
    xis: List[np.ndarray] = [None] * len(sequences)  # type: ignore[list-item]
    sequence_lls: List[float] = [None] * len(sequences)  # type: ignore[list-item]
    total_ll = 0.0

    for L, indices in by_length.items():
        group = [sequences[i] for i in indices]
        n = len(group)
        log_b = np.stack([_emission_log_probs(seq, artifact) for seq in group], axis=0)  # [n, L, K]

        log_alpha = np.zeros((n, L, K))
        log_alpha[:, 0, :] = log_pi0[None, :] + log_b[:, 0, :]
        for t in range(1, L):
            prev = log_alpha[:, t - 1, :, None] + log_Pi[None, :, :]  # [n, K(j), K(k)]
            log_alpha[:, t, :] = log_b[:, t, :] + logsumexp(prev, axis=1)

        log_beta = np.zeros((n, L, K))
        for t in range(L - 2, -1, -1):
            nxt = log_Pi[None, :, :] + (log_b[:, t + 1, :] + log_beta[:, t + 1, :])[:, None, :]  # [n, K(k), K(j)]
            log_beta[:, t, :] = logsumexp(nxt, axis=2)

        seq_ll = logsumexp(log_alpha[:, -1, :], axis=1)  # [n]
        total_ll += float(seq_ll.sum())

        gamma = np.exp(np.clip(log_alpha + log_beta - seq_ll[:, None, None], -700, 0))
        gamma = gamma / np.maximum(gamma.sum(axis=2, keepdims=True), 1e-300)

        if L >= 2:
            mat = (
                log_alpha[:, :-1, :, None] + log_Pi[None, None, :, :]
                + log_b[:, 1:, None, :] + log_beta[:, 1:, None, :] - seq_ll[:, None, None, None]
            )
            xi = np.exp(np.clip(mat, -700, 0))
            s = xi.sum(axis=(2, 3), keepdims=True)
            safe_s = np.where(s > 1e-300, s, 1.0)  # avoid div-by-zero warnings; matches original's if/else exactly
            xi = np.where(s > 1e-300, xi / safe_s, xi)
        else:
            xi = np.zeros((n, 0, K, K))

        for local_i, global_i in enumerate(indices):
            gammas[global_i] = gamma[local_i]
            xis[global_i] = xi[local_i]
            sequence_lls[global_i] = float(seq_ll[local_i])

    return {"log_likelihood": total_ll, "gammas": gammas, "xis": xis, "sequence_log_likelihoods": sequence_lls}


def m_step(sequences: List[ARHMMSequence], e_result: Dict[str, object], K: int, config: ARHMMConfig, dt: float) -> ARHMMArtifact:
    """Closed-form MAP update of ``(A_k, B_k, C_k, d_k, Q_k, Pi,
    initial_distribution)`` given ``sequences``, the E-step's
    responsibilities, and ``config``'s sticky/MNIW priors.

    2026-08-03 second audit round fix: this is now a TRUE conjugate
    matrix-normal/inverse-Wishart (MNIW) update, not the earlier
    Q-decoupled ridge + separately-updated-Q approximation. Standard
    conjugate multivariate-regression result (e.g. Bayesian VAR literature)
    for ``Y = X W^T + E``, ``E_t ~ N(0, Q)``, prior ``W|Q ~
    MatrixNormal(0, Q, V0)`` with ``V0 = shrinkage_scale * I``, ``Q ~
    InverseWishart(Psi0, nu0)``:

    ``V_n^{-1} = V0^{-1} + Sxx``  (``Sxx = sum_t gamma_t x_t x_t^T``)
    ``W_n = Sxy^T @ V_n``          (``Sxy = sum_t gamma_t x_t y_t^T``) --
        note this does NOT depend on Q (a known property of matrix-normal
        regression: Q cancels out of the W-conditional's posterior mean),
        so it is numerically the SAME formula the earlier (wrongly
        decoupled) version used -- what was missing was the next line:
    ``Psi_n = Psi0 + Syy - W_n @ V_n^{-1} @ W_n^T``  (``Syy = sum_t
        gamma_t y_t y_t^T``) -- an earlier version omitted the
        ``- W_n @ V_n^{-1} @ W_n^T`` correction and used a plain weighted
        residual sum instead, which is only equal to this for an
        unregularized (``V0^{-1}=0``) fit. This ALGEBRAIC form is
        mathematically identical to ``Psi0 + sum_t gamma_t (y_t - W_n
        x_t)(y_t - W_n x_t)^T`` (the standard weighted-least-squares
        residual-sum-of-squares identity), but 2026-08-03 (U3 numerical-
        stability fix) switched the IMPLEMENTATION to the direct residual
        form: the algebraic form requires computing Syy and
        ``W_n@Vn_inv@W_n^T`` SEPARATELY (each an O(n_eff)-scale sum) and
        subtracting, which loses catastrophic precision once a mode's true
        residual covariance is small relative to its data scale -- exactly
        what happens when a mode's Q shrinks toward near-singularity during
        EM (a real, observed pathology on real formal data at K=2: a
        well-populated mode's two formulas agreed to 5 significant figures,
        but a collapsing mode's disagreed by a full order of magnitude,
        directly causing a genuine ``EMMonotonicityError``). The direct
        residual form never subtracts two large near-equal quantities, so
        it does not have this failure mode; see ``m_step``'s implementation
        comment for the exact diagnostic numbers.
    ``Q_k = Psi_n / (nu0 + n_eff + D + d + 1)`` (posterior MODE; ``D =
        d_full``, the design-vector width -- 2026-08-03 third audit round
        fix: an earlier version omitted ``+D`` here. The matrix-normal
        prior on ``W_k`` has its OWN ``|Q_k|^{-D/2}`` normalizing constant;
        profiling ``W_k`` out at its optimum ``W_n`` does not remove this
        term, since ``p(W_n|Q_k)``'s normalizing constant still depends on
        ``Q_k`` even though ``W_n`` itself does not. Valid for any
        ``nu0 > 0``; the PRIOR's own propriety separately requires
        ``nu0 > d-1 = 1``, enforced by ``ARHMMConfig.__post_init__``).

    Pi/initial_distribution: Dirichlet posterior MODE (not mean -- must
    match what ``log_prior_density`` tracks as the MAP objective), i.e.
    ``(alpha_prior + counts - 1) / sum(alpha_prior + counts - 1)``, valid
    whenever ``alpha_prior > 1`` everywhere, which
    ``ARHMMConfig.__post_init__``'s ``dirichlet_alpha > 1`` enforces."""
    d_v, d_u, d_c, d_full = _design_dims()
    gammas = e_result["gammas"]
    xis = e_result["xis"]

    Sxx = [np.zeros((d_full, d_full)) for _ in range(K)]
    Sxy = [np.zeros((d_full, 2)) for _ in range(K)]
    neff = np.zeros(K)
    transition_counts = np.zeros((K, K))
    initial_counts = np.zeros(K)

    all_xs = []
    all_ys = []
    for seq, gamma, xi in zip(sequences, gammas, xis):
        L = gamma.shape[0]
        # 2026-08-03 fifth audit round performance fix: seq.v_current/u_robot/
        # context are already [L,2]/[L,2]/[L,6] numpy arrays, so building
        # ``xs`` via a per-row Python for-loop (the original implementation)
        # does 5 small-array assignments PER ROW for no numerical reason --
        # a single np.concatenate produces the byte-identical array. This
        # loop is the dominant cost at real-data scale (a 390,000-row/
        # K=1,2 real-data fit took ~712s wall time with the per-row loop;
        # confirmed via em_max_iters=1-style regression that results are
        # unchanged after vectorizing, only wall time differs).
        xs = np.concatenate([seq.v_current, seq.u_robot, seq.context, np.ones((L, 1))], axis=1)
        ys = seq.v_next
        all_xs.append(xs)
        all_ys.append(ys)
        for k in range(K):
            w = gamma[:, k]
            Sxx[k] += (xs * w[:, None]).T @ xs
            Sxy[k] += (xs * w[:, None]).T @ ys
            neff[k] += w.sum()
        initial_counts += gamma[0]
        if xi.shape[0] > 0:
            transition_counts += xi.sum(axis=0)

    ridge = 1.0 / config.shrinkage_scale
    W = []
    for k in range(K):
        Vn_inv = Sxx[k] + ridge * np.eye(d_full)
        Wk = np.linalg.solve(Vn_inv, Sxy[k]).T  # [2, d_full] -- Q cancels out of this mean, see docstring
        W.append(Wk)

    # 2026-08-03 (U3 numerical-stability fix): Psi_n is computed via DIRECT
    # weighted residual sum-of-squares PLUS an explicit ridge/prior
    # correction term, instead of the algebraically-equivalent identity
    # ``Psi0 + Syy - W_k @ Vn_inv @ W_k^T`` an earlier version used. That
    # identity requires near-total cancellation between two O(neff)
    # quantities whenever a mode's TRUE residual covariance is small
    # relative to its data scale -- exactly what happens as a mode's Q
    # shrinks toward (near-)singularity during EM (a well-known GMM-style
    # pathology: one mode's covariance can shrink onto an increasingly
    # tight sub-cluster as it iterates).
    #
    # IMPORTANT (caught by re-running the FULL selftest suite, not just the
    # targeted diagnostic): the naive "just use the direct residual
    # sum-of-squares" reformulation, i.e. ``Psi0 + sum_t gamma_t (y_t-W_k
    # x_t)(y_t-W_k x_t)^T`` with NO further correction, is only equal to the
    # original algebraic identity when ridge=0 (unregularized) -- a first
    # attempt at this fix used exactly that, and a small-data synthetic
    # test (`arhmm_artifact_roundtrip_exact`, where ridge is NOT negligible
    # relative to Sxx) immediately caught a huge, genuine monotonicity
    # violation, proving the two forms are NOT interchangeable in general.
    # Re-deriving algebraically: since ``Vn_inv @ W_k^T = Sxy[k]`` (W_k's own
    # defining equation), ``W_k @ Vn_inv @ W_k^T = W_k @ Sxy[k]``, and
    # substituting ``Sxy[k] = Sxx[k] @ W_k^T + ridge * W_k^T`` (from
    # ``Vn_inv = Sxx+ridge*I``) into the direct-residual expansion shows the
    # exact identity:
    #     Psi0 + Syy - W_k@Vn_inv@W_k^T
    #   = Psi0 + [sum_t gamma_t (y_t-W_k x_t)(y_t-W_k x_t)^T] + ridge * W_k @ W_k^T
    # i.e. the CORRECT numerically-stable form needs the data-residual SSE
    # (safe: never subtracts two large near-equal quantities) PLUS an
    # explicit ridge-correction term for the prior's own pull of W_k toward
    # 0 (cheap and exact: W_k is only [2, d_full], no cancellation risk).
    # See guide.md's write-up for the numerical comparison against the
    # original algebraic formula on the real U3 K=2 run that surfaced this.
    xs_cat = np.concatenate(all_xs, axis=0)
    ys_cat = np.concatenate(all_ys, axis=0)
    gamma_cat = np.concatenate(gammas, axis=0)

    Q = []
    for k in range(K):
        psi0 = config.inverse_wishart_scale * np.eye(2)
        resid = ys_cat - xs_cat @ W[k].T
        w = gamma_cat[:, k]
        sse = (resid * w[:, None]).T @ resid
        psi_n = psi0 + sse + ridge * (W[k] @ W[k].T)
        psi_n = 0.5 * (psi_n + psi_n.T)
        # 2026-08-03 third audit round fix: the joint mode over (W,Q) needs
        # Q's own denominator to include D=d_full, not just nu0+neff+d+1.
        # The matrix-normal prior on W has its OWN |Q|^{-D/2} normalizing
        # constant (D = number of columns of W) -- profiling W out at its
        # optimum W_n does not remove this term (W_n's own location is
        # independent of Q, but p(W_n|Q)'s normalizing constant still
        # depends on Q). Omitting "+d_full" here made Q systematically too
        # large, which the multi-seed anisotropic-Q monotonicity stress
        # test (arhmm_monotonicity_holds_across_seeds_and_anisotropic_q)
        # caught as a slow, smooth objective DECREASE over several
        # iterations near convergence -- not floating-point noise, a real
        # missing term.
        Qk = psi_n / (config.inverse_wishart_dof + neff[k] + d_full + 2.0 + 1.0)
        Qk = 0.5 * (Qk + Qk.T)
        Q.append(Qk)

    alpha_prior = np.full((K, K), config.dirichlet_alpha) + np.eye(K) * config.sticky_kappa
    Pi_mode_unnorm = np.maximum(transition_counts + alpha_prior - 1.0, 1e-12)
    Pi = Pi_mode_unnorm / Pi_mode_unnorm.sum(axis=1, keepdims=True)

    pi0_mode_unnorm = np.maximum(initial_counts + config.dirichlet_alpha - 1.0, 1e-12)
    initial_distribution = pi0_mode_unnorm / pi0_mode_unnorm.sum()

    A = [W[k][:, :d_v] for k in range(K)]
    B = [W[k][:, d_v:d_v + d_u] for k in range(K)]
    C = [W[k][:, d_v + d_u:d_v + d_u + d_c] for k in range(K)]
    d_vec = [W[k][:, -1] for k in range(K)]

    return ARHMMArtifact(K=K, A=A, B=B, C=C, d=d_vec, Q=Q, Pi=Pi, initial_distribution=initial_distribution, dt=dt)


def _initialize_artifact(sequences: List[ARHMMSequence], K: int, config: ARHMMConfig, dt: float) -> ARHMMArtifact:
    """K-means-on-the-design-vector initialization -- ONLY used to seed the
    very first M-step's hard-assignment responsibilities; every artifact
    returned by ``fit`` afterward comes from real EM iterations, never from
    this initialization directly (guide.md Upgrade U1's explicit
    requirement)."""
    from crowd_nav.bayesian_brne.mode_model import _kmeans_pp, _standardize

    d_v, d_u, d_c, _ = _design_dims()
    feats = []
    owners = []
    for si, seq in enumerate(sequences):
        L = seq.v_current.shape[0]
        for t in range(L):
            feats.append(np.concatenate([seq.v_current[t], seq.u_robot[t], seq.context[t]]))
            owners.append(si)
    feats = np.array(feats) if feats else np.zeros((0, d_v + d_u + d_c))

    if K == 1 or feats.shape[0] < K:
        labels = np.zeros(len(owners), dtype=np.int64)
    else:
        standardized, _, _ = _standardize(feats)
        labels, _ = _kmeans_pp(standardized, K, seed=config.seed)

    gammas_init: List[np.ndarray] = []
    xis_init: List[np.ndarray] = []
    idx = 0
    for seq in sequences:
        L = seq.v_current.shape[0]
        g = np.zeros((L, K))
        for t in range(L):
            g[t, labels[idx] if idx < len(labels) else 0] = 1.0
            idx += 1
        gammas_init.append(g)
        xis_init.append(np.einsum("tj,tk->tjk", g[:-1], g[1:]) if g.shape[0] >= 2 else np.zeros((0, K, K)))

    init_stats = {"log_likelihood": float("-inf"), "gammas": gammas_init, "xis": xis_init}
    return m_step(sequences, init_stats, K, config, dt)


def log_prior_density(artifact: ARHMMArtifact, config: ARHMMConfig) -> float:
    """Log density (up to an additive constant that does not depend on the
    parameters, hence irrelevant to monotonicity checks across iterations)
    of the CURRENT ``(A, B, C, d, Q, Pi, initial_distribution)`` under this
    M-step's own MNIW / sticky-Dirichlet priors.

    2026-08-03 second audit round fix: this now matches the TRUE joint
    matrix-normal/inverse-Wishart log-density (not the earlier
    Q-independent isotropic-Gaussian-on-W approximation). For
    ``W_k|Q_k ~ MatrixNormal(0, Q_k, V0)`` with ``V0=shrinkage_scale*I``
    (``D = d_full`` columns) and ``Q_k ~ InverseWishart(Psi0, nu0)``, the
    joint log-density (dropping additive constants) is:

    ``log p(W_k, Q_k) = -0.5*(D + nu0 + d + 1)*log|Q_k|
                         - 0.5*tr(Q_k^{-1} @ (W_k @ W_k^T / shrinkage_scale + Psi0))``

    (``d=2``, the output dimension). This REPLACES the earlier
    ``-0.5*sum(W_k**2)/shrinkage_scale`` term, which ignored ``Q_k``
    entirely and did not correspond to what ``m_step`` actually
    maximizes. Also now includes ``initial_distribution``'s own Dirichlet
    log-density (an earlier version omitted it, even though
    ``initial_distribution`` is smoothed/updated every M-step)."""
    K = artifact.K
    d_full = artifact.A[0].shape[1] + artifact.B[0].shape[1] + artifact.C[0].shape[1] + 1
    log_prior = 0.0
    for k in range(K):
        Wk = np.concatenate([artifact.A[k], artifact.B[k], artifact.C[k], artifact.d[k][:, None]], axis=1)
        Q_inv = np.linalg.inv(artifact.Q[k])
        _, logdet_Q = np.linalg.slogdet(artifact.Q[k])
        psi0 = config.inverse_wishart_scale * np.eye(2)
        nu0 = config.inverse_wishart_dof

        quad_term = Wk @ Wk.T / config.shrinkage_scale + psi0
        log_prior += -0.5 * (d_full + nu0 + 2.0 + 1.0) * logdet_Q - 0.5 * float(np.trace(Q_inv @ quad_term))

    alpha_prior = np.full((K, K), config.dirichlet_alpha) + np.eye(K) * config.sticky_kappa
    for j in range(K):
        log_prior += float(np.sum((alpha_prior[j] - 1.0) * np.log(np.maximum(artifact.Pi[j], 1e-300))))

    log_prior += float(np.sum((config.dirichlet_alpha - 1.0) * np.log(np.maximum(artifact.initial_distribution, 1e-300))))
    return log_prior


def fit(
    sequences_train: List[ARHMMSequence],
    sequences_validation: List[ARHMMSequence],
    config: ARHMMConfig,
    dt: float = 0.25,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[ARHMMArtifact, Dict[str, object]]:
    """Top-level MAP-EM fit entry point (mirrors
    ``mode_model.fit_and_select``'s role, but sequence-level and EM-based
    rather than K-means-then-regression). K-means is used ONLY to
    initialize the first M-step's responsibilities (via
    ``_initialize_artifact``); every subsequent artifact comes from real
    E-step/M-step iterations. Convergence is checked on the PENALIZED
    objective (``log_likelihood + log_prior_density``, see
    ``log_prior_density``'s docstring for why), not the bare data
    log-likelihood. Returns ``(best_artifact_by_validation_ll, {K:
    {"artifact", "train_ll_history", "train_objective_history",
    "validation_ll", "convergence"}})``. ``convergence`` (Order F1,
    2026-08-03) is ``{"converged": bool, "reason": "em_tol_met"|
    "em_max_iters_exhausted", "n_iters", "em_max_iters", "em_tol",
    "final_objective", "final_objective_delta"}`` -- ``ARHMMArtifact.save``
    with ``tier="production"`` requires ``convergence["converged"]`` to be
    True, refusing to let an em_max_iters-exhausted fit (like the U3
    pilot's K=5/K=6) masquerade as a production artifact.

    ``best_K = argmax(validation_ll)`` here is a PLACEHOLDER selection
    rule for Upgrade U2 testing only -- it is exactly the naive
    "eligible-with-lowest-NLL" rule that let the legacy K-means model climb
    to its search boundary (Order 9S.1). Upgrade U3 must replace this with
    a pre-registered held-out sequential-NLL one-standard-error/
    minimal-sufficient-K rule before any K selected by this function is
    used as a real result."""
    results: Dict[int, Dict[str, object]] = {}
    for K in config.k_candidates:
        artifact = _initialize_artifact(sequences_train, K, config, dt)
        ll_history: List[float] = []
        objective_history: List[float] = []
        prev_objective = float("-inf")

        def _evaluate_and_check(current_artifact: ARHMMArtifact, prev_obj: float) -> Tuple[float, float]:
            # 2026-08-03 audit fix: ``ll`` and ``log_prior_density`` must be
            # evaluated at the SAME parameter values. An earlier version
            # computed ``ll`` via e_step on the artifact BEFORE this
            # iteration's m_step, then paired it with
            # log_prior_density(artifact_AFTER_m_step, ...) -- silently
            # mixing two different parameter states into one "objective"
            # number. Evaluating both terms on the SAME ``current_artifact``
            # keeps every recorded (ll, objective) pair internally
            # consistent.
            e_result_local = e_step(sequences_train, current_artifact)
            ll_local = e_result_local["log_likelihood"]
            objective_local = ll_local + log_prior_density(current_artifact, config)
            ll_history.append(ll_local)
            objective_history.append(objective_local)
            if progress_callback is not None:
                progress_callback({
                    "K": int(K), "iteration": len(objective_history),
                    "log_likelihood": float(ll_local), "penalized_objective": float(objective_local),
                    "previous_objective": float(prev_obj) if np.isfinite(prev_obj) else None,
                })
            if objective_local < prev_obj - _EM_MONOTONICITY_TOLERANCE:
                # 2026-08-03 second audit round: with the corrected TRUE
                # conjugate MNIW M-step, this objective has a real EM
                # monotonicity GUARANTEE (not just an empirical hope on one
                # fixture) -- a genuine decrease beyond floating-point noise
                # means a bug in e_step/m_step/log_prior_density, not
                # "expected occasional dips." Fail loudly rather than
                # silently return a fit that violated its own guarantee.
                raise EMMonotonicityError(
                    f"K={K}: penalized objective decreased from {prev_obj:.6f} to {objective_local:.6f} "
                    f"(tolerance {_EM_MONOTONICITY_TOLERANCE}) -- this indicates a bug in e_step/m_step/"
                    "log_prior_density, since the corrected MNIW M-step is supposed to guarantee "
                    "non-decreasing objective."
                )
            return objective_local, e_result_local

        converged = False
        for _ in range(config.em_max_iters):
            objective, e_result = _evaluate_and_check(artifact, prev_objective)
            if abs(objective - prev_objective) < config.em_tol:
                converged = True
                break
            prev_objective = objective
            artifact = m_step(sequences_train, e_result, K, config, dt)
        else:
            # 2026-08-03 fourth audit round fix: the for-loop above only
            # evaluates/checks the artifact BEFORE each iteration's m_step
            # call. If em_max_iters is exhausted WITHOUT ever triggering
            # the convergence break, the LAST m_step's output (the
            # artifact this function is about to return) was never itself
            # evaluated or monotonicity-checked -- Python's for/else runs
            # this block only when the loop completed without `break`,
            # exactly the case that needs one final check.
            _evaluate_and_check(artifact, prev_objective)
            # converged stays False: em_max_iters was exhausted without the
            # em_tol convergence criterion ever being met (Order F1,
            # 2026-08-03 -- this is what ARHMMArtifact.save's tier='production'
            # check refuses to accept; a real example is the U3 pilot's
            # K=5/K=6 fits, which both hit em_max_iters=100 without meeting
            # em_tol=1e-4).

        validation_ll = float("nan")
        if sequences_validation:
            val_result = e_step(sequences_validation, artifact)
            validation_ll = val_result["log_likelihood"]

        convergence = {
            "converged": converged,
            "reason": "em_tol_met" if converged else "em_max_iters_exhausted",
            "n_iters": len(objective_history),
            "em_max_iters": config.em_max_iters,
            "em_tol": config.em_tol,
            "final_objective": objective_history[-1],
            "final_objective_delta": (
                objective_history[-1] - objective_history[-2] if len(objective_history) >= 2 else float("nan")
            ),
        }
        results[K] = {
            "artifact": artifact, "train_ll_history": ll_history,
            "train_objective_history": objective_history, "validation_ll": validation_ll,
            "convergence": convergence,
        }

    if sequences_validation:
        best_K = max(results, key=lambda k: results[k]["validation_ll"])
    else:
        best_K = config.k_candidates[0]
    return results[best_K]["artifact"], results
