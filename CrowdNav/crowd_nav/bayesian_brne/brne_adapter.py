"""The ONLY module allowed to import upstream BRNE code or touch its RNG.

Wraps BRNE's recursive Bayesian trajectory-weight equilibrium solver (Sun,
Baldini, Hughes, Trautman, Murphey -- *Mixed strategy Nash equilibrium for
crowd navigation*, IJRR 2024, GPLv3, https://github.com/MurpheyLab/brne,
pinned commit ``633a5cdcb39ab27f18b596cb8cb1968644f82391``). Not vendored
into this repo; loaded by file path from the location recorded in
``policy_bayesian_brne.config``'s ``[brne_upstream]`` section (guide.md 4.1's
explicitly-sanctioned development-period fallback to a full git submodule).

Three solver modes, all operating on the SAME public signature
(``BRNESolver.solve(trajectories, radii, edge_mask)`` -- guide.md 7.1):

- ``official_exact``: calls the upstream module's own ``brne_nav`` function
  completely unmodified (including its hardcoded 10-iteration loop), via a
  lossless mean+perturbation decomposition of ``trajectories`` that
  reconstructs the exact input samples before upstream ever sees them (see
  ``_decompose_mean_perturbation``). This is the <1e-10 equivalence target
  from guide.md 7.2. It does not support ``radii`` or ``edge_mask`` --
  upstream's cost function is a dense, point-agent formula; accepting either
  argument here would silently stop being "official".
- ``stable``: our own log-space, NaN/Inf-guarded reimplementation of the
  same iterative update, derived from and attributed to the same algorithm.
  Adds the numerical safeguard upstream lacks (log-sum-exp-style
  max-subtraction before exponentiating, so an "all trajectories overlap"
  fixture cannot silently underflow every sample's weight to zero and then
  divide-by-zero into NaN -- guide.md 8.3's stress fixture). Supports
  ``edge_mask`` for the sparse interaction graph (guide.md 2.6).
- ``stable_clearance``: the production candidate solver. It keeps the stable
  update and sparse graph handling, but uses an entity-aware clearance cost
  with agent radii, safety margin, sigmoid slope, and explicit scale.

Everything else in this package must go through ``BRNESolver`` -- no other
module may import the upstream module or call ``numpy.random`` globally.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from crowd_nav.bayesian_brne.schemas import EquilibriumResult, require_finite, require_shape

_UPSTREAM_CACHE: Dict[str, object] = {}


def load_upstream_brne(brne_root: str):
    """Load the pinned upstream ``socnavbench/brne.py`` module by file path.
    Cached by resolved path so repeated calls do not re-exec the module
    (re-exec would also reset upstream's module-level ``rng``, which would
    silently change ``mvn_sample_normal``'s stream mid-run)."""
    module_path = str(Path(brne_root).expanduser().resolve() / "socnavbench" / "brne.py")
    if module_path in _UPSTREAM_CACHE:
        return _UPSTREAM_CACHE[module_path]
    if not Path(module_path).exists():
        raise FileNotFoundError(
            f"upstream BRNE not found at {module_path}. Check policy_bayesian_brne.config's "
            "[brne_upstream] root, or that the pinned checkout still exists."
        )
    spec = importlib.util.spec_from_file_location("_sm_brne_upstream", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _UPSTREAM_CACHE[module_path] = module
    return module


def _decompose_mean_perturbation(trajectories: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``[A, M, H] -> (mean [A, H], perturbation [A*M, H])`` such that
    ``mean[i] + perturbation[i*M:(i+1)*M] == trajectories[i]`` exactly.

    This is a pure bookkeeping split -- upstream's ``brne_nav``/``costs_nb``
    only ever see the reconstructed sum, never the split itself. It lets a
    genuinely multi-modal sample set (some samples clustered around a
    "turn left" rollout, others around "turn right") be fed into the
    UNMODIFIED upstream functions: upstream does not require its "mean" to
    be a meaningful central tendency, it only ever adds mean + perturbation
    back together before computing costs.
    """
    A, M, H = trajectories.shape
    mean = trajectories.mean(axis=1)  # [A, H]
    perturbation = (trajectories - mean[:, None, :]).reshape(A * M, H)
    return mean, perturbation


def _index_table(num_agents: int) -> np.ndarray:
    table = np.zeros((num_agents, num_agents), dtype=np.int64)
    for i in range(num_agents):
        table[i, 0] = i
        idx = 1
        for j in range(num_agents):
            if i == j:
                continue
            table[i, idx] = j
            idx += 1
    return table


def _costs_dense(traj_x: np.ndarray, traj_y: np.ndarray) -> np.ndarray:
    """Reproduces upstream ``costs_nb``'s exact formula in plain numpy (no
    numba, no parallelism) so ``stable`` mode's numerics can be reasoned
    about directly: ``cost_ij = 100 * max_t(2 - 2/(1+exp(-10*dist2_ij(t))))``
    where ``dist2`` is squared Euclidean distance. Point-agent, no radius
    term -- matches upstream exactly. The radius-aware
    ``stable_clearance`` cost is intentionally separate and must never be
    silently substituted into this baseline.
    """
    n = traj_x.shape[0]
    dx = traj_x[:, None, :] - traj_x[None, :, :]
    dy = traj_y[:, None, :] - traj_y[None, :, :]
    dist2 = dx ** 2 + dy ** 2
    per_step_cost = 2.0 - 2.0 / (1.0 + np.exp(-10.0 * dist2))
    return per_step_cost.max(axis=-1) * 100.0


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Overflow-safe sigmoid for clearance logits."""
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0.0
    out = np.empty_like(values)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _costs_clearance(
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    radii: np.ndarray,
    num_agents: int,
    num_pts: int,
    safe_distance: float,
    cost_sigma: float,
    cost_scale: float,
) -> np.ndarray:
    """Entity-aware pair cost for ``stable_clearance``.

    The flattened agent/sample axes repeat each agent radius across its M
    samples. The diagonal is zero because an agent is never its own partner.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if radii.shape != (num_agents,) or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError(f"radii must be finite and >0 with shape ({num_agents},), got {radii}")
    if not np.isfinite(safe_distance) or safe_distance < 0.0:
        raise ValueError(f"safe_distance must be finite and >=0, got {safe_distance}")
    if not np.isfinite(cost_sigma) or cost_sigma <= 0.0:
        raise ValueError(f"cost_sigma must be finite and >0, got {cost_sigma}")
    if not np.isfinite(cost_scale) or cost_scale <= 0.0:
        raise ValueError(f"cost_scale must be finite and >0, got {cost_scale}")

    dx = traj_x[:, None, :] - traj_x[None, :, :]
    dy = traj_y[:, None, :] - traj_y[None, :, :]
    distance = np.sqrt(np.maximum(dx * dx + dy * dy, 0.0))
    expanded_radii = np.repeat(radii, num_pts)
    pair_boundary = expanded_radii[:, None] + expanded_radii[None, :] + safe_distance
    clearance = distance - pair_boundary[..., None]
    costs = (cost_scale * _stable_sigmoid(-clearance / cost_sigma)).max(axis=-1)
    np.fill_diagonal(costs, 0.0)
    return costs


def _apply_edge_mask(costs: np.ndarray, edge_mask: Optional[np.ndarray], num_agents: int, num_pts: int) -> np.ndarray:
    if edge_mask is None:
        return costs
    require_shape(edge_mask, (num_agents, num_agents), "edge_mask")
    expanded = np.repeat(np.repeat(edge_mask, num_pts, axis=0), num_pts, axis=1)
    return np.where(expanded, costs, 0.0)


def _weights_update_stable(
    costs: np.ndarray,
    weights: np.ndarray,
    index_table: np.ndarray,
    num_agents: int,
    num_pts: int,
) -> Tuple[np.ndarray, bool]:
    """Log-space, max-subtracted reimplementation of upstream's
    ``weights_update_nb``. Mathematically the same update
    (``weights[i] = exp(-cost1)`` then normalized so the row MEAN is 1, not
    the row sum -- matching upstream's convention, required downstream by
    the weighted-mean-trajectory reconstruction), but subtracting each row's
    max log-weight before exponentiating means the best sample always maps
    to exactly 1.0 before normalization, so a row can never underflow to
    all-zero and divide-by-zero into NaN -- upstream's literal
    ``exp(-cost1)`` has no such guard.
    """
    new_weights = weights.copy()
    fallback = False
    all_pt_index = np.arange(num_agents * num_pts).reshape(num_agents, num_pts)
    for i in range(num_agents):
        row = index_table[i]
        other_rows = row[1:]
        other_idx = all_pt_index[other_rows].reshape(-1)
        other_weights = weights[other_rows].reshape(-1)
        this_idx = all_pt_index[row[0]]
        cost_block = costs[np.ix_(this_idx, other_idx)]  # [num_pts, (num_agents-1)*num_pts]
        cost1 = (cost_block * other_weights[None, :]).sum(axis=1) / max(1, (num_agents - 1) * num_pts)
        log_weight = -cost1
        shifted = log_weight - log_weight.max()
        row_weights = np.exp(shifted)
        row_mean = row_weights.mean()
        if not np.isfinite(row_mean) or row_mean <= 0.0:
            row_weights = np.ones(num_pts, dtype=np.float64)
            fallback = True
        else:
            row_weights = row_weights / row_mean
        if not np.all(np.isfinite(row_weights)):
            row_weights = np.ones(num_pts, dtype=np.float64)
            fallback = True
        new_weights[i] = row_weights
    return new_weights, fallback


class BRNESolver:
    def __init__(
        self,
        solver_mode: str = "official_exact",
        brne_root: Optional[str] = None,
        safe_distance: Optional[float] = None,
        cost_sigma: Optional[float] = None,
        cost_scale: Optional[float] = None,
    ):
        if solver_mode not in ("official_exact", "stable", "stable_clearance"):
            raise ValueError(f"unknown solver_mode: {solver_mode!r}")
        if solver_mode == "official_exact" and any(v is not None for v in (safe_distance, cost_sigma, cost_scale)):
            raise ValueError("official_exact cannot accept radius-aware collision-cost parameters")
        self.solver_mode = solver_mode
        self.brne_root = brne_root
        self.safe_distance = 0.20 if safe_distance is None else float(safe_distance)
        self.cost_sigma = 0.10 if cost_sigma is None else float(cost_sigma)
        self.cost_scale = 100.0 if cost_scale is None else float(cost_scale)
        if solver_mode == "stable_clearance":
            if not np.isfinite(self.safe_distance) or self.safe_distance < 0.0:
                raise ValueError(f"safe_distance must be finite and >=0, got {self.safe_distance}")
            if not np.isfinite(self.cost_sigma) or self.cost_sigma <= 0.0:
                raise ValueError(f"cost_sigma must be finite and >0, got {self.cost_sigma}")
            if not np.isfinite(self.cost_scale) or self.cost_scale <= 0.0:
                raise ValueError(f"cost_scale must be finite and >0, got {self.cost_scale}")

    def solve(
        self,
        trajectories: np.ndarray,  # [A, M, H, 2]
        radii: np.ndarray,  # [A]
        edge_mask: Optional[np.ndarray],  # [A, A] bool or None
        equilibrium_iterations: int = 10,
    ) -> EquilibriumResult:
        require_shape(trajectories, (None, None, None, 2), "trajectories")
        require_finite(trajectories, "trajectories")
        num_agents = trajectories.shape[0]
        require_shape(radii, (num_agents,), "radii")
        require_finite(radii, "radii")
        if np.any(np.asarray(radii) <= 0.0):
            raise ValueError(f"radii must be strictly positive, got {radii}")

        start = time.time()
        if self.solver_mode == "official_exact":
            if edge_mask is not None:
                raise ValueError(
                    "official_exact mode does not support edge_mask (upstream's cost function "
                    "is dense/point-agent); use solver_mode='stable' or "
                    "'stable_clearance' for sparse graphs."
                )
            result = self._solve_official_exact(trajectories)
        elif self.solver_mode == "stable":
            result = self._solve_stable(trajectories, edge_mask, equilibrium_iterations)
        else:
            result = self._solve_stable_clearance(trajectories, radii, edge_mask, equilibrium_iterations)
        result.elapsed_ms = (time.time() - start) * 1000.0
        return result

    def _solve_official_exact(self, trajectories: np.ndarray) -> EquilibriumResult:
        upstream = load_upstream_brne(self.brne_root)
        num_agents, num_pts, _, _ = trajectories.shape

        xmean, x_pts = _decompose_mean_perturbation(trajectories[..., 0])
        ymean, y_pts = _decompose_mean_perturbation(trajectories[..., 1])

        # upstream.brne_nav hardcodes its own 10-iteration loop internally --
        # "official_exact" means calling it exactly as published, not
        # reproducing its logic with a configurable iteration count (that is
        # what solver_mode="stable" is for).
        _x_opt, _y_opt, weights = upstream.brne_nav(
            list(xmean), list(ymean), x_pts, y_pts, num_agents, trajectories.shape[2], num_pts
        )
        return EquilibriumResult(
            weights=np.asarray(weights, dtype=np.float64),
            iterations=10,
            converged=True,
            max_weight_change=float("nan"),
            pair_cost_summary={},
            elapsed_ms=0.0,
            numeric_fallback_used=False,
        )

    def _solve_stable(
        self, trajectories: np.ndarray, edge_mask: Optional[np.ndarray], equilibrium_iterations: int
    ) -> EquilibriumResult:
        num_agents, num_pts, _horizon, _ = trajectories.shape
        traj_x = trajectories[..., 0].reshape(num_agents * num_pts, -1)
        traj_y = trajectories[..., 1].reshape(num_agents * num_pts, -1)

        costs = _costs_dense(traj_x, traj_y)
        costs = _apply_edge_mask(costs, edge_mask, num_agents, num_pts)
        index_table = _index_table(num_agents)

        weights = np.ones((num_agents, num_pts), dtype=np.float64)
        max_weight_change = 0.0
        numeric_fallback_used = False
        converged = False
        iteration = 0
        for iteration in range(equilibrium_iterations):
            new_weights, fallback = _weights_update_stable(costs, weights, index_table, num_agents, num_pts)
            max_weight_change = float(np.max(np.abs(new_weights - weights)))
            weights = new_weights
            numeric_fallback_used = numeric_fallback_used or fallback
            if max_weight_change < 1e-9:
                converged = True
                break

        return EquilibriumResult(
            weights=weights,
            iterations=iteration + 1,
            converged=converged,
            max_weight_change=max_weight_change,
            pair_cost_summary={"mean_cost": float(costs.mean()), "max_cost": float(costs.max())},
            elapsed_ms=0.0,
            numeric_fallback_used=numeric_fallback_used,
        )

    def _solve_stable_clearance(
        self, trajectories: np.ndarray, radii: np.ndarray,
        edge_mask: Optional[np.ndarray], equilibrium_iterations: int,
    ) -> EquilibriumResult:
        num_agents, num_pts, _horizon, _ = trajectories.shape
        traj_x = trajectories[..., 0].reshape(num_agents * num_pts, -1)
        traj_y = trajectories[..., 1].reshape(num_agents * num_pts, -1)
        costs = _costs_clearance(
            traj_x, traj_y, radii, num_agents, num_pts,
            self.safe_distance, self.cost_sigma, self.cost_scale,
        )
        costs = _apply_edge_mask(costs, edge_mask, num_agents, num_pts)
        index_table = _index_table(num_agents)

        weights = np.ones((num_agents, num_pts), dtype=np.float64)
        max_weight_change = 0.0
        numeric_fallback_used = False
        converged = False
        iteration = 0
        for iteration in range(equilibrium_iterations):
            new_weights, fallback = _weights_update_stable(costs, weights, index_table, num_agents, num_pts)
            max_weight_change = float(np.max(np.abs(new_weights - weights)))
            weights = new_weights
            numeric_fallback_used = numeric_fallback_used or fallback
            if max_weight_change < 1e-9:
                converged = True
                break

        return EquilibriumResult(
            weights=weights,
            iterations=iteration + 1,
            converged=converged,
            max_weight_change=max_weight_change,
            pair_cost_summary={
                "mean_cost": float(costs.mean()),
                "max_cost": float(costs.max()),
                "cost_mode": "stable_clearance",
                "safe_distance": self.safe_distance,
                "cost_sigma": self.cost_sigma,
                "cost_scale": self.cost_scale,
            },
            elapsed_ms=0.0,
            numeric_fallback_used=numeric_fallback_used,
        )


def weighted_first_control(weights: np.ndarray, robot_controls: np.ndarray) -> np.ndarray:
    """``u_t = sum_m normalized(q_robot[m]) * u_robot[m, 0]`` (guide.md 2.4).
    ``weights`` is the robot's own row (index 0) from an EquilibriumResult;
    ``robot_controls`` is ``[M, H, control_dim]``. Returns ``[control_dim]``.
    """
    normalized = weights / max(float(weights.sum()), 1e-12)
    return (normalized[:, None] * robot_controls[:, 0, :]).sum(axis=0)
