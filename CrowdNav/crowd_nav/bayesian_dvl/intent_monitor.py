"""Durable, resume-safe training telemetry for Intent-BDVL.

The model/training math lives in ``intent_train.py``. This module only
observes it and produces the same practical signals used by the mature
Mamba-VL loop: an append-only log, machine-readable episode rows,
TensorBoard scalars, rolling outcome rates, development validation, and
``curves.png``. It never changes actions, rewards, replay, or gradients.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


class IntentMonitorError(RuntimeError):
    pass


def append_durable_log(run_dir: Path, message: str) -> None:
    """Append one fsync'd line before ``TrainingMonitor`` exists.

    Formal runs spend substantial time collecting the shared ORCA corpus
    before the model monitor can be constructed.  Those episodes must be
    visible in the same ``train.log`` instead of creating a multi-hour
    silent interval at startup.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(line, flush=True)


@dataclass(frozen=True)
class DevelopmentSummary:
    online_episode: int
    n: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_return: float
    mean_navigation_time: float
    mean_path_ratio: float
    mean_min_clearance: float
    mean_discomfort_frequency: float
    by_scenario: Dict[str, Dict[str, float]]


def _rates(rows: Sequence[dict]) -> Dict[str, float]:
    n = len(rows)
    if not n:
        return {"success_rate": 0.0, "collision_rate": 0.0, "timeout_rate": 0.0}
    return {
        "success_rate": sum(r["outcome"] == "success" for r in rows) / n,
        "collision_rate": sum(r["outcome"] == "collision" for r in rows) / n,
        "timeout_rate": sum(r["outcome"] == "timeout" for r in rows) / n,
    }


def _mean(rows: Sequence[dict], key: str) -> float:
    values = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return float(np.mean(values)) if values else 0.0


def summarize_development(online_episode: int, rows: Sequence[dict]) -> DevelopmentSummary:
    if not rows:
        raise IntentMonitorError("development validation produced no episodes")
    by_scenario: Dict[str, Dict[str, float]] = {}
    for scenario in sorted({str(r["scenario"]) for r in rows}):
        subset = [r for r in rows if r["scenario"] == scenario]
        by_scenario[scenario] = {
            "n": len(subset),
            **_rates(subset),
            "mean_return": _mean(subset, "episode_return"),
            "mean_navigation_time": _mean(subset, "navigation_time"),
            "mean_path_ratio": _mean(subset, "path_ratio"),
            "mean_min_clearance": _mean(subset, "min_clearance"),
            "mean_discomfort_frequency": _mean(subset, "discomfort_frequency"),
        }
    return DevelopmentSummary(
        online_episode=int(online_episode),
        n=len(rows),
        **_rates(rows),
        mean_return=_mean(rows, "episode_return"),
        mean_navigation_time=_mean(rows, "navigation_time"),
        mean_path_ratio=_mean(rows, "path_ratio"),
        mean_min_clearance=_mean(rows, "min_clearance"),
        mean_discomfort_frequency=_mean(rows, "discomfort_frequency"),
        by_scenario=by_scenario,
    )


def run_development_validation(
    env_config_path: Path,
    model,
    action_table: np.ndarray,
    standard_validation_seeds: Sequence[int],
    junction_validation_seeds: Sequence[int],
    belief_mode: str,
    n_samples: int,
    horizon: int,
    device: str,
) -> List[dict]:
    """Greedy evaluation on development-only seeds.

    Each seed is evaluated once in both training scenarios. Planner RNGs
    are local to this function, so validation cannot advance any training
    RNG or alter replay/resume identity.
    """
    from crowd_nav.bayesian_dvl.intent_train import collect_online_episode

    rows: List[dict] = []
    was_training = bool(model.training)
    model.eval()
    try:
        scenario_seeds = {
            "standard": tuple(standard_validation_seeds),
            "junction_crowd": tuple(junction_validation_seeds),
        }
        if not scenario_seeds["standard"] or not scenario_seeds["junction_crowd"]:
            raise IntentMonitorError("both development scenarios require non-empty seed sets")
        for scenario_index, (scenario, seeds) in enumerate(scenario_seeds.items()):
            for seed in seeds:
                planner_seed = 7_000_000 + scenario_index * 1_000_000 + int(seed)
                result = collect_online_episode(
                    env_config_path,
                    model,
                    action_table,
                    scenario,
                    int(seed),
                    epsilon=0.0,
                    explore_rng=np.random.default_rng(planner_seed),
                    is_heldout=False,
                    gamma=1.0,
                    n_samples=n_samples,
                    horizon=horizon,
                    device=device,
                    belief_mode=belief_mode,
                )
                rows.append({
                    "scenario": scenario,
                    "episode_seed": int(seed),
                    "outcome": result.outcome,
                    "episode_return": result.episode_return,
                    "steps": result.steps,
                    "navigation_time": result.navigation_time,
                    "path_length": result.path_length,
                    "path_ratio": result.path_ratio,
                    "min_clearance": result.min_clearance,
                    "discomfort_frequency": result.discomfort_frequency,
                })
    finally:
        model.train(was_training)
    return rows


class TrainingMonitor:
    """One monitor per run directory.

    ``metrics.jsonl`` is the source of truth. On resume, rows ahead of the
    checkpoint cursor are atomically removed because the model/replay has
    rolled back to that checkpoint. A cursor ahead of the log fails closed
    instead of silently inventing missing history.
    """

    def __init__(
        self,
        run_dir: Path,
        online_cursor: int,
        il_cursor: int,
        resume: bool,
        rolling_windows: Sequence[int],
        tensorboard_enabled: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train.log"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.curves_path = self.run_dir / "curves.png"
        self.rolling_windows = tuple(sorted(set(int(v) for v in rolling_windows)))
        if not self.rolling_windows or any(v <= 0 for v in self.rolling_windows):
            raise IntentMonitorError(f"rolling windows must be positive: {rolling_windows}")
        self.records = self._read_records()
        self._reconcile(int(online_cursor), int(il_cursor), resume)
        self._log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._metrics_handle = self.metrics_path.open("a", encoding="utf-8", buffering=1)
        self._writer = None
        if tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except Exception as exc:  # monitoring must not kill a multi-hour run
                self.log(f"WARNING TensorBoard unavailable: {exc!r}")
        self.log(
            f"MONITOR {'RESUME' if resume else 'START'} online_cursor={online_cursor} "
            f"il_cursor={il_cursor} windows={self.rolling_windows}"
        )

    def _read_records(self) -> List[dict]:
        if not self.metrics_path.exists():
            return []
        rows = []
        lines = self.metrics_path.read_text().splitlines()
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                # A power loss can interrupt the one append currently in
                # progress. Only a malformed FINAL line is recoverable;
                # corruption in the middle remains a hard failure.
                if line_no == len(lines):
                    tmp = self.metrics_path.with_suffix(".jsonl.tmp")
                    with tmp.open("w", encoding="utf-8") as fh:
                        for row in rows:
                            fh.write(json.dumps(row, sort_keys=True) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, self.metrics_path)
                    break
                raise IntentMonitorError(f"invalid JSON in {self.metrics_path}:{line_no}: {exc}") from exc
        return rows

    def _reconcile(self, online_cursor: int, il_cursor: int, resume: bool) -> None:
        online = [int(r["online_episode"]) for r in self.records if r.get("record_type") == "online"]
        il = [int(r["il_pass"]) for r in self.records if r.get("record_type") == "il"]
        if online != sorted(set(online)) or il != sorted(set(il)):
            raise IntentMonitorError("metrics history contains duplicate or non-monotone cursors")
        if not resume:
            if self.records:
                raise IntentMonitorError(
                    f"fresh training requested in non-empty monitor directory {self.run_dir}; use resume or a new run dir"
                )
            return
        max_online = max(online, default=0)
        max_il = max(il, default=0)
        if max_online < online_cursor or max_il < il_cursor:
            raise IntentMonitorError(
                f"checkpoint cursor is ahead of durable metrics: online {online_cursor}>{max_online} "
                f"or IL {il_cursor}>{max_il}"
            )
        kept = [
            r for r in self.records
            if not (
                (r.get("record_type") == "online" and int(r["online_episode"]) > online_cursor)
                or (r.get("record_type") == "il" and int(r["il_pass"]) > il_cursor)
                or (r.get("record_type") == "validation" and int(r["online_episode"]) > online_cursor)
            )
        ]
        if len(kept) != len(self.records):
            tmp = self.metrics_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for row in kept:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.metrics_path)
        self.records = kept

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        if hasattr(self, "_log_handle"):
            self._log_handle.write(line + "\n")
            self._log_handle.flush()
            os.fsync(self._log_handle.fileno())
        print(line, flush=True)

    def _append(self, row: dict) -> None:
        self._metrics_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._metrics_handle.flush()
        os.fsync(self._metrics_handle.fileno())
        self.records.append(row)

    def record_diagnostics(self, il_pass: int, result) -> None:
        """Per-step gradient/optimizer diagnostics for the 2x2 experiment.

        Written as its own record_type so the training curve files stay
        readable and a diagnostic run's extra rows can never be mistaken for
        the loss series.
        """
        row = {
            "record_type": "il_diagnostics", "il_pass": int(il_pass),
            "mc_grad_norm": float(result.mc_grad_norm),
            "rank_grad_norm": float(result.rank_grad_norm),
            "projected_rank_norm": float(result.projected_rank_norm),
            "raw_cosine": float(result.gradient_cosine),
            "active_hinge_count": int(result.active_hinge_count),
            "active_hinge_fraction": float(result.active_hinge_fraction),
            "rank_scale": float(result.lambda_used),
            "combined_grad_norm": float(result.combined_grad_norm),
            "post_adam_cos_mc": float(result.post_adam_cos_mc),
            "post_adam_dot_mc": float(result.post_adam_dot_mc),
            "post_adam_cos_rank": float(result.post_adam_cos_rank),
            "adam_step": int(result.adam_step),
            "adam_exp_avg_norm": float(result.adam_exp_avg_norm),
            "adam_exp_avg_sq_norm": float(result.adam_exp_avg_sq_norm),
            "modules": result.module_diagnostics,
        }
        self._append(row)

    def record_il(self, il_pass: int, total: int, result) -> None:
        row = {
            "record_type": "il",
            "il_pass": int(il_pass),
            "il_total": int(total),
            "loss": float(result.loss),
            "mc_loss": float(result.mc_loss),
            "rank_loss": float(result.rank_loss),
            "grad_norm": float(result.grad_norm_preclip),
            "clipped": bool(result.clipped),
        }
        self._append(row)
        if self._writer is not None:
            for key in ("loss", "mc_loss", "rank_loss", "grad_norm"):
                self._writer.add_scalar(f"il/{key}", row[key], il_pass)

    def record_online(self, online_episode: int, target: int, result) -> Dict[int, Dict[str, float]]:
        if result.outcome not in ("success", "collision", "timeout"):
            raise IntentMonitorError(f"invalid online outcome {result.outcome!r}")
        row = {
            "record_type": "online",
            "online_episode": int(online_episode),
            "online_target": int(target),
            "scenario": result.scenario,
            "episode_seed": int(result.episode_seed),
            "outcome": result.outcome,
            "episode_return": float(result.episode_return),
            "steps": int(result.episode_steps),
            "navigation_time": float(result.navigation_time),
            "path_length": float(result.path_length),
            "path_ratio": float(result.path_ratio),
            "min_clearance": float(result.min_clearance),
            "discomfort_frequency": float(result.discomfort_frequency),
            "epsilon": float(result.epsilon),
            "loss": float(result.loss),
            "mc_loss": float(result.mc_loss),
            "rank_loss": float(result.rank_loss),
            "grad_norm": float(result.grad_norm_preclip),
            "clipped": bool(result.clipped),
            "n_demo": int(result.n_demo),
            "n_online": int(result.n_online),
        }
        self._append(row)
        online_rows = [r for r in self.records if r.get("record_type") == "online"]
        rolling: Dict[int, Dict[str, float]] = {}
        for window in self.rolling_windows:
            subset = online_rows[-window:]
            rolling[window] = {
                **_rates(subset),
                "mean_return": _mean(subset, "episode_return"),
                "mean_loss": _mean(subset, "loss"),
            }
        if self._writer is not None:
            self._writer.add_scalar("train/episode_return", row["episode_return"], online_episode)
            self._writer.add_scalar("train/loss", row["loss"], online_episode)
            self._writer.add_scalar("train/mc_loss", row["mc_loss"], online_episode)
            self._writer.add_scalar("train/rank_loss", row["rank_loss"], online_episode)
            self._writer.add_scalar("train/epsilon", row["epsilon"], online_episode)
            for window, stats in rolling.items():
                for key, value in stats.items():
                    self._writer.add_scalar(f"rolling_{window}/{key}", value, online_episode)
        return rolling

    def record_validation(self, summary: DevelopmentSummary) -> None:
        row = {"record_type": "validation", **asdict(summary)}
        self._append(row)
        if self._writer is not None:
            for key in (
                "success_rate", "collision_rate", "timeout_rate", "mean_return",
                "mean_navigation_time", "mean_path_ratio", "mean_min_clearance",
                "mean_discomfort_frequency",
            ):
                self._writer.add_scalar(f"development/{key}", row[key], summary.online_episode)
        self.log(
            f"DEV[{summary.online_episode}] n={summary.n} SR={summary.success_rate:.3f} "
            f"CR={summary.collision_rate:.3f} TR={summary.timeout_rate:.3f} "
            f"return={summary.mean_return:.3f} time={summary.mean_navigation_time:.2f}s "
            f"path={summary.mean_path_ratio:.3f} clearance={summary.mean_min_clearance:.3f} "
            f"discomfort={summary.mean_discomfort_frequency:.3f}"
        )

    def has_validation(self, online_episode: int) -> bool:
        return any(
            r.get("record_type") == "validation" and int(r["online_episode"]) == int(online_episode)
            for r in self.records
        )

    def plot(self) -> None:
        online = [r for r in self.records if r.get("record_type") == "online"]
        if not online:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.asarray([r["online_episode"] for r in online], dtype=np.int64)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        window = min(50, len(online))

        def rolling_mean(values: np.ndarray) -> np.ndarray:
            return np.asarray([
                float(np.mean(values[max(0, i - window + 1):i + 1]))
                for i in range(len(values))
            ])

        outcomes = {
            "Success": np.asarray([r["outcome"] == "success" for r in online], dtype=float),
            "Collision": np.asarray([r["outcome"] == "collision" for r in online], dtype=float),
            "Timeout": np.asarray([r["outcome"] == "timeout" for r in online], dtype=float),
        }
        for label, values in outcomes.items():
            axes[0, 0].plot(x, rolling_mean(values), label=label)
        axes[0, 0].set(title=f"Training outcomes (rolling {window})", xlabel="Online episode", ylabel="Rate", ylim=(-0.03, 1.03))
        axes[0, 0].legend()

        returns = np.asarray([r["episode_return"] for r in online], dtype=float)
        axes[0, 1].plot(x, returns, color="0.75", linewidth=0.7, label="Episode")
        axes[0, 1].plot(x, rolling_mean(returns), color="tab:blue", label=f"Rolling {window}")
        axes[0, 1].set(title="Episode return", xlabel="Online episode", ylabel="Return")
        axes[0, 1].legend()

        axes[1, 0].plot(x, [r["mc_loss"] for r in online], label="MC")
        axes[1, 0].plot(x, [r["rank_loss"] for r in online], label="Rank")
        axes[1, 0].set(title="Training losses", xlabel="Online episode", ylabel="Loss")
        axes[1, 0].legend()

        validation = [r for r in self.records if r.get("record_type") == "validation"]
        if validation:
            vx = [r["online_episode"] for r in validation]
            axes[1, 1].plot(vx, [r["success_rate"] for r in validation], marker="o", label="Success")
            axes[1, 1].plot(vx, [r["collision_rate"] for r in validation], marker="o", label="Collision")
            axes[1, 1].plot(vx, [r["timeout_rate"] for r in validation], marker="o", label="Timeout")
            axes[1, 1].set_ylim(-0.03, 1.03)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, "Development validation pending", ha="center", va="center")
        axes[1, 1].set(title="Development validation", xlabel="Online episode", ylabel="Rate")
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        tmp = self.curves_path.with_suffix(".png.tmp")
        fig.savefig(tmp, format="png", dpi=140)
        plt.close(fig)
        os.replace(tmp, self.curves_path)

    def close(self) -> None:
        self.plot()
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        self._metrics_handle.close()
        self._log_handle.close()
