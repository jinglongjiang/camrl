#!/usr/bin/env python3
"""Run the frozen 128-batch ranking/MC gradient audit on real ORCA data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_nav.bayesian_dvl.config import ActionGridSpec, FROZEN_VALUES, build_frozen_registry, derive_return_bounds  # noqa: E402
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork  # noqa: E402
from crowd_nav.bayesian_dvl.replay import DemoOnlineReplay  # noqa: E402
from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, SetEncoder  # noqa: E402
from crowd_nav.bayesian_dvl.transition import RewardConfig  # noqa: E402
from crowd_nav.bayesian_dvl.trainer import stage1_train_step  # noqa: E402
from crowd_nav.bayesian_dvl.world_model import SBKHMMArtifact  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, build_run_manifest  # noqa: E402
from crowd_nav.tools.train_bdvl import _collect_orca_episode, _make_reward_config, _seed_everything  # noqa: E402
from crowd_nav.bayesian_dvl.ranking import derive_action_equivalence_tolerance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=int(FROZEN_VALUES["gradient_audit_batches"]))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ranking-batch-size", type=int, choices=(8, 16, 32), default=int(FROZEN_VALUES["ranking_batch_size"]))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=96001)
    args = parser.parse_args()
    if args.batches != int(FROZEN_VALUES["gradient_audit_batches"]):
        raise SystemExit("gradient audit batch count is frozen at 128")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    _seed_everything(args.seed)
    artifact = SBKHMMArtifact.load(str(Path(args.artifact_path)), expect_tier="production")
    action_table = ActionGridSpec.from_env_config(str(PACKAGE_ROOT / "crowd_nav/configs/env_bayesian_dvl.config")).build_action_table()
    registry = build_frozen_registry(str(PACKAGE_ROOT / "crowd_nav/configs/env_bayesian_dvl.config")).to_json_dict()
    audit_spec = registry["gradient_audit"]
    if int(audit_spec["required_batches"]) != args.batches:
        raise SystemExit("gradient audit batch count does not match the frozen registry spec")
    if float(audit_spec["ratio_min"]) != float(FROZEN_VALUES["rank_gradient_ratio_min"]) or float(audit_spec["ratio_max"]) != float(FROZEN_VALUES["rank_gradient_ratio_max"]):
        raise SystemExit("gradient audit ratio bounds drifted from the frozen registry spec")
    if args.ranking_batch_size not in tuple(audit_spec["ranking_batch_sizes_to_compare"]):
        raise SystemExit("ranking batch size is not in the frozen comparison set")
    v_min, v_max = derive_return_bounds(FROZEN_VALUES)
    encoder = SetEncoder().to(device)
    action_encoder = ActionEncoder().to(device)
    value_network = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim, v_min=v_min, v_max=v_max).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(value_network.parameters()) + list(action_encoder.parameters()), lr=float(FROZEN_VALUES["learning_rate"]))
    replay = DemoOnlineReplay(int(FROZEN_VALUES["replay_capacity"]), 50000, float(FROZEN_VALUES["demo_sample_ratio"]))
    tolerance = derive_action_equivalence_tolerance(action_table)
    ratios = []
    grad_rows = []
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    for batch_index in range(args.batches):
        samples, _ = _collect_orca_episode(
            PACKAGE_ROOT / "crowd_nav/configs/env_bayesian_dvl.config", action_table, artifact,
            args.seed + batch_index, "nominal", tolerance,
        )
        for sample in samples:
            replay.demo.add(sample)
        batch, _, _ = replay.sample_batch(args.batch_size, __import__("random").Random(args.seed + batch_index))
        result = stage1_train_step(
            encoder, value_network, action_encoder, artifact, action_table, _make_reward_config(),
            dt=float(FROZEN_VALUES["dt"]), time_limit=float(FROZEN_VALUES["time_limit"]),
            max_human_speed=float(FROZEN_VALUES["max_human_speed"]), n_world_samples=int(FROZEN_VALUES["world_samples_train"]),
            n_iqn_quantiles=int(FROZEN_VALUES["iqn_train_quantiles"]), posterior_source="full", device=device,
            gamma=float(FROZEN_VALUES["gamma"]), demo_batch=batch,
            ranking_margin=float(FROZEN_VALUES["ranking_margin"]), lambda_rank=float(FROZEN_VALUES["lambda_rank"]),
            n_train_quantiles=int(FROZEN_VALUES["iqn_train_quantiles"]), optimizer=optimizer,
            ranking_batch_size=args.ranking_batch_size,
        )
        if result.aborted:
            raise SystemExit(f"gradient audit aborted at batch {batch_index}: {result.abort_reason}")
        ratios.append(float(result.gradient_ratio))
        grad_rows.append({"batch": batch_index + 1, "mc_grad_norm": result.mc_grad_norm, "rank_grad_norm": result.rank_grad_norm, "weighted_rank_grad_norm": result.weighted_rank_grad_norm, "gradient_ratio": result.gradient_ratio})
        if (batch_index + 1) % 16 == 0:
            print(f"R3R2_PROGRESS batch={batch_index + 1}/{args.batches} ratio={result.gradient_ratio:.4f}", flush=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        peak_memory_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_memory_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_memory_allocated = None
        peak_memory_reserved = None
    elapsed_seconds = time.perf_counter() - started_at
    report = {
        "schema": "bdvl_r3r2_gradient_audit_v1", "batches": args.batches,
        "batch_size": args.batch_size, "ranking_batch_size": args.ranking_batch_size,
        "seed": args.seed, "device": device,
        "elapsed_seconds": float(elapsed_seconds),
        "batches_per_second": float(args.batches / elapsed_seconds),
        "peak_memory_allocated_bytes": peak_memory_allocated,
        "peak_memory_reserved_bytes": peak_memory_reserved,
        "ratios": {"min": float(np.min(ratios)), "median": float(np.median(ratios)), "max": float(np.max(ratios))},
        "frozen_bounds": {"min": FROZEN_VALUES["rank_gradient_ratio_min"], "max": FROZEN_VALUES["rank_gradient_ratio_max"]},
        "gradient_audit_spec": audit_spec,
        "rows": grad_rows, "registry_content_sha256": registry["content_sha256"],
        "artifact_sha256": artifact.content_sha256(),
    }
    output = Path(args.output) if Path(args.output).is_absolute() else PACKAGE_ROOT / args.output
    atomic_write_json(str(output), report)
    print(f"R3R2_GRADIENT_AUDIT_PASS report={output} median={np.median(ratios):.4f}")


if __name__ == "__main__":
    main()
