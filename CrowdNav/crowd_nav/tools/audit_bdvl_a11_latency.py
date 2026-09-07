#!/usr/bin/env python3
"""BDVL A11 latency benchmark (guide.md 9/A11 hard gate):
80 actions x 32 world samples x 64 IQN quantiles, 20 humans, formal
config. Hard thresholds: p95 <= 250ms, p99 <= 400ms -- guide.md
specifies this MUST be measured on the 4090; this script can run
anywhere, but a result from this local machine is NOT a substitute for
the formal 4090 measurement and must be labeled as such.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

import torch  # noqa: E402

from crowd_nav.bayesian_dvl.config import ActionGridSpec, FROZEN_VALUES  # noqa: E402
from crowd_nav.bayesian_dvl.world_model import Track, fit_sbk_hmm  # noqa: E402
from crowd_nav.bayesian_dvl.set_encoder import ActionEncoder, SetEncoder  # noqa: E402
from crowd_nav.bayesian_dvl.iqn import IQNValueNetwork  # noqa: E402
from crowd_nav.bayesian_dvl.transition import RewardConfig  # noqa: E402
from crowd_nav.bayesian_dvl.policy import BDVLPolicy  # noqa: E402
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import sha256_of_file  # noqa: E402


def _synthetic_artifact():
    dt = FROZEN_VALUES["dt"]

    def track(speed0, accel, omega):
        pos = [np.array([0.0, 0.0])]
        speed, heading = speed0, 0.0
        for _ in range(30):
            speed = max(speed + accel * dt, 0.0)
            heading += omega * dt
            vel = speed * np.array([np.cos(heading), np.sin(heading)])
            pos.append(pos[-1] + vel * dt)
        return Track(positions=np.array(pos), dt=dt)

    tracks = [track(1.0, 0.0, 0.0), track(0.3, 0.8, 0.0), track(1.5, -0.8, 0.0), track(1.0, 0.0, 1.0), track(1.0, 0.0, -1.0)]
    return fit_sbk_hmm(tracks, train_data_sha256="a11_latency_fixture", max_iterations=15)


def run(n_trials: int, device: str, env_config_path: Path, warmup: int = 10, n_humans: int = 20) -> dict:
    if n_trials < 20:
        raise ValueError("formal latency benchmark requires at least 20 timed trials")
    if warmup < 1:
        raise ValueError("warmup must be positive")
    if n_humans != 20:
        raise ValueError("A11 hard gate is frozen at 20 humans")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    env_config_path = env_config_path.resolve()
    expected_config = PACKAGE_ROOT / "crowd_nav" / "configs" / "env_bayesian_dvl.config"
    if env_config_path != expected_config.resolve():
        raise ValueError(f"A11 must use the frozen BDVL environment config: {expected_config}")
    action_grid = ActionGridSpec.from_env_config(str(env_config_path))
    action_table = action_grid.build_action_table()
    if len(action_table) != 80:
        raise RuntimeError(f"frozen BDVL action grid must contain 80 actions, got {len(action_table)}")
    artifact = _synthetic_artifact()
    encoder = SetEncoder().to(device).eval()
    action_encoder = ActionEncoder().to(device).eval()
    net = IQNValueNetwork(state_embedding_dim=128, action_embedding_dim=action_encoder.embed_dim).to(device).eval()
    reward_config = RewardConfig(
        success_reward=FROZEN_VALUES["success_reward"], collision_penalty=FROZEN_VALUES["collision_penalty"],
        timeout_penalty=FROZEN_VALUES["timeout_penalty"], progress_reward=FROZEN_VALUES["progress_reward"],
        time_penalty=FROZEN_VALUES["time_penalty"], stand_penalty=FROZEN_VALUES["stand_penalty"],
        stand_speed_threshold=FROZEN_VALUES["stand_speed_threshold"],
        discomfort_distance=FROZEN_VALUES["discomfort_distance"],
        discomfort_penalty_factor=FROZEN_VALUES["discomfort_penalty_factor"],
    )
    policy = BDVLPolicy(
        artifact=artifact, set_encoder=encoder, value_network=net, action_encoder=action_encoder, action_table=action_table,
        reward_config=reward_config, dt=FROZEN_VALUES["dt"], time_limit=FROZEN_VALUES["time_limit"],
        max_human_speed=FROZEN_VALUES["max_human_speed"], cvar_alpha=FROZEN_VALUES["cvar_alpha"],
        n_world_samples=FROZEN_VALUES["world_samples_formal"], n_iqn_quantiles=FROZEN_VALUES["iqn_quantiles_formal"],
    )

    rng = np.random.default_rng(0)
    robot = RobotObservation(px=0.0, py=0.0, vx=0.0, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0, theta=0.0)
    humans = [
        HumanObservation(track_id=i, px=float(rng.uniform(-4, 4)), py=float(rng.uniform(-4, 4)),
                          vx=float(rng.uniform(-0.5, 0.5)), vy=float(rng.uniform(-0.5, 0.5)), radius=0.3)
        for i in range(n_humans)  # formal config: 20 humans
    ]

    dt = FROZEN_VALUES["dt"]
    step_counter = 0

    # Warmup (JIT/allocator warmup should not count toward latency stats).
    for _ in range(warmup):
        policy.decide(robot, humans, global_time=1.0 + step_counter * dt, suite_seed=1, episode_seed=1)
        step_counter += 1

    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []
    for _ in range(n_trials):
        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        policy.decide(robot, humans, global_time=1.0 + step_counter * dt, suite_seed=1, episode_seed=1)
        if is_cuda:
            torch.cuda.synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        step_counter += 1

    latencies_ms = np.array(latencies_ms)
    if is_cuda:
        torch.cuda.synchronize(device)
        peak_memory_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_memory_reserved = int(torch.cuda.max_memory_reserved(device))
        gpu_name = torch.cuda.get_device_name(device)
    else:
        peak_memory_allocated = None
        peak_memory_reserved = None
        gpu_name = None
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    source_files = [
        Path(__file__),
        PACKAGE_ROOT / "crowd_nav" / "bayesian_dvl" / "policy.py",
        PACKAGE_ROOT / "crowd_nav" / "bayesian_dvl" / "rollout.py",
        PACKAGE_ROOT / "crowd_nav" / "bayesian_dvl" / "set_encoder.py",
        PACKAGE_ROOT / "crowd_nav" / "bayesian_dvl" / "iqn.py",
        PACKAGE_ROOT / "crowd_nav" / "bayesian_dvl" / "transition.py",
    ]
    source_sha256 = {
        str(path.resolve().relative_to(PACKAGE_ROOT.resolve())): sha256_of_file(str(path))
        for path in source_files
    }
    return {
        "schema_version": 2,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": device,
        "gpu_name": gpu_name,
        "n_trials": n_trials,
        "warmup": warmup,
        "config": {
            "n_actions": len(action_table), "n_humans": 20,
            "world_samples": FROZEN_VALUES["world_samples_formal"],
            "iqn_quantiles": FROZEN_VALUES["iqn_quantiles_formal"],
        },
        "mean_ms": float(latencies_ms.mean()),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "max_ms": float(latencies_ms.max()),
        "peak_memory_allocated_bytes": peak_memory_allocated,
        "peak_memory_reserved_bytes": peak_memory_reserved,
        "env_config_sha256": sha256_of_file(str(env_config_path)),
        "action_grid_sha256": action_grid.table_hash(),
        "source_sha256": source_sha256,
        "hard_gate": {"p95_threshold_ms": 250, "p99_threshold_ms": 400, "gate_applies_to": "4090 only (guide.md A11)"},
        "hard_gate_pass": bool(p95_ms <= 250.0 and p99_ms <= 400.0),
        "timing_contract": "CUDA synchronize before and after every timed decision; warmup excluded",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--env-config", default="crowd_nav/configs/env_bayesian_dvl.config")
    parser.add_argument("--output", default="runs/bayesian_dvl/a11_latency/result.json")
    args = parser.parse_args()

    env_config_path = Path(args.env_config)
    if not env_config_path.is_absolute():
        env_config_path = PACKAGE_ROOT / env_config_path
    result = run(args.n_trials, args.device, env_config_path, warmup=args.warmup)
    output_path = PACKAGE_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
