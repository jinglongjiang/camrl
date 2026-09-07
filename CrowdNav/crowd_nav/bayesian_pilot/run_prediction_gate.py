#!/usr/bin/env python3
"""Run the complete data-collection and prediction-gate pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, log_handle):
    rendered = " ".join(str(part) for part in command)
    print(f"[PILOT] {rendered}", flush=True)
    log_handle.write(f"\n$ {rendered}\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
    return_code = process.wait()
    log_handle.flush()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/bayesian_belief_pilot")
    parser.add_argument("--env_config", default="configs/env_gdbn.config")
    parser.add_argument("--seed", type=int, default=2407)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    gate_dir = output_dir / "gate"
    counts = (
        {"train": 18, "nominal": 8, "nonstationary": 10}
        if args.smoke
        else {"train": 300, "nominal": 80, "nonstationary": 120}
    )
    source_files = [
        Path(__file__).with_name("protocol.py"),
        Path(__file__).with_name("collect_nonstationary.py"),
        Path(__file__).with_name("evaluate_prediction_gate.py"),
        Path(__file__),
        Path(args.env_config).expanduser().resolve(),
    ]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "seed": args.seed,
        "smoke": args.smoke,
        "episode_counts": counts,
        "source_sha256": {
            str(path.resolve()): sha256(path.resolve())
            for path in source_files
        },
        "gate_rule": "All predeclared checks in prediction_gate.json must pass.",
        "next_step": "RL is prohibited unless the prediction gate passes.",
        "status": "running",
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log_path = output_dir / "pilot_controller.log"
    try:
        with log_path.open("a", encoding="utf-8") as log_handle:
            run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).with_name("collect_nonstationary.py")),
                    "--env_config",
                    args.env_config,
                    "--output_dir",
                    str(data_dir),
                    "--train_episodes",
                    str(counts["train"]),
                    "--nominal_test_episodes",
                    str(counts["nominal"]),
                    "--nonstationary_test_episodes",
                    str(counts["nonstationary"]),
                    "--seed",
                    str(args.seed),
                ],
                log_handle,
            )
            run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).with_name("evaluate_prediction_gate.py")),
                    "--data_dir",
                    str(data_dir),
                    "--output_dir",
                    str(gate_dir),
                    "--seed",
                    str(args.seed),
                ],
                log_handle,
            )
        gate = json.loads((gate_dir / "prediction_gate.json").read_text())
        manifest["status"] = "completed"
        manifest["gate_passed"] = bool(gate["passed"])
        manifest["decision"] = gate["decision"]
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
