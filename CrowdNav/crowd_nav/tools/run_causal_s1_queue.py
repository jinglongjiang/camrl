"""Run every CR-S1 stage serially with resumable stage markers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from crowd_nav.bayesian_brne.causal_s1_pipeline import STAGES


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--brne-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "queue_status.json"
    for index, stage in enumerate(STAGES, 1):
        marker = output / "stages" / f"{stage}.json"
        if marker.exists() and json.loads(marker.read_text()).get("status") == "PASS":
            print(f"[CR-S1-QUEUE] stage {index}/{len(STAGES)} already complete: {stage}", flush=True)
            continue
        command = [
            args.python, "-u", "-m", "crowd_nav.bayesian_brne.causal_s1_pipeline", stage,
            "--registry", args.registry, "--output", str(output), "--brne-root", args.brne_root,
        ]
        _write(status_path, {
            "schema_version": 1, "status": "running", "queue_pid": None,
            "stage_pid": None, "current_stage": stage, "stage_index": index,
            "stage_count": len(STAGES), "command": command, "updated_at": time.time(),
        })
        print(f"[CR-S1-QUEUE] stage {index}/{len(STAGES)} start: {stage}", flush=True)
        process = subprocess.Popen(command)
        status = json.loads(status_path.read_text())
        status["stage_pid"] = process.pid
        status["queue_pid"] = __import__("os").getpid()
        _write(status_path, status)
        return_code = process.wait()
        if return_code != 0:
            stop_status = "scientific_stop" if return_code == 20 else "error"
            _write(status_path, {
                "schema_version": 1, "status": stop_status, "queue_pid": __import__("os").getpid(),
                "stage_pid": process.pid, "current_stage": stage, "stage_index": index,
                "stage_count": len(STAGES), "return_code": return_code, "updated_at": time.time(),
            })
            print(f"[CR-S1-QUEUE] STOP stage={stage} return_code={return_code}", flush=True)
            raise SystemExit(return_code)
        print(f"[CR-S1-QUEUE] stage {index}/{len(STAGES)} done: {stage}", flush=True)
    _write(status_path, {
        "schema_version": 1, "status": "complete", "queue_pid": __import__("os").getpid(),
        "stage_pid": None, "current_stage": None, "stage_index": len(STAGES),
        "stage_count": len(STAGES), "updated_at": time.time(),
    })
    print("[CR-S1-QUEUE] all stages complete", flush=True)


if __name__ == "__main__":
    main()
