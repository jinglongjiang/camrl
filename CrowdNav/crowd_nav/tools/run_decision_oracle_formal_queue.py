#!/usr/bin/env python3
"""Monitor frozen decision-oracle shards and run merge plus D4 gate exactly once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _write_status(path: Path, status: str, **extra) -> None:
    payload = {"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def _live_audit_processes(base: Path) -> list[int]:
    needle = "crowd_nav.tools.audit_candidate_decision_oracle"
    roots = {str(base)}
    try:
        roots.add(str(base.relative_to(Path.cwd())))
    except ValueError:
        pass
    live = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            cmdline = (entry / "cmdline").read_bytes().decode(errors="ignore").replace("\x00", " ")
        except OSError:
            continue
        if needle in cmdline and any(root in cmdline for root in roots):
            live.append(int(entry.name))
    return sorted(live)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--registry", default="crowd_nav/configs/decision_oracle_registry.json")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-polls", type=int, default=720)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    status_path = base / "queue_status.json"
    shard_dirs = sorted(path for path in base.glob("shard_*") if path.is_dir())
    expected = 10
    if len(shard_dirs) != expected:
        _write_status(status_path, "failed", reason="wrong_shard_count", found=len(shard_dirs), expected=expected)
        raise SystemExit(21)
    _write_status(status_path, "monitoring", shards=[str(path) for path in shard_dirs])
    for poll in range(args.max_polls):
        completed = sorted(path for path in shard_dirs if (path / "status.json").exists())
        if len(completed) == expected:
            _write_status(status_path, "merging", completed=len(completed))
            merge_cmd = [
                os.environ.get("PYTHON", "python3"), "-m", "crowd_nav.tools.merge_decision_oracle_shards",
                "--registry", args.registry, "--output", str(base / "merged"),
            ]
            merge_cmd.extend(item for path in shard_dirs for item in ("--shard", str(path)))
            merge = subprocess.run(merge_cmd, cwd=Path.cwd(), check=False)
            if merge.returncode != 0:
                _write_status(status_path, "failed", reason="merge_failed", returncode=merge.returncode)
                raise SystemExit(merge.returncode)
            gate_cmd = [
                os.environ.get("PYTHON", "python3"), "-m", "crowd_nav.tools.evaluate_decision_oracle_gate",
                "--registry", args.registry, "--input", str(base / "merged"),
                "--output", str(base / "merged" / "gate_result.json"),
            ]
            gate = subprocess.run(gate_cmd, cwd=Path.cwd(), check=False)
            _write_status(status_path, "completed", completed=len(completed), gate_returncode=gate.returncode)
            raise SystemExit(0)
        live = _live_audit_processes(base)
        if not live:
            _write_status(status_path, "failed", reason="shard_failure_or_incomplete", completed=len(completed))
            raise SystemExit(21)
        if poll == 0 or poll % 5 == 0:
            _write_status(status_path, "monitoring", completed=len(completed), live_pids=live, poll=poll)
        time.sleep(float(args.poll_seconds))
    _write_status(status_path, "failed", reason="monitor_timeout", polls=args.max_polls)
    raise SystemExit(22)


if __name__ == "__main__":
    main()
