#!/usr/bin/env python3
"""run_s1_queue.py: the single automated queue entry point for S1's heavy
stages (guide.md section 13, Order S1-0/S1-0R, 2026-08-04).

This script only calls ``crowd_nav.tools.s1_strict_gate``'s stages in
order -- it must never reimplement or duplicate any statistics/identity
logic (that all lives in ``crowd_nav.bayesian_brne.s1_protocol``, per
guide.md: "统计逻辑不得散落在CLI脚本里").

Order S1-0R fixes (independent audit findings):
  - The repo root is ALWAYS derived from this file's own on-disk location
    (``s1_protocol.repo_root()``), never a hardcoded host path -- the
    original version hardcoded one developer machine's absolute checkout
    path, which does not exist on the 4090 training host at all.
  - ``--output-dir`` must resolve to EXACTLY ``<repo_root>/runs/
    bayesian_brne/<registry.experiment_id>`` -- any other value fails
    closed immediately, rather than silently writing status/pid files
    somewhere ``s1_strict_gate``'s own per-stage manifests are not.
  - ``status.json`` is written ONLY via ``s1_protocol.update_status_atomic``
    (never this script's own ad-hoc JSON write), so the queue's own
    progress and each stage's per-run result share one schema and merge
    instead of clobbering each other.

The full frozen stage sequence is implemented. Resume skips only stages
recorded complete under the same registry identity; scientific NO_GO or
INCONCLUSIVE stops cleanly before controls/audit are exposed.

This is the ONLY sanctioned automated entry point for S1's 4090 run
(guide.md section 8/S1-7): it must be launched as a host-visible `nohup`
process, never inside a CC-invisible sandbox background task.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from crowd_nav.bayesian_brne import s1_protocol as sp

QUEUE_STAGES = (
    "preflight",
    "collect_necessity",
    "fit_ac",
    "select_k",
    "fit_controls",
    "necessity",
    "collect_audit",
    "audit",
    "promote",
)


def _run_stage(registry: str, stage: str) -> int:
    cmd = [
        sys.executable, "-m", "crowd_nav.tools.s1_strict_gate",
        "--registry", registry, "--stage", stage, "--invocation-mode", "queue",
    ]
    print(f"[run_s1_queue] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(sp.repo_root()))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stop-after", default=None, choices=QUEUE_STAGES,
        help="Run only up through this stage, then stop (default: run until the first not-yet-implemented stage).",
    )
    args = parser.parse_args()

    repo_root = sp.repo_root()
    registry = sp.load_registry(args.registry)
    registry_hash = sp.registry_content_sha256(registry)
    expected_output_dir = (repo_root / "runs" / "bayesian_brne" / registry["experiment_id"]).resolve()
    given_output_dir = Path(args.output_dir).resolve()
    if given_output_dir != expected_output_dir:
        print(
            f"[run_s1_queue] --output-dir={given_output_dir} does not match the registry-derived "
            f"experiment output root {expected_output_dir} -- refusing to run: a mismatched output "
            "directory would let the queue's status/pid live somewhere s1_strict_gate's own "
            "per-stage manifests do not, silently splitting one experiment's state across two places.",
            file=sys.stderr,
        )
        return 2

    out_dir = expected_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pid").write_text(str(os.getpid()) + "\n")

    stages_to_run = QUEUE_STAGES
    if args.stop_after is not None:
        stages_to_run = QUEUE_STAGES[: QUEUE_STAGES.index(args.stop_after) + 1]

    # Order S1-0R-A (A4): queue_started_at is set ONCE here, before the
    # loop -- the original version recomputed "started_at" on every stage
    # iteration, so a queue that ran 3 stages ended up with a timestamp
    # only describing its LAST stage, not when the queue itself began.
    status_path = out_dir / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    completed_stages = list(previous.get("completed_stages", []))
    queue_started_at = previous.get("queue_started_at") or time.strftime("%Y-%m-%d %H:%M:%S")
    for stage in stages_to_run:
        if stage in completed_stages:
            print(f"[run_s1_queue] resume: stage={stage!r} already completed; skipping", flush=True)
            continue
        sp.update_status_atomic(
            str(out_dir), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
            queue_pid=os.getpid(), queue_status="RUNNING", queue_started_at=queue_started_at,
            current_stage=stage, completed_stages=completed_stages,
        )
        rc = _run_stage(args.registry, stage)
        if rc != 0:
            print(f"[run_s1_queue] stage={stage!r} returned {rc}; stopping queue (never skip ahead).", file=sys.stderr)
            sp.update_status_atomic(
                str(out_dir), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
                queue_pid=None, queue_status="STOPPED", current_stage=stage, returncode=rc,
                completed_stages=completed_stages,
            )
            return rc
        completed_stages = completed_stages + [stage]
        # Scientific NO_GO/INCONCLUSIVE is a valid completed experiment,
        # not an execution crash. Stop before controls/audit are exposed.
        if stage == "select_k":
            report = json.loads((out_dir / "selection_report.json").read_text())
            if report.get("status") != "SELECTED":
                print(f"[run_s1_queue] selection ended with {report.get('status')}; stopping by protocol", flush=True)
                sp.update_status_atomic(
                    str(out_dir), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
                    queue_pid=None, queue_status="COMPLETED_SCIENTIFIC_STOP", current_stage=stage,
                    returncode=0, completed_stages=completed_stages,
                )
                return 0
        if stage == "necessity":
            report = json.loads((out_dir / "necessity_report.json").read_text())
            if report.get("status") != "PASS":
                print(f"[run_s1_queue] necessity ended with {report.get('status')}; audit remains locked", flush=True)
                sp.update_status_atomic(
                    str(out_dir), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
                    queue_pid=None, queue_status="COMPLETED_SCIENTIFIC_STOP", current_stage=stage,
                    returncode=0, completed_stages=completed_stages,
                )
                return 0

    sp.update_status_atomic(
        str(out_dir), experiment_id=registry["experiment_id"], registry_sha256=registry_hash,
        queue_pid=None, queue_status="COMPLETED_REQUESTED_STAGES", current_stage=stages_to_run[-1],
        returncode=0, completed_stages=completed_stages,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
