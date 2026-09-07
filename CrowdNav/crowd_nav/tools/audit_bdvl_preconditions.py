#!/usr/bin/env python3
"""BDVL P0 precondition audit (guide.md section 9, stage P0).

P0-1 (implemented here): freeze the inherited 80-action grid's *actual*
production output into ``action_table.json`` with source/table hashes,
and prove the result is independent of the caller's working directory.

P0-2 (local coverage: branch all 80 actions from real conflict
snapshots) and P0-3 (frozen-Mamba-VL closed-loop witness) are NOT
implemented by this script -- they need real environment rollouts
(P0-2) and a loaded trained checkpoint on GPU (P0-3), out of scope for
this pass. Do not report P0 as complete without them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256_of_obj(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_package_root(start: Path) -> Path:
    """Locate the CrowdNav package root by its ``setup.py``, not by the
    nearest ``.git``: this repo's actual git root is one level up
    (``camrl/.git``), with a sibling backup directory
    (``CrowdNav_bayesian_replaced_.../setup.py``) that also matches a
    naive "nearest setup.py" walk from certain CWDs, so we additionally
    require the candidate to contain a ``crowd_nav`` subpackage."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root (setup.py + crowd_nav/) above {start}")


def run_p0_1(env_config_relpath: str, output_relpath: str) -> dict:
    """Build and freeze action_table.json. Paths are resolved against
    the git root, never the caller's CWD, so this is CWD-independent
    by construction (guide.md P0-1's anti-fallback requirement)."""
    # Importing here (not at module load) keeps this script runnable
    # via `python3 path/to/audit_bdvl_preconditions.py` from any CWD
    # without requiring the package to already be on sys.path.
    git_root = _find_package_root(Path(__file__).parent)
    sys.path.insert(0, str(git_root))
    from crowd_nav.bayesian_dvl.config import ActionGridSpec  # noqa: E402

    env_config_path = git_root / env_config_relpath
    grid = ActionGridSpec.from_env_config(str(env_config_path))
    table = grid.build_action_table()

    payload = {
        "n_speeds": grid.n_speeds,
        "n_headings": grid.n_headings,
        "sampling": grid.sampling,
        "include_stop": grid.include_stop,
        "n_actions": grid.n_actions,
        "index_order": "heading_idx * n_speeds + speed_idx",
        "action_table": [[round(vx, 15), round(vy, 15)] for vx, vy in table],
        "source_config_relpath": env_config_relpath,
        "source_config_sha256": _sha256_of_file(env_config_path),
    }
    payload["table_sha256"] = _sha256_of_obj(payload["action_table"])

    output_path = git_root / output_relpath
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify_cwd_independence(env_config_relpath: str, output_relpath: str) -> None:
    """Re-invoke this script's P0-1 path from three different CWDs and
    assert byte-identical action_table.json content each time."""
    git_root = _find_package_root(Path(__file__).parent)
    script_path = Path(__file__).resolve()
    cwds = [git_root, git_root / "crowd_nav", Path("/tmp")]

    hashes = []
    for cwd in cwds:
        cwd.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(script_path), "p0-1",
             "--env-config", env_config_relpath, "--output", output_relpath],
            cwd=str(cwd), capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"P0-1 run from CWD={cwd} failed:\nstdout={result.stdout}\nstderr={result.stderr}"
            )
        data = json.loads((git_root / output_relpath).read_text())
        hashes.append(data["table_sha256"])
        print(f"CWD={cwd}: table_sha256={data['table_sha256'][:16]}...")

    if len(set(hashes)) != 1:
        raise SystemExit(f"action_table.json differs across CWDs: {hashes}")
    print(f"CWD_INDEPENDENCE_OK all {len(cwds)} runs produced identical table_sha256={hashes[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p0_1 = sub.add_parser("p0-1", help="freeze action_table.json")
    p0_1.add_argument("--env-config", default="crowd_nav/configs/env.config")
    p0_1.add_argument("--output", default="crowd_nav/configs/bdvl_action_table.json")

    verify = sub.add_parser("verify-cwd-independence", help="run p0-1 from 3 CWDs, compare hashes")
    verify.add_argument("--env-config", default="crowd_nav/configs/env.config")
    verify.add_argument("--output", default="crowd_nav/configs/bdvl_action_table.json")

    args = parser.parse_args()
    if args.stage == "p0-1":
        payload = run_p0_1(args.env_config, args.output)
        print(json.dumps({"n_actions": payload["n_actions"], "table_sha256": payload["table_sha256"]}, indent=2))
    elif args.stage == "verify-cwd-independence":
        verify_cwd_independence(args.env_config, args.output)


if __name__ == "__main__":
    main()
