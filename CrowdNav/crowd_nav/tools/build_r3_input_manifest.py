#!/usr/bin/env python3
"""Freeze the R2 evidence R3 was built on (guide.md Order R3-0 / R3R-0
point 1): a read-only manifest listing the 21 bdvl_r2_final_20260807
checkpoints, the selection JSON, the 200-episode validation CSV, the
calibration JSON, and every hash needed to independently re-verify them.

This tool does not GUESS which files exist -- every path is passed in
explicitly and hashed; anything missing is a hard failure, never a
silent omission from the manifest. It also enforces guide.md 2532's
requirement that the buggy-selector output and the true best checkpoint
are never conflated: --selected-checkpoint (the SELECTOR's output,
known to have been seed93003/checkpoint_ep500.pth, 86.5% SR) and
--best-checkpoint (the true best available candidate, known to have
been seed93003/checkpoint_ep2000.pth, 99.5% SR) are required as two
DISTINCT, separately labeled arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

from crowd_nav.bayesian_dvl.config import BDVL_PRODUCTION_SOURCES  # noqa: E402
from crowd_nav.bayesian_dvl.provenance import atomic_write_json, sha256_of_file  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def _hash_or_fail(path: Path, label: str) -> str:
    if not path.is_file():
        raise SystemExit(f"R3 input manifest: required {label} not found: {path}")
    return sha256_of_file(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="bdvl_r2_final_20260807")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="all 21 R2 candidate checkpoint paths")
    parser.add_argument("--selection-json", required=True, help="select_bdvl_checkpoint.py's output report")
    parser.add_argument("--validation-csv", required=True, help="the 200-episode validation CSV")
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--selected-checkpoint", required=True, help="the SELECTOR's actual output (known buggy: seed93003/checkpoint_ep500.pth, 86.5%% SR)")
    parser.add_argument("--best-checkpoint", required=True, help="the true best available candidate (known: seed93003/checkpoint_ep2000.pth, 99.5%% SR)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.checkpoints) != 21:
        raise SystemExit(f"R3-0 requires exactly 21 R2 checkpoints; got {len(args.checkpoints)}")

    selected_path = _resolve(args.selected_checkpoint)
    best_path = _resolve(args.best_checkpoint)
    if selected_path.resolve() == best_path.resolve():
        raise SystemExit(
            "--selected-checkpoint and --best-checkpoint resolved to the same file -- "
            "guide.md 2532 requires these be recorded as two DISTINCT candidates "
            "(the buggy selector's output vs. the true best), never silently merged"
        )

    checkpoints = {}
    for raw_path in args.checkpoints:
        path = _resolve(raw_path)
        checkpoints[str(path)] = _hash_or_fail(path, f"checkpoint {path}")

    source_hashes = {}
    for relative in BDVL_PRODUCTION_SOURCES:
        source_path = PACKAGE_ROOT / relative
        source_hashes[relative] = _hash_or_fail(source_path, f"production source {relative}")

    manifest = {
        "run_name": args.run_name,
        "n_checkpoints": len(checkpoints),
        "checkpoint_sha256": checkpoints,
        "selection_json": str(_resolve(args.selection_json)),
        "selection_json_sha256": _hash_or_fail(_resolve(args.selection_json), "selection JSON"),
        "validation_csv": str(_resolve(args.validation_csv)),
        "validation_csv_sha256": _hash_or_fail(_resolve(args.validation_csv), "200-episode validation CSV"),
        "calibration_json": str(_resolve(args.calibration_json)),
        "calibration_json_sha256": _hash_or_fail(_resolve(args.calibration_json), "calibration JSON"),
        "selector_output_checkpoint": str(selected_path),
        "selector_output_checkpoint_sha256": _hash_or_fail(selected_path, "selector output checkpoint"),
        "true_best_checkpoint": str(best_path),
        "true_best_checkpoint_sha256": _hash_or_fail(best_path, "true best checkpoint"),
        "note": (
            "selector_output_checkpoint is what the PRE-R3-1 buggy selector actually "
            "wrote (known: seed93003/checkpoint_ep500.pth, 86.5% SR); true_best_checkpoint "
            "is the candidate the fixed R3-1 selection_rank_key would have chosen "
            "(known: seed93003/checkpoint_ep2000.pth, 99.5% SR). guide.md 2532: these "
            "must never be referred to interchangeably."
        ),
        "production_source_sha256": source_hashes,
    }
    output_path = _resolve(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing R3 input manifest: {output_path}")
    atomic_write_json(str(output_path), manifest)
    print(f"R3_INPUT_MANIFEST_DONE output={output_path} n_checkpoints={len(checkpoints)}")


if __name__ == "__main__":
    main()
