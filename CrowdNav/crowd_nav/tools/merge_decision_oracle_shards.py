#!/usr/bin/env python3
"""Merge complete frozen-suite-seed decision-oracle shards without re-fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CSV_NAMES = ("state_records.csv", "candidate_records.csv", "oracle_episode_records.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty merged CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="crowd_nav/configs/decision_oracle_registry.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", action="append", required=True,
                        help="One completed audit shard directory; repeat once per frozen suite seed.")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text())
    expected_seeds = {int(value) for value in registry["data_protocol"]["suite_seeds"]}
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    shards = [Path(value).resolve() for value in args.shard]
    expected_splits = set(str(value) for value in registry["data_protocol"]["splits"])
    expected_pairs = {(seed, split) for seed in expected_seeds for split in expected_splits}
    if len(shards) < len(expected_seeds):
        raise ValueError(f"at least {len(expected_seeds)} shards are required, got {len(shards)}")
    seen_pairs: set[tuple[int, str]] = set()
    shard_records = []
    all_rows: dict[str, list[dict]] = {name: [] for name in CSV_NAMES}
    fieldnames: dict[str, list[str]] = {}
    expected_registry_sha = _sha256(registry_path)

    for shard in shards:
        manifest_path = shard / "manifest.json"
        status_path = shard / "status.json"
        if not manifest_path.exists() or not status_path.exists():
            raise RuntimeError(f"incomplete shard: {shard}")
        status = json.loads(status_path.read_text())
        if status.get("status") != "completed":
            raise RuntimeError(f"shard is not completed: {shard}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("registry_sha256") != expected_registry_sha:
            raise RuntimeError(f"registry drift in shard: {shard}")
        shard_pairs: set[tuple[int, str]] = set()
        for name in CSV_NAMES:
            path = shard / name
            if not path.exists() or _sha256(path) != manifest.get("outputs_sha256", {}).get(name):
                raise RuntimeError(f"output hash mismatch in shard: {path}")
            names, rows = _read(path)
            if name not in fieldnames:
                fieldnames[name] = names
            elif fieldnames[name] != names:
                raise RuntimeError(f"CSV schema mismatch for {name}: {shard}")
            all_rows[name].extend(rows)
            if name in ("state_records.csv", "oracle_episode_records.csv"):
                shard_pairs.update((int(row["suite_seed"]), str(row["split"])) for row in rows)
        if not shard_pairs:
            raise RuntimeError(f"shard has no identifiable suite seed/split rows: {shard}")
        shard_seeds = {seed for seed, _ in shard_pairs}
        if len(shard_seeds) != 1:
            raise RuntimeError(f"each shard must contain exactly one suite seed: {shard} -> {shard_pairs}")
        if not shard_pairs <= expected_pairs:
            raise RuntimeError(f"shard contains non-frozen suite seed/split pair: {shard} -> {shard_pairs}")
        if seen_pairs & shard_pairs:
            raise RuntimeError(f"duplicate suite seed/split across shards: {seen_pairs & shard_pairs}")
        seen_pairs.update(shard_pairs)
        shard_records.append({"path": str(shard), "suite_seed": sorted(shard_seeds)[0],
                              "splits": sorted({split for _, split in shard_pairs}),
                              "manifest_sha256": _sha256(manifest_path)})

    if seen_pairs != expected_pairs:
        missing = sorted(expected_pairs - seen_pairs)
        extra = sorted(seen_pairs - expected_pairs)
        raise RuntimeError(f"shard pairs do not match frozen registry: missing={missing}, extra={extra}")
    output.mkdir(parents=True, exist_ok=True)
    for name in CSV_NAMES:
        _write(output / name, fieldnames[name], all_rows[name])
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "merge_type": "frozen_suite_seed_shards",
        "registry_sha256": expected_registry_sha,
        "shards": shard_records,
        "suite_seeds": sorted(expected_seeds),
        "splits": sorted(expected_splits),
        "outputs_sha256": {name: _sha256(output / name) for name in CSV_NAMES},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    status = {"status": "completed", "states": len(all_rows["state_records.csv"]),
              "candidates": len(all_rows["candidate_records.csv"]),
              "oracle_episodes": len(all_rows["oracle_episode_records.csv"])}
    (output / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
