#!/usr/bin/env python3
"""Migrate the frozen R0 engineering artifact to the R3 schema.

This is intentionally a one-way engineering migration: it reads the old
schema without trusting its model card, reuses only the numeric arrays, and
writes a new atomic R3 artifact with explicit engineering placeholders for
the provenance fields that were not recorded by the old build.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact


def migrate(source: Path, destination: Path) -> None:
    data = np.load(source / "arhmm.npz")
    old_card = json.loads((source / "model_card.json").read_text(encoding="utf-8"))
    K = int(data["K"])
    artifact = ARHMMArtifact(
        K=K,
        A=[data[f"A_{k}"] for k in range(K)],
        B=[data[f"B_{k}"] for k in range(K)],
        C=[data[f"C_{k}"] for k in range(K)],
        d=[data[f"d_{k}"] for k in range(K)],
        Q=[data[f"Q_{k}"] for k in range(K)],
        Pi=data["Pi"],
        initial_distribution=data["initial_distribution"],
        dt=float(data["dt"]),
        model_card={
            "purpose": old_card.get("purpose", "migrated engineering-only artifact"),
            "migrated_from_schema_version": old_card.get("schema_version"),
            "legacy_model_card": {
                "fit_data": old_card.get("fit_data"),
                "k_candidates_fit": old_card.get("k_candidates_fit"),
            },
        },
    )
    convergence = old_card.get("convergence", {"converged": False, "reason": "legacy card missing"})
    artifact.save(str(destination), tier="engineering_only", convergence=convergence)
    print(json.dumps({
        "source": str(source),
        "destination": str(destination),
        "schema_version": artifact.model_card["schema_version"],
        "content_sha256": artifact.model_card["content_sha256"],
        "tier": artifact.model_card["tier"],
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    migrate(Path(args.source), Path(args.destination))


if __name__ == "__main__":
    main()
