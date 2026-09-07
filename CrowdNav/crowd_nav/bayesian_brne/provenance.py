"""Hashing, manifest construction, and artifact-hash verification.

Every SM-BRNE artifact directory (``runs/bayesian_brne/<run_id>/``) carries a
``manifest.json`` recording exactly what produced it (guide.md 4.4). Loading
an artifact whose recorded hashes do not match the files on disk is refused
by default -- silent drift between "the model that was fit" and "the model
that gets loaded" is exactly the kind of provenance gap this module exists
to prevent (see belief_mdp/'s and bayesian_decision_gate/'s own hashing
history in this project).
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


class ArtifactMismatchError(RuntimeError):
    """Raised when a loaded artifact's hashes do not match its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path) -> str:
    digest = hashlib.sha256()
    path = Path(path)
    if not path.exists():
        return ""
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            digest.update(str(file_path.relative_to(path)).encode("utf-8"))
            digest.update(sha256_file(file_path).encode("utf-8"))
    return digest.hexdigest()


def git_head(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return ""


def git_dirty_diff_hash(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "diff", "--binary"], cwd=str(repo_dir), capture_output=True, text=True, check=True
        )
        return hashlib.sha256(out.stdout.encode("utf-8")).hexdigest()
    except Exception:
        return ""


def package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "not_installed"


def build_manifest(
    *,
    project_dir: Path,
    brne_root: Path,
    brne_commit: str,
    data_files: Dict[str, Path],
    model_files: Dict[str, Path],
    config_files: Dict[str, Path],
    suite_seed: int,
    episode_seed: Optional[int],
    extra: Optional[dict] = None,
) -> dict:
    manifest = {
        "project_git_head": git_head(project_dir),
        "project_git_dirty_diff_sha256": git_dirty_diff_hash(project_dir),
        "brne_root": str(brne_root),
        "brne_commit": brne_commit,
        "brne_license_sha256": sha256_file(Path(brne_root) / "LICENSE") if (Path(brne_root) / "LICENSE").exists() else "",
        "data_files_sha256": {name: sha256_file(Path(p)) for name, p in data_files.items()},
        "model_files_sha256": {name: sha256_file(Path(p)) for name, p in model_files.items()},
        "config_files_sha256": {name: sha256_file(Path(p)) for name, p in config_files.items()},
        "python_version": sys.version,
        "numpy_version": package_version("numpy"),
        "numba_version": package_version("numba"),
        "rvo2_version": package_version("rvo2"),
        "suite_seed": suite_seed,
        "episode_seed": episode_seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def verify_artifact_hashes(manifest: dict, model_files: Dict[str, Path], allow_override: bool = False) -> None:
    """Raise ArtifactMismatchError unless every file in ``model_files``
    matches the hash recorded in ``manifest['model_files_sha256']`` --
    unless ``allow_override`` is set, in which case mismatches are printed
    as warnings instead of raised (guide.md 4.4)."""
    recorded = manifest.get("model_files_sha256", {})
    for name, path in model_files.items():
        path = Path(path)
        if name not in recorded:
            message = f"[PROVENANCE] {name} not present in manifest's recorded hashes"
            if allow_override:
                print(f"WARNING: {message}")
                continue
            raise ArtifactMismatchError(message)
        if not path.exists():
            message = f"[PROVENANCE] {name} missing on disk: {path}"
            if allow_override:
                print(f"WARNING: {message}")
                continue
            raise ArtifactMismatchError(message)
        actual = sha256_file(path)
        if actual != recorded[name]:
            message = f"[PROVENANCE] {name} hash mismatch: manifest={recorded[name][:16]}... actual={actual[:16]}..."
            if allow_override:
                print(f"WARNING: {message}")
                continue
            raise ArtifactMismatchError(message)
