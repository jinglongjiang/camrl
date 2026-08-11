"""Shared manifest/hash/atomic-status helpers (guide.md 8.1, B4).

Consolidates the pattern repeated ad hoc across every ``crowd_nav/tools/
audit_bdvl_*.py`` script written so far (compute source/config/output
hashes, write an atomic manifest, record git commit) into one place so
new tooling does not reimplement it slightly differently each time.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_obj(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def git_commit(repo_root: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_manifest(
    repo_root: str,
    command: str,
    source_files: Iterable[str],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """A standard provenance block: git commit, hostname, pid, platform,
    python/numpy/torch versions if importable, and sha256 of every path
    in ``source_files`` (relative to ``repo_root``)."""
    source_hashes = {}
    for relative in source_files:
        path = Path(repo_root) / relative
        if not path.is_file():
            raise FileNotFoundError(f"required provenance source missing: {path}")
        source_hashes[relative] = sha256_of_file(str(path))

    versions = {"python": sys.version}
    try:
        import numpy as _np
        versions["numpy"] = _np.__version__
    except ImportError:
        pass
    try:
        import torch as _torch
        versions["torch"] = _torch.__version__
        versions["cuda_available"] = _torch.cuda.is_available()
    except ImportError:
        pass

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cwd": str(Path(repo_root).resolve()),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "platform": platform.platform(),
        "versions": versions,
        "git_commit": git_commit(repo_root),
        "source_sha256": source_hashes,
    }
    if extra:
        manifest.update(extra)
    return manifest


def atomic_write_json(path: str, payload: Dict[object, object]) -> None:
    """Write-to-temp-then-rename so a reader never observes a partially
    written file (guide.md 12.3: "status.json原子更新")."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(tmp_path), str(target))
