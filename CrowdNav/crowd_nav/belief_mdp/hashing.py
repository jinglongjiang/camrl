"""Shared content-hashing helpers so train.py and evaluate.py agree on what
"the same file/directory" means.

Used to record, at training time, exactly which Mamba checkpoint, GDBN
parameter directory, and config files produced a given belief-MDP
checkpoint -- and to verify, at evaluation time, that nothing referenced by
those same CLI arguments has changed since, even if the path string is
identical (a config file can be silently edited in place).
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: str) -> str:
    """Deterministic hash of every file under ``path``, order-independent.

    Hashes each file individually, then hashes the sorted list of
    "relative_path:file_hash" lines -- so moving files around without
    changing content changes the result (paths matter), but directory
    listing order never does.
    """
    root = Path(path)
    entries = []
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file():
            relative = file_path.relative_to(root).as_posix()
            entries.append(f"{relative}:{sha256_file(str(file_path))}")
    digest = hashlib.sha256()
    digest.update("\n".join(entries).encode("utf-8"))
    return digest.hexdigest()
