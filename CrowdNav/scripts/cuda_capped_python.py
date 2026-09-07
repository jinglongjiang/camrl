#!/usr/bin/env python3
"""Run a Python entry point with a PyTorch CUDA allocator limit.

The cap protects an already-running peer process from this process consuming
all remaining framebuffer memory.  It does not claim to reserve compute time
or to cap allocations made outside PyTorch's caching allocator.
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: cuda_capped_python.py PROGRAM [ARGS ...]")
    raw_fraction = os.environ.get("CUDA_MEMORY_FRACTION")
    if raw_fraction is None:
        raise SystemExit("CUDA_MEMORY_FRACTION is required")
    fraction = float(raw_fraction)
    if not 0.05 <= fraction <= 0.90:
        raise SystemExit(f"unsafe CUDA_MEMORY_FRACTION={fraction}")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    total = torch.cuda.get_device_properties(0).total_memory
    print(
        f"[CUDA-CAP] fraction={fraction:.3f} "
        f"allocator_limit_bytes={int(total * fraction)} program={sys.argv[1]}",
        flush=True,
    )

    program = str(Path(sys.argv[1]).resolve())
    sys.argv = sys.argv[1:]
    try:
        runpy.run_path(program, run_name="__main__")
    finally:
        if torch.cuda.is_initialized():
            print(
                f"[CUDA-CAP] peak_allocated_bytes={torch.cuda.max_memory_allocated(0)} "
                f"peak_reserved_bytes={torch.cuda.max_memory_reserved(0)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
