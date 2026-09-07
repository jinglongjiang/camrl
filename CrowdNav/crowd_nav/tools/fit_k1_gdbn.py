#!/usr/bin/env python3
"""Fit the single-mode GDBN baseline from the existing ORCA demonstrations."""

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crowd_nav.gdbn import GDBNIntegration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', default='../orca_demos_seq.npz')
    parser.add_argument('--output', default='runs/mamba_vl/gdbn_params_k1')
    parser.add_argument('--particles', type=int, default=50)
    parser.add_argument('--max_peds', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    demo_path = Path(args.demo).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not demo_path.is_file():
        raise FileNotFoundError(f"ORCA demonstration file not found: {demo_path}")

    model = GDBNIntegration(
        K=1,
        n_particles=args.particles,
        max_peds=args.max_peds,
    )
    model.train_offline(
        str(demo_path),
        save_dir=str(output_path),
        gng_epochs=args.epochs,
    )

    check = GDBNIntegration(
        K=1,
        n_particles=args.particles,
        params_dir=str(output_path),
        max_peds=args.max_peds,
    )
    expected = {'gng.npz', 'gdbn.npz', 'action_model.npz'}
    actual = {path.name for path in output_path.glob('*.npz')}
    if check.K != 1 or not check.is_fitted or not check.action_fitted:
        raise RuntimeError(
            f"Invalid K=1 parameters: K={check.K}, "
            f"fitted={check.is_fitted}, action_fitted={check.action_fitted}"
        )
    if not expected.issubset(actual):
        raise RuntimeError(f"Missing K=1 parameter files: {sorted(expected - actual)}")

    with np.load(output_path / 'gdbn.npz') as payload:
        saved_k = int(payload['K'])
    print(
        f"[K1-GDBN] ready: K={saved_k}, action_fitted={check.action_fitted}, "
        f"output={output_path}"
    )


if __name__ == '__main__':
    main()
