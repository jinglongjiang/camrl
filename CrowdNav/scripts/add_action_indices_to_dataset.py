#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为已存在的IL数据集回填 action_indices（与当前GRID一致）。

仅支持 merged 数据集（含 'trajectories'）。如果是 chunked 元数据，请先生成 merged 文件。
"""
import argparse
import configparser
import logging
import os
import sys
from typing import Any

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crowd_nav.contracts import init_grid_from_cfg, action_to_discrete_index, GRID


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M:%S',
        force=True,
    )


def _action_to_idx(a: Any):
    try:
        if hasattr(a, 'vx'):
            vx, vy = float(a.vx), float(a.vy)
        elif isinstance(a, (list, tuple, np.ndarray)) and len(a) >= 2:
            vx, vy = float(a[0]), float(a[1])
        else:
            return None
        return int(action_to_discrete_index(vx, vy))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Backfill action_indices into IL dataset")
    parser.add_argument('--input', type=str, required=True, help='Input merged dataset (.pth)')
    parser.add_argument('--output', type=str, default=None, help='Output dataset path')
    parser.add_argument('--env-config', type=str, default='crowd_nav/configs/env.config')
    args = parser.parse_args()

    setup_logging()

    if not os.path.exists(args.input):
        logging.error(f"Input dataset not found: {args.input}")
        sys.exit(1)

    # Init GRID from env.config
    env_cfg_raw = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    env_cfg_raw.read(args.env_config, encoding='utf-8')
    if not env_cfg_raw.has_section('policy'):
        logging.error(f"Missing [policy] in {args.env_config}")
        sys.exit(1)
    init_grid_from_cfg(env_cfg_raw)
    logging.info(f"[GRID] {GRID}")

    output_path = args.output
    if output_path is None:
        output_path = args.input.replace('.pth', '_with_idx.pth')

    logging.info(f"Loading dataset: {args.input}")
    dataset = torch.load(args.input, map_location='cpu', weights_only=False)

    if isinstance(dataset, dict) and 'chunk_files' in dataset and 'trajectories' not in dataset:
        logging.error("Chunked metadata detected. Please merge chunks first (need a merged dataset).")
        sys.exit(1)

    if not (isinstance(dataset, dict) and 'trajectories' in dataset):
        logging.error("Unsupported dataset format: missing 'trajectories'")
        sys.exit(1)

    trajs = dataset['trajectories']
    if not trajs:
        logging.error("No trajectories found")
        sys.exit(1)

    sample = trajs[0]
    if ('action_indices' in sample) or ('action_indices' in sample.get('meta', {})):
        logging.info("action_indices already present; nothing to do.")
        if args.input != output_path and not os.path.exists(output_path):
            torch.save(dataset, output_path, pickle_protocol=4)
            logging.info(f"Saved copy to: {output_path}")
        return

    logging.info(f"Backfilling action_indices for {len(trajs)} trajectories...")
    updated = 0
    for traj in trajs:
        actions = traj.get('actions', [])
        indices = []
        ok = True
        for act in actions:
            idx = _action_to_idx(act)
            if idx is None:
                ok = False
                break
            indices.append(idx)
        if not ok:
            continue
        traj['action_indices'] = indices
        meta = traj.get('meta', {})
        meta['action_indices'] = indices
        traj['meta'] = meta
        updated += 1

    logging.info(f"action_indices added for {updated} trajectories")
    logging.info(f"Saving -> {output_path}")
    torch.save(dataset, output_path, pickle_protocol=4)
    logging.info("Done.")


if __name__ == '__main__':
    main()
