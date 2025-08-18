#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_orca_traj.py
-----------------
可视化 orca_demos_seq.npz 中多条 expert 轨迹。

默认：每次随机抽取 rows*cols 条并作图。
想固定重现：加 --seed 42
想看前几条：--mode first
想只看第 idx 条：--idx 17（同时设 --rows 1 --cols 1 更合适）
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='orca_demos_seq.npz', help='npz 文件路径')
    p.add_argument('--rows', type=int, default=3, help='子图行数')
    p.add_argument('--cols', type=int, default=3, help='子图列数')
    p.add_argument('--idx',  type=int, default=None, help='只看第 idx 条 (0-based)')
    p.add_argument('--mode', choices=['random', 'first'], default='random',
                  help='抽取方式：random(默认) 或 first(前几条)')
    p.add_argument('--seed', type=int, default=None, help='随机种子（给了就可复现）')
    p.add_argument('--save', default='orca_traj_grid.png', help='保存文件名')
    return p.parse_args()

def pick_indices(E, K, idx, mode, seed):
    if idx is not None:
        return [max(0, min(E-1, idx))]
    K = min(K, E)
    if mode == 'first':
        return list(range(K))
    rng = np.random.default_rng(seed)
    return rng.choice(E, size=K, replace=False).tolist()

def main():
    args = parse_args()

    data = np.load(args.file, allow_pickle=True)
    obs_arr = data['obs']     # object array: list of (T, D)
    E = len(obs_arr)
    print(f'loaded {E} episodes from {args.file}')

    K = args.rows * args.cols
    idx_list = pick_indices(E, K, args.idx, args.mode, args.seed)
    print(f'plot indices: {idx_list}')

    rows, cols = args.rows, args.cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for k, idx in enumerate(idx_list):
        ep = obs_arr[idx]
        ax = axes[k]
        T  = ep.shape[0]

        # 机器人轨迹
        rxy = ep[:, :2]
        ax.plot(rxy[:,0], rxy[:,1], 'b.-', lw=1.5, ms=3, label='Robot')
        ax.scatter(rxy[0,0], rxy[0,1], c='g', s=60, label='Start')
        ax.scatter(rxy[-1,0], rxy[-1,1], c='r', s=60, label='Goal')

        # 人类轨迹（按 CrowdNav 的 [robot(6) + humans(4*nh)] 结构推算）
        nh = max(0, (ep.shape[1] - 6) // 4)
        for h in range(nh):
            hxy = ep[:, 6 + 4*h : 6 + 4*h + 2]
            ax.plot(hxy[:,0], hxy[:,1], '--', lw=1, alpha=0.7, label=f'H{h}')

        ax.set_title(f'Ep {idx} | len={T}')
        ax.axis('equal'); ax.grid(True)

    # 关掉多余子图
    for ax in axes[len(idx_list):]:
        ax.axis('off')

    # 只取关键图例项（Robot/Start/Goal/H0）
    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        # 取前三个固定项 + 第一个人类（若存在）
        keep = []
        for name in ['Robot', 'Start', 'Goal']:
            for h, l in zip(handles, labels):
                if l == name:
                    keep.append((h, l)); break
        # 加一个 H0（如果存在）
        for h, l in zip(handles, labels):
            if l.startswith('H0'):
                keep.append((h, l)); break
        if keep:
            fig.legend([h for h, _ in keep], [l for _, l in keep], loc='upper right')

    plt.tight_layout()

    # 若未指定 seed，自动在文件名里加时间戳，避免覆盖
    save_name = args.save
    if args.seed is None and args.idx is None and args.mode == 'random':
        stem, ext = (save_name.rsplit('.', 1) + ['png'])[:2]
        ts = datetime.now().strftime('%H%M%S')
        save_name = f'{stem}_{ts}.{ext}'

    plt.savefig(save_name, dpi=150)
    print(f'saved to: {save_name}')
    plt.show()

if __name__ == '__main__':
    main()

