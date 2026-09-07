# -*- coding: utf-8 -*-
"""

、、
"""
from __future__ import annotations

import os
import logging
import numpy as np
from collections import deque
from typing import Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False


# =========================================================
# =========================================================
class TrainingStats:
    def __init__(self, roll: int = 50):
        self.roll = roll
        self.win = deque(maxlen=roll)
        self.hist_s, self.hist_c, self.hist_t, self.hist_r = [], [], [], []
        self.q_spread = deque(maxlen=200)

        self.short_win = deque(maxlen=10)
        self.med_win = deque(maxlen=25)
        self.long_win = deque(maxlen=100)

        self.hist_time = []           # Time taken (s)
        self.hist_discomfort_freq = []  # Discomfort frequency
        self.hist_discomfort_dist = []  # Discomfort distance (m)

        # SAC loss tracking
        self.hist_v_loss = []         # Value/Critic loss (Q1+Q2)
        self.hist_p_loss = []         # Policy/Actor loss

    def update(self, succ, coll, timeout, reward, episode=None, time_taken=None, discomfort_freq=None, discomfort_dist=None, v_loss=None, p_loss=None):
        entry = (succ, coll, timeout)
        self.win.append(entry); self.hist_r.append(reward)
        s = np.mean([x[0] for x in self.win]) if self.win else 0.0
        c = np.mean([x[1] for x in self.win]) if self.win else 0.0
        t = np.mean([x[2] for x in self.win]) if self.win else 0.0
        self.hist_s.append(s); self.hist_c.append(c); self.hist_t.append(t)

        self.short_win.append(entry)
        self.med_win.append(entry)
        self.long_win.append(entry)

        if time_taken is not None:
            self.hist_time.append(time_taken)
        if discomfort_freq is not None:
            self.hist_discomfort_freq.append(discomfort_freq)
        if discomfort_dist is not None:
            self.hist_discomfort_dist.append(discomfort_dist)
        if v_loss is not None:
            self.hist_v_loss.append(v_loss)
        if p_loss is not None:
            self.hist_p_loss.append(p_loss)

    def get_multi_scale_metrics(self):
        def _window_stats(window):
            if not window: return {"succ": 0, "coll": 0, "timeout": 0}
            s = np.mean([x[0] for x in window])
            c = np.mean([x[1] for x in window])
            t = np.mean([x[2] for x in window])
            return {"succ": s, "coll": c, "timeout": t}

        peak_perf = max(self.hist_s) if self.hist_s else 0.0
        recent_succ = self.hist_s[-10:] if len(self.hist_s) >= 10 else self.hist_s
        stable_90 = sum(1 for s in recent_succ if s >= 0.9)

        return {
            "short_term": _window_stats(self.short_win),
            "medium_term": _window_stats(self.med_win),
            "long_term": _window_stats(self.long_win),
            "learning_rate": 0.0,
            "stability": 0.8,
            "peak_performance": peak_perf,
            "stable_90_episodes": stable_90,
            "total_90_achievements": stable_90,
            "stagnation_risk": False
        }

    def upd_spread(self, v):
        self.q_spread.append(v)

    def format_percentage(self, s, c, t):
        return f"{s:.2f}/{c:.2f}/{t:.2f}"


# =========================================================
# =========================================================
class Plotter:
    def __init__(self, outdir: str, plot_every: int = 25):
        self.outdir = outdir; self.plot_every = plot_every; self.png = os.path.join(outdir, 'curves.png')
        os.makedirs(outdir, exist_ok=True)

    def plot(self, ep: int, stats: TrainingStats):
        if not _HAVE_PLT: return
        if ep < 2 or (ep % self.plot_every != 0): return
        xs = list(range(1, len(stats.hist_s)+1))
        if len(xs) < 2: return
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), dpi=140)

        # 上方：成功率/碰撞率/超时率（保留原样）
        ax1.plot(xs, stats.hist_s, label=f"ROLL@{stats.roll} succ", color='green', lw=2)
        ax1.plot(xs, stats.hist_c, label=f"ROLL@{stats.roll} coll", color='red', lw=2)
        ax1.plot(xs, stats.hist_t, label=f"ROLL@{stats.roll} timeout", color='orange', lw=1)
        ax1.set_ylim(-0.02,1.02)
        ax1.set_yticks(np.arange(0, 1.1, 0.1))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.set_title('Success/Collision/Timeout')

        # 下方：V Loss 和 P Loss（替换原来的reward）
        if stats.hist_v_loss and stats.hist_p_loss:
            xs_loss = list(range(1, len(stats.hist_v_loss)+1))
            # V Loss (Critic Loss) - 蓝色
            ax2.plot(xs_loss, stats.hist_v_loss, alpha=0.3, color='blue', label='V Loss (raw)')
            if len(stats.hist_v_loss) > stats.roll:
                ma_v = np.convolve(stats.hist_v_loss, np.ones(stats.roll)/stats.roll, 'valid')
                ax2.plot(list(range(stats.roll, stats.roll+len(ma_v))), ma_v, color='blue', lw=2, label=f'V Loss MA@{stats.roll}')

            # P Loss (Actor Loss) - 红色
            xs_ploss = list(range(1, len(stats.hist_p_loss)+1))
            ax2.plot(xs_ploss, stats.hist_p_loss, alpha=0.3, color='red', label='P Loss (raw)')
            if len(stats.hist_p_loss) > stats.roll:
                ma_p = np.convolve(stats.hist_p_loss, np.ones(stats.roll)/stats.roll, 'valid')
                ax2.plot(list(range(stats.roll, stats.roll+len(ma_p))), ma_p, color='red', lw=2, label=f'P Loss MA@{stats.roll}')

            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            ax2.set_title('Training Loss (Value/Policy)')
        else:
            # 如果没有loss数据，显示提示
            ax2.text(0.5, 0.5, 'Loss data not available yet\n(will show after first update)',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Training Loss (waiting for data)')

        plt.tight_layout(); plt.savefig(self.png); plt.close()
        logging.info(f"[PLOT] curves updated ep={ep}")


class MetricsPlotter:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.metrics_png = os.path.join(outdir, 'curves_3metrics.png')

    def plot(self, ep: int, stats: TrainingStats, plot_every: int = 25):
        if not _HAVE_PLT: return
        if ep < 2 or (ep % plot_every != 0): return
        if not stats.hist_time or not stats.hist_discomfort_freq or not stats.hist_discomfort_dist:
            return

        episodes = list(range(1, len(stats.hist_time)+1))

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=140)

        axes[0].plot(episodes, stats.hist_time, color='blue', lw=1.5, alpha=0.7)
        axes[0].set_ylabel('Time Taken (s)', fontsize=11)
        axes[0].set_title('Time Taken over Episodes', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, stats.hist_discomfort_freq, color='orange', lw=1.5, alpha=0.7)
        axes[1].set_ylabel('Discomfort Freq', fontsize=11)
        axes[1].set_title('Discomfort Frequency over Episodes', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(episodes, stats.hist_discomfort_dist, color='purple', lw=1.5, alpha=0.7)
        axes[2].set_xlabel('Episode', fontsize=11)
        axes[2].set_ylabel('Discomfort Dist (m)', fontsize=11)
        axes[2].set_title('Discomfort Distance over Episodes', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.metrics_png)
        plt.close()
        logging.info(f"[METRICS-PLOT] 3-metrics curve -> {self.metrics_png}")


# =========================================================
# =========================================================
def log_periodic_metrics(ep: int, stats: TrainingStats, window: int = 200):
    if ep % window != 0:
        return

    total_eps = len(stats.hist_s)
    if total_eps == 0:
        return

    avg_succ = np.mean(stats.hist_s)
    avg_coll = np.mean(stats.hist_c)
    avg_timeout = np.mean(stats.hist_t)
    avg_reward = np.mean(stats.hist_r)
    avg_time = np.mean(stats.hist_time) if stats.hist_time else 0.0
    avg_dfreq = np.mean(stats.hist_discomfort_freq) if stats.hist_discomfort_freq else 0.0
    avg_ddist = np.mean(stats.hist_discomfort_dist) if stats.hist_discomfort_dist else 0.0

    logging.info("=" * 80)
    logging.info(f"[CUMULATIVE-METRICS] Episode {ep}: Cumulative average (all {total_eps} episodes)")
    logging.info(f"  Success: {avg_succ:.3f} ({avg_succ*100:.1f}%), Collision: {avg_coll:.3f} ({avg_coll*100:.1f}%), Timeout: {avg_timeout:.3f} ({avg_timeout*100:.1f}%)")
    logging.info(f"  Avg Reward: {avg_reward:.2f}")
    logging.info(f"  Avg Time Taken: {avg_time:.2f}s")
    logging.info(f"  Avg Discomfort Freq: {avg_dfreq:.3f}")
    logging.info(f"  Avg Discomfort Dist: {avg_ddist:.3f}m")
    logging.info("=" * 80)


def log_final_metrics(stats: TrainingStats, best_succ: float):
    if not stats.hist_s or not stats.hist_c or not stats.hist_t:
        logging.warning("[FINAL-STATS] No training statistics available")
        return

    final_success_rate = np.mean(stats.hist_s)
    final_collision_rate = np.mean(stats.hist_c)
    final_timeout_rate = np.mean(stats.hist_t)
    final_reward = np.mean(stats.hist_r) if stats.hist_r else 0.0

    final_time_taken = np.mean(stats.hist_time) if stats.hist_time else 0.0
    final_discomfort_freq = np.mean(stats.hist_discomfort_freq) if stats.hist_discomfort_freq else 0.0
    final_discomfort_dist = np.mean(stats.hist_discomfort_dist) if stats.hist_discomfort_dist else 0.0

    logging.info("=" * 80)
    logging.info("")
    logging.info("=" * 80)
    logging.info(f"[FINAL-STATS] Episodes completed: {len(stats.hist_s)}")
    logging.info(f"[FINAL-STATS] Average Success Rate:   {final_success_rate:.3f} ({final_success_rate*100:.1f}%)")
    logging.info(f"[FINAL-STATS] Average Collision Rate: {final_collision_rate:.3f} ({final_collision_rate*100:.1f}%)")
    logging.info(f"[FINAL-STATS] Average Timeout Rate:   {final_timeout_rate:.3f} ({final_timeout_rate*100:.1f}%)")
    logging.info(f"[FINAL-STATS] Average Reward:         {final_reward:.2f}")
    logging.info(f"[FINAL-STATS] Average Time Taken:     {final_time_taken:.2f}s")
    logging.info(f"[FINAL-STATS] Average Discomfort Freq: {final_discomfort_freq:.3f}")
    logging.info(f"[FINAL-STATS] Average Discomfort Dist: {final_discomfort_dist:.3f}m")
    logging.info(f"[FINAL-STATS] Best Evaluation Success: {best_succ:.3f} ({best_succ*100:.1f}%)")

    if final_success_rate >= 0.9:
        logging.info(f"[ACHIEVEMENT] 🎯 TARGET REACHED! Average success rate {final_success_rate:.1%} >= 90%")
    elif final_success_rate >= 0.8:
        logging.info(f"[PROGRESS] 📈 GOOD PROGRESS! Average success rate {final_success_rate:.1%}, approaching 90% target")
    else:
        logging.info(f"[STATUS] 📊 Current performance: {final_success_rate:.1%}, continuing towards 90% target")

    logging.info("=" * 80)
