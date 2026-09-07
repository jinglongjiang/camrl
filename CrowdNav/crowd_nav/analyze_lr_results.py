#!/usr/bin/env python3
"""
Learning Rate Search Results Analyzer
Analyzes training logs from LR search and recommends best learning rate
"""
import os
import re
import glob
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = "lr_search_results"

def parse_train_log(log_path):
    """Extract metrics from training log"""
    metrics = {
        'lr': None,
        'episodes': [],
        'success_rates': [],
        'collision_rates': [],
        'timeout_rates': [],
        'value_losses': [],
        'policy_losses': [],
        'final_success': None,
        'avg_success_last_50': None,
        'loss_stability': None,
    }

    # Extract LR from path
    lr_match = re.search(r'lr_([\d.e-]+)', str(log_path))
    if lr_match:
        metrics['lr'] = lr_match.group(1)

    with open(log_path, 'r') as f:
        for line in f:
            # Extract episode metrics
            if '[STAT]' in line and 'roll@25' in line:
                # Parse: roll@25(s/c/t)=0.8/0.1/0.1
                match = re.search(r'roll@25\(s/c/t\)=([\d.]+)/([\d.]+)/([\d.]+)', line)
                if match:
                    metrics['success_rates'].append(float(match.group(1)))
                    metrics['collision_rates'].append(float(match.group(2)))
                    metrics['timeout_rates'].append(float(match.group(3)))

            # Extract loss values
            if 'vloss=' in line and 'ploss=' in line:
                vloss_match = re.search(r'vloss=([\d.]+)', line)
                ploss_match = re.search(r'ploss=([\d.]+)', line)
                if vloss_match and ploss_match:
                    metrics['value_losses'].append(float(vloss_match.group(1)))
                    metrics['policy_losses'].append(float(ploss_match.group(1)))

            # Extract episode number
            ep_match = re.search(r'ep=(\d+)', line)
            if ep_match and len(metrics['success_rates']) == len(metrics['episodes']) + 1:
                metrics['episodes'].append(int(ep_match.group(1)))

    # Calculate summary statistics
    if metrics['success_rates']:
        metrics['final_success'] = metrics['success_rates'][-1]
        if len(metrics['success_rates']) >= 50:
            metrics['avg_success_last_50'] = np.mean(metrics['success_rates'][-50:])
        else:
            metrics['avg_success_last_50'] = np.mean(metrics['success_rates'])

    if metrics['value_losses']:
        # Loss stability: lower std = more stable
        metrics['loss_stability'] = np.std(metrics['value_losses'])

    return metrics


def analyze_all_results():
    """Analyze all LR search results"""
    results_path = Path(RESULTS_DIR)
    if not results_path.exists():
        print(f"❌ Results directory not found: {RESULTS_DIR}")
        return None

    # Find all train.log files
    log_files = list(results_path.glob("lr_*/train.log"))

    if not log_files:
        print(f"❌ No training logs found in {RESULTS_DIR}")
        return None

    print(f"Found {len(log_files)} training logs")
    print("=" * 80)

    # Parse all logs
    all_metrics = []
    for log_file in log_files:
        print(f"Parsing: {log_file}")
        metrics = parse_train_log(log_file)
        all_metrics.append(metrics)

    # Create DataFrame for analysis
    df = pd.DataFrame([{
        'LR': m['lr'],
        'Final Success': m['final_success'],
        'Avg Success (Last 50)': m['avg_success_last_50'],
        'Loss Stability': m['loss_stability'],
        'Total Episodes': len(m['episodes']),
    } for m in all_metrics])

    # Sort by learning rate
    df['LR_float'] = df['LR'].astype(float)
    df = df.sort_values('LR_float')

    return df, all_metrics


def recommend_best_lr(df):
    """Recommend best learning rate based on multiple criteria"""
    print("\n" + "=" * 80)
    print("LEARNING RATE SEARCH RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Scoring system (normalized 0-1, higher is better)
    df['Success_Score'] = df['Avg Success (Last 50)']  # Already 0-1

    # Stability score (lower std is better, invert and normalize)
    if df['Loss Stability'].notna().any():
        max_std = df['Loss Stability'].max()
        df['Stability_Score'] = 1 - (df['Loss Stability'] / max_std)
    else:
        df['Stability_Score'] = 0.5

    # Combined score (weighted average)
    df['Total_Score'] = (
        0.7 * df['Success_Score'] +      # 70% weight on success
        0.3 * df['Stability_Score']      # 30% weight on stability
    )

    # Find best LR
    best_idx = df['Total_Score'].idxmax()
    best_lr = df.loc[best_idx, 'LR']
    best_success = df.loc[best_idx, 'Avg Success (Last 50)']
    best_stability = df.loc[best_idx, 'Loss Stability']

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"🏆 Best Learning Rate: {best_lr}")
    print(f"   - Avg Success Rate: {best_success:.1%}")
    print(f"   - Loss Stability (std): {best_stability:.4f}")
    print(f"   - Overall Score: {df.loc[best_idx, 'Total_Score']:.3f}")
    print("=" * 80)

    # Show top 3
    print("\nTop 3 Learning Rates:")
    top3 = df.nlargest(3, 'Total_Score')[['LR', 'Avg Success (Last 50)', 'Loss Stability', 'Total_Score']]
    print(top3.to_string(index=False))
    print("=" * 80)

    return best_lr


def main():
    print("Analyzing Learning Rate Search Results...")
    print("=" * 80)

    df, all_metrics = analyze_all_results()

    if df is None or len(df) == 0:
        print("❌ No results to analyze")
        return

    best_lr = recommend_best_lr(df)

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print(f"1. Update configs/train.config with: learning_rate = {best_lr}")
    print(f"2. Run full training with optimal LR")
    print(f"3. Monitor for improved convergence and stability")
    print("=" * 80)


if __name__ == "__main__":
    main()
