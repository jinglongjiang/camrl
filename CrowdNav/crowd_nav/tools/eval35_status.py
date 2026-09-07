#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


GROUPS = [
    ("main", [
        "ours_s1_10",
        "lstm_s1_10",
        "sarl_s1_10",
        "orca_adapt_s1_10",
        "cadrl_adapt_s1_10",
        "dsrnn_example_adapt_s1_10",
    ]),
    ("ablation", [
        "ours_no_vl_s1_10",
        "ours_no_gdbn_veto_s1_10",
        "ours_nearest_s1_10",
        "ours_veto_025_s1_10",
        "ours_veto_045_s1_10",
        "ours_qonly_s1_10",
    ]),
    ("stress", [
        "ours_stress_square_s1_10",
        "lstm_stress_square_s1_10",
        "sarl_stress_square_s1_10",
    ]),
]


def parse_overall(summary_path):
    if not summary_path.exists():
        return None
    last = ""
    for line in summary_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("overall_mean_SR,"):
            last = line
    if not last:
        return None
    parts = last.split(",")
    try:
        return {
            "sr": float(parts[1]),
            "std": float(parts[3]),
            "seeds": int(parts[5]),
            "raw": last,
        }
    except Exception:
        return {"raw": last}


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return True


def print_summaries(root):
    for group, names in GROUPS:
        print(f"== {group} ==")
        for name in names:
            data = parse_overall(root / f"{name}.summary.txt")
            if data is None:
                print(f"{name:34s} not done")
            elif "sr" in data:
                status = "done" if data["seeds"] >= 10 else "partial"
                print(f"{name:34s} SR={data['sr']:6.2f} std={data['std']:5.2f} n={data['seeds']:2d} {status}")
            else:
                print(f"{name:34s} {data['raw']}")
        print()


def print_locks(root):
    locks = sorted(root.glob("*.lock"))
    print("== active locks ==")
    if not locks:
        print("none")
        return
    for lock in locks:
        raw = lock.read_text(encoding="utf-8", errors="ignore").strip()
        state = "active" if raw and pid_alive(raw) else "stale"
        print(f"{lock.name:44s} pid={raw or '?'} {state}")


def main():
    parser = argparse.ArgumentParser(description="Summarize eval35 progress.")
    parser.add_argument("--root", default="runs/eval35")
    args = parser.parse_args()
    root = Path(args.root)
    print_summaries(root)
    print_locks(root)


if __name__ == "__main__":
    main()
