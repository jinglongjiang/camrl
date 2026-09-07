#!/usr/bin/env python3
"""Summarize candidate-diagnostic JSONL files from test.py."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def _finite_score(candidate):
    value = candidate.get("final_score")
    return isinstance(value, (int, float)) and math.isfinite(value) and value > -1e8


def _pct(num, den):
    return 100.0 * num / den if den else 0.0


def _safe_key(threshold):
    return f"safe_ge_{float(threshold):.2f}"


def _record_stats(records, threshold):
    key = _safe_key(threshold)
    n = len(records)
    if n == 0:
        return None

    global_any = 0
    selected_safe = 0
    topk_finite_any = 0
    safest_in_topk = 0
    selected_unsafe_global_safe = 0
    max_clearances = []
    selected_clearances = []

    for rec in records:
        global_clearance = rec.get("global_clearance") or {}
        safety = global_clearance.get(key) or {}
        has_global_safe = bool(safety.get("any"))
        is_selected_safe = bool(safety.get("selected_safe"))

        global_any += has_global_safe
        selected_safe += is_selected_safe
        selected_unsafe_global_safe += bool(has_global_safe and not is_selected_safe)
        safest_in_topk += bool(global_clearance.get("safest_in_recorded_topk"))

        max_clear = global_clearance.get("max_clearance")
        if isinstance(max_clear, (int, float)) and math.isfinite(max_clear):
            max_clearances.append(float(max_clear))
        selected_clear = global_clearance.get("selected_clearance")
        if isinstance(selected_clear, (int, float)) and math.isfinite(selected_clear):
            selected_clearances.append(float(selected_clear))

        topk = rec.get("topk") or []
        topk_finite_any += any(
            _finite_score(c)
            and isinstance(c.get("clearance"), (int, float))
            and float(c["clearance"]) >= float(threshold)
            for c in topk
        )

    return {
        "steps": n,
        "global_any_safe_pct": _pct(global_any, n),
        "topk_finite_any_safe_pct": _pct(topk_finite_any, n),
        "selected_safe_pct": _pct(selected_safe, n),
        "selected_unsafe_despite_global_safe_pct": _pct(selected_unsafe_global_safe, n),
        "safest_in_recorded_topk_pct": _pct(safest_in_topk, n),
        "max_clearance_mean": mean(max_clearances) if max_clearances else float("nan"),
        "selected_clearance_mean": mean(selected_clearances) if selected_clearances else float("nan"),
    }


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Candidate diagnostic JSONL files")
    parser.add_argument("--thresholds", default="0.22,0.20,0.15")
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    for file_name in args.files:
        path = Path(file_name)
        episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        case = episodes[0].get("case", path.stem) if episodes else path.stem
        outcomes = {name: sum(ep.get("outcome") == name for ep in episodes) for name in ("collision", "timeout")}
        records = [rec for ep in episodes for rec in ep.get("records", [])]

        print(f"\nCASE {case} file={path.name}")
        print(f"episodes={len(episodes)} collision={outcomes['collision']} timeout={outcomes['timeout']} steps={len(records)}")

        groups = [("all", records)]
        for outcome in ("collision", "timeout"):
            groups.append((outcome, [rec for ep in episodes if ep.get("outcome") == outcome for rec in ep.get("records", [])]))
        for tail in (1, 3, 5, 10):
            groups.append((f"collision_tail{tail}", [
                rec
                for ep in episodes
                if ep.get("outcome") == "collision"
                for rec in ep.get("records", [])[-tail:]
            ]))

        for threshold in thresholds:
            print(f"  threshold={threshold:.2f}")
            for label, subset in groups:
                stats = _record_stats(subset, threshold)
                if not stats:
                    continue
                print(
                    "    {label:<18} steps={steps:<5} global_safe={global_any_safe_pct:>6} "
                    "topk_safe={topk_finite_any_safe_pct:>6} selected_safe={selected_safe_pct:>6} "
                    "missed_global_safe={selected_unsafe_despite_global_safe_pct:>6} "
                    "safest_in_topk={safest_in_recorded_topk_pct:>6} "
                    "clear(sel/max)={selected_clearance_mean}/{max_clearance_mean}".format(
                        label=label,
                        **{k: _fmt(v) for k, v in stats.items()},
                    )
                )


if __name__ == "__main__":
    main()
