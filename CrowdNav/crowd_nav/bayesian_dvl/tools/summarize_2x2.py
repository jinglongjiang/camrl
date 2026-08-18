"""12-row summary for the 3-seed 2x2, plus the windowed alignment detail.

Reads only what the branches wrote. A branch that aborted on the audit gate
is reported like any other -- the abort IS the experiment's answer, not a
missing row.
"""
import json, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/root/workspace/nav_data/mamba/camrl/bdvl_v6/CrowdNav/runs/v2/diag2x2")
BRANCHES = [("rho050", "0.5", "keep"), ("rho100", "1.0", "keep")]
SEEDS = [98211, 98212, 98213]
# Synchronised windows: alignment early in joint IL is not comparable to
# alignment late in it, because Adam's own bias correction and second-moment
# build-up change the realised step direction independently of the branch.
# Branches may only be compared window against matching window.
WINDOWS = [(1, 200), (201, 500), (501, 1100), (1101, 2000)]
MODULES = ("encoder", "action_encoder", "value_network")


def audit_series(log: Path):
    """One dict per AUDIT line, including the per-scenario MC and the fixed
    training-set MC when the run recorded them."""
    out = []
    if not log.exists():
        return out
    for line in log.read_text(errors="replace").splitlines():
        if "AUDIT pass=" not in line:
            continue
        f = dict(re.findall(r"([A-Za-z_0-9]+)=([-+0-9.eE]+)", line))
        try:
            row = {"pass": int(float(f["pass"])), "mc": float(f["mc"]),
                   "rank": float(f.get("rank", "nan")), "margin": float(f.get("margin", "nan")),
                   "top1": float(f.get("top1", "nan")), "range": float(f.get("range", "nan"))}
        except (KeyError, ValueError):
            continue
        for k, v in f.items():
            if k.startswith("mc_") or k == "trainfix":
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        out.append(row)
    return out


def abort_type(run_dir: Path):
    """The STRUCTURED reason a branch stopped, or None if it finished."""
    f = run_dir / "abort_reason.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("abort_type")
    except ValueError:
        return None


def read_metrics(path: Path):
    diag, il = [], []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("record_type") == "il_diagnostics":
                diag.append(r)
            elif r.get("record_type") == "il":
                il.append(r)
    return diag, il


def post_best_max(series):
    """The largest audit MC AFTER the best was reached.

    max() over the whole series is always the pass-0 value (~11.5, an
    untrained network) and says nothing; what the gate is about is how far
    the fit falls back once it has been good.
    """
    if not series:
        return float("nan")
    mc = [s["mc"] for s in series]
    i = int(np.argmin(mc))
    return max(mc[i:]) if i < len(mc) else mc[-1]


def windowed(diag, key):
    out = []
    for lo, hi in WINDOWS:
        v = [d[key] for d in diag if lo <= d["il_pass"] <= hi and key in d]
        out.append((np.median(v), np.percentile(v, 10)) if v else (float("nan"), float("nan")))
    return out


rows = []
hdr = (f"{'seed':>6}{'br':>4}{'shr':>5}{'adam':>6}{'best':>8}{'final':>8}{'postmax':>9}{'f/b':>7}"
       f"{'trainfix':>9}{'std':>8}{'junc':>8}{'rank':>8}{'margin':>8}{'top1':>7}{'range':>7}"
       f"{'cosMC':>7}{'p10':>7}{'dot>0':>7}{'hinge':>7}{'clip':>6}{'end':>16}")
print(hdr)
print("-" * len(hdr))
for seed in SEEDS:
    for name, share, adam in BRANCHES:
        out = ROOT / f"{name}_seed{seed}"
        ser = audit_series(Path(str(out) + ".log"))
        diag, il = read_metrics(out / "metrics.jsonl")
        if not ser:
            print(f"{seed:>6}{name:>4}{share:>5}{adam:>6}{'-- not run --':>30}")
            continue
        mc = [s["mc"] for s in ser]
        best, final = min(mc), mc[-1]
        last = ser[-1]
        cos = np.array([d["post_adam_cos_mc"] for d in diag]) if diag else np.array([np.nan])
        dot_pos = float(np.mean([d["post_adam_dot_mc"] > 0 for d in diag])) if diag else float("nan")
        hinge = float(np.mean([d["active_hinge_fraction"] for d in diag])) if diag else float("nan")
        clip = float(np.mean([bool(r.get("clipped")) for r in il])) if il else float("nan")
        at = abort_type(out)
        verdict = at if at else ("OK" if last["pass"] >= 2000 else f"SHORT@{last['pass']}")
        print(f"{seed:>6}{name:>4}{share:>5}{adam:>6}{best:>8.4f}{final:>8.4f}"
              f"{post_best_max(ser):>9.4f}{final/best:>7.3f}"
              f"{last.get('trainfix', float('nan')):>9.4f}"
              f"{last.get('mc_standard', float('nan')):>8.4f}"
              f"{last.get('mc_junction_crowd', last.get('mc_junction', float('nan'))):>8.4f}"
              f"{last['rank']:>8.4f}{last['margin']:>+8.4f}{last['top1']:>7.3f}{last['range']:>7.3f}"
              f"{np.nanmedian(cos):>7.3f}{np.nanpercentile(cos, 10):>7.3f}"
              f"{dot_pos:>7.2f}{hinge:>7.2f}{clip:>6.2f}{verdict:>16}")
        rows.append((seed, name, diag))

print("\n=== cos(-delta, g_MC) by synchronised window: median / p10 ===")
print("Adam's own warm-up changes the realised step direction independently of the")
print("branch, so a branch may only be compared against the SAME window of another.")
w_hdr = f"{'seed':>6}{'br':>4}" + "".join(f"{f'{lo}-{hi}':>16}" for lo, hi in WINDOWS)
print(w_hdr); print("-" * len(w_hdr))
for seed, name, diag in rows:
    cells = "".join(f"{m:>8.3f}/{p:<7.3f}" for m, p in windowed(diag, "post_adam_cos_mc"))
    print(f"{seed:>6}{name:>4}{cells}")

print("\n=== per-module cos(-delta, g_MC), median over the whole run ===")
m_hdr = f"{'seed':>6}{'br':>4}" + "".join(f"{m:>18}" for m in MODULES)
print(m_hdr); print("-" * len(m_hdr))
for seed, name, diag in rows:
    cells = ""
    for mod in MODULES:
        v = [d["modules"][mod]["cos_mc"] for d in diag if d.get("modules", {}).get(mod)]
        cells += f"{(np.median(v) if v else float('nan')):>18.3f}"
    print(f"{seed:>6}{name:>4}{cells}")

print("\nPRE-REGISTERED SELECTION RULE, frozen before this ran. A candidate passes only")
print("if on 3/3 seeds it: triggers no mc_regression / non_finite / clipping abort;")
print("ends with audit_rank <= 0.075 and margin > 0; and reaches a final audit MC")
print("better than the SAME seed's old R0 (0.1818 / 0.1530 / 0.1351). Both pass ->")
print("take rho=0.5, the smaller budget. One passes -> take it. Neither -> STOP, do")
print("not sweep more rho. The ranking-gate exemption must not mask a final failure.")
