"""Job P: multi-horizon prediction accuracy, CV vs MEAN vs FULL.

Question: at what time scale, if any, is the Bayesian intent prediction
actually closer to where the pedestrian goes than constant velocity?

No Mamba checkpoint is loaded and no simulator is run. The recorded
trajectories are replayed, the belief bank is re-run in frame order per
episode, and each prediction is scored against the SAME track's true future
position.

FULL is enumerated exactly over its candidates -- there is no Monte-Carlo
sampling here, so this table carries no estimator noise.

All three arms are scored with one metric, the energy score

    ES = sum_k b_k ||p_k - y|| - 0.5 sum_{k,l} b_k b_l ||p_k - p_l||

which is proper for a predictive distribution and degenerates to plain
endpoint error for the single-point CV and MEAN arms. Best-of-K is deliberately
NOT used: it would flatter FULL for merely being a set.

Leakage: the bank is fed observed positions and public scene anchors only; a
pedestrian's hidden gx/gy is never read, here or anywhere else.
"""
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from crowd_nav.bsw.vendor.intent_tracker import IntentBeliefBank
from crowd_nav.bsw.vendor.scene_candidates import (
    circle_scene, make_candidate_fn, square_scene)

STATES = os.environ.get("D0_STATES", "/root/bdvl_diagnostics/d0/states_v2.pkl")
OUT = os.environ.get("P_OUT", "/root/bdvl_diagnostics/p/p_horizon.json")
HORIZONS = [1, 2, 4, 8]
DT = 0.25
VMAX = 1.0
CRITICAL = 0.30
BOOT = 2000


def _bank(scene, dt):
    sim = (scene.get("sim") or "").strip()
    if sim == "circle_crossing":
        sc = circle_scene(float(scene["circle_radius"]), n_sectors=8)
    elif sim == "square_crossing":
        sc = square_scene(float(scene["square_width"]), n_rows=4)
    else:
        raise ValueError(f"no public scene for sim={sim!r}")
    return IntentBeliefBank(candidate_fn=make_candidate_fn(sc), dt=dt, speed=1.0)


def energy_score(b, P, y):
    """b: [K] weights, P: [K,2] predicted points, y: [2] truth."""
    d1 = float((b * np.linalg.norm(P - y, axis=1)).sum())
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    d2 = float(0.5 * (b[:, None] * b[None, :] * D).sum())
    return d1 - d2


def main():
    with open(STATES, "rb") as f:
        recs = pickle.load(f)["records"]
    by_ep = defaultdict(list)
    for i, r in enumerate(recs):
        by_ep[int(r["ep"])].append(i)
    for e in by_ep:
        by_ep[e].sort(key=lambda i: int(recs[i]["frame"]))

    rows = []
    n_gap = 0
    for ep, idxs in sorted(by_ep.items()):
        bank = _bank(recs[idxs[0]]["scene"], DT)
        bank.reset()
        # true trajectory of every stable track in this episode, by frame
        traj = np.array([recs[i]["env_humans"] for i in idxs])   # [F, N, 5]
        F, N, _ = traj.shape
        for f, i in enumerate(idxs):
            rec = recs[i]
            envh = rec["env_humans"]
            bank.update({t: (float(envh[t][0]), float(envh[t][1])) for t in range(N)})
            ttc_rank = {int(t): s for s, t in enumerate(rec["track_ids"].tolist())}
            crit = float(rec["dmin_all"]) < CRITICAL
            for t in range(N):
                pos = envh[t][:2].astype(np.float64)
                vel = envh[t][2:4].astype(np.float64)
                try:
                    tr = bank.tracker_for(int(t))
                    b = np.asarray(bank.belief_for(int(t)), np.float64)
                except Exception:
                    continue
                for H in HORIZONS:
                    if f + H >= F:
                        continue
                    # identity must survive: CrowdSim can respawn a pedestrian
                    # that reaches its goal, which would look like a teleport.
                    seg = traj[f:f + H + 1, t, :2].astype(np.float64)
                    if np.max(np.linalg.norm(np.diff(seg, axis=0), axis=1)) > VMAX * DT * 1.5:
                        n_gap += 1
                        continue
                    y = seg[-1]
                    P = np.array([tr.roll_candidate_future(k, tuple(pos), H)[-1]
                                  for k in range(len(b))], dtype=np.float64)
                    mu = (b[:, None] * P).sum(0)
                    spread = float(np.sqrt((b * ((P - mu) ** 2).sum(1)).sum()))
                    cv = pos + H * DT * vel
                    rows.append({
                        "H": H, "top5": int(ttc_rank.get(t, 99) < 5), "crit": int(crit),
                        "spread": spread,
                        "cv": float(np.linalg.norm(cv - y)),
                        "mean": float(np.linalg.norm(mu - y)),
                        "full": energy_score(b, P, y),
                    })
    print(f"[P] {len(rows)} (state,human,H) predictions; {n_gap} dropped on identity gaps")

    rng = np.random.default_rng(7)

    def boot(d):
        if len(d) < 20:
            return (float("nan"), float("nan"))
        idx = rng.integers(0, len(d), (BOOT, len(d)))
        m = d[idx].mean(axis=1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    hdr = (f"\n{'H':>3}{'subset':<14}{'N':>7}{'CV ES':>9}{'MEAN ES':>9}{'FULL ES':>9}"
           f"{'MEAN-CV':>10}{'  95% CI':>18}{'FULL-CV':>10}{'  95% CI':>18}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    allr = rows
    sp_all = np.array([r["spread"] for r in rows])
    out = []
    for H in HORIZONS:
        hs = [r for r in allr if r["H"] == H]
        sp = np.array([r["spread"] for r in hs])
        qs = np.percentile(sp, [25, 50, 75]) if len(sp) else [0, 0, 0]
        subsets = [("top5", [r for r in hs if r["top5"]]),
                   ("all", hs),
                   ("spread Q1", [r for r, s in zip(hs, sp) if s <= qs[0]]),
                   ("spread Q4", [r for r, s in zip(hs, sp) if s > qs[2]]),
                   ("critical", [r for r in hs if r["crit"]])]
        for name, sub in subsets:
            if not sub:
                continue
            cv = np.array([r["cv"] for r in sub])
            mn = np.array([r["mean"] for r in sub])
            fl = np.array([r["full"] for r in sub])
            dm, df = mn - cv, fl - cv
            lm, hm = boot(dm)
            lf, hf = boot(df)
            print(f"{H:>3}{name:<14}{len(sub):>7}{cv.mean():>9.4f}{mn.mean():>9.4f}"
                  f"{fl.mean():>9.4f}{dm.mean():>10.4f}  [{lm:>6.4f},{hm:>7.4f}]"
                  f"{df.mean():>10.4f}  [{lf:>6.4f},{hf:>7.4f}]")
            out.append({"H": H, "subset": name, "n": len(sub),
                        "cv": cv.mean(), "mean": mn.mean(), "full": fl.mean(),
                        "mean_minus_cv": dm.mean(), "mean_ci": [lm, hm],
                        "full_minus_cv": df.mean(), "full_ci": [lf, hf]})
        print()
    print("Energy score, metres, LOWER IS BETTER. Negative MEAN-CV / FULL-CV means "
          "the Bayesian prediction beats constant velocity. CIs are paired "
          "bootstrap over the per-prediction differences.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[P] -> {OUT}")


if __name__ == "__main__":
    main()
