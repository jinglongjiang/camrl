"""Job V: is the D-1 regression caused by the velocity field?

D-1's Bayesian arms changed two things at once. They moved each pedestrian's
assumed successor POSITION, and they also replaced the observed velocity with
one reconstructed from that position, (p_hat - p) / dt. CV keeps the observed
velocity. Velocity is a direct feature of the 13-D token, so the comparison so
far was not single-variable.

This probe isolates it. Four arms, identical posteriors, identical candidate
futures, identical positions within each pair:

    MEAN-RV  posterior mean position + reconstructed velocity   (D-1 as run)
    MEAN-OV  posterior mean position + OBSERVED velocity        (control)
    FULL-RV  sampled positions       + reconstructed velocity   (D-1 as run)
    FULL-OV  the SAME sampled positions + observed velocity     (control)

FULL-RV and FULL-OV use common random numbers: the identical M joint worlds,
drawn once. Independent draws would let Monte-Carlo noise masquerade as a
velocity effect.

Because position is untouched within a pair, R and dmin must come out bitwise
equal; that is asserted, not assumed. Any J difference is therefore purely
gamma * (V_OV - V_RV), which makes this a clean causal test of the velocity
channel into the frozen value network.
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch

from crowd_nav.bsw.vendor.intent_tracker import IntentBeliefBank
from crowd_nav.bsw.vendor.scene_candidates import (
    circle_scene, make_candidate_fn, square_scene)
from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState

STATES = os.environ.get("D0_STATES", "/root/bdvl_diagnostics/d0/states_v2.pkl")
OUT = os.environ.get("V_OUT", "/root/bdvl_diagnostics/v/v_velocity.json")
M_SAMPLES = int(os.environ.get("V_M", "16"))
SEED = int(os.environ.get("V_SEED", "12345"))
LIMIT = int(os.environ.get("V_LIMIT", "0"))
CRITICAL = 0.30
HORIZON = 1


def _fs(a):
    return FullState(*[float(x) for x in a])


def _bank(scene, dt):
    sim = (scene.get("sim") or "").strip()
    if sim == "circle_crossing":
        sc = circle_scene(float(scene["circle_radius"]), n_sectors=8)
    elif sim == "square_crossing":
        sc = square_scene(float(scene["square_width"]), n_rows=4)
    else:
        raise ValueError(f"no public scene for sim={sim!r}")
    return IntentBeliefBank(candidate_fn=make_candidate_fn(sc), dt=dt, speed=1.0)


def score(policy, robot, humans, base_hist):
    gamma = float(policy.gamma)
    seqs, rew, dm = [], [], []
    for act in policy.action_space:
        ns = policy.propagate(robot, act)
        rew.append(policy.compute_reward(ns, humans, prev_nav=robot, action=act))
        dm.append(min((np.hypot(ns.px - h.px, ns.py - h.py) - ns.radius - h.radius)
                      for h in humans) if humans else np.inf)
        s34 = policy._build_joint_state_34(ns, humans)
        tok = _batch_joint34_to_tokens_vectorized(s34.reshape(1, -1))[0]
        s = (base_hist + [tok])[-policy.seq_len:]
        if len(s) < policy.seq_len:
            s = [s[0]] * (policy.seq_len - len(s)) + s
        seqs.append(np.array(s))
    with torch.no_grad():
        v = policy.forward_value(
            torch.from_numpy(np.array(seqs)).float().to(policy.device)).cpu().numpy()
    return (np.array(rew, np.float64), np.asarray(v, np.float64).ravel(),
            np.array(dm, np.float64))


def deployed(policy, r, v, d):
    dep = r + float(policy.gamma) * v
    if policy.test_min_clearance > 0.0:
        m = d >= policy.test_min_clearance
        if m.any():
            dep = np.where(m, dep, -1e9)
    if policy.test_risk_lambda > 0.0:
        mg = policy.test_min_clearance if policy.test_min_clearance > 0.0 else policy.discomfort_dist
        dep = dep - policy.test_risk_lambda * np.clip(mg - d, 0.0, None)
    dep = dep.copy()
    dep[0] -= 1e-3
    return dep


def run(policy):
    with open(STATES, "rb") as f:
        recs = pickle.load(f)["records"]
    dt = float(policy.time_step)
    gamma = float(policy.gamma)
    rng = np.random.default_rng(SEED)
    by_ep = defaultdict(list)
    for i, r in enumerate(recs):
        by_ep[int(r["ep"])].append(i)
    for e in by_ep:
        by_ep[e].sort(key=lambda i: int(recs[i]["frame"]))
    if LIMIT:
        by_ep = {e: by_ep[e] for e in sorted(by_ep)[:LIMIT]}
    print(f"[V] M={M_SAMPLES} seed={SEED} H={HORIZON} dt={dt}", flush=True)

    out = []
    bad_rd = 0
    for ep, idxs in sorted(by_ep.items()):
        bank = _bank(recs[idxs[0]]["scene"], dt)
        bank.reset()
        for i in idxs:
            rec = recs[i]
            envh = rec["env_humans"]
            bank.update({t: (float(envh[t][0]), float(envh[t][1]))
                         for t in range(envh.shape[0])})
            robot = _fs(rec["robot"])
            tids = rec["track_ids"].tolist()
            obs = rec["humans"]
            if not tids:
                continue
            if policy.action_space is None:
                policy.build_action_space(robot.v_pref)
            bh = [np.asarray(t, np.float32) for t in rec["history"]]
            if not bh:
                continue
            s34 = policy._build_joint_state_34(robot, [ObservableState(*[float(x) for x in o])
                                                       for o in obs])
            bh = bh + [_batch_joint34_to_tokens_vectorized(s34.reshape(1, -1))[0]]

            beliefs, futs, ok = [], [], True
            for t in tids:
                try:
                    tr = bank.tracker_for(int(t))
                    b = np.asarray(bank.belief_for(int(t)), np.float64)
                except Exception:
                    ok = False
                    break
                p = (float(envh[t][0]), float(envh[t][1]))
                futs.append(np.array([tr.roll_candidate_future(k, p, HORIZON)[-1]
                                      for k in range(len(b))]))
                beliefs.append(b)
            if not ok:
                continue
            top = min(5, len(beliefs))
            spread = max((float(np.sqrt((beliefs[s] * ((futs[s] -
                          (beliefs[s][:, None] * futs[s]).sum(0)) ** 2).sum(1)).sum()))
                          for s in range(top)), default=0.0)

            def mk(pa, observed_velocity):
                return [ObservableState(
                    float(pa[s][0]), float(pa[s][1]),
                    float(obs[s][2]) if observed_velocity else float((pa[s][0] - obs[s][0]) / dt),
                    float(obs[s][3]) if observed_velocity else float((pa[s][1] - obs[s][1]) / dt),
                    float(obs[s][4])) for s in range(len(tids))]

            hn = [ObservableState(*[float(x) for x in obs[s]]) for s in range(len(tids))]
            r_cv, v_cv, d_cv = score(policy, robot, [policy.propagate(h, ActionXY(h.vx, h.vy))
                                                     for h in hn], bh)
            a_cv = int(deployed(policy, r_cv, v_cv, d_cv).argmax())

            pm = np.array([(beliefs[s][:, None] * futs[s]).sum(0) for s in range(len(tids))])
            res = {}
            for tag, ov in (("MEAN-RV", False), ("MEAN-OV", True)):
                r_, v_, d_ = score(policy, robot, mk(pm, ov), bh)
                res[tag] = (r_, v_, d_, deployed(policy, r_, v_, d_))

            # common random numbers: draw the M worlds ONCE, reuse for RV and OV
            worlds = [np.array([futs[s][int(rng.choice(len(beliefs[s]), p=beliefs[s]))]
                                for s in range(len(tids))]) for _ in range(M_SAMPLES)]
            for tag, ov in (("FULL-RV", False), ("FULL-OV", True)):
                R = np.zeros(80); V = np.zeros(80); D = np.zeros(80); DEP = np.zeros(80)
                for w in worlds:
                    r_, v_, d_ = score(policy, robot, mk(w, ov), bh)
                    R += r_; V += v_; D += d_; DEP += deployed(policy, r_, v_, d_)
                res[tag] = (R / M_SAMPLES, V / M_SAMPLES, D / M_SAMPLES, DEP / M_SAMPLES)

            row = {"spread": spread, "crit": int(float(rec["dmin_all"]) < CRITICAL)}
            for fam in ("MEAN", "FULL"):
                rv, ov = res[f"{fam}-RV"], res[f"{fam}-OV"]
                # position is identical within a pair, so the geometric channels
                # must be bitwise equal; this is a self-check, not an assumption
                if float(np.abs(rv[0] - ov[0]).max()) > 1e-12 or \
                   float(np.abs(rv[2] - ov[2]).max()) > 1e-12:
                    bad_rd += 1
                aR, aO = int(rv[3].argmax()), int(ov[3].argmax())
                row[f"{fam}_dV"] = float(gamma * np.abs(ov[1] - rv[1]).max())
                row[f"{fam}_flip"] = int(aR != aO)
                row[f"{fam}_agree_rv"] = int(aR == a_cv)
                row[f"{fam}_agree_ov"] = int(aO == a_cv)
                row[f"{fam}_recov"] = (int(aO == a_cv) if aR != a_cv else -1)
            out.append(row)
        if (ep + 1) % 5 == 0:
            print(f"  ... episode {ep + 1}/{len(by_ep)}", flush=True)

    print(f"\n[V] {len(out)} states; R/dmin mismatches between RV and OV: {bad_rd} (must be 0)")
    a = {k: np.array([o.get(k, 0) for o in out], np.float64) for k in out[0]}
    q75 = np.percentile(a["spread"], 75)
    hdr = (f"\n{'subset':<14}{'fam':<6}{'n':>6}{'g*dV':>9}{'flip%':>8}"
           f"{'agreeRV%':>10}{'agreeOV%':>10}{'disagr drop%':>13}{'recovery%':>11}")
    print(hdr); print("-" * (len(hdr) - 1))
    res_json = []
    for name, m in (("all", np.ones(len(out), bool)),
                    ("spread Q4", a["spread"] > q75),
                    ("critical", a["crit"] > 0)):
        for fam in ("MEAN", "FULL"):
            if m.sum() == 0:
                continue
            dv = a[f"{fam}_dV"][m]
            fl = a[f"{fam}_flip"][m]
            arv = a[f"{fam}_agree_rv"][m]
            aov = a[f"{fam}_agree_ov"][m]
            rc = a[f"{fam}_recov"][m]
            rc = rc[rc >= 0]
            drv, dov = 1 - arv.mean(), 1 - aov.mean()
            drop = 100.0 * (drv - dov) / drv if drv > 0 else float("nan")
            recov = 100.0 * rc.mean() if rc.size else float("nan")
            print(f"{name:<14}{fam:<6}{int(m.sum()):>6}{dv.mean():>9.4f}"
                  f"{100*fl.mean():>8.1f}{100*arv.mean():>10.1f}{100*aov.mean():>10.1f}"
                  f"{drop:>13.1f}{recov:>11.1f}")
            res_json.append({"subset": name, "family": fam, "n": int(m.sum()),
                             "gdV": dv.mean(), "flip": float(fl.mean()),
                             "agree_rv": float(arv.mean()), "agree_ov": float(aov.mean()),
                             "disagreement_drop_pct": drop, "recovery_pct": recov})
    print("\ng*dV is the mean over states of the max-over-80-actions value change "
          "caused by the velocity field alone. flip% is how often that alone moves "
          "the argmax. agreeRV/agreeOV are agreement with the frozen CV action. "
          "disagr drop% is the relative reduction in CV-disagreement from RV to OV. "
          "recovery% = P(a_OV == a_CV | a_RV != a_CV).")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res_json, f, indent=1)
    print(f"[V] -> {OUT}")


_first = {}
_orig = MambaRLPolicy.predict_sarl_style


def _hook(self, state):
    if not _first:
        _first["p"] = self
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
        run(self)
        sys.stdout.flush()
        os._exit(0)
    return _orig(self, state)


def main():
    import runpy
    MambaRLPolicy.predict_sarl_style = _hook
    sys.argv = ["test.py"] + sys.argv[1:]
    runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "test.py"), run_name="__main__")


if __name__ == "__main__":
    main()
