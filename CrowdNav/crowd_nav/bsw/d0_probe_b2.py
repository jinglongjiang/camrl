"""D-0 measurement B: does a REAL, leakage-free posterior move the decision?

A established that the frozen Mamba-VL has a live human-future -> learned-value
-> action channel, using a deliberate +/-45 deg perturbation. That measured the
channel's bandwidth, not whether a genuine posterior produces a signal large
enough to use it. B answers that.

Everything is held fixed except how the assumed human future is formed:

    CV    each human continues at its current velocity        (= paper 1 exactly)
    MEAN  one world; each human placed at sum_k b_k * p_k     (expectation over POSITIONS)
    FULL  M joint worlds sampled from the same posteriors     (expectation over VALUES)

FULL and MEAN share the identical posterior and the identical per-candidate
rollouts. Their difference is only whether the expectation is taken inside or
outside the value function, i.e. exactly the Jensen gap. Averaging is never
applied to actions -- that is the SM-BRNE pathology this project already
rejected.

Leakage contract (verified in the source, not assumed):
  intent_tracker.py:16   a human's gx/gy MUST NEVER be passed to the tracker
  intent_tracker.py:289  the bank never accepts a Human object
The tracker and the scene provider are vendored read-only into bsw/vendor so
the sealed bayesian_dvl package is not imported; only their import lines differ
from the originals.
  candidate_fn(track_id, first_position) can only see public geometry
Candidate anchors come from the scene's PUBLIC parameters: circle_scene(radius)
for circle_crossing, square_scene(width) for square_crossing. A pedestrian's
true goal need not coincide with an anchor; it only has to lie in a region an
anchor represents. Hidden goals are never read here, not even for diagnostics.

The belief bank is keyed by the stable env.humans track_id, never by the
policy's TTC slot, which is re-sorted every frame.
"""
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
SEED_B = SEED_B_DEFAULT = 7919      # independent stream for the FULL' control
M_SAMPLES = int(os.environ.get("D0_M", "16"))
SEED = int(os.environ.get("D0_SEED", "12345"))
CRITICAL = 0.30
LIMIT = int(os.environ.get("D0_LIMIT", "0"))   # 0 = all states
HORIZON = 1          # H stays at 1 in D-0. Multi-step is explicitly out of scope.


def _fs(a):
    return FullState(*[float(x) for x in a])


def score_world(policy, robot, next_humans_fn, base_hist, gamma):
    """Score all 80 actions where next_humans_fn(action) supplies the assumed
    human successor states. Uses the policy's own reward/tokeniser/value."""
    seqs, rewards, dmins = [], [], []
    for act in policy.action_space:
        nxt_self = policy.propagate(robot, act)
        nxt_h = next_humans_fn()
        rewards.append(policy.compute_reward(nxt_self, nxt_h, prev_nav=robot, action=act))
        d = min((np.hypot(nxt_self.px - h.px, nxt_self.py - h.py)
                 - nxt_self.radius - h.radius) for h in nxt_h) if nxt_h else np.inf
        dmins.append(d)
        s34 = policy._build_joint_state_34(nxt_self, nxt_h)
        tok = _batch_joint34_to_tokens_vectorized(s34.reshape(1, -1))[0]
        seq = base_hist + [tok]
        seq = seq[-policy.seq_len:]
        if len(seq) < policy.seq_len:
            seq = [seq[0]] * (policy.seq_len - len(seq)) + seq
        seqs.append(np.array(seq))
    with torch.no_grad():
        v = policy.forward_value(
            torch.from_numpy(np.array(seqs)).float().to(policy.device)).cpu().numpy()
    return (np.array(rewards, np.float64), v.astype(np.float64).ravel(),
            np.array(dmins, np.float64))


def deployed(policy, r, v, d, gamma):
    raw = r + gamma * v
    dep = raw.copy()
    if policy.test_min_clearance > 0.0:
        m = d >= policy.test_min_clearance
        if m.any():
            dep = np.where(m, dep, -1e9)
    if policy.test_risk_lambda > 0.0:
        mg = policy.test_min_clearance if policy.test_min_clearance > 0.0 else policy.discomfort_dist
        dep = dep - policy.test_risk_lambda * np.clip(mg - d, 0.0, None)
    dep = dep.copy()
    dep[0] -= 1e-3
    return raw, dep


def _bank_for(scene_info, dt):
    sim = (scene_info.get("sim") or "").strip()
    if sim == "circle_crossing":
        sc = circle_scene(float(scene_info["circle_radius"]), n_sectors=8)
    elif sim == "square_crossing":
        sc = square_scene(float(scene_info["square_width"]), n_rows=4)
    else:
        raise ValueError(f"no public scene defined for sim={sim!r}")
    return IntentBeliefBank(candidate_fn=make_candidate_fn(sc), dt=dt, speed=1.0)


def _spread(b, f):
    """Spatial dispersion of one human's candidate one-step futures under its
    own posterior: sqrt(tr(Cov_{k~b}[p_k])), in metres.

    This replaces posterior entropy as the ambiguity variable. Entropy is the
    wrong quantity here: a flat posterior over candidates that all point the
    same way produces futures that coincide, so FULL and MEAN agree however
    uncertain the goal is. What separates FULL from MEAN is how far apart the
    hypothesised futures actually lie, which is what this measures.
    """
    mu = (b[:, None] * f).sum(0)
    return float(np.sqrt((b * ((f - mu) ** 2).sum(1)).sum()))


def run(policy):
    with open(STATES, "rb") as f:
        data = pickle.load(f)
    recs = data["records"]
    gamma = float(policy.gamma)
    rngA = np.random.default_rng(SEED)
    rngB = np.random.default_rng(SEED + SEED_B)
    dt = float(policy.time_step)
    print(f"[B2] {len(recs)} states | gamma={gamma} dt={dt} | M={M_SAMPLES} | H={HORIZON}")
    print(f"[B2] FULL seed={SEED}, FULL' seed={SEED + SEED_B} -- independent "
          f"streams, identical posterior. FULL vs FULL' is the Monte-Carlo "
          f"noise floor that FULL vs MEAN must clear.")

    by_ep = defaultdict(list)
    for i, r in enumerate(recs):
        by_ep[int(r["ep"])].append(i)
    if LIMIT:
        by_ep = {e: by_ep[e] for e in sorted(by_ep)[:LIMIT]}
        print(f"[B2] LIMIT set: first {LIMIT} episodes only")

    out = []
    skipped = 0
    for ep, idxs in sorted(by_ep.items()):
        idxs.sort(key=lambda i: int(recs[i]["frame"]))
        try:
            bank = _bank_for(recs[idxs[0]]["scene"], dt)
        except Exception:
            skipped += len(idxs)
            continue
        bank.reset()
        for i in idxs:
            rec = recs[i]
            envh = rec["env_humans"]
            bank.update({int(t): (float(envh[t][0]), float(envh[t][1]))
                         for t in range(envh.shape[0])})
            robot = _fs(rec["robot"])
            tids = rec["track_ids"].tolist()
            obs = rec["humans"]
            if not tids:
                continue
            if policy.action_space is None:
                policy.build_action_space(robot.v_pref)
            base_hist = [np.asarray(t, np.float32) for t in rec["history"]]
            if not base_hist:
                continue

            beliefs, futs, ok = [], [], True
            for t in tids:
                try:
                    tr = bank.tracker_for(int(t))
                    b = np.asarray(bank.belief_for(int(t)), np.float64)
                except Exception:
                    ok = False
                    break
                pos = (float(envh[t][0]), float(envh[t][1]))
                futs.append(np.array([tr.roll_candidate_future(k, pos, HORIZON)[-1]
                                      for k in range(len(b))]))
                beliefs.append(b)
            if not ok:
                skipped += 1
                continue

            top = min(5, len(beliefs))
            ent = [float(-(np.clip(b, 1e-12, None) * np.log(np.clip(b, 1e-12, None))).sum()
                         / np.log(len(b))) for b in beliefs[:top] if len(b) > 1]
            amb = float(np.max(ent)) if ent else 0.0
            spr = max((_spread(beliefs[s], futs[s]) for s in range(top)), default=0.0)

            def mk(pa):
                return [ObservableState(float(pa[s][0]), float(pa[s][1]),
                                        float((pa[s][0] - obs[s][0]) / dt),
                                        float((pa[s][1] - obs[s][1]) / dt),
                                        float(obs[s][4])) for s in range(len(tids))]

            humans_now = [ObservableState(*[float(x) for x in obs[s]])
                          for s in range(len(tids))]
            r_cv, v_cv, d_cv = score_world(
                policy, robot,
                lambda: [policy.propagate(h, ActionXY(h.vx, h.vy)) for h in humans_now],
                base_hist, gamma)
            _, dep_cv = deployed(policy, r_cv, v_cv, d_cv, gamma)

            pm = np.array([(beliefs[s][:, None] * futs[s]).sum(0) for s in range(len(tids))])
            hm = mk(pm)
            r_mn, v_mn, d_mn = score_world(policy, robot, lambda: hm, base_hist, gamma)
            _, dep_mn = deployed(policy, r_mn, v_mn, d_mn, gamma)

            arms = {}
            for tag, rng in (("A", rngA), ("B", rngB)):
                R = np.zeros(80); V = np.zeros(80); DEP = np.zeros(80)
                for _ in range(M_SAMPLES):
                    ps = np.array([futs[s][int(rng.choice(len(beliefs[s]), p=beliefs[s]))]
                                   for s in range(len(tids))])
                    hs = mk(ps)
                    a, b_, c = score_world(policy, robot, lambda: hs, base_hist, gamma)
                    _, dp = deployed(policy, a, b_, c, gamma)
                    R += a; V += b_; DEP += dp
                arms[tag] = (R / M_SAMPLES, V / M_SAMPLES, DEP / M_SAMPLES)

            RA, VA, DA = arms["A"]
            RB, VB, DB = arms["B"]
            JA, JB, JM = RA + gamma * VA, RB + gamma * VB, r_mn + gamma * v_mn
            srt = np.sort(dep_mn)[::-1]
            margin = float(srt[0] - srt[1])
            AS = np.array([[a.vx, a.vy] for a in policy.action_space])
            aA, aB, aM, aC = int(DA.argmax()), int(DB.argmax()), int(dep_mn.argmax()), int(dep_cv.argmax())
            out.append({
                "dJ_fm": float(np.abs(JA - JM).max()),
                "dJ_ff": float(np.abs(JA - JB).max()),
                "dV_fm": float(gamma * np.abs(VA - v_mn).max()),
                "dV_ff": float(gamma * np.abs(VA - VB).max()),
                "dR_fm": float(np.abs(RA - r_mn).max()),
                "flip_fm": int(aA != aM),
                "flip_ff": int(aA != aB),
                "flip_mc": int(aM != aC),
                "big_fm": int(np.linalg.norm(AS[aA] - AS[aM]) > 0.39),
                "big_ff": int(np.linalg.norm(AS[aA] - AS[aB]) > 0.39),
                "margin": margin,
                "amb": amb,
                "spread": spr,
                "critical": int(rec["dmin_all"] < CRITICAL),
            })
        if (ep + 1) % 5 == 0:
            print(f"  ... episode {ep + 1}/{len(by_ep)}", flush=True)

    a = {k: np.array([o[k] for o in out], np.float64) for k in out[0]}
    print(f"\n[B2] {len(out)} states scored, {skipped} skipped")
    print(f"[B2] spread (m): p25={np.percentile(a['spread'],25):.4f} "
          f"p50={np.percentile(a['spread'],50):.4f} p75={np.percentile(a['spread'],75):.4f} "
          f"max={a['spread'].max():.4f}")

    def block(name, m):
        if m.sum() == 0:
            print(f"{name:<22}{0:>6}")
            return
        s = lambda k: a[k][m]
        print(f"{name:<22}{int(m.sum()):>6}"
              f"{np.median(s('dJ_fm')):>10.4f}{np.median(s('dJ_ff')):>10.4f}"
              f"{np.median(s('dV_fm')):>10.4f}{np.median(s('dR_fm')):>9.4f}"
              f"{np.median(s('margin')):>10.5f}"
              f"{100*s('flip_fm').mean():>9.1f}{100*s('flip_ff').mean():>9.1f}"
              f"{100*s('big_fm').mean():>9.1f}{100*s('big_ff').mean():>9.1f}"
              f"{100*s('flip_mc').mean():>9.1f}")

    hdr = (f"\n{'subset':<22}{'n':>6}{'dJ_FM':>10}{'dJ_FF':>10}{'dV_FM':>10}"
           f"{'dR_FM':>9}{'margin':>10}{'flipFM%':>9}{'flipFF%':>9}"
           f"{'bigFM%':>9}{'bigFF%':>9}{'MNvsCV%':>9}")
    print(hdr); print("-" * len(hdr))
    n = len(out)
    block("all", np.ones(n, bool))
    q = np.percentile(a["spread"], [25, 50, 75])
    block("spread Q1 (least)", a["spread"] <= q[0])
    block("spread Q2", (a["spread"] > q[0]) & (a["spread"] <= q[1]))
    block("spread Q3", (a["spread"] > q[1]) & (a["spread"] <= q[2]))
    block("spread Q4 (most)", a["spread"] > q[2])
    block("entropy high", a["amb"] >= 0.5)
    block("entropy low", a["amb"] < 0.5)
    block("critical", a["critical"] > 0)
    block("critical & spreadQ4", (a["critical"] > 0) & (a["spread"] > q[2]))
    print("\nFM = FULL vs MEAN (the signal). FF = FULL vs FULL', same posterior, "
          "independent sampling stream (the Monte-Carlo noise floor). The FM "
          "column is only meaningful to the extent it exceeds the FF column.")
    print("spread = sqrt(tr(Cov_k[candidate one-step futures])) over the humans "
          "the value network can see; quartiles are of that quantity.")
    print("big* = the argmax moved more than 0.39 m/s, one heading step on the "
          "fastest ring. MNvsCV% = how often the posterior moves paper 1 at all.")


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
