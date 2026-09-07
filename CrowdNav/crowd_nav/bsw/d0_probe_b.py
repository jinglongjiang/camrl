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


def run(policy):
    with open(STATES, "rb") as f:
        data = pickle.load(f)
    recs = data["records"]
    gamma = float(policy.gamma)
    rng = np.random.default_rng(SEED)
    dt = float(policy.time_step)
    print(f"[B] {len(recs)} states | gamma={gamma} dt={dt} | FULL uses M={M_SAMPLES} "
          f"joint posterior samples | H={HORIZON}")
    print(f"[B] test_min_clearance={policy.test_min_clearance} "
          f"test_risk_lambda={policy.test_risk_lambda}")

    by_ep = defaultdict(list)
    for i, r in enumerate(recs):
        by_ep[int(r["ep"])].append(i)
    if LIMIT:
        keep = sorted(by_ep)[:LIMIT]
        by_ep = {e: by_ep[e] for e in keep}
        print(f"[B] LIMIT set: first {LIMIT} episodes only")

    rows = {k: [] for k in ("all", "high_amb", "low_amb", "critical",
                            "critical_high_amb")}
    ents = []
    skipped = 0

    for ep, idxs in sorted(by_ep.items()):
        idxs.sort(key=lambda i: int(recs[i]["frame"]))
        try:
            bank = _bank_for(recs[idxs[0]]["scene"], dt)
        except Exception as e:
            skipped += len(idxs)
            continue
        bank.reset()
        for i in idxs:
            rec = recs[i]
            envh = rec["env_humans"]
            # feed the posterior by STABLE identity, in frame order
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

            # per-human posterior and per-candidate one-step futures
            beliefs, futs, ok = [], [], True
            for slot, t in enumerate(tids):
                try:
                    tr = bank.tracker_for(int(t))
                    b = np.asarray(bank.belief_for(int(t)), np.float64)
                except Exception:
                    ok = False
                    break
                pos = (float(envh[t][0]), float(envh[t][1]))
                f = np.array([tr.roll_candidate_future(k, pos, HORIZON)[-1]
                              for k in range(len(b))])
                beliefs.append(b)
                futs.append(f)
            if not ok:
                skipped += 1
                continue

            # ambiguity of the humans the value network can actually see
            top = min(5, len(beliefs))
            e = [float(-(np.clip(b, 1e-12, None) * np.log(np.clip(b, 1e-12, None))).sum()
                       / np.log(len(b))) for b in beliefs[:top] if len(b) > 1]
            amb = float(np.max(e)) if e else 0.0
            ents.append(amb)

            def mk(pos_arr):
                return [ObservableState(float(pos_arr[s][0]), float(pos_arr[s][1]),
                                        float((pos_arr[s][0] - obs[s][0]) / dt),
                                        float((pos_arr[s][1] - obs[s][1]) / dt),
                                        float(obs[s][4])) for s in range(len(tids))]

            # ---- CV (paper 1 exactly) ----
            humans_now = [ObservableState(*[float(x) for x in obs[s]])
                          for s in range(len(tids))]
            r_cv, v_cv, d_cv = score_world(
                policy, robot,
                lambda: [policy.propagate(h, ActionXY(h.vx, h.vy)) for h in humans_now],
                base_hist, gamma)
            _, dep_cv = deployed(policy, r_cv, v_cv, d_cv, gamma)

            # ---- MEAN: expectation over positions ----
            pm = np.array([(beliefs[s][:, None] * futs[s]).sum(0)
                           for s in range(len(tids))])
            hm = mk(pm)
            r_mn, v_mn, d_mn = score_world(policy, robot, lambda: hm, base_hist, gamma)
            raw_mn, dep_mn = deployed(policy, r_mn, v_mn, d_mn, gamma)

            # ---- FULL: expectation over values, same posterior ----
            R = np.zeros(80); V = np.zeros(80); D = np.zeros(80); DEP = np.zeros(80)
            for _ in range(M_SAMPLES):
                ps = np.array([futs[s][int(rng.choice(len(beliefs[s]), p=beliefs[s]))]
                               for s in range(len(tids))])
                hs = mk(ps)
                a, b_, c = score_world(policy, robot, lambda: hs, base_hist, gamma)
                rw, dp = deployed(policy, a, b_, c, gamma)
                R += a; V += b_; D += c; DEP += dp
            R /= M_SAMPLES; V /= M_SAMPLES; D /= M_SAMPLES; DEP /= M_SAMPLES

            srt = np.sort(dep_mn)[::-1]
            margin = float(srt[0] - srt[1])
            AS = np.array([[a.vx, a.vy] for a in policy.action_space])
            a_mn, a_fl, a_cv = int(dep_mn.argmax()), int(DEP.argmax()), int(dep_cv.argmax())
            row = (float(np.abs(R - r_mn).max()),
                   float(gamma * np.abs(V - v_mn).max()),
                   float(np.abs((R + gamma * V) - (r_mn + gamma * v_mn)).max()),
                   float(np.abs(D - d_mn).max()),
                   margin,
                   int(a_fl != a_mn),
                   float(np.linalg.norm(AS[a_fl] - AS[a_mn])),
                   int(a_mn != a_cv),
                   amb)
            rows["all"].append(row)
            rows["high_amb" if amb >= 0.5 else "low_amb"].append(row)
            if rec["dmin_all"] < CRITICAL:
                rows["critical"].append(row)
                if amb >= 0.5:
                    rows["critical_high_amb"].append(row)
        if (ep + 1) % 5 == 0:
            print(f"  ... episode {ep + 1}/{len(by_ep)}", flush=True)

    print(f"\n[B] skipped {skipped} states; posterior entropy median="
          f"{np.median(ents) if ents else float('nan'):.3f}")
    hdr = (f"\n{'subset':<20}{'n':>6}{'dR':>9}{'g*dV':>9}{'dJ':>9}{'dJ_p90':>9}"
           f"{'d_dmin':>9}{'margin':>10}{'dJ/mg':>9}{'flip%':>7}{'flipdst':>9}"
           f"{'bigflip%':>9}{'mnVScv%':>9}")
    print(hdr)
    print("-" * len(hdr))
    for k in ("all", "high_amb", "low_amb", "critical", "critical_high_amb"):
        a = np.array(rows[k], np.float64)
        if not a.size:
            print(f"{k:<20}{0:>6}")
            continue
        dR, dV, dJ, dD, mg, fl, fd, cvf, am = (a[:, c] for c in range(9))
        f1 = fd[fl > 0]
        print(f"{k:<20}{len(a):>6}{np.median(dR):>9.4f}{np.median(dV):>9.4f}"
              f"{np.median(dJ):>9.4f}{np.percentile(dJ,90):>9.4f}{np.median(dD):>9.4f}"
              f"{np.median(mg):>10.5f}{np.median(dJ/np.maximum(mg,1e-12)):>9.2f}"
              f"{100*fl.mean():>7.1f}{(np.median(f1) if f1.size else 0.0):>9.3f}"
              f"{100*float((fd>0.39).mean()):>9.1f}{100*cvf.mean():>9.1f}")
    print("\nFULL vs MEAN. dR/g*dV/dJ/d_dmin are max-over-80-actions per state, "
          "medians across states. flip% = argmax(FULL) != argmax(MEAN) under the "
          "deployed score. mnVScv% = argmax(MEAN) != argmax(CV), i.e. how much "
          "the posterior moves paper 1 at all. high_amb = normalised posterior "
          "entropy >= 0.5 among the humans the value network can see.")


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
