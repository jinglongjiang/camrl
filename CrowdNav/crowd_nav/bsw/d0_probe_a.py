"""D-0 measurement A: three-channel sensitivity of the frozen Mamba-VL.

Question: when the ASSUMED human future changes, does the frozen controller's
action score move, and through which channel?

    channel 1  R      compute_reward(next_self, next_humans) -- geometric,
                      no network, sees EVERY human
    channel 2  V      forward_value on the successor token -- sees only
                      human_states[:5] (mamba_rl.py:737), 1 of T frames
    channel 3  dmin   test-time safety mask / risk penalty -- geometric,
                      sees EVERY human, and can delete an action outright

The probe never runs the simulator. It replays recorded on-policy states and
calls the policy's OWN propagate / compute_reward / _build_joint_state_34 /
forward_value, so the numbers come from the deployed code path.

Stratification (K=5 visibility is not an edge case: 86.2% of recorded states
have more than five humans):
    visible    perturb a human at env-list index < 5, which V can see
    invisible  perturb a human at index >= 5, which V cannot see
A near-zero gamma*dV in the invisible group is an interface fact, not evidence
that V ignores people. Only the visible group tests channel 2.
"""
import os
import pickle
import sys

import numpy as np
import torch

from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState

STATES = os.environ.get("D0_STATES", "/root/bdvl_diagnostics/d0/states_t24.pkl")
ROT_DEG = [float(x) for x in os.environ.get("D0_ROT", "45").split(",")]
CRITICAL = 0.30


def _fs(a):
    return FullState(*[float(x) for x in a])


def _os_list(h):
    return [ObservableState(*[float(x) for x in r]) for r in h]


def _rotate(h, deg):
    """Speed-preserving heading change: what a different intent looks like at
    the next step. Position follows from the policy's own propagate()."""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return ObservableState(h.px, h.py, c * h.vx - s * h.vy, s * h.vx + c * h.vy, h.radius)


def score_all(policy, robot, humans, base_hist, gamma):
    """Reproduce predict_sarl_style's scoring for all 80 actions."""
    n = len(policy.action_space)
    seqs, rewards, dmins = [], [], []
    for act in policy.action_space:
        nxt_self = policy.propagate(robot, act)
        nxt_h = [policy.propagate(h, ActionXY(h.vx, h.vy)) for h in humans]
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
    r = np.array(rewards, dtype=np.float64)
    d = np.array(dmins, dtype=np.float64)
    return r, v.astype(np.float64).reshape(n), d


def deployed_score(policy, r, v, d, gamma):
    """Raw score, then the test-time safety layer, exactly as deployed."""
    raw = r + gamma * v
    dep = raw.copy()
    mask = np.ones_like(dep, dtype=bool)
    if policy.test_min_clearance > 0.0:
        mask = d >= policy.test_min_clearance
        if mask.any():
            dep = np.where(mask, dep, -1e9)
    if policy.test_risk_lambda > 0.0:
        m = policy.test_min_clearance if policy.test_min_clearance > 0.0 else policy.discomfort_dist
        dep = dep - policy.test_risk_lambda * np.clip(m - d, 0.0, None)
    dep = dep.copy()
    dep[0] -= 1e-3          # the STOP tie-break
    return raw, dep, mask


def run(policy):
    with open(STATES, "rb") as f:
        data = pickle.load(f)
    recs = data["records"]
    gamma = float(policy.gamma)
    print(f"[A] {len(recs)} states | seq_len={policy.seq_len} | gamma={gamma}")
    print(f"[A] test_min_clearance={policy.test_min_clearance} "
          f"test_risk_lambda={policy.test_risk_lambda} "
          f"discomfort_dist={policy.discomfort_dist}")
    print(f"[A] perturbation: heading rotation of +/-{ROT_DEG} deg, speed preserved")

    rows = {k: [] for k in ("all", "visible", "invisible", "critical",
                            "critical_visible", "critical_invisible")}
    for i, rec in enumerate(recs):
        robot = _fs(rec["robot"])
        humans = _os_list(rec["humans"])
        if not humans:
            continue
        if policy.action_space is None:
            policy.build_action_space(robot.v_pref)
        base_hist = [np.asarray(t, dtype=np.float32) for t in rec["history"]]
        if not base_hist:
            continue
        s34 = policy._build_joint_state_34(robot, humans)
        cur = _batch_joint34_to_tokens_vectorized(s34.reshape(1, -1))[0]
        bh = base_hist + [cur]

        r0, v0, d0 = score_all(policy, robot, humans, bh, gamma)
        raw0, dep0, m0 = deployed_score(policy, r0, v0, d0, gamma)
        srt = np.sort(dep0)[::-1]
        margin = float(srt[0] - srt[1])
        a0_dep = int(dep0.argmax())
        a0_raw = int(raw0.argmax())
        # how degenerate is the top of the ranking? a flip between two actions
        # that are nearly the same velocity is not a decision change.
        ties = int((dep0 >= srt[0] - 1e-3).sum())
        AS = np.array([[a.vx, a.vy] for a in policy.action_space])

        # nearest human within each visibility group
        dist = np.hypot(rec["robot"][0] - rec["humans"][:, 0],
                        rec["robot"][1] - rec["humans"][:, 1])
        groups = {}
        vis = [j for j in range(len(humans)) if j < 5]
        inv = [j for j in range(len(humans)) if j >= 5]
        if vis:
            groups["visible"] = int(min(vis, key=lambda j: dist[j]))
        if inv:
            groups["invisible"] = int(min(inv, key=lambda j: dist[j]))

        for gname, j in groups.items():
            dR = dV = dD = 0.0
            flip_dep = flip_raw = mask_ch = 0
            flip_d = 0.0
            for deg in ROT_DEG:
                for sgn in (+1.0, -1.0):
                    hp = list(humans)
                    hp[j] = _rotate(humans[j], sgn * deg)
                    r1, v1, d1 = score_all(policy, robot, hp, bh, gamma)
                    raw1, dep1, m1 = deployed_score(policy, r1, v1, d1, gamma)
                    dR = max(dR, float(np.abs(r1 - r0).max()))
                    dV = max(dV, float(gamma * np.abs(v1 - v0).max()))
                    dD = max(dD, float(np.abs(d1 - d0).max()))
                    mask_ch = max(mask_ch, int((m1 != m0).any()))
                    a1 = int(dep1.argmax())
                    flip_dep = max(flip_dep, int(a1 != a0_dep))
                    flip_raw = max(flip_raw, int(raw1.argmax() != a0_raw))
                    flip_d = max(flip_d, float(np.linalg.norm(AS[a1] - AS[a0_dep])))
            row = (dR, dV, dD, mask_ch, margin, flip_dep, flip_raw, flip_d, ties)
            rows[gname].append(row)
            rows["all"].append(row)
            if rec["dmin_all"] < CRITICAL:
                rows["critical"].append(row)
                rows["critical_" + gname].append(row)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(recs)}", flush=True)

    hdr = (f"\n{'subset':<22}{'n':>6}{'dR':>9}{'dR_p90':>9}{'g*dV':>9}{'gdV_p90':>9}"
           f"{'d_dmin':>8}{'dD_p90':>8}{'mask%':>7}{'margin':>10}{'dV/mg':>9}"
           f"{'flip%':>7}{'fliprw%':>8}{'flipdst':>9}{'bigflip%':>9}{'ties':>7}")
    print(hdr)
    print("-" * len(hdr))
    for k in ("all", "visible", "invisible", "critical",
              "critical_visible", "critical_invisible"):
        a = np.array(rows[k], dtype=np.float64)
        if not a.size:
            print(f"{k:<22}{0:>6}")
            continue
        dR, dV, dD, mk, mg, fd, fr, fdist, ties = (a[:, c] for c in range(9))
        ratio = np.median(dV / np.maximum(mg, 1e-12))
        fl = fdist[fd > 0]
        fmed = np.median(fl) if fl.size else 0.0
        # 0.39 m/s is the heading-neighbour spacing on the fastest ring; a flip
        # below it is at most one grid step, i.e. a tie broken, not a decision
        # changed.
        big = 100.0 * float((fdist > 0.39).mean())
        print(f"{k:<22}{len(a):>6}{np.median(dR):>9.4f}{np.percentile(dR,90):>9.4f}"
              f"{np.median(dV):>9.4f}{np.percentile(dV,90):>9.4f}"
              f"{np.median(dD):>8.4f}{np.percentile(dD,90):>8.4f}"
              f"{100 * mk.mean():>7.1f}{np.median(mg):>10.5f}"
              f"{ratio:>9.2f}{100 * fd.mean():>7.1f}{100 * fr.mean():>8.1f}"
              f"{fmed:>9.3f}{big:>9.1f}{np.median(ties):>7.0f}")
    print("\nmedians except mask%/flip%/bigflip% which are rates. "
          "dR/g*dV/d_dmin are max-over-80-actions per state.")
    print("flipdst = |v_perturbed - v_baseline| of the chosen action, "
          "conditioned on a flip actually happening. bigflip% = share of ALL "
          "states whose flip exceeds 0.39 m/s, the heading-neighbour spacing "
          "on the fastest ring -- below that a flip is at most one grid step.")
    print("ties = number of actions within 1e-3 of the top score.")
    print("flip% uses the deployed score (safety mask + tie-break on); "
          "flip_raw% uses R + gamma*V only, so the gap between them is the "
          "part of the decision change that comes from the safety filter.")


_captured = {}
_orig = MambaRLPolicy.predict_sarl_style


def _hook(self, state):
    """Capture the fully-configured deployed policy, then run the probe."""
    if not _captured:
        _captured["p"] = self
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
