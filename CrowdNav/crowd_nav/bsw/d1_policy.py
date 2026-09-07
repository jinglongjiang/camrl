"""D-1: closed-loop FULL / MEAN / CV on the frozen paper-1 controller.

D-0 established that the frozen value-lookahead controller perceives the
assumed human future through its learned value, that a real leakage-free
posterior produces a decision difference exceeding its own estimator noise, and
that the difference grows with the spatial spread of the candidate futures.
D-1 asks the only remaining question: does that difference produce better
navigation.

The three arms differ in exactly one thing, the assumed human successor state:

    cv    every human continues at its current velocity. This arm calls the
          frozen policy's own method untouched, so it is bit-identical to
          paper 1 and doubles as that paper's reproduction.
    mean  one world; every human placed at sum_k b_k * p_k, the posterior-
          weighted mean of its candidate one-step futures.
    full  M joint worlds drawn from the same posteriors; the expectation is
          taken over VALUES, never over actions.

Everything else is shared and frozen: weights, action grid, reward, history
length, the deployed safety layer, the stop tie-break, and action smoothing.
Any difference between arms therefore cannot be attributed to the backbone.

Leakage: the belief bank is fed observed positions and public scene anchors
only, keyed by the stable env.humans index. A pedestrian's gx/gy is never read.
The value network's top-K still follows the time-to-collision ranking, which is
a separate identity system and is left alone.
"""
import json
import os
import sys

import numpy as np
import torch

from crowd_nav.bsw.vendor.intent_tracker import IntentBeliefBank
from crowd_nav.bsw.vendor.scene_candidates import (
    circle_scene, make_candidate_fn, square_scene)
from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import ObservableState

ARM = os.environ.get("D1_ARM", "cv").strip().lower()
M_SAMPLES = int(os.environ.get("D1_M", "16"))
SEED = int(os.environ.get("D1_SEED", "12345"))
HORIZON = 1

_orig_predict = MambaRLPolicy.predict_sarl_style
_orig_reset = CrowdSim.reset
_orig_step = CrowdSim.step

_st = {"env": None, "bank": None, "rng": np.random.default_rng(SEED),
       "spreads": [], "n_steps": 0, "n_nobank": 0,
       # per-episode outcomes, so arms can be compared on the SAME episode
       # rather than only on aggregate rates
       "episodes": [], "cur": None}
EPOUT = os.environ.get("D1_EPOUT", "")


def _scene_bank(env, dt):
    sim = (getattr(env, "test_sim", "") or "").strip()
    if sim == "circle_crossing":
        sc = circle_scene(float(env.circle_radius), n_sectors=8)
    elif sim == "square_crossing":
        sc = square_scene(float(env.square_width), n_rows=4)
    else:
        raise ValueError(f"no public scene defined for sim={sim!r}")
    return IntentBeliefBank(candidate_fn=make_candidate_fn(sc), dt=dt, speed=1.0)


def _close_episode():
    c = _st["cur"]
    if c and c["n"]:
        c["spread_med"] = float(np.median(c.pop("spreads"))) if c["spreads_any"] else 0.0
        _st["episodes"].append(c)


def _patched_reset(self, *a, **kw):
    _close_episode()
    _st["env"] = self
    _st["cur"] = {"case": f"{getattr(self,'test_sim','?')}/{getattr(self,'human_num','?')}",
                  "radius": getattr(self, "circle_radius", None),
                  "width": getattr(self, "square_width", None),
                  "idx": len(_st["episodes"]), "n": 0, "outcome": None,
                  "spreads": [], "spreads_any": False}
    if ARM != "cv":
        _st["bank"] = _scene_bank(self, float(getattr(self, "time_step", 0.25)))
    return _orig_reset(self, *a, **kw)


def _patched_step(self, *a, **kw):
    out = _orig_step(self, *a, **kw)
    c = _st["cur"]
    if c is not None:
        c["n"] += 1
        try:
            info = out[4]
            c["outcome"] = info.get("event") if isinstance(info, dict) else str(info)[:40]
        except Exception:
            pass
    return out


def _track_ids(env, policy_humans):
    """Map each TTC-sorted observation back to its stable env.humans index by
    exact position match. The belief bank must follow the person, not the slot,
    which is re-sorted every frame."""
    table = {(float(h.px), float(h.py)): i for i, h in enumerate(env.humans)}
    return [table.get((float(h.px), float(h.py)), -1) for h in policy_humans]


def _worlds(policy, state, env):
    """Return a list of assumed successor human-state lists, one per world, and
    the spatial spread of the candidate futures the value network can see."""
    bank = _st["bank"]
    dt = float(policy.time_step)
    humans = list(state.human_states)
    tids = _track_ids(env, humans)
    if bank is None or any(t < 0 for t in tids):
        _st["n_nobank"] += 1
        return None, 0.0
    envh = env.humans
    bank.update({i: (float(envh[i].px), float(envh[i].py)) for i in range(len(envh))})

    beliefs, futs = [], []
    for t in tids:
        try:
            tr = bank.tracker_for(int(t))
            b = np.asarray(bank.belief_for(int(t)), np.float64)
        except Exception:
            _st["n_nobank"] += 1
            return None, 0.0
        pos = (float(envh[t].px), float(envh[t].py))
        futs.append(np.array([tr.roll_candidate_future(k, pos, HORIZON)[-1]
                              for k in range(len(b))]))
        beliefs.append(b)

    top = min(5, len(beliefs))
    spread = 0.0
    for s in range(top):
        mu = (beliefs[s][:, None] * futs[s]).sum(0)
        spread = max(spread, float(np.sqrt((beliefs[s] * ((futs[s] - mu) ** 2).sum(1)).sum())))

    def mk(pa):
        return [ObservableState(float(pa[s][0]), float(pa[s][1]),
                                float((pa[s][0] - humans[s].px) / dt),
                                float((pa[s][1] - humans[s].py) / dt),
                                float(humans[s].radius)) for s in range(len(humans))]

    if ARM == "mean":
        return [mk(np.array([(beliefs[s][:, None] * futs[s]).sum(0)
                             for s in range(len(humans))]))], spread
    rng = _st["rng"]
    return [mk(np.array([futs[s][int(rng.choice(len(beliefs[s]), p=beliefs[s]))]
                         for s in range(len(humans))]))
            for _ in range(M_SAMPLES)], spread


def _hook(self, state):
    # cv is the frozen paper-1 path, run untouched
    if ARM == "cv":
        return _orig_predict(self, state)
    if self.reach_destination(state):
        return ActionXY(0, 0)
    if self.action_space is None:
        self.build_action_space(state.self_state.v_pref)

    env = _st["env"]
    worlds, spread = _worlds(self, state, env) if env is not None else (None, 0.0)
    if worlds is None:
        # fail open to the frozen behaviour rather than inventing one
        return _orig_predict(self, state)
    _st["spreads"].append(spread)
    if _st["cur"] is not None:
        _st["cur"]["spreads"].append(spread)
        _st["cur"]["spreads_any"] = True
    _st["n_steps"] += 1

    cur34 = self._build_joint_state_34(state.self_state, state.human_states)
    cur_tok = _batch_joint34_to_tokens_vectorized(cur34.reshape(1, -1))[0]
    base = list(self._history) + [cur_tok]
    n_a = len(self.action_space)

    # the robot successor depends only on the action, not on the world
    selves = [self.propagate(state.self_state, act) for act in self.action_space]
    s34s, rews, dmins = [], [], []
    for w in worlds:
        hp = np.array([[h.px, h.py, h.radius] for h in w]) if w else np.zeros((0, 3))
        for act, nxt_self in zip(self.action_space, selves):
            # compute_reward reads `action` for the stand penalty; it must be
            # the real action, not None.
            rews.append(self.compute_reward(nxt_self, w, prev_nav=state.self_state,
                                            action=act))
            if hp.shape[0]:
                dd = (np.hypot(nxt_self.px - hp[:, 0], nxt_self.py - hp[:, 1])
                      - nxt_self.radius - hp[:, 2])
                dmins.append(float(dd.min()))
            else:
                dmins.append(np.inf)
            s34s.append(self._build_joint_state_34(nxt_self, w))
    # one batched tokenisation instead of 1280 single-row calls
    toks = _batch_joint34_to_tokens_vectorized(np.asarray(s34s, dtype=np.float32))

    # every sequence shares the same history; only the final frame differs
    hist = base[-(self.seq_len - 1):] if self.seq_len > 1 else []
    if len(hist) < self.seq_len - 1:
        hist = [hist[0] if hist else toks[0]] * (self.seq_len - 1 - len(hist)) + hist
    H = np.asarray(hist, dtype=np.float32)
    T = np.asarray(toks, dtype=np.float32)
    seqs = np.concatenate(
        [np.broadcast_to(H, (T.shape[0],) + H.shape), T[:, None]], axis=1)
    with torch.no_grad():
        v = self.forward_value(torch.from_numpy(seqs).float().to(self.device)).cpu().numpy()

    r = np.asarray(rews, np.float64).reshape(len(worlds), n_a)
    d = np.asarray(dmins, np.float64).reshape(len(worlds), n_a)
    v = np.asarray(v, np.float64).reshape(len(worlds), n_a)

    gamma = self.gamma
    if self.lookahead_ablation_mode == "reward_only":
        tv = r
    elif self.lookahead_ablation_mode == "value_only":
        tv = gamma * v
    else:
        tv = r + gamma * v
    if self._phase in ("test", "val", "eval") and (
            self.test_min_clearance > 0.0 or self.test_risk_lambda > 0.0):
        if self.test_min_clearance > 0.0:
            m = d >= self.test_min_clearance
            keep = m.any(axis=1)
            tv = np.where(m | ~keep[:, None], tv, -1e9)
        if self.test_risk_lambda > 0.0:
            mg = self.test_min_clearance if self.test_min_clearance > 0.0 else self.discomfort_dist
            tv = tv - self.test_risk_lambda * np.clip(mg - d, 0.0, None)
    # expectation over VALUES across worlds, never over actions
    score = tv.mean(axis=0)
    score[0] -= 1e-3
    best = self.action_space[int(score.argmax())]

    self._history.append(cur_tok)
    if self._phase in ("test", "val", "eval") and self.test_action_smoothing > 0.0:
        if self._last_action is not None:
            al = float(self.test_action_smoothing)
            best = ActionXY(al * self._last_action.vx + (1 - al) * best.vx,
                            al * self._last_action.vy + (1 - al) * best.vy)
        self._last_action = best
    return best


def main():
    import runpy
    if ARM not in ("cv", "mean", "full"):
        raise SystemExit(f"D1_ARM must be cv/mean/full, got {ARM!r}")
    print(f"[D1] arm={ARM} M={M_SAMPLES if ARM == 'full' else 1} seed={SEED} H={HORIZON}",
          flush=True)
    MambaRLPolicy.predict_sarl_style = _hook
    CrowdSim.reset = _patched_reset
    CrowdSim.step = _patched_step
    try:
        sys.argv = ["test.py"] + sys.argv[1:]
        runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "test.py"), run_name="__main__")
    finally:
        _close_episode()
        if EPOUT:
            os.makedirs(os.path.dirname(EPOUT), exist_ok=True)
            with open(EPOUT, "w") as f:
                json.dump({"arm": ARM, "M": M_SAMPLES, "seed": SEED,
                           "episodes": _st["episodes"]}, f)
            print(f"[D1] per-episode outcomes -> {EPOUT} ({len(_st['episodes'])} episodes)")
        sp = np.array(_st["spreads"]) if _st["spreads"] else np.zeros(1)
        print(f"\n[D1] arm={ARM} steps_with_posterior={_st['n_steps']} "
              f"fell_back_to_frozen={_st['n_nobank']}")
        print(f"[D1] spread p25={np.percentile(sp,25):.4f} "
              f"p50={np.percentile(sp,50):.4f} p75={np.percentile(sp,75):.4f}")


if __name__ == "__main__":
    main()
