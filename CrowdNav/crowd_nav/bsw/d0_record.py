"""D-0 state collector (v2: identity-carrying, for the Bayesian posterior).

Records the states the frozen Mamba-VL actually decides on, by hooking its own
``predict_sarl_style``. The hook calls the original method and returns its
action unchanged, so the trajectory is bit-identical to a normal run and no
paper-1 code is modified.

v2 exists because a recursive goal posterior needs IDENTITY CONTINUITY. The
evaluation path (test.py:532-559) re-sorts pedestrians by time-to-collision
before every decision, so a pedestrian's slot in ``state.human_states`` changes
from frame to frame. Feeding the belief bank by slot index would splice
different people's trajectories together -- exactly what intent_tracker.py:302
forbids ("NEVER by list position"). So each observation is mapped back to its
stable ``env.humans`` index, and THAT is the track_id.

Two identity systems are kept deliberately separate:
    track_id   stable per-pedestrian identity, follows the person; the belief
               bank is keyed by this
    ttc_order  the TTC ranking the policy uses; ``ttc_order[:5]`` are the only
               track_ids whose future reaches the value network at all

Every outcome is kept. Near-misses and collisions are the informative samples;
keeping only successes would bias the probe toward open-field cruising.
"""
import os
import pickle
import runpy
import sys

import numpy as np

from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim

OUT = os.environ.get("D0_OUT", "/root/bdvl_diagnostics/d0/states_v2.pkl")
MAX_STATES = int(os.environ.get("D0_MAX_STATES", "20000"))

_records = []
_episodes = []
_cur = {"steps": [], "infos": []}
_env = {}
_mismatch = {"n": 0}

_orig_predict = MambaRLPolicy.predict_sarl_style
_orig_reset = CrowdSim.reset
_orig_step = CrowdSim.step


def _robot_arr(s):
    return np.array([s.px, s.py, s.vx, s.vy, s.radius,
                     s.gx, s.gy, s.v_pref, s.theta], dtype=np.float32)


def _obs_arr(hs):
    if not hs:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array([[h.px, h.py, h.vx, h.vy, h.radius] for h in hs],
                    dtype=np.float32)


def _track_ids(policy_humans):
    """Map each observation the policy received back to its stable env.humans
    index by exact position match. Exact is correct here: the observation is
    built from the same float coordinates (obs-noise-std defaults to 0). A
    miss is counted, never silently approximated."""
    env = _env.get("e")
    if env is None or not hasattr(env, "humans"):
        return np.full(len(policy_humans), -1, dtype=np.int32)
    table = {}
    for i, h in enumerate(env.humans):
        table[(float(h.px), float(h.py))] = i
    out = []
    for h in policy_humans:
        k = (float(h.px), float(h.py))
        if k in table:
            out.append(table[k])
        else:
            _mismatch["n"] += 1
            out.append(-1)
    return np.array(out, dtype=np.int32)


def _scene_info():
    env = _env.get("e")
    if env is None:
        return {}
    return {"sim": getattr(env, "test_sim", None),
            "circle_radius": getattr(env, "circle_radius", None),
            "square_width": getattr(env, "square_width", None),
            "human_num": getattr(env, "human_num", None),
            "time_step": getattr(env, "time_step", None)}


def _patched_predict(self, state):
    hist = (np.array([np.asarray(t, dtype=np.float32) for t in self._history],
                     dtype=np.float32)
            if len(self._history) else np.zeros((0, 8, 13), np.float32))
    r = _robot_arr(state.self_state)
    h = _obs_arr(state.human_states)
    tid = _track_ids(state.human_states)
    env = _env.get("e")
    env_h = _obs_arr([x.get_observable_state() for x in env.humans]) if env is not None else h
    action = _orig_predict(self, state)
    if len(_records) < MAX_STATES:
        if h.shape[0]:
            d = np.hypot(r[0] - h[:, 0], r[1] - h[:, 1]) - r[4] - h[:, 4]
            dmin_all = float(d.min())
        else:
            dmin_all = float("inf")
        _records.append({
            "ep": len(_episodes),
            "frame": len(_cur["steps"]),
            "robot": r,
            "humans": h,             # TTC-sorted, exactly as the policy saw them
            "track_ids": tid,        # stable env.humans index of each of those
            "env_humans": env_h,     # ALL humans in stable env order; index == track_id
            "history": hist,
            "action": np.array([action.vx, action.vy], dtype=np.float32),
            "dmin_all": dmin_all,
            "scene": _scene_info(),
        })
        _cur["steps"].append(len(_records) - 1)
    return action


def _close_episode():
    if _cur["steps"]:
        _episodes.append({"state_ids": list(_cur["steps"]),
                          "outcome": _cur["infos"][-1] if _cur["infos"] else "unknown",
                          "scene": _scene_info()})


def _patched_reset(self, *a, **kw):
    # close the OLD episode before rebinding the env, or it gets labelled with
    # the NEXT scenario's geometry -- and B would then build circle candidates
    # for a square-crossing scene, the exact defect square_scene() documents.
    _close_episode()
    _env["e"] = self
    _cur["steps"] = []
    _cur["infos"] = []
    return _orig_reset(self, *a, **kw)


def _patched_step(self, *a, **kw):
    out = _orig_step(self, *a, **kw)
    # CrowdSim.step returns the gymnasium 5-tuple
    # (ob, reward, terminated, truncated, info); info is index 4, not 3.
    try:
        _cur["infos"].append(str(out[4])[:60])
    except Exception:
        pass
    return out


def _flush():
    _close_episode()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump({"records": _records, "episodes": _episodes}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    from collections import Counter
    print(f"\n[D0] {len(_records)} states from {len(_episodes)} episodes -> {OUT}")
    print(f"[D0] outcomes: {dict(Counter(e['outcome'] for e in _episodes))}")
    print(f"[D0] scenes: {dict(Counter(str(e['scene'].get('sim')) + '/' + str(e['scene'].get('human_num')) for e in _episodes))}")
    print(f"[D0] track-id match failures: {_mismatch['n']}  (must be 0)")
    bad = sum(1 for r in _records if (r["track_ids"] < 0).any())
    print(f"[D0] records with an unmatched track: {bad}  (must be 0)")
    uniq = [len(set(r["track_ids"].tolist())) == len(r["track_ids"]) for r in _records]
    print(f"[D0] records with duplicate track_ids: {sum(1 for u in uniq if not u)}  (must be 0)")
    n_dense = sum(1 for r in _records if r["humans"].shape[0] > 5)
    print(f"[D0] states with >5 humans: {n_dense} "
          f"({100.0 * n_dense / max(len(_records), 1):.1f}%)")
    q = np.array([r["dmin_all"] for r in _records if np.isfinite(r["dmin_all"])])
    if q.size:
        print(f"[D0] dmin_all p1={np.percentile(q,1):.3f} p5={np.percentile(q,5):.3f} "
              f"p25={np.percentile(q,25):.3f} p50={np.percentile(q,50):.3f}")
        print(f"[D0] near-critical states (dmin<0.3 m): {int((q<0.3).sum())}")


def main():
    MambaRLPolicy.predict_sarl_style = _patched_predict
    CrowdSim.reset = _patched_reset
    CrowdSim.step = _patched_step
    try:
        sys.argv = ["test.py"] + sys.argv[1:]
        runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "test.py"), run_name="__main__")
    finally:
        _flush()


if __name__ == "__main__":
    main()
