"""One round of dataset aggregation for the policy actor.

The actor drives its own greedy policy for the whole episode; ORCA is queried at
every visited state but never takes control. Only the label comes from the
teacher -- letting it steer would replace the states being collected with the
expert manifold that is already covered.

Why this and not more expert data. The BC actor reaches 92.4% expert-equivalence
on ORCA's own states and its speed distribution there matches ORCA ring for ring
(7.0/14.4/22.6/22.9/33.2 against 6.5/14.0/23.3/22.6/33.6). On the states it
reaches itself that collapses: 45.4% of actions take the slowest ring and only
1.3% the fastest, against ORCA's 33.6%. With an invisible robot, slow is how you
get hit -- which is what the gate shows, 93% collisions and almost no timeouts.
More demonstrations of what it already does correctly cannot fix that.

Records are stored in the SAME layout as the expert rows (robot features, packed
human rows, mask, ORCA equivalence class) so the trainer treats both sources
identically and only the sampling ratio differs.

Known gap, deliberately not handled this round: ORCA's own recovery rate falls
the deeper the student has drifted (measured 0.867 at 5 student steps, 0.522 at
20), so labels from deep learner states are worth less. Filtering on that needs a
teacher rollout per state, which is a new mechanism. If this round underperforms,
that is the first thing to check.
"""
from __future__ import annotations

import argparse, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from crowd_nav.bayesian_dvl.actor.model import PolicyActor
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.geometry_features import _robot_feature_vector
from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
from crowd_nav.bayesian_dvl.intent_policy import (
    build_intent_human_feature_batch, remaining_time_fraction,
)
from crowd_nav.bayesian_dvl.intent_runtime_config import (
    ActionGridSpec, FROZEN_VALUES, TRACKER_DEFAULTS,
)
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank
from crowd_nav.bayesian_dvl.intent_train import _ScenarioEpisode
from crowd_nav.bayesian_dvl.ranking import (
    build_action_equivalence_class, derive_action_equivalence_tolerance,
)
from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn
from crowd_sim.envs.utils.action import ActionXY

REPO = Path(__file__).resolve().parents[3]
ENVCFG = REPO / "crowd_nav/configs/env_bayesian_dvl.config"
CFGPATH = REPO / "crowd_nav/configs/train_intent_bdvl.config"
ACTORS = Path("/root/bdvl_diagnostics/actor")
# training-distribution layouts, disjoint from the 90-layout DEV gate (3_0xx_xxx)
BLOCKS = {"circle": 2_600_000, "square": 2_610_000, "junction_crowd": 2_100_000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=50000)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--out", type=Path, default=ACTORS / "dagger")
    a = ap.parse_args()

    cfg = load_intent_training_config(str(CFGPATH))
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    at = np.asarray(ActionGridSpec.from_env_config(str(ENVCFG)).build_action_table(), dtype=np.float64)
    tol = derive_action_equivalence_tolerance(at)
    SP = np.unique(np.round(np.hypot(at[:, 0], at[:, 1]), 4))
    ring = lambda i: int(np.argmin(np.abs(SP - round(float(np.hypot(*at[i])), 4))))

    models = {}
    for s in (98204, 98205, 98206):
        p = ACTORS / f"bc_s{s}/actor_u005000.pth"
        if p.exists():
            m = PolicyActor(n_actions=len(at)).to(dev)
            m.load_state_dict(torch.load(str(p), map_location=dev)["model_state_dict"])
            m.eval(); models[f"s{s}"] = m
    if not models:
        raise SystemExit("no BC actor checkpoints found")
    names = list(models)
    print(f"collecting {a.n_states} learner states with {names}, mode={a.mode}, device={dev}")

    R, H, M, E = [], [], [], []
    by_ck, by_sc, rings = defaultdict(int), defaultdict(int), defaultdict(int)
    t0, ep_i = time.time(), 0
    max_steps = int(round(FROZEN_VALUES["time_limit"] / FROZEN_VALUES["dt"])) + 1
    while len(E) < a.n_states:
        name = names[ep_i % len(names)]
        sc = list(BLOCKS)[(ep_i // len(names)) % len(BLOCKS)]
        seed = BLOCKS[sc] + (ep_i // (len(names) * len(BLOCKS)))
        ep_i += 1
        ep = _ScenarioEpisode(ENVCFG, sc, int(seed), is_heldout=False)
        env = ep.env
        bank = IntentBeliefBank(make_candidate_fn(ep.scene), dt=FROZEN_VALUES["dt"],
                                speed=TRACKER_DEFAULTS["speed_prior"])
        rng = np.random.default_rng(int(seed))
        for _ in range(max_steps):
            ep.advance_hidden_state()
            hs = [HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
                  for i, h in enumerate(env.humans)]
            bank.update({h.track_id: (h.px, h.py) for h in hs})
            ro = RobotObservation.from_full_state(env.robot.get_full_state())
            rem = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])
            hf, hm = build_intent_human_feature_batch(
                bank, ro, hs, mode=a.mode, rng=rng,
                horizon=cfg.future_horizon, n_samples=cfg.future_n_samples)
            with torch.no_grad():
                lg = models[name](
                    torch.as_tensor(_robot_feature_vector(ro, rem)[None], dtype=torch.float32, device=dev),
                    torch.as_tensor(hf[None], dtype=torch.float32, device=dev),
                    torch.as_tensor(hm[None], device=dev))
            k = int(lg[0].argmax())

            o = env.robot.act([h.get_observable_state() for h in env.humans])
            ex = build_action_equivalence_class(o.vx, o.vy, at, tol)
            if len(ex) < len(at):                      # need negatives; never empty by construction
                R.append(_robot_feature_vector(ro, rem)); H.append(hf); M.append(hm); E.append(tuple(ex))
                by_ck[name] += 1; by_sc[sc] += 1; rings[ring(k)] += 1
            _, _, term, trunc, _ = env.step(ActionXY(float(at[k][0]), float(at[k][1])))
            if term or trunc or len(E) >= a.n_states:
                break
        if ep_i % 100 == 0:
            print(f"  {len(E)}/{a.n_states} states, {ep_i} episodes, "
                  f"{time.time()-t0:.0f}s", flush=True)

    tot = sum(rings.values())
    print(f"\ncollected {len(E)} learner states from {ep_i} episodes in {time.time()-t0:.0f}s")
    print(f"  by checkpoint: {dict(by_ck)}")
    print(f"  by scenario  : {dict(by_sc)}")
    print("  actor speed rings while collecting: " + "  ".join(
        f"{SP[r]:.3f}={100*rings[r]/max(tot,1):.1f}%" for r in range(len(SP))))
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"R": np.stack(R).astype(np.float32), "H": np.stack(H).astype(np.float32),
                "M": np.stack(M), "E": E, "mode": a.mode,
                "future_horizon": int(cfg.future_horizon),
                "future_n_samples": int(cfg.future_n_samples)},
               str(a.out / "learner_data.pth"))
    print(f"saved -> {a.out/'learner_data.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
