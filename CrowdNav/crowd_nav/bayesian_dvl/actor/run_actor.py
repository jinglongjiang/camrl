"""Behaviour cloning for the belief-conditioned policy actor, with a fixed
multi-checkpoint evaluation protocol.

Stage 1 asks one question and nothing else:

    can a clean Bayesian-conditioned classification actor, trained only on
    successful ORCA demonstrations, learn to navigate STABLY?

No DAgger, no value function, no ranking, no online RL. Layers 1-3 are frozen at
the audited versions (18/18 interface checks passed) and are not touched.

Evaluation protocol is fixed BEFORE training, because a measured drift makes any
single number unreadable: the same V0 checkpoint moved 0.300 -> 0.511 over 500
further updates while its training loss stayed flat. So every run reports three
equidistant late checkpoints, chosen by position and never by score, and the
headline is a mean over seeds and checkpoints -- never a best-of.
"""
from __future__ import annotations

import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from crowd_nav.bayesian_dvl.actor.model import PolicyActor, policy_loss, soft_targets
from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.evaluation_protocol import STAGE_ACCEPT_PLAN, STAGE_ACCEPT_SEEDS
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
from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn
from crowd_sim.envs.utils.action import ActionXY

REPO = Path(__file__).resolve().parents[3]
ENVCFG = REPO / "crowd_nav/configs/env_bayesian_dvl.config"
CFGPATH = REPO / "crowd_nav/configs/train_intent_bdvl.config"
CACHE_FOR = {"full": REPO / "runs/v3_domain_randomized/materialized_full.pth"}
ACTION_GRID_HASH = ActionGridSpec.from_env_config(str(ENVCFG)).table_hash()


def build_dataset(mode: str, device: str):
    """Successful-episode rows only, straight from the frozen materialised cache.

    The filter is the frozen Order 13 contract, not a new rule: a row's
    expert_action_indices is empty exactly when its ORCA episode ended in a
    collision or timeout. Behaviour cloning has no negative-return channel, so a
    failed teacher's actions would enter as correct answers at precisely the
    decision points that matter most.
    """
    cache = CACHE_FOR.get(mode)
    if cache is None or not cache.exists():
        raise SystemExit(f"no materialised cache for arm {mode!r}")
    blob = torch.load(str(cache), map_location="cpu", weights_only=False)
    tx = blob["transitions"]
    keep = [t for t in tx if t.expert_action_indices]
    print(f"materialised cache: {len(tx)} rows -> {len(keep)} from successful ORCA "
          f"episodes ({100*len(keep)/len(tx):.1f}%)", flush=True)
    R = torch.as_tensor(np.stack([t.robot_features for t in keep]).astype(np.float32), device=device)
    H = torch.as_tensor(np.stack([t.human_features for t in keep]).astype(np.float32), device=device)
    M = torch.as_tensor(np.stack([t.human_mask for t in keep]), device=device)
    E = [tuple(t.expert_action_indices) for t in keep]
    # The SINGLE nearest grid action, kept separately from the equivalence class
    # by the frozen contract: action_index is what the environment actually
    # executed, the class is the tolerance-widened set L_rank used to supervise.
    A = [int(t.action_index) for t in keep]
    return R, H, M, E, A


def train(model, data, updates: int, batch: int, lr: float, seed: int,
          ckpt_at: list, out: Path, device: str, log_every: int = 500,
          learner=None, hard_label: bool = False):
    """Behaviour cloning, optionally aggregated with learner states at a fixed
    1:1 ratio per minibatch.

    The ratio is enforced by SAMPLING, not by a loss weight. With 220997 expert
    rows, simply concatenating 50000 learner rows would leave the learner
    distribution at 18% of every batch -- and a null result would then be
    unreadable, because "the method does not work" and "the dose was too small"
    look identical. Half the batch from each source removes that ambiguity
    without touching the network, the loss, or introducing any coefficient.
    """
    R, H, M, E, A = data
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = len(E)
    n_l = 0
    if learner is not None:
        Rl, Hl, Ml, El = learner
        n_l = len(El)
        half = batch // 2
        print(f"aggregated training: {half} expert + {half} learner per batch "
              f"(pools {n} / {n_l})", flush=True)
    hist, saved = [], {}
    model.train()
    for u in range(1, updates + 1):
        if n_l:
            half = batch // 2
            ie = torch.randint(0, n, (half,), generator=gen)
            il = torch.randint(0, n_l, (batch - half,), generator=gen)
            rb = torch.cat([R[ie.to(R.device)], Rl[il.to(Rl.device)]])
            hb = torch.cat([H[ie.to(H.device)], Hl[il.to(Hl.device)]])
            mb = torch.cat([M[ie.to(M.device)], Ml[il.to(Ml.device)]])
            ex = ([(A[k],) if hard_label else E[k] for k in ie.tolist()]
                  + [El[k] for k in il.tolist()])
            logits = model(rb, hb, mb)
            tgt = soft_targets(ex, model.n_actions, logits.device)
        else:
            idx = torch.randint(0, n, (min(batch, n),), generator=gen)
            j = idx.tolist()
            logits = model(R[idx.to(R.device)], H[idx.to(R.device)], M[idx.to(R.device)])
            ex = [(A[k],) if hard_label else E[k] for k in j]
            tgt = soft_targets(ex, model.n_actions, logits.device)
        loss = policy_loss(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            am = logits.detach().argmax(dim=-1).tolist()
            # exact: argmax == ORCA's single nearest grid action
            # cls:   argmax is anywhere inside ORCA's equivalence class
            if n_l:
                jj = ie.tolist()
                ka = [A[k] for k in jj] + [-1] * (batch - half)
                ke = [E[k] for k in jj] + [()] * (batch - half)
            else:
                ka = [A[k] for k in j]
                ke = [E[k] for k in j]
            m = [i for i in range(len(am)) if ka[i] >= 0]
            exact = sum(am[i] == ka[i] for i in m) / max(len(m), 1)
            cls = sum(am[i] in ke[i] for i in m) / max(len(m), 1)
        hist.append((u, float(loss.detach()), exact, cls))
        if u in ckpt_at:
            p = out / f"actor_u{u:06d}.pth"
            torch.save({"model_state_dict": model.state_dict(), "updates": u, "seed": seed,
                        "action_grid_hash": ACTION_GRID_HASH}, str(p))
            saved[u] = p
            print(f"  checkpoint @ {u} -> {p.name}", flush=True)
        if u % log_every == 0 or u == 1:
            print(f"  BC[{u}/{updates}] loss={hist[-1][1]:.5f} "
                  f"exact-top1={exact:.3f} class-hit={cls:.3f}", flush=True)
    return saved, hist


@torch.no_grad()
def gate(model, cfg, mode: str, device: str, tag: str):
    """The single go/no-go: greedy closed loop on the 90 frozen layouts."""
    at = np.asarray(ActionGridSpec.from_env_config(str(ENVCFG)).build_action_table(), dtype=np.float64)
    SP = np.unique(np.round(np.hypot(at[:, 0], at[:, 1]), 4))
    ring = lambda i: int(np.argmin(np.abs(SP - round(float(np.hypot(*at[i])), 4))))
    max_steps = int(round(FROZEN_VALUES["time_limit"] / FROZEN_VALUES["dt"])) + 1
    per = defaultdict(lambda: defaultdict(int))
    rings = defaultdict(int)
    prog = []
    model.eval()
    for sc, seed in zip(STAGE_ACCEPT_PLAN, STAGE_ACCEPT_SEEDS):
        ep = _ScenarioEpisode(ENVCFG, sc, int(seed), is_heldout=False)
        env, robot = ep.env, ep.robot
        bank = IntentBeliefBank(make_candidate_fn(ep.scene), dt=FROZEN_VALUES["dt"],
                                speed=TRACKER_DEFAULTS["speed_prior"])
        rng = np.random.default_rng(int(seed))
        goal = np.array([robot.gx, robot.gy], dtype=np.float64)
        outcome = None
        for _ in range(max_steps):
            ep.advance_hidden_state()
            hs = [HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
                  for i, h in enumerate(env.humans)]
            bank.update({h.track_id: (h.px, h.py) for h in hs})
            ro = RobotObservation.from_full_state(env.robot.get_full_state())
            rem = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])
            hf, hm = build_intent_human_feature_batch(
                bank, ro, hs, mode=mode, rng=rng,
                horizon=cfg.future_horizon, n_samples=cfg.future_n_samples)
            logits = model(
                torch.as_tensor(_robot_feature_vector(ro, rem)[None], dtype=torch.float32, device=device),
                torch.as_tensor(hf[None], dtype=torch.float32, device=device),
                torch.as_tensor(hm[None], device=device))
            k = int(logits[0].argmax())
            rings[ring(k)] += 1
            before = float(np.linalg.norm(np.array([ro.px, ro.py]) - goal))
            _, _, term, trunc, info = env.step(ActionXY(float(at[k][0]), float(at[k][1])))
            prog.append(before - float(np.linalg.norm(np.array([robot.px, robot.py]) - goal)))
            if term or trunc:
                outcome = {"reach_goal": "success", "collision": "collision",
                           "timeout": "timeout"}.get(info.get("event"), "timeout")
                break
        outcome = outcome or "timeout"
        per[sc][outcome] += 1
        per[sc]["n"] += 1
    srs = []
    print(f"\n[{tag}] {'scenario':<18}{'n':>5}{'SR':>9}{'CR':>9}{'TR':>9}")
    for sc in sorted(per):
        d = per[sc]; n = d["n"]; srs.append(d["success"] / n)
        print(f"[{tag}] {sc:<18}{n:>5}{d['success']/n:>9.3f}{d['collision']/n:>9.3f}{d['timeout']/n:>9.3f}")
    macro = float(np.mean(srs))
    tot = sum(rings.values())
    print(f"[{tag}] {'MACRO':<18}{'':>5}{macro:>9.3f}   mean goal progress {np.mean(prog):+.4f} m/step")
    print(f"[{tag}] speed rings: " + "  ".join(
        f"{SP[r]:.3f}={100*rings[r]/max(tot,1):.1f}%" for r in range(len(SP))))
    return dict(macro=macro, per_scenario={sc: per[sc]["success"] / per[sc]["n"] for sc in sorted(per)},
                cr={sc: per[sc]["collision"] / per[sc]["n"] for sc in sorted(per)},
                tr={sc: per[sc]["timeout"] / per[sc]["n"] for sc in sorted(per)},
                goal_progress=float(np.mean(prog)),
                speed_rings={float(SP[r]): rings[r] / max(tot, 1) for r in range(len(SP))})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "mean", "cv", "uniform"])
    ap.add_argument("--updates", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-at", default=None,
                    help="comma-separated update counts to checkpoint and evaluate. "
                         "Fixed before training starts, never chosen by score.")
    ap.add_argument("--hard-label", action="store_true",
                    help="supervise ONLY the single nearest grid action instead of the "
                         "tolerance-widened equivalence class (mean 8.27 of 80 actions). "
                         "A class that wide calls ~10%% of the grid correct, so a high hit "
                         "rate can coexist with picking the member that collides.")
    ap.add_argument("--learner-data", type=Path, default=None,
                    help="learner_data.pth from one DAgger round; when given, every "
                         "minibatch is half expert and half learner")
    a = ap.parse_args()

    cfg = load_intent_training_config(str(CFGPATH))
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    at = ActionGridSpec.from_env_config(str(ENVCFG)).build_action_table()
    torch.manual_seed(a.seed)
    model = PolicyActor(n_actions=len(at)).to(dev)
    a.out.mkdir(parents=True, exist_ok=True)

    # THREE equidistant LATE checkpoints, fixed here, before any result exists.
    # Never a best-of: the drift measured on the value chain means a maximum over
    # checkpoints reports the luckiest point of a noisy trajectory.
    if a.ckpt_at:
        ckpt_at = [int(x) for x in a.ckpt_at.split(",")]
    else:
        ckpt_at = [a.updates - 2 * (a.updates // 10), a.updates - (a.updates // 10), a.updates]
    assert all(0 < c <= a.updates for c in ckpt_at), ckpt_at
    print(f"ACTOR BC  mode={a.mode}  seed={a.seed}  device={dev}  updates={a.updates}  "
          f"params={sum(p.numel() for p in model.parameters())}")
    print(f"fixed evaluation checkpoints: {ckpt_at}")
    print(f"label: {'single nearest action' if a.hard_label else 'uniform over the ORCA equivalence class'}")

    learner = None
    if a.learner_data:
        b = torch.load(str(a.learner_data), map_location="cpu", weights_only=False)
        if (b.get("mode") != a.mode or b.get("future_horizon") != cfg.future_horizon
                or b.get("future_n_samples") != cfg.future_n_samples):
            raise SystemExit(f"REFUSING learner data {a.learner_data.name}: built for a "
                             "different arm or belief horizon.")
        learner = (torch.as_tensor(b["R"], device=dev), torch.as_tensor(b["H"], device=dev),
                   torch.as_tensor(b["M"], device=dev), b["E"])
        print(f"learner data: {len(b['E'])} states from {a.learner_data}")

    t0 = time.time()
    data = build_dataset(a.mode, dev)
    print(f"dataset ready in {time.time()-t0:.0f}s")
    saved, hist = train(model, data, a.updates, a.batch, a.lr, a.seed, ckpt_at, a.out, dev,
                        learner=learner, hard_label=a.hard_label)

    results = {}
    for u in ckpt_at:
        blob = torch.load(str(saved[u]), map_location=dev)
        # class i of the training target must be action_table[i] at deployment.
        # Nothing else enforces that, and it is the shape of every contract bug
        # this project has hit (tau 16/32, horizon 8/1, t vs t+1).
        stored = blob.get("action_grid_hash")
        if stored is not None and stored != ACTION_GRID_HASH:
            raise SystemExit(f"action grid changed since training: checkpoint {stored[:12]} "
                             f"!= current {ACTION_GRID_HASH[:12]}; class i is no longer "
                             "action_table[i]")
        m = PolicyActor(n_actions=len(at)).to(dev)
        m.load_state_dict(blob["model_state_dict"])
        results[u] = gate(m, cfg, a.mode, dev, f"s{a.seed}@{u}")
    macs = [results[u]["macro"] for u in ckpt_at]
    print(f"\nseed {a.seed}: checkpoint macros {[round(x,3) for x in macs]}  "
          f"mean {np.mean(macs):.3f} +- {np.std(macs):.3f}")
    (a.out / "result.json").write_text(json.dumps(
        {"seed": a.seed, "mode": a.mode, "updates": a.updates, "checkpoints": ckpt_at,
         "results": {str(k): v for k, v in results.items()},
         "checkpoint_mean": float(np.mean(macs)), "checkpoint_sd": float(np.std(macs)),
         "loss_history": hist[-50:]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
