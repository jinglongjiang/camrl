"""V0: train V(s,b) on the frozen ORCA corpus, then run the 90-episode gate.

Deliberately minimal. No IQN, no ranking loss, no gradient composer, no online
RL, no DAgger. One objective:

    V_theta(s, b)  <-  MC return of ORCA from s

and one decision rule, a*= argmax_a [ r(s,a) + gamma V(s'_a, b) ]. If closed-loop
success rises off 0.078 with nothing but this, the frozen chain's bottleneck was
structural, not a training-signal problem -- and every module deleted to get
here can be reintroduced one at a time against a working baseline instead of
against another 0.078.

Usage:
  run_v0.py --mode full --passes 2400 --out runs/v0/full
  run_v0.py --mode full --eval-only runs/v0/full/v0.pth
"""
from __future__ import annotations

import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.evaluation_protocol import STAGE_ACCEPT_PLAN, STAGE_ACCEPT_SEEDS
from crowd_nav.bayesian_dvl.geometry_features import _robot_feature_vector
from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
from crowd_nav.bayesian_dvl.intent_policy import (
    build_intent_human_feature_batch, remaining_time_fraction,
)
from crowd_nav.bayesian_dvl.intent_runtime_config import (
    ActionGridSpec, FROZEN_VALUES, HUMAN_FEATURE_DIM_V7, TRACKER_DEFAULTS,
)
from crowd_nav.bayesian_dvl.intent_train import _ScenarioEpisode, compute_mc_returns
from crowd_nav.bayesian_dvl.intent_tracker import IntentBeliefBank
from crowd_nav.bayesian_dvl.scene_candidates import make_candidate_fn
from crowd_nav.bayesian_dvl.v0 import lookahead as LA
from crowd_nav.bayesian_dvl.v0.scoring import score_actions
from crowd_nav.bayesian_dvl.v0.value_model import ScalarValueModel
from crowd_sim.envs.utils.action import ActionXY

REPO = Path(__file__).resolve().parents[3]
ENVCFG = REPO / "crowd_nav/configs/env_bayesian_dvl.config"
CFGPATH = REPO / "crowd_nav/configs/train_intent_bdvl.config"
CORPUS = REPO / "runs/v3_domain_randomized/corpus/il_corpus_raw_24ffe4448151.pth"
# One materialised cache per arm. Only `full` exists today; mean/cv are a
# separate materialisation run, not something to fake from the full rows.
CACHE_FOR = {"full": REPO / "runs/v3_domain_randomized/materialized_full.pth"}
DATASET_DIR = Path("/root/bdvl_diagnostics/v0/datasets")


# ---------------------------------------------------------------- training --
def dataset_identity(cfg, mode: str, limit: int | None) -> dict:
    """What a cached V0 dataset is valid FOR.

    Everything that decides what a row CONTAINS: the arm, the episode budget,
    the belief horizon and sample count, the feature schema and width, gamma
    (the MC targets depend on it), and the hash of the modules that actually
    produce the rows. Anything outside that list is a training knob and does
    not invalidate the data.
    """
    from crowd_nav.bayesian_dvl.intent_train_cli import materialization_code_sha256
    return {
        "dataset_schema": "bdvl_v0_state_value_dataset_v1",
        "mode": mode, "episodes": limit,
        "future_horizon": int(cfg.future_horizon),
        "future_n_samples": int(cfg.future_n_samples),
        "gamma": float(cfg.gamma),
        "feature_schema": cfg.feature_schema,
        "human_feature_dim": int(HUMAN_FEATURE_DIM_V7),
        "materialization_code_sha256": materialization_code_sha256(),
    }


def build_dataset(cfg, mode: str, device: str, limit: int | None):
    """State rows + ORCA MC returns for one arm.

    Two sources, and which one is used is NOT a free choice: the rows a model
    trains on must be built by the same call, in the same arm, that will build
    them again at deployment. Mixing arms here would recreate the exact defect
    just removed from the scoring path -- one quantity learned, a different
    quantity read.

      * the frozen materialised cache when this arm has one (the `full` arm
        does: 256177 rows, already built by build_intent_human_feature_batch);
      * otherwise a replay of the raw corpus for this arm. At ~80 ms/row in
        `full` that is 5.7 hours for all 5000 episodes, which is why the cache
        exists -- but for a 30-episode pilot it is two minutes.

    The result is cached on disk per (arm, episode budget) so the three
    optimizer seeds of an arm share one dataset and differ only in training
    randomness.
    """
    ident = dataset_identity(cfg, mode, limit)
    tag = f"{mode}_{limit if limit else 'all'}"
    disk = DATASET_DIR / f"v0_dataset_{tag}.pth"
    if disk.exists():
        d = torch.load(str(disk), map_location="cpu", weights_only=False)
        stored = d.get("identity")
        if stored != ident:
            drift = sorted(k for k in ident if (stored or {}).get(k) != ident[k])
            raise SystemExit(
                f"REFUSING the cached dataset {disk.name}: it was built under different "
                f"{drift}. Delete it and rebuild -- silently reusing rows built by other\n"
                "code, another arm or another horizon is the exact failure this project\n"
                "has already paid for twice (the 16-vs-32 quantile split and the\n"
                "1-vs-8 future horizon).")
        print(f"dataset reuse: {len(d['G'])} rows from {disk.name}", flush=True)
        return tuple(torch.as_tensor(d[k], device=device) for k in ("R", "H", "M", "G"))

    cache = CACHE_FOR.get(mode)
    from_materialised = cache is not None and cache.exists() and not limit
    if from_materialised:
        blob = torch.load(str(cache), map_location="cpu", weights_only=False)
        tx = blob["transitions"]
        print(f"materialised cache: {len(tx)} rows from {cache.name}", flush=True)
        R = np.stack([t.robot_features for t in tx]).astype(np.float32)
        H = np.stack([t.human_features for t in tx]).astype(np.float32)
        M = np.stack([t.human_mask for t in tx])
        G = np.asarray([t.mc_return for t in tx], dtype=np.float32)
    else:
        payload = torch.load(str(CORPUS), map_location="cpu", weights_only=False)
        eps = payload["episodes"][:limit] if limit else payload["episodes"]
        print(f"replaying {len(eps)} corpus episodes for arm {mode!r}", flush=True)
        Rs, Hs, Ms, Gs = [], [], [], []
        t0 = time.time()
        for n, raw in enumerate(eps, 1):
            ep = _ScenarioEpisode(ENVCFG, str(raw.scenario), int(raw.episode_seed),
                                  is_heldout=bool(raw.is_heldout))
            bank = IntentBeliefBank(make_candidate_fn(ep.scene), dt=FROZEN_VALUES["dt"],
                                    speed=TRACKER_DEFAULTS["speed_prior"])
            rng = np.random.default_rng(int(raw.episode_seed))
            for st, g in zip(raw.steps, compute_mc_returns([s.reward for s in raw.steps], cfg.gamma)):
                bank.update({h.track_id: (h.px, h.py) for h in st.humans})
                hf, hm = build_intent_human_feature_batch(
                    bank, st.robot, st.humans, mode=mode, rng=rng,
                    horizon=cfg.future_horizon, n_samples=cfg.future_n_samples)
                Rs.append(_robot_feature_vector(st.robot, st.remaining_fraction))
                Hs.append(hf); Ms.append(hm); Gs.append(g)
            if n % 10 == 0:
                print(f"  {n}/{len(eps)} episodes, {len(Gs)} rows, {time.time()-t0:.0f}s", flush=True)
        R = np.stack(Rs).astype(np.float32); H = np.stack(Hs).astype(np.float32)
        M = np.stack(Ms); G = np.asarray(Gs, dtype=np.float32)

    # Only the expensive path is worth caching. Rows taken from the materialised
    # cache are already on disk in another form; writing a second 1.44 GB copy
    # of them buys nothing and this box is at 93%.
    if not from_materialised:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"R": R, "H": H, "M": M, "G": G, "identity": ident}, str(disk))
        print(f"dataset cached -> {disk}", flush=True)
    return (torch.as_tensor(R, device=device), torch.as_tensor(H, device=device),
            torch.as_tensor(M, device=device), torch.as_tensor(G, device=device))


def train(model, data, passes: int, batch: int, lr: float, seed: int,
          rank_pool: dict | None = None, gamma: float = 0.99, device: str = "cpu",
          log_every: int = 200):
    """MSE on states, plus an optional LOCAL RANKING term on one state's 80
    successors per update.

        L = MSE(V(s), G)  +  relu(S_neg - S_pos)^2,     S_a = r(s,a) + gamma V(s'_a)

    The ranking term constrains the SCORE ordering, not V directly, because the
    score is what deployment takes an argmax over. r(s,a) is analytic and exact,
    so it enters as a constant offset and every gradient flows through V -- the
    known physics is not re-learned, only the part that has to be.

    Why it is needed at all: MSE is pointwise regression on states that appear
    on ORCA trajectories. Nothing in it refers to two candidates from the same
    state, so nothing constrains their relative order -- and the relative order
    is the entire decision. Measured: three seeds reached indistinguishable MSE
    (0.104-0.118) and produced closed-loop macro 0.556 / 0.300 / 0.356, with
    Spearman agreement of only ~0.71 on identical successor sets, one model
    advancing +0.12 m/step toward the goal and the others -0.16 to -0.24.

    Deliberately no margin, no loss weight, no gradient projection. Squared
    hinge keeps it in the same units as the MSE term. Once the expert set
    already ranks first the term and its gradient are exactly zero, so this
    never pushes the policy to imitate ORCA beyond fixing an inversion.
    """
    R, H, M, G = data
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # TWO generators, deliberately. `gen` draws the expert minibatch and nothing
    # else, so an A/B pair that differs only in whether the ranking term is on
    # still sees a bit-identical minibatch sequence. Drawing the ranking state
    # from the same stream would desynchronise every subsequent batch and the
    # comparison would no longer isolate the term being tested.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rank_gen = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    n = len(G)
    rank_n = 0 if rank_pool is None else int(rank_pool["n_states"])
    if rank_n:
        rp = {k: torch.as_tensor(rank_pool[k], device=device)
              for k in ("rows", "mask", "robot_feats", "rewards", "terminal", "expert_mask")}
        print(f"ranking term ON: pool of {rank_n} expert states x "
              f"{rp['rows'].shape[1]} candidates", flush=True)
    model.train()
    hist = []
    for p_i in range(1, passes + 1):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen).to(G.device)
        value_loss = torch.nn.functional.mse_loss(model(R[idx], H[idx], M[idx]), G[idx])
        rank_loss = torch.zeros((), device=value_loss.device)
        if rank_n:
            j = int(torch.randint(0, rank_n, (1,), generator=rank_gen))
            n_a = rp["rows"].shape[1]
            v = model(rp["robot_feats"][j], rp["rows"][j],
                      rp["mask"][j].unsqueeze(0).expand(n_a, -1))
            v = torch.where(rp["terminal"][j], torch.zeros_like(v), v)
            scores = rp["rewards"][j] + gamma * v
            em = rp["expert_mask"][j]
            rank_loss = torch.relu(scores[~em].max() - scores[em].max()).square()
        loss = value_loss + rank_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        hist.append((p_i, float(value_loss.detach()), float(rank_loss.detach())))
        if p_i % log_every == 0 or p_i == 1:
            print(f"  IL[{p_i}/{passes}] mse={hist[-1][1]:.6f} rank={hist[-1][2]:.6f}", flush=True)
    return model, hist


# -------------------------------------------------------------- evaluation --
def gate(model, cfg, mode: str, device: str, tag: str, feature_horizon: int,
         decomp_csv: Path | None = None):
    at = np.asarray(ActionGridSpec.from_env_config(str(ENVCFG)).build_action_table(), dtype=np.float64)
    SP = np.unique(np.round(np.hypot(at[:, 0], at[:, 1]), 4))
    ring = lambda i: int(np.argmin(np.abs(SP - round(float(np.hypot(*at[i])), 4))))
    max_steps = int(round(FROZEN_VALUES["time_limit"] / FROZEN_VALUES["dt"])) + 1
    per = defaultdict(lambda: defaultdict(int)); rings = defaultdict(int)
    # Score decomposition. Who is actually driving -- the analytic reward or
    # the learned value? range(gamma V)/range(r) answers that per decision,
    # and it is the difference between 'the value head is flat so progress
    # reward decides' and 'the value head genuinely prefers speed'.
    decomp = []
    checked_layout = False
    speeds_of = np.hypot(at[:, 0], at[:, 1])
    print(f"\nGATE {tag}  mode={mode}  {len(STAGE_ACCEPT_SEEDS)} layouts", flush=True)
    for sc, seed in zip(STAGE_ACCEPT_PLAN, STAGE_ACCEPT_SEEDS):
        ep = _ScenarioEpisode(ENVCFG, sc, int(seed), is_heldout=False)
        env, robot = ep.env, ep.robot
        bank = IntentBeliefBank(make_candidate_fn(ep.scene), dt=FROZEN_VALUES["dt"],
                                speed=TRACKER_DEFAULTS["speed_prior"])
        rng = np.random.default_rng(int(seed))
        outcome = None
        first = len(decomp)   # where this episode's decomposition rows start
        for _ in range(max_steps):
            ep.advance_hidden_state()
            hs = [HumanObservation(i, float(h.px), float(h.py), float(h.vx), float(h.vy), float(h.radius))
                  for i, h in enumerate(env.humans)]
            bank.update({h.track_id: (h.px, h.py) for h in hs})
            ro = RobotObservation.from_full_state(env.robot.get_full_state())
            rem = remaining_time_fraction(env.global_time, FROZEN_VALUES["time_limit"])
            if not checked_layout:
                # once per gate, against the real feature builder
                LA.assert_robot_relative_columns(bank, ro, hs, mode, np.random.default_rng(0),
                                                 cfg.future_horizon, cfg.future_n_samples)
                checked_layout = True
            s, rw, vv = score_actions(model, bank, ro, hs, at, rem, mode=mode, rng=rng,
                                      n_samples=cfg.future_n_samples, gamma=cfg.gamma,
                                      device=device, feature_horizon=feature_horizon)
            k = int(np.argmax(s)); rings[ring(k)] += 1
            gv = cfg.gamma * vv
            rr, rv = float(rw.max() - rw.min()), float(gv.max() - gv.min())
            cc = lambda x: (float(np.corrcoef(speeds_of, x)[0, 1])
                            if float(np.std(x)) > 1e-12 else float('nan'))
            decomp.append(dict(
                scenario=sc, seed=int(seed), step=len(decomp) - first,
                range_r=round(rr, 6), range_v=round(rv, 6),
                range_total=round(float(s.max() - s.min()), 6),
                rho=round(rv / rr, 4) if rr > 1e-12 else '',
                argmax_r=int(np.argmax(rw)), argmax_v=int(np.argmax(gv)), argmax_total=k,
                ring_total=ring(k), corr_speed_r=round(cc(rw), 4),
                corr_speed_v=round(cc(gv), 4), corr_speed_total=round(cc(s), 4)))
            _, _, term, trunc, info = env.step(ActionXY(float(at[k][0]), float(at[k][1])))
            if term or trunc:
                outcome = {"reach_goal": "success", "collision": "collision",
                           "timeout": "timeout"}.get(info.get("event"), "timeout")
                break
        outcome = outcome or "timeout"
        per[sc][outcome] += 1; per[sc]["n"] += 1
        for d in decomp[first:]:
            d['outcome'] = outcome
        print(f"  {sc:<15} {seed}  {outcome}", flush=True)
    srs = []
    print(f"\n{'scenario':<18}{'n':>5}{'SR':>9}{'CR':>9}{'TR':>9}")
    for sc in sorted(per):
        d = per[sc]; n = d["n"]; srs.append(d["success"] / n)
        print(f"{sc:<18}{n:>5}{d['success']/n:>9.3f}{d['collision']/n:>9.3f}{d['timeout']/n:>9.3f}")
    if decomp_csv is not None:
        import csv as _csv
        with open(decomp_csv, 'w', newline='') as f:
            w = _csv.DictWriter(f, list(decomp[0]))
            w.writeheader(); w.writerows(decomp)
        rho = np.array([d['rho'] for d in decomp if d['rho'] != ''], dtype=float)
        same_r = np.mean([d['argmax_total'] == d['argmax_r'] for d in decomp])
        same_v = np.mean([d['argmax_total'] == d['argmax_v'] for d in decomp])
        print(f"\nscore decomposition ({len(decomp)} decisions) -> {decomp_csv.name}")
        print(f"  rho = range(gammaV)/range(r):  median {np.median(rho):.3f}  "
              f"p10 {np.percentile(rho,10):.3f}  p90 {np.percentile(rho,90):.3f}")
        print(f"  argmax_total == argmax_r : {100*same_r:.1f}%   "
              f"argmax_total == argmax_v : {100*same_v:.1f}%")
        for nm in ('corr_speed_r','corr_speed_v','corr_speed_total'):
            v = np.array([d[nm] for d in decomp], dtype=float)
            print(f"  mean {nm:<18} {np.nanmean(v):+.3f}")
    macro = float(np.mean(srs))
    tot = sum(rings.values())
    print(f"{'MACRO':<18}{'':>5}{macro:>9.3f}")
    print("executed speed rings: " + "  ".join(
        f"{SP[r]:.3f}={100*rings[r]/max(tot,1):.1f}%" for r in range(len(SP))))
    print(f"\nGATE {'PASS' if (macro >= 0.70 and min(srs) >= 0.50) else 'FAIL'}"
          f"  (macro {macro:.3f} vs 0.70, min {min(srs):.3f} vs 0.50)")
    return macro, srs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "mean", "cv", "uniform"])
    ap.add_argument("--passes", type=int, default=2400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=98204)
    ap.add_argument("--episodes", type=int, default=None, help="corpus episodes (default: all)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--eval-only", type=Path, default=None)
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="continue training from an existing v0.pth (repair test)")
    ap.add_argument("--rank-states", type=int, default=0,
                    help="size of the local-ranking candidate pool; 0 disables the term")
    ap.add_argument("--rank-stride", type=int, default=3)
    ap.add_argument("--learner-pool", type=Path, default=None,
                    help="learner_states.pth -- ranking supervision on states the "
                         "STUDENT reaches, which is where the ordering was measured "
                         "to be wrong (62%% violation) while expert states were "
                         "already correct (loss 1e-8..1e-5)")
    ap.add_argument("--score-horizon", type=int, default=None,
                    help="posterior-future horizon used when SCORING. Defaults to the\n                          config's future_horizon, which is what the training rows used;\n                          override only to reproduce an earlier mismatched run.")
    a = ap.parse_args()

    cfg = load_intent_training_config(str(CFGPATH))
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    sh = cfg.future_horizon if a.score_horizon is None else a.score_horizon
    if sh != cfg.future_horizon:
        print(f"WARNING: scoring horizon {sh} != training horizon {cfg.future_horizon}; "
              "the fdx/fdy/spread columns will not mean the same thing in both places.")
    model = ScalarValueModel().to(dev)
    if a.resume_from:
        model.load_state_dict(torch.load(str(a.resume_from), map_location=dev)["model_state_dict"])
        print(f"resumed weights from {a.resume_from}")
    print(f"V0  mode={a.mode}  device={dev}  gamma={cfg.gamma}  n_samples={cfg.future_n_samples}  "
          f"train_horizon={cfg.future_horizon}  score_horizon={sh}  "
          f"params={sum(p.numel() for p in model.parameters())}")

    if a.eval_only:
        model.load_state_dict(torch.load(str(a.eval_only), map_location=dev)["model_state_dict"])
        gate(model, cfg, a.mode, dev, a.eval_only.parent.name, sh,
             a.eval_only.parent / "decomposition.csv")
        return 0

    t0 = time.time()
    data = build_dataset(cfg, a.mode, dev, a.episodes)
    print(f"dataset: {len(data[3])} rows in {time.time()-t0:.0f}s")
    pool = None
    if a.learner_pool:
        blob = torch.load(str(a.learner_pool), map_location="cpu", weights_only=False)
        if (blob.get("mode") != a.mode or blob.get("future_horizon") != cfg.future_horizon
                or blob.get("future_n_samples") != cfg.future_n_samples):
            raise SystemExit(f"REFUSING learner pool {a.learner_pool.name}: built for a "
                             "different arm or belief horizon.")
        from crowd_nav.bayesian_dvl.v0.learner_states import rebuild_rows
        recs = blob["records"]
        print(f"learner pool: rebuilding {len(recs)} candidate sets "
              f"(patch_robot_columns, the deployment function)", flush=True)
        pool = {
            "rows": np.stack([rebuild_rows(r) for r in recs]).astype(np.float32),
            "mask": np.stack([np.asarray(r["mask"]) for r in recs]),
            "robot_feats": np.stack([np.asarray(r["robot_feats"]) for r in recs]).astype(np.float32),
            "rewards": np.stack([np.asarray(r["rewards"]) for r in recs]).astype(np.float32),
            "terminal": np.stack([np.asarray(r["terminal"]) for r in recs]),
            "expert_mask": np.stack([np.asarray(r["expert_mask"]) for r in recs]),
            "n_states": len(recs),
        }
        print(f"learner pool ready: {pool['rows'].nbytes/2**30:.2f} GiB", flush=True)
    elif a.rank_states:
        from crowd_nav.bayesian_dvl.v0.rank_data import build_rank_pool
        pcache = DATASET_DIR / f"v0_rankpool_{a.mode}_{a.rank_states}_{a.rank_stride}.pth"
        if pcache.exists():
            pool = torch.load(str(pcache), map_location="cpu", weights_only=False)
            if (pool.get("mode") != a.mode
                    or pool.get("future_horizon") != cfg.future_horizon
                    or pool.get("future_n_samples") != cfg.future_n_samples):
                raise SystemExit(f"REFUSING cached rank pool {pcache.name}: built for a "
                                 "different arm or belief horizon.")
            print(f"rank pool reuse: {pool['n_states']} states from {pcache.name}")
        else:
            pool = build_rank_pool(cfg, CORPUS, ENVCFG, a.mode, a.rank_states,
                                   a.seed, a.rank_stride)
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(pool, str(pcache))
    _, hist = train(model, data, a.passes, a.batch, a.lr, a.seed,
                    rank_pool=pool, gamma=cfg.gamma, device=dev)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "mode": a.mode,
                    "passes": a.passes, "seed": a.seed, "episodes": a.episodes,
                    "score_horizon": sh, "train_horizon": cfg.future_horizon,
                    "rank_states": a.rank_states,
                    "learner_pool": str(a.learner_pool) if a.learner_pool else None,
                    "resumed_from": str(a.resume_from) if a.resume_from else None,
                    "loss_history": hist},
                   str(a.out / "v0.pth"))
        print(f"saved -> {a.out / 'v0.pth'}")
    macro, srs = gate(model, cfg, a.mode, dev, a.out.name if a.out else a.mode, sh,
                      (a.out / "decomposition.csv") if a.out else None)
    if a.out:
        (a.out / "gate.json").write_text(json.dumps(
            {"mode": a.mode, "macro": macro, "per_scenario": srs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
