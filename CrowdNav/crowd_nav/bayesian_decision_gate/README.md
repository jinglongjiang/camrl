# Bayesian Decision Gate

## Why this exists

The `belief_mdp/` experiments (Mamba-as-teacher + GDBN-risk-assistant, twin-head
Q-decomposition) never cleanly demonstrated that the Bayesian branch earns its
architectural complexity: results were close to a frozen Mamba-VL baseline,
easy to tie, and hard to attribute to the belief branch specifically. Before
sinking more time into either **Route A** (keep Mamba as the primary policy,
GDBN as a risk-assistant -- everything already built in `belief_mdp/`) or
**Route B** (drop Mamba/IL/RL entirely, go pure Bayesian belief + risk-aware
MPC), this module runs one cheap, decisive, pre-registered experiment:

> Does the fitted GDBN belief provide genuine **decision-level** advantage
> over constant-velocity (CV) reasoning -- can it rank the SAME candidate
> robot actions by true counterfactual risk better than CV, especially during
> stop/slow/turn stress events -- or does it only look better on one-step
> prediction metrics (NLL/Brier) while making equivalent decisions?

**A better NLL or Brier score alone does not count as passing.** Prior
Bayesian-pilot work already showed GDBN beats CV on those (see
`bayesian_pilot/BMA_EXPERIMENT.md`), and that was not enough to make Route A's
navigation results decisively better than the frozen baseline. This gate only
looks at whether GDBN changes *which action gets chosen*, and whether that
change is actually safer.

This directory is deliberately isolated: it does not modify `belief_mdp/`,
`bayesian_pilot/`, or any existing paper-experiment results. It only *reads*
the frozen K=3 GDBN checkpoint (`runs/bayesian_distributional/gdbn_params_cv_residual_k3`)
that Route A already uses.

**Revision note:** this module went through TWO rounds of review before any
formal run, both before looking at any real result. Round 1 (8 issues) found
problems that would have made a formal PASS/FAIL untrustworthy even though
the pipeline ran correctly: Gate-A/Gate-B split, tie-tolerant ranking,
multi-horizon sensitivity, frozen CV covariance, event_near vs event_any,
seed-level bootstrap blocks, corrected episode-count budget, reproducibility
hashing. Round 2 (after Round 1's fixes were verified) found three more,
all now fixed and described inline below:
  - the "seed-block bootstrap" was silently grouping by *episode*, not by
    *experiment seed* (see "suite_seed vs. episode_seed" below);
  - the gate collapsed FAIL and INCONCLUSIVE into one bool, producing
    self-contradictory reports (a "FAIL" whose own text said the sample was
    too small to conclude anything) -- fixed with a proper three-state
    `PASS`/`FAIL`/`INCONCLUSIVE` result and a pre-registered `opportunity_rate`
    floor;
  - only the 5-person density constrained the PASS/FAIL decision even though
    the paper's final scenarios include 20-person crowds -- fixed by requiring
    20-person non-inferiority inside each gate.
Do not revert any of these without re-reading the reasoning.

## Two independent gates, not one

**Gate-A** and **Gate-B** answer different questions and are reported
separately, then combined:

- **Gate-A**: `frozen_k3` (the model *already deployed* in `belief_mdp/`, i.e.
  Route A exactly as it exists today) vs. CV.
- **Gate-B**: `refit_k{K}` (freshly fit on this experiment's own data -- the
  best a GDBN belief could plausibly do) vs. CV.

A frozen-K3 loss does not by itself mean "Bayesian belief is a dead end" -- it
may just mean the *current fit* is bad. A refit-K loss means the belief
*framework itself* has no decision-level signal to extract here, which is a
materially stronger and different conclusion.

Each gate is itself a **three-state result** -- `PASS` / `FAIL` /
`INCONCLUSIVE`, never a bare bool (see the pre-registered criteria section
below for exactly how the state is decided). `run_gate.py::combine_gates`
combines the two gates' states:

- **If either gate is `INCONCLUSIVE`**, the combined result is
  `INCONCLUSIVE` -- never silently resolved to PASS or FAIL. More data or a
  stronger protocol is needed before any Route A/B conclusion.
- **Otherwise** (both gates are a definite PASS or FAIL), the 2x2 table
  applies:

| Gate-A | Gate-B | Conclusion |
|---|---|---|
| PASS | PASS | Continue Route A citing this gate as decisive evidence; Route B is *also* scientifically justified if the team prefers the simpler architecture. |
| FAIL | PASS | The *currently deployed* model isn't earning its complexity, but the belief framework has real signal -- refit Route A's GDBN branch on this protocol, or seriously consider Route B. Do NOT conclude "Bayesian belief is a dead end" from Gate-A alone. |
| PASS | FAIL | Keep Route A as-is; do NOT switch to Route B -- refitting on fresh data made things worse or no-better, so the frozen model's edge may be fragile rather than a robust property of the framework. |
| FAIL | FAIL | Reject Route B as specified. Keep Route A's GDBN branch only as a generic calibration/robustness component, not a decision-quality claim. Stop investing in the GDBN direction without a materially different protocol or model. |

## What gets compared

Four models, all exposed through the identical
`predict_action_rollout_batch(states, actions, belief_vecs, horizon, dt,
safe_distance, cvar_alpha, pedestrian_aggregation)` interface
(`gdbn.GDBNIntegration` / `risk_models.ConstantVelocityRiskModel`):

- **`cv`** -- `ConstantVelocityRiskModel`, no fitting, no belief.
- **`frozen_k3`** -- the actual Route-A model, loaded read-only. Never refit. (Gate-A)
- **`refit_k{2,3,4}`** -- freshly fit on *this* experiment's own
  `train_nonstationary` data; one K is selected on the disjoint
  `validation_nonstationary` split by NLL (subject to the same eligibility
  rule as `bayesian_pilot/evaluate_prediction_gate.py`: minimum per-mode
  sample count, minimum pairwise dynamics distance) and reported as
  `refit_k{K}` throughout. (Gate-B)

The candidate action grid (80 actions: 5 speeds x 16 headings) is locked
explicitly from `configs/env_belief_mdp.config` (`protocol.py::lock_action_grid`)
-- the same config `belief_mdp/train.py` uses for the actual deployed policy
and for fitting `frozen_k3`. Do not rely on `contracts.py`'s own
auto-discovery (`configs/env.config` from cwd): an earlier version of this
module pointed at `configs/env_gdbn.config` instead, which has a *different*
grid (`include_stop=true`, 81 actions, different `v_min`) -- silently
comparing a different action set than "the one the real policy considers."

## Data

Reuses `bayesian_pilot/protocol.py`'s `PROFILES` (`nominal`,
`train_nonstationary`, `heldout_nonstationary`), `BehaviorScheduler`, and
`InterventionORCA` verbatim -- no occlusion, no goal-switching in this round.

| Split                      | Profile               | Purpose                          | Episodes/seed/density |
|-----------------------------|------------------------|-----------------------------------|-------------------|
| `train_nonstationary`       | `train_nonstationary`  | fit refit-K GDBN                  | 80 |
| `validation_nonstationary`  | `train_nonstationary`  | select K + hyperparameters        | 20 |
| `heldout_nonstationary`     | `heldout_nonstationary`| final decision-level test         | 60 |
| `nominal`                   | `nominal`               | check for regression when nothing is happening | 40 |

With the default 5 seeds, this sums to the originally intended 400/100/300/200
**per density** (2000 episodes total across both densities) -- an earlier
version of this file used these same numbers as PER-SEED-PER-DENSITY counts,
silently inflating the real budget to ~10,000 episodes. Both **5-person** and
**20-person** densities are collected for every split
(`collect_dataset.py --density`). Five independent seeds by default
(`2407,3407,4407,5407,6407`).

Every episode records the full identity-stable observation sequence, the
actions taken, and per-pedestrian intervention-mode labels -- so
`obs[t+1 : t+1+horizon]` IS the real recorded pedestrian future, not a
prediction of it.

**`suite_seed` vs. `episode_seed`.** Each episode stores TWO seed fields, and
evaluation code must never conflate them: `episode_seed` is the per-episode
simulation RNG seed (unique per episode, via `protocol.episode_seed`);
`suite_seed` is the experiment-level identity passed via `--seed` (one of
2407/3407/4407/5407/6407 in the default 5-seed run), IDENTICAL across every
episode collected by one `collect_dataset.py` invocation.
`evaluate_action_ranking.py`'s bootstrap groups by `suite_seed` -- an earlier
version grouped by `episode_seed` instead (which is unique per episode), so
the "seed-block bootstrap" silently collapsed into per-episode resampling:
its block count equaled the episode count (e.g. 300), not the intended 5
experiment seeds, understating variance and making small real effects look
more significant than they are.

If the first formal run reports insufficient event decisions, run
`run_gate.py --topup_heldout --run_id <existing_run_id>` to collect
*additional* `heldout_nonstationary` episodes only (new `--seed` offset so
files/episode_seeds never collide with the first round, but `--suite_seed`
is passed explicitly so the extra episodes are still grouped under the
ORIGINAL experiment seed, not a spurious 6th one) and re-evaluate -- never do
a full rerun just to add statistical power.

## The counterfactual action-ranking evaluation

For sampled timesteps (`evaluate_action_ranking.py`, default stride 5, up to
40 decisions/episode), the SAME 80 candidate actions the real policy
considers are scored two ways:

1. **Ground truth risk (open-loop counterfactual oracle)**: roll the ROBOT
   forward under the candidate action (simple constant-velocity kinematics)
   and combine with the REAL recorded pedestrian future positions -- never a
   model's own prediction of them -- using the same clearance-risk transform
   every model uses (`GDBNIntegration._clearance_risk`).

   **This is explicitly an open-loop approximation, not closed-loop ground
   truth**: it assumes the recorded pedestrians' trajectory would not change
   had the robot taken a different action, which is false in general (ORCA
   pedestrians react to the robot). Two required mitigations:
   - Every evaluation runs at **H = 1, 3, and 5** (`--horizons`). The gate
     requires the regret-reduction sign to **agree across all horizons with
     enough data** (`horizon_agreement` check) -- if H=1 says GDBN is better
     and H=5 says CV is better, the approximation error likely dominates the
     signal and neither result can be trusted.
   - **If Gate-B passes**, the next step (not implemented in this module) is
     to re-verify on a sample of high-risk states using a real simulator
     branch rollout (actually re-running ORCA with the candidate action)
     before building a full Bayesian-MPC system on this result alone.

2. **Predicted risk**: each model's `predict_action_rollout_batch`, given its
   own online belief built by replaying the real observation history up to
   time `t` (`model.update()` each step, exactly as the online policy does).

Both are computed only after restricting to the **near-optimal-by-progress**
subset (default: top 20% of the 80 candidates by how much closer to the goal
they get the robot over the horizon) -- otherwise "always stand still"
trivially minimizes risk and the comparison is meaningless.

**Tie-tolerant ranking (Fix 2).** Top-1/top-k hit and regret are computed
against the true-risk *value* with an explicit tolerance
(`rank_and_hit`, `--rank_tolerance`, default 1e-3), NOT by argsort-ranking the
subset. Argsort forces a total order even among actions with essentially
identical true risk (very common when the robot is far from every
pedestrian) -- it would count a model's equally-safe-but-different choice as
a Top-1 *failure*, which is wrong.

**`event_near` vs. `event_any` (Fix 5).** A decision is stratified into the
`event_near` window only if an intervention-affected pedestrian is actually
within `--event_near_radius` (default 3m) of the robot at the time the event
is active (`event_window_labels`). `event_any` (any pedestrian anywhere has a
non-nominal mode) is reported for reference only -- at 20-person density it is
dominated by irrelevant far-away pedestrians and would dilute the gate with
"stress" that could not possibly affect the robot's decision.

**`opportunity_rate`.** Fraction of near-optimal-subset decisions where true
risk actually varies meaningfully across candidates (range >
`--opportunity_threshold`, default 0.05) -- reported per window, independent
of any model, as a sanity check that the protocol creates real decisions to
rank.

**Meaningful disagreement (Fix 6).** `disagreement_rate` (any different
choice) is reported, but the gate uses `meaningful_disagreement_rate`, which
excludes disagreements where the two chosen actions' true risk differs by
less than `--meaningful_diff_tolerance` (default 0.02) -- a "disagreement"
between two equally-safe actions is not evidence of anything.

**Collision/near-miss metrics.** Alongside the exponential clearance-risk
index, each decision also records `true_collision` (min clearance <= 0) and
`true_near_miss` (min clearance <= 0.10m) for the chosen action, so the
report never reduces safety to a single continuous index.

All confidence intervals are **seed-block bootstrap** (`bootstrap.py`): the
top-level resampling unit is the `suite_seed` (all of a seed's episodes
concatenated into one block), matching `belief_mdp/evaluate.py`'s convention
-- episodes within a seed, and decisions within an episode, are correlated.
Per-seed means are also reported (`per_seed_means`) alongside the pooled
bootstrap CI.

## Pre-registered pass/fail criteria (`run_gate.py::evaluate_one_gate`)

Applied identically to Gate-A (`frozen_k3` vs. `cv`) and Gate-B (`refit_k{K}`
vs. `cv`), on `heldout_nonstationary`'s **`event_near`** window at
**H = 5** (`PRIMARY_HORIZON`), **5-person** density (primary; matches the
current production scenario). Each gate returns a proper **three-state**
result, decided in this fixed priority order (never a bare bool -- collapsing
FAIL and INCONCLUSIVE produced self-contradictory reports in an earlier
version, e.g. a "FAIL" whose own explanation said the sample was too small to
conclude anything):

1. `sufficient_event_decisions` -- at least 200 `event_near`-window decisions
   at H=5. If not: **INCONCLUSIVE** (underpowered sample, not evidence).
2. `sufficient_opportunity` -- `opportunity_rate` at H=5 is at least 5% (the
   near-optimal action subset must actually contain meaningfully different
   true-risk options; this was computed but never checked in an earlier
   version). If not: **INCONCLUSIVE** (the protocol posed no real decisions
   to rank, so neither PASS nor FAIL is a fair reading).
3. `sufficient_disagreement` -- `meaningful_disagreement_rate` at H=5 is at
   least 5%. If real opportunities existed (step 2 passed) but the model
   almost never chose differently from CV anyway: **FAIL** (the belief
   branch had chances to matter and didn't take them -- this is a genuine
   negative result, not an underpowered one).
4. With data and real opportunities both adequate, everything below must
   hold or the result is **FAIL**:
   - `regret_reduction_significant` -- the 95% bootstrap CI of (CV's decision
     regret − model's decision regret) at H=5 has its **lower bound strictly
     above zero**. The single decisive criterion: genuine superiority, not
     "no worse than."
   - `top1_beats_cv_point_estimate` -- the model's top-1 safest-action hit
     rate point estimate exceeds CV's at H=5 (corroborating signal).
   - `spearman_noninferior` -- the model's ranking correlation with true risk
     at H=5 is not more than 0.02 below CV's (sanity check).
   - `horizon_agreement` -- among H in {1,3,5} with >= 200 decisions, the
     regret-reduction mean's sign never flips (see the open-loop-oracle
     discussion above).
   - `no_nominal_regression` -- on the `nominal` split (no stress events), the
     95% CI **lower bound** of (CV's regret − model's regret) is >= -0.01.
   - `no_20person_regression` -- **20-person density must also be checked, not
     just reported** (an earlier version let 20-person regress silently since
     only 5-person constrained PASS/FAIL, even though the paper's final
     scenarios include high-density crowds). At H=5, 20-person's
     regret-reduction 95% CI **lower bound** must be >= -0.01 -- a clear
     20-person regression forces FAIL even if 5-person looks perfect. If
     20-person has fewer than 200 `event_near` decisions, the would-be PASS
     is downgraded to **INCONCLUSIVE** instead of passing by omission.
5. If every check above holds: **PASS**.

## Running it

```bash
cd crowd_nav
PYTHONPATH=.. python3 -m py_compile bayesian_decision_gate/*.py
PYTHONPATH=.. python3 -m crowd_nav.bayesian_decision_gate.selftest        # GPU-free, no simulation
PYTHONPATH=.. python3 -m crowd_nav.bayesian_decision_gate.run_gate --smoke --run_id smoketest   # structural check ONLY, never a go/no-go signal
PYTHONPATH=.. python3 -m crowd_nav.bayesian_decision_gate.run_gate --formal --run_id <descriptive_run_id> --seeds 2407,3407,4407,5407,6407
```

(`PYTHONPATH=..` is required when invoking these `-m crowd_nav.bayesian_decision_gate.*`
modules directly from inside `crowd_nav/` -- from that directory `crowd_nav`
itself is not on `sys.path`. `run_gate.py`'s internal subprocess calls already
set this automatically; only manual/standalone invocations need it.)

Every `--formal` (and `--topup_heldout`) run requires an explicit `--run_id`,
which becomes a dedicated subdirectory
(`runs/bayesian_decision_gate/<run_id>/{data,models,prediction,action_ranking,report}/`)
-- this guarantees `collect_dataset.py`'s glob patterns can never silently
pick up stale seed files left over from a previous run.

`run_gate.py --formal` records code/config/dependency/data/model-checkpoint
SHA256 hashes (`report/provenance.json`) before evaluating: this module's own
`.py` files, `contracts.py`, `gdbn.py`, `risk_models.py`,
`bayesian_pilot/protocol.py`, `configs/env_belief_mdp.config`, the collected
data directory, and both the frozen and refit GDBN parameter directories. The
final `report/gate_result.json` has the full Gate-A/Gate-B `checks`
breakdown and the plain-language combined decision.

## Files

- `protocol.py` -- environment building (variable pedestrian count), thin
  reuse of `bayesian_pilot.protocol`'s profiles/scheduler/intervention-ORCA,
  `lock_action_grid` (deterministic 80-action grid), disjoint seed-base
  bookkeeping for this experiment.
- `collect_dataset.py` -- collects one (split, density, seed) episode file.
- `fit_models.py` -- fits/selects the `refit_k{K}` model; sanity-loads all
  four model families.
- `evaluate_prediction.py` -- ADE/FDE/NLL/Brier/ECE/AUC with CV's covariance
  FROZEN from train/validation only; reported, never gating.
- `evaluate_action_ranking.py` -- the decisive counterfactual action-ranking
  evaluation described above, run at H=1/3/5.
- `bootstrap.py` -- seed-block bootstrap CI, Spearman correlation, top-`k`
  hit rate.
- `run_gate.py` -- end-to-end orchestration, unique `--run_id`s, hashing,
  Gate-A/Gate-B three-state (PASS/FAIL/INCONCLUSIVE) pre-registered criteria
  + combined decision table, `--topup_heldout` for adding statistical power
  without a full rerun (passes `--suite_seed` explicitly to keep the extra
  episodes grouped under the original experiment seed).
- `selftest.py` -- GPU-free correctness checks for all of the above,
  including the suite_seed-vs-episode_seed bootstrap grouping test and
  synthetic PASS/FAIL/INCONCLUSIVE coverage (both gates, both densities, the
  combined decision table, and INCONCLUSIVE propagation).
