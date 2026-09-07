# Belief-MDP: Bayesian Belief-Conditioned Constrained Navigation

Isolated pilot. Nothing under `belief_mdp/` touches the frozen full-crowd
calibration paper (`policy/bayesian_fullcrowd_risk_value.py`, its results in
`runs/eval35_fullcrowd_*`, or the LaTeX draft). That paper stays the
submission fallback regardless of what happens here.

**Revision history:** two rounds of code review, both before any full
training was trusted. Round 1 found the 80/81-action config mismatch, the
broken `state_only` ablation, the cancelable risk penalty, the mismatched
ablation dimensions, the per-step corruption permutation, the nominal-only
evaluation, and the per-episode bootstrap (see "Fixed in round 1" below).
Round 2, after a real smoke run succeeded, found that round 1's fix for the
cancelable penalty still had Q_C learn to predict GDBN's own risk feature
rather than real simulator safety outcomes (circular), that the resulting
risk scale could still let beta dominate task value, that the "held-out"
stress profile was not actually held out, that the formal protocol's time
limit didn't match the paper's, and that nothing let two belief-MDP
checkpoints be compared directly against each other (only each against the
frozen baseline) (see "Fixed in round 2" below). Round 3, after round 2's
smoke test actually ran clean on real GPU hardware (80-action grid
confirmed, no NaNs, byte-identical repeated evaluation, Q_C initializing at
~0.01 as intended -- see "Round 2 smoke verification" below), found that
`particles` (the GDBN particle-filter count, which changes the Bayesian
belief's own stochastic approximation quality) was never added to the
checkpoint-parameter lock, and that nothing in training distinguished "Q_C
is correctly initialized" from "Q_C has actually learned to predict risk"
(see "Fixed in round 3" below). Round 4, after a full real training run
(`action_conditioned`, `beta=0.5`, seed 2407, 2700 episodes) actually
completed on the 4090: `q_c_action_std` and `beta_top1_change_rate` both
confirmed real, stable, non-constant Bayesian participation in decisions
throughout training -- but `Q_R` inflated from O(1) to ~20 then collapsed
to ~3-4, BC loss rose toward chance level (`log(80)=4.38`), and navigation
performance peaked mid-training (episode 1900) then degraded by the final
episode. Root cause: the expert imitation loss was ordinary cross-entropy
on `Q_score` itself, which has no stopping point and kept inflating Q
values in search of an ever-larger margin, independent of whether they
still meant anything as returns (see "Fixed in round 4" below). None of
the results referenced anywhere in this project came from any pre-fix
version — the only completed full run came from the code round 4 replaced.

## Why this exists

Every previous integration in this project (hard veto, CVaR-adaptive
threshold, belief-conditioned successor injection, trained distributional
value heads, and the 5-pedestrian `belief_space_rl` pilot) added Bayesian
information to an already-converged, frozen Mamba-VL decision head. The
open question this pilot tests is whether making belief part of the
decision state from the first gradient step of training -- not injected
after convergence -- changes the outcome. It is not a claim that this will
beat the frozen baseline; it is the most literal test of that specific
hypothesis this project has run.

## Architecture

As of round 8/9 (see below), the network is two fully separate encoder
stacks with **no shared layer or joint tensor** between them:

```
                     TaskEncoder (-> Q_R only)
history (5 nearest peds, frozen Mamba view)
        |
        v
frozen Mamba spatial+temporal encoder ---> context (256-d) --\
                                                                \
action kinematics: vx, vy, speed, goal-alignment,               v
progress, turn-cost (6 features) ------------------------> [concat] -> Q_R(b,a)

                     RiskEncoder (-> Q_C only)
full crowd (<=20 peds) -> GDBN belief b_t ---------------------\
                              |                                 \
                              v (per candidate action)           v
        action-conditioned posterior rollout                [concat] -> Q_C(b,a)
        (6 Bayesian risk features) ------------------------/

                     combined at decision time only:
          Q_score(b,a) = Q_R(b,a) - beta * Q_C(b,a)
          Q_R: task-return Bellman + MC-return target (real env reward)
          Q_C: risk-cost Bellman + MC-return target (REAL simulator
               collision event + real minimum clearance, never GDBN's own
               risk prediction -- see round 2, item 1)
                                  |
                                  v
                            argmax_a Q_score(b,a)
```

`Q_R` and `Q_C` are both learned, each with its **own** Bellman target (see
round 1 item 3 and round 2 items 1-2 below). `beta` is a **fixed** hyperparameter,
swept over `{0, 0.25, 0.5, 1.0}` across separate full training runs — not a
Lagrangian dual variable auto-tuned during training. A learned dual
variable can converge toward zero for the same reason a learned mixing
weight might, silently re-litigating "is Bayesian information big enough to
matter" instead of answering "how does behavior change as Bayesian weight
increases," which is what a fixed sweep answers directly.

An earlier design (through round 7/V5) fused Mamba's context into the
belief representation through a bounded, belief-gated correction
(`BeliefMDPQNetwork.refine_belief`), which — despite being bounded — did
give Q_C a real gradient path from Mamba context, and gave Q_R a real
gradient path from belief, through the shared joint tensor both heads read
from. That cross-talk was removed in round 8/9: TaskEncoder (Mamba context
+ kinematics) feeds Q_R exclusively; RiskEncoder (GDBN belief + risk
features) feeds Q_C exclusively. Verified via gradient-isolation tests
(zero `d(Q_R)/d(belief)` and zero `d(Q_C)/d(context)`).

Training keeps the project's established two-stage curriculum: Stage 1 is
teacher behavior cloning with belief already present in every input; Stage
2 is Double-DQN online RL with a decaying margin-loss weight. Since round
8/9 the Stage 1/DAgger teacher is the frozen Mamba-VL checkpoint's own
one-step value-lookahead (`runtime.BeliefMDPFeatureEngine.teacher_scores`),
not ORCA — see "Round 7 full training verdict" below for why. The frozen
checkpoint's value head is used for this and for the paired evaluation
baseline in `evaluate.py`; it is never fine-tuned or backpropagated into.

## Fixed in round 1 (all found by code review before any full run)

1. **Action grid mismatch (81 vs 80 actions).** `configs/env_gdbn.config`
   defines `include_stop = true` (5 speeds x 16 headings + stop = 81), but
   the frozen checkpoint (`rl_model_ep10000_T24.pth`) was trained on the
   80-action grid in `configs/env.config` (`sampling = exponential`,
   `include_stop = false`). Loading the wrong grid silently changed the
   action count and broke the checkpoint's q-head shape (`[80,256]` vs
   `[81,256]`) — this is what the first smoke test actually failed on, not
   a theoretical concern. Fixed by adding `configs/env_belief_mdp.config`,
   a dedicated config with the correct 80-action `[policy]` section (copied
   verbatim from `env.config`) and the full-crowd `[reward]/[sim]/[humans]
   /[robot]/[orca]` sections (copied from `env_gdbn.config`; `[sim]` is
   overridden at runtime per scenario regardless).
2. **Candidate features had no action identity.** The six Bayesian rollout
   features (risk, tail risk, tail excess, entropy, klda, epistemic value)
   were the *only* action-dependent signal reaching the network. This made
   `state_only` (which broadcasts the same risk to every action by design)
   produce 80 identical Q-values, so argmax always returned action 0 — the
   ablation was not measuring anything. Fixed by adding six real kinematic
   features per action (`vx, vy, speed, goal_alignment, progress,
   turn_cost`, computed directly from robot state and the action vector,
   independent of any risk model) so every belief mode can still
   distinguish actions by their actual motion.
3. **Q_task could silently cancel the risk penalty.** The original
   `Q = Q_task - beta*Risk` was trained with a single Bellman target built
   only from environment reward. Nothing stopped `Q_task` from learning
   `+beta*Risk` on top of the true task value, exactly canceling the
   penalty in the combined output — sweeping beta would then have no
   reliable behavioral effect. Fixed (round 1) by splitting into twin
   heads with separate Bellman targets, combined only at decision time.
   Round 1's `Q_C` target was still the GDBN rollout's own risk feature,
   which round 2 found to be a separate, deeper problem — see round 2
   item 1.
4. **`beta=0` did not mean "no Bayesian information."** Even with the
   explicit penalty zeroed, GDBN's risk/entropy/klda features still
   entered `candidate_encoder`, and the belief vector still entered
   `belief_encoder` — a learned `Q_R` could still exploit them freely. This
   confused "no explicit penalty" with "no Bayesian information at all."
   Fixed by adding a fifth, distinct `no_belief` condition that zeros both
   the belief vector and the six risk features (keeping only kinematics and
   Mamba context) and forces `beta=0`; `beta=0` on `action_conditioned`
   remains a separate, meaningful condition ("no explicit risk term, but
   Bayesian features are still visible to the learned heads").
5. **The four ablations were not the same architecture.**
   `ConstantVelocityRiskModel` hardcodes `K=1` (`risk_models.py:16`)
   regardless of what's passed in, so the `cv` condition produced an
   11-dimensional belief vs GDBN's original 7 — literally a smaller
   network, not an ablation of the same one. Fixed by padding every belief
   vector to a canonical `MAX_K=3` (GDBN's own K) regardless of the
   underlying filter's real mode count, so `belief_dim` and `candidate_dim`
   are now identical across all five conditions (verified: a K=1 filter's
   belief vector pads to the same 11-d shape as K=3's).
6. **Corrupted-belief permutation was resampled every step.** Reshuffling
   pedestrian identities at every timestep is closer to independent
   per-step noise than a stable wrong mapping, and is a weaker, different
   corruption than intended. Fixed: the permutation is now drawn once per
   episode in `reset()` and reused for every `encode()` call within that
   episode.
7. **Bootstrap resampled individual episodes, not blocks.** Episodes
   sharing a seed are not independent draws; treating them as i.i.d.
   understates uncertainty and doesn't match the project's own established
   block-bootstrap convention. Fixed: `evaluate.py` now resamples whole
   `(scenario, profile, seed)` blocks with replacement.
8. **Formal evaluation was nominal-only.** ORCA pedestrians are close to
   constant-velocity most of the time under the `nominal` profile — exactly
   the regime a CV-based ablation looks best in. Fixed in round 1 by adding
   a second profile; round 2 found that profile ("decision_stress") was
   mislabeled -- see round 2 item 3.

## Fixed in round 2 (found after round 1's smoke test actually ran)

1. **Q_C was trained on GDBN's own prediction as ground truth.**
   `train.py`'s `risk_cost` was `features.candidates[action_index,
   risk_feature_index]` — the GDBN rollout's own tail-risk estimate for the
   chosen action. `Q_C` then learned "accumulated predicted GDBN risk," not
   "accumulated real collision risk." GDBN was simultaneously the input and
   its own supervision target — even a strong result could not show GDBN
   tracks real safety, since nothing independent of GDBN was ever checked
   against it. Fixed: `risk_cost` is now `1.0` if the environment's own
   `info["event"] == "collision"` at that step, else a graded near-miss
   term from the environment's own `dmin` (`clip((safe_distance - dmin) /
   safe_distance, 0, 1)` when `dmin < safe_distance`, weighted `0.2` so it
   doesn't dominate the terminal collision signal, `0` otherwise). GDBN's
   risk features remain visible to the network only as *inputs* to
   `candidate_encoder`; they no longer touch any training label.
2. **Risk scale could let beta dominate regardless of its value.** Per-step
   GDBN risk in `[0,1]` accumulated under `gamma=0.99` Bellman recursion can
   reach the tens, while `Q_R` (built from `success_reward=1.0`,
   `progress_reward=0.01`/step, etc.) stays roughly `O(1)`; `beta=0.25`
   could already have overwhelmed task value. Fixed as a side effect of
   item 1: cost is `0` at every non-terminal step except a small bounded
   near-miss term, so `Q_C` now approximates a discounted eventual-collision
   probability, staying within roughly `[0,1]` — the same scale as `Q_R`,
   so `beta` sweeps now have an interpretable, comparable meaning.
3. **The "held-out" stress profile was not held out.** The removed
   `train_risk`/`decision_stress` profiles both copied
   `heldout_nonstationary`'s exact `duration_steps`, `turn_degrees`, and
   `slow_scale` and only changed `event_rate` (0.06 / 0.08) — a
   higher-intensity replay of a distribution training already saw, not an
   unseen one. Fixed: training now alternates `nominal` and the project's
   own `train_nonstationary` profile; formal evaluation uses the project's
   own `heldout_nonstationary` profile (both from
   `bayesian_pilot/protocol.py`, which differ from each other in
   `duration_steps`, `turn_degrees`, and `slow_scale`, not just rate — a
   genuine train/test split, not a relabeled intensity knob).
4. **Formal protocol's time limit didn't match the paper's.**
   `env_belief_mdp.config` had `time_limit = 25` (100 internal steps at
   `time_step=0.25`), while the actual full-crowd paper protocol
   (`tools/run_eval_queue.py`) uses `--time-limit 35` (140 steps).
   `evaluate.py --max_steps 200` never overrode this, since the
   environment's own internal cutoff fires first. Fixed:
   `env_belief_mdp.config` now sets `time_limit = 35`.
5. **No way to prove `action_conditioned` beats the other ablations.**
   `evaluate.py` only ever paired each checkpoint against the frozen
   Mamba-VL baseline; two belief-MDP checkpoints (e.g.
   `action_conditioned` vs `cv`) were never compared against each other on
   the same episodes, which is what the ablation battery's central claim
   requires. Fixed: `evaluate.py --episode_records_output` now writes every
   episode's `(checkpoint, method, scenario, profile, seed, test_case,
   outcome, steps, min_dmin)` to CSV; the new `compare_checkpoints.py`
   joins two checkpoints' records on `(scenario, profile, seed, test_case)`
   and runs the same seed-block bootstrap directly on their difference.
6. **Bootstrap blocks were finer than necessary and mixed profiles.**
   Round 1 resampled `(scenario, profile, seed)` blocks independently, so a
   single bootstrap replicate could draw one scenario's block for a seed
   without its other scenarios, discarding whatever correlation that seed
   carries across scenarios, and pooled `nominal`/held-out differences into
   one CI. Fixed: the resampling unit is now the seed; each replicate keeps
   all six scenarios for a resampled seed together, and `nominal` /
   `heldout_nonstationary` get separate reported CIs.

## Round 2 smoke verification (real GPU run, not a claim from reading code)

`action_conditioned --smoke` was actually run on the project's 4090
(`/root/miniconda3/envs/mamba`, torch 2.9.1+cu128): 80-action grid
confirmed (`belief_dim=11`, `candidate_dim=12`), no NaNs in any real
optimizer step, checkpoint save/load round-tripped, all five
`artifact_hashes` recorded. `evaluate.py --smoke` was run twice
independently; the two runs' per-episode CSV records were byte-identical
(`diff` on the raw files, not just the aggregate summary). `Q_C` measured
~0.0099 mean / ~0.0106 p95 across the first real optimizer steps, matching
the ~0.01 initialization target instead of `softplus(0) ~= 0.693`.

This confirms the round-1/round-2 fixes run without crashing and that
`Q_C` starts at the intended scale -- it does **not** confirm `Q_C` has
learned anything about risk yet (see round 3, item 2): with only a few
smoke episodes, `q_c_std` stayed near zero and `td_target_c` stayed near
its initial value, which is expected from so little data and is not
evidence either way about whether the risk head trains correctly at real
scale.

## Fixed in round 3 (found after the round-2 smoke run's real numbers were reviewed)

1. **`particles` was not locked to the checkpoint.** The GDBN
   particle-filter count changes the belief's own stochastic approximation
   (more particles = lower-variance mode/entropy estimates), so evaluating
   with a different particle count than training used is evaluating a
   subtly different Bayesian feature distribution, not just a slower or
   faster run. `particles` was missing from `PARAMS_FROM_CHECKPOINT`, so
   `evaluate.py` silently used its own CLI default (or whatever the user
   passed) with no check against what the checkpoint was trained with.
   Fixed: `particles` is now in `PARAMS_FROM_CHECKPOINT`, defaults to
   `None` (must come from the checkpoint) like every other locked
   parameter, and is subject to the same mismatch-is-an-error-unless-
   `--allow_feature_override` rule. `--smoke` forces its own reduced
   particle count for speed, and correspondingly auto-enables
   `--allow_feature_override` (smoke is documented everywhere as
   correctness-only, not a decision; the formal, non-smoke path still
   enforces every locked parameter strictly).
2. **No signal distinguished "Q_C is correctly initialized" from "Q_C has
   learned to predict risk."** The round-2 smoke numbers (`q_c_mean`,
   `q_c_std`, `td_target_c_mean` all sitting close to the ~0.01 init value)
   are exactly what a *correctly initialized but untrained* `Q_C` looks
   like, and the smoke run was far too short to tell the difference from a
   working one. Fixed: `train.py`'s per-update log now also reports
   `positive_risk_fraction` (fraction of the sampled minibatch with a
   collision label -- `Q_C` cannot learn from an all-zero batch),
   `q_c_action_std` (how much `Q_C` varies across the 80 candidate actions
   for the same state -- near zero means it still cannot tell dangerous
   actions from safe ones), and `beta_top1_change_rate` (how often
   `beta*Q_C` actually flips which action `argmax` prefers, vs `Q_R` alone
   -- zero forever means beta is inert; near 100% forever means risk has
   swamped task value). A real (non-smoke) `action_conditioned, beta=0.5`
   training run should show `positive_risk_fraction` occasionally nonzero,
   `q_c_action_std` rising visibly above its near-zero initial value, and
   `beta_top1_change_rate` nonzero but not pinned near 100%. If `Q_C` stays
   effectively constant through a full run, collision-trajectory
   prioritized replay sampling is the indicated next fix -- before that is
   confirmed necessary, it is not implemented here, since a full run's
   real metrics are the only way to know whether it is actually needed.

Comparisons between `action_conditioned`, `cv`, and `no_belief` for the
paper-level claim need at least 3 independent training seeds per condition,
not one -- 10 evaluation seeds exercise the same trained policy repeatedly
and cannot substitute for training-time randomness (initialization, replay
sampling order, exploration noise all differ seed to seed). This is a
scope note for when the full ablation battery is launched, not something
`train.py`/`evaluate.py` enforce automatically; each condition's 3 seeds
are 3 separate `train.py` invocations with different `--seed`.

## Round 3 full training verdict (real 2700-episode GPU run, seed 2407)

Two separate, independent findings, not one:

1. **The Bayesian branch genuinely participates in RL decisions.**
   `q_c_action_std` stabilized at ~0.23-0.26 for the entire second half of
   training (not decaying toward the ~0 it starts at) and
   `beta_top1_change_rate` stayed nonzero throughout (peak ~8%, never near
   0% or 100%) -- `Q_C` is not a dead constant, and `beta` measurably
   changes which action gets chosen without ever dominating `Q_R`.
2. **The task-value head diverged.** `q_r_mean` rose from ~13 to ~19 then
   fell to ~4; BC loss rose from ~2.39 toward `log(80)=4.38` (chance
   level); evaluated SR/CR peaked at episode 1900 (nominal 50%/0%, stress
   57%/7%) then degraded by episode 2700 (nominal 10%/17%, stress 17%/17%).
   `positive_risk_fraction` also fell from ~0.6% to ~0.14% over training --
   this is *not* mainly because the policy got safer (the last 200
   episodes still ran roughly 17-20% collision rate): `risk_cost` is 1.0
   only at the single terminal transition of a colliding episode, so even
   at a constant ~20% episode-level collision rate, positive-labeled
   *transitions* are diluted to roughly (0.20 / average episode length) of
   the replay buffer -- a few tenths of a percent is the expected dilution
   from the sparse one-step label, not evidence risk was being resolved.

Finding 1 is why this is still worth pursuing. Finding 2 is why the
un-fixed code must not be used for the ablation battery or trusted as
"the policy converged."

## Fixed in round 4 (root cause of the round-3 divergence)

1. **Cross-entropy BC loss inflated Q without bound.** `bc_loss =
   F.cross_entropy(q_score, expert_action)` used the same `Q_score` both as
   a Bellman-fitted return estimate and as a classification logit.
   Cross-entropy has no "close enough" -- it keeps demanding a larger gap
   between the expert action's score and every other action's, forever,
   regardless of whether the values still mean anything as returns. This
   is what pushed `Q_R` to ~20. Fixed: replaced with a DQfD-style
   **fixed-margin** expert loss (`expert_margin_loss`, margin=0.05) that
   contributes zero once the expert action already leads by the margin --
   verified directly: an expert already leading by 0.5 gives loss 0.0; an
   expert leading by only 0.01 (< the 0.05 margin) gives a small positive
   loss; an expert that is not even the arg max gives a large loss capped
   by how far behind it is, never unbounded.
2. **Only the single terminal transition of a colliding episode supervised
   `Q_C`.** Fixed: `compute_mc_returns` does a reverse pass over each
   finished episode and gives every preceding transition a geometrically-
   decaying Monte-Carlo collision return (verified: a 4-step trajectory
   ending in collision gives all 4 steps a nonzero `mc_collision_return`,
   not just the last one; a non-colliding trajectory gives all zeros). Both
   `Q_R` and `Q_C` now train against **both** a one-step TD target and this
   per-episode MC target (`L = L_TD_R + 0.25*L_MC_R + L_TD_C + 1.00*L_MC_C
   + lambda_e*L_margin`).
3. **ORCA demonstrations could be crowded out by online experience.** The
   single circular replay buffer let 300 episodes of demonstrations get
   diluted and eventually evicted as more online experience accumulated.
   Fixed: a `permanent=True` demo buffer (verified: stops accepting new
   transitions once at capacity instead of overwriting the oldest ones) is
   mixed with the circular online buffer at a fixed 25%/75% ratio
   (`sample_mixed`) for every gradient step in Stage 2, so demonstrations
   are never diluted away regardless of how long training runs.
4. **Nothing would have caught the same divergence recurring.** Fixed:
   `train.py` now logs `q_r_abs_p95`, `positive_mc_collision_fraction`,
   `margin_violation_rate`, and `expert_top1_agreement` every update, and
   aborts the run automatically (writing `aborted_reason` to the manifest)
   if `q_r_abs_p95 > 3.0`, any loss/diagnostic is non-finite, or mean SR
   drops by more than 20 percentage points between two consecutive
   evaluations -- the same failure mode should stop the run within one
   `eval_interval`, not silently produce another 1900-episode round trip.
5. **Checkpoint selection used a noisy 30-episode in-training quick-eval.**
   The round-3 run's picked "best" checkpoint (episode 1900) happened to be
   the actual peak, but a 30-episode estimate picking a lucky point rather
   than a stable one is a real risk, and the picked checkpoint was never
   independently re-validated. Fixed: `train.py` now saves an unconditional
   checkpoint every `eval_interval` (`checkpoint_ep{N}.pth`) and no longer
   picks a "best" during training; `select_checkpoint.py` evaluates every
   saved checkpoint afterward on independent *validation* seeds (base
   2,000,000 -- disjoint from both training's own quick-eval seed offsets
   and `evaluate.py`'s formal *test* seeds, base 5,000,000) with at least
   20 episodes per scenario per profile, and picks the checkpoint with the
   best SR/CR that is not badly imbalanced between `nominal` and
   `heldout_nonstationary`. The formal test seeds are spent exactly once,
   on whichever checkpoint this script selects -- never during selection.

## Round 4 full training verdict (V3, seed 2407) and round 5 fixes

Round 4's fixed loss composition ran clean (no NaN, no Q_R blowup) but
Stage 1 itself did not converge: held-out top-3 expert-set agreement
70.2% (borderline), margin violation 67.1% (target <30%), and post-Stage-1
navigation was 6.7%/16.7% (nominal SR/CR) and 6.7%/40.0% (stress) --
clearly not basic navigation competence yet, let alone something to spend
a 2400-episode RL budget refining. **The bottleneck was never "did Bayesian
belief enter RL" (round 3 already confirmed it did) -- it is that Stage 1
imitation had not converged before RL started.**

1. **Single-action margin was too strict; fixed to a top-K expert set.**
   ORCA's continuous velocity rarely lands exactly on the single nearest of
   80 discrete actions; several neighbors are often near-equivalent.
   `nearest_action_indices(k=3)` (default `--expert_set_size 3`) and
   `expert_set_margin_loss` now only require the arg max to be *any* of the
   k nearest actions, not the single nearest one specifically (verified:
   argmax landing on any of a 3-action expert set gives zero loss; landing
   outside it gives positive loss).
2. **DQfD margin loss was applied to online (self-generated) samples too.**
   75% of every batch is online experience; supervising it with a
   large-margin "must match ORCA" loss keeps that whole 75% anchored to
   ORCA's instantaneous suggestion for the rest of training, contradicting
   "the learned policy drives decisions." Fixed: the margin loss is now
   weighted by `is_demo` so only the permanent demo buffer's transitions
   contribute (verified: the demo-weighted loss over a batch equals exactly
   the demo rows' own loss, ignoring online rows entirely; a batch with
   zero demo rows gives a finite 0, not a NaN from dividing by zero).
3. **Stage 1b's Q_R divergence check used a single update, not an EMA.**
   Inconsistent with the online-phase check (round 4 already made that one
   EMA-based). Fixed: Stage 1b now tracks its own EMA and only aborts on
   sustained divergence (verified: a single spike 100x the limit, surrounded
   by safe updates, does not trip the abort; genuinely sustained divergence
   does).
4. **Online-phase diagnostics mixed demo and online rows, making their
   meaning unclear** (e.g. "expert agreement" computed over 75% online
   experience, which was never asked to imitate anything). Fixed: added
   `demo_top1_in_set` / `demo_margin_violation` (is_demo==1 only) and
   `online_beta_top1_change` / `online_qc_action_std` (is_demo==0 only)
   alongside the existing whole-batch versions. Also added `velocity_error`
   / `angle_error_deg` between the chosen and expert action, read directly
   from the candidate features' kinematic columns.
5. **Stage 1 failure only printed a WARN and burned the full RL budget
   anyway.** Fixed: a hard gate now runs right after Stage 1b, on a
   dedicated 120-episode (20 per scenario x 6) navigation check plus the
   held-out demo validation set. All four of `--stage1_min_top1_in_set`
   (0.85), `--stage1_max_margin_violation` (0.30), `--stage1_min_sr` (0.75),
   `--stage1_max_cr` (0.15) must pass or the run stops there, saving
   `stage1_failed_model.pth` and writing `status=stage1_failed` to the
   manifest, instead of spending 2400 RL episodes on a policy that cannot
   navigate yet. `--smoke` disables this gate (a few dozen updates cannot
   meaningfully pass or fail it); the non-smoke path enforces it strictly.
   Strengthened Stage 1 hyperparameters to match: `--demo_episodes 600`,
   `--demo_updates 10000`, `--lambda_e_demo 5.0` (up from 300/3000/1.0).
6. **`select_checkpoint.py`'s CR/TR caps (15%/60%) were arbitrary fixed
   numbers with no reference point**, and silently produced a
   `selected_model.pth` even when nothing was eligible. Fixed: it now
   evaluates the frozen Mamba-VL baseline on the exact same validation
   episodes and defines eligibility *relative to that baseline*
   (`--sr_slack 0.02`, `--cr_slack 0.01`, `--tr_slack 0.02` per profile);
   if no checkpoint is within slack of baseline on both profiles, the
   script exits nonzero and does **not** write `selected_model.pth` --
   there is no silent "least-bad" fallback anymore for this specific
   failure mode (verified via synthetic baseline/candidate rates).
7. `selftest.py` adds five GPU-free regression checks (top-3 margin
   zero/positive, demo-only margin weighting including the zero-demo-row
   edge case, MC returns covering a whole colliding trajectory, EMA
   ignoring a single spike but catching sustained divergence, and no
   `selected_model.pth` without an eligible checkpoint) -- run with
   `python3 -u belief_mdp/selftest.py` before any GPU smoke test.

## Round 6: V4 (strengthened Stage 1 alone) still failed -- DAgger added (round 7)

V4 (600 demo episodes, 10000 offline updates, `lambda_e_demo=5.0`) ran clean
(Q_R EMA stayed ~1.05 for all 10000 updates -- the round-4/5 fix holds) but
the gate still failed: held-out top-3 agreement plateaued at ~78% after
~5000 updates (more updates did not help further), and, notably, **nav-level
collision rate got worse than the smaller V3 attempt** (nominal/stress CR
70.8%/85.0% vs V3's 16.7%/40.0%), despite better per-step imitation
agreement. Root cause: **sequential distribution shift, not insufficient
training.** All Stage 1 training states come from ORCA's own trajectory;
the first time the learned policy picks a different action than ORCA would,
it can reach a state ORCA's demonstrations never covered, and errors
compound step over step -- a per-step success probability of 78.4% compounds
to `0.784^50 ~= 0.000005` over a 50-step trajectory, which is consistent with
good per-step numbers and terrible full-episode outcomes. More offline
updates on the same fixed demo set cannot fix this; only visiting the states
the *policy* reaches and getting them labeled can.

1. **Fixed the velocity/angle-error diagnostic** (round 5 computed it
   against `batch["action"]`, which during forced-expert Stage 1 collection
   *is* the expert action by construction, making the "error" tautologically
   zero). Now uses `score_action` (`q_score.argmax`) -- what the network
   would actually choose -- against the expert action.
2. **Fixed `evaluate_demo_validation`'s RNG.** It was sampling with
   replacement from the *training* RNG, silently perturbing every later
   training batch's draw order and potentially over/under-counting held-out
   transitions. Replaced with `ReplayBuffer.deterministic_chunks` -- a fixed,
   no-replacement pass over every stored transition, no RNG at all
   (verified: covers every transition exactly once, no duplicates/gaps).
3. **Added DAgger (Stage 1c).** Runs only if the Stage 1b gate fails.
   `--dagger_orca_prob_schedule` (default `0.8,0.6,0.4,0.2,0.0`) sets, per
   round, the probability ORCA's action is actually *executed* --
   `run_episode`'s `expert_prob` parameter (renamed from the old boolean
   `force_expert`). ORCA is still queried for its label at every step
   regardless of who acts; DAgger's whole point is aggregating that label at
   states the *policy* visits. Each round collects
   `--dagger_episodes_per_round` (150) new episodes into the permanent demo
   buffer, then runs `--dagger_updates_per_round` (2000) offline updates and
   re-checks the gate, stopping early the first round it passes.
4. **Split `is_demo` into two flags** (`model.ReplayBuffer`):
   `has_expert_label` (does this transition have a valid ORCA label to
   imitate -- true for Stage 1a and every DAgger round regardless of who
   acted, false for pure Stage 2) gates the margin loss exactly where
   `is_demo` used to; `is_expert_action` (was the actually-*executed* action
   ORCA's) is new bookkeeping that does not gate anything -- TD/MC targets
   always supervise whatever action was actually taken and its real outcome,
   independent of this flag.
5. **Added a gradient-conflict diagnostic**, `compute_gradient_conflict`:
   cosine similarity between `grad(value_loss)` and `grad(margin_loss)` over
   every shared parameter, via `torch.autograd.grad` (never touches
   `.grad`/`optimizer.step()`, so it cannot affect real training). Logged
   every `stage1_log_every` updates during Stage 1b and every DAgger round.
   **Dueling Q (separating `V_R`/`A_R`) is deliberately not implemented
   yet** -- it addresses a specific, different failure mode (the same Q_R
   output serving both return regression and action-ranking) than the
   distribution-shift problem DAgger targets, and should only be built once
   this cosine is confirmed persistently negative (e.g. mean < -0.2) across
   many measurements, not assumed from the DAgger result alone.
6. Ablations (`cv`, `no_belief`, etc.) must reuse the exact same Stage
   1a/1b/DAgger procedure when their turn comes -- otherwise a difference
   between conditions could be DAgger's own contribution to closing the
   distribution-shift gap, not anything about the belief source itself.

## Round 7 full training verdict (V5, DAgger, seed 2407) and the round 8/9 redesign

V5 ran all 5 scheduled DAgger rounds (`teacher_prob` 0.8/0.6/0.4/0.2/0.0);
none passed the Stage 1 gate. DAgger produced real but highly non-monotonic
improvement: round 1 jumped nominal/stress SR from the Stage-1b baseline
(3.3%) to 37.5%, round 2 *crashed* to 8.3%, and round 5 (the best of the
five) reached 44.2%/49.2% nominal SR/CR -- clearly better than no DAgger at
all, but still far short of the 75%/15% gate. The gradient-conflict
diagnostic across all 5 rounds' logged measurements showed a mean of
**+0.051** (not the sustained negative correlation, e.g. mean < -0.2, that
would justify Dueling Q), so that architecture change is still not
implemented.

Two structural issues, not just "needs more DAgger rounds," were diagnosed
from these numbers and fixed in the same pass (round 8/9, see the module
docstring in `train.py` for the full rationale):

1. **ORCA replaced as the Stage 1/DAgger teacher with the frozen strong
   Mamba-VL checkpoint's own one-step value-lookahead**
   (`runtime.BeliefMDPFeatureEngine.teacher_scores`, ported from
   `belief_space_rl.runtime.BeliefFeatureEngine.teacher_scores`). ORCA is a
   classical planner with its own risk model, unrelated to the Mamba-VL
   policy this project is trying to approximate and augment with belief --
   V5's imitation target and its eventual RL objective could legitimately
   disagree about what a good action looks like for reasons that have
   nothing to do with the belief branch at all. `is_expert_action` was also
   fixed to record the literal `use_expert` draw rather than an
   `action_index == expert_index` equality check (which could be
   coincidentally true under exploration).
2. **`BeliefMDPQNetwork` rebuilt with fully separate TaskEncoder/RiskEncoder
   paths** (`model.py`): Mamba context + action kinematics feed Q_R only;
   GDBN belief + per-action risk features feed Q_C only; no shared layer or
   joint tensor between them (the previous `refine_belief`/`context_gate`
   design let Mamba context leak into Q_C and belief leak into Q_R).
   Verified via gradient-isolation tests (zero gradient of Q_R w.r.t.
   belief/risk inputs and vice versa).
3. **Five-way stratified replay** (`model.sample_stratified`,
   25/35/15/15/10 for teacher_demo/recent_dagger/older_dagger/collision/
   online_rl) replacing the single permanent "demo" buffer DAgger's rounds
   and Stage 1a's fixed demonstrations previously all shared undifferentiated.
   Collision trajectories are additionally prioritized into their own
   rolling-window stratum regardless of which stage produced them.
4. **Convergence-based per-round stopping**
   (`train.run_dagger_round_updates`) replacing the fixed
   `--dagger_updates_per_round` update count: checks every
   `--stage1_log_every` updates whether a nav SR/CR composite improved,
   stops after `--dagger_convergence_patience` consecutive non-improving
   checks, and reverts to the round's best-seen checkpoint (V5's round-2
   crash is exactly the kind of non-monotonic dip this is meant to recover
   from automatically instead of carrying a regressed checkpoint into the
   gate check). Five more diagnostics are logged at each check for
   visibility (teacher/DAgger-state top-3 hit rate, Q_C collision AUC/Brier,
   Q_R scale, beta-top1-change fraction) but are not folded into the
   stop/revert decision itself.
5. **`k1_belief` added as a seventh ablation condition** -- see the battery
   table below.

This redesign has not yet been validated by a full GPU training run as of
this writing; `selftest.py` and `--smoke` pass, but the actual question --
does a Mamba-consistent teacher plus a cleaner replay/stopping design close
enough of V5's gap to clear the Stage 1 gate -- is still open.

## Round 10: train/validate on 5 people only -- fixing test-distribution leakage

Round 8/9's redesign still had all Stage 1/DAgger/Stage 2 data collection,
the Stage 1 gate's nav check, and DAgger's convergence-based revert
decision drawing on `TRAIN_SCENARIOS = tuple(SIX_SCENARIOS.keys())` (all
six scenarios, 5-20 people) and the `heldout_nonstationary` profile --
exactly the scenario/profile combinations `evaluate.py` later reports the
formal zero-shot-generalization result on. Every automated decision that
used those numbers (gate pass/fail, which DAgger-round checkpoint to keep)
had therefore already "seen" performance on the distribution the central
claim is about generalizing to, which invalidates that claim regardless of
what the frozen model's formal numbers turn out to be -- a distinct problem
from ordinary seed reuse, since even fresh seeds on the *same scenario
distribution* still let checkpoint selection optimize for it.

Fixed by hard-separating three scenario/profile sets, enforced by threading
an explicit `scenarios` argument through every in-training nav-eval call
(`runtime.VALIDATION_SCENARIOS`, never a hardcoded default that could
silently drift back to the six-scenario battery):

- `TRAIN_SCENARIOS` / `VALIDATION_SCENARIOS` (`runtime.py`, both
  `("baseline_circle",)`, 5 people) -- all Stage 1a/DAgger/Stage 2 data
  collection, the Stage 1 gate's nav check, DAgger's convergence check, and
  Stage 2's periodic eval. `VALIDATION_PROFILES = ("nominal",
  "train_nonstationary")` -- both already part of what the policy trains
  under (`run_episode` samples between them), so validation shares a
  distribution with training the way an ordinary train/val split does.
- `select_checkpoint.py`'s post-hoc checkpoint selection: same fix, same
  `VALIDATION_SCENARIOS`/`VALIDATION_PROFILES` (it previously validated on
  all six scenarios x `heldout_nonstationary`, the same leak).
- `SIX_SCENARIOS` x `(nominal, heldout_nonstationary)` remains exclusively
  in `evaluate.py`'s one-time formal evaluation of the final frozen model --
  nowhere else in the codebase touches either the 10/12/20-person scenarios
  or the `heldout_nonstationary` profile.

Also split `dagger_validation` into `dagger_validation_recent` (current
round's held-out episodes only, moved into `dagger_validation_historical`
at the start of the next round -- mirrors `recent_dagger_buffer`/
`older_dagger_buffer`) and `dagger_validation_historical` (accumulates
across rounds, logged for reporting only): the convergence check needs "did
*this* round's updates help on states *this* round's policy visits," which
a metric diluted by earlier, weaker rounds cannot answer.

Verified via a real GPU smoke run: `metrics.jsonl`'s `scenario` field is
`baseline_circle` on every single recorded episode across Stage 1a, both
DAgger rounds, and Stage 2, with no crash and `status=completed`.

## Round 10 full training verdict and the Round 11 fix (Q_R-only teacher supervision)

The Round 10 run (`belief_mdp_action_conditioned_beta0.5_round10_seed2407`,
seed 2407) cleared the nav bar easily on **every single check** -- nominal/
stress SR 92-100%, CR 0-7%, across Stage 1b and all 5 DAgger rounds -- but
never passed the gate, because `top1_in_expert_set` (was the network's
*final* decision among the teacher's top-3 actions) stayed flat at ~63-66%
throughout, even trending slightly down over 5 rounds of DAgger. **This
run's checkpoints and logs are preserved, not discarded** -- see
`round10_verdict.md` inside that run's output directory: navigation passed;
the run was blocked by an imitation criterion that, on inspection, was
measuring the wrong thing.

Root cause, two compounding problems:

1. **Teacher supervision was attached to the wrong head.** The margin loss
   was applied to `Q_score = Q_R - beta*Q_C`, not `Q_R`. Demanding the
   *combined* score agree with the teacher creates a direct incentive to
   shrink `Q_C` wherever its risk signal would otherwise move `Q_score`'s
   argmax away from the teacher's pick -- `Q_C` penalized for doing exactly
   what it exists to do. The Mamba teacher only ever demonstrated "how to
   get to the goal" (599/600 of its own demos succeeded, 0 collided); it
   has no business supervising the risk head at all.
2. **The hard top-3 gate could not tell "picked an equally good alternative"
   from "picked something clearly worse."** Nav performance was already
   near ceiling while imitation agreement sat at 64% -- exactly the
   signature of a metric that penalizes reasonable disagreement, not one
   that is catching real failures.

Fixed (Round 11, `train.py`):

1. **Margin loss moved to `Q_R` only** (`optimize()`). `Q_C` is untouched by
   this change -- it still learns only from real simulator collision/near-
   miss cost (`td_loss_c`/`mc_loss_c`), never from the teacher, exactly as
   before.
2. **`model.ReplayBuffer` now stores the full 80-action teacher score
   vector** per transition (`teacher_scores`), not just the top-K indices.
   This powers `normalized_teacher_regret` -- (teacher's best score minus
   its score for the chosen action) / the teacher's own score range for
   that state -- a continuous measure that treats near-ties as near-zero
   regret instead of a binary miss.
3. **The Stage 1 hard gate is now a paired non-inferiority comparison
   against the frozen Mamba-VL baseline** (`evaluate_quick_baseline`, a
   dedicated baseline-policy instance -- never the shared `mamba` object
   `teacher_scores` uses, since `run_mamba_baseline_episode` mutates
   persistent policy state), computed once at start-up on the exact
   seed/scenario/profile formula every gate check reuses:
   `--gate_sr_slack`/`--gate_cr_slack`/`--gate_tr_slack` (default
   0.03/0.02/0.02) replace the old absolute `--stage1_min_sr`/
   `--stage1_max_cr`. Teacher-imitation agreement (`Q_R`-only and
   `Q_score`/final-decision versions, both with regret) is still computed
   and logged in full every gate check and every `stage1_log_every`
   update -- it just never appears in `gate_passed` anymore.
4. **A "risk reduction" diagnostic**: the same validation episodes replayed
   with a counterfactual policy that ignores `Q_C` entirely
   (`evaluate_quick(..., head="task_only")`), reporting the CR delta
   against the actual (`Q_score`-driven) policy -- direct evidence of
   whether the Bayesian branch is preventing collisions the task head alone
   would not have, logged at every gate check, never used to select or gate
   anything.

Not changed: `Q_C`'s training target (already simulator-only), the 5-way
stratified replay, DAgger's convergence-based per-round stopping, and the
Round 10 train/validation scenario isolation.

## Round 12: teacher-label bug found before Round 11's rerun finished (stopped at episode 156)

Independent review of the Round 11 rerun (still in progress, Stage 1a,
episode 156) found a confirmed, serious bug in the teacher itself -- not a
parameter or gate-design issue. The run was stopped and its checkpoints/
logs kept (no gradient updates had happened yet at that point, so nothing
was trained on the corrupted labels, but Stage 1a's already-collected 156
episodes were discarded since they used the buggy teacher).

**1. `teacher_scores()` dropped the current frame.** `runtime.py`'s
sequence construction excluded the just-appended current-step token
(reasoning that each candidate action's own `next_token` made it
redundant) and fed Mamba `[...older history] + [next_token]` -- silently
skipping the current frame `t` and jumping straight to the candidate's
projected `t+1`. Confirmed against `MambaRLPolicy.predict_sarl_style`
(the same production-validated lookahead `test.py` uses, whose own code
comment reads "🔥 修复：先把当前state token加入历史（Mamba需要当前帧）" --
fix: add the current state token to history first, Mamba needs the
current frame) on 60 real states: **36/60 (60%) had a different top-1
teacher action** with the frame included vs excluded. This affected every
run since Round 8 introduced the Mamba teacher (Rounds 8, 9, 10, and the
now-discarded start of Round 11) -- ORCA-era rounds (1-7) are unaffected,
since they never used `teacher_scores()`. Fixed: include the current frame
(`recent_raw = self.history[-self.seq_len:]`, not `self.history[:-1][...]`).

**2. The frozen-Mamba baseline (`evaluate.py`'s `run_mamba_baseline_episode`,
used by `select_checkpoint.py` and Round 11's new gate baseline) fed Mamba
humans in raw simulation order via `environment.joint_state()`, not
TTC-sorted.** The candidate policy always uses TTC-sorted order
(`runtime.sort_humans_by_ttc`, matching `test.py`'s own production path
line-for-line). A 30-episode spot check found no resulting action
difference, but the input-construction code path must match the
candidate's exactly for a paired comparison to be trustworthy -- fixed by
building a TTC-sorted `JointState` directly in `run_mamba_baseline_episode`.

**3. `test_teacher_equivalence.py` (new, GPU-only, not a substitute for
`selftest.py`)**: drives real rollouts and, via a temporary monkeypatch of
`mamba.forward_value` (no changes to either function under test), captures
the exact token tensor `teacher_scores()` and `predict_sarl_style()` each
feed to it. Requires >=100 real states (spanning both the history-still-
filling-up phase and steady state, across 3 scenarios), the input tensors
to match near-exactly, and top-1 action agreement to be 100%. This is the
test that would have caught bug 1 directly, rather than requiring a
by-hand 60-state audit.

**4. `test_margin_loss_on_qr_never_reaches_qc_head` (new, `selftest.py`,
CPU-only)**: the Round 11 report's claim ("Q_R gradient nonzero, Q_C
gradient strictly zero") was verified by hand once; this fixes it
permanently into the regression suite so it cannot silently regress.

**5. `normalized_teacher_regret` was letting the `-1e4` safety-mask
sentinel dominate its normalization range.** Found on 6/60 sampled real
states: a single masked (unsafe) action dragged `teacher_worst` down to
~-1e4, inflating the range and compressing every other action's regret
toward zero regardless of how different their real scores actually were.
Fixed: masked actions are excluded from the min/max range (falling back to
the full range only if literally every action is masked); choosing a
masked action itself now clamps to a regret of exactly 1.0 instead of an
unbounded value that would dominate any batch average.

**6. Documentation correction**: `risk_cost` (`Q_C`'s one-step target) is
currently a pure terminal-collision indicator with **no near-miss/dmin
component** -- an earlier round ("Fixed in round 2" above) added a graded
near-miss term, but it is not present in the current `run_episode`
(`risk_cost = float(result.done and result.outcome == "collision")`).
Comments describing it as "collision/near-miss cost" have been corrected;
the round-2 section above is left as an accurate historical record of what
that round did, not a description of current behavior.

**7. `compute_artifact_hashes` now also hashes `train.py`/`model.py`/
`runtime.py`/`evaluate.py` themselves** (not just configs/GDBN-params/
checkpoint), stored in every checkpoint's `artifact_hashes` and in
`run_manifest.json`. `evaluate.py` now warns (never blocks -- an old
checkpoint must stay evaluable under newer code) if any of these source
files have changed since the checkpoint was trained.

**Confirmed still correct** (independently re-checked, not just re-stated):
the twin-head separation (`Q_R` gradient nonzero / `Q_C` gradient strictly
zero from the margin loss, per item 4's new test), `Q_C` still trained only
on real simulator collision outcome, `Q_score = Q_R - beta*Q_C` still the
final decision rule, teacher-agreement metrics correctly out of the hard
gate, the gate's paired seed/test_case/profile construction, and local-vs-
remote source parity (5 core files' SHA256 matched exactly at review time).
The twin-head belief-MDP architecture itself does not need to change again;
this round fixed teacher-replication correctness and an input-protocol
mismatch, not the decision architecture.

Item 8 of the fix list: run `test_teacher_equivalence.py` and a full
`--smoke` after these fixes, and only start a fresh `round12` output
directory from scratch if both pass -- Round 11's partial data is not
carried forward.

## Ablation battery (`--belief_mode`)

All six conditions now share the exact same network architecture and
input dimensions; only the information reaching the network changes
(`runtime.py`). Each requires the same teacher, same training volume, and
same random seed as `action_conditioned` when its turn comes -- otherwise a
difference between conditions could be an artifact of unequal training, not
anything about the belief source itself.

| Mode | What changes | Tests |
|---|---|---|
| `action_conditioned` | Real fitted K=3 GDBN, action-conditioned risk | Proposed method |
| `cv` | `ConstantVelocityRiskModel` (K=1, padded to the same dims) | Is it Bayesian multi-modality, or just any uncertainty signal? |
| `k1_belief` | Same GDBN code path as `action_conditioned`, but fitted with K=1 (`tools/fit_k1_gdbn.py`, `runs/mamba_vl/gdbn_params_k1`, same `orca_demos_seq.npz` source as the K=3 fit) | Isolates multi-modality itself from "having a GDBN at all" (distinct from `cv`, which also changes the model class) |
| `state_only` | Real GDBN belief, but the risk portion of every candidate scored with the same neutral rollout (kinematics still vary per action) | Is action-conditioning of risk itself doing the work? |
| `corrupted` | Real GDBN belief with pedestrian identity permuted once per episode | Necessity check: performance must drop if belief matters at all |
| `no_belief` | Belief vector and all six risk features zeroed (forces `beta=0`) | The only condition with zero Bayesian information reaching the network |

A `beta=0` run on `action_conditioned` is the seventh condition in the
battery (see the falsifiable-criterion discussion in "Round 8/9" below) --
not a `--belief_mode`, just `--beta 0.0` on the proposed method itself.

`post_hoc_calibration` (the frozen paper's own score-correction interface)
is not re-implemented here — its results already exist in
`runs/eval35_fullcrowd_*` and are the comparison point, not a condition to
rerun.

## Protocol (per project convention: smoke test, then the full protocol directly)

0. `python3 -u belief_mdp/selftest.py` — GPU/CrowdSim-free logic checks
   (margin loss, MC returns, EMA abort, checkpoint eligibility). Seconds to
   run; do this before spending any GPU time.
1. `python3 -u belief_mdp/train.py --smoke --output_dir runs/belief_mdp_smoke`
   — a few episodes, checks the code path end to end (no NaNs, correct
   tensor shapes at `num_humans=20`, checkpoint save/load, correct 80-action
   grid). Not a decision — 12 episodes all timing out, as the first
   corrected smoke run showed, is an expected artifact of near-zero training
   and is not evidence about performance either way.
2. Full training run per condition:
   `python3 -u belief_mdp/train.py --output_dir runs/belief_mdp_<condition> --belief_mode <mode> --beta <value>`
   Stage 1 (imitation) runs first; if it does not clear the hard gate
   (`--stage1_min_top1_in_set`/`--stage1_max_margin_violation`/
   `--stage1_min_sr`/`--stage1_max_cr`), the run stops there with
   `status=stage1_failed` and never spends the 2400-episode RL budget --
   check `[STAGE1-GATE]` in the console log before assuming a run in
   progress is doing anything useful past that point.
3. Full six-scenario x two-profile formal evaluation directly (no
   intermediate statistical gate), writing per-episode records so ablations
   can be compared to each other, not just to the frozen baseline:
   ```
   python3 -u belief_mdp/evaluate.py --checkpoint runs/belief_mdp_<condition>/best_model.pth \
       --seeds 10 --episodes_per_seed 100 \
       --episode_records_output runs/belief_mdp_eval_records.csv \
       --output runs/belief_mdp_<condition>/formal_eval.json
   ```
   Run this once per condition (`action_conditioned`, `cv`, `k1_belief`,
   `state_only`, `corrupted`, `no_belief`, and each `beta` value), pointing
   `--episode_records_output` at the **same** CSV file every time (it
   appends) so every checkpoint's episodes end up in one place, all sharing
   identical `(scenario, profile, seed, test_case)` keys since `--seeds`,
   `--episodes_per_seed`, and `--eval_seed_base` are the same by default.
   Six scenarios (`baseline_circle`, `baseline_square`, `dense_circle`,
   `dense_square`, `large_circle`, `large_square`) x two profiles (`nominal`,
   `heldout_nonstationary`) x 10 seeds x 100 episodes, paired against the
   frozen Mamba-VL baseline on identical seeds/test cases, with a
   100,000-replicate seed-block bootstrap CI on the SR/CR/TR differences.
4. Head-to-head comparison between two conditions (this is what actually
   supports "`action_conditioned` beats `cv`", not just "beats the frozen
   baseline"):
   ```
   python3 -u belief_mdp/compare_checkpoints.py \
       --records runs/belief_mdp_eval_records.csv \
       --checkpoint_a runs/belief_mdp_action_conditioned/best_model.pth \
       --checkpoint_b runs/belief_mdp_cv/best_model.pth
   ```

## Reading the result

This is not gated by a pass/fail rule the way the 5-pedestrian pilot was —
the point of running the full protocol directly is to get the actual effect
size and its confidence interval, not a binary verdict. What would make
this a genuinely new result, not a repeat of five prior "no decisive
advantage" findings:

- `action_conditioned` beats `cv` **and** `k1_belief` **and** `state_only`
  **and** `corrupted` **and** `no_belief` — via `compare_checkpoints.py`'s
  direct head-to-head CI, not just each one's separate CI against the frozen
  baseline — by an amount whose CI excludes zero, in **both** the nominal
  and `heldout_nonstationary` profiles. This is the only outcome that
  supports "action-conditioned Bayesian belief in the decision core
  provides information no simpler alternative provides."
- The `beta` sweep on `action_conditioned` (including `beta=0`, the seventh
  battery condition) shows a non-flat relationship between beta and SR/CR
  (reviewer question: "does more Bayesian weight help" gets an actual
  answer either way) — and this is now a real test, since `Q_C` can no
  longer be canceled by `Q_R`.

If `action_conditioned` does not clearly separate from `cv`, `k1_belief`,
`state_only`, and `no_belief`, that is the same finding this project has
reached five times before, now demonstrated under an architecture whose
specific bugs that could have hidden a real effect have been fixed — and
should be reported as such, not re-litigated with an eighth variant. This
is a pre-registered, falsifiable criterion: if the method cannot beat these
simpler alternatives, the line of research stops here rather than trying a
ninth architecture change in search of a positive result.
