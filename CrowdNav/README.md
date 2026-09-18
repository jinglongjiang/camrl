# CrowdNav

## Temporal-composition learning study (2026-09-18, completed negative)

**Frozen result: neither new head improves over the original IL reference.**
3600 executions, three seeds, same 50 layouts per cell. Entries below are
success/collision/timeout out of150. These results do not justify a positive
claim about ordered features or Bayesian generalization.

| Scene | Humans | Original IL | Moments + IL/RL | Ordered + IL/RL |
| --- | ---: | --- | --- | --- |
| Circle | 5 | 145/5/0 | 139/11/0 | 134/16/0 |
| Circle | 10 | 129/14/7 | 104/36/10 | 101/44/5 |
| Circle | 12 | 130/14/6 | 95/42/13 | 92/50/8 |
| Circle | 20 | 111/31/8 | 86/47/17 | 92/49/9 |
| Square | 5 | 137/13/0 | 145/5/0 | 136/14/0 |
| Square | 10 | 117/29/4 | 123/25/2 | 126/23/1 |
| Square | 12 | 115/32/3 | 117/31/2 | 103/44/3 |
| Square | 20 | 78/59/13 | 77/68/5 | 66/80/4 |

Unlike the earlier scalar correction, these heads changed behavior, including
an unfavorable increase in high-density collisions. Stronger representation
and hard-example IL were therefore NOT sufficient. The local offset stress
test changed2/200 five-human actions under arbitrary per-person additive cost
offsets; sum and min-centered composition changed0. This is only a structural
sensitivity observation, not proof that offsets caused these navigation errors.
Artifacts: `/home/abc/temp/local_ordered_20260918/`, including full checkpoints,
`frozen_eval.json`, `frozen_eval.log`, and `offset_audit.json`.

Source commit `2802f9d`. Hypothesis: preserving the three smallest per-person
predicted clearances at four time offsets provides a more transferable action
descriptor than compressing each time slice to minimum/mean/standard deviation.
This is a composition hypothesis, NOT an asserted Bayesian advantage or novelty.

Both `moments` and `ordered` have 22917 parameters, the same zero-initialized
18 -> 64 -> 32 -> 1 residual combination head, and identical initial scores
from each seed's existing `prior` IL model. The original per-person costs still
include ALL locally supported humans. Only the new descriptor uses three order
statistics. Horizons are .25/.5/1/2 seconds; clearances are prediction-weighted
means per person, capped to [-.6,4] metres. These are geometric features, NOT
calibrated collision probabilities or an independence-product risk model.

Frozen protocol before opening the new test matrix:

- Seeds 2407/4807/7207; 500 existing five-human teacher episodes per run.
- Local/base networks frozen during 4000 combination-head IL updates, LR3e-4,
  batch64, evaluate each1000 on five-human cases20000--20099.
- Half of each IL minibatch samples initial-model action disagreements with
  teacher demonstrations, half samples all demonstrations. Hard indices are
  fixed before learning and hashed. No student rollouts are teacher-labelled:
  this is offline hard-example imitation, NOT DAgger.
- Initial policy is a separate reference; select among TRAINED IL checkpoints.
  Require >=90/100 development success to enter RL. This intentionally allows
  lower five-human performance than the original reference; that cost must be
  reported, not concealed by claiming identical final IL competence.
- Unfreeze for self-critical RL, no imitation loss, cap10000 candidate steps
  per run, LR3e-5, KL limit.002, temperature.05. Reference steps counted separately.
- Finish all six runs and seal model selection before testing cases
  9600000--9600049 at5/10/12/20 humans in circle/square. Original IL references
  are evaluated on the same layouts; no high-density model selection.
- Primary controlled contrast: ordered versus moments, separately for each
  geometry/count. Original IL is an engineering reference with no extra head
  training budget, not an equal-budget architecture baseline. A pooled-attention
  baseline and a validated Bayesian benefit remain separate requirements.

Remote run directory: `/root/local_ordered_20260918/`. No source files beyond
the existing Bayesian core and training entry point were added. Robot visibility,
human dynamics, action set and reward are unchanged. All six training runs
finished before any of the above evaluations. This experiment is closed;
its test layouts must not be reused for model selection.

### Next bounded experiment: supervise the local event before composing it

In progress, with a fresh run directory and later fresh test layouts. The local
network is trained on an explicit two-second event: would a constant candidate
robot velocity intersect a recorded human trajectory? Labels use exact swept
segments from five-human teacher trajectories. Because robot visibility is
FALSE, changing the hypothetical robot velocity does not alter human motion.
Future observations are TRAINING TARGETS ONLY, never deployed inputs. The event
is defined over the fixed horizon, not over goal-terminated episode returns.

Local event BCE pretraining is followed by ordinary imitation and then pure
policy-gradient RL. The local event model stays frozen during both later stages
to avoid relabelling an arbitrary RL score as a collision probability. Controls
are summed event probabilities, worst-person log-survival cost, and summed
log-survival cost. The last is an independence approximation, not automatically
Bayesian inference or a guaranteed joint collision probability. Five-human
calibration is checked on separate teacher cases25000--25019. No positive result
is asserted until held-out navigation supports it.

## Composition-only minimal experiment (2026-09-18)

**Result: composition interventions change closed-loop behavior, but the new
zero-start conflict correction has NOT demonstrated an improvement.** This is
a one-seed engineering/minimal-learning experiment, not a positive method result
and not evidence against every local-composition architecture.

Only existing `bayesian.py` and `train.py` were changed (source commit
`17f0a0c`). No additional source file, PDF, reward change, human-response model,
Mamba, PPO or DAgger was introduced. Robot visibility remains FALSE. Prediction
was fixed to the `prior` arm to isolate composition from Bayesian estimation.

### Architecture change

The shared local network and original learned max/sum combination are retained.
The optional `conflict` composition adds ONE zero-initialized scalar parameter:

```text
F(a) = max over i<j [ min(c_i(a), c_j(a))
                     * exp(-abs(t_i(a)-t_j(a)) / 1 second)
                     * (1 - dot(u_i,u_j)) / 2 ]
score(a) = original_score(a) - tanh(gain) * F(a)
```

Here c is the existing learned nonnegative local cost, t is the predictive-point
average of closest-approach time (clipped to 0--2 seconds), and u is the current
robot-to-human unit direction. It encodes a narrow hypothesis: opposing local
constraints occurring at similar times may require a combination correction.
It is NOT a calibrated collision probability, a general model of all joint
conflicts, or a new Bayesian inference algorithm. All humans still contribute
to the original score; the correction considers all pairs, O(N^2). Its sign is
learned and its magnitude is bounded by the largest local cost. Single-person
and empty crowds receive zero correction. Parameter count: 19588 -> 19589.

`--composition mixture|sum|max|conflict` is supported, with composition stored
in checkpoints. Old v3 checkpoints default to mixture; explicit
`--initialize-composition` is required to warm-start a different composition.
The two training conditions have exactly identical initial action scores.

### Locked minimal training

- Initialization: prior/2407 IL checkpoint from the preceding study, development
  97 successes / 3 collisions out of 100. SAME weights for both conditions.
- Five-human circle only; no additional IL updates; 5000 candidate RL steps
  per condition, LR 3e-5, temperature .05, KL limit .002, batch size 64.
- Reference rollouts: 5017 extra steps per condition; 23 effective RL updates
  per condition. No imitation loss during RL.
- Development SR at successive evaluations: 97, 96, 95, 96, 96 out of 100,
  for BOTH conditions. Neither improved the selection criterion over IL.
- Consequently BOTH selected checkpoints remain IL (selected RL step 0).
  The frozen selected-policy comparison cannot be presented as trained
  conflict-policy performance; it is the same restored behavior by construction.
- Only after both training runs finished: 50 layouts, IDs 9500000--9500049,
  each at 5/10/12/20 humans in circle/square. These differ from the preceding
  study's test IDs. Remote study runtime: 683 seconds; 800 evaluations.

Selected mixture and conflict scores are identical in every cell:

| Scene | Humans | Success / collision / timeout, per condition |
| --- | ---: | --- |
| Circle | 5 | 48 / 2 / 0 |
| Circle | 10 | 43 / 6 / 1 |
| Circle | 12 | 40 / 9 / 1 |
| Circle | 20 | 39 / 6 / 5 |
| Square | 5 | 47 / 3 / 0 |
| Square | 10 | 34 / 16 / 0 |
| Square | 12 | 33 / 14 / 3 |
| Square | 20 | 23 / 17 / 10 |

Post-training sensitivity diagnosis used the LAST, actually trained weights,
not the restored IL checkpoint. The learned tanh coefficient was 0.000141765.
Across 200 five-human and 196 twenty-human teacher-rollout states, setting that
coefficient back to zero changed ZERO greedy actions. Maximum action-score
corrections were 0.000317 and 0.000359 respectively. The two last-trained models
also selected identical actions in this sample. This limited sample does not
prove global behavioral equivalence, but it explains why this minimal run
provides no evidence of effective conflict handling. More environment steps
alone are not established as a remedy.

### Fixed-weight operator intervention

Separately, the SAME original IL weights were evaluated with mixture/sum/max,
without retraining or selecting an operator using test results. 1200 additional
executions (including a repeated 400-episode mixture control); not independent
replications and NOT independently trained baseline comparisons.

| Scene | Humans | Mixture S/C/T | Sum S/C/T | Max S/C/T |
| --- | ---: | --- | --- | --- |
| Circle | 5 | 48/2/0 | 47/3/0 | 46/4/0 |
| Circle | 10 | 43/6/1 | 41/6/3 | 27/23/0 |
| Circle | 12 | 40/9/1 | 38/7/5 | 26/24/0 |
| Circle | 20 | 39/6/5 | 26/6/18 | 24/26/0 |
| Square | 5 | 47/3/0 | 36/12/2 | 45/5/0 |
| Square | 10 | 34/16/0 | 31/14/5 | 40/8/2 |
| Square | 12 | 33/14/3 | 25/23/2 | 37/11/2 |
| Square | 20 | 23/17/10 | 16/19/15 | 29/20/1 |

Interpretation is limited: operator changes also change score scale relative to
the robot base value, and the weights were trained for mixture. Results show
sensitivity to composition/calibration, NOT that max or sum cannot learn, NOT
that the bottleneck is exclusively composition, and NOT a Bayesian advantage.
An independently trained attention/pooled baseline remains missing. A useful
next architecture must cause meaningful, validated action-ranking changes on
five-human development conflicts before spending on a larger generalization
study; the present scalar correction has not met that condition.

### Artifacts and verification

Local: `/home/abc/temp/local_composition_min_20260918/` contains full run
checkpoints/results, `fixed_weight_probe.json`, `sensitivity.json`, source
snapshot and log. Remote: `/root/local_composition_min_20260918/`.
All selected checkpoint hashes and 800 selected-policy outcome counts were
verified. CPU/CUDA smoke tests passed, including permutation/padding invariance,
zero-start equality, single/empty-crowd neutrality, time-overlap/opposition
identities and finite correction gradient. Server had approximately 699 MiB
free before the run; no unnecessary deletion was performed. The unrelated ECG
GPU process was not changed. Main experiment command:

```bash
python crowd_nav/train.py --mode study --device cuda \
  --checkpoint /root/local_predictive_overnight_20260918/runs/study_v3/prior_2407/il.pt \
  --initialize-composition --study-arms prior --study-seeds 2407 \
  --study-compositions mixture conflict --il-updates 0 --rl-steps 5000 \
  --eval-steps 1000 --lr 0.00003 --eval-episodes 100 --batch-size 64 \
  --case-start 9500000 --study-eval-episodes 50 --out runs/composition_min
```

## Completed overnight result (2026-09-18)

**Verdict: the implementation and bounded IL/RL study completed, but this
Bayesian update did NOT establish a generalization advantage over its fixed
predictive-prior control. Do not promote it as a validated Bayesian method.**

All training and model selection used five-human circle layouts. The robot
remained invisible; human response types, occlusion, Mamba, DAgger, PPO and
GDBN were not used. No high-density training or test-set tuning was performed.
Twelve runs (four arms, three seeds) completed their prescribed training/gates,
followed by 6000 frozen evaluation episodes. Main study wall time: 9643 seconds
(2 h 41 min), excluding preceding implementation and five-human development.

### Frozen results

Each entry is successes / collisions / timeouts over 150 executions:
three training seeds on the SAME 50 layouts, not 150 independent layouts.
The square tests also change geometry, so they are not a pure density intervention.

| Scene | Humans | Bayesian FULL, IL+RL | Fixed prior, IL+RL | FULL, IL only |
| --- | ---: | --- | --- | --- |
| Circle | 5 | 136 / 14 / 0 | 144 / 6 / 0 | 127 / 23 / 0 |
| Circle | 10 | 115 / 33 / 2 | 130 / 18 / 2 | 107 / 42 / 1 |
| Circle | 12 | 119 / 27 / 4 | 121 / 24 / 5 | 109 / 36 / 5 |
| Circle | 20 | 108 / 16 / 26 | 120 / 14 / 16 | 108 / 23 / 19 |
| Square | 5 | 137 / 13 / 0 | 146 / 4 / 0 | 130 / 20 / 0 |
| Square | 10 | 129 / 20 / 1 | 130 / 17 / 3 | 117 / 29 / 4 |
| Square | 12 | 112 / 33 / 5 | 121 / 27 / 2 | 106 / 44 / 0 |
| Square | 20 | 101 / 29 / 20 | 96 / 37 / 17 | 90 / 46 / 14 |

Pooling the two specified geometries with equal weight gives:

| Humans | FULL success | Prior success | FULL collision | Prior collision |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 91.00% | 96.67% | 9.00% | 3.33% |
| 10 | 81.33% | 86.67% | 17.67% | 11.67% |
| 12 | 77.00% | 80.67% | 20.00% | 17.00% |
| 20 | 69.67% | 72.00% | 15.00% | 17.00% |

At 20 humans, FULL has fewer pooled collisions but more timeouts (46 versus
33), and fewer successes (209 versus 216). That is not a general safety and
efficiency improvement. FULL wins only the 20-human square aggregate among
the eight scene/count comparisons. Its square gain is +3.33 percentage points;
an exploratory paired two-way bootstrap over seeds and layouts gives
[-8.67, +15.33] pp. The circle difference at 20 is -8.00 pp, with
[-23.33, +6.00] pp. These 10000-resample intervals (NumPy RNG seed 918,
resampling shared case indices across seeds) are exploratory, unadjusted for
multiple comparisons, and have only three seeds. No equivalence claim follows.

FULL's selected RL policies versus their own IL policies recover 11 successes
in the 20-human square test, but zero in the 20-human circle test. RL is doing
real work, yet its benefit is not consistently a high-density success gain.
We did NOT run a pooled-network control, so this study cannot assign the
generalization level specifically to local factorization. Nor can its results
be directly compared with historical 3--5% scores under different protocols.

### Training gates and actual budgets

| Arm | Seed | Best IL SR /100 | Selected IL update | Candidate RL steps | Reference steps | Effective RL updates | Selected RL step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 2407 | 91 | 4000 | 20000 | 19442 | 95 | 7058 |
| full | 4807 | 91 | 12000 | 20000 | 19554 | 105 | 19022 |
| full | 7207 | 92 | 12000 | 20000 | 19921 | 99 | 11016 |
| prior | 2407 | 97 | 4000 | 20000 | 20278 | 100 | 16020 |
| prior | 4807 | 96 | 6000 | 20000 | 19038 | 86 | 18008 |
| prior | 7207 | 96 | 12000 | 20000 | 18872 | 96 | 17121 |
| map | 2407 | 88 | 4000 | 0 | 0 | 0 | 0 |
| map | 4807 | 90 | 12000 | 20000 | 19672 | 107 | 20000 |
| map | 7207 | 89 | 12000 | 0 | 0 | 0 | 0 |
| history | 2407 | 86 | 6000 | 0 | 0 | 0 | 0 |
| history | 4807 | 90 | 10000 | 1018 | 986 | 5 | 0 |
| history | 7207 | 82 | 12000 | 0 | 0 | 0 | 0 |

All runs received 500 teacher episodes and 12000 IL updates. Total candidate
RL steps: 141018; additional frozen-reference steps: 137763; effective
parameter-changing RL updates: 693. Four runs failed the 90% IL gate and did
not train RL. History/4807 regressed at its first RL evaluation and restored IL.
No gate was lowered. Thus FULL versus prior has matching RL caps and all seeds
qualify, but comparisons to history/map are NOT equal completed-RL comparisons.
Each arm learns IL separately from the same teacher cases and initial weights;
the resulting IL weights and competence are not identical across arms.

For completeness, diagnostic selected-policy counts for the weaker controls:

| Scene | Humans | Posterior-mean point (`map`) | Four-increment history |
| --- | ---: | --- | --- |
| Circle | 5 | 127 / 23 / 0 | 115 / 35 / 0 |
| Circle | 10 | 92 / 56 / 2 | 94 / 54 / 2 |
| Circle | 12 | 97 / 47 / 6 | 98 / 44 / 8 |
| Circle | 20 | 99 / 32 / 19 | 92 / 38 / 20 |
| Square | 5 | 134 / 16 / 0 | 128 / 20 / 2 |
| Square | 10 | 106 / 39 / 5 | 107 / 41 / 2 |
| Square | 12 | 94 / 51 / 5 | 87 / 60 / 3 |
| Square | 20 | 74 / 63 / 13 | 85 / 45 / 20 |

These include gate-failed IL and a rolled-back policy. They cannot establish
that Bayes is better than a successfully trained short-history RL baseline.

### Reproduction and audit

Executed source commit: `ec7b029` (five scoped implementation/report paths).
The source snapshot additionally records the actual ORCA dependency, whose
uncommitted local difference is a docstring change. Unrelated colleague changes
were preserved, not swept into the commit. The final report is committed
separately. GitHub SSH synchronization succeeded on `fix-junction-candidates`.

Remote artifacts: `/root/local_predictive_overnight_20260918/runs/study_v3/`.
Local complete copy, including every run, checkpoint, source snapshot and log:
`/home/abc/temp/local_predictive_overnight_results_20260918/`.
`runs/study_v3/study.json` contains all 6000 per-episode records, all checkpoint
SHA256 hashes, configuration, seeds, selected stages and actual budgets.
All twelve selected checkpoint hashes, 120 evaluation cell counts and paired
case IDs were checked after download. No PDF was created.

```bash
python crowd_nav/train.py --mode study --device cuda \
  --demo-episodes 500 --il-updates 12000 --il-eval-every 2000 --il-lr 0.0003 \
  --rl-steps 20000 --eval-steps 1000 --lr 0.00003 \
  --eval-episodes 100 --batch-size 64 \
  --case-start 9400000 --study-eval-episodes 50 --out runs/new_study
```

System-disk maintenance removed package download cache and 376 MiB of archived
system journals; no datasets, environments, checkpoints or unrelated jobs were
deleted. The existing ECG job was left running. The navigation training job
finished normally. Implementation, training, frozen evaluation and reporting
are complete; new experiments must use new test layouts, not tune against this
matrix. The unresolved research requirement is a reproducible advantage from
online Bayesian updating, not merely a functioning IL/RL pipeline.

## Active v3 local-composition IL/RL (2026-09-18 overnight)

Robot visibility is FALSE, human reciprocity types and occlusion are OFF.
Only five-human circle layouts are used for training and checkpoint selection.
The only new source file remains `crowd_nav/bayesian.py`; existing `train.py`,
configuration and simulator files are used. No PDF, new source directory,
Mamba, DAgger, PPO, GDBN, learned safety critic or deployed planner is added.

### Current algorithm

```text
public motion history -> per-person velocity-innovation posterior
 -> five moment-matched predictive velocities
 -> shared local action-cost network -> posterior-expected local costs
 -> learned mixture of maximum and sum -> base value minus local costs
 -> 80 discrete action preferences

successful ORCA demos -> softmax/MC imitation initialization
 -> self-critical REINFORCE with observed trajectory returns
 -> deterministic argmax deployment
```

The 19,588-parameter network includes CURRENT observed relative velocity as
well as predicted candidate-relative velocity. V2 omitted the former from
the local factor. V3 also increases hidden width to 128. These changes were
developed using five-human data only. The local factorization is an inductive
bias, not an exact Bellman decomposition or a safety guarantee. Expectations
are computed per person before max/sum aggregation, not over a joint crowd
posterior. The diagonal stationary residual model and moment quadrature remain
approximations, not a validated model of hidden intentions or exact tail risk.

IL uses cross entropy at temperature 0.1 plus an executed-action Monte Carlo
value initialization term; RL uses NO imitation loss. During policy-gradient
training, network outputs are action preferences, not calibrated Q values.
The frozen IL policy rolls out the same training case to supply a return-only
baseline. Candidate and reference simulator steps are both reported. A measured
old-policy KL limit and surrogate-improvement backtracking bound each gradient
step; this is first-order REINFORCE, not PPO or an exact TRPO implementation.
No private goals or future trajectories enter deployed inputs. Zero-advantage
batches do not advance Adam or drift parameters.

Controls, with identical network capacity and training budgets:

- `full`: conjugate Normal-Inverse-Gamma predictive mean and variance.
- `prior`: fixed zero-mean innovation model with standard deviation 0.15.
- `map`: posterior-mean point prediction; this legacy CLI name is NOT a mode MAP.
- `history`: frequentist mean/variance of the last four velocity increments,
  using the same five predictive points and the same downstream network.
- `current`: CV point prediction, available but not in the locked four-arm run.

### Completed five-human development

- V2, 200 demos and 3000 margin-IL updates: 40/100 success, 39 collisions.
- Teacher-only safety-space screen on development cases 22000--22049:
  0.10 -> 41 successes/9 collisions; 0.25 -> 46/3; 0.40 -> 39/1.
  Only the ROBOT teacher parameter changed; human dynamics/reward did not.
- Selected teacher 0.25: 466/500 successful training demonstrations.
- V3, 500 demos, 12000 IL updates: best development checkpoint at update 4000,
  91/100 success and 9 collisions. Later update 12000 scored 82/100; it was NOT
  substituted for the better checkpoint.
- Double-DQN negative control: 1000 actual updates -> 49/100 success,
  0 collisions, 51 timeouts. Automatically rolled back; no Q-bootstrap RL
  checkpoint is promoted as the main method.
- Self-critical policy gradient: 5000 candidate environment steps, 5131 reference
  steps, 27 effective updates. Development curve: 90/92/92/91/92 successes out
  of 100. Selected step 3141: 92 successes, 7 collisions, 1 timeout.

These are development results, NOT a claim of Bayesian generalization gains.
The 90% IL gate is not a 5% collision-rate safety certificate.

### Locked overnight study

Remote directory: `/root/local_predictive_overnight_20260918`.
Study output: `runs/study_v3/study.json`, with per-run results/checkpoints and
a source snapshot. Startup PID: 1194807. Final status: COMPLETED.

- Arms: full/prior/map/history. Seeds: 2407/4807/7207.
- Per run: 500 five-human teacher episodes, 12000 IL updates, batch 64,
  IL LR 3e-4; evaluate every 2000 IL updates on cases 20000--20099.
- Qualify at 90/100 before RL; do not lower the gate for failing arms.
- RL cap: 20000 candidate steps, LR 3e-5, temperature 0.05, gamma 0.99,
  four trajectories per update, sampled-policy KL <= 0.002.
- Evaluate about every 1000 RL steps after complete trajectories. A development
  drop below 90% stops that run and restores the best qualified checkpoint.
  Thus caps match, but early-stopped actual budgets need not match; both are logged.
- ALL model selection finishes before any test matrix is opened.
- Frozen tests: cases 9400000--9400049, 5/10/12/20 humans, circle and square.
  FULL also tests its IL-only checkpoint when an RL checkpoint was selected.
- Gate-failed IL policies may be evaluated diagnostically but remain explicitly
  marked IL_GATE_FAILED, not presented as qualified IL+RL competitors.

Engineering checks include analytic conjugate updates, quadrature moments,
history window length, initial full/prior equality, permutation/padding and
distant-human invariance, actual swept collision, timeout timing, invisible
robot trajectory invariance, terminal bootstrap masking and checkpoint reload.
An exact 17-step DQN test recorded 17 updates and evaluations at 10/17. A mocked
regression test verified rollback restores every IL tensor. Policy-gradient
tests checked a nonzero accepted step under the KL bound and no zero-advantage
drift. A miniature two-arm study completed all eight test configurations per arm.

No datasets, checkpoints or unrelated jobs were deleted. Apt download-cache
cleanup freed about 365 MiB; this training process uses a 15% CUDA allocator cap.
Do not modify frozen code/configuration or choose hyperparameters using the
pending 10/12/20-human tests. A pooled-network architecture control is still
needed before attributing gains specifically to local composition.

## Historical v2 smoke (2026-09-18)

This section supersedes the interaction-type design and diagnostics below.
Robot visibility is FALSE. Persistent reciprocity types are OFF. There is no
claim that an invisible robot can infer how humans respond to its motion.

The active entry remains `crowd_nav/train.py`; the only added source file
remains `crowd_nav/bayesian.py`. No Mamba, DAgger, PPO or GDBN is imported.

```text
public consecutive human velocities
 -> per-track Normal-Inverse-Gamma velocity-innovation posterior
 -> five moment-matched predictive velocity points
 -> shared nonlinear local action-cost network, then posterior expectation
 -> learned positive mixture of maximum and summed local costs
 -> base navigation value minus composed local costs
 -> 80 discrete actions

ORCA demonstrations -> margin/MC initialization -> online Double-DQN
```

The method is a hypothesis about transferable local interactions, NOT a proven
factorization of optimal Q or a collision-probability guarantee. The filter
assumes stationary diagonal velocity innovations within an episode. It does
not identify intentions, model human-human dependence, or guarantee calibrated
long-horizon predictions. Five quadrature points match predictive moments but
do not integrate exact Student-t tails. These are explicit limitations.

Arms share the network: `current` uses CV point prediction; `prior` uses the
same fixed innovation scale without updating; `map` is a posterior-mean point
approximation (historical CLI name, NOT a type MAP); `full` integrates updated
predictive spread. All training and validation use five humans only. Larger
crowds are evaluation-only and must not select checkpoints or hyperparameters.

The default IL gate remains 90%. Passing `--il-gate 0` is ONLY a connectivity
smoke and is marked `smoke_only` in results. It does not qualify RL training.
Actual IL and RL optimizer update counts are recorded. A new schema rejects
old interaction-type checkpoints. Executed-motion swept collision checks and
exact terminal timeout handling are enabled independently of reciprocity.

Local and RTX4090 smoke tests pass finite outputs for 0/1/5/10/12/20 people,
permutation and padding invariance, duplicate-frame idempotence, IL/RL gradients,
frozen target-network parameters, and identical human trajectories under two
different robot actions. This is engineering verification, not performance.

Remote: `/root/local_predictive_20260918`; conda environment `mamba`.
Before deployment, apt download caches were cleared: root free space rose
from 207 MiB to 572 MiB. Conda had no unused archive/index cache. No datasets,
checkpoints, installed environments or unrelated processes were removed.
The existing ECG job was left running; this job caps CUDA allocator usage
at 15% (framework/context allocations are additional).

Completed minimal runs: seed 2407, 30 teacher episodes, 600 IL updates, batch 32,
3 RL episodes, 10 five-human development episodes, explicit smoke-only gate.
Each arm collected 24 successful demos / 30 (1162 successful transitions).
Both IL policies scored 3/10 on development, BELOW the default qualification
gate. The deliberately bypassed gate tests connectivity only, not a qualified
IL-to-RL transition. Full performed 166 real RL updates; prior performed 167.
Development after RL: full 8/10 (one collision), prior 6/10. The episode budget
is identical but the step/update budget is not exactly matched.

Frozen final checkpoints, no OOD selection, circle layouts 710000--710009:

| Humans | Full success/collision/timeout | Fixed prior success/collision/timeout |
| --- | --- | --- |
| 5 | 7/2/1 | 4/5/1 |
| 10 | 3/5/2 | 2/7/1 |
| 12 | 4/5/1 | 2/7/1 |
| 20 | 3/4/3 | 4/5/1 |

Conclusion: implementation and nonzero IL/RL optimization work. Navigation
quality and 5-to-20 generalization are NOT qualified. Full does not consistently
outperform the fixed prior. One seed and ten layouts per density cannot establish
a Bayesian contribution, and the IL base was intentionally undertrained here.
Do not tune against these OOD cases. Next performance work must first establish
a qualified five-human IL base, then use equal-step RL and untouched test cases.
An end-to-end set-network control, posterior-mean control, and short-history
control are still needed to separate representation, history, and uncertainty.

Remote artifacts: `runs/full_smoke/`, `runs/prior_smoke/`, and
`runs/paired_smoke_evaluation.json` under `/root/local_predictive_20260918`.
No PDF was generated. No claim of method novelty or publication readiness.

## Historical interaction-belief IL/RL branch (2026-09-17)

The active entry point is `python crowd_nav/train.py`. It no longer imports the
old sequence-policy trainer. The original upstream project description below
is retained as provenance, not as this branch's algorithm specification.

Only one source file is added: `crowd_nav/bayesian.py`. The training entry,
environment configuration and simulator are modified in place. No new planner,
external RL framework, recurrent neural backbone or old belief package is used.

### Architecture

```text
public robot state + public human positions/velocities/radii
    -> two observable-only ORCA response hypotheses per person
    -> cumulative per-person likelihoods
    -> shared population prior + persistent individual type posterior
    -> human tokens + robot state + posterior joint proximity features
    -> shared human encoder + action-conditioned attention and max pooling
    -> shared action-Q scorer -> one discrete velocity action

ORCA successful demonstrations -> fixed-margin IL + Monte Carlo value initialization
    -> independent target network -> Double-DQN with actual online transitions
```

- Actions: the existing 5 speeds x 16 headings grid (80 actions), interpreted in
  the robot-to-goal frame and rotated to world coordinates at execution.
- Human tokens: five normalized physical quantities, four predicted relative
  velocity components (two hypotheses), one posterior probability. Shared MLP:
  10 -> 128 -> 128. No fixed number of human slots and no top-k truncation.
- Robot token: goal distance, velocity (2), radius, preferred speed, remaining
  time, population posterior mean and variance (8 dimensions).
- Each action queries the whole human set. Attention pooling + max pooling,
  robot/action features and two joint soft-proximity features feed a shared
  268 -> 256 -> 128 -> 1 Q scorer. Action values are NOT summed person-wise.
- Joint proximity integrates the shared population variable after multiplying
  conditional individual survival factors. It retains the dependence induced
  by the population posterior; it is a feature, not a collision guarantee.

### Bayesian model and boundaries

`theta ~ Beta(alpha,beta)` is the reciprocal fraction, and each persistent type
`z_i | theta ~ Bernoulli(theta)`. A fixed 21-point midpoint grid approximates
the one-dimensional integral. The configured Beta(7,3) prior has the same mean
0.7 as the training population; the fixed-prior control uses that same mean.
Neither prior is changed when evaluating a different population. Given each person's cumulative likelihood `L_i`,

```text
p(theta | history) proportional to p(theta) product_i [(1-theta)L_i(0)+theta L_i(1)]
p(z_i=1 | theta,history) = theta L_i(1) / [(1-theta)L_i(0)+theta L_i(1)]
```

Every person contributes one cumulative factor, not one pseudo-person per frame.
Equal likelihoods leave the population posterior unchanged. Track IDs are stable
row indices within an episode; reset clears all evidence. Track association,
occlusion and identity switches are not implemented or claimed in this protocol.

The two ORCA predictions include/exclude the observed robot. Human preferred
velocities are inferred from past observed velocity, never from private goals.
The approximate Gaussian velocity likelihood uses configured std=0.2: this is
an initial hyperparameter, NOT an already calibrated or qualified model.
Per-step residual independence and the ORCA behavioral model are approximations.
Human actions are simultaneous with the robot step, so they respond to the
previously observable robot state, not an unexecuted current candidate action.

### Environment and controls

The active config explicitly makes the robot visible and enables persistent
reciprocal types, sampled with a separate seeded RNG. Types never enter public
observations, IL labels or network inputs. This is a changed physical protocol,
not a continuation of the earlier robot-invisible benchmark. Reward coefficients
remain unchanged. Actual executed human velocities are used for swept collision
checks and timeout occurs on the last legal step in this protocol.

Arms, with identical network dimensions:

- `current`: current geometry, CV hypotheses, no history evidence.
- `prior`: the same two ORCA predictions as full, fixed probability 0.7, no
  accumulated individual type evidence or population adaptation. Preferred
  velocity estimation still uses the same observable history as full.
- `fixed`: cumulative individual evidence with fixed population fraction.
- `map`: hard individual MAP decisions, with the population statistics retained.
- `full`: hierarchical posterior probabilities and joint integration.
- `oracle`: evaluation-only true individual type intervention. The filter never
  receives labels, training/IL collection reject this arm, and all input fields
  except conditional type probability stay equal to `prior`. Feeding these
  inputs to an existing non-oracle-trained network is NOT a performance upper bound.

`current` differs in both evidence and prediction model; it alone cannot attribute
gains to Bayesian integration. `prior` versus `fixed` isolates individual evidence;
`fixed` versus `full` isolates population adaptation, and `map`
tests individual posterior compression. These are implemented controls, not
positive experimental findings. Short-history and calibrated alternative controls
are still needed before a paper-level claim.

### Training and checks

Training and model selection reject human counts other than five. Evaluation can
load the same checkpoint at 10/12/20 humans; type-fraction shifts are separately
controlled through `--reciprocal-probability` in evaluation only.

IL uses successful teacher episodes only. Failed teacher transitions may enter
reward replay but never become imitation labels. IL fits a fixed action margin
plus executed-action Monte Carlo returns. RL stops using IL loss, learns from
actual executed actions with Huber Double-DQN targets, and treats episode timeout
as terminal because remaining time is part of the state. IL validation must reach
the configured gate (default 90%) before RL starts. Checkpoints reject incompatible
schema/arm and preserve the action table and environment configuration.

```bash
# No artifacts written; CPU is sufficient.
PYTHONDONTWRITEBYTECODE=1 python crowd_nav/train.py --self-test

# Explicit training command; NOT launched automatically during architecture work.
python crowd_nav/train.py --mode train --arm full --out runs/interaction_full

# Freeze first; evaluate crowd size without updating weights.
python crowd_nav/train.py --mode evaluate --arm full \
  --checkpoint runs/interaction_full/final.pt --humans 20 --eval-episodes 100
```

Architecture validation completed: 28 network/filter checks plus runtime checks
for real IL and TD updates, frozen targets, terminal bootstrap masking, identical
expert actions across input arms, deterministic replay, in-memory
checkpoint round trip, private-type isolation, neighbor selection, exact timeout
and swept collision. The same model executed short rollouts with 5/10/12/20 humans.
This is a connectivity result, NOT trained navigation performance, calibrated
type inference, density generalization evidence or a demonstrated new algorithm.
No formal training or OOD-based tuning has been performed on this new architecture.

### Identification preflight (2026-09-17)

Run `python crowd_nav/train.py --mode diagnose --diagnostic-episodes 12`.
This writes no files and performs no optimization. Four disjoint cohorts of 12
ORCA-controlled episodes: five-human fitting 610000--610011, five-human validation
620000--620011, ten-human audit 630000--630011, twenty-human audit 640000--640011.
All are circle scenarios with the same configured reciprocal fraction (0.7).
Counts are 48 physical layouts, not 480 independent episodes. People within an
episode are dependent. The run was repeated to verify the reported numbers.

Velocity-likelihood std candidates were 0.1/0.2/0.4/0.8/1.6. Selection used only
five-human fitting endpoint log loss and selected **0.8**. This is a diagnostic
candidate; the active config remains 0.2, not silently promoted to a qualified
model. Validation and larger crowds were not used to choose the value.

| Held-out crowd | Individuals | Prior Brier | Fixed AUC / Brier | Full AUC / Brier |
| --- | --- | --- | --- | --- |
| 5 | 60 | 0.19667 | 0.54403 / 0.18609 | 0.51989 / 0.18568 |
| 10 | 120 | 0.19000 | 0.63926 / 0.18308 | 0.51759 / 0.18446 |
| 20 | 240 | 0.21333 | 0.75586 / 0.20332 | 0.71927 / 0.20333 |

Full-minus-prior Brier differences with exploratory episode-block bootstrap
95% intervals (10,000 resamples): 5 humans -0.01098 [-0.02977,0.00375];
10 humans -0.00554 [-0.01432,0.00278]; 20 humans -0.01000 [-0.01490,-0.00517].
These are small-sample exploratory results without multiplicity adjustment.

The twenty-human likelihood carries some type information; the five-human
validation does not establish reliable discrimination. Hierarchical adaptation
does not beat fixed-prior individual inference here. This is not proof that the
RL network cannot learn, nor proof of navigation improvement.

Only 12/240 twenty-human endpoint predictions lie outside the 0.6--0.8 bin.
For full the three occupied bins have (count, mean prediction, observed rate):
(4,0.5794,0), (228,0.7147,0.6930), (8,0.8514,1). Thus AUC alone would overstate
how decisive the inferred types are. At 2/4/8 seconds twenty-human AUC is
0.4850/0.5684/0.6611: end-of-episode information may arrive too late for decisions.
Fixed-time metrics omit episodes already ended; endpoint metrics include all.

At the original std=0.2, full endpoint AUC is 0.5455/0.5802/0.7369 and log loss
0.7000/0.6760/0.5681 (5/10/20 respectively). The original prior log losses are
0.5826/0.5685/0.6179. Raising std reduces damaging certainty in sparse validation,
but does not fix the underlying preferred-velocity likelihood mismatch.

Sensitivity uses the first observed near-human state (<2m, after 8 steps), not
future outcomes, and flips one person's probability 0 -> 1 while keeping the
other fields fixed. On 36 states, three random network initializations change
their argmax in 3/36, 14/36 and 2/36 cases; maximum absolute Q changes are
0.00614/0.00670/0.01023. This confirms a live information path, NOT correct
decision use. Random-network sensitivity is not a training gate. To inspect an
IL checkpoint use the same diagnostic with `--checkpoint`; no IL checkpoint has
been trained or evaluated for this new architecture yet.

Do not add a fixed risk penalty merely to force action changes. No long training,
oracle performance claim, population-shift claim or final likelihood qualification
is supported by this preflight. The next engineering target is likelihood quality
at decision time, not a claim that all Bayesian/RL approaches have failed.

**[`Website`](https://www.epfl.ch/labs/vita/research/planning/crowd-robot-interaction/) | [`Paper`](https://arxiv.org/abs/1809.08835) | [`Video`](https://youtu.be/0sNVtQ9eqjA)**

This repository contains the codes for our ICRA 2019 paper. For more details, please refer to the paper
[Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning](https://arxiv.org/abs/1809.08835).

Please find our more recent work in the following links 
- [Relational Graph Learning for Crowd Navigation, IROS, 2020](https://github.com/ChanganVR/RelationalGraphLearning).
- [Social NCE: Contrastive Learning of Socially-aware Motion Representations, ICCV, 2021](https://github.com/vita-epfl/social-nce).

## Abstract
Mobility in an effective and socially-compliant manner is an essential yet challenging task for robots operating in crowded spaces.
Recent works have shown the power of deep reinforcement learning techniques to learn socially cooperative policies.
However, their cooperation ability deteriorates as the crowd grows since they typically relax the problem as a one-way Human-Robot interaction problem.
In this work, we want to go beyond first-order Human-Robot interaction and more explicitly model Crowd-Robot Interaction (CRI).
We propose to (i) rethink pairwise interactions with a self-attention mechanism, and
(ii) jointly model Human-Robot as well as Human-Human interactions in the deep reinforcement learning framework.
Our model captures the Human-Human interactions occurring in dense crowds that indirectly affects the robot's anticipation capability.
Our proposed attentive pooling mechanism learns the collective importance of neighboring humans with respect to their future states.
Various experiments demonstrate that our model can anticipate human dynamics and navigate in crowds with time efficiency,
outperforming state-of-the-art methods.


## Method Overview
<img src="https://i.imgur.com/YOPHXD1.png" width="1000" />

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy sarl
```
2. Test policies with 500 test cases.
```
python test.py --policy orca --phase test
python test.py --policy sarl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
5. Plot training curve.
```
python utils/plot.py data/output/output.log
```


## Simulation Videos
CADRL             | LSTM-RL
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/vrWsxPM.gif" width="400" />|<img src="https://i.imgur.com/6gjT0nG.gif" width="400" />
SARL             |  OM-SARL
<img src="https://i.imgur.com/rUtAGVP.gif" width="400" />|<img src="https://i.imgur.com/UXhcvZL.gif" width="400" />


## Learning Curve
Learning curve comparison between different methods in an invisible setting.

<img src="https://i.imgur.com/l5UC3qa.png" width="600" />

## Citation
If you find the codes or paper useful for your research, please cite our paper:
```bibtex
@inproceedings{chen2019crowd,
  title={Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning},
  author={Chen, Changan and Liu, Yuejiang and Kreiss, Sven and Alahi, Alexandre},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={6015--6022},
  year={2019},
  organization={IEEE}
}
```
