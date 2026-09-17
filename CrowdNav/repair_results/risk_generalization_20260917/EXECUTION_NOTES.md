# Execution notes

## Frozen scope

Four arms (physical baseline, CV risk, EWMA risk, Bayesian predictive risk),
three seeds, five-human training and development only. The primary question is
whether this specific risk-attention mechanism improves frozen larger-crowd
deployment, not whether every possible Bayesian architecture can work.

The protocol was frozen before the first completed training result. The branch
is `bayes-risk-generalization-20260917`. Implementation and audit revisions are
`3985cb3` and `30b5fe8`.

## Preflight

- Three `RiskGeneralizationContracts` unit tests passed (0.741 seconds).
- After the case-range correction, all four tests in that class passed again
  (0.792 seconds), including a physical-layout aliasing regression test.
- Nine existing regression tests also passed (0.597 seconds), covering the
  default set encoder, cached teacher, prior/seed, termination, executor, action
  history, Bayesian moment propagation and Gaussian PPO semantics.
- Checked 5/10/12/20 valid entities, padding isolation, permutation invariance,
  deterministic episode risk reset, and one expert query per recorded step.
- SB3 environment checks passed for all four arms.
- Before training, transferred deterministic means matched the source actor
  within 1e-6 on the checked observations. Risk gains start at zero.
- All methods use isolated SB3 2.3.2 on local Python 3.8, rather than the
  historical SB3 2.7.1 training runtime. The source weights are copied directly;
  no old critic or optimizer is transferred. No cross-version training
  trajectory equivalence is claimed.
- The existing SciPy/NumPy installation emits an ABI warning. Finite-value and
  deterministic risk checks passed. This is recorded, not silently dismissed as
  a verified supported dependency combination.

## Interrupted attempt and correction of the initial diagnosis

The first attempt was interrupted during BC after noticing two calls to
`expert_action()` at a teacher-controlled step. That was initially described as
a potential label/execution mismatch. Inspection of `BeliefEnv.expert_action()`
then established that it caches the action until the environment advances.
Therefore the suspected contamination did NOT occur. The interruption was
unnecessary. The new collector directly reuses the executed teacher action,
and a regression test checks that stronger explicit contract.

All 24 corresponding ten-episode collection batches were compared after the
restart: observations, four risk channels, labels and episode records were
identical in every batch. The interrupted run is kept locally in
`risk_generalization_20260917_interrupted_cached_query`; it is not used to pick
models or pool results.

The merged dataset order is pool-completion order, so the two complete tensors
are not byte-identical despite identical episode content. This can alter SGD
sampling. All four arms and three seeds in the reported attempt share the SAME
saved merged dataset. Its exact artifact and hash must be retained to reproduce
this training attempt; recreating episodes alone need not reproduce its order.

## Teacher collection

240 attempted episodes: 234 successes, 5 collisions, 1 timeout.
Circle/square x nominal/train_nonstationary each has 60 episodes, always five
humans. Only successful teacher trajectories enter BC: 10,322 transitions.

## Physical layout aliasing: actual issue, corrected before final analysis

The first complete bounded run failed the IL gate, but its subsequent audit
also found 240 shared physical layouts between teacher collection and DAgger.
Different `layout_seed` values do not suffice: CrowdSim reseeds physical layout
generation from `test_case`. Development layouts did not overlap training.
The discarded run is kept in `risk_generalization_20260917_case_overlap`.

The corrected run reuses ALL twelve exact BC checkpoints and the exact original
merged teacher dataset, but repeats the DAgger stage with case numbers
625000--625359 instead of 620000--620359. Teacher cases remain 620000--620239;
development remains 630000--630099. This is a sampling-contract repair, not a
change to losses, budgets, gate thresholds or a selection of better checkpoints.
Both old and corrected attempts must be disclosed. `restart_receipt.json`
records the reused checkpoint and dataset hashes. Final training results must
come only from the corrected attempt.

The corrected collection has 360 unique DAgger layouts, with zero overlap
against the 240 teacher layouts and the 100 development layouts. It contains
311 successes, 43 collisions and 6 timeouts, yielding 16,134 labeled visited
states. All of these student states are retained; combined IL data size is
26,456 transitions. This count supersedes the discarded attempt's count.

## Interpretation limits

### Separate terminal IL diagnostic

The corrected original experiment also failed its all-runs qualification gate:
CV seed 7207 and EWMA seed 4807 each scored 88/100 nominal. The Bayesian arm
and the physical baseline passed their own gates for all three seeds. PPO was
NOT started. This distinction matters: the cohort gate failure is not evidence
that the Bayesian actor itself failed to learn five-human navigation.

To answer the user's actual generalization question rather than stop at that
gate, a separate protocol was committed as `a5a70a7` before completing the new
deployment evaluation. It evaluates all twelve frozen DAgger policies, with no
checkpoint selection or further training, on the originally planned six scenes
and two profiles (50 layouts per cell, 7,200 episodes). It is stored under
`il_diagnostic/` and must be called a supplemental IL-only diagnostic, not a
passed original experiment or an IL+PPO result. The original failed gate and
its analysis are retained unchanged. Original audit assertions about the absence
of high-density testing refer only to that original protocol's output folder,
not to this explicitly separate supplemental evaluation.

- The Bayesian channel uses the existing fitted GDBN's predictive mean and
  isotropized covariance, not a newly validated likelihood or a full mixture
  collision integral.
- Risk is the maximum marginal disc-overlap probability over seven probe arcs
  and eight time steps. It is not a calibrated probability of any collision.
- All arms retain the same mask, physical inputs, architecture and data.
- A masked set encoder is permutation invariant, not guaranteed invariant to
  adding pedestrians. Neither these risk features nor their attention pooling
  are proved sufficient statistics or a guarantee of crowd-size generalization.
- Five-human square training is an explicit expansion of training geometry,
  not training on ten- or twenty-human cases.
- A failed five-human qualification gate makes the generalization question
  inconclusive. It does not prove that Bayesian generalization is impossible.
- No PDF is generated; the FCS1 manuscript is not edited.

## Final checks

- All 7,200 deployment records are present, covering 300 physical layouts;
  episode action-bound violation counts sum to zero.
- Four sampled episodes were replayed across different arms, seeds, crowd sizes
  and profiles. Their entire saved episode dictionaries matched exactly.
- All 50 files in `il_artifacts.tar.gz` matched their on-disk SHA-256 digests.
- The evaluation auditor's code digest matches the digest frozen in the
  supplemental protocol. The original training-code hashes also still match.
- FCS2 source copies match; static checks found 12 unique labels, four resolved
  references, ten cited bibliography keys and balanced environments. No PDF
  compilation was performed.
- FCS1 was not edited by this work. Its hash changed externally near closeout;
  that concurrent change was neither overwritten nor reverted.
