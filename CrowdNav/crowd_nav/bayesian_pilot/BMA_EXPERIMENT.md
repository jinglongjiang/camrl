# Online Bayesian Model-Average Navigation Experiment

## Question

Does online model evidence improve full-crowd risk correction by selecting
between a constant-velocity predictor and the fixed paper K=3 GDBN when
pedestrian motion becomes nonstationary?

## Frozen design

- Navigation backbone: the unchanged Mamba-VL T=24 value-lookahead checkpoint.
- Candidate action space: the existing 80 actions.
- Risk correction: the existing bounded continuous value correction.
- Full-crowd input: identity-stable observations of up to 20 pedestrians.
- GDBN: `runs/bayesian_distributional/gdbn_params_cv_residual_k3`.
- CV covariance: fitted once on the disjoint pilot training split.
- Model prior: 0.5/0.5.
- Online evidence: one-step GDBN mixture likelihood versus CV Gaussian
  likelihood, centered and robustly scaled using nominal training transitions.
- Model weights: maintained per pedestrian and used to mix per-pedestrian
  action risks before the existing max-over-crowd aggregation.
- No navigation outcome, collision label, or test split is used to update the
  online model weights.

## Conditions

1. Frozen Mamba-VL base policy.
2. Full-crowd CV risk correction.
3. Full-crowd fixed K=3 GDBN risk correction.
4. Fixed 50/50 CV-GDBN risk mixture.
5. Online Bayesian CV-GDBN model average.

All conditions use the same environment cases, evaluation seeds, intervention
seeds, action grid, time limit, and Mamba weights.

## Protocols

Primary:

- held-out nonstationary pedestrian profile;
- baseline circle with 5 pedestrians and dense square with 20 pedestrians;
- 10 seeds, 100 episodes per scenario and seed.

Noninferiority:

- nominal pedestrian profile;
- the same two scenarios;
- 3 seeds, 100 episodes per scenario and seed.

## Interpretation

The Bayesian-necessity claim requires the online model average to improve the
nonstationary safety-efficiency tradeoff over both fixed GDBN and CV, while not
materially degrading nominal success. A result that only matches fixed GDBN,
or improves over the base but not over CV, does not establish that online
Bayesian model selection is necessary.

The fixed 50/50 condition separates online evidence from the generic benefit
of ensembling two predictors.
