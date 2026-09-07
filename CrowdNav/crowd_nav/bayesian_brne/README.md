# SM-BRNE: Switching-Mode Bayesian Recursive Nash Equilibrium

## What this is, and what it is not

SM-BRNE is a from-scratch, Bayesian-belief-native crowd navigation method.
It is **not**:

- `Mamba score - beta * Bayesian risk` (that is `belief_mdp/`, a separate,
  untouched module).
- A veto/shield/calibration/residual bolted onto the end of another policy.
- GDBN's multiple modes averaged into one mean trajectory before planning.
- BRNE used as a teacher to train a Q-network.
- A modification of `belief_mdp/`, `belief_space_rl/`, or
  `bayesian_decision_gate/` -- those directories are frozen and unrelated to
  this package.

The question this package exists to answer:

> Can an online switching-mode Bayesian posterior be turned *completely*
> into each pedestrian's mixed-strategy trajectory prior, and recursively
> coupled with the robot's own trajectory distribution inside BRNE, to get
> safer, explainable, 20-person-scalable navigation when behavioral
> ambiguity and genuine robot-pedestrian interaction are present?

## Four structural invariants (never violate these)

1. Each pedestrian's Bayesian posterior `b_i(z)` must determine the
   distribution its future trajectory samples are drawn from.
2. Multiple modes must survive into BRNE as distinct sample clusters --
   **never** collapsed into one weighted-mean trajectory before planning.
   `x_mean = sum(prob[k] * prediction[k] for k in modes)` is banned from the
   main method; it exists only as the `posterior_mean_brne` ablation, whose
   entire purpose is to demonstrate this collapse is worse.
3. BRNE's recursive weight update must depend jointly on the robot's and
   every pedestrian's trajectory distribution; the executed action comes
   from the equilibrium robot mixed strategy, not a separately-computed
   score.
4. No Mamba/SARL/LSTM weights are loaded at deployment time. They are
   baselines only.

The test for "is Bayesian actually load-bearing" is **not** "can you set its
weight to zero and still get an answer" (every additive module passes that
trivially). The real test: **does the posterior define the pedestrian
mixed-strategy prior that the equilibrium is solved over, or is it a
separable score added outside the equilibrium?** Only the former counts.

## The mean-collapse trap (why invariant 2 exists)

If a pedestrian is 50% likely to turn left and 50% likely to turn right, the
*probability-weighted mean* of those two trajectories passes straight through
the middle -- a path the pedestrian will almost certainly never take. Feeding
BRNE that single mean trajectory plus ordinary unimodal GP noise reproduces
exactly the constant-velocity-like prior BRNE's own paper defaults to; it is
not a use of the posterior's actual multimodality. `trajectory_sampler.py`'s
`selftest` includes a synthetic two-mode fixture that visually and
numerically proves `full_posterior` forms two separate trajectory clusters
while `posterior_mean` collapses to one, and that BRNE's equilibrium
treats the two inputs differently -- this is both a regression test and the
paper's core motivating figure.

## Status

As of 2026-08-04, the dense end-to-end engineering path is implemented and
accepted through R5: Bayesian filtering, action-conditioned pedestrian
sampling, recursive BRNE equilibrium, entity-clearance cost, CrowdSim
integration, paired evaluation, and auditable diagnostics. The regression
suite has 96 passing tests and the pinned upstream BRNE equivalence suite has
15 passing checks.

This is not yet a paper-ready method. Only an `engineering_only` artifact
exists; the strict independent necessity gate (S1), navigation ablation pilot
(S2), six-scenario evaluation (S3), and Gazebo integration (S4) have not been
completed. The 4090 F7 benchmark is numerically stable but not real-time
(approximately 4.27 s/step median with 5 pedestrians and 19.68 s/step with
20 pedestrians), so sparse/accelerated solving remains conditional on S1
passing. `/home/abc/temp/guide.md` is the authoritative order and status
document.

## Upstream dependency

Core equilibrium math is adapted from BRNE (Sun, Baldini, Hughes, Trautman,
Murphey -- *Mixed strategy Nash equilibrium for crowd navigation*, IJRR 2024,
GPLv3, https://github.com/MurpheyLab/brne). Pinned commit:
`633a5cdcb39ab27f18b596cb8cb1968644f82391`. This package does not vendor the
upstream wrapper's CV pedestrian prior, fixed GP covariance, nearest-N
truncation, corridor-specific bounds, closest-pedestrian safety mask, or ROS
control caching as part of the proposed method -- those are upstream-demo
conveniences, not part of the equilibrium algorithm itself, and are
reimplemented explicitly (and shared fairly across every BRNE-family
ablation) where this project actually needs them.
