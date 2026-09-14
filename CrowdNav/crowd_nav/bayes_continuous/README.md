# Continuous Bayesian Set RL prototype

Independent execution path: observed human states -> repaired GDBN mode posterior
-> shared human MLP -> masked mean/max pooling -> continuous TD3 actor/critics.
No Mamba imports, weights, teacher, sequence context, or enumerated action grid.
Legacy code remains in the repository for historical reproduction only.

The actor outputs physical `(v, omega)` in `[0, 1] x [-1.2, 1.2]`.
SB3 internally scales actions to `[-1, 1]`. The custom set encoder handles padded
human tensors; invalid entries do not participate in pooling. Capacity is 20,
not an unlimited-cardinality implementation. Training and model selection are
restricted to baseline_circle (5 humans). Evaluation supports 5/10/20 humans.

## Training

ORCA planar velocities are projected to the same bounded unicycle executor used
by the student. BC pretrains the actor with normalized-action MSE. SB3 TD3 then
provides twin critics, target updates, exploration, and delayed actor updates.
An optimizer pre-hook adds a decaying demo BC gradient to actor updates.
This is online TD3 with BC regularization, not the original offline TD3+BC method.
See https://stable-baselines3.readthedocs.io/en/master/modules/td3.html .

The smoke uses four shared demonstration episodes (three train, one held out),
80 BC updates and 256 environment steps per arm. Demonstrations are also loaded
into replay. The fixed demo dataset remains available for BC; ordinary replay
may evict demonstrations in longer runs. No long-run training claim is made.

Arms share geometry, capacity, initialization seed and demonstrations:

- no_belief: mode probabilities and entropy zeroed, geometry retained.
- map: one-hot most probable mode, entropy zeroed.
- full: all three posterior probabilities and entropy retained.

There is no oracle-mode arm: fitted GDBN modes do not have simulator truth labels.
There is no explicit collision-probability feature or hard risk shield in this
actor. FULL denotes its mode inputs, not exact future trajectory integration.

## Run

From CrowdNav, in an environment with torch, numpy, scipy and local RVO2:

```bash
python -m pip install -r crowd_nav/bayes_continuous/requirements.txt
python -m crowd_nav.bayes_continuous.tests
python -m crowd_nav.bayes_continuous.train_smoke --params repair_results/params --out repair_results/new_smoke
python -m crowd_nav.bayes_continuous.audit_saved --params repair_results/params --results repair_results/new_smoke
```

The runner requires CUDA and limits its allocator to 20% of GPU memory.
Existing result directories are rejected, not overwritten. Remote dependencies
were installed into an isolated target directory, preserving other GPU jobs.
Exact versions, module paths, source and checkpoint hashes accompany the result.

## Scope and limitations

This is a FULL-OBSERVATION architecture smoke using the legacy full-state training
wrapper, not an occlusion experiment or a legal-detection tracker evaluation.
Simulator human positions/velocities and associated identities are available.
The GDBN parameters are reused frozen fixtures, not newly calibrated on this
protocol. Their data provenance must be resolved before any independent test.

Humans do not respond to the robot (`robot.visible=false`); B_action is zeroed.
Native CrowdNav executes rotation followed by straight translation over each
0.25-second step, not exact constant-twist arc integration. ORCA projection does
not preserve holonomic ORCA's collision-avoidance properties. Neither acceleration
limits nor TurtleBot hardware dynamics are newly implemented or certified.
`bound_violations` in these new results counts ACTION bounds only, not workspace
boundary crossings. Timeout is a terminal finite-horizon failure; remaining time
is included in observations. Original reward coefficients are retained.

The included smoke is one seed and three evaluation layouts per arm. These
layouts are now development diagnostics, not an untouched future confirmation
set. It validates execution, not convergence, generalization, or Bayesian value.
