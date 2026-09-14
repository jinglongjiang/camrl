# Continuous Bayesian Set RL: Repair Status

Current entry is `train_smoke.py`. It owns teacher qualification, collection,
five-human model refitting, BC-only evaluation, supervised critic warm-up and
the gated online entry (`--online`). Do not create separate entry scripts for
each stage. Only `teacher.py` and `algorithm.py` are added functional modules.
The default stage now stops after BC evaluation, even on success. The frozen
teacher qualification receipt is mandatory; the teacher is not re-tuned.

Current changes relative to dcc4fe5:

- Existing unicycle CEM controller reused in teacher.py; projected ORCA and the
  simplified constant-turn teacher are no longer used. Seven core classes are
  copied unchanged from bayes_occ_mpc (source hashes in teacher.py).
  Adapter configuration: 512 candidates, four iterations, 16 steps, v<=1 m/s,
  omega<=1.2 rad/s, a<=2 m/s^2, 0.50 m development-calibrated human buffer.
- Fixed scaling (position 10 m, velocity 2 m/s, radius 0.3 m), seven ego inputs,
  robot-conditioned attention plus max pooling, no heading or explicit count.
- Balanced uniform mode prior and deterministic per-episode filter RNG.
- Five-human-only GNG/GDBN refit, world coordinates in discovery and dynamics;
  no robot-action coupling. This fixes a coordinate contract, not calibration.
- Corrected terminal order is opt-in for this environment, preserving legacy
  experiment semantics. Timeout is now at step 140; post-terminal step rejects.
- Collision qualification uses the union of native collision reports and swept
  overlap of actually executed robot/human segments. Native checks use previous
  human velocity, so an additional check is necessary; no native collision is
  forgiven. The copied planner's `first_physical_clearance` diagnostic includes
  its configured margin and must not be mistaken for measured body clearance.
- Explicit TD3 reward/BC/collision-cost objectives. No optimizer hooks in the
  active entry. Cost critic is a soft penalty, not certified safe control.
- BC only uses successful teacher episodes; all failures remain in raw data
  and critic data. BC-only runs 100 distinct five-human development episodes.
- Reward critic warm-up uses complete teacher episodes and a 20-episode holdout.
  Cost warm-up requires separate safety trajectories, with at least 20 collision
  and 20 non-collision episodes in each split. Missing positives fail closed.
  Sigmoid cost heads represent undiscounted collision reachability, using the
  maximum of both heads for bootstrapping and actor penalties. Actor parameters
  must remain bitwise unchanged during warm-up. Class-enriched validation is
  not evidence of calibration under the deployment state distribution.
- TD3 explicitly uses learning_starts=0, normalized Gaussian action noise 0.1,
  one step per update, policy_delay=2, tau=0.005, gamma=0.99. It is NOT started
  by the BC-only experiment. All checkpoints and receipts record the arm;
  mismatched online arms are rejected.
- Actor RL requires teacher and BC success >=90%, collision <=2%, 100 episodes,
  1000 warm-up updates and a passed held-out critic check. These are development
  thresholds, not statistical safety certificates. Long RL is not auto-started.

```bash
python -m crowd_nav.bayes_continuous.tests
python -m crowd_nav.bayes_continuous.train_smoke --params repair_results/params --arm no_belief --teacher-receipt repair_results/teacher_qualification/teacher_summary.json --out repair_results/new_development
# Qualification-only audit, same fixed 100 development layouts:
python -m crowd_nav.bayes_continuous.train_smoke --params repair_results/params --out repair_results/new_teacher_audit --teacher-only
# Fixed qualification reproduction, not new independent data:
python -m crowd_nav.bayes_continuous.train_smoke --params repair_results/params --out repair_results/reproduce_teacher_confirmation --teacher-only --evaluation-episodes 200 --case-offset 700001
# Refuses failed stage receipts:
python -m crowd_nav.bayes_continuous.train_smoke --online --arm no_belief --stages repair_results/new_development --out repair_results/new_online --steps 10000
```

Layout generation uses test_case, not the filter/layout_seed. Current collection
uses cases 20000..20199, BC validation 30000..30099. Teacher development uses
0..99; a fresh qualification run must pass an unused `--case-offset`. Results
store initial physical-layout hashes. Earlier dcc4fe5/960ddda claims of layout
independence based only on different layout_seed offsets are withdrawn.
`--evaluation-episodes` defaults to 100; smaller probes cannot qualify a teacher.

Nonstationary training probability defaults to zero (nominal acquisition).
The online CLI can explicitly schedule a fixed nonstationary mixture while
keeping five humans. No 10/20-human layout is used for current model selection.
Full observations remain an explicit scope limitation, not occlusion sensing.
No-Belief masks every belief input before BC and evaluation, and does not refit
GDBN. The old filter fixture runs only behind that masked interface, not as an
input to the student. MAP/FULL refit on the same five-human training cases.
Old checkpoints with nine ego inputs are incompatible; they are not hot-loaded.

## Historical dcc4fe5 Smoke (Not the Current Training Protocol)

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
