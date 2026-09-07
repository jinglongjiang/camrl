# Belief-Space RL Pilot

This isolated pilot tests the Bayesian-first architecture only after the
prediction gate has passed.

Decision structure:

1. a calibrated action-conditioned GDBN maintains the primary belief state;
2. Bayesian per-action caution-zone probabilities drive the Q towers;
3. the verified Mamba checkpoint is frozen and supplies only a bounded,
   belief-gated history-context residual;
4. the frozen Mamba value-lookahead ranking is used only as a training teacher;
5. low-risk states are anchored to that teacher, while calibrated high-risk
   states may prefer a safer near-progress action;
6. actions preceding a simulator collision receive explicit backward credit;
7. no pretrained Mamba Q/value output enters deployed action selection;
8. conservative Double-DQN refines the distilled policy.

The pilot remains restricted to five pedestrians. Training alternates nominal
behavior with a `train_risk` protocol at event rate 0.06. Evaluation uses fixed
paired nominal cases and a preregistered `decision_stress` protocol at event
rate 0.08. Every checkpoint faces the same cases as the frozen baseline.
Only a checkpoint satisfying the full SR/CR gate is additionally saved as
`best_eligible_model.pth`. This is a go/no-go test, not a paper result.

Smoke test:

```bash
python3 -u belief_space_rl/train.py \
  --smoke \
  --gdbn_params runs/bayesian_belief_pilot_smoke/gate/selected_gdbn \
  --output_dir runs/belief_space_rl_smoke
```

Controlled run:

```bash
python3 -u belief_space_rl/train.py \
  --output_dir runs/belief_space_rl_teacher_guided_final_chance_20260730
```

The default run collects 300 frozen-teacher episodes, performs 3,000 offline
updates, and then runs 2,400 conservative online episodes. It writes
`run_manifest.json`, `metrics.jsonl`, `best_model.pth`, `final_model.pth`, and
conditionally `best_eligible_model.pth`. Do not expand to the formal
six-scenario protocol unless the eligible checkpoint exists and passes an
independent paired reevaluation.
