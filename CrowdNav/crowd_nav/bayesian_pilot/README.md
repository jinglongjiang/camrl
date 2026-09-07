# Bayesian Belief Pilot

This directory is isolated from the frozen full-crowd calibration paper.

The first gate asks whether a CV-anchored GDBN provides decision-relevant
predictive information when pedestrians stop, slow, or turn outside the
training parameter ranges. Training, model selection, nominal testing, and
held-out nonstationary testing use disjoint episodes.

Run a local smoke test:

```bash
python3 -u bayesian_pilot/run_prediction_gate.py --smoke
```

Run the preregistered full prediction gate:

```bash
python3 -u bayesian_pilot/run_prediction_gate.py
```

The next RL stage is prohibited unless every check in
`runs/bayesian_belief_pilot/gate/prediction_gate.json` passes. Prediction
density is assessed with NLL. Decision-relevant proximity calibration uses
Brier score and AUC for a fixed 0.30 m surface-clearance caution zone. FDE is
reported but is not a gate because the mean of a valid multi-modal predictive
density need not be a better point forecast than CV.
