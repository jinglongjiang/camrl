"""SM-BRNE: Switching-Mode Bayesian Recursive Nash Equilibrium.

A Bayesian-belief-native crowd navigation policy: each pedestrian's full
discrete-mode posterior is turned directly into a multi-modal trajectory
prior, which is fed into BRNE's recursive equilibrium solver alongside the
robot's own sampled trajectories. Mamba-VL/SARL/LSTM-RL/DSRNN are baselines
only -- this package never loads their weights to produce an action.

See README.md for the full architecture and the four structural invariants
this package must never violate (no posterior-mean collapse before BRNE, no
"nearest-5" truncation presented as full-crowd, no risk-head/veto bolted onto
another policy's action, no closed-form BRNE without upstream equivalence
verified first).
"""
