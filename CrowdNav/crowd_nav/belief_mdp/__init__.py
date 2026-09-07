"""Belief-MDP: belief-conditioned constrained navigation policy.

Bayesian posteriors over full-crowd behaviour are part of the decision state
from the first training step. Mamba is a frozen history encoder only; it
refines the belief representation but cannot feed the value/cost heads
through any other path. See README.md for the full design and the
pre-registered go/no-go protocol.
"""
