# legacy_kmeans_mode -- frozen baseline, read-only

This directory exists ONLY to pin the hashes of the pre-upgrade mode model
and its evaluation reports (Upgrade U0, 2026-08-03). It contains no code of
its own -- `mode_model.py` and `fit_bayesian_brne.py` stay in place and
remain importable (other tools still reference them as the historical
K-means-then-per-cluster-regression baseline), but from this date forward
they are **not** the main method under development. Do not edit the frozen
files to "fix" the findings below; if a change is needed, make it in the
new `action_conditioned_arhmm.py` module instead.

## Verdict on the frozen method (Order 9 / Order 9S.1-4)

`mode_model.fit_and_select` fits K-means on the instantaneous 10-dim
feature vector `phi`, then fits per-cluster linear dynamics, then derives
`Pi` post-hoc from the hard cluster labels. Five independent checks against
this method's K=6 selection on synthetic 6-behavior-type data:

- **Order 9S.1** (fresh audit seeds, K=1..10): NLL improves monotonically
  through K=10 with no natural plateau; K=9/10 are only rejected by an
  arbitrary `min_mode_fraction` floor, not by loss of signal.
- **Order 9S.2** (ORCA negative control): near-unimodal real ORCA data
  still gets K=3 selected (K=4-10 also show CI-supported "improvement",
  blocked only by the same mode-fraction floor).
- **Order 9S.3** (shuffled robot-human pairing, refit): the K>1 advantage
  survives almost unchanged even when the robot trajectory paired with
  each human is wrong.
- **Order 9S.3b** (frozen model, no refit, feature ablation): self-motion
  features alone recover 86.7% of the improvement; robot-relative features
  alone recover 93.5%; even a wrong (shuffled) robot trajectory recovers
  99.7%. The two feature groups are highly redundant with each other.
- **Order 9S.4** (frozen model, scored -- not refit -- on an independent
  raw-RVO2 generator): the K=6-over-K=1 improvement is *larger* on the
  transfer domain (0.58 nats vs 0.11 nats natively), consistent with K=6
  simply being a higher-capacity piecewise-linear regressor rather than a
  semantically meaningful, transferable interaction-mode discovery.

**Conclusion**: for this frozen method, "action-conditioned Bayesian
interaction-mode discovery" is NO-GO. K behaves like "number of
piecewise-linear regression regions over smooth, continuous kinematic
variation," not a discrete latent intent/interaction-type variable. See
`/home/abc/temp/guide.md`'s 2026-08-03 14:10-15:05 KST entries for the
full audit trail and raw numbers; see `FROZEN_MANIFEST.sha256` in this
directory for the exact file/report hashes this verdict is based on.

## What comes next

The user's explicit decision (2026-08-03) was to upgrade the core model
rather than retreat to a weaker claim: replace this K-means-then-regression
model with an **action-conditioned sticky Bayesian switching autoregressive
model** (`action_conditioned_arhmm.py`), trained with sequence-level
forward-backward EM (not K-means-then-regression), with the robot's
actually-executed action `u_R,t` as an explicit, irreplaceable regression
term (`B_k @ u_R,t`), sticky-prior-regularized mode persistence, and a
proper necessity gate (does action-conditioning beat a same-capacity
self-only model? does shuffling the action destroy that specific gain?)
before any claim of "discovered interaction mode" is made again.
