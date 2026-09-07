"""Read-only copies of the two leakage-safe BDVL modules D-0 needs.

The old ``bayesian_dvl`` package is sealed: the Intent-BDVL line ended with
the fourth-layer experiments recorded in the project report. Rather than
importing from a sealed package (whose copy inside the paper-1 tree is a
different, older revision anyway), the goal-intent tracker and the public
scene-candidate provider are vendored here verbatim.

Nothing is edited except the module paths in the import lines. The leakage
contract is preserved exactly:
    a pedestrian's gx/gy is never passed in; the bank never accepts a Human;
    candidate_fn sees only a stable track_id and a public position.
"""
