# L-corner exact POMDP gate

This is a falsification test, not the proposed paper system.  It uses at most
one side-corridor pedestrian and an episode-level existence prior.  The learned
Mamba policy and CrowdSim are intentionally absent.

Run Phase 2A tests:

```bash
python -m unittest crowd_nav.gate_lcorner.test_gate -v
```

Run the independent red-team tests (raw observation-history enumeration rather
than the production solver recursion):

```bash
python -m unittest crowd_nav.gate_lcorner.test_redteam -v
python -m crowd_nav.gate_lcorner.audit_gate bruteforce --output /tmp/bruteforce.json
```

Run the preregistered Phase 2B cell:

```bash
python -m crowd_nav.gate_lcorner.run_gate --stage single
```

The 135-cell Phase 2C sweep is refused until the single-cell JSON records a
passing oracle-minus-best-fixed gap.
