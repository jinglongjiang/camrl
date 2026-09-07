"""Order 14C: prove the materialized cache is still valid, then re-stamp it.

materialization_code_sha256() hashes intent_policy.py, and Order 14C edited
that file -- so the cache identity no longer matches and a run would spend
6.7 hours re-materializing 256177 rows.

But the edit is confined to symbol names (CHECKPOINT_SCHEMA_V8 -> V9,
TRAINING_CONTRACT_V9 -> V10) and one default value that is numerically
unchanged (n_taus 32 -> ACTION_QUANTILES, which IS 32). Nothing in
build_intent_human_feature_batch moved. That is an argument, not evidence,
so this script MEASURES it: it re-materializes a sample of episodes with the
new code and compares every array of every row bit-for-bit against what the
cache holds for the same episode seeds.

Only if every field of every sampled row is byte-identical does it rewrite
the cache -- transitions untouched, identity stamp refreshed. Any mismatch
and it exits non-zero having written nothing, because then the cache really
is stale and the 6.7 hours are owed.

Usage: order14_cache_restamp.py [--apply]
"""
import sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, ".")
from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
from crowd_nav.bayesian_dvl.intent_runtime_config import ActionGridSpec
from crowd_nav.bayesian_dvl.intent_train import materialize_arm_transitions
from crowd_nav.bayesian_dvl.intent_train_cli import materialized_cache_identity

APPLY = "--apply" in sys.argv
R = Path("runs/v3_domain_randomized")
CACHE, RAW = R / "materialized_full.pth", R / "corpus/il_corpus_raw_24ffe4448151.pth"
PER_SCENARIO = 6
FIELDS = ("robot_features", "human_features", "human_mask", "action_features",
          "all_action_features", "action_index", "expert_action_indices",
          "remaining_fraction", "mc_return", "source_role")

cfg = load_intent_training_config("crowd_nav/configs/train_intent_bdvl.config")
at = np.asarray(ActionGridSpec.from_env_config("crowd_nav/configs/env_bayesian_dvl.config")
                .build_action_table(), dtype=np.float64)

cached = torch.load(str(CACHE), map_location="cpu", weights_only=False)
tx, ep_row = cached["transitions"], cached["episode_of_row"]
print(f"cache rows={len(tx)}")
old_id = cached["identity"]
import hashlib
new_id = materialized_cache_identity(cfg, "full", hashlib.sha256(RAW.read_bytes()).hexdigest())
drift = sorted(k for k in new_id if old_id.get(k) != new_id[k])
print(f"identity drift on: {drift or 'NOTHING (cache would already HIT)'}")
if not drift:
    print("nothing to do"); sys.exit(0)
if drift != ["materialization_code_sha256"]:
    print(f"REFUSING: drift is not confined to the code hash -> {drift}"); sys.exit(2)

payload = torch.load(str(RAW), map_location="cpu", weights_only=False)
eps = payload["episodes"]
by_sc = {}
for e in eps:
    by_sc.setdefault(str(e.scenario), []).append(e)
start = {}
for i, ep in enumerate(ep_row):
    start.setdefault(int(ep), i)

sample = []
for sc in sorted(by_sc):
    lst = by_sc[sc]
    for j in np.linspace(0, len(lst) - 1, PER_SCENARIO).astype(int):
        sample.append(lst[int(j)])
print(f"re-materializing {len(sample)} episodes ({PER_SCENARIO}/scenario, spread over the corpus)")

rows_checked, bad = 0, []
for e in sample:
    fresh = materialize_arm_transitions(e, "full", at, horizon=cfg.future_horizon,
                                        n_samples=cfg.future_n_samples)
    s = start.get(int(e.episode_seed))
    if s is None:
        bad.append((e.scenario, e.episode_seed, "episode not in cache")); continue
    old = tx[s:s + len(fresh)]
    if len(old) != len(fresh):
        bad.append((e.scenario, e.episode_seed, f"row count {len(old)} != {len(fresh)}")); continue
    for k, (a, b) in enumerate(zip(old, fresh)):
        for f in FIELDS:
            va, vb = getattr(a, f, None), getattr(b, f, None)
            same = (np.array_equal(np.asarray(va), np.asarray(vb))
                    if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray) else va == vb)
            if not same:
                bad.append((e.scenario, e.episode_seed, f"row {k} field {f}"))
        rows_checked += 1
    print(f"  {e.scenario:<16} seed={e.episode_seed:<9} rows={len(fresh):>4} "
          f"{'MISMATCH' if bad else 'identical'}", flush=True)

print(f"\nrows compared={rows_checked}  fields/row={len(FIELDS)}  mismatches={len(bad)}")
if bad:
    for x in bad[:10]:
        print("  ", x)
    print("\nCACHE IS STALE -- re-materialization is genuinely owed. Nothing written.")
    sys.exit(3)

print("every sampled row is byte-identical under the new code.")
if not APPLY:
    print("dry run; pass --apply to re-stamp"); sys.exit(0)
cached["identity"] = new_id
tmp = Path(str(CACHE) + ".tmp")
torch.save(cached, str(tmp)); tmp.replace(CACHE)
print(f"re-stamped {CACHE}: materialization_code_sha256 "
      f"{old_id['materialization_code_sha256'][:12]} -> {new_id['materialization_code_sha256'][:12]}")
