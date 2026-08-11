"""Complete source manifest for the goal-intent (V6) main chain.

Why this exists (pre-C5 requirement): the whole ``crowd_nav/bayesian_dvl/``
package and ``configs/train_intent_bdvl.config`` were Git-UNTRACKED. Before
anything is rsync'd to the 4090 there has to be one artifact that answers
"exactly which files, at exactly which contents, constitute this chain" --
otherwise a remote run cannot be tied back to a local state, and a missing
file only surfaces as a confusing failure mid-training.

It also captures a real dependency that is easy to miss: two files OUTSIDE
this package are READ AT TEST TIME --
``configs/policy_bayesian_fullcrowd_tail.config`` (the Test5 six-scenario
protocol the paper table must match) and ``tools/evaluate_bdvl_paper_main.py``
(the Test8 episode-seed formula). Both were untracked, so a fresh clone or a
partial rsync would fail those equivalence tests. They are listed here as
first-class dependencies, not incidental extras.

Run:  python3 -m crowd_nav.bayesian_dvl.source_manifest [--out PATH] [--check PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List


class SourceManifestError(ValueError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = "bdvl_intent_source_manifest_v1"

# The goal-intent MAIN CHAIN: every module the V6 training/eval/deploy path
# actually imports. Kept explicit (not a glob) so an accidental new file
# cannot silently join the chain without being declared.
MAIN_CHAIN_SOURCES: List[str] = [
    "crowd_nav/bayesian_dvl/__init__.py",
    "crowd_nav/bayesian_dvl/intent_tracker.py",
    "crowd_nav/bayesian_dvl/scene_candidates.py",
    "crowd_nav/bayesian_dvl/intent_policy.py",
    "crowd_nav/bayesian_dvl/junction_scenario.py",
    "crowd_nav/bayesian_dvl/intent_train.py",
    "crowd_nav/bayesian_dvl/intent_config.py",
    "crowd_nav/bayesian_dvl/intent_train_cli.py",
    "crowd_nav/bayesian_dvl/intent_evaluate.py",
    "crowd_nav/bayesian_dvl/intent_crowdnav_policy.py",
    "crowd_nav/bayesian_dvl/geometry_features.py",
    "crowd_nav/bayesian_dvl/set_encoder.py",
    "crowd_nav/bayesian_dvl/iqn.py",
    "crowd_nav/bayesian_dvl/model.py",
    "crowd_nav/bayesian_dvl/ranking.py",
    "crowd_nav/bayesian_dvl/normalization.py",
    "crowd_nav/bayesian_dvl/contracts.py",
    "crowd_nav/bayesian_dvl/intent_runtime_config.py",
    "crowd_nav/bayesian_dvl/statistics.py",
    "crowd_nav/bayesian_dvl/evaluate.py",
    "crowd_nav/bayesian_dvl/source_manifest.py",
]

# Configs read at runtime.
CONFIGS: List[str] = [
    "crowd_nav/configs/train_intent_bdvl.config",
    "crowd_nav/configs/env_bayesian_dvl.config",
]

# Files OUTSIDE the package that the chain or its tests depend on. All of
# these were untracked; a partial sync silently breaks the paper-protocol
# equivalence tests.
# Determined EMPIRICALLY by tracing every crowd_nav/crowd_sim module
# imported and every config opened while running representative
# env-building tests -- not by guessing. The first version of this list
# was incomplete and a remote acceptance run surfaced it as 36 failures
# (orca.py's unset time_step, env.config's missing [policy] section, and
# three untracked tools modules), so the closure is now declared in full.
EXTERNAL_DEPENDENCIES: List[str] = [
    # protocol definitions the tests read
    "crowd_nav/configs/policy_bayesian_fullcrowd_tail.config",  # Test5 [eval_envs]
    "crowd_nav/tools/evaluate_bdvl_paper_main.py",              # Test8 seed formula
    "crowd_nav/configs/env.config",                             # [policy] -> ActionGridSpec
    # CrowdNav integration
    "crowd_nav/__init__.py",
    "crowd_nav/policy/policy_factory.py",                       # registers IntentBDVLPolicy
    # the simulator package, in full -- every module in the traced closure
    "crowd_sim/__init__.py",
    "crowd_sim/envs/__init__.py",
    "crowd_sim/envs/crowd_sim.py",
    "crowd_sim/envs/policy/__init__.py",
    "crowd_sim/envs/policy/linear.py",
    "crowd_sim/envs/policy/orca.py",                            # time_step default; rvo2 ctor
    "crowd_sim/envs/policy/policy.py",
    "crowd_sim/envs/policy/policy_factory.py",
    "crowd_sim/envs/utils/__init__.py",
    "crowd_sim/envs/utils/action.py",
    "crowd_sim/envs/utils/agent.py",
    "crowd_sim/envs/utils/human.py",
    "crowd_sim/envs/utils/info.py",
    "crowd_sim/envs/utils/robot.py",                            # expects_joint_state dispatch
    "crowd_sim/envs/utils/state.py",
    "crowd_sim/envs/utils/utils.py",
]

TESTS: List[str] = [
    "crowd_nav/bayesian_dvl/selftest.py",
    "crowd_nav/bayesian_dvl/tests/__init__.py",
    "crowd_nav/bayesian_dvl/tests/_common.py",
    "crowd_nav/bayesian_dvl/tests/test_intent_tracker.py",
    "crowd_nav/bayesian_dvl/tests/test_scene_candidates.py",
    "crowd_nav/bayesian_dvl/tests/test_intent_policy.py",
    "crowd_nav/bayesian_dvl/tests/test_intent_training.py",
    "crowd_nav/bayesian_dvl/tests/test_intent_cli.py",
    "crowd_nav/bayesian_dvl/tests/test_intent_integration.py",
]

# B6: the manifest now covers the V6 RUNTIME CLOSURE ONLY.
#
# Two groups were removed rather than trimmed:
#
#   * ``legacy_chain`` -- belief/rollout/world_model/trainer/... are no
#     longer on this branch at all, so hashing them could only fail closed.
#   * ``registry_hashed_sources`` -- it derived from
#     ``config.BDVL_PRODUCTION_SOURCES``, and B4 removed the V6 chain's
#     dependency on ``config.py`` entirely (the six symbols V6 actually
#     needs now live in ``intent_runtime_config.py``). Keeping the group
#     would reintroduce the exact import the shrink deleted -- which is
#     how this test failed: ModuleNotFoundError on a module that is
#     correctly gone.
#
# The remaining groups are verified against the closure computed by
# actually importing the V6 entry points (test_c5_source_manifest_...),
# so a file that the chain really loads cannot be silently omitted.
GROUPS = {
    "main_chain": MAIN_CHAIN_SOURCES,
    "configs": CONFIGS,
    "external_dependencies": EXTERNAL_DEPENDENCIES,
    "tests": TESTS,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(repo_root: Path = REPO_ROOT) -> Dict[str, object]:
    """Hash every declared file. FAILS CLOSED on a missing one -- a manifest
    that silently skips absent files is worse than no manifest."""
    repo_root = Path(repo_root)
    groups: Dict[str, Dict[str, Dict[str, object]]] = {}
    missing: List[str] = []
    seen: set = set()
    for group, rels in GROUPS.items():
        entries = {}
        for rel in rels:
            if rel in seen:
                continue  # already declared by an earlier group
            seen.add(rel)
            p = repo_root / rel
            if not p.exists():
                missing.append(rel)
                continue
            entries[rel] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
        groups[group] = entries
    if missing:
        raise SourceManifestError(f"declared source files are missing: {missing}")

    # one hash over the whole set, order-independent
    blob = json.dumps(
        {g: {r: e["sha256"] for r, e in sorted(v.items())} for g, v in sorted(groups.items())},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")

    from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
    from crowd_nav.bayesian_dvl.intent_train_cli import code_sha256, scene_registry_sha256
    from crowd_nav.bayesian_dvl.intent_runtime_config import ActionGridSpec

    cfg = load_intent_training_config(repo_root / "crowd_nav" / "configs" / "train_intent_bdvl.config")
    grid = ActionGridSpec.from_env_config(str(repo_root / "crowd_nav" / "configs" / "env_bayesian_dvl.config"))
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "manifest_sha256": hashlib.sha256(blob).hexdigest(),
        "n_files": sum(len(v) for v in groups.values()),
        "groups": groups,
        "code_hash": code_sha256(),
        "config_content_hash": cfg.content_hash(),
        "config_sha256": cfg.source_sha256,
        "action_grid_hash": grid.table_hash(),
        "scene_registry_hash": scene_registry_sha256(cfg),
        "feature_schema": cfg.feature_schema,
        "training_contract_schema": cfg.training_contract_schema,
        "checkpoint_schema": cfg.checkpoint_schema,
        "python": platform.python_version(),
    }


def check_manifest(manifest: Dict[str, object], repo_root: Path = REPO_ROOT) -> List[str]:
    """Return a list of human-readable differences ([] == the tree matches)."""
    repo_root = Path(repo_root)
    problems: List[str] = []
    for group, entries in manifest["groups"].items():
        for rel, meta in entries.items():
            p = repo_root / rel
            if not p.exists():
                problems.append(f"MISSING {rel}")
            elif _sha256(p) != meta["sha256"]:
                problems.append(f"CHANGED {rel}")
    fresh = build_manifest(repo_root)
    for key in ("code_hash", "config_content_hash", "action_grid_hash", "scene_registry_hash"):
        if fresh[key] != manifest.get(key):
            problems.append(f"{key}: {manifest.get(key)} -> {fresh[key]}")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build or verify the BDVL goal-intent source manifest.")
    p.add_argument("--out", type=Path, default=None, help="write the manifest JSON here")
    p.add_argument("--check", type=Path, default=None, help="verify the working tree against this manifest")
    args = p.parse_args(argv)

    if args.check:
        manifest = json.loads(Path(args.check).read_text())
        problems = check_manifest(manifest)
        if problems:
            print("MANIFEST MISMATCH:")
            for x in problems:
                print(f"  {x}")
            return 2
        print(f"manifest OK: {manifest['n_files']} files match, manifest_sha256 {manifest['manifest_sha256'][:16]}")
        return 0

    manifest = build_manifest()
    text = json.dumps(manifest, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    print(f"files             : {manifest['n_files']}")
    print(f"manifest_sha256   : {manifest['manifest_sha256']}")
    print(f"code_hash         : {manifest['code_hash']}")
    print(f"config_content    : {manifest['config_content_hash']}")
    print(f"action_grid_hash  : {manifest['action_grid_hash']}")
    print(f"scene_registry    : {manifest['scene_registry_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
