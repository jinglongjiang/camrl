"""Artifact-bound pretraining gate, not a navigation performance claim."""
import json
from pathlib import Path

from crowd_nav.belief_mdp.hashing import sha256_file, sha256_dir

FEATURE_CONTRACT = "density5_mask_nonreactive_adjacent_v1"


def teacher_fingerprint(args):
    root = Path(__file__).resolve().parents[1]
    sources = [root / "belief_mdp" / name for name in
               ("runtime.py", "test_teacher_equivalence.py", "protocol.py")]
    sources += [root / "gdbn.py", root / "policy/mamba_rl.py",
                root / "belief_space_rl/runtime.py", root / "contracts.py",
                root.parent / "crowd_sim/envs/crowd_sim.py",
                root.parent / "crowd_sim/envs/policy/orca.py"]
    result = {str(p.relative_to(root.parent)): sha256_file(str(p)) for p in sources}
    for name in ("policy_config", "base_env_config", "env_config", "base_checkpoint"):
        result[name] = sha256_file(str(getattr(args, name)))
    result["gdbn_params"] = sha256_dir(str(args.gdbn_params))
    return result


def validate_teacher_receipt(path, args):
    receipt = json.loads(Path(path).read_text())
    if (receipt.get("passed") is not True or receipt.get("states", 0) < 100
            or receipt.get("tensor_mismatches") != 0
            or receipt.get("value_mismatches") != 0
            or receipt.get("score_mismatches") != 0
            or receipt.get("top1_mismatches") != 0
            or receipt.get("fingerprint") != teacher_fingerprint(args)):
        raise ValueError("Teacher gate missing, failed, or stale: rerun real teacher equivalence")
    return receipt
