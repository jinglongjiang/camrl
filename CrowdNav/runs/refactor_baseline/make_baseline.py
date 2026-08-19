"""Order 0: freeze the numerical baseline the cleanup must not move.

Deliberately fixed everything an ordinary run varies: the episodes, the
batch, both RNG streams and the model init. Anything that differs after a
refactor is a real behaviour change, not sampling noise.
"""
import hashlib, json
import sys
from pathlib import Path

# runnable from anywhere: the repo root is three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from crowd_nav.bayesian_dvl.intent_config import load_intent_training_config
from crowd_nav.bayesian_dvl.intent_policy import (
    HUMAN_FEATURE_DIM_V6, load_intent_checkpoint, save_intent_checkpoint)
from crowd_nav.bayesian_dvl.intent_runtime_config import ActionGridSpec
from crowd_nav.bayesian_dvl.model import DistributionalValueModel
import crowd_nav.bayesian_dvl.intent_train as T

OUT = Path("runs/refactor_baseline")
OUT.mkdir(parents=True, exist_ok=True)
ENV = Path("crowd_nav/configs/env_bayesian_dvl.config")
cfg = load_intent_training_config()


def h(x):
    if isinstance(x, torch.Tensor):
        return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def grads_hash(gs):
    m = hashlib.sha256()
    for g in gs:
        m.update(b"none" if g is None else g.detach().cpu().numpy().tobytes())
    return m.hexdigest()


tr = []
for sc, sd in (("standard", 2_600_000), ("standard", 2_600_001), ("junction", 96001)):
    tr += T.collect_orca_episode(ENV, sc, sd).transitions
batch = T.batch_to_tensors(tr, device="cpu")

torch.manual_seed(12345)
model = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
base = {"n_transitions": len(tr), "init_param_hash": h(torch.cat([p.detach().flatten() for p in model.parameters()]))}

# --- forward ---
model.eval()
tau = ((torch.arange(16, dtype=torch.float32) + 0.5) / 16).unsqueeze(0).expand(len(tr), 16)
with torch.no_grad():
    q = model(batch.robot_feats, batch.human_feats, batch.human_mask, batch.action_feats, tau)
base["forward_hash"] = h(q)
base["forward_mean"] = float(q.mean())
model.train()

# --- losses + both raw gradients + combined ---
params = [p for p in model.parameters() if p.requires_grad]
gen = torch.Generator().manual_seed(777)
res = T.train_step(model, torch.optim.SGD(params, lr=0.0), batch, gen,
                   rho=cfg.rank_cap_rho, ranking_margin=cfg.ranking_margin,
                   ranking_batch_size=cfg.ranking_batch_size)
base["mc_loss"] = res.mc_loss
base["rank_loss"] = res.rank_loss
base["mc_grad_norm"] = res.mc_grad_norm
base["rank_grad_norm"] = res.rank_grad_norm
base["rank_scale"] = res.rank_scale
base["combined_grad_hash"] = grads_hash([p.grad for p in params])
base["combined_grad_norm"] = res.grad_norm_preclip

# --- ONE real optimizer step ---
torch.manual_seed(12345)
model2 = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
opt = torch.optim.Adam(model2.parameters(), lr=cfg.learning_rate)
gen2 = torch.Generator().manual_seed(777)
T.train_step(model2, opt, batch, gen2, rho=cfg.rank_cap_rho,
             ranking_margin=cfg.ranking_margin, ranking_batch_size=cfg.ranking_batch_size)
base["param_hash_after_one_step"] = h(torch.cat([p.detach().flatten() for p in model2.parameters()]))

# --- checkpoint round trip ---
ck = OUT / "baseline_checkpoint.pth"
grid = ActionGridSpec.from_env_config(str(ENV))
save_intent_checkpoint(model2, str(ck), action_grid_hash=grid.table_hash(),
                       scene_registry_sha256="baseline")
torch.manual_seed(999)
reloaded = DistributionalValueModel(human_feature_dim=HUMAN_FEATURE_DIM_V6)
load_intent_checkpoint(str(ck), reloaded)
reloaded.eval()
with torch.no_grad():
    q2 = reloaded(batch.robot_feats, batch.human_feats, batch.human_mask, batch.action_feats, tau)
base["checkpoint_roundtrip_hash"] = h(q2)

(OUT / "baseline.json").write_text(json.dumps(base, indent=2, sort_keys=True))
for k in sorted(base):
    v = base[k]
    print(f"  {k:34} {v if not isinstance(v, str) else v[:32]}")
