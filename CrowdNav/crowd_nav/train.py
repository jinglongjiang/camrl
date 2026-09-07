

from __future__ import annotations

import os, sys, time, argparse, logging, random, configparser, math, re, shutil
import glob
import json
import subprocess
from typing import Optional, Tuple, Dict, List

# Running this file directly makes Python prefer a sibling CrowdNav checkout on
# this machine. Pin imports to the repository that owns this entry point.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# These variables must be set before importing torch or creating a CUDA
# context; setting them later only prints deterministic-mode warnings.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF',
    'max_split_size_mb:64,expandable_segments:False',
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as tb_err:
    SummaryWriter = None
    print(f"[TENSORBOARD] Disabled due to import failure: {tb_err}")

_SUCCESS_TOKENS = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
_COLLISION_TOKENS = {"collision"}
_TIMEOUT_TOKENS = {"timeout"}

def _done_for_bootstrap(event_str: str, bootstrap_on_timeout: bool = False) -> bool:
    """Return done flag used for bootstrap masking.

    - success/collision -> True
    - timeout -> True unless bootstrap_on_timeout=True
    """
    if not event_str:
        return True
    ev = str(event_str).lower()
    is_timeout = 'timeout' in ev
    if is_timeout and bootstrap_on_timeout:
        return False
    return True

def _event_token(info):

    try:
        if info is None:
            return ""
        if isinstance(info, dict):
            val = info.get("event") or info.get("Event") or info.get("status") or info.get("done_event")
            if val is not None:
                return str(val).replace(" ", "_").lower().strip()
        for attr in ("event", "name", "value"):
            if hasattr(info, attr):
                try:
                    return str(getattr(info, attr)).replace(" ", "_").lower().strip()
                except Exception:
                    pass
        s = str(info)
        if "." in s:
            s = s.split(".")[-1]
        return s.replace(" ", "_").lower().strip()
    except Exception:
        return ""

def _info_get(info, key, default=None):
    if isinstance(info, dict):
        return info.get(key, default)
    if hasattr(info, key):
        try:
            return getattr(info, key)
        except Exception:
            return default
    return default

_TMP_DIR = '/root/tmp'
try:
    os.makedirs(_TMP_DIR, exist_ok=True)
except Exception:
    _TMP_DIR = os.getcwd()
for _tmp_env in ('TMPDIR', 'TEMP', 'TMP'):
    os.environ.setdefault(_tmp_env, _TMP_DIR)
import tempfile
tempfile.tempdir = _TMP_DIR
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.ppo_buffer import ReplayBufferIQL
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.log_formatter import get_log_formatter

from crowd_nav.contracts import (
    tokens_to_34d, joint34_to_tokens, simulate_next_frames,
    pick_last
)

from crowd_nav.utils.metrics import (
    TrainingStats, Plotter, MetricsPlotter,
    log_periodic_metrics, log_final_metrics
)

def linear_schedule(start: float, end: float, total_steps: int, current_step: int) -> float:

    if current_step >= total_steps:
        return end
    return start + (end - start) * (current_step / total_steps)

# il_ratio_schedule已删除 - PPO不混合IL数据

class TrainConfig:

    def __init__(self, cfg: configparser.RawConfigParser, outdir: str):

        self.seed = int(cfg.getfloat('train', 'seed', fallback=42))
        self.gamma = cfg.getfloat('train', 'gamma', fallback=0.99)
        self.learning_rate = cfg.getfloat('train', 'learning_rate', fallback=1e-4)
        self.batch_size = int(cfg.getfloat('train', 'batch_size', fallback=96))
        self.updates_per_ep = int(cfg.getfloat('train', 'updates_per_ep', fallback=12))

        self.clip_grad_norm = cfg.getfloat('train', 'clip_grad_norm', fallback=1.0)

        self.n_step = cfg.getint('train', 'n_step', fallback=1)
        self.bootstrap_on_timeout = cfg.getboolean('train', 'bootstrap_on_timeout', fallback=False)

        self.il_ckpt = cfg.get('train', 'il_ckpt', fallback=os.path.join(outdir, 'il_policy.pth'))
        self.il_epochs = int(cfg.getfloat('train', 'il_epochs', fallback=120))
        self.il_batch_size = int(cfg.getfloat('train', 'il_batch_size', fallback=256))
        # [FIX] Default False
        force_retrain_raw = str(cfg.get('train', 'il_force_retrain', fallback='false')).strip().lower()
        self.il_force_retrain = force_retrain_raw in ('true', '1', 'yes')

        self.teacher_neighbor_dist = cfg.getfloat('imitation_learning', 'teacher_neighbor_dist', fallback=6.0)
        self.teacher_max_neighbors = cfg.getint('imitation_learning', 'teacher_max_neighbors', fallback=15)
        self.teacher_time_horizon = cfg.getfloat('imitation_learning', 'teacher_time_horizon', fallback=4.5)
        self.teacher_time_horizon_obst = cfg.getfloat('imitation_learning', 'teacher_time_horizon_obst', fallback=4.5)
        self.teacher_safety_space = cfg.getfloat('imitation_learning', 'teacher_safety_space', fallback=0.10)
        self.success_target = int(cfg.getfloat('imitation_learning', 'success_target', fallback=2000))
        self.max_prefill_episodes = int(cfg.getfloat('imitation_learning', 'max_prefill_episodes', fallback=12000))
        self.prefill_batch_episodes = int(cfg.getfloat('imitation_learning', 'prefill_batch_episodes', fallback=64))
        self.prefill_patience_batches = int(cfg.getfloat('imitation_learning', 'prefill_patience_batches', fallback=10))
        self.max_il_prefill = int(cfg.getfloat('imitation_learning', 'max_il_prefill', fallback=500))

        self.warmup_episodes = int(cfg.getfloat('buffer', 'warmup_episodes', fallback=50))

        # Fix: Read train_episodes from [train] (SSOT), then [trainer] (legacy)
        self.episodes = int(cfg.getfloat('train', 'train_episodes',
                                         fallback=cfg.getfloat('trainer', 'episodes', fallback=3000)))
        self.save_every = int(cfg.getfloat('train', 'save_every',
                                           fallback=cfg.getfloat('trainer', 'save_every', fallback=200)))
        self.roll_window = int(cfg.getfloat('train', 'roll_window', fallback=50))
        self.plot_every = int(cfg.getfloat('train', 'plot_every', fallback=25))

        self.reward_profile = cfg.get('train', 'reward_profile', fallback='benchmark')

class EnvConfig:

    def __init__(self, cfg: configparser.RawConfigParser):

        self.time_step = cfg.getfloat('env', 'time_step', fallback=0.25)
        self.time_limit = cfg.getint('env', 'time_limit', fallback=25)

        self.human_num = cfg.getint('sim', 'human_num', fallback=5)
        self.circle_radius = cfg.getfloat('sim', 'circle_radius', fallback=4.0)

        self.robot_visible = cfg.get('robot', 'visible', fallback='false').lower() == 'true'
        self.robot_success_radius = cfg.getfloat('robot', 'success_radius', fallback=0.25)

        self.success_reward = cfg.getfloat('reward', 'success_reward', fallback=1.0)
        self.collision_penalty = cfg.getfloat('reward', 'collision_penalty', fallback=-0.25)
        self.timeout_penalty = cfg.getfloat('reward', 'timeout_penalty', fallback=-0.6)
        self.success_radius = cfg.getfloat('reward', 'success_radius', fallback=0.25)

        self.orca_neighbor_dist = cfg.getfloat('orca', 'neighbor_dist', fallback=3.0)
        self.orca_max_neighbors = cfg.getint('orca', 'max_neighbors', fallback=10)
        self.orca_time_horizon = cfg.getfloat('orca', 'time_horizon', fallback=3.0)
        self.orca_time_horizon_obst = cfg.getfloat('orca', 'time_horizon_obst', fallback=3.0)
        self.orca_safety_space = cfg.getfloat('orca', 'safety_space', fallback=0.20)

class PolicyConfig:

    def __init__(self, cfg: configparser.RawConfigParser):

        self.policy_name = str(cfg.get('policy', 'key', fallback='mamba_rl')).lower()

        self.seq_len = int(cfg.getfloat('buffer', 'seq_len', fallback=1))

        self.d_model = int(cfg.getfloat('policy', 'd_model', fallback=256)) if cfg.has_option('policy', 'd_model') else 256
        self.n_layers = int(cfg.getfloat('policy', 'n_layers', fallback=4)) if cfg.has_option('policy', 'n_layers') else 4

def safe_torch_load(path, map_location='cpu'):
    """安全加载checkpoint，兼容不同PyTorch版本和CUDA环境

    解决3060/4090服务器间PyTorch版本差异导致的加载问题：
    - PyTorch 2.6+: weights_only默认为True，导致加载包含非tensor对象的checkpoint失败
    - 旧版PyTorch: 不支持weights_only参数
    """
    try:
        # 优先尝试 weights_only=False（适配新版PyTorch 2.6+）
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 旧版PyTorch不支持weights_only参数
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # 其他加载错误，记录并重新抛出
        logging.warning(f"[COMPAT] Checkpoint加载异常: {e}")
        raise

def set_seed(seed: int, deterministic: bool = False):
    """
    设置全局随机种子

    Args:
        seed: 随机种子
        deterministic: 是否启用CUDA确定性模式
            - True: 完全可复现，但速度慢20-30%（测评用）
            - False: 允许CUDA优化，速度快但有微小数值差异（训练用，默认）
    """
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        # 关闭TF32以确保完全确定性（TF32有舍入误差）
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logging.info("[SEED] 🔒 CUDA确定性模式已启用（完全可复现，速度较慢）")
    elif torch.cuda.is_available():
        # 训练模式：保持性能优化
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logging.info("[SEED] ⚡ CUDA确定性模式已关闭（训练模式，速度优先）")

def load_il_ckpt(policy, value, path, device="cpu"):

    ckpt = safe_torch_load(path, map_location=device)

    sd_val = ckpt.get("value") or ckpt.get("value_state")
    assert sd_val, f"IL ckpt缺少value state_dict: {path}"

    value.load_state_dict(sd_val, strict=False)

    # PPO不需要target_value network

    meta = ckpt.get("meta", {})
    stage = str(meta.get("stage", "")).lower() if isinstance(meta.get("stage"), str) else str(ckpt.get("stage", "")).lower()
    assert "il" in stage, f"不是IL权重或元信息缺失: stage={stage}"

    success_rate = meta.get("il_success_rate") or ckpt.get("il_success_rate", 0.0)
    arch = meta.get("arch", "unknown")
    ts = meta.get("ts", 0.0)

    return {
        "success_rate": float(success_rate),
        "arch": str(arch),
        "ts": float(ts),
        "meta": meta
    }

def apply_occlusion_override(cfg, args):
    """Order 17 item 14: --occlusion-mode is the single switch that selects an
    arm. It writes into the one [occlusion] section rather than creating a
    per-arm config file, so the four arms cannot diverge on anything else.

    Occlusion runs are hard-wired to value-lookahead: the direct-action
    interface was measured at 18.5% success against 87.9% for lookahead, and
    running an occlusion arm on it would confound the experiment with a known
    broken interface.
    """
    mode = getattr(args, 'occlusion_mode', None)
    if mode is not None:
        if not cfg.has_section('occlusion'):
            cfg.add_section('occlusion')
        cfg.set('occlusion', 'mode', mode)
        cfg.set('occlusion', 'enabled', 'false' if mode == 'off' else 'true')
    belief_features = getattr(args, 'belief_features', None)
    if belief_features is not None:
        if not cfg.has_section('occlusion'):
            cfg.add_section('occlusion')
        cfg.set('occlusion', 'belief_features', belief_features)
    active_mode = _occlusion_mode(cfg)
    if active_mode != 'off' and bool(getattr(args, 'discrete_mamba', False)):
        raise SystemExit("--occlusion-mode requires value-lookahead; "
                         "--discrete-mamba is refused for occlusion arms")
    return cfg


def _occlusion_mode(cfg):
    if not cfg.has_section('occlusion'):
        return 'off'
    enabled = cfg.getboolean('occlusion', 'enabled', fallback=False)
    return cfg.get('occlusion', 'mode', fallback='off').strip().lower() if enabled else 'off'


def _occlusion_meta(cfg, policy=None):
    from crowd_nav.policy.mamba_rl import occlusion_checkpoint_meta
    backbone = getattr(
        policy,
        'temporal_backbone',
        cfg.get('mamba', 'temporal_backbone', fallback='mamba'),
    )
    return occlusion_checkpoint_meta(cfg, _occlusion_mode(cfg), str(backbone).lower())


def _assert_occlusion_resume(cfg, policy, checkpoint):
    """Reject arm/config drift for an occlusion checkpoint resume."""
    if _occlusion_mode(cfg) == 'off':
        return
    from crowd_nav.policy.mamba_rl import assert_checkpoint_compatible
    expected = _occlusion_meta(cfg, policy)
    outer_meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
    saved = outer_meta.get('occlusion') if isinstance(outer_meta, dict) else None
    assert_checkpoint_compatible(
        saved,
        expected['occlusion_mode'],
        expected['backbone'],
        expected_meta=expected,
    )


def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ('policy_state', 'model_state_dict', 'value', 'policy', 'model', 'value_state'):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_config(path: str) -> configparser.RawConfigParser:
    cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)

    canonical_parts = {'env.config', 'policy.config', 'train.config'}
    if os.path.basename(path) in canonical_parts or path.endswith('config.txt'):
        config_dir = os.path.dirname(path) if os.path.dirname(path) else './configs'
        config_files = [
            os.path.join(config_dir, 'env.config'),
            os.path.join(config_dir, 'policy.config'),
            os.path.join(config_dir, 'train.config')
        ]

        missing = [config_file for config_file in config_files
                   if not os.path.exists(config_file)]
        if missing:
            raise FileNotFoundError(
                f"canonical config bundle is incomplete under {config_dir}: {missing}"
            )
        for config_file in config_files:
            cfg.read(config_file, encoding='utf-8')
    else:
        cfg.read(path, encoding='utf-8')

    # ✅ 兼容：若缺少[policy]，则从[action_space]/[discrete_actions]补齐（保证动作表单一口径）
    if not cfg.has_section('policy'):
        src = None
        if cfg.has_section('action_space'):
            src = 'action_space'
        elif cfg.has_section('discrete_actions'):
            src = 'discrete_actions'
        if src is not None:
            cfg.add_section('policy')
            for key in ('n_speeds', 'n_headings', 'v_min', 'v_max', 'sampling', 'include_stop', 'stop_eps'):
                if cfg.has_option(src, key):
                    cfg.set('policy', key, cfg.get(src, key))
    return cfg

def setup_logger(outdir: str, level=logging.INFO):
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, 'train.log')

    if os.path.exists(log_path):
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(outdir, f'train_backup_{timestamp}.log')
        try:
            os.rename(log_path, backup_path)
            print(f"LOG_BACKUP: Previous log backed up to: {backup_path}")
        except Exception as e:
            print(f"LOG_BACKUP: Log backup failed: {e}")

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    h1 = logging.StreamHandler(sys.stdout); h1.setFormatter(fmt)
    h2 = logging.FileHandler(log_path); h2.setFormatter(fmt)
    root = logging.getLogger(); root.handlers = []; root.setLevel(level)

    root.addHandler(h1); root.addHandler(h2)
    return logging.getLogger(__name__)


def _ablation_done_path(outdir: str) -> str:
    return os.path.join(outdir, "ablation_done.json")


def _write_ablation_done(outdir: str, payload: dict):
    os.makedirs(outdir, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(_ablation_done_path(outdir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _latest_rl_checkpoint(outdir: str) -> Optional[str]:
    candidates = []
    for path in glob.glob(os.path.join(outdir, "rl_model_ep*.pth")):
        match = re.search(r"rl_model_ep(\d+)\.pth$", os.path.basename(path))
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _latest_valid_discrete_checkpoint(outdir: str) -> Optional[str]:
    candidates = []
    for path in glob.glob(os.path.join(outdir, "rl_model_ep*.pth")):
        match = re.search(r"rl_model_ep(\d+)\.pth$", os.path.basename(path))
        if match:
            candidates.append((int(match.group(1)), path))

    required = {'policy_state', 'target_q_net_state', 'optim_q_state'}
    for _, path in sorted(candidates, reverse=True):
        try:
            checkpoint = safe_torch_load(path, map_location='cpu')
        except Exception as e:
            print(f"[AUTO-RESUME] Ignoring unreadable checkpoint {path}: {e}", flush=True)
            continue
        if str(checkpoint.get('algo', '')).lower() != 'discrete_mamba':
            continue
        if int(checkpoint.get('checkpoint_version', 0)) < 3:
            print(
                f"[AUTO-RESUME] Ignoring legacy checkpoint without constrained-DQN fix: {path}",
                flush=True,
            )
            continue
        missing = required.difference(checkpoint)
        if missing:
            print(f"[AUTO-RESUME] Ignoring incomplete checkpoint {path}: missing {sorted(missing)}", flush=True)
            continue
        return path
    return None


def _prune_rl_checkpoints(outdir: str, keep: int = 3):
    valid_candidates = []
    for path in glob.glob(os.path.join(outdir, "rl_model_ep*.pth")):
        match = re.search(r"rl_model_ep(\d+)\.pth$", os.path.basename(path))
        if not match:
            continue
        try:
            checkpoint = safe_torch_load(path, map_location='cpu')
            is_valid = (
                int(checkpoint.get('checkpoint_version', 0)) >= 3
                and str(checkpoint.get('algo', '')).lower() == 'discrete_mamba'
                and all(
                    key in checkpoint
                    for key in ('policy_state', 'target_q_net_state', 'optim_q_state')
                )
            )
        except Exception:
            is_valid = False

        if is_valid:
            valid_candidates.append((int(match.group(1)), path))
            continue

        try:
            os.remove(path)
            logging.info(f"[CHECKPOINT-PRUNE] removed legacy/incomplete {path}")
        except OSError as e:
            logging.warning(f"[CHECKPOINT-PRUNE] failed to remove {path}: {e}")

    valid_candidates.sort(key=lambda item: item[0], reverse=True)
    for _, path in valid_candidates[max(1, int(keep)):]:
        try:
            os.remove(path)
            logging.info(f"[CHECKPOINT-PRUNE] removed old valid checkpoint {path}")
        except OSError as e:
            logging.warning(f"[CHECKPOINT-PRUNE] failed to remove {path}: {e}")


def _require_free_disk_space(path: str, minimum_gib: float = 2.0):
    os.makedirs(path, exist_ok=True)
    free_bytes = shutil.disk_usage(path).free
    minimum_bytes = int(float(minimum_gib) * (1024 ** 3))
    if free_bytes < minimum_bytes:
        raise RuntimeError(
            f"Insufficient disk space for training: {free_bytes / (1024 ** 3):.2f} GiB free "
            f"at {os.path.abspath(path)}; require at least {minimum_gib:.1f} GiB"
        )


def _checkpoint_episode(path: Optional[str]) -> int:
    if not path:
        return 0
    match = re.search(r"rl_model_ep(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else 0


def _run_two_stage_ablation_suite(args) -> int:
    """Run reviewer-requested two-stage ablations in clean subprocesses."""
    script = os.path.abspath(__file__)
    base_outdir = os.path.abspath(args.outdir)
    variants_root = os.path.join(base_outdir, "two_stage_ablations")
    os.makedirs(variants_root, exist_ok=True)

    cfg = load_config(args.config)
    target_episodes = int(cfg.getfloat('train', 'train_episodes',
                                       fallback=cfg.getfloat('trainer', 'episodes', fallback=3000)))

    base_cmd = [sys.executable, script, "--config", args.config, "--device", args.device]
    if args.policy:
        base_cmd += ["--policy", args.policy]
    if args.seed is not None:
        base_cmd += ["--seed", str(args.seed)]
    if args.gpu:
        base_cmd.append("--gpu")
    if args.cpu:
        base_cmd.append("--cpu")
    if args.deterministic:
        base_cmd.append("--deterministic")

    variants = [
        {
            "name": "orca_only",
            "flag": "--pretrain-only",
            "description": "ORCA pretraining only / w/o online fine-tuning",
        },
        {
            "name": "no_orca_pretrain",
            "flag": "--skip-il-pretrain",
            "description": "w/o ORCA pretraining, online training from scratch",
        },
    ]

    print(f"[ABLATION-SUITE] root={variants_root}", flush=True)
    for variant in variants:
        outdir = os.path.join(variants_root, variant["name"])
        done_marker = _ablation_done_path(outdir)
        il_policy = os.path.join(outdir, "il_policy.pth")

        if os.path.exists(done_marker):
            print(f"[ABLATION-SUITE] skip {variant['name']}: done marker exists", flush=True)
            continue

        if variant["name"] == "orca_only" and os.path.exists(il_policy) and os.path.getsize(il_policy) > 1024:
            _write_ablation_done(outdir, {
                "variant": variant["name"],
                "description": variant["description"],
                "status": "done",
                "reason": "existing_il_policy",
                "il_policy": il_policy,
            })
            print(f"[ABLATION-SUITE] skip {variant['name']}: existing IL policy found", flush=True)
            continue

        cmd = base_cmd + ["--outdir", outdir, variant["flag"]]
        if variant["name"] == "no_orca_pretrain":
            latest = _latest_rl_checkpoint(outdir)
            if latest and _checkpoint_episode(latest) >= target_episodes:
                _write_ablation_done(outdir, {
                    "variant": variant["name"],
                    "description": variant["description"],
                    "status": "done",
                    "reason": "latest_checkpoint_reached_target_episode",
                    "latest_checkpoint": latest,
                    "target_episodes": target_episodes,
                })
                print(f"[ABLATION-SUITE] skip {variant['name']}: checkpoint already reached target episode", flush=True)
                continue
            if latest:
                cmd += ["--resume", latest]
                print(f"[ABLATION-SUITE] resume {variant['name']} from {latest}", flush=True)

        print(f"[ABLATION-SUITE] start {variant['name']}: {variant['description']}", flush=True)
        print("[ABLATION-SUITE] command: " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ABLATION-SUITE] {variant['name']} failed with code {result.returncode}", flush=True)
            return result.returncode
        print(f"[ABLATION-SUITE] finished {variant['name']}", flush=True)

    with open(os.path.join(variants_root, "suite_done.json"), "w", encoding="utf-8") as f:
        json.dump({
            "status": "done",
            "variants_root": variants_root,
            "variants": [v["name"] for v in variants],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, sort_keys=True)
    print("[ABLATION-SUITE] all requested ablations finished", flush=True)
    return 0

class GradientMonitor:
    def __init__(self, sample_every: int = 200, mode: str = 'lite', watch=None):
        self.sample_every = sample_every; self.mode = mode
        self.watch = watch or ['spatial_mamba', 'temporal_mamba', 'value_head']
        self.last_sample_step = 0; self.history = []
        self.cache = {'total_norm': 0.0, 'last': -1}
    def analyze(self, net: nn.Module, step: int) -> Dict[str, float]:
        if self.mode == 'off':
            return {'grad_norm_total': 0.0}
        if step - self.last_sample_step < self.sample_every:
            return {'grad_norm_total': self.cache['total_norm']}
        self.last_sample_step = step
        tot2 = 0.0
        for n, p in net.named_parameters():
            if p.grad is None: continue
            g = p.grad.data
            tot2 += float(g.norm(2).item() ** 2)
        total = float(tot2 ** 0.5) if tot2 > 0 else 0.0
        self.cache['total_norm'] = total
        self.history.append(total); self.history = self.history[-50:]
        if step % (self.sample_every * 4) == 0:
            logging.info(f"[GRAD] step={step} total={total:.3f}")
        return {'grad_norm_total': total}

def sarl_style_update(policy, target_policy, memory, optimizer, batch_size, gamma, device):
    """SARL Update: Regression on MC Returns (Stable Target)"""
    import torch
    import torch.nn.functional as F

    if len(memory) < batch_size:
        return 0.0

    batch = memory.sample(batch_size, device)
    states = batch['states']
    # next_states = batch['next_states'] # Not needed for MC regression
    # rewards = batch['rewards']         # Not needed for MC regression
    # dones = batch['dones']             # Not needed for MC regression
    returns = batch['returns']           # ✅ Use Monte Carlo Return G_t

    values_pred = policy.forward_value(states).squeeze(-1)

    # ❌ Disable TD(0) Bootstrapping (Unstable with full IL buffer)
    # with torch.no_grad():
    #     next_values = target_policy.forward_value(next_states).squeeze(-1)
    #     values_target = rewards + gamma * next_values * (1.0 - dones)

    # ✅ Use stable MC target
    values_target = returns

    loss = F.mse_loss(values_pred, values_target)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def discrete_q_update(
    policy,
    target_policy,
    memory,
    optimizer,
    batch_size,
    gamma,
    device,
    teacher_policy=None,
    td_weight: float = 1.0,
    bc_weight: float = 0.0,
    bc_temperature: float = 1.0,
    q_target_min: float = -0.5,
    q_target_max: float = 1.0,
):
    """Teacher-regularized Double-DQN update for direct discrete actions."""
    import torch
    import torch.nn.functional as F

    if len(memory) < batch_size:
        return 0.0

    # cuDNN RNN/GRU backward requires the module to be in training mode.
    # Evaluation inside the RL loop may leave the policy in eval() mode.
    policy.train()
    target_policy.eval()
    if teacher_policy is not None:
        teacher_policy.eval()

    batch = memory.sample(batch_size, device)
    states = batch['states']
    action_indices = batch['action_indices'].long()
    rewards = batch['rewards']
    next_states = batch['next_states']
    dones = batch['dones']

    q_values = policy.forward_q(states)
    q_selected = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_online_actions = policy.forward_q(next_states).argmax(dim=1, keepdim=True)
        next_target_values = target_policy.forward_q(next_states)
        next_q = next_target_values.gather(1, next_online_actions).squeeze(1)
        q_target = rewards + float(gamma) * (1.0 - dones) * next_q
        q_target = q_target.clamp(
            min=float(q_target_min),
            max=float(q_target_max),
        )

    td_loss = F.smooth_l1_loss(q_selected, q_target)
    bc_loss = torch.zeros((), dtype=td_loss.dtype, device=device)
    if teacher_policy is not None and bc_weight > 0.0:
        temperature = max(1e-3, float(bc_temperature))
        with torch.no_grad():
            teacher_logits = teacher_policy.forward_q(states)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(q_values / temperature, dim=-1)
        bc_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
        ) * (temperature ** 2)

    loss = float(td_weight) * td_loss + float(bc_weight) * bc_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return {
        'loss': float(loss.item()),
        'q_loss': float(td_loss.item()),
        'bc_loss': float(bc_loss.item()),
    }


def build_env_and_robot(cfg) -> Tuple[CrowdSim, Robot]:

    env = CrowdSim(); env.configure(cfg)
    robot = Robot(cfg, 'robot'); env.set_robot(robot)
    return env, robot

def build_policy(policy_key: str, policy_cfg, device: torch.device):
    Policy = policy_factory[policy_key]
    try:
        policy = Policy(policy_cfg, device=device)
    except TypeError:
        try:
            policy = Policy(policy_cfg)
        except TypeError:
            policy = Policy()

    if hasattr(policy,'configure'):
        policy.configure(policy_cfg)

    if hasattr(policy, 'to'):
        policy.to(device)
    elif hasattr(policy,'set_device'):
        policy.set_device(device)

    return policy

def bind_policy(robot: Robot, policy, env, epsilon: Optional[float]=None):
    if hasattr(policy,'set_env'): policy.set_env(env)
    if hasattr(policy,'set_robot'): policy.set_robot(robot)
    if epsilon is not None and hasattr(policy,'set_epsilon'): policy.set_epsilon(epsilon)
    robot.set_policy(policy)
    return policy

def _log_action_table(GRID):

    from crowd_nav.contracts import grid_action_dim
    n_actions = grid_action_dim(GRID)
    v_max_grid = GRID['v_max']
    v_min_grid = GRID['v_min']
    include_stop = bool(GRID.get('include_stop', False))
    sampling = str(GRID.get('sampling', 'even')).strip().lower()

    # 验证动作空间配置
    assert GRID['n_speeds'] == 5, f"Expected 5 speeds, got {GRID['n_speeds']}"
    assert GRID['n_headings'] == 16, f"Expected 16 headings, got {GRID['n_headings']}"
    assert v_min_grid >= 0.0, f"v_min should be >= 0.0, got {v_min_grid}"
    if include_stop:
        assert n_actions == GRID['n_speeds'] * GRID['n_headings'] + 1, \
            f"Expected {GRID['n_speeds']*GRID['n_headings']+1} actions, got {n_actions}"
        assert sampling in ('exponential', 'even'), f"Unexpected sampling: {sampling}"

    return n_actions

def _log_policy_tag(phase, tag):

    logging.info(f"[POLICY-TAG] phase={phase} tag={tag}")

def _clamp_policy_std(policy, min_std: float = 0.05, max_std: float = 0.25):
    """将策略的动作标准差限制在安全范围，避免BC权重被大探索噪声破坏"""
    if not hasattr(policy, 'action_log_std'):
        return
    try:
        with torch.no_grad():
            current_std = policy.action_log_std.detach().exp()
            clamped_std = current_std.clamp(min=min_std, max=max_std)
            if not torch.allclose(clamped_std, current_std):
                policy.action_log_std.copy_(clamped_std.log())
                logging.info(f"[ACTION-STD] Clamped std {current_std.cpu().numpy()} -> {clamped_std.cpu().numpy()}")
            else:
                logging.info(f"[ACTION-STD] std unchanged {current_std.cpu().numpy()}")
    except Exception as e:
        logging.warning(f"[ACTION-STD] Failed to clamp std: {e}")

def _run_policy_il_pretrain(policy, device, seq_len: int, il_epochs: int, il_batch_size: int, il_ckpt_path: str,
                            explorer, gamma: float = 0.97, success_trajs: Optional[List] = None, config=None,
                            direct_discrete: bool = False):

    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from crowd_nav.contracts import joint34_to_tokens

    def _pad_state_to_34(state):

        if hasattr(state, 'to_array'):
            arr = state.to_array()

        elif isinstance(state, list) and len(state) > 0 and hasattr(state[0], 'px'):

            robot_state = np.zeros(9, dtype=np.float32)

            human_states = []
            for obs in state[:5]:
                human_states.extend([obs.px, obs.py, obs.vx, obs.vy, obs.radius])

            while len(human_states) < 25:
                human_states.append(0.0)

            arr = np.concatenate([robot_state, human_states[:25]], axis=0)
        else:

            arr = np.asarray(state, dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 1:
            if arr.shape[0] < 34:
                return np.pad(arr, (0, 34 - arr.shape[0]), mode='constant')
            return arr[:34]
        elif arr.ndim == 2:
            pad_width = [(0, 0), (0, max(0, 34 - arr.shape[1]))]
            arr = np.pad(arr, pad_width, mode='constant') if arr.shape[1] < 34 else arr[:, :34]
            return arr
        return arr.reshape(-1)[:34]

    def _build_il_dataset(trajs, gamma_value, require_action_indices: bool = True,
                          requantize_actions: bool = False):

        seq_len_local = max(1, int(seq_len))
        state_windows = []
        teacher_indices = []
        returns = []

        total_steps = 0

        from crowd_nav.contracts import GRID, action_to_discrete_index, grid_action_dim
        n_actions = int(grid_action_dim(GRID))
        requantized_count = 0
        changed_label_count = 0

        for (traj, _info) in trajs:
            states, actions, rewards = traj
            T = len(states)
            if T == 0:
                continue

            action_indices = None
            if isinstance(_info, dict):
                action_indices = _info.get('action_indices')
            if action_indices is None and isinstance(traj, dict):
                action_indices = traj.get('action_indices')

            if require_action_indices:
                if requantize_actions:
                    if actions is None or len(actions) < T:
                        raise RuntimeError(
                            "[IL-DISCRETE] Continuous teacher actions are required to remap labels "
                            "to the current action grid."
                        )
                    remapped_indices = []
                    for t in range(T):
                        action_t = actions[t]
                        if hasattr(action_t, 'vx'):
                            vx, vy = float(action_t.vx), float(action_t.vy)
                        else:
                            action_arr = np.asarray(action_t, dtype=np.float32).reshape(-1)
                            if action_arr.size < 2:
                                raise RuntimeError(
                                    f"[IL-DISCRETE] Invalid teacher action at step {t}: {action_t!r}"
                                )
                            vx, vy = float(action_arr[0]), float(action_arr[1])
                        new_idx = int(action_to_discrete_index(vx, vy, grid=GRID))
                        remapped_indices.append(new_idx)
                        requantized_count += 1
                        if action_indices is not None and t < len(action_indices):
                            changed_label_count += int(int(action_indices[t]) != new_idx)
                    action_indices = remapped_indices
                elif action_indices is None:
                    raise RuntimeError("[IL-VALUE] Missing action_indices in IL trajectories. "
                                       "Please recollect IL online with discrete action indices.")
                if len(action_indices) < T or any(a is None for a in action_indices):
                    raise RuntimeError("[IL-VALUE] action_indices incomplete or contains None. "
                                       "Please recollect IL online with discrete action indices.")
            else:
                # SARL-style IL: ignore action indices entirely (continuous ORCA trajectories)
                action_indices = None

            state_frames = []
            for state in states:
                arr = np.asarray(state, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 13 and arr.shape[0] >= 4:
                    state_frames.append(arr)
                else:
                    state_frames.append(_pad_state_to_34(state))
            frame_shapes = {frame.shape for frame in state_frames}
            if len(frame_shapes) != 1:
                raise RuntimeError(
                    f"[IL-DATASET] mixed state contracts in one episode: {frame_shapes}")

            mc = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma_value * G
                mc.insert(0, G)

            for t in range(T):
                window = state_frames[max(0, t - seq_len_local + 1): t + 1]
                if not window:
                    continue
                if len(window) < seq_len_local:
                    pad_needed = seq_len_local - len(window)
                    pad_frame = window[0]
                    window = [pad_frame.copy() for _ in range(pad_needed)] + [w.copy() for w in window]
                else:
                    window = [w.copy() for w in window[-seq_len_local:]]

                state_windows.append(np.stack(window, axis=0).astype(np.float32))

                if require_action_indices and action_indices is not None:
                    idx_t = int(action_indices[t])
                    if not (0 <= idx_t < n_actions):
                        raise RuntimeError(f"[IL-VALUE] action_idx out of range: {idx_t} (0..{n_actions-1})")
                    teacher_indices.append(idx_t)
                returns.append(mc[t])
                total_steps += 1

        if not state_windows:
            raise RuntimeError("[IL-VALUE] No data available for IL value regression")

        state_windows = np.asarray(state_windows, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)
        teacher_indices = np.asarray(teacher_indices, dtype=np.int64) if teacher_indices else None

        logging.info(f"[IL-ACTION-PROCESS] Total steps: {total_steps}, n_actions={n_actions}")
        if requantize_actions:
            logging.info(
                f"[IL-ACTION-REMAP] Re-quantized {requantized_count} teacher actions to the current "
                f"{n_actions}-action grid; changed_labels={changed_label_count}"
            )

        if (state_windows.ndim == 4 and state_windows.shape[-1] == 13 and
                state_windows.shape[-2] >= 4):
            tokens = torch.from_numpy(state_windows).contiguous()
            total_states = state_windows.shape[0] * state_windows.shape[1]
            logging.info(
                f"[IL-TOKEN-CONTRACT] Preserved {total_states} native belief "
                "tokens without a 34-D round trip")
        else:
            flat_states = state_windows.reshape(-1, state_windows.shape[-1])
            total_states = flat_states.shape[0]
            batch_size_token = 100000
            if total_states > batch_size_token:
                logging.info(
                    f"[IL-TOKEN-CONVERT] Converting {total_states} states to tokens in batches "
                    f"(batch_size={batch_size_token}, please wait)..."
                )
                token_batches = []
                with tqdm(total=total_states, desc="[IL-TOKEN-CONVERT]", ncols=100, unit="states") as pbar:
                    for start_idx in range(0, total_states, batch_size_token):
                        end_idx = min(start_idx + batch_size_token, total_states)
                        batch_states = flat_states[start_idx:end_idx]
                        batch_tokens = joint34_to_tokens(batch_states)
                        token_batches.append(batch_tokens)
                        pbar.update(end_idx - start_idx)
                if isinstance(token_batches[0], torch.Tensor):
                    tokens = torch.cat(token_batches, dim=0)
                else:
                    tokens = np.concatenate(token_batches, axis=0)
                logging.info(f"[IL-TOKEN-CONVERT] Token conversion complete ({total_states} states)")
            else:
                logging.info(f"[IL-TOKEN-CONVERT] Converting {total_states} states to tokens...")
                tokens = joint34_to_tokens(flat_states)
                logging.info(f"[IL-TOKEN-CONVERT] Token conversion complete")
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.view(
                    state_windows.shape[0], seq_len_local, 1, 8, 13,
                ).squeeze(2).contiguous()
            else:
                tokens = torch.from_numpy(tokens).view(
                    state_windows.shape[0], seq_len_local, 1, 8, 13,
                ).squeeze(2).contiguous()

        logging.info(f"[IL-DATASET] IL dataset: {state_windows.shape[0]} sequences (seq_len={seq_len_local}, with MC returns)")

        out = {
            'states_tokens': tokens.float(),
            'returns': torch.from_numpy(returns).float()
        }
        if teacher_indices is not None:
            out['action_indices'] = torch.from_numpy(teacher_indices).long()
        return out

    ac_net = policy

    def _get_cfg_float(cfg_obj, section, key, default):
        try:
            return cfg_obj.getfloat(section, key, fallback=default)
        except Exception:
            return getattr(cfg_obj, key, default)

    # 🔥 关键修复：IL阶段也用AdamW（与RL保持一致）
    # Mamba需要AdamW，不是SGD
    bc_lr = _get_cfg_float(config, 'train', 'il_learning_rate', 1e-4)
    optim = torch.optim.AdamW(
        ac_net.parameters(),
        lr=bc_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # LR Schedule: Cosine annealing (AdamW标准)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=il_epochs,
        eta_min=1e-6
    )

    logging.info(f"[IL-BC] Training policy network - {sum(p.numel() for p in ac_net.parameters())} params")

    ac_net.train()

    # ========== Critical Fix: BC training uses RL mode (full sequence modeling) ==========
    # BC should mimic full seq_len input, not just last 1 frame
    ac_net.set_training_mode('rl')
    logging.info(f"[IL-BC-MODE] Using RL mode for BC training (full {seq_len}-frame temporal modeling)")
    logging.info(f"[IL-BC-MODE] This ensures BC learns sequence dependencies, not just single-frame actions")

    il_objective = "discrete_action_ce" if direct_discrete else "value_regression"
    logging.info(f"[IL-OBJECTIVE] mode={il_objective}, lr={bc_lr}, gamma={gamma}")

    dataset_trajs = success_trajs if success_trajs is not None else getattr(explorer, "_last_trajectories", [])
    if not dataset_trajs:
        logging.warning("[IL-CONTINUOUS-BC] No trajectories available, skipping BC training")
        return

    data_source = "Provided trajectories" if success_trajs is not None else "Online collection"
    logging.info(f"[IL-BC] Data source: {data_source}")
    logging.info(f"[IL-BC] Trajectories: {len(dataset_trajs)}")

    # Direct discrete baseline imitates the quantized ORCA action. The full
    # Mamba-VL path keeps its original value-regression objective.
    require_indices = bool(direct_discrete)
    dataset = _build_il_dataset(
        dataset_trajs,
        gamma,
        require_action_indices=require_indices,
        requantize_actions=direct_discrete
    )
    states_tokens = dataset['states_tokens'].to(device, non_blocking=True)
    mc_returns_full = dataset['returns'].to(device, non_blocking=True)
    action_indices_full = dataset.get('action_indices')  # 可能为 None
    if action_indices_full is not None:
        action_indices_full = action_indices_full.to(device, non_blocking=True)
        logging.info("[IL-BC] Action indices loaded for discrete-action imitation")
    if direct_discrete and action_indices_full is None:
        raise RuntimeError("[IL-DISCRETE] Offline trajectories do not contain action_indices")
    teacher_indices_full = None
    dataset_size = states_tokens.size(0)

    logging.info(f"[IL-BC] Prepared dataset: N={dataset_size}, batch_size={il_batch_size}, batches={dataset_size // il_batch_size}")

    # [Step 5] Clamp log_std during BC
    def _clamp_policy_std(net, min_log_std=-2.5, max_log_std=-1.0):
        """Clamp action_log_std to keep log_std in [min_log_std, max_log_std] during BC."""
        if hasattr(net, 'action_log_std'):
            with torch.no_grad():
                net.action_log_std.data.clamp_(min=min_log_std, max=max_log_std)

    # [Step 6] Best-epoch tracking + Early stop
    import copy
    best_sr = 0.0
    best_state = None
    best_epoch = 0
    min_epochs = 15  # Train at least 15 epochs
    patience = 3  # Stop if no improvement for 3 evals
    eval_interval = 5  # Eval every 5 epochs
    progress_path = il_ckpt_path + ".progress"
    start_il_epoch = 1

    if (os.path.exists(progress_path)
            and not config.getboolean('train', 'il_force_retrain', fallback=False)):
        try:
            progress = safe_torch_load(progress_path, map_location=device)
            _assert_occlusion_resume(config, policy, progress)
            if progress.get('objective') == il_objective:
                ac_net.load_state_dict(progress['model_state_dict'], strict=True)
                optim.load_state_dict(progress['optimizer_state'])
                scheduler.load_state_dict(progress['scheduler_state'])
                start_il_epoch = int(progress.get('epoch', 0)) + 1
                best_sr = float(progress.get('best_sr', 0.0))
                best_epoch = int(progress.get('best_epoch', 0))
                best_state = progress.get('best_state')
                logging.info(f"[IL-RESUME] objective={il_objective}, resume epoch={start_il_epoch}/{il_epochs}")
            else:
                logging.warning("[IL-RESUME] Progress objective mismatch; starting IL from scratch")
        except Exception as e:
            logging.warning(f"[IL-RESUME] Failed to restore {progress_path}: {e}")
            if _occlusion_mode(config) != 'off':
                raise

    def _quick_eval_bc(net, expl, dev, n_episodes=20):
        """Quick eval of BC success rate"""
        net.eval()
        policy.set_phase('eval')
        old_sarl_predict = getattr(policy, 'use_sarl_predict', False)
        policy.use_sarl_predict = not direct_discrete
        try:
            stats = expl.run_k_episodes(
                n_episodes, 'test',
                update_memory=False, show_tqdm=False,
                return_stats=True, imitation_learning=False,
                force_joint_state_policy=False
            )
            sr = stats.get('success_rate', 0.0) if isinstance(stats, dict) else 0.0
        except Exception as e:
            logging.warning(f"[IL-BC-EVAL] Quick eval failed: {e}")
            sr = 0.0
        finally:
            net.train()
            policy.set_phase('train')
            policy.use_sarl_predict = old_sarl_predict  # 恢复
        return sr

    avg_value_loss = 0.0
    avg_action_loss = 0.0
    avg_total_loss = 0.0
    pbar = tqdm(range(start_il_epoch, il_epochs + 1), desc="[IL-BC]", ncols=120, leave=True)
    for ep in pbar:
        # [Step 5] Clamp log_std at start of each epoch
        _clamp_policy_std(ac_net, min_log_std=-2.5, max_log_std=-1.0)

        if dataset_size < il_batch_size:
            logging.warning(f"[IL-BC] Dataset too small (N={dataset_size}), reducing batch size to {dataset_size}")
        perm = torch.randperm(dataset_size, device=device)

        total_value_loss = torch.tensor(0.0, device=device)
        total_action_loss = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        n_batches = 0

        for start in range(0, dataset_size, il_batch_size):
            idx = perm[start:start + il_batch_size]
            if idx.numel() < 8:
                continue

            states_batch = states_tokens[idx].contiguous()
            mc_returns = mc_returns_full[idx].contiguous()
            action_targets = action_indices_full[idx].contiguous() if action_indices_full is not None else None

            if states_batch.device != device:
                states_batch = states_batch.to(device, non_blocking=True)
            if mc_returns.device != device:
                mc_returns = mc_returns.to(device, non_blocking=True)
            if action_targets is not None and action_targets.device != device:
                action_targets = action_targets.to(device, non_blocking=True)

            if direct_discrete:
                logits = ac_net.forward_q(states_batch)
                action_loss = F.cross_entropy(logits, action_targets)
                value_loss = torch.tensor(0.0, device=device)
                loss = action_loss
            else:
                values_pred = ac_net.forward_value(states_batch).squeeze(-1)
                value_loss = F.mse_loss(values_pred, mc_returns)
                action_loss = torch.tensor(0.0, device=device)
                loss = value_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac_net.parameters(), 1.0)
            optim.step()

            total_value_loss += value_loss.detach()
            total_action_loss += action_loss.detach()
            total_loss += loss.detach()
            n_batches += 1

        avg_value_loss = (total_value_loss / max(1, n_batches)).item()
        avg_action_loss = (total_action_loss / max(1, n_batches)).item()
        avg_total_loss = (total_loss / max(1, n_batches)).item()

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        pbar.set_postfix({
            'v': f'{avg_value_loss:.4f}',
            'a': f'{avg_action_loss:.4f}',
            'total': f'{avg_total_loss:.4f}',
            'lr': f'{current_lr:.6f}'
        })

        if ep % max(1, il_epochs // 5) == 0 or ep == il_epochs:
            logging.info(
                f"[IL-{('DISCRETE' if direct_discrete else 'VALUE')}] epoch={ep}/{il_epochs} "
                f"value_loss={avg_value_loss:.4f} action_loss={avg_action_loss:.4f} "
                f"total={avg_total_loss:.4f} lr={current_lr:.6f}"
            )

        # [Step 6] Quick eval + early stop check every eval_interval epochs
        if ep % eval_interval == 0:
            sr = _quick_eval_bc(ac_net, explorer, device, n_episodes=20)
            logging.info(f"[IL-BC-EVAL] epoch={ep} quick_eval sr={sr:.1%} (best={best_sr:.1%} @ epoch {best_epoch})")

            # Update best
            if sr > best_sr:
                best_sr = sr
                best_epoch = ep
                best_state = copy.deepcopy(ac_net.state_dict())
                logging.info(f"[IL-BC] ★ New best! epoch={ep}, sr={best_sr:.1%}")

            # Early stop check
            if ep >= min_epochs and ep - best_epoch >= patience * eval_interval:
                logging.info(f"[IL-BC] ✓ Early stop: no improvement for {patience} evals (best @ epoch {best_epoch}, sr={best_sr:.1%})")
                torch.save({
                    'objective': il_objective,
                    'epoch': ep,
                    'model_state_dict': ac_net.state_dict(),
                    'optimizer_state': optim.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_sr': best_sr,
                    'best_epoch': best_epoch,
                    'best_state': best_state,
                    'meta': {'occlusion': _occlusion_meta(config, policy)},
                }, progress_path)
                break

        torch.save({
            'objective': il_objective,
            'epoch': ep,
            'model_state_dict': ac_net.state_dict(),
            'optimizer_state': optim.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_sr': best_sr,
            'best_epoch': best_epoch,
            'best_state': best_state,
            'meta': {'occlusion': _occlusion_meta(config, policy)},
        }, progress_path)

    pbar.close()

    # ========== Save BC Checkpoint (with metadata) ==========
    # [Step 6] Use best_state if available
    final_state = best_state if best_state is not None else ac_net.state_dict()
    final_epoch = best_epoch if best_state is not None else il_epochs
    final_sr = best_sr if best_state is not None else None

    if best_state is not None:
        logging.info(f"[IL-BC] Using best checkpoint from epoch {best_epoch} (sr={best_sr:.1%})")
        ac_net.load_state_dict(best_state)  # Restore best state
    else:
        logging.info(f"[IL-BC] No best checkpoint found, using final epoch {il_epochs}")

    logging.info(f"[IL-BC] Saving BC checkpoint with metadata...")
    try:
        import time
        bc_checkpoint = {
            'value': final_state,  # Legacy
            'model_state_dict': final_state,  # Standard PyTorch
            'meta': {
                'stage': 'bc_policy_only',
                'arch': 'policy_network',
                'epochs': final_epoch,
                'value_loss': avg_value_loss,
                'action_loss': avg_action_loss,
                'total_loss': avg_total_loss,
                'objective': il_objective,
                'eval_success_rate': final_sr,  # Has best sr
                'best_epoch': best_epoch,
                'best_sr': best_sr,
                'seq_len': seq_len,
                'dataset_size': dataset_size,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'occlusion': _occlusion_meta(config, policy),
            }
        }
        # 🔥 修复：创建目录（如果不存在）
        os.makedirs(os.path.dirname(il_ckpt_path), exist_ok=True)
        torch.save(bc_checkpoint, il_ckpt_path)
        logging.info(f"[IL-BC] ✓ BC checkpoint saved to {il_ckpt_path}")
        if os.path.exists(progress_path):
            os.remove(progress_path)
    except Exception as e:
        logging.warning(f"[IL-BC] Failed to save checkpoint: {e}")

    logging.info(f"[IL-CONTINUOUS-BC] BC training completed")

    # 🔥 IL evaluation改为可选（默认跳过，因为Q-head未训练）
    # 🔥 修复：config可能是TrainConfig对象，需要用hasattr检查
    if hasattr(config, 'getboolean'):
        skip_il_eval = config.getboolean('imitation_learning', 'skip_il_eval', fallback=True)
    else:
        skip_il_eval = True  # TrainConfig对象，默认跳过
    if skip_il_eval:
        logging.info("[IL-BC-EVAL] Skipping IL evaluation (Q-head not trained, will validate in RL phase)")
    else:
        logging.info(f"[IL-CONTINUOUS-BC] Evaluating BC-trained model (in BC mode)...")
        # Fix: Restore training environment config before eval (ensure human_num=5)
        if hasattr(explorer, 'env'):
            explorer.env.human_num = 5
            explorer.env.config.set('sim', 'human_num', '5')
            logging.info(f"[IL-BC-EVAL] Environment reset to human_num=5 (matching BC training data)")

        # 🔥 IL evaluation禁用SARL lookahead（value网络未训练完成）
        old_use_sarl = getattr(policy, 'use_sarl_predict', False)
        if hasattr(policy, 'use_sarl_predict'):
            policy.use_sarl_predict = False
            logging.info("[IL-BC-EVAL] Disabled SARL lookahead for IL evaluation (value net not ready)")

        ac_net.eval()
        policy.set_phase('eval')

        stats = explorer.run_k_episodes(
            100,
            'test',  # Change to test mode
            update_memory=False,
            show_tqdm=False,
            return_stats=True,
            imitation_learning=False,
            force_joint_state_policy=False
        )

        policy.set_phase('train')
        ac_net.train()

        # 恢复SARL predict设置
        if hasattr(policy, 'use_sarl_predict'):
            policy.use_sarl_predict = old_use_sarl

        try:
            if isinstance(stats, dict):
                success_rate = stats.get('success_rate', 0.0)
                avg_return = stats.get('total_reward', stats.get('avg_return', 0.0))
                collision_rate = stats.get('collision_rate', 0.0)
                timeout_rate = stats.get('timeout_rate', 0.0)

                logging.info(
                    f"[IL-BC-EVAL] 100 episodes | success={success_rate:.1%} | "
                    f"collision={collision_rate:.1%} | timeout={timeout_rate:.1%} | reward={avg_return:.4f}"
                )

                # Update checkpoint with eval results
                try:
                    if os.path.exists(il_ckpt_path):
                        bc_ckpt = safe_torch_load(il_ckpt_path, map_location=device)
                        if 'meta' in bc_ckpt:
                            bc_ckpt['meta']['eval_success_rate'] = success_rate
                            bc_ckpt['meta']['eval_collision_rate'] = collision_rate
                            bc_ckpt['meta']['eval_timeout_rate'] = timeout_rate
                            torch.save(bc_ckpt, il_ckpt_path)
                            logging.info(f"[IL-BC] Updated checkpoint with eval metrics")
                except Exception as e:
                    logging.warning(f"[IL-BC] Failed to update checkpoint with eval: {e}")
            else:
                logging.info(f"[IL-BC-EVAL] Evaluation completed (stats not available)")

        except Exception as e:
            logging.warning(f"[IL-BC-EVAL] Evaluation failed: {e}")

        except Exception as e:
            logging.warning(f"[IL-BC-EVAL] Evaluation failed: {e}")
        finally:
            ac_net.train()
            policy.set_phase('train')

    ac_net.set_training_mode('rl')
    logging.info(f"[IL-BC-MODE] BC training finished, keeping RL mode for RL phase")
    logging.info(f"[IL-BC-MODE] Temporal encoder is now trained (unlike old BC-only baseline)")

    logging.info(f"[IL-BC-ADAPT] Skipping warm adaptation - temporal encoder already trained during BC")

    # 🔥 第二次IL evaluation也跳过（与第一次保持一致）
    if not skip_il_eval:
        logging.info(f"[IL-MAMBA-EVAL] Evaluating BC-trained model in RL mode (with temporal encoder)...")
        try:
            # 🔥 IL evaluation禁用SARL lookahead（value网络未训练完成）
            old_use_sarl_rl = getattr(policy, 'use_sarl_predict', False)
            if hasattr(policy, 'use_sarl_predict'):
                policy.use_sarl_predict = False
                logging.info("[IL-MAMBA-EVAL] Disabled SARL lookahead for IL evaluation (value net not ready)")

            ac_net.eval()
            policy.set_phase('eval')

            stats_rl = explorer.run_k_episodes(
                100,
                'test',  # Change to test
                update_memory=False,
                show_tqdm=False,
                imitation_learning=False,
                force_joint_state_policy=False,
                return_stats=True
            )

            if isinstance(stats_rl, dict):
                success_rate_rl = stats_rl.get('success_rate', 0.0)
                avg_return_rl = stats_rl.get('total_reward', stats_rl.get('avg_return', 0.0))
                collision_rate_rl = stats_rl.get('collision_rate', 0.0)
                timeout_rate_rl = stats_rl.get('timeout_rate', 0.0)

                logging.info(
                    f"[IL-MAMBA-EVAL] 100 episodes (RL mode) | success={success_rate_rl:.1%} | "
                    f"collision={collision_rate_rl:.1%} | timeout={timeout_rate_rl:.1%} | reward={avg_return_rl:.4f}"
                )
            else:
                logging.info(f"[IL-MAMBA-EVAL] RL mode evaluation completed (stats not available)")

        except Exception as e:
            logging.warning(f"[IL-MAMBA-EVAL] RL mode evaluation failed: {e}")
        finally:
            ac_net.train()
            policy.set_phase('train')
            # 恢复SARL predict设置
            if hasattr(policy, 'use_sarl_predict'):
                policy.use_sarl_predict = old_use_sarl_rl

    # BC checkpoint已在训练循环后保存（带完整元信息），此处不重复保存

def _run_il_phase(train_cfg, env_cfg, policy_cfg, env, explorer, policy, device, batch_size, replay_iql, cfg=None,
                  direct_discrete: bool = False):

    import math
    import copy

    # IL: 使用统一的cfg对象（RawConfigParser）
    # 所有配置通过cfg.getfloat()/cfg.getint()读取
    config = cfg if cfg is not None else train_cfg

    # 保存原始环境配置（用于恢复）
    original_time_limit = config.getfloat('env', 'time_limit', fallback=25.0)
    original_success_radius = config.getfloat('robot', 'success_radius', fallback=0.25)
    original_human_num = config.getint('sim', 'human_num', fallback=5)
    original_neighbor_dist = config.getfloat('orca', 'neighbor_dist', fallback=4.5)
    original_max_neighbors = config.getint('orca', 'max_neighbors', fallback=12)
    original_time_horizon = config.getfloat('orca', 'time_horizon', fallback=5.0)
    original_time_horizon_obst = config.getfloat('orca', 'time_horizon_obst', fallback=3.0)
    original_safety_space = config.getfloat('orca', 'safety_space', fallback=0.10)
    original_robot_visible = config.getboolean('robot', 'visible', fallback=False)

    # Teacher ORCA配置（从[imitation_learning] section读取）
    teacher_neighbor_dist = config.getfloat('imitation_learning', 'teacher_neighbor_dist', fallback=6.0)
    teacher_max_neighbors = config.getint('imitation_learning', 'teacher_max_neighbors', fallback=15)
    teacher_time_horizon = config.getfloat('imitation_learning', 'teacher_time_horizon', fallback=4.5)
    teacher_time_horizon_obst = config.getfloat('imitation_learning', 'teacher_time_horizon_obst', fallback=4.5)
    teacher_safety_space = config.getfloat('imitation_learning', 'teacher_safety_space', fallback=0.10)

    def _safe_getint(section: str, option: str, fallback: int) -> int:
        try:
            # 🔥 修复：直接使用getint而不是get+int转换
            return config.getint(section, option, fallback=fallback)
        except Exception as e:
            logging.error(f"[IL-CONFIG] Failed to read {section}.{option}: {e}, using fallback={fallback}")
            return int(fallback)

    success_target = _safe_getint('imitation_learning', 'success_target', 2000)
    max_total_episodes = _safe_getint('imitation_learning', 'max_prefill_episodes', 12000)
    batch_collect_episodes = _safe_getint('imitation_learning', 'prefill_batch_episodes', 64)
    patience_batches = _safe_getint('imitation_learning', 'prefill_patience_batches', 5)
    logging.info(
        "[IL-CONFIG] success_target=%s max_total_episodes=%s prefill_batch_episodes=%s patience_batches=%s",
        success_target, max_total_episodes, batch_collect_episodes, patience_batches
    )

    # 🔥 调试：验证读取的值
    logging.info(f"[IL-CONFIG] success_target={success_target} (type={type(success_target).__name__})")
    logging.info(f"[IL-CONFIG] max_total_episodes={max_total_episodes} (type={type(max_total_episodes).__name__})")
    logging.info(f"[IL-CONFIG] batch_collect_episodes={batch_collect_episodes} (type={type(batch_collect_episodes).__name__})")
    logging.info(f"[IL-CONFIG] patience_batches={patience_batches} (type={type(patience_batches).__name__})")

    logging.info(
        f"[IL-TEACHER] Applying ORCA teacher profile: neighbor_dist={teacher_neighbor_dist}, max_neighbors={teacher_max_neighbors}, "
        f"time_horizon={teacher_time_horizon}, time_horizon_obst={teacher_time_horizon_obst}, safety_space={teacher_safety_space}"
    )

    # 固定IL环境：5人 + 不随机属性（对齐SARL口径）
    try:
        env.config.set('env', 'randomize_attributes', 'false')
        env.config.set('sim', 'human_num', '5')
        if env.config.has_option('env', 'human_num'):
            env.config.set('env', 'human_num', '5')
        # IL阶段临时开启robot可见性（提高ORCA teacher成功率）
        if env.config.has_option('robot', 'visible'):
            env.config.set('robot', 'visible', 'true')
        if hasattr(env, 'robot'):
            try:
                env.robot.visible = True
            except Exception:
                pass
        env.human_num = 5
    except Exception as e:
        logging.warning(f"[IL-ENV] Failed to enforce fixed 5-human setting: {e}")

    # Keep environment ORCA params (for humans) at baseline; apply teacher params only to teacher policy.
    try:
        env._teacher_orca_params = {
            'neighbor_dist': float(teacher_neighbor_dist),
            'max_neighbors': int(teacher_max_neighbors),
            'time_horizon': float(teacher_time_horizon),
            'time_horizon_obst': float(teacher_time_horizon_obst),
            'safety_space': float(teacher_safety_space),
        }
    except Exception:
        env._teacher_orca_params = None
    env.configure(env.config)

    target_buffer = getattr(explorer, 'memory', None)
    if target_buffer is None and hasattr(explorer, 'replay_buffer'):
        target_buffer = explorer.replay_buffer
    if target_buffer is None:
        logging.warning("[IL] Explorer has no replay buffer reference; IL data will stay in-flight only")

    initial_buf_size = len(target_buffer) if target_buffer else 0
    logging.info(f"[IL-START] Buffer size before IL: {initial_buf_size}")

    # IL: 是否强制在线采集（不使用离线缓存）
    force_online = config.getboolean('imitation_learning', 'force_online', fallback=False)
    offline_dataset_path = None
    if force_online:
        logging.info("[IL-CACHE] force_online=true -> skip offline dataset cache, recollect IL online")
    else:
        # IL: 从main函数传入的offline_dataset_path读取
        offline_dataset_path = config.get('train', 'offline_il_dataset', fallback=None)
        if not offline_dataset_path or offline_dataset_path == 'None':
            # ========== V5.0优先：使用merged文件 ==========
            # V5.0: 15000条轨迹，极端密度/速度组合，最优质数据集
            for version in ['v5.0_merged_15000_with_idx', 'v5.0_merged_15000', 'v5.0_merged', 'v4.0', 'v3.1', 'v3.0']:
                candidates = [
                    f'data/il_dataset_diverse_{version}.pth',
                    f'../data/il_dataset_diverse_{version}.pth',
                    os.path.join(os.path.dirname(__file__), f'../data/il_dataset_diverse_{version}.pth'),
                ]
                for candidate in candidates:
                    if os.path.exists(candidate) and os.path.getsize(candidate) > 1024:  # 确保非空文件
                        offline_dataset_path = os.path.abspath(candidate)
                        logging.info(f"[IL-CACHE] 找到数据集候选: {offline_dataset_path}")
                        break
                if offline_dataset_path:
                    break

            if offline_dataset_path is None:
                # Fallback：默认路径（通常不会用到）
                offline_dataset_path = 'data/il_dataset_diverse_v5.0_merged_15000_with_idx.pth'

    success_trajs_all: List = []
    fail_trajs_all: List = []
    success_ratio = 0.0
    total_sampled_episodes = 0
    loaded_from_cache = False

    require_discrete_il = config.getboolean('imitation_learning', 'require_discrete', fallback=True)

    if (offline_dataset_path is not None) and os.path.exists(offline_dataset_path):
        try:
            logging.info(f"[IL-CACHE] 发现离线数据集缓存: {offline_dataset_path}")
            dataset = safe_torch_load(offline_dataset_path, map_location='cpu')

            if isinstance(dataset, dict) and 'trajectories' in dataset:
                cached_trajs = dataset['trajectories']
                config_info = dataset.get('config', {})
                stats_info = dataset.get('stats', {})

                logging.info(f"[IL-CACHE] 数据集版本: {dataset.get('version', 'unknown')}")
                logging.info(f"[IL-CACHE] 生成时间: {dataset.get('timestamp', 'unknown')}")
                logging.info(f"[IL-CACHE] Seeds: {config_info.get('seeds', [])}")
                logging.info(f"[IL-CACHE] ORCA configs: {list(config_info.get('orca_configs', {}).keys())}")
                logging.info(f"[IL-CACHE] 总轨迹数: {len(cached_trajs)}")
                logging.info(f"[IL-CACHE] 原始成功率: {stats_info.get('success_rate', 0.0):.2%}")

                # 离散一致性检查：若要求离散而数据集非离散，则强制回退在线采集
                dataset_discrete = False
                try:
                    action_mode = config_info.get('action_mode') or config_info.get('actions_mode')
                    if isinstance(action_mode, str) and action_mode.lower() == 'discrete':
                        dataset_discrete = True
                except Exception:
                    pass
                if not dataset_discrete and cached_trajs:
                    meta0 = cached_trajs[0].get('meta', {})
                    if meta0.get('action_mode') == 'discrete' or meta0.get('actions_discrete') is True:
                        dataset_discrete = True

                if require_discrete_il and not dataset_discrete:
                    logging.warning("[IL-CACHE] Offline dataset is not discrete; require_discrete=true -> ignore cache and recollect IL online")
                    success_trajs_all = []
                    loaded_from_cache = False
                    cached_trajs = []

                if cached_trajs:
                    # [FIX] 从config读取IL预填充上限，避免IL基数过大稀释RL梯度
                    max_il_prefill = config.getint('imitation_learning', 'max_il_prefill', fallback=15000)
                    cached_trajs = cached_trajs[:max_il_prefill]
                    logging.info(f"[IL-CACHE] 限制加载: {len(cached_trajs)} episodes (max_il_prefill={max_il_prefill})")

                    for traj in cached_trajs:
                        states = traj['states']
                        actions = traj['actions']
                        rewards = traj.get('rewards', [0.0] * len(states))
                        meta = traj.get('meta', {})
                        # 确保action_indices透传到meta
                        if 'action_indices' not in meta and traj.get('action_indices') is not None:
                            meta['action_indices'] = traj.get('action_indices')

                        # ✅ 只保留成功轨迹（过滤失败轨迹）
                        # 检查多种outcome格式
                        outcome = meta.get('outcome', None)
                        flag_success = meta.get('flag_success', False)
                        event = meta.get('event', '')

                        # 判断是否成功：优先用flag_success，其次用outcome，最后用event
                        is_success = False
                        if flag_success:
                            is_success = True
                        elif outcome in ['success', 'ReachGoal']:
                            is_success = True
                        elif 'success' in str(event).lower() or 'reachgoal' in str(event).lower():
                            is_success = True

                        traj_tuple = ((states, actions, rewards), meta)
                        if is_success:
                            success_trajs_all.append(traj_tuple)
                        else:
                            fail_trajs_all.append(traj_tuple)

                    total_sampled_episodes = stats_info.get('total_attempts', len(cached_trajs))
                    success_ratio = len(success_trajs_all) / max(1, total_sampled_episodes)
                    loaded_from_cache = True

                    # 限制失败轨迹数量（用于RL预填充）
                    max_il_fail_prefill = config.getint(
                        'imitation_learning', 'max_il_fail_prefill',
                        fallback=max(0, int(max_il_prefill * 0.5))
                    )
                    if max_il_fail_prefill > 0 and len(fail_trajs_all) > max_il_fail_prefill:
                        fail_trajs_all = fail_trajs_all[:max_il_fail_prefill]

                    logging.info(f"[IL-CACHE] ✓ 成功加载 {len(success_trajs_all)} 条轨迹")
                    if fail_trajs_all:
                        logging.info(f"[IL-CACHE] 失败轨迹保留 {len(fail_trajs_all)} 条用于RL预填充")
                    logging.info(f"[IL-CACHE] 跳过在线采集，直接使用离线数据集")
            else:
                logging.warning(f"[IL-CACHE] 数据集格式不正确，回退到在线采集")
        except Exception as e:
            logging.warning(f"[IL-CACHE] 加载失败: {e}，回退到在线采集")
            success_trajs_all = []
            loaded_from_cache = False
    else:
        logging.info(f"[IL-CACHE] 未发现离线数据集 ({offline_dataset_path})，使用在线采集")

    if not loaded_from_cache:
        # SARL原版逻辑：只保留success和collision，丢弃timeout
        # 这样负样本（collision）不会被大量timeout稀释
        store_only = ('success', 'collision')
        consecutive_no_gain = 0

        if hasattr(policy, 'set_epsilon'):
            orig_eps = getattr(policy, 'epsilon', None)
            policy.set_epsilon(0.0)
        else:
            orig_eps = None

        _log_policy_tag('il', 'IL-ORCA-TEACHER')

        while total_sampled_episodes < max_total_episodes and len(success_trajs_all) < success_target:
            try:
                batch_collect_episodes = int(batch_collect_episodes)
            except Exception as e:
                logging.error(f"[IL-CONFIG] prefill_batch_episodes invalid at runtime: {batch_collect_episodes!r} ({e}), fallback=64")
                batch_collect_episodes = 64
            batch = min(batch_collect_episodes, max_total_episodes - total_sampled_episodes)
            explorer.run_k_episodes(batch, 'il', update_memory=True, show_tqdm=False,
                                    imitation_learning=True, force_joint_state_policy=True,
                                    store_on=store_only)

            new_trajs = list(getattr(explorer, '_last_trajectories', []) or [])
            if new_trajs:
                added_success = 0
                for traj_data, meta in new_trajs:
                    event = (meta.get('event', '') if isinstance(meta, dict) else '').lower()
                    is_success = ('success' in event) or ('reachgoal' in event) or ('reach_goal' in event)
                    if is_success:
                        success_trajs_all.append((traj_data, meta))
                        added_success += 1
                    else:
                        fail_trajs_all.append((traj_data, meta))
                consecutive_no_gain = 0 if added_success > 0 else (consecutive_no_gain + 1)
            else:
                consecutive_no_gain += 1

            total_sampled_episodes += batch
            success_ratio = len(success_trajs_all) / max(1, total_sampled_episodes)
            logging.info(
                f"[IL-COLLECT] episodes={total_sampled_episodes}/{max_total_episodes} | "
                f"success={len(success_trajs_all)}/{success_target} ({success_ratio:.2%})"
            )

            if consecutive_no_gain >= patience_batches:
                logging.warning("[IL-COLLECT] No new successes for several batches, continuing but check teacher settings")
                consecutive_no_gain = 0

        if orig_eps is not None and hasattr(policy, 'set_epsilon'):
            policy.set_epsilon(orig_eps)

    # [FIX] 限制IL收集数量（在线采集模式）
    max_il_prefill = train_cfg.max_il_prefill
    if len(success_trajs_all) > max_il_prefill:
        logging.info(f"[IL-COLLECT] 限制收集数量: {len(success_trajs_all)} → {max_il_prefill}")
        success_trajs_all = success_trajs_all[:max_il_prefill]

    max_il_fail_prefill = config.getint(
        'imitation_learning', 'max_il_fail_prefill',
        fallback=max(0, int(max_il_prefill * 0.5))
    )
    if max_il_fail_prefill > 0 and len(fail_trajs_all) > max_il_fail_prefill:
        logging.info(f"[IL-COLLECT] 限制失败轨迹: {len(fail_trajs_all)} → {max_il_fail_prefill}")
        fail_trajs_all = fail_trajs_all[:max_il_fail_prefill]

    collected_success_count = len(success_trajs_all)
    collected_attempts = total_sampled_episodes

    if collected_success_count == 0:
        logging.error("[IL-COLLECT] ORCA teacher failed to produce any successful episodes. Aborting IL stage.")
        success_ratio = 0.0
    else:
        success_ratio = collected_success_count / max(1, collected_attempts)
        logging.info(
            f"[IL-COLLECT] Final success set: {collected_success_count} trajectories from {collected_attempts} attempts"
            f" (success_ratio={success_ratio:.2%})"
        )
        if fail_trajs_all:
            logging.info(f"[IL-COLLECT] Fail trajectories kept for RL prefill: {len(fail_trajs_all)}")
        # IL value regression: use success + collision (exclude timeouts), SARL-style
        def _is_collision(meta):
            if not isinstance(meta, dict):
                return False
            if meta.get('flag_collision'):
                return True
            event = str(meta.get('event', '')).lower()
            outcome = str(meta.get('outcome', '')).lower()
            return ('collision' in event) or ('collision' in outcome)

        collision_trajs_all = [(traj, meta) for (traj, meta) in fail_trajs_all if _is_collision(meta)]
        il_trajs_for_bc = list(success_trajs_all) + list(collision_trajs_all)
        logging.info(f"[IL-DATA] BC dataset uses success+collision: success={len(success_trajs_all)}, collision={len(collision_trajs_all)}")

        explorer._last_trajectories = il_trajs_for_bc

        # DoubleQ: BC训练完成后，不需要将IL数据注入buffer
        # RL使用纯在线环境数据训练

        # 收集成功轨迹数量统计
        # 1. IL轨迹缺少log_prob/value（不是当前策略采集）
        # 2. 违背on-policy前提（PPO要求所有数据来自当前策略）
        # 3. IL数据仅用于BC预训练，不应混入RL训练
        # 保留success_trajs_all供_run_policy_il_pretrain使用即可
        logging.info(f"[IL-BUFFER] IL data reserved for BC pretraining only ({len(explorer._last_trajectories)} trajectories)")

    env.config.set('orca', 'neighbor_dist', str(original_neighbor_dist))
    env.config.set('orca', 'max_neighbors', str(original_max_neighbors))
    env.config.set('orca', 'time_horizon', str(original_time_horizon))
    env.config.set('orca', 'time_horizon_obst', str(original_time_horizon_obst))
    env.config.set('orca', 'safety_space', str(original_safety_space))
    env.configure(env.config)
    logging.info("[IL-TEACHER] Restored ORCA parameters for RL phase")

    if collected_success_count == 0:
        return {'success_rate': 0.0, 'transitions': 0, 'il_trajectories': [], 'il_fail_trajectories': fail_trajs_all}

    il_ckpt = config.get('train', 'il_ckpt', fallback=os.path.join(ARGS.outdir, 'il_policy.pth'))
    il_epochs = config.getint('train', 'il_epochs', fallback=20)
    il_batch_size = config.getint('train', 'il_batch_size', fallback=512)
    seq_len = config.getint('buffer', 'seq_len', fallback=12)
    gamma = config.getfloat('train', 'gamma', fallback=0.99)

    # ========== BC Checkpoint机制：避免重复训练 ==========
    bc_checkpoint_exists = os.path.exists(il_ckpt) and os.path.getsize(il_ckpt) > 1024  # 确保非空

    if bc_checkpoint_exists and config.getboolean('train', 'il_force_retrain', fallback=False):
        logging.info(f"[IL-BC] il_force_retrain=True → 忽略现有checkpoint，强制重新训练BC")
        bc_checkpoint_exists = False

    if bc_checkpoint_exists:
        # 存在checkpoint：直接加载，跳过120轮训练
        logging.info(f"[IL-BC] ✓ Found existing BC checkpoint: {il_ckpt}")
        logging.info(f"[IL-BC] Loading checkpoint to skip {il_epochs} epochs of BC training")

        try:
            checkpoint = safe_torch_load(il_ckpt, map_location=device)

            # 兼容多种checkpoint格式
            state_dict_to_load = None
            bc_meta = {}
            partial_load = False

            if isinstance(checkpoint, dict):
                if 'value' in checkpoint:
                    # 旧格式1: {'value': state_dict, 'meta': ...}
                    logging.info(f"[IL-BC] Detected old format (dict with 'value' key)")
                    state_dict_to_load = checkpoint['value']
                    bc_meta = checkpoint.get('meta', {})
                elif 'policy' in checkpoint:
                    # 旧格式2: {'policy': state_dict, ...}
                    logging.info(f"[IL-BC] Detected format with 'policy' key")
                    state_dict_to_load = checkpoint['policy']
                    bc_meta = checkpoint.get('meta', {})
                elif 'model_state_dict' in checkpoint:
                    # 标准PyTorch格式
                    state_dict_to_load = checkpoint['model_state_dict']
                    bc_meta = checkpoint.get('meta', {})
                else:
                    # 新格式: 直接是state_dict
                    state_dict_to_load = checkpoint
            else:
                # 直接是state_dict
                state_dict_to_load = checkpoint

            expected_objective = "discrete_action_ce" if direct_discrete else "value_regression"
            saved_objective = bc_meta.get('objective') if isinstance(bc_meta, dict) else None
            if direct_discrete and saved_objective != expected_objective:
                raise RuntimeError(
                    f"BC checkpoint objective mismatch: expected={expected_objective}, saved={saved_objective}"
                )

            if _occlusion_mode(config) != 'off':
                _assert_occlusion_resume(config, policy, checkpoint)

            # Occlusion checkpoints are resumable only under an exact schema.
            # Legacy weights use the explicit --legacy-warm-start path above.
            try:
                policy.load_state_dict(state_dict_to_load, strict=True)
                logging.info(f"[IL-BC] ✓ Loaded all parameters (strict mode)")
            except RuntimeError as e:
                if _occlusion_mode(config) != 'off':
                    raise
                if "Missing key" in str(e) or "Unexpected key" in str(e):
                    logging.warning(f"[IL-BC] ⚠ Model structure mismatch, trying partial load (strict=False)")
                    logging.warning(f"[IL-BC] Error details: {str(e)[:200]}...")
                    result = policy.load_state_dict(state_dict_to_load, strict=False)
                    if result.missing_keys:
                        logging.warning(f"[IL-BC] Missing {len(result.missing_keys)} parameters (will use random init)")
                    if result.unexpected_keys:
                        logging.warning(f"[IL-BC] Ignored {len(result.unexpected_keys)} unexpected parameters")
                    logging.info(f"[IL-BC] ⚠ Partial load completed (may need retraining)")
                    partial_load = True
                else:
                    raise

            # 打印checkpoint元信息
            if bc_meta:
                logging.info(f"[IL-BC] Checkpoint metadata:")
                logging.info(f"  - Training epochs: {bc_meta.get('epochs', 'N/A')}")
                logging.info(f"  - BC loss: {bc_meta.get('bc_loss', 'N/A')}")
                logging.info(f"  - Eval success rate: {bc_meta.get('eval_success_rate', 'N/A')}")
                logging.info(f"  - Timestamp: {bc_meta.get('timestamp', 'N/A')}")

            logging.info(f"[IL-BC] ✓ Successfully loaded BC weights from {il_ckpt}")

            # PPO不需要target_value（已删除）
            setattr(policy, "_il_warm_started", True)

            # 评估加载的模型（快速验证）
            logging.info("[IL-BC] Evaluating loaded BC model (100 episodes)...")
            policy.set_phase('eval')
            policy.eval()
            old_sarl_predict = getattr(policy, 'use_sarl_predict', False)
            if hasattr(policy, 'use_sarl_predict'):
                policy.use_sarl_predict = not direct_discrete
            try:
                explorer.run_k_episodes(100, 'val', update_memory=False, show_tqdm=False,
                                        imitation_learning=False, force_joint_state_policy=False)
            finally:
                if hasattr(policy, 'use_sarl_predict'):
                    policy.use_sarl_predict = old_sarl_predict

            def _mamba_event_token(x):
                try:
                    if x is None:
                        return ""
                    if isinstance(x, dict):
                        v = x.get("event") or x.get("Event") or x.get("status") or x.get("done_event")
                        if v is not None:
                            return str(v).replace(" ", "_").lower().strip()
                    for attr in ("event", "name", "value"):
                        if hasattr(x, attr):
                            try:
                                return str(getattr(x, attr)).replace(" ", "_").lower().strip()
                            except Exception:
                                pass
                    s = str(x)
                    if "." in s:
                        s = s.split(".")[-1]
                    return s.replace(" ", "_").lower().strip()
                except Exception:
                    return ""

            _MAMBA_SUCCESS_TOKENS = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
            mamba_succ = sum(1 for _, info in explorer._last_trajectories
                            if _mamba_event_token(info) in _MAMBA_SUCCESS_TOKENS)
            mamba_succ_rate = mamba_succ / max(1, len(explorer._last_trajectories))
            logging.info(f"[IL-BC] Loaded BC model success rate: {mamba_succ_rate:.1%} ({mamba_succ}/{len(explorer._last_trajectories)} episodes)")
            policy.set_phase('train')
            policy.train()

        except Exception as e:
            logging.error(f"[IL-BC] ✗ Failed to load checkpoint: {e}")
            if _occlusion_mode(config) != 'off':
                raise
            logging.info(f"[IL-BC] Will train from scratch due to loading error")
            bc_checkpoint_exists = False  # 加载失败，重新训练

    if not bc_checkpoint_exists:
        # 不存在checkpoint：训练120轮
        logging.info(f"[IL-BC] ✗ No BC checkpoint found at {il_ckpt}")
        logging.info(f"[IL-BC] Starting {il_epochs} epochs of BC training from scratch")

        _run_policy_il_pretrain(
            policy, device, seq_len, il_epochs, il_batch_size, il_ckpt, explorer, gamma,
            getattr(explorer, '_last_trajectories', None), config,
            direct_discrete=direct_discrete
        )

        # PPO不需要target_value（已删除）
        setattr(policy, "_il_warm_started", True)

        post_il_eval_episodes = config.getint(
            'imitation_learning', 'post_il_eval_episodes', fallback=100
        )
        logging.info(
            "[IL-BC] Evaluating newly trained BC model (%d episodes)...",
            post_il_eval_episodes,
        )
        policy.set_phase('eval')
        policy.eval()
        old_sarl_predict = getattr(policy, 'use_sarl_predict', False)
        if hasattr(policy, 'use_sarl_predict'):
            policy.use_sarl_predict = not direct_discrete
        try:
            explorer.run_k_episodes(post_il_eval_episodes, 'val', update_memory=False, show_tqdm=False,
                                    imitation_learning=False, force_joint_state_policy=False)
        finally:
            if hasattr(policy, 'use_sarl_predict'):
                policy.use_sarl_predict = old_sarl_predict

        def _mamba_event_token(x):
            try:
                if x is None:
                    return ""
                if isinstance(x, dict):
                    v = x.get("event") or x.get("Event") or x.get("status") or x.get("done_event")
                    if v is not None:
                        return str(v).replace(" ", "_").lower().strip()
                for attr in ("event", "name", "value"):
                    if hasattr(x, attr):
                        try:
                            return str(getattr(x, attr)).replace(" ", "_").lower().strip()
                        except Exception:
                            pass
                s = str(x)
                if "." in s:
                    s = s.split(".")[-1]
                return s.replace(" ", "_").lower().strip()
            except Exception:
                return ""

        _MAMBA_SUCCESS_TOKENS = {"reachgoal", "reach_goal", "reaching_goal", "goal_reached", "success"}
        mamba_succ = sum(1 for _, info in explorer._last_trajectories
                        if _mamba_event_token(info) in _MAMBA_SUCCESS_TOKENS)
        mamba_succ_rate = mamba_succ / max(1, len(explorer._last_trajectories))
        logging.info(f"[IL-BC] Newly trained BC model success rate: {mamba_succ_rate:.1%} ({mamba_succ}/{len(explorer._last_trajectories)} episodes)")
        policy.set_phase('train')
        policy.train()

    # 删除冗余日志：IL distillation DISABLED
    # 删除冗余日志：Continuous action replay chain

    il_succ_rate = success_ratio
    logging.info(
        f"[IL-ORCA-DATA] Teacher success stats: attempts={collected_attempts}, successes={collected_success_count}, ratio={il_succ_rate:.3f}"
    )

    if il_succ_rate < 0.2:
        logging.warning(f"[IL-WARNING] Low success rate: {il_succ_rate:.3f}, but continuing to RL")

    # 恢复原始环境配置（注意类型：time_limit是int，其他可能是float）
    env.config.set('env', 'time_limit', str(int(original_time_limit)))
    env.config.set('reward', 'success_radius', str(original_success_radius))
    env.config.set('orca', 'neighbor_dist', str(original_neighbor_dist))
    env.config.set('orca', 'max_neighbors', str(int(original_max_neighbors)))
    env.config.set('orca', 'time_horizon', str(original_time_horizon))
    env.config.set('orca', 'safety_space', str(original_safety_space))
    env.config.set('sim', 'human_num', str(int(original_human_num)))
    if env.config.has_option('robot', 'visible'):
        env.config.set('robot', 'visible', 'true' if original_robot_visible else 'false')
    if hasattr(env, 'robot'):
        try:
            env.robot.visible = bool(original_robot_visible)
        except Exception:
            pass
    env.configure(env.config)
    restored_success_radius = env.config.getfloat('reward', 'success_radius', fallback=original_success_radius)
    logging.info(
        f"[IL-PARAM-RESTORE] Parameters restored for RL: time_limit={env.time_limit}, success_radius={restored_success_radius}, human_num={original_human_num}"
    )

    # 返回IL轨迹数据用于注入RL buffer
    return {
        'success_rate': il_succ_rate,
        'transitions': len(replay_iql) if replay_iql else 0,
        'il_trajectories': success_trajs_all,     # BC: 成功轨迹
        'il_fail_trajectories': fail_trajs_all    # RL预填充: 失败轨迹
    }

# _prefill_buffer_with_bc已删除 - PPO不需要BC prefill

def _evaluate_policy_quick(explorer, policy, episodes: int, tag: str = 'POLICY-EVAL', phase: str = 'val'):
    if episodes <= 0:
        return
    logging.info(f"[{tag}] Running evaluation: episodes={episodes}")
    prev_phase = getattr(policy, '_phase', None)
    prev_eps = getattr(policy, 'epsilon', None)
    prev_mode = getattr(policy, 'training_mode', None)
    try:
        if hasattr(policy, 'set_phase'):
            policy.set_phase(phase)
        if hasattr(policy, 'set_epsilon') and prev_eps is not None:
            policy.set_epsilon(0.0)
        stats = explorer.run_k_episodes(episodes, phase, update_memory=False, show_tqdm=False,
                                        imitation_learning=False, force_joint_state_policy=False,
                                        return_stats=True)
        if isinstance(stats, dict):
            logging.info(f"[{tag}] success={stats.get('success_rate', 0.0):.1%} collision={stats.get('collision_rate', 0.0):.1%} timeout={stats.get('timeout_rate', 0.0):.1%} reward={stats.get('total_reward', 0.0):.3f}")
        else:
            logging.info(f"[{tag}] Evaluation completed")
    except Exception as e:
        logging.warning(f"[{tag}] Evaluation failed: {e}")
    finally:
        if prev_mode is not None and hasattr(policy, 'set_training_mode'):
            policy.set_training_mode(prev_mode)
        if prev_phase is not None and hasattr(policy, 'set_phase'):
            policy.set_phase(prev_phase)
        if prev_eps is not None and hasattr(policy, 'set_epsilon'):
            policy.set_epsilon(prev_eps)

def _convert_state_sequence_to_array(states):
    import numpy as _np
    from crowd_nav.contracts import ensure_btnd
    if isinstance(states, _np.ndarray):
        arr = _np.asarray(states, dtype=_np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr
    converted = []
    for s in states:
        if hasattr(s, 'to_array'):
            arr = s.to_array()
        else:
            arr = _np.asarray(s, dtype=_np.float32)
        arr = _np.asarray(arr, dtype=_np.float32)
        if arr.ndim == 1:
            if arr.shape[0] < 34:
                arr = _np.pad(arr, (0, 34 - arr.shape[0]), mode='constant')
            arr = arr[:34]
        converted.append(arr)
    arr = _np.stack(converted, axis=0)
    if arr.shape[-1] < 34:
        pad = _np.zeros((arr.shape[0], 34 - arr.shape[-1]), dtype=_np.float32)
        arr = _np.concatenate([arr, pad], axis=-1)
    return arr

def _convert_actions_to_array(actions):
    import numpy as _np
    if actions is None:
        return None
    arr = _np.asarray(actions, dtype=_np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _meta_from_event(meta, event_token, length):
    meta = dict(meta) if meta else {}
    success_tokens = {'success', 'reachgoal', 'reach_goal', 'reaching_goal', 'goal_reached'}
    is_success = any(tok for tok in success_tokens if tok in event_token)
    meta.setdefault('source', 'IL')
    meta['event'] = event_token
    meta['flag_success'] = int(is_success)
    meta['flag_timeout'] = int('timeout' in event_token)
    meta['flag_collision'] = int('collision' in event_token)
    meta['length'] = int(length)
    return meta

# _warmup_rl_buffer已删除 - PPO不需要buffer预热

def _run_rl_phase(cfg, env, policy, explorer, rl_buffer, stats, plotter, device, episodes, batch_size, save_every, ARGS, tb_writer, train_step_counter, start_episode=1, metrics_plotter=None, trajectory_file=None, resume_checkpoint=None):

    from crowd_nav.contracts import ensure_btnd, pick_last, joint34_to_tokens
    import copy, math, os, random, torch, numpy as np
    from crowd_nav.contracts import GRID
    direct_discrete = bool(getattr(ARGS, 'discrete_mamba', False))
    algo = 'discrete_mamba' if direct_discrete else 'sarl'
    policy_net = policy
    policy_net_base = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    temporal_backbone = getattr(
        policy_net_base,
        'temporal_backbone',
        cfg.get('mamba', 'temporal_backbone', fallback='mamba'),
    ).strip().lower()
    target_net = copy.deepcopy(policy_net_base).to(device)
    target_net.eval()
    rl_section = 'doubleq' if direct_discrete else 'sarl'
    rl_lr = cfg.getfloat(
        rl_section,
        'learning_rate' if direct_discrete else 'value_lr',
        fallback=cfg.getfloat(
            'train',
            'rl_learning_rate',
            fallback=cfg.getfloat('train', 'learning_rate', fallback=1e-4),
        ),
    )
    backbone_lr = cfg.getfloat(
        rl_section,
        'backbone_learning_rate',
        fallback=rl_lr * 0.1,
    )
    freeze_backbone_episodes = cfg.getint(
        rl_section,
        'freeze_backbone_episodes',
        fallback=1000,
    ) if direct_discrete else 0

    if direct_discrete:
        q_head_params = list(policy_net_base.q_head.parameters())
        q_head_param_ids = {id(param) for param in q_head_params}
        backbone_params = [
            param for param in policy_net_base.parameters()
            if id(param) not in q_head_param_ids
        ]
        backbone_frozen = start_episode <= freeze_backbone_episodes
        for param in backbone_params:
            param.requires_grad_(not backbone_frozen)
        rl_optimizer = torch.optim.AdamW(
            [
                {'params': q_head_params, 'lr': rl_lr, 'name': 'q_head'},
                {
                    'params': backbone_params,
                    'lr': 0.0 if backbone_frozen else backbone_lr,
                    'name': 'backbone',
                },
            ],
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
    else:
        backbone_params = []
        backbone_frozen = False
        rl_optimizer = torch.optim.AdamW(
            policy_net.parameters(),
            lr=rl_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

    bc_teacher = None
    if direct_discrete:
        bc_teacher = copy.deepcopy(policy_net_base).to(device)
        il_teacher_path = os.path.join(os.path.abspath(ARGS.outdir), 'il_policy.pth')
        if os.path.exists(il_teacher_path):
            teacher_checkpoint = safe_torch_load(il_teacher_path, map_location=device)
            teacher_state = teacher_checkpoint
            if isinstance(teacher_checkpoint, dict):
                teacher_state = teacher_checkpoint.get(
                    'model_state_dict',
                    teacher_checkpoint.get(
                        'value',
                        teacher_checkpoint.get('policy', teacher_checkpoint),
                    ),
                )
            if any(key.startswith('_orig_mod.') for key in teacher_state):
                teacher_state = {
                    key.replace('_orig_mod.', '', 1): value
                    for key, value in teacher_state.items()
                }
            bc_teacher.load_state_dict(teacher_state, strict=True)
            logging.info(f"[BC-TEACHER] Loaded frozen teacher from {il_teacher_path}")
        else:
            logging.warning("[BC-TEACHER] il_policy.pth not found; using the current policy snapshot")
        bc_teacher.eval()
        for param in bc_teacher.parameters():
            param.requires_grad_(False)

    target_value_net = target_net if not direct_discrete else None
    target_q_net = target_net if direct_discrete else None
    optim_value = rl_optimizer if not direct_discrete else None
    optim_q = rl_optimizer if direct_discrete else None

    policy_mode_owner = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    if direct_discrete:
        policy_mode_owner.use_sarl_predict = False
        logging.info(
            f"[ALGO] Direct-Q ({temporal_backbone.upper()}): constrained Double-DQN, "
            f"q_head_lr={rl_lr}, backbone_lr={backbone_lr}, "
            f"freeze_backbone_episodes={freeze_backbone_episodes}"
        )
    else:
        policy_mode_owner.use_sarl_predict = True
        logging.info(f"[ALGO] Value-lookahead ({temporal_backbone.upper()}): value regression + one-step lookahead, lr={rl_lr}")

    resume_stage = str(resume_checkpoint.get('stage', '')).lower() if resume_checkpoint is not None else ''
    if resume_checkpoint is not None and resume_stage.startswith('rl_training'):
        logging.info(f"[{algo.upper()}-RESUME] Loading target network and optimizer from checkpoint...")
        try:
            target_key = 'target_q_net_state' if direct_discrete else 'target_value_net_state'
            optim_key = 'optim_q_state' if direct_discrete else 'optim_value_state'
            if target_key in resume_checkpoint:
                target_net.load_state_dict(resume_checkpoint[target_key], strict=False)
                logging.info(f"[{algo.upper()}-RESUME] Target network loaded")
            if optim_key in resume_checkpoint:
                rl_optimizer.load_state_dict(resume_checkpoint[optim_key])
                logging.info(f"[{algo.upper()}-RESUME] Optimizer loaded")
        except Exception as e:
            logging.warning(f"[{algo.upper()}-RESUME] Failed to load some components: {e}")
    elif resume_checkpoint is not None:
        logging.warning(f"[{algo.upper()}-RESUME] Checkpoint stage={resume_stage!r} is not an RL checkpoint; only policy/start episode were restored")

    # ============================================================
    # Replay warmup机制：先采集数据，再开始训练
    # ============================================================
    if cfg.has_option('train', 'replay_warmup'):
        replay_warmup = cfg.getint('train', 'replay_warmup', fallback=3000)
    else:
        replay_warmup = cfg.getint('buffer', 'replay_warmup', fallback=3000)
    logging.info(f"[{algo.upper()}-WARMUP] Replay warmup threshold: {replay_warmup} transitions")

    # n-step / PER config（🔥 SARL强制n_step=1, use_per=False）
    n_step = 1
    use_per = False
    logging.info(f"[{algo.upper()}] n_step=1, use_per=False")

    best_succ = float(resume_checkpoint.get('best_success', 0.0)) if resume_checkpoint else 0.0
    buffer_store_ok = 0
    buffer_store_skip = 0
    buffer_store_steps = 0
    updates_per_ep = cfg.getint('train', 'updates_per_ep', fallback=20)
    eval_every = cfg.getint(
        rl_section,
        'eval_every',
        fallback=cfg.getint('train', 'eval_every', fallback=100),
    )
    eval_episodes = cfg.getint(
        rl_section,
        'eval_episodes',
        fallback=cfg.getint('train', 'eval_episodes', fallback=10),
    )
    td_weight = cfg.getfloat(rl_section, 'td_weight', fallback=0.25) if direct_discrete else 1.0
    bc_weight_start = cfg.getfloat(rl_section, 'bc_weight_start', fallback=1.0) if direct_discrete else 0.0
    bc_weight_end = cfg.getfloat(rl_section, 'bc_weight_end', fallback=0.5) if direct_discrete else 0.0
    bc_weight_decay_episodes = cfg.getint(
        rl_section,
        'bc_weight_decay_episodes',
        fallback=3000,
    ) if direct_discrete else 1
    bc_temperature = cfg.getfloat(rl_section, 'bc_temperature', fallback=1.0) if direct_discrete else 1.0
    q_target_min = cfg.getfloat(
        rl_section,
        'q_target_min',
        fallback=cfg.getfloat('reward', 'collision_penalty', fallback=-0.5),
    ) if direct_discrete else -0.5
    q_target_max = cfg.getfloat(
        rl_section,
        'q_target_max',
        fallback=cfg.getfloat('reward', 'success_reward', fallback=1.0),
    ) if direct_discrete else 1.0
    collapse_ratio = cfg.getfloat(rl_section, 'collapse_ratio', fallback=0.60) if direct_discrete else 0.0
    collapse_patience = cfg.getint(rl_section, 'collapse_patience', fallback=2) if direct_discrete else 0
    early_stop_patience = cfg.getint(rl_section, 'early_stop_patience', fallback=6) if direct_discrete else 0
    min_evals_before_stop = cfg.getint(rl_section, 'min_evals_before_stop', fallback=3) if direct_discrete else 0
    safe_stop_disabled = bool(getattr(ARGS, 'disable_safe_stop', False)) and direct_discrete
    checkpoint_interval = int(save_every)
    logging.info(f"[CHECKPOINT] interval={checkpoint_interval} episodes")
    if direct_discrete:
        logging.info(
            f"[CONSTRAINED-DQN] td_weight={td_weight}, "
            f"bc_weight={bc_weight_start}->{bc_weight_end} over {bc_weight_decay_episodes}ep, "
            f"temperature={bc_temperature}, q_target=[{q_target_min}, {q_target_max}], "
            f"eval_every={eval_every}"
        )
        if safe_stop_disabled:
            logging.warning("[SAFE-IMPROVEMENT] Safe early stopping disabled; diagnostics will be logged only")
    # 🔥 修复：batch_size根据algo读取
    batch_size = cfg.getint('sarl', 'batch_size',
                 fallback=cfg.getint('train', 'batch_size', fallback=batch_size))

    sot_monitor_interval = 100
    timeout_alert_episodes = []

    def _build_rl_checkpoint(ep_number, stats_history):
        payload = {
            'checkpoint_version': 3,
            'episode': int(ep_number),
            'algo': algo,
            'policy_state': policy.state_dict(),
            'stats_history': stats_history,
            'config': {
                'batch_size': batch_size,
                'lr': rl_optimizer.param_groups[0]['lr'],
                'gamma': cfg.getfloat('train', 'gamma', fallback=0.99),
                'td_weight': td_weight,
                'bc_weight_start': bc_weight_start,
                'bc_weight_end': bc_weight_end,
                'temporal_backbone': temporal_backbone,
            },
            'stage': f'rl_training_{algo}',
            'best_success': float(best_succ),
            'meta': {
                'save_time': time.time(),
                'episode_count': int(ep_number),
                'model_type': f'{algo}_checkpoint',
                'temporal_backbone': temporal_backbone,
                'occlusion': _occlusion_meta(cfg, policy_net_base),
            },
        }
        if direct_discrete:
            payload['target_q_net_state'] = target_q_net.state_dict()
            payload['optim_q_state'] = optim_q.state_dict()
        else:
            payload['target_value_net_state'] = target_value_net.state_dict()
            payload['optim_value_state'] = optim_value.state_dict()
        return payload

    vectorize_enable = cfg.getboolean('vectorize', 'enable', fallback=False)
    num_workers = cfg.getint('vectorize', 'num_workers', fallback=8)
    episodes_per_worker = cfg.getint('vectorize', 'episodes_per_worker', fallback=2)
    broadcast_interval = cfg.getint('vectorize', 'broadcast_interval', fallback=1)

    vectorized_sampler = None
    if vectorize_enable:
        logging.info(f"[VECTORIZE] Enabling parallel sampling: {num_workers} workers × {episodes_per_worker} episodes/worker")
        from crowd_nav.utils.vectorized_sampler import VectorizedSampler

        import os
        if ARGS.config and os.path.exists(ARGS.config):
            config_path = os.path.abspath(ARGS.config)
        elif os.path.exists('configs/env.config'):
            config_path = os.path.abspath('configs/env.config')
        elif os.path.exists('crowd_nav/configs/env.config'):
            config_path = os.path.abspath('crowd_nav/configs/env.config')
        else:
            config_path = os.path.abspath(ARGS.config) if ARGS.config else 'configs/env.config'
            logging.warning(f"[VECTORIZE] Config file not found, trying {config_path}")

        policy_name = ARGS.policy or cfg.get('policy', 'key', fallback='mamba_rl')

        logging.info(f"[VECTORIZE] Config path: {config_path}")
        logging.info(f"[VECTORIZE] Policy name: {policy_name}")

        worker_log_dir = os.path.abspath(ARGS.outdir)

        vectorized_sampler = VectorizedSampler(
            num_workers=num_workers,
            episodes_per_worker=episodes_per_worker,
            config_path=config_path,
            policy_name=policy_name,
            device=device,
            log_dir=worker_log_dir,
        )
        vectorized_sampler.start()
        logging.info(f"[VECTORIZE] Parallel sampler started successfully")

        # 立即广播BC权重到workers，避免首批采样使用随机权重
        if hasattr(vectorized_sampler, 'broadcast_params'):
            bc_state_dict = policy.state_dict()
            num_params = sum(p.numel() for p in policy.parameters())
            logging.info(f"[VECTORIZE] Broadcasting BC policy to workers: {num_params:,} parameters, {len(bc_state_dict)} tensors")
            vectorized_sampler.broadcast_params(bc_state_dict)
            logging.info("[VECTORIZE] ✓ Initial BC policy weights broadcast to all workers")
    else:
        logging.info("[VECTORIZE] Parallel sampling disabled, using serial explorer")

    _first_action_logged = [False]

    # DoubleQ: Q网络通过online RL直接学习
    policy.train()

    # 诊断日志
    if vectorized_sampler is not None:
        logging.info("[DIAGNOSTIC] ✓ Using VECTORIZED sampling (parallel workers)")
        logging.info(f"[DIAGNOSTIC] Workers: {num_workers}, Episodes/worker: {episodes_per_worker}")
    else:
        logging.info("[DIAGNOSTIC] ⚠ Using SERIAL sampling (no parallelization)")

    # Establish a deterministic BC baseline before any online update.
    baseline_eval_episodes = (
        eval_episodes if direct_discrete and start_episode == 1
        else cfg.getint('train', 'pre_rl_eval_episodes', fallback=5)
    )
    logging.info(f"[DIAGNOSTIC] Pre-RL policy evaluation ({baseline_eval_episodes} episodes)...")
    previous_epsilon = float(getattr(policy, 'epsilon', 0.0))
    policy.set_phase('eval')
    policy.eval()
    if hasattr(policy, 'set_epsilon'):
        policy.set_epsilon(0.0)
    test_stats = explorer.run_k_episodes(
        baseline_eval_episodes,
        'val',
        update_memory=False,
        show_tqdm=False,
        imitation_learning=False,
        force_joint_state_policy=False,
    )
    logging.info(f"[DIAGNOSTIC] Pre-RL test performance: s/c/t={test_stats[0]:.1%}/{test_stats[1]:.1%}/{test_stats[2]:.1%}")
    policy.set_phase('train')
    policy.train()
    if hasattr(policy, 'set_epsilon'):
        policy.set_epsilon(previous_epsilon)

    baseline_succ = float(test_stats[0]) if direct_discrete else 0.0
    eval_count = 0
    evals_without_improvement = 0
    collapse_eval_count = 0
    stop_training_early = False
    best_path = os.path.join(ARGS.outdir, 'best_model.pth')
    if direct_discrete and start_episode == 1:
        best_succ = baseline_succ
        baseline_tmp_path = best_path + ".tmp"
        try:
            baseline_history = {
                'hist_s': [],
                'hist_c': [],
                'hist_t': [],
                'hist_r': [],
            }
            torch.save(_build_rl_checkpoint(0, baseline_history), baseline_tmp_path)
            os.replace(baseline_tmp_path, best_path)
            logging.info(f"[BEST-SAVE] BC baseline success={best_succ:.3f} -> {best_path}")
        finally:
            if os.path.exists(baseline_tmp_path):
                try:
                    os.remove(baseline_tmp_path)
                except OSError:
                    pass
    elif direct_discrete:
        baseline_succ = best_succ

    # algo已在n_step/PER配置前读取，此处仅输出确认
    logging.info(f"[TRAIN] Algorithm: {algo}")

    for ep in range(start_episode, int(episodes)+1):
        policy.train()
        if direct_discrete:
            policy_mode_owner.use_sarl_predict = False
        else:
            policy_mode_owner.use_sarl_predict = True

        if direct_discrete and backbone_frozen and ep > freeze_backbone_episodes:
            for param in backbone_params:
                param.requires_grad_(True)
            for group in rl_optimizer.param_groups:
                if group.get('name') == 'backbone':
                    group['lr'] = backbone_lr
            backbone_frozen = False
            logging.info(
                f"[BACKBONE] Unfrozen at episode {ep}; learning_rate={backbone_lr}"
            )

        standard_human_num = cfg.getint('sim', 'human_num', fallback=5)
        standard_circle_radius = cfg.getfloat('sim', 'circle_radius', fallback=4.0)
        standard_success_radius = cfg.getfloat('robot', 'success_radius', fallback=0.25)

        # 固定标准场景（5人circle），禁用所有混合训练
        current_human_num = standard_human_num
        current_sim_type = 'circle_crossing'
        current_circle_radius = standard_circle_radius
        current_square_width = None

        # 应用场景配置到环境
        if hasattr(env, '_raw_config'):
            if hasattr(env._raw_config, 'set'):
                env._raw_config.set('sim', 'human_num', str(current_human_num))
                if current_circle_radius is not None:
                    env._raw_config.set('sim', 'circle_radius', str(current_circle_radius))
                if current_square_width is not None:
                    env._raw_config.set('sim', 'square_width', str(current_square_width))
                env._raw_config.set('sim', 'train_sim', current_sim_type)
            elif isinstance(env._raw_config, dict):
                if 'sim' not in env._raw_config:
                    env._raw_config['sim'] = {}
                env._raw_config['sim']['human_num'] = current_human_num
                if current_circle_radius is not None:
                    env._raw_config['sim']['circle_radius'] = current_circle_radius
                if current_square_width is not None:
                    env._raw_config['sim']['square_width'] = current_square_width
                env._raw_config['sim']['train_sim'] = current_sim_type

        # 直接设置环境属性（更可靠）
        if hasattr(env, 'human_num'):
            env.human_num = current_human_num
        if hasattr(env, 'train_sim'):
            env.train_sim = current_sim_type
        if current_circle_radius is not None and hasattr(env, 'circle_radius'):
            env.circle_radius = current_circle_radius
        if current_square_width is not None and hasattr(env, 'square_width'):
            env.square_width = current_square_width

        if hasattr(env, 'rwd'):
            env.rwd.success_radius = standard_success_radius

        if ep == 1:
            logging.info(f"[P1-RL] Base config: human_num={standard_human_num}, circle_radius={standard_circle_radius}, success_radius={standard_success_radius}")

        env._current_episode = ep
        epsilon = float(getattr(policy, 'epsilon', 0.0))

        # [FIX-REWARD-ULTIMATE] Block REMOVED.
        # Rewards are now strictly controlled by env.config loaded at startup.
        # updates_per_ep is strictly controlled by train.config.

        if ep % 50 == 0:
             logging.info(f"[CONFIG-CHECK] ep={ep} updates_per_ep={updates_per_ep} (from config)")

        if vectorized_sampler is not None:

            episodes_batch = vectorized_sampler.collect_episodes(
                episode_start=ep,
                phase='train',
                # PPO: 不需要epsilon参数（使用entropy探索）
                timeout=300.0
            )

            succ_count = coll_count = timeout_count = 0
            total_reward = 0.0
            nav_time_sum = 0.0
            time_taken_list = []
            discomfort_freq_list = []
            discomfort_dist_list = []
            path_len_list = []
            smooth_list = []
            speed_list = []

            for ep_data in episodes_batch:
                states = ep_data['states']
                actions = ep_data['actions']
                rewards = ep_data['rewards']
                info = ep_data['info']

                if len(states) > 0:

                    states_arr = np.array(states, dtype=np.float32)
                    rewards_arr = np.array(rewards, dtype=np.float32)

                    episode_length = len(states)
                    dones = np.zeros(episode_length, dtype=bool)

                    event_str = (_event_token(info) or '').lower()
                    timeout_flag = ('timeout' in event_str)
                    bootstrap_on_timeout = cfg.getboolean('train', 'bootstrap_on_timeout', fallback=False)
                    dones[-1] = _done_for_bootstrap(event_str, bootstrap_on_timeout)

                    if len(actions) > 0:
                        actions_continuous = np.asarray(actions, dtype=np.float32)
                        if actions_continuous.ndim == 1:
                            actions_continuous = actions_continuous.reshape(1, -1)
                    else:
                        actions_continuous = None

                    success_flag = (event_str in _SUCCESS_TOKENS) or ('success' in event_str) or ('reachgoal' in event_str) or ('reaching_goal' in event_str)
                    collision_flag = ('collision' in event_str)
                    # 分层探索：使用episode_source标签（来自vectorized_sampler）
                    actual_source = ep_data.get('source', 'RL')  # 从ep_data中提取source字段
                    meta = {
                        'source': actual_source,
                        'event': event_str,
                        'is_success': success_flag,
                        'is_timeout': timeout_flag,
                        'is_terminal': dones[-1],
                        'length': episode_length,
                    }

                    # 存入RL buffer（在线数据）
                    # 只使用policy输出的离散动作索引，禁止由连续动作反推
                    action_indices = ep_data.get('action_indices', None)
                    if action_indices is None and isinstance(info, dict):
                        action_indices = info.get('action_indices')
                    if action_indices is None or any(a is None for a in action_indices):
                        buffer_store_skip += 1
                        logging.warning("[REPLAY] Missing action_indices in vectorized episode; skip storing.")
                    else:
                        rl_buffer.store_episode({
                            'states': states_arr,
                            'actions_continuous': actions_continuous,  # 保留用于兼容性
                            'action_indices': np.array(action_indices, dtype=np.int64),
                            'rewards': rewards_arr,
                            'dones': dones
                        })
                        buffer_store_ok += 1
                        buffer_store_steps += int(episode_length)

                event_str = (_event_token(info) or '').lower()
                if (event_str in _SUCCESS_TOKENS) or ('success' in event_str) or ('reachgoal' in event_str) or ('reaching_goal' in event_str):
                    succ_count += 1
                elif 'collision' in event_str:
                    coll_count += 1
                elif 'timeout' in event_str:
                    timeout_count += 1

                total_reward += sum(rewards)
                nav_time_sum += len(states) * getattr(env, 'time_step', 0.25)

                time_taken_val = _info_get(info, 'time_taken')
                if time_taken_val is not None:
                    time_taken_list.append(time_taken_val)
                discomfort_freq_val = _info_get(info, 'discomfort_freq')
                if discomfort_freq_val is not None:
                    discomfort_freq_list.append(discomfort_freq_val)
                discomfort_dist_val = _info_get(info, 'discomfort_dist')
                if discomfort_dist_val is not None:
                    discomfort_dist_list.append(discomfort_dist_val)
                path_len_val = _info_get(info, 'path_length')
                if path_len_val is not None:
                    path_len_list.append(path_len_val)
                smooth_val = _info_get(info, 'smoothness')
                if smooth_val is not None:
                    smooth_list.append(smooth_val)
                speed_val = _info_get(info, 'speed')
                if speed_val is not None:
                    speed_list.append(speed_val)

                # Trajectory logging: 记录轨迹到trajectory.log
                if trajectory_file is not None and 'trajectory_positions' in info:
                    try:
                        traj_positions = info['trajectory_positions']
                        start_pos = info.get('start_pos', (0.0, 0.0))
                        goal_pos = info.get('goal_pos', (0.0, 0.0))
                        event = info.get('event', 'unknown').upper()
                        ep_id = ep_data.get('episode', ep)

                        # 写入header: [EP=X] START=(x,y) GOAL=(x,y) RESULT=status STEPS=n
                        trajectory_file.write(f"[EP={ep_id}] START={start_pos} GOAL={goal_pos} RESULT={event} STEPS={len(traj_positions)}\n")

                        # 写入trajectory: TRAJECTORY: (x1, y1) (x2, y2) ...
                        trajectory_str = ' '.join(f"{pos}" for pos in traj_positions)
                        trajectory_file.write(f"TRAJECTORY: {trajectory_str}\n\n")
                        trajectory_file.flush()
                    except Exception as e:
                        logging.warning(f"[TRAJECTORY-LOG] Failed to write trajectory: {e}")

            num_episodes = len(episodes_batch)
            succ = succ_count / max(1, num_episodes)
            coll = coll_count / max(1, num_episodes)
            timeout = timeout_count / max(1, num_episodes)
            ep_reward = total_reward / max(1, num_episodes)
            nav_t = nav_time_sum / max(1, num_episodes)
            time_taken = np.mean(time_taken_list) if time_taken_list else 0.0
            discomfort_freq = np.mean(discomfort_freq_list) if discomfort_freq_list else 0.0
            discomfort_dist = np.mean(discomfort_dist_list) if discomfort_dist_list else 0.0
            path_len = np.mean(path_len_list) if path_len_list else 0.0
            smooth = np.mean(smooth_list) if smooth_list else 0.0
            speed = np.mean(speed_list) if speed_list else 0.0

            if ep % broadcast_interval == 0:
                vectorized_sampler.broadcast_params(policy.state_dict())
                if ep % 10 == 0:  # More frequent logging for debugging
                    logging.info(f"[VECTORIZE] Model weights synced to workers (ep={ep})")

        else:

            eps_section = 'doubleq' if direct_discrete and cfg.has_section('doubleq') else 'sarl'
            eps_start = cfg.getfloat(eps_section, 'epsilon_start', fallback=0.30)
            eps_end = cfg.getfloat(eps_section, 'epsilon_end', fallback=0.05)
            eps_decay = cfg.getint(eps_section, 'epsilon_decay_episodes', fallback=3000)

            epsilon = linear_schedule(eps_start, eps_end, eps_decay, ep)
            if hasattr(policy, 'set_epsilon'):
                policy.set_epsilon(epsilon)

            if ep % 100 == 0:
                logging.info(f"[{algo.upper()}-EPSILON] ep={ep} epsilon={epsilon:.4f} (start={eps_start}, end={eps_end}, decay={eps_decay})")

            # 【精简】删除 expert_policy 和 student_prob 参数
            succ, coll, timeout, ep_reward, nav_t, time_taken, discomfort_freq, discomfort_dist, path_len, smooth, speed = explorer.run_k_episodes(
                k=1, phase='train', update_memory=False, show_tqdm=False,
                imitation_learning=False, force_joint_state_policy=False
            )

            # 手动存储数据到buffer（确保数据不丢失）
            # 即使Explorer.update_memory=False，我们也要确保RL数据被存入buffer
            if hasattr(explorer, '_last_trajectories') and explorer._last_trajectories:
                for traj_data, info in explorer._last_trajectories:
                    try:
                        # 解析轨迹数据结构
                        states, actions, rewards = traj_data

                        # 所有轨迹都进buffer（包括timeout）
                        # timeout轨迹让网络学到"慢=坏"，防止策略过于保守
                        event_str = (_event_token(info) or '').lower()
                        is_timeout = 'timeout' in event_str

                        # 构造dones数组：终止状态done=True
                        dones = np.zeros(len(rewards), dtype=bool)
                        if len(rewards) > 0:
                            dones[-1] = True  # 所有终止状态都是done

                        action_indices = info.get('action_indices') if isinstance(info, dict) else None
                        if (
                            action_indices is None
                            or len(action_indices) != len(rewards)
                            or any(a is None for a in action_indices)
                        ):
                            buffer_store_skip += 1
                            message = (f"[STORE] Missing discrete action indices in episode {ep}; "
                                       "trajectory skipped")
                            if _occlusion_mode(cfg) != 'off':
                                raise RuntimeError(message)
                            logging.warning(message)
                            continue
                        if action_indices is None:
                            action_indices = np.zeros(len(rewards), dtype=np.int64)

                        rl_buffer.push_episode({
                            'states': states,
                            'actions': actions,
                            'action_indices': np.asarray(action_indices, dtype=np.int64),
                            'rewards': rewards,
                            'dones': dones
                        })
                        buffer_store_ok += 1
                        buffer_store_steps += int(len(rewards))
                    except Exception as e:
                        logging.warning(f"[STORE] Failed to store trajectory in episode {ep}: {e}")
                        if _occlusion_mode(cfg) != 'off':
                            raise

                # 清空explorer缓存（访问内部属性）
                explorer._last_trajectories = []

        if ep == 1 and not _first_action_logged[0]:

            if hasattr(policy, '_last_continuous_action'):
                vx, vy = policy._last_continuous_action
                speed = math.sqrt(vx*vx + vy*vy)
                heading_deg = math.degrees(math.atan2(vy, vx))
                logging.info(f"FIRST_ACTION (Continuous): speed={speed:.3f}, heading={heading_deg:.1f}°, (vx={vx:+.3f}, vy={vy:+.3f})")
                _first_action_logged[0] = True
            else:
                logging.info("FIRST_ACTION: Policy doesn't expose continuous action info, skipping debug")
                _first_action_logged[0] = True

        # 【渐进更新门槛】根据训练表现动态调整updates
        # 当前简化版：固定使用warm_updates（6次），等稳定后再提升
        # TODO: 实现完整的渐进式提升逻辑（6→8→10）

        # 计算最近50集的滑动窗口统计
        window_size = min(50, len(stats.hist_s))
        if window_size > 0:
            recent_succ = sum(stats.hist_s[-window_size:]) / window_size
            recent_tout = sum(stats.hist_t[-window_size:]) / window_size
        else:
            recent_succ, recent_tout = 0.0, 0.0

        # Warm-update机制已删除，使用固定更新次数
        current_updates = updates_per_ep

        total_loss = 0.0
        total_policy_loss = 0.0
        total_v_loss = 0.0
        total_q_loss = 0.0
        total_bc_loss = 0.0
        total_aux_loss = 0.0
        # 统计累加器
        total_adv_mean = 0.0
        total_adv_std = 0.0
        total_weight = 0.0
        grad_norm_sum = 0.0
        updates_done = 0
        # PER beta schedule (if enabled)
        if use_per:
            per_beta = linear_schedule(per_beta_start, per_beta_end, per_beta_episodes, ep)
            rl_buffer.set_per_beta(per_beta)
            if ep % 100 == 0:
                logging.info(f"[PER] ep={ep} beta={per_beta:.3f}")

        # ========== Value or direct discrete Q update ==========
        min_update_size = max(batch_size, replay_warmup if direct_discrete else batch_size)
        if len(rl_buffer) < min_update_size:
            if ep == start_episode or ep % 100 == 0:
                logging.info(f"[{algo.upper()}-WARMUP] RL buffer size: {len(rl_buffer)}, waiting for {min_update_size}...")
            updates_done = 0
        else:
            n_updates = current_updates
            for _ in range(n_updates):
                if len(rl_buffer) < batch_size:
                    break

                update_gamma = cfg.getfloat('train', 'gamma', fallback=0.99)
                if direct_discrete:
                    bc_progress = min(
                        1.0,
                        max(0.0, float(ep - 1) / max(1, bc_weight_decay_episodes)),
                    )
                    current_bc_weight = (
                        bc_weight_start
                        + (bc_weight_end - bc_weight_start) * bc_progress
                    )
                    update_metrics = discrete_q_update(
                        policy=policy_net,
                        target_policy=target_q_net,
                        memory=rl_buffer,
                        optimizer=optim_q,
                        batch_size=batch_size,
                        gamma=update_gamma,
                        device=device,
                        teacher_policy=bc_teacher,
                        td_weight=td_weight,
                        bc_weight=current_bc_weight,
                        bc_temperature=bc_temperature,
                        q_target_min=q_target_min,
                        q_target_max=q_target_max,
                    )
                    update_loss = update_metrics['loss']
                    total_q_loss += update_metrics['q_loss']
                    total_bc_loss += update_metrics['bc_loss']
                else:
                    update_loss = sarl_style_update(
                        policy=policy_net,
                        target_policy=target_value_net,
                        memory=rl_buffer,
                        optimizer=optim_value,
                        batch_size=batch_size,
                        gamma=update_gamma,
                        device=device
                    )
                    total_v_loss += update_loss

                total_loss += update_loss
                updates_done += 1

                tau = 0.005
                with torch.no_grad():
                    for param, target_param in zip(policy_net.parameters(), target_net.parameters()):
                        target_param.data.mul_(1.0 - tau).add_(tau * param.data)

        # Buffer信息
        if ep % 100 == 0:
            logging.info(f"[{algo.upper()}-REPLAY] ep={ep} buffer_size={len(rl_buffer)} updates={updates_done}")

        # 计算平均loss
        avg_v_loss = total_v_loss / max(1, updates_done) if updates_done > 0 else 0.0
        avg_q_loss = total_q_loss / max(1, updates_done) if updates_done > 0 else 0.0
        avg_bc_loss = total_bc_loss / max(1, updates_done) if updates_done > 0 else 0.0
        avg_policy_loss = total_policy_loss / max(1, updates_done) if updates_done > 0 else 0.0
        avg_aux_loss = 0.0
        avg_adv_mean = 0.0
        avg_adv_std = 0.0
        avg_weight = 0.0

        # 每25 episodes输出loss到日志和TensorBoard
        if updates_done > 0 and ep % 25 == 0:
            if direct_discrete:
                logging.info(
                    f"[DISCRETE_MAMBA-LOSS] ep={ep} q_loss={avg_q_loss:.4f} "
                    f"bc_loss={avg_bc_loss:.4f} bc_weight={current_bc_weight:.3f} "
                    f"epsilon={epsilon:.4f}"
                )
                tb_writer.add_scalar("discrete_mamba/q_loss", avg_q_loss, ep)
                tb_writer.add_scalar("discrete_mamba/bc_loss", avg_bc_loss, ep)
                tb_writer.add_scalar("discrete_mamba/bc_weight", current_bc_weight, ep)
                tb_writer.add_scalar("discrete_mamba/epsilon", epsilon, ep)
            else:
                avg_value_loss = total_v_loss / max(1, updates_done)
                logging.info(f"[SARL-LOSS] ep={ep} value_loss={avg_value_loss:.4f} epsilon={epsilon:.4f}")
                tb_writer.add_scalar("sarl/value_loss", avg_value_loss, ep)
                tb_writer.add_scalar("sarl/epsilon", epsilon, ep)
            if ep % 100 == 0:
                logging.info(f"[{algo.upper()}-STATS] ep={ep} buffer_size={len(rl_buffer)}")

            if avg_aux_loss > 0:
                tb_writer.add_scalar(f"{algo}/aux_loss", avg_aux_loss, ep)

        metrics = {
            'loss': total_loss / max(1, updates_done),
            'policy_loss': avg_policy_loss,
            'value_loss': avg_v_loss,
            'q_loss': avg_q_loss,
            'bc_loss': avg_bc_loss,
            'grad_norm': grad_norm_sum / max(1, updates_done),
            'updates': updates_done,
            'architecture': f'{algo}_mamba'  # 🔥 动态设置architecture
        }

        train_step_counter['episode'] = ep

        # 🔥 根据算法类型选择loss metric用于绘图
        v_loss_metric = (avg_q_loss if direct_discrete else avg_v_loss) if updates_done > 0 else None
        p_loss_metric = avg_policy_loss if updates_done > 0 else None
        stats.update(succ, coll, timeout, ep_reward, episode=ep,
                    time_taken=time_taken, discomfort_freq=discomfort_freq, discomfort_dist=discomfort_dist,
                    v_loss=v_loss_metric, p_loss=p_loss_metric)

        tb_writer.flush()

        if ep % 25 == 0:

            roll_succ = stats.hist_s[-1] if stats.hist_s else 0.0
            roll_coll = stats.hist_c[-1] if stats.hist_c else 0.0
            roll_timeout = stats.hist_t[-1] if stats.hist_t else 0.0
            roll_reward = np.mean(stats.hist_r[-stats.roll:]) if stats.hist_r else 0.0

            pct_format = stats.format_percentage(roll_succ, roll_coll, roll_timeout)

            recent_time = np.mean(stats.hist_time[-stats.roll:]) if stats.hist_time else 0.0
            recent_dfreq = np.mean(stats.hist_discomfort_freq[-stats.roll:]) if stats.hist_discomfort_freq else 0.0
            recent_ddist = np.mean(stats.hist_discomfort_dist[-stats.roll:]) if stats.hist_discomfort_dist else 0.0

            logging.info(f"[STAT] ep={ep:03d} | roll@25(s/c/t)={roll_succ:.1f}/{roll_coll:.1f}/{roll_timeout:.1f} ({pct_format}) | R.mean={roll_reward:.1f}")
            logging.info(f"[METRICS-25] time={recent_time:.2f}s | dfreq={recent_dfreq:.3f} | ddist={recent_ddist:.3f}m")

            tb_writer.add_scalar("train/avg_reward", roll_reward, ep)
            tb_writer.add_scalar("train/success_rate", roll_succ, ep)

            if metrics_plotter is not None:
                try:
                    metrics_plotter.plot(ep, stats, plot_every=25)
                except OSError as e:
                    logging.warning(f"[METRICS-PLOT] disabled after write failure: {e}")
                    metrics_plotter = None

        current_succ = 1.0 if succ else 0.0
        current_coll = 1.0 if coll else 0.0
        current_timeout = 1.0 if timeout else 0.0

        roll = (stats.hist_s[-1] if stats.hist_s else 0.0,
                stats.hist_c[-1] if stats.hist_c else 0.0,
                stats.hist_t[-1] if stats.hist_t else 0.0)

        rl_episode_count = ep
        total_episodes_target = int(episodes)

        cumulative_pct = f"{roll[0]:.1%}/{roll[1]:.1%}/{roll[2]:.1%}"

        if ep <= 50:
            loss_name = 'qloss' if direct_discrete else 'vloss'
            loss_value = metrics['q_loss'] if direct_discrete else metrics['value_loss']
            logging.info(
                f"[RL-EP-{ep:04d}] ({rl_episode_count}/{total_episodes_target}) "
                f"s/c/t={current_succ:.0f}/{current_coll:.0f}/{current_timeout:.0f} "
                f"{loss_name}={loss_value:.4f} updates={metrics['updates']} "
                f"buf={len(rl_buffer)}"
            )
            logging.info(f"              cumulative s/c/t%: {cumulative_pct}")
            logging.info(f"[REPLAY-STORE] ok={buffer_store_ok} skip={buffer_store_skip} steps={buffer_store_steps} buf={len(rl_buffer)}")
        elif ep % 25 == 0:
            loss_name = 'qloss' if direct_discrete else 'vloss'
            loss_value = metrics['q_loss'] if direct_discrete else metrics['value_loss']
            logging.info(
                f"[RL-EP-{ep:04d}] ({rl_episode_count}/{total_episodes_target}) "
                f"s/c/t={current_succ:.0f}/{current_coll:.0f}/{current_timeout:.0f} "
                f"{loss_name}={loss_value:.4f} updates={metrics['updates']} "
                f"buf={len(rl_buffer)}"
            )
            logging.info(f"              cumulative s/c/t%: {cumulative_pct}")
            logging.info(f"[REPLAY-STORE] ok={buffer_store_ok} skip={buffer_store_skip} steps={buffer_store_steps} buf={len(rl_buffer)}")

            logging.info(f"[STAGE0-MONITOR] timeout_ratio={roll[2]:.3f} (target: decrease over time)")
        elif ep % 5 == 0:
            logging.info(f"[RL-EP-{ep:04d}] roll={roll[0]:.3f}/{roll[1]:.3f}/{roll[2]:.3f}")
        else:
            logging.info(f"[RL-EP-{ep:04d}] ({rl_episode_count})")

        if roll[1] <= 0.01 and roll[2] >= 0.95:
            timeout_alert_episodes.append(ep)

            timeout_alert_episodes = timeout_alert_episodes[-200:]

        if ep % sot_monitor_interval == 0:
            # PPO: 不需要epsilon监控（使用entropy探索）
            reward_config = f"succ={cfg.getfloat('reward', 'success_reward', fallback=1.0):.1f}/coll={cfg.getfloat('reward', 'collision_penalty', fallback=-0.25):.2f}"
            from crowd_nav.contracts import grid_action_dim
            _stop_label = "+stop" if GRID.get('include_stop', False) else ""
            grid_config = f"{GRID['n_speeds']}×{GRID['n_headings']}{_stop_label}={grid_action_dim(GRID)}"
            event_dist = f"s={roll[0]:.2f}/c={roll[1]:.2f}/t={roll[2]:.2f}"

            recent_timeouts = len([x for x in timeout_alert_episodes if x > ep - 50])
            timeout_warning = f" WARNINGTIMEOUT_ALERT({recent_timeouts}/50)" if recent_timeouts >= 30 else ""

            logging.info(f"[SOT-{ep:04d}] "
                        f"REWARD {reward_config} | GRID {grid_config} | "
                        f"EVENTS {event_dist}{timeout_warning}")

            if recent_timeouts >= 40:
                logging.warning(f"[P2-SOT-ALERT] Detected {recent_timeouts}/50 timeout episodes! Checking configuration...")
                logging.warning(f"[P2-SOT-ALERT] Current time_limit={cfg.getfloat('env', 'time_limit', fallback=25):.1f}s, "
                               f"dt={cfg.getfloat('env', 'time_step', fallback=0.25):.2f}s = {cfg.getfloat('env', 'time_limit', fallback=25)/cfg.getfloat('env', 'time_step', fallback=0.25):.0f} max steps")
                logging.warning(f"[P2-SOT-ALERT] Current circle_radius={cfg.getfloat('sim', 'circle_radius', fallback=4.0):.1f}m, "
                               f"human_num={cfg.getint('sim', 'human_num', fallback=5)}")
                logging.warning("[P2-SOT-ALERT] Consider increasing time_limit or decreasing difficulty if task is infeasible")

        if ep % 25 == 0:
            multi_metrics = stats.get_multi_scale_metrics()
            logging.info("=" * 80)
            logging.info(f"[MULTI-SCALE-{ep}] TARGET: peak={multi_metrics['peak_performance']:.3f} stable_90={multi_metrics['stable_90_episodes']}ep")
            logging.info(f"[MULTI-SCALE-{ep}] WINDOWS: short={multi_metrics['short_term']['succ']:.3f} med={multi_metrics['medium_term']['succ']:.3f} long={multi_metrics['long_term']['succ']:.3f}")
            logging.info(f"[MULTI-SCALE-{ep}] QUALITY: stability={multi_metrics['stability']:.3f} learn_rate={multi_metrics['learning_rate']:.5f}")

            if multi_metrics['stable_90_episodes'] >= 5:
                logging.info(f"[ACHIEVEMENT] Stable 90%+ success rate achieved for {multi_metrics['stable_90_episodes']} episodes!")
            elif multi_metrics['peak_performance'] >= 0.9:
                logging.info(f"[MILESTONE] Peak performance {multi_metrics['peak_performance']:.1%} reached! Working towards stability...")

            if multi_metrics['stagnation_risk']:
                logging.warning(f"[ALERT] Learning stagnation detected. Consider parameter adjustment.")

            overall_health = (multi_metrics['stability'] + min(multi_metrics['learning_rate']*10, 1.0)) / 2
            health_status = "EXCELLENT" if overall_health > 0.8 else "GOOD" if overall_health > 0.6 else "NEEDS_IMPROVEMENT"
            logging.info(f"[HEALTH-CHECK-{ep}] Overall: {health_status} (score={overall_health:.3f})")
            logging.info("=" * 80)

        if plotter is not None:
            try:
                plotter.plot(ep, stats)
            except OSError as e:
                logging.warning(f"[PLOT] disabled after write failure: {e}")
                plotter = None

        if ep % eval_every == 0:

            rng_state_random = random.getstate()
            rng_state_numpy = np.random.get_state()
            rng_state_torch = torch.get_rng_state()
            if torch.cuda.is_available():
                rng_state_cuda = torch.cuda.get_rng_state()

            # PPO: 不需要epsilon和tau（evaluation时也不需要）
            policy_state_dict = None
            if hasattr(policy, 'get_state') and callable(policy.get_state):
                try:
                    policy_state_dict = copy.deepcopy(policy.get_state())
                except Exception:
                    pass

            model_cache_backup = None
            if hasattr(policy, 'reset_sequence_cache'):
                try:

                    if hasattr(policy, '_cache_states'):
                        model_cache_backup = copy.deepcopy(getattr(policy, '_cache_states', None))
                except Exception:
                    pass

            if hasattr(policy,'set_epsilon'):
                policy.set_epsilon(0.0)

            if hasattr(policy, 'reset_sequence_cache'):
                policy.reset_sequence_cache()

            logging.info(f"[EVAL] Starting evaluation: ep={ep}, ε=0.0, k={eval_episodes}")

            succ_e, coll_e, timeout_e, rew_e, nav_e, time_e, discomfort_freq_e, discomfort_dist_e, path_len_e, smooth_e, speed_e = explorer.run_k_episodes(
                k=eval_episodes, phase='val', update_memory=False, show_tqdm=False,
                imitation_learning=False, force_joint_state_policy=False)

            random.setstate(rng_state_random)
            np.random.set_state(rng_state_numpy)
            torch.set_rng_state(rng_state_torch)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(rng_state_cuda)

            # PPO: 不需要恢复epsilon（evaluation时epsilon=0也是通过set_epsilon(0.0)设置的）
            if policy_state_dict is not None and hasattr(policy, 'set_state'):
                try:
                    policy.set_state(policy_state_dict)
                except Exception:
                    pass

            if model_cache_backup is not None:
                try:
                    if hasattr(policy, '_cache_states'):
                        policy._cache_states = model_cache_backup
                except Exception:
                    pass

            # PPO修复：先计算eval_sum再使用（避免UnboundLocalError）
            eval_sum = succ_e + coll_e + timeout_e

            logging.info(f"[EVAL] Completed evaluation: succ={succ_e:.3f} coll={coll_e:.3f} timeout={timeout_e:.3f} sum={eval_sum:.3f} nav={nav_e:.2f}")

            logging.info(f"[EVAL-EXTENDED] path_len={path_len_e:.2f}m smoothness={smooth_e:.3f} avg_speed={speed_e:.2f}m/s")
            if abs(eval_sum - 1.0) > 0.05:
                logging.warning(f"[EVAL-SUM-CHECK] Evaluation sum deviation: {eval_sum:.3f} != 1.00, check evaluation calibration")

            eval_count += 1
            is_new_best = succ_e > best_succ
            best_succ = max(best_succ, succ_e)
            if is_new_best:
                evals_without_improvement = 0
                best_stats_history = {
                    'hist_s': getattr(stats, 'hist_s', []) or [],
                    'hist_c': getattr(stats, 'hist_c', []) or [],
                    'hist_t': getattr(stats, 'hist_t', []) or [],
                    'hist_r': getattr(stats, 'hist_r', []) or [],
                }
                best_tmp_path = best_path + ".tmp"
                try:
                    torch.save(_build_rl_checkpoint(ep, best_stats_history), best_tmp_path)
                    os.replace(best_tmp_path, best_path)
                finally:
                    if os.path.exists(best_tmp_path):
                        try:
                            os.remove(best_tmp_path)
                        except OSError:
                            pass
                logging.info(f"[BEST-SAVE] success={best_succ:.3f} -> {best_path}")
            else:
                evals_without_improvement += 1

            tb_writer.add_scalar("eval/success_rate", succ_e, ep)

            tb_writer.add_scalar("eval/path_length", path_len_e, ep)
            tb_writer.add_scalar("eval/smoothness", smooth_e, ep)
            tb_writer.add_scalar("eval/avg_speed", speed_e, ep)

            if direct_discrete:
                collapse_threshold = baseline_succ * collapse_ratio
                if succ_e < collapse_threshold:
                    collapse_eval_count += 1
                else:
                    collapse_eval_count = 0

                logging.info(
                    f"[SAFE-IMPROVEMENT] baseline={baseline_succ:.3f} best={best_succ:.3f} "
                    f"current={succ_e:.3f} collapse_threshold={collapse_threshold:.3f} "
                    f"collapse_count={collapse_eval_count}/{collapse_patience} "
                    f"no_improve={evals_without_improvement}/{early_stop_patience}"
                )

                collapsed = collapse_eval_count >= collapse_patience
                plateaued = (
                    eval_count >= min_evals_before_stop
                    and evals_without_improvement >= early_stop_patience
                )
                if safe_stop_disabled and (collapsed or plateaued):
                    reason = "performance collapse" if collapsed else "no safe improvement"
                    logging.warning(
                        f"[SAFE-IMPROVEMENT] Would stop at episode {ep}: {reason}; "
                        f"continuing because --disable-safe-stop is set"
                    )
                    collapse_eval_count = 0
                    evals_without_improvement = 0
                elif collapsed or plateaued:
                    reason = "performance collapse" if collapsed else "no safe improvement"
                    logging.warning(
                        f"[SAFE-IMPROVEMENT] Stopping at episode {ep}: {reason}; "
                        f"restoring {best_path}"
                    )
                    best_checkpoint = safe_torch_load(best_path, map_location=device)
                    best_policy_state = best_checkpoint['policy_state']
                    try:
                        policy.load_state_dict(best_policy_state, strict=True)
                    except RuntimeError:
                        normalized_state = {
                            key.replace('_orig_mod.', '', 1): value
                            for key, value in best_policy_state.items()
                        }
                        policy_net_base.load_state_dict(normalized_state, strict=True)
                    if 'target_q_net_state' in best_checkpoint:
                        target_q_net.load_state_dict(
                            best_checkpoint['target_q_net_state'],
                            strict=True,
                        )
                    stop_training_early = True

        if stop_training_early:
            break

        if (ep % checkpoint_interval == 0):

            log_periodic_metrics(ep, stats, window=200)

        # Progress reward统计已删除（简化版）
        # if (ep % 25 == 0) and (ep > 0):
        #     if hasattr(env, '_rprog_stats'):
        #         rprog = env._rprog_stats
        #         if rprog['count'] > 0:
        #             coverage = rprog['nonzero'] / rprog['count']
        #             pos_avg = rprog['pos_sum'] / max(1, rprog['pos_count'])
        #             neg_avg = rprog['neg_sum'] / max(1, rprog['neg_count'])
        #             episode_num = getattr(env, 'current_episode', ep)
        #             current_lambda_p = getattr(env, 'progress_reward_weight', 0.12)
        #             logging.info(f"[R_PROG@{ep}] λ_p={current_lambda_p:.3f} | cover={coverage:.1%} pos={pos_avg:.4f} neg={neg_avg:.4f}")
        #
        #             env._rprog_stats = {'count': 0, 'nonzero': 0, 'pos_sum': 0.0, 'pos_count': 0, 'neg_sum': 0.0, 'neg_count': 0}

        if (ep % checkpoint_interval == 0):

            rl_model_path = os.path.join(ARGS.outdir, f"rl_model_ep{ep}.pth")
            try:
                current_gamma = cfg.getfloat('train', 'gamma', fallback=0.995)

                # 安全获取stats历史（防止None或属性不存在）
                stats_history = {
                    'hist_s': getattr(stats, 'hist_s', []) or [],
                    'hist_c': getattr(stats, 'hist_c', []) or [],
                    'hist_t': getattr(stats, 'hist_t', []) or [],
                    'hist_r': getattr(stats, 'hist_r', []) or []
                }

                rl_checkpoint = _build_rl_checkpoint(ep, stats_history)
                tmp_model_path = rl_model_path + ".tmp"
                torch.save(rl_checkpoint, tmp_model_path)
                os.replace(tmp_model_path, rl_model_path)
                logging.info(f"[RL-SAVE] {algo} checkpoint -> {rl_model_path}")
                logging.info(f"[RL-SAVE] Saved components: policy, target network, optimizer")
                if direct_discrete:
                    _prune_rl_checkpoints(ARGS.outdir, keep=3)

            except Exception as e:
                import traceback
                tmp_model_path = rl_model_path + ".tmp"
                if os.path.exists(tmp_model_path):
                    try:
                        os.remove(tmp_model_path)
                    except OSError:
                        pass
                logging.error(f"[RL-SAVE] RL checkpoint failed: {e}")
                logging.error(f"[RL-SAVE] Full traceback:\n{traceback.format_exc()}")

                # 尝试分步保存，定位具体失败的组件
                try:
                    logging.info("[RL-SAVE] Attempting component-by-component save to diagnose...")
                    components = {
                        'policy_state': policy.state_dict(),
                    }
                    target_key = 'target_q_net_state' if direct_discrete else 'target_value_net_state'
                    optim_key = 'optim_q_state' if direct_discrete else 'optim_value_state'
                    try:
                        components[target_key] = target_net.state_dict()
                        logging.info(f"[RL-SAVE] {target_key} OK")
                    except Exception as etv:
                        logging.error(f"[RL-SAVE] {target_key} failed: {etv}")
                    try:
                        components[optim_key] = rl_optimizer.state_dict()
                        logging.info(f"[RL-SAVE] {optim_key} OK")
                    except Exception as eopt:
                        logging.error(f"[RL-SAVE] {optim_key} failed: {eopt}")

                    components['episode'] = ep
                    components['stage'] = f'rl_training_{algo}_partial'
                    components['meta'] = {
                        'occlusion': _occlusion_meta(cfg, policy_net_base),
                    }
                    torch.save(components, rl_model_path)
                    logging.info(f"[RL-SAVE] Partial checkpoint saved with {len(components)} components -> {rl_model_path}")

                except Exception as e2:
                    logging.error(f"[RL-SAVE] Partial checkpoint also failed: {e2}")
                    # 最后兜底：只保存policy
                    try:
                        minimal_checkpoint = {
                            'episode': ep,
                            'policy_state': policy.state_dict(),
                            'stage': 'rl_training_minimal',
                            'meta': {
                                'occlusion': _occlusion_meta(cfg, policy_net_base),
                            },
                        }
                        torch.save(minimal_checkpoint, rl_model_path)
                        logging.warning(f"[RL-SAVE] Minimal RL checkpoint (policy only) -> {rl_model_path}")
                    except Exception as e3:
                        logging.error(f"[RL-SAVE] Minimal checkpoint failed: {e3}")
                        if direct_discrete:
                            raise RuntimeError(
                                "Discrete-Mamba checkpoint could not be saved; stopping to avoid uncheckpointed training"
                            ) from e3

    log_final_metrics(stats, best_succ)

    if vectorized_sampler is not None:
        logging.info("[VECTORIZE] Stopping vectorized sampler...")
        vectorized_sampler.stop()
        logging.info("[VECTORIZE] Vectorized sampler stopped")

    logging.info(f"[DONE] Training completed - best_eval_success={best_succ:.3f}")
    return best_succ

def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.txt')
    parser.add_argument('--outdir', type=str, default='runs/mamba_vl')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--policy', type=str, default=None, help='override policy key (e.g., mamba or mamba_rl)')
    parser.add_argument('--gpu', action='store_true', help='force CUDA if available')
    parser.add_argument('--cpu', action='store_true', help='force CPU even if CUDA available')

    parser.add_argument('--stage', type=str, choices=['il','rl','auto'], default='auto',
                        help='Training stage: il=IL pretraining only, rl=RL only, auto=IL then RL')
    parser.add_argument('--two-stage-ablation-suite', action='store_true',
                        help='Run the two retraining ablations serially: ORCA-only and w/o ORCA pretraining')
    parser.add_argument('--pretrain-only', action='store_true',
                        help='Run ORCA/IL pretraining only, save il_policy.pth, then exit before online RL')
    parser.add_argument('--skip-il-pretrain', action='store_true',
                        help='Skip ORCA/IL pretraining and train online from random initialization')
    parser.add_argument('--discrete-mamba', action='store_true',
                        help='Train the direct discrete-action Mamba baseline: ORCA action IL then Double-DQN')
    parser.add_argument('--temporal-backbone', choices=['mamba', 'gru', 'mlp'], default=None,
                        help='Override the temporal backbone for Mamba/GRU/stateless-MLP ablations')
    parser.add_argument('--disable-safe-stop', action='store_true',
                        help='Disable direct-Q safe early stopping while keeping diagnostics/checkpoints')

    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint file (e.g., --resume policy_ep1000.pt)')
    parser.add_argument('--start_episode', type=int, default=None,
                        help='Override start episode number when resuming')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: from config)')
    parser.add_argument('--train-episodes', type=int, default=None,
                        help='Override [train] train_episodes for a registered pilot')
    parser.add_argument('--il-success-target', type=int, default=None,
                        help='Override [imitation_learning] success_target for a pilot')
    parser.add_argument('--il-epochs', type=int, default=None,
                        help='Override [train] il_epochs for a registered pilot or smoke test')
    parser.add_argument('--il-collect-batch', type=int, default=None,
                        help='Override [imitation_learning] prefill_batch_episodes')
    parser.add_argument('--post-il-eval-episodes', type=int, default=None,
                        help='Override post-IL closed-loop evaluation episode count')
    parser.add_argument('--pre-rl-eval-episodes', type=int, default=None,
                        help='Override pre-RL diagnostic evaluation episode count')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Override checkpoint interval for a registered pilot or smoke test')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile (used for CPU and smoke validation)')
    # Order 17 item 14: one flag selects the arm. All four arms share this
    # entry point and the same model class, so an arm cannot acquire a private
    # training script and drift.
    parser.add_argument('--occlusion-mode', type=str, default=None,
                        choices=['off', 'sensor', 'deterministic', 'bayes', 'gt',
                                 'oracle_belief'],
                        help='occlusion arm; overrides [occlusion] mode in env.config')
    parser.add_argument('--legacy-warm-start', type=str, default=None,
                        help='Explicit pre-occlusion checkpoint used only to initialize an occlusion arm')
    parser.add_argument('--belief-features',
                        choices=['full', 'fixed_confidence'], default=None,
                        help='paired belief input: posterior confidence or fixed point confidence')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable CUDA deterministic mode (slower but reproducible)')
    ARGS = parser.parse_args()

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if ARGS.gpu and torch.cuda.is_available():
        ARGS.device = 'cuda'
    elif ARGS.cpu:
        ARGS.device = 'cpu'

    if ARGS.two_stage_ablation_suite:
        sys.exit(_run_two_stage_ablation_suite(ARGS))

    _require_free_disk_space(ARGS.outdir, minimum_gib=2.0)

    if ARGS.discrete_mamba and not ARGS.resume and ARGS.temporal_backbone in (None, 'mamba'):
        latest = _latest_valid_discrete_checkpoint(os.path.abspath(ARGS.outdir))
        if latest:
            ARGS.resume = latest
            print(f"[AUTO-RESUME] Valid Discrete-Mamba checkpoint detected: {latest}", flush=True)
    elif (ARGS.discrete_mamba and not ARGS.resume and
          ARGS.temporal_backbone in ('gru', 'mlp')):
        print("[AUTO-RESUME] Disabled for non-Mamba direct-Q override; "
              "pass --resume explicitly if needed.", flush=True)

    # RTX 4090: 启用 BF16 AMP 支持
    if ARGS.device == 'cuda':
        torch.set_float32_matmul_precision('high')
        logging.info("[AMP] BF16 matmul precision enabled for RTX 4090")

    logger = setup_logger(ARGS.outdir)

    tb_log_dir = os.path.join(ARGS.outdir, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)

    if not ARGS.resume:
        import glob
        for old_file in glob.glob(os.path.join(tb_log_dir, "events.out.tfevents.*")):
            try:
                os.remove(old_file)
            except Exception as e:
                logging.warning(f"[TENSORBOARD] Failed to remove {old_file}: {e}")

    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    logging.info(f"[TENSORBOARD] Fresh events will be saved to: {tb_log_dir}")

    cfg = load_config(ARGS.config)
    cfg = apply_occlusion_override(cfg, ARGS)
    if ARGS.train_episodes is not None:
        if ARGS.train_episodes < 1:
            raise SystemExit('--train-episodes must be >= 1')
        cfg.set('train', 'train_episodes', str(ARGS.train_episodes))
    if ARGS.il_success_target is not None:
        if ARGS.il_success_target < 1:
            raise SystemExit('--il-success-target must be >= 1')
        cfg.set('imitation_learning', 'success_target',
                str(ARGS.il_success_target))
    if ARGS.il_epochs is not None:
        if ARGS.il_epochs < 1:
            raise SystemExit('--il-epochs must be >= 1')
        cfg.set('train', 'il_epochs', str(ARGS.il_epochs))
    if ARGS.il_collect_batch is not None:
        if ARGS.il_collect_batch < 1:
            raise SystemExit('--il-collect-batch must be >= 1')
        cfg.set('imitation_learning', 'prefill_batch_episodes',
                str(ARGS.il_collect_batch))
    if ARGS.post_il_eval_episodes is not None:
        if ARGS.post_il_eval_episodes < 1:
            raise SystemExit('--post-il-eval-episodes must be >= 1')
        cfg.set('imitation_learning', 'post_il_eval_episodes',
                str(ARGS.post_il_eval_episodes))
    if ARGS.pre_rl_eval_episodes is not None:
        if ARGS.pre_rl_eval_episodes < 1:
            raise SystemExit('--pre-rl-eval-episodes must be >= 1')
        cfg.set('train', 'pre_rl_eval_episodes',
                str(ARGS.pre_rl_eval_episodes))
    if ARGS.save_every is not None:
        if ARGS.save_every < 1:
            raise SystemExit('--save-every must be >= 1')
        cfg.set('train', 'save_every', str(ARGS.save_every))
    if _occlusion_mode(cfg) != 'off':
        if not cfg.has_section('imitation_learning'):
            cfg.add_section('imitation_learning')
        cfg.set('imitation_learning', 'force_online', 'true')
        logging.info(
            "[OCCLUSION] Online IL collection forced: legacy 34-D/full-observation "
            "datasets are incompatible with occlusion policy tokens"
        )
    if ARGS.legacy_warm_start:
        if ARGS.resume or ARGS.skip_il_pretrain:
            raise SystemExit(
                "--legacy-warm-start cannot be combined with --resume or --skip-il-pretrain"
            )
        if not cfg.has_section('train'):
            cfg.add_section('train')
        cfg.set('train', 'il_force_retrain', 'true')
    if ARGS.temporal_backbone is not None:
        if not cfg.has_section('mamba'):
            cfg.add_section('mamba')
        cfg.set('mamba', 'temporal_backbone', ARGS.temporal_backbone)
        logging.info(f"[TEMPORAL-BACKBONE] CLI override: {ARGS.temporal_backbone}")

    # ✅ 仅从 configs/env.config 的 [policy] 初始化动作网格（SSOT）
    env_cfg_raw = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    config_dir = os.path.dirname(ARGS.config) if os.path.dirname(ARGS.config) else './configs'
    env_config_path = os.path.join(config_dir, 'env.config')
    if not os.path.exists(env_config_path):
        logging.error(f"[CONTRACTS] env.config not found: {env_config_path}")
        sys.exit(1)
    env_cfg_raw.read(env_config_path, encoding='utf-8')
    if not env_cfg_raw.has_section('policy'):
        if env_cfg_raw.has_section('action_space'):
            env_cfg_raw.add_section('policy')
            for key in ('n_speeds', 'n_headings', 'v_min', 'v_max', 'sampling', 'include_stop', 'stop_eps'):
                if env_cfg_raw.has_option('action_space', key):
                    env_cfg_raw.set('policy', key, env_cfg_raw.get('action_space', key))
            logging.info(f"[CONTRACTS] [policy] missing; copied from [action_space] in {env_config_path}")
        else:
            logging.error(f"[CONTRACTS] Missing [policy] or [action_space] section in {env_config_path}")
            sys.exit(1)

    from crowd_nav.contracts import init_grid_from_cfg, GRID
    init_grid_from_cfg(env_cfg_raw)

    # 校验GRID与env.config一致，否则直接退出
    try:
        expected_grid = {
            'n_speeds': env_cfg_raw.getint('policy', 'n_speeds'),
            'n_headings': env_cfg_raw.getint('policy', 'n_headings'),
            'v_min': env_cfg_raw.getfloat('policy', 'v_min'),
            'v_max': env_cfg_raw.getfloat('policy', 'v_max'),
        }
    except Exception as e:
        logging.error(f"[CONTRACTS] Failed to parse [policy] from {env_config_path}: {e}")
        sys.exit(1)

    mismatches = []
    for k, v in expected_grid.items():
        if float(GRID.get(k)) != float(v):
            mismatches.append((k, GRID.get(k), v))
    if mismatches:
        logging.error(f"[CONTRACTS] GRID mismatch vs {env_config_path}: {mismatches}")
        sys.exit(1)

    from crowd_nav.contracts import grid_action_dim
    n_actions = int(grid_action_dim(GRID))
    logging.info(f"[CONTRACTS] GRID locked from env.config: {GRID}, num_actions={n_actions}")

    train_cfg = TrainConfig(cfg, ARGS.outdir)
    env_cfg = EnvConfig(cfg)
    policy_cfg = PolicyConfig(cfg)

    seed = train_cfg.seed
    # 命令行参数覆盖配置文件
    if ARGS.seed is not None:
        seed = ARGS.seed
        logging.info(f"[SEED] 使用命令行指定的seed: {seed}")
    else:
        logging.info(f"[SEED] 使用配置文件的seed: {seed}")

    episodes = train_cfg.episodes
    updates_ep = train_cfg.updates_per_ep
    batch_size = train_cfg.batch_size
    lr = train_cfg.learning_rate
    gamma = train_cfg.gamma
    # PPO: 不需要epsilon（使用entropy探索）、huber_delta（使用MSE）
    dt = env_cfg.time_step
    roll_window = train_cfg.roll_window
    plot_every = train_cfg.plot_every
    save_every = train_cfg.save_every

    # PPO: grad_clip在config中定义为clip_grad_norm
    grad_clip = train_cfg.clip_grad_norm

    if not (dt>0):
        raise RuntimeError('配置 dt 必须 > 0，并与环境步长一致')

    # 设置随机种子和CUDA确定性模式
    set_seed(int(seed), deterministic=ARGS.deterministic)
    device = torch.device(ARGS.device)

    # TF32加速（仅在非确定性模式下启用）
    if torch.cuda.is_available() and not ARGS.deterministic:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 删除冗余日志：TF32 Enabled

    # ✅ Reward profile对齐：仅在显式配置时才覆盖，避免“口径混乱”
    if cfg.has_option('train', 'reward_profile'):
        profile = cfg.get('train', 'reward_profile', fallback='benchmark').strip().lower()
        if profile == 'benchmark':
            for k in ('progress_reward', 'time_penalty', 'stand_penalty', 'discomfort_penalty_factor'):
                if cfg.has_option('reward', k):
                    cfg.set('reward', k, '0.0')
            logging.info("[REWARD-PROFILE] benchmark -> shaping disabled")
        elif profile in ('dense', 'shaped'):
            logging.info("[REWARD-PROFILE] dense -> shaping from config")
        else:
            logging.info(f"[REWARD-PROFILE] unknown profile='{profile}', using config as-is")

    env, robot = build_env_and_robot(cfg)
    pol_key = (ARGS.policy or policy_cfg.policy_name).lower()

    logging.info(f"[BOOT] policy={pol_key} device={ARGS.device}")

    robot_visible = env_cfg.robot_visible
    success_reward = env_cfg.success_reward
    collision_penalty = env_cfg.collision_penalty

    from crowd_nav.contracts import GRID, grid_action_dim
    n_actions = grid_action_dim(GRID)
    logging.info("=" * 60)
    logging.info(
        f"[SOT] env: circle R={env_cfg.circle_radius} Nh={env_cfg.human_num} dt={env_cfg.time_step} "
        f"T={env_cfg.time_limit} | reward: {success_reward}/{collision_penalty}/{env_cfg.timeout_penalty if hasattr(env_cfg,'timeout_penalty') else 'N/A'} "
        f"| actions: {n_actions}(sampling={GRID.get('sampling')}, stop={GRID.get('include_stop', False)}) | "
        f"success_radius={env_cfg.robot_success_radius} | eval eps=0"
    )
    logging.info("=" * 60)
    if ARGS.discrete_mamba:
        temporal_backbone = cfg.get('mamba', 'temporal_backbone', fallback='mamba').strip().lower()
        logging.info(f"[ALGO] Direct-Q ({temporal_backbone.upper()}): q_head action prediction + Double-DQN")
    else:
        temporal_backbone = cfg.get('mamba', 'temporal_backbone', fallback='mamba').strip().lower()
        logging.info(f"[ALGO] Value-lookahead ({temporal_backbone.upper()}): one-step lookahead + epsilon-greedy")

    logging.info(f"REWARD: success={success_reward}, collision={collision_penalty}, timeout={env_cfg.timeout_penalty}")
    logging.info(f"TRAINING: lr={lr}, gamma={gamma}, batch={batch_size}, updates_per_ep={updates_ep}")
    logging.info(f"TIMESTEP: dt={dt}s")
    logging.info("=" * 60)

    # ============================================================
    # 构建主策略网络 (必须在 TEACHER-LOAD 之前)
    # ============================================================
    policy = build_policy(pol_key, cfg, device)
    bind_policy(robot, policy, env)
    logging.info(f"[POLICY-INIT] Main policy built: {pol_key}")

    # 根源修复：CrowdNav是多智能体场景，policy必须设置multiagent_training=True
    # 否则env.reset()会强制human_num=1，导致34维训练 vs 14维推理的维度不匹配
    if hasattr(policy, 'multiagent_training'):
        policy.multiagent_training = True
        logging.info(f"[POLICY-INIT] Set multiagent_training=True (5-human crowd navigation)")

    if ARGS.legacy_warm_start:
        if _occlusion_mode(cfg) == 'off':
            raise RuntimeError("--legacy-warm-start is only valid for an active occlusion arm")
        legacy_path = os.path.abspath(ARGS.legacy_warm_start)
        if not os.path.isfile(legacy_path):
            raise FileNotFoundError(legacy_path)
        from crowd_nav.policy.mamba_rl import warm_start_from_legacy
        legacy_checkpoint = safe_torch_load(legacy_path, map_location=device)
        legacy_state = _checkpoint_state_dict(legacy_checkpoint)
        warm_start_from_legacy(policy, legacy_state, strict=False, verbose=True)
        policy._legacy_occlusion_warm_started = True
        logging.info(
            "[OCCLUSION] Legacy checkpoint initialized the shared network; "
            "new IL is still required and will not be skipped"
        )

    if hasattr(policy,'set_env_dt'): policy.set_env_dt(dt)

    logging.info("✅ debug.md清理：奖励整形器已删除（未实际使用）")

    logging.info("=" * 60)
    logging.info("[CONFIG-SUMMARY] Training Configuration (Runtime Values)")
    logging.info(f"  Policy: {pol_key}")
    if hasattr(policy, 'model'):
        model = policy.model

        if hasattr(model, 'temporal_encoder'):
            actual_n_layers = getattr(model.temporal_encoder, 'n_layers', 'N/A')
        else:
            actual_n_layers = 'N/A'
        actual_d_model = getattr(model, 'd_model', 'N/A')
        logging.info(f"  Network: n_layers={actual_n_layers}, d_model={actual_d_model}")
    logging.info(f"  Sequence: seq_len={policy_cfg.seq_len}")
    logging.info(f"  Training: batch_size={batch_size}, updates_per_ep={updates_ep}")
    logging.info(f"  Learning: lr={lr:.6f}, gamma={gamma}")
    logging.info(f"  Episodes: {episodes}, device={device}")
    logging.info("=" * 60)

    if hasattr(policy, 'set_phase'):
        policy.set_phase('train')
    if hasattr(policy, 'set_goal_prior'):
        policy.set_goal_prior(False)
    algo = cfg.get('train', 'algo', fallback='sarl').lower()
    logging.info(f"[{algo.upper()}-TRAIN] goal prior=OFF")

    circle_radius = cfg.getfloat('sim', 'circle_radius', fallback=4.0)
    human_num = cfg.getint('sim', 'human_num', fallback=5)
    dt = cfg.getfloat('env', 'time_step', fallback=0.25)
    time_limit = cfg.getfloat('env', 'time_limit', fallback=25.0)
    success_radius = cfg.getfloat('robot', 'success_radius', fallback=0.25)
    robot_visible = cfg.get('robot', 'visible', fallback='false').lower() == 'true'

    n_actions = _log_action_table(GRID)

    logging.info(f"[CFG-LOCK] circle R=4 Nh=5 dt=0.25 T=50 succ_r=0.25 visible=False actions={n_actions} eval_eps=0")

    # 验证配置一致性（无日志输出）
    # 🔥 修复：支持include_stop=false配置（80个动作）
    expected_actions = 80 if not GRID.get('include_stop', False) else 81
    assert circle_radius == 4 and human_num == 5 and abs(dt - 0.25) < 1e-6 and time_limit == 50 \
           and abs(success_radius - 0.25) < 1e-6 and not robot_visible and n_actions == expected_actions \
           and str(GRID.get('sampling', '')).lower() == 'exponential', \
           f"Config mismatch: R={circle_radius}, Nh={human_num}, dt={dt}, T={time_limit}, succ_r={success_radius}, visible={robot_visible}, actions={n_actions} (expected={expected_actions}), sampling={GRID.get('sampling')}, stop={GRID.get('include_stop', False)}"

    from crowd_nav.contracts import discrete_index_to_action
    import math

    v_pref = cfg.getfloat('robot', 'v_pref', fallback=1.0)
    v_max_grid = GRID['v_max']

    v_threshold = 0.8 * v_pref
    if v_max_grid < v_threshold:
        logging.error("FATAL: 动作表熔断触发！Action table circuit breaker triggered!")
        logging.error(f"FATAL: v_max={v_max_grid:.3f} < threshold={v_threshold:.3f} (0.8 × v_pref)")
        logging.error("FATAL: 请检查 configs/env.config 中的 [policy] v_min/v_max 设置")
        sys.exit(1)

    cfg_print_count = getattr(main, '_cfg_print_count', 0)
    if cfg_print_count > 0:
        logging.error("FATAL: 评测口径熔断触发！Multiple [CFG] prints detected!")
        logging.error("FATAL: 只允许打印一次[CFG]配置，发现多次打印可能存在口径不一致")
        sys.exit(1)
    main._cfg_print_count = cfg_print_count + 1

    # 删除不合理的time_limit熔断：time_limit是环境配置，不应与reward_profile绑定
    # SARL原版用50秒，但reward shaping与time_limit无关
    profile = cfg.get('train', 'reward_profile', fallback='benchmark')
    logging.info(f"[REWARD-PROFILE] Using profile={profile}, time_limit={time_limit}s")

    REWARD = {
        'success': cfg.getfloat('reward', 'success_reward', fallback=1.0),
        'collision': cfg.getfloat('reward', 'collision_penalty', fallback=-0.25),
        'timeout': cfg.getfloat('reward', 'timeout_penalty', fallback=-0.6)
    }

    logging.info(f"[REWARD-{profile.upper()}] {REWARD}")
    # Progress reward已删除（简化版）
    # logging.info(f"[REWARD-PROGRESS] r_prog ENABLED | warm-up: ep<150 λ_p=0.0 → ep 150-1150 linear → ep≥1150 λ_p=0.12")
    # logging.info(f"[REWARD-PROGRESS] r_max=0.06 (clip单步进度) | m_t gate: 1[dmin≥0.6m] (拥挤区不刷进度)")

    config_success = cfg.getfloat('reward', 'success_reward', fallback=1.0)
    config_collision = cfg.getfloat('reward', 'collision_penalty', fallback=-1.0)
    config_timeout = cfg.getfloat('reward', 'timeout_penalty', fallback=-0.8)

    logging.info(f"[CONFIG-FALLBACK] success={config_success}, collision={config_collision}, timeout={config_timeout}")
    logging.info(f"[ACTIVE-REWARD] Using profile-based rewards: {REWARD}")

    logging.info("=" * 60)

    _first_action_logged = [False]

    timeout_penalty = cfg.getfloat('reward', 'timeout_penalty', fallback=-0.6)

    # DoubleQ不需要PPO buffer
    explorer = Explorer(env, robot, device, memory=None, gamma=gamma)

    trajectory_log_path = os.path.join(ARGS.outdir, 'trajectory.log')
    trajectory_file = None  # 初始化为None，避免NameError
    try:
        if ARGS.discrete_mamba:
            raise RuntimeError("disabled for Discrete-Mamba training")

        trajectory_file = open(trajectory_log_path, 'w', encoding='utf-8')
        explorer.trajectory_logger = trajectory_file
        explorer.enable_trajectory_logging = True
        logging.info(f"[TRAJECTORY-LOG] Recording enabled: {trajectory_log_path}")
    except Exception as e:
        if ARGS.discrete_mamba:
            logging.info("[TRAJECTORY-LOG] Disabled for Discrete-Mamba training")
        else:
            logging.warning(f"[TRAJECTORY-LOG] Failed to initialize trajectory logging: {e}")
        trajectory_file = None
        explorer.trajectory_logger = None
        explorer.enable_trajectory_logging = False

    import copy

    # 延迟 torch.compile：在所有 checkpoint 加载完成后执行
    # 原因：torch.compile 会改变 state_dict key 的前缀（_orig_mod.*），导致加载失败
    # 在加载 checkpoint 之前编译会导致参数不匹配，使用随机初始化而不是已训练的权重
    compile_policy_later = (
        not ARGS.discrete_mamba
        and device.type == 'cuda'
        and not ARGS.no_compile
        and cfg.getboolean('train', 'torch_compile', fallback=True)
    )
    if ARGS.discrete_mamba:
        logging.info("[JIT] Disabled for constrained Discrete-Mamba (supports staged backbone unfreezing)")

    # DoubleQ不需要单独的policy optimizer（Q网络在RL阶段更新）

    train_step_counter = {'step': 0}

    il_weights_path = './models/il_model.pth'
    il_weights_loaded = False

    if (os.path.exists(il_weights_path) and not ARGS.discrete_mamba
            and _occlusion_mode(cfg) == 'off'):
        try:

            ckpt = safe_torch_load(il_weights_path, map_location=device)
            missing, unexpected = policy.load_state_dict(ckpt.get('model', ckpt), strict=False)
            if missing:
                logging.info(f"[DUAL-HEAD] Missing keys in checkpoint (expected for new continuous_head): {missing}")
            if unexpected:
                logging.warning(f"[DUAL-HEAD] Unexpected keys in checkpoint: {unexpected}")
            logging.info(f"[DEBUG.MD] Successfully loaded IL weights from {il_weights_path} (strict=False)")
            il_weights_loaded = True
        except Exception as e:
            logging.warning(f"[DEBUG.MD] IL weight loading failed, continuing without IL warm-start: {e}")
            il_weights_loaded = False

    # 删除冗余日志：Mixed buffer strategy

    stats = TrainingStats(roll_window)
    plotter = Plotter(ARGS.outdir, plot_every)
    metrics_plotter = MetricsPlotter(ARGS.outdir)

    fmt = get_log_formatter()

    outdir_abs = os.path.abspath(ARGS.outdir)
    ckpt_path = os.path.join(outdir_abs, 'ckpt_il.pt')
    il_policy_path = os.path.join(outdir_abs, 'il_policy.pth')

    force_online = cfg.getboolean('imitation_learning', 'force_online', fallback=False)
    offline_dataset_path = None
    if force_online:
        logging.info("[IL-AUTO] force_online=true -> skip offline dataset auto-detect")
    else:
        for version in ['v5.0_merged_15000_with_idx', 'v5.0_merged_15000', 'v5.0_merged', 'v4.0', 'v3.1', 'v3.0', 'v2.1', 'v2.0', 'v1.0']:
            candidates = [
                f'data/il_dataset_diverse_{version}.pth',
                f'../data/il_dataset_diverse_{version}.pth',
                os.path.join(os.path.dirname(__file__), f'../data/il_dataset_diverse_{version}.pth'),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    offline_dataset_path = os.path.abspath(candidate)
                    break
            if offline_dataset_path:
                break

    skip_il = False
    logging.info(f"[IL-AUTO] 自动检测训练数据...")
    logging.info(f"[IL-AUTO]   离线数据集: {offline_dataset_path or 'None'} (exists: {os.path.exists(offline_dataset_path) if offline_dataset_path else False})")
    logging.info(f"[IL-AUTO]   BC checkpoint: {il_policy_path} (exists: {os.path.exists(il_policy_path)})")
    logging.info(f"[IL-AUTO]   IL checkpoint: {ckpt_path} (exists: {os.path.exists(ckpt_path)})")

    if ARGS.discrete_mamba and ARGS.resume:
        logging.info("[AUTO-RESUME] RL checkpoint found: skip IL and restore the complete training state")
        skip_il = True
    elif ARGS.skip_il_pretrain:
        logging.info("[ABLATION] --skip-il-pretrain enabled: skip ORCA/IL pretraining and IL replay prefill")
        skip_il = True
    elif force_online:
        logging.info("[IL-AUTO] force_online=true -> will run IL+BC online (ignore checkpoints/offline cache)")
        skip_il = False
    # PPO修复：修正IL跳过条件（offline_dataset_path可能是非空字符串但文件不存在）
    elif os.path.exists(ckpt_path) and os.path.exists(il_policy_path) and not (offline_dataset_path and os.path.exists(offline_dataset_path)):

        logging.info(f"[IL-AUTO] ✓ 发现checkpoint但无离线数据集，加载checkpoint跳过IL+BC")
        try:
            ckpt = safe_torch_load(ckpt_path, map_location=device)
            _assert_occlusion_resume(cfg, policy, ckpt)
            policy.load_state_dict(
                ckpt['policy'],
                strict=(_occlusion_mode(cfg) != 'off'),
            )
            # PPO: policy已是actor-critic合一，不需要加载value_net和target_net
            il_success_rate = ckpt.get('il_success_rate', 0.0)
            logging.info(f"[IL-AUTO] ✓ 加载checkpoint成功 (IL成功率: {il_success_rate:.1%})")

            # PPO: 不需要warmup buffer（每次更新后清空）

            skip_il = True
            logging.info("[IL-AUTO] ✓ 跳过IL+BC，直接进入RL训练")
        except Exception as e:
            logging.warning(f"[IL-AUTO] Checkpoint加载失败: {e}")
            if _occlusion_mode(cfg) != 'off':
                raise
            skip_il = False
            checkpoint = None # Initialize to None on failure
    elif offline_dataset_path and os.path.exists(offline_dataset_path):

        logging.info(f"[IL-AUTO] ✓ 检测到离线数据集: {offline_dataset_path}")
        logging.info(f"[IL-AUTO]   将跳过IL在线收集，强制执行BC训练")
        skip_il = False
    else:
        logging.info("[IL-AUTO] ✗ 无离线数据集也无checkpoint，将执行完整IL+BC训练")
        skip_il = False

    # ====== Buffer 初始化 ======
    from crowd_nav.utils.ppo_buffer import ReplayBufferIQL
    from crowd_nav.contracts import token_shape_for_contract

    seq_len = cfg.getint('buffer', 'seq_len', fallback=12)

    algo = cfg.get('train', 'algo', fallback='sarl').lower()
    if algo == 'sarl':
        n_step = 1
        use_per = False
        per_alpha = 0.6
        per_beta_start = 0.4
        per_eps = 1e-6
        buffer_tag = 'SARL-BUFFER'
    else:
        # Fallback/Legacy config (should not be reached if forced to sarl)
        n_step = 1
        use_per = False
        per_alpha = 0.6
        per_beta_start = 0.4
        per_eps = 1e-6
        buffer_tag = 'LEGACY-BUFFER'

    replay_obs_shape = (
        token_shape_for_contract(
            cfg.get('occlusion', 'token_contract', fallback='legacy_top5')
            .strip().lower(),
            cfg.getint('occlusion', 'visible_slots', fallback=5),
            cfg.getint('occlusion', 'hidden_slots', fallback=10),
        ) if _occlusion_mode(cfg) != 'off' else (8, 13)
    )
    rl_buffer = ReplayBufferIQL(
        capacity=int(cfg.getfloat('buffer', 'capacity', fallback=800000)),
        seq_len=seq_len,
        obs_shape=replay_obs_shape,
        n_step=n_step,
        gamma=gamma,
        use_per=use_per,
        per_alpha=per_alpha,
        per_beta=per_beta_start,
        per_eps=per_eps,
        occlusion_mode=_occlusion_mode(cfg),
        belief_features=cfg.get(
            'occlusion', 'belief_features', fallback='full').strip().lower(),
    )
    logging.info(f"[{buffer_tag}] rl_buffer capacity={rl_buffer.capacity}, n_step={n_step}, per={use_per}")

    stats = TrainingStats(roll_window)
    logging.info("[STATS-RESET] TrainingStats reset for RL phase")

    # ========== IL/BC训练逻辑 ==========
    if not skip_il:
        logging.info("[IL-BC] Starting BC training phase...")
        il_result = _run_il_phase(
            train_cfg=train_cfg,  # 传递wrapper对象，但函数内部用cfg(RawConfigParser)
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
            env=env,
            explorer=explorer,
            policy=policy,
            device=device,
            batch_size=batch_size,
            replay_iql=rl_buffer,  # DoubleQ: 复用rl_buffer用于BC训练参数传递
            cfg=cfg,  # 传递RawConfigParser用于config读取
            direct_discrete=ARGS.discrete_mamba
        )
        logging.info(f"[IL-BC] BC training completed: success_rate={il_result.get('success_rate', 0.0):.1%}")
    else:
        logging.info("[IL-BC] Skipping BC training (checkpoint loaded)")
        il_result = {'success_rate': 0.0, 'il_trajectories': [], 'il_fail_trajectories': []}

    if ARGS.pretrain_only:
        _write_ablation_done(ARGS.outdir, {
            "variant": "orca_only",
            "description": "ORCA pretraining only / w/o online fine-tuning",
            "status": "done",
            "il_success_rate": float(il_result.get('success_rate', 0.0)) if isinstance(il_result, dict) else 0.0,
            "il_policy": os.path.join(os.path.abspath(ARGS.outdir), "il_policy.pth"),
        })
        logging.info("[ABLATION] --pretrain-only complete; exiting before online RL")
        try:
            tb_writer.close()
        except Exception:
            pass
        return

    if ARGS.skip_il_pretrain:
        logging.info("[ABLATION] Continuing directly to online RL from random initialization")
    elif not ARGS.discrete_mamba:
        # ====== ABLATION: 恢复有限制的IL prefill（原版逻辑） ======
        # 方案B已注释：不再注入所有IL数据，改用replay_warmup限制
        try:
            il_succ_trajs = il_result.get('il_trajectories', []) if isinstance(il_result, dict) else []
            il_fail_trajs = il_result.get('il_fail_trajectories', []) if isinstance(il_result, dict) else []
            if il_succ_trajs or il_fail_trajs:
                import numpy as _np
                prefill_added = 0
                prefill_skipped = 0
                # ABLATION: 恢复replay_warmup限制
                if cfg.has_option('train', 'replay_warmup'):
                    max_prefill = cfg.getint('train', 'replay_warmup', fallback=50000)
                else:
                    max_prefill = cfg.getint('buffer', 'replay_warmup', fallback=50000)

                def _inject_limited(trajs, limit):
                    """注入轨迹，有数量限制"""
                    nonlocal prefill_added, prefill_skipped
                    for traj_data, info in trajs:
                        if prefill_added >= limit:
                            break
                        try:
                            states, actions, rewards = traj_data
                            dones = _np.zeros(len(rewards), dtype=bool)
                            if len(rewards) > 0:
                                event_str = (_event_token(info) or '').lower()
                                bootstrap_on_timeout = cfg.getboolean('train', 'bootstrap_on_timeout', fallback=False)
                                dones[-1] = _done_for_bootstrap(event_str, bootstrap_on_timeout)
                            rl_buffer.push_episode({
                                'states': states,
                                'actions': actions,
                                'action_indices': np.zeros(len(rewards), dtype=np.int64),
                                'rewards': rewards,
                                'dones': dones
                            })
                            prefill_added += 1
                        except Exception:
                            prefill_skipped += 1
                            continue

                # ABLATION: 有限制地注入成功轨迹
                _inject_limited(il_succ_trajs, max_prefill)
                # 失败轨迹也有限制
                remaining = max(0, max_prefill - prefill_added)
                _inject_limited(il_fail_trajs, remaining)

                logging.info(
                    f"[SARL-PREFILL-LIMITED] IL episodes injected={prefill_added}, skipped={prefill_skipped}, "
                    f"buffer_size={len(rl_buffer)} (max_prefill={max_prefill})"
                )
        except Exception as e:
            logging.warning(f"[SARL-PREFILL-LIMITED] Failed to prefill buffer with IL data: {e}")

    if ARGS.skip_il_pretrain:
        logging.info("[CKPT-LOAD] IL warm-start intentionally disabled for w/o ORCA pretraining ablation")
    elif ARGS.discrete_mamba and ARGS.resume:
        logging.info("[CKPT-LOAD] Discrete-Mamba policy will be restored from the RL checkpoint")
    elif not getattr(policy, "_il_warm_started", False):
        logging.warning("[CKPT-LOAD] IL warm-start not detected, should have been done in IL phase")
    else:
        logging.info("[CKPT-LOAD] skipped (already warm-started with fresh IL ckpt)")

    start_episode = 1
    checkpoint = None
    if ARGS.resume:
        resume_path = ARGS.resume
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(ARGS.outdir, resume_path)

        if os.path.exists(resume_path):
            try:
                logging.info(f"[RESUME] Loading checkpoint from {resume_path}")
                checkpoint = safe_torch_load(resume_path, map_location=device)
                _assert_occlusion_resume(cfg, policy, checkpoint)
                if ARGS.discrete_mamba and str(checkpoint.get('algo', '')).lower() != 'discrete_mamba':
                    raise RuntimeError(
                        f"Refusing incompatible checkpoint for --discrete-mamba: algo={checkpoint.get('algo')!r}"
                    )

                if checkpoint.get('policy_state'):
                    resume_state = checkpoint['policy_state']
                    if any(k.startswith('_orig_mod.') for k in resume_state):
                        resume_state = {k.replace('_orig_mod.', '', 1): v for k, v in resume_state.items()}
                    strict_resume = _occlusion_mode(cfg) != 'off'
                    missing, unexpected = policy.load_state_dict(
                        resume_state,
                        strict=strict_resume,
                    )
                    if missing:
                        logging.warning(f"[RESUME] Missing keys: {missing}")
                    if unexpected:
                        logging.warning(f"[RESUME] Unexpected keys: {unexpected}")
                    logging.info("[RESUME] Policy state loaded")

                # DoubleQ: optimizer在RL阶段由Q网络内部创建与恢复

                start_episode = checkpoint.get('episode', 1) + 1
                if ARGS.start_episode:
                    start_episode = ARGS.start_episode
                    logging.info(f"[RESUME] Episode number overridden to {start_episode}")

                if checkpoint.get('stats_history'):
                    hist = checkpoint['stats_history']
                    stats.hist_s = hist.get('hist_s', [])
                    stats.hist_c = hist.get('hist_c', [])
                    stats.hist_t = hist.get('hist_t', [])
                    stats.hist_r = hist.get('hist_r', [])
                    logging.info(f"[RESUME] Statistics history loaded: {len(stats.hist_s)} episodes")

                logging.info(f"[RESUME] ✅ Successfully resumed from episode {start_episode}")

            except Exception as e:
                logging.error(f"[RESUME] Failed to load checkpoint: {e}")
                if ARGS.discrete_mamba or _occlusion_mode(cfg) != 'off':
                    raise
                logging.info("[RESUME] Starting fresh training...")
                start_episode = 1
        else:
            logging.error(f"[RESUME] Checkpoint file not found: {resume_path}")
            if ARGS.discrete_mamba:
                raise FileNotFoundError(resume_path)
            logging.info("[RESUME] Starting fresh training...")
            start_episode = 1

    # PPO: 不需要warmup buffer（每次更新后清空，无需预热）

    # 保持BC学到的低噪声，不强制clamp
    # _clamp_policy_std(policy, min_std=0.05, max_std=0.10)  # 已禁用：让初始std=0.01生效

    # 现在执行 torch.compile：所有 checkpoint 加载已完成
    # 延迟编译确保 state_dict key 匹配，BC 权重正确加载
    if compile_policy_later:
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
            logging.info("[JIT] 4090 UNLEASHED: torch.compile FORCED ON! (reduce-overhead mode, after checkpoint loads)")
            # 重新绑定到robot，确保采样使用编译后的模型
            bind_policy(robot, policy, env)
        except Exception as e:
            logging.info(f"[JIT] torch.compile skipped: {e}")

    # ========== 关闭 RL Curriculum (todo.md修复) ==========
    # 原因：BC模型只在5人环境训练，RL应该先在相同环境超越BC（>80%）
    # 再增加难度。现在curriculum太激进，导致RL在4-20人随机变化中困顿
    # 修复：回归标准5人圆桌环境
    env.use_rl_curriculum = False
    logging.info("[CURRICULUM] DISABLED - Focusing on standard environment")
    logging.info("[CURRICULUM] RL will use baseline 5-human circle environment (Nh=5, R=4)")

    best_succ = _run_rl_phase(
        cfg, env, policy, explorer, rl_buffer, stats, plotter, device,
        episodes, batch_size, save_every, ARGS, tb_writer, train_step_counter,
        start_episode, metrics_plotter, trajectory_file,
        resume_checkpoint=checkpoint
    )

    logging.info(f"[MAIN-COMPLETE] Training finished with best evaluation success: {best_succ:.3f}")
    if ARGS.discrete_mamba:
        temporal_backbone = cfg.get('mamba', 'temporal_backbone', fallback='mamba').strip().lower()
        _write_ablation_done(ARGS.outdir, {
            "variant": f"{temporal_backbone}_direct_q",
            "description": f"{temporal_backbone.upper()} backbone with direct discrete action prediction",
            "status": "done",
            "best_success": float(best_succ),
            "temporal_backbone": temporal_backbone,
            "latest_checkpoint": _latest_rl_checkpoint(ARGS.outdir),
            "best_checkpoint": os.path.join(os.path.abspath(ARGS.outdir), "best_model.pth"),
        })
        logging.info(f"[DIRECT-Q-{temporal_backbone.upper()}] Completion marker written")
    if ARGS.skip_il_pretrain:
        _write_ablation_done(ARGS.outdir, {
            "variant": "no_orca_pretrain",
            "description": "w/o ORCA pretraining, online training from scratch",
            "status": "done",
            "best_success": float(best_succ),
            "latest_checkpoint": _latest_rl_checkpoint(ARGS.outdir),
        })
        logging.info("[ABLATION] w/o ORCA pretraining marker written")

    if hasattr(explorer, 'trajectory_logger') and explorer.trajectory_logger:
        try:
            explorer.trajectory_logger.close()
            logging.info("[TRAJECTORY-LOG] Trajectory file closed successfully")
        except Exception as e:
            logging.warning(f"[TRAJECTORY-LOG] Failed to close trajectory file: {e}")

    tb_writer.close()
    logging.info("[TENSORBOARD] Events file closed successfully")

if __name__ == '__main__':
    main()
