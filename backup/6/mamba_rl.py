import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot

# ===== 默认维度（可被 config 覆盖）=====
ROBOT_FEAT_DIM = 9
HUMAN_FEAT_DIM = 5
HUMAN_NUM = 5
STATE_DIM = ROBOT_FEAT_DIM + HUMAN_FEAT_DIM * HUMAN_NUM
EPS = 1e-6


# ---------------- Utilities ----------------
def _split_state(x, robot_dim, human_dim, human_num):
    """
    x: [B, D] 或 [B, T, D]
    return: robot_x [B,1,Fr], human_x [B,N,Fh], mask [B,N]
    """
    if x.dim() == 3:
        x = x[:, -1, :]
    b = x.size(0)
    robot = x[:, :robot_dim].view(b, 1, robot_dim)
    human = x[:, robot_dim:]

    need = human_num * human_dim
    orig = human.size(1)  # 记录补零前的人类维度
    if orig < need:
        pad = x.new_zeros(b, need - orig)
        human = torch.cat([human, pad], dim=1)
        valid = torch.zeros(b, human_num, dtype=torch.bool, device=x.device)
        fill_n = min(orig // human_dim, human_num)  # 用原始维度算有效人数
        if fill_n > 0:
            valid[:, :fill_n] = True
    else:
        valid = torch.ones(b, human_num, dtype=torch.bool, device=x.device)

    if human.size(1) > need:
        human = human[:, :need]
    human = human.view(b, human_num, human_dim)
    return robot, human, valid


def _sort_by_distance(robot_xy, human_xy, mask):
    dist = torch.norm(human_xy - robot_xy.unsqueeze(1), dim=-1)  # [B,N]
    dist = dist + (~mask).to(dist.dtype) * 1e6                   # 明确转 dtype，安全叠加
    idx = torch.argsort(dist, dim=1)
    return idx


def orthogonal_init(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------- Blocks ----------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, robot_tok, human_tok, mask):
        """
        robot_tok: [B,1,H]
        human_tok: [B,N,H]
        mask:      [B,N]  bool，True 表示有效
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()

        q = self.q(robot_tok)                        # [B,1,H]
        k = self.k(human_tok)                        # [B,N,H]
        v = self.v(human_tok)                        # [B,N,H]

        logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(max(k.size(-1), 1))  # [B,1,N]
        # fp16 安全的掩码：用 dtype 的最小值当作 -inf
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask.unsqueeze(1), neg_inf)

        # masked softmax -> 权重在无效位为 0；若全无有效项，归一化后仍为 0
        w = torch.softmax(logits, dim=-1)                               # [B,1,N]
        w = w * mask.unsqueeze(1).to(w.dtype)                           # 置零无效位置
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)            # [B,1,1]
        w = w / w_sum                                                   # 全无有效项 -> 全 0（被 clamp 成 0/1e-6）

        pooled = torch.matmul(w, v).squeeze(1)       # [B,H]
        return self.fc(pooled)


class MambaStateEncoder(nn.Module):
    """
    robot+humans -> token 序列 -> Mamba -> robot_token + 人群摘要
    """
    def __init__(self, robot_feat_dim, human_feat_dim, human_num,
                 hidden_dim=64, n_blocks=4, d_state=None, conv_dim=4, expand=2,
                 dropout=0.0, topk=None):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise ImportError("缺少 mamba_ssm，请先安装：pip install mamba-ssm") from e

        self.robot_feat_dim = robot_feat_dim
        self.human_feat_dim = human_feat_dim
        self.human_num = human_num
        self.hidden_dim = hidden_dim
        self.topk = topk  # 只编码最近 topk 个行人（None 表示全用）

        if d_state is None:
            d_state = hidden_dim

        self.robot_enc = nn.Sequential(
            nn.Linear(robot_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.human_enc = nn.Sequential(
            nn.Linear(human_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=d_state, d_conv=conv_dim, expand=expand)
            for _ in range(n_blocks)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_blocks)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = AttentionPooling(hidden_dim)

    def forward(self, x):
        # x: [B,D] 或 [B,T,D]
        robot_x, human_x, mask = _split_state(
            x, self.robot_feat_dim, self.human_feat_dim, self.human_num
        )  # [B,1,Fr], [B,N,Fh], [B,N]

        # 基于位置排序（假设前2维是 px,py）
        rxy = robot_x[..., :2].squeeze(1)       # [B,2]
        hxy = human_x[..., :2]                  # [B,N,2]
        idx = _sort_by_distance(rxy, hxy, mask) # [B,N]
        b, n, fh = human_x.size()
        batch_idx = torch.arange(b, device=x.device).unsqueeze(-1)
        human_x = human_x[batch_idx, idx, :]
        mask = mask[batch_idx, idx]

        # 只取最近 topk（其余置零并 mask=False）
        if self.topk is not None and self.topk < n:
            keep = self.topk
            drop = n - keep
            keep_x = human_x[:, :keep, :]
            drop_x = human_x.new_zeros(b, drop, fh)
            human_x = torch.cat([keep_x, drop_x], dim=1)
            keep_m = mask[:, :keep]
            drop_m = torch.zeros_like(mask[:, keep:], dtype=torch.bool)
            mask = torch.cat([keep_m, drop_m], dim=1)

        r_tok = self.robot_enc(robot_x)               # [B,1,H]
        h_tok = self.human_enc(human_x)               # [B,N,H]
        seq = torch.cat([r_tok, h_tok], dim=1)        # [B,1+N,H]

        for blk, ln in zip(self.blocks, self.norms):
            seq = seq + blk(ln(seq))
            seq = self.dropout(seq)

        r_out = seq[:, :1, :]                         # [B,1,H]
        h_out = seq[:, 1:, :]                         # [B,N,H]
        pooled = self.pool(r_out, h_out, mask)        # [B,H]
        fused = torch.cat([r_out.squeeze(1), pooled], dim=-1)  # [B,2H]
        return fused, r_out.squeeze(1), h_out, mask


# ---------------- Actor / Critic ----------------
class TanhGaussianActor(nn.Module):
    """
    SAC 兼容演员头：两层 MLP + tanh 高斯，带稳定的 log_prob 计算。
    """
    def __init__(self, in_dim, act_dim=2, hidden_dim=128, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        this_log_std = nn.Linear(hidden_dim, act_dim)
        self.log_std = this_log_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        orthogonal_init(self.mu, gain=0.01)
        orthogonal_init(self.log_std, gain=0.01)

    def forward(self, h):
        z = self.net(h)
        mu = self.mu(z)
        log_std = self.log_std(z).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, h):
        mu, log_std = self.forward(h)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()                 # reparameterization
        a = torch.tanh(z)
        # tanh 修正（数值安全）
        logp = normal.log_prob(z) - torch.log(torch.clamp(1 - a.pow(2), min=1e-6))
        logp = logp.sum(-1, keepdim=True)   # [B,1]
        a_det = torch.tanh(mu)
        return a, logp, a_det


class QCritic(nn.Module):
    """
    双 Q 之一：两层 MLP
    """
    def __init__(self, in_dim, act_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        orthogonal_init(self.net[-1], gain=0.01)

    def forward(self, h, a):
        x = torch.cat([h, a], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------- Policy ----------------
class MambaRL(Policy):
    """
    共享编码器 + TanhGaussian Actor + 双 Q
    """
    def __init__(self, config, device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None \
            else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.trainable = True
        self.multiagent_training = True

        # ---- 读取配置 ----
        if hasattr(config, "get"):
            try:
                self.kinematics = config.get("action_space", "kinematics", fallback="holonomic")
            except Exception:
                self.kinematics = "holonomic"

            sec = "mamba"
            _gi = lambda opt, d: config.getint(sec, opt, fallback=d)
            _gf = lambda opt, d: config.getfloat(sec, opt, fallback=d)
            _gs = lambda opt, d: config.get(sec, opt, fallback=d)

            hidden_dim     = _gi("hidden_dim", 64)
            n_blocks       = _gi("n_blocks", 4)
            state_dim      = _gi("state_dim", STATE_DIM)
            conv_dim       = _gi("conv_dim", 4)
            expand         = _gi("expand", 2)
            robot_fd       = _gi("robot_feat_dim", ROBOT_FEAT_DIM)
            human_fd       = _gi("human_feat_dim", HUMAN_FEAT_DIM)
            human_num      = _gi("human_num", HUMAN_NUM)
            topk           = _gi("topk", human_num)  # <= human_num
            dropout        = _gf("dropout", 0.0)
            d_state_raw    = _gs("d_state", "").strip()
            d_state        = int(d_state_raw) if d_state_raw != "" else None

            self.gamma     = _gf("gamma", 0.99)
            self.action_scale = _gf("action_scale", 1.0)
            self.multiagent_training = bool(_gs("multiagent_training", "true").strip().lower() in ("1","true","yes"))
            self.env_stochastic = bool(_gs("env_stochastic", "true").strip().lower() in ("1","true","yes"))
        else:
            self.kinematics = getattr(config, "kinematics", "holonomic")
            hidden_dim     = getattr(config, "hidden_dim", 64)
            n_blocks       = getattr(config, "n_blocks", 4)
            state_dim      = getattr(config, "state_dim", STATE_DIM)
            conv_dim       = getattr(config, "conv_dim", 4)
            expand         = getattr(config, "expand", 2)
            robot_fd       = getattr(config, "robot_feat_dim", ROBOT_FEAT_DIM)
            human_fd       = getattr(config, "human_feat_dim", HUMAN_FEAT_DIM)
            human_num      = getattr(config, "human_num", HUMAN_NUM)
            topk           = getattr(config, "topk", human_num)
            dropout        = getattr(config, "dropout", 0.0)
            d_state        = getattr(config, "d_state", None)
            self.gamma     = getattr(config, "gamma", 0.99)
            self.action_scale = getattr(config, "action_scale", 1.0)
            self.multiagent_training = getattr(config, "multiagent_training", True)
            self.env_stochastic = getattr(config, "env_stochastic", True)

        self.epsilon = 0.0

        # ---- 模块 ----
        self.encoder = MambaStateEncoder(
            robot_feat_dim=robot_fd, human_feat_dim=human_fd, human_num=human_num,
            hidden_dim=hidden_dim, n_blocks=n_blocks, d_state=d_state,
            conv_dim=conv_dim, expand=expand, dropout=dropout, topk=topk
        ).to(self.device)

        fused_dim = hidden_dim * 2
        head_hid = max(128, hidden_dim)  # 稍加容量
        self.actor = TanhGaussianActor(fused_dim, act_dim=2, hidden_dim=head_hid).to(self.device)
        self.q1 = QCritic(fused_dim, act_dim=2, hidden_dim=head_hid).to(self.device)
        self.q2 = QCritic(fused_dim, act_dim=2, hidden_dim=head_hid).to(self.device)

        # 观测归一化：运行统计（mean/var/count）
        self.obs_mean = torch.zeros(state_dim, dtype=torch.float32, device=self.device)
        self.obs_var  = torch.ones(state_dim, dtype=torch.float32, device=self.device)
        self.obs_count = 1e-4

        self.phase = None
        self.robot_v_pref = 1.0

    # ---------- phases ----------
    def set_phase(self, phase):
        self.phase = phase
        train = (phase == 'train')
        self.encoder.train(train)
        self.actor.train(train)
        self.q1.train(train)
        self.q2.train(train)

    def set_robot(self, robot):
        self.robot_v_pref = getattr(robot, 'v_pref', 1.0)

    def set_device(self, device):
        self.device = torch.device(device)
        self.encoder.to(self.device)
        self.actor.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.obs_mean = self.obs_mean.to(self.device)
        self.obs_var  = self.obs_var.to(self.device)

    def set_epsilon(self, epsilon: float):
        self.epsilon = float(epsilon)

    # ---------- obs norm ----------
    def _norm_obs(self, obs_tensor):
        mean = self.obs_mean
        std = torch.sqrt(self.obs_var) + 1e-8
        return (obs_tensor - mean) / std

    @torch.no_grad()
    def update_obs_rms(self, batch_np):
        x = torch.as_tensor(batch_np, dtype=torch.float32, device=self.device)
        b = x.size(0)
        if b == 0:
            return
        n0 = self.obs_count
        n1 = n0 + b

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        delta = batch_mean - self.obs_mean
        new_mean = self.obs_mean + delta * (b / n1)

        m2_old = self.obs_var * n0
        m2_batch = batch_var * b
        m2_delta = (delta * delta) * (n0 * b / n1)
        m2_new = m2_old + m2_batch + m2_delta
        new_var = m2_new / n1

        self.obs_mean = new_mean
        self.obs_var = new_var
        self.obs_count = float(n1)

    # ---------- core enc/act ----------
    def _encode(self, obs_tensor):
        fused, _, _, _ = self.encoder(obs_tensor)
        return fused

    @torch.no_grad()
    def sample_action(self, obs_np):
        """
        训练用：返回 action[-1,1]、log_prob[B,1]、deterministic[-1,1]
        """
        if isinstance(obs_np, np.ndarray):
            obs_t = torch.from_numpy(obs_np).to(self.device)
        else:
            obs_t = obs_np.to(self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        obs_t = self._norm_obs(obs_t)
        h = self._encode(obs_t)
        a, logp, a_det = self.actor.sample(h)
        return a.cpu().numpy(), logp.cpu().numpy(), a_det.cpu().numpy()

    @torch.no_grad()
    def q_values(self, obs_np, act_np):
        if isinstance(obs_np, np.ndarray):
            obs_t = torch.from_numpy(obs_np).to(self.device)
        else:
            obs_t = obs_np.to(self.device)
        if isinstance(act_np, np.ndarray):
            a_t = torch.from_numpy(act_np).to(self.device)
        else:
            a_t = act_np.to(self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0); a_t = a_t.unsqueeze(0)
        obs_t = self._norm_obs(obs_t)
        h = self._encode(obs_t)
        return self.q1(h, a_t).cpu().numpy(), self.q2(h, a_t).cpu().numpy()

    # ---------- env-facing ----------
    def predict(self, obs):
        """
        环境调用：
        - 训练期默认使用随机采样（与 SAC 一致），可通过 env_stochastic=False 改为确定性。
        - 若仍想用 epsilon 额外噪声，可保留 self.epsilon>0（不推荐 SAC 下叠加）。
        """
        def to_array(state):
            from crowd_sim.envs.utils.state import JointState
            if hasattr(state, "to_array"):
                return state.to_array().astype(np.float32)
            if "JointState" in str(type(state)):
                try:
                    return state.to_array().astype(np.float32)
                except Exception:
                    pass
            if isinstance(state, (list, tuple)):
                parts = [to_array(s) for s in state]
                return np.concatenate(parts).astype(np.float32)
            return np.asarray(state, dtype=np.float32)

        obs_arr = to_array(obs)
        obs_t = torch.from_numpy(obs_arr).to(self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        obs_t = self._norm_obs(obs_t)

        with torch.no_grad():
            h = self._encode(obs_t)
            if self.phase == 'train' and self.env_stochastic:
                act, _, _ = self.actor.sample(h)
                act = act[0].cpu().numpy()
            else:
                mu, _ = self.actor(h)
                act = torch.tanh(mu)[0].cpu().numpy()

        if self.phase == 'train' and self.epsilon > 0:
            act = np.clip(act + np.random.normal(0, self.epsilon, size=act.shape), -1.0, 1.0)

        vx, vy = float(act[0]), float(act[1])
        scale = self.robot_v_pref * self.action_scale
        vx *= scale; vy *= scale

        if self.kinematics == 'holonomic':
            return ActionXY(vx, vy)
        else:
            return ActionRot(vx, vy)

    def transform(self, state):
        return state

    # ---------- save/load ----------
    def get_state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "obs_count": self.obs_count,
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.actor.load_state_dict(state_dict["actor"])
        self.q1.load_state_dict(state_dict["q1"])
        self.q2.load_state_dict(state_dict["q2"])
        if "obs_mean" in state_dict:
            self.obs_mean = state_dict["obs_mean"].to(self.device)
            self.obs_var  = state_dict["obs_var"].to(self.device)
            self.obs_count = float(state_dict.get("obs_count", 1e-4))

    def save_model(self, path):
        torch.save(self.get_state_dict(), path)

    def load_model(self, path):
        sd = torch.load(path, map_location=self.device)
        self.load_state_dict(sd)

    def get_model(self):
        return self.encoder

