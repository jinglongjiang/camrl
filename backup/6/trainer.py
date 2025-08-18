# crowd_nav/utils/trainer.py
import copy
import torch
import torch.nn as nn

class Trainer:
    """
    SAC + 双Q + 可选 BC 正则（SACfD 风格）
    - AMP 友好（unscale 后裁剪）
    - 前向图分离，避免“second backward”错误
    - TD 目标与 Q 输出对齐到 [B]，消除 size 警告
    依赖 policy 暴露：
      encoder/actor/q1/q2、_norm_obs(tensor)、_encode(tensor)
      update_obs_rms(np.ndarray)（可选）、action_scale（可选）
    """
    def __init__(self, policy, device,
                 gamma=0.99, tau=0.005,
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 batch_size=512,
                 target_entropy: float = -3.0,  # 动作维=2，-2~-3 常用
                 use_amp: bool = True,
                 grad_clip: float = 1.0,
                 v_pref_idx: int = 6):          # 你环境里 v_pref 的索引（与 train.py 保持一致）
        self.policy = policy
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.use_amp = bool(use_amp)
        self.grad_clip = float(grad_clip) if grad_clip is not None else None
        self.v_pref_idx = int(v_pref_idx)
        self.target_entropy = float(target_entropy)

        # 模块
        self.actor = policy.actor
        self.q1 = policy.q1
        self.q2 = policy.q2

        # 目标网络
        self.q1_targ = copy.deepcopy(self.q1).to(self.device)
        self.q2_targ = copy.deepcopy(self.q2).to(self.device)
        for p in self.q1_targ.parameters(): p.requires_grad_(False)
        for p in self.q2_targ.parameters(): p.requires_grad_(False)

        # 优化器
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_q1    = torch.optim.Adam(self.q1.parameters(),    lr=lr_critic)
        self.opt_q2    = torch.optim.Adam(self.q2.parameters(),    lr=lr_critic)

        # 温度 α（自动调节）
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    # ---------- helpers ----------
    def _obs_norm(self, s: torch.Tensor) -> torch.Tensor:
        return self.policy._norm_obs(s) if hasattr(self.policy, "_norm_obs") else s

    def _encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.policy._encode(s) if hasattr(self.policy, "_encode") else s

    def _to_t(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    @torch.no_grad()
    def _polyak(self, src: nn.Module, tgt: nn.Module, tau: float):
        for p, pt in zip(src.parameters(), tgt.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)

    # ---------- sampling ----------
    def _sample_rl_batch(self, rl_buf, n):
        if n <= 0 or len(rl_buf) == 0:
            return None
        s, a, r, s2, d = rl_buf.sample(n)  # r:(B,1)  d:(B,)
        return (self._to_t(s),
                self._to_t(a),
                self._to_t(r, dtype=torch.float32),
                self._to_t(s2),
                self._to_t(d, dtype=torch.float32))  # d→float，后续算术更方便

    def _sample_demo_batch(self, exp_buf, n):
        if exp_buf is None or n <= 0 or len(exp_buf) == 0:
            return None
        s, a = exp_buf.sample(n)
        return (self._to_t(s), self._to_t(a))

    # 物理动作 → [-1,1]（除以 v_pref * action_scale）
    def _normalize_actions(self, s_t: torch.Tensor, a_phys: torch.Tensor) -> torch.Tensor:
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        v_pref = s_t[:, self.v_pref_idx:self.v_pref_idx+1].detach().clamp(min=1e-6)
        scale = v_pref * float(getattr(self.policy, 'action_scale', 1.0))
        return torch.clamp(a_phys / scale, -1.0, 1.0)

    def set_learning_rate(self, lr: float):
        for opt in (self.opt_actor, self.opt_q1, self.opt_q2, self.opt_alpha):
            for g in opt.param_groups:
                g['lr'] = lr

    # ---------- optimize ----------
    def optimize_batch(self, rl_buf, exp_buf=None, updates=1,
                       p_demo=0.0, lambda_bc=0.0, use_q_filter=True):
        """
        rl_buf: ReplayMemory(state, action(phys), reward, next_state, done)
        exp_buf: ExpertReplayMemory(state, expert_action(phys))
        p_demo:  每次更新从专家缓冲采样的比例（0~1）
        lambda_bc: BC 正则权重
        """
        meter = {}
        for _ in range(int(updates)):
            B = self.batch_size
            B_demo = int(B * float(p_demo))
            B_rl   = max(1, B - B_demo)

            rl = self._sample_rl_batch(rl_buf, B_rl)
            if rl is None:
                continue
            s_rl, a_rl_phys, r_rl, s2_rl, d_rl = rl         # (B,D),(B,2),(B,1),(B,D),(B,)

            # 统一把物理动作归一到 [-1,1]
            a_rl = self._normalize_actions(s_rl, a_rl_phys)  # (B,2)

            # 在线更新观测统计（若 policy 支持）
            if hasattr(self.policy, "update_obs_rms"):
                with torch.no_grad():
                    self.policy.update_obs_rms(s_rl.detach().cpu().numpy())
                    self.policy.update_obs_rms(s2_rl.detach().cpu().numpy())

            # 规范化观测
            s_rl_n  = self._obs_norm(s_rl)
            s2_rl_n = self._obs_norm(s2_rl)

            # ------- Critic（图 A + 图 B）-------
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                with torch.no_grad():
                    h2 = self._encode(s2_rl_n)                           # 图 A
                    a2, logp2, _ = self.policy.actor.sample(h2)          # (B,2),(B,1)
                    q1_t = self.q1_targ(h2, a2)                          # (B,)
                    q2_t = self.q2_targ(h2, a2)                          # (B,)
                    q_min = torch.min(q1_t, q2_t) - self.log_alpha.exp() * logp2.squeeze(-1)
                    y = r_rl.squeeze(-1) + (1.0 - d_rl) * self.gamma * q_min  # (B,)

                h_q = self._encode(s_rl_n)                                # 图 B
                q1 = self.q1(h_q, a_rl)                                   # (B,)
                q2 = self.q2(h_q, a_rl)                                   # (B,)
                loss_q = torch.nn.functional.mse_loss(q1, y.detach()) + \
                         torch.nn.functional.mse_loss(q2, y.detach())

            self.opt_q1.zero_grad(set_to_none=True)
            self.opt_q2.zero_grad(set_to_none=True)
            self.scaler.scale(loss_q).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.opt_q1)
                self.scaler.unscale_(self.opt_q2)
                nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
                nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
            self.scaler.step(self.opt_q1)
            self.scaler.step(self.opt_q2)
            self.scaler.update()

            # ------- Actor（图 C + 可选图 D）-------
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                h_pi = self._encode(self._obs_norm(s_rl))                 # 图 C
                a_pi, logp_pi, _ = self.policy.actor.sample(h_pi)         # (B,2),(B,1)
                q1_pi = self.q1(h_pi, a_pi); q2_pi = self.q2(h_pi, a_pi)  # (B,),(B,)
                q_pi_min = torch.min(q1_pi, q2_pi)                         # (B,)
                sac_actor = (self.log_alpha.exp() * logp_pi.squeeze(-1) - q_pi_min).mean()

                # BC 正则（来自 exp_buf，带可选 Q-filter）
                bc_loss = torch.zeros((), device=self.device)
                if exp_buf is not None and B_demo > 0 and float(lambda_bc) > 0.0:
                    demo = self._sample_demo_batch(exp_buf, B_demo)
                    if demo is not None:
                        s_demo, a_demo_phys = demo
                        s_demo_n = self._obs_norm(s_demo)
                        h_demo   = self._encode(s_demo_n)                 # 图 D
                        a_pi_demo, _, _ = self.policy.actor.sample(h_demo)
                        a_demo_norm = self._normalize_actions(s_demo, a_demo_phys)

                        if use_q_filter:
                            with torch.no_grad():
                                q_pi_demo  = torch.min(self.q1(h_demo, a_pi_demo),
                                                       self.q2(h_demo, a_pi_demo))          # (B_demo,)
                                q_exp_demo = torch.min(self.q1(h_demo, a_demo_norm),
                                                       self.q2(h_demo, a_demo_norm))        # (B_demo,)
                                mask = (q_exp_demo > q_pi_demo).float().unsqueeze(-1)       # (B_demo,1)
                        else:
                            mask = 1.0

                        bc_loss = ((a_pi_demo - a_demo_norm) ** 2 * mask).mean()

                loss_actor = sac_actor + float(lambda_bc) * bc_loss

            self.opt_actor.zero_grad(set_to_none=True)
            self.scaler.scale(loss_actor).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.opt_actor)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.scaler.step(self.opt_actor)
            self.scaler.update()

            # ------- 温度 α（单独图，logp_pi.detach()）-------
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                alpha_loss = -(self.log_alpha * (logp_pi.squeeze(-1).detach() + self.target_entropy)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.opt_alpha)
            self.scaler.update()

            # ------- 目标网络 -------
            self._polyak(self.q1, self.q1_targ, self.tau)
            self._polyak(self.q2, self.q2_targ, self.tau)

            meter = {
                "loss_q": float(loss_q.item()),
                "loss_actor": float(loss_actor.item()),
                "bc_loss": float(bc_loss.item()) if torch.is_tensor(bc_loss) else 0.0,
                "alpha": float(self.log_alpha.exp().item()),
            }
        return meter

