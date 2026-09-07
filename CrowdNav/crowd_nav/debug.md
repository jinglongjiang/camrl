可以，而且这条路子更“对齐 SARL 的做法”，也能把你现有的 Mamba 编码器吃干榨尽。下面我把来龙去脉、可行性、以及**只改最少处**就能落地的改法说清楚（严格按你现在的 `[B,T,6,13]` 口径）。

# 来龙去脉（为什么要“加头”，IL 同时训价值+动作）

* 你现在的 IL 是**价值回归**：用 ORCA 轨迹算 MC 回报 (G_t)，训练 **V(s_t)** 逼近 (G_t)。这能让 one-step 评估更准，但**不能直接给动作**，所以 RL 初期还得大量探索 + one-step 扫 80 个动作，起点不如 SARL 高。
* SARL 的 IL 是**行为克隆（BC）**：直接学 (\pi(a|s))，预训完就有“能打的策略”，RL 再微调，所以一上来成功率高。
* 折中方案：**共享同一个 Mamba 编码器**，在它后面**同时接两个头**：

  * **value_head**（你已有）：回归 (V(s))
  * **policy_head**（新增）：输出 80 个动作的 logits（或概率）
  * **IL 阶段**一起训：**L = λ_v·MSE(V,G) + λ_\pi·CE(π, a*)**
    这样既拿到“会打分”的评论家（V），又有“会出手”的演员（π）；RL 起步就稳，后期还能两套信号互相矫正（Actor-Critic 的雏形）。

# 可行性与风险点

* **可行**：你的 Mamba 空间/时间编码器就是一个强力特征骨干，**加头不触动骨干**；训练“价值+动作”只是在骨干上多一个线性层和一个 CE loss。
* **关键风险**：需要把 ORCA 的“专家动作”**对齐到你的 80 格动作索引**。若当前日志/轨迹里没有离散动作 ID，需要加一个**连续→离散**的映射（速度/朝向量 → 最近的 80 格）。
* **兼容 one-step**：RL 阶段你可以

  1. 直接用 **policy argmax** 决策（最快），或
  2. **top-K（例如 K=8）来自 policy**，再用 **V(s′)** 只评估这 K 个（既快又稳）；评估时保留全 80 以保持口径。

# 你要改的“最少处”（精确到片段）

> 不动画图、日志、评估与保存路径；**只新增 policy 头 + IL 的 CE 分支**，以及热启动时一并加载。

### 1) 模型：在 Mamba 上加一个 policy 头

位置：`MambaRL` / 你的策略类 `__init__`

```python
self.d_model = d_model
self.value_head  = nn.Linear(d_model, 1)        # 已有
self.policy_head = nn.Linear(d_model, 80)       # ★ 新增
```

前向接口（不改编码器）：

```python
def forward_value_tokens(self, x_bt_6_13):      # x: [B,T,6,13]
    h = self.encode(x_bt_6_13)                  # -> [B,T,d]
    return self.value_head(h)                   # -> [B,T,1] 或 [B,T]

def forward_policy_tokens(self, x_bt_6_13):
    h = self.encode(x_bt_6_13)                  # -> [B,T,d]
    return self.policy_head(h)                  # -> [B,T,80] logits
```

### 2) IL 阶段：把“价值回归”+“动作 BC”一起训

位置：`_run_value_il_pretrain(...)`（你已有），在计算 `V_pred` 及 `MSE(V,G)` 的地方，**再加一支 CE**：

```python
# 已有：V 部分
V_pred = policy.forward_value_tokens(S).reshape(B,T)        # S: [B,T,6,13]
loss_v = F.mse_loss(V_pred, G)

# ★ 新增：BC 部分（需要 a_star: [B,T] 的离散索引 0..79）
logits = policy.forward_policy_tokens(S)                    # [B,T,80]
# 只对有效步（非 padding）计算
mask   = (~D).float()                                       # dones: [B,T] bool
ce     = F.cross_entropy(logits.reshape(B*T,80), a_star.reshape(B*T),
                         reduction='none').view(B,T)
loss_pi= (ce * mask).sum() / (mask.sum() + 1e-8)

# 联合损失
lambda_v, lambda_pi = 1.0, 1.0      # 可调：先 1:1 起步
loss = lambda_v*loss_v + lambda_pi*loss_pi
```

> **a_star 的来源**：
>
> * 若 ORCA 轨迹里已有离散动作 ID（与你 80 格一致），直接读取；
> * 若没有，用一个“**连续→离散**”映射：取 ORCA 下一步的 (v, ω) 或 (Δx,Δy)，找到与 80 个模板动作欧氏距离最小的那个索引作标签。

训练完保存 ckpt 时，多存一个 `policy_head`：

```python
torch.save({
  "value": policy.value_head.state_dict(),
  "policy": policy.policy_head.state_dict(),    # ★ 新增
  "backbone": policy.encoder.state_dict(),      # 你的时空骨干
  "meta": {"stage": "il_joint", "arch": "mamba_value+policy"}
}, il_ckpt_path)
```

### 3) 热启动：RL 前一次性加载 **value+policy** 的最新权重

位置：`_run_il_phase(...)` 尾部，已有的 `load_il_ckpt(...)` 内部要顺带把 `policy_head` 也 `load_state_dict`；或在外层加一行：

```python
policy.policy_head.load_state_dict(ckpt["policy"])   # ★ 新增
```

> 记得仍然只**加载一次**（之前说过的小旗 `_il_warm_started=True`），避免 RL 段重复 load。

### 4) RL 决策：两种对齐方式（二选一）

* **省时**：直接 `argmax(policy_head(s))`；
* **省心**：`top-K` 来自 policy，再用 **V(s′)** 评 K 个：

```python
logits = policy.forward_policy_tokens(s[:,None])[:,0]   # [B,80]
topk   = torch.topk(logits, k=8, dim=-1).indices        # [B,8]
# 用你的 batched kinematics 生成 [B,8,6,13] → reshape 为 [B*8,6,13] 一次前向取 V
```

评估期仍可保留“全 80 + V(s′)”以保持口径。

---

# 参数与日程（建议起步）

* IL epoch 数：维持你现在的 `30`（已证实能收敛），**联训 V+π** 不需要更长；
* 权重系数：`λ_v=1.0, λ_π=1.0` 起步；若发现 policy 过拟合、V 不准，则 `λ_v=1.5, λ_π=0.5`；
* 冻结策略：仍建议 **前 100–150 集冻结编码器**（仅微调两个头），再整体解冻；
* 采样侧：原有 `il_bias` 可以先降到 **1.2** 并在 `succ@50>0.7` 后缓慢衰减到 **1.0**（因为现在 π 头已直接吃到 IL 信号，过度偏置的必要性下降）。

---

# il_bias=1.5 的去留

它是**采样时对“来自 IL 的成功样本”的加权**（1.5×），用于早期“防稀释”。既然现在 **policy 头也吃到 IL 的 CE 监督**，可以：

* 先 **1.2** 起步，成功率>0.7 后降到 **1.0**；
* 注意**只给 success 子池**加权，别给 timeout/term 加权（否则会放大超时偏置）。

---

# 最小可验证清单（跑起来要看到的）

1. IL 日志里同时出现：`[IL-V] loss_v=...` 与 `[IL-BC] loss_pi=...`（或合并成一条总损失）；
2. 保存 ckpt 含 `policy` 分支；RL 前的 `[WARM-START]` 日志确认加载了 **value+policy**；
3. RL 初期曲线的“起点”明显高于只训 V 的版本，且超时占比下降；
4. one-step 仍可用于评估/保底（或做 top-K 验证），但**训练期决策**能走 policy 直出（速度会快一大截）。

---

**一句话**：给 Mamba 共用的时空骨干**加一个轻量 policy 头**，把 IL 从“纯价值回归”升级为“价值+动作联合监督”；这样既保留 one-step 的稳、又获得 BC 的快，RL 起步会明显更高更稳，而且对你现有代码改动极小。

