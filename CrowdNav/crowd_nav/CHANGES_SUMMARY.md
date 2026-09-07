# 回归简单范式 - 修改总结

## 已完成修改

### 1. train.config 超参数调整

**梯度强度提升** (lr × 60, updates × 6.7):
- learning_rate: 5e-6 → 3e-4 (SARL 0.001的0.3倍，适配AdamW)
- updates_per_ep: 6 → 40 (SARL 100的0.4倍，适配大batch=512)

**探索率降低**:
- epsilon_start: 0.5 → 0.35 (BC已91%，无需暴力探索)
- epsilon_decay: 4000 → 2000 ep (快速收敛)

**IL数据退场**:
- il_ratio_start: 0.4 → 0.0 (SARL范式，RL阶段纯RL)
- il_ratio_end: 0.4 → 0.0

### 2. memory.py 采样简化（需手动完成）

**需要替换sample函数（387行开始）为简化版本**:

```python
def sample(self, batch_size: int, seq_len: int = None, device: torch.device = None,
           il_ratio: float = 0.0, exp_ratio: float = 0.0, frame_idx: int = 0):
    """【回归简单】均匀随机采样RL数据（SARL范式）"""
    # 完整代码见 /tmp/simple_sample_replace.txt
    # 核心逻辑：
    # 1. 找到所有RL episodes
    # 2. 均匀随机采样（_pick方法）
    # 3. 构建batch tensor返回
```

**删除内容**：
- 第410-512行：三层采样逻辑（IL/success/timeout/collision分桶）
- 第70-73行：PER参数（per_alpha等）
- 第590-597行：PER辅助方法（update_priorities等）

### 3. train.py AWR权重删除（需手动完成）

**optimizer_step函数（326行）修改**:

删除AWR加权（411行）：
```python
# 旧代码：
weights = 0.10 + torch.relu(advantage / awr_scale)
policy_loss = (weights.unsqueeze(-1) * policy_loss_raw).mean()

# 新代码：
policy_loss = policy_loss_raw.mean()  # SARL范式：均等MSE，无加权
```

删除PER参数（326行函数签名）：
```python
# 旧：def optimize_step(..., per_info=None):
# 新：def optimize_step(...):  # 删除per_info参数
```

删除336-390行PER相关逻辑

## 预期效果

**训练强度对比**:
| 维度 | 修改前 | 修改后 | SARL |
|------|--------|--------|------|
| lr × updates | 5e-6 × 6 = 3e-5 | 3e-4 × 40 = 1.2e-2 | 1e-3 × 100 = 0.1 |
| 相对SARL | 1/3300 | 1/8.3 | 1.0 |

**曲线预期**:
- ep1-50: 成功率稳定在0.85+ (不再从0.88掉到0.78)
- ep50-200: 碰撞率降到0.10以下 (不再上升到0.20)
- ep200+: Reward MA@50持平或缓慢上升 (不再负向漂移)

## 手动修改步骤

1. ✅ train.config已自动修改完成
2. ⚠️ memory.py需手动替换sample函数（参考/tmp/simple_sample_replace.txt）
3. ⚠️ train.py需手动删除AWR权重和PER逻辑（见上文）
4. ⚠️ 删除training启动参数中的PER相关配置

## 验证方法

重启训练后检查日志：
```bash
grep "UNIFORM-SAMPLE\|learning_rate\|updates_per_ep" runs/mamba_vl/train.log
# 应看到：
# - learning_rate=0.0003
# - updates_per_ep=40
# - [UNIFORM-SAMPLE] Sampled XX/512 episodes
```
