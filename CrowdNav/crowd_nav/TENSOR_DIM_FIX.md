# Tensor Dimension Bug Fix - 2025-11-17

## 问题描述

训练时遇到tensor维度错误：
```
RuntimeError: expand(torch.cuda.LongTensor{[256, 12, 0, 6, 1]}, size=[-1, -1, -1, 13]):
the number of sizes provided (4) must be greater or equal to the number of dimensions in the tensor (5)
```

错误发生在 `mamba_rl.py:104` 的spatial encoder中。

## 根本原因

**Replay Buffer返回了5D tensor，但训练代码期望4D tensor！**

### 问题链路：

1. **Episode存储**: `ep["tokens"]` 存储为 `[T, 1, 6, 13]` (4D，多了中间维度1)
2. **Buffer采样**: `memory.py:487-488`
   ```python
   token_shape = tuple(any_ep["tokens"].shape[1:])  # (1, 6, 13)
   states = np.zeros((b, seq_len) + token_shape)    # [B, T, 1, 6, 13] ← 5D!
   ```
3. **训练期望**: 所有训练代码期望 `[B, T, 6, 13]` (4D)
   - Warmup: `train.py:765-773`
   - RL training: `train.py:1811-1812`
   - Spatial encoder: `mamba_rl.py:82-91`

### 错误表现：

- Buffer返回 `[256, 12, 1, 6, 13]` (5D)
- Spatial encoder提取 `humans_all = joint_state[:, :, 3:6, :]`
- 因为有额外维度，索引错位导致 `distances` 形状错误
- `torch.sort()` 返回错误维度的 `sorted_indices`
- `unsqueeze(-1).expand()` 操作失败

## 修复方案

### 文件1: `utils/memory.py:531-539`

在 `sample()` 方法返回前，检测并squeeze掉多余维度：

```python
# ⚠️ 关键修复：ep["tokens"]存储为[T, 1, 6, 13]（多了中间维度1），需要squeeze掉
states_tensor = torch.as_tensor(states).to(device, non_blocking=True)
nstates_tensor = torch.as_tensor(nstates).to(device, non_blocking=True)

# 如果是5D[B,T,1,6,13]，squeeze掉维度2（索引2）
if states_tensor.dim() == 5 and states_tensor.shape[2] == 1:
    states_tensor = states_tensor.squeeze(2)  # [B, T, 6, 13]
    nstates_tensor = nstates_tensor.squeeze(2)  # [B, T, 6, 13]
```

### 文件2: `utils/memory.py:404-405`

更新docstring反映正确的输出格式：

```python
返回：
  states       [B, T, 6, 13] ← 已squeeze中间维度1
  next_states  [B, T, 6, 13] ← 已squeeze中间维度1
```

### 文件3: `train.py:765`

更新注释说明buffer已修复：

```python
states_raw = batch['states'].to(device)  # [B, T, 6, 13] (buffer已squeeze维度)
```

### 文件4: `policy/mamba_rl.py:87-91`

添加输入验证，确保spatial encoder收到正确维度：

```python
# 输入验证
if joint_state.dim() != 4:
    raise ValueError(f"❌ Spatial encoder expects 4D input [B,T,6,13], got {joint_state.dim()}D: {joint_state.shape}")
if joint_state.shape[2:] != (6, 13):
    raise ValueError(f"❌ Spatial encoder expects [..., 6, 13], got [..., {joint_state.shape[2]}, {joint_state.shape[3]}]: {joint_state.shape}")
```

## 验证测试

运行 `test_buffer_shape.py` 验证修复：

```bash
python test_buffer_shape.py
```

测试结果：
- ✅ 5D→4D squeeze逻辑正确
- ✅ Robot特征提取 [B, T, 13]
- ✅ Human特征提取 [B, T, 3, 13]
- ✅ 距离排序操作 [B, T, 3]
- ✅ 索引扩展操作 [B, T, 3, 13]
- ✅ 边界情况测试通过

## 影响范围

### 已修复的流程：
1. **IL Actor Warmup**: `train.py:756-790` - 使用完整序列 [B, T, 6, 13]
2. **RL Training**: `train.py:1809-1831` - 使用完整序列 [B, T, 6, 13]
3. **Spatial Encoder**: `mamba_rl.py:82-140` - 接收正确4D输入
4. **Temporal Encoder**: `mamba_rl.py:188-229` - 接收正确特征维度

### 不受影响的部分：
- **Rollout/Predict**: `mamba_rl.py:464-478` - 不使用buffer，自己构建序列
- **Episode存储**: 不需要修改，保持向后兼容性

## 后续注意事项

1. **新Buffer实现**: 如果将来重写buffer，直接存储 `[T, 6, 13]` 格式更简洁
2. **类型检查**: 所有接收states的函数都应验证维度
3. **文档更新**: 确保所有维度注释保持一致

## 测试检查清单

同步到服务器后验证：

- [ ] Warmup训练正常启动（无维度错误）
- [ ] RL训练正常启动（无维度错误）
- [ ] Rollout采样正常（机器人能移动）
- [ ] Loss正常下降（不是NaN或0）
- [ ] Success rate逐渐提升（不是100% timeout）

## 修复前后对比

| 阶段 | 修复前 | 修复后 |
|------|--------|--------|
| Buffer采样 | `[B, T, 1, 6, 13]` (5D) | `[B, T, 6, 13]` (4D) ✅ |
| Warmup输入 | ❌ 维度错误 | `[B, T, 6, 13]` ✅ |
| RL训练输入 | ❌ 维度错误 | `[B, T, 6, 13]` ✅ |
| Spatial编码 | ❌ RuntimeError | 正常处理 ✅ |
| 排序操作 | `[B, T, 0, 6, 1]` 错误 | `[B, T, 3]` 正确 ✅ |

---

**修复完成时间**: 2025-11-17
**测试状态**: ✅ 本地验证通过
**服务器测试**: 待用户同步后验证
