# 部署版本说明

## 模型文件
- `il_model.pth`: 模仿学习预训练模型（可选）
- `rl_model.pth`: **推荐使用** - 最佳性能的RL模型（rl_model_ep14500.pth）
- `rl_model_final.pth`: 备选 - 最终训练版本（更保守）

## 使用方法
```bash
# 使用最佳模型
python test.py --policy mamba --model_dir deploy/mamba_vl --weights rl_model.pth

# 使用最终模型（更保守）
python test.py --policy mamba --model_dir deploy/mamba_vl --weights rl_model_final.pth
```

## 模型特点
- 最佳模型：平衡性好，适合大多数场景
- 最终模型：更加谨慎，适合安全要求高的场景
