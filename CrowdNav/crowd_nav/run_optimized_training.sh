#!/bin/bash

# 优化训练启动脚本
# 解决训练效果低下和速度慢的问题

echo "=== CrowdNav Mamba-RL 优化训练启动 ==="
echo "主要改进："
echo "1. 更大的batch size (4096) 提高训练稳定性"
echo "2. 优化的学习率调度 (actor:1.5e-4, critic:2e-4)"
echo "3. 增强的Mamba架构 (hidden_dim=128, n_blocks=4)"
echo "4. 改进的奖励函数 (success=30, collision=-15)"
echo "5. 增加IL专家数据 (300 episodes, 5000 epochs)"
echo "6. 并行采样加速 (8 workers, 6 episodes/worker)"
echo ""

# 检查GPU可用性
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU，启用GPU训练"
    GPU_FLAG="--gpu"
else
    echo "未检测到NVIDIA GPU，使用CPU训练"
    GPU_FLAG=""
fi

# 创建输出目录
OUTPUT_DIR="data/output_optimized_$(date +%Y%m%d_%H%M%S)"
echo "输出目录: $OUTPUT_DIR"

# 启动训练
echo "开始优化训练..."
python train_optimized.py \
    --env_config configs/env.config \
    --policy_config configs/policy.config \
    --train_config configs/train.config \
    --output_dir "$OUTPUT_DIR" \
    $GPU_FLAG \
    --seed 42

echo "训练完成！结果保存在: $OUTPUT_DIR"
echo "可以查看训练曲线: $OUTPUT_DIR/train_curves_ep*.png"
echo "查看详细日志: $OUTPUT_DIR/output.log"


