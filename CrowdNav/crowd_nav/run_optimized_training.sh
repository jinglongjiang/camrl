#!/bin/bash

# 优化训练启动脚本
# 解决训练效果低下和速度慢的问题

echo "=== CrowdNav 优化训练启动 ==="
echo "主要改进："
echo "1. 更大的batch size (2048) 提高训练稳定性"
echo "2. 降低学习率 (1e-4) 提高收敛稳定性"
echo "3. 自适应探索策略"
echo "4. 早停机制防止过拟合"
echo "5. 更频繁的评估和检查点"
echo "6. 优化的奖励函数"
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
    --env_config configs/env_optimized.config \
    --train_config configs/train_optimized.config \
    --output_dir "$OUTPUT_DIR" \
    $GPU_FLAG \
    --seed 42

echo "训练完成！结果保存在: $OUTPUT_DIR"
echo "可以查看训练曲线: $OUTPUT_DIR/train_curves_ep*.png"
echo "查看详细日志: $OUTPUT_DIR/output.log"
