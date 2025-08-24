#!/bin/bash

# 超快速训练启动脚本 - 6小时内达到85%成功率
echo "=== CrowdNav 超快速训练 (目标: 6小时内85%+成功率) ==="
echo ""
echo "关键优化策略："
echo "1. 超大batch size (8192) - 极大提升训练效率"
echo "2. 增强IL预训练 (500 episodes, 8000 epochs) - 确保高起点"
echo "3. 激进奖励函数 (success=50, goal_weight=12) - 强目标导向"
echo "4. 超大Mamba模型 (hidden=256, blocks=6) - 强大表达能力"
echo "5. 12并行workers - 最大化采样速度"
echo "6. 仅400训练episodes - 快速收敛"
echo "7. 高学习率 (3e-4 to 4e-4) - 加速学习"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU 检测成功"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_FLAG="--gpu"
else
    echo "✗ 未检测到GPU，将使用CPU训练（速度会很慢）"
    GPU_FLAG=""
fi

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="data/output_fast_${TIMESTAMP}"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""

# 启动训练
echo "=========================================="
echo "开始训练 (预计时间: <6小时)"
echo "=========================================="

python train_fast.py \
    --env_config configs/env.config \
    --policy_config configs/policy.config \
    --train_config configs/train.config \
    --output_dir "$OUTPUT_DIR" \
    $GPU_FLAG \
    --seed 42

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo "结果保存在: $OUTPUT_DIR"
echo "查看训练日志: tail -f $OUTPUT_DIR/output.log"
echo "查看最终结果: grep 'Final Success Rate' $OUTPUT_DIR/output.log"

# 显示最终结果
if [ -f "$OUTPUT_DIR/output.log" ]; then
    echo ""
    echo "训练结果摘要："
    grep -E "(Final Success Rate|Total Training Time|Best Success Rate)" "$OUTPUT_DIR/output.log"
fi