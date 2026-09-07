#!/bin/bash
# 种子搜索启动脚本

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Random Seed Search for CrowdNav Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 设置环境变量（确保可复现）
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 检查BC checkpoint是否存在
BC_CHECKPOINT="runs/mamba_vl/il_policy.pth"

if [ ! -f "$BC_CHECKPOINT" ]; then
    echo "❌ BC checkpoint not found: $BC_CHECKPOINT"
    echo ""
    echo "Please ensure BC training has completed."
    echo "Expected location: $BC_CHECKPOINT"
    exit 1
fi

echo ""
echo "✅ Found BC checkpoint: $BC_CHECKPOINT"
echo ""

# 选择测试模式
echo "Select testing mode:"
echo "  1) Quick test (用BC模型快速测试16个seed，~5分钟)"
echo "  2) Full training test (完整训练8个seed到500 episodes，~2-4小时)"
echo ""
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 Starting quick seed test (BC model evaluation)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        python quick_seed_test.py \
            --checkpoint "$BC_CHECKPOINT" \
            --num_seeds 16 \
            --test_episodes 100 \
            --output seed_test_results_$(date +%Y%m%d_%H%M%S).json

        echo ""
        echo "✅ Quick test completed!"
        ;;

    2)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 Starting full training seed search"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "⚠️  Warning: This will take 2-4 hours!"
        echo ""
        read -p "Continue? [y/N]: " confirm

        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "Cancelled."
            exit 0
        fi

        python find_best_seed.py \
            --milestone 500 \
            --num_seeds 8 \
            --max_parallel 2 \
            --gpu \
            --output_dir ./seed_search_results

        echo ""
        echo "✅ Full training test completed!"
        ;;

    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Seed search completed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
