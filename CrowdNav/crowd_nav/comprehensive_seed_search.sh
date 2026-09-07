#!/bin/bash
# 全面种子搜索脚本

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 全面种子搜索 - 选择测试规模"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "选择测试规模："
echo "  1) 快速测试    (16个种子 × 100 episodes, ~5分钟)"
echo "  2) 标准测试    (50个种子 × 100 episodes, ~15分钟)"
echo "  3) 全面测试    (100个种子 × 100 episodes, ~30分钟)"
echo "  4) 自定义"
echo ""
read -p "请选择 [1-4]: " choice

case $choice in
    1)
        NUM_SEEDS=16
        NUM_EPS=100
        ;;
    2)
        NUM_SEEDS=50
        NUM_EPS=100
        ;;
    3)
        NUM_SEEDS=100
        NUM_EPS=100
        ;;
    4)
        read -p "请输入种子数量 (推荐16-100): " NUM_SEEDS
        read -p "请输入每个种子的episodes (推荐50-100): " NUM_EPS
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 测试配置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  种子数量: $NUM_SEEDS"
echo "  每个种子episodes: $NUM_EPS"
echo "  总测试episodes: $((NUM_SEEDS * NUM_EPS))"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "确认开始? [y/N]: " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🚀 开始测试..."
echo ""

# 运行测试
python simple_seed_test.py \
    --num_seeds $NUM_SEEDS \
    --test_episodes $NUM_EPS

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 测试完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
