#!/bin/bash
# 使用IL训练的10个固定种子测试，验证是否比随机测试更困难

echo "=========================================="
echo "Testing with IL training seeds (should be harder?)"
echo "=========================================="

IL_SEEDS=(1000 10000 100000 1000000 10000000 50000000 100000000 500000000 1000000000 2000000000)

for seed in "${IL_SEEDS[@]}"; do
    echo ""
    echo "=== Testing with IL seed: $seed ==="
    PYTHONPATH=/home/abc/workspace/nav_data/mamba/camrl/CrowdNav:$PYTHONPATH \
        python test.py --policy mamba_rl --model_dir runs/mamba_vl --gpu \
        --episodes 50 --deterministic --seed $seed 2>&1 | grep -E "Success:|Collision:|Timeout:"
done
