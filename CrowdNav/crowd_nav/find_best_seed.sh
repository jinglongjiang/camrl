#!/bin/bash
# 寻找成功率最高的seed（用于展示视频）

OUTFILE="seed_search_results.txt"
echo "Seed,Success_Rate,Collision_Rate,Timeout_Rate" > $OUTFILE

echo "Testing 30 different seeds with 20 episodes each..."

for i in {1..30}; do
    SEED=$((1000 + i * 1000))
    echo "Testing seed $SEED ($i/30)..."
    
    RESULT=$(PYTHONPATH=/home/abc/workspace/nav_data/mamba/camrl/CrowdNav:$PYTHONPATH \
        python test.py --policy mamba_rl --model_dir runs/mamba_vl --gpu \
        --episodes 20 --deterministic --seed $SEED 2>&1 | \
        grep -A3 "AVERAGE ACROSS" | tail -3)
    
    SUCCESS=$(echo "$RESULT" | grep "Success:" | awk '{print $2}')
    COLLISION=$(echo "$RESULT" | grep "Collision:" | awk '{print $2}')
    TIMEOUT=$(echo "$RESULT" | grep "Timeout:" | awk '{print $2}')
    
    echo "$SEED,$SUCCESS,$COLLISION,$TIMEOUT" >> $OUTFILE
    echo "  Seed $SEED: Success=$SUCCESS"
done

echo ""
echo "=========================================="
echo "Top 5 seeds by success rate:"
echo "=========================================="
tail -n +2 $OUTFILE | sort -t',' -k2 -rn | head -5 | \
    awk -F',' '{printf "Seed: %s  Success: %.1f%%  Collision: %.1f%%  Timeout: %.1f%%\n", $1, $2*100, $3*100, $4*100}'

echo ""
echo "Full results saved to: $OUTFILE"
