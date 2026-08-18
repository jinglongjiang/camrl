#!/bin/bash
# 3-seed 2x2 causal diagnostic. SERIAL by design: the branches are compared
# against each other, so GPU contention between them would be a confound.
#
# Every branch of a seed starts from the SAME warm-up fork -- weights, EMA,
# replay, sample RNG and tau RNG identical -- and differs in exactly two
# declared ways: whether Adam state is carried, and rank_share.
set -u
cd /root/workspace/nav_data/mamba/camrl/bdvl_v6/CrowdNav || exit 2
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
CLI="/root/miniconda3/envs/mamba/bin/python -u -m crowd_nav.bayesian_dvl.intent_train_cli"
ROOT=runs/v2/diag2x2
CORPUS=runs/v2/formal_corpus/il_corpus_raw_0ebebd0d1029.pth
CACHE=$ROOT/materialized_full.pth
AUDIT=runs/v2/candidate_audit.json
SUMMARY=/root/diag2x2_summary.txt
mkdir -p $ROOT

# The ONLY abort type that counts as an experimental result. Everything else
# -- non_finite, clipping, ranking_quality, warmup_failure -- means the run
# broke rather than answered, and must stop the whole queue.
#
# Read from the TYPE the trainer emits (ABORT_TYPE=... on stdout and
# abort_reason.json in the run dir), never from the message: three failures
# used to share one English prefix, and matching prose meant a reworded
# sentence could silently turn a broken run into a result.
EXPECTED_ABORT_TYPE='mc_regression'

# R0 keep/2.0 is the CURRENT implementation: it must reproduce the failure,
# or nothing measured here can be attributed to anything.
BRANCHES="R0:2.0:keep R1:2.0:reset R2:0.0:keep R3:0.0:reset"

fail() { echo "FATAL: $*"; exit 2; }

check_cache_hit() {   # $1 = log file, $2 = label
  grep -q "materialized cache HIT" "$1" \
    || fail "$2 did not report a materialized cache HIT; every branch must reuse ONE cache identity"
  ! grep -q "materialized cache MISS" "$1" \
    || fail "$2 re-materialized (cache MISS); the branches would not share their inputs"
}

for SEED in 98211 98212 98213; do
  FORK=$ROOT/fork_seed${SEED}.pth
  WLOG=$ROOT/warmup_seed${SEED}.log
  if [ ! -f "$FORK" ]; then
    echo "=== seed $SEED: warm-up (once) ==="
    $CLI --device cuda train --run-dir $ROOT/warmup_seed${SEED} \
      --training-arm full --seed $SEED --il-corpus $CORPUS --materialized-cache $CACHE \
      --candidate-audit $AUDIT --save-warmup-fork $FORK \
      --il-passes 0 --target-online-episodes 0 > $WLOG 2>&1
    [ -f "$FORK" ] || { tail -20 $WLOG; fail "warm-up produced no fork for seed $SEED"; }
  fi
  # The FIRST seed writes the cache; every later warm-up must hit it.
  if [ "$SEED" != "98211" ]; then check_cache_hit "$WLOG" "warm-up seed $SEED"; fi

  for B in $BRANCHES; do
    NAME=${B%%:*}; REST=${B#*:}; SHARE=${REST%%:*}; ADAM=${REST##*:}
    OUT=$ROOT/${NAME}_seed${SEED}
    LOG=$OUT.log
    [ -f "$OUT/BRANCH_DONE" ] && { echo "skip $NAME/$SEED (already done)"; continue; }
    RESET=""; [ "$ADAM" = "reset" ] && RESET="--reset-optimizer"
    echo "=== seed $SEED branch $NAME (share=$SHARE adam=$ADAM) ==="
    mkdir -p $OUT
    $CLI --device cuda train --run-dir $OUT \
      --training-arm full --seed $SEED --il-corpus $CORPUS --materialized-cache $CACHE \
      --candidate-audit $AUDIT --fork-from $FORK $RESET --rank-share $SHARE \
      --diagnostic-disable-ranking-gate \
      --il-passes 2000 --target-online-episodes 0 \
      --diagnostic-interval 50 --diagnostic-checkpoint-dir $OUT/diag_ckpt > $LOG 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      TYPE=$(grep -o 'ABORT_TYPE=[a-z_]*' $LOG | tail -1 | cut -d= -f2)
      if [ "${TYPE:-}" != "$EXPECTED_ABORT_TYPE" ]; then
        echo "--- last 25 lines of $LOG ---"; tail -25 $LOG
        fail "$NAME/$SEED exited $rc with abort type '${TYPE:-<none>}', expected '$EXPECTED_ABORT_TYPE'"
      fi
      echo "  -> aborted with type $TYPE (that is a RESULT, continuing)"
    fi
    check_cache_hit "$LOG" "$NAME/$SEED"
    touch $OUT/BRANCH_DONE
  done
done
echo "2x2 COMPLETE"
/root/miniconda3/envs/mamba/bin/python /root/summarize_2x2.py \
  /root/workspace/nav_data/mamba/camrl/bdvl_v6/CrowdNav/runs/v2/diag2x2 > $SUMMARY 2>&1
echo "summary written -> $SUMMARY"
cat $SUMMARY
