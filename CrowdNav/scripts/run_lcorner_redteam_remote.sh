#!/bin/bash
set -euo pipefail

ROOT=/root/workspace/nav_data/mamba/camrl/CrowdNav
PYTHON=/root/miniconda3/envs/mamba/bin/python
REGISTRY="$ROOT/crowd_nav/gate_lcorner/registered_grid.json"
EXPECTED_REGISTRY_SHA=da3005093b57438648a473d834aeda2b0dfec8081cfcd9d353bce099caf06ebd

if (($# != 2)); then
  echo "usage: $0 OUTPUT_DIR REFERENCE_SWEEP_JSON" >&2
  exit 64
fi
OUTPUT=$1
REFERENCE=$2

if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite existing audit output: $OUTPUT" >&2
  exit 73
fi
if [[ ! -f "$REFERENCE" ]]; then
  echo "missing reference sweep: $REFERENCE" >&2
  exit 66
fi

actual_registry_sha=$(sha256sum "$REGISTRY" | awk '{print $1}')
if [[ "$actual_registry_sha" != "$EXPECTED_REGISTRY_SHA" ]]; then
  echo "registered grid changed: $actual_registry_sha" >&2
  exit 65
fi

mkdir -p "$OUTPUT/rerun"
cd "$ROOT"

set -o pipefail
"$PYTHON" -m unittest \
  crowd_nav.gate_lcorner.test_gate \
  crowd_nav.gate_lcorner.test_redteam -v \
  2>&1 | tee "$OUTPUT/unit.log"
unit_rc=${PIPESTATUS[0]}
if ((unit_rc != 0)); then
  exit "$unit_rc"
fi

"$PYTHON" -m crowd_nav.gate_lcorner.audit_gate bruteforce \
  --output "$OUTPUT/bruteforce.json" \
  2>&1 | tee "$OUTPUT/bruteforce.log"
brute_rc=${PIPESTATUS[0]}
if ((brute_rc != 0)); then
  exit "$brute_rc"
fi

"$PYTHON" -m crowd_nav.gate_lcorner.audit_gate traces \
  --registry "$REGISTRY" \
  --output "$OUTPUT/representative_traces.json" \
  2>&1 | tee "$OUTPUT/representative_traces.log"
trace_rc=${PIPESTATUS[0]}
if ((trace_rc != 0)); then
  exit "$trace_rc"
fi

"$PYTHON" -m crowd_nav.gate_lcorner.run_gate --stage single \
  --registry "$REGISTRY" --output "$OUTPUT/rerun" \
  2>&1 | tee "$OUTPUT/rerun_single.log"
single_rc=${PIPESTATUS[0]}
if ((single_rc != 0)); then
  exit "$single_rc"
fi

"$PYTHON" -m crowd_nav.gate_lcorner.run_gate --stage sweep \
  --registry "$REGISTRY" --output "$OUTPUT/rerun" \
  2>&1 | tee "$OUTPUT/rerun_sweep.log"
sweep_rc=${PIPESTATUS[0]}
if ((sweep_rc != 0)); then
  exit "$sweep_rc"
fi

"$PYTHON" -m crowd_nav.gate_lcorner.audit_gate compare \
  --reference "$REFERENCE" \
  --candidate "$OUTPUT/rerun/registered_sweep.json" \
  --output "$OUTPUT/determinism.json" \
  2>&1 | tee "$OUTPUT/determinism.log"
compare_rc=${PIPESTATUS[0]}
if ((compare_rc != 0)); then
  exit "$compare_rc"
fi

sha256sum \
  "$REGISTRY" \
  "$OUTPUT/bruteforce.json" \
  "$OUTPUT/representative_traces.json" \
  "$OUTPUT/rerun/registered_sweep.json" \
  "$OUTPUT/determinism.json" > "$OUTPUT/sha256.txt"

printf 'REDTEAM_STATUS=PASS\n' | tee "$OUTPUT/status.txt"
