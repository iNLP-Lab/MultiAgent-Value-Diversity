#!/bin/bash
# run_multiturn.sh
# Usage: bash run_multiturn.sh <model> <max_round>
# Example: bash run_multiturn.sh grok-4.3 3
#
# Runs rounds 2..max_round for the given model.
# Within each round, runs all 5 cultures sequentially.
# A round must fully complete before the next round starts
# (because round k reads from round k-1's outputs).

set -e   # exit on any error

MODEL=${1:?"Usage: bash sec6_multi_turn_api.sh <model> <max_round>"}
MAX_ROUND=${2:?"Usage: bash sec6_multi_turn_api.sh <model> <max_round>"}
CULTURES=("BRA" "CHN" "MEX" "NGA" "NZL")
SCRIPT="sec6_multi_turn_api.py"

for R in $(seq 2 $MAX_ROUND); do
    echo "=========================================="
    echo " Round $R  (model: $MODEL)"
    echo "=========================================="
    for CUL in "${CULTURES[@]}"; do
        echo "--- [round $R | $CUL] ---"
        python $SCRIPT --model $MODEL --culture $CUL --round_idx $R
    done
done

echo "All rounds done for $MODEL."