#!/bin/bash
# run_multiturn_qwen.sh
# Usage: bash run_multiturn_qwen.sh <hf_model_path> <max_round>

set -e

MODEL=${1:?"Usage: bash sec6_multi_turn_torch.sh <model_path> <max_round>"}
MAX_ROUND=${2:?"Usage: bash sec6_multi_turn_torch.sh <model_path> <max_round>"}
CULTURES=("BRA" "CHN" "MEX" "NGA" "NZL")
SCRIPT="sec6_multi_turn_torch.py"

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