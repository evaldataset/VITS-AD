#!/bin/bash
# Run CATCH baseline on all 4 datasets sequentially.
#
# Usage: bash scripts/run_catch_all.sh GPU_ID
#
# Skips datasets that already have results.

set -euo pipefail

GPU_ID="${1:?Usage: $0 GPU_ID}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

for dataset in psm msl smap smd; do
    OUTPUT_DIR="results/reports/dl_baselines/catch_${dataset}"
    if [ -f "$OUTPUT_DIR/catch_results.json" ]; then
        echo ">>> Skipping $dataset (already done)"
        continue
    fi
    echo ">>> Running CATCH on $dataset (GPU $GPU_ID)..."
    python scripts/run_catch_baseline.py \
        --dataset "$dataset" \
        --gpu "$GPU_ID" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR.log"
    echo ">>> $dataset done."
done

echo "All CATCH baselines complete."
