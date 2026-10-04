#!/bin/bash

set -euo pipefail

GPU_ID="${1:?Usage: $0 GPU_ID}"
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"

for dataset in psm msl smap smd; do
    AT_OUTPUT_DIR="results/reports/dl_baselines/at_${dataset}"
    AT_RESULT_JSON="$AT_OUTPUT_DIR/at_results.json"

    if [ -f "$AT_RESULT_JSON" ]; then
        echo ">>> Skipping AnomalyTransformer on $dataset (already done)"
    else
        echo ">>> Running AnomalyTransformer on $dataset (GPU $GPU_ID)..."
        python scripts/run_at_baseline.py \
            --dataset "$dataset" \
            --gpu "$GPU_ID" \
            --output_dir "$AT_OUTPUT_DIR" \
            2>&1 | tee "$AT_OUTPUT_DIR.log"
        echo ">>> AnomalyTransformer $dataset done."
    fi

    DCD_OUTPUT_DIR="results/reports/dl_baselines/dcdetector_${dataset}"
    DCD_RESULT_JSON="$DCD_OUTPUT_DIR/dcdetector_results.json"

    if [ -f "$DCD_RESULT_JSON" ]; then
        echo ">>> Skipping DCdetector on $dataset (already done)"
    else
        echo ">>> Running DCdetector on $dataset (GPU $GPU_ID)..."
        python scripts/run_dcdetector_baseline.py \
            --dataset "$dataset" \
            --gpu "$GPU_ID" \
            --output_dir "$DCD_OUTPUT_DIR" \
            2>&1 | tee "$DCD_OUTPUT_DIR.log"
        echo ">>> DCdetector $dataset done."
    fi

    TS2VEC_OUTPUT_DIR="results/reports/ts2vec_baselines/ts2vec_${dataset}"
    TS2VEC_RESULT_JSON="$TS2VEC_OUTPUT_DIR/ts2vec_results.json"

    if [ -f "$TS2VEC_RESULT_JSON" ]; then
        echo ">>> Skipping TS2Vec on $dataset (already done)"
    else
        echo ">>> Running TS2Vec on $dataset (GPU $GPU_ID)..."
        python scripts/run_ts2vec_baseline.py \
            --dataset "$dataset" \
            --gpu "$GPU_ID" \
            --output_dir "$TS2VEC_OUTPUT_DIR" \
            2>&1 | tee "$TS2VEC_OUTPUT_DIR.log"
        echo ">>> TS2Vec $dataset done."
    fi
done

echo "All AT, DCdetector, and TS2Vec baselines complete."
