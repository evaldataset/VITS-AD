#!/bin/bash
# Multi-scale PatchTraj experiment: train with multiple window sizes,
# then ensemble scores across scales for improved detection.
#
# Theory: anomalies manifest at different temporal scales. A short window
# catches sudden spikes, while a long window catches gradual drifts.
#
# Usage:
#   bash scripts/run_multiscale_experiment.sh GPU_ID DATASET [ENTITY]
#
# Examples:
#   bash scripts/run_multiscale_experiment.sh 2 smd machine-1-1
#   bash scripts/run_multiscale_experiment.sh 3 psm

set -euo pipefail

GPU_ID="${1:?Usage: $0 GPU_ID DATASET [ENTITY]}"
DATASET="${2:?Usage: $0 GPU_ID DATASET [ENTITY]}"
ENTITY="${3:-}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Multi-scale window sizes
WINDOW_SIZES=(50 100 200)
STRIDE_RATIO=10  # stride = window_size / STRIDE_RATIO

EPOCHS=100
PATIENCE=15
BATCH_SIZE=32
LR="5e-4"

RESULTS_BASE="results/multiscale_${DATASET}"
mkdir -p "$RESULTS_BASE"

echo "=========================================="
echo "Multi-Scale PatchTraj on GPU $GPU_ID"
echo "Dataset: $DATASET | Entity: ${ENTITY:-'(single)'}"
echo "Window sizes: ${WINDOW_SIZES[*]}"
echo "=========================================="

# Build entity args
ENTITY_ARGS=""
if [ -n "$ENTITY" ]; then
    ENTITY_ARGS="data.entity=$ENTITY"
fi

for ws in "${WINDOW_SIZES[@]}"; do
    stride=$((ws / STRIDE_RATIO))

    for renderer in line_plot recurrence_plot; do
        OUTPUT_DIR="$RESULTS_BASE/${ENTITY:-default}/w${ws}/${renderer}"
        mkdir -p "$OUTPUT_DIR"

        # Skip if already done
        if [ -f "$OUTPUT_DIR/metrics.json" ] && [ -f "$OUTPUT_DIR/scores.npy" ]; then
            echo ">>> Skipping w=$ws/$renderer (already done)"
            continue
        fi

        echo ">>> Training w=$ws, renderer=$renderer..."
        python scripts/train_patchtraj.py \
            data="$DATASET" \
            $ENTITY_ARGS \
            render="$renderer" \
            data.window_size="$ws" \
            data.stride="$stride" \
            training.epochs="$EPOCHS" \
            training.patience="$PATIENCE" \
            training.batch_size="$BATCH_SIZE" \
            training.lr="$LR" \
            output_dir="$OUTPUT_DIR" \
            2>&1 | tail -3

        echo ">>> Detecting w=$ws, renderer=$renderer..."
        python scripts/detect.py \
            data="$DATASET" \
            $ENTITY_ARGS \
            render="$renderer" \
            data.window_size="$ws" \
            data.stride="$stride" \
            training.batch_size="$BATCH_SIZE" \
            output_dir="$OUTPUT_DIR" \
            scoring.smooth_window=21 \
            scoring.smooth_method=mean \
            2>&1 | tail -5
    done
done

echo ""
echo "=========================================="
echo "Multi-scale training complete."
echo "Run: python scripts/ensemble_multiscale.py --results_dir $RESULTS_BASE"
echo "=========================================="
