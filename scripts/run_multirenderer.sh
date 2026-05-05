#!/bin/bash
# Train GAF and RP renderers for specified entities on a given GPU.
# The existing line_plot results from benchmark_smd are reused.
#
# Usage:
#   bash scripts/run_multirenderer.sh GPU_ID ENTITY1 ENTITY2 ...
#
# Example:
#   bash scripts/run_multirenderer.sh 1 machine-1-1 machine-1-2 machine-1-3 machine-1-4

set -euo pipefail

GPU_ID="${1:?Usage: $0 GPU_ID ENTITY1 ENTITY2 ...}"
shift
ENTITIES=("$@")

EPOCHS=100
PATIENCE=15
BATCH_SIZE=64
LR="5e-4"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="$GPU_ID"

ENSEMBLE_DIR="results/ensemble_smd"
BENCHMARK_DIR="results/benchmark_smd"

echo "=========================================="
echo "Multi-renderer training on GPU $GPU_ID"
echo "Entities: ${ENTITIES[*]}"
echo "=========================================="

for entity in "${ENTITIES[@]}"; do
    # Copy existing line_plot results
    LP_SRC="$BENCHMARK_DIR/$entity"
    LP_DST="$ENSEMBLE_DIR/$entity/line_plot"
    mkdir -p "$LP_DST"
    if [ -f "$LP_SRC/best_model.pt" ]; then
        cp -n "$LP_SRC/best_model.pt" "$LP_DST/" 2>/dev/null || true
        cp -n "$LP_SRC/metrics.json" "$LP_DST/" 2>/dev/null || true
        cp -n "$LP_SRC/scores.npy" "$LP_DST/" 2>/dev/null || true
        cp -n "$LP_SRC/test_patch_tokens.npy" "$LP_DST/" 2>/dev/null || true
        echo ">>> Copied existing line_plot results for $entity"
    fi

    # Need to re-run detect for line_plot to get labels.npy
    if [ ! -f "$LP_DST/labels.npy" ]; then
        echo ">>> Re-running line_plot detect for $entity (need labels.npy)..."
        python scripts/detect.py \
            data.entity="$entity" \
            render=line_plot \
            training.batch_size="$BATCH_SIZE" \
            output_dir="$LP_DST" \
            scoring.smooth_window=7 \
            2>&1 | tail -5
    fi

    for renderer in gaf recurrence_plot; do
        OUTPUT_DIR="$ENSEMBLE_DIR/$entity/$renderer"
        mkdir -p "$OUTPUT_DIR"

        echo ""
        echo ">>> Training $entity with $renderer on GPU $GPU_ID..."

        python scripts/train_patchtraj.py \
            data.entity="$entity" \
            render="$renderer" \
            training.epochs="$EPOCHS" \
            training.patience="$PATIENCE" \
            training.batch_size="$BATCH_SIZE" \
            training.lr="$LR" \
            output_dir="$OUTPUT_DIR" \
            2>&1 | tail -3

        echo ">>> Detecting $entity with $renderer..."
        python scripts/detect.py \
            data.entity="$entity" \
            render="$renderer" \
            training.batch_size="$BATCH_SIZE" \
            output_dir="$OUTPUT_DIR" \
            scoring.smooth_window=7 \
            2>&1 | tail -8
    done

    # Run ensemble
    echo ""
    echo ">>> Ensembling $entity..."
    python scripts/ensemble_scores.py \
        --entity "$entity" \
        --results_dir "$ENSEMBLE_DIR" \
        --renderers line_plot gaf recurrence_plot \
        --smooth_window 7 \
        2>&1

    echo ""
done

echo "=========================================="
echo "Multi-renderer training complete on GPU $GPU_ID"
echo "=========================================="
