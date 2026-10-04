#!/bin/bash
# Run LP+RP PatchTraj benchmark across ALL 28 SMD entities on a given GPU.
# Trains line_plot and recurrence_plot renderers, then ensembles via rank_mean.
#
# Usage:
#   bash scripts/run_full_smd_benchmark.sh GPU_ID ENTITY1 ENTITY2 ...
#
# Example:
#   bash scripts/run_full_smd_benchmark.sh 1 machine-1-1 machine-2-1

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

RESULTS_DIR="results/full_smd"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Full SMD LP+RP Benchmark on GPU $GPU_ID"
echo "Entities: ${#ENTITIES[@]}"
echo "=========================================="

SUMMARY_FILE="$RESULTS_DIR/summary_gpu${GPU_ID}.csv"
echo "entity,lp_auc_roc,rp_auc_roc,ensemble_auc_roc,ensemble_method" > "$SUMMARY_FILE"

for entity in "${ENTITIES[@]}"; do
    echo ""
    echo "=========================================="
    echo ">>> Entity: $entity"
    echo "=========================================="

    ENTITY_DIR="$RESULTS_DIR/$entity"

    for renderer in line_plot recurrence_plot; do
        OUTPUT_DIR="$ENTITY_DIR/$renderer"
        mkdir -p "$OUTPUT_DIR"

        # Skip if already done
        if [ -f "$OUTPUT_DIR/metrics.json" ] && [ -f "$OUTPUT_DIR/scores.npy" ] && [ -f "$OUTPUT_DIR/labels.npy" ]; then
            echo ">>> Skipping $entity/$renderer (already done)"
            continue
        fi

        echo ">>> Training $entity with $renderer..."
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

    # Ensemble
    echo ">>> Ensembling $entity (LP+RP)..."
    python scripts/ensemble_scores.py \
        --entity "$entity" \
        --results_dir "$RESULTS_DIR" \
        --renderers line_plot recurrence_plot \
        --smooth_window 7 \
        2>&1 | grep -E "Best method|AUC-ROC" | tail -5

    # Extract metrics for summary
    LP_AUC="N/A"
    RP_AUC="N/A"
    ENS_AUC="N/A"
    ENS_METHOD="N/A"

    if [ -f "$ENTITY_DIR/line_plot/metrics.json" ]; then
        LP_AUC=$(python3 -c "import json; print(f\"{json.load(open('$ENTITY_DIR/line_plot/metrics.json'))['auc_roc']:.6f}\")")
    fi
    if [ -f "$ENTITY_DIR/recurrence_plot/metrics.json" ]; then
        RP_AUC=$(python3 -c "import json; print(f\"{json.load(open('$ENTITY_DIR/recurrence_plot/metrics.json'))['auc_roc']:.6f}\")")
    fi
    if [ -f "$ENTITY_DIR/ensemble_metrics.json" ]; then
        ENS_AUC=$(python3 -c "import json; print(f\"{json.load(open('$ENTITY_DIR/ensemble_metrics.json'))['auc_roc']:.6f}\")")
    fi

    echo "$entity,$LP_AUC,$RP_AUC,$ENS_AUC,$ENS_METHOD" >> "$SUMMARY_FILE"
    echo ">>> $entity: LP=$LP_AUC | RP=$RP_AUC | Ensemble=$ENS_AUC"
done

echo ""
echo "=========================================="
echo "Benchmark Complete for GPU $GPU_ID"
echo "=========================================="
cat "$SUMMARY_FILE" | column -t -s ','
