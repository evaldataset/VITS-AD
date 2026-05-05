#!/bin/bash
# Run PatchTraj multi-renderer ensemble benchmark across SMD entities.
#
# For each entity, trains a separate PatchTraj model per renderer
# (line_plot, gaf, recurrence_plot), then ensembles scores.
#
# Usage:
#   bash scripts/run_ensemble_benchmark.sh [GPU_ID] [EPOCHS]
#
# Example:
#   bash scripts/run_ensemble_benchmark.sh 1 100

set -euo pipefail

GPU_ID="${1:-1}"
EPOCHS="${2:-100}"
PATIENCE=15
BATCH_SIZE=64
LR="5e-4"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="$GPU_ID"

RENDERERS=("line_plot" "gaf" "recurrence_plot")
ENTITIES=(
    "machine-1-1" "machine-1-2" "machine-1-3" "machine-1-4"
    "machine-1-5" "machine-1-6" "machine-1-7" "machine-1-8"
)

RESULTS_DIR="results/ensemble_smd"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "PatchTraj Ensemble Benchmark"
echo "GPU: $GPU_ID | Epochs: $EPOCHS | Renderers: ${RENDERERS[*]}"
echo "Entities: ${#ENTITIES[@]}"
echo "=========================================="

SUMMARY_FILE="$RESULTS_DIR/summary.csv"
echo "entity,renderer,auc_roc,auc_pr,best_f1,f1_pa" > "$SUMMARY_FILE"

for entity in "${ENTITIES[@]}"; do
    echo ""
    echo "=========================================="
    echo ">>> Entity: $entity"
    echo "=========================================="

    for renderer in "${RENDERERS[@]}"; do
        OUTPUT_DIR="$RESULTS_DIR/$entity/$renderer"
        mkdir -p "$OUTPUT_DIR"

        echo ""
        echo ">>> Training $entity with $renderer renderer..."

        python scripts/train_patchtraj.py \
            data.entity="$entity" \
            render="$renderer" \
            training.epochs="$EPOCHS" \
            training.patience="$PATIENCE" \
            training.batch_size="$BATCH_SIZE" \
            training.lr="$LR" \
            output_dir="$OUTPUT_DIR" \
            2>&1 | tee "$OUTPUT_DIR/train.log"

        echo ">>> Detecting $entity with $renderer renderer..."

        python scripts/detect.py \
            data.entity="$entity" \
            render="$renderer" \
            training.batch_size="$BATCH_SIZE" \
            output_dir="$OUTPUT_DIR" \
            scoring.smooth_window=7 \
            2>&1 | tee "$OUTPUT_DIR/detect.log"

        if [ -f "$OUTPUT_DIR/metrics.json" ]; then
            AUC_ROC=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{m['auc_roc']:.6f}\")")
            AUC_PR=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{m['auc_pr']:.6f}\")")
            BEST_F1=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{m['best_f1']:.6f}\")")
            F1_PA=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{m['f1_pa']:.6f}\")")
            echo "$entity,$renderer,$AUC_ROC,$AUC_PR,$BEST_F1,$F1_PA" >> "$SUMMARY_FILE"
            echo ">>> $entity/$renderer: AUC-ROC=$AUC_ROC | AUC-PR=$AUC_PR | F1-PA=$F1_PA"
        else
            echo "WARNING: No metrics.json for $entity/$renderer"
            echo "$entity,$renderer,N/A,N/A,N/A,N/A" >> "$SUMMARY_FILE"
        fi
    done

    # Ensemble scores for this entity
    echo ""
    echo ">>> Ensembling scores for $entity..."
    python scripts/ensemble_scores.py \
        --entity "$entity" \
        --results_dir "$RESULTS_DIR" \
        --renderers line_plot gaf recurrence_plot \
        --smooth_window 7 \
        2>&1 | tee "$RESULTS_DIR/$entity/ensemble.log"

    if [ -f "$RESULTS_DIR/$entity/ensemble_metrics.json" ]; then
        AUC_ROC=$(python -c "import json; m=json.load(open('$RESULTS_DIR/$entity/ensemble_metrics.json')); print(f\"{m['auc_roc']:.6f}\")")
        AUC_PR=$(python -c "import json; m=json.load(open('$RESULTS_DIR/$entity/ensemble_metrics.json')); print(f\"{m['auc_pr']:.6f}\")")
        BEST_F1=$(python -c "import json; m=json.load(open('$RESULTS_DIR/$entity/ensemble_metrics.json')); print(f\"{m['best_f1']:.6f}\")")
        F1_PA=$(python -c "import json; m=json.load(open('$RESULTS_DIR/$entity/ensemble_metrics.json')); print(f\"{m['f1_pa']:.6f}\")")
        echo "$entity,ensemble,$AUC_ROC,$AUC_PR,$BEST_F1,$F1_PA" >> "$SUMMARY_FILE"
        echo ">>> $entity/ENSEMBLE: AUC-ROC=$AUC_ROC | AUC-PR=$AUC_PR | F1-PA=$F1_PA"
    fi
done

echo ""
echo "=========================================="
echo "Ensemble Benchmark Complete. Summary:"
echo "=========================================="
cat "$SUMMARY_FILE" | column -t -s ','

# Compute averages per renderer and ensemble
python -c "
import csv, statistics

with open('$SUMMARY_FILE') as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r['auc_roc'] != 'N/A']

by_renderer = {}
for r in rows:
    renderer = r['renderer']
    if renderer not in by_renderer:
        by_renderer[renderer] = []
    by_renderer[renderer].append(r)

for renderer in ['line_plot', 'gaf', 'recurrence_plot', 'ensemble']:
    if renderer not in by_renderer:
        continue
    rr = by_renderer[renderer]
    avg_roc = statistics.mean(float(r['auc_roc']) for r in rr)
    avg_pr = statistics.mean(float(r['auc_pr']) for r in rr)
    avg_f1 = statistics.mean(float(r['best_f1']) for r in rr)
    avg_pa = statistics.mean(float(r['f1_pa']) for r in rr)
    status = 'PASS' if avg_roc > 0.7 else 'NEEDS IMPROVEMENT'
    print(f'\n{renderer.upper()} ({len(rr)} entities):')
    print(f'  AUC-ROC: {avg_roc:.4f} ({status})')
    print(f'  AUC-PR:  {avg_pr:.4f}')
    print(f'  Best-F1: {avg_f1:.4f}')
    print(f'  F1-PA:   {avg_pa:.4f}')
"
