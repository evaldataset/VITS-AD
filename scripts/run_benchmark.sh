#!/bin/bash
# Run PatchTraj benchmark across SMD entities.
#
# Defaults to the FULL 28-entity SMD benchmark used for the headline numbers
# in Table 1 of the paper.  Pass `smoke` as the third argument to run only the
# 8-entity subset (machine-1-*) historically used for quick smoke tests.
#
# Usage:
#   bash scripts/run_benchmark.sh [GPU_ID] [EPOCHS] [smoke|full]
#
# Examples:
#   bash scripts/run_benchmark.sh 0 100        # full 28 entities
#   bash scripts/run_benchmark.sh 0 100 smoke  # 8-entity smoke test

set -euo pipefail

GPU_ID="${1:-0}"
EPOCHS="${2:-100}"
MODE="${3:-full}"
PATIENCE=15
BATCH_SIZE=64
LR="5e-4"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# SMD entities — full 28-entity benchmark or 8-entity smoke subset.
if [ "$MODE" = "smoke" ]; then
    ENTITIES=(
        "machine-1-1" "machine-1-2" "machine-1-3" "machine-1-4"
        "machine-1-5" "machine-1-6" "machine-1-7" "machine-1-8"
    )
else
    ENTITIES=(
        "machine-1-1" "machine-1-2" "machine-1-3" "machine-1-4"
        "machine-1-5" "machine-1-6" "machine-1-7" "machine-1-8"
        "machine-2-1" "machine-2-2" "machine-2-3" "machine-2-4"
        "machine-2-5" "machine-2-6" "machine-2-7" "machine-2-8"
        "machine-2-9"
        "machine-3-1" "machine-3-2" "machine-3-3" "machine-3-4"
        "machine-3-5" "machine-3-6" "machine-3-7" "machine-3-8"
        "machine-3-9" "machine-3-10" "machine-3-11"
    )
fi

RESULTS_DIR="results/benchmark_smd"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "PatchTraj SMD Benchmark"
echo "GPU: $GPU_ID | Epochs: $EPOCHS | Entities: ${#ENTITIES[@]}"
echo "=========================================="

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.csv"
echo "entity,auc_roc,auc_pr,best_f1,f1_pa" > "$SUMMARY_FILE"

for entity in "${ENTITIES[@]}"; do
    echo ""
    echo ">>> Training $entity ..."
    OUTPUT_DIR="$RESULTS_DIR/$entity"
    mkdir -p "$OUTPUT_DIR"

    # Train
    python scripts/train_patchtraj.py \
        data.entity="$entity" \
        training.epochs="$EPOCHS" \
        training.patience="$PATIENCE" \
        training.batch_size="$BATCH_SIZE" \
        training.lr="$LR" \
        output_dir="$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/train.log"

    echo ">>> Detecting $entity ..."

    # Detect
    python scripts/detect.py \
        data.entity="$entity" \
        training.batch_size="$BATCH_SIZE" \
        output_dir="$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/detect.log"

    # Extract metrics from JSON
    if [ -f "$OUTPUT_DIR/metrics.json" ]; then
        AUC_ROC=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{m['auc_roc']:.6f}\")")
        AUC_PR=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{m['auc_pr']:.6f}\")")
        BEST_F1=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{m['best_f1']:.6f}\")")
        F1_PA=$(python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"{m['f1_pa']:.6f}\")")
        echo "$entity,$AUC_ROC,$AUC_PR,$BEST_F1,$F1_PA" >> "$SUMMARY_FILE"
        echo ">>> $entity: AUC-ROC=$AUC_ROC | AUC-PR=$AUC_PR | Best-F1=$BEST_F1 | F1-PA=$F1_PA"
    else
        echo "WARNING: No metrics.json for $entity"
        echo "$entity,N/A,N/A,N/A,N/A" >> "$SUMMARY_FILE"
    fi
done

echo ""
echo "=========================================="
echo "Benchmark Complete. Summary:"
echo "=========================================="
cat "$SUMMARY_FILE" | column -t -s ','

# Compute averages
python -c "
import csv, statistics
with open('$SUMMARY_FILE') as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r['auc_roc'] != 'N/A']

if rows:
    avg_roc = statistics.mean(float(r['auc_roc']) for r in rows)
    avg_pr = statistics.mean(float(r['auc_pr']) for r in rows)
    avg_f1 = statistics.mean(float(r['best_f1']) for r in rows)
    avg_pa = statistics.mean(float(r['f1_pa']) for r in rows)
    print(f'\nAverages ({len(rows)} entities):')
    print(f'  AUC-ROC: {avg_roc:.4f}')
    print(f'  AUC-PR:  {avg_pr:.4f}')
    print(f'  Best-F1: {avg_f1:.4f}')
    print(f'  F1-PA:   {avg_pa:.4f}')
    print(f'\nGo/No-Go (AUC-ROC > 0.7): {\"PASS\" if avg_roc > 0.7 else \"NEEDS IMPROVEMENT\"} ({avg_roc:.4f})')
"
