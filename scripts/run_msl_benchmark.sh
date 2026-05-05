#!/bin/bash
# Run LP+RP PatchTraj benchmark on the MSL dataset.
# MSL: 55 features, official train/test split, single entity.
#
# Usage:
#   bash scripts/run_msl_benchmark.sh GPU_ID
#
# Example:
#   bash scripts/run_msl_benchmark.sh 3

set -euo pipefail

GPU_ID="${1:?Usage: $0 GPU_ID}"

EPOCHS=100
PATIENCE=15
BATCH_SIZE=32
LR="5e-4"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="$GPU_ID"

RESULTS_DIR="results/msl_benchmark"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "MSL LP+RP Benchmark on GPU $GPU_ID"
echo "=========================================="

for renderer in line_plot recurrence_plot; do
    OUTPUT_DIR="$RESULTS_DIR/$renderer"
    mkdir -p "$OUTPUT_DIR"

    # Skip if already done
    if [ -f "$OUTPUT_DIR/metrics.json" ] && [ -f "$OUTPUT_DIR/scores.npy" ] && [ -f "$OUTPUT_DIR/labels.npy" ]; then
        echo ">>> Skipping $renderer (already done)"
        continue
    fi

    echo ">>> Training MSL with $renderer..."
    python scripts/train_patchtraj.py \
        data=msl \
        render="$renderer" \
        training.epochs="$EPOCHS" \
        training.patience="$PATIENCE" \
        training.batch_size="$BATCH_SIZE" \
        training.lr="$LR" \
        output_dir="$OUTPUT_DIR" \
        2>&1 | tail -5

    echo ">>> Detecting MSL with $renderer..."
    python scripts/detect.py \
        data=msl \
        render="$renderer" \
        training.batch_size="$BATCH_SIZE" \
        output_dir="$OUTPUT_DIR" \
        scoring.smooth_window=7 \
        2>&1 | tail -8
done

# Ensemble LP+RP
echo ""
echo ">>> Ensembling MSL (LP+RP)..."

python3 -c "
import json
import numpy as np
from scipy.stats import rankdata

lp_scores = np.load('$RESULTS_DIR/line_plot/scores.npy')
rp_scores = np.load('$RESULTS_DIR/recurrence_plot/scores.npy')
labels = np.load('$RESULTS_DIR/line_plot/labels.npy')

# Ensure same length
min_len = min(len(lp_scores), len(rp_scores), len(labels))
lp_scores = lp_scores[:min_len]
rp_scores = rp_scores[:min_len]
labels = labels[:min_len]

# rank_mean ensemble
lp_ranks = rankdata(lp_scores) / len(lp_scores)
rp_ranks = rankdata(rp_scores) / len(rp_scores)
ensemble_scores = (lp_ranks + rp_ranks) / 2.0

# Compute AUC-ROC
from sklearn.metrics import roc_auc_score
lp_auc = roc_auc_score(labels, lp_scores)
rp_auc = roc_auc_score(labels, rp_scores)
ens_auc = roc_auc_score(labels, ensemble_scores)

print(f'line_plot AUC-ROC: {lp_auc:.6f}')
print(f'recurrence_plot AUC-ROC: {rp_auc:.6f}')
print(f'LP+RP ensemble (rank_mean) AUC-ROC: {ens_auc:.6f}')

# Save ensemble results
np.save('$RESULTS_DIR/ensemble_scores.npy', ensemble_scores)
json.dump({'auc_roc': ens_auc, 'lp_auc_roc': lp_auc, 'rp_auc_roc': rp_auc, 'method': 'rank_mean'},
          open('$RESULTS_DIR/ensemble_metrics.json', 'w'), indent=2)
"

echo ""
echo "=========================================="
echo "MSL Benchmark Results"
echo "=========================================="

# Print summary
if [ -f "$RESULTS_DIR/line_plot/metrics.json" ]; then
    echo -n "line_plot:        AUC-ROC="
    python3 -c "import json; m=json.load(open('$RESULTS_DIR/line_plot/metrics.json')); print(f\"{m['auc_roc']:.6f}\")"
fi
if [ -f "$RESULTS_DIR/recurrence_plot/metrics.json" ]; then
    echo -n "recurrence_plot:  AUC-ROC="
    python3 -c "import json; m=json.load(open('$RESULTS_DIR/recurrence_plot/metrics.json')); print(f\"{m['auc_roc']:.6f}\")"
fi
if [ -f "$RESULTS_DIR/ensemble_metrics.json" ]; then
    echo -n "LP+RP ensemble:   AUC-ROC="
    python3 -c "import json; m=json.load(open('$RESULTS_DIR/ensemble_metrics.json')); print(f\"{m['auc_roc']:.6f}\")"
fi

echo ""
echo "Done. Results in $RESULTS_DIR/"
