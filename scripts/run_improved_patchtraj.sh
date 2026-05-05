#!/bin/bash
# Run improved PatchTraj (larger model, d=384, L=3, K=12) on all 4 datasets.
#
# Usage:
#   bash scripts/run_improved_patchtraj.sh GPU_ID [DATASET]
#
# Examples:
#   bash scripts/run_improved_patchtraj.sh 2          # all datasets
#   bash scripts/run_improved_patchtraj.sh 2 smd      # SMD only

set -euo pipefail

GPU_ID="${1:?Usage: $0 GPU_ID [DATASET]}"
SINGLE_DATASET="${2:-}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="$GPU_ID"

EPOCHS=150
PATIENCE=20
BATCH_SIZE=32
LR="3e-4"

if [ -n "$SINGLE_DATASET" ]; then
    DATASETS=("$SINGLE_DATASET")
else
    DATASETS=(smd psm msl smap)
fi

echo "=========================================="
echo "Improved PatchTraj on GPU $GPU_ID"
echo "Datasets: ${DATASETS[*]}"
echo "Config: d_model=384, n_layers=3, K=12"
echo "=========================================="

# SMD entities (machine-1-1 through machine-3-11)
SMD_ENTITIES=(
    machine-1-1 machine-1-2 machine-1-3 machine-1-4 machine-1-5
    machine-1-6 machine-1-7 machine-1-8
    machine-2-1 machine-2-2 machine-2-3 machine-2-4 machine-2-5
    machine-2-6 machine-2-7 machine-2-8 machine-2-9
    machine-3-1 machine-3-2 machine-3-3 machine-3-4 machine-3-5
    machine-3-6 machine-3-7 machine-3-8 machine-3-9 machine-3-10
    machine-3-11
)

run_single() {
    local dataset="$1"
    local entity="$2"
    local renderer="$3"
    local entity_args=""
    local entity_tag="default"

    if [ -n "$entity" ]; then
        entity_args="data.entity=$entity"
        entity_tag="$entity"
    fi

    local OUTPUT_DIR="results/improved_${dataset}/${entity_tag}/${renderer}"
    mkdir -p "$OUTPUT_DIR"

    if [ -f "$OUTPUT_DIR/metrics.json" ] && [ -f "$OUTPUT_DIR/scores.npy" ]; then
        echo ">>> Skipping ${dataset}/${entity_tag}/${renderer} (already done)"
        return 0
    fi

    echo ">>> Training ${dataset}/${entity_tag}/${renderer}..."
    python scripts/train_patchtraj.py \
        data="$dataset" \
        $entity_args \
        render="$renderer" \
        patchtraj.K=12 \
        patchtraj.d_model=384 \
        patchtraj.n_heads=6 \
        patchtraj.n_layers=3 \
        patchtraj.dim_feedforward=1536 \
        training.epochs="$EPOCHS" \
        training.patience="$PATIENCE" \
        training.batch_size="$BATCH_SIZE" \
        training.lr="$LR" \
        training.warmup_epochs=10 \
        training.scheduler=cosine \
        output_dir="$OUTPUT_DIR" \
        2>&1 | tail -3

    echo ">>> Detecting ${dataset}/${entity_tag}/${renderer}..."
    python scripts/detect.py \
        data="$dataset" \
        $entity_args \
        render="$renderer" \
        patchtraj.K=12 \
        patchtraj.d_model=384 \
        patchtraj.n_heads=6 \
        patchtraj.n_layers=3 \
        patchtraj.dim_feedforward=1536 \
        training.batch_size="$BATCH_SIZE" \
        output_dir="$OUTPUT_DIR" \
        scoring.smooth_window=21 \
        scoring.smooth_method=mean \
        2>&1 | tail -5

    if [ -f "$OUTPUT_DIR/metrics.json" ]; then
        local AUC
        AUC=$(python3 -c "import json; print(f\"{json.load(open('$OUTPUT_DIR/metrics.json'))['auc_roc']:.4f}\")")
        echo ">>> ${dataset}/${entity_tag}/${renderer} AUC-ROC = $AUC"
    fi
}

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "--- Dataset: $dataset ---"

    if [ "$dataset" = "smd" ]; then
        for entity in "${SMD_ENTITIES[@]}"; do
            for renderer in line_plot recurrence_plot; do
                run_single "$dataset" "$entity" "$renderer"
            done
        done
    else
        for renderer in line_plot recurrence_plot; do
            run_single "$dataset" "" "$renderer"
        done
    fi
done

echo ""
echo "=========================================="
echo "Improved PatchTraj training complete!"
echo "=========================================="

# Collect summary
echo ""
echo "Results summary:"
echo "dataset,entity,renderer,auc_roc"
for dataset in "${DATASETS[@]}"; do
    if [ "$dataset" = "smd" ]; then
        for entity in "${SMD_ENTITIES[@]}"; do
            for renderer in line_plot recurrence_plot; do
                metrics_file="results/improved_${dataset}/${entity}/${renderer}/metrics.json"
                if [ -f "$metrics_file" ]; then
                    AUC=$(python3 -c "import json; print(f\"{json.load(open('$metrics_file'))['auc_roc']:.6f}\")")
                    echo "${dataset},${entity},${renderer},${AUC}"
                fi
            done
        done
    else
        for renderer in line_plot recurrence_plot; do
            metrics_file="results/improved_${dataset}/default/${renderer}/metrics.json"
            if [ -f "$metrics_file" ]; then
                AUC=$(python3 -c "import json; print(f\"{json.load(open('$metrics_file'))['auc_roc']:.6f}\")")
                echo "${dataset},default,${renderer},${AUC}"
            fi
        done
    fi
done
