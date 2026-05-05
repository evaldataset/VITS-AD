#!/bin/bash
# Re-run detect on existing SMD 28-entity spatial checkpoints with smooth_window=21.
# Uses existing best_model.pt and test_patch_tokens cache - fast (~2min per entity).

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3

RESULTS_DIR="results/benchmark_smd_spatial_smooth21"
SOURCE_DIR="results/benchmark_smd_spatial"
mkdir -p "$RESULTS_DIR"

ALL_ENTITIES=(
    "machine-1-1" "machine-1-2" "machine-1-3" "machine-1-4"
    "machine-1-5" "machine-1-6" "machine-1-7" "machine-1-8"
    "machine-2-1" "machine-2-2" "machine-2-3" "machine-2-4"
    "machine-2-5" "machine-2-6" "machine-2-7" "machine-2-8"
    "machine-2-9"
    "machine-3-1" "machine-3-2" "machine-3-3" "machine-3-4"
    "machine-3-5" "machine-3-6" "machine-3-7" "machine-3-8"
    "machine-3-9" "machine-3-10" "machine-3-11"
)

redetect_entity() {
    local gpu_id=$1
    local entity=$2
    local render=$3
    local src="$SOURCE_DIR/${entity}/${render}"
    local dst="$RESULTS_DIR/${entity}/${render}"
    mkdir -p "$dst"

    # Copy checkpoint to destination
    cp -f "$src/best_model.pt" "$dst/best_model.pt"

    CUDA_VISIBLE_DEVICES=$gpu_id .venv/bin/python scripts/detect.py \
        --config-name experiment/patchtraj_spatial \
        data=smd data.entity="$entity" render=$render \
        output_dir="$dst" \
        scoring.smooth_window=21 \
        training.batch_size=16 \
        2>&1 > "$dst/detect.log"

    # Clean up large intermediate files
    rm -f "$dst/test_patch_tokens.npy" "$dst/best_model.pt"

    if [ -f "$dst/metrics.json" ]; then
        auc=$(.venv/bin/python -c "import json; print(f'{json.load(open(\"$dst/metrics.json\"))[\"auc_roc\"]:.4f}')")
        echo "[GPU $gpu_id] $entity/$render: AUC-ROC=$auc"
    fi
}

echo "=========================================="
echo "Spatial+Dual SMD Re-detect (smooth=21)"
echo "=========================================="

# LP phase — 3 GPUs (1,2,3 — GPU 0 occupied by other process)
gpu_id=1
pids=()
for entity in "${ALL_ENTITIES[@]}"; do
    redetect_entity $gpu_id "$entity" "line_plot" &
    pids+=($!)
    gpu_id=$(( (gpu_id % 3) + 1 ))
    if [ ${#pids[@]} -ge 3 ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
wait

echo ""
echo "LP done. Starting RP..."

gpu_id=1
pids=()
for entity in "${ALL_ENTITIES[@]}"; do
    redetect_entity $gpu_id "$entity" "recurrence_plot" &
    pids+=($!)
    gpu_id=$(( (gpu_id % 3) + 1 ))
    if [ ${#pids[@]} -ge 3 ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
wait

echo "=========================================="
echo "All detect done."
echo "=========================================="
