#!/bin/bash
# Run Spatial+Dual-Signal PatchTraj benchmark across all 28 SMD entities.
# Uses 4 GPUs in parallel, each handling 7 entities with LP renderer.
#
# Usage:
#   bash scripts/run_spatial_benchmark.sh

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3

RESULTS_DIR="results/benchmark_smd_spatial"
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

run_entity() {
    local gpu_id=$1
    local entity=$2
    local render=$3
    local outdir="$RESULTS_DIR/${entity}/${render}"
    mkdir -p "$outdir"

    echo "[GPU $gpu_id] Training $entity / $render ..."
    CUDA_VISIBLE_DEVICES=$gpu_id .venv/bin/python scripts/train_patchtraj.py \
        --config-name experiment/patchtraj_spatial \
        data=smd data.entity="$entity" render=$render \
        output_dir="$outdir" \
        2>&1 > "$outdir/train.log"

    echo "[GPU $gpu_id] Detecting $entity / $render ..."
    CUDA_VISIBLE_DEVICES=$gpu_id .venv/bin/python scripts/detect.py \
        --config-name experiment/patchtraj_spatial \
        data=smd data.entity="$entity" render=$render \
        output_dir="$outdir" \
        2>&1 > "$outdir/detect.log"

    # Clean up large intermediate file
    rm -f "$outdir/test_patch_tokens.npy"

    if [ -f "$outdir/metrics.json" ]; then
        auc=$(.venv/bin/python -c "import json; print(f'{json.load(open(\"$outdir/metrics.json\"))[\"auc_roc\"]:.6f}')")
        echo "[GPU $gpu_id] $entity/$render: AUC-ROC=$auc"
    else
        echo "[GPU $gpu_id] $entity/$render: FAILED (no metrics)"
    fi
}

# Run LP on all entities, distribute across 4 GPUs
echo "=========================================="
echo "Spatial+Dual SMD Benchmark (28 entities, LP)"
echo "=========================================="

gpu_id=1
pids=()
for entity in "${ALL_ENTITIES[@]}"; do
    run_entity $gpu_id "$entity" "line_plot" &
    pids+=($!)
    gpu_id=$(( (gpu_id % 3) + 1 ))

    # Limit parallel jobs to 3 (GPUs 1-3)
    if [ ${#pids[@]} -ge 3 ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
wait

echo ""
echo "=========================================="
echo "LP Phase Complete. Starting RP Phase..."
echo "=========================================="

# Run RP on all entities
gpu_id=1
pids=()
for entity in "${ALL_ENTITIES[@]}"; do
    run_entity $gpu_id "$entity" "recurrence_plot" &
    pids+=($!)
    gpu_id=$(( (gpu_id % 3) + 1 ))

    if [ ${#pids[@]} -ge 4 ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
wait

echo ""
echo "=========================================="
echo "All Done. Collecting results..."
echo "=========================================="

# Collect summary
SUMMARY="$RESULTS_DIR/summary.csv"
echo "entity,render,auc_roc,auc_pr,best_f1,f1_pa" > "$SUMMARY"
for entity in "${ALL_ENTITIES[@]}"; do
    for render in line_plot recurrence_plot; do
        mf="$RESULTS_DIR/${entity}/${render}/metrics.json"
        if [ -f "$mf" ]; then
            .venv/bin/python -c "
import json
m = json.load(open('$mf'))
print(f'$entity,$render,{m[\"auc_roc\"]:.6f},{m[\"auc_pr\"]:.6f},{m[\"best_f1\"]:.6f},{m[\"f1_pa\"]:.6f}')
" >> "$SUMMARY"
        fi
    done
done

.venv/bin/python -c "
import csv, statistics
with open('$SUMMARY') as f:
    rows = list(csv.DictReader(f))
lp = [float(r['auc_roc']) for r in rows if r['render'] == 'line_plot']
rp = [float(r['auc_roc']) for r in rows if r['render'] == 'recurrence_plot']
print(f'LP avg AUC-ROC: {statistics.mean(lp):.4f} ({len(lp)} entities)')
print(f'RP avg AUC-ROC: {statistics.mean(rp):.4f} ({len(rp)} entities)')
print(f'Overall avg:    {statistics.mean(lp+rp):.4f}')
"
