#!/bin/bash
# Run multi-layer pooled Mahalanobis benchmark on all 28 SMD entities.
# Uses GPU 1 and GPU 3 in parallel.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
RESULTS_DIR="results/benchmark_smd_multilayer"
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
    local outdir="$RESULTS_DIR/$entity"
    mkdir -p "$outdir"

    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/pilot_perpatch.py \
        --dataset smd --entity "$entity" --renderer line_plot \
        --layers 4 8 12 --aggregation mean --smooth 5 \
        --batch-size 16 --output-dir "$outdir" \
        > "$outdir/run.log" 2>&1

    if [ -f "$outdir/metrics.json" ]; then
        auc=$(python -c "import json; print(f'{json.load(open(\"$outdir/metrics.json\"))[\"auc_roc\"]:.6f}')")
        echo "[GPU $gpu_id] $entity: AUC-ROC=$auc"
    else
        echo "[GPU $gpu_id] $entity: FAILED"
    fi
}

echo "=========================================="
echo "Multi-Layer Pooled Mahalanobis SMD Benchmark"
echo "GPUs: 1, 3 | Entities: ${#ALL_ENTITIES[@]}"
echo "=========================================="

gpu_ids=(1 3)
pids=()
gpu_idx=0

for entity in "${ALL_ENTITIES[@]}"; do
    gpu=${gpu_ids[$gpu_idx]}
    run_entity $gpu "$entity" &
    pids+=($!)
    gpu_idx=$(( (gpu_idx + 1) % 2 ))

    if [ ${#pids[@]} -ge 2 ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
wait

echo ""
echo "=========================================="
echo "Collecting results..."
echo "=========================================="

python -c "
import json, os, statistics
base = '$RESULTS_DIR'
aucs = []
for entity in sorted(os.listdir(base)):
    mf = os.path.join(base, entity, 'metrics.json')
    if os.path.exists(mf):
        m = json.load(open(mf))
        auc = m['auc_roc']
        aucs.append(auc)
        print(f'{entity}: {auc:.6f}')
print(f'---')
print(f'Mean AUC-ROC: {statistics.mean(aucs):.4f} ({len(aucs)} entities)')
print(f'Std:          {statistics.stdev(aucs):.4f}')
"
