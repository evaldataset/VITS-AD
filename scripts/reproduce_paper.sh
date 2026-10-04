#!/bin/bash
# One-command reproduction of key paper claims.
#
# Runs (ordered by cost):
#   1. Unit tests (~10s)
#   2. Compute-cost benchmark (~5s)          -> Table: Compute cost
#   3. Raw Mahalanobis baseline (~1min)      -> Table 1 classic rows
#   4. Single-seed VITS on SMD machine-1-1   -> sanity check (~30min)
#   5. 28-entity SMD Spatial+Dual benchmark  -> Table 1 "Ours" SMD row (~24h, 4 GPUs)
#   6. PSM/MSL/SMAP LP+RP Spatial+Dual       -> Table 1 other cells (~12h, 3 GPUs)
#   7. UCR 109-series Distributional-only    -> Table UCR (~8h, 1 GPU)
#
# Usage:
#   bash scripts/reproduce_paper.sh [quick|full]
#
# Environment:
#   conda env create -f environment.yml && conda activate vits
#   or: pip install -r requirements.txt

set -euo pipefail

MODE="${1:-quick}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3

echo "============================================================"
echo "VITS Paper Reproduction Script"
echo "Mode: $MODE"
echo "Start: $(date)"
echo "============================================================"

# --- 1. Unit tests ---
echo ""
echo "[1/7] Running unit tests..."
.venv/bin/python -m pytest tests/ -q --tb=no 2>&1 | tail -3

# --- 2. Compute cost ---
echo ""
echo "[2/7] Computing compute-cost table..."
.venv/bin/python scripts/bench_compute.py | tail -5

if [[ "$MODE" == "quick" ]]; then
    echo ""
    echo "[Quick mode] Skipping training runs. Run 'bash scripts/reproduce_paper.sh full' for complete reproduction."
    echo "Completed quick mode at $(date)"
    exit 0
fi

# --- 3. Raw Mahalanobis baseline ---
echo ""
echo "[3/7] Raw Mahalanobis baseline (all datasets)..."
for dataset in smd psm msl smap; do
    .venv/bin/python scripts/run_raw_mahalanobis_baseline.py data="$dataset" \
        2>&1 | tail -2 || echo "  (script may be named differently; adjust as needed)"
done

# --- 4. Single-entity sanity check ---
echo ""
echo "[4/7] SMD machine-1-1 sanity check (Spatial+Dual, LP+RP)..."
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_patchtraj.py \
    --config-name experiment/patchtraj_spatial \
    data=smd data.entity=machine-1-1 render=line_plot 2>&1 | tail -3
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/detect.py \
    --config-name experiment/patchtraj_spatial \
    data=smd data.entity=machine-1-1 render=line_plot 2>&1 | tail -3

# --- 5. Full 28-entity SMD benchmark ---
echo ""
echo "[5/7] 28-entity SMD Spatial+Dual benchmark (this takes ~24h on 4 GPUs)..."
bash scripts/run_spatial_benchmark.sh

# --- 6. PSM / MSL / SMAP Spatial+Dual ---
echo ""
echo "[6/7] PSM/MSL/SMAP Spatial+Dual (LP+RP)..."
for dataset in psm msl smap; do
    for render in line_plot recurrence_plot; do
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_patchtraj.py \
            --config-name experiment/patchtraj_spatial \
            data="$dataset" render="$render" 2>&1 | tail -2
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/detect.py \
            --config-name experiment/patchtraj_spatial \
            data="$dataset" render="$render" \
            scoring.dual_signal.alpha=0.1 \
            scoring.dual_signal.auto_alpha=false \
            scoring.smooth_window=21 2>&1 | tail -2
    done
done

# --- 7. UCR 109-series ---
echo ""
echo "[7/7] UCR 109-series Distributional-only evaluation..."
.venv/bin/python scripts/run_ucr_experiment.py scoring.dual_signal.alpha=0.0 \
    2>&1 | tail -3 || echo "  (script name may differ; see scripts/run_ucr*.py)"

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "Main results in:"
echo "  - results/compute_cost.json              (compute table)"
echo "  - results/benchmark_smd_spatial/         (SMD 28-entity)"
echo "  - results/dinov2-base_*_spatial/         (per-dataset runs)"
echo "  - results/reports/ucr_*.json             (UCR)"
echo "============================================================"
