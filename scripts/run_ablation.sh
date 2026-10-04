#!/bin/bash
# Ablation study for PatchTraj components (Exp 6 from RESEARCH.md).
#
# Runs 6 ablation variants:
#   A. π = identity (no geometric correspondence)
#   B. Smaller model (d_model=128, n_layers=1, GRU-like)
#   C. No trimmed loss (trimmed_ratio=0)
#   D. No smoothing (smooth_window=1)
#   E. CLS token only (instead of patch tokens) — requires modified detect.py
#   F. Shorter context (K=4)
#
# Usage:
#   bash scripts/run_ablation.sh GPU_ID DATASET RENDERER [ENTITY]
#
# Example:
#   bash scripts/run_ablation.sh 2 smd line_plot machine-1-1
#   bash scripts/run_ablation.sh 3 psm recurrence_plot

set -euo pipefail

GPU_ID="${1:?Usage: $0 GPU_ID DATASET RENDERER [ENTITY]}"
DATASET="${2:?Usage: $0 GPU_ID DATASET RENDERER [ENTITY]}"
RENDERER="${3:?Usage: $0 GPU_ID DATASET RENDERER [ENTITY]}"
ENTITY="${4:-}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="$GPU_ID"

EPOCHS=100
PATIENCE=15
BATCH_SIZE=32
LR="5e-4"

ABLATION_DIR="results/ablation_${DATASET}"
mkdir -p "$ABLATION_DIR"

ENTITY_ARGS=""
ENTITY_TAG=""
if [ -n "$ENTITY" ]; then
    ENTITY_ARGS="data.entity=$ENTITY"
    ENTITY_TAG="$ENTITY"
else
    ENTITY_TAG="default"
fi

echo "=========================================="
echo "Ablation Study on GPU $GPU_ID"
echo "Dataset: $DATASET | Renderer: $RENDERER | Entity: $ENTITY_TAG"
echo "=========================================="

run_variant() {
    local variant_name="$1"
    shift
    local extra_args=("$@")

    local OUTPUT_DIR="$ABLATION_DIR/${ENTITY_TAG}/${variant_name}/${RENDERER}"
    mkdir -p "$OUTPUT_DIR"

    if [ -f "$OUTPUT_DIR/metrics.json" ] && [ -f "$OUTPUT_DIR/scores.npy" ]; then
        echo ">>> Skipping $variant_name (already done)"
        return 0
    fi

    echo ">>> [$variant_name] Training..."
    python scripts/train_patchtraj.py \
        data="$DATASET" \
        $ENTITY_ARGS \
        render="$RENDERER" \
        training.epochs="$EPOCHS" \
        training.patience="$PATIENCE" \
        training.batch_size="$BATCH_SIZE" \
        training.lr="$LR" \
        output_dir="$OUTPUT_DIR" \
        "${extra_args[@]}" \
        2>&1 | tail -3

    echo ">>> [$variant_name] Detecting..."
    python scripts/detect.py \
        data="$DATASET" \
        $ENTITY_ARGS \
        render="$RENDERER" \
        training.batch_size="$BATCH_SIZE" \
        output_dir="$OUTPUT_DIR" \
        scoring.smooth_window=7 \
        "${extra_args[@]}" \
        2>&1 | tail -5

    if [ -f "$OUTPUT_DIR/metrics.json" ]; then
        local AUC
        AUC=$(python3 -c "import json; print(f\"{json.load(open('$OUTPUT_DIR/metrics.json'))['auc_roc']:.4f}\")")
        echo ">>> [$variant_name] AUC-ROC = $AUC"
    fi
}

# === Baseline (full model, default config) ===
echo ""
echo "--- Baseline (full model) ---"
run_variant "baseline"

# === A. π = identity ===
echo ""
echo "--- A. π = identity (no geometric correspondence) ---"
run_variant "A_pi_identity" "patchtraj.use_identity_pi=true"

# === B. Smaller model (GRU-like) ===
echo ""
echo "--- B. Smaller model (d_model=128, n_layers=1) ---"
run_variant "B_small_model" \
    "patchtraj.d_model=128" \
    "patchtraj.n_heads=4" \
    "patchtraj.n_layers=1" \
    "patchtraj.dim_feedforward=512"

# === C. No trimmed loss ===
echo ""
echo "--- C. No trimmed loss (trimmed_ratio=0) ---"
run_variant "C_no_trim" "training.trimmed_ratio=0.0"

# === D. No smoothing ===
echo ""
echo "--- D. No smoothing (smooth_window=1) ---"
# Train with default, but detect with no smoothing
OUTPUT_DIR="$ABLATION_DIR/${ENTITY_TAG}/D_no_smooth/${RENDERER}"
mkdir -p "$OUTPUT_DIR"
if [ ! -f "$OUTPUT_DIR/metrics.json" ]; then
    # Copy model from baseline if available
    BASELINE_DIR="$ABLATION_DIR/${ENTITY_TAG}/baseline/${RENDERER}"
    if [ -f "$BASELINE_DIR/best_model.pt" ]; then
        cp "$BASELINE_DIR/best_model.pt" "$OUTPUT_DIR/"
        echo ">>> [D_no_smooth] Using baseline model, detecting with no smoothing..."
        python scripts/detect.py \
            data="$DATASET" \
            $ENTITY_ARGS \
            render="$RENDERER" \
            training.batch_size="$BATCH_SIZE" \
            output_dir="$OUTPUT_DIR" \
            scoring.smooth_window=1 \
            2>&1 | tail -5
    else
        echo ">>> [D_no_smooth] No baseline model found, training fresh..."
        run_variant "D_no_smooth" "scoring.smooth_window=1"
    fi
fi

# === F. Shorter context (K=4) ===
echo ""
echo "--- F. Shorter context (K=4 instead of K=8) ---"
run_variant "F_short_context" "patchtraj.K=4"

echo ""
echo "=========================================="
echo "Ablation study complete!"
echo "=========================================="

# Collect results
echo ""
echo "Results summary:"
echo "variant,renderer,auc_roc"
for variant_dir in "$ABLATION_DIR/${ENTITY_TAG}"/*/; do
    variant_name=$(basename "$variant_dir")
    metrics_file="$variant_dir/${RENDERER}/metrics.json"
    if [ -f "$metrics_file" ]; then
        AUC=$(python3 -c "import json; print(f\"{json.load(open('$metrics_file'))['auc_roc']:.6f}\")")
        echo "$variant_name,$RENDERER,$AUC"
    fi
done
