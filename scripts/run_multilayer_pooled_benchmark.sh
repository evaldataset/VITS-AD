#!/bin/bash
# Run multi-layer POOLED Mahalanobis benchmark on all 28 SMD entities.
# Uses GPU 1 and GPU 3 in parallel. Much faster than per-patch version.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
RESULTS_DIR="results/benchmark_smd_ml_pooled"
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

    CUDA_VISIBLE_DEVICES=$gpu_id python -c "
import sys, json, numpy as np, torch, logging
from pathlib import Path
sys.path.insert(0, '.')
from src.data.base import create_sliding_windows, normalize_data, time_based_split
from src.data.smd import _load_smd_matrix, _load_smd_labels
from src.evaluation.metrics import compute_all_metrics
from src.models.backbone import VisionBackbone
from src.rendering.line_plot import render_line_plot_batch
from src.scoring.patchtraj_scorer import normalize_scores, smooth_scores
from src.utils.reproducibility import seed_everything, get_device
from sklearn.covariance import LedoitWolf

logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')
LOG = logging.getLogger()
seed_everything(42)
device = get_device()

raw = Path('data/raw/smd')
entity = '$entity'
tr = _load_smd_matrix(raw/'train'/f'{entity}.txt')
te = _load_smd_matrix(raw/'test'/f'{entity}.txt')
trl = np.zeros(tr.shape[0], dtype=np.int64)
tel = _load_smd_labels(raw/'test_label'/f'{entity}.txt')
data = np.concatenate([tr, te]); labels = np.concatenate([trl, tel])
train_d, train_l, test_d, test_l = time_based_split(data, labels, 0.5)
train_d, test_d = normalize_data(train_d, test_d, 'standard')

train_w, train_wl = create_sliding_windows(train_d, train_l, 100, 10)
train_w = train_w[train_wl == 0]
test_w, _ = create_sliding_windows(test_d, test_l, 100, 10)
starts = np.arange(0, test_l.shape[0]-99, 10)
test_wl = np.array([int(np.any(test_l[s:s+100]==1)) for s in starts], dtype=np.int64)

backbone = VisionBackbone(model_name='facebook/dinov2-base', device=device)
rkw = dict(image_size=224, dpi=56, colormap='tab20', line_width=0.8,
           background_color='white', show_axes=False, show_grid=False)
bs = 32

# Random projection 2304 -> 256
rng = np.random.RandomState(42)
PROJ = rng.randn(2304, 256).astype(np.float64)
PROJ /= np.linalg.norm(PROJ, axis=0, keepdims=True)

def extract_pooled(windows):
    chunks = []
    for s in range(0, windows.shape[0], bs):
        e = min(s+bs, windows.shape[0])
        imgs = render_line_plot_batch(windows[s:e], **rkw)
        t = torch.from_numpy(imgs.astype(np.float32))
        tok = backbone.extract_multilayer_tokens(t, layers=(4,8,12))
        pooled = tok.cpu().numpy().mean(axis=1).astype(np.float64)
        projected = pooled @ PROJ
        chunks.append(projected)
        del tok, pooled
    return np.concatenate(chunks)

train_p = extract_pooled(train_w)
test_p = extract_pooled(test_w)

mu = train_p.mean(axis=0)
lw = LedoitWolf().fit(train_p)
prec = lw.precision_
diff = test_p - mu
raw_scores = np.einsum('bd,de,be->b', diff, prec, diff)
raw_scores = np.maximum(raw_scores, 0.0)
raw_scores = smooth_scores(raw_scores, window_size=5, method='mean')
norm = normalize_scores(raw_scores, method='minmax')
mn = min(norm.shape[0], test_wl.shape[0])
metrics = compute_all_metrics(norm[:mn], test_wl[:mn])

outd = Path('$outdir')
with (outd/'metrics.json').open('w') as f: json.dump(metrics, f, indent=2)
LOG.info('$entity: auc_roc=%.6f', metrics['auc_roc'])
" > "$outdir/run.log" 2>&1

    if [ -f "$outdir/metrics.json" ]; then
        auc=$(python -c "import json; print(f'{json.load(open(\"$outdir/metrics.json\"))[\"auc_roc\"]:.6f}')")
        echo "[GPU $gpu_id] $entity: AUC-ROC=$auc"
    else
        echo "[GPU $gpu_id] $entity: FAILED"
    fi
}

echo "=========================================="
echo "Multi-Layer POOLED Mahalanobis Benchmark"
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
echo "Results:"
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
if aucs:
    print(f'Mean AUC-ROC: {statistics.mean(aucs):.4f} ({len(aucs)} entities)')
    print(f'Std:          {statistics.stdev(aucs):.4f}')
"
