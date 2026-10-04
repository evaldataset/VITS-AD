"""Pilot: Per-Patch Mahalanobis + Multi-Layer Features anomaly detection.

Standalone script that evaluates the PaDiM-style per-patch distributional
scoring on existing datasets.  Does NOT require PatchTraj training — uses
only frozen DINOv2 features.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/pilot_perpatch.py \
        --dataset psm --renderer line_plot --layers 4 8 12 --group-size 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.base import create_sliding_windows, normalize_data, time_based_split
from src.evaluation.metrics import compute_all_metrics
from src.models.backbone import VisionBackbone
from src.rendering.line_plot import render_line_plot_batch
from src.rendering.recurrence_plot import render_recurrence_plot_batch
from src.scoring.dual_signal_scorer import PerPatchScorer
from src.scoring.patchtraj_scorer import normalize_scores, smooth_scores
from src.utils.reproducibility import get_device, seed_everything

LOGGER = logging.getLogger(__name__)

DATASET_LOADERS = {
    "smd": ("data/raw/smd", "smd"),
    "psm": ("data/raw/psm", "psm"),
    "msl": ("data/raw/msl", "msl"),
    "smap": ("data/raw/smap", "smap"),
}


def _load_dataset(name: str, entity: str = "machine-1-1") -> tuple[np.ndarray, np.ndarray, float]:
    """Load train+test arrays and return (data, labels, train_ratio)."""
    if name == "smd":
        from src.data.smd import _load_smd_labels, _load_smd_matrix
        raw = Path(DATASET_LOADERS[name][0])
        train = _load_smd_matrix(raw / "train" / f"{entity}.txt")
        test = _load_smd_matrix(raw / "test" / f"{entity}.txt")
        train_labels = np.zeros(train.shape[0], dtype=np.int64)
        test_labels = _load_smd_labels(raw / "test_label" / f"{entity}.txt")
        data = np.concatenate([train, test])
        labels = np.concatenate([train_labels, test_labels])
        return data, labels, 0.5
    if name == "psm":
        from src.data.psm import _load_psm_features, _load_psm_labels
        raw = Path(DATASET_LOADERS[name][0])
        train = _load_psm_features(raw / "train.csv")
        test = _load_psm_features(raw / "test.csv")
        train_labels = np.zeros(train.shape[0], dtype=np.int64)
        test_labels = _load_psm_labels(raw / "test_label.csv")
        data = np.concatenate([train, test])
        labels = np.concatenate([train_labels, test_labels])
        return data, labels, 0.5
    if name == "msl":
        from src.data.msl import _load_msl_labels, _load_msl_matrix
        raw = Path(DATASET_LOADERS[name][0])
        train = _load_msl_matrix(raw / "MSL_train.npy")
        test = _load_msl_matrix(raw / "MSL_test.npy")
        train_labels = np.zeros(train.shape[0], dtype=np.int64)
        test_labels = _load_msl_labels(raw / "MSL_test_label.npy")
        data = np.concatenate([train, test])
        labels = np.concatenate([train_labels, test_labels])
        ratio = train.shape[0] / data.shape[0]
        return data, labels, ratio
    if name == "smap":
        from src.data.smap import _load_smap_labels, _load_smap_matrix
        raw = Path(DATASET_LOADERS[name][0])
        train = _load_smap_matrix(raw / "SMAP_train.npy")
        test = _load_smap_matrix(raw / "SMAP_test.npy")
        train_labels = np.zeros(train.shape[0], dtype=np.int64)
        test_labels = _load_smap_labels(raw / "SMAP_test_label.npy")
        data = np.concatenate([train, test])
        labels = np.concatenate([train_labels, test_labels])
        ratio = train.shape[0] / data.shape[0]
        return data, labels, ratio
    raise ValueError(f"Unknown dataset: {name}")


def _get_renderer(name: str):
    if name == "line_plot":
        return render_line_plot_batch
    if name == "recurrence_plot":
        return render_recurrence_plot_batch
    raise ValueError(f"Unknown renderer: {name}")


def _render_kwargs(renderer: str) -> dict:
    if renderer == "line_plot":
        return {"image_size": 224, "dpi": 56, "colormap": "tab20",
                "line_width": 0.8, "background_color": "white",
                "show_axes": False, "show_grid": False}
    if renderer == "recurrence_plot":
        return {"image_size": 224, "metric": "euclidean"}
    return {}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["smd", "psm", "msl", "smap"])
    parser.add_argument("--entity", default="machine-1-1")
    parser.add_argument("--renderer", default="line_plot", choices=["line_plot", "recurrence_plot"])
    parser.add_argument("--layers", nargs="+", type=int, default=[12])
    parser.add_argument("--group-size", type=int, default=0, help="0=no grouping")
    parser.add_argument("--aggregation", default="mean", choices=["mean", "max", "p95"])
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    seed_everything(args.seed)
    device = get_device()
    use_multilayer = len(args.layers) > 1 or args.layers != [12]

    tag = f"perpatch_{'ml' if use_multilayer else 'sl'}_L{'_'.join(map(str, args.layers))}"
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"results/pilot_{tag}_{args.dataset}_{args.renderer}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    LOGGER.info("Loading dataset %s (entity=%s)", args.dataset, args.entity)
    data, labels, train_ratio = _load_dataset(args.dataset, args.entity)
    train_data, train_labels, test_data, test_labels = time_based_split(
        data, labels, train_ratio
    )
    train_data, test_data = normalize_data(train_data, test_data, method="standard")

    train_windows, train_wlabels = create_sliding_windows(
        train_data, train_labels, args.window_size, args.stride
    )
    train_windows = train_windows[train_wlabels == 0]  # normal only

    test_windows, _ = create_sliding_windows(
        test_data, test_labels, args.window_size, args.stride
    )

    # Window labels for evaluation
    test_starts = np.arange(0, test_labels.shape[0] - args.window_size + 1, args.stride)
    test_wlabels = np.array(
        [int(np.any(test_labels[s : s + args.window_size] == 1)) for s in test_starts],
        dtype=np.int64,
    )

    LOGGER.info("Train windows: %d (normal), Test windows: %d", train_windows.shape[0], test_windows.shape[0])

    # --- Extract tokens ---
    backbone = VisionBackbone(model_name="facebook/dinov2-base", device=device)
    render_fn = _get_renderer(args.renderer)
    rkw = _render_kwargs(args.renderer)
    bs = args.batch_size
    layers_tuple = tuple(args.layers)

    def _extract_tokens(windows: np.ndarray) -> np.ndarray:
        chunks = []
        for start in range(0, windows.shape[0], bs):
            end = min(start + bs, windows.shape[0])
            images = render_fn(windows[start:end], **rkw)
            img_tensor = torch.from_numpy(images.astype(np.float32))
            if use_multilayer:
                tokens = backbone.extract_multilayer_tokens(img_tensor, layers=layers_tuple)
            else:
                tokens = backbone.extract_patch_tokens(img_tensor)
            chunks.append(tokens.cpu().numpy().astype(np.float64))
        return np.concatenate(chunks, axis=0)

    LOGGER.info("Extracting train tokens (layers=%s)...", args.layers)
    train_tokens = _extract_tokens(train_windows)
    LOGGER.info("Extracting test tokens...")
    test_tokens = _extract_tokens(test_windows)
    LOGGER.info("Token shapes: train=%s, test=%s", train_tokens.shape, test_tokens.shape)

    # --- Fit PerPatchScorer ---
    scorer = PerPatchScorer(aggregation=args.aggregation)
    LOGGER.info("Fitting PerPatchScorer (aggregation=%s)...", args.aggregation)
    scorer.fit(train_tokens)

    # --- Score ---
    LOGGER.info("Scoring test windows...")
    score_chunks = []
    for start in range(0, test_tokens.shape[0], 4096):
        end = min(start + 4096, test_tokens.shape[0])
        score_chunks.append(scorer.score(test_tokens[start:end]))
    raw_scores = np.concatenate(score_chunks)

    # Smooth + normalize
    if args.smooth > 1:
        sw = args.smooth if args.smooth % 2 == 1 else args.smooth + 1
        raw_scores = smooth_scores(raw_scores, window_size=sw, method="mean")
    normalized = normalize_scores(raw_scores, method="minmax")

    # Align lengths
    min_len = min(normalized.shape[0], test_wlabels.shape[0])
    normalized = normalized[:min_len]
    eval_labels = test_wlabels[:min_len]

    # --- Metrics ---
    metrics = compute_all_metrics(normalized, eval_labels)
    LOGGER.info("=== Results (%s %s, layers=%s) ===", args.dataset, args.renderer, args.layers)
    for k, v in metrics.items():
        LOGGER.info("  %s = %.6f", k, v)

    # Save
    np.save(out_dir / "scores.npy", normalized)
    np.save(out_dir / "labels.npy", eval_labels)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    LOGGER.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
