"""Standalone PerPatchScorer vs DualSignalScorer (pooled Mahalanobis) ablation.

Renders + extracts tokens + scores test windows directly. Doesn't depend on
cached test_patch_tokens.npy. Training tokens loaded from data/processed/ cache.

Results: AUC-ROC of pooled Mahalanobis vs per-patch PaDiM-style scoring on
test windows for SMD/PSM/MSL/SMAP × LP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data.base import create_sliding_windows, normalize_data
from src.models.backbone import VisionBackbone
from src.rendering.line_plot import render_line_plot_batch
from src.rendering.recurrence_plot import render_recurrence_plot_batch
from src.scoring.dual_signal_scorer import DualSignalScorer, PerPatchScorer


def _load_test_data(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Return test data and timestep labels."""
    raw_dir = Path(f"data/raw/{dataset}")
    if dataset == "smd":
        from src.data.smd import _load_smd_labels, _load_smd_matrix
        entity = "machine-1-1"
        test_data = _load_smd_matrix(raw_dir / "test" / f"{entity}.txt")
        test_labels = _load_smd_labels(raw_dir / "test_label" / f"{entity}.txt")
        train_data = _load_smd_matrix(raw_dir / "train" / f"{entity}.txt")
        # Normalize test using train statistics
        train_data, test_data = normalize_data(
            train_data=train_data, test_data=test_data, method="standard"
        )
        return test_data, test_labels
    if dataset == "psm":
        from src.data.psm import _load_psm_features, _load_psm_labels
        test_data = _load_psm_features(raw_dir / "test.csv")
        test_labels = _load_psm_labels(raw_dir / "test_label.csv")
        train_data = _load_psm_features(raw_dir / "train.csv")
        train_data, test_data = normalize_data(
            train_data=train_data, test_data=test_data, method="standard"
        )
        return test_data, test_labels
    if dataset == "msl":
        from src.data.msl import _load_msl_labels, _load_msl_matrix
        test_data = _load_msl_matrix(raw_dir / "MSL_test.npy")
        test_labels = _load_msl_labels(raw_dir / "MSL_test_label.npy")
        train_data = _load_msl_matrix(raw_dir / "MSL_train.npy")
        train_data, test_data = normalize_data(
            train_data=train_data, test_data=test_data, method="standard"
        )
        return test_data, test_labels
    if dataset == "smap":
        from src.data.smap import _load_smap_labels, _load_smap_matrix
        test_data = _load_smap_matrix(raw_dir / "SMAP_test.npy")
        test_labels = _load_smap_labels(raw_dir / "SMAP_test_label.npy")
        train_data = _load_smap_matrix(raw_dir / "SMAP_train.npy")
        train_data, test_data = normalize_data(
            train_data=train_data, test_data=test_data, method="standard"
        )
        return test_data, test_labels
    raise ValueError(dataset)


def _compute_window_labels(labels, window_size, stride):
    starts = np.arange(0, labels.shape[0] - window_size + 1, stride, dtype=np.int64)
    return np.asarray(
        [int(np.any(labels[s : s + window_size] == 1)) for s in starts],
        dtype=np.int64,
    )


def _load_train_tokens(dataset: str, entity: str, render: str) -> np.ndarray:
    p = Path(f"data/processed/{dataset}/tokens_facebook_dinov2-base_{entity}_{render}.pt")
    tokens = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(tokens, dict):
        tokens = tokens["tokens"]
    return tokens.numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)


@torch.no_grad()
def _extract_test_tokens(
    windows: np.ndarray, backbone: VisionBackbone, batch_size: int = 32, render: str = "line_plot"
) -> np.ndarray:
    chunks = []
    renderer = render_line_plot_batch if render == "line_plot" else render_recurrence_plot_batch
    kwargs = {"image_size": 224}
    if render == "line_plot":
        kwargs.update({"dpi": 100, "colormap": "viridis", "line_width": 1.5,
                       "background_color": "white", "show_axes": False, "show_grid": False})
    elif render == "recurrence_plot":
        kwargs.update({"metric": "euclidean"})
    for start in range(0, windows.shape[0], batch_size):
        end = min(start + batch_size, windows.shape[0])
        imgs = renderer(windows[start:end], **kwargs)
        chunks.append(backbone.extract_patch_tokens_from_numpy(imgs).astype(np.float64))
    return np.concatenate(chunks, axis=0)


def _smooth(x, w=21):
    if w <= 1 or w >= len(x):
        return x
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = float(np.mean(padded[i : i + w]))
    return out


def run(dataset: str, render: str, device: str = "cuda") -> dict:
    entity = {"smd": "machine-1-1", "psm": "psm", "msl": "msl", "smap": "smap"}[dataset]
    window_size = 100
    stride = 1 if dataset == "smap" else 10

    print(f"\n=== {dataset}/{render} ===")

    # 1. Load training tokens from cache (these are normal only already)
    print("Loading training tokens from cache...")
    train_tokens = _load_train_tokens(dataset, entity, render).astype(np.float64)
    print(f"  Train tokens: {train_tokens.shape}")

    # 2. Fit pooled and per-patch scorers
    pooled = DualSignalScorer(alpha=0.0)
    pooled.fit(train_tokens)

    perpatch_mean = PerPatchScorer(aggregation="mean", max_dim=256)
    perpatch_mean.fit(train_tokens)

    perpatch_max = PerPatchScorer(aggregation="max", max_dim=256)
    perpatch_max.fit(train_tokens)

    # 3. Load test data, create windows
    print("Loading test data + creating sliding windows...")
    test_data, test_labels = _load_test_data(dataset)
    windows, _ = create_sliding_windows(
        data=test_data, labels=test_labels, window_size=window_size, stride=stride
    )
    window_labels = _compute_window_labels(test_labels, window_size, stride)
    n_anom = int(window_labels.sum())
    print(f"  Test windows: {windows.shape[0]}, anomalous: {n_anom}")

    # 4. Render + backbone extract
    print("Rendering + extracting test tokens (GPU)...")
    backbone = VisionBackbone(model_name="facebook/dinov2-base", device=torch.device(device))
    test_tokens = _extract_test_tokens(windows, backbone, batch_size=32, render=render)
    print(f"  Test tokens: {test_tokens.shape}")

    # 5. Score
    results = {}
    for name, scorer_fn in [
        ("pooled_maha", lambda: pooled.score_distributional(test_tokens)),
        ("perpatch_mean", lambda: perpatch_mean.score(test_tokens)),
        ("perpatch_max", lambda: perpatch_max.score(test_tokens)),
    ]:
        scores = scorer_fn()
        smoothed = _smooth(scores.astype(np.float64), 21)
        if len(set(window_labels)) < 2:
            auc = float("nan")
        else:
            try:
                auc = float(roc_auc_score(window_labels, smoothed))
            except ValueError:
                auc = float("nan")
        results[name] = auc
        print(f"  {name}: AUC-ROC={auc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["smd", "psm", "msl", "smap"])
    parser.add_argument("--renders", nargs="+", default=["line_plot"])
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    all_results = {}
    for ds in args.datasets:
        for rd in args.renders:
            try:
                r = run(ds, rd, args.device)
                all_results[f"{ds}_{rd}"] = r
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                all_results[f"{ds}_{rd}"] = {"error": str(e)}

    out = Path("results/perpatch_ablation.json")
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out}")

    print(f"\n{'='*70}\nSummary (AUC-ROC, smooth=21):")
    print(f"{'Dataset/Render':<25}{'Pooled':>10}{'PP-Mean':>10}{'PP-Max':>10}")
    for key, res in all_results.items():
        if "error" in res:
            continue
        print(f"{key:<25}{res.get('pooled_maha', float('nan')):>10.4f}{res.get('perpatch_mean', float('nan')):>10.4f}{res.get('perpatch_max', float('nan')):>10.4f}")


if __name__ == "__main__":
    main()
