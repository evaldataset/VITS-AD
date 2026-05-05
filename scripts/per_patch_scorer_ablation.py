"""PerPatchScorer (PaDiM-style) ablation on SMD/PSM/MSL/SMAP.

Fits per-patch Gaussians on cached training tokens, scores cached test tokens,
and reports AUC-ROC alongside DualSignalScorer baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.scoring.dual_signal_scorer import DualSignalScorer, PerPatchScorer


def _load_full_series(dataset: str) -> tuple[np.ndarray, np.ndarray, int | None]:
    raw_dir = Path(f"data/raw/{dataset}")
    if dataset == "smd":
        from src.data.smd import _load_smd_labels, _load_smd_matrix
        entity = "machine-1-1"
        train_data = _load_smd_matrix(raw_dir / "train" / f"{entity}.txt")
        test_data = _load_smd_matrix(raw_dir / "test" / f"{entity}.txt")
        train_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
        test_labels = _load_smd_labels(raw_dir / "test_label" / f"{entity}.txt")
        return (
            np.concatenate([train_data, test_data], axis=0),
            np.concatenate([train_labels, test_labels], axis=0),
            train_data.shape[0],
        )
    if dataset == "psm":
        from src.data.psm import _load_psm_features, _load_psm_labels
        train_data = _load_psm_features(raw_dir / "train.csv")
        test_data = _load_psm_features(raw_dir / "test.csv")
        train_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
        test_labels = _load_psm_labels(raw_dir / "test_label.csv")
        return (
            np.concatenate([train_data, test_data], axis=0),
            np.concatenate([train_labels, test_labels], axis=0),
            train_data.shape[0],
        )
    if dataset == "msl":
        from src.data.msl import _load_msl_labels, _load_msl_matrix
        train_data = _load_msl_matrix(raw_dir / "MSL_train.npy")
        test_data = _load_msl_matrix(raw_dir / "MSL_test.npy")
        train_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
        test_labels = _load_msl_labels(raw_dir / "MSL_test_label.npy")
        return (
            np.concatenate([train_data, test_data], axis=0),
            np.concatenate([train_labels, test_labels], axis=0),
            train_data.shape[0],
        )
    if dataset == "smap":
        from src.data.smap import _load_smap_labels, _load_smap_matrix
        train_data = _load_smap_matrix(raw_dir / "SMAP_train.npy")
        test_data = _load_smap_matrix(raw_dir / "SMAP_test.npy")
        train_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
        test_labels = _load_smap_labels(raw_dir / "SMAP_test_label.npy")
        return (
            np.concatenate([train_data, test_data], axis=0),
            np.concatenate([train_labels, test_labels], axis=0),
            train_data.shape[0],
        )
    raise ValueError(f"Unknown dataset {dataset}")


def _get_test_tokens_from_cache(
    dataset: str, entity: str, render: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached training tokens (for fit) and reconstruct test tokens."""
    train_tokens_p = Path(f"data/processed/{dataset}/tokens_facebook_dinov2-base_{entity}_{render}.pt")
    if not train_tokens_p.exists():
        raise FileNotFoundError(f"Training tokens not found: {train_tokens_p}")

    tokens = torch.load(train_tokens_p, map_location="cpu", weights_only=False)
    if isinstance(tokens, dict):
        tokens = tokens["tokens"]
    train_arr = tokens.numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
    return train_arr.astype(np.float64)


def _get_test_tokens(results_dir: Path, entity: str, render: str) -> np.ndarray | None:
    """Try to load test_patch_tokens.npy."""
    p = results_dir / entity / render / "test_patch_tokens.npy"
    if p.exists():
        return np.load(p).astype(np.float64)
    # Standalone datasets
    p2 = Path(f"results/dinov2-base_{entity}_{render}_spatial/test_patch_tokens.npy")
    if p2.exists():
        return np.load(p2).astype(np.float64)
    return None


def _compute_window_labels(labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    starts = np.arange(0, labels.shape[0] - window_size + 1, stride, dtype=np.int64)
    return np.asarray(
        [int(np.any(labels[start : start + window_size] == 1)) for start in starts],
        dtype=np.int64,
    )


def _smooth(x: np.ndarray, window: int = 21) -> np.ndarray:
    if window <= 1 or window >= len(x):
        return x
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = float(np.mean(padded[i : i + window]))
    return out


def run_ablation(dataset: str, render: str = "line_plot") -> dict:
    entity = {"smd": "machine-1-1", "psm": "psm", "msl": "msl", "smap": "smap"}[dataset]

    # 1. Load training tokens from cache (training data is normal only)
    train_tokens = _get_test_tokens_from_cache(dataset, entity, render)
    print(f"[{dataset}/{render}] Train tokens shape: {train_tokens.shape}")

    # 2. Fit scorers on training tokens
    pooled_scorer = DualSignalScorer()
    pooled_scorer.fit(train_tokens)

    perpatch = PerPatchScorer(aggregation="mean", max_dim=256)
    perpatch.fit(train_tokens)

    perpatch_max = PerPatchScorer(aggregation="max", max_dim=256)
    perpatch_max.fit(train_tokens)

    # 3. Get test tokens. Re-use detection result directory structure.
    results_dir = Path("results/benchmark_smd_spatial_smooth21") if dataset == "smd" else Path(".")
    test_tokens = _get_test_tokens(results_dir, entity, render)
    if test_tokens is None:
        print(f"[{dataset}/{render}] Test tokens not cached; skipping.")
        return {}

    # 4. Load test labels from labels.npy
    if dataset == "smd":
        labels_p = results_dir / entity / render / "labels.npy"
    else:
        labels_p = Path(f"results/dinov2-base_{entity}_{render}_spatial/labels.npy")
    if not labels_p.exists():
        print(f"[{dataset}/{render}] Labels not found at {labels_p}; skipping.")
        return {}
    labels = np.load(labels_p).astype(np.int64)

    # Align lengths
    n = min(len(test_tokens), len(labels))
    test_tokens, labels = test_tokens[:n], labels[:n]
    if len(set(labels)) < 2:
        return {}

    # 5. Score with both methods
    pooled_scores = pooled_scorer.score_distributional(test_tokens)
    perpatch_mean_scores = perpatch.score(test_tokens)
    perpatch_max_scores = perpatch_max.score(test_tokens)

    # 6. Apply smoothing + AUC
    results = {}
    for name, scores in [
        ("pooled_maha", pooled_scores),
        ("perpatch_mean", perpatch_mean_scores),
        ("perpatch_max", perpatch_max_scores),
    ]:
        smoothed = _smooth(scores.astype(np.float64), window=21)
        try:
            auc = float(roc_auc_score(labels, smoothed))
        except ValueError:
            auc = float("nan")
        results[name] = auc

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["smd", "psm", "msl", "smap"],
    )
    parser.add_argument(
        "--renders", nargs="+", default=["line_plot", "recurrence_plot"],
    )
    args = parser.parse_args()

    all_results = {}
    for dataset in args.datasets:
        for render in args.renders:
            key = f"{dataset}_{render}"
            print(f"\n=== {key} ===")
            try:
                res = run_ablation(dataset, render)
                all_results[key] = res
                if res:
                    for method, auc in res.items():
                        print(f"  {method}: AUC-ROC={auc:.4f}")
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                all_results[key] = {"error": str(e)}

    # Save
    out = Path("results/perpatch_ablation_summary.json")
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out}")

    # Summary table
    print(f"\n{'='*70}")
    print("Summary (AUC-ROC with smooth=21):")
    print(f"{'='*70}")
    print(f"{'Dataset/Render':<25} {'Pooled':>8} {'PP-Mean':>8} {'PP-Max':>8}")
    for key, res in all_results.items():
        if "error" in res:
            continue
        pp = res.get("pooled_maha", float("nan"))
        pm = res.get("perpatch_mean", float("nan"))
        px = res.get("perpatch_max", float("nan"))
        print(f"{key:<25} {pp:>8.4f} {pm:>8.4f} {px:>8.4f}")


if __name__ == "__main__":
    main()
