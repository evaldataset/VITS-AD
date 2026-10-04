"""Per-patch Mahalanobis baseline comparison for VITS.

Compares mean-pooled Mahalanobis (existing DualSignalScorer) against
per-patch Mahalanobis (PerPatchMahalanobisScorer) with max / mean / top-k
aggregation modes on four datasets: SMD machine-1-1, PSM, MSL, SMAP.

Token extraction:
    If a test_patch_tokens.npy cache exists in the spatial_pilot experiment
    directory, it is loaded directly.  Train tokens are always re-extracted
    from the training windows using the frozen backbone (requires the vision
    backbone, but no GPU computation beyond forward passes).

Results are saved to results/perpatch_mahalanobis/{dataset}/metrics.json.
A comparison table is printed to stdout.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/run_perpatch_baseline.py

    # Skip datasets whose token cache is missing (no backbone call):
    PYTHONPATH=. .venv/bin/python scripts/run_perpatch_baseline.py \\
        --skip-missing-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.msl import MSLDataset
from src.data.psm import PSMDataset
from src.data.smap import SMAPDataset
from src.data.smd import SMDDataset
from src.evaluation.metrics import compute_all_metrics
from src.scoring.dual_signal_scorer import DualSignalScorer
from src.scoring.patchtraj_scorer import smooth_scores
from src.scoring.perpatch_scorer import PerPatchMahalanobisScorer

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT: Path = Path("data/raw")
RESULTS_ROOT: Path = Path("results/perpatch_mahalanobis")
SPATIAL_PILOT_ROOT: Path = Path("results/spatial_pilot")

WINDOW_SIZE: int = 100
STRIDE: int = 10
SMOOTH_WINDOW: int = 5
SMOOTH_METHOD: str = "mean"
TOPK: int = 10

# Mapping from dataset key to spatial_pilot subdirectory name.
PILOT_DIRS: dict[str, str] = {
    "smd_machine-1-1": "smd_m11_lp",
    "psm": "psm_lp",
    "msl": "msl_lp",
    "smap": "smap_lp",
}

# Reference metrics from dual_only_pilot (mean-pooled Mahalanobis, alpha=0.0).
MEAN_POOLED_REFERENCE: dict[str, dict[str, float]] = {
    "smd_machine-1-1": {"auc_roc": 0.9259, "auc_pr": 0.7210, "f1_pa": 0.9620},
    "psm": {"auc_roc": 0.5563, "auc_pr": 0.3253, "f1_pa": 0.8565},
    "msl": {"auc_roc": 0.5448, "auc_pr": 0.1575, "f1_pa": 0.5306},
    "smap": {"auc_roc": 0.6850, "auc_pr": 0.3337, "f1_pa": 0.7263},
}


# ---------------------------------------------------------------------------
# Dataset loaders  (returns train_windows, test_windows, test_labels)
# ---------------------------------------------------------------------------


def load_smd_machine_1_1() -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
]:
    """Load SMD machine-1-1 train/test windows."""
    LOGGER.info("Loading SMD machine-1-1...")
    ds = SMDDataset(
        raw_dir=DATA_ROOT / "smd",
        entity="machine-1-1",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        normalize=True,
        norm_method="standard",
    )
    return ds.train_windows, ds.test_windows, ds.test_labels


def load_psm() -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
]:
    """Load PSM train/test windows."""
    LOGGER.info("Loading PSM...")
    ds = PSMDataset(
        raw_dir=DATA_ROOT / "psm",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        normalize=True,
        norm_method="standard",
    )
    return ds.train_windows, ds.test_windows, ds.test_labels


def load_msl() -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
]:
    """Load MSL train/test windows."""
    LOGGER.info("Loading MSL...")
    ds = MSLDataset(
        raw_dir=DATA_ROOT / "msl",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        normalize=True,
        norm_method="standard",
    )
    return ds.train_windows, ds.test_windows, ds.test_labels


def load_smap() -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
]:
    """Load SMAP train/test windows."""
    LOGGER.info("Loading SMAP...")
    ds = SMAPDataset(
        raw_dir=DATA_ROOT / "smap",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        normalize=True,
        norm_method="standard",
    )
    return ds.train_windows, ds.test_windows, ds.test_labels


def _compute_window_labels(
    labels: npt.NDArray[np.int64],
    window_size: int,
    stride: int,
) -> npt.NDArray[np.int64]:
    """Map pointwise labels to window labels (any-positive convention).

    Args:
        labels: Pointwise binary label array of shape ``(T,)``.
        window_size: Window length.
        stride: Stride between windows.

    Returns:
        Window-level binary labels of shape ``(N_windows,)``.
    """
    n_windows = max(0, (len(labels) - window_size) // stride + 1)
    window_labels = np.zeros(n_windows, dtype=np.int64)
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window_labels[i] = int(labels[start:end].any())
    return window_labels


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


def _load_or_extract_tokens(
    dataset_key: str,
    train_windows: npt.NDArray[np.float64],
    test_windows: npt.NDArray[np.float64],
    skip_missing_cache: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Load cached test tokens and extract train tokens.

    Tries to load test tokens from the spatial_pilot cache.  Train tokens
    are always re-extracted via the backbone (no per-run cache).

    Args:
        dataset_key: Dataset identifier string.
        train_windows: Training windows of shape ``(N_train, W, D)``.
        test_windows: Test windows of shape ``(N_test, W, D)``.
        skip_missing_cache: If True, return None when no cache exists
            rather than running backbone extraction.

    Returns:
        ``(train_tokens, test_tokens)`` each of shape ``(N, 256, 768)``,
        or None if extraction is skipped.
    """
    pilot_subdir = PILOT_DIRS.get(dataset_key)
    test_cache_path: Path | None = None
    if pilot_subdir is not None:
        candidate = SPATIAL_PILOT_ROOT / pilot_subdir / "test_patch_tokens.npy"
        if candidate.exists():
            test_cache_path = candidate

    if test_cache_path is None and skip_missing_cache:
        LOGGER.warning(
            "%s: no test token cache found and --skip-missing-cache set; skipping.",
            dataset_key,
        )
        return None

    # Load backbone.  Deferred import so the script can run without GPU if
    # all caches are available (though train tokens always need extraction).
    try:
        from src.models.backbone import VisionBackbone
        from src.rendering.line_plot import render_line_plot_batch
        from src.utils.reproducibility import get_device
    except ImportError as exc:
        LOGGER.error("Failed to import backbone modules: %s", exc)
        if skip_missing_cache:
            return None
        raise

    device = get_device()
    backbone = VisionBackbone(
        model_name="facebook/dinov2-base",
        device=device,
    )
    render_kwargs: dict[str, Any] = {
        "image_size": 224,
        "dpi": 100,
        "colormap": "tab10",
        "background_color": "white",
        "line_width": 1.0,
        "show_axes": False,
        "show_grid": False,
    }

    batch_size = 32

    def _extract(windows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        chunks: list[npt.NDArray[np.float64]] = []
        with torch.no_grad():
            for start in range(0, windows.shape[0], batch_size):
                end = min(start + batch_size, windows.shape[0])
                images = render_line_plot_batch(windows[start:end], **render_kwargs)
                tokens = backbone.extract_patch_tokens_from_numpy(images)
                chunks.append(tokens.astype(np.float64, copy=False))
        return np.concatenate(chunks, axis=0)

    # Extract train tokens.
    LOGGER.info("%s: extracting train tokens (%d windows)...", dataset_key, train_windows.shape[0])
    train_tokens = _extract(train_windows)
    LOGGER.info("%s: train_tokens shape=%s", dataset_key, train_tokens.shape)

    # Load or extract test tokens.
    if test_cache_path is not None:
        LOGGER.info("%s: loading cached test tokens from %s", dataset_key, test_cache_path)
        test_tokens = np.load(test_cache_path).astype(np.float64)
    else:
        LOGGER.info("%s: extracting test tokens (%d windows)...", dataset_key, test_windows.shape[0])
        test_tokens = _extract(test_windows)

    LOGGER.info("%s: test_tokens shape=%s", dataset_key, test_tokens.shape)
    return train_tokens, test_tokens


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------


def run_dataset(
    dataset_key: str,
    train_tokens: npt.NDArray[np.float64],
    test_tokens: npt.NDArray[np.float64],
    test_labels: npt.NDArray[np.int64],
) -> dict[str, dict[str, Any]]:
    """Run mean-pooled and all per-patch variants for one dataset.

    Args:
        dataset_key: Human-readable dataset identifier.
        train_tokens: Training patch tokens of shape ``(N_train, 256, D)``.
        test_tokens: Test patch tokens of shape ``(N_test, 256, D)``.
        test_labels: Per-window binary labels of shape ``(N_test,)``.

    Returns:
        Dict mapping variant name to metrics dict.
    """
    n_train = train_tokens.shape[0]
    n_test = test_tokens.shape[0]
    n_patches = train_tokens.shape[1]
    hidden_dim = train_tokens.shape[2]
    n_anomaly = int(test_labels.sum())

    LOGGER.info(
        "%s: train=%d, test=%d (anomaly_rate=%.1f%%), patches=%d, D=%d",
        dataset_key,
        n_train,
        n_test,
        100.0 * n_anomaly / max(n_test, 1),
        n_patches,
        hidden_dim,
    )

    # Align test_labels length to test_tokens if needed (cache may have
    # different stride than locally computed windows).
    if len(test_labels) != n_test:
        LOGGER.warning(
            "%s: test_labels length %d != test_tokens length %d; truncating to min.",
            dataset_key,
            len(test_labels),
            n_test,
        )
        min_len = min(len(test_labels), n_test)
        test_labels = test_labels[:min_len]
        test_tokens = test_tokens[:min_len]

    results: dict[str, dict[str, Any]] = {}

    # --- Variant 0: mean-pooled (replicate DualSignalScorer distributional) ---
    LOGGER.info("%s: mean-pooled Mahalanobis...", dataset_key)
    mean_scorer = DualSignalScorer(alpha=0.0)
    mean_scorer.fit(train_tokens)
    raw_mean = mean_scorer.score_distributional(test_tokens)
    smoothed_mean = smooth_scores(raw_mean.astype(np.float64), window_size=SMOOTH_WINDOW, method=SMOOTH_METHOD)
    metrics_mean = compute_all_metrics(scores=smoothed_mean, labels=test_labels)
    results["mean_pooled"] = metrics_mean
    LOGGER.info(
        "%s mean_pooled: AUC-ROC=%.4f, AUC-PR=%.4f, F1-PA=%.4f",
        dataset_key, metrics_mean["auc_roc"], metrics_mean["auc_pr"], metrics_mean["f1_pa"],
    )

    # --- Variants 1-3: per-patch (max / mean / topk) ---
    for agg in ("max", "mean", "topk"):
        variant_key = f"perpatch_{agg}"
        kw: dict[str, Any] = {"aggregation": agg}
        if agg == "topk":
            kw["topk"] = TOPK
        LOGGER.info("%s: fitting per-patch Mahalanobis (agg=%s)...", dataset_key, agg)
        pp_scorer = PerPatchMahalanobisScorer(**kw)
        pp_scorer.fit(train_tokens)
        raw_pp = pp_scorer.score(test_tokens)
        smoothed_pp = smooth_scores(raw_pp.astype(np.float64), window_size=SMOOTH_WINDOW, method=SMOOTH_METHOD)
        metrics_pp = compute_all_metrics(scores=smoothed_pp, labels=test_labels)
        results[variant_key] = metrics_pp
        LOGGER.info(
            "%s %s: AUC-ROC=%.4f, AUC-PR=%.4f, F1-PA=%.4f",
            dataset_key, variant_key,
            metrics_pp["auc_roc"], metrics_pp["auc_pr"], metrics_pp["f1_pa"],
        )

        # Save scorer checkpoint.
        out_dir = RESULTS_ROOT / dataset_key / variant_key
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "scorer.npz"
        state = pp_scorer.state_dict()
        np.savez_compressed(
            ckpt_path,
            aggregation=np.array(state["aggregation"]),
            topk=np.array(state["topk"]),
            eps=np.array(state["eps"]),
            n_patches=np.array(state["n_patches"]),
            hidden_dim=np.array(state["hidden_dim"]),
        )
        LOGGER.info("Saved scorer checkpoint: %s", ckpt_path)

    return results


def save_results(
    dataset_key: str,
    results: dict[str, dict[str, Any]],
) -> None:
    """Save per-dataset variant results to JSON files.

    Args:
        dataset_key: Filesystem-safe dataset identifier.
        results: Dict returned by run_dataset().
    """
    for variant, metrics in results.items():
        out_dir = RESULTS_ROOT / dataset_key / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        LOGGER.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def print_comparison_table(
    all_results: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Print AUC-ROC comparison: mean-pooled vs per-patch variants.

    Args:
        all_results: Mapping dataset_key -> variant -> metrics.
    """
    datasets = list(all_results.keys())
    variants = ["mean_pooled", "perpatch_max", "perpatch_mean", "perpatch_topk"]
    variant_labels = [
        "Mean-Pooled Maha",
        "Per-Patch (max)",
        "Per-Patch (mean)",
        f"Per-Patch (top-{TOPK})",
    ]
    metric_keys = ["auc_roc", "auc_pr", "f1_pa"]
    metric_labels = ["AUC-ROC", "AUC-PR ", "F1-PA  "]

    col_w = 14
    n_cols = len(datasets)
    header_sep = "=" * (32 + n_cols * (col_w + 2))

    print()
    print(header_sep)
    print("  PER-PATCH vs. MEAN-POOLED MAHALANOBIS COMPARISON")
    print(header_sep)

    for metric_key, metric_label in zip(metric_keys, metric_labels):
        print(f"\n  Metric: {metric_label}")
        row = f"  {'Variant':<32}"
        for ds in datasets:
            ds_short = (
                ds.replace("smd_machine-1-1", "SMD/m-1-1")
                .replace("psm", "PSM")
                .replace("msl", "MSL")
                .replace("smap", "SMAP")
            )
            row += f"  {ds_short:>{col_w}}"
        print(row)
        print("  " + "-" * (32 + n_cols * (col_w + 2)))

        for variant, label in zip(variants, variant_labels):
            row = f"  {label:<32}"
            for ds in datasets:
                val = all_results.get(ds, {}).get(variant, {}).get(metric_key, float("nan"))
                row += f"  {val:>{col_w}.4f}"
            print(row)

        # Reference row from pilot
        row = f"  {'[Pilot] Mean-Pooled Ref':<32}"
        for ds in datasets:
            val = MEAN_POOLED_REFERENCE.get(ds, {}).get(metric_key, float("nan"))
            row += f"  {val:>{col_w}.4f}"
        print(row)

    print()
    print("  Notes:")
    print(f"  - window_size={WINDOW_SIZE}, stride={STRIDE}, smooth_window={SMOOTH_WINDOW}")
    print(f"  - Per-patch top-k uses k={TOPK} out of 256 patches")
    print("  - Backbone: facebook/dinov2-base (frozen), renderer: line_plot")
    print("  - Pilot reference: dual_only_pilot (recurrence_plot, DINOv2)")
    print(header_sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Per-patch Mahalanobis vs mean-pooled comparison."
    )
    parser.add_argument(
        "--skip-missing-cache",
        action="store_true",
        default=False,
        help="Skip datasets whose test token cache does not exist.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["smd_machine-1-1", "psm", "msl", "smap"],
        metavar="DATASET",
        help="Datasets to run (default: all four).",
    )
    return parser.parse_args()


def main() -> None:
    """Run per-patch Mahalanobis baseline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    LOGGER.info("Per-patch Mahalanobis baseline starting.")
    LOGGER.info(
        "Config: window_size=%d, stride=%d, smooth_window=%d, topk=%d",
        WINDOW_SIZE, STRIDE, SMOOTH_WINDOW, TOPK,
    )

    dataset_loaders: dict[
        str,
        Any,
    ] = {
        "smd_machine-1-1": load_smd_machine_1_1,
        "psm": load_psm,
        "msl": load_msl,
        "smap": load_smap,
    }

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for dataset_key in args.datasets:
        loader_fn = dataset_loaders.get(dataset_key)
        if loader_fn is None:
            LOGGER.error("Unknown dataset: %s. Skipping.", dataset_key)
            continue

        LOGGER.info("=" * 60)
        LOGGER.info("Dataset: %s", dataset_key)
        LOGGER.info("=" * 60)

        try:
            train_windows, test_windows, test_labels = loader_fn()
        except (FileNotFoundError, OSError) as exc:
            LOGGER.warning("%s: data not found (%s). Skipping.", dataset_key, exc)
            continue

        token_result = _load_or_extract_tokens(
            dataset_key=dataset_key,
            train_windows=train_windows,
            test_windows=test_windows,
            skip_missing_cache=args.skip_missing_cache,
        )
        if token_result is None:
            LOGGER.info("%s: skipped (no tokens).", dataset_key)
            continue

        train_tokens, test_tokens = token_result
        results = run_dataset(
            dataset_key=dataset_key,
            train_tokens=train_tokens,
            test_tokens=test_tokens,
            test_labels=test_labels,
        )
        save_results(dataset_key=dataset_key, results=results)
        all_results[dataset_key] = results

    if not all_results:
        LOGGER.warning("No datasets completed. Check data/cache availability.")
        return

    # Save aggregate summary.
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_ROOT / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    LOGGER.info("Aggregate summary saved: %s", summary_path)

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
