"""Raw-space Mahalanobis distance baseline for VITS comparison.

Computes Mahalanobis distance on raw time series features (mean-pooled and
flattened window variants) to quantify the added value of the VITS vision
pipeline over a purely statistical raw-feature baseline.

Two variants:
    mean_pooled: Window mean across timesteps → D-dimensional vector.
    flattened:   Full window flattened → W*D-dimensional vector.
                 Tests whether vision backbone compression is beneficial.

Results saved to results/raw_mahalanobis/{dataset}/metrics.json.
Final comparison table printed to stdout.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.covariance import LedoitWolf

from src.data.msl import MSLDataset
from src.data.psm import PSMDataset
from src.data.smd import SMDDataset
from src.data.smap import SMAPDataset
from src.evaluation.metrics import compute_all_metrics
from src.scoring.patchtraj_scorer import smooth_scores

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE: int = 100
STRIDE: int = 1
SMOOTH_WINDOW: int = 5
SMOOTH_METHOD: str = "mean"
DATA_ROOT: Path = Path("data/raw")
RESULTS_ROOT: Path = Path("results/raw_mahalanobis")

# Existing VITS results for comparison table (from dual_only_pilot + benchmarks).
# These are the distributional-only (alpha=0.0, recurrence_plot) results from
# the dual_only_pilot experiment, which isolate the vision-space Mahalanobis signal.
VITS_REFERENCE: dict[str, dict[str, float]] = {
    "smd_machine-1-1": {
        "dist_only_auc_roc": 0.9259,
        "dist_only_auc_pr": 0.7210,
        "dist_only_f1_pa": 0.9620,
        "dual_auc_roc": 0.9275,
        "dual_auc_pr": 0.7270,
        "dual_f1_pa": 0.9509,
    },
    "psm": {
        "dist_only_auc_roc": 0.5563,
        "dist_only_auc_pr": 0.3253,
        "dist_only_f1_pa": 0.8565,
        "dual_auc_roc": 0.5909,
        "dual_auc_pr": 0.3284,
        "dual_f1_pa": 0.8814,
    },
    "msl": {
        "dist_only_auc_roc": 0.5448,
        "dist_only_auc_pr": 0.1575,
        "dist_only_f1_pa": 0.5306,
        "dual_auc_roc": 0.5025,
        "dual_auc_pr": 0.1481,
        "dual_f1_pa": 0.6131,
    },
    "smap": {
        "dist_only_auc_roc": 0.6850,
        "dist_only_auc_pr": 0.3337,
        "dist_only_f1_pa": 0.7263,
        "dual_auc_roc": 0.6462,
        "dual_auc_pr": 0.2347,
        "dual_f1_pa": 0.7288,
    },
}


# ---------------------------------------------------------------------------
# Mahalanobis scorer (pure numpy/sklearn, no GPU)
# ---------------------------------------------------------------------------


class RawMahalanobisScorer:
    """Mahalanobis distance anomaly scorer for raw window feature vectors.

    Fits Ledoit-Wolf shrinkage covariance on training features and scores
    test windows by their squared Mahalanobis distance from the training mean.

    Args:
        eps: Numerical stability floor for degenerate covariances.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self._train_mu: npt.NDArray[np.float64] | None = None
        self._precision: npt.NDArray[np.float64] | None = None
        self._shrinkage: float = float("nan")

    def fit(self, features: npt.NDArray[np.float64]) -> None:
        """Fit Ledoit-Wolf Mahalanobis parameters on training feature vectors.

        Args:
            features: Training feature matrix of shape (N_train, d).

        Raises:
            ValueError: If features is not 2D or has fewer samples than features.
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2D with shape (N, d), got ndim={features.ndim}."
            )
        n_samples, n_features = features.shape
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 training samples, got {n_samples}."
            )
        if n_samples < n_features:
            LOGGER.warning(
                "Fewer training samples (%d) than feature dimensions (%d); "
                "Ledoit-Wolf will regularise heavily.",
                n_samples,
                n_features,
            )

        feats = features.astype(np.float64)
        self._train_mu = feats.mean(axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lw = LedoitWolf().fit(feats)

        self._precision = lw.precision_.astype(np.float64)
        self._shrinkage = float(lw.shrinkage_)
        LOGGER.info(
            "RawMahalanobisScorer fitted: N=%d, d=%d, shrinkage=%.4f",
            n_samples,
            n_features,
            self._shrinkage,
        )

    def score(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute squared Mahalanobis distance for each test feature vector.

        Args:
            features: Test feature matrix of shape (N_test, d).

        Returns:
            Anomaly scores of shape (N_test,).

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If features has wrong shape.
        """
        if self._train_mu is None or self._precision is None:
            raise RuntimeError("RawMahalanobisScorer has not been fitted yet.")
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2D with shape (N, d), got ndim={features.ndim}."
            )

        feats = features.astype(np.float64)
        diff = feats - self._train_mu  # (N, d)
        # Mahalanobis^2: (x-mu)^T Sigma^{-1} (x-mu) for each sample.
        # Decompose into BLAS gemm + element-wise sum so very large d (e.g.
        # 3800-dim flattened SMD windows) does not blow up the triple-tensor
        # einsum cost. Process in chunks of N to keep peak memory bounded.
        n_total = diff.shape[0]
        chunk_size = 4096
        out: npt.NDArray[np.float64] = np.empty(n_total, dtype=np.float64)
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            inter = diff[start:end] @ self._precision  # (chunk, d), BLAS gemm
            out[start:end] = np.einsum("cd,cd->c", inter, diff[start:end])
        return np.maximum(out, 0.0)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def extract_mean_pooled(windows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Mean-pool windows across the time axis.

    Args:
        windows: Window array of shape (N, W, D).

    Returns:
        Feature matrix of shape (N, D).
    """
    return windows.mean(axis=1).astype(np.float64)


def extract_flattened(windows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Flatten each window into a 1D vector.

    Args:
        windows: Window array of shape (N, W, D).

    Returns:
        Feature matrix of shape (N, W*D).
    """
    n = windows.shape[0]
    return windows.reshape(n, -1).astype(np.float64)


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------


def run_dataset(
    dataset_name: str,
    train_windows: npt.NDArray[np.float64],
    test_windows: npt.NDArray[np.float64],
    test_labels: npt.NDArray[np.int64],
) -> dict[str, dict[str, Any]]:
    """Run both Mahalanobis variants for a single dataset.

    Args:
        dataset_name: Human-readable dataset name for logging.
        train_windows: Normal training windows of shape (N_train, W, D).
        test_windows: Test windows of shape (N_test, W, D).
        test_labels: Per-window binary labels of shape (N_test,).

    Returns:
        Dict with keys 'mean_pooled' and 'flattened', each mapping to a
        metrics dict with keys: auc_roc, auc_pr, best_f1, f1_pa.
    """
    n_train, W, D = train_windows.shape
    n_test = test_windows.shape[0]
    n_anomaly = int(test_labels.sum())
    LOGGER.info(
        "%s: train=%d windows, test=%d windows, anomaly_rate=%.1f%%",
        dataset_name,
        n_train,
        n_test,
        100.0 * n_anomaly / max(n_test, 1),
    )

    results: dict[str, dict[str, Any]] = {}

    # --- Variant 1: mean-pooled (D-dimensional) ---
    LOGGER.info("%s: fitting mean-pooled Mahalanobis (d=%d)...", dataset_name, D)
    train_mean = extract_mean_pooled(train_windows)
    test_mean = extract_mean_pooled(test_windows)

    scorer_mean = RawMahalanobisScorer()
    scorer_mean.fit(train_mean)
    raw_scores_mean = scorer_mean.score(test_mean)
    smoothed_mean = smooth_scores(
        raw_scores_mean.astype(np.float64),
        window_size=SMOOTH_WINDOW,
        method=SMOOTH_METHOD,
    )
    metrics_mean = compute_all_metrics(scores=smoothed_mean, labels=test_labels)
    metrics_mean["shrinkage"] = scorer_mean._shrinkage
    metrics_mean["feature_dim"] = D
    results["mean_pooled"] = metrics_mean
    LOGGER.info(
        "%s mean-pooled: AUC-ROC=%.4f, AUC-PR=%.4f, F1-PA=%.4f",
        dataset_name,
        metrics_mean["auc_roc"],
        metrics_mean["auc_pr"],
        metrics_mean["f1_pa"],
    )

    # --- Variant 2: flattened (W*D-dimensional) ---
    flat_dim = W * D
    LOGGER.info(
        "%s: fitting flattened Mahalanobis (d=%d = %d×%d)...",
        dataset_name,
        flat_dim,
        W,
        D,
    )

    # Ledoit-Wolf is feasible only when flat_dim is not catastrophically large.
    # For W=100, D=38 (SMD) → 3800 dims vs ~28k train windows: fine.
    # For W=100, D=55 (MSL) → 5500 dims: Ledoit-Wolf handles this.
    train_flat = extract_flattened(train_windows)
    test_flat = extract_flattened(test_windows)

    scorer_flat = RawMahalanobisScorer()
    scorer_flat.fit(train_flat)
    raw_scores_flat = scorer_flat.score(test_flat)
    smoothed_flat = smooth_scores(
        raw_scores_flat.astype(np.float64),
        window_size=SMOOTH_WINDOW,
        method=SMOOTH_METHOD,
    )
    metrics_flat = compute_all_metrics(scores=smoothed_flat, labels=test_labels)
    metrics_flat["shrinkage"] = scorer_flat._shrinkage
    metrics_flat["feature_dim"] = flat_dim
    results["flattened"] = metrics_flat
    LOGGER.info(
        "%s flattened: AUC-ROC=%.4f, AUC-PR=%.4f, F1-PA=%.4f",
        dataset_name,
        metrics_flat["auc_roc"],
        metrics_flat["auc_pr"],
        metrics_flat["f1_pa"],
    )

    return results


def save_results(
    dataset_key: str,
    results: dict[str, dict[str, Any]],
) -> None:
    """Save per-dataset results to JSON files.

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
    """Print a formatted comparison table.

    Args:
        all_results: Mapping dataset_key → variant → metrics.
    """
    col_w = 12
    metric_keys = ["auc_roc", "auc_pr", "f1_pa"]
    metric_labels = ["AUC-ROC", "AUC-PR ", "F1-PA  "]

    datasets = list(all_results.keys())

    header_sep = "-" * (22 + len(datasets) * (col_w + 3))
    print()
    print("=" * len(header_sep))
    print("  RAW-SPACE MAHALANOBIS BASELINE vs. VITS")
    print("=" * len(header_sep))

    for metric_key, metric_label in zip(metric_keys, metric_labels):
        print(f"\n  {metric_label}")
        # Header row
        row = f"  {'Method':<30}"
        for ds in datasets:
            ds_short = ds.replace("smd_", "SMD/").replace("psm", "PSM").replace("msl", "MSL").replace("smap", "SMAP")
            row += f"  {ds_short:>{col_w}}"
        print(row)
        print("  " + "-" * (30 + len(datasets) * (col_w + 2)))

        # Raw mean-pooled row
        row = f"  {'Raw Mean-Pooled Maha':<30}"
        for ds in datasets:
            val = all_results[ds].get("mean_pooled", {}).get(metric_key, float("nan"))
            row += f"  {val:>{col_w}.4f}"
        print(row)

        # Raw flattened row
        row = f"  {'Raw Flattened Maha':<30}"
        for ds in datasets:
            val = all_results[ds].get("flattened", {}).get(metric_key, float("nan"))
            row += f"  {val:>{col_w}.4f}"
        print(row)

        # VITS distributional-only row
        row = f"  {'VITS Dist-Only (α=0.0)':<30}"
        for ds in datasets:
            ref = VITS_REFERENCE.get(ds, {})
            key_map = {
                "auc_roc": "dist_only_auc_roc",
                "auc_pr": "dist_only_auc_pr",
                "f1_pa": "dist_only_f1_pa",
            }
            val = ref.get(key_map[metric_key], float("nan"))
            row += f"  {val:>{col_w}.4f}"
        print(row)

        # VITS dual-signal row
        row = f"  {'VITS Dual-Signal (α=0.1)':<30}"
        for ds in datasets:
            ref = VITS_REFERENCE.get(ds, {})
            key_map = {
                "auc_roc": "dual_auc_roc",
                "auc_pr": "dual_auc_pr",
                "f1_pa": "dual_f1_pa",
            }
            val = ref.get(key_map[metric_key], float("nan"))
            row += f"  {val:>{col_w}.4f}"
        print(row)

    print()
    print("  Notes:")
    print("  - Raw Mahalanobis uses same window_size=100, stride=1, smooth_window=5.")
    print("  - VITS results: dual_only_pilot (recurrence_plot, DINOv2 patch tokens).")
    print("  - All metrics computed on per-window labels (any-positive convention).")
    print("=" * len(header_sep))
    print()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


# SMD has 28 server-machine entities; the paper's main SMD result is the
# macro-average across all of them. The paper used to report a single-entity
# baseline (machine-1-1) — that is preserved as a per-entity row, but the
# headline "smd" key in summary.json is now the 28-entity macro.
ALL_SMD_ENTITIES: tuple[str, ...] = (
    "machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4",
    "machine-1-5", "machine-1-6", "machine-1-7", "machine-1-8",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4",
    "machine-2-5", "machine-2-6", "machine-2-7", "machine-2-8",
    "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3", "machine-3-4",
    "machine-3-5", "machine-3-6", "machine-3-7", "machine-3-8",
    "machine-3-9", "machine-3-10", "machine-3-11",
)


def load_smd_entity(entity: str) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
]:
    """Load a single SMD entity's train/test windows.

    Args:
        entity: SMD entity name, e.g. ``"machine-1-1"``.

    Returns:
        ``(train_windows, test_windows, test_labels)``.
    """
    LOGGER.info("Loading SMD %s...", entity)
    ds = SMDDataset(
        raw_dir=DATA_ROOT / "smd",
        entity=entity,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        normalize=True,
        norm_method="standard",
    )
    return ds.train_windows, ds.test_windows, ds.test_labels


def load_smd_machine_1_1() -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
]:
    """Backwards-compatible loader for the historical single-entity baseline."""
    return load_smd_entity("machine-1-1")


def load_psm() -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
]:
    """Load PSM train/test windows.

    Returns:
        (train_windows, test_windows, test_labels)
    """
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
    """Load MSL train/test windows.

    Returns:
        (train_windows, test_windows, test_labels)
    """
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
    """Load SMAP train/test windows.

    Returns:
        (train_windows, test_windows, test_labels)
    """
    LOGGER.info("Loading SMAP...")
    ds = SMAPDataset(
        raw_dir=DATA_ROOT / "smap",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        normalize=True,
        norm_method="standard",
    )
    return ds.train_windows, ds.test_windows, ds.test_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run raw-space Mahalanobis baseline across all target datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    LOGGER.info("Raw-space Mahalanobis baseline starting.")
    LOGGER.info(
        "Config: window_size=%d, stride=%d, smooth_window=%d, method=%s",
        WINDOW_SIZE,
        STRIDE,
        SMOOTH_WINDOW,
        SMOOTH_METHOD,
    )

    # PSM / MSL / SMAP are single-entity datasets at this benchmark scale;
    # SMD is run across all 28 entities and aggregated into a macro mean.
    single_entity_loaders = [
        ("psm", load_psm),
        ("msl", load_msl),
        ("smap", load_smap),
    ]

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    # --- SMD: run all 28 entities, then aggregate macro mean/std ---
    LOGGER.info("=" * 60)
    LOGGER.info("Dataset: SMD (%d entities)", len(ALL_SMD_ENTITIES))
    LOGGER.info("=" * 60)

    smd_per_entity: dict[str, dict[str, dict[str, Any]]] = {}
    for entity in ALL_SMD_ENTITIES:
        entity_key = f"smd_{entity}"
        train_windows, test_windows, test_labels = load_smd_entity(entity)
        entity_results = run_dataset(
            dataset_name=entity_key,
            train_windows=train_windows,
            test_windows=test_windows,
            test_labels=test_labels,
        )
        save_results(dataset_key=entity_key, results=entity_results)
        smd_per_entity[entity] = entity_results
        # Preserve legacy machine-1-1 top-level key for back-compat with
        # any tooling that reads the old summary schema.
        if entity == "machine-1-1":
            all_results["smd_machine-1-1"] = entity_results

    # Macro aggregate across the 28 entities.
    metric_keys = ("auc_roc", "auc_pr", "best_f1", "f1_pa")
    smd_macro: dict[str, dict[str, Any]] = {}
    for variant in ("mean_pooled", "flattened"):
        macro: dict[str, Any] = {
            "n": len(smd_per_entity),
            "entities": list(smd_per_entity.keys()),
            "per_entity": {
                entity: {k: smd_per_entity[entity][variant].get(k)
                         for k in metric_keys}
                for entity in smd_per_entity
            },
        }
        for k in metric_keys:
            vals = [
                smd_per_entity[entity][variant].get(k)
                for entity in smd_per_entity
            ]
            vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                macro[f"macro_mean_{k}"] = float(arr.mean())
                macro[f"macro_std_{k}"] = (
                    float(arr.std(ddof=1)) if len(vals) > 1 else 0.0
                )
            else:
                macro[f"macro_mean_{k}"] = float("nan")
                macro[f"macro_std_{k}"] = float("nan")
        smd_macro[variant] = macro
    all_results["smd"] = smd_macro

    # Persist SMD aggregate at results/raw_mahalanobis/smd/summary.json
    smd_dir = RESULTS_ROOT / "smd"
    smd_dir.mkdir(parents=True, exist_ok=True)
    (smd_dir / "summary.json").write_text(
        json.dumps(smd_macro, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "SMD 28-entity macro AUC-ROC: mean_pooled=%.4f, flattened=%.4f",
        smd_macro["mean_pooled"]["macro_mean_auc_roc"],
        smd_macro["flattened"]["macro_mean_auc_roc"],
    )

    # --- Other datasets: single run each ---
    for dataset_key, loader_fn in single_entity_loaders:
        LOGGER.info("=" * 60)
        LOGGER.info("Dataset: %s", dataset_key)
        LOGGER.info("=" * 60)

        train_windows, test_windows, test_labels = loader_fn()
        results = run_dataset(
            dataset_name=dataset_key,
            train_windows=train_windows,
            test_windows=test_windows,
            test_labels=test_labels,
        )
        save_results(dataset_key=dataset_key, results=results)
        all_results[dataset_key] = results

    # Save aggregate summary
    summary_path = RESULTS_ROOT / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    LOGGER.info("Aggregate summary saved: %s", summary_path)

    # Comparison-table view: keep legacy keys so the existing layout works.
    table_view = {
        "smd_macro_28": {
            variant: {
                k: smd_macro[variant].get(f"macro_mean_{k}")
                for k in metric_keys
            }
            for variant in ("mean_pooled", "flattened")
        },
        **{ds: all_results[ds] for ds in ("psm", "msl", "smap")},
    }
    print_comparison_table(table_view)


if __name__ == "__main__":
    main()
