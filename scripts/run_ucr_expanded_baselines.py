from __future__ import annotations

# pyright: basic, reportMissingImports=false, reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportIndexIssue=false, reportImplicitOverride=false, reportUntypedFunctionDecorator=false, reportArgumentType=false

"""Expanded UCR baselines: Raw Mahalanobis + LOF + IF + OCSVM on all eligible series.

Runs purely statistical baselines (no GPU) on the same UCR series set as
run_ucr_expanded.py:
  - Raw Mean-Pooled Mahalanobis (Ledoit-Wolf)
  - Raw Flattened Mahalanobis   (Ledoit-Wolf)
  - LOF (Local Outlier Factor)
  - Isolation Forest
  - One-Class SVM

Results:
  results/ucr_expanded_raw/per_series/<series_name>.json  — per-series metrics
  results/ucr_expanded_raw/summary.json                   — mean±std across all series
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.base import create_sliding_windows, normalize_data
from src.data.ucr import list_ucr_files, load_ucr_series
from src.evaluation.metrics import compute_all_metrics
from src.scoring.patchtraj_scorer import smooth_scores


LOGGER = logging.getLogger(__name__)

# ---- Hyperparameters --------------------------------------------------------
SEED = 42
WINDOW_SIZE = 100
STRIDE = 1
SMOOTH_WINDOW = 5
SMOOTH_METHOD = "mean"
OCSVM_MAX_TRAIN = 50000

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


# ---- Mahalanobis scorer -----------------------------------------------------


class _RawMahalanobisScorer:
    """Ledoit-Wolf Mahalanobis scorer for raw window feature vectors."""

    def __init__(self) -> None:
        self._train_mu: FloatArray | None = None
        self._precision: FloatArray | None = None

    def fit(self, features: FloatArray) -> None:
        feats = features.astype(np.float64)
        self._train_mu = feats.mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lw = LedoitWolf().fit(feats)
        self._precision = lw.precision_.astype(np.float64)

    def score(self, features: FloatArray) -> FloatArray:
        assert self._train_mu is not None and self._precision is not None
        feats = features.astype(np.float64)
        diff = feats - self._train_mu
        raw: FloatArray = np.einsum("nd,de,ne->n", diff, self._precision, diff)
        return np.maximum(raw, 0.0)


# ---- Feature extractors -----------------------------------------------------


def _mean_pooled(windows: FloatArray) -> FloatArray:
    return windows.mean(axis=1).astype(np.float64)


def _flattened(windows: FloatArray) -> FloatArray:
    n = windows.shape[0]
    return windows.reshape(n, -1).astype(np.float64)


def _last_step(windows: FloatArray) -> FloatArray:
    """Last time-step of each window — cheap feature for LOF/IF/OCSVM."""
    return windows[:, -1, :].astype(np.float64)


# ---- Classical scorers ------------------------------------------------------


def _score_lof(train_feats: FloatArray, test_feats: FloatArray) -> FloatArray:
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_feats)
    test_s = scaler.transform(test_feats)
    n_neighbors = max(2, min(20, train_s.shape[0] - 1))
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination="auto")
    model.fit(train_s)
    return -model.decision_function(test_s)


def _score_if(train_feats: FloatArray, test_feats: FloatArray) -> FloatArray:
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_feats)
    test_s = scaler.transform(test_feats)
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=SEED)
    model.fit(train_s)
    return -model.decision_function(test_s)


def _score_ocsvm(train_feats: FloatArray, test_feats: FloatArray) -> FloatArray:
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_feats)
    test_s = scaler.transform(test_feats)
    fit_data = train_s
    if train_s.shape[0] > OCSVM_MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(train_s.shape[0], size=OCSVM_MAX_TRAIN, replace=False)
        fit_data = train_s[idx]
    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.01)
    model.fit(fit_data)
    return -model.decision_function(test_s)


# ---- Eligibility filter (mirrors run_ucr_expanded.py) ----------------------


def _has_both_classes_in_test_windows(labels: IntArray) -> bool:
    split_idx = labels.shape[0] // 2
    test_labels = labels[split_idx:]
    if np.unique(test_labels).size < 2:
        return False
    dummy = np.zeros((test_labels.shape[0], 1), dtype=np.float64)
    _, window_labels = create_sliding_windows(dummy, test_labels, WINDOW_SIZE, STRIDE)
    return np.unique(window_labels).size == 2


def _find_all_eligible_files(ucr_dir: Path) -> list[Path]:
    all_files = list_ucr_files(ucr_dir)
    eligible: list[Path] = []
    for path in all_files:
        upper = path.name.upper()
        if "DISTORTED" in upper or "NOISE" in upper:
            continue
        try:
            _, labels, _, _ = load_ucr_series(path)
        except (ValueError, OSError) as exc:
            LOGGER.warning("Skipping %s: %s", path.name, exc)
            continue
        if _has_both_classes_in_test_windows(labels):
            eligible.append(path)
    LOGGER.info("Found %d eligible UCR series.", len(eligible))
    return eligible


# ---- Per-series runner ------------------------------------------------------


def _run_one_series(
    path: Path,
    out_dir: Path,
) -> dict[str, object] | None:
    series_name = path.stem
    out_file = out_dir / "per_series" / f"{series_name}.json"
    if out_file.exists():
        LOGGER.info("Skipping %s (already done).", series_name)
        with out_file.open() as fh:
            return json.load(fh)

    try:
        data, labels, anomaly_start, anomaly_end = load_ucr_series(path)
    except (ValueError, OSError) as exc:
        LOGGER.warning("Failed to load %s: %s", path.name, exc)
        return None

    split_idx = data.shape[0] // 2
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    test_labels_ts = labels[split_idx:]

    if train_data.shape[0] <= WINDOW_SIZE or test_data.shape[0] <= WINDOW_SIZE:
        LOGGER.warning("Series %s too short, skipping.", series_name)
        return None

    train_norm, test_norm = normalize_data(train_data, test_data, method="standard")
    train_labels_dummy = np.zeros((train_norm.shape[0],), dtype=np.int64)

    train_windows, _ = create_sliding_windows(train_norm, train_labels_dummy, WINDOW_SIZE, STRIDE)
    test_windows, test_window_labels = create_sliding_windows(test_norm, test_labels_ts, WINDOW_SIZE, STRIDE)

    # --- Raw Mahalanobis (mean-pooled) ---
    train_mp = _mean_pooled(train_windows)
    test_mp = _mean_pooled(test_windows)
    scorer_mp = _RawMahalanobisScorer()
    scorer_mp.fit(train_mp)
    raw_mp = scorer_mp.score(test_mp)
    smooth_mp = smooth_scores(raw_mp, window_size=SMOOTH_WINDOW, method=SMOOTH_METHOD)
    auc_mp = float(compute_all_metrics(smooth_mp, test_window_labels)["auc_roc"])

    # --- Raw Mahalanobis (flattened) — skip if dim too large ---
    flat_dim = WINDOW_SIZE * int(train_windows.shape[2])
    auc_flat: float | None = None
    if flat_dim <= 5000:
        train_fl = _flattened(train_windows)
        test_fl = _flattened(test_windows)
        scorer_fl = _RawMahalanobisScorer()
        try:
            scorer_fl.fit(train_fl)
            raw_fl = scorer_fl.score(test_fl)
            smooth_fl = smooth_scores(raw_fl, window_size=SMOOTH_WINDOW, method=SMOOTH_METHOD)
            auc_flat = float(compute_all_metrics(smooth_fl, test_window_labels)["auc_roc"])
        except Exception as exc:
            LOGGER.warning("%s flattened Mahalanobis failed: %s", series_name, exc)
    else:
        LOGGER.info("%s: skipping flattened Mahalanobis (dim=%d > 5000).", series_name, flat_dim)

    # --- Classical baselines (use last-step feature for speed) ---
    train_ls = _last_step(train_windows)
    test_ls = _last_step(test_windows)

    auc_lof: float | None = None
    auc_if: float | None = None
    auc_ocsvm: float | None = None

    try:
        auc_lof = float(compute_all_metrics(_score_lof(train_ls, test_ls), test_window_labels)["auc_roc"])
    except Exception as exc:
        LOGGER.warning("%s LOF failed: %s", series_name, exc)

    try:
        auc_if = float(compute_all_metrics(_score_if(train_ls, test_ls), test_window_labels)["auc_roc"])
    except Exception as exc:
        LOGGER.warning("%s IsolationForest failed: %s", series_name, exc)

    try:
        auc_ocsvm = float(compute_all_metrics(_score_ocsvm(train_ls, test_ls), test_window_labels)["auc_roc"])
    except Exception as exc:
        LOGGER.warning("%s OCSVM failed: %s", series_name, exc)

    auc_roc: dict[str, float | None] = {
        "RawMaha_MeanPooled": auc_mp,
        "RawMaha_Flattened": auc_flat,
        "LOF": auc_lof,
        "IsolationForest": auc_if,
        "OneClassSVM": auc_ocsvm,
    }

    result: dict[str, object] = {
        "series": series_name,
        "file": str(path),
        "anomaly_start": int(anomaly_start),
        "anomaly_end": int(anomaly_end),
        "train_length": int(train_data.shape[0]),
        "test_length": int(test_data.shape[0]),
        "auc_roc": auc_roc,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    LOGGER.info(
        "%s  maha_mp=%.4f  lof=%s  if=%s  ocsvm=%s",
        series_name,
        auc_mp,
        f"{auc_lof:.4f}" if auc_lof is not None else "N/A",
        f"{auc_if:.4f}" if auc_if is not None else "N/A",
        f"{auc_ocsvm:.4f}" if auc_ocsvm is not None else "N/A",
    )
    return result


# ---- Summary ----------------------------------------------------------------


def _compute_summary(
    per_series: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    methods = ["RawMaha_MeanPooled", "RawMaha_Flattened", "LOF", "IsolationForest", "OneClassSVM"]
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        vals = [
            float(v)
            for item in per_series
            for k, v in item.get("auc_roc", {}).items()
            if k == method and v is not None
        ]
        if vals:
            summary[method] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
    return summary


# ---- Entry point ------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expanded UCR baselines: Raw Mahalanobis + LOF + IF + OCSVM."
    )
    parser.add_argument(
        "--ucr_dir",
        type=str,
        default="data/UCR",
        help="Root directory containing extracted UCR files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ucr_expanded_raw",
        help="Output directory for per-series and summary results.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    np.random.seed(SEED)

    args = _build_parser().parse_args()
    ucr_dir = Path(args.ucr_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_series").mkdir(parents=True, exist_ok=True)

    eligible_files = _find_all_eligible_files(ucr_dir)
    LOGGER.info("Running baselines on %d UCR series.", len(eligible_files))

    per_series: list[dict[str, Any]] = []
    for i, path in enumerate(eligible_files):
        LOGGER.info("[%d/%d] Processing %s", i + 1, len(eligible_files), path.name)
        result = _run_one_series(path, out_dir)
        if result is not None:
            per_series.append(result)

    summary = _compute_summary(per_series)

    payload = {
        "dataset": "UCR Anomaly Archive 2021 (expanded baselines)",
        "n_series": len(per_series),
        "config": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "smooth_window": SMOOTH_WINDOW,
            "smooth_method": SMOOTH_METHOD,
            "lof_n_neighbors": 20,
            "if_n_estimators": 100,
            "ocsvm_kernel": "rbf",
            "ocsvm_nu": 0.01,
        },
        "summary_auc_roc": summary,
        "per_series": per_series,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    LOGGER.info("Saved summary to %s", summary_path)

    print("\n=== UCR Expanded Baselines (mean±std AUC-ROC) ===")
    for method, stats in summary.items():
        print(f"  {method:<30}  {stats['mean']:.4f} ± {stats['std']:.4f}  (n={stats['n']})")
    print()


if __name__ == "__main__":
    main()
