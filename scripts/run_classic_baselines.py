from __future__ import annotations

# pyright: basic, reportMissingImports=false, reportMissingTypeStubs=false

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src.data.msl import _load_msl_labels, _load_msl_matrix
from src.data.psm import _load_psm_features, _load_psm_labels
from src.data.smap import _load_smap_labels, _load_smap_matrix
from src.data.smd import _load_smd_labels, _load_smd_matrix
from src.evaluation.metrics import (
    compute_auc_pr,
    compute_auc_roc,
    compute_best_f1,
    compute_f1_pa,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger(__name__)


def _standardize(
    train_data: np.ndarray[Any, Any], test_data: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (train_data - mean) / std, (test_data - mean) / std


def _sanitize_non_finite(
    train_data: np.ndarray[Any, Any], test_data: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    train = np.asarray(train_data, dtype=np.float64).copy()
    test = np.asarray(test_data, dtype=np.float64).copy()

    col_means = np.nanmean(train, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    train_mask = ~np.isfinite(train)
    if np.any(train_mask):
        train[train_mask] = np.take(col_means, np.where(train_mask)[1])

    test_mask = ~np.isfinite(test)
    if np.any(test_mask):
        test[test_mask] = np.take(col_means, np.where(test_mask)[1])

    train = np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0)
    test = np.nan_to_num(test, nan=0.0, posinf=0.0, neginf=0.0)
    return train, test


def _score_zscore_meanabs(
    train_data: np.ndarray[Any, Any], test_data: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    _, z_test = _standardize(train_data=train_data, test_data=test_data)
    return np.mean(np.abs(z_test), axis=1)


def _score_pca_recon(
    train_data: np.ndarray[Any, Any], test_data: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    train_norm, test_norm = _standardize(train_data=train_data, test_data=test_data)
    n_components = max(2, min(train_norm.shape[1] // 2, 20, train_norm.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_norm)
    test_proj = pca.inverse_transform(pca.transform(test_norm))
    return np.mean((test_norm - test_proj) ** 2, axis=1)


def _score_isolation_forest(
    train_data: np.ndarray[Any, Any], test_data: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    train_norm, test_norm = _standardize(train_data=train_data, test_data=test_data)
    max_train = 50000
    if train_norm.shape[0] > max_train:
        rng = np.random.default_rng(42)
        indices = rng.choice(train_norm.shape[0], size=max_train, replace=False)
        fit_data = train_norm[indices]
    else:
        fit_data = train_norm

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(fit_data)
    return -model.decision_function(test_norm)


def _evaluate_scores(
    scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
) -> dict[str, float]:
    auc_roc = compute_auc_roc(scores=scores, labels=labels)
    auc_pr = compute_auc_pr(scores=scores, labels=labels)
    best_f1, best_threshold = compute_best_f1(scores=scores, labels=labels)
    f1_pa = compute_f1_pa(scores=scores, labels=labels, threshold=best_threshold)
    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "f1_pa": float(f1_pa),
    }


def _load_smd_entity(
    entity: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    raw = Path("data/raw/smd")
    train_data = _load_smd_matrix(raw / "train" / f"{entity}.txt")
    test_data = _load_smd_matrix(raw / "test" / f"{entity}.txt")
    test_labels = _load_smd_labels(raw / "test_label" / f"{entity}.txt")
    return train_data, test_data, test_labels.astype(np.int64)


def _load_psm() -> tuple[
    np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
]:
    raw = Path("data/raw/psm")
    return (
        _load_psm_features(raw / "train.csv"),
        _load_psm_features(raw / "test.csv"),
        _load_psm_labels(raw / "test_label.csv").astype(np.int64),
    )


def _load_msl() -> tuple[
    np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
]:
    raw = Path("data/raw/msl")
    return (
        _load_msl_matrix(raw / "MSL_train.npy"),
        _load_msl_matrix(raw / "MSL_test.npy"),
        _load_msl_labels(raw / "MSL_test_label.npy").astype(np.int64),
    )


def _load_smap() -> tuple[
    np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
]:
    raw = Path("data/raw/smap")
    return (
        _load_smap_matrix(raw / "SMAP_train.npy"),
        _load_smap_matrix(raw / "SMAP_test.npy"),
        _load_smap_labels(raw / "SMAP_test_label.npy").astype(np.int64),
    )


def _run_methods(
    train_data: np.ndarray[Any, Any],
    test_data: np.ndarray[Any, Any],
    test_labels: np.ndarray[Any, Any],
) -> dict[str, dict[str, float]]:
    train_clean, test_clean = _sanitize_non_finite(
        train_data=train_data,
        test_data=test_data,
    )
    return {
        "zscore_meanabs": _evaluate_scores(
            scores=_score_zscore_meanabs(train_data=train_clean, test_data=test_clean),
            labels=test_labels,
        ),
        "pca_recon": _evaluate_scores(
            scores=_score_pca_recon(train_data=train_clean, test_data=test_clean),
            labels=test_labels,
        ),
        "isolation_forest": _evaluate_scores(
            scores=_score_isolation_forest(
                train_data=train_clean, test_data=test_clean
            ),
            labels=test_labels,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical baselines.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/reports",
        help="Directory for baseline outputs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    smd_entities = sorted((Path("data/raw/smd/test")).glob("*.txt"))
    smd_entities = [p.stem for p in smd_entities]
    LOGGER.info("Running baselines on %d SMD entities.", len(smd_entities))
    for entity in smd_entities:
        train_data, test_data, test_labels = _load_smd_entity(entity=entity)
        method_results = _run_methods(
            train_data=train_data,
            test_data=test_data,
            test_labels=test_labels,
        )
        for method_name, metrics in method_results.items():
            all_rows.append(
                {
                    "dataset": "SMD",
                    "entity": entity,
                    "method": method_name,
                    **metrics,
                }
            )

    for dataset_name, loader in [
        ("PSM", _load_psm),
        ("MSL", _load_msl),
        ("SMAP", _load_smap),
    ]:
        LOGGER.info("Running baselines on %s.", dataset_name)
        train_data, test_data, test_labels = loader()
        method_results = _run_methods(
            train_data=train_data,
            test_data=test_data,
            test_labels=test_labels,
        )
        for method_name, metrics in method_results.items():
            all_rows.append(
                {
                    "dataset": dataset_name,
                    "entity": dataset_name.lower(),
                    "method": method_name,
                    **metrics,
                }
            )

    detail_path = output_dir / "baselines_classic_detail.csv"
    fieldnames = [
        "dataset",
        "entity",
        "method",
        "auc_roc",
        "auc_pr",
        "best_f1",
        "best_threshold",
        "f1_pa",
    ]
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        key = (str(row["dataset"]), str(row["method"]))
        summary.setdefault(key, []).append(row)

    summary_path = output_dir / "baselines_classic_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "method",
                "n",
                "auc_roc_mean",
                "auc_pr_mean",
                "best_f1_mean",
                "f1_pa_mean",
            ],
        )
        writer.writeheader()
        for (dataset, method), rows in sorted(summary.items()):
            writer.writerow(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": len(rows),
                    "auc_roc_mean": float(np.mean([float(r["auc_roc"]) for r in rows])),
                    "auc_pr_mean": float(np.mean([float(r["auc_pr"]) for r in rows])),
                    "best_f1_mean": float(np.mean([float(r["best_f1"]) for r in rows])),
                    "f1_pa_mean": float(np.mean([float(r["f1_pa"]) for r in rows])),
                }
            )

    LOGGER.info("Wrote detailed baseline results: %s", detail_path)
    LOGGER.info("Wrote baseline summary: %s", summary_path)


if __name__ == "__main__":
    main()
