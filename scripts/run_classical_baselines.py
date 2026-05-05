from __future__ import annotations

# pyright: basic, reportMissingImports=false, reportMissingTypeStubs=false

import matplotlib

matplotlib.use("Agg")

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.msl import MSLDataset
from src.data.psm import PSMDataset
from src.data.smap import SMAPDataset
from src.data.smd import SMDDataset
from src.evaluation.metrics import compute_all_metrics

LOGGER = logging.getLogger(__name__)
SEED = 42
WINDOW_SIZE = 100
OCSVM_MAX_TRAIN = 50000
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _set_seeds() -> None:
    np.random.seed(SEED)
    random.seed(SEED)


def _sanitize_features(
    train_features: FloatArray,
    test_features: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    train_clean = np.nan_to_num(
        train_features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    test_clean = np.nan_to_num(
        test_features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    return train_clean, test_clean


def _windows_to_features(windows: FloatArray) -> FloatArray:
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (N, W, D), got {windows.shape}.")
    return windows[:, -1, :]


def _score_lof(
    train_features: FloatArray,
    test_features: FloatArray,
) -> FloatArray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    n_neighbors = max(2, min(20, train_scaled.shape[0] - 1))
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        contamination=0.05,  # pyright: ignore[reportArgumentType]
    )
    model.fit(train_scaled)
    return -model.decision_function(test_scaled)


def _score_isolation_forest(
    train_features: FloatArray,
    test_features: FloatArray,
) -> FloatArray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # pyright: ignore[reportArgumentType]
        random_state=SEED,
    )
    model.fit(train_scaled)
    return -model.decision_function(test_scaled)


def _score_ocsvm(
    train_features: FloatArray,
    test_features: FloatArray,
) -> FloatArray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    fit_data = train_scaled
    if train_scaled.shape[0] > OCSVM_MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(
            train_scaled.shape[0], size=OCSVM_MAX_TRAIN, replace=False
        )
        fit_data = train_scaled[sample_idx]
        LOGGER.info(
            "OCSVM train subsampling: %d -> %d",
            train_scaled.shape[0],
            fit_data.shape[0],
        )

    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
    model.fit(fit_data)
    return -model.decision_function(test_scaled)


def _evaluate_methods(
    train_windows: FloatArray,
    test_windows: FloatArray,
    test_labels: IntArray,
) -> dict[str, dict[str, float]]:
    train_features = _windows_to_features(train_windows)
    test_features = _windows_to_features(test_windows)
    train_clean, test_clean = _sanitize_features(train_features, test_features)
    labels = np.asarray(test_labels, dtype=np.int64)

    method_scores = {
        "LOF": _score_lof(train_clean, test_clean),
        "IsolationForest": _score_isolation_forest(train_clean, test_clean),
        "OneClassSVM": _score_ocsvm(train_clean, test_clean),
    }

    results: dict[str, dict[str, float]] = {}
    for method, scores in method_scores.items():
        metrics = compute_all_metrics(scores=scores, labels=labels)
        results[method] = {
            "auc_roc": float(metrics["auc_roc"]),
            "auc_pr": float(metrics["auc_pr"]),
            "f1_pa": float(metrics["f1_pa"]),
        }
    return results


def _load_single_dataset(
    dataset: str,
    raw_root: Path,
    window_size: int,
) -> tuple[FloatArray, FloatArray, IntArray]:
    if dataset == "psm":
        ds = PSMDataset(
            raw_dir=raw_root / "psm", window_size=window_size, normalize=False
        )
    elif dataset == "msl":
        ds = MSLDataset(
            raw_dir=raw_root / "msl", window_size=window_size, normalize=False
        )
    elif dataset == "smap":
        ds = SMAPDataset(
            raw_dir=raw_root / "smap", window_size=window_size, normalize=False
        )
    else:
        raise ValueError(f"Unsupported single-entity dataset: {dataset}")

    return ds.train_windows, ds.test_windows, ds.test_labels


def _list_smd_entities(raw_root: Path) -> list[str]:
    train_dir = raw_root / "smd" / "train"
    entities = sorted(path.stem for path in train_dir.glob("*.txt"))
    if not entities:
        raise ValueError(f"No SMD entities found in {train_dir}")
    if len(entities) != 28:
        LOGGER.warning("Expected 28 SMD entities, found %d.", len(entities))
    return entities


def _save_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _run_single_entity_dataset(
    dataset: str,
    raw_root: Path,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    train_windows, test_windows, test_labels = _load_single_dataset(
        dataset=dataset,
        raw_root=raw_root,
        window_size=WINDOW_SIZE,
    )
    method_metrics = _evaluate_methods(train_windows, test_windows, test_labels)

    for method, metrics in method_metrics.items():
        payload: dict[str, Any] = {
            "method": method,
            "dataset": dataset,
            "metrics": metrics,
        }
        _save_json(output_dir / f"{method.lower()}_{dataset}.json", payload)
    return method_metrics


def _run_smd(raw_root: Path, output_dir: Path) -> dict[str, dict[str, float]]:
    entities = _list_smd_entities(raw_root)
    per_method_per_entity: dict[str, dict[str, dict[str, float]]] = {
        "LOF": {},
        "IsolationForest": {},
        "OneClassSVM": {},
    }

    for entity in entities:
        ds = SMDDataset(
            raw_dir=raw_root / "smd",
            entity=entity,
            window_size=WINDOW_SIZE,
            normalize=False,
        )
        method_metrics = _evaluate_methods(
            ds.train_windows, ds.test_windows, ds.test_labels
        )
        for method, metrics in method_metrics.items():
            per_method_per_entity[method][entity] = metrics

    method_averages: dict[str, dict[str, float]] = {}
    for method, entity_map in per_method_per_entity.items():
        auc_rocs = [item["auc_roc"] for item in entity_map.values()]
        auc_prs = [item["auc_pr"] for item in entity_map.values()]
        f1_pas = [item["f1_pa"] for item in entity_map.values()]
        avg_metrics = {
            "auc_roc": float(np.mean(auc_rocs)),
            "auc_pr": float(np.mean(auc_prs)),
            "f1_pa": float(np.mean(f1_pas)),
        }
        method_averages[method] = avg_metrics

        per_entity_auc = {
            entity: {"auc_roc": float(values["auc_roc"])}
            for entity, values in sorted(entity_map.items())
        }
        payload: dict[str, Any] = {
            "method": method,
            "dataset": "smd",
            "metrics": avg_metrics,
            "per_entity": per_entity_auc,
        }
        _save_json(output_dir / f"{method.lower()}_smd.json", payload)

    return method_averages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sklearn classical TSAD baselines."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["psm", "msl", "smap", "smd", "all"],
        help="Dataset to run.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Unused argument for script interface consistency.",
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="data/raw",
        help="Root directory containing raw datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/reports/classical_baselines",
        help="Directory to save JSON reports.",
    )
    return parser


def _print_summary(summary: dict[str, dict[str, dict[str, float]]]) -> None:
    LOGGER.info("Summary (AUC-ROC / AUC-PR / F1-PA)")
    LOGGER.info(
        "%-6s | %-15s | %-8s | %-8s | %-8s",
        "Dataset",
        "Method",
        "AUCROC",
        "AUCPR",
        "F1PA",
    )
    for dataset in sorted(summary.keys()):
        for method, metrics in sorted(summary[dataset].items()):
            LOGGER.info(
                "%-6s | %-15s | %.4f   | %.4f   | %.4f",
                dataset,
                method,
                metrics["auc_roc"],
                metrics["auc_pr"],
                metrics["f1_pa"],
            )


def main() -> None:
    _setup_logging()
    _set_seeds()

    args = _build_parser().parse_args()
    _ = args.gpu

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = (
        ["psm", "msl", "smap", "smd"] if args.dataset == "all" else [args.dataset]
    )
    summary: dict[str, dict[str, dict[str, float]]] = {}

    for dataset in datasets:
        LOGGER.info("Running classical baselines on %s", dataset)
        if dataset == "smd":
            summary[dataset] = _run_smd(raw_root=raw_root, output_dir=output_dir)
        else:
            summary[dataset] = _run_single_entity_dataset(
                dataset=dataset,
                raw_root=raw_root,
                output_dir=output_dir,
            )

    _print_summary(summary)
    LOGGER.info("Saved classical baseline reports to %s", output_dir)


if __name__ == "__main__":
    main()
