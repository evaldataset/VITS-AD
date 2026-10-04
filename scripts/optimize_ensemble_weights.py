#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportMissingTypeArgument=false
"""Optimize LP/RP ensemble weights on validation splits.

This script loads precomputed LP/RP anomaly scores from improved result folders,
creates temporal validation/test splits, optimizes LP weight on validation AUC-ROC,
and reports test performance for fixed, per-dataset optimized, and global
optimized settings.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_all_metrics
from src.scoring.patchtraj_scorer import smooth_scores

LOGGER = logging.getLogger(__name__)

DATASETS = ("smd", "psm", "msl", "smap")
SMOOTH_METHODS = ("mean", "median")
SMOOTH_WINDOWS = (7, 15, 21)
WEIGHT_GRID = tuple(round(i * 0.05, 2) for i in range(21))
OPT_ENSEMBLE_METHODS = ("zscore_weighted",)

FIXED_CONFIGS: dict[str, dict[str, Any]] = {
    "smd": {
        "method": "zscore_weighted",
        "smooth_window": 7,
        "smooth_method": "median",
        "w_lp": 0.40,
    },
    "smap": {
        "method": "zscore_weighted",
        "smooth_window": 7,
        "smooth_method": "mean",
        "w_lp": 0.10,
    },
    "psm": {
        "method": "rank_weighted",
        "smooth_window": 21,
        "smooth_method": "mean",
        "w_lp": 0.50,
    },
    "msl": {
        "method": "zscore_weighted",
        "smooth_window": 15,
        "smooth_method": "median",
        "w_lp": 0.30,
    },
}


def _zscore(scores: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1D score vector.

    Args:
        scores: Input scores of shape (T,).

    Returns:
        Z-score normalized scores of shape (T,).
    """
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    if sigma < 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return ((scores - mu) / sigma).astype(np.float64)


def _combine_scores(
    lp_scores: np.ndarray,
    rp_scores: np.ndarray,
    method: str,
    w_lp: float,
    smooth_window: int,
    smooth_method: str,
) -> np.ndarray:
    """Combine LP/RP scores with smoothing and weighted ensemble.

    Args:
        lp_scores: LP scores of shape (T,).
        rp_scores: RP scores of shape (T,).
        method: Ensemble method (zscore_weighted or rank_weighted).
        w_lp: LP weight in [0, 1].
        smooth_window: Smoothing window size.
        smooth_method: Smoothing method (mean or median).

    Returns:
        Combined scores of shape (T,).
    """
    w_rp = 1.0 - w_lp
    lp_smooth = smooth_scores(
        lp_scores, window_size=smooth_window, method=smooth_method
    )
    rp_smooth = smooth_scores(
        rp_scores, window_size=smooth_window, method=smooth_method
    )

    if method == "zscore_weighted":
        lp_norm = _zscore(lp_smooth)
        rp_norm = _zscore(rp_smooth)
        return (w_lp * lp_norm + w_rp * rp_norm).astype(np.float64)

    if method == "rank_weighted":
        n = lp_smooth.shape[0]
        lp_rank = rankdata(lp_smooth) / n
        rp_rank = rankdata(rp_smooth) / n
        return (w_lp * lp_rank + w_rp * rp_rank).astype(np.float64)

    raise ValueError(f"Unknown ensemble method: {method}")


def _temporal_split(
    lp_scores: np.ndarray,
    rp_scores: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """Split aligned scores into temporal validation/test partitions.

    Args:
        lp_scores: LP scores of shape (T,).
        rp_scores: RP scores of shape (T,).
        labels: Labels of shape (T,).
        val_ratio: Fraction used as validation prefix.

    Returns:
        Split dict with keys val/test, or None if sequence is too short.
    """
    min_len = min(lp_scores.size, rp_scores.size, labels.size)
    if min_len < 2:
        return None

    lp = lp_scores[:min_len]
    rp = rp_scores[:min_len]
    y = labels[:min_len]

    split_idx = int(np.floor(min_len * val_ratio))
    split_idx = max(1, min(split_idx, min_len - 1))

    return {
        "val": (lp[:split_idx], rp[:split_idx], y[:split_idx]),
        "test": (lp[split_idx:], rp[split_idx:], y[split_idx:]),
    }


def _load_dataset_entities(
    dataset_dir: Path,
    val_ratio: float,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Load and split all valid entities from a dataset directory.

    Args:
        dataset_dir: Directory like results/improved_smd.
        val_ratio: Validation ratio for temporal split.

    Returns:
        Entity map: entity_name -> {"val": (...), "test": (...)}.
    """
    entities: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for entity_dir in sorted(dataset_dir.iterdir()):
        if not entity_dir.is_dir():
            continue

        lp_path = entity_dir / "line_plot" / "scores.npy"
        rp_path = entity_dir / "recurrence_plot" / "scores.npy"
        label_candidates = (
            entity_dir / "line_plot" / "labels.npy",
            entity_dir / "recurrence_plot" / "labels.npy",
        )

        label_path = next((path for path in label_candidates if path.exists()), None)
        if not lp_path.exists() or not rp_path.exists() or label_path is None:
            LOGGER.warning("Skipping %s due to missing score/label files", entity_dir)
            continue

        lp_scores = np.load(lp_path).astype(np.float64).reshape(-1)
        rp_scores = np.load(rp_path).astype(np.float64).reshape(-1)
        labels = np.load(label_path).astype(np.int64).reshape(-1)

        split_data = _temporal_split(
            lp_scores=lp_scores,
            rp_scores=rp_scores,
            labels=labels,
            val_ratio=val_ratio,
        )
        if split_data is None:
            LOGGER.warning("Skipping %s due to sequence length < 2", entity_dir)
            continue

        entities[entity_dir.name] = split_data

    return entities


def _evaluate_dataset(
    entities: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    config: dict[str, Any],
    split_name: str,
) -> dict[str, float]:
    """Evaluate one config on one dataset split.

    Args:
        entities: Entity split map.
        config: Ensemble config dictionary.
        split_name: One of "val" or "test".

    Returns:
        Aggregated metric dictionary with average values and entity count.
    """
    auc_roc_values: list[float] = []
    auc_pr_values: list[float] = []
    f1_pa_values: list[float] = []

    for entity_name, split_data in entities.items():
        lp_scores, rp_scores, labels = split_data[split_name]
        try:
            combined = _combine_scores(
                lp_scores=lp_scores,
                rp_scores=rp_scores,
                method=str(config["method"]),
                w_lp=float(config["w_lp"]),
                smooth_window=int(config["smooth_window"]),
                smooth_method=str(config["smooth_method"]),
            )
            metrics = compute_all_metrics(scores=combined, labels=labels)
        except ValueError:
            LOGGER.debug(
                "Skipping %s (%s split) due to invalid metric inputs",
                entity_name,
                split_name,
            )
            continue

        auc_roc_values.append(float(metrics["auc_roc"]))
        auc_pr_values.append(float(metrics["auc_pr"]))
        f1_pa_values.append(float(metrics["f1_pa"]))

    if not auc_roc_values:
        return {
            "auc_roc": float("nan"),
            "auc_pr": float("nan"),
            "f1_pa": float("nan"),
            "n_entities": 0.0,
        }

    return {
        "auc_roc": float(np.mean(auc_roc_values)),
        "auc_pr": float(np.mean(auc_pr_values)),
        "f1_pa": float(np.mean(f1_pa_values)),
        "n_entities": float(len(auc_roc_values)),
    }


def _search_best_config_for_dataset(
    entities: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> dict[str, Any] | None:
    """Grid-search best per-dataset config using validation AUC-ROC.

    Args:
        entities: Dataset entities with val/test splits.

    Returns:
        Best config dict with val/test metrics, or None if unavailable.
    """
    best_val_auc = -1.0
    best_config: dict[str, Any] | None = None

    for method in OPT_ENSEMBLE_METHODS:
        for smooth_method in SMOOTH_METHODS:
            for smooth_window in SMOOTH_WINDOWS:
                for w_lp in WEIGHT_GRID:
                    config = {
                        "method": method,
                        "smooth_method": smooth_method,
                        "smooth_window": smooth_window,
                        "w_lp": w_lp,
                    }
                    val_metrics = _evaluate_dataset(
                        entities=entities,
                        config=config,
                        split_name="val",
                    )
                    val_auc = float(val_metrics["auc_roc"])
                    if np.isnan(val_auc):
                        continue
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_config = {
                            **config,
                            "val_metrics": val_metrics,
                        }

    if best_config is None:
        return None

    best_config["test_metrics"] = _evaluate_dataset(
        entities=entities,
        config=best_config,
        split_name="test",
    )
    return best_config


def _search_best_global_config(
    dataset_entities: dict[
        str, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ],
) -> dict[str, Any] | None:
    """Grid-search one global config maximizing mean validation AUC-ROC.

    Args:
        dataset_entities: Map from dataset name to entity split map.

    Returns:
        Best global config dictionary or None.
    """
    best_val_auc = -1.0
    best_config: dict[str, Any] | None = None

    for method in OPT_ENSEMBLE_METHODS:
        for smooth_method in SMOOTH_METHODS:
            for smooth_window in SMOOTH_WINDOWS:
                for w_lp in WEIGHT_GRID:
                    config = {
                        "method": method,
                        "smooth_method": smooth_method,
                        "smooth_window": smooth_window,
                        "w_lp": w_lp,
                    }

                    per_dataset_val: dict[str, float] = {}
                    for dataset_name, entities in dataset_entities.items():
                        metrics = _evaluate_dataset(
                            entities=entities,
                            config=config,
                            split_name="val",
                        )
                        auc_roc = float(metrics["auc_roc"])
                        if np.isnan(auc_roc):
                            break
                        per_dataset_val[dataset_name] = auc_roc

                    if len(per_dataset_val) != len(dataset_entities):
                        continue

                    mean_val_auc = float(np.mean(list(per_dataset_val.values())))
                    if mean_val_auc > best_val_auc:
                        best_val_auc = mean_val_auc
                        best_config = {
                            **config,
                            "val_auc_roc": mean_val_auc,
                            "per_dataset_val_auc_roc": per_dataset_val,
                        }

    return best_config


def _delta_string(new_score: float, base_score: float) -> str:
    """Format score delta as signed string with 4 decimals."""
    return f"{(new_score - base_score):+.4f}"


def _build_dataset_report(
    dataset_name: str,
    entities: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    global_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble report block for one dataset.

    Args:
        dataset_name: Dataset name.
        entities: Entity split map.
        global_config: Optional global optimized config.

    Returns:
        Report dictionary for JSON output.
    """
    fixed_config = FIXED_CONFIGS[dataset_name]
    fixed_test = _evaluate_dataset(
        entities=entities, config=fixed_config, split_name="test"
    )

    optimized = _search_best_config_for_dataset(entities=entities)
    if optimized is None:
        return {
            "error": "no_valid_entities",
            "fixed_weights": {
                "w_lp": float(fixed_config["w_lp"]),
                "auc_roc": float(fixed_test["auc_roc"]),
            },
        }

    report: dict[str, Any] = {
        "fixed_weights": {
            "method": fixed_config["method"],
            "smooth_method": fixed_config["smooth_method"],
            "smooth_window": int(fixed_config["smooth_window"]),
            "w_lp": float(fixed_config["w_lp"]),
            "auc_roc": float(fixed_test["auc_roc"]),
            "auc_pr": float(fixed_test["auc_pr"]),
            "f1_pa": float(fixed_test["f1_pa"]),
            "n_entities": int(fixed_test["n_entities"]),
        },
        "optimized_weights": {
            "method": optimized["method"],
            "smooth_method": optimized["smooth_method"],
            "smooth_window": int(optimized["smooth_window"]),
            "w_lp": float(optimized["w_lp"]),
            "val_auc_roc": float(optimized["val_metrics"]["auc_roc"]),
            "auc_roc": float(optimized["test_metrics"]["auc_roc"]),
            "auc_pr": float(optimized["test_metrics"]["auc_pr"]),
            "f1_pa": float(optimized["test_metrics"]["f1_pa"]),
            "n_entities": int(optimized["test_metrics"]["n_entities"]),
        },
        "improvement": _delta_string(
            new_score=float(optimized["test_metrics"]["auc_roc"]),
            base_score=float(fixed_test["auc_roc"]),
        ),
    }

    if global_config is not None:
        global_test = _evaluate_dataset(
            entities=entities,
            config=global_config,
            split_name="test",
        )
        report["global_weights"] = {
            "method": global_config["method"],
            "smooth_method": global_config["smooth_method"],
            "smooth_window": int(global_config["smooth_window"]),
            "w_lp": float(global_config["w_lp"]),
            "val_auc_roc_global_mean": float(global_config["val_auc_roc"]),
            "auc_roc": float(global_test["auc_roc"]),
            "auc_pr": float(global_test["auc_pr"]),
            "f1_pa": float(global_test["f1_pa"]),
            "n_entities": int(global_test["n_entities"]),
            "improvement_vs_fixed": _delta_string(
                new_score=float(global_test["auc_roc"]),
                base_score=float(fixed_test["auc_roc"]),
            ),
        }

    return report


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize LP/RP ensemble weights using validation splits",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing improved_* folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="Datasets to process (default: smd psm msl smap).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.30,
        help="Temporal prefix ratio used as validation split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <results-root>/reports/optimized_ensemble.json).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    """Run validation-based ensemble weight optimization."""
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in (0, 1), got {args.val_ratio}.")

    datasets = [dataset.lower() for dataset in args.datasets]
    output_path = (
        args.output
        if args.output is not None
        else args.results_root / "reports" / "optimized_ensemble.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_entities: dict[
        str,
        dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ] = {}
    for dataset in datasets:
        dataset_dir = args.results_root / f"improved_{dataset}"
        if not dataset_dir.exists():
            LOGGER.warning("Missing dataset directory: %s", dataset_dir)
            continue
        entities = _load_dataset_entities(
            dataset_dir=dataset_dir, val_ratio=args.val_ratio
        )
        if not entities:
            LOGGER.warning("No valid entities for dataset: %s", dataset)
            continue
        dataset_entities[dataset] = entities

    if not dataset_entities:
        raise RuntimeError("No valid dataset entities were loaded.")

    global_config = _search_best_global_config(dataset_entities=dataset_entities)
    if global_config is not None:
        LOGGER.info(
            "Global best config: method=%s smooth=%s/%d w_lp=%.2f val_auc=%.4f",
            global_config["method"],
            global_config["smooth_method"],
            int(global_config["smooth_window"]),
            float(global_config["w_lp"]),
            float(global_config["val_auc_roc"]),
        )

    report: dict[str, Any] = {
        "_meta": {
            "results_root": str(args.results_root),
            "val_ratio": float(args.val_ratio),
            "weight_grid": list(WEIGHT_GRID),
            "smooth_methods": list(SMOOTH_METHODS),
            "smooth_windows": list(SMOOTH_WINDOWS),
            "optimization_methods": list(OPT_ENSEMBLE_METHODS),
            "global_optimization": global_config,
        }
    }

    for dataset_name in datasets:
        if dataset_name not in dataset_entities:
            continue
        LOGGER.info("Optimizing dataset: %s", dataset_name)
        report[dataset_name] = _build_dataset_report(
            dataset_name=dataset_name,
            entities=dataset_entities[dataset_name],
            global_config=global_config,
        )

        dataset_block = report[dataset_name]
        if "optimized_weights" in dataset_block:
            LOGGER.info(
                "%s fixed AUC=%.4f -> optimized AUC=%.4f (delta %s)",
                dataset_name.upper(),
                float(dataset_block["fixed_weights"]["auc_roc"]),
                float(dataset_block["optimized_weights"]["auc_roc"]),
                str(dataset_block["improvement"]),
            )

    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, sort_keys=True)

    LOGGER.info("Saved optimized ensemble report to %s", output_path)


if __name__ == "__main__":
    main()
