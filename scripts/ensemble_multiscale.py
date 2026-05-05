"""Ensemble anomaly scores from multi-scale PatchTraj experiments."""

from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.evaluation.metrics import compute_all_metrics
from src.scoring.multiscale_ensemble import MultiScaleEnsemble, MultiScaleScoreEntry
from src.scoring.patchtraj_scorer import normalize_scores, smooth_scores

LOGGER = logging.getLogger(__name__)


def _evaluate_method(
    ensemble: MultiScaleEnsemble,
    entries: Sequence[MultiScaleScoreEntry],
    method: str,
    smooth_window: int,
    smooth_method: str,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float] | None]:
    scores, labels = ensemble.combine(entries=entries, method=method)
    if smooth_window > 1:
        scores = smooth_scores(scores, window_size=smooth_window, method=smooth_method)
    scores = normalize_scores(scores, method="minmax")
    if labels is None:
        return scores, labels, None
    return scores, labels, compute_all_metrics(scores, labels)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ensemble multi-scale PatchTraj scores")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--entity", type=str, default="default")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", *MultiScaleEnsemble.SUPPORTED_METHODS],
    )
    parser.add_argument("--smooth_window", type=int, default=21)
    parser.add_argument(
        "--smooth_method", type=str, default="mean", choices=["mean", "median"]
    )
    args = parser.parse_args()

    smooth_window = int(args.smooth_window)
    if smooth_window > 1 and smooth_window % 2 == 0:
        smooth_window += 1

    ensemble = MultiScaleEnsemble(window_sizes=(50, 100, 200))
    entries = ensemble.find_score_entries(
        results_dir=Path(args.results_dir),
        entity=str(args.entity).strip() or None,
    )

    methods = (
        list(MultiScaleEnsemble.SUPPORTED_METHODS)
        if args.method == "auto"
        else [str(args.method)]
    )
    best_scores: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None
    best_method = methods[0]
    best_auc = float("-inf")

    for method in methods:
        scores, labels, metrics = _evaluate_method(
            ensemble=ensemble,
            entries=entries,
            method=method,
            smooth_window=smooth_window,
            smooth_method=str(args.smooth_method),
        )
        if metrics is None:
            best_scores = scores
            best_labels = labels
            best_method = method
            break
        auc = float(metrics["auc_roc"])
        LOGGER.info("%s: AUC-ROC=%.4f", method, auc)
        if auc > best_auc:
            best_scores = scores
            best_labels = labels
            best_metrics = metrics
            best_method = method
            best_auc = auc

    output_dir = Path(args.results_dir) / args.entity
    np.save(output_dir / "multiscale_ensemble_scores.npy", best_scores)
    if best_labels is not None:
        np.save(output_dir / "multiscale_ensemble_labels.npy", best_labels)
    if best_metrics is not None:
        with (output_dir / "multiscale_ensemble_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(best_metrics, handle, indent=2, sort_keys=True)
        LOGGER.info("Best method: %s (AUC-ROC=%.4f)", best_method, best_auc)
        for metric_name, metric_value in best_metrics.items():
            LOGGER.info("%s=%.6f", metric_name, metric_value)


if __name__ == "__main__":
    main()
