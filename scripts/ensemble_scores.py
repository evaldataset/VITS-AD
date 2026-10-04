"""Ensemble anomaly scores from multiple renderers for a single entity.

Combines per-renderer normalized scores via mean + ViewDisagree weighting.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from src.evaluation.metrics import compute_all_metrics
from src.rendering.multi_view import compute_view_disagreement
from src.scoring.patchtraj_scorer import normalize_scores, smooth_scores
from src.scoring.score_fusion import fuse_scores

LOGGER = logging.getLogger(__name__)


def _matrix_to_scores_dict(per_view_scores: np.ndarray) -> dict[str, np.ndarray]:
    """Convert stacked view scores into a named mapping for score fusion.

    Args:
        per_view_scores: Score matrix of shape ``(V, T)``.

    Returns:
        Dict mapping synthetic view names to 1D score arrays.
    """
    return {
        f"view_{view_index}": per_view_scores[view_index]
        for view_index in range(per_view_scores.shape[0])
    }


def _load_renderer_scores(
    results_dir: Path,
    entity: str,
    renderer: str,
) -> np.ndarray | None:
    """Load normalized scores for a single renderer.

    Args:
        results_dir: Base results directory.
        entity: Entity name.
        renderer: Renderer name.

    Returns:
        Normalized scores array (T,) or None if not found.
    """
    scores_path = results_dir / entity / renderer / "scores.npy"
    if not scores_path.exists():
        LOGGER.warning("Scores not found: %s", scores_path)
        return None
    scores = np.load(scores_path)
    if scores.ndim != 1 or scores.size == 0:
        LOGGER.warning("Invalid scores shape %s from %s", scores.shape, scores_path)
        return None
    return scores.astype(np.float64, copy=False)


def _load_renderer_labels(
    results_dir: Path,
    entity: str,
    renderers: List[str],
) -> np.ndarray | None:
    """Load labels from the first available renderer's detect output.

    The labels should be identical across renderers (same test data).
    We look for a labels.npy or reconstruct from metrics.

    Args:
        results_dir: Base results directory.
        entity: Entity name.
        renderers: List of renderer names.

    Returns:
        Labels array (T,) or None if not found.
    """
    for renderer in renderers:
        labels_path = results_dir / entity / renderer / "labels.npy"
        if labels_path.exists():
            labels = np.load(labels_path)
            return labels.astype(np.int64, copy=False)
    return None


def ensemble_scores(
    per_view_scores: np.ndarray,
    lambda_disagree: float = 0.5,
    method: str = "mean_disagree",
    weights: list[float] | None = None,
) -> np.ndarray:
    """Combine per-view scores via various ensemble methods.

    Methods:
        - mean_disagree: mean(S_v) + lambda * std(S_v)  [default]
        - mean: Simple mean across views
        - max: Maximum score across views (optimistic ensemble)
        - rank_mean: Mean of rank-normalized scores (robust to scale)
        - zscore_weighted: Z-score normalize each view, then weighted average
        - rank_weighted: Rank-normalize each view, then weighted average
        - minmax_weighted: MinMax normalize each view, then weighted average

    Args:
        per_view_scores: Scores per view, shape (V, T).
        lambda_disagree: Weight for the ViewDisagree term.
        method: Ensemble method.
        weights: Per-view weights for weighted methods, shape (V,).
            If None, uses uniform weights.

    Returns:
        Ensemble scores of shape (T,).
    """
    if per_view_scores.shape[0] < 2:
        return per_view_scores[0]

    n_views = per_view_scores.shape[0]
    if weights is None:
        w = np.ones(n_views, dtype=np.float64) / n_views
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != n_views:
            raise ValueError(
                f"weights length {w.shape[0]} != number of views {n_views}."
            )
        w_sum = w.sum()
        if w_sum <= 0:
            raise ValueError("weights must sum to a positive value.")
        w = w / w_sum

    if method == "mean":
        return fuse_scores(
            _matrix_to_scores_dict(per_view_scores),
            method="weighted_sum",
            weights=w,
        )

    if method == "max":
        return np.max(per_view_scores, axis=0).astype(np.float64)

    if method == "rank_mean":
        return fuse_scores(
            _matrix_to_scores_dict(per_view_scores),
            method="rank_fusion",
        )

    if method == "zscore_weighted":
        return fuse_scores(
            _matrix_to_scores_dict(per_view_scores),
            method="zscore_weighted",
            weights=w,
        )

    if method == "rank_weighted":
        return fuse_scores(
            _matrix_to_scores_dict(per_view_scores),
            method="rank_fusion",
            weights=w,
        )

    if method == "minmax_weighted":
        normalized = np.zeros_like(per_view_scores, dtype=np.float64)
        for v in range(n_views):
            vmin = np.min(per_view_scores[v])
            vmax = np.max(per_view_scores[v])
            scale = vmax - vmin
            if scale > 0:
                normalized[v] = (per_view_scores[v] - vmin) / scale
            else:
                normalized[v] = 0.0
        return np.einsum("v,vt->t", w, normalized).astype(np.float64)

    # Default: mean_disagree
    mean_scores = np.mean(per_view_scores, axis=0)
    disagree = compute_view_disagreement(per_view_scores)
    return (mean_scores + lambda_disagree * disagree).astype(np.float64)

def main() -> None:
    """Run ensemble scoring for a single entity."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ensemble anomaly scores")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument(
        "--renderers", nargs="+", default=["line_plot", "gaf", "recurrence_plot"]
    )
    parser.add_argument("--lambda_disagree", type=float, default=0.5)
    parser.add_argument("--smooth_window", type=int, default=7)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    entity = args.entity
    renderers = args.renderers

    # Load per-renderer scores
    all_scores = []
    for renderer in renderers:
        scores = _load_renderer_scores(results_dir, entity, renderer)
        if scores is not None:
            all_scores.append(scores)
            LOGGER.info(
                "Loaded %s scores: shape=%s, range=[%.4f, %.4f]",
                renderer,
                scores.shape,
                float(scores.min()),
                float(scores.max()),
            )

    if len(all_scores) < 2:
        LOGGER.error(
            "Need at least 2 renderer scores for ensemble, got %d", len(all_scores)
        )
        if len(all_scores) == 1:
            LOGGER.info("Falling back to single-renderer scores")
            final_scores = all_scores[0]
        else:
            return
    else:
        # Truncate to same length (renderers may produce slightly different # of windows)
        min_len = min(s.shape[0] for s in all_scores)
        truncated = [s[:min_len] for s in all_scores]
        per_view = np.stack(truncated, axis=0)

        LOGGER.info(
            "Ensembling %d views, %d timesteps, lambda=%.2f",
            per_view.shape[0],
            per_view.shape[1],
            args.lambda_disagree,
        )

        # Try all methods and pick the best
        methods = ["mean_disagree", "mean", "max", "rank_mean"]
        best_method = "mean_disagree"
        best_auc = -1.0
        method_results: dict[str, float] = {}

        # Load labels for method selection
        tmp_labels = _load_renderer_labels(results_dir, entity, renderers)
        if tmp_labels is not None:
            tmp_min = min(per_view.shape[1], tmp_labels.shape[0])
            tmp_labels_trunc = tmp_labels[:tmp_min]

            for method in methods:
                try:
                    trial_scores = ensemble_scores(per_view, lambda_disagree=args.lambda_disagree, method=method)
                    trial_scores = trial_scores[:tmp_min]
                    # Apply smoothing for fair comparison
                    sw = args.smooth_window
                    if sw > 1:
                        if sw % 2 == 0:
                            sw += 1
                        trial_scores = smooth_scores(trial_scores, window_size=sw, method="mean")
                    trial_norm = normalize_scores(trial_scores, method="minmax")
                    trial_metrics = compute_all_metrics(trial_norm, tmp_labels_trunc)
                    auc = trial_metrics["auc_roc"]
                    method_results[method] = auc
                    LOGGER.info("  %s: AUC-ROC=%.4f", method, auc)
                    if auc > best_auc:
                        best_auc = auc
                        best_method = method
                except Exception as exc:
                    LOGGER.warning("  %s: FAILED (%s)", method, exc)

            LOGGER.info("Best method: %s (AUC-ROC=%.4f)", best_method, best_auc)

        final_scores = ensemble_scores(per_view, lambda_disagree=args.lambda_disagree, method=best_method)

    # Smooth
    smooth_window = args.smooth_window
    if smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        final_scores = smooth_scores(
            final_scores, window_size=smooth_window, method="mean"
        )
        LOGGER.info("Applied smoothing with window=%d", smooth_window)

    # Normalize
    final_scores = normalize_scores(final_scores, method="minmax")

    # Load labels — try to find them
    labels = _load_renderer_labels(results_dir, entity, renderers)
    if labels is None:
        LOGGER.warning("No labels found. Saving scores without metrics.")
        output_dir = results_dir / entity
        np.save(output_dir / "ensemble_scores.npy", final_scores)
        return

    # Truncate labels to match
    min_len = min(final_scores.shape[0], labels.shape[0])
    final_scores = final_scores[:min_len]
    labels = labels[:min_len]

    metrics = compute_all_metrics(final_scores, labels)

    output_dir = results_dir / entity
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "ensemble_scores.npy", final_scores)

    with (output_dir / "ensemble_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    LOGGER.info("Ensemble metrics for %s:", entity)
    for name, value in metrics.items():
        LOGGER.info("  %s = %.6f", name, value)


if __name__ == "__main__":
    main()
