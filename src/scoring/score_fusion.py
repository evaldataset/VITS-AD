from __future__ import annotations

# pyright: reportMissingImports=false, reportInvalidTypeForm=false

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]
_MIN_STD = 1e-12


def _align_scores(
    scores_dict: Mapping[str, npt.ArrayLike],
) -> tuple[list[str], list[FloatArray]]:
    """Validate score sources and align them to a common length.

    Args:
        scores_dict: Mapping from source name to 1D anomaly score array.

    Returns:
        Tuple of ordered source names and aligned score arrays.

    Raises:
        ValueError: If inputs are empty or contain invalid score arrays.
    """
    if not scores_dict:
        raise ValueError("scores_dict must contain at least one score source.")

    source_names: list[str] = []
    score_arrays: list[FloatArray] = []

    for source_name, raw_scores in scores_dict.items():
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError(
                f"scores for source '{source_name}' must be 1D, got ndim={scores.ndim}."
            )
        if scores.size == 0:
            raise ValueError(
                f"scores for source '{source_name}' must contain at least one value."
            )
        if not np.isfinite(scores).all():
            raise ValueError(
                f"scores for source '{source_name}' contains non-finite values."
            )

        source_names.append(source_name)
        score_arrays.append(scores.astype(np.float64, copy=False))

    min_length = min(scores.size for scores in score_arrays)
    aligned_scores = [scores[:min_length] for scores in score_arrays]
    return source_names, aligned_scores


def _resolve_weights(
    source_names: Sequence[str],
    weights: Mapping[str, float] | Sequence[float] | None,
) -> FloatArray:
    """Resolve fusion weights into a normalized vector.

    Args:
        source_names: Ordered score source names.
        weights: Optional mapping keyed by source name or ordered sequence.

    Returns:
        Normalized weights of shape ``(V,)``.

    Raises:
        ValueError: If weights are invalid.
    """
    n_sources = len(source_names)
    if weights is None:
        return np.full(n_sources, 1.0 / n_sources, dtype=np.float64)

    if isinstance(weights, Mapping):
        expected_names = set(source_names)
        provided_names = set(weights.keys())
        if provided_names != expected_names:
            missing = sorted(expected_names - provided_names)
            extra = sorted(provided_names - expected_names)
            raise ValueError(
                "weights keys must match score sources exactly. "
                f"missing={missing}, extra={extra}."
            )
        weight_values = np.asarray(
            [weights[source_name] for source_name in source_names],
            dtype=np.float64,
        )
    else:
        weight_values = np.asarray(weights, dtype=np.float64)
        if weight_values.ndim != 1 or weight_values.size != n_sources:
            raise ValueError(
                f"weights must have length {n_sources}, got shape {weight_values.shape}."
            )

    if not np.isfinite(weight_values).all():
        raise ValueError("weights must contain only finite values.")
    if np.any(weight_values < 0.0):
        raise ValueError("weights must be non-negative.")

    weight_sum = float(np.sum(weight_values))
    if weight_sum <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return weight_values / weight_sum


def _zscore(scores: FloatArray) -> FloatArray:
    """Z-score normalize a 1D anomaly score sequence.

    Args:
        scores: Raw anomaly scores of shape ``(T,)``.

    Returns:
        Z-scored values of shape ``(T,)``.
    """
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    if std < _MIN_STD:
        return np.zeros_like(scores, dtype=np.float64)
    return ((scores - mean) / std).astype(np.float64)


def _rank_normalize(scores: FloatArray) -> FloatArray:
    """Convert raw anomaly scores to normalized ranks.

    Args:
        scores: Raw anomaly scores of shape ``(T,)``.

    Returns:
        Rank-normalized scores in ``(0, 1]`` with shape ``(T,)``.
    """
    return (rankdata(scores, method="average") / float(scores.size)).astype(np.float64)


def fuse_scores(
    scores_dict: Mapping[str, npt.ArrayLike],
    method: str = "weighted_sum",
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> FloatArray:
    """Fuse multiple anomaly score sources into a single score sequence.

    Arrays are aligned by truncating each source to the shortest available
    length, which keeps the interface compatible with score sources that may
    differ slightly in window count.

    Args:
        scores_dict: Mapping like ``{"lp": lp_scores, "rp": rp_scores}`` where
            each score array has shape ``(T,)``.
        method: One of ``"weighted_sum"``, ``"rank_fusion"``, or
            ``"zscore_weighted"``.
        weights: Optional source weights as a mapping keyed by source name or as
            an ordered sequence matching ``scores_dict`` iteration order.

    Returns:
        Fused anomaly scores of shape ``(T,)`` and dtype ``np.float64``.

    Raises:
        ValueError: If inputs are invalid or the fusion method is unsupported.
    """
    source_names, aligned_scores = _align_scores(scores_dict)
    weight_vector = _resolve_weights(source_names, weights)
    method_normalized = method.strip().lower()
    stacked_scores = np.stack(aligned_scores, axis=0)

    if method_normalized == "weighted_sum":
        fused = np.einsum("v,vt->t", weight_vector, stacked_scores)
        return fused.astype(np.float64)

    if method_normalized == "rank_fusion":
        ranked_scores = np.stack(
            [_rank_normalize(scores) for scores in aligned_scores],
            axis=0,
        )
        fused = np.einsum("v,vt->t", weight_vector, ranked_scores)
        return fused.astype(np.float64)

    if method_normalized == "zscore_weighted":
        normalized_scores = np.stack([_zscore(scores) for scores in aligned_scores], axis=0)
        fused = np.einsum("v,vt->t", weight_vector, normalized_scores)
        return fused.astype(np.float64)

    raise ValueError(
        "Unsupported fusion method "
        f"'{method}'. Expected one of: ['weighted_sum', 'rank_fusion', 'zscore_weighted']."
    )


def fuse_scores_confidence_weighted(
    renderer_scores: Mapping[str, npt.ArrayLike],
    renderer_residuals: Mapping[str, npt.ArrayLike],
    eps: float = 1e-8,
) -> FloatArray:
    """Fuse renderer scores weighted by inverse per-window uncertainty.

    For each window, the variance of per-patch residuals from a renderer
    provides a natural uncertainty estimate.  Lower variance means higher
    confidence in that renderer's score.

    Args:
        renderer_scores: Mapping from renderer name to anomaly scores ``(T,)``.
        renderer_residuals: Mapping from renderer name to per-patch residuals
            ``(T, N_valid)``.  Must share keys with ``renderer_scores``.
        eps: Small constant for numerical stability.

    Returns:
        Fused anomaly scores of shape ``(T,)``.

    Raises:
        ValueError: If inputs are empty, mismatched, or invalid.
    """
    if not renderer_scores or not renderer_residuals:
        raise ValueError("renderer_scores and renderer_residuals must be non-empty.")

    score_keys = set(renderer_scores.keys())
    resid_keys = set(renderer_residuals.keys())
    if score_keys != resid_keys:
        raise ValueError(
            f"Key mismatch: scores has {sorted(score_keys)}, "
            f"residuals has {sorted(resid_keys)}."
        )

    # Convert and align
    names = sorted(renderer_scores.keys())
    score_arrays: list[FloatArray] = []
    uncertainty_arrays: list[FloatArray] = []

    for name in names:
        scores = np.asarray(renderer_scores[name], dtype=np.float64)
        residuals = np.asarray(renderer_residuals[name], dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError(
                f"Scores for '{name}' must be 1D, got ndim={scores.ndim}."
            )
        if residuals.ndim != 2:
            raise ValueError(
                f"Residuals for '{name}' must be 2D (T, N_valid), got ndim={residuals.ndim}."
            )
        if scores.shape[0] != residuals.shape[0]:
            raise ValueError(
                f"Length mismatch for '{name}': scores={scores.shape[0]}, "
                f"residuals={residuals.shape[0]}."
            )
        score_arrays.append(scores)
        # Per-window uncertainty = variance across patches
        uncertainty_arrays.append(residuals.var(axis=1))

    # Align to shortest common length
    min_len = min(s.size for s in score_arrays)
    score_matrix = np.stack([s[:min_len] for s in score_arrays], axis=1)  # (T, V)
    uncert_matrix = np.stack([u[:min_len] for u in uncertainty_arrays], axis=1)  # (T, V)

    # Confidence = inverse uncertainty
    confidence = 1.0 / (uncert_matrix + eps)  # (T, V)
    weights = confidence / confidence.sum(axis=1, keepdims=True)  # (T, V)

    fused = (weights * score_matrix).sum(axis=1)  # (T,)
    return fused.astype(np.float64)


__all__ = ["fuse_scores", "fuse_scores_confidence_weighted"]
