"""Evaluation metrics for time series anomaly detection (TSAD)."""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _to_binary_1d(
    array: np.ndarray[Any, Any],
    name: str,
    require_both_classes: bool,
) -> IntArray:
    casted = np.asarray(array, dtype=np.int64)
    if casted.ndim != 1:
        raise ValueError(
            f"{name} must be 1D with shape (T,), got shape {casted.shape}."
        )
    if casted.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    unique_values = np.unique(casted)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"{name} must be binary with values in {{0, 1}}, got {unique_values.tolist()}."
        )
    if require_both_classes and unique_values.size < 2:
        raise ValueError(f"{name} must contain both classes 0 and 1.")
    return casted


def _to_scores_and_labels(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
) -> tuple[FloatArray, IntArray]:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError(
            f"scores must be 1D with shape (T,), got shape {score_array.shape}."
        )
    if score_array.size == 0:
        raise ValueError("scores must be non-empty.")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite values.")

    label_array = _to_binary_1d(
        array=labels,
        name="labels",
        require_both_classes=True,
    )
    if score_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            f"scores and labels must have the same length, got {score_array.shape[0]} "
            f"and {label_array.shape[0]}."
        )
    return score_array, label_array


def _find_anomaly_segments(labels: IntArray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    in_segment = False
    start_idx = 0

    for idx in range(labels.shape[0]):
        value = int(labels[idx])
        if value == 1 and not in_segment:
            start_idx = idx
            in_segment = True
        elif value == 0 and in_segment:
            segments.append((start_idx, idx - 1))
            in_segment = False

    if in_segment:
        segments.append((start_idx, labels.shape[0] - 1))
    return segments


def compute_auc_roc(
    scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
) -> float:
    """Compute Area Under ROC Curve.

    Args:
        scores: Anomaly scores of shape (T,). Higher = more anomalous.
        labels: Binary ground truth of shape (T,). 1 = anomaly.

    Returns:
        AUC-ROC value in [0, 1].
    """
    score_array, label_array = _to_scores_and_labels(scores=scores, labels=labels)
    return float(roc_auc_score(label_array, score_array))


def compute_auc_pr(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Compute Area Under Precision-Recall Curve.

    More robust than AUC-ROC for imbalanced datasets (common in TSAD).

    Args:
        scores: Anomaly scores (T,).
        labels: Binary labels (T,).

    Returns:
        AUC-PR value in [0, 1].
    """
    score_array, label_array = _to_scores_and_labels(scores=scores, labels=labels)
    return float(average_precision_score(label_array, score_array))


def compute_best_f1(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
) -> tuple[float, float]:
    """Compute best F1 score over all possible thresholds.

    .. warning::
        This is an **oracle** metric: the threshold is selected on the test
        labels themselves. It is suitable for upper-bound analysis only and
        must NOT be reported as the deployment-time F1. Use
        :func:`compute_f1_with_validation_threshold` for the
        leakage-free variant.

    Args:
        scores: Anomaly scores (T,).
        labels: Binary labels (T,).

    Returns:
        Tuple of (best_f1, best_threshold).
    """
    score_array, label_array = _to_scores_and_labels(scores=scores, labels=labels)
    precision, recall, thresholds = precision_recall_curve(label_array, score_array)

    precision_array = np.asarray(precision, dtype=np.float64)
    recall_array = np.asarray(recall, dtype=np.float64)
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    if threshold_array.size == 0:
        logger.warning(
            "No thresholds produced by precision_recall_curve; returning zeros."
        )
        return 0.0, float(np.max(score_array))

    denom = precision_array[:-1] + recall_array[:-1]
    f1_scores = np.zeros_like(threshold_array, dtype=np.float64)
    valid = denom > 0.0
    f1_scores[valid] = (
        2.0 * precision_array[:-1][valid] * recall_array[:-1][valid] / denom[valid]
    )

    best_idx = int(np.argmax(f1_scores))
    return float(f1_scores[best_idx]), float(threshold_array[best_idx])


def point_adjust(
    predictions: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Apply point-adjust to binary predictions.

    For each contiguous anomaly segment in labels, if any prediction
    within that segment is 1 (detected), set all predictions in that
    segment to 1.

    Args:
        predictions: Binary predictions (T,). 1 = detected anomaly.
        labels: Binary ground truth (T,). 1 = anomaly.

    Returns:
        Adjusted predictions (T,).
    """
    prediction_array = _to_binary_1d(
        array=predictions,
        name="predictions",
        require_both_classes=False,
    )
    label_array = _to_binary_1d(
        array=labels,
        name="labels",
        require_both_classes=False,
    )
    if prediction_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            f"predictions and labels must have the same length, got {prediction_array.shape[0]} "
            f"and {label_array.shape[0]}."
        )

    adjusted = prediction_array.copy()
    segments = _find_anomaly_segments(labels=label_array)
    for start, end in segments:
        if np.any(adjusted[start : end + 1] == 1):
            adjusted[start : end + 1] = 1

    return adjusted


def compute_f1_pa(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    threshold: float | None = None,
) -> float:
    """Compute F1 score with point-adjust protocol.

    Point-adjust (PA): If ANY point in a contiguous anomaly segment is detected,
    ALL points in that segment are considered detected (TP).
    This is the standard TSAD evaluation protocol used by most papers.

    .. warning::
        When ``threshold`` is None, this function selects the threshold that
        maximises F1-PA on the test labels themselves, making the result an
        **oracle** upper bound. Pass an explicit threshold derived from a
        validation split (or use
        :func:`compute_f1_pa_with_validation_threshold`) for a
        leakage-free measurement.

    Args:
        scores: Anomaly scores (T,).
        labels: Binary labels (T,).
        threshold: Detection threshold. If None, uses the threshold that
            maximizes F1-PA (searches over score percentiles) — *oracle mode*.

    Returns:
        F1-PA score.
    """
    score_array, label_array = _to_scores_and_labels(scores=scores, labels=labels)

    if threshold is None:
        percentiles = np.concatenate(
            [np.arange(90.0, 100.0, 1.0, dtype=np.float64), np.array([99.5, 99.9])]
        )
        candidate_thresholds = np.unique(np.percentile(score_array, percentiles))

        best_f1_pa = 0.0
        for candidate_threshold in candidate_thresholds:
            predictions = (score_array >= float(candidate_threshold)).astype(np.int64)
            adjusted_predictions = point_adjust(
                predictions=predictions,
                labels=label_array,
            )
            current_f1_pa = float(
                f1_score(label_array, adjusted_predictions, zero_division="warn")
            )
            best_f1_pa = max(best_f1_pa, current_f1_pa)

        return best_f1_pa

    predictions = (score_array >= float(threshold)).astype(np.int64)
    adjusted_predictions = point_adjust(predictions=predictions, labels=label_array)
    precision = float(
        precision_score(label_array, adjusted_predictions, zero_division="warn")
    )
    recall = float(
        recall_score(label_array, adjusted_predictions, zero_division="warn")
    )

    if precision + recall == 0.0:
        return 0.0

    return float(2.0 * precision * recall / (precision + recall))


def select_threshold_from_validation(
    val_scores: np.ndarray[Any, Any],
    far_target: float = 0.05,
) -> float:
    """Pick a detection threshold from validation (normal) scores only.

    Uses a quantile rule: ``threshold = quantile(val_scores, 1 - far_target)``.
    This guarantees that no test labels enter the threshold-selection path,
    which is the minimum requirement for a leakage-free F1 / F1-PA report.

    Args:
        val_scores: Validation-set anomaly scores (T_val,). Should come from a
            held-out slice of training data (no test contamination).
        far_target: Target false alarm rate on the validation distribution.
            Default 0.05 matches the 5%-FAR convention.

    Returns:
        Detection threshold (float).
    """
    val_array = np.asarray(val_scores, dtype=np.float64).reshape(-1)
    if val_array.size == 0:
        raise ValueError("val_scores must be non-empty for validation threshold selection.")
    quantile = float(np.clip(1.0 - far_target, 0.0, 1.0))
    return float(np.quantile(val_array, quantile))


def compute_f1_with_validation_threshold(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    val_scores: np.ndarray[Any, Any],
    far_target: float = 0.05,
) -> tuple[float, float]:
    """Leakage-free F1: threshold is chosen from validation scores only.

    Args:
        scores: Test anomaly scores (T,).
        labels: Test binary labels (T,).
        val_scores: Validation (normal-only) anomaly scores.
        far_target: Target false alarm rate used to derive the threshold.

    Returns:
        Tuple ``(f1, threshold)``.
    """
    score_array, label_array = _to_scores_and_labels(scores=scores, labels=labels)
    threshold = select_threshold_from_validation(val_scores, far_target=far_target)
    predictions = (score_array >= threshold).astype(np.int64)
    if predictions.sum() == 0 or predictions.sum() == predictions.size:
        return 0.0, threshold
    f1 = float(f1_score(label_array, predictions, zero_division="warn"))
    return f1, threshold


def compute_f1_pa_with_validation_threshold(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    val_scores: np.ndarray[Any, Any],
    far_target: float = 0.05,
) -> tuple[float, float]:
    """Leakage-free F1-PA: validation-selected threshold + point-adjust."""
    score_array, label_array = _to_scores_and_labels(scores=scores, labels=labels)
    threshold = select_threshold_from_validation(val_scores, far_target=far_target)
    f1_pa = compute_f1_pa(scores=score_array, labels=label_array, threshold=threshold)
    return f1_pa, threshold


def compute_all_metrics(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    val_scores: np.ndarray[Any, Any] | None = None,
    far_target: float = 0.05,
) -> dict[str, float]:
    """Compute all standard TSAD metrics.

    Reports two distinct flavors of F1 / F1-PA:

    * ``best_f1`` / ``f1_pa_oracle``: threshold optimised on the test labels
      themselves. **Oracle** numbers — upper bounds only, do not deploy.
    * ``f1_validation`` / ``f1_pa_validation``: threshold derived from
      ``val_scores`` (normal-only validation slice). Leakage-free.

    Args:
        scores: Test anomaly scores (T,).
        labels: Test binary labels (T,).
        val_scores: Optional validation (normal-only) scores. If None, the
            validation-based metrics are reported as ``None`` and only the
            oracle metrics are returned.
        far_target: Target FAR for the validation threshold rule.

    Returns:
        Dict with keys: ``auc_roc``, ``auc_pr``, ``best_f1`` (oracle),
        ``best_threshold``, ``f1_pa_oracle``, ``f1_validation``,
        ``f1_pa_validation``, ``threshold_validation``.
    """
    _to_scores_and_labels(scores=scores, labels=labels)

    best_f1, best_threshold = compute_best_f1(scores=scores, labels=labels)
    f1_pa_oracle = compute_f1_pa(scores=scores, labels=labels, threshold=None)

    out: dict[str, float] = {
        "auc_roc": compute_auc_roc(scores=scores, labels=labels),
        "auc_pr": compute_auc_pr(scores=scores, labels=labels),
        # Oracle (test-label-tuned) — upper bound only
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "f1_pa_oracle": f1_pa_oracle,
        # Backward-compatible alias retained until callers migrate
        "f1_pa": f1_pa_oracle,
    }

    if val_scores is not None:
        f1_val, thresh_val = compute_f1_with_validation_threshold(
            scores=scores, labels=labels, val_scores=val_scores, far_target=far_target,
        )
        f1_pa_val, _ = compute_f1_pa_with_validation_threshold(
            scores=scores, labels=labels, val_scores=val_scores, far_target=far_target,
        )
        out["f1_validation"] = f1_val
        out["f1_pa_validation"] = f1_pa_val
        out["threshold_validation"] = thresh_val
    else:
        out["f1_validation"] = float("nan")
        out["f1_pa_validation"] = float("nan")
        out["threshold_validation"] = float("nan")

    return out
