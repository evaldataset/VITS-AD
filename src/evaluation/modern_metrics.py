"""Modern threshold-free TSAD metrics (VUS-PR, VUS-ROC, Affiliation-F, range-F1).

The community has moved away from point-adjusted F1, which inflates scores and
leaks test labels through its threshold. The TSB-AD benchmark identifies
**VUS-PR** as the most reliable measure. Rather than re-implement these subtle
metrics, this module wraps the reference implementation shipped with the
``TSB_AD`` package and cross-checks its AUC-ROC/AUC-PR against our own
:mod:`src.evaluation.metrics` so a mis-wired adapter cannot pass silently.

``TSB_AD`` is an optional dependency; import errors are surfaced with an
actionable message instead of failing at module import time.

Example:
    >>> result = compute_modern_metrics(scores, labels, sliding_window=100)
    >>> result["VUS-PR"]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from src.evaluation.metrics import compute_auc_pr, compute_auc_roc

LOGGER = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Metrics we surface from the reference implementation.
MODERN_KEYS: tuple[str, ...] = (
    "VUS-PR",
    "VUS-ROC",
    "Affiliation-F",
    "Event-based-F1",
    "R-based-F1",
    "Standard-F1",
    "PA-F1",
)

#: Tolerance for agreement between the reference AUCs and our own.
_AUC_TOLERANCE: float = 1e-6


def _require_tsb_ad() -> Any:
    """Import and return ``TSB_AD.evaluation.metrics.get_metrics``.

    Returns:
        The reference ``get_metrics`` callable.

    Raises:
        ImportError: If the optional ``TSB_AD`` dependency is unavailable.
    """
    try:
        from TSB_AD.evaluation.metrics import get_metrics
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TSB_AD is required for modern metrics. Install it with "
            "`pip install TSB-AD` (note: it pins numpy<2)."
        ) from exc
    return get_metrics


def compute_modern_metrics(
    scores: npt.NDArray[Any],
    labels: npt.NDArray[Any],
    sliding_window: int = 100,
    verify_against_own: bool = True,
) -> dict[str, float]:
    """Compute the modern threshold-free metric stack for one score series.

    Args:
        scores: Anomaly scores of shape ``(T,)``. Higher means more anomalous.
        labels: Binary ground truth of shape ``(T,)`` with values in ``{0, 1}``.
        sliding_window: Buffer size used by the VUS/range-based measures. Should
            reflect the typical anomaly length; TSB-AD's default is 100.
        verify_against_own: If True, cross-check the reference AUC-ROC/AUC-PR
            against :mod:`src.evaluation.metrics` and log a warning on mismatch.

    Returns:
        Mapping from metric name to value. Contains the keys in
        :data:`MODERN_KEYS` plus ``AUC-ROC`` and ``AUC-PR``.

    Raises:
        ImportError: If the optional ``TSB_AD`` dependency is unavailable.
        ValueError: If inputs are not 1D, differ in length, or labels are
            not binary with both classes present.
    """
    get_metrics = _require_tsb_ad()

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError(
            f"scores and labels must be 1D, got {score_array.shape} and "
            f"{label_array.shape}."
        )
    if score_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            f"scores and labels must have equal length, got "
            f"{score_array.shape[0]} and {label_array.shape[0]}."
        )
    unique = np.unique(label_array)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"labels must be binary, got values {unique.tolist()}.")
    if unique.size < 2:
        raise ValueError("labels must contain both classes 0 and 1.")
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}.")

    raw = get_metrics(score_array, label_array, slidingWindow=sliding_window)
    result: dict[str, float] = {
        key: float(raw[key]) for key in MODERN_KEYS if key in raw
    }
    result["AUC-ROC"] = float(raw["AUC-ROC"])
    result["AUC-PR"] = float(raw["AUC-PR"])

    if verify_against_own:
        own_roc = compute_auc_roc(scores=score_array, labels=label_array)
        own_pr = compute_auc_pr(scores=score_array, labels=label_array)
        for name, reference, own in (
            ("AUC-ROC", result["AUC-ROC"], own_roc),
            ("AUC-PR", result["AUC-PR"], own_pr),
        ):
            if abs(reference - own) > _AUC_TOLERANCE:
                LOGGER.warning(
                    "%s mismatch between TSB-AD reference (%.6f) and src.evaluation "
                    "(%.6f); check score/label alignment.",
                    name,
                    reference,
                    own,
                )
    return result


def summarize_modern_metrics(
    per_series: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-series metric dicts into mean/std per metric.

    Args:
        per_series: Mapping ``series_id -> {metric_name: value}``.

    Returns:
        Mapping ``metric_name -> {"mean": ..., "std": ..., "n": ...}``.

    Raises:
        ValueError: If ``per_series`` is empty.
    """
    if not per_series:
        raise ValueError("per_series must be non-empty.")

    names: set[str] = set()
    for values in per_series.values():
        names.update(values)

    summary: dict[str, dict[str, float]] = {}
    for name in sorted(names):
        collected = [
            float(values[name])
            for values in per_series.values()
            if name in values and np.isfinite(values[name])
        ]
        if not collected:
            continue
        array = np.asarray(collected, dtype=np.float64)
        summary[name] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "n": float(array.size),
        }
    return summary
