from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import numpy as np
import pytest

from src.evaluation.metrics import (
    _find_anomaly_segments,
    compute_all_metrics,
    compute_auc_pr,
    compute_auc_roc,
    compute_best_f1,
    compute_f1_pa,
    point_adjust,
)


def test_compute_auc_roc_perfect_scores() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    auc = compute_auc_roc(scores=scores, labels=labels)
    assert auc == pytest.approx(1.0)


def test_compute_auc_roc_random_scores_near_half() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 100)
    scores = rng.random(labels.shape[0])
    auc = compute_auc_roc(scores=scores, labels=labels)
    assert 0.35 <= auc <= 0.65


def test_compute_auc_roc_inverted_scores() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.95, 0.9, 0.8, 0.2, 0.1])
    auc = compute_auc_roc(scores=scores, labels=labels)
    assert auc == pytest.approx(0.0)


def test_compute_auc_pr_perfect_scores() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    auc_pr = compute_auc_pr(scores=scores, labels=labels)
    assert auc_pr == pytest.approx(1.0)


def test_compute_auc_pr_imperfect_is_positive() -> None:
    labels = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.3, 0.2, 0.7, 0.4, 0.6, 0.8, 0.5, 0.9, 0.05])
    auc_pr = compute_auc_pr(scores=scores, labels=labels)
    assert 0.0 < auc_pr < 1.0


def test_compute_best_f1_perfect_detection() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    best_f1, threshold = compute_best_f1(scores=scores, labels=labels)
    assert best_f1 == pytest.approx(1.0)
    assert 0.3 < threshold <= 0.95


def test_compute_best_f1_returns_float_tuple() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    result = compute_best_f1(scores=scores, labels=labels)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_point_adjust_single_segment_all_detected() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0])
    preds = np.array([0, 0, 1, 1, 1, 0])
    adjusted = point_adjust(predictions=preds, labels=labels)
    np.testing.assert_array_equal(adjusted, np.array([0, 0, 1, 1, 1, 0]))


def test_point_adjust_single_segment_partial_detected_becomes_all_tp() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0])
    preds = np.array([0, 0, 0, 0, 1, 0])
    adjusted = point_adjust(predictions=preds, labels=labels)
    np.testing.assert_array_equal(adjusted, np.array([0, 0, 1, 1, 1, 0]))


def test_point_adjust_single_segment_no_detection_no_change() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0])
    preds = np.array([0, 0, 0, 0, 0, 0])
    adjusted = point_adjust(predictions=preds, labels=labels)
    np.testing.assert_array_equal(adjusted, preds)


def test_point_adjust_multiple_segments() -> None:
    labels = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1])
    preds = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0])
    adjusted = point_adjust(predictions=preds, labels=labels)
    expected = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(adjusted, expected)


def test_compute_f1_pa_perfect_with_explicit_threshold() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    f1_pa = compute_f1_pa(scores=scores, labels=labels, threshold=0.9)
    assert f1_pa == pytest.approx(1.0)


def test_compute_f1_pa_auto_threshold_search() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.95, 0.4, 0.2, 0.1])
    f1_pa = compute_f1_pa(scores=scores, labels=labels, threshold=None)
    assert f1_pa == pytest.approx(1.0)


def test_compute_f1_pa_with_explicit_threshold_imperfect() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.95, 0.4, 0.2, 0.1])
    f1_pa = compute_f1_pa(scores=scores, labels=labels, threshold=0.99)
    assert f1_pa == pytest.approx(0.0)


def test_compute_all_metrics_returns_expected_keys() -> None:
    labels = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.15, 0.25, 0.7, 0.85, 0.95])
    metrics = compute_all_metrics(scores=scores, labels=labels)
    expected_keys = {
        "auc_roc",
        "auc_pr",
        "best_f1",
        "best_threshold",
        "f1_pa",
        "f1_pa_oracle",
        "f1_validation",
        "f1_pa_validation",
        "threshold_validation",
    }
    assert set(metrics.keys()) == expected_keys


def test_compute_all_metrics_with_validation_threshold() -> None:
    labels = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.15, 0.25, 0.7, 0.85, 0.95])
    val_scores = np.array([0.05, 0.08, 0.12, 0.15, 0.2, 0.18, 0.1, 0.22])
    metrics = compute_all_metrics(
        scores=scores, labels=labels, val_scores=val_scores, far_target=0.05,
    )
    assert metrics["f1_validation"] >= 0.0
    assert metrics["f1_pa_validation"] >= 0.0
    # Validation threshold should never come from test labels.
    assert np.isfinite(metrics["threshold_validation"])
    # Oracle and validation may differ; oracle should be >= validation by definition.
    assert metrics["best_f1"] >= metrics["f1_validation"] - 1e-9


def test_find_anomaly_segments_all_required_patterns() -> None:
    assert _find_anomaly_segments(np.array([0, 0, 0, 0])) == []
    assert _find_anomaly_segments(np.array([0, 1, 1, 1, 0])) == [(1, 3)]
    assert _find_anomaly_segments(np.array([0, 1, 1, 0, 1, 0, 1, 1])) == [
        (1, 2),
        (4, 4),
        (6, 7),
    ]
    assert _find_anomaly_segments(np.array([1, 1, 1, 1])) == [(0, 3)]
    assert _find_anomaly_segments(np.array([1, 1, 0, 0, 1])) == [(0, 1), (4, 4)]


def test_validation_errors_for_scores_and_labels() -> None:
    with pytest.raises(ValueError, match="scores must be 1D"):
        compute_auc_roc(scores=np.array([[0.1, 0.2]]), labels=np.array([0, 1]))

    with pytest.raises(ValueError, match="same length"):
        compute_auc_pr(scores=np.array([0.1, 0.2]), labels=np.array([0, 1, 0]))

    with pytest.raises(ValueError, match="scores must be non-empty"):
        compute_best_f1(scores=np.array([]), labels=np.array([]))

    with pytest.raises(ValueError, match="binary"):
        compute_auc_roc(scores=np.array([0.1, 0.2, 0.3]), labels=np.array([0, 2, 1]))

    with pytest.raises(ValueError, match="both classes"):
        compute_auc_roc(scores=np.array([0.1, 0.2, 0.3]), labels=np.array([0, 0, 0]))

    with pytest.raises(ValueError, match="finite"):
        compute_auc_pr(scores=np.array([0.1, np.nan, 0.2]), labels=np.array([0, 1, 0]))


def test_validation_errors_for_point_adjust_inputs() -> None:
    with pytest.raises(ValueError, match="predictions must be 1D"):
        point_adjust(predictions=np.array([[0, 1]]), labels=np.array([0, 1]))

    with pytest.raises(ValueError, match="same length"):
        point_adjust(predictions=np.array([0, 1]), labels=np.array([0, 1, 1]))

    with pytest.raises(ValueError, match="predictions must be non-empty"):
        point_adjust(predictions=np.array([]), labels=np.array([]))

    with pytest.raises(ValueError, match="labels must be binary"):
        point_adjust(predictions=np.array([0, 1]), labels=np.array([0, 2]))
