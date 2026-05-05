"""Tests for smooth_scores function in patchtraj_scorer.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring.patchtraj_scorer import smooth_scores


def test_smooth_scores_mean_known_values() -> None:
    """Smooth scores with method='mean' on known values."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = smooth_scores(scores, window_size=3, method="mean")
    assert smoothed.shape == (5,)
    assert smoothed.dtype == np.float64


def test_smooth_scores_median_outlier_rejection() -> None:
    """Smooth scores with method='median' rejects outliers."""
    scores = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    smoothed = smooth_scores(scores, window_size=3, method="median")
    # Median of [0, 0, 10] is 0.0, middle should be suppressed
    assert smoothed[2] < 2.0  # Should be lower than original


def test_smooth_scores_preserves_length() -> None:
    """Smooth scores preserves the input length."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = smooth_scores(scores, window_size=3)
    assert len(smoothed) == len(scores)


def test_smooth_scores_window_size_one() -> None:
    """Smooth scores with window_size=1 is identity."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = smooth_scores(scores, window_size=1)
    np.testing.assert_array_equal(smoothed, scores)


def test_smooth_scores_full_window() -> None:
    """Smooth scores with window_size >= len(scores) returns constant."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = smooth_scores(scores, window_size=11)  # Must be odd
    assert np.allclose(smoothed, np.full_like(scores, float(np.mean(scores))))


def test_smooth_scores_mean_decreases_spike() -> None:
    """Mean smoothing reduces spikes in input array."""
    scores = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    smoothed = smooth_scores(scores, window_size=3, method="mean")
    # Middle value should be reduced compared to original spike (10 -> 3.33)
    assert smoothed[2] < 5.0


def test_smooth_scores_even_window_size_error() -> None:
    """Smooth scores raises ValueError for even window_size."""
    scores = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="odd"):
        smooth_scores(scores, window_size=2)


def test_smooth_scores_negative_window_size_error() -> None:
    """Smooth scores raises ValueError for negative window_size."""
    scores = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="positive"):
        smooth_scores(scores, window_size=-1)


def test_smooth_scores_non_1d_error() -> None:
    """Smooth scores raises ValueError for non-1D input."""
    scores_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="1D"):
        smooth_scores(scores_2d)


def test_smooth_scores_empty_error() -> None:
    """Smooth scores raises ValueError for empty array."""
    scores = np.array([])
    with pytest.raises(ValueError, match="at least one value"):
        smooth_scores(scores)


def test_smooth_scores_non_finite_error() -> None:
    """Smooth scores raises ValueError for non-finite values."""
    scores = np.array([1.0, 2.0, float("nan"), 4.0])
    with pytest.raises(ValueError, match="non-finite"):
        smooth_scores(scores)


def test_smooth_scores_invalid_method_error() -> None:
    """Smooth scores raises ValueError for invalid method."""
    scores = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Unsupported"):
        smooth_scores(scores, method="invalid")


def test_smooth_scores_median_rejects_extreme_outliers() -> None:
    """Median smoothing rejects extreme outliers effectively."""
    scores = np.array([1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0])
    smoothed = smooth_scores(scores, window_size=3, method="median")
    # Median of [1, 1, 100] is 1.0, so spike should be suppressed
    assert smoothed[3] < 10.0


def test_smooth_scores_custom_window_size() -> None:
    """Smooth scores works with different window sizes."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for ws in [3, 5, 7]:
        smoothed = smooth_scores(scores, window_size=ws, method="mean")
        assert smoothed.shape == (5,)
