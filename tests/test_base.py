"""Tests for src/data/base.py — sliding window, time-based split, normalization."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.base import create_sliding_windows, normalize_data, time_based_split


# ---------------------------------------------------------------------------
# create_sliding_windows
# ---------------------------------------------------------------------------


def test_sliding_window_basic() -> None:
    data = np.random.randn(10, 3)
    labels = np.zeros(10, dtype=np.int64)
    windows, wlabels = create_sliding_windows(data, labels, window_size=5, stride=1)
    assert windows.shape == (6, 5, 3)
    assert wlabels.shape == (6,)


def test_sliding_window_stride() -> None:
    data = np.random.randn(10, 2)
    labels = np.zeros(10, dtype=np.int64)
    windows, wlabels = create_sliding_windows(data, labels, window_size=5, stride=3)
    # starts: 0, 3 → 2 windows
    assert windows.shape[0] == 2
    assert wlabels.shape[0] == 2


def test_sliding_window_label_any() -> None:
    data = np.ones((10, 2))
    labels = np.zeros(10, dtype=np.int64)
    labels[3] = 1  # anomaly at index 3
    _, wlabels = create_sliding_windows(data, labels, window_size=5, stride=1)
    # Windows covering index 3: starts 0,1,2,3 → indices 0-3 should be 1
    assert wlabels[0] == 1  # covers [0:5], includes 3
    assert wlabels[3] == 1  # covers [3:8], includes 3
    # Window starting at 4: covers [4:9], excludes 3
    assert wlabels[4] == 0


def test_sliding_window_all_normal() -> None:
    data = np.ones((20, 4))
    labels = np.zeros(20, dtype=np.int64)
    _, wlabels = create_sliding_windows(data, labels, window_size=5, stride=1)
    assert np.all(wlabels == 0)


def test_sliding_window_data_too_short() -> None:
    data = np.ones((3, 2))
    labels = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError, match="window_size"):
        create_sliding_windows(data, labels, window_size=5)


def test_sliding_window_shapes_mismatch() -> None:
    data = np.ones((10, 2))
    labels = np.zeros(8, dtype=np.int64)
    with pytest.raises(ValueError, match="same length"):
        create_sliding_windows(data, labels, window_size=5)


def test_sliding_window_content_correct() -> None:
    """Verify that window content matches the original data slice."""
    data = np.arange(20).reshape(10, 2).astype(np.float64)
    labels = np.zeros(10, dtype=np.int64)
    windows, _ = create_sliding_windows(data, labels, window_size=3, stride=2)
    np.testing.assert_array_equal(windows[0], data[0:3])
    np.testing.assert_array_equal(windows[1], data[2:5])


# ---------------------------------------------------------------------------
# time_based_split
# ---------------------------------------------------------------------------


def test_time_based_split_basic() -> None:
    data = np.random.randn(100, 5)
    labels = np.zeros(100, dtype=np.int64)
    train_d, train_l, test_d, test_l = time_based_split(data, labels, train_ratio=0.5)
    assert test_d.shape[0] == 50
    assert train_d.shape[0] == 50  # all normal in first 50
    assert np.all(train_l == 0)


def test_time_based_split_filters_anomalies() -> None:
    data = np.random.randn(100, 3)
    labels = np.zeros(100, dtype=np.int64)
    labels[10] = 1
    labels[20] = 1
    labels[30] = 1
    train_d, train_l, test_d, test_l = time_based_split(data, labels, train_ratio=0.5)
    # 3 anomaly timesteps in first 50 → train has 47 rows
    assert train_d.shape[0] == 47
    assert np.all(train_l == 0)
    assert test_d.shape[0] == 50


def test_time_based_split_invalid_ratio() -> None:
    data = np.random.randn(100, 3)
    labels = np.zeros(100, dtype=np.int64)
    with pytest.raises(ValueError, match="train_ratio"):
        time_based_split(data, labels, train_ratio=0.0)
    with pytest.raises(ValueError, match="train_ratio"):
        time_based_split(data, labels, train_ratio=1.0)


# ---------------------------------------------------------------------------
# normalize_data
# ---------------------------------------------------------------------------


def test_normalize_standard() -> None:
    np.random.seed(42)
    train = np.random.randn(500, 4) * 10 + 5
    test = np.random.randn(200, 4) * 10 + 5
    norm_train, norm_test = normalize_data(train, test, method="standard")
    np.testing.assert_allclose(np.mean(norm_train, axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.std(norm_train, axis=0), 1.0, atol=1e-10)


def test_normalize_minmax() -> None:
    np.random.seed(42)
    train = np.random.randn(500, 3) * 10 + 5
    test = np.random.randn(200, 3) * 10 + 5
    norm_train, _ = normalize_data(train, test, method="minmax")
    assert np.min(norm_train) >= -1e-10
    assert np.max(norm_train) <= 1.0 + 1e-10


def test_normalize_no_leakage() -> None:
    """Stats should be computed from train only, not from combined data."""
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    test = np.array([[100.0, 200.0]])
    norm_train, norm_test = normalize_data(train, test, method="standard")
    # train mean=[2,3], std=[1,1]
    np.testing.assert_allclose(np.mean(norm_train, axis=0), 0.0, atol=1e-10)
    # test should use train stats → test values will be large
    assert np.all(norm_test > 10)


def test_normalize_unsupported_method() -> None:
    train = np.random.randn(10, 2)
    test = np.random.randn(5, 2)
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_data(train, test, method="invalid")


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_handling_sliding_windows() -> None:
    data = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan], [5.0, 6.0]])
    labels = np.zeros(4, dtype=np.int64)
    windows, _ = create_sliding_windows(data, labels, window_size=2, stride=1)
    assert not np.any(np.isnan(windows))
    # Forward fill: row 1 col 0 should be 1.0, row 2 col 1 should be 3.0
    assert windows[0, 1, 0] == 1.0  # NaN forward-filled from row 0
    assert windows[1, 1, 1] == 3.0  # NaN forward-filled from row 1
