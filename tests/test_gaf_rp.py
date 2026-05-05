"""Tests for GAF and Recurrence Plot renderers."""

from __future__ import annotations

import numpy as np
import pytest

from src.rendering.gaf import render_gaf, render_gaf_batch
from src.rendering.recurrence_plot import (
    render_recurrence_plot,
    render_recurrence_plot_batch,
)
from src.rendering.multi_view import render_multi_view, render_multi_view_batch


def _make_window(length: int = 100, features: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(length, features).astype(np.float64)


# ---------------------------------------------------------------------------
# GAF Tests
# ---------------------------------------------------------------------------


def test_gaf_deterministic() -> None:
    """GAF render is deterministic."""
    window = _make_window()
    img1 = render_gaf(window)
    img2 = render_gaf(window)
    np.testing.assert_array_equal(img1, img2)


def test_gaf_output_shape() -> None:
    """GAF output shape is (3, 224, 224)."""
    window = _make_window()
    img = render_gaf(window, image_size=224)
    assert img.shape == (3, 224, 224)


def test_gaf_output_dtype() -> None:
    """GAF output dtype is float32."""
    window = _make_window()
    img = render_gaf(window)
    assert img.dtype == np.float32


def test_gaf_output_range() -> None:
    """GAF output values are in [0, 1]."""
    window = _make_window()
    img = render_gaf(window)
    assert float(np.min(img)) >= 0.0
    assert float(np.max(img)) <= 1.0


def test_gaf_batch_shape() -> None:
    """GAF batch output shape is (N, 3, 224, 224)."""
    windows = np.stack([_make_window(seed=i) for i in range(4)])
    batch = render_gaf_batch(windows)
    assert batch.shape == (4, 3, 224, 224)


def test_gaf_batch_consistency() -> None:
    """GAF batch is consistent with individual renders."""
    windows = np.stack([_make_window(seed=i) for i in range(3)])
    batch = render_gaf_batch(windows)
    for i in range(3):
        single = render_gaf(windows[i])
        np.testing.assert_array_equal(batch[i], single)


def test_gaf_summation_method() -> None:
    """GAF with method='summation' produces valid output."""
    window = _make_window()
    img = render_gaf(window, method="summation")
    assert img.shape == (3, 224, 224)
    assert img.dtype == np.float32
    assert float(np.min(img)) >= 0.0
    assert float(np.max(img)) <= 1.0


def test_gaf_difference_method() -> None:
    """GAF with method='difference' produces valid output."""
    window = _make_window()
    img = render_gaf(window, method="difference")
    assert img.shape == (3, 224, 224)
    assert img.dtype == np.float32
    assert float(np.min(img)) >= 0.0
    assert float(np.max(img)) <= 1.0


def test_gaf_different_image_sizes() -> None:
    """GAF works with different image sizes."""
    window = _make_window()
    img_64 = render_gaf(window, image_size=64)
    img_224 = render_gaf(window, image_size=224)
    assert img_64.shape == (3, 64, 64)
    assert img_224.shape == (3, 224, 224)


def test_gaf_single_feature() -> None:
    """GAF works with single feature."""
    window = _make_window(features=1)
    img = render_gaf(window)
    assert img.shape == (3, 224, 224)


def test_gaf_many_features() -> None:
    """GAF works with many features."""
    window = _make_window(features=38)
    img = render_gaf(window)
    assert img.shape == (3, 224, 224)


def test_gaf_invalid_window_1d() -> None:
    """GAF raises ValueError for 1D input."""
    with pytest.raises(ValueError, match="ndim"):
        render_gaf(np.array([1.0, 2.0, 3.0]))


def test_gaf_empty_window() -> None:
    """GAF raises ValueError for empty window."""
    with pytest.raises(ValueError, match="non-empty"):
        render_gaf(np.empty((0, 3)))


def test_gaf_non_finite_values() -> None:
    """GAF raises ValueError for non-finite values."""
    window = np.array([[1.0, 2.0], [3.0, float("inf")]])
    with pytest.raises(ValueError, match="non-finite"):
        render_gaf(window)


def test_gaf_invalid_method() -> None:
    """GAF raises ValueError for invalid method."""
    window = _make_window()
    with pytest.raises(ValueError, match="method"):
        render_gaf(window, method="invalid")


def test_gaf_invalid_image_size() -> None:
    """GAF raises ValueError for invalid image_size."""
    window = _make_window()
    with pytest.raises(ValueError, match="positive"):
        render_gaf(window, image_size=0)


# ---------------------------------------------------------------------------
# Recurrence Plot Tests
# ---------------------------------------------------------------------------


def test_rp_deterministic() -> None:
    """Recurrence plot render is deterministic."""
    window = _make_window()
    img1 = render_recurrence_plot(window)
    img2 = render_recurrence_plot(window)
    np.testing.assert_array_equal(img1, img2)


def test_rp_output_shape() -> None:
    """RP output shape is (3, 224, 224)."""
    window = _make_window()
    img = render_recurrence_plot(window, image_size=224)
    assert img.shape == (3, 224, 224)


def test_rp_output_dtype() -> None:
    """RP output dtype is float32."""
    window = _make_window()
    img = render_recurrence_plot(window)
    assert img.dtype == np.float32


def test_rp_output_range() -> None:
    """RP output values are in [0, 1]."""
    window = _make_window()
    img = render_recurrence_plot(window)
    assert float(np.min(img)) >= 0.0
    assert float(np.max(img)) <= 1.0


def test_rp_batch_shape() -> None:
    """RP batch output shape is (N, 3, 224, 224)."""
    windows = np.stack([_make_window(seed=i) for i in range(4)])
    batch = render_recurrence_plot_batch(windows)
    assert batch.shape == (4, 3, 224, 224)


def test_rp_batch_consistency() -> None:
    """RP batch is consistent with individual renders."""
    windows = np.stack([_make_window(seed=i) for i in range(3)])
    batch = render_recurrence_plot_batch(windows)
    for i in range(3):
        single = render_recurrence_plot(windows[i])
        np.testing.assert_array_equal(batch[i], single)


def test_rp_different_metrics() -> None:
    """RP works with different distance metrics."""
    window = _make_window()
    img_euclidean = render_recurrence_plot(window, metric="euclidean")
    img_sqeuclidean = render_recurrence_plot(window, metric="sqeuclidean")
    assert img_euclidean.shape == (3, 224, 224)
    assert img_sqeuclidean.shape == (3, 224, 224)


def test_rp_with_threshold() -> None:
    """RP works with binary threshold."""
    window = _make_window()
    img = render_recurrence_plot(window, threshold=0.5)
    assert img.shape == (3, 224, 224)
    assert img.dtype == np.float32


def test_rp_no_threshold() -> None:
    """RP works without threshold (continuous mode)."""
    window = _make_window()
    img = render_recurrence_plot(window, threshold=None)
    assert img.shape == (3, 224, 224)
    assert img.dtype == np.float32


def test_rp_invalid_window_1d() -> None:
    """RP raises ValueError for 1D input."""
    with pytest.raises(ValueError, match="ndim"):
        render_recurrence_plot(np.array([1.0, 2.0, 3.0]))


def test_rp_empty_window() -> None:
    """RP raises ValueError for empty window."""
    with pytest.raises(ValueError, match="non-empty"):
        render_recurrence_plot(np.empty((0, 3)))


def test_rp_non_finite_values() -> None:
    """RP raises ValueError for non-finite values."""
    window = np.array([[1.0, 2.0], [3.0, float("inf")]])
    with pytest.raises(ValueError, match="non-finite"):
        render_recurrence_plot(window)


def test_rp_invalid_image_size() -> None:
    """RP raises ValueError for invalid image_size."""
    window = _make_window()
    with pytest.raises(ValueError, match="positive"):
        render_recurrence_plot(window, image_size=0)


def test_rp_invalid_threshold() -> None:
    """RP raises ValueError for negative threshold."""
    window = _make_window()
    with pytest.raises(ValueError, match="non-negative"):
        render_recurrence_plot(window, threshold=-1.0)


# ---------------------------------------------------------------------------
# Multi-view Tests
# ---------------------------------------------------------------------------


def test_multi_view_three_renderers() -> None:
    """Multi-view with all three renderers produces shape (9, 224, 224)."""
    window = _make_window()
    img = render_multi_view(window, renderers=["line_plot", "gaf", "recurrence_plot"])
    assert img.shape == (9, 224, 224)


def test_multi_view_single_renderer() -> None:
    """Multi-view with single renderer produces shape (3, 224, 224)."""
    window = _make_window()
    img = render_multi_view(window, renderers=["line_plot"])
    assert img.shape == (3, 224, 224)


def test_multi_view_batch_shape() -> None:
    """Multi-view batch shape is (N, 9, 224, 224)."""
    windows = np.stack([_make_window(seed=i) for i in range(3)])
    batch = render_multi_view_batch(windows)
    assert batch.shape == (3, 9, 224, 224)


def test_multi_view_unknown_renderer() -> None:
    """Multi-view raises ValueError for unknown renderer."""
    window = _make_window()
    with pytest.raises(ValueError, match="Unknown renderer"):
        render_multi_view(window, renderers=["unknown"])


def test_multi_view_default_renderers() -> None:
    """Multi-view uses default renderers when none specified."""
    window = _make_window()
    img1 = render_multi_view(window)
    img2 = render_multi_view(window, renderers=["line_plot", "gaf", "recurrence_plot"])
    np.testing.assert_array_equal(img1, img2)


def test_multi_view_different_sizes() -> None:
    """Multi-view works with different image sizes."""
    window = _make_window()
    img_64 = render_multi_view(window, image_size=64)
    img_224 = render_multi_view(window, image_size=224)
    assert img_64.shape == (9, 64, 64)
    assert img_224.shape == (9, 224, 224)
