"""Tests for src/rendering/line_plot.py — deterministic rendering."""

from __future__ import annotations

import numpy as np
import pytest

from src.rendering.line_plot import render_line_plot, render_line_plot_batch


def _make_window(length: int = 50, features: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(length, features).astype(np.float64)


# ---------------------------------------------------------------------------
# Single render
# ---------------------------------------------------------------------------


def test_render_deterministic() -> None:
    window = _make_window()
    img1 = render_line_plot(window)
    img2 = render_line_plot(window)
    np.testing.assert_array_equal(img1, img2)


def test_render_output_shape() -> None:
    window = _make_window()
    img = render_line_plot(window, image_size=224)
    assert img.shape == (3, 224, 224)


def test_render_output_range() -> None:
    window = _make_window()
    img = render_line_plot(window)
    assert float(np.min(img)) >= 0.0
    assert float(np.max(img)) <= 1.0


def test_render_output_dtype() -> None:
    window = _make_window()
    img = render_line_plot(window)
    assert img.dtype == np.float32


def test_render_different_inputs() -> None:
    w1 = _make_window(seed=0)
    w2 = _make_window(seed=1)
    img1 = render_line_plot(w1)
    img2 = render_line_plot(w2)
    assert not np.array_equal(img1, img2)


def test_render_single_feature() -> None:
    window = _make_window(features=1)
    img = render_line_plot(window)
    assert img.shape == (3, 224, 224)


def test_render_many_features() -> None:
    window = _make_window(features=38)
    img = render_line_plot(window)
    assert img.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# Batch render
# ---------------------------------------------------------------------------


def test_render_batch_shape() -> None:
    windows = np.stack([_make_window(seed=i) for i in range(4)])
    batch = render_line_plot_batch(windows)
    assert batch.shape == (4, 3, 224, 224)


def test_render_batch_consistency() -> None:
    windows = np.stack([_make_window(seed=i) for i in range(3)])
    batch = render_line_plot_batch(windows)
    for i in range(3):
        single = render_line_plot(windows[i])
        np.testing.assert_array_equal(batch[i], single)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_render_invalid_window_1d() -> None:
    with pytest.raises(ValueError, match="shape"):
        render_line_plot(np.array([1.0, 2.0, 3.0]))


def test_render_empty_window() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        render_line_plot(np.empty((0, 3)))
