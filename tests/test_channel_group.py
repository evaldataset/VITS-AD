"""Tests for src/rendering/channel_group.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.rendering.channel_group import (
    render_channel_groups,
    render_channel_groups_batch,
)


def _dummy_render(window: np.ndarray, image_size: int = 8, **kwargs: object) -> np.ndarray:
    """Minimal renderer: returns a (3, H, W) image encoding channel count."""
    L, D = window.shape
    img = np.full((3, image_size, image_size), D / 100.0, dtype=np.float32)
    return img


class TestRenderChannelGroups:
    def test_output_shape_exact_division(self) -> None:
        window = np.random.randn(100, 12).astype(np.float32)
        images = render_channel_groups(window, group_size=4, render_fn=_dummy_render)
        assert images.shape == (3, 3, 8, 8)  # 12/4 = 3 groups

    def test_output_shape_remainder(self) -> None:
        window = np.random.randn(100, 10).astype(np.float32)
        images = render_channel_groups(window, group_size=4, render_fn=_dummy_render)
        assert images.shape == (3, 3, 8, 8)  # ceil(10/4) = 3 groups

    def test_single_group(self) -> None:
        window = np.random.randn(100, 3).astype(np.float32)
        images = render_channel_groups(window, group_size=6, render_fn=_dummy_render)
        assert images.shape == (1, 3, 8, 8)

    def test_each_group_receives_correct_channels(self) -> None:
        window = np.random.randn(50, 10).astype(np.float32)
        # _dummy_render encodes D as pixel value D/100
        images = render_channel_groups(window, group_size=4, render_fn=_dummy_render)
        # Group 0: 4 channels -> 0.04, Group 1: 4 channels -> 0.04, Group 2: 2 channels -> 0.02
        np.testing.assert_allclose(images[0, 0, 0, 0], 0.04, atol=1e-6)
        np.testing.assert_allclose(images[2, 0, 0, 0], 0.02, atol=1e-6)

    def test_invalid_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="ndim"):
            render_channel_groups(np.zeros((10,)), group_size=4, render_fn=_dummy_render)

    def test_invalid_group_size_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            render_channel_groups(np.zeros((10, 5)), group_size=0, render_fn=_dummy_render)

    def test_kwargs_passed_through(self) -> None:
        window = np.random.randn(50, 6).astype(np.float32)
        images = render_channel_groups(
            window, group_size=3, render_fn=_dummy_render, image_size=16
        )
        assert images.shape == (2, 3, 16, 16)


class TestRenderChannelGroupsBatch:
    def test_batch_output_shape(self) -> None:
        windows = np.random.randn(5, 100, 12).astype(np.float32)
        images = render_channel_groups_batch(
            windows, group_size=4, render_fn=_dummy_render
        )
        assert images.shape == (5, 3, 3, 8, 8)  # (B, G, 3, H, W)

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            render_channel_groups_batch(
                np.zeros((0, 100, 12)), group_size=4, render_fn=_dummy_render
            )

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="ndim"):
            render_channel_groups_batch(
                np.zeros((100, 12)), group_size=4, render_fn=_dummy_render
            )
