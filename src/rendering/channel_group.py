"""Channel-group rendering for high-dimensional multivariate time series.

Partitions D channels into groups of size k, renders each group independently,
and returns per-group images.  This avoids compressing D>>3 channels into a
single 3-channel RGB image, preserving per-channel amplitude information that
would otherwise be lost.

Typical usage::

    images = render_channel_groups(window, group_size=6, render_fn=render_line_plot)
    # images.shape = (ceil(D/6), 3, 224, 224)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

FloatImage = NDArray[np.float32]


def render_channel_groups(
    window: np.ndarray,
    group_size: int,
    render_fn: Callable[..., FloatImage],
    **render_kwargs: Any,
) -> FloatImage:
    """Render a single window by channel groups.

    Args:
        window: Time-series window ``(L, D)``.
        group_size: Number of channels per rendering group.
            Must be positive.  The last group may have fewer channels.
        render_fn: Single-window renderer, e.g. ``render_line_plot``.
            Signature: ``render_fn(window: ndarray, **kwargs) -> (3, H, W)``.
        **render_kwargs: Passed to ``render_fn``.

    Returns:
        Rendered images ``(G, 3, H, W)`` where ``G = ceil(D / group_size)``.

    Raises:
        ValueError: If inputs are invalid.
    """
    window = np.asarray(window)
    if window.ndim != 2:
        raise ValueError(f"window must have shape (L, D), got ndim={window.ndim}.")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    L, D = window.shape
    num_groups = math.ceil(D / group_size)

    images: list[FloatImage] = []
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, D)
        group_window = window[:, start:end]  # (L, k) where k <= group_size
        img = render_fn(window=group_window, **render_kwargs)
        images.append(img)

    return np.stack(images, axis=0).astype(np.float32, copy=False)


def render_channel_groups_batch(
    windows: np.ndarray,
    group_size: int,
    render_fn: Callable[..., FloatImage],
    **render_kwargs: Any,
) -> FloatImage:
    """Render a batch of windows by channel groups.

    Args:
        windows: Batch of windows ``(B, L, D)``.
        group_size: Channels per group.
        render_fn: Single-window renderer.
        **render_kwargs: Passed to ``render_fn``.

    Returns:
        Rendered images ``(B, G, 3, H, W)`` where ``G = ceil(D / group_size)``.

    Raises:
        ValueError: If inputs are invalid.
    """
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError(
            f"windows must have shape (B, L, D), got ndim={windows.ndim}."
        )
    if windows.shape[0] == 0:
        raise ValueError("Empty batch (B=0).")

    batch_images = [
        render_channel_groups(w, group_size, render_fn, **render_kwargs)
        for w in windows
    ]
    return np.stack(batch_images, axis=0).astype(np.float32, copy=False)
