"""Deterministic recurrence-plot rendering for time-series windows."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist

LOGGER = logging.getLogger(__name__)

FloatImage = NDArray[np.float32]


def _validate_window(window: np.ndarray) -> np.ndarray:
    window_array = np.asarray(window)
    if window_array.ndim != 2:
        raise ValueError(
            f"window must have shape (L, D), but got ndim={window_array.ndim}."
        )
    if window_array.shape[0] == 0 or window_array.shape[1] == 0:
        raise ValueError(
            f"window must be non-empty in both dimensions, got shape={window_array.shape}."
        )
    if not np.isfinite(window_array).all():
        raise ValueError("window contains non-finite values.")
    return window_array


def _validate_metric(metric: str) -> str:
    if not metric.strip():
        raise ValueError("metric must be a non-empty string.")
    return metric.strip()


def _validate_threshold(threshold: float) -> float:
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite when provided.")
    if threshold_value < 0.0:
        raise ValueError(f"threshold must be non-negative, got {threshold_value}.")
    return threshold_value


def _distance_to_continuous_recurrence(
    distance_matrix: np.ndarray,
) -> FloatImage:
    distances = distance_matrix.astype(np.float32, copy=False)
    min_val = float(np.min(distances))
    max_val = float(np.max(distances))
    if np.isclose(max_val, min_val):
        return np.ones_like(distances, dtype=np.float32)

    normalized = (distances - min_val) / (max_val - min_val)
    recurrence = 1.0 - normalized
    return np.clip(recurrence, 0.0, 1.0).astype(np.float32, copy=False)


def _distance_to_binary_recurrence(
    distance_matrix: np.ndarray,
    threshold: float,
) -> FloatImage:
    return (distance_matrix < threshold).astype(np.float32)


def _resize_square_image(image_hw: np.ndarray, image_size: int) -> FloatImage:
    source_size = int(image_hw.shape[0])
    scale = image_size / source_size
    resized = zoom(image_hw.astype(np.float32, copy=False), zoom=scale, order=1)
    return np.clip(resized.astype(np.float32, copy=False), 0.0, 1.0)


def render_recurrence_plot(
    window: np.ndarray,
    image_size: int = 224,
    threshold: float | None = None,
    metric: str = "euclidean",
) -> FloatImage:
    """Render a time-series window as a recurrence-plot image.

    Uses continuous recurrence by default for better neural network inputs.
    If a threshold is provided, binary recurrence is produced.

    Args:
        window: Time series data of shape (L, D).
        image_size: Output image size (square). Default 224 for CLIP/DINOv2.
        threshold: Optional threshold for binary recurrence.
            If None, continuous recurrence mode is used.
        metric: Distance metric passed to scipy.spatial.distance.cdist.

    Returns:
        RGB image as np.ndarray of shape (3, image_size, image_size), dtype
        float32, values in [0, 1].

    Raises:
        ValueError: If inputs are invalid.
    """
    window_array = _validate_window(window)
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")

    metric_name = _validate_metric(metric)
    window_values = window_array.astype(np.float32, copy=False)
    distance_matrix = cast(
        np.ndarray,
        cdist(window_values, window_values, metric=cast(Any, metric_name)),
    )
    if not np.isfinite(distance_matrix).all():
        raise ValueError("distance matrix contains non-finite values.")

    default_threshold = float(np.percentile(distance_matrix, 10.0))
    if threshold is None:
        recurrence_hw = _distance_to_continuous_recurrence(distance_matrix)
        threshold_used = default_threshold
    else:
        threshold_used = _validate_threshold(threshold)
        recurrence_hw = _distance_to_binary_recurrence(distance_matrix, threshold_used)

    resized_hw = _resize_square_image(recurrence_hw, image_size=image_size)
    image_chw = np.repeat(resized_hw[None, :, :], repeats=3, axis=0)

    LOGGER.debug(
        "Rendered recurrence plot: window_shape=%s image_size=%d metric=%s threshold=%s",
        window_array.shape,
        image_size,
        metric_name,
        threshold_used,
    )
    return np.clip(image_chw.astype(np.float32, copy=False), 0.0, 1.0)


def render_recurrence_plot_batch(
    windows: np.ndarray, **kwargs: Any
) -> FloatImage:
    """Render multiple windows as recurrence-plot images.

    Args:
        windows: Shape (N, L, D).
        **kwargs: Passed to render_recurrence_plot.

    Returns:
        Batch of images, shape (N, 3, image_size, image_size), float32.

    Raises:
        ValueError: If windows has invalid shape.
    """
    windows_array = np.asarray(windows)
    if windows_array.ndim != 3:
        raise ValueError(
            f"windows must have shape (N, L, D), but got ndim={windows_array.ndim}."
        )
    if windows_array.shape[0] == 0:
        raise ValueError("windows batch is empty (N=0).")

    rendered_images = [
        render_recurrence_plot(window=window, **kwargs) for window in windows_array
    ]
    return np.stack(rendered_images, axis=0).astype(np.float32, copy=False)
