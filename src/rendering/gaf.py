"""Deterministic Gramian Angular Field rendering for time-series windows."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

LOGGER = logging.getLogger(__name__)


FloatImage = NDArray[np.float32]


def _validate_window(window: NDArray[np.float32]) -> NDArray[np.float32]:
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


def _normalize_feature_to_minus_one_one(
    feature: NDArray[np.float32],
) -> NDArray[np.float32]:
    values = feature.astype(np.float32, copy=False)
    min_val = float(np.min(values))
    max_val = float(np.max(values))

    if np.isclose(max_val, min_val):
        return np.zeros_like(values, dtype=np.float32)

    normalized = (values - min_val) / (max_val - min_val)
    scaled = (normalized * 2.0) - 1.0
    return np.clip(scaled, -1.0, 1.0).astype(np.float32, copy=False)


def _compute_gaf_feature(
    feature: NDArray[np.float32], method: str
) -> NDArray[np.float32]:
    normalized = _normalize_feature_to_minus_one_one(feature)
    angles = np.arccos(np.clip(normalized, -1.0, 1.0)).astype(np.float32, copy=False)

    if method == "summation":
        matrix = np.cos(np.add.outer(angles, angles))
    else:
        matrix = np.sin(np.subtract.outer(angles, angles))

    return matrix.astype(np.float32, copy=False)


def _resize_square(matrix: NDArray[np.float32], image_size: int) -> NDArray[np.float32]:
    height, width = matrix.shape
    if height == image_size and width == image_size:
        return matrix.astype(np.float32, copy=False)

    zoom_factors = (image_size / float(height), image_size / float(width))
    resized = zoom(
        matrix,
        zoom=zoom_factors,
        order=1,
        mode="nearest",
        prefilter=False,
    )

    if resized.shape != (image_size, image_size):
        resized = resized[:image_size, :image_size]
        pad_h = image_size - resized.shape[0]
        pad_w = image_size - resized.shape[1]
        if pad_h > 0 or pad_w > 0:
            resized = np.pad(
                resized,
                pad_width=((0, max(pad_h, 0)), (0, max(pad_w, 0))),
                mode="edge",
            )
    return resized.astype(np.float32, copy=False)


def _normalize_image(image: NDArray[np.float32]) -> NDArray[np.float32]:
    min_val = float(np.min(image))
    max_val = float(np.max(image))

    if np.isclose(max_val, min_val):
        return np.full_like(image, 0.5, dtype=np.float32)

    normalized = (image - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def render_gaf(
    window: NDArray[np.float32],
    image_size: int = 224,
    method: str = "summation",
) -> FloatImage:
    """Render a time series window as a Gramian Angular Field image.

    MUST be deterministic: same input -> same output (pixel-perfect).

    Args:
        window: Time series data of shape (L, D) where L=window length, D=features.
            For multivariate data, one GAF is computed per feature and then averaged.
        image_size: Output image size (square). Default 224 for CLIP/DINOv2.
        method: GAF method, one of {"summation", "difference"}.

    Returns:
        RGB image as np.ndarray of shape (3, image_size, image_size), dtype float32,
        values in [0, 1].

    Raises:
        ValueError: If window or renderer arguments are invalid.
    """
    window_array = _validate_window(window)
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    if method not in {"summation", "difference"}:
        raise ValueError(
            f"method must be one of {{'summation', 'difference'}}, got '{method}'."
        )

    _, num_features = window_array.shape
    gaf_matrices = [
        _compute_gaf_feature(window_array[:, feature_idx], method=method)
        for feature_idx in range(num_features)
    ]

    gaf_mean = np.mean(np.stack(gaf_matrices, axis=0), axis=0).astype(
        np.float32, copy=False
    )
    gaf_resized = _resize_square(gaf_mean, image_size=image_size)
    gaf_normalized = _normalize_image(gaf_resized)
    image_chw = np.repeat(gaf_normalized[np.newaxis, :, :], repeats=3, axis=0)

    LOGGER.debug(
        "Rendered GAF image with method=%s, input_shape=%s, output_shape=%s",
        method,
        tuple(window_array.shape),
        tuple(image_chw.shape),
    )
    return np.clip(image_chw.astype(np.float32, copy=False), 0.0, 1.0)


def render_gaf_batch(windows: NDArray[np.float32], **kwargs: Any) -> FloatImage:
    """Render multiple windows as Gramian Angular Field images.

    Args:
        windows: Shape (N, L, D).
        **kwargs: Passed to render_gaf.

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
    if windows_array.shape[1] == 0 or windows_array.shape[2] == 0:
        raise ValueError(
            f"windows must be non-empty in L and D dimensions, got shape={windows_array.shape}."
        )
    if not np.isfinite(windows_array).all():
        raise ValueError("windows contains non-finite values.")

    rendered_images = [render_gaf(window=window, **kwargs) for window in windows_array]
    return np.stack(rendered_images, axis=0).astype(np.float32, copy=False)
