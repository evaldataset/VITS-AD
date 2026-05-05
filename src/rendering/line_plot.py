"""Deterministic line-plot rendering for time-series windows."""

from __future__ import annotations

import io
import logging

import matplotlib
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


FloatImage = NDArray[np.float32]


def _validate_window(window: np.ndarray) -> np.ndarray:
    """Validate a single time-series window.

    Args:
        window: Time-series window expected to have shape (L, D).

    Raises:
        ValueError: If the input shape or values are invalid.
    """
    window_array = np.asarray(window)
    if window_array.ndim != 2:
        raise ValueError(f"window must have shape (L, D), but got ndim={window.ndim}.")
    if window_array.shape[0] == 0 or window_array.shape[1] == 0:
        raise ValueError(
            f"window must be non-empty in both dimensions, got shape={window_array.shape}."
        )
    if not np.isfinite(window_array).all():
        raise ValueError("window contains non-finite values.")
    return window_array


def _normalize_window(window: np.ndarray) -> FloatImage:
    """Min-max normalize a window to [0, 1] across all features.

    Args:
        window: Input window of shape (L, D).

    Returns:
        Normalized array of shape (L, D), dtype float32.
    """
    values = window.astype(np.float32, copy=False)
    min_val = float(np.min(values))
    max_val = float(np.max(values))

    if np.isclose(max_val, min_val):
        return np.full_like(values, 0.5, dtype=np.float32)

    normalized = (values - min_val) / (max_val - min_val)
    return normalized.astype(np.float32, copy=False)


def render_line_plot(
    window: np.ndarray,
    image_size: int = 224,
    dpi: int = 100,
    colormap: str = "tab10",
    line_width: float = 1.0,
    background_color: str = "white",
    show_axes: bool = False,
    show_grid: bool = False,
) -> FloatImage:
    """Render a time series window as a line plot image.

    MUST be deterministic: same input -> same output (pixel-perfect).

    Args:
        window: Time series data of shape (L, D) where L=window length, D=features.
            Each feature is plotted as a separate line.
        image_size: Output image size (square). Default 224 for CLIP/DINOv2.
        dpi: Figure DPI for matplotlib rendering.
        colormap: Matplotlib colormap name for line colors.
        line_width: Width of plotted lines.
        background_color: Figure background color.
        show_axes: Whether to show axis ticks/labels.
        show_grid: Whether to show grid lines.

    Returns:
        RGB image as np.ndarray of shape (3, image_size, image_size), dtype float32,
        values in [0, 1].

    Raises:
        ValueError: If window is empty or has wrong dimensions.
    """
    window_array = _validate_window(window)
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}.")
    if line_width <= 0:
        raise ValueError(f"line_width must be positive, got {line_width}.")

    normalized_window = _normalize_window(window_array)
    sequence_length, num_features = normalized_window.shape
    x_axis = np.arange(sequence_length, dtype=np.float32)

    render_rc_params: dict[str, object] = {
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.pad_inches": 0.0,
        "font.family": "DejaVu Sans",
        "axes.facecolor": background_color,
        "figure.facecolor": background_color,
    }

    with matplotlib.rc_context(render_rc_params):
        fig = plt.figure(
            figsize=(image_size / dpi, image_size / dpi),
            dpi=dpi,
            facecolor=background_color,
        )
        ax = fig.add_subplot(111)

        cmap = plt.get_cmap(colormap, num_features)
        for feature_idx in range(num_features):
            ax.plot(
                x_axis,
                normalized_window[:, feature_idx],
                color=cmap(feature_idx),
                linewidth=line_width,
                antialiased=True,
            )

        ax.set_xlim(0, max(sequence_length - 1, 1))
        ax.set_ylim(0.0, 1.0)
        ax.margins(x=0.0, y=0.0)

        if show_grid:
            ax.grid(True, linewidth=0.3, alpha=0.5)
        else:
            ax.grid(False)

        if show_axes:
            ax.tick_params(which="both", labelbottom=False, labelleft=False)
        else:
            ax.set_axis_off()

        fig.tight_layout(pad=0.0)

        buffer = io.BytesIO()
        try:
            fig.savefig(buffer, format="raw", dpi=dpi, pad_inches=0.0)
            buffer.seek(0)
            width, height = fig.canvas.get_width_height()
            image_rgba = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            image_rgba = image_rgba.reshape((height, width, -1))
            image_rgb = image_rgba[:, :, :3]
        finally:
            buffer.close()
            plt.close(fig)

    if image_rgb.shape[0] != image_size or image_rgb.shape[1] != image_size:
        raise ValueError(
            f"Rendered image size mismatch: expected ({image_size}, {image_size}), "
            f"got {image_rgb.shape[:2]}."
        )

    image_chw = np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1))
    return np.clip(image_chw, 0.0, 1.0)


def render_line_plot_batch(windows: np.ndarray, **kwargs: object) -> FloatImage:
    """Render multiple windows as line plots.

    Args:
        windows: Shape (N, L, D).
        **kwargs: Passed to render_line_plot.

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
        render_line_plot(window=window, **kwargs) for window in windows_array
    ]
    return np.stack(rendered_images, axis=0).astype(np.float32, copy=False)
