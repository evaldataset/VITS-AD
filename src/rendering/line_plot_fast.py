"""Fast deterministic line-plot rendering using pure numpy (no matplotlib).

Strategy: fully vectorized Wu anti-aliased line drawing.
All segments for a given channel are rasterized in a single numpy pass —
no Python loops over individual pixels.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

FloatImage = NDArray[np.float32]

# Tab20-like color palette (RGB float32, 20 colors)
_TAB20_COLORS: list[tuple[float, float, float]] = [
    (0.122, 0.467, 0.706),  # blue
    (1.000, 0.498, 0.055),  # orange
    (0.173, 0.627, 0.173),  # green
    (0.839, 0.153, 0.157),  # red
    (0.580, 0.404, 0.741),  # purple
    (0.549, 0.337, 0.294),  # brown
    (0.890, 0.467, 0.761),  # pink
    (0.498, 0.498, 0.498),  # gray
    (0.737, 0.741, 0.133),  # olive
    (0.090, 0.745, 0.812),  # cyan
    (0.682, 0.780, 0.910),  # light blue
    (1.000, 0.733, 0.471),  # light orange
    (0.596, 0.875, 0.541),  # light green
    (1.000, 0.596, 0.588),  # light red
    (0.773, 0.690, 0.835),  # light purple
    (0.769, 0.612, 0.580),  # light brown
    (0.969, 0.714, 0.824),  # light pink
    (0.780, 0.780, 0.780),  # light gray
    (0.859, 0.859, 0.553),  # light olive
    (0.620, 0.855, 0.898),  # light cyan
]

_COLORS_ARRAY = np.array(_TAB20_COLORS, dtype=np.float32)

_MARGIN = 4  # pixel margin top/bottom


def _rasterize_segments_wu(
    img: np.ndarray,
    x0s: np.ndarray,
    y0s: np.ndarray,
    x1s: np.ndarray,
    y1s: np.ndarray,
    color: np.ndarray,
) -> None:
    """Rasterize multiple line segments simultaneously using Wu's algorithm.

    All segments are handled with a single set of vectorized numpy operations.
    Pixels from multiple segments that land on the same location are blended
    cumulatively (last writer wins per segment ordering, using np.minimum for
    brightness accumulation on a white background).

    Args:
        img: HxWx3 float32 image (white background = 1.0).
        x0s: Start x coords, shape (S,) int32.
        y0s: Start y coords, shape (S,) int32.
        x1s: End x coords, shape (S,) int32.
        y1s: End y coords, shape (S,) int32.
        color: RGB color (3,) float32 [0, 1].
    """
    H, W = img.shape[:2]

    dx = (x1s - x0s).astype(np.float32)
    dy = (y1s - y0s).astype(np.float32)
    steep = np.abs(dy) > np.abs(dx)

    # Swap x/y for steep segments
    x0s = x0s.copy()
    y0s = y0s.copy()
    x1s = x1s.copy()
    y1s = y1s.copy()

    x0s[steep], y0s[steep] = y0s[steep].copy(), x0s[steep].copy()
    x1s[steep], y1s[steep] = y1s[steep].copy(), x1s[steep].copy()

    # Ensure left-to-right
    swap = x0s > x1s
    x0s[swap], x1s[swap] = x1s[swap].copy(), x0s[swap].copy()
    y0s[swap], y1s[swap] = y1s[swap].copy(), y0s[swap].copy()

    dx = (x1s - x0s).astype(np.float32)
    dy = (y1s - y0s).astype(np.float32)

    # Skip degenerate (zero-length) segments
    valid = dx != 0
    if not valid.any():
        return

    x0s, y0s, x1s, y1s = x0s[valid], y0s[valid], x1s[valid], y1s[valid]
    dx, dy, steep_v = dx[valid], dy[valid], steep[valid]

    gradients = dy / dx  # (S,)

    # Expand each segment into its pixel run.
    # segment_lengths[s] = dx[s] + 1 pixels per segment.
    lengths = (dx + 1).astype(np.int32)  # (S,)
    total = int(lengths.sum())

    # Build flat arrays of (seg_idx, local_offset) for all pixels
    seg_idx = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)  # (total,)
    local_t = np.arange(total, dtype=np.int32) - np.repeat(
        np.concatenate([[0], lengths[:-1].cumsum()]), lengths
    )  # offset within each segment

    # Pixel x along the dominant axis
    px = x0s[seg_idx] + local_t  # (total,)

    # Floating-point y
    py_f = y0s[seg_idx].astype(np.float32) + gradients[seg_idx] * local_t.astype(np.float32)
    py_floor = py_f.astype(np.int32)
    frac = (py_f - py_floor).astype(np.float32)

    steep_pix = steep_v[seg_idx]  # (total,)

    # Row/col for the two anti-aliased sub-pixels
    # Non-steep: row=py, col=px; steep: row=px, col=py (swapped back)
    row0 = np.where(steep_pix, px, py_floor)
    col0 = np.where(steep_pix, py_floor, px)
    row1 = np.where(steep_pix, px, py_floor + 1)
    col1 = np.where(steep_pix, py_floor + 1, px)

    alpha0 = (1.0 - frac).astype(np.float32)
    alpha1 = frac

    # Clamp to image bounds
    mask0 = (row0 >= 0) & (row0 < H) & (col0 >= 0) & (col0 < W)
    mask1 = (row1 >= 0) & (row1 < H) & (col1 >= 0) & (col1 < W)

    # On a white background, blending color c with alpha a gives:
    # pixel = pixel * (1 - a) + color * a
    # For accumulation across segments we use np.minimum (darkest wins on white bg)
    # which correctly accumulates overlapping lines without double-brightening.

    # Pixel 0 contribution: minimum of existing pixel and blended value
    r0, c0 = row0[mask0], col0[mask0]
    a0 = alpha0[mask0, np.newaxis]
    blended0 = img[r0, c0] * (1.0 - a0) + color * a0
    np.minimum(img[r0, c0], blended0, out=img[r0, c0])

    # Pixel 1 contribution
    r1, c1 = row1[mask1], col1[mask1]
    a1 = alpha1[mask1, np.newaxis]
    blended1 = img[r1, c1] * (1.0 - a1) + color * a1
    np.minimum(img[r1, c1], blended1, out=img[r1, c1])


def render_line_plot_fast(
    window: np.ndarray,
    image_size: int = 224,
) -> FloatImage:
    """Render a time series window as a line plot using pure numpy.

    DETERMINISTIC: same input -> identical pixel output.

    Args:
        window: Time series data of shape (L, D) where L=window length, D=features.
        image_size: Output image size (square). Default 224 for CLIP/DINOv2.

    Returns:
        RGB image as np.ndarray of shape (3, image_size, image_size), dtype float32,
        values in [0, 1].

    Raises:
        ValueError: If window is empty, has wrong dimensions, or contains non-finite values.
    """
    window_array = np.asarray(window, dtype=np.float32)
    if window_array.ndim != 2:
        raise ValueError(f"window must have shape (L, D), but got ndim={window_array.ndim}.")
    if window_array.shape[0] == 0 or window_array.shape[1] == 0:
        raise ValueError(
            f"window must be non-empty in both dimensions, got shape={window_array.shape}."
        )
    if not np.isfinite(window_array).all():
        raise ValueError("window contains non-finite values.")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")

    H = image_size
    W_img = image_size
    T, D = window_array.shape
    margin = _MARGIN

    # White background, HWC layout for in-place drawing
    img = np.ones((H, W_img, 3), dtype=np.float32)

    # x-coordinates: T points evenly spanning [0, W_img-1]
    xs = np.round(np.linspace(0, W_img - 1, T)).astype(np.int32)

    # Global min/max for consistent y-scaling across all channels
    global_min = float(window_array.min())
    global_max = float(window_array.max())
    global_range = global_max - global_min

    draw_height = H - 2 * margin

    # Precompute all y-coords: shape (T, D)
    if global_range < 1e-8:
        ys_all = np.full((T, D), H // 2, dtype=np.int32)
    else:
        ys_all = (
            (1.0 - (window_array - global_min) / global_range) * draw_height + margin
        ).astype(np.int32)
        ys_all = np.clip(ys_all, 0, H - 1)

    # Segment endpoints: each channel d has (T-1) segments
    # x0s/x1s are the same for all channels
    x0s_base = xs[:-1]  # (T-1,)
    x1s_base = xs[1:]   # (T-1,)

    for d in range(D):
        color = _COLORS_ARRAY[d % len(_COLORS_ARRAY)]
        y0s = ys_all[:-1, d]  # (T-1,)
        y1s = ys_all[1:, d]   # (T-1,)
        _rasterize_segments_wu(img, x0s_base, y0s, x1s_base, y1s, color)

    # Convert HWC -> CHW
    return np.clip(img.transpose(2, 0, 1), 0.0, 1.0)


def render_line_plot_fast_batch(windows: np.ndarray, **kwargs: object) -> FloatImage:
    """Render multiple windows as fast line plots.

    Args:
        windows: Shape (N, L, D).
        **kwargs: Passed to render_line_plot_fast.

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

    rendered_images = [render_line_plot_fast(window=window, **kwargs) for window in windows_array]
    return np.stack(rendered_images, axis=0).astype(np.float32, copy=False)
