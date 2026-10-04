"""Patch token correspondence map construction for sliding windows."""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

SUPPORTED_RENDERERS = {"line_plot", "gaf", "recurrence_plot"}


def compute_correspondence_map(
    renderer_type: str,
    window_size: int,
    stride: int,
    patch_grid: tuple[int, int] = (14, 14),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute patch token correspondence map for a given renderer.

    For `line_plot` (time on x-axis), correspondence is a horizontal shift.
    For `gaf` and `recurrence_plot` (time on both axes), correspondence is a
    diagonal shift.

    Args:
        renderer_type: One of ``line_plot``, ``gaf``, ``recurrence_plot``.
        window_size: Length of the sliding window.
        stride: Sliding window stride.
        patch_grid: Patch grid as ``(height, width)``.

    Returns:
        Tuple ``(pi, valid_mask)`` where:
            - ``pi`` has shape ``(num_patches,)`` and maps source patch index to
              target patch index in the next window.
            - ``valid_mask`` has shape ``(num_patches,)`` and indicates whether
              each mapping is valid.

    Raises:
        ValueError: If inputs are invalid.
    """
    normalized_renderer = renderer_type.strip().lower()
    if normalized_renderer not in SUPPORTED_RENDERERS:
        raise ValueError(
            "Unsupported renderer_type "
            f"'{renderer_type}'. Supported values: {sorted(SUPPORTED_RENDERERS)}."
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    if stride >= window_size:
        raise ValueError(
            f"stride must be smaller than window_size, got stride={stride}, "
            f"window_size={window_size}."
        )
    if len(patch_grid) != 2:
        raise ValueError(
            f"patch_grid must contain two integers (height, width), got {patch_grid}."
        )

    grid_h, grid_w = patch_grid
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"patch_grid dimensions must be positive, got {patch_grid}.")

    num_patches = grid_h * grid_w
    pi = np.full((num_patches,), -1, dtype=np.int64)
    valid_mask = np.zeros((num_patches,), dtype=bool)

    if normalized_renderer == "line_plot":
        delta_col = int(np.round((grid_w * stride) / window_size))
        for row in range(grid_h):
            for col in range(grid_w):
                source_index = row * grid_w + col
                new_col = col - delta_col
                if 0 <= new_col < grid_w:
                    pi[source_index] = row * grid_w + new_col
                    valid_mask[source_index] = True
        return pi, valid_mask

    if grid_h != grid_w:
        raise ValueError(
            "gaf and recurrence_plot require a square patch_grid for diagonal "
            f"correspondence, got {patch_grid}."
        )

    delta = int(np.round((grid_h * stride) / window_size))
    for row in range(grid_h):
        for col in range(grid_w):
            source_index = row * grid_w + col
            new_row = row - delta
            new_col = col - delta
            if 0 <= new_row < grid_h and 0 <= new_col < grid_w:
                pi[source_index] = new_row * grid_w + new_col
                valid_mask[source_index] = True

    return pi, valid_mask


def get_valid_patch_count(valid_mask: np.ndarray) -> int:
    """Return the number of valid correspondence entries.

    Args:
        valid_mask: Boolean mask of valid patch correspondences.

    Returns:
        Number of valid patches.

    Raises:
        ValueError: If ``valid_mask`` is not a one-dimensional array.
    """
    if not isinstance(valid_mask, np.ndarray):
        raise ValueError("valid_mask must be a numpy.ndarray.")
    if valid_mask.ndim != 1:
        raise ValueError(f"valid_mask must be 1D, got ndim={valid_mask.ndim}.")
    return int(np.sum(valid_mask.astype(bool, copy=False)))


def compute_identity_map(
    patch_grid: tuple[int, int] = (14, 14),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute identity correspondence map (ablation: no geometric alignment).

    Each patch maps to itself. Used to measure the contribution of the
    geometric correspondence map π in ablation studies.

    Args:
        patch_grid: Patch grid as ``(height, width)``.

    Returns:
        Tuple ``(pi, valid_mask)`` where pi[i] = i for all patches.
    """
    num_patches = patch_grid[0] * patch_grid[1]
    pi = np.arange(num_patches, dtype=np.int64)
    valid_mask = np.ones(num_patches, dtype=bool)
    return pi, valid_mask
