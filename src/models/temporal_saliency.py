from __future__ import annotations

import logging
from typing import Final

import numpy as np
import numpy.typing as npt
import torch


LOGGER = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
_SUPPORTED_HEAD_FUSION: Final[set[str]] = {"mean", "max", "min"}
_SUPPORTED_RENDERERS: Final[set[str]] = {"line_plot", "gaf", "recurrence_plot"}


def _validate_attentions(attentions: tuple[torch.Tensor, ...]) -> tuple[int, int, int]:
    """Validate attention tuple shape consistency.

    Args:
        attentions: Per-layer attention tensors with shape ``(B, H, S, S)``.

    Returns:
        Tuple ``(batch_size, seq_len, patch_side)``.

    Raises:
        ValueError: If attention tensors are empty or have incompatible shapes.
    """
    if not attentions:
        raise ValueError("attentions must contain at least one layer tensor.")

    first = attentions[0]
    if first.ndim != 4:
        raise ValueError(
            f"Each attention tensor must have shape (B, H, S, S), got first tensor shape {tuple(first.shape)}."
        )

    batch_size, _, seq_len, seq_len_2 = first.shape
    if batch_size <= 0:
        raise ValueError("attention tensors must have B > 0.")
    if seq_len <= 1:
        raise ValueError(
            "attention sequence length must be > 1 to include CLS and patch tokens."
        )
    if seq_len != seq_len_2:
        raise ValueError(
            f"attention tensors must be square over sequence dimension, got S={seq_len} and S2={seq_len_2}."
        )

    for layer_idx, layer_attn in enumerate(attentions):
        if layer_attn.ndim != 4:
            raise ValueError(
                f"attention tensor at layer {layer_idx} must be 4D, got ndim={layer_attn.ndim}."
            )
        if layer_attn.shape[0] != batch_size:
            raise ValueError(
                f"All attention tensors must have matching batch size, expected {batch_size}, got {layer_attn.shape[0]} at layer {layer_idx}."
            )
        if layer_attn.shape[2] != seq_len or layer_attn.shape[3] != seq_len:
            raise ValueError(
                f"All attention tensors must have matching sequence dimensions, expected ({seq_len}, {seq_len}), got ({layer_attn.shape[2]}, {layer_attn.shape[3]}) at layer {layer_idx}."
            )

    num_patches = seq_len - 1
    patch_side = int(np.sqrt(num_patches))
    if patch_side * patch_side != num_patches:
        raise ValueError(
            f"Number of patch tokens must be a perfect square, got num_patches={num_patches}."
        )
    return batch_size, seq_len, patch_side


def _fuse_heads(attention: torch.Tensor, head_fusion: str) -> torch.Tensor:
    """Fuse multi-head attention into one matrix per sample.

    Args:
        attention: Attention tensor of shape ``(B, H, S, S)``.
        head_fusion: Head fusion method.

    Returns:
        Fused attention tensor of shape ``(B, S, S)``.

    Raises:
        ValueError: If ``head_fusion`` is unsupported.
    """
    if head_fusion == "mean":
        return attention.mean(dim=1)
    if head_fusion == "max":
        return attention.max(dim=1).values
    if head_fusion == "min":
        return attention.min(dim=1).values
    raise ValueError(
        f"Unsupported head_fusion '{head_fusion}'. Supported values: {sorted(_SUPPORTED_HEAD_FUSION)}."
    )


def _discard_low_attention(
    attention: torch.Tensor, discard_ratio: float
) -> torch.Tensor:
    """Discard low attention weights in non-CLS submatrix.

    Args:
        attention: Fused attention tensor of shape ``(B, S, S)``.
        discard_ratio: Fraction of smallest values to zero-out.

    Returns:
        Attention tensor with low entries zeroed in the ``[1:, 1:]`` block.

    Raises:
        ValueError: If ``discard_ratio`` is outside ``[0, 1)``.
    """
    if discard_ratio == 0.0:
        return attention
    if discard_ratio < 0.0 or discard_ratio >= 1.0:
        raise ValueError(
            f"discard_ratio must satisfy 0 <= discard_ratio < 1, got {discard_ratio}."
        )

    batch_size, seq_len, _ = attention.shape
    flattened = attention[:, 1:, 1:].reshape(batch_size, -1)
    num_discard = int(flattened.shape[1] * discard_ratio)
    if num_discard <= 0:
        return attention

    _, smallest_indices = torch.topk(
        flattened,
        k=num_discard,
        dim=1,
        largest=False,
        sorted=False,
    )
    keep_mask = torch.ones_like(flattened, dtype=torch.bool)
    _ = keep_mask.scatter_(1, smallest_indices, False)

    retained = torch.where(keep_mask, flattened, torch.zeros_like(flattened))
    attention_retained = attention.clone()
    attention_retained[:, 1:, 1:] = retained.reshape(
        batch_size, seq_len - 1, seq_len - 1
    )
    return attention_retained


def _normalize_to_unit_interval(values: FloatArray) -> FloatArray:
    """Normalize a non-negative array to ``[0, 1]`` by dividing by the max.

    This preserves the zero/non-zero distinction: a positive raw value always
    maps to a positive normalized value.  When all values are zero the result
    is all zeros.

    Args:
        values: Input numeric values (must be non-negative).

    Returns:
        Array with same shape and dtype ``float64``.
    """
    max_val = float(np.max(values))
    if np.isclose(max_val, 0.0):
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values / max_val, 0.0, 1.0).astype(np.float64, copy=False)


def compute_attention_rollout(
    attentions: tuple[torch.Tensor, ...],
    discard_ratio: float = 0.9,
    head_fusion: str = "mean",
) -> FloatArray:
    """Compute attention rollout and return patch-level saliency.

    Args:
        attentions: Tuple of per-layer attention tensors, each with shape
            ``(B, H, S, S)`` where ``S = 1 + num_patches``.
        discard_ratio: Fraction of smallest non-CLS attention entries to zero-out
            before residual addition. Must satisfy ``0 <= discard_ratio < 1``.
        head_fusion: Strategy to merge attention heads. One of
            ``{"mean", "max", "min"}``.

    Returns:
        Patch saliency map of shape ``(patch_grid_h, patch_grid_w)`` with values
        normalized to ``[0, 1]``.

    Raises:
        ValueError: If attention shapes or arguments are invalid.
    """
    _, seq_len, patch_side = _validate_attentions(attentions)
    fusion = head_fusion.strip().lower()
    if fusion not in _SUPPORTED_HEAD_FUSION:
        raise ValueError(
            f"Unsupported head_fusion '{head_fusion}'. Supported values: {sorted(_SUPPORTED_HEAD_FUSION)}."
        )

    rollout: torch.Tensor | None = None
    identity = torch.eye(
        seq_len, device=attentions[0].device, dtype=torch.float32
    ).unsqueeze(0)

    for layer_idx, layer_attention in enumerate(attentions):
        attention_float = layer_attention.detach().to(dtype=torch.float32)
        fused = _fuse_heads(attention_float, head_fusion=fusion)
        fused = _discard_low_attention(fused, discard_ratio=discard_ratio)

        augmented = fused + identity
        normalized = augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if rollout is None:
            rollout = normalized
        else:
            rollout = normalized @ rollout

        LOGGER.debug(
            "Computed rollout for layer %d with shape %s",
            layer_idx,
            tuple(normalized.shape),
        )

    if rollout is None:
        raise ValueError("No rollout computed from empty attentions.")

    cls_to_patch = rollout[:, 0, 1:]
    cls_to_patch_mean = cls_to_patch.mean(dim=0)
    patch_importance = (
        cls_to_patch_mean.reshape(patch_side, patch_side).detach().cpu().numpy()
    )
    patch_importance_float = patch_importance.astype(np.float64, copy=False)
    return _normalize_to_unit_interval(patch_importance_float)


class TemporalSaliencyMapper:
    """Map patch-level saliency back to time-series timesteps."""

    renderer_type: str
    window_size: int
    image_size: int
    patch_grid: tuple[int, int]
    _patch_height_pixels: float
    _patch_width_pixels: float

    def __init__(
        self,
        renderer_type: str,
        window_size: int,
        image_size: int = 224,
        patch_grid: tuple[int, int] = (14, 14),
    ) -> None:
        """Initialize temporal saliency mapper.

        Args:
            renderer_type: Renderer used to create model input image.
            window_size: Number of timesteps in the original input window.
            image_size: Square rendered image size in pixels.
            patch_grid: Patch grid shape as ``(height, width)``.

        Raises:
            ValueError: If configuration values are invalid.
        """
        normalized_renderer = renderer_type.strip().lower()
        if normalized_renderer not in _SUPPORTED_RENDERERS:
            raise ValueError(
                f"Unsupported renderer_type '{renderer_type}'. Supported values: {sorted(_SUPPORTED_RENDERERS)}."
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        if image_size <= 1:
            raise ValueError(f"image_size must be > 1, got {image_size}.")
        if len(patch_grid) != 2:
            raise ValueError(
                f"patch_grid must contain two integers (height, width), got {patch_grid}."
            )

        patch_h, patch_w = patch_grid
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"patch_grid dimensions must be positive, got {patch_grid}."
            )

        self.renderer_type = normalized_renderer
        self.window_size = int(window_size)
        self.image_size = int(image_size)
        self.patch_grid = (int(patch_h), int(patch_w))
        self._patch_height_pixels = self.image_size / float(self.patch_grid[0])
        self._patch_width_pixels = self.image_size / float(self.patch_grid[1])

    def map_to_timesteps(self, patch_importance: FloatArray) -> FloatArray:
        """Map patch saliency map to timestep saliency.

        Args:
            patch_importance: Patch importance map of shape ``patch_grid``.

        Returns:
            Timestep saliency of shape ``(window_size,)`` normalized to ``[0, 1]``.

        Raises:
            ValueError: If patch saliency input is invalid.
        """
        patch_values = np.asarray(patch_importance, dtype=np.float64)
        if patch_values.shape != self.patch_grid:
            raise ValueError(
                f"patch_importance shape must match patch_grid, expected {self.patch_grid}, got {patch_values.shape}."
            )
        if not np.isfinite(patch_values).all():
            raise ValueError("patch_importance contains non-finite values.")
        if np.any(patch_values < 0.0):
            raise ValueError("patch_importance must be non-negative.")

        if np.allclose(patch_values, 0.0):
            return np.zeros((self.window_size,), dtype=np.float64)

        if self.renderer_type == "line_plot":
            timestep_importance = self._map_line_plot(patch_values)
        else:
            timestep_importance = self._map_pairwise_renderer(patch_values)
        return _normalize_to_unit_interval(timestep_importance)

    def _map_line_plot(self, patch_importance: FloatArray) -> FloatArray:
        """Map line-plot patch saliency to timestep saliency.

        For line plots the x-axis represents time.  We iterate over each
        timestep and compute which fractional pixel position it corresponds
        to, then find the enclosing patch column(s) and accumulate their
        importance.  This guarantees every timestep receives a value.

        Args:
            patch_importance: Patch saliency map with shape ``patch_grid``.

        Returns:
            Unnormalized timestep saliency with shape ``(window_size,)``.
        """
        timestep_importance = np.zeros((self.window_size,), dtype=np.float64)
        pixels_per_patch = self._patch_width_pixels
        num_patch_cols = self.patch_grid[1]

        # Pre-compute column importance (sum over all rows for each column)
        col_importance = np.array(
            [float(np.sum(patch_importance[:, c])) for c in range(num_patch_cols)],
            dtype=np.float64,
        )

        for ts in range(self.window_size):
            # Map timestep to fractional pixel x-coordinate
            pixel_x = ts * (self.image_size - 1) / (self.window_size - 1) if self.window_size > 1 else 0.0
            # Determine which patch column this pixel falls in
            patch_col = int(pixel_x / pixels_per_patch)
            patch_col = max(0, min(patch_col, num_patch_cols - 1))
            timestep_importance[ts] = col_importance[patch_col]

        return timestep_importance

    def _map_pairwise_renderer(self, patch_importance: FloatArray) -> FloatArray:
        """Map pairwise-renderer patch saliency to timestep saliency.

        For pairwise renderers (GAF, recurrence plot) both the row and column
        axes represent time.  We iterate over all (ts_i, ts_j) pairs and find
        which patch cell covers that pair, then accumulate importance to both
        ts_i and ts_j.  To avoid O(window_size^2) cost, we precompute a
        timestep→patch mapping vector and use vectorized outer accumulation.

        Args:
            patch_importance: Patch saliency map with shape ``patch_grid``.

        Returns:
            Unnormalized timestep saliency with shape ``(window_size,)``.
        """
        timestep_importance = np.zeros((self.window_size,), dtype=np.float64)
        pixels_per_patch_h = self._patch_height_pixels
        pixels_per_patch_w = self._patch_width_pixels
        num_patch_rows = self.patch_grid[0]
        num_patch_cols = self.patch_grid[1]

        # Map each timestep to its patch row and patch col index
        # ts -> pixel -> patch_index
        ts_to_patch_row = np.zeros((self.window_size,), dtype=np.int64)
        ts_to_patch_col = np.zeros((self.window_size,), dtype=np.int64)
        for ts in range(self.window_size):
            pixel = ts * (self.image_size - 1) / (self.window_size - 1) if self.window_size > 1 else 0.0
            pr = int(pixel / pixels_per_patch_h)
            pc = int(pixel / pixels_per_patch_w)
            ts_to_patch_row[ts] = max(0, min(pr, num_patch_rows - 1))
            ts_to_patch_col[ts] = max(0, min(pc, num_patch_cols - 1))

        # For each timestep pair (i, j), contribution = patch_importance[row_i, col_j]
        # We accumulate to both timestep i and timestep j.
        # Vectorized: for each ts_i, sum over all ts_j
        for ts_i in range(self.window_size):
            row_i = ts_to_patch_row[ts_i]
            # importance of row_i interacting with each column patch
            col_importances = patch_importance[row_i, ts_to_patch_col]  # shape (W,)
            timestep_importance[ts_i] += float(np.sum(col_importances))
            timestep_importance += col_importances  # ts_j also gets credit

        return timestep_importance
