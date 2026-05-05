"""PatchTraj scoring utilities for training and detection."""

from __future__ import annotations

# pyright: reportMissingImports=false

import logging

import numpy as np
import numpy.typing as npt
import torch


LOGGER = logging.getLogger(__name__)


def _validate_patchtraj_inputs(
    predicted_tokens: torch.Tensor,
    actual_tokens: torch.Tensor,
    pi: npt.NDArray[np.int64],
    valid_mask: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Validate PatchTraj scorer inputs and return normalized numpy masks.

    Args:
        predicted_tokens: Predicted patch tokens with shape ``(B, N, D)``.
        actual_tokens: Actual patch tokens with shape ``(B, N, D)``.
        pi: Correspondence map with shape ``(N,)``.
        valid_mask: Boolean validity mask with shape ``(N,)``.

    Returns:
        Tuple of ``(pi_int64, valid_mask_bool)``.

    Raises:
        ValueError: If any input is invalid or shapes are incompatible.
    """
    if predicted_tokens.ndim != 3:
        raise ValueError(
            f"predicted_tokens must have shape (B, N, D), got {tuple(predicted_tokens.shape)}."
        )
    if actual_tokens.ndim != 3:
        raise ValueError(
            f"actual_tokens must have shape (B, N, D), got {tuple(actual_tokens.shape)}."
        )
    if predicted_tokens.shape != actual_tokens.shape:
        raise ValueError(
            f"predicted_tokens and actual_tokens must have identical shapes, got {tuple(predicted_tokens.shape)} and {tuple(actual_tokens.shape)}."
        )

    batch_size, num_patches, _ = predicted_tokens.shape
    if batch_size == 0:
        raise ValueError("predicted_tokens and actual_tokens must have B > 0.")
    if num_patches == 0:
        raise ValueError("predicted_tokens and actual_tokens must have N > 0.")

    if pi.ndim != 1:
        raise ValueError(f"pi must be 1D, got ndim={pi.ndim}.")
    if pi.shape[0] != num_patches:
        raise ValueError(
            f"pi length must match token dimension N, got len(pi)={pi.shape[0]} and N={num_patches}."
        )

    if valid_mask.ndim != 1:
        raise ValueError(f"valid_mask must be 1D, got ndim={valid_mask.ndim}.")
    if valid_mask.shape[0] != num_patches:
        raise ValueError(
            f"valid_mask length must match token dimension N, got len(valid_mask)={valid_mask.shape[0]} and N={num_patches}."
        )

    pi_int64 = pi.astype(np.int64, copy=False)
    valid_mask_bool = valid_mask.astype(bool, copy=False)
    valid_indices = np.flatnonzero(valid_mask_bool)
    if valid_indices.size == 0:
        raise ValueError("valid_mask contains no valid correspondences.")

    target_indices = pi_int64[valid_indices]
    if np.any(target_indices < 0):
        raise ValueError("pi contains negative indices for valid correspondences.")
    if np.any(target_indices >= num_patches):
        raise ValueError(
            f"pi contains out-of-range indices for valid correspondences: expected < {num_patches}."
        )

    return pi_int64, valid_mask_bool


def compute_patchtraj_score(
    predicted_tokens: torch.Tensor,
    actual_tokens: torch.Tensor,
    pi: npt.NDArray[np.int64],
    valid_mask: npt.NDArray[np.bool_],
) -> torch.Tensor:
    """Compute per-window PatchTraj anomaly scores.

    The score is the mean squared error between predicted patch tokens and
    actual patch tokens, aligned using the correspondence map ``pi``.

    Score formula:
        s_t = (1/N_valid) * Σ_{i∈valid} ||P_{t+Δ,π(i)} - P̂_{t+Δ,i}||²

    Args:
        predicted_tokens: Predicted next-window tokens of shape ``(B, N, D)``.
        actual_tokens: Actual next-window tokens of shape ``(B, N, D)``.
        pi: Correspondence map of shape ``(N,)`` where ``pi[i]`` is the target index.
        valid_mask: Boolean validity mask of shape ``(N,)``.

    Returns:
        Anomaly scores of shape ``(B,)``.

    Raises:
        ValueError: If shapes are incompatible.
    """
    pi_int64, valid_mask_bool = _validate_patchtraj_inputs(
        predicted_tokens=predicted_tokens,
        actual_tokens=actual_tokens,
        pi=pi,
        valid_mask=valid_mask,
    )

    valid_indices_np = np.flatnonzero(valid_mask_bool)
    target_indices_np = pi_int64[valid_indices_np]

    device = predicted_tokens.device
    valid_indices = torch.as_tensor(valid_indices_np, device=device, dtype=torch.long)
    target_indices = torch.as_tensor(target_indices_np, device=device, dtype=torch.long)

    actual_aligned = actual_tokens.index_select(dim=1, index=target_indices)
    pred_valid = predicted_tokens.index_select(dim=1, index=valid_indices)
    residuals = (actual_aligned - pred_valid).pow(2).sum(dim=-1)
    scores = residuals.mean(dim=-1)
    return scores


def compute_patchtraj_score_extended(
    predicted_tokens: torch.Tensor,
    actual_tokens: torch.Tensor,
    pi: npt.NDArray[np.int64],
    valid_mask: npt.NDArray[np.bool_],
) -> dict[str, torch.Tensor]:
    """Compute extended PatchTraj anomaly scores with multiple statistics.

    Returns mean, max, std, and p95 of per-patch residuals. These richer
    statistics often improve detection when combined via ensemble.

    Args:
        predicted_tokens: Predicted tokens of shape ``(B, N, D)``.
        actual_tokens: Actual tokens of shape ``(B, N, D)``.
        pi: Correspondence map of shape ``(N,)``.
        valid_mask: Boolean validity mask of shape ``(N,)``.

    Returns:
        Dict mapping statistic name to score tensor of shape ``(B,)``.
        Keys: 'mean', 'max', 'std', 'p95'.

    Raises:
        ValueError: If shapes are incompatible.
    """
    pi_int64, valid_mask_bool = _validate_patchtraj_inputs(
        predicted_tokens=predicted_tokens,
        actual_tokens=actual_tokens,
        pi=pi,
        valid_mask=valid_mask,
    )

    valid_indices_np = np.flatnonzero(valid_mask_bool)
    target_indices_np = pi_int64[valid_indices_np]

    device = predicted_tokens.device
    valid_indices = torch.as_tensor(valid_indices_np, device=device, dtype=torch.long)
    target_indices = torch.as_tensor(target_indices_np, device=device, dtype=torch.long)

    actual_aligned = actual_tokens.index_select(dim=1, index=target_indices)
    pred_valid = predicted_tokens.index_select(dim=1, index=valid_indices)
    residuals = (actual_aligned - pred_valid).pow(2).sum(dim=-1)  # (B, N_valid)

    score_mean = residuals.mean(dim=-1)
    score_max = residuals.max(dim=-1).values
    score_std = residuals.std(dim=-1)

    # Percentile-95: approximate via sorted selection
    n_valid = residuals.shape[1]
    p95_idx = min(int(n_valid * 0.95), n_valid - 1)
    sorted_res, _ = residuals.sort(dim=-1)
    score_p95 = sorted_res[:, p95_idx]

    return {
        "mean": score_mean,
        "max": score_max,
        "std": score_std,
        "p95": score_p95,
    }


def compute_patchtraj_residuals(
    predicted_tokens: torch.Tensor,
    actual_tokens: torch.Tensor,
    pi: npt.NDArray[np.int64],
    valid_mask: npt.NDArray[np.bool_],
) -> torch.Tensor:
    """Compute per-patch residuals for PatchTraj.

    Returns squared L2 residuals per valid patch, without averaging.

    Args:
        predicted_tokens: Predicted next-window tokens of shape ``(B, N, D)``.
        actual_tokens: Actual next-window tokens of shape ``(B, N, D)``.
        pi: Correspondence map of shape ``(N,)``.
        valid_mask: Boolean validity mask of shape ``(N,)``.

    Returns:
        Residual tensor with shape ``(B, N_valid)``.

    Raises:
        ValueError: If shapes are incompatible.
    """
    pi_int64, valid_mask_bool = _validate_patchtraj_inputs(
        predicted_tokens=predicted_tokens,
        actual_tokens=actual_tokens,
        pi=pi,
        valid_mask=valid_mask,
    )

    valid_indices_np = np.flatnonzero(valid_mask_bool)
    target_indices_np = pi_int64[valid_indices_np]

    device = predicted_tokens.device
    valid_indices = torch.as_tensor(valid_indices_np, device=device, dtype=torch.long)
    target_indices = torch.as_tensor(target_indices_np, device=device, dtype=torch.long)

    actual_aligned = actual_tokens.index_select(dim=1, index=target_indices)
    pred_valid = predicted_tokens.index_select(dim=1, index=valid_indices)
    residuals = (actual_aligned - pred_valid).pow(2).sum(dim=-1)
    return residuals


def compute_patchtraj_residuals_soft(
    predicted_tokens: torch.Tensor,
    actual_tokens: torch.Tensor,
    soft_pi: torch.Tensor,
) -> torch.Tensor:
    """Compute per-patch residuals with a soft OT correspondence matrix.

    Args:
        predicted_tokens: Predicted next-window tokens of shape ``(B, N, D)``.
        actual_tokens: Actual next-window tokens of shape ``(B, N, D)``.
        soft_pi: Soft correspondence matrix of shape ``(N, N)``.

    Returns:
        Residual tensor with shape ``(B, N)``.

    Raises:
        ValueError: If shapes are incompatible.
    """
    if predicted_tokens.ndim != 3 or actual_tokens.ndim != 3:
        raise ValueError(
            "predicted_tokens and actual_tokens must have shape (B, N, D), got "
            f"{tuple(predicted_tokens.shape)} and {tuple(actual_tokens.shape)}."
        )
    if predicted_tokens.shape != actual_tokens.shape:
        raise ValueError(
            f"predicted_tokens and actual_tokens must have identical shapes, got {tuple(predicted_tokens.shape)} and {tuple(actual_tokens.shape)}."
        )
    if soft_pi.ndim != 2:
        raise ValueError(f"soft_pi must be 2D with shape (N, N), got {soft_pi.ndim}D.")

    _, num_patches, _ = predicted_tokens.shape
    if soft_pi.shape != (num_patches, num_patches):
        raise ValueError(
            "soft_pi shape must match token dimension N, got "
            f"{tuple(soft_pi.shape)} and N={num_patches}."
        )
    if not torch.isfinite(soft_pi).all().item():
        raise ValueError("soft_pi contains non-finite values.")

    soft_pi_device = soft_pi.to(device=predicted_tokens.device, dtype=predicted_tokens.dtype)
    actual_aligned = torch.einsum("ij,bjd->bid", soft_pi_device, actual_tokens)
    residuals = (actual_aligned - predicted_tokens).pow(2).sum(dim=-1)
    return residuals


def trimmed_huber_loss(
    residuals: torch.Tensor,
    delta: float = 1.0,
    trim_ratio: float = 0.10,
) -> torch.Tensor:
    """Compute trimmed Huber loss for robust PatchTraj training.

    Steps:
      1. Convert squared residuals to L2 errors.
      2. Apply element-wise Huber loss.
      3. Remove the top ``trim_ratio`` fraction of largest losses.
      4. Return the mean over remaining losses.

    Args:
        residuals: Per-patch squared residuals of shape ``(B, N_valid)``.
        delta: Huber transition threshold. Must be positive.
        trim_ratio: Fraction to trim from highest losses. Must satisfy ``0 <= r < 1``.

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If arguments are invalid.
    """
    if residuals.ndim != 2:
        raise ValueError(
            f"residuals must have shape (B, N_valid), got {tuple(residuals.shape)}."
        )
    if residuals.numel() == 0:
        raise ValueError("residuals must contain at least one value.")
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}.")
    if trim_ratio < 0.0 or trim_ratio >= 1.0:
        raise ValueError(
            f"trim_ratio must satisfy 0 <= trim_ratio < 1, got {trim_ratio}."
        )
    if not torch.isfinite(residuals).all().item():
        raise ValueError("residuals contains non-finite values.")
    if (residuals < 0).any().item():
        raise ValueError("residuals must be non-negative squared errors.")

    errors = residuals.sqrt()
    huber = torch.where(
        errors <= delta,
        0.5 * errors.pow(2),
        delta * (errors - (0.5 * delta)),
    )

    flat_losses = huber.reshape(-1)
    if trim_ratio == 0.0:
        return flat_losses.mean()

    num_losses = flat_losses.numel()
    num_trim = int(num_losses * trim_ratio)
    num_keep = num_losses - num_trim
    if num_keep <= 0:
        raise ValueError(
            "trim_ratio removes all elements; decrease trim_ratio or provide more data."
        )

    sorted_losses, _ = torch.sort(flat_losses)
    kept_losses = sorted_losses[:num_keep]
    return kept_losses.mean()


def normalize_scores(
    scores: npt.NDArray[np.float64],
    method: str = "minmax",
) -> npt.NDArray[np.float64]:
    """Normalize anomaly scores to the ``[0, 1]`` range.

    Args:
        scores: Raw anomaly scores of shape ``(T,)``.
        method: One of ``"minmax"`` or ``"zscore"``.

    Returns:
        Normalized scores with shape ``(T,)``.

    Raises:
        ValueError: If inputs are invalid.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got ndim={scores.ndim}.")
    if scores.size == 0:
        raise ValueError("scores must contain at least one value.")
    if not np.isfinite(scores).all():
        raise ValueError("scores contains non-finite values.")

    method_normalized = method.strip().lower()
    scores_float = scores.astype(np.float64, copy=False)

    if method_normalized == "minmax":
        min_val = float(np.min(scores_float))
        max_val = float(np.max(scores_float))
        scale = max_val - min_val
        if scale == 0.0:
            return np.zeros_like(scores_float)
        return (scores_float - min_val) / scale

    if method_normalized == "zscore":
        mean_val = float(np.mean(scores_float))
        std_val = float(np.std(scores_float))
        if std_val == 0.0:
            return np.zeros_like(scores_float)

        zscores = (scores_float - mean_val) / std_val
        z_min = float(np.min(zscores))
        z_max = float(np.max(zscores))
        z_scale = z_max - z_min
        if z_scale == 0.0:
            return np.zeros_like(scores_float)
        return (zscores - z_min) / z_scale

    raise ValueError(
        f"Unsupported normalization method '{method}'. Expected one of: ['minmax', 'zscore']."
    )


def smooth_scores(
    scores: np.ndarray,
    window_size: int = 5,
    method: str = "mean",
) -> np.ndarray:
    """Apply sliding window smoothing to anomaly scores.

    Smoothing reduces noise in per-window scores and improves
    AUC-ROC by reducing false positives from isolated spikes.

    Args:
        scores: Raw anomaly scores of shape ``(T,)``.
        window_size: Smoothing window size. Must be positive odd integer.
        method: One of ``"mean"`` (moving average) or ``"median"`` (moving median).

    Returns:
        Smoothed scores with shape ``(T,)``.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(scores, np.ndarray):
        raise ValueError(f"scores must be a numpy.ndarray, got {type(scores)!r}.")
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got ndim={scores.ndim}.")
    if scores.size == 0:
        raise ValueError("scores must contain at least one value.")
    if not np.isfinite(scores).all():
        raise ValueError("scores contains non-finite values.")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}.")

    method_lower = method.strip().lower()
    if method_lower not in ("mean", "median"):
        raise ValueError(
            f"Unsupported smoothing method '{method}'. "
            "Expected one of: ['mean', 'median']."
        )

    if window_size >= scores.size:
        if method_lower == "mean":
            return np.full_like(scores, float(np.mean(scores)))
        return np.full_like(scores, float(np.median(scores)))

    scores_float = scores.astype(np.float64, copy=True)
    pad = window_size // 2
    padded = np.pad(scores_float, (pad, pad), mode="edge")

    smoothed = np.empty_like(scores_float)
    for i in range(scores_float.size):
        window = padded[i : i + window_size]
        if method_lower == "mean":
            smoothed[i] = float(np.mean(window))
        else:
            smoothed[i] = float(np.median(window))

    return smoothed
