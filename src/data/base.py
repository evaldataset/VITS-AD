from __future__ import annotations

import logging

import numpy as np


logger = logging.getLogger(__name__)


def _validate_data_and_labels(data: np.ndarray, labels: np.ndarray) -> None:
    if data.ndim != 2:
        raise ValueError(
            f"data must be a 2D array of shape (T, D), got shape {data.shape}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"labels must be a 1D array of shape (T,), got shape {labels.shape}."
        )
    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            "data and labels must have the same length along axis 0, "
            f"got {data.shape[0]} and {labels.shape[0]}."
        )
    if data.shape[0] == 0:
        raise ValueError("data and labels must be non-empty.")


def _forward_fill_nan_then_zero(data: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(data, dtype=np.float64).copy()
    if cleaned.size == 0:
        return cleaned

    nan_mask = np.isnan(cleaned)
    if not np.any(nan_mask):
        return cleaned

    logger.debug("Detected NaN values; applying forward-fill then zero-fill.")
    for feature_idx in range(cleaned.shape[1]):
        column = cleaned[:, feature_idx]
        for row_idx in range(1, column.shape[0]):
            if np.isnan(column[row_idx]) and not np.isnan(column[row_idx - 1]):
                column[row_idx] = column[row_idx - 1]
        column[np.isnan(column)] = 0.0
        cleaned[:, feature_idx] = column

    return cleaned


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from time series data.

    Args:
        data: Time series data of shape (T, D) where T=timesteps, D=features.
        labels: Binary anomaly labels of shape (T,). 1=anomaly, 0=normal.
        window_size: Length of each window.
        stride: Step size between consecutive windows.

    Returns:
        Tuple of (windows, window_labels) where:
            windows: np.ndarray of shape (N, window_size, D)
            window_labels: np.ndarray of shape (N,), 1 if ANY timestep in window is anomaly

    Raises:
        ValueError: If data length < window_size or shapes mismatch.
    """
    _validate_data_and_labels(data=data, labels=labels)

    if window_size <= 0:
        raise ValueError(f"window_size must be a positive integer, got {window_size}.")
    if stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride}.")
    if data.shape[0] < window_size:
        raise ValueError(
            f"data length ({data.shape[0]}) must be >= window_size ({window_size})."
        )

    cleaned_data = _forward_fill_nan_then_zero(data)
    cleaned_labels = _forward_fill_nan_then_zero(labels.reshape(-1, 1)).reshape(-1)

    starts = np.arange(
        0, cleaned_data.shape[0] - window_size + 1, stride, dtype=np.int64
    )
    num_windows = starts.shape[0]

    windows = np.empty(
        (num_windows, window_size, cleaned_data.shape[1]), dtype=cleaned_data.dtype
    )
    window_labels = np.empty((num_windows,), dtype=np.int64)

    for idx, start in enumerate(starts):
        end = int(start + window_size)
        windows[idx] = cleaned_data[start:end]
        window_labels[idx] = int(np.any(cleaned_labels[start:end] > 0))

    return windows, window_labels


def time_based_split(
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time series by time (NOT random). Train set uses only normal data.

    Args:
        data: Full time series (T, D).
        labels: Binary labels (T,).
        train_ratio: Fraction of data for training (from the beginning).

    Returns:
        (train_data, train_labels, test_data, test_labels)
        train_data contains ONLY normal timesteps from the first train_ratio portion.
        test_data is the remaining (1 - train_ratio) portion (all labels kept).

    Raises:
        ValueError: If train_ratio not in (0, 1).
    """
    _validate_data_and_labels(data=data, labels=labels)

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")

    cleaned_data = _forward_fill_nan_then_zero(data)
    cleaned_labels = _forward_fill_nan_then_zero(labels.reshape(-1, 1)).reshape(-1)

    split_idx = int(cleaned_data.shape[0] * train_ratio)
    if split_idx <= 0 or split_idx >= cleaned_data.shape[0]:
        raise ValueError(
            "train_ratio produces an empty train or test split. "
            f"Computed split index {split_idx} for total length {cleaned_data.shape[0]}."
        )

    initial_train_data = cleaned_data[:split_idx]
    initial_train_labels = cleaned_labels[:split_idx]
    normal_mask = initial_train_labels == 0

    train_data = initial_train_data[normal_mask]
    train_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
    test_data = cleaned_data[split_idx:]
    test_labels = (cleaned_labels[split_idx:] > 0).astype(np.int64)

    if train_data.shape[0] == 0:
        logger.warning(
            "No normal samples found in the training segment after time-based split."
        )

    return train_data, train_labels, test_data, test_labels


def normalize_data(
    train_data: np.ndarray,
    test_data: np.ndarray,
    method: str = "standard",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize using statistics from train_data only (no leakage).

    Args:
        train_data: (T_train, D)
        test_data: (T_test, D)
        method: "standard" (zero mean, unit var) or "minmax" (scale to [0, 1])

    Returns:
        (normalized_train, normalized_test)
    """
    if train_data.ndim != 2:
        raise ValueError(
            "train_data must be a 2D array of shape (T_train, D), "
            f"got shape {train_data.shape}."
        )
    if test_data.ndim != 2:
        raise ValueError(
            "test_data must be a 2D array of shape (T_test, D), "
            f"got shape {test_data.shape}."
        )
    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError(
            "train_data and test_data must have the same feature dimension, "
            f"got {train_data.shape[1]} and {test_data.shape[1]}."
        )
    if train_data.shape[0] == 0:
        raise ValueError("train_data must contain at least one row for normalization.")

    cleaned_train = _forward_fill_nan_then_zero(train_data)
    cleaned_test = _forward_fill_nan_then_zero(test_data)

    if method == "standard":
        mean = np.mean(cleaned_train, axis=0)
        std = np.std(cleaned_train, axis=0)
        std = np.where(std == 0.0, 1.0, std)
        normalized_train = (cleaned_train - mean) / std
        normalized_test = (cleaned_test - mean) / std
    elif method == "minmax":
        min_val = np.min(cleaned_train, axis=0)
        max_val = np.max(cleaned_train, axis=0)
        scale = max_val - min_val
        scale = np.where(scale == 0.0, 1.0, scale)
        normalized_train = (cleaned_train - min_val) / scale
        normalized_test = (cleaned_test - min_val) / scale
    else:
        raise ValueError(
            f"Unsupported normalization method '{method}'. "
            "Expected one of: ['standard', 'minmax']."
        )

    return normalized_train, normalized_test
