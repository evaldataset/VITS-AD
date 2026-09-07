from __future__ import annotations

# pyright: reportMissingImports=false

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.data.base import create_sliding_windows, normalize_data


logger = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _load_smap_matrix(file_path: Path) -> FloatArray:
    """Load an SMAP feature matrix from a .npy file.

    Args:
        file_path: Path to the .npy feature file.

    Returns:
        Array of shape (T, D) with dtype float64.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the loaded array is invalid.
    """
    try:
        matrix = np.load(file_path)
    except OSError as exc:
        raise OSError(f"Failed to read SMAP file: {file_path}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid SMAP .npy file: {file_path}") from exc

    if matrix.ndim != 2:
        raise ValueError(
            f"SMAP matrix must be 2D, got shape {matrix.shape} in {file_path}."
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"SMAP matrix must be non-empty: {file_path}")

    return np.asarray(matrix, dtype=np.float64)


def _load_smap_labels(file_path: Path) -> IntArray:
    """Load binary timestep labels for SMAP from a .npy file.

    Args:
        file_path: Path to the .npy label file.

    Returns:
        Binary label array of shape (T,) and dtype int64.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the loaded array is invalid.
    """
    try:
        labels = np.load(file_path)
    except OSError as exc:
        raise OSError(f"Failed to read SMAP label file: {file_path}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid SMAP label .npy file: {file_path}") from exc

    if labels.ndim != 1:
        raise ValueError(
            f"SMAP labels must be 1D, got shape {labels.shape} in {file_path}."
        )
    if labels.shape[0] == 0:
        raise ValueError(f"SMAP labels must be non-empty: {file_path}")

    return (labels > 0).astype(np.int64)


class SMAPDataset:
    """SMAP (Soil Moisture Active Passive) loader.

    Args:
        raw_dir: Path to raw SMAP data directory.
        window_size: Sliding window size.
        stride: Sliding window stride.
        normalize: Whether to normalize data.
        norm_method: Normalization method ('standard' or 'minmax').
    """

    def __init__(
        self,
        raw_dir: str | Path,
        window_size: int,
        stride: int = 1,
        normalize: bool = True,
        norm_method: str = "standard",
    ) -> None:
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            raise ValueError(f"SMAP raw_dir does not exist: {raw_path}")
        if not raw_path.is_dir():
            raise ValueError(f"SMAP raw_dir must be a directory: {raw_path}")

        train_file = raw_path / "SMAP_train.npy"
        test_file = raw_path / "SMAP_test.npy"
        label_file = raw_path / "SMAP_test_label.npy"

        for file_path in (train_file, test_file, label_file):
            if not file_path.exists():
                raise ValueError(f"Required SMAP file does not exist: {file_path}")
            if not file_path.is_file():
                raise ValueError(f"Required SMAP path is not a file: {file_path}")

        train_data = _load_smap_matrix(train_file)
        test_data = _load_smap_matrix(test_file)
        test_labels_timestep = _load_smap_labels(label_file)

        if train_data.ndim != 2 or test_data.ndim != 2:
            raise ValueError("SMAP train/test data must be 2D arrays.")
        if train_data.shape[0] == 0 or test_data.shape[0] == 0:
            raise ValueError("SMAP train/test data must be non-empty.")
        if train_data.shape[1] != test_data.shape[1]:
            raise ValueError(
                "SMAP train and test feature dimensions must match, "
                f"got {train_data.shape[1]} and {test_data.shape[1]}."
            )
        if test_data.shape[0] != test_labels_timestep.shape[0]:
            raise ValueError(
                "SMAP test data and test labels length mismatch, "
                f"got {test_data.shape[0]} and {test_labels_timestep.shape[0]}."
            )

        if normalize:
            logger.info("Normalizing SMAP using method '%s'.", norm_method)
            train_data, test_data = normalize_data(
                train_data=train_data,
                test_data=test_data,
                method=norm_method,
            )

        train_timestep_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
        all_train_windows, all_train_window_labels = create_sliding_windows(
            data=train_data,
            labels=train_timestep_labels,
            window_size=window_size,
            stride=stride,
        )

        normal_window_mask = all_train_window_labels == 0
        self._train_windows = all_train_windows[normal_window_mask]

        self._test_windows, self._test_labels = create_sliding_windows(
            data=test_data,
            labels=test_labels_timestep,
            window_size=window_size,
            stride=stride,
        )

        if self._train_windows.shape[0] == 0:
            raise ValueError(
                "No normal training windows available after windowing. "
                "Consider changing window_size or stride."
            )

        self._num_features = int(train_data.shape[1])

    @property
    def train_windows(self) -> FloatArray:
        """Return normal-only training windows of shape (N_train, L, D)."""
        return self._train_windows

    @property
    def test_windows(self) -> FloatArray:
        """Return test windows of shape (N_test, L, D)."""
        return self._test_windows

    @property
    def test_labels(self) -> IntArray:
        """Return per-window binary test labels of shape (N_test,)."""
        return self._test_labels

    @property
    def num_features(self) -> int:
        """Return the number of features per timestep."""
        return self._num_features
