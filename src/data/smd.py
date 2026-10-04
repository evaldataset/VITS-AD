from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.data.base import create_sliding_windows, normalize_data


logger = logging.getLogger(__name__)


def _load_smd_matrix(file_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                tokens = stripped.replace(",", " ").split()
                try:
                    row = [float(token) if token != "" else np.nan for token in tokens]
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value in {file_path} at line {line_number}."
                    ) from exc
                rows.append(row)
    except OSError as exc:
        raise OSError(f"Failed to read SMD file: {file_path}") from exc

    if not rows:
        raise ValueError(f"SMD file is empty: {file_path}")

    num_columns = len(rows[0])
    if num_columns == 0:
        raise ValueError(f"SMD file has no columns: {file_path}")

    for row_index, row in enumerate(rows, start=1):
        if len(row) != num_columns:
            raise ValueError(
                f"Inconsistent number of columns in {file_path} at row {row_index}: "
                f"expected {num_columns}, got {len(row)}."
            )

    return np.asarray(rows, dtype=np.float64)


def _load_smd_labels(file_path: Path) -> np.ndarray:
    labels_matrix = _load_smd_matrix(file_path)
    if labels_matrix.shape[1] != 1:
        raise ValueError(
            f"SMD label file must have one column, got {labels_matrix.shape[1]} in {file_path}."
        )
    return (labels_matrix[:, 0] > 0).astype(np.int64)


class SMDDataset:
    """SMD (Server Machine Dataset) loader.

    Args:
        raw_dir: Path to raw SMD data directory.
        entity: Machine entity name (e.g., 'machine-1-1').
        window_size: Sliding window size.
        stride: Sliding window stride.
        normalize: Whether to normalize data.
        norm_method: Normalization method ('standard' or 'minmax').
    """

    def __init__(
        self,
        raw_dir: str | Path,
        entity: str,
        window_size: int,
        stride: int = 1,
        normalize: bool = True,
        norm_method: str = "standard",
    ) -> None:
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            raise ValueError(f"SMD raw_dir does not exist: {raw_path}")
        if not raw_path.is_dir():
            raise ValueError(f"SMD raw_dir must be a directory: {raw_path}")
        if not entity:
            raise ValueError("entity must be a non-empty string.")

        train_file = raw_path / "train" / f"{entity}.txt"
        test_file = raw_path / "test" / f"{entity}.txt"
        label_file = raw_path / "test_label" / f"{entity}.txt"

        for file_path in (train_file, test_file, label_file):
            if not file_path.exists():
                raise ValueError(f"Required SMD file does not exist: {file_path}")
            if not file_path.is_file():
                raise ValueError(f"Required SMD path is not a file: {file_path}")

        train_data = _load_smd_matrix(train_file)
        test_data = _load_smd_matrix(test_file)
        test_labels_timestep = _load_smd_labels(label_file)

        if train_data.ndim != 2 or test_data.ndim != 2:
            raise ValueError("SMD train/test data must be 2D arrays.")
        if train_data.shape[0] == 0 or test_data.shape[0] == 0:
            raise ValueError("SMD train/test data must be non-empty.")
        if train_data.shape[1] != test_data.shape[1]:
            raise ValueError(
                "SMD train and test feature dimensions must match, "
                f"got {train_data.shape[1]} and {test_data.shape[1]}."
            )
        if test_data.shape[0] != test_labels_timestep.shape[0]:
            raise ValueError(
                "SMD test data and test labels length mismatch, "
                f"got {test_data.shape[0]} and {test_labels_timestep.shape[0]}."
            )

        if normalize:
            logger.info(
                "Normalizing SMD entity %s using method '%s'.", entity, norm_method
            )
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
    def train_windows(self) -> np.ndarray:
        """Return normal-only training windows of shape (N_train, L, D)."""
        return self._train_windows

    @property
    def test_windows(self) -> np.ndarray:
        """Return test windows of shape (N_test, L, D)."""
        return self._test_windows

    @property
    def test_labels(self) -> np.ndarray:
        """Return per-window binary test labels of shape (N_test,)."""
        return self._test_labels

    @property
    def num_features(self) -> int:
        """Return the number of features per timestep."""
        return self._num_features
