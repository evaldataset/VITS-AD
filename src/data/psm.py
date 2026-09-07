from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from src.data.base import create_sliding_windows, normalize_data


logger = logging.getLogger(__name__)


def _parse_float(
    cell_value: str, file_path: Path, row_index: int, col_index: int
) -> float:
    if cell_value == "":
        return np.nan
    try:
        return float(cell_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value '{cell_value}' in {file_path} at row {row_index}, "
            f"column {col_index}."
        ) from exc


def _load_psm_features(file_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    try:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"CSV file is empty: {file_path}")
            if len(header) < 2:
                raise ValueError(
                    f"PSM feature file must contain timestamp + features: {file_path}"
                )

            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"Row length mismatch in {file_path} at line {line_number}: "
                        f"expected {len(header)} columns, got {len(row)}."
                    )

                feature_cells = row[1:]
                parsed_row = [
                    _parse_float(cell, file_path, line_number, col_idx + 2)
                    for col_idx, cell in enumerate(feature_cells)
                ]
                rows.append(parsed_row)
    except OSError as exc:
        raise OSError(f"Failed to read PSM file: {file_path}") from exc

    if not rows:
        raise ValueError(f"No feature rows found in PSM file: {file_path}")

    return np.asarray(rows, dtype=np.float64)


def _load_psm_labels(file_path: Path) -> np.ndarray:
    labels: list[int] = []
    try:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"CSV file is empty: {file_path}")

            try:
                label_col_idx = header.index("label")
            except ValueError as exc:
                raise ValueError(
                    f"PSM label file must contain a 'label' column: {file_path}"
                ) from exc

            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                if len(row) <= label_col_idx:
                    raise ValueError(
                        f"Missing label value in {file_path} at line {line_number}."
                    )

                value = _parse_float(
                    row[label_col_idx],
                    file_path,
                    line_number,
                    label_col_idx + 1,
                )
                labels.append(int(value > 0))
    except OSError as exc:
        raise OSError(f"Failed to read PSM label file: {file_path}") from exc

    if not labels:
        raise ValueError(f"No labels found in PSM label file: {file_path}")

    return np.asarray(labels, dtype=np.int64)


class PSMDataset:
    """PSM (Pooled Server Metrics) loader.

    Args:
        raw_dir: Path to raw PSM data directory.
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
            raise ValueError(f"PSM raw_dir does not exist: {raw_path}")
        if not raw_path.is_dir():
            raise ValueError(f"PSM raw_dir must be a directory: {raw_path}")

        train_file = raw_path / "train.csv"
        test_file = raw_path / "test.csv"
        label_file = raw_path / "test_label.csv"

        for file_path in (train_file, test_file, label_file):
            if not file_path.exists():
                raise ValueError(f"Required PSM file does not exist: {file_path}")
            if not file_path.is_file():
                raise ValueError(f"Required PSM path is not a file: {file_path}")

        train_data = _load_psm_features(train_file)
        test_data = _load_psm_features(test_file)
        test_labels_timestep = _load_psm_labels(label_file)

        if train_data.ndim != 2 or test_data.ndim != 2:
            raise ValueError("PSM train/test data must be 2D arrays.")
        if train_data.shape[0] == 0 or test_data.shape[0] == 0:
            raise ValueError("PSM train/test data must be non-empty.")
        if train_data.shape[1] != test_data.shape[1]:
            raise ValueError(
                "PSM train and test feature dimensions must match, "
                f"got {train_data.shape[1]} and {test_data.shape[1]}."
            )
        if test_data.shape[0] != test_labels_timestep.shape[0]:
            raise ValueError(
                "PSM test data and test labels length mismatch, "
                f"got {test_data.shape[0]} and {test_labels_timestep.shape[0]}."
            )

        if normalize:
            logger.info("Normalizing PSM using method '%s'.", norm_method)
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
