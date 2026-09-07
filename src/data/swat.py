from __future__ import annotations

# pyright: reportMissingImports=false

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data.base import create_sliding_windows, normalize_data


LOGGER = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# SWaT's public (iTrust) distribution ships under inconsistent filenames
# across releases (v0/v1) and either CSV or Excel format. Candidates are
# tried in priority order; extend this list if a release uses a different
# name. See paper/rebuttal_package/swat_runbook.md for placement details.
_TRAIN_FILENAME_CANDIDATES: tuple[str, ...] = (
    "SWaT_Dataset_Normal_v1.csv",
    "SWaT_Dataset_Normal_v0.csv",
    "SWaT_Dataset_Normal_v1.xlsx",
    "SWaT_Dataset_Normal_v0.xlsx",
    "normal.csv",
)
_TEST_FILENAME_CANDIDATES: tuple[str, ...] = (
    "SWaT_Dataset_Attack_v0.csv",
    "SWaT_Dataset_Attack_v1.csv",
    "SWaT_Dataset_Attack_v0.xlsx",
    "SWaT_Dataset_Attack_v1.xlsx",
    "attack.csv",
)

# Column-name matching is case-insensitive and whitespace-stripped since
# SWaT headers are notoriously space-padded (e.g. " Timestamp").
_TIMESTAMP_COLUMN_NAMES: frozenset[str] = frozenset({"timestamp"})
_LABEL_COLUMN_NAMES: frozenset[str] = frozenset({"normal/attack"})
_NORMAL_LABEL_VALUE: str = "NORMAL"


def _resolve_swat_file(raw_dir: Path, candidates: tuple[str, ...]) -> Path:
    """Resolve the first existing SWaT file among naming candidates.

    Args:
        raw_dir: Directory expected to contain a SWaT raw file.
        candidates: Candidate filenames to try, in priority order.

    Returns:
        Path to the first candidate that exists as a file.

    Raises:
        ValueError: If none of the candidate filenames exist in raw_dir.
    """
    for name in candidates:
        candidate_path = raw_dir / name
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path

    raise ValueError(
        f"No SWaT file found in {raw_dir}. Tried: {list(candidates)}. "
        "SWaT filenames vary by release; place the raw file under one of "
        "these names or extend the candidate list in src/data/swat.py."
    )


def _read_swat_table(file_path: Path) -> pd.DataFrame:
    """Read a raw SWaT table from CSV or Excel with whitespace-safe columns.

    Args:
        file_path: Path to a SWaT .csv, .xlsx, or .xls file.

    Returns:
        DataFrame with stripped column names.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the extension is unsupported or the table is empty.
    """
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            table = pd.read_csv(file_path)
        elif suffix in (".xlsx", ".xls"):
            table = pd.read_excel(file_path)
        else:
            raise ValueError(
                f"Unsupported SWaT file extension '{suffix}': {file_path}. "
                "Expected .csv, .xlsx, or .xls."
            )
    except OSError as exc:
        raise OSError(f"Failed to read SWaT file: {file_path}") from exc

    if table.shape[0] == 0 or table.shape[1] == 0:
        raise ValueError(f"SWaT file is empty: {file_path}")

    table.columns = [str(col).strip() for col in table.columns]
    return table


def _drop_non_feature_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Drop the timestamp and 'Normal/Attack' label columns if present.

    Args:
        table: SWaT table with stripped column names.

    Returns:
        DataFrame containing only sensor feature columns.
    """
    drop_columns = [
        col
        for col in table.columns
        if col.strip().lower() in _TIMESTAMP_COLUMN_NAMES | _LABEL_COLUMN_NAMES
    ]
    return table.drop(columns=drop_columns)


def _load_swat_matrix(file_path: Path) -> FloatArray:
    """Load a SWaT sensor feature matrix, dropping timestamp/label columns.

    Args:
        file_path: Path to a SWaT .csv or .xlsx file (train or test).

    Returns:
        Array of shape (T, D) with dtype float64.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the resulting matrix is empty or has no feature columns.
    """
    table = _read_swat_table(file_path)
    feature_table = _drop_non_feature_columns(table)

    if feature_table.shape[1] == 0:
        raise ValueError(
            "No sensor feature columns remain after dropping timestamp/label "
            f"columns in {file_path}. Columns found: {list(table.columns)}."
        )

    # SWaT cells are sometimes read as whitespace-padded strings; coerce
    # defensively and let invalid values become NaN (forward-filled by
    # base.create_sliding_windows downstream).
    numeric_table = feature_table.apply(
        lambda column: pd.to_numeric(
            column.astype(str).str.strip() if column.dtype == object else column,
            errors="coerce",
        )
    )

    matrix = numeric_table.to_numpy(dtype=np.float64)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"SWaT matrix must be non-empty: {file_path}")

    return matrix


def _load_swat_labels(file_path: Path) -> IntArray:
    """Load binary timestep labels from the SWaT 'Normal/Attack' column.

    Args:
        file_path: Path to a SWaT .csv or .xlsx file containing labels.

    Returns:
        Binary label array of shape (T,) and dtype int64 (1=attack, 0=normal).

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the label column is missing or empty.
    """
    table = _read_swat_table(file_path)
    label_columns = [
        col for col in table.columns if col.strip().lower() in _LABEL_COLUMN_NAMES
    ]
    if not label_columns:
        raise ValueError(
            f"SWaT file must contain a 'Normal/Attack' label column: {file_path}. "
            f"Columns found: {list(table.columns)}."
        )

    raw_labels = table[label_columns[0]].astype(str).str.strip().str.upper()
    labels = (raw_labels != _NORMAL_LABEL_VALUE).astype(np.int64).to_numpy()

    if labels.shape[0] == 0:
        raise ValueError(f"SWaT labels must be non-empty: {file_path}")

    return labels


class SWaTDataset:
    """SWaT (Secure Water Treatment) loader.

    SWaT's public distribution ships as two files: a normal-operation file
    (no attacks, used for training) and an attack file (used for testing,
    with a 'Normal/Attack' string label column). Both are typically CSV or
    Excel exports with a leading timestamp column and space-padded headers.

    Args:
        raw_dir: Path to raw SWaT data directory.
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
            raise ValueError(f"SWaT raw_dir does not exist: {raw_path}")
        if not raw_path.is_dir():
            raise ValueError(f"SWaT raw_dir must be a directory: {raw_path}")

        train_file = _resolve_swat_file(raw_path, _TRAIN_FILENAME_CANDIDATES)
        test_file = _resolve_swat_file(raw_path, _TEST_FILENAME_CANDIDATES)

        train_data = _load_swat_matrix(train_file)
        test_data = _load_swat_matrix(test_file)
        test_labels_timestep = _load_swat_labels(test_file)

        if train_data.ndim != 2 or test_data.ndim != 2:
            raise ValueError("SWaT train/test data must be 2D arrays.")
        if train_data.shape[0] == 0 or test_data.shape[0] == 0:
            raise ValueError("SWaT train/test data must be non-empty.")
        if train_data.shape[1] != test_data.shape[1]:
            raise ValueError(
                "SWaT train and test feature dimensions must match, "
                f"got {train_data.shape[1]} and {test_data.shape[1]}."
            )
        if test_data.shape[0] != test_labels_timestep.shape[0]:
            raise ValueError(
                "SWaT test data and test labels length mismatch, "
                f"got {test_data.shape[0]} and {test_labels_timestep.shape[0]}."
            )

        if normalize:
            LOGGER.info("Normalizing SWaT using method '%s'.", norm_method)
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
