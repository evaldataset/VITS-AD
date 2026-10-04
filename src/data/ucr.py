from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt


_UCR_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_UCR_Anomaly_(?P<description>.+)_(?P<total>\d+)_(?P<start>\d+)_(?P<end>\d+)\.txt$"
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _parse_ucr_filename(file_path: Path) -> tuple[int, int, int]:
    """Parse UCR anomaly metadata from filename.

    Args:
        file_path: UCR series file path.

    Returns:
        Tuple of ``(total_length, anomaly_start, anomaly_end)``.

    Raises:
        ValueError: If filename does not match expected UCR naming convention.
    """
    match = _UCR_FILENAME_PATTERN.match(file_path.name)
    if match is None:
        raise ValueError(
            f"UCR filename does not match expected format '{{ID}}_UCR_Anomaly_{{description}}_{{total_length}}_{{anomaly_start}}_{{anomaly_end}}.txt': {file_path.name}"
        )

    total_length = int(match.group("total"))
    anomaly_start = int(match.group("start"))
    anomaly_end = int(match.group("end"))
    return total_length, anomaly_start, anomaly_end


def load_ucr_series(file_path: Path) -> tuple[FloatArray, IntArray, int, int]:
    """Load one UCR anomaly series and its timestep labels.

    Args:
        file_path: Path to one UCR text file.

    Returns:
        Tuple ``(data, labels, anomaly_start, anomaly_end)`` where:
        - ``data`` has shape ``(T, 1)`` with dtype ``np.float64``.
        - ``labels`` has shape ``(T,)`` with dtype ``np.int64``.
        - ``anomaly_start`` and ``anomaly_end`` are inclusive indices.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If file contents or anomaly indices are invalid.
    """
    total_length, anomaly_start, anomaly_end = _parse_ucr_filename(file_path)

    values: list[float] = []
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                tokens = stripped.replace(",", " ").split()
                for token in tokens:
                    try:
                        values.append(float(token))
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-numeric value in {file_path} at line {line_number}: '{token}'."
                        ) from exc
    except OSError as exc:
        raise OSError(f"Failed to read UCR file: {file_path}") from exc

    if not values:
        raise ValueError(f"UCR file is empty: {file_path}")

    data = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    length = int(data.shape[0])
    _ = total_length

    if not (0 <= anomaly_start < length):
        raise ValueError(
            f"anomaly_start={anomaly_start} is out of bounds for series length {length} in {file_path}."
        )
    if not (0 <= anomaly_end < length):
        raise ValueError(
            f"anomaly_end={anomaly_end} is out of bounds for series length {length} in {file_path}."
        )
    if anomaly_end < anomaly_start:
        raise ValueError(
            f"anomaly_end ({anomaly_end}) must be >= anomaly_start ({anomaly_start}) in {file_path}."
        )

    labels = np.zeros((length,), dtype=np.int64)
    labels[anomaly_start : anomaly_end + 1] = 1
    return data, labels, anomaly_start, anomaly_end


def list_ucr_files(ucr_dir: Path) -> list[Path]:
    """List UCR anomaly series files under a UCR root directory.

    Args:
        ucr_dir: Root directory that contains extracted UCR files.

    Returns:
        Sorted list of UCR text file paths.

    Raises:
        ValueError: If ``ucr_dir`` is invalid or no files are found.
    """
    if not ucr_dir.exists():
        raise ValueError(f"UCR directory does not exist: {ucr_dir}")
    if not ucr_dir.is_dir():
        raise ValueError(f"UCR path must be a directory: {ucr_dir}")

    files = sorted(ucr_dir.rglob("*_UCR_Anomaly_*.txt"))
    if not files:
        raise ValueError(f"No UCR anomaly files found under: {ucr_dir}")
    return files
