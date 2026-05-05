"""Download MSL and SMAP datasets and organize .npy files.

Usage:
    python scripts/download_msl_smap.py [--output_dir data/raw]

The script downloads preprocessed NASA telemetry datasets in .npy format and
organizes them into:

    data/raw/msl/
    ├── MSL_train.npy
    ├── MSL_test.npy
    └── MSL_test_label.npy

    data/raw/smap/
    ├── SMAP_train.npy
    ├── SMAP_test.npy
    └── SMAP_test_label.npy
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


DATASET_URLS: dict[str, str] = {
    "MSL": "https://drive.usercontent.google.com/download?id=14STjpszyi6D0B7BUHZ1L4GLUkhhPXE0G&confirm=t",
    "SMAP": "https://drive.usercontent.google.com/download?id=1kxiTMOouw1p-yJMkb_Q_CGMjakVNtg3X&confirm=t",
}

EXPECTED_FILES: dict[str, tuple[str, str, str]] = {
    "MSL": ("MSL_train.npy", "MSL_test.npy", "MSL_test_label.npy"),
    "SMAP": ("SMAP_train.npy", "SMAP_test.npy", "SMAP_test_label.npy"),
}

EXPECTED_FEATURES: dict[str, int] = {
    "MSL": 55,
    "SMAP": 25,
}


def _download_file(url: str, destination: Path) -> None:
    """Download a file from URL to local path.

    Args:
        url: Source URL.
        destination: Destination file path.

    Raises:
        RuntimeError: If download fails.
    """
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download from {url}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write downloaded file to {destination}: {exc}"
        ) from exc


def _find_single_file(root: Path, file_name: str) -> Path:
    """Find a single file recursively by name.

    Args:
        root: Root directory to search.
        file_name: Target file name.

    Returns:
        Absolute path to found file.

    Raises:
        RuntimeError: If file is missing or ambiguous.
    """
    matches = [path for path in root.rglob(file_name) if path.is_file()]
    if len(matches) == 0:
        raise RuntimeError(
            f"Required file '{file_name}' not found in extracted archive {root}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files named '{file_name}' found in extracted archive {root}: {matches}"
        )
    return matches[0]


def _validate_dataset_files(dataset_name: str, dataset_dir: Path) -> None:
    """Validate downloaded dataset files.

    Args:
        dataset_name: Dataset name ('MSL' or 'SMAP').
        dataset_dir: Directory containing expected .npy files.

    Raises:
        RuntimeError: If validation fails.
    """
    expected_files = EXPECTED_FILES[dataset_name]
    train_path = dataset_dir / expected_files[0]
    test_path = dataset_dir / expected_files[1]
    label_path = dataset_dir / expected_files[2]

    for file_path in (train_path, test_path, label_path):
        if not file_path.exists() or not file_path.is_file():
            raise RuntimeError(f"Missing required file: {file_path}")

    train_data = np.load(train_path)
    test_data = np.load(test_path)
    test_labels = np.load(label_path)

    if train_data.ndim != 2 or test_data.ndim != 2:
        raise RuntimeError(f"{dataset_name} train/test must be 2D arrays.")
    if test_labels.ndim != 1:
        raise RuntimeError(f"{dataset_name} test labels must be 1D array.")
    if train_data.shape[1] != EXPECTED_FEATURES[dataset_name]:
        raise RuntimeError(
            f"{dataset_name} train feature dimension mismatch: expected "
            f"{EXPECTED_FEATURES[dataset_name]}, got {train_data.shape[1]}."
        )
    if test_data.shape[1] != EXPECTED_FEATURES[dataset_name]:
        raise RuntimeError(
            f"{dataset_name} test feature dimension mismatch: expected "
            f"{EXPECTED_FEATURES[dataset_name]}, got {test_data.shape[1]}."
        )
    if test_data.shape[0] != test_labels.shape[0]:
        raise RuntimeError(
            f"{dataset_name} test/test_label length mismatch: "
            f"{test_data.shape[0]} vs {test_labels.shape[0]}."
        )

    logger.info(
        "%s validation passed: train=%s, test=%s, labels=%s",
        dataset_name,
        train_data.shape,
        test_data.shape,
        test_labels.shape,
    )


def _download_dataset(dataset_name: str, output_root: Path) -> None:
    """Download a dataset archive and place files in expected directory.

    Args:
        dataset_name: Dataset name ('MSL' or 'SMAP').
        output_root: Root output directory (typically data/raw).

    Raises:
        RuntimeError: If download, extraction, or validation fails.
    """
    dataset_dir = output_root / dataset_name.lower()
    expected_files = EXPECTED_FILES[dataset_name]
    if dataset_dir.exists() and all(
        (dataset_dir / name).is_file() for name in expected_files
    ):
        logger.info(
            "%s files already exist in %s, skipping download.",
            dataset_name,
            dataset_dir,
        )
        _validate_dataset_files(dataset_name, dataset_dir)
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / f"{dataset_name}.zip"

        logger.info("Downloading %s archive...", dataset_name)
        _download_file(DATASET_URLS[dataset_name], archive_path)

        logger.info("Extracting %s archive...", dataset_name)
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                f"Downloaded archive for {dataset_name} is not a valid zip file."
            ) from exc

        for file_name in expected_files:
            src_file = _find_single_file(extract_dir, file_name)
            dst_file = dataset_dir / file_name
            shutil.copy2(src_file, dst_file)
            logger.info("Copied %s to %s", src_file, dst_file)

    _validate_dataset_files(dataset_name, dataset_dir)


def main() -> None:
    """Entry point for MSL/SMAP download script."""
    parser = argparse.ArgumentParser(
        description="Download MSL and SMAP datasets (NASA telemetry, preprocessed .npy)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Root directory for raw data (default: data/raw)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MSL/SMAP datasets to %s", output_root.resolve())

    try:
        _download_dataset("MSL", output_root)
        _download_dataset("SMAP", output_root)
        logger.info("MSL/SMAP download complete.")
    except RuntimeError as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
