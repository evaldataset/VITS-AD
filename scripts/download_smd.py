"""Download SMD (Server Machine Dataset) from OmniAnomaly repository.

Usage:
    python scripts/download_smd.py [--output_dir data/raw/smd]

The script downloads the ServerMachineDataset from the OmniAnomaly GitHub
repository (NetManAIOps/OmniAnomaly) and organizes it into the expected
directory structure:

    data/raw/smd/
    ├── train/
    │   ├── machine-1-1.txt
    │   └── ...
    ├── test/
    │   ├── machine-1-1.txt
    │   └── ...
    └── test_label/
        ├── machine-1-1.txt
        └── ...
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OMNIANOMALY_REPO = "https://github.com/NetManAIOps/OmniAnomaly.git"
SMD_SUBDIR = "ServerMachineDataset"

# All 28 SMD entities
SMD_ENTITIES: list[str] = [
    f"machine-{group}-{idx}"
    for group in range(1, 4)
    for idx in range(1, 9 if group < 3 else 12)
]

EXPECTED_SUBDIRS = ("train", "test", "test_label")


def download_smd_via_git(output_dir: Path) -> None:
    """Download SMD using git sparse checkout.

    Args:
        output_dir: Target directory for SMD data.

    Raises:
        RuntimeError: If git operations fail.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("SMD data already exists at %s, skipping download.", output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        logger.info("Cloning OmniAnomaly repo (sparse checkout)...")

        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    OMNIANOMALY_REPO,
                    str(tmp_path / "repo"),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "sparse-checkout", "set", SMD_SUBDIR],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(tmp_path / "repo"),
                timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Git operation failed: {exc.stderr}") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                "git is not installed. Please install git and retry."
            ) from exc

        src_dir = tmp_path / "repo" / SMD_SUBDIR
        if not src_dir.exists():
            raise RuntimeError(
                f"Expected directory {SMD_SUBDIR} not found in cloned repo."
            )

        for subdir_name in EXPECTED_SUBDIRS:
            src_subdir = src_dir / subdir_name
            dst_subdir = output_dir / subdir_name
            if src_subdir.exists():
                shutil.copytree(str(src_subdir), str(dst_subdir), dirs_exist_ok=True)
                file_count = len(list(dst_subdir.glob("*.txt")))
                logger.info(
                    "Copied %s/ (%d files) to %s", subdir_name, file_count, dst_subdir
                )
            else:
                logger.warning("Expected subdirectory not found: %s", src_subdir)

    _validate_smd_data(output_dir)


def _validate_smd_data(data_dir: Path) -> None:
    """Validate downloaded SMD data structure.

    Args:
        data_dir: Path to SMD data directory.

    Raises:
        RuntimeError: If validation fails.
    """
    missing_dirs: list[str] = []
    for subdir_name in EXPECTED_SUBDIRS:
        subdir = data_dir / subdir_name
        if not subdir.exists() or not subdir.is_dir():
            missing_dirs.append(subdir_name)

    if missing_dirs:
        raise RuntimeError(
            f"Missing expected subdirectories: {missing_dirs} in {data_dir}"
        )

    train_files = sorted((data_dir / "train").glob("*.txt"))
    test_files = sorted((data_dir / "test").glob("*.txt"))
    label_files = sorted((data_dir / "test_label").glob("*.txt"))

    logger.info(
        "SMD validation: train=%d, test=%d, label=%d files",
        len(train_files),
        len(test_files),
        len(label_files),
    )

    if len(train_files) == 0:
        raise RuntimeError("No training files found in SMD dataset.")
    if len(train_files) != len(test_files):
        raise RuntimeError(
            f"Train/test file count mismatch: {len(train_files)} vs {len(test_files)}"
        )
    if len(test_files) != len(label_files):
        raise RuntimeError(
            f"Test/label file count mismatch: {len(test_files)} vs {len(label_files)}"
        )

    logger.info("SMD data validation passed. %d entities available.", len(train_files))


def main() -> None:
    """Entry point for SMD data download."""
    parser = argparse.ArgumentParser(
        description="Download SMD dataset from OmniAnomaly repository."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/smd",
        help="Target directory for SMD data (default: data/raw/smd)",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    logger.info("Downloading SMD dataset to %s", output_path.resolve())

    try:
        download_smd_via_git(output_path)
        logger.info("SMD download complete.")
    except RuntimeError as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
