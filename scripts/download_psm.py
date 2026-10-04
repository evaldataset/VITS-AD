"""Download PSM (Pooled Server Metrics) dataset.

Usage:
    python scripts/download_psm.py [--output_dir data/raw/psm]

The PSM dataset was introduced by Abdulaal et al. (2021) in
"Practical Approach to Asynchronous Multivariate Time Series Anomaly
Detection and Localization" from eBay.

The script downloads train.csv, test.csv, and test_label.csv from the
official eBay RANSynCoders GitHub repository:
    https://github.com/eBay/RANSynCoders

Expected output structure:
    data/raw/psm/
    ├── train.csv       # 132481 rows × 26 cols (timestamp + 25 features)
    ├── test.csv        # 87841 rows × 26 cols
    └── test_label.csv  # 87841 rows (label column)
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Raw GitHub URLs from eBay/RANSynCoders repository
BASE_URL = "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data"
PSM_FILES = ["train.csv", "test.csv", "test_label.csv"]


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    """Download a single file from URL to destination.

    Args:
        url: Source URL.
        dest: Destination file path.
        timeout: Download timeout in seconds.

    Raises:
        urllib.error.URLError: If download fails.
    """
    logger.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, str(dest))
    file_size = dest.stat().st_size
    logger.info("  Downloaded %.2f MB", file_size / (1024 * 1024))


def download_psm(output_dir: Path) -> None:
    """Download all PSM data files.

    Args:
        output_dir: Directory to save the files.

    Raises:
        RuntimeError: If any download fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in PSM_FILES:
        dest = output_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            logger.info(
                "Skipping %s (already exists, %.2f MB)",
                filename,
                dest.stat().st_size / (1024 * 1024),
            )
            continue

        url = f"{BASE_URL}/{filename}"
        try:
            download_file(url, dest)
        except Exception as exc:
            logger.error("Failed to download %s: %s", url, exc)
            # Try alternative source: clone repo
            logger.info("Trying alternative download via git clone...")
            _download_via_git(output_dir)
            return

    _validate_download(output_dir)


def _download_via_git(output_dir: Path) -> None:
    """Fallback: clone the RANSynCoders repo and extract PSM data.

    Args:
        output_dir: Directory to save the files.
    """
    import shutil
    import subprocess
    import tempfile

    repo_url = "https://github.com/eBay/RANSynCoders.git"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir) / "RANSynCoders"
        logger.info("Cloning %s (sparse checkout)...", repo_url)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                repo_url,
                str(tmppath),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "data"],
            cwd=str(tmppath),
            check=True,
            capture_output=True,
        )

        data_dir = tmppath / "data"
        if not data_dir.exists():
            raise RuntimeError(f"data/ directory not found in cloned repo at {tmppath}")

        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in PSM_FILES:
            src = data_dir / filename
            if not src.exists():
                raise RuntimeError(f"Expected file not found: {src}")
            shutil.copy2(str(src), str(output_dir / filename))
            logger.info("Copied %s → %s", src, output_dir / filename)

    _validate_download(output_dir)


def _validate_download(output_dir: Path) -> None:
    """Validate that all required files exist and have reasonable sizes.

    Args:
        output_dir: Directory containing downloaded files.

    Raises:
        RuntimeError: If validation fails.
    """
    for filename in PSM_FILES:
        filepath = output_dir / filename
        if not filepath.exists():
            raise RuntimeError(f"Missing required file: {filepath}")
        if filepath.stat().st_size < 1024:
            raise RuntimeError(
                f"File suspiciously small ({filepath.stat().st_size} bytes): {filepath}"
            )
        logger.info(
            "Validated: %s (%.2f MB)", filename, filepath.stat().st_size / (1024 * 1024)
        )

    logger.info("PSM download complete. Files in: %s", output_dir)


def main() -> None:
    """Entry point for PSM download script."""
    parser = argparse.ArgumentParser(
        description="Download PSM (Pooled Server Metrics) dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/psm",
        help="Output directory for PSM data files.",
    )
    args = parser.parse_args()
    output_path = Path(args.output_dir)

    logger.info("Downloading PSM dataset to %s", output_path)
    try:
        download_psm(output_path)
    except Exception as exc:
        logger.error("PSM download failed: %s", exc)
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
