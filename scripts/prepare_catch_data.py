#!/usr/bin/env python3
"""Convert raw VITS datasets to TAB/CATCH format.

The TAB framework expects a "long" CSV format where each feature is stacked
vertically. For a dataset with T timesteps and D features + 1 label column:
  - Total rows = T * (D + 1)
  - Columns: [value, cols]
  - 'cols' contains feature names like "0", "1", ..., "label"

Additionally, a DETECT_META.csv file is needed with at least:
  file_name, train_lens
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _convert_array_to_tab(
    train_data: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    output_path: Path,
) -> int:
    """Convert numpy arrays to TAB long-format CSV.

    Args:
        train_data: (T_train, D) training features.
        test_data: (T_test, D) test features.
        test_labels: (T_test,) binary anomaly labels.
        output_path: Where to write the CSV.

    Returns:
        Training length (number of timesteps).
    """
    n_features = train_data.shape[1]
    train_len = train_data.shape[0]

    # Concatenate train + test
    all_data = np.concatenate([train_data, test_data], axis=0)  # (T, D)
    total_len = all_data.shape[0]

    # Build labels: 0 for train, actual labels for test
    all_labels = np.zeros(total_len, dtype=np.float64)
    all_labels[train_len:] = test_labels.astype(np.float64)

    # Build long-format: stack features vertically
    rows_value = []
    rows_cols = []

    for d in range(n_features):
        rows_value.extend(all_data[:, d].tolist())
        rows_cols.extend([str(d)] * total_len)

    # Add label column
    rows_value.extend(all_labels.tolist())
    rows_cols.extend(["label"] * total_len)

    df = pd.DataFrame({"value": rows_value, "cols": rows_cols})
    df.to_csv(output_path, index=False)

    LOGGER.info(
        "Wrote %s: T=%d, D=%d, rows=%d",
        output_path.name,
        total_len,
        n_features,
        len(df),
    )
    return train_len


def convert_smd(raw_dir: Path, output_dir: Path) -> dict[str, int]:
    """Convert SMD dataset. Uses all entities concatenated."""
    train_dir = raw_dir / "smd" / "train"
    test_dir = raw_dir / "smd" / "test"
    label_dir = raw_dir / "smd" / "test_label"

    all_train = []
    all_test = []
    all_labels = []

    entities = sorted(f.stem for f in train_dir.glob("*.txt"))
    for entity in entities:
        train = np.loadtxt(train_dir / f"{entity}.txt", delimiter=",")
        test = np.loadtxt(test_dir / f"{entity}.txt", delimiter=",")
        labels = np.loadtxt(label_dir / f"{entity}.txt", delimiter=",")
        all_train.append(train)
        all_test.append(test)
        all_labels.append(labels)
        LOGGER.info("SMD %s: train=%s, test=%s", entity, train.shape, test.shape)

    train_data = np.concatenate(all_train, axis=0)
    test_data = np.concatenate(all_test, axis=0)
    test_labels = np.concatenate(all_labels, axis=0)

    train_len = _convert_array_to_tab(
        train_data, test_data, test_labels, output_dir / "SMD.csv"
    )
    return {"SMD.csv": train_len}


def convert_psm(raw_dir: Path, output_dir: Path) -> dict[str, int]:
    """Convert PSM dataset."""
    train_df = pd.read_csv(raw_dir / "psm" / "train.csv")
    test_df = pd.read_csv(raw_dir / "psm" / "test.csv")
    label_df = pd.read_csv(raw_dir / "psm" / "test_label.csv")

    # Drop timestamp column if present
    if train_df.columns[0].lower() in ("timestamp", "date", "time"):
        train_df = train_df.iloc[:, 1:]
    if test_df.columns[0].lower() in ("timestamp", "date", "time"):
        test_df = test_df.iloc[:, 1:]

    # Fill NaN
    train_data = train_df.values.astype(np.float64)
    test_data = test_df.values.astype(np.float64)
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)

    # Labels
    if label_df.shape[1] > 1:
        test_labels = label_df.iloc[:, -1].values.astype(np.float64)
    else:
        test_labels = label_df.values.reshape(-1).astype(np.float64)

    train_len = _convert_array_to_tab(
        train_data, test_data, test_labels, output_dir / "PSM.csv"
    )
    return {"PSM.csv": train_len}


def convert_npy_dataset(
    raw_dir: Path,
    output_dir: Path,
    name: str,
    train_file: str,
    test_file: str,
    label_file: str,
) -> dict[str, int]:
    """Convert numpy-based dataset (MSL, SMAP)."""
    train_data = np.load(raw_dir / train_file).astype(np.float64)
    test_data = np.load(raw_dir / test_file).astype(np.float64)
    test_labels = np.load(raw_dir / label_file).astype(np.float64)

    # Handle NaN
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)

    csv_name = f"{name}.csv"
    train_len = _convert_array_to_tab(
        train_data, test_data, test_labels, output_dir / csv_name
    )
    return {csv_name: train_len}


def write_detect_meta(output_dir: Path, train_lens: dict[str, int]) -> None:
    """Write DETECT_META.csv for CATCH."""
    rows = []
    for file_name, tlen in train_lens.items():
        rows.append(
            {
                "file_name": file_name,
                "train_lens": tlen,
                "freq": "UNKNOWN",
                "if_univariate": False,
                "size": "user",
            }
        )
    meta_df = pd.DataFrame(rows).set_index("file_name")
    meta_path = output_dir / "DETECT_META.csv"
    meta_df.to_csv(meta_path)
    LOGGER.info("Wrote %s with %d entries", meta_path, len(rows))


def main() -> None:
    """Run data conversion."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Convert raw data to CATCH/TAB format")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Raw data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="baselines/CATCH/dataset/anomaly_detect",
        help="Output directory for CATCH data",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_base = Path(args.output_dir)
    data_dir = output_base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_train_lens: dict[str, int] = {}

    # SMD
    LOGGER.info("Converting SMD...")
    all_train_lens.update(convert_smd(raw_dir, data_dir))

    # PSM
    LOGGER.info("Converting PSM...")
    all_train_lens.update(convert_psm(raw_dir, data_dir))

    # MSL
    LOGGER.info("Converting MSL...")
    all_train_lens.update(
        convert_npy_dataset(
            raw_dir / "msl",
            data_dir,
            "MSL",
            "MSL_train.npy",
            "MSL_test.npy",
            "MSL_test_label.npy",
        )
    )

    # SMAP
    LOGGER.info("Converting SMAP...")
    all_train_lens.update(
        convert_npy_dataset(
            raw_dir / "smap",
            data_dir,
            "SMAP",
            "SMAP_train.npy",
            "SMAP_test.npy",
            "SMAP_test_label.npy",
        )
    )

    # Write metadata
    write_detect_meta(output_base, all_train_lens)
    LOGGER.info("Done! Data written to %s", output_base)


if __name__ == "__main__":
    main()
