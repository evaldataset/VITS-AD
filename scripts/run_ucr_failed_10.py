"""Re-run the 10 UCR series that previously failed VITS pipeline.

Series: 239-241 (taichi), 243, 245-247 (tilt), 248-250 (weallwalk).

Usage:
    python scripts/run_ucr_failed_10.py --ucr_dir data/UCR --output_dir results/ucr_failed10
"""

from __future__ import annotations

# pyright: basic, reportMissingImports=false

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Reuse the existing pipeline by importing the function
import scripts.run_ucr_expanded as ru
from scripts.run_ucr_expanded import (
    LOGGER, SEED, _set_seeds, _setup_logging, _run_one_series, _compute_summary
)
from src.data.ucr import list_ucr_files
from src.models.backbone import VisionBackbone


FAILED_PREFIXES = {"239", "240", "241", "243", "245", "246", "247", "248", "249", "250"}


def _stratified_subsample(windows, labels, max_windows):
    """Stratified subsample preserving both classes.

    Maintains roughly equal anomaly/normal mix while respecting max_windows.
    Falls back gracefully when one class dominates.
    """
    if windows.shape[0] <= max_windows:
        return windows, labels
    anom_idx = np.where(labels == 1)[0]
    norm_idx = np.where(labels == 0)[0]
    n_anom_total = len(anom_idx)
    n_norm_total = len(norm_idx)

    # If single class, uniform sample
    if n_anom_total == 0 or n_norm_total == 0:
        idx = np.linspace(0, windows.shape[0] - 1, max_windows, dtype=np.int64)
        return windows[idx], labels[idx]

    # Cap each class to half max_windows, then redistribute leftover
    half = max_windows // 2
    n_anom_keep = min(n_anom_total, half)
    n_norm_keep = min(n_norm_total, max_windows - n_anom_keep)
    # If anomaly was capped low and norm has slack, use slack on anomaly
    if n_anom_keep < n_anom_total and n_norm_keep < (max_windows - half):
        slack = (max_windows - half) - n_norm_keep
        n_anom_keep = min(n_anom_total, n_anom_keep + slack)

    if n_anom_keep < n_anom_total:
        sel = np.linspace(0, n_anom_total - 1, n_anom_keep, dtype=np.int64)
        anom_idx = anom_idx[sel]
    if n_norm_keep < n_norm_total:
        sel = np.linspace(0, n_norm_total - 1, n_norm_keep, dtype=np.int64)
        norm_idx = norm_idx[sel]

    keep = np.sort(np.concatenate([anom_idx, norm_idx]))
    return windows[keep], labels[keep]


# Monkey-patch the subsampler in the expanded script
ru._subsample_windows = _stratified_subsample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucr_dir", type=str, default="data/UCR")
    parser.add_argument("--output_dir", type=str, default="results/ucr_failed10")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _setup_logging()
    _set_seeds(SEED)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_series").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backbone = VisionBackbone(model_name="facebook/dinov2-base", device=device)

    all_files = list_ucr_files(Path(args.ucr_dir))
    target_files = []
    for path in all_files:
        prefix = path.name.split("_")[0]
        if prefix not in FAILED_PREFIXES:
            continue
        upper = path.name.upper()
        if "DISTORTED" in upper or "NOISE" in upper:
            continue
        target_files.append(path)

    LOGGER.info("Running on %d failed UCR series.", len(target_files))

    per_series = []
    for i, path in enumerate(target_files):
        LOGGER.info("[%d/%d] %s", i + 1, len(target_files), path.name)
        try:
            result = _run_one_series(path, backbone, device, out_dir)
        except Exception as e:
            LOGGER.exception("Failed: %s", e)
            result = None
        if result is not None:
            per_series.append(result)

    summary = _compute_summary(per_series)
    payload = {
        "n_series": len(per_series),
        "n_target": len(target_files),
        "summary": summary,
        "per_series": per_series,
    }
    out_path = out_dir / "summary.json"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    LOGGER.info("Saved summary to %s", out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
