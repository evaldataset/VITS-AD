"""TSB-AD raw-space Mahalanobis audit (roadmap 1.3/1.4, stage (a)).

Runs the paper's raw-space control across the TSB-AD benchmark (870 univariate +
200 multivariate curated series) and scores it with the modern metric stack
(VUS-PR, VUS-ROC, Affiliation-F, range-based F1) via the reference
implementation in :mod:`src.evaluation.modern_metrics`.

This is the CPU-only stage: it establishes coverage at benchmark scale without
competing for GPUs with the vision runs. The vision arm is added later on a
stratified subset (stage (b)).

TSB-AD conventions (verified on 12 files, 12/12 agreement):
    filename ``..._tr_<T>_1st_<F>.csv`` -> the first ``T`` rows are the
    anomaly-free training split; ``F`` is the index of the first anomaly.
    Final column is ``Label``; all preceding columns are channels.

Two raw variants, matching the paper:
    mean_pooled : window mean across time -> D-dim vector
    flattened   : whole window flattened  -> W*D-dim vector

Output: results/tsb_ad_raw_audit/{subset}/summary.json (+ per-series records)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from scripts.run_raw_mahalanobis_baseline import RawMahalanobisScorer
from src.data.base import create_sliding_windows
from src.evaluation.modern_metrics import compute_modern_metrics

LOGGER = logging.getLogger(__name__)

WINDOW_SIZE = 100
#: Benchmark-scale protocol. The paper's canonical raw protocol uses stride=1,
#: but TSB-AD series average ~50k (U) to ~109k (M) timesteps, so stride=1 would
#: produce >100k windows per series. We use stride=10 here (matching VITS-AD's
#: own tuned protocol) and disclose it; both arms use the identical protocol so
#: the raw-vs-vision comparison stays internally consistent.
STRIDE = 10
#: Ledoit-Wolf inversion is O(d^3); the flattened variant has d = W * D, so it
#: becomes intractable for wide multivariate series. Above this cap we report
#: mean-pooled only and record the flattened variant as skipped (disclosed).
MAX_FLATTEN_DIM = 2000
DATA_ROOT = Path("data/tsb_ad")
OUT_ROOT = Path("results/tsb_ad_raw_audit")

_NAME_RE = re.compile(r"_tr_(\d+)_1st_(\d+)")


def parse_split(path: Path) -> tuple[int, int]:
    """Extract the train boundary and first-anomaly index from a filename.

    Args:
        path: CSV path following the TSB-AD naming convention.

    Returns:
        ``(train_end, first_anomaly)`` indices.

    Raises:
        ValueError: If the filename does not follow the convention.
    """
    match = _NAME_RE.search(path.name)
    if match is None:
        raise ValueError(f"filename does not encode a split: {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_series(path: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Load one TSB-AD CSV into ``(values, labels)``.

    Args:
        path: CSV path. Final column must be ``Label``.

    Returns:
        ``(values (T, D) float64, labels (T,) int64)``.
    """
    frame = pd.read_csv(path)
    labels = frame["Label"].to_numpy(dtype=np.int64)
    values = frame.drop(columns=["Label"]).to_numpy(dtype=np.float64)
    return values, labels


def score_series(path: Path) -> dict[str, Any] | None:
    """Run both raw-Mahalanobis variants on one series.

    Args:
        path: CSV path.

    Returns:
        Record dict, or ``None`` if the series is unusable (too short, no
        anomalies in the test split, or degenerate).
    """
    train_end, _ = parse_split(path)
    values, labels = load_series(path)
    total, n_channels = values.shape

    if train_end < WINDOW_SIZE + 1 or total - train_end < WINDOW_SIZE + 1:
        LOGGER.debug("skip %s: too short (T=%d, tr=%d)", path.name, total, train_end)
        return None

    train_raw, test_raw = values[:train_end], values[train_end:]
    test_labels_raw = labels[train_end:]

    # Standardise using TRAIN statistics only (no leakage).
    mu = train_raw.mean(axis=0)
    sigma = train_raw.std(axis=0) + 1e-8
    train_norm = (train_raw - mu) / sigma
    test_norm = (test_raw - mu) / sigma

    # Training split is anomaly-free by construction (verified on the naming
    # convention); labels are still passed so the repo's validated windowing
    # helper produces the any-positive window labels for the test split.
    train_windows, _ = create_sliding_windows(
        train_norm, labels[:train_end], WINDOW_SIZE, STRIDE
    )
    test_windows, win_labels = create_sliding_windows(
        test_norm, test_labels_raw, WINDOW_SIZE, STRIDE
    )
    if train_windows.shape[0] < 2 or test_windows.shape[0] < 2:
        return None

    if win_labels.sum() == 0 or win_labels.sum() == win_labels.shape[0]:
        LOGGER.debug("skip %s: degenerate window labels", path.name)
        return None

    record: dict[str, Any] = {
        "series": path.name,
        "n_channels": int(n_channels),
        "length": int(total),
        "train_end": int(train_end),
        "n_test_windows": int(test_windows.shape[0]),
        "anomaly_window_rate": float(win_labels.mean()),
    }

    flatten_dim = WINDOW_SIZE * n_channels
    record["flatten_dim"] = int(flatten_dim)
    record["flatten_skipped_too_wide"] = bool(flatten_dim > MAX_FLATTEN_DIM)

    for variant, featurize in (
        ("mean_pooled", lambda w: w.mean(axis=1)),
        ("flattened", lambda w: w.reshape(w.shape[0], -1)),
    ):
        if variant == "flattened" and flatten_dim > MAX_FLATTEN_DIM:
            LOGGER.debug(
                "%s: skipping flattened (d=%d > %d)",
                path.name, flatten_dim, MAX_FLATTEN_DIM,
            )
            record[variant] = None
            continue
        try:
            train_feat = featurize(train_windows).astype(np.float64)
            test_feat = featurize(test_windows).astype(np.float64)
            scorer = RawMahalanobisScorer()
            scorer.fit(train_feat)
            scores = scorer.score(test_feat)
            if not np.all(np.isfinite(scores)):
                raise ValueError("non-finite scores")
            metrics = compute_modern_metrics(
                scores, win_labels, sliding_window=WINDOW_SIZE,
                verify_against_own=False,
            )
            record[variant] = metrics
        except Exception as exc:  # one bad variant must not kill the series
            LOGGER.warning("%s [%s] failed: %s", path.name, variant, exc)
            record[variant] = None

    if record.get("mean_pooled") is None and record.get("flattened") is None:
        return None
    return record


def run_subset(subset: str, limit: int | None) -> None:
    """Run the audit over one TSB-AD subset.

    Args:
        subset: ``"TSB-AD-U"`` or ``"TSB-AD-M"``.
        limit: Optional cap on the number of series (for smoke runs).
    """
    files = sorted((DATA_ROOT / subset).glob("*.csv"))
    if limit is not None:
        files = files[:limit]
    LOGGER.info("%s: %d series", subset, len(files))

    records: list[dict[str, Any]] = []
    skipped = 0
    for index, path in enumerate(files, start=1):
        try:
            record = score_series(path)
        except Exception as exc:
            LOGGER.warning("%s failed: %s", path.name, exc)
            record = None
        if record is None:
            skipped += 1
        else:
            records.append(record)
        if index % 50 == 0 or index == len(files):
            LOGGER.info("%s: %d/%d done (%d skipped)", subset, index, len(files), skipped)

    out_dir = OUT_ROOT / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_series.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "subset": subset,
        "n_attempted": len(files),
        "n_scored": len(records),
        "n_skipped": skipped,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "max_flatten_dim": MAX_FLATTEN_DIM,
        "n_flatten_skipped_too_wide": sum(
            1 for r in records if r.get("flatten_skipped_too_wide")
        ),
        "protocol_note": (
            "stride=10 (not the paper's canonical stride=1) because TSB-AD series "
            "average ~50k-109k timesteps; flattened variant reported only where "
            "W*D <= max_flatten_dim (Ledoit-Wolf inversion is O(d^3))."
        ),
    }
    for variant in ("mean_pooled", "flattened"):
        per_metric: dict[str, list[float]] = {}
        for record in records:
            metrics = record.get(variant)
            if not metrics:
                continue
            for name, value in metrics.items():
                if np.isfinite(value):
                    per_metric.setdefault(name, []).append(float(value))
        summary[variant] = {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
            for name, values in sorted(per_metric.items())
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    LOGGER.info("=== %s summary (n=%d scored, %d skipped) ===",
                subset, len(records), skipped)
    for variant in ("mean_pooled", "flattened"):
        stats = summary.get(variant, {})
        if stats:
            LOGGER.info(
                "  %-11s VUS-PR=%.4f  AUC-PR=%.4f  AUC-ROC=%.4f  Affil-F=%.4f",
                variant,
                stats.get("VUS-PR", {}).get("mean", float("nan")),
                stats.get("AUC-PR", {}).get("mean", float("nan")),
                stats.get("AUC-ROC", {}).get("mean", float("nan")),
                stats.get("Affiliation-F", {}).get("mean", float("nan")),
            )
    LOGGER.info("wrote %s", out_dir / "summary.json")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--subsets", type=str, default="TSB-AD-U,TSB-AD-M")
    _ = parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for subset in str(args.subsets).split(","):
        subset = subset.strip()
        if subset:
            run_subset(subset, args.limit)


if __name__ == "__main__":
    main()
