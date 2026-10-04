"""Trained deep baselines on the TSB-AD paired subset (closes review gap W1).

The benchmark-scale audit originally compared only raw-space variants and our
rendered arm, while the trained detectors appeared solely on the four classic
datasets. That made the claim "the missing control is a gap in the state of the
art" rest on a narrower experiment than the 1,044-series framing implies.

This script runs deep detectors bundled with TSB-AD on exactly the series of the
paired subset, so their numbers are directly comparable to the raw controls and the
rendered arm already scored there.

Scoring is aligned to our protocol: each detector produces point-level scores over
the test split, which we aggregate into the same windows (max within window) and
evaluate against identical any-positive window labels and the same metric stack.

Output: results/tsb_ad_trained/{per_series,summary}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_tsb_ad_raw_audit import load_series, parse_split
from src.data.base import create_sliding_windows
from src.evaluation.modern_metrics import compute_modern_metrics

LOGGER = logging.getLogger(__name__)

WINDOW_SIZE = 100
MIN_STRIDE = 10
MAX_WINDOWS = 1000
DATA_ROOT = Path("data/tsb_ad")
PAIRED = Path("results/tsb_ad_vision_paired/per_series.json")
OUT_ROOT = Path("results/tsb_ad_trained")

#: Semi-supervised detectors get the anomaly-free training split, matching how our
#: raw and rendered arms are fitted. Names must exist in TSB-AD's model pool.
DEFAULT_DETECTORS = ("USAD", "TimesNet", "AnomalyTransformer")


def _window_scores(point_scores: np.ndarray, n_windows: int, stride: int) -> np.ndarray:
    """Aggregate point-level scores into our windows (max within window)."""
    out = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        seg = point_scores[i * stride : i * stride + WINDOW_SIZE]
        out[i] = float(np.nanmax(seg)) if seg.size else 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def run_series(entry: dict[str, Any], detectors: tuple[str, ...]) -> dict[str, Any] | None:
    """Score one series with each trained detector on our windows."""
    from TSB_AD.model_wrapper import run_Semisupervise_AD, Semisupervise_AD_Pool

    path = DATA_ROOT / entry["subset"] / entry["series"]
    train_end, _ = parse_split(path)
    values, labels = load_series(path)

    n_test_points = values.shape[0] - train_end
    stride = max(MIN_STRIDE, math.ceil(n_test_points / MAX_WINDOWS))
    test_values = values[train_end:]
    _, win_labels = create_sliding_windows(
        test_values, labels[train_end:], WINDOW_SIZE, stride
    )
    if win_labels.size < 2 or win_labels.sum() in (0, win_labels.size):
        return None

    record: dict[str, Any] = {
        "subset": entry["subset"], "series": entry["series"],
        "stratum": entry["stratum"], "n_channels": entry["n_channels"],
        "stride": int(stride), "n_test_windows": int(win_labels.size),
    }
    for name in detectors:
        if name not in Semisupervise_AD_Pool:
            LOGGER.warning("%s not in TSB-AD pool; skipping", name)
            record[name] = None
            continue
        try:
            scores = run_Semisupervise_AD(
                name, values[:train_end].astype(float), test_values.astype(float)
            )
            if isinstance(scores, str):          # TSB-AD returns the error text
                raise RuntimeError(scores)
            scores = np.asarray(scores, dtype=np.float64).ravel()
            if scores.size < test_values.shape[0]:
                scores = np.pad(scores, (0, test_values.shape[0] - scores.size),
                                mode="edge")
            win_scores = _window_scores(scores[: test_values.shape[0]],
                                        win_labels.size, stride)
            if float(np.nanstd(win_scores)) == 0.0:
                raise ValueError("degenerate scores")
            record[name] = compute_modern_metrics(
                win_scores, win_labels, sliding_window=WINDOW_SIZE,
                verify_against_own=False,
            )
        except Exception as exc:
            LOGGER.warning("%s [%s] failed: %s", entry["series"], name,
                           str(exc)[:120])
            record[name] = None
    return record


def summarize(records: list[dict[str, Any]], detectors: tuple[str, ...],
              paired_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Paired comparison of each detector against the raw-flatten control."""
    from scipy import stats

    summary: dict[str, Any] = {"n_series": len(records), "detectors": {}}
    for name in detectors:
        for arm in ("raw_flatten", "vision"):
            for metric in ("VUS-PR", "AUC-ROC"):
                pairs = []
                for r in records:
                    own = r.get(name)
                    ref = paired_lookup.get(r["series"], {}).get(arm)
                    if own and ref:
                        pairs.append((own[metric], ref[metric]))
                if len(pairs) < 2:
                    continue
                a = np.array([p[0] for p in pairs])
                b = np.array([p[1] for p in pairs])
                d = a - b
                entry: dict[str, Any] = {
                    "n": len(pairs), "detector_mean": float(a.mean()),
                    "reference_mean": float(b.mean()), "delta": float(d.mean()),
                    "detector_wins": int((d > 0).sum()),
                }
                if np.any(d != 0):
                    entry["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
                key = f"vs_{arm}"
                summary["detectors"].setdefault(name, {}).setdefault(key, {})[metric] = entry
    return summary


def _report(summary: dict[str, Any]) -> None:
    """Log one line per detector x reference arm x metric."""
    for name, per_arm in summary["detectors"].items():
        for arm, per_metric in per_arm.items():
            for metric, st in per_metric.items():
                LOGGER.info("  %-20s %-15s [%s] n=%3d det=%.4f ref=%.4f d=%+.4f "
                            "wins=%d p=%s", name, arm, metric, st["n"],
                            st["detector_mean"], st["reference_mean"], st["delta"],
                            st["detector_wins"],
                            f"{st.get('wilcoxon_p', float('nan')):.3g}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--detectors", type=str,
                            default=",".join(DEFAULT_DETECTORS))
    _ = parser.add_argument("--limit", type=int, default=None)
    _ = parser.add_argument("--shard", type=int, default=0)
    _ = parser.add_argument("--num-shards", type=int, default=1)
    _ = parser.add_argument("--aggregate", action="store_true",
                            help="merge existing shard files and summarise only")
    args = parser.parse_args()

    detectors = tuple(d.strip() for d in str(args.detectors).split(",") if d.strip())
    paired = [e for e in json.loads(PAIRED.read_text())
              if e.get("vision") and e.get("raw_flatten")]
    lookup = {e["series"]: e for e in paired}
    if args.limit is not None:
        paired = paired[: int(args.limit)]
    if bool(args.aggregate):
        records: list[dict[str, Any]] = []
        for shard_path in sorted(OUT_ROOT.glob("per_series_shard*.json")):
            records.extend(json.loads(shard_path.read_text()))
        LOGGER.info("aggregated %d series from %d shards", len(records),
                    len(list(OUT_ROOT.glob("per_series_shard*.json"))))
        summary = summarize(records, detectors, lookup)
        (OUT_ROOT / "per_series.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8")
        (OUT_ROOT / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _report(summary)
        return

    mine = [e for i, e in enumerate(paired) if i % int(args.num_shards) == int(args.shard)]
    LOGGER.info("shard %d/%d: %d series, detectors=%s",
                args.shard, args.num_shards, len(mine), detectors)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"per_series_shard{args.shard}.json"
    records: list[dict[str, Any]] = []
    if out_path.exists():
        try:
            records = json.loads(out_path.read_text())
            LOGGER.info("resuming with %d records", len(records))
        except Exception:
            records = []
    done = {r["series"] for r in records}
    mine = [e for e in mine if e["series"] not in done]

    for index, entry in enumerate(mine, start=1):
        rec = run_series(entry, detectors)
        if rec is not None:
            records.append(rec)
        if index % 5 == 0 or index == len(mine):
            out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            LOGGER.info("%d/%d (%d scored)", index, len(mine), len(records))

    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    if int(args.num_shards) == 1:
        summary = summarize(records, detectors, lookup)
        (OUT_ROOT / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _report(summary)
        LOGGER.info("wrote %s", OUT_ROOT / "summary.json")


if __name__ == "__main__":
    main()
