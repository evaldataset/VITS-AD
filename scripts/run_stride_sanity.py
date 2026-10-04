"""Stride sanity check (closes review gap W6).

The classic benchmarks use the canonical stride of 1, while TSB-AD uses stride 10
because its series are long. Internal validity is protected --- the stride is identical
across arms --- but a reviewer can reasonably ask whether the headline conclusion is a
stride artefact.

This re-runs the stage-(a) raw comparison (flattened versus mean-pooled Mahalanobis) at
stride 1 on a subset of TSB-AD series short enough to make stride 1 tractable, and
reports the same paired statistic. If the flatten-over-mean-pool advantage survives, the
conclusion is not a consequence of the stride choice.

CPU only: no rendering and no backbone are involved.

Output: results/stride_sanity/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from scripts.run_raw_mahalanobis_baseline import RawMahalanobisScorer
from scripts.run_tsb_ad_raw_audit import (
    MAX_FLATTEN_DIM,
    WINDOW_SIZE,
    load_series,
    parse_split,
)
from src.data.base import create_sliding_windows
from src.evaluation.modern_metrics import compute_modern_metrics

LOGGER = logging.getLogger(__name__)

DATA_ROOT = Path("data/tsb_ad")
AUDIT = Path("results/tsb_ad_raw_audit")
OUT_ROOT = Path("results/stride_sanity")
#: Stride 1 is only tractable on short series; this cap keeps window counts sane.
MAX_TEST_POINTS = 8000


def score_at_stride(path: Path, stride: int) -> dict[str, Any] | None:
    """Run both raw variants on one series at the requested stride."""
    train_end, _ = parse_split(path)
    values, labels = load_series(path)
    n_test = values.shape[0] - train_end
    if n_test > MAX_TEST_POINTS or train_end < WINDOW_SIZE + 1:
        return None
    if WINDOW_SIZE * values.shape[1] > MAX_FLATTEN_DIM:
        return None

    mu = values[:train_end].mean(axis=0)
    sigma = values[:train_end].std(axis=0) + 1e-8
    train_w, _ = create_sliding_windows((values[:train_end] - mu) / sigma,
                                        labels[:train_end], WINDOW_SIZE, stride)
    test_w, win_labels = create_sliding_windows((values[train_end:] - mu) / sigma,
                                                labels[train_end:], WINDOW_SIZE, stride)
    if train_w.shape[0] < 2 or test_w.shape[0] < 2:
        return None
    if win_labels.sum() in (0, win_labels.shape[0]):
        return None

    out: dict[str, Any] = {"series": path.name, "stride": stride,
                           "n_test_windows": int(test_w.shape[0])}
    for name, fx in (("mean_pooled", lambda w: w.mean(axis=1)),
                     ("flattened", lambda w: w.reshape(w.shape[0], -1))):
        try:
            scorer = RawMahalanobisScorer()
            scorer.fit(fx(train_w).astype(np.float64))
            scores = scorer.score(fx(test_w).astype(np.float64))
            out[name] = compute_modern_metrics(scores, win_labels,
                                               sliding_window=WINDOW_SIZE,
                                               verify_against_own=False)
        except Exception as exc:
            LOGGER.warning("%s [%s@%d] failed: %s", path.name, name, stride,
                           str(exc)[:90])
            out[name] = None
    return out if (out.get("mean_pooled") and out.get("flattened")) else None


def paired_delta(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Paired flatten-minus-mean-pool statistic over the given records."""
    pairs = [(r["flattened"][metric], r["mean_pooled"][metric]) for r in records]
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    d = a - b
    out: dict[str, Any] = {
        "n": len(pairs), "flatten_mean": float(a.mean()),
        "mean_pool_mean": float(b.mean()), "delta": float(d.mean()),
        "flatten_wins": int((d > 0).sum()),
    }
    if np.any(d != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--limit", type=int, default=40)
    _ = parser.add_argument("--subset", type=str, default="TSB-AD-U")
    args = parser.parse_args()

    scored = json.loads((AUDIT / args.subset / "per_series.json").read_text())
    candidates = [r["series"] for r in scored]

    results: dict[int, list[dict[str, Any]]] = {1: [], 10: []}
    used: list[str] = []
    for name in candidates:
        if len(used) >= int(args.limit):
            break
        path = DATA_ROOT / args.subset / name
        at1 = score_at_stride(path, 1)
        if at1 is None:
            continue
        at10 = score_at_stride(path, 10)
        if at10 is None:
            continue
        results[1].append(at1)
        results[10].append(at10)
        used.append(name)
        if len(used) % 10 == 0:
            LOGGER.info("%d series done", len(used))

    if len(used) < 5:
        LOGGER.error("too few comparable series (%d)", len(used))
        return

    summary: dict[str, Any] = {"subset": args.subset, "n_series": len(used),
                               "max_test_points": MAX_TEST_POINTS, "by_stride": {}}
    for stride in (1, 10):
        summary["by_stride"][str(stride)] = {
            m: paired_delta(results[stride], m) for m in ("VUS-PR", "AUC-ROC")
        }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                           encoding="utf-8")
    LOGGER.info("=== stride sanity (n=%d, same series at both strides) ===", len(used))
    for stride in (1, 10):
        for m in ("VUS-PR", "AUC-ROC"):
            st = summary["by_stride"][str(stride)][m]
            LOGGER.info("  stride %2d [%s] flatten=%.4f mean-pool=%.4f d=%+.4f "
                        "wins=%d/%d p=%.3g", stride, m, st["flatten_mean"],
                        st["mean_pool_mean"], st["delta"], st["flatten_wins"],
                        st["n"], st.get("wilcoxon_p", float("nan")))
    LOGGER.info("wrote %s", OUT_ROOT / "summary.json")


if __name__ == "__main__":
    main()
