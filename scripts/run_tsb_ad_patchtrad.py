"""PatchTrAD on the TSB-AD paired subset.

PatchTrAD is a patch-wise reconstruction Transformer that works on raw 1D patches, with no
rendering and no vision encoder, which makes it the natural trained baseline from the
raw-space family. Its authors released a TSB-AD integration (github.com/vilhess/PatchTrAD,
``tsbad/PatchTrAD.py``). That repository carries no licence, so we do not vendor it: point
``PATCHTRAD_DIR`` at a clone and this script imports the detector from there, with the
authors' default hyperparameters.

The authors state that their TSB-AD port supports univariate series and that the
multivariate port is work in progress. We therefore report univariate series as the
primary result and keep multivariate runs in a separately labelled block.

Everything else matches ``run_tsb_ad_trained_baselines.py``: same series, anomaly-free
training split, point scores aggregated to our windows by max, same metric stack.

Output: results/tsb_ad_patchtrad/{per_series,summary}.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from scripts.run_tsb_ad_raw_audit import load_series, parse_split
from scripts.run_tsb_ad_trained_baselines import (
    DATA_ROOT, MAX_WINDOWS, MIN_STRIDE, PAIRED, WINDOW_SIZE, _window_scores,
)
from src.data.base import create_sliding_windows
from src.evaluation.modern_metrics import compute_modern_metrics

LOGGER = logging.getLogger(__name__)
OUT_ROOT = Path("results/tsb_ad_patchtrad")
UPSTREAM_COMMIT = "012667d57cc052a58fdd28ffba3a0a56c1af7643"


def _load_patchtrad() -> type:
    """Import the PatchTrAD detector class from the authors' clone."""
    root = os.environ.get("PATCHTRAD_DIR")
    if not root:
        raise SystemExit("set PATCHTRAD_DIR to a clone of github.com/vilhess/PatchTrAD")
    # The authors' port imports ``TSB_AD.base``; TSB-AD 1.5 (installed here) keeps the same
    # class at ``TSB_AD.models.base``. Alias the module rather than editing their code.
    import sys

    import TSB_AD.models.base as _tsbad_base
    sys.modules.setdefault("TSB_AD.base", _tsbad_base)
    spec = importlib.util.spec_from_file_location(
        "patchtrad_tsbad", Path(root) / "tsbad" / "PatchTrAD.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PatchTrAD


def run_series(entry: dict[str, Any], detector_cls: type) -> dict[str, Any] | None:
    """Score one series with PatchTrAD on our windows."""
    import torch

    path = DATA_ROOT / entry["subset"] / entry["series"]
    train_end, _ = parse_split(path)
    values, labels = load_series(path)
    test_values = values[train_end:]
    stride = max(MIN_STRIDE, math.ceil(test_values.shape[0] / MAX_WINDOWS))
    _, win_labels = create_sliding_windows(test_values, labels[train_end:], WINDOW_SIZE, stride)
    if win_labels.size < 2 or win_labels.sum() in (0, win_labels.size):
        return None
    record: dict[str, Any] = {
        "subset": entry["subset"], "series": entry["series"], "stratum": entry["stratum"],
        "n_channels": entry["n_channels"], "stride": int(stride),
        "n_test_windows": int(win_labels.size),
    }
    try:
        torch.manual_seed(0)
        np.random.seed(0)
        det = detector_cls(win_size=WINDOW_SIZE, feats=int(values.shape[1]))
        det.fit(values[:train_end].astype(float))
        scores = np.asarray(det.decision_function(test_values.astype(float)),
                            dtype=np.float64).ravel()
        if scores.size < test_values.shape[0]:
            scores = np.pad(scores, (0, test_values.shape[0] - scores.size), mode="edge")
        win_scores = _window_scores(scores[: test_values.shape[0]], win_labels.size, stride)
        if float(np.nanstd(win_scores)) == 0.0:
            raise ValueError("degenerate scores")
        record["PatchTrAD"] = compute_modern_metrics(
            win_scores, win_labels, sliding_window=WINDOW_SIZE, verify_against_own=False)
    except Exception as exc:
        LOGGER.warning("%s failed: %s", entry["series"], str(exc)[:120])
        record["PatchTrAD"] = None
    return record


def _paired(records: list[dict[str, Any]], lookup: dict[str, dict[str, Any]],
            arm: str, metric: str) -> dict[str, Any] | None:
    pairs = [(r["PatchTrAD"][metric], lookup[r["series"]][arm][metric]) for r in records
             if r.get("PatchTrAD") and lookup.get(r["series"], {}).get(arm)]
    if len(pairs) < 2:
        return None
    a, b = np.array(pairs).T
    out: dict[str, Any] = {"n": len(pairs), "patchtrad_mean": float(a.mean()),
                           "reference_mean": float(b.mean()), "delta": float((a - b).mean()),
                           "patchtrad_wins": int((a > b).sum())}
    if np.any(a != b):
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    return out


def summarize(records: list[dict[str, Any]], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Paired PatchTrAD comparisons, univariate (primary) and multivariate (secondary)."""
    blocks = {"all": records,
              "univariate": [r for r in records if r["n_channels"] == 1],
              "multivariate_unsupported_upstream": [r for r in records if r["n_channels"] > 1]}
    out: dict[str, Any] = {"upstream": "github.com/vilhess/PatchTrAD",
                           "upstream_commit": UPSTREAM_COMMIT, "hyperparameters": "authors' defaults",
                           "n_scored": sum(1 for r in records if r.get("PatchTrAD")),
                           "n_failed": sum(1 for r in records if not r.get("PatchTrAD"))}
    for name, recs in blocks.items():
        out[name] = {f"vs_{arm}": {m: _paired(recs, lookup, arm, m) for m in ("VUS-PR", "AUC-ROC")}
                     for arm in ("raw_flatten", "vision")}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--shard", type=int, default=0)
    _ = parser.add_argument("--num-shards", type=int, default=1)
    _ = parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    paired = [e for e in json.loads(PAIRED.read_text()) if e.get("vision") and e.get("raw_flatten")]
    lookup = {e["series"]: e for e in paired}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        records: list[dict[str, Any]] = []
        for f in sorted(OUT_ROOT.glob("per_series_shard*.json")):
            records.extend(json.loads(f.read_text()))
        summary = summarize(records, lookup)
        (OUT_ROOT / "per_series.json").write_text(json.dumps(records, indent=2) + "\n")
        (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        LOGGER.info("aggregated %d series (%d scored)", len(records), summary["n_scored"])
        LOGGER.info("%s", json.dumps(summary, indent=1))
        return

    detector_cls = _load_patchtrad()
    mine = [e for i, e in enumerate(paired) if i % args.num_shards == args.shard]
    out_path = OUT_ROOT / f"per_series_shard{args.shard}.json"
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["series"] for r in records}
    todo = [e for e in mine if e["series"] not in done]
    LOGGER.info("shard %d/%d: %d series (%d already done)", args.shard, args.num_shards,
                len(todo), len(done))
    for i, entry in enumerate(todo, start=1):
        rec = run_series(entry, detector_cls)
        if rec is not None:
            records.append(rec)
        out_path.write_text(json.dumps(records, indent=2) + "\n")
        LOGGER.info("%d/%d (%d scored)", i, len(todo), sum(1 for r in records if r.get("PatchTrAD")))


if __name__ == "__main__":
    main()
