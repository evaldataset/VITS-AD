"""One table of every detector compared with the flattened raw control on the paired subset.

Rows cover three sources, all scored on the same series, windows and metric stack:
  * trained detectors at default hyperparameters (artifacts/tsb_ad_trained, tsb_ad_patchtrad);
  * the same trained detectors with TSB-AD's tuned hyperparameters, and TSB-AD's statistical
    detectors with their tuned settings (artifacts/tsb_ad_pool).
For each row: mean paired difference against the flattened control and against the rendered
arm, a 95% bootstrap CI (10,000 resamples, seed 0), Wilcoxon p, wins and ties, split by
univariate / multivariate where the detector applies to both.

Reads only artifacts/; writes artifacts/detectors/summary.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts")
METRICS = ("VUS-PR", "AUC-ROC")


def _paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any] | None:
    if a.size < 2:
        return None
    d = a - b
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(10000)]
    out: dict[str, Any] = {"n": int(d.size), "detector_mean": float(a.mean()),
                           "reference_mean": float(b.mean()), "delta": float(d.mean()),
                           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                           "wins": int((d > 0).sum()), "ties": int((d == 0).sum())}
    out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    return out


def _rows(records: list[dict[str, Any]], key: str, lookup: dict[str, Any]) -> dict[str, Any]:
    blocks = {"all": records, "univariate": [r for r in records if r["n_channels"] == 1],
              "multivariate": [r for r in records if r["n_channels"] > 1]}
    out: dict[str, Any] = {}
    for blk, recs in blocks.items():
        for arm in ("raw_flatten", "vision"):
            for m in METRICS:
                pairs = [(r[key][m], lookup[r["series"]][arm][m]) for r in recs
                         if r.get(key) and lookup.get(r["series"], {}).get(arm)]
                if len(pairs) >= 2:
                    a, b = np.array(pairs).T
                    out.setdefault(blk, {}).setdefault(f"vs_{arm}", {})[m] = _paired(a, b)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lookup = {e["series"]: e for e in json.loads((ART / "tsb_ad_vision_paired/per_series.json").read_text())}
    table: dict[str, Any] = {}
    default = json.loads((ART / "tsb_ad_trained/per_series.json").read_text())
    for name in ("USAD", "TimesNet", "AnomalyTransformer"):
        table[f"{name} (default)"] = _rows(default, name, lookup)
    table["PatchTrAD (authors' defaults)"] = _rows(
        json.loads((ART / "tsb_ad_patchtrad/per_series.json").read_text()), "PatchTrAD", lookup)
    for d in sorted((ART / "tsb_ad_pool").iterdir()):
        recs = json.loads((d / "per_series.json").read_text())
        table[f"{d.name} (TSB-AD tuned)"] = _rows(recs, d.name, lookup)
    (ART / "detectors").mkdir(exist_ok=True)
    (ART / "detectors/summary.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    for name, blocks in table.items():
        for blk, arms in blocks.items():
            x = arms.get("vs_raw_flatten", {}).get("VUS-PR")
            if x:
                LOGGER.info("%-34s %-12s n=%3d d=%+.3f CI[%+.3f,%+.3f] p=%.2g", name, blk, x["n"],
                            x["delta"], x["ci95"][0], x["ci95"][1], x["wilcoxon_p"])


if __name__ == "__main__":
    main()
