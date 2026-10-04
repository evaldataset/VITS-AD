"""UCR paired comparison, as reported in the paper, from the released artifacts.

The canonical UCR run scored 109 eligible series with the raw controls but only 99 with the
rendered pipeline; the other ten (long, nearly constant activity and gait recordings) failed.
Those ten were later completed with a stratified test-window subsample, which differs from the
uniform subsample of the canonical run. This script therefore reports three things and keeps
them apart:

  * paired_99   -- identical protocol on both arms; the headline figure.
  * completed_10 / all_109 -- including the later runs; a mixed-protocol check.
  * imputation bounds -- ignoring the later runs and imputing chance or zero AUC for the ten,
    with BOTH arms averaged over the same 109 series.

Reads only artifacts/ucr/, so it needs neither the data nor a GPU.

Output: artifacts/ucr/paper_summary.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts/ucr")
VISION = "VITS_Dist"
RAW = "RawMaha_Flattened"


def _paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    return {"n": int(a.size), "rendered_mean": float(a.mean()), "raw_flatten_mean": float(b.mean()),
            "delta": float((a - b).mean()), "rendered_wins": int((a > b).sum()),
            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    per = json.loads((ART / "canonical_per_series.json").read_text())
    done10 = {r["series"]: r["auc_roc"][VISION]
              for r in json.loads((ART / "completed10_summary.json").read_text())["per_series"]}

    paired = [(v[VISION], v[RAW]) for v in per.values() if v.get(VISION) is not None]
    missing = [k for k, v in per.items() if v.get(VISION) is None]
    a, b = (np.array(x) for x in zip(*paired))
    raw_all = np.array([v[RAW] for v in per.values()])

    full = np.array([per[k][VISION] if per[k].get(VISION) is not None else done10[k] for k in per])
    ten_v = np.array([done10[k] for k in missing])
    ten_r = np.array([per[k][RAW] for k in missing])

    out: dict[str, Any] = {
        "paired_99": _paired(a, b),
        "completed_10": {"n": len(missing), "rendered_mean": float(ten_v.mean()),
                         "raw_flatten_mean": float(ten_r.mean()),
                         "rendered_wins": int((ten_v > ten_r).sum()),
                         "note": "stratified window subsample; differs from canonical protocol"},
        "all_109_mixed_protocol": _paired(full, raw_all),
        "imputation_bounds_same_109": {
            f"impute_{name}": float(np.array([per[k][VISION] if per[k].get(VISION) is not None
                                              else val for k in per]).mean() - raw_all.mean())
            for name, val in (("chance", 0.5), ("zero", 0.0))},
        "missing_series": missing,
    }
    (ART / "paper_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    p = out["paired_99"]
    LOGGER.info("paired-99  delta %+.4f (%d/%d, p=%.1e)", p["delta"], p["rendered_wins"], p["n"],
                p["wilcoxon_p"])
    LOGGER.info("all-109    delta %+.4f", out["all_109_mixed_protocol"]["delta"])
    LOGGER.info("bounds     %s", out["imputation_bounds_same_109"])


if __name__ == "__main__":
    main()
