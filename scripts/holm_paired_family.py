"""Holm correction over the family of paired tests reported on the TSB-AD paired subset.

The paper reports many paired Wilcoxon tests on the same 100-120 series: rendered vs each raw
control, ViT4TS vs rendered and vs raw, four trained detectors vs raw, the random-init encoder
control, per-stratum rendered vs raw, and every detector in artifacts/detectors/summary.json
(default and TSB-AD-tuned settings) vs raw. They are treated as one family; this script collects
their p-values from the released artifacts and applies Holm's step-down correction.

Reads only artifacts/; writes artifacts/stats/holm_paired_family.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts")


def _walk(obj: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("wilcoxon_p", "p") and isinstance(v, float):
                yield prefix.rstrip("/"), v
            else:
                yield from _walk(v, f"{prefix}{k}/")


def collect() -> dict[str, float]:
    p: dict[str, float] = {}
    detectors = json.loads((ART / "detectors/summary.json").read_text())
    for name, blocks in detectors.items():
        for m in ("VUS-PR", "AUC-ROC"):
            row = blocks.get("all", {}).get("vs_raw_flatten", {}).get(m)
            if row:
                p[f"{name} vs raw_flatten {m}"] = row["wilcoxon_p"]
    for k, v in _walk(json.loads((ART / "vit4ts_paired/summary.json").read_text())):
        p[f"ViT4TS {k}"] = v
    rb = json.loads((ART / "random_backbone/summary.json").read_text())
    for m in ("VUS-PR", "AUC-ROC"):
        p[f"random-init vs pretrained {m}"] = rb[m]["random_vs_pretrained"]["wilcoxon_p"]
    recs = json.loads((ART / "tsb_ad_vision_paired/per_series.json").read_text())
    for arm in ("raw_flatten", "raw_meanpool"):
        for m in ("VUS-PR", "AUC-ROC"):
            a, b = np.array([(r["vision"][m], r[arm][m]) for r in recs
                             if r.get("vision") and r.get(arm)]).T
            p[f"rendered vs {arm} {m}"] = float(stats.wilcoxon(a, b).pvalue)
    for s in sorted({r["stratum"] for r in recs}):
        pairs = [(r["vision"]["VUS-PR"], r["raw_flatten"]["VUS-PR"]) for r in recs
                 if r["stratum"] == s and r.get("vision") and r.get("raw_flatten")]
        if len(pairs) > 5:
            a, b = np.array(pairs).T
            p[f"stratum {s} rendered vs raw_flatten VUS-PR"] = float(stats.wilcoxon(a, b).pvalue)
    return p


def holm(p: dict[str, float]) -> list[dict[str, Any]]:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, []
    for i, (name, raw) in enumerate(items):
        running = min(1.0, max(running, (m - i) * raw))
        out.append({"test": name, "p_raw": raw, "p_holm": running, "survives_0.05": running < 0.05})
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = holm(collect())
    (ART / "stats").mkdir(exist_ok=True)
    (ART / "stats/holm_paired_family.json").write_text(
        json.dumps({"family_size": len(rows), "tests": rows}, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        LOGGER.info("%s p_holm=%.3g  %s", "SURVIVES" if r["survives_0.05"] else "--------",
                    r["p_holm"], r["test"])


if __name__ == "__main__":
    main()
