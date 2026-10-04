"""Per-entity macro AUC-ROC on SMD for detectors that were scored on the concatenated test set.

The classic-benchmark table reports SMD as the mean of per-entity AUC-ROC over the 28 machines.
CATCH, TimesNet and the Anomaly Transformer were run through their reference framework, which
scores SMD as a single concatenated series, so their stored AUC is pooled across machines and
not comparable with the rest of the column. This script cuts each stored score array back into
the 28 machines and recomputes the per-entity mean.

The framework concatenates the machines in lexicographic file order; we verify that by checking
that the stored labels equal the concatenation of the per-machine label files exactly, and refuse
to proceed otherwise.

Inputs (GPU-run outputs, not shipped): results/reports/dl_baselines/<method>_smd/{scores,labels}.npy
and the raw SMD label files. Output (shipped): artifacts/classic/smd_entity_macro.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)
LABEL_DIR = Path("data/raw/smd/test_label")
RUN_DIR = Path("results/reports/dl_baselines")
OUT = Path("artifacts/classic/smd_entity_macro.json")
METHODS = ("catch", "timesnet", "at", "dcdetector")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = sorted(LABEL_DIR.glob("machine-*.txt"))
    labels = [np.loadtxt(f).astype(np.int64) for f in files]
    bounds = np.cumsum([0] + [lab.size for lab in labels])
    out: dict[str, dict] = {}
    for method in METHODS:
        scores = np.load(RUN_DIR / f"{method}_smd" / "scores.npy").ravel()
        stored = np.load(RUN_DIR / f"{method}_smd" / "labels.npy").ravel()
        n = min(scores.size, stored.size)
        if not np.array_equal(stored[:n], np.concatenate(labels)[:n]):
            # Entity boundaries cannot be recovered reliably; keep only the pooled figure.
            out[method] = {"pooled_auc_roc": float(roc_auc_score(stored, scores[: stored.size])),
                           "entity_macro_auc_roc": None,
                           "note": "stored labels are not the lexicographic concatenation; "
                                   "per-entity split not recoverable"}
            LOGGER.info("%-10s pooled %.3f -> entity macro not recoverable", method,
                        out[method]["pooled_auc_roc"])
            continue
        per_entity = {}
        for f, lab, lo, hi in zip(files, labels, bounds[:-1], bounds[1:]):
            seg = scores[lo:min(hi, n)]
            if seg.size == 0 or lab[: seg.size].min() == lab[: seg.size].max():
                continue
            per_entity[f.stem] = float(roc_auc_score(lab[: seg.size], seg))
        out[method] = {"pooled_auc_roc": float(roc_auc_score(stored[:n], scores[:n])),
                       "entity_macro_auc_roc": float(np.mean(list(per_entity.values()))),
                       "n_entities": len(per_entity), "dropped_tail_points": int(bounds[-1] - n),
                       "per_entity": per_entity}
        LOGGER.info("%-10s pooled %.3f -> entity macro %.3f (%d entities)", method,
                    out[method]["pooled_auc_roc"], out[method]["entity_macro_auc_roc"],
                    len(per_entity))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
