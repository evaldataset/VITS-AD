"""Controlled synthetic regime study v2 — adds an ANOMALY-DURATION factor.

Extends `controlled_regime_synthetic.py` (which established that channel count D is
a causal driver of the raw-vs-vision split) with a second controlled factor the
reviewers flagged as a confound: the duration of the injected anomaly. We hold the
other factors at representative levels and vary D, anomaly type, and duration.

Both anomalies are now LOCALIZED over a segment of length `seg_len` (so duration is
a clean knob for both types):
  amplitude   : the segment is multiplied by AMP_GAIN (a localized energy bump).
  structural  : the segment is overwritten by an amplitude-matched sawtooth
                morphology (shape change, marginal amplitude preserved).

Reuses v1's renderer / frozen-DINOv2 / Ledoit-Wolf-Mahalanobis helpers unchanged,
so results are directly comparable in methodology. Writes to a NEW json; the v1
rebuttal artifact is left untouched.

Output: paper/rebuttal_package/synthetic_regime_v2_duration.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import scripts.controlled_regime_synthetic as v1
from scripts.controlled_regime_synthetic import (
    _auc,
    _make_window,
    _standardize,
    _vision_features,
)
from src.models.backbone import VisionBackbone

LOGGER = logging.getLogger(__name__)

# Held-fixed levels (representative; see v1 for the D/noise sweeps)
NOISE: float = 0.6
N_NORMAL_TRAIN: int = 120
N_NORMAL_TEST: int = 80
N_ANOM_TEST: int = 80
SEEDS: tuple[int, ...] = (42, 123, 456)

# Swept factors
D_GRID: tuple[int, ...] = (1, 25)
TYPES: tuple[str, ...] = ("amplitude", "structural")
DURATIONS: dict[str, int] = {"short": 10, "long": 38}   # anomaly segment length
RHO_D25: float = 0.9

OUT_PATH = Path("paper/rebuttal_package/synthetic_regime_v2_duration.json")


def _sawtooth_segment(rng: np.random.Generator, seg: int) -> np.ndarray:
    tt = np.linspace(0.0, 1.0, seg, dtype=np.float64)
    burst = 2.0 * (tt * v1.STRUCT_K % 1.0) - 1.0
    burst = burst + NOISE * rng.standard_normal(seg)
    return (burst - burst.mean()) / (burst.std() + 1e-8)


def _build_split_v2(
    rng: np.random.Generator, d: int, rho: float, anom_type: str, seg_len: int
):
    """Return standardized (train_win, test_win, test_labels) with a localized
    anomaly of length `seg_len`."""
    v1.NOISE = NOISE  # _make_window reads this module-global at call time
    train = np.stack(
        [_make_window(rng, d, rho, v1.F0, 1.0) for _ in range(N_NORMAL_TRAIN)]
    )
    test_normal = np.stack(
        [_make_window(rng, d, rho, v1.F0, 1.0) for _ in range(N_NORMAL_TEST)]
    )
    anom = []
    for _ in range(N_ANOM_TEST):
        w = _make_window(rng, d, rho, v1.F0, 1.0)
        seg = min(seg_len, v1.L)
        start = int(rng.integers(0, v1.L - seg + 1))
        if anom_type == "amplitude":
            w[start:start + seg, :] *= v1.AMP_GAIN            # localized energy bump
        else:                                                  # localized morphology
            for c in range(d):
                mu_c, sd_c = w[:, c].mean(), w[:, c].std() + 1e-8
                local = _sawtooth_segment(rng, seg)
                w[start:start + seg, c] = local * sd_c + mu_c
        anom.append(w)
    anom = np.stack(anom)

    test = np.concatenate([test_normal, anom], axis=0)
    labels = np.concatenate(
        [np.zeros(N_NORMAL_TEST, dtype=np.int64), np.ones(N_ANOM_TEST, dtype=np.int64)]
    )
    train_s, test_s = _standardize(train, test)
    return train_s, test_s, labels


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    bb = VisionBackbone("facebook/dinov2-base")
    records: list[dict[str, Any]] = []

    for anom_type in TYPES:
        for d in D_GRID:
            rho = RHO_D25 if d > 1 else 0.0
            for dur_name, seg_len in DURATIONS.items():
                per_seed = {"raw_flatten": [], "raw_meanpool": [], "vision": []}
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    tr, te, lab = _build_split_v2(rng, d, rho, anom_type, seg_len)
                    per_seed["raw_flatten"].append(
                        _auc(tr.reshape(tr.shape[0], -1), te.reshape(te.shape[0], -1), lab))
                    per_seed["raw_meanpool"].append(
                        _auc(tr.mean(axis=1), te.mean(axis=1), lab))
                    tr_v = _vision_features(bb, tr)
                    te_v = _vision_features(bb, te)
                    per_seed["vision"].append(_auc(tr_v, te_v, lab))

                rec: dict[str, Any] = {"D": d, "rho": rho, "anomaly_type": anom_type,
                                       "duration": dur_name, "seg_len": seg_len,
                                       "noise": NOISE, "n_seeds": len(SEEDS)}
                for arm, vals in per_seed.items():
                    a = np.asarray(vals, dtype=np.float64)
                    rec[f"{arm}_mean"] = float(a.mean())
                    rec[f"{arm}_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
                rec["vision_minus_rawflatten"] = rec["vision_mean"] - rec["raw_flatten_mean"]
                records.append(rec)
                LOGGER.info(
                    "type=%-10s D=%-2d dur=%-5s(seg=%2d) | raw_flat=%.3f raw_mp=%.3f "
                    "vision=%.3f | Δ=%+.3f",
                    anom_type, d, dur_name, seg_len, rec["raw_flatten_mean"],
                    rec["raw_meanpool_mean"], rec["vision_mean"],
                    rec["vision_minus_rawflatten"],
                )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "meta": {
            "factor_added": "anomaly_duration (localized segment length)",
            "durations": DURATIONS, "noise": NOISE, "seeds": list(SEEDS),
            "held_fixed": "noise, N; swept: D, anomaly_type, duration",
            "backbone": "facebook/dinov2-base",
            "note": "amplitude/structural both localized so duration is a clean knob; "
                    "reuses v1 helpers for methodological comparability.",
        },
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (%d configs)", OUT_PATH, len(records))


if __name__ == "__main__":
    run()
