"""Consolidated controlled regime study — full factorial (roadmap 2.1).

Unifies the earlier v1 (channel count, noise) and v2 (duration) probes into one
principled factorial, and adds the non-stationarity factor the reviewers asked for.
Orthogonal factors, everything else held fixed:

  D        in {1, 25}            channel count / RGB-compression axis
  type     in {amplitude, structural}
  locality in {whole, local}     anomaly spans whole window vs a short segment
                                 (this is exactly what differed between v1 and v2,
                                  now made an explicit factor)
  drift    in {stationary, drift} per-window random linear trend (non-stationarity)

Both arms reuse v1's renderer / frozen-DINOv2 / Ledoit-Wolf-Mahalanobis helpers, so
results are methodologically comparable. Hypothesis for drift: `render_line_plot`
normalizes each window, so the vision arm should tolerate per-window trends better
than raw-flatten Mahalanobis, which compares absolute values against train statistics.

Output: paper/rebuttal_package/synthetic_regime_full.json
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

NOISE: float = 0.6
LOCAL_SEG: int = 15
DRIFT_LEVEL: float = 1.5
N_NORMAL_TRAIN: int = 120
N_NORMAL_TEST: int = 80
N_ANOM_TEST: int = 80
SEEDS: tuple[int, ...] = (42, 123, 456)

D_GRID: tuple[int, ...] = (1, 25)
TYPES: tuple[str, ...] = ("amplitude", "structural")
LOCALITIES: tuple[str, ...] = ("whole", "local")
DRIFTS: dict[str, float] = {"stationary": 0.0, "drift": DRIFT_LEVEL}
RHO_D25: float = 0.9

OUT_PATH = Path("paper/rebuttal_package/synthetic_regime_full.json")


def _add_drift(w: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """Add an independent random linear trend per channel (non-stationarity)."""
    if level <= 0.0:
        return w
    n, d = w.shape
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
    slopes = rng.uniform(-level, level, size=(1, d))
    return w + t * slopes


def _sawtooth(rng: np.random.Generator, seg: int) -> np.ndarray:
    tt = np.linspace(0.0, 1.0, seg, dtype=np.float64)
    b = 2.0 * (tt * v1.STRUCT_K % 1.0) - 1.0 + NOISE * rng.standard_normal(seg)
    return (b - b.mean()) / (b.std() + 1e-8)


def _make_anomaly(rng, d, rho, anom_type, locality):
    """One anomalous window (before drift/standardization)."""
    if anom_type == "amplitude":
        if locality == "whole":
            return _make_window(rng, d, rho, v1.F0, v1.AMP_GAIN)
        w = _make_window(rng, d, rho, v1.F0, 1.0)
        s = int(rng.integers(0, v1.L - LOCAL_SEG + 1))
        w[s:s + LOCAL_SEG, :] *= v1.AMP_GAIN
        return w
    # structural
    if locality == "whole":
        w = _make_window(rng, d, rho, v1.F0 * v1.STRUCT_K, 1.0)
        return (w - w.mean(axis=0)) / (w.std(axis=0) + 1e-8)  # amplitude-matched
    w = _make_window(rng, d, rho, v1.F0, 1.0)
    s = int(rng.integers(0, v1.L - LOCAL_SEG + 1))
    for c in range(d):
        mu_c, sd_c = w[:, c].mean(), w[:, c].std() + 1e-8
        w[s:s + LOCAL_SEG, c] = _sawtooth(rng, LOCAL_SEG) * sd_c + mu_c
    return w


def _build_split(rng, d, rho, anom_type, locality, drift):
    v1.NOISE = NOISE
    train = np.stack([_add_drift(_make_window(rng, d, rho, v1.F0, 1.0), rng, drift)
                      for _ in range(N_NORMAL_TRAIN)])
    test_normal = np.stack([_add_drift(_make_window(rng, d, rho, v1.F0, 1.0), rng, drift)
                            for _ in range(N_NORMAL_TEST)])
    anom = np.stack([_add_drift(_make_anomaly(rng, d, rho, anom_type, locality), rng, drift)
                     for _ in range(N_ANOM_TEST)])
    test = np.concatenate([test_normal, anom], axis=0)
    labels = np.concatenate([np.zeros(N_NORMAL_TEST, dtype=np.int64),
                             np.ones(N_ANOM_TEST, dtype=np.int64)])
    return (*_standardize(train, test), labels)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    bb = VisionBackbone("facebook/dinov2-base")
    records: list[dict[str, Any]] = []
    for anom_type in TYPES:
        for d in D_GRID:
            rho = RHO_D25 if d > 1 else 0.0
            for locality in LOCALITIES:
                for drift_name, drift in DRIFTS.items():
                    per = {"raw_flatten": [], "raw_meanpool": [], "vision": []}
                    for seed in SEEDS:
                        rng = np.random.default_rng(seed)
                        tr, te, lab = _build_split(rng, d, rho, anom_type, locality, drift)
                        per["raw_flatten"].append(
                            _auc(tr.reshape(tr.shape[0], -1), te.reshape(te.shape[0], -1), lab))
                        per["raw_meanpool"].append(_auc(tr.mean(1), te.mean(1), lab))
                        per["vision"].append(
                            _auc(_vision_features(bb, tr), _vision_features(bb, te), lab))
                    rec: dict[str, Any] = {"D": d, "anomaly_type": anom_type,
                                           "locality": locality, "drift": drift_name,
                                           "noise": NOISE, "n_seeds": len(SEEDS)}
                    for arm, vals in per.items():
                        a = np.asarray(vals, dtype=np.float64)
                        rec[f"{arm}_mean"] = float(a.mean())
                        rec[f"{arm}_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
                    rec["vision_minus_rawflatten"] = rec["vision_mean"] - rec["raw_flatten_mean"]
                    records.append(rec)
                    LOGGER.info(
                        "type=%-10s D=%-2d loc=%-5s drift=%-10s | raw_flat=%.3f "
                        "raw_mp=%.3f vision=%.3f | Δ=%+.3f",
                        anom_type, d, locality, drift_name, rec["raw_flatten_mean"],
                        rec["raw_meanpool_mean"], rec["vision_mean"],
                        rec["vision_minus_rawflatten"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "meta": {"factors": "D x type x locality x drift", "noise": NOISE,
                 "local_seg": LOCAL_SEG, "drift_level": DRIFT_LEVEL, "seeds": list(SEEDS),
                 "backbone": "facebook/dinov2-base",
                 "note": "consolidates v1(channel/noise)+v2(duration); locality explicit; "
                         "adds per-window random-trend non-stationarity."},
        "records": records}, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (%d configs)", OUT_PATH, len(records))


if __name__ == "__main__":
    run()
