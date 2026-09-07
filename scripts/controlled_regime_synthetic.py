"""Controlled synthetic regime experiment for the VITS-AD rebuttal.

Reviewers (CmYN Q3, fUP4 W4, meta) note that the five public datasets confound
anomaly *type* with channel count, sequence length, and correlation, so the
observed raw-vs-vision "regime split" is correlational. This script isolates the
factors by generating synthetic multivariate series where we vary ONE factor at
a time and measure raw-space Mahalanobis vs. frozen-vision Mahalanobis AUC-ROC.

Controlled factors
------------------
- anomaly_type in {amplitude, structural}
    amplitude : anomalous segment is amplitude/energy-shifted (gain g), same shape.
    structural: anomalous segment changes temporal *shape* (frequency k*f0) but is
                re-standardised to the normal segment's per-channel mean/std, so its
                marginal amplitude statistics MATCH normal. Only shape differs.
- D (channel count) in {1, 5, 25}
- rho (inter-channel correlation) in {0.0, 0.9}   (ignored for D=1)

Everything else is held fixed: window L=100, base frequency f0, #normal/#anom
windows, standardisation (train stats), smoothing off, seed grid {42,123,456}.

Arms (both reuse the paper's own code paths)
--------------------------------------------
- raw_flatten  : RawMahalanobisScorer on flattened windows (W*D)   [paper's strongest raw]
- raw_meanpool : RawMahalanobisScorer on mean-pooled windows (D)
- vision       : line_plot render -> frozen DINOv2 patch tokens -> mean-pool over
                 patches -> RawMahalanobisScorer  [paper's distributional vision channel]

Hypothesis: vision-minus-raw AUC is >0 for structural anomalies and <0 for
amplitude anomalies, and the vision advantage grows as D shrinks (less channel
information lost to the 3-channel RGB rendering).

Output: JSON to paper/rebuttal_package/synthetic_regime_results.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.evaluation.metrics import compute_auc_roc
from src.models.backbone import VisionBackbone
from src.rendering.line_plot import render_line_plot
from scripts.run_raw_mahalanobis_baseline import RawMahalanobisScorer

LOGGER = logging.getLogger(__name__)

# ----------------------------- fixed config --------------------------------
L: int = 100                      # window length
F0: float = 3.0                   # base cycles per window for normal signal
STRUCT_K: float = 4.0             # frequency multiplier for structural burst
AMP_GAIN: float = 1.4             # subtle amplitude gain for amplitude anomaly
N_NORMAL_TRAIN: int = 120
N_NORMAL_TEST: int = 80
N_ANOM_TEST: int = 80
NOISE: float = 0.3                # overwritten by NOISE_GRID sweep
SEEDS: tuple[int, ...] = (42, 123, 456)
# Difficulty axis (a-priori, reported in full — NOT tuned until vision wins):
NOISE_GRID: tuple[float, ...] = (0.3, 0.6, 1.0)
# Dimensionality / RGB-compression axis (two extremes):
D_GRID: tuple[int, ...] = (1, 25)
RHO_GRID: tuple[float, ...] = (0.9,)   # for D>1
TYPES: tuple[str, ...] = ("amplitude", "structural")
OUT_PATH = Path("paper/rebuttal_package/synthetic_regime_results.json")


def _make_window(
    rng: np.random.Generator, d: int, rho: float, freq: float, gain: float
) -> npt.NDArray[np.float64]:
    """Generate one (L, D) window.

    Each channel is a phase-shifted sinusoid at `freq` cycles/window plus a
    shared latent (weight sqrt(rho)) and per-channel noise. `gain` scales the
    amplitude (used for amplitude anomalies).
    """
    t = np.linspace(0.0, 1.0, L, dtype=np.float64)
    shared = np.sin(2.0 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    cols = []
    for _ in range(d):
        phase = rng.uniform(0, 2 * np.pi)
        indep = np.sin(2.0 * np.pi * freq * t + phase)
        mix = np.sqrt(rho) * shared + np.sqrt(max(1.0 - rho, 0.0)) * indep
        col = gain * mix + NOISE * rng.standard_normal(L)
        cols.append(col)
    return np.stack(cols, axis=1)  # (L, D)


def _standardize(train: npt.NDArray[np.float64], test: npt.NDArray[np.float64]):
    """Standardise per channel using TRAIN statistics (no leakage)."""
    mu = train.reshape(-1, train.shape[-1]).mean(axis=0)
    sd = train.reshape(-1, train.shape[-1]).std(axis=0) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


def _build_split(rng: np.random.Generator, d: int, rho: float, anom_type: str):
    """Return standardized (train_win, test_win, test_labels)."""
    train = np.stack(
        [_make_window(rng, d, rho, F0, 1.0) for _ in range(N_NORMAL_TRAIN)]
    )
    test_normal = np.stack(
        [_make_window(rng, d, rho, F0, 1.0) for _ in range(N_NORMAL_TEST)]
    )
    anom = []
    for _ in range(N_ANOM_TEST):
        if anom_type == "amplitude":
            w = _make_window(rng, d, rho, F0, AMP_GAIN)  # global energy shift, same shape
        else:
            # structural: a LOCALIZED, amplitude-matched morphology burst at a
            # random position. The normal carrier is unchanged; inside a short
            # random sub-segment we overwrite the waveform with a different
            # (sawtooth) micro-morphology, then rescale that segment to the
            # normal per-channel mean/std so GLOBAL amplitude statistics match.
            # Only the local temporal *shape* differs -> visually salient but
            # small in global L2/covariance terms.
            w = _make_window(rng, d, rho, F0, 1.0)
            seg = rng.integers(12, 22)
            start = int(rng.integers(0, L - seg))
            tt = np.linspace(0.0, 1.0, seg, dtype=np.float64)
            # sawtooth burst: different harmonic content than the sine carrier
            burst = 2.0 * (tt * STRUCT_K % 1.0) - 1.0  # (seg,)
            for c in range(d):
                mu_c, sd_c = w[:, c].mean(), w[:, c].std() + 1e-8
                local = burst + NOISE * rng.standard_normal(seg)
                local = (local - local.mean()) / (local.std() + 1e-8)
                w[start:start + seg, c] = local * sd_c + mu_c
        anom.append(w)
    anom = np.stack(anom)

    test = np.concatenate([test_normal, anom], axis=0)
    labels = np.concatenate(
        [np.zeros(N_NORMAL_TEST, dtype=np.int64), np.ones(N_ANOM_TEST, dtype=np.int64)]
    )
    train_s, test_s = _standardize(train, test)
    return train_s, test_s, labels


def _vision_features(
    bb: VisionBackbone, windows: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Render each window as a line plot and mean-pool DINOv2 patch tokens."""
    feats = []
    batch_imgs: list[npt.NDArray[np.float32]] = []
    for i in range(windows.shape[0]):
        img = render_line_plot(windows[i].astype(np.float32))  # (3,224,224)
        batch_imgs.append(img)
        if len(batch_imgs) == 64 or i == windows.shape[0] - 1:
            arr = np.stack(batch_imgs).astype(np.float32)
            toks = bb.extract_patch_tokens_from_numpy(arr)  # (B, P, H)
            feats.append(toks.mean(axis=1))                 # mean-pool -> (B, H)
            batch_imgs = []
    return np.concatenate(feats, axis=0).astype(np.float64)


def _auc(train_f, test_f, labels) -> float:
    scorer = RawMahalanobisScorer()
    scorer.fit(train_f)
    scores = scorer.score(test_f)
    return compute_auc_roc(scores=scores.astype(np.float64), labels=labels)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    bb = VisionBackbone("facebook/dinov2-base")
    records: list[dict[str, Any]] = []

    configs: list[tuple[float, int, float, str]] = []
    for noise in NOISE_GRID:
        for anom_type in TYPES:
            for d in D_GRID:
                rhos = (0.0,) if d == 1 else RHO_GRID
                for rho in rhos:
                    configs.append((noise, d, rho, anom_type))

    global NOISE
    for (noise, d, rho, anom_type) in configs:
        NOISE = noise  # difficulty knob read by _make_window at call time
        per_seed: dict[str, list[float]] = {"raw_flatten": [], "raw_meanpool": [],
                                            "vision": []}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            tr, te, lab = _build_split(rng, d, rho, anom_type)
            tr_flat = tr.reshape(tr.shape[0], -1)
            te_flat = te.reshape(te.shape[0], -1)
            tr_mp = tr.mean(axis=1)
            te_mp = te.mean(axis=1)
            per_seed["raw_flatten"].append(_auc(tr_flat, te_flat, lab))
            per_seed["raw_meanpool"].append(_auc(tr_mp, te_mp, lab))
            tr_v = _vision_features(bb, tr)
            te_v = _vision_features(bb, te)
            per_seed["vision"].append(_auc(tr_v, te_v, lab))

        rec: dict[str, Any] = {"noise": noise, "D": d, "rho": rho,
                               "anomaly_type": anom_type, "n_seeds": len(SEEDS)}
        for arm, vals in per_seed.items():
            a = np.asarray(vals, dtype=np.float64)
            rec[f"{arm}_mean"] = float(a.mean())
            rec[f"{arm}_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        rec["vision_minus_rawflatten"] = rec["vision_mean"] - rec["raw_flatten_mean"]
        records.append(rec)
        LOGGER.info(
            "noise=%.1f type=%-10s D=%-2d | raw_flat=%.3f raw_mp=%.3f vision=%.3f | "
            "Δ(vision-rawflat)=%+.3f",
            noise, anom_type, d, rec["raw_flatten_mean"], rec["raw_meanpool_mean"],
            rec["vision_mean"], rec["vision_minus_rawflatten"],
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "meta": {
            "L": L, "f0": F0, "struct_k": STRUCT_K, "amp_gain": AMP_GAIN,
            "n_normal_train": N_NORMAL_TRAIN, "n_normal_test": N_NORMAL_TEST,
            "n_anom_test": N_ANOM_TEST, "noise": NOISE, "seeds": list(SEEDS),
            "backbone": "facebook/dinov2-base",
            "vision_arm": "line_plot -> DINOv2 patch tokens -> mean-pool -> LedoitWolf Mahalanobis",
            "raw_arm": "LedoitWolf Mahalanobis on flattened / mean-pooled windows",
        },
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (%d configs)", OUT_PATH, len(records))


if __name__ == "__main__":
    run()
