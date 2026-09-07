"""Label-free proxies for the amplitude-subtlety regime law (deployability).

`amplitude_subtlety_law.md` established that raw-space Mahalanobis is an
amplitude-extremity detector: its accuracy tracks

    z_ratio = median(|z| on anomalous points) / median(|z| on normal points)

with Spearman 0.60, while the vision arm is nearly insensitive (0.18). That
statistic needs **labels**, so it characterises a regime but cannot select a
detector at deployment time — exactly the "descriptive, not deployable"
criticism reviewers raised about the paper's regime classifier.

This script searches for **label-free** proxies computable from the training
split plus unlabeled test data, and evaluates whether a proxy can drive a
deployable selection rule ("render, or score raw?").

Candidate proxies (none uses labels):
    tail_ratio   q99.9 / median of the raw-Mahalanobis test scores
    kurtosis     excess kurtosis of the raw-Mahalanobis test scores
    excursion    max|z| on test / max|z| on train
    novelty_frac fraction of test timesteps with |z| beyond the train max
    z_disp       IQR of |z| on test divided by its median

Evaluation:
    1. rank correlation of each proxy with the (label-derived) z_ratio;
    2. rank correlation with the realised advantage delta = vision - raw AUC-ROC;
    3. a deployable rule "use vision iff proxy < threshold", scored by the
       realised mean AUC-ROC and compared against always-raw, always-vision and
       the oracle, with the threshold chosen by leave-one-out cross-validation
       so the reported number is not fitted on the series it grades.

Output: results/regime_proxy/proxy_search.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from scripts.run_raw_mahalanobis_baseline import RawMahalanobisScorer
from scripts.run_tsb_ad_raw_audit import load_series, parse_split
from src.data.base import create_sliding_windows

LOGGER = logging.getLogger(__name__)

WINDOW_SIZE = 100
MIN_STRIDE = 10
MAX_WINDOWS = 1000
PAIRED_ROOT = Path("results/tsb_ad_vision_paired")
DATA_ROOT = Path("data/tsb_ad")
OUT_DIR = Path("results/regime_proxy")

PROXY_NAMES = ("tail_ratio", "kurtosis", "excursion", "novelty_frac", "z_disp")


def _windows_for(path: Path) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64],
    npt.NDArray[np.float64], npt.NDArray[np.float64],
    npt.NDArray[np.float64], npt.NDArray[np.int64]
]:
    """Recreate the stage-(b) windows, train/test |z|, and the full |z| + labels.

    Returns everything a caller needs from one CSV read: the train/test windows,
    the per-timestep |z| for the train and test splits, the |z| over the whole
    series, and the point labels.
    """
    train_end, _ = parse_split(path)
    values, labels = load_series(path)
    n_test_points = values.shape[0] - train_end
    stride = max(MIN_STRIDE, int(np.ceil(n_test_points / MAX_WINDOWS)))

    train_raw, test_raw = values[:train_end], values[train_end:]
    mu = train_raw.mean(axis=0)
    sigma = train_raw.std(axis=0) + 1e-8
    train_norm = (train_raw - mu) / sigma
    test_norm = (test_raw - mu) / sigma

    train_windows, _ = create_sliding_windows(
        train_norm, labels[:train_end], WINDOW_SIZE, stride
    )
    test_windows, win_labels = create_sliding_windows(
        test_norm, labels[train_end:], WINDOW_SIZE, stride
    )
    # Aggregate across channels so |z| is per timestep and stays aligned with
    # the point labels (ravel() would give T*D entries for multivariate series).
    z_all = np.abs((values - mu) / sigma).mean(axis=1)
    return (train_windows, test_windows, win_labels,
            np.abs(train_norm).ravel(), np.abs(test_norm).ravel(),
            z_all, labels)


def compute_proxies(path: Path) -> dict[str, float] | None:
    """Compute all label-free proxies plus the label-derived z_ratio reference."""
    (train_windows, test_windows, win_labels,
     z_train, z_test, z_all, labels_full) = _windows_for(path)
    if train_windows.shape[0] < 2 or test_windows.shape[0] < 2:
        return None
    if win_labels.sum() in (0, win_labels.shape[0]):
        return None

    train_feat = train_windows.reshape(train_windows.shape[0], -1).astype(np.float64)
    test_feat = test_windows.reshape(test_windows.shape[0], -1).astype(np.float64)
    scorer = RawMahalanobisScorer()
    scorer.fit(train_feat)
    raw_scores = scorer.score(test_feat)
    if not np.all(np.isfinite(raw_scores)):
        return None

    median_score = float(np.median(raw_scores)) + 1e-12
    z_train_max = float(z_train.max()) + 1e-12
    z_test_median = float(np.median(z_test)) + 1e-12

    proxies: dict[str, float] = {
        "tail_ratio": float(np.quantile(raw_scores, 0.999) / median_score),
        "kurtosis": float(stats.kurtosis(raw_scores, fisher=True, bias=False)),
        "excursion": float(z_test.max() / z_train_max),
        "novelty_frac": float((z_test > z_train.max()).mean()),
        "z_disp": float(stats.iqr(z_test) / z_test_median),
    }

    # Label-derived reference (NOT usable at deployment; for validation only).
    mask = labels_full.astype(bool)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return None
    proxies["z_ratio_labelled"] = float(
        np.median(z_all[mask]) / (np.median(z_all[~mask]) + 1e-9)
    )
    return proxies


def _spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _rule_score(records: list[dict[str, Any]], proxy: str,
                threshold: float) -> float:
    """Mean realised AUC-ROC of 'use vision iff proxy < threshold'."""
    chosen = [
        r["vision_auc"] if r["proxies"][proxy] < threshold else r["raw_auc"]
        for r in records
    ]
    return float(np.mean(chosen))


def _loo_rule(records: list[dict[str, Any]], proxy: str) -> dict[str, float]:
    """Leave-one-out evaluation of the threshold rule (no fitting on the graded series)."""
    values = sorted({r["proxies"][proxy] for r in records})
    candidates = [
        (values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)
    ] or values
    realised: list[float] = []
    picked_vision = 0
    for index in range(len(records)):
        train = records[:index] + records[index + 1:]
        best = max(candidates, key=lambda t: _rule_score(train, proxy, t))
        held = records[index]
        use_vision = held["proxies"][proxy] < best
        picked_vision += int(use_vision)
        realised.append(held["vision_auc"] if use_vision else held["raw_auc"])
    return {
        "loo_mean_auc": float(np.mean(realised)),
        "loo_vision_fraction": picked_vision / len(records),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    paired = json.loads((PAIRED_ROOT / "per_series.json").read_text())
    records: list[dict[str, Any]] = []

    for index, entry in enumerate(paired, start=1):
        if not (entry.get("vision") and entry.get("raw_flatten")):
            continue
        path = DATA_ROOT / entry["subset"] / entry["series"]
        try:
            proxies = compute_proxies(path)
        except Exception as exc:
            LOGGER.warning("%s failed: %s", entry["series"], exc)
            proxies = None
        if proxies is None:
            continue
        records.append({
            "subset": entry["subset"],
            "series": entry["series"],
            "stratum": entry["stratum"],
            "n_channels": entry["n_channels"],
            "proxies": proxies,
            "raw_auc": float(entry["raw_flatten"]["AUC-ROC"]),
            "vision_auc": float(entry["vision"]["AUC-ROC"]),
            "delta": float(entry["vision"]["AUC-ROC"] - entry["raw_flatten"]["AUC-ROC"]),
        })
        if index % 20 == 0:
            LOGGER.info("%d/%d processed (%d usable)", index, len(paired), len(records))

    LOGGER.info("usable series: %d", len(records))
    if len(records) < 10:
        LOGGER.error("too few series to analyse")
        return

    summary: dict[str, Any] = {"n_series": len(records), "proxies": {}}
    always_raw = float(np.mean([r["raw_auc"] for r in records]))
    always_vision = float(np.mean([r["vision_auc"] for r in records]))
    oracle = float(np.mean([max(r["raw_auc"], r["vision_auc"]) for r in records]))
    summary["baselines"] = {
        "always_raw": always_raw,
        "always_vision": always_vision,
        "oracle": oracle,
    }
    LOGGER.info("baselines: always_raw=%.4f always_vision=%.4f oracle=%.4f",
                always_raw, always_vision, oracle)

    z_ratio = [r["proxies"]["z_ratio_labelled"] for r in records]
    deltas = [r["delta"] for r in records]
    rho_z, p_z = _spearman(z_ratio, deltas)
    summary["z_ratio_labelled"] = {
        "spearman_with_delta": rho_z, "p": p_z,
    }
    LOGGER.info("z_ratio (labelled) vs delta: rho=%.3f p=%.3g", rho_z, p_z)

    for proxy in PROXY_NAMES:
        xs = [r["proxies"][proxy] for r in records]
        rho_zr, p_zr = _spearman(xs, z_ratio)
        rho_d, p_d = _spearman(xs, deltas)
        loo = _loo_rule(records, proxy)
        summary["proxies"][proxy] = {
            "spearman_with_z_ratio": rho_zr, "p_z_ratio": p_zr,
            "spearman_with_delta": rho_d, "p_delta": p_d,
            **loo,
        }
        LOGGER.info(
            "%-13s rho(z_ratio)=%+.3f (p=%.3g)  rho(delta)=%+.3f (p=%.3g)  "
            "LOO_AUC=%.4f (vision %.0f%%)",
            proxy, rho_zr, p_zr, rho_d, p_d,
            loo["loo_mean_auc"], 100 * loo["loo_vision_fraction"],
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "proxy_search.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote %s", OUT_DIR / "proxy_search.json")


if __name__ == "__main__":
    main()
