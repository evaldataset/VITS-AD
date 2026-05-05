"""Regime map and entity-level gain predictor for VITS.

For each available entity/dataset, computes signal features and VITS vs raw
Mahalanobis AUC-ROC delta, then generates:
  - Figure A: 2D regime scatter map (paper/figures/regime_map.pdf)
  - Figure B: Spearman correlation bar chart (paper/figures/gain_predictor.pdf)
  - Logistic regression: predict(delta > 0) from features
  - results/regime_analysis/regime_map.json
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats

matplotlib.use("Agg")

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DATA_RAW = ROOT / "data" / "raw"
DATA_UCR = ROOT / "data" / "UCR"
PAPER_FIG = ROOT / "paper" / "figures"
OUT_DIR = RESULTS / "regime_analysis"


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _spectral_entropy(signal: NDArray[np.float64]) -> float:
    """Compute normalised spectral entropy of a 1-D signal."""
    if signal.std() < 1e-10:
        return 0.0
    freqs = np.fft.rfft(signal - signal.mean())
    power = np.abs(freqs) ** 2
    total = power.sum()
    if total < 1e-30:
        return 0.0
    p = power / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(p)) if len(p) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _periodicity(signal: NDArray[np.float64]) -> float:
    """Return dominant autocorrelation peak strength (lag 1 … T/2).

    Uses FFT-based autocorrelation (O(N log N)) to avoid O(N^2) slowness on
    long time series.
    """
    if signal.std() < 1e-10:
        return 0.0
    n = len(signal)
    if n < 4:
        return 0.0
    norm = signal - signal.mean()
    # FFT-based circular autocorrelation, zero-padded to avoid wrap-around
    fft_len = 2 * n
    f = np.fft.rfft(norm, n=fft_len)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    ac = ac / (ac[0] + 1e-30)
    # look at lags 1 .. n//2
    lags = ac[1 : n // 2]
    if len(lags) == 0:
        return 0.0
    return float(np.max(np.abs(lags)))


def _compute_signal_features(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> dict[str, float]:
    """Compute regime features from (T, D) data and (T,) labels."""
    n_channels = data.shape[1]
    anomaly_ratio = float(labels.mean())

    # mean over channels for scalar stats
    mean_entropy = float(np.mean([_spectral_entropy(data[:, c]) for c in range(n_channels)]))
    mean_period = float(np.mean([_periodicity(data[:, c]) for c in range(n_channels)]))
    rendering_compression = n_channels / 3.0

    return {
        "n_channels": float(n_channels),
        "anomaly_ratio": anomaly_ratio,
        "periodicity": mean_period,
        "signal_entropy": mean_entropy,
        "rendering_compression": rendering_compression,
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_smd_entity(entity: str) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    """Load SMD test data and labels for one entity name like 'machine-1-1'."""
    test_path = DATA_RAW / "smd" / "test" / f"{entity}.txt"
    label_path = DATA_RAW / "smd" / "test_label" / f"{entity}.txt"
    if not test_path.exists() or not label_path.exists():
        LOGGER.warning("SMD data not found for %s", entity)
        return None
    data = np.loadtxt(str(test_path), delimiter=",", dtype=np.float64)
    labels = np.loadtxt(str(label_path), dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data, labels


def _load_psm() -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    try:
        import pandas as pd  # type: ignore[import]
    except ImportError:
        LOGGER.warning("pandas not available; skipping PSM")
        return None
    test_path = DATA_RAW / "psm" / "test.csv"
    label_path = DATA_RAW / "psm" / "test_label.csv"
    if not test_path.exists():
        LOGGER.warning("PSM data not found at %s", test_path)
        return None
    data_df = pd.read_csv(str(test_path))
    data = data_df.drop(columns=["timestamp_(min)"], errors="ignore").values.astype(np.float64)
    if label_path.exists():
        labels = pd.read_csv(str(label_path))["label"].values.astype(np.int64)
    else:
        labels = np.zeros(len(data), dtype=np.int64)
    return data, labels


def _load_msl_smap(dataset: str) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    base = DATA_RAW / dataset.lower()
    test_path = base / "test.npy"
    label_path = base / "test_label.npy"
    # also try processed
    if not test_path.exists():
        base = ROOT / "data" / "processed" / dataset.lower()
        test_path = base / "test.npy"
        label_path = base / "test_label.npy"
    if not test_path.exists():
        LOGGER.warning("%s data not found at %s", dataset, test_path)
        return None
    data = np.load(str(test_path)).astype(np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if label_path.exists():
        labels = np.load(str(label_path)).astype(np.int64)
    else:
        labels = np.zeros(len(data), dtype=np.int64)
    return data, labels


def _load_ucr_series(filepath: str) -> tuple[NDArray[np.float64], NDArray[np.int64], int, int] | None:
    """Load a UCR series file; returns (data_1d, labels, anomaly_start, anomaly_end)."""
    fpath = Path(filepath)
    if not fpath.exists():
        fpath = ROOT / filepath
    if not fpath.exists():
        LOGGER.warning("UCR file not found: %s", filepath)
        return None
    raw = np.loadtxt(str(fpath), dtype=np.float64)
    # parse anomaly positions from filename: *_trainLen_anomalyStart_anomalyEnd.txt
    parts = fpath.stem.split("_")
    try:
        anomaly_end = int(parts[-1])
        anomaly_start = int(parts[-2])
    except (ValueError, IndexError):
        LOGGER.warning("Cannot parse anomaly positions from %s", fpath.name)
        return None
    data = raw.reshape(-1, 1)
    labels = np.zeros(len(data), dtype=np.int64)
    labels[anomaly_start:anomaly_end] = 1
    return data, labels, anomaly_start, anomaly_end


# ---------------------------------------------------------------------------
# Result loaders
# ---------------------------------------------------------------------------

def _load_smd_vits_auc(entity: str) -> float | None:
    """Load VITS AUC-ROC for SMD entity from benchmark_smd_spatial/line_plot."""
    metrics_path = RESULTS / "benchmark_smd_spatial" / entity / "line_plot" / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        m = json.load(f)
    return float(m.get("auc_roc", np.nan))


def _load_smd_raw_auc(entity: str) -> float | None:
    """Load raw Mahalanobis AUC-ROC for SMD entity."""
    # try per-entity dir first
    mp = RESULTS / "raw_mahalanobis" / f"smd_{entity}" / "mean_pooled" / "metrics.json"
    if mp.exists():
        with open(mp) as f:
            m = json.load(f)
        return float(m.get("auc_roc", np.nan))
    # fall back to summary.json
    summary = RESULTS / "raw_mahalanobis" / "summary.json"
    if summary.exists():
        with open(summary) as f:
            s = json.load(f)
        key = f"smd_{entity}"
        if key in s:
            return float(s[key].get("mean_pooled", {}).get("auc_roc", np.nan))
    return None


def _load_dataset_vits_auc(dataset: str, renderer: str = "lp") -> float | None:
    """Load VITS AUC-ROC for PSM/MSL/SMAP from spatial_pilot."""
    key = f"{dataset.lower()}_{renderer}"
    metrics_path = RESULTS / "spatial_pilot" / key / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        return float(m.get("auc_roc", np.nan))
    return None


def _load_dataset_raw_auc(dataset: str) -> float | None:
    """Load raw Mahalanobis AUC-ROC for PSM/MSL/SMAP."""
    mp = RESULTS / "raw_mahalanobis" / dataset.lower() / "mean_pooled" / "metrics.json"
    if mp.exists():
        with open(mp) as f:
            m = json.load(f)
        return float(m.get("auc_roc", np.nan))
    # fall back summary
    summary = RESULTS / "raw_mahalanobis" / "summary.json"
    if summary.exists():
        with open(summary) as f:
            s = json.load(f)
        key = dataset.lower()
        if key in s:
            return float(s[key].get("mean_pooled", {}).get("auc_roc", np.nan))
    return None


def _load_ucr_auc_pair(series_name: str) -> tuple[float, float] | None:
    """Return (vits_auc, raw_auc) for a UCR series name."""
    vits_path = RESULTS / "ucr_expanded" / "per_series" / f"{series_name}.json"
    raw_path = RESULTS / "ucr_expanded_raw" / "per_series" / f"{series_name}.json"
    if not vits_path.exists() or not raw_path.exists():
        return None
    with open(vits_path) as f:
        v = json.load(f)
    with open(raw_path) as f:
        r = json.load(f)
    vits_auc = v.get("auc_roc", {})
    # prefer Dual, then Dist, then Traj
    if isinstance(vits_auc, dict):
        vits_val = vits_auc.get("VITS_Dual") or vits_auc.get("VITS_Dist") or vits_auc.get("VITS_Traj")
    else:
        vits_val = float(vits_auc)
    raw_auc = r.get("auc_roc", {})
    if isinstance(raw_auc, dict):
        raw_val = raw_auc.get("RawMaha_MeanPooled")
    else:
        raw_val = float(raw_auc)
    if vits_val is None or raw_val is None:
        return None
    return float(vits_val), float(raw_val)


# ---------------------------------------------------------------------------
# Entity collection
# ---------------------------------------------------------------------------

def _collect_smd_entities() -> list[dict[str, Any]]:
    entities = []
    smd_spatial = RESULTS / "benchmark_smd_spatial"
    if not smd_spatial.exists():
        return entities
    for entity_dir in sorted(smd_spatial.iterdir()):
        if not entity_dir.is_dir():
            continue
        entity = entity_dir.name
        vits_auc = _load_smd_vits_auc(entity)
        raw_auc = _load_smd_raw_auc(entity)
        if vits_auc is None or raw_auc is None:
            LOGGER.warning("Missing AUC for SMD %s (vits=%s, raw=%s)", entity, vits_auc, raw_auc)
            continue
        loaded = _load_smd_entity(entity)
        if loaded is None:
            continue
        data, labels = loaded
        feats = _compute_signal_features(data, labels)
        delta = vits_auc - raw_auc
        entities.append({
            "name": entity,
            "dataset": "SMD",
            "vits_auc": vits_auc,
            "raw_auc": raw_auc,
            "delta": delta,
            **feats,
        })
        LOGGER.info("SMD %-16s  vits=%.4f  raw=%.4f  delta=%+.4f", entity, vits_auc, raw_auc, delta)
    return entities


def _collect_multivar_datasets() -> list[dict[str, Any]]:
    records = []
    for dataset, n_ch in [("PSM", 25), ("MSL", 55), ("SMAP", 25)]:
        vits_auc = _load_dataset_vits_auc(dataset, renderer="lp")
        raw_auc = _load_dataset_raw_auc(dataset)
        if vits_auc is None or raw_auc is None:
            LOGGER.warning("Missing AUC for %s (vits=%s, raw=%s)", dataset, vits_auc, raw_auc)
            continue

        # Try loading actual data for features
        if dataset == "PSM":
            loaded = _load_psm()
        else:
            loaded = _load_msl_smap(dataset)

        if loaded is not None:
            data, labels = loaded
            feats = _compute_signal_features(data, labels)
        else:
            # Use known n_channels and placeholder features
            feats = {
                "n_channels": float(n_ch),
                "anomaly_ratio": float("nan"),
                "periodicity": float("nan"),
                "signal_entropy": float("nan"),
                "rendering_compression": n_ch / 3.0,
            }

        delta = vits_auc - raw_auc
        records.append({
            "name": dataset,
            "dataset": dataset,
            "vits_auc": vits_auc,
            "raw_auc": raw_auc,
            "delta": delta,
            **feats,
        })
        LOGGER.info("%-6s  vits=%.4f  raw=%.4f  delta=%+.4f", dataset, vits_auc, raw_auc, delta)
    return records


def _collect_ucr_entities() -> list[dict[str, Any]]:
    per_series_dir = RESULTS / "ucr_expanded" / "per_series"
    if not per_series_dir.exists():
        return []
    records = []
    # also load from summary for file paths
    raw_summary_path = RESULTS / "ucr_expanded_raw" / "summary.json"
    file_map: dict[str, str] = {}
    if raw_summary_path.exists():
        with open(raw_summary_path) as f:
            raw_summary = json.load(f)
        for entry in raw_summary.get("per_series", []):
            file_map[entry["series"]] = entry.get("file", "")

    for json_file in sorted(per_series_dir.iterdir()):
        if json_file.suffix != ".json":
            continue
        series_name = json_file.stem
        auc_pair = _load_ucr_auc_pair(series_name)
        if auc_pair is None:
            continue
        vits_auc, raw_auc = auc_pair

        # load actual series data for features
        filepath = file_map.get(series_name, "")
        feats: dict[str, float] = {
            "n_channels": 1.0,
            "anomaly_ratio": float("nan"),
            "periodicity": float("nan"),
            "signal_entropy": float("nan"),
            "rendering_compression": 1 / 3.0,
        }
        if filepath:
            result = _load_ucr_series(filepath)
            if result is not None:
                data, labels, _, _ = result
                feats = _compute_signal_features(data, labels)

        delta = vits_auc - raw_auc
        records.append({
            "name": series_name,
            "dataset": "UCR",
            "vits_auc": vits_auc,
            "raw_auc": raw_auc,
            "delta": delta,
            **feats,
        })
    LOGGER.info("UCR: collected %d series", len(records))
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "n_channels",
    "anomaly_ratio",
    "periodicity",
    "signal_entropy",
    "rendering_compression",
]


def _spearman_correlations(records: list[dict[str, Any]]) -> dict[str, float]:
    deltas = np.array([r["delta"] for r in records], dtype=np.float64)
    corrs: dict[str, float] = {}
    for feat in FEATURE_COLS:
        vals = np.array([r.get(feat, np.nan) for r in records], dtype=np.float64)
        mask = np.isfinite(vals) & np.isfinite(deltas)
        if mask.sum() < 3:
            corrs[feat] = float("nan")
            continue
        rho, _ = stats.spearmanr(vals[mask], deltas[mask])
        corrs[feat] = float(rho)
    return corrs


def _fit_logistic_regression(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit logistic regression predicting delta > 0 from features."""
    from sklearn.linear_model import LogisticRegression  # type: ignore[import]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import]
    from sklearn.pipeline import Pipeline  # type: ignore[import]
    from sklearn.metrics import accuracy_score  # type: ignore[import]

    rows = []
    targets = []
    groups = []
    for r in records:
        row = [r.get(f, np.nan) for f in FEATURE_COLS]
        if any(not np.isfinite(v) for v in row):
            continue
        rows.append(row)
        targets.append(1 if r["delta"] > 0 else 0)
        groups.append(r["dataset"])

    if len(rows) < 5:
        LOGGER.warning("Too few complete records (%d) for logistic regression", len(rows))
        return {"n_samples": len(rows), "error": "insufficient data"}

    X = np.array(rows, dtype=np.float64)
    y = np.array(targets, dtype=np.int64)
    groups_arr = np.array(groups, dtype=object)

    def _make_pipe() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])

    pipe = _make_pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)

    y_pred = pipe.predict(X)
    in_sample_acc = float(accuracy_score(y, y_pred))
    coefs = dict(zip(FEATURE_COLS, pipe.named_steps["lr"].coef_[0].tolist()))

    # --- Leave-One-Dataset-Out (LODO) cross-validation ---
    # Train on all datasets except one, evaluate on the held-out dataset.
    # This guards against in-sample overfitting on the dataset axis.
    unique_groups = sorted(set(groups))
    fold_results: list[dict[str, Any]] = []
    all_lodo_preds = np.zeros_like(y)
    all_lodo_mask = np.zeros_like(y, dtype=bool)
    for held_out in unique_groups:
        train_mask = groups_arr != held_out
        test_mask = groups_arr == held_out
        if train_mask.sum() < 5 or test_mask.sum() == 0:
            continue
        # Skip degenerate folds where training labels collapse to one class
        if len(set(y[train_mask].tolist())) < 2:
            fold_results.append({
                "held_out": held_out,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "accuracy": None,
                "note": "degenerate (single-class train)",
            })
            continue
        fold_pipe = _make_pipe()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_pipe.fit(X[train_mask], y[train_mask])
        fold_pred = fold_pipe.predict(X[test_mask])
        fold_acc = float(accuracy_score(y[test_mask], fold_pred))
        all_lodo_preds[test_mask] = fold_pred
        all_lodo_mask[test_mask] = True
        fold_results.append({
            "held_out": held_out,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "accuracy": fold_acc,
        })

    valid_folds = [f for f in fold_results if f.get("accuracy") is not None]
    if valid_folds:
        fold_accs = np.array([f["accuracy"] for f in valid_folds])
        lodo_mean = float(fold_accs.mean())
        lodo_std = float(fold_accs.std(ddof=1)) if len(fold_accs) > 1 else 0.0
        # 95% CI from Wilson interval on pooled LODO predictions
        if all_lodo_mask.any():
            pooled_acc = float(accuracy_score(y[all_lodo_mask], all_lodo_preds[all_lodo_mask]))
            n_pool = int(all_lodo_mask.sum())
            # Normal approximation 95% CI
            se = np.sqrt(pooled_acc * (1 - pooled_acc) / n_pool) if n_pool > 0 else 0.0
            ci95 = (max(0.0, pooled_acc - 1.96 * se), min(1.0, pooled_acc + 1.96 * se))
        else:
            pooled_acc, n_pool, ci95 = float("nan"), 0, (float("nan"), float("nan"))
    else:
        lodo_mean = lodo_std = pooled_acc = float("nan")
        n_pool = 0
        ci95 = (float("nan"), float("nan"))

    LOGGER.info(
        "Logistic regression: in_sample=%.3f  LODO_mean=%.3f  LODO_std=%.3f  "
        "pooled_LODO=%.3f  95%%CI=[%.3f, %.3f]  n=%d",
        in_sample_acc, lodo_mean, lodo_std, pooled_acc, ci95[0], ci95[1], len(y),
    )
    for feat, coef in sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True):
        LOGGER.info("  %-28s coef=%+.4f", feat, coef)

    return {
        "n_samples": len(y),
        "in_sample_accuracy": in_sample_acc,
        "lodo_mean_accuracy": lodo_mean,
        "lodo_std_accuracy": lodo_std,
        "lodo_pooled_accuracy": pooled_acc,
        "lodo_pooled_n": n_pool,
        "lodo_95ci_low": ci95[0],
        "lodo_95ci_high": ci95[1],
        "lodo_folds": fold_results,
        "coefficients": coefs,
        "class_balance": {"n_positive": int(y.sum()), "n_negative": int((1 - y).sum())},
        # Backward compatibility: keep "accuracy" key pointing to in_sample
        # so callers that read it do not break, but flag it explicitly.
        "accuracy": in_sample_acc,
        "accuracy_kind": "in_sample (use lodo_pooled_accuracy for held-out claim)",
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_regime_map(records: list[dict[str, Any]], out_path: Path) -> None:
    """Figure A: 2D regime scatter map."""
    fig, ax = plt.subplots(figsize=(7.0, 2.5))

    x_key = "n_channels"
    y_key = "signal_entropy"

    xs, ys, ds, szs, names, datasets = [], [], [], [], [], []
    for r in records:
        xv = r.get(x_key)
        yv = r.get(y_key)
        dv = r.get("delta")
        if xv is None or yv is None or dv is None:
            continue
        if not (np.isfinite(xv) and np.isfinite(yv) and np.isfinite(dv)):
            continue
        xs.append(xv)
        ys.append(yv)
        ds.append(dv)
        szs.append(abs(dv) * 600 + 20)
        names.append(r["name"])
        datasets.append(r["dataset"])

    if not xs:
        LOGGER.warning("No data for regime map")
        return

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    ds_arr = np.array(ds)

    # clip delta for symmetric colormap
    vmax = max(abs(ds_arr.min()), abs(ds_arr.max()), 0.01)
    sc = ax.scatter(
        xs_arr, ys_arr,
        c=ds_arr, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        s=szs, alpha=0.75, edgecolors="0.2", linewidths=0.4, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$\Delta$ AUC-ROC (VITS $-$ Raw)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Label dataset clusters
    dataset_centroids: dict[str, list[list[float]]] = {}
    for xv, yv, ds_name in zip(xs, ys, datasets):
        dataset_centroids.setdefault(ds_name, [[], []])
        dataset_centroids[ds_name][0].append(xv)
        dataset_centroids[ds_name][1].append(yv)

    for ds_name, (cxs, cys) in dataset_centroids.items():
        cx, cy = float(np.mean(cxs)), float(np.mean(cys))
        ax.text(
            cx, cy, ds_name,
            ha="center", va="bottom", fontsize=6, fontweight="bold",
            color="0.2",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6),
        )

    # Dashed decision boundary at x=median n_channels if bimodal
    median_x = float(np.median(xs_arr))
    above_mask = xs_arr > median_x
    below_mask = ~above_mask
    if above_mask.sum() >= 2 and below_mask.sum() >= 2:
        above_mean = ds_arr[above_mask].mean()
        below_mean = ds_arr[below_mask].mean()
        if above_mean * below_mean < 0:  # sign flip → meaningful boundary
            ax.axvline(median_x, color="0.4", lw=0.8, ls="--", zorder=2, label=f"n_ch={median_x:.0f}")
            ax.legend(fontsize=6, loc="upper right")

    ax.set_xlabel("Number of channels ($D$)", fontsize=8)
    ax.set_ylabel("Signal entropy", fontsize=8)
    ax.set_title("Regime map: when do frozen vision representations help TSAD?", fontsize=8, pad=4)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved regime map: %s", out_path)


def _make_gain_predictor(corrs: dict[str, float], out_path: Path) -> None:
    """Figure B: Spearman rank correlation bar chart."""
    feat_labels = {
        "n_channels": "No. channels ($D$)",
        "anomaly_ratio": "Anomaly ratio",
        "periodicity": "Periodicity",
        "signal_entropy": "Signal entropy",
        "rendering_compression": "Rendering compression ($D/3$)",
    }
    feats = [f for f in FEATURE_COLS if np.isfinite(corrs.get(f, np.nan))]
    rhos = [corrs[f] for f in feats]
    labels = [feat_labels.get(f, f) for f in feats]

    # Sort by absolute value
    order = np.argsort(np.abs(rhos))[::-1]
    feats = [feats[i] for i in order]
    rhos = [rhos[i] for i in order]
    labels = [labels[i] for i in order]

    colors = ["#d62728" if r > 0 else "#1f77b4" for r in rhos]

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    bars = ax.barh(range(len(feats)), rhos, color=colors, edgecolor="0.2", linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("Spearman $\\rho$ with $\\Delta$ AUC-ROC", fontsize=8)
    ax.set_title("Feature correlation with VITS gain over Raw Mahalanobis", fontsize=8, pad=4)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-1, 1)
    ax.grid(True, axis="x", lw=0.3, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, rho in zip(bars, rhos):
        xpos = rho + (0.03 if rho >= 0 else -0.03)
        ha = "left" if rho >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{rho:+.3f}", va="center", ha=ha, fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved gain predictor: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    LOGGER.info("=== Regime Map: collecting entities ===")
    records: list[dict[str, Any]] = []

    smd = _collect_smd_entities()
    LOGGER.info("SMD: %d entities collected", len(smd))
    records.extend(smd)

    mv = _collect_multivar_datasets()
    LOGGER.info("Multivariate datasets: %d collected", len(mv))
    records.extend(mv)

    ucr = _collect_ucr_entities()
    LOGGER.info("UCR: %d series collected", len(ucr))
    records.extend(ucr)

    LOGGER.info("Total entities: %d", len(records))
    if not records:
        LOGGER.error("No records collected; exiting.")
        return

    # Print summary
    deltas = [r["delta"] for r in records]
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d <= 0)
    LOGGER.info(
        "Delta summary: mean=%+.4f  std=%.4f  min=%+.4f  max=%+.4f  "
        "vision_wins=%d  raw_wins=%d",
        np.mean(deltas), np.std(deltas), np.min(deltas), np.max(deltas), n_pos, n_neg,
    )

    # Spearman correlations
    corrs = _spearman_correlations(records)
    LOGGER.info("=== Spearman correlations with delta ===")
    for feat, rho in sorted(corrs.items(), key=lambda kv: abs(kv[1] if np.isfinite(kv[1]) else 0), reverse=True):
        LOGGER.info("  %-28s rho=%+.4f", feat, rho if np.isfinite(rho) else float("nan"))

    # Logistic regression
    LOGGER.info("=== Logistic regression ===")
    try:
        lr_results = _fit_logistic_regression(records)
    except ImportError:
        LOGGER.warning("scikit-learn not available; skipping logistic regression")
        lr_results = {"error": "scikit-learn not available"}

    # Save JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "n_entities": len(records),
        "summary": {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "vision_wins": n_pos,
            "raw_wins": n_neg,
        },
        "spearman_correlations": {k: (v if np.isfinite(v) else None) for k, v in corrs.items()},
        "logistic_regression": lr_results,
        "records": records,
    }
    json_path = OUT_DIR / "regime_map.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)
    LOGGER.info("Saved JSON: %s", json_path)

    # Figures
    LOGGER.info("=== Generating figures ===")
    _make_regime_map(records, PAPER_FIG / "regime_map.pdf")
    _make_gain_predictor(corrs, PAPER_FIG / "gain_predictor.pdf")

    LOGGER.info("=== Done ===")


if __name__ == "__main__":
    main()
