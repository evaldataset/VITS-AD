"""Post-hoc alpha tuning and confidence-weighted ensemble for SMD 28-entity.

Reads traj_scores.npy + dist_scores.npy (per-entity, per-renderer), sweeps alpha
to find per-entity optimal alpha based on labels (oracle upper bound) and on
score-spread heuristic (data-driven selection without labels). Then computes
LP+RP ensemble using rank_mean and confidence-weighted approaches.

Usage:
    python scripts/posthoc_alpha_ensemble.py --results-dir results/benchmark_smd_spatial_smooth21
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    std = float(np.std(x))
    if std < eps:
        return np.zeros_like(x, dtype=np.float64)
    return ((x - float(np.mean(x))) / std).astype(np.float64)


def _smooth(x: np.ndarray, window: int = 21) -> np.ndarray:
    if window <= 1 or window >= len(x):
        return x.astype(np.float64, copy=True)
    pad = window // 2
    padded = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        out[i] = float(np.mean(padded[i : i + window]))
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return ((x - lo) / (hi - lo)).astype(np.float64)


def fuse_alpha(
    traj: np.ndarray, dist: np.ndarray, alpha: float, smooth_window: int = 21
) -> np.ndarray:
    """Fuse trajectory + distributional scores with given alpha, then smooth + minmax."""
    fused = alpha * _zscore(traj) + (1.0 - alpha) * _zscore(dist)
    fused = _smooth(fused, window=smooth_window)
    return _minmax(fused)


def oracle_alpha(
    traj: np.ndarray, dist: np.ndarray, labels: np.ndarray, smooth_window: int = 21
) -> tuple[float, float]:
    """Best alpha given test labels (oracle upper bound)."""
    best_a, best_auc = 0.5, -1.0
    for a in np.arange(0.0, 1.01, 0.05):
        scores = fuse_alpha(traj, dist, float(a), smooth_window)
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            continue
        if auc > best_auc:
            best_auc = auc
            best_a = float(a)
    return best_a, best_auc


def heuristic_alpha(
    traj: np.ndarray, dist: np.ndarray, smooth_window: int = 21
) -> float:
    """Heuristic alpha based on score spread (data-driven, no labels)."""
    best_a, best_spread = 0.5, -1.0
    for a in np.arange(0.0, 1.01, 0.1):
        fused = float(a) * _zscore(traj) + (1.0 - float(a)) * _zscore(dist)
        fused_abs = np.abs(fused)
        mu = float(np.mean(fused_abs))
        std = float(np.std(fused_abs))
        spread = std / max(mu, 1e-12)
        if spread > best_spread:
            best_spread = spread
            best_a = float(a)
    return best_a


def confidence_weighted_ensemble(
    lp_scores: np.ndarray,
    rp_scores: np.ndarray,
    lp_traj_var: float,
    rp_traj_var: float,
) -> np.ndarray:
    """Confidence-weighted ensemble (inverse trajectory variance = confidence)."""
    lp_conf = 1.0 / (lp_traj_var + 1e-8)
    rp_conf = 1.0 / (rp_traj_var + 1e-8)
    total = lp_conf + rp_conf
    w_lp = lp_conf / total
    w_rp = rp_conf / total
    lp_r = rankdata(lp_scores) / len(lp_scores)
    rp_r = rankdata(rp_scores) / len(rp_scores)
    return w_lp * lp_r + w_rp * rp_r


def analyze(results_dir: Path) -> None:
    entities = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])

    rows = []
    for ent in entities:
        lp_dir = results_dir / ent / "line_plot"
        rp_dir = results_dir / ent / "recurrence_plot"
        labels_path = lp_dir / "labels.npy"
        if not labels_path.exists():
            continue
        labels = np.load(labels_path)

        # Check for traj/dist breakdown
        lp_traj_p = lp_dir / "traj_scores.npy"
        lp_dist_p = lp_dir / "dist_scores.npy"
        rp_traj_p = rp_dir / "traj_scores.npy"
        rp_dist_p = rp_dir / "dist_scores.npy"

        # Fallback to final fused scores if traj/dist breakdown missing
        lp_scores_p = lp_dir / "scores.npy"
        rp_scores_p = rp_dir / "scores.npy"

        entry = {"entity": ent}

        if lp_traj_p.exists() and lp_dist_p.exists():
            lp_traj = np.load(lp_traj_p)
            lp_dist = np.load(lp_dist_p)
            n = min(len(lp_traj), len(lp_dist), len(labels))
            entry["lp_oracle_alpha"], entry["lp_oracle_auc"] = oracle_alpha(
                lp_traj[:n], lp_dist[:n], labels[:n]
            )
            entry["lp_heuristic_alpha"] = heuristic_alpha(lp_traj[:n], lp_dist[:n])
            lp_heur = fuse_alpha(lp_traj[:n], lp_dist[:n], entry["lp_heuristic_alpha"])
            try:
                entry["lp_heuristic_auc"] = float(roc_auc_score(labels[:n], lp_heur))
            except ValueError:
                entry["lp_heuristic_auc"] = float("nan")
        else:
            if lp_scores_p.exists():
                lp_s = np.load(lp_scores_p)
                n = min(len(lp_s), len(labels))
                try:
                    entry["lp_current_auc"] = float(roc_auc_score(labels[:n], lp_s[:n]))
                except ValueError:
                    entry["lp_current_auc"] = float("nan")

        if rp_traj_p.exists() and rp_dist_p.exists():
            rp_traj = np.load(rp_traj_p)
            rp_dist = np.load(rp_dist_p)
            n = min(len(rp_traj), len(rp_dist), len(labels))
            entry["rp_oracle_alpha"], entry["rp_oracle_auc"] = oracle_alpha(
                rp_traj[:n], rp_dist[:n], labels[:n]
            )
            entry["rp_heuristic_alpha"] = heuristic_alpha(rp_traj[:n], rp_dist[:n])
            rp_heur = fuse_alpha(rp_traj[:n], rp_dist[:n], entry["rp_heuristic_alpha"])
            try:
                entry["rp_heuristic_auc"] = float(roc_auc_score(labels[:n], rp_heur))
            except ValueError:
                entry["rp_heuristic_auc"] = float("nan")

        # Ensemble
        if lp_scores_p.exists() and rp_scores_p.exists():
            lp_s = np.load(lp_scores_p)
            rp_s = np.load(rp_scores_p)
            n = min(len(lp_s), len(rp_s), len(labels))
            lp_s, rp_s, lab = lp_s[:n], rp_s[:n], labels[:n]

            # Rank-mean ensemble (baseline)
            lp_r = rankdata(lp_s) / len(lp_s)
            rp_r = rankdata(rp_s) / len(rp_s)
            rm_ens = 0.5 * lp_r + 0.5 * rp_r
            try:
                entry["ensemble_rank_mean_auc"] = float(roc_auc_score(lab, rm_ens))
            except ValueError:
                entry["ensemble_rank_mean_auc"] = float("nan")

            # Confidence-weighted (using trajectory variance as inverse confidence)
            if lp_traj_p.exists() and rp_traj_p.exists():
                lp_traj = np.load(lp_traj_p)[:n]
                rp_traj = np.load(rp_traj_p)[:n]
                cw_ens = confidence_weighted_ensemble(
                    lp_s, rp_s, float(np.var(lp_traj)), float(np.var(rp_traj))
                )
                try:
                    entry["ensemble_conf_weighted_auc"] = float(
                        roc_auc_score(lab, cw_ens)
                    )
                except ValueError:
                    entry["ensemble_conf_weighted_auc"] = float("nan")

        rows.append(entry)

    # Summary
    print(f"{'='*70}")
    print(f"Post-hoc analysis: {results_dir}")
    print(f"{'='*70}")

    def _avg(key: str) -> str:
        vals = [r[key] for r in rows if key in r and not np.isnan(r.get(key, float("nan")))]
        return f"{statistics.mean(vals):.4f} ({len(vals)}/28)" if vals else "N/A"

    print("\nPer-renderer AUC-ROC (heuristic alpha + smooth=21):")
    print(f"  LP heuristic: {_avg('lp_heuristic_auc')}")
    print(f"  RP heuristic: {_avg('rp_heuristic_auc')}")

    print("\nPer-renderer AUC-ROC (oracle alpha + smooth=21) [upper bound]:")
    print(f"  LP oracle:    {_avg('lp_oracle_auc')}")
    print(f"  RP oracle:    {_avg('rp_oracle_auc')}")

    print("\nEnsemble AUC-ROC:")
    print(f"  Rank-mean:         {_avg('ensemble_rank_mean_auc')}")
    print(f"  Confidence-weighted: {_avg('ensemble_conf_weighted_auc')}")

    # Also reference current scores
    print("\nCurrent scores (fused at detect-time alpha):")
    print(f"  LP current AUC: {_avg('lp_current_auc')}")

    # Save detailed per-entity results
    out_path = results_dir / "posthoc_analysis.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved detailed per-entity results: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/benchmark_smd_spatial_smooth21",
    )
    args = parser.parse_args()
    analyze(Path(args.results_dir))
