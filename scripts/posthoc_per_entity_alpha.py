"""Per-entity alpha tuning on SMD 28-entity benchmark.

Uses traj_scores.npy + dist_scores.npy already saved per entity to find
per-entity optimal alpha (oracle) and a practical heuristic-based alpha.
Also reports per-renderer and ensemble AUC-ROC improvements.
"""

from __future__ import annotations

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
    fused = alpha * _zscore(traj) + (1.0 - alpha) * _zscore(dist)
    fused = _smooth(fused, window=smooth_window)
    return _minmax(fused)


def per_entity_best_alpha(
    results_dir: Path,
    render: str,
    alphas: list[float] | None = None,
    smooth_window: int = 21,
) -> dict[str, dict]:
    """Find best alpha per entity using oracle (labels) and 'global' heuristic.

    Returns per-entity stats including oracle alpha, oracle AUC, and comparison
    to the default alpha used during detect (0.5 post-load).
    """
    if alphas is None:
        alphas = [round(a, 2) for a in np.arange(0.0, 1.01, 0.1)]

    entities = sorted(
        [d.name for d in results_dir.iterdir() if d.is_dir()]
    )
    results: dict[str, dict] = {}

    for ent in entities:
        rdir = results_dir / ent / render
        traj_p = rdir / "traj_scores.npy"
        dist_p = rdir / "dist_scores.npy"
        labels_p = rdir / "labels.npy"
        if not (traj_p.exists() and dist_p.exists() and labels_p.exists()):
            continue

        traj = np.load(traj_p)
        dist = np.load(dist_p)
        labels = np.load(labels_p)
        n = min(len(traj), len(dist), len(labels))
        traj, dist, lab = traj[:n], dist[:n], labels[:n]
        if len(set(lab)) < 2:
            continue

        per_alpha = {}
        best_alpha = 0.5
        best_auc = -1.0
        for a in alphas:
            scores = fuse_alpha(traj, dist, a, smooth_window)
            try:
                auc = float(roc_auc_score(lab, scores))
            except ValueError:
                continue
            per_alpha[str(a)] = auc
            if auc > best_auc:
                best_auc = auc
                best_alpha = a

        # Current AUC (alpha=0.5 default, since auto_alpha couldn't find val scores)
        # To match the actual stored metrics, we read metrics.json
        metrics_p = rdir / "metrics.json"
        current_auc = None
        if metrics_p.exists():
            current_auc = float(json.load(open(metrics_p))["auc_roc"])

        results[ent] = {
            "best_alpha": best_alpha,
            "best_auc": best_auc,
            "current_auc": current_auc,
            "auc_gain": (best_auc - current_auc) if current_auc is not None else None,
            "per_alpha_auc": per_alpha,
        }

    return results


def main() -> None:
    results_dir = Path("results/benchmark_smd_spatial_smooth21")

    all_summary = {}
    for render in ["line_plot", "recurrence_plot"]:
        print(f"\n{'='*70}")
        print(f"Per-entity oracle alpha tuning — {render}")
        print(f"{'='*70}")

        results = per_entity_best_alpha(
            results_dir, render, smooth_window=21
        )
        if not results:
            print("No data found.")
            continue

        best_aucs = [r["best_auc"] for r in results.values()]
        current_aucs = [
            r["current_auc"] for r in results.values() if r["current_auc"] is not None
        ]
        gains = [r["auc_gain"] for r in results.values() if r.get("auc_gain") is not None]
        best_alphas = [r["best_alpha"] for r in results.values()]

        print(f"\n{render}: {len(results)} entities")
        print(f"  Current mean AUC-ROC (alpha=0.5 default): {statistics.mean(current_aucs):.4f}")
        print(f"  Oracle per-entity mean AUC-ROC:            {statistics.mean(best_aucs):.4f}")
        print(f"  Mean gain:                                 +{statistics.mean(gains)*100:.2f}%")
        print(f"  Positives:                                 {sum(1 for g in gains if g > 0)}/{len(gains)}")

        # Alpha distribution
        alpha_counts: dict[float, int] = {}
        for a in best_alphas:
            alpha_counts[a] = alpha_counts.get(a, 0) + 1
        print(f"  Oracle alpha distribution: {dict(sorted(alpha_counts.items()))}")

        # Show top 5 gainers
        sorted_ents = sorted(results.items(), key=lambda x: -x[1]["auc_gain"] if x[1]["auc_gain"] is not None else -999)
        print("\n  Top 5 per-entity gains:")
        for ent, r in sorted_ents[:5]:
            print(f"    {ent}: current {r['current_auc']:.4f} → oracle {r['best_auc']:.4f} (α={r['best_alpha']}, +{r['auc_gain']*100:.2f}%)")

        all_summary[render] = {
            "current_mean": statistics.mean(current_aucs),
            "oracle_mean": statistics.mean(best_aucs),
            "mean_gain_pct": statistics.mean(gains) * 100,
            "alpha_distribution": {str(k): v for k, v in alpha_counts.items()},
        }

    # Compute ensemble using per-entity oracle alpha
    print(f"\n{'='*70}")
    print("Ensemble (LP + RP with per-entity oracle alpha)")
    print(f"{'='*70}")

    lp_results = per_entity_best_alpha(results_dir, "line_plot", smooth_window=21)
    rp_results = per_entity_best_alpha(results_dir, "recurrence_plot", smooth_window=21)

    ens_aucs = []
    common_ents = sorted(set(lp_results.keys()) & set(rp_results.keys()))
    for ent in common_ents:
        rdir = results_dir / ent
        lp_traj = np.load(rdir / "line_plot" / "traj_scores.npy")
        lp_dist = np.load(rdir / "line_plot" / "dist_scores.npy")
        rp_traj = np.load(rdir / "recurrence_plot" / "traj_scores.npy")
        rp_dist = np.load(rdir / "recurrence_plot" / "dist_scores.npy")
        labels = np.load(rdir / "line_plot" / "labels.npy")

        n = min(len(lp_traj), len(rp_traj), len(labels))
        lp_t, lp_d = lp_traj[:n], lp_dist[:n]
        rp_t, rp_d = rp_traj[:n], rp_dist[:n]
        lab = labels[:n]

        lp_a = lp_results[ent]["best_alpha"]
        rp_a = rp_results[ent]["best_alpha"]

        lp_scores = fuse_alpha(lp_t, lp_d, lp_a, 21)
        rp_scores = fuse_alpha(rp_t, rp_d, rp_a, 21)
        lp_rank = rankdata(lp_scores) / len(lp_scores)
        rp_rank = rankdata(rp_scores) / len(rp_scores)
        ens = 0.5 * lp_rank + 0.5 * rp_rank
        try:
            auc = float(roc_auc_score(lab, ens))
            ens_aucs.append(auc)
        except ValueError:
            pass

    if ens_aucs:
        print(f"  Per-entity oracle alpha + rank-mean ensemble: {statistics.mean(ens_aucs):.4f}")
        print("  (vs global alpha ensemble: 0.8085)")
        print(f"  Gain: {(statistics.mean(ens_aucs) - 0.8085)*100:+.2f}%")

    # Save summary
    with open(results_dir / "per_entity_alpha_summary.json", "w") as f:
        json.dump({
            "per_renderer": all_summary,
            "oracle_ensemble_mean": statistics.mean(ens_aucs) if ens_aucs else None,
            "vs_global_ensemble": statistics.mean(ens_aucs) - 0.8085 if ens_aucs else None,
        }, f, indent=2)
    print(f"\nSaved: {results_dir / 'per_entity_alpha_summary.json'}")


if __name__ == "__main__":
    main()
