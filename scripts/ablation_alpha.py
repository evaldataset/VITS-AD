"""Post-hoc alpha sensitivity ablation for dual-signal scoring.

This script loads pre-computed trajectory and distributional scores from
detection results and sweeps the fusion weight alpha to show sensitivity.
Results are clearly labelled as **post-hoc analysis on test set** and must
NOT be used for model selection in the main results table.

Usage:
    python scripts/ablation_alpha.py --results-dir results/dinov2_base_smd_line_plot_spatial
    python scripts/ablation_alpha.py --results-dir results/dinov2_base_smd_recurrence_plot_dual_only

Output:
    results/ablation_alpha/<experiment_name>/alpha_sweep.json
    results/ablation_alpha/<experiment_name>/alpha_sweep.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.evaluation.metrics import compute_all_metrics
from src.scoring.patchtraj_scorer import normalize_scores, smooth_scores

LOGGER = logging.getLogger(__name__)


def _zscore(x: npt.NDArray[np.float64], eps: float = 1e-8) -> npt.NDArray[np.float64]:
    """Z-score normalise, returning zeros for constant arrays."""
    mu = float(np.mean(x))
    std = float(np.std(x))
    if std < eps:
        return np.zeros_like(x)
    return (x - mu) / std


def run_alpha_sweep(
    traj_scores: npt.NDArray[np.float64],
    dist_scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    alpha_values: list[float],
    smooth_window: int = 5,
    smooth_method: str = "mean",
) -> list[dict[str, float]]:
    """Sweep alpha values and compute metrics for each.

    Args:
        traj_scores: Trajectory anomaly scores of shape ``(T,)``.
        dist_scores: Distributional anomaly scores of shape ``(T,)``.
        labels: Ground truth labels of shape ``(T,)``.
        alpha_values: List of alpha values to evaluate.
        smooth_window: Score smoothing window size.
        smooth_method: Smoothing method (mean or median).

    Returns:
        List of dicts with alpha, AUC-ROC, AUC-PR, best-F1, and F1-PA.
    """
    results: list[dict[str, float]] = []

    traj_z = _zscore(traj_scores)
    dist_z = _zscore(dist_scores)

    for alpha in alpha_values:
        fused = alpha * traj_z + (1.0 - alpha) * dist_z

        if smooth_window > 1:
            sw = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            fused = smooth_scores(fused, window_size=sw, method=smooth_method)

        normalized = normalize_scores(fused, method="minmax")
        metrics = compute_all_metrics(normalized, labels)

        row = {
            "alpha": round(alpha, 2),
            "auc_roc": float(metrics.get("auc_roc", 0.0)),
            "auc_pr": float(metrics.get("auc_pr", 0.0)),
            "best_f1": float(metrics.get("best_f1", 0.0)),
            "f1_pa": float(metrics.get("f1_pa", 0.0)),
        }
        results.append(row)
        LOGGER.info(
            "alpha=%.2f  AUC-ROC=%.4f  AUC-PR=%.4f  F1=%.4f  F1-PA=%.4f",
            row["alpha"],
            row["auc_roc"],
            row["auc_pr"],
            row["best_f1"],
            row["f1_pa"],
        )

    return results


def main() -> None:
    """Run alpha ablation sweep on saved detection results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Post-hoc alpha sensitivity ablation (test-set analysis)."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to detection results directory containing traj_scores.npy and dist_scores.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to results/ablation_alpha/<experiment_name>/.",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.0,
        help="Minimum alpha value (default: 0.0).",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=1.0,
        help="Maximum alpha value (default: 1.0).",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.1,
        help="Alpha step size (default: 0.1).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Score smoothing window (default: 5).",
    )
    parser.add_argument(
        "--smooth-method",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Score smoothing method (default: mean).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    traj_path = results_dir / "traj_scores.npy"
    dist_path = results_dir / "dist_scores.npy"
    labels_path = results_dir / "labels.npy"

    if not traj_path.exists():
        raise FileNotFoundError(
            f"Trajectory scores not found: {traj_path}. "
            "Run detect.py with dual_signal.enabled=true first."
        )
    if not dist_path.exists():
        raise FileNotFoundError(
            f"Distributional scores not found: {dist_path}. "
            "Run detect.py with dual_signal.enabled=true first."
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}.")

    traj_scores = np.load(traj_path).astype(np.float64)
    dist_scores = np.load(dist_path).astype(np.float64)
    labels = np.load(labels_path).astype(np.int64)

    LOGGER.info(
        "Loaded scores: traj=%s dist=%s labels=%s",
        traj_scores.shape,
        dist_scores.shape,
        labels.shape,
    )

    alpha_values = list(
        np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)
    )
    alpha_values = [round(a, 2) for a in alpha_values]

    results = run_alpha_sweep(
        traj_scores=traj_scores,
        dist_scores=dist_scores,
        labels=labels,
        alpha_values=alpha_values,
        smooth_window=args.smooth_window,
        smooth_method=args.smooth_method,
    )

    # Determine output directory
    experiment_name = results_dir.name
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("results") / "ablation_alpha" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    output_payload = {
        "NOTE": (
            "POST-HOC ANALYSIS on test set. These results must NOT be used "
            "for model selection. Use validation-based alpha selection in "
            "detect.py for the main results."
        ),
        "source_dir": str(results_dir),
        "smooth_window": args.smooth_window,
        "smooth_method": args.smooth_method,
        "results": results,
    }
    json_path = out_dir / "alpha_sweep.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, sort_keys=False)
    LOGGER.info("Saved JSON results to %s", json_path)

    # Save CSV
    csv_path = out_dir / "alpha_sweep.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("alpha,auc_roc,auc_pr,best_f1,f1_pa\n")
        for row in results:
            f.write(
                f"{row['alpha']:.2f},{row['auc_roc']:.4f},"
                f"{row['auc_pr']:.4f},{row['best_f1']:.4f},{row['f1_pa']:.4f}\n"
            )
    LOGGER.info("Saved CSV results to %s", csv_path)

    # Print summary
    best_row = max(results, key=lambda r: r["auc_roc"])
    LOGGER.info(
        "Best alpha=%.2f (AUC-ROC=%.4f). NOTE: This is post-hoc test-set analysis.",
        best_row["alpha"],
        best_row["auc_roc"],
    )


if __name__ == "__main__":
    main()
