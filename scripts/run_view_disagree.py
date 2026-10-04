#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeArgument=false
"""ViewDisagree λ sweep experiment.

Evaluates the effect of adding view disagreement (std across LP and RP scores)
as an auxiliary anomaly signal. The combined score is:

    s_combined = s_ensemble + λ * ViewDisagree(s_LP, s_RP)

where ViewDisagree = std(s_LP_z, s_RP_z) per timestep.

Sweeps λ ∈ {0, 0.1, 0.2, ..., 2.0} and reports AUC-ROC for each dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from src.evaluation.metrics import compute_all_metrics
from src.rendering.multi_view import compute_view_disagreement
from src.scoring.patchtraj_scorer import smooth_scores

LOGGER = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")

# Best ensemble configs from previous experiments
BEST_ENSEMBLE_CONFIGS: dict[str, dict[str, Any]] = {
    "smd": {
        "method": "zscore_weighted",
        "smooth_window": 7,
        "smooth_method": "median",
        "w_lp": 0.40,
    },
    "smap": {
        "method": "zscore_weighted",
        "smooth_window": 7,
        "smooth_method": "mean",
        "w_lp": 0.10,
    },
    "psm": {
        "method": "rank_weighted",
        "smooth_window": 21,
        "smooth_method": "mean",
        "w_lp": 0.50,
    },
    "msl": {
        "method": "zscore_weighted",
        "smooth_window": 15,
        "smooth_method": "median",
        "w_lp": 0.30,
    },
}

SMD_ENTITIES = [
    f"machine-{g}-{i}"
    for g, count in [(1, 8), (2, 9), (3, 11)]
    for i in range(1, count + 1)
]

LAMBDA_VALUES = [round(x * 0.1, 1) for x in range(21)]  # 0.0 to 2.0


def _zscore(scores: np.ndarray) -> np.ndarray:
    """Z-score normalize scores.

    Args:
        scores: Raw scores of shape (T,).

    Returns:
        Z-scored values of shape (T,).
    """
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    if sigma < 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return ((scores - mu) / sigma).astype(np.float64)


def _load_entity_scores(
    entity_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load LP scores, RP scores, and labels for one entity.

    Args:
        entity_dir: Directory containing line_plot/ and recurrence_plot/ subdirs.

    Returns:
        Tuple of (lp_scores, rp_scores, labels) or None if files missing.
    """
    # Support both naming conventions: scores.npy (legacy) and test_scores.npy
    lp_path = entity_dir / "line_plot" / "scores.npy"
    rp_path = entity_dir / "recurrence_plot" / "scores.npy"
    label_path = entity_dir / "line_plot" / "labels.npy"

    # Fall back to test_scores.npy / test_labels.npy if legacy names not found
    if not lp_path.exists():
        lp_path = entity_dir / "line_plot" / "test_scores.npy"
    if not rp_path.exists():
        rp_path = entity_dir / "recurrence_plot" / "test_scores.npy"
    if not label_path.exists():
        label_path = entity_dir / "line_plot" / "test_labels.npy"

    if not all(p.exists() for p in [lp_path, rp_path, label_path]):
        LOGGER.warning("Missing score files in %s", entity_dir)
        return None

    lp = np.load(lp_path).astype(np.float64).reshape(-1)
    rp = np.load(rp_path).astype(np.float64).reshape(-1)
    labels = np.load(label_path).astype(np.int64).reshape(-1)

    # Align lengths
    min_len = min(lp.size, rp.size, labels.size)
    return lp[:min_len], rp[:min_len], labels[:min_len]


def _compute_ensemble_with_disagree(
    lp_scores: np.ndarray,
    rp_scores: np.ndarray,
    method: str,
    w_lp: float,
    smooth_window: int,
    smooth_method: str,
    lambda_disagree: float,
) -> np.ndarray:
    """Compute ensemble score with ViewDisagree auxiliary signal.

    Args:
        lp_scores: Line plot raw scores (T,).
        rp_scores: Recurrence plot raw scores (T,).
        method: Ensemble method ('zscore_weighted' or 'rank_weighted').
        w_lp: Weight for LP in ensemble.
        smooth_window: Smoothing window size.
        smooth_method: Smoothing method ('mean' or 'median').
        lambda_disagree: Weight for ViewDisagree signal.

    Returns:
        Combined scores of shape (T,).
    """
    # Smooth individual scores
    lp_smooth = smooth_scores(
        lp_scores, window_size=smooth_window, method=smooth_method
    )
    rp_smooth = smooth_scores(
        rp_scores, window_size=smooth_window, method=smooth_method
    )

    # Z-score normalize
    lp_z = _zscore(lp_smooth)
    rp_z = _zscore(rp_smooth)

    # Base ensemble
    w_rp = 1.0 - w_lp
    if method == "zscore_weighted":
        ensemble = w_lp * lp_z + w_rp * rp_z
    elif method == "rank_weighted":
        n = lp_z.shape[0]
        lp_r = rankdata(lp_smooth) / n
        rp_r = rankdata(rp_smooth) / n
        ensemble = w_lp * lp_r + w_rp * rp_r
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    # ViewDisagree signal: std across views per timestep
    per_view = np.stack([lp_z, rp_z], axis=0)  # (2, T)
    disagree = compute_view_disagreement(per_view)  # (T,)

    # Combined score
    combined = ensemble + lambda_disagree * disagree
    return combined.astype(np.float64)


def run_smd_sweep() -> dict[str, Any]:
    """Run ViewDisagree λ sweep on all 28 SMD entities.

    Returns:
        Results dict with per-lambda aggregate AUC-ROC.
    """
    config = BEST_ENSEMBLE_CONFIGS["smd"]
    results_dir = RESULTS_ROOT / "improved_smd"

    lambda_results: dict[float, list[float]] = {lam: [] for lam in LAMBDA_VALUES}

    for entity in SMD_ENTITIES:
        entity_dir = results_dir / entity
        loaded = _load_entity_scores(entity_dir)
        if loaded is None:
            LOGGER.warning("Skipping %s (missing files)", entity)
            continue

        lp, rp, labels = loaded

        for lam in LAMBDA_VALUES:
            combined = _compute_ensemble_with_disagree(
                lp_scores=lp,
                rp_scores=rp,
                method=config["method"],
                w_lp=config["w_lp"],
                smooth_window=config["smooth_window"],
                smooth_method=config["smooth_method"],
                lambda_disagree=lam,
            )
            metrics = compute_all_metrics(scores=combined, labels=labels)
            lambda_results[lam].append(float(metrics["auc_roc"]))

    # Aggregate
    summary: dict[str, Any] = {}
    for lam in LAMBDA_VALUES:
        scores_list = lambda_results[lam]
        if scores_list:
            summary[str(lam)] = {
                "mean_auc_roc": round(float(np.mean(scores_list)), 4),
                "std_auc_roc": round(float(np.std(scores_list)), 4),
                "n_entities": len(scores_list),
            }

    return {"dataset": "smd", "results": summary}


def run_aggregate_sweep(dataset: str) -> dict[str, Any]:
    """Run ViewDisagree λ sweep on an aggregate dataset (PSM/MSL/SMAP).

    Args:
        dataset: Dataset name ('psm', 'msl', or 'smap').

    Returns:
        Results dict with per-lambda AUC-ROC.
    """
    config = BEST_ENSEMBLE_CONFIGS[dataset]
    entity_dir = RESULTS_ROOT / f"improved_{dataset}" / "default"

    loaded = _load_entity_scores(entity_dir)
    if loaded is None:
        LOGGER.error("Missing score files for %s", dataset)
        return {"dataset": dataset, "error": "missing_files"}

    lp, rp, labels = loaded

    summary: dict[str, Any] = {}
    for lam in LAMBDA_VALUES:
        combined = _compute_ensemble_with_disagree(
            lp_scores=lp,
            rp_scores=rp,
            method=config["method"],
            w_lp=config["w_lp"],
            smooth_window=config["smooth_window"],
            smooth_method=config["smooth_method"],
            lambda_disagree=lam,
        )
        metrics = compute_all_metrics(scores=combined, labels=labels)
        summary[str(lam)] = {
            "auc_roc": round(float(metrics["auc_roc"]), 4),
            "auc_pr": round(float(metrics["auc_pr"]), 4),
            "f1_pa": round(float(metrics["f1_pa"]), 4),
        }

    return {"dataset": dataset, "results": summary}


def find_best_lambda(results: dict[str, Any]) -> tuple[float, float]:
    """Find the λ that maximizes AUC-ROC from sweep results.

    Args:
        results: Sweep results dict with per-lambda metrics.

    Returns:
        Tuple of (best_lambda, best_auc_roc).
    """
    best_lam = 0.0
    best_auc = 0.0

    for lam_str, metrics in results.get("results", {}).items():
        if isinstance(metrics, dict) and "error" not in metrics:
            auc = metrics.get("mean_auc_roc", metrics.get("auc_roc", 0.0))
            if auc > best_auc:
                best_auc = auc
                best_lam = float(lam_str)

    return best_lam, best_auc


def main() -> None:
    """Run ViewDisagree λ sweep on all datasets and save report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir = RESULTS_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}

    # SMD (28 entities)
    LOGGER.info("Running ViewDisagree sweep on SMD...")
    smd_results = run_smd_sweep()
    all_results["smd"] = smd_results
    best_lam, best_auc = find_best_lambda(smd_results)
    LOGGER.info("SMD best λ=%.1f → AUC-ROC=%.4f", best_lam, best_auc)

    # Aggregate datasets
    for dataset in ["psm", "msl", "smap"]:
        LOGGER.info("Running ViewDisagree sweep on %s...", dataset.upper())
        ds_results = run_aggregate_sweep(dataset)
        all_results[dataset] = ds_results
        best_lam, best_auc = find_best_lambda(ds_results)
        LOGGER.info(
            "%s best λ=%.1f → AUC-ROC=%.4f", dataset.upper(), best_lam, best_auc
        )

    # Save JSON
    json_path = output_dir / "view_disagree_sweep.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    LOGGER.info("Saved results to %s", json_path)

    # Generate markdown report
    md_lines = [
        "# ViewDisagree λ Sweep Results\n",
        "",
        "Combined score: `s = s_ensemble + λ * ViewDisagree(s_LP, s_RP)`\n",
        "",
        "ViewDisagree = std(z(LP), z(RP)) per timestep.\n",
        "",
    ]

    for dataset in ["smd", "psm", "msl", "smap"]:
        ds_data = all_results.get(dataset, {})
        best_lam, best_auc = find_best_lambda(ds_data)
        baseline_auc_key = "0.0"
        baseline_metrics = ds_data.get("results", {}).get(baseline_auc_key, {})
        baseline_auc = baseline_metrics.get(
            "mean_auc_roc", baseline_metrics.get("auc_roc", 0.0)
        )

        md_lines.append(f"## {dataset.upper()}\n")
        md_lines.append("")
        md_lines.append(f"- Best λ = **{best_lam:.1f}** → AUC-ROC = **{best_auc:.4f}**")
        md_lines.append(f"- Baseline (λ=0) AUC-ROC = {baseline_auc:.4f}")
        delta = best_auc - baseline_auc
        md_lines.append(f"- Improvement: {delta:+.4f}")
        md_lines.append("")

        # Table of all λ values
        md_lines.append("| λ | AUC-ROC |")
        md_lines.append("|---|---------|")
        for lam in LAMBDA_VALUES:
            lam_str = str(lam)
            metrics = ds_data.get("results", {}).get(lam_str, {})
            auc = metrics.get("mean_auc_roc", metrics.get("auc_roc", 0.0))
            marker = " ← best" if lam == best_lam and lam != 0.0 else ""
            md_lines.append(f"| {lam:.1f} | {auc:.4f}{marker} |")
        md_lines.append("")

    md_path = output_dir / "view_disagree_sweep.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    LOGGER.info("Saved markdown report to %s", md_path)


if __name__ == "__main__":
    main()
