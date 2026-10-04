"""Apply optimal post-hoc ensemble configs and generate before/after comparison.

Re-ensembles per-renderer scores using the best configs found in the
quick-improvement grid search (Phase 2 analysis). Produces optimized
ensemble scores, metrics, and a before/after comparison report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import compute_all_metrics
from src.scoring.patchtraj_scorer import normalize_scores, smooth_scores

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimalConfig:
    """Dataset-specific optimal ensemble configuration."""

    dataset: str
    smooth_method: str
    smooth_window: int
    ensemble_method: str
    w_lp: float
    expected_auc: float


# Best configs from quick_improvement_grid.csv
OPTIMAL_CONFIGS: dict[str, OptimalConfig] = {
    "smd": OptimalConfig(
        dataset="smd",
        smooth_method="mean",
        smooth_window=21,
        ensemble_method="zscore_weighted",
        w_lp=0.35,
        expected_auc=0.7974,
    ),
    "psm": OptimalConfig(
        dataset="psm",
        smooth_method="mean",
        smooth_window=21,
        ensemble_method="rank_weighted",
        w_lp=0.80,
        expected_auc=0.6236,
    ),
    "msl": OptimalConfig(
        dataset="msl",
        smooth_method="median",
        smooth_window=21,
        ensemble_method="zscore_weighted",
        w_lp=0.20,
        expected_auc=0.5534,
    ),
    "smap": OptimalConfig(
        dataset="smap",
        smooth_method="mean",
        smooth_window=7,
        ensemble_method="zscore_weighted",
        w_lp=0.20,
        expected_auc=0.6787,
    ),
}


def _weighted_ensemble(
    lp_scores: np.ndarray,
    rp_scores: np.ndarray,
    method: str,
    w_lp: float,
) -> np.ndarray:
    """Combine LP and RP scores with weighted ensemble method.

    Args:
        lp_scores: Line plot scores, shape (T,).
        rp_scores: Recurrence plot scores, shape (T,).
        method: Ensemble method (zscore_weighted, rank_weighted, minmax_weighted).
        w_lp: Weight for line plot scores (RP weight = 1 - w_lp).

    Returns:
        Ensemble scores of shape (T,).
    """
    from scipy.stats import rankdata

    w_rp = 1.0 - w_lp

    if method == "zscore_weighted":
        lp_mu, lp_std = np.mean(lp_scores), np.std(lp_scores)
        rp_mu, rp_std = np.mean(rp_scores), np.std(rp_scores)
        lp_z = (lp_scores - lp_mu) / lp_std if lp_std > 0 else np.zeros_like(lp_scores)
        rp_z = (rp_scores - rp_mu) / rp_std if rp_std > 0 else np.zeros_like(rp_scores)
        return (w_lp * lp_z + w_rp * rp_z).astype(np.float64)

    if method == "rank_weighted":
        n = lp_scores.shape[0]
        lp_r = rankdata(lp_scores) / n
        rp_r = rankdata(rp_scores) / n
        return (w_lp * lp_r + w_rp * rp_r).astype(np.float64)

    if method == "minmax_weighted":
        lp_min, lp_max = np.min(lp_scores), np.max(lp_scores)
        rp_min, rp_max = np.min(rp_scores), np.max(rp_scores)
        lp_scale = lp_max - lp_min
        rp_scale = rp_max - rp_min
        lp_n = (
            (lp_scores - lp_min) / lp_scale
            if lp_scale > 0
            else np.zeros_like(lp_scores)
        )
        rp_n = (
            (rp_scores - rp_min) / rp_scale
            if rp_scale > 0
            else np.zeros_like(rp_scores)
        )
        return (w_lp * lp_n + w_rp * rp_n).astype(np.float64)

    raise ValueError(f"Unknown ensemble method: {method}")


def _process_entity(
    entity_dir: Path,
    config: OptimalConfig,
    entity_name: str,
) -> dict[str, Any] | None:
    """Re-ensemble a single entity with optimal config.

    Args:
        entity_dir: Directory containing line_plot/ and recurrence_plot/ subdirs.
        config: Optimal config to apply.
        entity_name: Name of entity for logging.

    Returns:
        Dict with before/after metrics or None if data missing.
    """
    lp_scores_path = entity_dir / "line_plot" / "scores.npy"
    rp_scores_path = entity_dir / "recurrence_plot" / "scores.npy"
    lp_labels_path = entity_dir / "line_plot" / "labels.npy"
    rp_labels_path = entity_dir / "recurrence_plot" / "labels.npy"

    if not lp_scores_path.exists() or not rp_scores_path.exists():
        LOGGER.warning("Missing scores for %s, skipping", entity_name)
        return None

    lp_scores = np.load(lp_scores_path).astype(np.float64)
    rp_scores = np.load(rp_scores_path).astype(np.float64)

    # Load labels from either renderer
    labels_path = lp_labels_path if lp_labels_path.exists() else rp_labels_path
    if not labels_path.exists():
        LOGGER.warning("No labels for %s, skipping", entity_name)
        return None
    labels = np.load(labels_path).astype(np.int64)

    # Truncate to common length
    min_len = min(lp_scores.shape[0], rp_scores.shape[0], labels.shape[0])
    lp_scores = lp_scores[:min_len]
    rp_scores = rp_scores[:min_len]
    labels = labels[:min_len]

    # --- Before (original ensemble from saved file) ---
    before_metrics_path = entity_dir / "ensemble_metrics.json"
    if before_metrics_path.exists():
        with before_metrics_path.open("r") as f:
            before_metrics = json.load(f)
    else:
        # Reconstruct original: simple rank_mean with smooth=7
        from scipy.stats import rankdata

        lp_r = rankdata(lp_scores) / min_len
        rp_r = rankdata(rp_scores) / min_len
        orig_ens = np.mean(np.stack([lp_r, rp_r]), axis=0)
        orig_smooth = smooth_scores(orig_ens, window_size=7, method="mean")
        orig_norm = normalize_scores(orig_smooth, method="minmax")
        before_metrics = compute_all_metrics(orig_norm, labels)

    # --- After (optimized ensemble) ---
    ensemble_raw = _weighted_ensemble(
        lp_scores=lp_scores,
        rp_scores=rp_scores,
        method=config.ensemble_method,
        w_lp=config.w_lp,
    )

    # Apply optimized smoothing
    sw = config.smooth_window
    if sw % 2 == 0:
        sw += 1
    if sw > 1:
        ensemble_raw = smooth_scores(
            ensemble_raw, window_size=sw, method=config.smooth_method
        )

    # Normalize
    ensemble_norm = normalize_scores(ensemble_raw, method="minmax")

    after_metrics = compute_all_metrics(ensemble_norm, labels)

    # Save optimized scores and metrics
    np.save(entity_dir / "optimized_ensemble_scores.npy", ensemble_norm)
    with (entity_dir / "optimized_ensemble_metrics.json").open("w") as f:
        json.dump(after_metrics, f, indent=2, sort_keys=True)

    LOGGER.info(
        "%s: before=%.4f → after=%.4f (delta=%+.4f)",
        entity_name,
        before_metrics["auc_roc"],
        after_metrics["auc_roc"],
        after_metrics["auc_roc"] - before_metrics["auc_roc"],
    )

    return {
        "entity": entity_name,
        "before_auc_roc": before_metrics["auc_roc"],
        "after_auc_roc": after_metrics["auc_roc"],
        "before_auc_pr": before_metrics.get("auc_pr", 0.0),
        "after_auc_pr": after_metrics.get("auc_pr", 0.0),
        "before_f1_pa": before_metrics.get("f1_pa", 0.0),
        "after_f1_pa": after_metrics.get("f1_pa", 0.0),
        "delta_auc_roc": after_metrics["auc_roc"] - before_metrics["auc_roc"],
    }


def process_smd(results_dir: Path, config: OptimalConfig) -> list[dict[str, Any]]:
    """Process all SMD entities.

    Args:
        results_dir: Path to results/full_smd/.
        config: Optimal config for SMD.

    Returns:
        List of per-entity before/after result dicts.
    """
    results = []
    smd_dir = results_dir / "full_smd"
    if not smd_dir.exists():
        LOGGER.warning("SMD results dir not found: %s", smd_dir)
        return results

    entities = sorted(
        [
            d.name
            for d in smd_dir.iterdir()
            if d.is_dir() and d.name.startswith("machine")
        ]
    )
    LOGGER.info("Processing %d SMD entities with config: %s", len(entities), config)

    for entity in entities:
        result = _process_entity(smd_dir / entity, config, f"smd/{entity}")
        if result is not None:
            results.append(result)

    return results


def process_single_dataset(
    results_dir: Path,
    dataset: str,
    config: OptimalConfig,
) -> dict[str, Any] | None:
    """Process a single-entity dataset (PSM, MSL, SMAP).

    Args:
        results_dir: Base results directory.
        dataset: Dataset name.
        config: Optimal config.

    Returns:
        Before/after result dict or None.
    """
    dataset_dir = results_dir / f"{dataset}_benchmark"
    if not dataset_dir.exists():
        LOGGER.warning("%s results dir not found: %s", dataset, dataset_dir)
        return None

    return _process_entity(dataset_dir, config, dataset)


def generate_report(
    all_results: dict[str, list[dict[str, Any]]],
    configs: dict[str, OptimalConfig],
    output_dir: Path,
) -> None:
    """Generate before/after comparison report.

    Args:
        all_results: Per-dataset results.
        configs: Optimal configs used.
        output_dir: Where to save reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV detail
    csv_lines = [
        "dataset,entity,before_auc_roc,after_auc_roc,delta_auc_roc,"
        "before_auc_pr,after_auc_pr,before_f1_pa,after_f1_pa"
    ]
    for dataset, results in all_results.items():
        for r in results:
            csv_lines.append(
                f"{dataset},{r['entity']},"
                f"{r['before_auc_roc']:.6f},{r['after_auc_roc']:.6f},"
                f"{r['delta_auc_roc']:+.6f},"
                f"{r['before_auc_pr']:.6f},{r['after_auc_pr']:.6f},"
                f"{r['before_f1_pa']:.6f},{r['after_f1_pa']:.6f}"
            )

    csv_path = output_dir / "before_after_detail.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved detail CSV: %s", csv_path)

    # Summary table
    summary_lines = [
        "dataset,before_auc_roc,after_auc_roc,delta,smooth,window,method,w_lp"
    ]
    for dataset, results in all_results.items():
        if not results:
            continue
        cfg = configs[dataset]
        before_avg = np.mean([r["before_auc_roc"] for r in results])
        after_avg = np.mean([r["after_auc_roc"] for r in results])
        delta = after_avg - before_avg
        summary_lines.append(
            f"{dataset},{before_avg:.6f},{after_avg:.6f},{delta:+.6f},"
            f"{cfg.smooth_method},{cfg.smooth_window},{cfg.ensemble_method},{cfg.w_lp}"
        )

    summary_path = output_dir / "before_after_summary.csv"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved summary CSV: %s", summary_path)

    # Markdown report
    md_lines = [
        "# Before vs After: Optimal Post-Hoc Ensemble Configs",
        "",
        "## Summary",
        "",
        "| Dataset | Before AUC-ROC | After AUC-ROC | Δ | Config |",
        "|---------|---------------|--------------|---|--------|",
    ]
    for dataset, results in all_results.items():
        if not results:
            continue
        cfg = configs[dataset]
        before_avg = np.mean([r["before_auc_roc"] for r in results])
        after_avg = np.mean([r["after_auc_roc"] for r in results])
        delta = after_avg - before_avg
        config_str = (
            f"{cfg.smooth_method}-{cfg.smooth_window}, "
            f"{cfg.ensemble_method}, w_lp={cfg.w_lp}"
        )
        md_lines.append(
            f"| {dataset.upper()} | {before_avg:.4f} | {after_avg:.4f} "
            f"| {delta:+.4f} | {config_str} |"
        )

    md_lines.extend(
        [
            "",
            "## Optimal Configs Applied",
            "",
        ]
    )
    for dataset, cfg in configs.items():
        md_lines.extend(
            [
                f"### {dataset.upper()}",
                f"- Smooth: {cfg.smooth_method}, window={cfg.smooth_window}",
                f"- Ensemble: {cfg.ensemble_method}, w_lp={cfg.w_lp}",
                f"- Expected AUC-ROC: {cfg.expected_auc:.4f}",
                "",
            ]
        )

    if "smd" in all_results and all_results["smd"]:
        md_lines.extend(
            [
                "## SMD Per-Entity Detail",
                "",
                "| Entity | Before | After | Δ |",
                "|--------|--------|-------|---|",
            ]
        )
        for r in sorted(all_results["smd"], key=lambda x: x["entity"]):
            md_lines.append(
                f"| {r['entity']} | {r['before_auc_roc']:.4f} "
                f"| {r['after_auc_roc']:.4f} | {r['delta_auc_roc']:+.4f} |"
            )

    md_path = output_dir / "before_after_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved markdown report: %s", md_path)


def main() -> None:
    """Run optimal ensemble application across all datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    results_dir = Path("results")
    output_dir = results_dir / "reports"

    all_results: dict[str, list[dict[str, Any]]] = {}

    # SMD (28 entities)
    LOGGER.info("=" * 60)
    LOGGER.info("Processing SMD...")
    smd_results = process_smd(results_dir, OPTIMAL_CONFIGS["smd"])
    all_results["smd"] = smd_results

    # PSM
    LOGGER.info("=" * 60)
    LOGGER.info("Processing PSM...")
    psm_result = process_single_dataset(results_dir, "psm", OPTIMAL_CONFIGS["psm"])
    all_results["psm"] = [psm_result] if psm_result else []

    # MSL
    LOGGER.info("=" * 60)
    LOGGER.info("Processing MSL...")
    msl_result = process_single_dataset(results_dir, "msl", OPTIMAL_CONFIGS["msl"])
    all_results["msl"] = [msl_result] if msl_result else []

    # SMAP
    LOGGER.info("=" * 60)
    LOGGER.info("Processing SMAP...")
    smap_result = process_single_dataset(results_dir, "smap", OPTIMAL_CONFIGS["smap"])
    all_results["smap"] = [smap_result] if smap_result else []

    # Generate reports
    LOGGER.info("=" * 60)
    LOGGER.info("Generating reports...")
    generate_report(all_results, OPTIMAL_CONFIGS, output_dir)

    # Print summary
    LOGGER.info("=" * 60)
    LOGGER.info("DONE — Before vs After Summary:")
    for dataset, results in all_results.items():
        if results:
            before = np.mean([r["before_auc_roc"] for r in results])
            after = np.mean([r["after_auc_roc"] for r in results])
            LOGGER.info(
                "  %s: %.4f → %.4f (%+.4f)",
                dataset.upper(),
                before,
                after,
                after - before,
            )


if __name__ == "__main__":
    main()
