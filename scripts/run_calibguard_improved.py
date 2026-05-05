#!/usr/bin/env python3
"""Run experimental CalibGuard FAR diagnostics on improved PatchTraj scores.

Evaluates conformal FAR control at α = {0.01, 0.05, 0.10} on:
1. All 28 SMD entities (per-entity + aggregate)
2. PSM, MSL, SMAP datasets

Uses the IMPROVED model scores (results/improved_*/) and also
the default model scores (results/full_smd/) for comparison.

Output: results/reports/experimental_calibguard_improved.md + .json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.scoring.calibguard import compute_far_at_alpha

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

RESULTS_ROOT = Path("results")
ALPHAS = [0.01, 0.05, 0.10]

SMD_ENTITIES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-1-4",
    "machine-1-5",
    "machine-1-6",
    "machine-1-7",
    "machine-1-8",
    "machine-2-1",
    "machine-2-2",
    "machine-2-3",
    "machine-2-4",
    "machine-2-5",
    "machine-2-6",
    "machine-2-7",
    "machine-2-8",
    "machine-2-9",
    "machine-3-1",
    "machine-3-2",
    "machine-3-3",
    "machine-3-4",
    "machine-3-5",
    "machine-3-6",
    "machine-3-7",
    "machine-3-8",
    "machine-3-9",
    "machine-3-10",
    "machine-3-11",
]


def load_scores_and_labels(
    entity_dir: Path,
    renderer: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load scores and labels for a given entity/renderer.

    Args:
        entity_dir: Path to entity directory.
        renderer: Renderer name.

    Returns:
        Tuple of (scores, labels) or None if files missing.
    """
    scores_path = entity_dir / renderer / "scores.npy"
    labels_path = entity_dir / renderer / "labels.npy"

    if not scores_path.exists() or not labels_path.exists():
        return None

    scores = np.load(scores_path).astype(np.float64)
    labels = np.load(labels_path).astype(np.int64)

    # Align lengths
    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[:min_len]

    # Validate
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None

    return scores, labels


def run_calibguard_entity(
    scores: np.ndarray,
    labels: np.ndarray,
    alphas: list[float],
    calibration_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """Run CalibGuard evaluation for multiple alpha values.

    Args:
        scores: Anomaly scores (T,).
        labels: Binary labels (T,).
        alphas: List of target FAR values.
        calibration_ratio: Fraction of normal scores for calibration.

    Returns:
        List of result dicts, one per alpha.
    """
    results = []
    for alpha in alphas:
        try:
            far_result = compute_far_at_alpha(
                scores=scores,
                labels=labels,
                alpha=alpha,
                calibration_ratio=calibration_ratio,
            )
            results.append(
                {
                    "alpha": alpha,
                    "target_far": alpha,
                    "actual_far": far_result["actual_far"],
                    "coverage": far_result["coverage"],
                    "threshold": far_result["threshold"],
                    "n_calibration": far_result["n_calibration"],
                    "n_test_normal": far_result["n_test_normal"],
                    "n_test_anomaly": far_result["n_test_anomaly"],
                    "far_violation": far_result["actual_far"] - alpha,
                }
            )
        except (ValueError, RuntimeError) as exc:
            LOGGER.warning("  CalibGuard failed for alpha=%.2f: %s", alpha, exc)
            results.append(
                {
                    "alpha": alpha,
                    "target_far": alpha,
                    "actual_far": None,
                    "coverage": None,
                    "error": str(exc),
                }
            )
    return results


def main() -> None:
    LOGGER.warning(
        "Running experimental CalibGuard diagnostics. These outputs are not claim-bearing because they calibrate from held-out test normals."
    )
    """Run CalibGuard experiments on all available results."""
    LOGGER.info("=" * 60)
    LOGGER.info("CalibGuard FAR Experiments (Improved Model)")
    LOGGER.info("=" * 60)

    all_results: dict[str, Any] = {}
    report_lines: list[str] = []
    report_lines.append("# CalibGuard FAR Experiments (Improved Model)")
    report_lines.append("")
    report_lines.append(f"Alphas tested: {ALPHAS}")
    report_lines.append("Calibration ratio: 0.5 (50% normal scores for calibration)")
    report_lines.append("")

    # ===== SMD Per-Entity =====
    report_lines.append(
        "## 1. SMD Per-Entity Results (Improved Model, Recurrence Plot)"
    )
    report_lines.append("")
    report_lines.append(
        "| Entity | α=0.01 FAR | α=0.01 Cov | α=0.05 FAR | α=0.05 Cov | α=0.10 FAR | α=0.10 Cov |"
    )
    report_lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    smd_entity_results: dict[str, list[dict]] = {}
    # Aggregate FAR and coverage across entities for each alpha
    agg_far: dict[float, list[float]] = {a: [] for a in ALPHAS}
    agg_cov: dict[float, list[float]] = {a: [] for a in ALPHAS}

    for entity in SMD_ENTITIES:
        entity_dir = RESULTS_ROOT / "improved_smd" / entity

        # Use recurrence_plot (generally better)
        data = load_scores_and_labels(entity_dir, "recurrence_plot")
        if data is None:
            data = load_scores_and_labels(entity_dir, "line_plot")
        if data is None:
            LOGGER.warning("  Skipping %s: no scores", entity)
            continue

        scores, labels = data
        results = run_calibguard_entity(scores, labels, ALPHAS)
        smd_entity_results[entity] = results

        row_parts = [f"| {entity}"]
        for r in results:
            if r.get("actual_far") is not None:
                row_parts.append(f" {r['actual_far']:.4f}")
                row_parts.append(f" {r['coverage']:.4f}")
                agg_far[r["alpha"]].append(r["actual_far"])
                agg_cov[r["alpha"]].append(r["coverage"])
            else:
                row_parts.append(" —")
                row_parts.append(" —")
        row_parts.append(" |")
        report_lines.append(" |".join(row_parts))

    # Aggregate row
    agg_row = ["| **28-Entity Avg**"]
    for alpha in ALPHAS:
        if agg_far[alpha]:
            agg_row.append(f" **{np.mean(agg_far[alpha]):.4f}**")
            agg_row.append(f" **{np.mean(agg_cov[alpha]):.4f}**")
        else:
            agg_row.append(" —")
            agg_row.append(" —")
    agg_row.append(" |")
    report_lines.append(" |".join(agg_row))

    report_lines.append("")
    all_results["smd_per_entity"] = smd_entity_results
    all_results["smd_aggregate"] = {
        str(a): {
            "mean_far": float(np.mean(agg_far[a])) if agg_far[a] else None,
            "mean_coverage": float(np.mean(agg_cov[a])) if agg_cov[a] else None,
            "std_far": float(np.std(agg_far[a])) if agg_far[a] else None,
            "n_entities": len(agg_far[a]),
        }
        for a in ALPHAS
    }

    # ===== FAR Guarantee Analysis =====
    report_lines.append("## 2. FAR Guarantee Analysis")
    report_lines.append("")
    report_lines.append("| α | Mean FAR | Std FAR | FAR ≤ α (%) | Mean Coverage | n |")
    report_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")

    for alpha in ALPHAS:
        if not agg_far[alpha]:
            continue
        fars = np.array(agg_far[alpha])
        covs = np.array(agg_cov[alpha])
        pct_valid = 100 * np.mean(fars <= alpha)
        report_lines.append(
            f"| {alpha} | {np.mean(fars):.4f} | {np.std(fars):.4f} | "
            f"{pct_valid:.1f}% | {np.mean(covs):.4f} | {len(fars)} |"
        )

    report_lines.append("")

    # ===== Other datasets =====
    report_lines.append("## 3. Other Datasets (Improved Model)")
    report_lines.append("")
    report_lines.append(
        "| Dataset | Renderer | α | Target FAR | Actual FAR | Coverage | n_calib |"
    )
    report_lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|")

    for dataset in ["psm", "msl", "smap"]:
        ds_dir = RESULTS_ROOT / f"improved_{dataset}" / "default"
        if not ds_dir.exists():
            continue

        for renderer in ["line_plot", "recurrence_plot"]:
            data = load_scores_and_labels(ds_dir, renderer)
            if data is None:
                continue

            scores, labels = data
            results = run_calibguard_entity(scores, labels, ALPHAS)

            for r in results:
                if r.get("actual_far") is not None:
                    report_lines.append(
                        f"| {dataset.upper()} | {renderer} | {r['alpha']} | "
                        f"{r['target_far']:.2f} | {r['actual_far']:.4f} | "
                        f"{r['coverage']:.4f} | {r.get('n_calibration', '—')} |"
                    )

            all_results[f"{dataset}_{renderer}"] = results

    report_lines.append("")

    # ===== Comparison: Default vs Improved CalibGuard =====
    report_lines.append("## 4. Default vs Improved Model CalibGuard (SMD)")
    report_lines.append("")
    report_lines.append("| Model | α | Mean FAR | Mean Coverage |")
    report_lines.append("|---|:---:|:---:|:---:|")

    # Default model CalibGuard
    for alpha in ALPHAS:
        default_fars: list[float] = []
        default_covs: list[float] = []
        for entity in SMD_ENTITIES:
            entity_dir = RESULTS_ROOT / "full_smd" / entity
            data = load_scores_and_labels(entity_dir, "recurrence_plot")
            if data is None:
                data = load_scores_and_labels(entity_dir, "line_plot")
            if data is None:
                continue
            scores, labels = data
            try:
                r = compute_far_at_alpha(scores, labels, alpha=alpha)
                default_fars.append(r["actual_far"])
                default_covs.append(r["coverage"])
            except (ValueError, RuntimeError):
                pass

        if default_fars:
            report_lines.append(
                f"| Default | {alpha} | {np.mean(default_fars):.4f} | {np.mean(default_covs):.4f} |"
            )

    for alpha in ALPHAS:
        if agg_far[alpha]:
            report_lines.append(
                f"| Improved | {alpha} | {np.mean(agg_far[alpha]):.4f} | {np.mean(agg_cov[alpha]):.4f} |"
            )

    report_lines.append("")

    # Save
    report_text = "\n".join(report_lines)
    report_path = RESULTS_ROOT / "reports" / "experimental_calibguard_improved.md"
    report_path.write_text(report_text)
    LOGGER.info("Report saved to %s", report_path)

    json_path = RESULTS_ROOT / "reports" / "experimental_calibguard_improved.json"
    with json_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    LOGGER.info("JSON saved to %s", json_path)

    print()
    print(report_text)


if __name__ == "__main__":
    main()
