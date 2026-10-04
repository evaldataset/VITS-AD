#!/usr/bin/env python3
"""Statistical significance tests for VITS/PatchTraj paper.

Runs:
1. Paired Wilcoxon signed-rank test: Default vs Improved PatchTraj (28 SMD entities)
2. Paired Wilcoxon: PatchTraj Ensemble vs individual renderers
3. Renderer comparison: LP vs RP
4. Effect sizes (Cohen's d)
5. Summary statistics with confidence intervals

Output: results/reports/statistical_tests.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

RESULTS_ROOT = Path("results")

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


def load_entity_aucroc(
    result_dir: Path,
    renderer: str,
) -> dict[str, float]:
    """Load per-entity AUC-ROC from results directory.

    Args:
        result_dir: Directory containing entity subdirs.
        renderer: Renderer name (line_plot or recurrence_plot).

    Returns:
        Dict mapping entity name -> AUC-ROC.
    """
    results: dict[str, float] = {}
    for entity in SMD_ENTITIES:
        metrics_path = result_dir / entity / renderer / "metrics.json"
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text())
            results[entity] = data.get("auc_roc", 0.0)
    return results


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Cohen's d value.
    """
    diff = x - y
    return float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-10))


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        data: 1D array of values.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            np.mean(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1 - confidence) / 2
    return float(np.percentile(means, 100 * alpha)), float(
        np.percentile(means, 100 * (1 - alpha))
    )


def run_tests() -> str:
    """Run all statistical tests and return markdown report.

    Returns:
        Markdown-formatted report string.
    """
    report_lines: list[str] = []
    report_lines.append("# Statistical Significance Tests")
    report_lines.append("")
    report_lines.append(f"Generated: {np.datetime64('now')}")
    report_lines.append(f"N = {len(SMD_ENTITIES)} SMD entities")
    report_lines.append("")

    # Load all data
    default_lp = load_entity_aucroc(RESULTS_ROOT / "full_smd", "line_plot")
    default_rp = load_entity_aucroc(RESULTS_ROOT / "full_smd", "recurrence_plot")
    improved_lp = load_entity_aucroc(RESULTS_ROOT / "improved_smd", "line_plot")
    improved_rp = load_entity_aucroc(RESULTS_ROOT / "improved_smd", "recurrence_plot")

    # Load improved ensemble
    ensemble_path = RESULTS_ROOT / "reports" / "improved_ensemble_results.json"
    improved_ensemble: dict[str, float] = {}
    if ensemble_path.exists():
        all_data = json.loads(ensemble_path.read_text())
        if "smd" in all_data:
            for entity, metrics in all_data["smd"].get("per_entity", {}).items():
                improved_ensemble[entity] = metrics["auc_roc"]

    # Common entities
    common = sorted(
        set(default_lp)
        & set(default_rp)
        & set(improved_lp)
        & set(improved_rp)
        & set(improved_ensemble)
    )
    LOGGER.info("Common entities across all methods: %d", len(common))

    # Build arrays
    d_lp = np.array([default_lp[e] for e in common])
    d_rp = np.array([default_rp[e] for e in common])
    i_lp = np.array([improved_lp[e] for e in common])
    i_rp = np.array([improved_rp[e] for e in common])
    i_ens = np.array([improved_ensemble[e] for e in common])

    # Best default per entity (max of LP, RP)
    d_best = np.maximum(d_lp, d_rp)
    i_best = np.maximum(i_lp, i_rp)

    # ===== Test 1: Default LP vs Improved LP =====
    report_lines.append("## 1. Default vs Improved PatchTraj (Paired Wilcoxon)")
    report_lines.append("")
    comparisons = [
        ("LP", d_lp, i_lp),
        ("RP", d_rp, i_rp),
        ("Best(LP,RP)", d_best, i_best),
        ("Ensemble", d_best, i_ens),
    ]

    n_comparisons = 4  # LP, RP, Best, Ensemble
    bonferroni_alpha = 0.05 / n_comparisons

    report_lines.append(
        f"Bonferroni-corrected α = {bonferroni_alpha:.4f} "
        f"(raw α = 0.05, {n_comparisons} comparisons)"
    )
    report_lines.append("")
    report_lines.append(
        "| Comparison | Default Mean | Improved Mean | Δ Mean | W-stat | p-value | Cohen's d | Sig (raw α=0.05) | Sig (corrected α={:.4f}) |".format(bonferroni_alpha)
    )
    report_lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for name, arr_default, arr_improved in comparisons:
        diff = arr_improved - arr_default
        mean_d = np.mean(arr_default)
        mean_i = np.mean(arr_improved)
        delta = np.mean(diff)
        d_effect = cohens_d(arr_improved, arr_default)

        try:
            w_stat, p_val = stats.wilcoxon(
                arr_improved, arr_default, alternative="two-sided"
            )
        except ValueError:
            w_stat, p_val = 0.0, 1.0

        sig_raw = "✅ Yes" if p_val < 0.05 else "❌ No"
        sig_corrected = "✅ Yes" if p_val < bonferroni_alpha else "❌ No"
        report_lines.append(
            f"| {name} | {mean_d:.4f} | {mean_i:.4f} | {delta:+.4f} | {w_stat:.1f} | {p_val:.4f} | {d_effect:+.3f} | {sig_raw} | {sig_corrected} |"
        )

    report_lines.append("")

    # ===== Test 2: LP vs RP =====
    report_lines.append("## 2. Renderer Comparison: LP vs RP (Paired Wilcoxon)")
    report_lines.append("")
    report_lines.append(
        "| Model | LP Mean | RP Mean | Δ (RP-LP) | W-stat | p-value | Significant? |"
    )
    report_lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for model_name, lp_arr, rp_arr in [
        ("Default", d_lp, d_rp),
        ("Improved", i_lp, i_rp),
    ]:
        diff = rp_arr - lp_arr
        try:
            w_stat, p_val = stats.wilcoxon(rp_arr, lp_arr, alternative="two-sided")
        except ValueError:
            w_stat, p_val = 0.0, 1.0
        sig = "✅ Yes" if p_val < 0.05 else "❌ No"
        report_lines.append(
            f"| {model_name} | {np.mean(lp_arr):.4f} | {np.mean(rp_arr):.4f} | {np.mean(diff):+.4f} | {w_stat:.1f} | {p_val:.4f} | {sig} |"
        )

    report_lines.append("")

    # ===== Test 3: Summary statistics with CI =====
    report_lines.append("## 3. Summary Statistics (95% Bootstrap CI)")
    report_lines.append("")
    report_lines.append("| Method | Mean | Std | Median | 95% CI | Min | Max |")
    report_lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for name, arr in [
        ("Default LP", d_lp),
        ("Default RP", d_rp),
        ("Default Best", d_best),
        ("Improved LP", i_lp),
        ("Improved RP", i_rp),
        ("Improved Best", i_best),
        ("Improved Ensemble", i_ens),
    ]:
        ci_lo, ci_hi = bootstrap_ci(arr)
        report_lines.append(
            f"| {name} | {np.mean(arr):.4f} | {np.std(arr):.4f} | {np.median(arr):.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] | {np.min(arr):.4f} | {np.max(arr):.4f} |"
        )

    report_lines.append("")

    # ===== Test 4: Per-entity improvement direction =====
    report_lines.append("## 4. Per-Entity Improvement Analysis")
    report_lines.append("")

    n_lp_improved = np.sum(i_lp > d_lp)
    n_rp_improved = np.sum(i_rp > d_rp)
    n_ens_improved = np.sum(i_ens > d_best)

    report_lines.append(
        f"- LP improved in **{n_lp_improved}/{len(common)}** entities ({100 * n_lp_improved / len(common):.1f}%)"
    )
    report_lines.append(
        f"- RP improved in **{n_rp_improved}/{len(common)}** entities ({100 * n_rp_improved / len(common):.1f}%)"
    )
    report_lines.append(
        f"- Ensemble > Default Best in **{n_ens_improved}/{len(common)}** entities ({100 * n_ens_improved / len(common):.1f}%)"
    )
    report_lines.append("")

    # Top improvements
    report_lines.append("### Largest Improvements (Ensemble vs Default Best)")
    report_lines.append("")
    ens_diffs = [(common[i], float(i_ens[i] - d_best[i])) for i in range(len(common))]
    ens_diffs.sort(key=lambda x: x[1], reverse=True)

    report_lines.append("| Entity | Default Best | Improved Ensemble | Δ |")
    report_lines.append("|---|:---:|:---:|:---:|")
    for entity, delta in ens_diffs[:5]:
        idx = common.index(entity)
        report_lines.append(
            f"| {entity} | {d_best[idx]:.4f} | {i_ens[idx]:.4f} | {delta:+.4f} |"
        )

    report_lines.append("")
    report_lines.append("### Largest Regressions")
    report_lines.append("")
    report_lines.append("| Entity | Default Best | Improved Ensemble | Δ |")
    report_lines.append("|---|:---:|:---:|:---:|")
    for entity, delta in ens_diffs[-5:]:
        idx = common.index(entity)
        report_lines.append(
            f"| {entity} | {d_best[idx]:.4f} | {i_ens[idx]:.4f} | {delta:+.4f} |"
        )

    report_lines.append("")

    # ===== Test 5: Sign test (non-parametric) =====
    report_lines.append("## 5. Sign Test (Alternative to Wilcoxon)")
    report_lines.append("")
    report_lines.append(
        "| Comparison | n+ | n- | n0 | p-value (binomial) | Significant? |"
    )
    report_lines.append("|---|:---:|:---:|:---:|:---:|:---:|")

    for name, arr1, arr2 in [
        ("Improved LP vs Default LP", i_lp, d_lp),
        ("Improved RP vs Default RP", i_rp, d_rp),
        ("Improved Ensemble vs Default Best", i_ens, d_best),
    ]:
        diff = arr1 - arr2
        n_pos = np.sum(diff > 0)
        n_neg = np.sum(diff < 0)
        n_zero = np.sum(diff == 0)
        n_nonzero = n_pos + n_neg
        if n_nonzero > 0:
            p_val = float(stats.binomtest(min(n_pos, n_neg), n_nonzero, 0.5).pvalue)
        else:
            p_val = 1.0
        sig = "✅ Yes" if p_val < 0.05 else "❌ No"
        report_lines.append(
            f"| {name} | {n_pos} | {n_neg} | {n_zero} | {p_val:.4f} | {sig} |"
        )

    return "\n".join(report_lines)


def main() -> None:
    """Run statistical tests and save report."""
    LOGGER.info("Running statistical significance tests...")
    report = run_tests()

    output_path = RESULTS_ROOT / "reports" / "statistical_tests.md"
    output_path.write_text(report)
    LOGGER.info("Report saved to %s", output_path)

    # Also log to console
    LOGGER.info("\n%s", report)


if __name__ == "__main__":
    main()
