"""Exchangeability analysis for CalibGuard conformal prediction scores.

CalibGuard assumes calibration and test scores are exchangeable (i.i.d.).
For time series, temporal autocorrelation can violate this assumption.
This script tests exchangeability via autocorrelation, runs test, and
block bootstrap comparison on normal-only anomaly scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULT_DIRS: dict[str, Path] = {
    "SMD-m11": Path("results/spatial_pilot/smd_m11_lp"),
    "PSM": Path("results/spatial_pilot/psm_lp"),
    "MSL": Path("results/spatial_pilot/msl_lp"),
    "SMAP": Path("results/spatial_pilot/smap_lp"),
}

MAX_LAG: int = 10
BLOCK_SIZE: int = 50
SIGNIFICANCE_LEVEL: float = 0.05
OUTPUT_PATH: Path = Path("results/exchangeability_analysis.json")


# ---------------------------------------------------------------------------
# Test implementations
# ---------------------------------------------------------------------------


def compute_autocorrelation(
    scores: NDArray[np.float64], max_lag: int = MAX_LAG
) -> dict[str, Any]:
    """Compute lag-1 to lag-K autocorrelation and test significance.

    Args:
        scores: 1-D array of normal-only anomaly scores.
        max_lag: Maximum lag to compute.

    Returns:
        Dictionary with per-lag autocorrelation values and significance flags.
    """
    n = len(scores)
    mean = scores.mean()
    var = np.sum((scores - mean) ** 2) / n
    if var == 0:
        return {
            "lags": list(range(1, max_lag + 1)),
            "values": [0.0] * max_lag,
            "significant": [False] * max_lag,
            "num_significant": 0,
        }

    acf_values: list[float] = []
    # Approximate 95% CI bound for white noise: +/- 1.96 / sqrt(n)
    ci_bound = 1.96 / np.sqrt(n)

    for lag in range(1, max_lag + 1):
        c = np.sum((scores[:n - lag] - mean) * (scores[lag:] - mean)) / n
        acf_values.append(float(c / var))

    significant = [abs(v) > ci_bound for v in acf_values]
    return {
        "lags": list(range(1, max_lag + 1)),
        "values": [round(v, 6) for v in acf_values],
        "ci_bound_95": round(float(ci_bound), 6),
        "significant": significant,
        "num_significant": sum(significant),
    }


def runs_test(scores: NDArray[np.float64]) -> dict[str, Any]:
    """Wald-Wolfowitz runs test for randomness around the median.

    Args:
        scores: 1-D array of normal-only anomaly scores.

    Returns:
        Dictionary with observed runs, expected runs, z-statistic, p-value.
    """
    median = np.median(scores)
    binary = (scores > median).astype(int)

    # Count runs (consecutive sequences of same value)
    runs = 1 + int(np.sum(np.diff(binary) != 0))

    n1 = int(np.sum(binary == 1))
    n0 = int(np.sum(binary == 0))
    n = n1 + n0

    if n1 == 0 or n0 == 0:
        return {
            "observed_runs": runs,
            "expected_runs": None,
            "z_statistic": None,
            "p_value": None,
            "conclusion": "degenerate — all values on one side of median",
        }

    expected_runs = 1.0 + (2.0 * n1 * n0) / n
    var_runs = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n)) / (n ** 2 * (n - 1))

    if var_runs <= 0:
        return {
            "observed_runs": runs,
            "expected_runs": round(expected_runs, 4),
            "z_statistic": None,
            "p_value": None,
            "conclusion": "degenerate variance",
        }

    z = (runs - expected_runs) / np.sqrt(var_runs)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return {
        "observed_runs": runs,
        "expected_runs": round(expected_runs, 4),
        "z_statistic": round(float(z), 4),
        "p_value": round(float(p_value), 6),
    }


def block_bootstrap_comparison(
    scores: NDArray[np.float64], block_size: int = BLOCK_SIZE
) -> dict[str, Any]:
    """Compare block means to assess stationarity.

    Under exchangeability, non-overlapping block means should be similar.
    We use a Kruskal-Wallis test on blocks.

    Args:
        scores: 1-D array of normal-only anomaly scores.
        block_size: Size of each non-overlapping block.

    Returns:
        Dictionary with number of blocks, block mean stats, test result.
    """
    n = len(scores)
    n_blocks = n // block_size
    if n_blocks < 3:
        return {
            "n_blocks": n_blocks,
            "block_size": block_size,
            "conclusion": f"too few blocks ({n_blocks}) for meaningful test",
        }

    # Create non-overlapping blocks
    blocks = [
        scores[i * block_size : (i + 1) * block_size] for i in range(n_blocks)
    ]
    block_means = np.array([b.mean() for b in blocks])

    # Kruskal-Wallis on the raw blocks (not just means)
    stat, p_value = stats.kruskal(*blocks)

    return {
        "n_blocks": n_blocks,
        "block_size": block_size,
        "block_mean_min": round(float(block_means.min()), 6),
        "block_mean_max": round(float(block_means.max()), 6),
        "block_mean_std": round(float(block_means.std()), 6),
        "kruskal_wallis_stat": round(float(stat), 4),
        "kruskal_wallis_p": round(float(p_value), 6),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_dataset(
    name: str, result_dir: Path
) -> dict[str, Any]:
    """Run all exchangeability tests on a single dataset.

    Args:
        name: Human-readable dataset name.
        result_dir: Path to the result directory containing scores.npy and labels.npy.

    Returns:
        Dictionary with all test results and overall conclusion.
    """
    scores_path = result_dir / "scores.npy"
    labels_path = result_dir / "labels.npy"

    if not scores_path.exists() or not labels_path.exists():
        LOGGER.warning("Missing files in %s, skipping.", result_dir)
        return {"error": f"missing scores.npy or labels.npy in {result_dir}"}

    scores: NDArray[np.float64] = np.load(scores_path).astype(np.float64)
    labels: NDArray[np.int64] = np.load(labels_path).astype(np.int64)

    # Use only normal (label=0) scores — exchangeability should hold on normals
    normal_mask = labels == 0
    normal_scores = scores[normal_mask]
    LOGGER.info(
        "%s: %d total, %d normal, %d anomaly",
        name,
        len(scores),
        int(normal_mask.sum()),
        int((~normal_mask).sum()),
    )

    acf_result = compute_autocorrelation(normal_scores)
    runs_result = runs_test(normal_scores)
    block_result = block_bootstrap_comparison(normal_scores)

    # Overall conclusion
    violations: list[str] = []
    if acf_result["num_significant"] >= 3:
        violations.append(
            f"autocorrelation significant at {acf_result['num_significant']}/{MAX_LAG} lags"
        )
    if runs_result.get("p_value") is not None and runs_result["p_value"] < SIGNIFICANCE_LEVEL:
        violations.append(
            f"runs test rejected (p={runs_result['p_value']:.4f})"
        )
    if block_result.get("kruskal_wallis_p") is not None and block_result["kruskal_wallis_p"] < SIGNIFICANCE_LEVEL:
        violations.append(
            f"block means differ (Kruskal-Wallis p={block_result['kruskal_wallis_p']:.4f})"
        )

    if violations:
        conclusion = "VIOLATED — " + "; ".join(violations)
    else:
        conclusion = "NOT REJECTED — no strong evidence against exchangeability"

    return {
        "n_total": len(scores),
        "n_normal": int(normal_mask.sum()),
        "autocorrelation": acf_result,
        "runs_test": runs_result,
        "block_bootstrap": block_result,
        "exchangeability_conclusion": conclusion,
    }


def print_summary(results: dict[str, Any]) -> None:
    """Print a formatted summary table of exchangeability analysis.

    Args:
        results: Full results dictionary keyed by dataset name.
    """
    header = f"{'Dataset':<12} {'N_normal':>8} {'ACF sig':>8} {'Runs p':>10} {'KW p':>10} {'Conclusion'}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("Exchangeability Analysis Summary")
    print(sep)
    print(header)
    print(sep)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<12} {'ERROR':>8}")
            continue

        n_normal = res["n_normal"]
        acf_sig = res["autocorrelation"]["num_significant"]
        runs_p = res["runs_test"].get("p_value")
        kw_p = res["block_bootstrap"].get("kruskal_wallis_p")

        runs_str = f"{runs_p:.4f}" if runs_p is not None else "N/A"
        kw_str = f"{kw_p:.4f}" if kw_p is not None else "N/A"

        violated = "VIOLATED" in res["exchangeability_conclusion"]
        tag = "VIOLATED" if violated else "OK"

        print(f"{name:<12} {n_normal:>8} {acf_sig:>6}/10 {runs_str:>10} {kw_str:>10} {tag}")

    print(sep)
    print(f"Significance level: alpha = {SIGNIFICANCE_LEVEL}")
    print(f"ACF violation threshold: >= 3/{MAX_LAG} lags significant")
    print(sep + "\n")


def main() -> None:
    """Run exchangeability analysis on all configured datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    results: dict[str, Any] = {}
    for name, result_dir in RESULT_DIRS.items():
        LOGGER.info("Analyzing %s ...", name)
        results[name] = analyze_dataset(name, result_dir)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    LOGGER.info("Results saved to %s", OUTPUT_PATH)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
