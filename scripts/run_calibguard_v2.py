#!/usr/bin/env python3
"""Run experimental CalibGuard v1 vs v2 FAR diagnostics across datasets."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

RESULTS_ROOT = Path("results")
REPORTS_DIR = RESULTS_ROOT / "reports"
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

Scalar = Union[float, int, str, None]
ResultRow = Dict[str, Scalar]
EvalReturn = Dict[str, Union[float, int]]
EvalFn = Callable[..., EvalReturn]


def _resolve_method(method_name: str) -> EvalFn:
    """Resolve evaluation function from method name.

    Args:
        method_name: Method identifier ("v1" or "v2").

    Returns:
        Callable FAR evaluation function.

    Raises:
        ValueError: If method name is unknown.
    """
    if method_name == "v1":
        module = importlib.import_module("src.scoring.calibguard")
        return cast(EvalFn, getattr(module, "compute_far_at_alpha"))
    if method_name == "v2":
        module = importlib.import_module("src.scoring.calibguard_v2")
        return cast(EvalFn, getattr(module, "compute_far_at_alpha_v2"))
    raise ValueError(f"Unsupported method_name: {method_name}")


def load_scores_and_labels(
    entity_dir: Path,
    renderer: str,
) -> Optional[Tuple[NDArray[np.float64], NDArray[np.int64]]]:
    """Load score and label arrays for one entity and renderer.

    Args:
        entity_dir: Entity or dataset result directory.
        renderer: Renderer directory name.

    Returns:
        Tuple of (scores, labels) when valid files exist; otherwise None.
    """
    score_candidates = ["test_scores.npy", "scores.npy"]
    label_candidates = ["test_labels.npy", "labels.npy"]

    scores_path = None
    labels_path = None
    for name in score_candidates:
        candidate = entity_dir / renderer / name
        if candidate.exists():
            scores_path = candidate
            break
    for name in label_candidates:
        candidate = entity_dir / renderer / name
        if candidate.exists():
            labels_path = candidate
            break

    if scores_path is None or labels_path is None:
        return None

    scores_raw = cast(NDArray[np.float64], np.load(scores_path))
    labels_raw = cast(NDArray[np.int64], np.load(labels_path))
    scores = np.asarray(scores_raw, dtype=np.float64)
    labels = np.asarray(labels_raw, dtype=np.int64)

    min_len = min(int(scores.shape[0]), int(labels.shape[0]))
    if min_len == 0:
        return None

    scores = scores[:min_len]
    labels = labels[:min_len]
    label_values = cast(List[int], labels.astype(np.int64).tolist())
    normal_count = sum(1 for x in label_values if x == 0)
    anomaly_count = sum(1 for x in label_values if x == 1)
    if normal_count == 0 or anomaly_count == 0:
        return None
    return scores, labels


def run_method(
    method_name: str,
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    alpha: float,
    calibration_ratio: float,
    bonferroni_n_tests: int,
) -> ResultRow:
    """Evaluate one method for one alpha.

    Args:
        method_name: Method identifier ("v1" or "v2").
        scores: Raw anomaly scores.
        labels: Binary labels aligned with scores.
        alpha: Target FAR.
        calibration_ratio: Fraction of normal samples for calibration.
        bonferroni_n_tests: Number of simultaneous tests for v2.

    Returns:
        Result row with metrics or an error field.
    """
    try:
        method_fn = _resolve_method(method_name)
        if method_name == "v2":
            result = method_fn(
                scores=scores,
                labels=labels,
                alpha=alpha,
                calibration_ratio=calibration_ratio,
                use_aci=True,
                aci_gamma=0.02,
                bonferroni_n_tests=bonferroni_n_tests,
                rolling_window=0,
            )
        else:
            result = method_fn(
                scores=scores,
                labels=labels,
                alpha=alpha,
                calibration_ratio=calibration_ratio,
            )
        actual_far = float(result["actual_far"])
        return {
            "alpha": alpha,
            "target_far": alpha,
            "actual_far": actual_far,
            "coverage": float(result["coverage"]),
            "threshold": float(result["threshold"]),
            "n_calibration": int(result["n_calibration"]),
            "n_test_normal": int(result["n_test_normal"]),
            "n_test_anomaly": int(result["n_test_anomaly"]),
            "far_violation": actual_far - alpha,
            "effective_alpha": float(result.get("effective_alpha", alpha)),
            "alpha_t": float(result.get("alpha_t", alpha)),
        }
    except (ImportError, AttributeError, ValueError, RuntimeError, KeyError) as exc:
        LOGGER.warning("%s failed at alpha %.3f: %s", method_name, alpha, exc)
        return {
            "alpha": alpha,
            "target_far": alpha,
            "actual_far": None,
            "coverage": None,
            "error": str(exc),
        }


def summarize_per_alpha(rows: List[ResultRow], alpha: float) -> ResultRow:
    """Aggregate per-entity results for one alpha.

    Args:
        rows: Per-entity method rows.
        alpha: Target FAR.

    Returns:
        Aggregated statistics for this alpha.
    """
    valid_rows = [
        r for r in rows if r.get("alpha") == alpha and r.get("actual_far") is not None
    ]
    if not valid_rows:
        return {
            "alpha": alpha,
            "mean_far": None,
            "std_far": None,
            "mean_coverage": None,
            "n_entities": 0,
            "far_guarantee_entity_coverage_pct": None,
        }

    fars = np.asarray(
        [float(cast(float, r["actual_far"])) for r in valid_rows], dtype=np.float64
    )
    covs = np.asarray(
        [float(cast(float, r["coverage"])) for r in valid_rows], dtype=np.float64
    )
    return {
        "alpha": alpha,
        "mean_far": float(np.mean(fars)),
        "std_far": float(np.std(fars)),
        "mean_coverage": float(np.mean(covs)),
        "n_entities": int(fars.size),
        "far_guarantee_entity_coverage_pct": 100.0 * float(np.mean(fars <= alpha)),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.warning(
        "Running experimental CalibGuard diagnostics. These outputs are not claim-bearing because they calibrate from held-out test normals."
    )

    results: Dict[str, object] = {
        "config": {
            "alphas": ALPHAS,
            "calibration_ratio_v1": 0.5,
            "calibration_ratio_v2": 0.8,
            "v2_use_aci": True,
            "v2_aci_gamma": 0.02,
            "v2_bonferroni_smd_n_tests": len(SMD_ENTITIES),
        },
        "smd": {"entities": {}, "aggregate": {"v1": {}, "v2": {}}},
        "other_datasets": {},
    }

    md_lines = [
        "# CalibGuard v1 vs v2 FAR Comparison",
        "",
        "## Setup",
        "- Alphas: [0.01, 0.05, 0.10]",
        "- v1: split conformal, calibration ratio = 0.5",
        "- v2: z-score normalization + calibration ratio = 0.8 + ACI + Bonferroni",
        "",
        "## SMD Per-Entity (recurrence_plot fallback line_plot)",
        "",
        "| Entity | Alpha | v1 FAR | v2 FAR | v1 Cov | v2 Cov | FAR<=alpha v1 | FAR<=alpha v2 |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    smd_rows_v1: List[ResultRow] = []
    smd_rows_v2: List[ResultRow] = []

    for entity in SMD_ENTITIES:
        entity_dir = RESULTS_ROOT / "improved_smd" / entity
        data = load_scores_and_labels(entity_dir, "recurrence_plot")
        renderer = "recurrence_plot"
        if data is None:
            data = load_scores_and_labels(entity_dir, "line_plot")
            renderer = "line_plot"
        if data is None:
            LOGGER.warning("Skipping %s: no score/label files", entity)
            continue

        scores, labels = data
        entity_result: Dict[str, Union[str, List[ResultRow]]] = {
            "renderer": renderer,
            "v1": [],
            "v2": [],
        }
        for alpha in ALPHAS:
            row_v1 = run_method(
                "v1", scores, labels, alpha, calibration_ratio=0.5, bonferroni_n_tests=1
            )
            row_v2 = run_method(
                "v2",
                scores,
                labels,
                alpha,
                calibration_ratio=0.8,
                bonferroni_n_tests=len(SMD_ENTITIES),
            )
            cast(List[ResultRow], entity_result["v1"]).append(row_v1)
            cast(List[ResultRow], entity_result["v2"]).append(row_v2)
            smd_rows_v1.append(row_v1)
            smd_rows_v2.append(row_v2)

            if row_v1.get("actual_far") is None or row_v2.get("actual_far") is None:
                continue
            v1_far = float(cast(float, row_v1["actual_far"]))
            v2_far = float(cast(float, row_v2["actual_far"]))
            v1_cov = float(cast(float, row_v1["coverage"]))
            v2_cov = float(cast(float, row_v2["coverage"]))
            v1_ok = "Y" if v1_far <= alpha else "N"
            v2_ok = "Y" if v2_far <= alpha else "N"
            md_lines.append(
                f"| {entity} | {alpha:.2f} | {v1_far:.4f} | {v2_far:.4f} | {v1_cov:.4f} | {v2_cov:.4f} | {v1_ok} | {v2_ok} |"
            )

        cast(Dict[str, object], cast(Dict[str, object], results["smd"])["entities"])[
            entity
        ] = entity_result

    md_lines.extend(
        [
            "",
            "## SMD Aggregate FAR Guarantee Coverage",
            "",
            "| Alpha | Method | Mean FAR | Std FAR | Mean Coverage | Entity FAR<=alpha (%) | n_entities |",
            "|:---:|---|:---:|:---:|:---:|:---:|:---:|",
        ]
    )

    smd_block = cast(Dict[str, object], results["smd"])
    aggregate = cast(Dict[str, object], smd_block["aggregate"])
    aggregate_v1 = cast(Dict[str, object], aggregate["v1"])
    aggregate_v2 = cast(Dict[str, object], aggregate["v2"])

    for alpha in ALPHAS:
        agg_v1 = summarize_per_alpha(smd_rows_v1, alpha)
        agg_v2 = summarize_per_alpha(smd_rows_v2, alpha)
        aggregate_v1[str(alpha)] = agg_v1
        aggregate_v2[str(alpha)] = agg_v2

        if agg_v1["mean_far"] is not None:
            md_lines.append(
                f"| {alpha:.2f} | v1 | {float(cast(float, agg_v1['mean_far'])):.4f} | {float(cast(float, agg_v1['std_far'])):.4f} | {float(cast(float, agg_v1['mean_coverage'])):.4f} | {float(cast(float, agg_v1['far_guarantee_entity_coverage_pct'])):.1f}% | {int(cast(int, agg_v1['n_entities']))} |"
            )
        if agg_v2["mean_far"] is not None:
            md_lines.append(
                f"| {alpha:.2f} | v2 | {float(cast(float, agg_v2['mean_far'])):.4f} | {float(cast(float, agg_v2['std_far'])):.4f} | {float(cast(float, agg_v2['mean_coverage'])):.4f} | {float(cast(float, agg_v2['far_guarantee_entity_coverage_pct'])):.1f}% | {int(cast(int, agg_v2['n_entities']))} |"
            )

    md_lines.extend(
        [
            "",
            "## Other Datasets (PSM/MSL/SMAP)",
            "",
            "| Dataset | Renderer | Alpha | v1 FAR | v2 FAR | v1 Cov | v2 Cov |",
            "|---|---|:---:|:---:|:---:|:---:|:---:|",
        ]
    )

    other_datasets = cast(Dict[str, object], results["other_datasets"])
    for dataset in ["psm", "msl", "smap"]:
        dataset_dir = RESULTS_ROOT / f"improved_{dataset}" / "default"
        if not dataset_dir.exists():
            LOGGER.warning("Skipping dataset %s: %s missing", dataset, dataset_dir)
            continue

        dataset_result: Dict[str, object] = {}
        for renderer in ["line_plot", "recurrence_plot"]:
            data = load_scores_and_labels(dataset_dir, renderer)
            if data is None:
                continue
            scores, labels = data

            renderer_rows: Dict[str, List[ResultRow]] = {"v1": [], "v2": []}
            for alpha in ALPHAS:
                row_v1 = run_method(
                    "v1",
                    scores,
                    labels,
                    alpha,
                    calibration_ratio=0.5,
                    bonferroni_n_tests=1,
                )
                row_v2 = run_method(
                    "v2",
                    scores,
                    labels,
                    alpha,
                    calibration_ratio=0.8,
                    bonferroni_n_tests=1,
                )
                renderer_rows["v1"].append(row_v1)
                renderer_rows["v2"].append(row_v2)

                if row_v1.get("actual_far") is None or row_v2.get("actual_far") is None:
                    continue
                md_lines.append(
                    f"| {dataset.upper()} | {renderer} | {alpha:.2f} | {float(cast(float, row_v1['actual_far'])):.4f} | {float(cast(float, row_v2['actual_far'])):.4f} | {float(cast(float, row_v1['coverage'])):.4f} | {float(cast(float, row_v2['coverage'])):.4f} |"
                )

            dataset_result[renderer] = renderer_rows
        other_datasets[dataset] = dataset_result

    json_path = REPORTS_DIR / "experimental_calibguard_v2.json"
    md_path = REPORTS_DIR / "experimental_calibguard_v2.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    _ = md_path.write_text("\n".join(md_lines), encoding="utf-8")

    LOGGER.info("Saved JSON report: %s", json_path)
    LOGGER.info("Saved markdown report: %s", md_path)


if __name__ == "__main__":
    main()
