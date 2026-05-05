#!/usr/bin/env python3
"""Ensemble improved PatchTraj scores (LP + RP) for each dataset.

Loads scores from results/improved_{dataset}/, applies weighted ensemble,
and reports per-entity and aggregate metrics. Tries multiple configs to
find the best for each dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from src.evaluation.metrics import compute_all_metrics
from src.scoring.patchtraj_scorer import normalize_scores, smooth_scores

LOGGER = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")

ENSEMBLE_METHODS = ["zscore_weighted", "rank_weighted"]
SMOOTH_METHODS = ["mean", "median"]
SMOOTH_WINDOWS = [7, 15, 21]
LP_WEIGHTS = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5]


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
        method: Ensemble method.
        w_lp: Weight for line plot scores.

    Returns:
        Ensemble scores of shape (T,).
    """
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

    raise ValueError(f"Unknown ensemble method: {method}")


def _process_entity(
    entity_dir: Path,
    smooth_method: str,
    smooth_window: int,
    ensemble_method: str,
    w_lp: float,
) -> dict[str, Any] | None:
    """Process a single entity with given config.

    Args:
        entity_dir: Directory with line_plot/ and recurrence_plot/ subdirs.
        smooth_method: Smoothing method.
        smooth_window: Smoothing window size.
        ensemble_method: Ensemble combination method.
        w_lp: Weight for line plot in ensemble.

    Returns:
        Dict with metrics or None if data missing.
    """
    lp_path = entity_dir / "line_plot" / "scores.npy"
    rp_path = entity_dir / "recurrence_plot" / "scores.npy"

    if not lp_path.exists() or not rp_path.exists():
        return None

    lp_scores = np.load(lp_path).astype(np.float64)
    rp_scores = np.load(rp_path).astype(np.float64)

    # Labels from either renderer
    for renderer in ["line_plot", "recurrence_plot"]:
        labels_path = entity_dir / renderer / "labels.npy"
        if labels_path.exists():
            labels = np.load(labels_path).astype(np.int64)
            break
    else:
        return None

    # Truncate to common length
    min_len = min(lp_scores.shape[0], rp_scores.shape[0], labels.shape[0])
    lp_scores = lp_scores[:min_len]
    rp_scores = rp_scores[:min_len]
    labels = labels[:min_len]

    # Check labels have both classes
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None

    # Ensemble
    ensemble = _weighted_ensemble(lp_scores, rp_scores, ensemble_method, w_lp)

    # Smooth
    sw = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    if sw > 1:
        ensemble = smooth_scores(ensemble, window_size=sw, method=smooth_method)

    ensemble = normalize_scores(ensemble, method="minmax")
    metrics = compute_all_metrics(ensemble, labels)

    return {
        "auc_roc": float(metrics["auc_roc"]),
        "auc_pr": float(metrics["auc_pr"]),
        "f1_pa": float(metrics["f1_pa"]),
    }


def find_best_config(
    dataset_dir: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Grid search for best ensemble config on completed entities.

    .. warning::
        This grid search optimises ensemble hyperparameters (smoothing window,
        smoothing method, ensemble method, LP weight) by maximising AUC-ROC
        averaged across all entities, where each entity's AUC-ROC is computed
        on its own test labels. Although AUC-ROC is threshold-free, the
        hyperparameter selection step uses the test set, so the resulting
        configuration represents an **oracle upper bound**. Numbers produced by
        this function must be reported as such; for leakage-free deployment,
        use :func:`find_best_config_loo` (leave-one-entity-out).

    Args:
        dataset_dir: e.g., results/improved_smd/

    Returns:
        Tuple of (best_config_dict, per_entity_results).
    """
    best_auc = -1.0
    best_config: dict[str, Any] = {}
    best_per_entity: dict[str, dict] = {}

    for sm in SMOOTH_METHODS:
        for sw in SMOOTH_WINDOWS:
            for em in ENSEMBLE_METHODS:
                for w in LP_WEIGHTS:
                    entity_aucs: list[float] = []
                    per_entity: dict[str, dict] = {}

                    for entity_dir in sorted(dataset_dir.iterdir()):
                        if not entity_dir.is_dir():
                            continue
                        result = _process_entity(entity_dir, sm, sw, em, w)
                        if result:
                            entity_aucs.append(result["auc_roc"])
                            per_entity[entity_dir.name] = result

                    if not entity_aucs:
                        continue

                    avg_auc = float(np.mean(entity_aucs))
                    if avg_auc > best_auc:
                        best_auc = avg_auc
                        best_config = {
                            "smooth_method": sm,
                            "smooth_window": sw,
                            "ensemble_method": em,
                            "w_lp": w,
                            "avg_auc_roc": avg_auc,
                            "n_entities": len(entity_aucs),
                        }
                        best_per_entity = per_entity

    best_config["oracle_warning"] = (
        "Hyperparameters were selected by maximising AUC-ROC over the same "
        "entities used for evaluation. Numbers reflect an oracle upper bound, "
        "not deployment performance."
    )
    return best_config, best_per_entity


def find_best_config_loo(
    dataset_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict]]:
    """Leave-one-entity-out hyperparameter selection (leakage-free).

    For each held-out entity:
      1. Run the same grid as :func:`find_best_config` over the **other**
         entities, picking the configuration with the best mean AUC-ROC.
      2. Apply that configuration to the held-out entity and record the result.

    The returned per-entity dict contains scores under leakage-free
    hyperparameters; the aggregate is the mean across entities.

    Args:
        dataset_dir: e.g., results/improved_smd/

    Returns:
        Tuple of (summary_dict, per_entity_results) where summary_dict reports
        ``mean_auc_roc`` aggregated under leave-one-out selection.
    """
    entities = sorted(
        e for e in dataset_dir.iterdir() if e.is_dir()
    )
    if not entities:
        return {"mean_auc_roc": float("nan"), "n_entities": 0}, {}

    per_entity_loo: dict[str, dict] = {}
    for held_out in entities:
        train_entities = [e for e in entities if e != held_out]

        # Inner grid: pick best config on training entities
        best_inner_auc = -1.0
        best_inner_cfg: dict[str, Any] | None = None
        for sm in SMOOTH_METHODS:
            for sw in SMOOTH_WINDOWS:
                for em in ENSEMBLE_METHODS:
                    for w in LP_WEIGHTS:
                        train_aucs: list[float] = []
                        for entity in train_entities:
                            res = _process_entity(entity, sm, sw, em, w)
                            if res:
                                train_aucs.append(res["auc_roc"])
                        if not train_aucs:
                            continue
                        mean_auc = float(np.mean(train_aucs))
                        if mean_auc > best_inner_auc:
                            best_inner_auc = mean_auc
                            best_inner_cfg = {
                                "smooth_method": sm,
                                "smooth_window": sw,
                                "ensemble_method": em,
                                "w_lp": w,
                            }
        if best_inner_cfg is None:
            continue
        held_result = _process_entity(
            held_out,
            best_inner_cfg["smooth_method"],
            best_inner_cfg["smooth_window"],
            best_inner_cfg["ensemble_method"],
            best_inner_cfg["w_lp"],
        )
        if held_result:
            per_entity_loo[held_out.name] = {
                **held_result,
                "selected_config": best_inner_cfg,
            }

    if not per_entity_loo:
        return {"mean_auc_roc": float("nan"), "n_entities": 0}, {}

    aucs = [r["auc_roc"] for r in per_entity_loo.values()]
    summary = {
        "mean_auc_roc": float(np.mean(aucs)),
        "std_auc_roc": float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0),
        "n_entities": len(aucs),
        "selection_protocol": "leave-one-entity-out",
        "note": "Hyperparameters selected on N-1 entities, evaluated on held-out entity.",
    }
    return summary, per_entity_loo


def main() -> None:
    """Run improved model ensemble for all available datasets."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ensemble improved PatchTraj scores")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset (smd/psm/msl/smap). Default: all available.",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["smd", "psm", "msl", "smap"]
    output_dir = RESULTS_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}

    for ds in datasets:
        ds_dir = RESULTS_ROOT / f"improved_{ds}"
        if not ds_dir.exists():
            LOGGER.info("No improved results for %s, skipping", ds)
            continue

        LOGGER.info("Processing %s...", ds)

        # Also report individual renderer performance
        lp_aucs = []
        rp_aucs = []
        for entity_dir in sorted(ds_dir.iterdir()):
            if not entity_dir.is_dir():
                continue
            lp_m = entity_dir / "line_plot" / "metrics.json"
            rp_m = entity_dir / "recurrence_plot" / "metrics.json"
            if lp_m.exists():
                lp_aucs.append(json.loads(lp_m.read_text())["auc_roc"])
            if rp_m.exists():
                rp_aucs.append(json.loads(rp_m.read_text())["auc_roc"])

        LOGGER.info(
            "  Individual: LP avg=%.4f (n=%d), RP avg=%.4f (n=%d)",
            np.mean(lp_aucs) if lp_aucs else 0,
            len(lp_aucs),
            np.mean(rp_aucs) if rp_aucs else 0,
            len(rp_aucs),
        )

        # Grid search for best ensemble
        best_config, per_entity = find_best_config(ds_dir)

        if not best_config:
            LOGGER.info("  No complete entities (need both LP+RP), skipping ensemble")
            continue

        LOGGER.info(
            "  Best ensemble: %s (sw=%d, %s, w_lp=%.2f) → AUC-ROC=%.4f (n=%d)",
            best_config["ensemble_method"],
            best_config["smooth_window"],
            best_config["smooth_method"],
            best_config["w_lp"],
            best_config["avg_auc_roc"],
            best_config["n_entities"],
        )

        for entity_name, metrics in sorted(per_entity.items()):
            LOGGER.info(
                "    %s: AUC-ROC=%.4f, AUC-PR=%.4f, F1-PA=%.4f",
                entity_name,
                metrics["auc_roc"],
                metrics["auc_pr"],
                metrics["f1_pa"],
            )

        all_results[ds] = {
            "best_config": best_config,
            "per_entity": per_entity,
            "individual_lp_avg": float(np.mean(lp_aucs)) if lp_aucs else None,
            "individual_rp_avg": float(np.mean(rp_aucs)) if rp_aucs else None,
            "individual_lp_count": len(lp_aucs),
            "individual_rp_count": len(rp_aucs),
        }

    # Save
    output_path = output_dir / "improved_ensemble_results.json"
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    LOGGER.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
