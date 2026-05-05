#!/usr/bin/env python3
from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnusedFunction=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false

import json
import logging
import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patchtraj import PatchTrajPredictor
from src.rendering.token_correspondence import compute_correspondence_map
from src.scoring.calibguard_v3 import CalibGuardV3
from src.scoring.patchtraj_scorer import (
    compute_patchtraj_score,
    normalize_scores,
    smooth_scores,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

RESULTS_ROOT = Path("results")
DEFAULT_REPORT_DIR = RESULTS_ROOT / "reports" / "calibguard_v3"
DEFAULT_EVIDENCE_PATH = Path(".sisyphus/evidence/task-10-calibguard-v3-results.json")

ALPHAS = [0.01, 0.05, 0.10]
CALIB_RATIO = 0.20
WINDOW_SIZE = 100
STRIDE = 10
SEQUENCE_LENGTH = 12
PREDICTION_DELTA = 1
SMOOTH_WINDOW = 21
SMOOTH_METHOD = "mean"
INFERENCE_BATCH_SIZE = 64

PATCHTRAJ_MODEL_CONFIG = {
    "d_model": 384,
    "n_heads": 6,
    "n_layers": 3,
    "dim_feedforward": 1536,
    "dropout": 0.1,
    "activation": "gelu",
}

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    _ = parser.add_argument(
        "--evidence-path", type=Path, default=DEFAULT_EVIDENCE_PATH
    )
    return parser.parse_args()


def _load_metric_auc(metrics_path: Path) -> float:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return float(metrics["auc_roc"])


def _select_dataset_renderer(dataset: str) -> str:
    dataset_root = RESULTS_ROOT / f"improved_{dataset}" / "default"
    candidates = ["line_plot", "recurrence_plot"]
    auc_by_renderer = {
        renderer: _load_metric_auc(dataset_root / renderer / "metrics.json")
        for renderer in candidates
    }
    sorted_renderers = sorted(
        auc_by_renderer.items(), key=lambda item: item[1], reverse=True
    )
    return sorted_renderers[0][0]


def _resolve_smd_renderer(entity: str) -> str:
    entity_root = RESULTS_ROOT / "improved_smd" / entity
    preferred = entity_root / "recurrence_plot"
    fallback = entity_root / "line_plot"
    if preferred.exists():
        return "recurrence_plot"
    if fallback.exists():
        return "line_plot"
    raise FileNotFoundError(f"No saved SMD renderer directory found for {entity}.")


def _load_token_cache(cache_path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    tokens = torch.as_tensor(payload["tokens"], dtype=torch.float32)
    patch_grid_raw = payload["patch_grid"]
    patch_grid = (int(patch_grid_raw[0]), int(patch_grid_raw[1]))
    return tokens, patch_grid


def _load_model(checkpoint_path: Path, hidden_dim: int) -> PatchTrajPredictor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model = PatchTrajPredictor(hidden_dim=hidden_dim, **PATCHTRAJ_MODEL_CONFIG)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _compute_stream_scores(
    tokens: torch.Tensor,
    model: PatchTrajPredictor,
    renderer: str,
    patch_grid: tuple[int, int],
) -> np.ndarray:
    pi, valid_mask = compute_correspondence_map(
        renderer_type=renderer,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        patch_grid=patch_grid,
    )

    num_samples = int(tokens.shape[0]) - SEQUENCE_LENGTH - PREDICTION_DELTA + 1
    if num_samples <= 0:
        raise ValueError(
            "Not enough tokens for score computation: "
            f"T={tokens.shape[0]}, K={SEQUENCE_LENGTH}, delta={PREDICTION_DELTA}."
        )

    windows = tokens.unfold(0, SEQUENCE_LENGTH + PREDICTION_DELTA, 1).permute(
        0, 3, 1, 2
    )
    score_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, num_samples, INFERENCE_BATCH_SIZE):
            end = min(start + INFERENCE_BATCH_SIZE, num_samples)
            batch_windows = windows[start:end]
            sequence_batch = batch_windows[:, :SEQUENCE_LENGTH]
            target_batch = batch_windows[:, SEQUENCE_LENGTH + PREDICTION_DELTA - 1]
            predicted_tokens = model(sequence_batch)
            batch_scores = compute_patchtraj_score(
                predicted_tokens=predicted_tokens,
                actual_tokens=target_batch,
                pi=pi,
                valid_mask=valid_mask,
            )
            score_chunks.append(batch_scores.cpu().numpy().astype(np.float64))

    scores = np.concatenate(score_chunks, axis=0)
    scores = smooth_scores(scores, window_size=SMOOTH_WINDOW, method=SMOOTH_METHOD)
    return normalize_scores(scores, method="minmax")


def _evaluate_alpha(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    guard = CalibGuardV3.from_train_split(
        train_scores=train_scores,
        calib_ratio=CALIB_RATIO,
        alpha=alpha,
        rolling_window=0,
        use_aci=False,
        bonferroni_n_tests=1,
    )
    flags, p_values, thresholds = guard.predict_batch(test_scores)
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    actual_far = float(np.mean(flags[normal_mask])) if np.any(normal_mask) else 0.0
    coverage = float(np.mean(flags[anomaly_mask])) if np.any(anomaly_mask) else 0.0
    abs_far_error = abs(actual_far - alpha)
    return {
        "alpha": alpha,
        "target_far": alpha,
        "actual_far": actual_far,
        "abs_far_error": abs_far_error,
        "coverage": coverage,
        "threshold": float(thresholds[0]) if thresholds.size > 0 else None,
        "mean_p_value": float(np.mean(p_values)) if p_values.size > 0 else None,
        "n_train_total": guard.n_train_total,
        "n_train_calibration": guard.n_train_calibration,
        "n_test_scores": int(test_scores.size),
        "n_test_normal": int(np.count_nonzero(normal_mask)),
        "n_test_anomaly": int(np.count_nonzero(anomaly_mask)),
        "far_le_alpha": bool(actual_far <= alpha),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _run_dataset_experiment(dataset: str, renderer: str) -> dict[str, Any]:
    result_dir = RESULTS_ROOT / f"improved_{dataset}" / "default" / renderer
    cache_path = (
        Path("data")
        / "processed"
        / dataset
        / f"tokens_facebook_dinov2-base_{dataset}_{renderer}.pt"
    )
    checkpoint_path = result_dir / "best_model.pt"
    test_scores = np.load(result_dir / "scores.npy").astype(np.float64)
    labels = np.load(result_dir / "labels.npy").astype(np.int64)
    train_tokens, patch_grid = _load_token_cache(cache_path)
    model = _load_model(checkpoint_path=checkpoint_path, hidden_dim=int(train_tokens.shape[-1]))
    train_scores = _compute_stream_scores(
        tokens=train_tokens,
        model=model,
        renderer=renderer,
        patch_grid=patch_grid,
    )

    rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        row = _evaluate_alpha(
            train_scores=train_scores,
            test_scores=test_scores,
            labels=labels,
            alpha=alpha,
        )
        row.update(
            {
                "dataset": dataset,
                "renderer": renderer,
                "checkpoint_path": str(checkpoint_path),
                "train_cache_path": str(cache_path),
            }
        )
        rows.append(row)

    return {
        "dataset": dataset,
        "renderer": renderer,
        "train_score_pipeline": {
            "source": "recomputed_from_train_token_cache",
            "smooth_window": SMOOTH_WINDOW,
            "smooth_method": SMOOTH_METHOD,
            "normalization": "minmax",
        },
        "test_score_pipeline": {
            "source": "saved_detection_scores",
            "path": str(result_dir / "scores.npy"),
        },
        "rows": rows,
    }


def _run_smd_experiment() -> dict[str, Any]:
    entity_results: list[dict[str, Any]] = []
    for entity in SMD_ENTITIES:
        renderer = _resolve_smd_renderer(entity)
        result_dir = RESULTS_ROOT / "improved_smd" / entity / renderer
        cache_path = (
            Path("data")
            / "processed"
            / "smd"
            / f"tokens_facebook_dinov2-base_{entity}_{renderer}.pt"
        )
        checkpoint_path = result_dir / "best_model.pt"
        test_scores = np.load(result_dir / "scores.npy").astype(np.float64)
        labels = np.load(result_dir / "labels.npy").astype(np.int64)
        train_tokens, patch_grid = _load_token_cache(cache_path)
        model = _load_model(
            checkpoint_path=checkpoint_path,
            hidden_dim=int(train_tokens.shape[-1]),
        )
        train_scores = _compute_stream_scores(
            tokens=train_tokens,
            model=model,
            renderer=renderer,
            patch_grid=patch_grid,
        )

        rows: list[dict[str, Any]] = []
        for alpha in ALPHAS:
            row = _evaluate_alpha(
                train_scores=train_scores,
                test_scores=test_scores,
                labels=labels,
                alpha=alpha,
            )
            row.update(
                {
                    "entity": entity,
                    "renderer": renderer,
                    "checkpoint_path": str(checkpoint_path),
                    "train_cache_path": str(cache_path),
                }
            )
            rows.append(row)

        entity_results.append(
            {
                "entity": entity,
                "renderer": renderer,
                "rows": rows,
            }
        )

    return {
        "dataset": "smd",
        "renderer_policy": "prefer recurrence_plot, fallback line_plot",
        "train_score_pipeline": {
            "source": "recomputed_from_train_token_cache",
            "smooth_window": SMOOTH_WINDOW,
            "smooth_method": SMOOTH_METHOD,
            "normalization": "minmax",
        },
        "test_score_pipeline": {
            "source": "saved_detection_scores",
            "path_template": "results/improved_smd/{entity}/{renderer}/scores.npy",
        },
        "entities": entity_results,
    }


def _summarize_smd_alpha(smd_result: dict[str, Any], alpha: float) -> dict[str, Any]:
    rows = []
    for entity_result in smd_result["entities"]:
        for row in entity_result["rows"]:
            if float(row["alpha"]) == alpha:
                rows.append(row)

    fars = np.asarray([float(row["actual_far"]) for row in rows], dtype=np.float64)
    coverages = np.asarray([float(row["coverage"]) for row in rows], dtype=np.float64)
    abs_errors = np.asarray([float(row["abs_far_error"]) for row in rows], dtype=np.float64)
    guarantee_mask = np.asarray([bool(row["far_le_alpha"]) for row in rows], dtype=bool)
    return {
        "alpha": alpha,
        "mean_far": float(np.mean(fars)),
        "std_far": float(np.std(fars)),
        "mean_abs_far_error": float(np.mean(abs_errors)),
        "mean_coverage": float(np.mean(coverages)),
        "entity_far_le_alpha_pct": 100.0 * float(np.mean(guarantee_mask)),
        "n_entities": int(len(rows)),
    }


def _build_summary(
    smd_result: dict[str, Any],
    dataset_results: list[dict[str, Any]],
) -> dict[str, Any]:
    smd_aggregate = {
        f"{alpha:.2f}": _summarize_smd_alpha(smd_result=smd_result, alpha=alpha)
        for alpha in ALPHAS
    }

    dataset_aggregate: dict[str, dict[str, Any]] = {
        "smd": {
            "per_alpha": smd_aggregate,
            "all_alphas_abs_far_error_lt_0_02": all(
                float(smd_aggregate[f"{alpha:.2f}"]["mean_abs_far_error"]) < 0.02
                for alpha in ALPHAS
            ),
        }
    }

    guarantee_checks: list[bool] = []
    guarantee_checks_by_alpha: dict[str, list[bool]] = {
        f"{alpha:.2f}": [] for alpha in ALPHAS
    }

    for alpha in ALPHAS:
        alpha_key = f"{alpha:.2f}"
        for entity_result in smd_result["entities"]:
            row = next(
                row_item
                for row_item in entity_result["rows"]
                if float(row_item["alpha"]) == alpha
            )
            guarantee_checks.append(bool(row["far_le_alpha"]))
            guarantee_checks_by_alpha[alpha_key].append(bool(row["far_le_alpha"]))

    for dataset_result in dataset_results:
        per_alpha: dict[str, dict[str, Any]] = {}
        for row in dataset_result["rows"]:
            alpha_key = f"{float(row['alpha']):.2f}"
            per_alpha[alpha_key] = {
                "renderer": dataset_result["renderer"],
                "actual_far": float(row["actual_far"]),
                "abs_far_error": float(row["abs_far_error"]),
                "coverage": float(row["coverage"]),
                "far_le_alpha": bool(row["far_le_alpha"]),
            }
            guarantee_checks.append(bool(row["far_le_alpha"]))
            guarantee_checks_by_alpha[alpha_key].append(bool(row["far_le_alpha"]))

        dataset_aggregate[dataset_result["dataset"]] = {
            "renderer": dataset_result["renderer"],
            "per_alpha": per_alpha,
            "all_alphas_abs_far_error_lt_0_02": all(
                float(per_alpha[f"{alpha:.2f}"]["abs_far_error"]) < 0.02
                for alpha in ALPHAS
            ),
        }

    guarantee_summary_by_alpha = {
        alpha_key: {
            "coverage_pct": 100.0 * float(np.mean(np.asarray(values, dtype=np.float64))),
            "n_combinations": len(values),
        }
        for alpha_key, values in guarantee_checks_by_alpha.items()
    }

    overall_guarantee_pct = 100.0 * float(
        np.mean(np.asarray(guarantee_checks, dtype=np.float64))
    )

    return {
        "config": {
            "alphas": ALPHAS,
            "calib_ratio": CALIB_RATIO,
            "rolling_window": 0,
            "use_aci": False,
            "bonferroni_n_tests": 1,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "sequence_length": SEQUENCE_LENGTH,
            "prediction_delta": PREDICTION_DELTA,
            "smooth_window": SMOOTH_WINDOW,
            "smooth_method": SMOOTH_METHOD,
            "expected_experiments": 12,
        },
        "datasets": dataset_aggregate,
        "expected_outcome_checks": {
            "completed_12_experiments": True,
            "all_datasets_abs_far_error_lt_0_02": all(
                bool(dataset_aggregate[name]["all_alphas_abs_far_error_lt_0_02"])
                for name in ["smd", "psm", "msl", "smap"]
            ),
            "guarantee_coverage_pct_overall": overall_guarantee_pct,
            "guarantee_coverage_pct_by_alpha": guarantee_summary_by_alpha,
            "guarantee_coverage_ge_80_overall": overall_guarantee_pct >= 80.0,
        },
    }


def _write_dataset_alpha_reports(
    smd_result: dict[str, Any],
    dataset_results: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    for alpha in ALPHAS:
        alpha_key = f"{alpha:.2f}"
        smd_rows = []
        for entity_result in smd_result["entities"]:
            row = next(
                row_item
                for row_item in entity_result["rows"]
                if float(row_item["alpha"]) == alpha
            )
            smd_rows.append(row)
        smd_payload = {
            "dataset": "smd",
            "alpha": alpha,
            "renderer_policy": smd_result["renderer_policy"],
            "entities": smd_rows,
            "aggregate": _summarize_smd_alpha(smd_result=smd_result, alpha=alpha),
        }
        _save_json(report_dir / f"smd_alpha_{alpha_key}.json", smd_payload)

        for dataset_result in dataset_results:
            row = next(
                row_item
                for row_item in dataset_result["rows"]
                if float(row_item["alpha"]) == alpha
            )
            payload = {
                "dataset": dataset_result["dataset"],
                "alpha": alpha,
                "renderer": dataset_result["renderer"],
                "result": row,
            }
            _save_json(
                report_dir / f"{dataset_result['dataset']}_alpha_{alpha_key}.json",
                payload,
            )


def main() -> None:
    args = _parse_args()
    report_dir = Path(args.report_dir)
    evidence_path = Path(args.evidence_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Running CalibGuard v3 experiments with leak-free train calibration.")

    smd_result = _run_smd_experiment()
    dataset_results = [
        _run_dataset_experiment(dataset="psm", renderer=_select_dataset_renderer("psm")),
        _run_dataset_experiment(dataset="msl", renderer=_select_dataset_renderer("msl")),
        _run_dataset_experiment(dataset="smap", renderer=_select_dataset_renderer("smap")),
    ]

    _write_dataset_alpha_reports(
        smd_result=smd_result,
        dataset_results=dataset_results,
        report_dir=report_dir,
    )
    summary = _build_summary(smd_result=smd_result, dataset_results=dataset_results)
    _save_json(report_dir / "summary.json", summary)
    _save_json(evidence_path, summary)

    LOGGER.info("Saved CalibGuard v3 reports to %s", report_dir)
    LOGGER.info("Saved QA evidence to %s", evidence_path)
    LOGGER.info(
        "Overall FAR<=alpha coverage: %.1f%%",
        summary["expected_outcome_checks"]["guarantee_coverage_pct_overall"],
    )


if __name__ == "__main__":
    main()
