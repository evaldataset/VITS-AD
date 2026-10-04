#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger(__name__)


def _config_root() -> str:
    return str((Path(__file__).resolve().parents[1] / "configs").resolve())


def _load_cfg(dataset: str, renderer: str, model: str, seed: int) -> DictConfig:
    with initialize_config_dir(config_dir=_config_root(), version_base=None):
        cfg = compose(
            config_name="experiment/patchtraj_improved",
            overrides=[
                f"data={dataset}",
                f"render={renderer}",
                f"model={model}",
                f"training.seed={seed}",
            ],
        )
    return cfg


def _pool_tokens(tokens: np.ndarray) -> np.ndarray:
    token_array = np.asarray(tokens, dtype=np.float64)
    if token_array.ndim != 3:
        raise ValueError(f"tokens must have shape (B, N, D), got {token_array.shape}.")
    return np.mean(token_array, axis=1, dtype=np.float64)


def _safe_metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[dict[str, float | None], str | None]:
    from src.evaluation.metrics import compute_all_metrics

    label_array = np.asarray(labels, dtype=np.int64)
    if np.unique(label_array).size < 2:
        return {
            "auc_roc": None,
            "auc_pr": None,
            "f1_pa": None,
        }, "smoke slice contains only one label class; metrics omitted"

    raw = compute_all_metrics(scores=scores, labels=label_array)
    return {
        "auc_roc": float(raw["auc_roc"]),
        "auc_pr": float(raw["auc_pr"]),
        "f1_pa": float(raw["f1_pa"]),
    }, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pooled frozen-feature baseline using the PatchTraj visual pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["smd", "psm", "msl", "smap"],
        help="Dataset key.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="line_plot",
        choices=["line_plot", "gaf", "recurrence_plot"],
        help="Renderer name reused from PatchTraj.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_base",
        choices=["dinov2_base", "clip_base", "siglip_base"],
        help="Frozen backbone config.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="k for train-window nearest-neighbor scoring.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for consistency with the main pipeline.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store pooled-feature outputs.",
    )
    parser.add_argument(
        "--max_train_windows",
        type=int,
        default=None,
        help="Optional cap for train windows used in smoke checks.",
    )
    parser.add_argument(
        "--max_test_windows",
        type=int,
        default=None,
        help="Optional cap for test windows used in smoke checks.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    from src.data.base import create_sliding_windows, normalize_data, time_based_split
    from src.utils.reproducibility import get_device, seed_everything

    from scripts.train_patchtraj import extract_tokens_from_windows, load_dataset_arrays

    seed_everything(int(args.seed))

    cfg = _load_cfg(
        dataset=str(args.dataset),
        renderer=str(args.renderer),
        model=str(args.model),
        seed=int(args.seed),
    )
    device = get_device(None)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, labels, entity, official_test_start = load_dataset_arrays(cfg)
    train_ratio = (
        float(official_test_start) / float(len(labels))
        if official_test_start is not None
        else float(cfg.data.train_ratio)
    )
    train_data, train_labels, test_data, test_labels = time_based_split(
        data=data,
        labels=labels,
        train_ratio=train_ratio,
    )
    if bool(cfg.data.normalize):
        train_data, test_data = normalize_data(
            train_data=train_data,
            test_data=test_data,
            method=str(cfg.data.norm_method),
        )

    train_windows, _ = create_sliding_windows(
        data=train_data,
        labels=train_labels,
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.stride),
    )
    test_windows, test_window_labels = create_sliding_windows(
        data=test_data,
        labels=test_labels,
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.stride),
    )
    if args.max_train_windows is not None:
        train_windows = train_windows[: int(args.max_train_windows)]
    if args.max_test_windows is not None:
        test_windows = test_windows[: int(args.max_test_windows)]
        test_window_labels = test_window_labels[: int(args.max_test_windows)]

    LOGGER.info(
        "Extracting pooled frozen features for dataset=%s entity=%s renderer=%s model=%s",
        args.dataset,
        entity,
        args.renderer,
        args.model,
    )
    train_tokens, _ = extract_tokens_from_windows(train_windows, cfg, device)
    test_tokens, _ = extract_tokens_from_windows(test_windows, cfg, device)

    train_features = _pool_tokens(train_tokens.numpy())
    test_features = _pool_tokens(test_tokens.numpy())

    knn = NearestNeighbors(n_neighbors=int(args.neighbors), metric="euclidean")
    knn.fit(train_features)
    distances, _ = knn.kneighbors(test_features)
    scores = np.mean(distances, axis=1).astype(np.float64)
    labels_array = np.asarray(test_window_labels, dtype=np.int64)
    metrics, metrics_note = _safe_metrics(scores=scores, labels=labels_array)

    np.save(output_dir / "scores.npy", scores)
    np.save(output_dir / "labels.npy", labels_array)
    np.save(output_dir / "train_features.npy", train_features)
    np.save(output_dir / "test_features.npy", test_features)

    payload: dict[str, Any] = {
        "dataset": str(args.dataset),
        "entity": entity,
        "method": "FrozenFeaturePoolKNN",
        "metrics": {
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1_pa": metrics["f1_pa"],
        },
        "baseline": {
            "renderer": str(args.renderer),
            "model": str(args.model),
            "neighbors": int(args.neighbors),
            "seed": int(args.seed),
            "window_size": int(cfg.data.window_size),
            "stride": int(cfg.data.stride),
            "normalization": str(cfg.data.norm_method),
            "feature_pooling": "mean_over_patches",
            "scoring": "mean_kNN_distance_over_train_windows",
            "notes": "Same visual pipeline as PatchTraj, but no token-trajectory modeling.",
            "config_root": to_absolute_path("../configs"),
            "max_train_windows": None
            if args.max_train_windows is None
            else int(args.max_train_windows),
            "max_test_windows": None
            if args.max_test_windows is None
            else int(args.max_test_windows),
            "metrics_note": metrics_note,
            "label_counts": {
                "normal": int(np.sum(labels_array == 0)),
                "anomaly": int(np.sum(labels_array == 1)),
            },
        },
    }
    result_path = output_dir / "pooled_frozen_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    LOGGER.info("Saved pooled frozen-feature baseline outputs to %s", output_dir)
    LOGGER.info(
        "Metrics: auc_roc=%s auc_pr=%s f1_pa=%s",
        payload["metrics"]["auc_roc"],
        payload["metrics"]["auc_pr"],
        payload["metrics"]["f1_pa"],
    )


if __name__ == "__main__":
    main()
