#!/usr/bin/env python3

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportExplicitAny=false, reportAny=false, reportUnusedCallResult=false

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors


LOGGER = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class TS2VecConfig:
    data_name: str
    input_dims: int
    window_size: int
    repr_dims: int
    epochs: int
    batch_size: int
    learning_rate: float
    k_neighbors: int


DATASET_CONFIGS: dict[str, TS2VecConfig] = {
    "smd": TS2VecConfig(
        data_name="SMD.csv",
        input_dims=38,
        window_size=100,
        repr_dims=320,
        epochs=30,
        batch_size=64,
        learning_rate=1e-3,
        k_neighbors=5,
    ),
    "psm": TS2VecConfig(
        data_name="PSM.csv",
        input_dims=25,
        window_size=100,
        repr_dims=320,
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        k_neighbors=5,
    ),
    "msl": TS2VecConfig(
        data_name="MSL.csv",
        input_dims=55,
        window_size=100,
        repr_dims=320,
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        k_neighbors=5,
    ),
    "smap": TS2VecConfig(
        data_name="SMAP.csv",
        input_dims=25,
        window_size=100,
        repr_dims=320,
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        k_neighbors=5,
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _catch_root(project_root: Path) -> Path:
    return project_root / "baselines" / "CATCH"


def _load_ts2vec_class(project_root: Path) -> type[Any]:
    local_repo = project_root / "baselines" / "TS2Vec"
    if local_repo.exists():
        sys.path.insert(0, str(local_repo))
    try:
        from ts2vec import TS2Vec  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "TS2Vec dependency is missing. Install the official package or clone the repository into baselines/TS2Vec so scripts/run_ts2vec_baseline.py can import it."
        ) from exc
    return TS2Vec


def _load_dataset(
    catch_root: Path,
    data_name: str,
) -> tuple[FloatArray, FloatArray, IntArray]:
    sys.path.insert(0, str(catch_root))

    from ts_benchmark.data.data_source import (  # type: ignore[import-not-found]
        LocalAnomalyDetectDataSource,
    )
    from ts_benchmark.utils.data_processing import split_before  # type: ignore[import-not-found]

    data_source = LocalAnomalyDetectDataSource()
    data_source.load_series_list([data_name])
    series = data_source.dataset.get_series(data_name)
    if series is None:
        raise FileNotFoundError(f"Dataset not found in CATCH source: {data_name}")

    meta = data_source.dataset.get_series_meta_info(data_name)
    if meta is None or "train_lens" not in meta:
        raise KeyError(f"Missing train_lens metadata for dataset: {data_name}")

    train_len = int(meta["train_lens"])
    full_series = series.reset_index(drop=True)
    train_frame, test_frame = split_before(full_series, train_len)

    train_values = train_frame.loc[:, train_frame.columns != "label"].to_numpy(dtype=np.float64)
    test_values = test_frame.loc[:, test_frame.columns != "label"].to_numpy(dtype=np.float64)
    test_labels = (test_frame.loc[:, ["label"]].to_numpy().reshape(-1) > 0).astype(np.int64)
    return train_values, test_values, test_labels


def _standardize(
    train_values: FloatArray,
    test_values: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    mean = np.mean(train_values, axis=0, keepdims=True)
    std = np.std(train_values, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_values - mean) / std, (test_values - mean) / std


def _sliding_windows(values: FloatArray, window_size: int) -> FloatArray:
    if values.ndim != 2:
        raise ValueError(f"values must have shape (T, D), got {values.shape}.")
    if values.shape[0] < window_size:
        raise ValueError(
            f"window_size={window_size} exceeds sequence length {values.shape[0]}."
        )
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=(window_size, values.shape[1])
    )
    return np.asarray(windows[:, 0], dtype=np.float64)


def _window_scores_to_timestep_scores(
    window_scores: FloatArray,
    sequence_length: int,
    window_size: int,
) -> FloatArray:
    sums = np.zeros(sequence_length, dtype=np.float64)
    counts = np.zeros(sequence_length, dtype=np.float64)
    for start, score in enumerate(window_scores):
        end = min(start + window_size, sequence_length)
        sums[start:end] += float(score)
        counts[start:end] += 1.0
    counts = np.where(counts == 0.0, 1.0, counts)
    return sums / counts


def _fit_and_score(
    project_root: Path,
    catch_root: Path,
    config: TS2VecConfig,
    gpu: int,
) -> tuple[FloatArray, IntArray, int]:
    train_values, test_values, test_labels = _load_dataset(
        catch_root=catch_root,
        data_name=config.data_name,
    )
    train_scaled, test_scaled = _standardize(train_values=train_values, test_values=test_values)
    train_windows = _sliding_windows(train_scaled, window_size=config.window_size)
    test_windows = _sliding_windows(test_scaled, window_size=config.window_size)

    TS2Vec = _load_ts2vec_class(project_root=project_root)
    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    actual_input_dims = int(train_scaled.shape[1])
    model = TS2Vec(
        input_dims=actual_input_dims,
        output_dims=config.repr_dims,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        device=device,
    )
    model.fit(train_windows, n_epochs=config.epochs)

    train_repr = np.asarray(
        model.encode(train_windows, encoding_window="full_series"),
        dtype=np.float64,
    )
    test_repr = np.asarray(
        model.encode(test_windows, encoding_window="full_series"),
        dtype=np.float64,
    )

    knn = NearestNeighbors(n_neighbors=config.k_neighbors, metric="euclidean")
    knn.fit(train_repr)
    distances, _ = knn.kneighbors(test_repr)
    window_scores = np.mean(distances, axis=1).astype(np.float64)
    timestep_scores = _window_scores_to_timestep_scores(
        window_scores=window_scores,
        sequence_length=test_scaled.shape[0],
        window_size=config.window_size,
    )
    return timestep_scores, test_labels, actual_input_dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TS2Vec+kNN baseline and export VITS-compatible metrics."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset key: smd/psm/msl/smap.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index passed to CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store TS2Vec outputs and metrics.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the number of TS2Vec training epochs.",
    )
    parser.add_argument(
        "--repr_dims",
        type=int,
        default=None,
        help="Optional override for TS2Vec output dimension.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override for TS2Vec batch size.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=None,
        help="Optional override for kNN scoring neighbors.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args()
    dataset_key = args.dataset.lower()
    base_config = DATASET_CONFIGS[dataset_key]
    config = TS2VecConfig(
        data_name=base_config.data_name,
        input_dims=base_config.input_dims,
        window_size=base_config.window_size,
        repr_dims=int(args.repr_dims) if args.repr_dims is not None else base_config.repr_dims,
        epochs=int(args.epochs) if args.epochs is not None else base_config.epochs,
        batch_size=int(args.batch_size) if args.batch_size is not None else base_config.batch_size,
        learning_rate=base_config.learning_rate,
        k_neighbors=int(args.k_neighbors) if args.k_neighbors is not None else base_config.k_neighbors,
    )
    project_root = _project_root()
    catch_root = _catch_root(project_root)
    if not catch_root.exists():
        raise FileNotFoundError(f"CATCH baseline repo not found: {catch_root}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sys.path.insert(0, str(project_root))
    from src.evaluation.metrics import compute_all_metrics

    LOGGER.info("Running TS2Vec baseline for data=%s on gpu=%d", config.data_name, args.gpu)
    scores, labels, actual_input_dims = _fit_and_score(
        project_root=project_root,
        catch_root=catch_root,
        config=config,
        gpu=args.gpu,
    )
    raw_metrics = compute_all_metrics(scores=scores, labels=labels)
    metrics = {
        "auc_roc": float(raw_metrics["auc_roc"]),
        "auc_pr": float(raw_metrics["auc_pr"]),
        "f1_pa": float(raw_metrics["f1_pa"]),
    }

    np.save(output_dir / "scores.npy", scores)
    np.save(output_dir / "labels.npy", labels)

    payload = {
        "dataset": dataset_key,
        "method": "TS2Vec",
        "metrics": metrics,
        "ts2vec": {
            "data": config.data_name,
            "input_dims": actual_input_dims,
            "window_size": config.window_size,
            "repr_dims": config.repr_dims,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "k_neighbors": config.k_neighbors,
            "scoring": "train-window representation kNN distance",
        },
    }
    result_path = output_dir / "ts2vec_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    LOGGER.info("Saved TS2Vec baseline outputs to %s", output_dir)
    LOGGER.info(
        "Metrics: auc_roc=%.6f auc_pr=%.6f f1_pa=%.6f",
        metrics["auc_roc"],
        metrics["auc_pr"],
        metrics["f1_pa"],
    )


if __name__ == "__main__":
    main()
