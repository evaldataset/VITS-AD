#!/usr/bin/env python3

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportExplicitAny=false, reportAny=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatchConfig:
    """Configuration for running CATCH on one dataset."""

    data_name: str
    enc_in: int
    config_path: str
    save_path: str
    model_hyper_params: dict[str, Any]


DATASET_CONFIGS: dict[str, CatchConfig] = {
    "smd": CatchConfig(
        data_name="SMD.csv",
        enc_in=38,
        config_path="unfixed_detect_score_multi_config.json",
        save_path="score/CATCH",
        model_hyper_params={
            "Mlr": 1e-05,
            "batch_size": 64,
            "cf_dim": 64,
            "d_ff": 128,
            "d_model": 128,
            "e_layers": 3,
            "head_dim": 32,
            "lr": 0.0001,
            "n_heads": 16,
            "num_epochs": 5,
            "patch_size": 8,
            "patch_stride": 8,
            "seq_len": 192,
        },
    ),
    "psm": CatchConfig(
        data_name="PSM.csv",
        enc_in=25,
        config_path="unfixed_detect_score_multi_config.json",
        save_path="score/CATCH",
        model_hyper_params={
            "Mlr": 0.001,
            "auxi_lambda": 0.01,
            "batch_size": 128,
            "cf_dim": 16,
            "d_ff": 32,
            "d_model": 16,
            "dc_lambda": 0.05,
            "dropout": 0.3,
            "e_layers": 1,
            "head_dim": 32,
            "inference_patch_size": 96,
            "lr": 0.005,
            "n_heads": 4,
            "num_epochs": 3,
            "patch_size": 16,
            "patch_stride": 8,
            "score_lambda": 0.5,
            "seq_len": 192,
        },
    ),
    "msl": CatchConfig(
        data_name="MSL.csv",
        enc_in=55,
        config_path="unfixed_detect_score_multi_config.json",
        save_path="score/CATCH",
        model_hyper_params={
            "Mlr": 5e-05,
            "batch_size": 128,
            "cf_dim": 64,
            "d_ff": 256,
            "d_model": 128,
            "e_layers": 3,
            "head_dim": 64,
            "lr": 0.0005,
            "n_heads": 2,
            "num_epochs": 5,
            "patch_size": 16,
            "patch_stride": 8,
            "seq_len": 192,
        },
    ),
    "smap": CatchConfig(
        data_name="SMAP.csv",
        enc_in=25,
        config_path="unfixed_detect_score_multi_config.json",
        save_path="score/CATCH",
        model_hyper_params={
            "Mlr": 0.01,
            "auxi_lambda": 1,
            "batch_size": 128,
            "cf_dim": 16,
            "d_ff": 32,
            "d_model": 64,
            "dc_lambda": 1,
            "dropout": 0.4,
            "e_layers": 3,
            "head_dim": 64,
            "inference_patch_size": 4,
            "lr": 0.005,
            "n_heads": 4,
            "num_epochs": 10,
            "patch_size": 16,
            "patch_stride": 8,
            "score_lambda": 1e-06,
            "seq_len": 192,
        },
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _catch_root(project_root: Path) -> Path:
    return project_root / "baselines" / "CATCH"


def _run_catch_training(catch_root: Path, config: CatchConfig, gpu: int) -> None:
    command = [
        sys.executable,
        "-u",
        "scripts/run_benchmark.py",
        "--config-path",
        config.config_path,
        "--data-name-list",
        config.data_name,
        "--model-name",
        "catch.CATCH",
        "--model-hyper-params",
        json.dumps(config.model_hyper_params, sort_keys=True),
        "--gpus",
        "0",
        "--num-workers",
        "1",
        "--timeout",
        "60000",
        "--save-path",
        config.save_path,
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    LOGGER.info("Running CATCH for data=%s on gpu=%d", config.data_name, gpu)
    subprocess.run(command, cwd=catch_root, env=env, check=True)


def _align_scores_and_labels(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    score_array = np.asarray(scores).reshape(-1)
    label_array = np.asarray(labels).reshape(-1)

    if score_array.size < label_array.size:
        padding = label_array.size - score_array.size
        score_array = np.pad(
            score_array, (0, padding), mode="constant", constant_values=0
        )
    elif score_array.size > label_array.size:
        score_array = score_array[: label_array.size]

    return score_array.astype(np.float64), (label_array > 0).astype(np.int64)


def _extract_scores_and_labels(
    catch_root: Path,
    config: CatchConfig,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    sys.path.insert(0, str(catch_root))

    from ts_benchmark.baselines.catch.CATCH import CATCH  # type: ignore[import-not-found]
    from ts_benchmark.data.data_source import (  # type: ignore[import-not-found]
        LocalAnomalyDetectDataSource,
    )
    from ts_benchmark.utils.data_processing import split_before  # type: ignore[import-not-found]

    data_source = LocalAnomalyDetectDataSource()
    data_source.load_series_list([config.data_name])

    series = data_source.dataset.get_series(config.data_name)
    if series is None:
        raise FileNotFoundError(f"CATCH dataset not found: {config.data_name}")

    meta = data_source.dataset.get_series_meta_info(config.data_name)
    if meta is None or "train_lens" not in meta:
        raise KeyError(f"Missing 'train_lens' metadata for dataset: {config.data_name}")

    train_length = int(meta["train_lens"])
    full_series = series.reset_index(drop=True)
    train_frame, test_frame = split_before(full_series, train_length)

    train_data = train_frame.loc[:, train_frame.columns != "label"]
    train_label = train_frame.loc[:, ["label"]]
    test_data = test_frame.loc[:, test_frame.columns != "label"]
    test_label = test_frame.loc[:, ["label"]].to_numpy().reshape(-1)

    model = CATCH(**config.model_hyper_params)
    model.detect_fit(train_data, train_label)
    scores, _ = model.detect_score(test_data)

    return _align_scores_and_labels(scores=np.asarray(scores), labels=test_label)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for CATCH baseline execution."""

    parser = argparse.ArgumentParser(
        description="Run CATCH anomaly baseline and export VITS-compatible metrics."
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
        help="Directory to store CATCH outputs and metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Run CATCH baseline, compute VITS metrics, and save JSON outputs."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    dataset_key = args.dataset.lower()
    config = DATASET_CONFIGS[dataset_key]

    project_root = _project_root()
    catch_root = _catch_root(project_root)
    if not catch_root.exists():
        raise FileNotFoundError(
            f"CATCH baseline repo not found: {catch_root}. "
            "Clone https://github.com/decisionintelligence/CATCH into baselines/CATCH first."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(project_root))
    from src.evaluation.metrics import compute_all_metrics

    _run_catch_training(catch_root=catch_root, config=config, gpu=args.gpu)
    scores, labels = _extract_scores_and_labels(catch_root=catch_root, config=config)

    raw_metrics = compute_all_metrics(scores=scores, labels=labels)
    metrics = {
        "auc_roc": float(raw_metrics["auc_roc"]),
        "auc_pr": float(raw_metrics["auc_pr"]),
        "f1_pa": float(raw_metrics["f1_pa"]),
    }

    np.save(output_dir / "scores.npy", scores)
    np.save(output_dir / "labels.npy", labels)

    result_payload = {
        "dataset": dataset_key,
        "method": "CATCH",
        "metrics": metrics,
        "catch": {
            "data": config.data_name,
            "enc_in": config.enc_in,
            "config_path": config.config_path,
            "save_path": config.save_path,
            "model_hyper_params": config.model_hyper_params,
        },
    }

    result_path = output_dir / "catch_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, sort_keys=True)

    LOGGER.info("Saved CATCH baseline outputs to %s", output_dir)
    LOGGER.info(
        "Metrics: auc_roc=%.6f auc_pr=%.6f f1_pa=%.6f",
        metrics["auc_roc"],
        metrics["auc_pr"],
        metrics["f1_pa"],
    )


if __name__ == "__main__":
    main()
