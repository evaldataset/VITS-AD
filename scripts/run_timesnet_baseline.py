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
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimesNetConfig:
    data_name: str
    root_path: str
    seq_len: int
    d_model: int
    d_ff: int
    e_layers: int
    enc_in: int
    c_out: int
    top_k: int
    anomaly_ratio: float
    train_epochs: int


DATASET_CONFIGS: dict[str, TimesNetConfig] = {
    "smd": TimesNetConfig(
        data_name="SMD",
        root_path="./dataset/SMD",
        seq_len=100,
        d_model=64,
        d_ff=64,
        e_layers=2,
        enc_in=38,
        c_out=38,
        top_k=5,
        anomaly_ratio=0.5,
        train_epochs=10,
    ),
    "psm": TimesNetConfig(
        data_name="PSM",
        root_path="./dataset/PSM",
        seq_len=100,
        d_model=64,
        d_ff=64,
        e_layers=2,
        enc_in=25,
        c_out=25,
        top_k=3,
        anomaly_ratio=1.0,
        train_epochs=3,
    ),
    "msl": TimesNetConfig(
        data_name="MSL",
        root_path="./dataset/MSL",
        seq_len=100,
        d_model=64,
        d_ff=64,
        e_layers=2,
        enc_in=55,
        c_out=55,
        top_k=3,
        anomaly_ratio=1.0,
        train_epochs=3,
    ),
    "smap": TimesNetConfig(
        data_name="SMAP",
        root_path="./dataset/SMAP",
        seq_len=100,
        d_model=64,
        d_ff=64,
        e_layers=2,
        enc_in=25,
        c_out=25,
        top_k=3,
        anomaly_ratio=1.0,
        train_epochs=3,
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timesnet_root(project_root: Path) -> Path:
    return project_root / "baselines" / "TimesNet"


def _build_setting_name(args: SimpleNamespace) -> str:
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_"
        f"df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_fc{args.factor}_"
        f"eb{args.embed}_dt{args.distil}_{args.des}_0"
    )


def _ts_args(config: TimesNetConfig, dataset_key: str, gpu: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="anomaly_detection",
        is_training=1,
        model_id=config.data_name,
        model="TimesNet",
        data=config.data_name,
        root_path=config.root_path,
        features="M",
        seq_len=config.seq_len,
        label_len=48,
        pred_len=0,
        d_model=config.d_model,
        n_heads=8,
        e_layers=config.e_layers,
        d_layers=1,
        d_ff=config.d_ff,
        factor=1,
        top_k=config.top_k,
        num_kernels=6,
        enc_in=config.enc_in,
        dec_in=config.enc_in,
        c_out=config.c_out,
        anomaly_ratio=config.anomaly_ratio,
        train_epochs=config.train_epochs,
        batch_size=128,
        learning_rate=1e-4,
        patience=3,
        dropout=0.1,
        embed="timeF",
        freq="h",
        des=f"vits_{dataset_key}",
        itr=1,
        augmentation_ratio=0,
        expand=2,
        d_conv=4,
        distil=True,
        gpu=gpu,
        gpu_type="cuda",
        num_workers=0,
        seasonal_patterns="Monthly",
    )


def _run_timesnet_training(
    timesnet_root: Path,
    config: TimesNetConfig,
    dataset_key: str,
    gpu: int,
) -> None:
    command = [
        sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "anomaly_detection",
        "--is_training",
        "1",
        "--root_path",
        config.root_path,
        "--model_id",
        config.data_name,
        "--model",
        "TimesNet",
        "--data",
        config.data_name,
        "--features",
        "M",
        "--seq_len",
        str(config.seq_len),
        "--pred_len",
        "0",
        "--d_model",
        str(config.d_model),
        "--d_ff",
        str(config.d_ff),
        "--e_layers",
        str(config.e_layers),
        "--enc_in",
        str(config.enc_in),
        "--c_out",
        str(config.c_out),
        "--top_k",
        str(config.top_k),
        "--anomaly_ratio",
        str(config.anomaly_ratio),
        "--train_epochs",
        str(config.train_epochs),
        "--batch_size",
        "128",
        "--des",
        f"vits_{dataset_key}",
        "--itr",
        "1",
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    LOGGER.info("Running TimesNet for dataset=%s on gpu=%d", dataset_key, gpu)
    subprocess.run(command, cwd=timesnet_root, env=env, check=True)


def _extract_scores_and_labels(
    timesnet_root: Path,
    ts_args: SimpleNamespace,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    sys.path.insert(0, str(timesnet_root))

    from data_provider.data_factory import data_provider  # type: ignore[import-not-found]
    from models import TimesNet as timesnet_model  # type: ignore[import-not-found]

    setting = _build_setting_name(ts_args)
    checkpoint_path = timesnet_root / "checkpoints" / setting / "checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TimesNet checkpoint not found: {checkpoint_path}")

    _, test_loader = data_provider(ts_args, flag="test")
    model = timesnet_model.Model(ts_args).float()

    device = (
        torch.device(f"cuda:{ts_args.gpu}")
        if torch.cuda.is_available() and ts_args.gpu >= 0
        else torch.device("cpu")
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)  # nosec: trusted local checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    criterion = torch.nn.MSELoss(reduction="none")
    score_chunks: list[np.ndarray[Any, Any]] = []
    label_chunks: list[np.ndarray[Any, Any]] = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            inputs = batch_x.float().to(device)
            outputs = model(inputs, None, None, None)
            score = torch.mean(criterion(inputs, outputs), dim=-1)
            score_chunks.append(score.detach().cpu().numpy())
            label_chunks.append(np.asarray(batch_y))

    scores = np.concatenate(score_chunks, axis=0).reshape(-1)
    labels = np.concatenate(label_chunks, axis=0).reshape(-1)
    labels = (labels > 0).astype(np.int64)
    return scores.astype(np.float64), labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TimesNet anomaly baseline and export VITS-compatible metrics."
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
        help="Directory to store TimesNet outputs and metrics.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    dataset_key = args.dataset.lower()
    config = DATASET_CONFIGS[dataset_key]

    project_root = _project_root()
    timesnet_root = _timesnet_root(project_root)
    if not timesnet_root.exists():
        raise FileNotFoundError(
            f"TimesNet baseline repo not found: {timesnet_root}. "
            "Run scripts/setup_baselines.sh first."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(project_root))
    from src.evaluation.metrics import compute_all_metrics

    _run_timesnet_training(
        timesnet_root=timesnet_root,
        config=config,
        dataset_key=dataset_key,
        gpu=args.gpu,
    )

    score_args = _ts_args(config=config, dataset_key=dataset_key, gpu=args.gpu)
    scores, labels = _extract_scores_and_labels(
        timesnet_root=timesnet_root, ts_args=score_args
    )

    metrics = compute_all_metrics(scores=scores, labels=labels)

    np.save(output_dir / "scores.npy", scores)
    np.save(output_dir / "labels.npy", labels)

    result_payload = {
        "dataset": dataset_key,
        "method": "TimesNet",
        "metrics": {k: float(v) for k, v in metrics.items()},
        "timesnet": {
            "data": config.data_name,
            "root_path": config.root_path,
            "seq_len": config.seq_len,
            "d_model": config.d_model,
            "d_ff": config.d_ff,
            "e_layers": config.e_layers,
            "enc_in": config.enc_in,
            "c_out": config.c_out,
            "top_k": config.top_k,
            "anomaly_ratio": config.anomaly_ratio,
            "train_epochs": config.train_epochs,
        },
    }

    result_path = output_dir / "timesnet_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, sort_keys=True)

    LOGGER.info("Saved TimesNet baseline outputs to %s", output_dir)
    LOGGER.info(
        "Metrics: auc_roc=%.6f auc_pr=%.6f f1_pa=%.6f",
        float(metrics["auc_roc"]),
        float(metrics["auc_pr"]),
        float(metrics["f1_pa"]),
    )


if __name__ == "__main__":
    main()
