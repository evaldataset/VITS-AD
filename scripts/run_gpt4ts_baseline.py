#!/usr/bin/env python3

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportExplicitAny=false, reportAny=false, reportUnusedCallResult=false, reportPrivateImportUsage=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.msl import MSLDataset
from src.data.psm import PSMDataset
from src.data.smap import SMAPDataset
from src.data.smd import SMDDataset
from src.evaluation.metrics import compute_all_metrics

LOGGER = logging.getLogger(__name__)

SEED = 42
WINDOW_SIZE = 100
GPT_LAYERS = 6
GPT_D_MODEL = 768
LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 64

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class GPT4TSAD(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = GPT_D_MODEL,
        n_layers: int = GPT_LAYERS,
    ) -> None:
        super().__init__()

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")

        gpt2 = GPT2Model.from_pretrained("gpt2")
        if n_layers > len(gpt2.h):
            raise ValueError(f"n_layers={n_layers} exceeds GPT-2 layers={len(gpt2.h)}")
        if seq_len > gpt2.config.n_positions:
            raise ValueError(
                f"seq_len={seq_len} exceeds GPT-2 max positions={gpt2.config.n_positions}"
            )

        self.seq_len = seq_len
        self.position_embeddings = gpt2.wpe
        self.gpt2_layers = nn.ModuleList(gpt2.h[:n_layers])

        for parameter in self.position_embeddings.parameters():
            parameter.requires_grad = False
        for parameter in self.gpt2_layers.parameters():
            parameter.requires_grad = False

        self.input_proj = nn.Linear(input_dim, d_model)
        self.ln_in = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, input_dim)

        self.position_embeddings.eval()
        self.gpt2_layers.eval()

    def train(self, mode: bool = True) -> "GPT4TSAD":
        super().train(mode)
        self.position_embeddings.eval()
        self.gpt2_layers.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, W, D), got {tuple(x.shape)}.")

        batch_size, seq_len, _ = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}.")

        h = self.input_proj(x)
        h = self.ln_in(h)

        position_ids = (
            torch.arange(seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        h = h + self.position_embeddings(position_ids)

        for layer in self.gpt2_layers:
            h = layer(h)[0]

        h = self.ln_out(h)
        return self.output_proj(h)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sanitize_windows(windows: FloatArray) -> FloatArray:
    return np.nan_to_num(windows.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _save_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _list_smd_entities(raw_root: Path) -> list[str]:
    train_dir = raw_root / "smd" / "train"
    entities = sorted(path.stem for path in train_dir.glob("*.txt"))
    if not entities:
        raise ValueError(f"No SMD entities found in {train_dir}")
    if len(entities) != 28:
        LOGGER.warning("Expected 28 SMD entities, found %d.", len(entities))
    return entities


def _load_single_dataset(
    dataset: str,
    raw_root: Path,
    window_size: int,
) -> tuple[FloatArray, FloatArray, IntArray]:
    if dataset == "psm":
        ds = PSMDataset(
            raw_dir=raw_root / "psm", window_size=window_size, normalize=False
        )
    elif dataset == "msl":
        ds = MSLDataset(
            raw_dir=raw_root / "msl", window_size=window_size, normalize=False
        )
    elif dataset == "smap":
        ds = SMAPDataset(
            raw_dir=raw_root / "smap", window_size=window_size, normalize=False
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return ds.train_windows, ds.test_windows, ds.test_labels


def _train_gpt4ts(
    train_windows: FloatArray,
    device: torch.device,
) -> GPT4TSAD:
    if train_windows.ndim != 3:
        raise ValueError(
            f"Expected train_windows (N, W, D), got {train_windows.shape}."
        )

    num_features = int(train_windows.shape[2])
    model = GPT4TSAD(input_dim=num_features, seq_len=WINDOW_SIZE).to(device)

    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
    )
    criterion = nn.MSELoss(reduction="mean")

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        total_samples = 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            batch_size = int(batch_x.shape[0])
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        avg_loss = running_loss / max(total_samples, 1)
        LOGGER.info(
            "Epoch %d/%d - reconstruction_loss=%.6f",
            epoch + 1,
            EPOCHS,
            avg_loss,
        )

    return model


def _compute_scores(
    model: GPT4TSAD,
    test_windows: FloatArray,
    device: torch.device,
) -> FloatArray:
    test_tensor = torch.tensor(test_windows, dtype=torch.float32)
    test_loader = DataLoader(
        TensorDataset(test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    score_chunks: list[np.ndarray[Any, Any]] = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            reconstructed = model(batch_x)
            squared_error = (batch_x - reconstructed).pow(2)
            batch_scores = squared_error.mean(dim=(1, 2))
            score_chunks.append(batch_scores.detach().cpu().numpy())

    scores = np.concatenate(score_chunks, axis=0).astype(np.float64)
    return scores


def _evaluate_dataset(
    train_windows: FloatArray,
    test_windows: FloatArray,
    test_labels: IntArray,
    device: torch.device,
) -> dict[str, float]:
    train_clean = _sanitize_windows(train_windows)
    test_clean = _sanitize_windows(test_windows)
    labels = np.asarray(test_labels, dtype=np.int64)

    model = _train_gpt4ts(train_windows=train_clean, device=device)
    scores = _compute_scores(model=model, test_windows=test_clean, device=device)
    metrics = compute_all_metrics(scores=scores, labels=labels)
    return {
        "auc_roc": float(metrics["auc_roc"]),
        "auc_pr": float(metrics["auc_pr"]),
        "f1_pa": float(metrics["f1_pa"]),
    }


def _run_single_entity_dataset(
    dataset: str,
    raw_root: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    train_windows, test_windows, test_labels = _load_single_dataset(
        dataset=dataset,
        raw_root=raw_root,
        window_size=WINDOW_SIZE,
    )
    metrics = _evaluate_dataset(
        train_windows=train_windows,
        test_windows=test_windows,
        test_labels=test_labels,
        device=device,
    )

    payload: dict[str, Any] = {
        "method": "GPT4TS",
        "dataset": dataset,
        "metrics": metrics,
        "gpt4ts": {
            "window_size": WINDOW_SIZE,
            "gpt_layers": GPT_LAYERS,
            "d_model": GPT_D_MODEL,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "frozen_gpt2": True,
            "model_name": "gpt2",
        },
    }
    _save_json(output_dir / f"gpt4ts_{dataset}.json", payload)
    return metrics


def _run_smd(
    raw_root: Path, output_dir: Path, device: torch.device
) -> dict[str, float]:
    entities = _list_smd_entities(raw_root)
    per_entity_metrics: dict[str, dict[str, float]] = {}

    for index, entity in enumerate(entities, start=1):
        LOGGER.info(
            "Running GPT4TS on SMD entity %d/%d: %s", index, len(entities), entity
        )
        ds = SMDDataset(
            raw_dir=raw_root / "smd",
            entity=entity,
            window_size=WINDOW_SIZE,
            normalize=False,
        )
        metrics = _evaluate_dataset(
            train_windows=ds.train_windows,
            test_windows=ds.test_windows,
            test_labels=ds.test_labels,
            device=device,
        )
        per_entity_metrics[entity] = metrics

    auc_rocs = [item["auc_roc"] for item in per_entity_metrics.values()]
    auc_prs = [item["auc_pr"] for item in per_entity_metrics.values()]
    f1_pas = [item["f1_pa"] for item in per_entity_metrics.values()]
    avg_metrics = {
        "auc_roc": float(np.mean(auc_rocs)),
        "auc_pr": float(np.mean(auc_prs)),
        "f1_pa": float(np.mean(f1_pas)),
    }

    per_entity_auc = {
        entity: {"auc_roc": float(values["auc_roc"])}
        for entity, values in sorted(per_entity_metrics.items())
    }
    payload: dict[str, Any] = {
        "method": "GPT4TS",
        "dataset": "smd",
        "metrics": avg_metrics,
        "per_entity": per_entity_auc,
        "gpt4ts": {
            "window_size": WINDOW_SIZE,
            "gpt_layers": GPT_LAYERS,
            "d_model": GPT_D_MODEL,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "frozen_gpt2": True,
            "model_name": "gpt2",
            "num_entities": len(entities),
        },
    }
    _save_json(output_dir / "gpt4ts_smd.json", payload)
    return avg_metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GPT4TS TSAD baseline.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["psm", "msl", "smap", "smd", "all"],
        help="Dataset to run.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU id for CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="data/raw",
        help="Root directory containing raw datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/reports/gpt4ts_baselines",
        help="Directory to save JSON reports.",
    )
    return parser


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    LOGGER.info("Summary (AUC-ROC / AUC-PR / F1-PA)")
    LOGGER.info("%-6s | %-8s | %-8s | %-8s", "Dataset", "AUCROC", "AUCPR", "F1PA")
    for dataset in sorted(summary.keys()):
        metrics = summary[dataset]
        LOGGER.info(
            "%-6s | %.4f   | %.4f   | %.4f",
            dataset,
            metrics["auc_roc"],
            metrics["auc_pr"],
            metrics["f1_pa"],
        )


def main() -> None:
    _setup_logging()

    args = _build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _set_seeds(SEED)

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device=%s with CUDA_VISIBLE_DEVICES=%s", device, args.gpu)

    datasets = (
        ["psm", "msl", "smap", "smd"] if args.dataset == "all" else [args.dataset]
    )
    summary: dict[str, dict[str, float]] = {}

    for dataset in datasets:
        LOGGER.info("Running GPT4TS baseline on %s", dataset)
        if dataset == "smd":
            summary[dataset] = _run_smd(
                raw_root=raw_root, output_dir=output_dir, device=device
            )
        else:
            summary[dataset] = _run_single_entity_dataset(
                dataset=dataset,
                raw_root=raw_root,
                output_dir=output_dir,
                device=device,
            )

    _print_summary(summary)
    LOGGER.info("Saved GPT4TS baseline reports to %s", output_dir)


if __name__ == "__main__":
    main()
