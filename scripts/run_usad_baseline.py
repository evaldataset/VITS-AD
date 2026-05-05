#!/usr/bin/env python3

from __future__ import annotations

# pyright: basic, reportMissingImports=false, reportMissingTypeStubs=false

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.optim.adam import Adam
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
LATENT_DIM = 128
INTERMEDIATE_DIM = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 256
ALPHA = 0.5
BETA = 0.5

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class USADModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, INTERMEDIATE_DIM),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_DIM, LATENT_DIM),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(LATENT_DIM, INTERMEDIATE_DIM),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_DIM, input_dim),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(LATENT_DIM, INTERMEDIATE_DIM),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_DIM, input_dim),
        )

    def forward(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(batch)
        recon_1 = self.decoder1(latent)
        recon_2 = self.decoder2(latent)
        recon_2_of_1 = self.decoder2(self.encoder(recon_1))
        return recon_1, recon_2, recon_2_of_1


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _set_seeds() -> None:
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _sanitize_and_flatten(windows: FloatArray) -> FloatArray:
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (N, W, D), got {windows.shape}.")
    clean = np.nan_to_num(windows.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return clean.reshape(clean.shape[0], -1)


def _create_loader(data: FloatArray, shuffle: bool) -> DataLoader[Any]:
    tensor = torch.from_numpy(data.astype(np.float32, copy=False))
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


def _train_usad(train_data: FloatArray, device: torch.device) -> USADModel:
    input_dim = int(train_data.shape[1])
    model = USADModel(input_dim=input_dim).to(device)
    optimizer_1 = Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        lr=LEARNING_RATE,
    )
    optimizer_2 = Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        lr=LEARNING_RATE,
    )

    train_loader = _create_loader(train_data, shuffle=True)
    mse = nn.MSELoss()

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss_1 = 0.0
        epoch_loss_2 = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            factor_a = 1.0 / float(epoch + 1)
            factor_b = float(epoch) / float(epoch + 1)

            recon_1, recon_2, recon_2_of_1 = model(batch)
            loss_1 = factor_a * mse(recon_1, batch) + factor_b * mse(
                recon_2_of_1, batch
            )
            optimizer_1.zero_grad(set_to_none=True)
            loss_1.backward()
            optimizer_1.step()

            recon_1, recon_2, recon_2_of_1 = model(batch)
            loss_2 = factor_a * mse(recon_2, batch) - factor_b * mse(
                recon_2_of_1, batch
            )
            optimizer_2.zero_grad(set_to_none=True)
            loss_2.backward()
            optimizer_2.step()

            epoch_loss_1 += float(loss_1.item())
            epoch_loss_2 += float(loss_2.item())

        avg_loss_1 = epoch_loss_1 / float(max(1, len(train_loader)))
        avg_loss_2 = epoch_loss_2 / float(max(1, len(train_loader)))
        monitor_loss = avg_loss_1 + abs(avg_loss_2)

        LOGGER.info(
            "USAD epoch %d/%d - loss1=%.6f loss2=%.6f",
            epoch,
            EPOCHS,
            avg_loss_1,
            avg_loss_2,
        )

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            LOGGER.info("Early stopping at epoch %d (patience=%d)", epoch, PATIENCE)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def _score_usad(
    model: USADModel, test_data: FloatArray, device: torch.device
) -> FloatArray:
    test_loader = _create_loader(test_data, shuffle=False)
    chunks: list[np.ndarray[Any, Any]] = []

    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(device)
            recon_1, _, recon_2_of_1 = model(batch)
            error_1 = torch.mean((recon_1 - batch) ** 2, dim=1)
            error_2 = torch.mean((recon_2_of_1 - batch) ** 2, dim=1)
            scores = ALPHA * error_1 + BETA * error_2
            chunks.append(scores.detach().cpu().numpy())

    return np.concatenate(chunks, axis=0).astype(np.float64)


def _run_usad(
    train_windows: FloatArray,
    test_windows: FloatArray,
    test_labels: IntArray,
    device: torch.device,
) -> dict[str, float]:
    train_data = _sanitize_and_flatten(train_windows)
    test_data = _sanitize_and_flatten(test_windows)
    labels = np.asarray(test_labels, dtype=np.int64)

    model = _train_usad(train_data=train_data, device=device)
    scores = _score_usad(model=model, test_data=test_data, device=device)
    metrics = compute_all_metrics(scores=scores, labels=labels)
    return {
        "auc_roc": float(metrics["auc_roc"]),
        "auc_pr": float(metrics["auc_pr"]),
        "f1_pa": float(metrics["f1_pa"]),
    }


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
        raise ValueError(f"Unsupported single-entity dataset: {dataset}")

    return ds.train_windows, ds.test_windows, ds.test_labels


def _list_smd_entities(raw_root: Path) -> list[str]:
    train_dir = raw_root / "smd" / "train"
    entities = sorted(path.stem for path in train_dir.glob("*.txt"))
    if not entities:
        raise ValueError(f"No SMD entities found in {train_dir}")
    if len(entities) != 28:
        LOGGER.warning("Expected 28 SMD entities, found %d.", len(entities))
    return entities


def _save_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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
    metrics = _run_usad(
        train_windows=train_windows,
        test_windows=test_windows,
        test_labels=test_labels,
        device=device,
    )
    payload: dict[str, Any] = {
        "method": "USAD",
        "dataset": dataset,
        "metrics": metrics,
    }
    _save_json(output_dir / f"usad_{dataset}.json", payload)
    return metrics


def _run_smd(
    raw_root: Path, output_dir: Path, device: torch.device
) -> dict[str, float]:
    entities = _list_smd_entities(raw_root)
    per_entity: dict[str, dict[str, float]] = {}

    for entity in entities:
        LOGGER.info("Running USAD on SMD entity %s", entity)
        ds = SMDDataset(
            raw_dir=raw_root / "smd",
            entity=entity,
            window_size=WINDOW_SIZE,
            normalize=False,
        )
        per_entity[entity] = _run_usad(
            train_windows=ds.train_windows,
            test_windows=ds.test_windows,
            test_labels=ds.test_labels,
            device=device,
        )

    avg_metrics = {
        "auc_roc": float(np.mean([item["auc_roc"] for item in per_entity.values()])),
        "auc_pr": float(np.mean([item["auc_pr"] for item in per_entity.values()])),
        "f1_pa": float(np.mean([item["f1_pa"] for item in per_entity.values()])),
    }

    per_entity_auc = {
        entity: {"auc_roc": float(values["auc_roc"])}
        for entity, values in sorted(per_entity.items())
    }
    payload: dict[str, Any] = {
        "method": "USAD",
        "dataset": "smd",
        "metrics": avg_metrics,
        "per_entity": per_entity_auc,
    }
    _save_json(output_dir / "usad_smd.json", payload)
    return avg_metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run USAD TSAD baseline.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["psm", "msl", "smap", "smd", "all"],
        help="Dataset to run.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index after CUDA_VISIBLE_DEVICES remapping.",
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
        default="results/reports/usad_baselines",
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
    _set_seeds()

    args = _build_parser().parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    datasets = (
        ["psm", "msl", "smap", "smd"] if args.dataset == "all" else [args.dataset]
    )
    summary: dict[str, dict[str, float]] = {}

    for dataset in datasets:
        LOGGER.info("Running USAD baseline on %s", dataset)
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
    LOGGER.info("Saved USAD baseline reports to %s", output_dir)


if __name__ == "__main__":
    main()
