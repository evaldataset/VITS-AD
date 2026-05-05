from __future__ import annotations

# pyright: basic, reportMissingImports=false, reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportIndexIssue=false, reportImplicitOverride=false, reportUntypedFunctionDecorator=false, reportArgumentType=false

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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.base import create_sliding_windows, normalize_data
from src.data.ucr import list_ucr_files, load_ucr_series
from src.evaluation.metrics import compute_all_metrics
from src.models.backbone import VisionBackbone
from src.models.patchtraj import PatchTrajPredictor
from src.rendering.line_plot import render_line_plot_batch
from src.rendering.token_correspondence import compute_correspondence_map
from src.scoring.patchtraj_scorer import compute_patchtraj_score, normalize_scores


LOGGER = logging.getLogger(__name__)
SEED = 42
WINDOW_SIZE = 100
STRIDE = 1
K_STEPS = 8
DELTA = 1
PATCHTRAJ_D_MODEL = 256
PATCHTRAJ_N_HEADS = 4
PATCHTRAJ_N_LAYERS = 2
PATCHTRAJ_DIM_FF = 1024
PATCHTRAJ_DROPOUT = 0.1
PATCHTRAJ_ACTIVATION = "gelu"
PATCHTRAJ_EPOCHS = 8
PATCHTRAJ_BATCH_SIZE = 32
PATCHTRAJ_LR = 1e-3
PATCHTRAJ_WEIGHT_DECAY = 1e-5
MAX_PATCHTRAJ_TRAIN_WINDOWS = 600
MAX_PATCHTRAJ_TEST_WINDOWS = 800
OCSVM_MAX_TRAIN = 50000

TARGET_SERIES_COUNT = 10
PREFERRED_KEYWORDS = [
    "ECG",
    "PowerDemand",
    "AirTemperature",
    "InternalBleeding",
    "apneaecg",
    "insectEPG",
    "Lab2Cmac",
    "WalkingAceleration",
    "CHARIS",
    "weallwalk",
    "gait",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class TokenSequenceDataset(Dataset):
    def __init__(
        self, tokens: torch.Tensor, anchors: list[int], k: int, delta: int
    ) -> None:
        self.tokens = tokens
        self.anchors = anchors
        self.k = k
        self.delta = delta

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = self.anchors[idx]
        return self.tokens[anchor - self.k + 1 : anchor + 1], self.tokens[
            anchor + self.delta
        ]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _windows_to_features(windows: FloatArray) -> FloatArray:
    return windows[:, -1, :]


def _sanitize_features(
    train_features: FloatArray, test_features: FloatArray
) -> tuple[FloatArray, FloatArray]:
    train_clean = np.nan_to_num(
        train_features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    test_clean = np.nan_to_num(
        test_features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    return train_clean, test_clean


def _score_lof(train_features: FloatArray, test_features: FloatArray) -> FloatArray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    n_neighbors = max(2, min(20, train_scaled.shape[0] - 1))
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        contamination="auto",
    )
    model.fit(train_scaled)
    return -model.decision_function(test_scaled)


def _score_isolation_forest(
    train_features: FloatArray, test_features: FloatArray
) -> FloatArray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=SEED,
    )
    model.fit(train_scaled)
    return -model.decision_function(test_scaled)


def _score_ocsvm(train_features: FloatArray, test_features: FloatArray) -> FloatArray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    fit_data = train_scaled
    if train_scaled.shape[0] > OCSVM_MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(
            train_scaled.shape[0], size=OCSVM_MAX_TRAIN, replace=False
        )
        fit_data = train_scaled[sample_idx]
    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.01)
    model.fit(fit_data)
    return -model.decision_function(test_scaled)


def _run_classical_baselines(
    train_windows: FloatArray, test_windows: FloatArray, test_labels: IntArray
) -> dict[str, float]:
    train_features = _windows_to_features(train_windows)
    test_features = _windows_to_features(test_windows)
    train_clean, test_clean = _sanitize_features(train_features, test_features)

    scores = {
        "LOF": _score_lof(train_clean, test_clean),
        "IsolationForest": _score_isolation_forest(train_clean, test_clean),
        "OneClassSVM": _score_ocsvm(train_clean, test_clean),
    }
    return {
        method: float(compute_all_metrics(method_scores, test_labels)["auc_roc"])
        for method, method_scores in scores.items()
    }


def _subsample_windows(
    windows: FloatArray, labels: IntArray, max_windows: int
) -> tuple[FloatArray, IntArray]:
    if windows.shape[0] <= max_windows:
        return windows, labels
    idx = np.linspace(0, windows.shape[0] - 1, max_windows, dtype=np.int64)
    return windows[idx], labels[idx]


@torch.no_grad()
def _extract_tokens(
    windows: FloatArray,
    backbone: VisionBackbone,
    render_batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, windows.shape[0], render_batch_size):
        end = min(start + render_batch_size, windows.shape[0])
        images = render_line_plot_batch(
            windows[start:end],
            image_size=224,
            dpi=100,
            colormap="tab10",
            line_width=1.0,
            background_color="white",
            show_axes=False,
            show_grid=False,
        )
        batch_tokens = backbone.extract_patch_tokens_from_numpy(
            images.astype(np.float32, copy=False)
        )
        chunks.append(torch.from_numpy(batch_tokens.astype(np.float32, copy=False)))
    return torch.cat(chunks, dim=0)


def _train_patchtraj(
    train_tokens: torch.Tensor,
    device: torch.device,
) -> PatchTrajPredictor:
    model = PatchTrajPredictor(
        hidden_dim=int(train_tokens.shape[-1]),
        d_model=PATCHTRAJ_D_MODEL,
        n_heads=PATCHTRAJ_N_HEADS,
        n_layers=PATCHTRAJ_N_LAYERS,
        dim_feedforward=PATCHTRAJ_DIM_FF,
        dropout=PATCHTRAJ_DROPOUT,
        activation=PATCHTRAJ_ACTIVATION,
    ).to(device)

    start_anchor = K_STEPS - 1
    end_anchor = int(train_tokens.shape[0]) - DELTA - 1
    anchors = list(range(start_anchor, end_anchor + 1))
    split = max(1, int(0.8 * len(anchors)))
    train_anchors = anchors[:split]

    dataset = TokenSequenceDataset(train_tokens, train_anchors, K_STEPS, DELTA)
    loader = DataLoader(
        dataset,
        batch_size=PATCHTRAJ_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = Adam(
        model.parameters(), lr=PATCHTRAJ_LR, weight_decay=PATCHTRAJ_WEIGHT_DECAY
    )
    pi, valid_mask = compute_correspondence_map(
        renderer_type="line_plot",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        patch_grid=(16, 16),
    )

    for _ in range(PATCHTRAJ_EPOCHS):
        model.train()
        for token_seq, target in loader:
            token_seq = token_seq.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(token_seq)
            loss = compute_patchtraj_score(predicted, target, pi, valid_mask).mean()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def _score_patchtraj(
    model: PatchTrajPredictor,
    test_tokens: torch.Tensor,
    test_labels: IntArray,
    device: torch.device,
) -> float:
    pi, valid_mask = compute_correspondence_map(
        renderer_type="line_plot",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        patch_grid=(16, 16),
    )
    scores: list[np.ndarray] = []
    seq_labels: list[np.ndarray] = []
    model.eval()

    for start in range(
        0, int(test_tokens.shape[0]) - K_STEPS - DELTA + 1, PATCHTRAJ_BATCH_SIZE
    ):
        end = min(
            start + PATCHTRAJ_BATCH_SIZE,
            int(test_tokens.shape[0]) - K_STEPS - DELTA + 1,
        )
        seq_batch: list[torch.Tensor] = []
        target_batch: list[torch.Tensor] = []
        labels_batch: list[int] = []
        for i in range(start, end):
            target_idx = i + K_STEPS + DELTA - 1
            seq_batch.append(test_tokens[i : i + K_STEPS])
            target_batch.append(test_tokens[target_idx])
            labels_batch.append(int(test_labels[target_idx]))

        token_seq = torch.stack(seq_batch, dim=0).to(device=device, dtype=torch.float32)
        target = torch.stack(target_batch, dim=0).to(device=device, dtype=torch.float32)
        predicted = model(token_seq)
        batch_scores = compute_patchtraj_score(predicted, target, pi, valid_mask)
        scores.append(
            batch_scores.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        seq_labels.append(np.asarray(labels_batch, dtype=np.int64))

    score_array = normalize_scores(np.concatenate(scores, axis=0), method="minmax")
    label_array = np.concatenate(seq_labels, axis=0)
    return float(compute_all_metrics(score_array, label_array)["auc_roc"])


def _has_both_classes_in_test_windows(labels: IntArray) -> bool:
    split_idx = labels.shape[0] // 2
    test_labels = labels[split_idx:]
    if np.unique(test_labels).size < 2:
        return False
    dummy = np.zeros((test_labels.shape[0], 1), dtype=np.float64)
    _, window_labels = create_sliding_windows(dummy, test_labels, WINDOW_SIZE, STRIDE)
    return np.unique(window_labels).size == 2


def _find_selected_files(ucr_dir: Path) -> list[Path]:
    all_files = list_ucr_files(ucr_dir)
    candidates: list[Path] = []
    for path in all_files:
        upper = path.name.upper()
        if "DISTORTED" in upper or "NOISE" in upper:
            continue
        _, labels, _, _ = load_ucr_series(path)
        if _has_both_classes_in_test_windows(labels):
            candidates.append(path)

    selected: list[Path] = []
    used = set()
    for keyword in PREFERRED_KEYWORDS:
        for path in candidates:
            if path in used:
                continue
            if keyword.lower() in path.name.lower():
                selected.append(path)
                used.add(path)
                break

    for path in candidates:
        if len(selected) >= TARGET_SERIES_COUNT:
            break
        if path in used:
            continue
        selected.append(path)
        used.add(path)

    if len(selected) < TARGET_SERIES_COUNT:
        raise ValueError(
            f"Could only select {len(selected)} eligible UCR series; expected {TARGET_SERIES_COUNT}."
        )
    selected = selected[:TARGET_SERIES_COUNT]
    return selected


def _run_one_series(
    path: Path, backbone: VisionBackbone, device: torch.device
) -> dict[str, object]:
    data, labels, anomaly_start, anomaly_end = load_ucr_series(path)

    split_idx = data.shape[0] // 2
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    test_labels_timestep = labels[split_idx:]

    if train_data.shape[0] <= WINDOW_SIZE or test_data.shape[0] <= WINDOW_SIZE:
        raise ValueError(
            f"Series {path.name} is too short for window_size={WINDOW_SIZE}."
        )

    train_norm, test_norm = normalize_data(train_data, test_data, method="standard")
    train_labels = np.zeros((train_norm.shape[0],), dtype=np.int64)

    train_windows, _ = create_sliding_windows(
        train_norm, train_labels, WINDOW_SIZE, STRIDE
    )
    test_windows, test_window_labels = create_sliding_windows(
        test_norm, test_labels_timestep, WINDOW_SIZE, STRIDE
    )

    classical = _run_classical_baselines(
        train_windows, test_windows, test_window_labels
    )

    patch_train_windows, _ = _subsample_windows(
        train_windows,
        np.zeros((train_windows.shape[0],), dtype=np.int64),
        MAX_PATCHTRAJ_TRAIN_WINDOWS,
    )
    patch_test_windows, patch_test_labels = _subsample_windows(
        test_windows,
        test_window_labels,
        MAX_PATCHTRAJ_TEST_WINDOWS,
    )

    train_tokens = _extract_tokens(patch_train_windows, backbone, PATCHTRAJ_BATCH_SIZE)
    test_tokens = _extract_tokens(patch_test_windows, backbone, PATCHTRAJ_BATCH_SIZE)
    model = _train_patchtraj(train_tokens, device)
    patchtraj_auc = _score_patchtraj(model, test_tokens, patch_test_labels, device)

    aucs = {**classical, "PatchTraj": patchtraj_auc}
    return {
        "file": str(path),
        "series": path.name,
        "anomaly_start": int(anomaly_start),
        "anomaly_end": int(anomaly_end),
        "train_length": int(train_data.shape[0]),
        "test_length": int(test_data.shape[0]),
        "auc_roc": aucs,
    }


def _mean_auc(per_series: list[dict[str, Any]]) -> dict[str, float]:
    methods = ["PatchTraj", "LOF", "IsolationForest", "OneClassSVM"]
    result: dict[str, float] = {}
    for method in methods:
        vals: list[float] = []
        for item in per_series:
            auc_map = item.get("auc_roc")
            if not isinstance(auc_map, dict):
                raise ValueError(
                    "Each series result must contain an 'auc_roc' dictionary."
                )
            vals.append(float(auc_map[method]))
        result[method] = float(np.mean(vals))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run UCR PatchTraj + classical baselines."
    )
    parser.add_argument(
        "--ucr_dir",
        type=str,
        default="data/UCR",
        help="Root directory containing extracted UCR files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/reports/ucr_results.json",
        help="Output JSON path.",
    )
    return parser


def main() -> None:
    _setup_logging()
    _set_seeds(SEED)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

    args = _build_parser().parse_args()
    ucr_dir = Path(args.ucr_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = VisionBackbone(model_name="facebook/dinov2-base", device=device)

    selected_files = _find_selected_files(ucr_dir)
    LOGGER.info(
        "Running UCR experiment on %d representative series.", len(selected_files)
    )

    per_series: list[dict[str, object]] = []
    for file_path in selected_files:
        LOGGER.info("Processing %s", file_path.name)
        per_series.append(_run_one_series(file_path, backbone, device))

    payload = {
        "dataset": "UCR Anomaly Archive 2021",
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "selected_series": [path.name for path in selected_files],
        "patchtraj": {
            "model": "facebook/dinov2-base",
            "renderer": "line_plot",
            "K": K_STEPS,
            "delta": DELTA,
            "d_model": PATCHTRAJ_D_MODEL,
            "n_heads": PATCHTRAJ_N_HEADS,
            "n_layers": PATCHTRAJ_N_LAYERS,
            "dim_feedforward": PATCHTRAJ_DIM_FF,
            "hidden_dim": 768,
            "epochs": PATCHTRAJ_EPOCHS,
            "max_train_windows_per_series": MAX_PATCHTRAJ_TRAIN_WINDOWS,
            "max_test_windows_per_series": MAX_PATCHTRAJ_TEST_WINDOWS,
        },
        "classical_baselines": {
            "LOF": {"n_neighbors": 20, "contamination": "auto"},
            "IsolationForest": {
                "n_estimators": 100,
                "contamination": "auto",
                "random_state": 42,
            },
            "OneClassSVM": {"kernel": "rbf", "gamma": "auto", "nu": 0.01},
        },
        "per_series": per_series,
        "mean_auc_roc": _mean_auc(per_series),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    LOGGER.info("Saved UCR results to %s", output_path)


if __name__ == "__main__":
    main()
