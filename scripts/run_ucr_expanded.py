from __future__ import annotations

# pyright: basic, reportMissingImports=false, reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportIndexIssue=false, reportImplicitOverride=false, reportUntypedFunctionDecorator=false, reportArgumentType=false

"""Expanded UCR benchmark: VITS (LP + DINOv2-base, spatial+dual α=0.5) on all eligible series.

Runs the full VITS pipeline with:
  - Renderer: line_plot (LP)
  - Backbone: facebook/dinov2-base (frozen)
  - Predictor: SpatialTemporalPatchTrajPredictor
  - Scorer: DualSignalScorer (alpha=0.5, trajectory + distributional)

Evaluates ALL non-DISTORTED, non-NOISE UCR series that have both anomaly and
normal windows in the test split (half-split protocol).

Results:
  results/ucr_expanded/per_series/<series_name>.json  — per-series metrics
  results/ucr_expanded/summary.json                   — mean±std across all series
"""

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
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.base import create_sliding_windows, normalize_data
from src.data.ucr import list_ucr_files, load_ucr_series
from src.evaluation.metrics import compute_all_metrics
from src.models.backbone import VisionBackbone
from src.models.patchtraj import SpatialTemporalPatchTrajPredictor
from src.rendering.line_plot import render_line_plot_batch
from src.rendering.token_correspondence import compute_correspondence_map
from src.scoring.dual_signal_scorer import DualSignalScorer
from src.scoring.patchtraj_scorer import compute_patchtraj_score, normalize_scores


LOGGER = logging.getLogger(__name__)

# ---- Hyperparameters --------------------------------------------------------
SEED = 42
WINDOW_SIZE = 100
STRIDE = 1
K_STEPS = 8
DELTA = 1
PATCH_GRID = (16, 16)

PATCHTRAJ_D_MODEL = 256
PATCHTRAJ_N_HEADS = 4
PATCHTRAJ_N_LAYERS = 2
PATCHTRAJ_DIM_FF = 1024
PATCHTRAJ_DROPOUT = 0.1
PATCHTRAJ_EPOCHS = 8
PATCHTRAJ_BATCH_SIZE = 32
PATCHTRAJ_LR = 1e-3
PATCHTRAJ_WEIGHT_DECAY = 1e-5

MAX_PATCHTRAJ_TRAIN_WINDOWS = 600
MAX_PATCHTRAJ_TEST_WINDOWS = 800

DUAL_ALPHA = 0.5  # 50% trajectory, 50% distributional

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


# ---- Dataset ----------------------------------------------------------------


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
        return (
            self.tokens[anchor - self.k + 1 : anchor + 1],
            self.tokens[anchor + self.delta],
        )


# ---- Utilities --------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _subsample_windows(
    windows: FloatArray, labels: IntArray, max_windows: int
) -> tuple[FloatArray, IntArray]:
    if windows.shape[0] <= max_windows:
        return windows, labels
    idx = np.linspace(0, windows.shape[0] - 1, max_windows, dtype=np.int64)
    return windows[idx], labels[idx]


def _has_both_classes_in_test_windows(labels: IntArray) -> bool:
    split_idx = labels.shape[0] // 2
    test_labels = labels[split_idx:]
    if np.unique(test_labels).size < 2:
        return False
    dummy = np.zeros((test_labels.shape[0], 1), dtype=np.float64)
    _, window_labels = create_sliding_windows(dummy, test_labels, WINDOW_SIZE, STRIDE)
    return np.unique(window_labels).size == 2


def _find_all_eligible_files(ucr_dir: Path) -> list[Path]:
    """Return all non-DISTORTED, non-NOISE UCR files that have both classes."""
    all_files = list_ucr_files(ucr_dir)
    eligible: list[Path] = []
    for path in all_files:
        upper = path.name.upper()
        if "DISTORTED" in upper or "NOISE" in upper:
            continue
        try:
            _, labels, _, _ = load_ucr_series(path)
        except (ValueError, OSError) as exc:
            LOGGER.warning("Skipping %s: %s", path.name, exc)
            continue
        if _has_both_classes_in_test_windows(labels):
            eligible.append(path)
    LOGGER.info("Found %d eligible UCR series (non-DISTORTED, non-NOISE).", len(eligible))
    return eligible


# ---- Token extraction -------------------------------------------------------


@torch.no_grad()
def _extract_tokens(
    windows: FloatArray,
    backbone: VisionBackbone,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, windows.shape[0], batch_size):
        end = min(start + batch_size, windows.shape[0])
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


# ---- Training ---------------------------------------------------------------


def _train_spatial_patchtraj(
    train_tokens: torch.Tensor,
    device: torch.device,
) -> SpatialTemporalPatchTrajPredictor:
    model = SpatialTemporalPatchTrajPredictor(
        hidden_dim=int(train_tokens.shape[-1]),
        d_model=PATCHTRAJ_D_MODEL,
        n_heads=PATCHTRAJ_N_HEADS,
        n_layers=PATCHTRAJ_N_LAYERS,
        dim_feedforward=PATCHTRAJ_DIM_FF,
        dropout=PATCHTRAJ_DROPOUT,
        patch_grid=PATCH_GRID,
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
        patch_grid=PATCH_GRID,
    )

    for epoch in range(PATCHTRAJ_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for token_seq, target in loader:
            token_seq = token_seq.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(token_seq)
            loss = compute_patchtraj_score(predicted, target, pi, valid_mask).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        LOGGER.debug("Epoch %d/%d loss=%.4f", epoch + 1, PATCHTRAJ_EPOCHS, epoch_loss / max(n_batches, 1))

    return model


# ---- Scoring ----------------------------------------------------------------


@torch.no_grad()
def _compute_scores(
    model: SpatialTemporalPatchTrajPredictor,
    test_tokens: torch.Tensor,
    test_labels: IntArray,
    dual_scorer: DualSignalScorer,
    device: torch.device,
) -> tuple[float, float, float]:
    """Compute AUC-ROC for trajectory-only, distributional-only, and dual-signal."""
    pi, valid_mask = compute_correspondence_map(
        renderer_type="line_plot",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        patch_grid=PATCH_GRID,
    )

    traj_scores_list: list[np.ndarray] = []
    dist_scores_list: list[np.ndarray] = []
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

        batch_traj = compute_patchtraj_score(predicted, target, pi, valid_mask)
        traj_scores_list.append(batch_traj.detach().cpu().numpy().astype(np.float64))

        # Distributional score on the target tokens (B, N, D)
        target_np = target.detach().cpu().numpy().astype(np.float64)
        batch_dist = dual_scorer.score_distributional(target_np)
        dist_scores_list.append(batch_dist)

        seq_labels.append(np.asarray(labels_batch, dtype=np.int64))

    traj_arr = normalize_scores(np.concatenate(traj_scores_list, axis=0), method="minmax")
    dist_arr = normalize_scores(np.concatenate(dist_scores_list, axis=0), method="minmax")
    label_arr = np.concatenate(seq_labels, axis=0)

    dual_fused = dual_scorer.fuse(traj_arr, dist_arr)

    traj_auc = float(compute_all_metrics(traj_arr, label_arr)["auc_roc"])
    dist_auc = float(compute_all_metrics(dist_arr, label_arr)["auc_roc"])
    dual_auc = float(compute_all_metrics(dual_fused, label_arr)["auc_roc"])
    return traj_auc, dist_auc, dual_auc


# ---- Per-series runner ------------------------------------------------------


def _run_one_series(
    path: Path,
    backbone: VisionBackbone,
    device: torch.device,
    out_dir: Path,
) -> dict[str, object] | None:
    series_name = path.stem
    out_file = out_dir / "per_series" / f"{series_name}.json"
    if out_file.exists():
        LOGGER.info("Skipping %s (already done).", series_name)
        with out_file.open() as fh:
            return json.load(fh)

    try:
        data, labels, anomaly_start, anomaly_end = load_ucr_series(path)
    except (ValueError, OSError) as exc:
        LOGGER.warning("Failed to load %s: %s", path.name, exc)
        return None

    split_idx = data.shape[0] // 2
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    test_labels_ts = labels[split_idx:]

    if train_data.shape[0] <= WINDOW_SIZE or test_data.shape[0] <= WINDOW_SIZE:
        LOGGER.warning("Series %s too short, skipping.", series_name)
        return None

    train_norm, test_norm = normalize_data(train_data, test_data, method="standard")
    train_labels_dummy = np.zeros((train_norm.shape[0],), dtype=np.int64)

    train_windows, _ = create_sliding_windows(train_norm, train_labels_dummy, WINDOW_SIZE, STRIDE)
    test_windows, test_window_labels = create_sliding_windows(test_norm, test_labels_ts, WINDOW_SIZE, STRIDE)

    # Subsample for GPU budget
    pt_train_windows, _ = _subsample_windows(
        train_windows,
        np.zeros((train_windows.shape[0],), dtype=np.int64),
        MAX_PATCHTRAJ_TRAIN_WINDOWS,
    )
    pt_test_windows, pt_test_labels = _subsample_windows(
        test_windows, test_window_labels, MAX_PATCHTRAJ_TEST_WINDOWS
    )

    train_tokens = _extract_tokens(pt_train_windows, backbone, PATCHTRAJ_BATCH_SIZE)
    test_tokens = _extract_tokens(pt_test_windows, backbone, PATCHTRAJ_BATCH_SIZE)

    # Fit dual scorer on training tokens
    dual_scorer = DualSignalScorer(alpha=DUAL_ALPHA)
    dual_scorer.fit(train_tokens.numpy().astype(np.float64))

    model = _train_spatial_patchtraj(train_tokens, device)

    traj_auc, dist_auc, dual_auc = _compute_scores(
        model, test_tokens, pt_test_labels, dual_scorer, device
    )

    result: dict[str, object] = {
        "series": series_name,
        "file": str(path),
        "anomaly_start": int(anomaly_start),
        "anomaly_end": int(anomaly_end),
        "train_length": int(train_data.shape[0]),
        "test_length": int(test_data.shape[0]),
        "auc_roc": {
            "VITS_Traj": traj_auc,
            "VITS_Dist": dist_auc,
            "VITS_Dual": dual_auc,
        },
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    LOGGER.info(
        "%s  traj=%.4f  dist=%.4f  dual=%.4f",
        series_name, traj_auc, dist_auc, dual_auc,
    )
    return result


# ---- Summary ----------------------------------------------------------------


def _compute_summary(
    per_series: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    methods = ["VITS_Traj", "VITS_Dist", "VITS_Dual"]
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        vals = [float(item["auc_roc"][method]) for item in per_series if method in item.get("auc_roc", {})]
        if vals:
            summary[method] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
    return summary


# ---- Entry point ------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expanded UCR benchmark: VITS (spatial+dual α=0.5) on all eligible series."
    )
    parser.add_argument(
        "--ucr_dir",
        type=str,
        default="data/UCR",
        help="Root directory containing extracted UCR files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ucr_expanded",
        help="Output directory for per-series and summary results.",
    )
    return parser


def main() -> None:
    _setup_logging()
    _set_seeds(SEED)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

    args = _build_parser().parse_args()
    ucr_dir = Path(args.ucr_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_series").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    backbone = VisionBackbone(model_name="facebook/dinov2-base", device=device)

    eligible_files = _find_all_eligible_files(ucr_dir)
    LOGGER.info("Running VITS on %d UCR series.", len(eligible_files))

    per_series: list[dict[str, Any]] = []
    for i, path in enumerate(eligible_files):
        LOGGER.info("[%d/%d] Processing %s", i + 1, len(eligible_files), path.name)
        result = _run_one_series(path, backbone, device, out_dir)
        if result is not None:
            per_series.append(result)

    summary = _compute_summary(per_series)

    payload = {
        "dataset": "UCR Anomaly Archive 2021 (expanded)",
        "n_series": len(per_series),
        "config": {
            "backbone": "facebook/dinov2-base",
            "renderer": "line_plot",
            "predictor": "SpatialTemporalPatchTrajPredictor",
            "dual_alpha": DUAL_ALPHA,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "K": K_STEPS,
            "delta": DELTA,
            "d_model": PATCHTRAJ_D_MODEL,
            "n_heads": PATCHTRAJ_N_HEADS,
            "n_layers": PATCHTRAJ_N_LAYERS,
            "epochs": PATCHTRAJ_EPOCHS,
        },
        "summary_auc_roc": summary,
        "per_series": per_series,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    LOGGER.info("Saved summary to %s", summary_path)

    print("\n=== UCR Expanded Results (mean±std AUC-ROC) ===")
    for method, stats in summary.items():
        print(f"  {method:<20}  {stats['mean']:.4f} ± {stats['std']:.4f}  (n={stats['n']})")
    print()


if __name__ == "__main__":
    main()
