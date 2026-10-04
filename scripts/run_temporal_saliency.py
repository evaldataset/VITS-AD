#!/usr/bin/env python3
"""Generate temporal saliency figures from PatchTraj + DINOv2 attention."""

from __future__ import annotations

# pyright: reportMissingImports=false,reportMissingTypeArgument=false,reportUnknownVariableType=false,reportUnknownMemberType=false,reportUnknownArgumentType=false,reportUnknownParameterType=false,reportUnusedCallResult=false,reportAny=false,reportExplicitAny=false,reportUntypedFunctionDecorator=false,reportAttributeAccessIssue=false,reportArgumentType=false,reportPossiblyUnboundVariable=false

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.base import create_sliding_windows, normalize_data
from src.data.smd import _load_smd_labels, _load_smd_matrix
from src.models.backbone import VisionBackbone
from src.models.patchtraj import PatchTrajPredictor
from src.models.temporal_saliency import (
    TemporalSaliencyMapper,
    compute_attention_rollout,
)
from src.rendering.line_plot import render_line_plot_batch
from src.rendering.recurrence_plot import render_recurrence_plot_batch
from src.rendering.token_correspondence import compute_correspondence_map
from src.scoring.patchtraj_scorer import compute_patchtraj_score
from src.utils.reproducibility import seed_everything


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SaliencyWindowResult:
    """Container for one selected window's saliency outputs."""

    split: str
    target_index: int
    label: int
    score: float
    window: np.ndarray
    rendered_image: np.ndarray
    patch_importance: np.ndarray
    timestep_importance: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run temporal saliency visualization for PatchTraj checkpoints.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/smd"))
    parser.add_argument("--entity", type=str, default="machine-1-1")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PatchTraj checkpoint path. If omitted, auto-resolve from results/.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory for outputs and checkpoint auto-resolution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="line_plot",
        choices=["line_plot", "recurrence_plot"],
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="facebook/dinov2-base")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda:1. Defaults to cuda:1 if available.",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--discard-ratio", type=float, default=0.9)
    parser.add_argument("--head-fusion", type=str, default="mean")
    parser.add_argument("--line-dpi", type=int, default=100)
    parser.add_argument("--recurrence-metric", type=str, default="euclidean")
    parser.add_argument(
        "--recurrence-threshold",
        type=float,
        default=None,
        help="Threshold for binary recurrence; omit for continuous recurrence.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure process logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_device(device_arg: str | None) -> torch.device:
    """Resolve runtime device with GPU-1 default when available."""
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        index = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    """Resolve checkpoint path from explicit arg or common result layouts."""
    if args.checkpoint is not None:
        return args.checkpoint

    candidates = [
        args.results_root
        / "improved_smd"
        / args.entity
        / args.renderer
        / "patchtraj_model.pt",
        args.results_root
        / "improved_smd"
        / args.entity
        / args.renderer
        / "best_model.pt",
        args.results_root / f"dinov2-base_smd_{args.renderer}" / "best_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = "\n - ".join(str(path) for path in candidates)
    raise FileNotFoundError("Could not resolve checkpoint. Tried:\n - " + joined)


def load_smd_windows(
    raw_dir: Path,
    entity: str,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized SMD test windows and binary window labels."""
    train_data = _load_smd_matrix(raw_dir / "train" / f"{entity}.txt")
    test_data = _load_smd_matrix(raw_dir / "test" / f"{entity}.txt")
    test_labels = _load_smd_labels(raw_dir / "test_label" / f"{entity}.txt")

    _, test_data_norm = normalize_data(
        train_data=train_data,
        test_data=test_data,
        method="standard",
    )

    windows, _ = create_sliding_windows(
        data=test_data_norm,
        labels=test_labels,
        window_size=window_size,
        stride=stride,
    )

    starts = np.arange(
        0, test_labels.shape[0] - window_size + 1, stride, dtype=np.int64
    )
    window_labels = np.asarray(
        [
            int(np.any(test_labels[start : start + window_size] == 1))
            for start in starts
        ],
        dtype=np.int64,
    )
    return windows.astype(np.float32, copy=False), window_labels


def get_renderer(
    renderer: str,
    image_size: int,
    line_dpi: int,
    recurrence_metric: str,
    recurrence_threshold: float | None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve renderer function and kwargs for image generation."""
    if renderer == "line_plot":
        return (
            render_line_plot_batch,
            {
                "image_size": image_size,
                "dpi": line_dpi,
                "colormap": "tab10",
                "line_width": 1.0,
                "background_color": "white",
                "show_axes": False,
                "show_grid": False,
            },
        )

    kwargs: dict[str, Any] = {
        "image_size": image_size,
        "metric": recurrence_metric,
    }
    if recurrence_threshold is not None:
        kwargs["threshold"] = recurrence_threshold
    return render_recurrence_plot_batch, kwargs


@torch.no_grad()
def extract_token_windows(
    windows: np.ndarray,
    backbone: VisionBackbone,
    renderer_fn: Any,
    renderer_kwargs: dict[str, Any],
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    """Render windows and extract patch tokens into a temporary cache."""
    token_chunks: list[np.ndarray] = []
    for start in range(0, int(windows.shape[0]), batch_size):
        end = min(start + batch_size, int(windows.shape[0]))
        images = renderer_fn(windows[start:end], **renderer_kwargs)
        image_tensor = torch.from_numpy(images.astype(np.float32, copy=False))
        tokens = backbone.extract_patch_tokens(image_tensor)
        token_chunks.append(
            tokens.detach().cpu().numpy().astype(np.float32, copy=False)
        )

    token_windows = np.concatenate(token_chunks, axis=0)
    np.save(cache_path, token_windows)
    LOGGER.info("Saved temporary token cache: %s", cache_path)
    return token_windows


def build_predictor(
    checkpoint: dict[str, Any],
    hidden_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[PatchTrajPredictor, int, int]:
    """Build PatchTraj predictor from checkpoint config with CLI fallback."""
    checkpoint_cfg = (
        checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    )
    patchtraj_cfg: dict[str, Any] = checkpoint_cfg.get("patchtraj", {})

    model = PatchTrajPredictor(
        hidden_dim=hidden_dim,
        d_model=int(patchtraj_cfg.get("d_model", args.d_model)),
        n_heads=int(patchtraj_cfg.get("n_heads", args.n_heads)),
        n_layers=int(patchtraj_cfg.get("n_layers", args.n_layers)),
        dim_feedforward=int(patchtraj_cfg.get("dim_feedforward", args.dim_feedforward)),
        dropout=float(patchtraj_cfg.get("dropout", args.dropout)),
        activation=str(patchtraj_cfg.get("activation", args.activation)),
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint format: expected state_dict mapping.")
    model.load_state_dict(state_dict)
    model.eval()

    k = int(patchtraj_cfg.get("K", args.k))
    delta = int(patchtraj_cfg.get("delta", args.delta))
    return model, k, delta


@torch.no_grad()
def score_windows(
    token_windows: np.ndarray,
    window_labels: np.ndarray,
    model: PatchTrajPredictor,
    k: int,
    delta: int,
    pi: np.ndarray,
    valid_mask: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute anomaly scores for target windows used by PatchTraj inference."""
    num_windows = int(token_windows.shape[0])
    start_target = k + delta - 1
    target_indices = np.arange(start_target, num_windows, dtype=np.int64)
    if target_indices.size == 0:
        raise ValueError(
            f"No target windows available for K={k}, delta={delta}, T={num_windows}."
        )

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []

    for start in range(0, int(target_indices.size), batch_size):
        end = min(start + batch_size, int(target_indices.size))
        batch_targets = target_indices[start:end]

        seq_batch_np = np.stack(
            [
                token_windows[target_idx - delta - k + 1 : target_idx - delta + 1]
                for target_idx in batch_targets
            ],
            axis=0,
        )
        target_batch_np = token_windows[batch_targets]

        seq_batch = torch.from_numpy(seq_batch_np).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        target_batch = torch.from_numpy(target_batch_np).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        pred_batch = model(seq_batch)
        batch_scores = compute_patchtraj_score(
            predicted_tokens=pred_batch,
            actual_tokens=target_batch,
            pi=pi,
            valid_mask=valid_mask,
        )
        scores.append(
            batch_scores.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        labels.append(window_labels[batch_targets].astype(np.int64, copy=False))
        indices.append(batch_targets)

    return (
        np.concatenate(scores, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(indices, axis=0),
    )


def select_indices(
    scores: np.ndarray,
    labels: np.ndarray,
    target_indices: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top anomalous and top normal target indices from scored windows."""
    anomaly_mask = labels == 1
    normal_mask = labels == 0

    if int(np.sum(anomaly_mask)) < top_k:
        raise ValueError(
            f"Not enough anomalous windows for top_k={top_k}. Found {int(np.sum(anomaly_mask))}."
        )
    if int(np.sum(normal_mask)) < top_k:
        raise ValueError(
            f"Not enough normal windows for top_k={top_k}. Found {int(np.sum(normal_mask))}."
        )

    anomaly_local = np.flatnonzero(anomaly_mask)
    normal_local = np.flatnonzero(normal_mask)

    anomaly_order = anomaly_local[np.argsort(scores[anomaly_local])[::-1]]
    normal_order = normal_local[np.argsort(scores[normal_local])]

    return target_indices[anomaly_order[:top_k]], target_indices[normal_order[:top_k]]


@torch.no_grad()
def compute_saliency_for_index(
    split: str,
    target_index: int,
    windows: np.ndarray,
    window_labels: np.ndarray,
    scores_by_index: dict[int, float],
    backbone: VisionBackbone,
    renderer_fn: Any,
    renderer_kwargs: dict[str, Any],
    mapper: TemporalSaliencyMapper,
    discard_ratio: float,
    head_fusion: str,
) -> SaliencyWindowResult:
    """Compute rendered image, patch saliency, and timestep saliency for one window."""
    window = windows[target_index]
    image_batch = renderer_fn(window[None, ...], **renderer_kwargs)
    image = image_batch[0]

    image_tensor = torch.from_numpy(image_batch.astype(np.float32, copy=False))
    _, attentions = backbone.extract_with_attention(image_tensor)

    patch_importance = compute_attention_rollout(
        attentions=attentions,
        discard_ratio=discard_ratio,
        head_fusion=head_fusion,
    )
    timestep_importance = mapper.map_to_timesteps(patch_importance)

    return SaliencyWindowResult(
        split=split,
        target_index=int(target_index),
        label=int(window_labels[target_index]),
        score=float(scores_by_index[int(target_index)]),
        window=window.astype(np.float64, copy=False),
        rendered_image=image.astype(np.float64, copy=False),
        patch_importance=patch_importance.astype(np.float64, copy=False),
        timestep_importance=timestep_importance.astype(np.float64, copy=False),
    )


def _aggregate_signal(window: np.ndarray) -> np.ndarray:
    """Aggregate multivariate window into a single robust visualization trace."""
    centered = window - np.median(window, axis=0, keepdims=True)
    trace = np.mean(centered, axis=1)
    std = float(np.std(trace))
    if np.isclose(std, 0.0):
        return np.zeros_like(trace, dtype=np.float64)
    return (trace / std).astype(np.float64, copy=False)


def save_figure(fig: plt.Figure, path_stem: Path, dpi: int) -> None:
    """Save one figure as PDF and PNG."""
    fig.savefig(path_stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved figure: %s.[pdf|png]", path_stem)


def make_overlay_figure(
    anomalous_results: list[SaliencyWindowResult],
    normal_results: list[SaliencyWindowResult],
    output_dir: Path,
    dpi: int,
) -> None:
    """Create figure with timestep saliency overlayed on time-series traces."""
    rows = len(anomalous_results) + len(normal_results)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 1.8 * rows), sharex=True)
    axes_array = np.atleast_1d(axes)

    ordered = [("Anomaly", result) for result in anomalous_results] + [
        ("Normal", result) for result in normal_results
    ]

    for ax, (group, result) in zip(axes_array, ordered):
        signal = _aggregate_signal(result.window)
        x_axis = np.arange(signal.shape[0], dtype=np.float64)

        ax.plot(x_axis, signal, color="#1f2937", linewidth=1.1)
        y_min = float(np.min(signal))
        y_max = float(np.max(signal))
        if np.isclose(y_min, y_max):
            y_max = y_min + 1.0

        saliency_strip = np.tile(result.timestep_importance[None, :], (8, 1))
        ax.imshow(
            saliency_strip,
            cmap="magma",
            aspect="auto",
            alpha=0.45,
            extent=(0, signal.shape[0] - 1, y_min, y_max),
            interpolation="nearest",
            origin="lower",
        )
        ax.set_ylabel("z", fontsize=10)
        ax.set_title(
            f"{group} | idx={result.target_index} | label={result.label} | score={result.score:.4f}",
            fontsize=10,
        )
        ax.tick_params(labelsize=9)

    axes_array[-1].set_xlabel("Timestep", fontsize=10)
    fig.tight_layout()
    save_figure(fig, output_dir / "temporal_saliency_overlay", dpi=dpi)


def make_patch_grid_figure(
    anomalous_results: list[SaliencyWindowResult],
    normal_results: list[SaliencyWindowResult],
    output_dir: Path,
    dpi: int,
) -> None:
    """Create figure comparing rendered image and patch-level saliency maps."""
    picks = [anomalous_results[0], normal_results[0]]
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    for row, result in enumerate(picks):
        image_hwc = np.transpose(result.rendered_image, (1, 2, 0))
        axes[row, 0].imshow(np.clip(image_hwc, 0.0, 1.0))
        axes[row, 0].set_title(
            f"Rendered ({result.split}) idx={result.target_index}", fontsize=10
        )
        axes[row, 0].axis("off")

        saliency_map = axes[row, 1].imshow(
            result.patch_importance,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        patch_h, patch_w = result.patch_importance.shape
        axes[row, 1].set_title(f"Attention rollout ({patch_h}x{patch_w})", fontsize=10)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

    colorbar = fig.colorbar(saliency_map, ax=axes[:, 1], fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    save_figure(fig, output_dir / "temporal_saliency_patch_grid", dpi=dpi)


def make_profile_figure(
    anomalous_results: list[SaliencyWindowResult],
    normal_results: list[SaliencyWindowResult],
    output_dir: Path,
    dpi: int,
) -> None:
    """Create average timestep saliency profile figure for normal vs anomaly."""
    anomalous_stack = np.stack(
        [result.timestep_importance for result in anomalous_results], axis=0
    )
    normal_stack = np.stack(
        [result.timestep_importance for result in normal_results], axis=0
    )
    x_axis = np.arange(anomalous_stack.shape[1], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    anomaly_mean = np.mean(anomalous_stack, axis=0)
    anomaly_std = np.std(anomalous_stack, axis=0)
    normal_mean = np.mean(normal_stack, axis=0)
    normal_std = np.std(normal_stack, axis=0)

    ax.plot(x_axis, anomaly_mean, color="#b91c1c", linewidth=1.6, label="Anomalous")
    ax.fill_between(
        x_axis,
        np.clip(anomaly_mean - anomaly_std, 0.0, 1.0),
        np.clip(anomaly_mean + anomaly_std, 0.0, 1.0),
        color="#ef4444",
        alpha=0.2,
    )
    ax.plot(x_axis, normal_mean, color="#1d4ed8", linewidth=1.6, label="Normal")
    ax.fill_between(
        x_axis,
        np.clip(normal_mean - normal_std, 0.0, 1.0),
        np.clip(normal_mean + normal_std, 0.0, 1.0),
        color="#60a5fa",
        alpha=0.25,
    )

    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_ylabel("Saliency", fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    save_figure(fig, output_dir / "temporal_saliency_profile", dpi=dpi)


def main() -> None:
    """Run temporal saliency figure generation pipeline."""
    configure_logging()
    args = parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    LOGGER.info("Using device: %s", device)

    checkpoint_path = resolve_checkpoint_path(args)
    LOGGER.info("Using checkpoint: %s", checkpoint_path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    windows, window_labels = load_smd_windows(
        raw_dir=args.raw_dir,
        entity=args.entity,
        window_size=args.window_size,
        stride=args.stride,
    )
    LOGGER.info("Loaded %d test windows for %s", int(windows.shape[0]), args.entity)

    renderer_fn, renderer_kwargs = get_renderer(
        renderer=args.renderer,
        image_size=args.image_size,
        line_dpi=args.line_dpi,
        recurrence_metric=args.recurrence_metric,
        recurrence_threshold=args.recurrence_threshold,
    )

    backbone = VisionBackbone(model_name=args.backbone, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # nosec: trusted local checkpoint
    model, k, delta = build_predictor(
        checkpoint=checkpoint,
        hidden_dim=backbone.hidden_dim,
        device=device,
        args=args,
    )

    pi, valid_mask = compute_correspondence_map(
        renderer_type=args.renderer,
        window_size=args.window_size,
        stride=args.stride,
        patch_grid=backbone.patch_grid,
    )

    tmp_cache_name = f".tmp_saliency_tokens_{args.entity}_{args.renderer}.npy"
    token_cache_path = output_dir / tmp_cache_name

    try:
        token_windows = extract_token_windows(
            windows=windows,
            backbone=backbone,
            renderer_fn=renderer_fn,
            renderer_kwargs=renderer_kwargs,
            batch_size=args.batch_size,
            cache_path=token_cache_path,
        )

        scores, eval_labels, target_indices = score_windows(
            token_windows=token_windows,
            window_labels=window_labels,
            model=model,
            k=k,
            delta=delta,
            pi=pi,
            valid_mask=valid_mask,
            device=device,
            batch_size=args.batch_size,
        )

        anomalous_targets, normal_targets = select_indices(
            scores=scores,
            labels=eval_labels,
            target_indices=target_indices,
            top_k=args.top_k,
        )

        score_by_target = {
            int(target_idx): float(score)
            for target_idx, score in zip(target_indices, scores)
        }
        mapper = TemporalSaliencyMapper(
            renderer_type=args.renderer,
            window_size=args.window_size,
            image_size=args.image_size,
            patch_grid=backbone.patch_grid,
        )

        anomalous_results = [
            compute_saliency_for_index(
                split="anomalous",
                target_index=int(target_idx),
                windows=windows,
                window_labels=window_labels,
                scores_by_index=score_by_target,
                backbone=backbone,
                renderer_fn=renderer_fn,
                renderer_kwargs=renderer_kwargs,
                mapper=mapper,
                discard_ratio=args.discard_ratio,
                head_fusion=args.head_fusion,
            )
            for target_idx in anomalous_targets
        ]
        normal_results = [
            compute_saliency_for_index(
                split="normal",
                target_index=int(target_idx),
                windows=windows,
                window_labels=window_labels,
                scores_by_index=score_by_target,
                backbone=backbone,
                renderer_fn=renderer_fn,
                renderer_kwargs=renderer_kwargs,
                mapper=mapper,
                discard_ratio=args.discard_ratio,
                head_fusion=args.head_fusion,
            )
            for target_idx in normal_targets
        ]

        make_overlay_figure(
            anomalous_results=anomalous_results,
            normal_results=normal_results,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        make_patch_grid_figure(
            anomalous_results=anomalous_results,
            normal_results=normal_results,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        make_profile_figure(
            anomalous_results=anomalous_results,
            normal_results=normal_results,
            output_dir=output_dir,
            dpi=args.dpi,
        )

        LOGGER.info("Temporal saliency figure generation complete: %s", output_dir)
    finally:
        if token_cache_path.exists():
            token_cache_path.unlink()
            LOGGER.info("Removed temporary token cache: %s", token_cache_path)


if __name__ == "__main__":
    main()
