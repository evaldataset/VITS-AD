#!/usr/bin/env python3
"""Benchmark PatchTraj inference FPS by pipeline stage."""

from __future__ import annotations

# pyright: reportMissingImports=false,reportUnknownVariableType=false,reportUnknownArgumentType=false,reportUnknownMemberType=false,reportUnknownLambdaType=false,reportUntypedFunctionDecorator=false,reportUnknownParameterType=false

import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.models.backbone import VisionBackbone
from src.models.patchtraj import PatchTrajPredictor
from src.rendering.line_plot import render_line_plot_batch
from src.rendering.recurrence_plot import render_recurrence_plot_batch


LOGGER = logging.getLogger(__name__)

_BATCH_SIZE = 64
_WINDOW_SIZE = 100
_NUM_FEATURES = 38
_WARMUP_BATCHES = 10
_TIMED_BATCHES = 100
_DEFAULT_K = 8
_IMPROVED_K = 12


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _measure_fps(
    func: Callable[[], object],
    device: torch.device,
    batch_size: int,
    warmup_batches: int,
    timed_batches: int,
) -> float:
    for _ in range(warmup_batches):
        _ = func()

    _synchronize_if_cuda(device)
    start_time = time.perf_counter()
    for _ in range(timed_batches):
        _ = func()
    _synchronize_if_cuda(device)
    elapsed_seconds = time.perf_counter() - start_time

    processed_windows = float(batch_size * timed_batches)
    return processed_windows / elapsed_seconds


def _build_predictor(
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    device: torch.device,
) -> PatchTrajPredictor:
    predictor = PatchTrajPredictor(
        hidden_dim=768,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        activation="gelu",
    ).to(device)
    predictor.eval()
    return predictor


@torch.no_grad()
def main() -> None:
    configure_logging()
    np.random.seed(42)
    _ = torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    LOGGER.info("Using device: %s", torch.cuda.get_device_name(device))

    windows_batch = np.random.randn(
        _BATCH_SIZE,
        _WINDOW_SIZE,
        _NUM_FEATURES,
    ).astype(np.float32)
    e2e_windows = np.random.randn(
        _BATCH_SIZE,
        _DEFAULT_K + 1,
        _WINDOW_SIZE,
        _NUM_FEATURES,
    ).astype(np.float32)

    line_plot_kwargs = {
        "image_size": 224,
        "dpi": 100,
        "colormap": "viridis",
        "line_width": 1.0,
        "background_color": "white",
        "show_axes": False,
        "show_grid": False,
    }
    recurrence_plot_kwargs = {
        "image_size": 224,
        "metric": "euclidean",
    }

    LOGGER.info("Benchmarking rendering stage.")
    line_render_fps = _measure_fps(
        func=lambda: render_line_plot_batch(windows_batch, **line_plot_kwargs),
        device=device,
        batch_size=_BATCH_SIZE,
        warmup_batches=_WARMUP_BATCHES,
        timed_batches=_TIMED_BATCHES,
    )
    recurrence_render_fps = _measure_fps(
        func=lambda: render_recurrence_plot_batch(
            windows_batch, **recurrence_plot_kwargs
        ),
        device=device,
        batch_size=_BATCH_SIZE,
        warmup_batches=_WARMUP_BATCHES,
        timed_batches=_TIMED_BATCHES,
    )

    images_line = render_line_plot_batch(windows_batch, **line_plot_kwargs)
    image_tensor = torch.from_numpy(images_line).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    LOGGER.info("Benchmarking backbone token extraction stage.")
    dinov2_backbone = VisionBackbone(
        model_name="facebook/dinov2-base",
        device=device,
    )
    clip_backbone = VisionBackbone(
        model_name="openai/clip-vit-base-patch16",
        device=device,
    )
    dinov2_backbone_fps = _measure_fps(
        func=lambda: dinov2_backbone.extract_patch_tokens(image_tensor),
        device=device,
        batch_size=_BATCH_SIZE,
        warmup_batches=_WARMUP_BATCHES,
        timed_batches=_TIMED_BATCHES,
    )
    clip_backbone_fps = _measure_fps(
        func=lambda: clip_backbone.extract_patch_tokens(image_tensor),
        device=device,
        batch_size=_BATCH_SIZE,
        warmup_batches=_WARMUP_BATCHES,
        timed_batches=_TIMED_BATCHES,
    )

    LOGGER.info("Benchmarking predictor stage.")
    predictor_default = _build_predictor(
        d_model=256,
        n_heads=4,
        n_layers=2,
        dim_feedforward=1024,
        device=device,
    )
    predictor_improved = _build_predictor(
        d_model=384,
        n_heads=6,
        n_layers=3,
        dim_feedforward=1536,
        device=device,
    )
    default_param_count = predictor_default.count_parameters()
    improved_param_count = predictor_improved.count_parameters()

    token_seq_default = torch.randn(
        _BATCH_SIZE,
        _DEFAULT_K,
        dinov2_backbone.num_patches,
        dinov2_backbone.hidden_dim,
        device=device,
        dtype=torch.float32,
    )
    token_seq_improved = torch.randn(
        _BATCH_SIZE,
        _IMPROVED_K,
        dinov2_backbone.num_patches,
        dinov2_backbone.hidden_dim,
        device=device,
        dtype=torch.float32,
    )
    predictor_default_fps = _measure_fps(
        func=lambda: predictor_default(token_seq_default),
        device=device,
        batch_size=_BATCH_SIZE,
        warmup_batches=_WARMUP_BATCHES,
        timed_batches=_TIMED_BATCHES,
    )
    predictor_improved_fps = _measure_fps(
        func=lambda: predictor_improved(token_seq_improved),
        device=device,
        batch_size=_BATCH_SIZE,
        warmup_batches=_WARMUP_BATCHES,
        timed_batches=_TIMED_BATCHES,
    )

    LOGGER.info(
        "Benchmarking end-to-end throughput (render->backbone->predict->score)."
    )

    def _run_e2e_line_plot() -> None:
        rendered_batches = [
            render_line_plot_batch(e2e_windows[:, idx], **line_plot_kwargs)
            for idx in range(_DEFAULT_K + 1)
        ]
        images = np.stack(rendered_batches, axis=1).astype(np.float32, copy=False)
        flat_images = torch.from_numpy(images.reshape(-1, 3, 224, 224)).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        tokens = dinov2_backbone.extract_patch_tokens(flat_images).reshape(
            _BATCH_SIZE,
            _DEFAULT_K + 1,
            dinov2_backbone.num_patches,
            dinov2_backbone.hidden_dim,
        )
        pred_tokens = predictor_default(tokens[:, :_DEFAULT_K])
        target_tokens = tokens[:, _DEFAULT_K]
        _ = (pred_tokens - target_tokens).pow(2).mean(dim=(1, 2))

    def _run_e2e_recurrence_plot() -> None:
        rendered_batches = [
            render_recurrence_plot_batch(e2e_windows[:, idx], **recurrence_plot_kwargs)
            for idx in range(_DEFAULT_K + 1)
        ]
        images = np.stack(rendered_batches, axis=1).astype(np.float32, copy=False)
        flat_images = torch.from_numpy(images.reshape(-1, 3, 224, 224)).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        tokens = dinov2_backbone.extract_patch_tokens(flat_images).reshape(
            _BATCH_SIZE,
            _DEFAULT_K + 1,
            dinov2_backbone.num_patches,
            dinov2_backbone.hidden_dim,
        )
        pred_tokens = predictor_default(tokens[:, :_DEFAULT_K])
        target_tokens = tokens[:, _DEFAULT_K]
        _ = (pred_tokens - target_tokens).pow(2).mean(dim=(1, 2))

    e2e_lp_fps = _measure_fps(
        func=_run_e2e_line_plot,
        device=device,
        batch_size=_BATCH_SIZE,
        warmup_batches=_WARMUP_BATCHES,
        timed_batches=_TIMED_BATCHES,
    )
    e2e_rp_fps = _measure_fps(
        func=_run_e2e_recurrence_plot,
        device=device,
        batch_size=_BATCH_SIZE,
        warmup_batches=_WARMUP_BATCHES,
        timed_batches=_TIMED_BATCHES,
    )

    report = {
        "rendering_fps": {
            "line_plot": line_render_fps,
            "recurrence_plot": recurrence_render_fps,
        },
        "backbone_fps": {
            "dinov2": dinov2_backbone_fps,
            "clip": clip_backbone_fps,
        },
        "predictor_fps": {
            "default_2M": predictor_default_fps,
            "improved_6M": predictor_improved_fps,
        },
        "e2e_fps": {
            "dinov2_lp": e2e_lp_fps,
            "dinov2_rp": e2e_rp_fps,
        },
        "params": {
            "default": default_param_count,
            "improved": improved_param_count,
        },
        "hardware": "RTX 3090",
        "batch_size": _BATCH_SIZE,
        "notes": "Single GPU, batch_size=64, averaged over 100 batches",
    }

    output_path = Path("results/reports/fps_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    LOGGER.info("FPS benchmark saved to %s", output_path)


if __name__ == "__main__":
    main()
