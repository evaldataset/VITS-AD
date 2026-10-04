from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

_TEMPORAL_SALIENCY = importlib.import_module("src.models.temporal_saliency")
TemporalSaliencyMapper = _TEMPORAL_SALIENCY.TemporalSaliencyMapper
compute_attention_rollout = _TEMPORAL_SALIENCY.compute_attention_rollout


def _build_uniform_attentions(
    num_layers: int = 3,
    batch_size: int = 1,
    num_heads: int = 12,
    seq_len: int = 197,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.ones((batch_size, num_heads, seq_len, seq_len), dtype=torch.float32)
        for _ in range(num_layers)
    )


def test_attention_rollout_returns_14x14_grid() -> None:
    attentions = _build_uniform_attentions()

    saliency = compute_attention_rollout(
        attentions=attentions,
        discard_ratio=0.0,
        head_fusion="mean",
    )

    assert saliency.shape == (14, 14)
    assert np.isfinite(saliency).all()
    assert float(np.min(saliency)) >= 0.0
    assert float(np.max(saliency)) <= 1.0


def test_line_plot_uniform_patch_importance_maps_to_uniform_timesteps() -> None:
    mapper = TemporalSaliencyMapper(
        renderer_type="line_plot",
        window_size=100,
        image_size=224,
        patch_grid=(14, 14),
    )
    patch_importance = np.ones((14, 14), dtype=np.float64)

    timestep_importance = mapper.map_to_timesteps(patch_importance)

    assert timestep_importance.shape == (100,)
    # With uniform patch importance, timestep importance should be
    # approximately uniform (not exactly, due to discrete patch-to-timestep mapping).
    # After normalization to [0,1], all values should be > 0.5 and max = 1.0.
    assert float(np.max(timestep_importance)) == pytest.approx(1.0, abs=1e-6)
    assert float(np.min(timestep_importance)) > 0.5
    # Coefficient of variation should be low (< 0.15)
    cv = float(np.std(timestep_importance) / np.mean(timestep_importance))
    assert cv < 0.15, f"Coefficient of variation {cv:.4f} too high for uniform input"


def test_gaf_diagonal_importance_spreads_to_all_timesteps() -> None:
    mapper = TemporalSaliencyMapper(
        renderer_type="gaf",
        window_size=100,
        image_size=224,
        patch_grid=(14, 14),
    )
    patch_importance = np.eye(14, dtype=np.float64)

    timestep_importance = mapper.map_to_timesteps(patch_importance)

    assert timestep_importance.shape == (100,)
    assert np.all(timestep_importance > 0.0)


def test_recurrence_diagonal_importance_spreads_to_all_timesteps() -> None:
    mapper = TemporalSaliencyMapper(
        renderer_type="recurrence_plot",
        window_size=100,
        image_size=224,
        patch_grid=(14, 14),
    )
    patch_importance = np.eye(14, dtype=np.float64)

    timestep_importance = mapper.map_to_timesteps(patch_importance)

    assert timestep_importance.shape == (100,)
    assert np.all(timestep_importance > 0.0)


def test_single_patch_important_line_plot_localizes_to_subset() -> None:
    mapper = TemporalSaliencyMapper(
        renderer_type="line_plot",
        window_size=100,
        image_size=224,
        patch_grid=(14, 14),
    )
    patch_importance = np.zeros((14, 14), dtype=np.float64)
    patch_importance[0, 0] = 1.0

    timestep_importance = mapper.map_to_timesteps(patch_importance)

    assert timestep_importance.shape == (100,)
    assert float(np.max(timestep_importance)) == pytest.approx(1.0)
    assert np.count_nonzero(timestep_importance > 0.0) < 40


def test_all_zero_patch_importance_returns_all_zero_timesteps() -> None:
    mapper = TemporalSaliencyMapper(
        renderer_type="gaf",
        window_size=100,
        image_size=224,
        patch_grid=(14, 14),
    )
    patch_importance = np.zeros((14, 14), dtype=np.float64)

    timestep_importance = mapper.map_to_timesteps(patch_importance)

    assert timestep_importance.shape == (100,)
    assert np.allclose(timestep_importance, np.zeros((100,), dtype=np.float64))
