from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.tcn_ae import TCNAutoencoder
from src.scoring.hybrid_scorer import compute_hybrid_score


def test_tcn_ae_forward_preserves_shape_for_2d_and_3d_inputs() -> None:
    model = TCNAutoencoder(input_dim=8, dropout=0.0)

    single_window = torch.randn(32, 8, dtype=torch.float32)
    batch_windows = torch.randn(4, 32, 8, dtype=torch.float32)

    single_reconstruction = model(single_window)
    batch_reconstruction = model(batch_windows)

    assert single_reconstruction.shape == (32, 8)
    assert batch_reconstruction.shape == (4, 32, 8)


def test_tcn_ae_reconstruction_score_is_1d_and_non_negative() -> None:
    model = TCNAutoencoder(input_dim=6, dropout=0.0)
    windows = torch.randn(5, 24, 6, dtype=torch.float32)

    scores = model.compute_reconstruction_score(windows)

    assert scores.shape == (5,)
    assert torch.all(scores >= 0.0)


def test_tcn_ae_reconstruction_score_matches_manual_mse() -> None:
    torch.manual_seed(0)
    model = TCNAutoencoder(input_dim=4, dropout=0.0)
    windows = torch.randn(3, 20, 4, dtype=torch.float32)

    reconstruction = model(windows)
    expected = ((reconstruction - windows) ** 2).mean(dim=(1, 2))
    actual = model.compute_reconstruction_score(windows)

    assert torch.allclose(actual, expected)


def test_tcn_ae_parameter_budget_stays_under_50k_for_psm_dim() -> None:
    model = TCNAutoencoder(input_dim=25, hidden_channels=32, bottleneck_channels=48)
    assert model.count_parameters() <= 50000


def test_compute_hybrid_score_weighted_sum_matches_manual_values() -> None:
    patchtraj = np.array([0.1, 0.8, 0.3], dtype=np.float64)
    reconstruction = np.array([0.9, 0.2, 0.7], dtype=np.float64)

    fused = compute_hybrid_score(
        patchtraj_score=patchtraj,
        recon_score=reconstruction,
        method="weighted_sum",
        weight=0.75,
    )

    expected = 0.75 * patchtraj + 0.25 * reconstruction
    np.testing.assert_allclose(fused, expected)


def test_compute_hybrid_score_max_uses_shortest_length_alignment() -> None:
    patchtraj = np.array([0.1, 0.8, 0.3, 0.2], dtype=np.float64)
    reconstruction = np.array([0.9, 0.2], dtype=np.float64)

    fused = compute_hybrid_score(
        patchtraj_score=patchtraj,
        recon_score=reconstruction,
        method="max",
    )

    expected = np.array([0.9, 0.8], dtype=np.float64)
    np.testing.assert_allclose(fused, expected)


def test_compute_hybrid_score_rejects_invalid_weight_and_method() -> None:
    with pytest.raises(ValueError, match=r"weight must be in \[0, 1\]"):
        _ = compute_hybrid_score(
            patchtraj_score=np.array([0.1, 0.2], dtype=np.float64),
            recon_score=np.array([0.3, 0.4], dtype=np.float64),
            method="weighted_sum",
            weight=1.5,
        )

    with pytest.raises(ValueError, match="Unsupported hybrid method"):
        _ = compute_hybrid_score(
            patchtraj_score=np.array([0.1, 0.2], dtype=np.float64),
            recon_score=np.array([0.3, 0.4], dtype=np.float64),
            method="median",
        )
