"""Tests for src/models/patchtraj.py (PatchTrajPredictor)."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportPrivateImportUsage=false

import pytest
import torch
from torch import nn

from src.models.patchtraj import (
    _MAX_TEMPORAL_STEPS,
    PatchTrajPredictor,
    SpatialTemporalPatchTrajPredictor,
)


def _create_predictor(
    *, hidden_dim: int = 32, dropout: float = 0.0
) -> PatchTrajPredictor:
    """Create a small CPU-only predictor for fast tests."""
    return PatchTrajPredictor(
        hidden_dim=hidden_dim,
        d_model=16,
        n_heads=2,
        n_layers=1,
        dim_feedforward=64,
        dropout=dropout,
        activation="gelu",
    ).to(torch.device("cpu"))


def test_forward_output_shape() -> None:
    """Return shape is (B, N, D) for input (B, K, N, D)."""
    model = _create_predictor()
    token_seq = torch.randn(3, 4, 196, 32, device=torch.device("cpu"))

    output = model(token_seq)

    assert output.shape == (3, 196, 32)


@pytest.mark.parametrize("temporal_steps", [1, 4, 8])
def test_forward_variable_k(temporal_steps: int) -> None:
    """Forward pass works with different temporal lengths K."""
    model = _create_predictor()
    token_seq = torch.randn(2, temporal_steps, 64, 32, device=torch.device("cpu"))

    output = model(token_seq)

    assert output.shape == (2, 64, 32)


@pytest.mark.parametrize("num_patches", [64, 196, 256])
def test_forward_variable_n(num_patches: int) -> None:
    """Forward pass works with different patch counts N."""
    model = _create_predictor()
    token_seq = torch.randn(2, 4, num_patches, 32, device=torch.device("cpu"))

    output = model(token_seq)

    assert output.shape == (2, num_patches, 32)


def test_overfit_single_sample_loss_decreases() -> None:
    """Model overfits one sample with strong loss decrease."""
    torch.manual_seed(7)
    model = _create_predictor(dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    token_seq = torch.randn(1, 4, 64, 32, device=torch.device("cpu"))
    target = torch.zeros(1, 64, 32, device=torch.device("cpu"))

    initial_loss = criterion(model(token_seq), target).item()

    for _ in range(50):
        optimizer.zero_grad()
        prediction = model(token_seq)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(token_seq), target).item()

    assert final_loss < initial_loss * 0.1


def test_backward_populates_all_parameter_gradients() -> None:
    """Backward pass creates non-None gradients for all parameters."""
    model = _create_predictor(dropout=0.0)
    token_seq = torch.randn(2, 4, 64, 32, device=torch.device("cpu"))
    target = torch.randn(2, 64, 32, device=torch.device("cpu"))
    criterion = nn.MSELoss()

    loss = criterion(model(token_seq), target)
    loss.backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_determinism_same_seed_same_output() -> None:
    """Equal seed and input produce identical outputs."""
    torch.manual_seed(123)
    model_a = _create_predictor(dropout=0.0)
    token_seq = torch.randn(2, 4, 64, 32, device=torch.device("cpu"))

    torch.manual_seed(123)
    model_b = _create_predictor(dropout=0.0)

    output_a = model_a(token_seq)
    output_b = model_b(token_seq.clone())

    assert torch.allclose(output_a, output_b)


def test_forward_raises_on_wrong_ndim() -> None:
    """Raise ValueError when input tensor rank is not 4."""
    model = _create_predictor()
    token_seq = torch.randn(2, 4, 64, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="shape"):
        model(token_seq)


def test_forward_raises_on_wrong_dtype() -> None:
    """Raise ValueError when input tensor is non-floating."""
    model = _create_predictor()
    token_seq = torch.randint(
        0, 10, (2, 4, 64, 32), dtype=torch.int64, device=torch.device("cpu")
    )

    with pytest.raises(ValueError, match="floating tensor"):
        model(token_seq)


def test_forward_raises_on_wrong_hidden_dim() -> None:
    """Raise ValueError when input hidden dimension mismatches model config."""
    model = _create_predictor(hidden_dim=32)
    token_seq = torch.randn(2, 4, 64, 31, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="Input hidden dimension"):
        model(token_seq)


def test_forward_raises_when_k_exceeds_max_temporal_steps() -> None:
    """Raise ValueError when K is larger than configured maximum."""
    model = _create_predictor()
    token_seq = torch.randn(
        1,
        _MAX_TEMPORAL_STEPS + 1,
        64,
        32,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="exceeds"):
        model(token_seq)


def test_forward_raises_on_zero_batch() -> None:
    """Raise ValueError when batch size is zero."""
    model = _create_predictor()
    token_seq = torch.randn(0, 4, 64, 32, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="Batch size"):
        model(token_seq)


def test_config_validation_negative_hidden_dim() -> None:
    """Raise ValueError for non-positive hidden_dim in constructor."""
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        PatchTrajPredictor(hidden_dim=-1, d_model=16, n_heads=2, n_layers=1)


def test_config_validation_d_model_not_divisible_by_n_heads() -> None:
    """Raise ValueError when d_model is not divisible by n_heads."""
    with pytest.raises(ValueError, match="must be divisible"):
        PatchTrajPredictor(hidden_dim=32, d_model=15, n_heads=2, n_layers=1)


def test_config_validation_dropout_out_of_range() -> None:
    """Raise ValueError for invalid dropout rates."""
    with pytest.raises(ValueError, match=r"dropout must be in \[0, 1\)"):
        PatchTrajPredictor(
            hidden_dim=32, d_model=16, n_heads=2, n_layers=1, dropout=1.0
        )


def test_count_parameters_matches_manual_count() -> None:
    """count_parameters returns positive int equal to manual total."""
    model = _create_predictor()
    manual_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    count = model.count_parameters()

    assert isinstance(count, int)
    assert count > 0
    assert count == manual_count


# =====================================================================
# SpatialTemporalPatchTrajPredictor tests
# =====================================================================


def _create_st_predictor(
    *, hidden_dim: int = 32, patch_grid: tuple[int, int] = (4, 4)
) -> SpatialTemporalPatchTrajPredictor:
    """Create a small spatial-temporal predictor for fast tests."""
    return SpatialTemporalPatchTrajPredictor(
        hidden_dim=hidden_dim,
        d_model=16,  # divisible by 4
        n_heads=2,
        n_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        patch_grid=patch_grid,
    ).to(torch.device("cpu"))


class TestSpatialTemporalForward:
    def test_output_shape(self) -> None:
        model = _create_st_predictor()
        x = torch.randn(2, 4, 16, 32)
        out = model(x)
        assert out.shape == (2, 16, 32)

    @pytest.mark.parametrize("k", [1, 4, 8])
    def test_variable_k(self, k: int) -> None:
        model = _create_st_predictor()
        x = torch.randn(2, k, 16, 32)
        out = model(x)
        assert out.shape == (2, 16, 32)

    @pytest.mark.parametrize("grid", [(4, 4), (7, 7), (14, 14)])
    def test_variable_grid(self, grid: tuple[int, int]) -> None:
        n = grid[0] * grid[1]
        model = _create_st_predictor(patch_grid=grid)
        x = torch.randn(2, 4, n, 32)
        out = model(x)
        assert out.shape == (2, n, 32)

    def test_gradient_flows(self) -> None:
        model = _create_st_predictor()
        x = torch.randn(2, 4, 16, 32, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_eval_deterministic(self) -> None:
        model = _create_st_predictor()
        model.eval()
        x = torch.randn(2, 4, 16, 32)
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)


class TestSpatialTemporalValidation:
    def test_wrong_ndim_raises(self) -> None:
        model = _create_st_predictor()
        with pytest.raises(ValueError, match="shape"):
            model(torch.randn(4, 16, 32))

    def test_wrong_hidden_dim_raises(self) -> None:
        model = _create_st_predictor(hidden_dim=32)
        with pytest.raises(ValueError, match="Hidden dim"):
            model(torch.randn(2, 4, 16, 64))

    def test_k_exceeds_max_raises(self) -> None:
        model = _create_st_predictor()
        with pytest.raises(ValueError, match="exceeds"):
            model(torch.randn(1, _MAX_TEMPORAL_STEPS + 1, 16, 32))

    def test_invalid_d_model_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible by 4"):
            SpatialTemporalPatchTrajPredictor(d_model=15)


class TestSpatialTemporalParams:
    def test_more_params_than_temporal_only(self) -> None:
        temporal = _create_predictor()
        spatial = _create_st_predictor()
        assert spatial.count_parameters() > temporal.count_parameters()

    def test_count_parameters_positive(self) -> None:
        model = _create_st_predictor()
        assert model.count_parameters() > 0
