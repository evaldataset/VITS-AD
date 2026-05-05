from __future__ import annotations

# pyright: reportMissingImports=false

import pytest
import torch

from src.rendering.token_correspondence_ot import compute_ot_correspondence


def _make_tokens(seed: int = 0, num_tokens: int = 196, dim: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((num_tokens, dim), generator=generator, dtype=torch.float32)


def test_ot_sinkhorn_converges_within_100_iterations() -> None:
    tokens_t = _make_tokens(seed=123)
    tokens_t1 = _make_tokens(seed=456)

    soft_pi, iterations = compute_ot_correspondence(
        tokens_t=tokens_t,
        tokens_t1=tokens_t1,
        reg=0.1,
        max_iterations=100,
        tolerance=1e-3,
        hard_assignment=False,
    )

    assert soft_pi.shape == (196, 196)
    assert 1 <= iterations <= 100


def test_ot_soft_pi_is_approximately_doubly_stochastic() -> None:
    tokens_t = _make_tokens(seed=7)
    tokens_t1 = _make_tokens(seed=8)

    soft_pi, _ = compute_ot_correspondence(
        tokens_t=tokens_t,
        tokens_t1=tokens_t1,
        reg=0.1,
        max_iterations=100,
        tolerance=1e-3,
    )

    uniform_mass = torch.full((196,), 1.0 / 196.0, dtype=torch.float32)
    torch.testing.assert_close(soft_pi.sum(dim=1), uniform_mass, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(soft_pi.sum(dim=0), uniform_mass, atol=1e-3, rtol=1e-3)


def test_ot_identity_for_identical_tokens_with_hard_assignment() -> None:
    tokens = _make_tokens(seed=42)

    hard_pi, iterations = compute_ot_correspondence(
        tokens_t=tokens,
        tokens_t1=tokens,
        reg=0.05,
        max_iterations=100,
        tolerance=1e-4,
        hard_assignment=True,
    )

    expected = torch.arange(196, dtype=torch.int64)
    assert 1 <= iterations <= 100
    torch.testing.assert_close(hard_pi, expected)


def test_ot_soft_correspondence_is_differentiable() -> None:
    tokens_t = _make_tokens(seed=1).requires_grad_(True)
    tokens_t1 = _make_tokens(seed=2)

    soft_pi, _ = compute_ot_correspondence(
        tokens_t=tokens_t,
        tokens_t1=tokens_t1,
        reg=0.1,
        max_iterations=40,
        tolerance=1e-2,
    )
    loss = (soft_pi.pow(2)).sum()
    loss.backward()

    assert tokens_t.grad is not None
    assert torch.isfinite(tokens_t.grad).all().item()


def test_ot_invalid_reg_raises_value_error() -> None:
    tokens = _make_tokens(seed=3)

    with pytest.raises(ValueError, match="reg must be positive"):
        compute_ot_correspondence(tokens, tokens, reg=0.0)
