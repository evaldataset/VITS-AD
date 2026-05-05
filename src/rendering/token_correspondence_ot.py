"""Optimal-transport token correspondence using Sinkhorn iterations."""

from __future__ import annotations

# pyright: reportMissingImports=false

import math

import torch


def _validate_ot_inputs(tokens_t: torch.Tensor, tokens_t1: torch.Tensor) -> None:
    if tokens_t.ndim != 2:
        raise ValueError(
            f"tokens_t must have shape (N, D), got {tuple(tokens_t.shape)}."
        )
    if tokens_t1.ndim != 2:
        raise ValueError(
            f"tokens_t1 must have shape (N, D), got {tuple(tokens_t1.shape)}."
        )
    if tokens_t.shape != tokens_t1.shape:
        raise ValueError(
            "tokens_t and tokens_t1 must have identical shapes, got "
            f"{tuple(tokens_t.shape)} and {tuple(tokens_t1.shape)}."
        )
    if tokens_t.shape[0] == 0:
        raise ValueError("tokens_t and tokens_t1 must contain at least one token.")
    if not torch.isfinite(tokens_t).all().item():
        raise ValueError("tokens_t contains non-finite values.")
    if not torch.isfinite(tokens_t1).all().item():
        raise ValueError("tokens_t1 contains non-finite values.")


def compute_ot_correspondence(
    tokens_t: torch.Tensor,
    tokens_t1: torch.Tensor,
    reg: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-3,
    hard_assignment: bool = False,
) -> tuple[torch.Tensor, int]:
    """Compute OT correspondence between two patch-token sets.

    Uses entropic OT with Sinkhorn updates in log space.

    Args:
        tokens_t: Source tokens of shape ``(N, D)``.
        tokens_t1: Target tokens of shape ``(N, D)``.
        reg: Entropic regularization epsilon.
        max_iterations: Maximum number of Sinkhorn iterations.
        tolerance: L-infinity tolerance for row/column marginal errors.
        hard_assignment: If ``True``, return hard correspondence indices ``(N,)``.

    Returns:
        Tuple ``(pi, num_iterations)`` where:
            - ``pi`` is soft correspondence matrix ``(N, N)`` when
              ``hard_assignment=False``.
            - ``pi`` is hard index map ``(N,)`` when ``hard_assignment=True``.
            - ``num_iterations`` is the number of iterations used.

    Raises:
        ValueError: If arguments are invalid.
    """
    _validate_ot_inputs(tokens_t=tokens_t, tokens_t1=tokens_t1)
    if reg <= 0.0:
        raise ValueError(f"reg must be positive, got {reg}.")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}.")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}.")

    num_tokens = int(tokens_t.shape[0])
    dtype = tokens_t.dtype
    device = tokens_t.device

    log_mu = torch.full((num_tokens,), -math.log(num_tokens) if num_tokens > 0 else 0.0, dtype=dtype, device=device)
    log_nu = log_mu.clone()
    uniform_mass = torch.full((num_tokens,), 1.0 / float(num_tokens), dtype=dtype, device=device)

    cost = torch.cdist(tokens_t.unsqueeze(0), tokens_t1.unsqueeze(0), p=2).squeeze(0).pow(2) / 2.0

    # Standard Sinkhorn in log-domain: iterate dual variables u, v
    u = torch.zeros(num_tokens, dtype=dtype, device=device)
    v = torch.zeros(num_tokens, dtype=dtype, device=device)

    soft_pi = torch.empty(0, device=device, dtype=dtype)
    iterations_used = max_iterations
    for iteration in range(1, max_iterations + 1):
        u = reg * (log_mu - torch.logsumexp((-cost + v) / reg, dim=1))
        v = reg * (log_nu - torch.logsumexp((-cost.T + u) / reg, dim=1))

        plan_logits = (u.unsqueeze(-1) + v.unsqueeze(-2) - cost) / reg
        soft_pi = torch.exp(plan_logits)

        row_error = torch.max(torch.abs(soft_pi.sum(dim=1) - uniform_mass))
        col_error = torch.max(torch.abs(soft_pi.sum(dim=0) - uniform_mass))
        if max(row_error.item(), col_error.item()) <= tolerance:
            iterations_used = iteration
            break

    if hard_assignment:
        return soft_pi.argmax(dim=-1).to(dtype=torch.int64), iterations_used
    return soft_pi, iterations_used
