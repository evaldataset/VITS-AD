from __future__ import annotations

import logging
from typing import cast

import torch
from torch import nn
from typing_extensions import override


LOGGER = logging.getLogger(__name__)


class _TemporalBlock(nn.Module):
    """Residual temporal convolution block with dilation."""

    conv1: nn.Conv1d
    act1: nn.GELU
    dropout1: nn.Dropout
    conv2: nn.Conv1d
    act2: nn.GELU
    dropout2: nn.Dropout
    residual: nn.Conv1d | nn.Identity

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}."
            )
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(p=dropout)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        y = cast(torch.Tensor, self.conv1(x))
        y = cast(torch.Tensor, self.act1(y))
        y = cast(torch.Tensor, self.dropout1(y))
        y = cast(torch.Tensor, self.conv2(y))
        y = cast(torch.Tensor, self.act2(y))
        y = cast(torch.Tensor, self.dropout2(y))
        return cast(torch.Tensor, y + residual)


class TCNAutoencoder(nn.Module):
    """Lightweight TCN autoencoder for 1D reconstruction anomaly scoring.

    Args:
        input_dim: Number of features ``D`` in each timestep.
        hidden_channels: Encoder stem width.
        bottleneck_channels: Bottleneck channel width.
        kernel_size: Temporal convolution kernel size (odd integer).
        dropout: Dropout applied in each temporal block.
    """

    input_dim: int
    encoder: nn.Sequential
    decoder: nn.Sequential
    output_projection: nn.Conv1d

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 32,
        bottleneck_channels: int = 48,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._validate_config(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            bottleneck_channels=bottleneck_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.input_dim = int(input_dim)

        self.encoder = nn.Sequential(
            _TemporalBlock(
                in_channels=self.input_dim,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=1,
                dropout=dropout,
            ),
            _TemporalBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=2,
                dropout=dropout,
            ),
            _TemporalBlock(
                in_channels=hidden_channels,
                out_channels=bottleneck_channels,
                kernel_size=kernel_size,
                dilation=4,
                dropout=dropout,
            ),
        )

        self.decoder = nn.Sequential(
            _TemporalBlock(
                in_channels=bottleneck_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=4,
                dropout=dropout,
            ),
            _TemporalBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=2,
                dropout=dropout,
            ),
            _TemporalBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=1,
                dropout=dropout,
            ),
        )

        self.output_projection = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=self.input_dim,
            kernel_size=1,
        )

        LOGGER.info(
            "Initialized TCNAutoencoder(input_dim=%d, params=%d)",
            self.input_dim,
            self.count_parameters(),
        )

    @override
    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """Reconstruct input sequence.

        Args:
            window: Input tensor with shape ``(L, D)`` or ``(B, L, D)``.

        Returns:
            Reconstructed tensor with the same shape as ``window``.

        Raises:
            ValueError: If input shape, dtype, or values are invalid.
        """
        sequence, squeeze_batch = self._as_batch(window)
        x = sequence.transpose(1, 2)
        x = cast(torch.Tensor, self.encoder(x))
        x = cast(torch.Tensor, self.decoder(x))
        reconstructed = cast(torch.Tensor, self.output_projection(x)).transpose(1, 2)
        if squeeze_batch:
            return reconstructed.squeeze(0)
        return reconstructed

    @torch.no_grad()
    def compute_reconstruction_score(self, window: torch.Tensor) -> torch.Tensor:
        """Compute per-window MSE reconstruction anomaly scores.

        Args:
            window: Input tensor with shape ``(L, D)`` or ``(B, L, D)``.

        Returns:
            1D tensor of shape ``(B,)`` with per-window MSE scores.
        """
        sequence, _ = self._as_batch(window)
        reconstruction = self.forward(sequence)
        squared_error = (reconstruction - sequence) ** 2
        return squared_error.mean(dim=(1, 2))

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _as_batch(self, window: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if window.ndim == 2:
            sequence = window.unsqueeze(0)
            squeeze_batch = True
        elif window.ndim == 3:
            sequence = window
            squeeze_batch = False
        else:
            raise ValueError(
                f"window must have shape (L, D) or (B, L, D), got {tuple(window.shape)}."
            )

        if not torch.is_floating_point(sequence):
            raise ValueError(f"window must be floating tensor, got dtype={sequence.dtype}.")
        if sequence.shape[0] <= 0:
            raise ValueError("window batch dimension must be positive.")
        if sequence.shape[1] <= 0:
            raise ValueError("window length L must be positive.")
        if sequence.shape[2] != self.input_dim:
            raise ValueError(
                f"window feature dimension must equal input_dim={self.input_dim}, got D={sequence.shape[2]}."
            )
        if not torch.isfinite(sequence).all():
            raise ValueError("window contains non-finite values.")

        return sequence, squeeze_batch

    def _validate_config(
        self,
        input_dim: int,
        hidden_channels: int,
        bottleneck_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive, got {hidden_channels}."
            )
        if bottleneck_channels <= 0:
            raise ValueError(
                f"bottleneck_channels must be positive, got {bottleneck_channels}."
            )
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")


__all__ = ["TCNAutoencoder"]
