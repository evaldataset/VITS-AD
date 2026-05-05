from __future__ import annotations

# pyright: reportUnknownMemberType=false

import logging
from typing import cast

import torch
from torch import nn
from typing_extensions import override


LOGGER = logging.getLogger(__name__)
_MAX_TEMPORAL_STEPS = 32
_MAX_SPATIAL_PATCHES = 256


class PatchTrajPredictor(nn.Module):
    """Lightweight Transformer for predicting next-window patch tokens.

    Takes K consecutive windows' patch tokens and predicts the patch tokens
    for the next window. Processes each patch position independently through
    a shared temporal Transformer encoder.

    Args:
        hidden_dim: Dimension of input patch tokens (e.g., 768 for DINOv2-base).
        d_model: Internal Transformer dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward dimension in Transformer.
        dropout: Dropout rate.
        activation: Activation function for Transformer encoder layers.
    """

    hidden_dim: int
    d_model: int
    max_temporal_steps: int
    w_in: nn.Sequential
    temporal_pos: nn.Parameter
    temporal_encoder: nn.TransformerEncoder
    w_out: nn.Linear

    def __init__(
        self,
        hidden_dim: int = 768,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """Initialize PatchTraj predictor.

        Args:
            hidden_dim: Dimension of input patch tokens (e.g., 768).
            d_model: Internal Transformer dimension.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer encoder layers.
            dim_feedforward: Feedforward layer dimension.
            dropout: Dropout rate for Transformer layers.
            activation: Activation function for Transformer encoder layers.

        Raises:
            ValueError: If configuration values are invalid.
        """
        super().__init__()

        self._validate_config(
            hidden_dim=hidden_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.max_temporal_steps = _MAX_TEMPORAL_STEPS

        self.w_in = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.temporal_pos = nn.Parameter(
            torch.randn(1, self.max_temporal_steps, 1, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.w_out = nn.Linear(d_model, hidden_dim)

        LOGGER.info(
            "Initialized PatchTrajPredictor (hidden_dim=%d, d_model=%d, n_heads=%d, n_layers=%d, params=%d)",
            self.hidden_dim,
            self.d_model,
            n_heads,
            n_layers,
            self.count_parameters(),
        )

    @override
    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        """Predict next window's patch tokens from K past windows.

        Args:
            token_seq: Patch token sequence of shape (B, K, N, D) where:
                B = batch size
                K = number of past windows
                N = number of patches (e.g., 196)
                D = hidden dimension (e.g., 768)

        Returns:
            Predicted patch tokens of shape (B, N, D).
            These are predictions for window_{t+1} given windows [t-K+1, ..., t].

        Raises:
            ValueError: If input shape, dtype, or dimensions are invalid.
        """
        if token_seq.ndim != 4:
            raise ValueError(
                f"token_seq must have shape (B, K, N, D), got {tuple(token_seq.shape)}."
            )
        if not torch.is_floating_point(token_seq):
            raise ValueError(
                f"token_seq must be a floating tensor, got dtype {token_seq.dtype}."
            )

        batch_size, temporal_steps, num_patches, hidden_dim = token_seq.shape

        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got B={batch_size}.")
        if temporal_steps <= 0:
            raise ValueError(
                f"Temporal length must be positive, got K={temporal_steps}."
            )
        if num_patches <= 0:
            raise ValueError(
                f"Number of patches must be positive, got N={num_patches}."
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Input hidden dimension does not match model configuration: expected {self.hidden_dim}, got {hidden_dim}."
            )
        if temporal_steps > self.max_temporal_steps:
            raise ValueError(
                f"Temporal length K exceeds configured positional embedding capacity: K={temporal_steps}, max={self.max_temporal_steps}."
            )

        x: torch.Tensor = cast(torch.Tensor, self.w_in(token_seq))
        x = x + self.temporal_pos[:, :temporal_steps, :, :]

        x = x.permute(1, 0, 2, 3)
        x = x.reshape(temporal_steps, batch_size * num_patches, self.d_model)

        x = cast(torch.Tensor, self.temporal_encoder(x))

        x = x[-1]
        x = x.reshape(batch_size, num_patches, self.d_model)

        pred_tokens: torch.Tensor = cast(torch.Tensor, self.w_out(x))
        return pred_tokens

    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns:
            Number of trainable model parameters.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _validate_config(
        self,
        hidden_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}.")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")
        if dim_feedforward <= 0:
            raise ValueError(
                f"dim_feedforward must be positive, got {dim_feedforward}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )


def _build_2d_sinusoidal_pos(
    grid_h: int, grid_w: int, d_model: int
) -> torch.Tensor:
    """Build 2D sinusoidal positional encoding for a patch grid.

    Args:
        grid_h: Grid height.
        grid_w: Grid width.
        d_model: Embedding dimension (must be divisible by 4).

    Returns:
        Positional encoding of shape ``(1, 1, grid_h * grid_w, d_model)``.
    """
    assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"
    d_model // 2
    quarter = d_model // 4

    omega = 1.0 / (10000.0 ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))

    rows = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1)  # (H, 1)
    cols = torch.arange(grid_w, dtype=torch.float32).unsqueeze(1)  # (W, 1)

    row_enc = torch.cat([torch.sin(rows * omega), torch.cos(rows * omega)], dim=-1)  # (H, half)
    col_enc = torch.cat([torch.sin(cols * omega), torch.cos(cols * omega)], dim=-1)  # (W, half)

    # Broadcast: (H, 1, half) + (1, W, half) -> (H, W, half) each
    pos = torch.cat(
        [
            row_enc.unsqueeze(1).expand(-1, grid_w, -1),
            col_enc.unsqueeze(0).expand(grid_h, -1, -1),
        ],
        dim=-1,
    )  # (H, W, d_model)
    return pos.reshape(1, 1, grid_h * grid_w, d_model)


class SpatialTemporalBlock(nn.Module):
    """One block of alternating temporal and spatial attention.

    Temporal attention: each patch position attends across K timesteps.
    Spatial attention: within each timestep, all N patches attend to each other.
    Both use pre-norm (LayerNorm before attention) for training stability.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Temporal attention (across K timesteps per patch)
        self.temporal_norm1 = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm2 = nn.LayerNorm(d_model)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Spatial attention (across N patches per timestep)
        self.spatial_norm1 = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm2 = nn.LayerNorm(d_model)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply temporal then spatial attention.

        Args:
            x: Input of shape ``(B, K, N, d_model)``.
            causal_mask: Optional causal mask for temporal attention, shape ``(K, K)``.

        Returns:
            Output of shape ``(B, K, N, d_model)``.
        """
        B, K, N, D = x.shape

        # --- Temporal attention: (B*N, K, D) ---
        xt = x.permute(0, 2, 1, 3).reshape(B * N, K, D)
        normed = self.temporal_norm1(xt)
        attn_out = self.temporal_attn(
            normed, normed, normed, attn_mask=causal_mask, need_weights=False
        )[0]
        xt = xt + attn_out
        xt = xt + self.temporal_ffn(self.temporal_norm2(xt))
        x = xt.reshape(B, N, K, D).permute(0, 2, 1, 3)  # (B, K, N, D)

        # --- Spatial attention: (B*K, N, D) ---
        xs = x.reshape(B * K, N, D)
        normed = self.spatial_norm1(xs)
        attn_out = self.spatial_attn(
            normed, normed, normed, need_weights=False
        )[0]
        xs = xs + attn_out
        xs = xs + self.spatial_ffn(self.spatial_norm2(xs))
        x = xs.reshape(B, K, N, D)

        return x


class SpatialTemporalPatchTrajPredictor(nn.Module):
    """PatchTraj predictor with spatial-temporal cross-attention.

    Extends :class:`PatchTrajPredictor` by adding spatial attention between
    patches within each timestep, interleaved with temporal attention across
    timesteps.  This captures cross-patch relationships (e.g., adjacent patches
    should co-vary), providing a regularising effect that reduces per-patch
    noise overfitting.

    Args:
        hidden_dim: Dimension of input patch tokens (e.g., 768).
        d_model: Internal model dimension (must be divisible by 4 for 2D pos enc).
        n_heads: Number of attention heads.
        n_layers: Number of spatial-temporal blocks.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.
        patch_grid: ``(H, W)`` patch grid dimensions for 2D spatial pos enc.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        patch_grid: tuple[int, int] = (14, 14),
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if d_model <= 0 or d_model % 4 != 0:
            raise ValueError(
                f"d_model must be positive and divisible by 4, got {d_model}."
            )
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")

        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.max_temporal_steps = _MAX_TEMPORAL_STEPS
        self.patch_grid = patch_grid

        self.w_in = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable temporal positional encoding
        self.temporal_pos = nn.Parameter(
            torch.randn(1, _MAX_TEMPORAL_STEPS, 1, d_model) * 0.02
        )

        # Fixed 2D sinusoidal spatial positional encoding
        grid_h, grid_w = patch_grid
        self.register_buffer(
            "spatial_pos",
            _build_2d_sinusoidal_pos(grid_h, grid_w, d_model),
        )

        # Spatial-temporal blocks
        self.blocks = nn.ModuleList(
            [
                SpatialTemporalBlock(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.w_out = nn.Linear(d_model, hidden_dim)

        LOGGER.info(
            "Initialized SpatialTemporalPatchTrajPredictor "
            "(hidden_dim=%d, d_model=%d, n_heads=%d, n_layers=%d, "
            "grid=%s, params=%d)",
            hidden_dim,
            d_model,
            n_heads,
            n_layers,
            patch_grid,
            self.count_parameters(),
        )

    @override
    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        """Predict next window's patch tokens from K past windows.

        Args:
            token_seq: Input of shape ``(B, K, N, D)`` where D = hidden_dim.

        Returns:
            Predicted patch tokens of shape ``(B, N, D)``.

        Raises:
            ValueError: If input shape or dimensions are invalid.
        """
        if token_seq.ndim != 4:
            raise ValueError(
                f"token_seq must have shape (B, K, N, D), got {tuple(token_seq.shape)}."
            )
        if not torch.is_floating_point(token_seq):
            raise ValueError(
                f"token_seq must be floating, got dtype {token_seq.dtype}."
            )

        B, K, N, D = token_seq.shape
        if D != self.hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch: expected {self.hidden_dim}, got {D}."
            )
        if K > self.max_temporal_steps:
            raise ValueError(
                f"K={K} exceeds max temporal steps {self.max_temporal_steps}."
            )

        # Project to d_model
        x: torch.Tensor = cast(torch.Tensor, self.w_in(token_seq))  # (B, K, N, d_model)

        # Add positional encodings
        x = x + self.temporal_pos[:, :K, :, :]
        x = x + self.spatial_pos[:, :, :N, :]

        # Build causal mask for temporal attention
        causal_mask = torch.triu(
            torch.full((K, K), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )

        # Apply spatial-temporal blocks
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask)

        # Take last timestep, project back
        x = self.final_norm(x[:, -1, :, :])  # (B, N, d_model)
        pred_tokens: torch.Tensor = cast(torch.Tensor, self.w_out(x))
        return pred_tokens

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
