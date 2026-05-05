from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

from src.models.backbone import VisionBackbone


LOGGER = logging.getLogger(__name__)

_BACKBONE_NAMES: dict[str, str] = {
    "clip": "openai/clip-vit-base-patch16",
    "dinov2": "facebook/dinov2-base",
}


class EnsembleBackbone:
    """Dual frozen backbone wrapper for CLIP+DINOv2 extraction.

    The class can operate in three output modes:
    - ``selected``: return tokens from a single selected backbone (drop-in mode).
    - ``dict``: return a dict with tokens from both backbones.
    - ``concat``: concatenate CLIP and DINOv2 tokens along hidden dimension.

    Args:
        model_name: Reserved for compatibility with ``VisionBackbone``.
        device: Torch device. Defaults to CUDA if available.
        dtype: Model dtype.
        clip_model_name: HuggingFace CLIP vision model identifier.
        dinov2_model_name: HuggingFace DINOv2 model identifier.
        sequential: If true, enforce one-active-backbone execution for lower memory.
        load_both: If true, preload both backbones at initialization.
        output_mode: Output mode for ``extract_patch_tokens``.
        default_backbone: Fallback backbone when renderer is not provided.
        renderer_backbone_map: Renderer-to-backbone mapping.
    """

    def __init__(
        self,
        model_name: str = "ensemble-clip-dinov2",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        clip_model_name: str = _BACKBONE_NAMES["clip"],
        dinov2_model_name: str = _BACKBONE_NAMES["dinov2"],
        sequential: bool = True,
        load_both: bool = False,
        output_mode: Literal["selected", "dict", "concat"] = "selected",
        default_backbone: Literal["clip", "dinov2"] = "clip",
        renderer_backbone_map: dict[str, Literal["clip", "dinov2"]] | None = None,
    ) -> None:
        """Initialize a dual-backbone ensemble extractor.

        Raises:
            ValueError: If mode or backbone settings are invalid.
        """
        self.model_name = model_name
        if output_mode not in {"selected", "dict", "concat"}:
            raise ValueError(
                f"output_mode must be one of selected/dict/concat, got {output_mode}."
            )
        if default_backbone not in {"clip", "dinov2"}:
            raise ValueError(
                f"default_backbone must be one of clip/dinov2, got {default_backbone}."
            )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.clip_model_name = clip_model_name
        self.dinov2_model_name = dinov2_model_name
        self.sequential = sequential
        self.load_both = load_both
        self.output_mode = output_mode
        self.default_backbone = default_backbone
        self.renderer_backbone_map = renderer_backbone_map or {
            "line_plot": "clip",
            "recurrence_plot": "dinov2",
        }

        self._backbones: dict[str, VisionBackbone | None] = {
            "clip": None,
            "dinov2": None,
        }
        self._active_backbone_name = default_backbone

        if self.load_both:
            self._ensure_backbone("clip")
            self._ensure_backbone("dinov2")
        else:
            self._ensure_backbone(self._active_backbone_name)

    @property
    def hidden_dim(self) -> int:
        """Return hidden dimension based on output mode."""
        return self._resolve_hidden_dim()

    @property
    def num_patches(self) -> int:
        """Return number of patch tokens per image."""
        if self.output_mode == "selected":
            return self._ensure_backbone(self._active_backbone_name).num_patches
        return self._ensure_backbone("clip").num_patches

    @property
    def patch_grid(self) -> tuple[int, int]:
        """Return (height, width) of patch grid."""
        if self.output_mode == "selected":
            return self._ensure_backbone(self._active_backbone_name).patch_grid
        return self._ensure_backbone("clip").patch_grid

    def set_renderer(self, renderer_name: str) -> None:
        """Select active backbone based on renderer.

        Args:
            renderer_name: Renderer identifier (e.g. ``line_plot``).
        """
        selected = self.renderer_backbone_map.get(renderer_name, self.default_backbone)
        self._active_backbone_name = selected

    @torch.no_grad()
    def extract_patch_tokens(
        self, images: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Extract patch tokens using configured ensemble mode.

        Args:
            images: Batch of images, shape (B, 3, 224, 224), float32, values in [0, 1].

        Returns:
            Depending on ``output_mode``:
            - ``selected``: Tensor ``(B, P, D)`` from active backbone.
            - ``dict``: ``{"clip": ..., "dinov2": ...}`` tensors.
            - ``concat``: Tensor ``(B, P, D_clip + D_dino)``.
        """
        if self.output_mode == "selected":
            backbone = self._prepare_backbone_for_inference(self._active_backbone_name)
            return backbone.extract_patch_tokens(images)

        clip_tokens = self._extract_from_backbone(images, "clip")
        dino_tokens = self._extract_from_backbone(images, "dinov2")
        if self.output_mode == "dict":
            return {"clip": clip_tokens, "dinov2": dino_tokens}
        if clip_tokens.shape[1] != dino_tokens.shape[1]:
            raise RuntimeError(
                "Cannot concatenate patch tokens with different patch counts: "
                f"clip={clip_tokens.shape[1]}, dinov2={dino_tokens.shape[1]}."
            )
        return torch.cat([clip_tokens, dino_tokens], dim=-1)

    @torch.no_grad()
    def extract_with_attention(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Extract tokens with attention from active backbone.

        This method intentionally mirrors ``VisionBackbone`` behavior and only
        supports the selected backbone pathway.

        Args:
            images: Batch of images, shape (B, 3, 224, 224), float32, values in [0, 1].

        Returns:
            Tuple of ``(patch_tokens, attentions)`` for active backbone.

        Raises:
            RuntimeError: If ``output_mode`` is not ``selected``.
        """
        if self.output_mode != "selected":
            raise RuntimeError(
                "extract_with_attention is only available when output_mode='selected'."
            )
        backbone = self._prepare_backbone_for_inference(self._active_backbone_name)
        return backbone.extract_with_attention(images)

    @torch.no_grad()
    def extract_patch_tokens_from_numpy(
        self, images: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32] | dict[str, npt.NDArray[np.float32]]:
        """Extract patch tokens from numpy images.

        Args:
            images: Array with shape (B, 3, H, W) or (3, H, W), float32, values in [0, 1].

        Returns:
            Numpy tokens in the same mode semantics as ``extract_patch_tokens``.
        """
        tensor_images = torch.from_numpy(images)
        tokens = self.extract_patch_tokens(tensor_images)
        if isinstance(tokens, dict):
            return {
                key: value.detach().cpu().numpy().astype(np.float32, copy=False)
                for key, value in tokens.items()
            }
        return tokens.detach().cpu().numpy().astype(np.float32, copy=False)

    def _resolve_hidden_dim(self) -> int:
        if self.output_mode == "selected":
            return self._ensure_backbone(self._active_backbone_name).hidden_dim
        clip_dim = self._ensure_backbone("clip").hidden_dim
        dino_dim = self._ensure_backbone("dinov2").hidden_dim
        return clip_dim + dino_dim if self.output_mode == "concat" else clip_dim

    def _extract_from_backbone(self, images: torch.Tensor, name: str) -> torch.Tensor:
        backbone = self._prepare_backbone_for_inference(name)
        return backbone.extract_patch_tokens(images)

    def _prepare_backbone_for_inference(self, name: str) -> VisionBackbone:
        backbone = self._ensure_backbone(name)
        if self.sequential and not self.load_both:
            other_name = "dinov2" if name == "clip" else "clip"
            self._backbones[other_name] = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return backbone

    def _ensure_backbone(self, name: str) -> VisionBackbone:
        existing = self._backbones[name]
        if existing is not None:
            return existing

        model_name = self.clip_model_name if name == "clip" else self.dinov2_model_name
        backbone = VisionBackbone(
            model_name=model_name,
            device=self.device,
            dtype=self.dtype,
        )
        self._backbones[name] = backbone
        LOGGER.info("Loaded %s backbone (%s).", name, model_name)
        return backbone
