from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from transformers import (
    AutoImageProcessor,
    CLIPVisionModel,
    Dinov2Model,
    SiglipVisionModel,
)


LOGGER = logging.getLogger(__name__)

_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)

_MODEL_REGISTRY: dict[str, str] = {
    "facebook/dinov2-base": "dinov2",
    "facebook/dinov2-large": "dinov2",
    "openai/clip-vit-base-patch16": "clip",
    "openai/clip-vit-large-patch14": "clip",
    "google/siglip-base-patch16-224": "siglip",
}


class VisionBackbone:
    """Frozen vision model wrapper for patch token extraction.

    Supports DINOv2 and CLIP vision encoders. The backbone is always frozen
    (no gradients) and used only for inference.

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device. Defaults to CUDA if available.
        dtype: Model dtype.

    Attributes:
        model: The underlying vision model (frozen).
        processor: Image preprocessor from HuggingFace.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize a frozen vision backbone.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to run inference on.
            dtype: Model parameter dtype.

        Raises:
            ValueError: If model name is unsupported.
        """
        if model_name not in _MODEL_REGISTRY:
            supported = ", ".join(sorted(_MODEL_REGISTRY))
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Supported models: {supported}."
            )

        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self._family = _MODEL_REGISTRY[model_name]

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = self._load_model(model_name=model_name, dtype=dtype)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device=self.device)

        self._hidden_dim = int(self._resolve_hidden_dim(self.model.config))
        self._patch_grid = self._resolve_patch_grid(self.model.config)
        self._num_patches = int(self._patch_grid[0] * self._patch_grid[1])

        mean, std = self._resolve_normalization()
        self._mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor(std, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

        LOGGER.info(
            "Loaded frozen backbone %s on %s (dtype=%s, hidden_dim=%d, num_patches=%d)",
            self.model_name,
            self.device,
            self.dtype,
            self._hidden_dim,
            self._num_patches,
        )

    @torch.no_grad()
    def extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from a batch of images.

        Args:
            images: Batch of images, shape (B, 3, 224, 224), float32, values in [0, 1].

        Returns:
            Patch tokens of shape (B, num_patches, hidden_dim), excluding CLS token.

        Raises:
            ValueError: If images have wrong shape, dtype, or value range.
            RuntimeError: If model output shape is unexpected.
        """
        patch_tokens, _ = self._extract_tokens_and_attentions(
            images, output_attentions=False
        )
        return patch_tokens

    @torch.no_grad()
    def extract_with_attention(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Extract patch tokens and attention tensors from a batch of images.

        Args:
            images: Batch of images, shape (B, 3, 224, 224), float32, values in [0, 1].

        Returns:
            Tuple ``(patch_tokens, attentions)`` where:
            - ``patch_tokens`` has shape ``(B, num_patches, hidden_dim)``.
            - ``attentions`` is a tuple of per-layer attention tensors, each with
              shape ``(B, num_heads, sequence_len, sequence_len)``.

        Raises:
            RuntimeError: If model output shape is unexpected or attention tensors
                are missing/invalid.
            ValueError: If images have wrong shape, dtype, or value range.
        """
        return self._extract_tokens_and_attentions(images, output_attentions=True)

    @torch.no_grad()
    def extract_patch_tokens_from_numpy(
        self, images: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Extract patch tokens from numpy images.

        Args:
            images: Array with shape (B, 3, H, W) or (3, H, W), float32, values in [0, 1].

        Returns:
            Patch tokens as numpy array with shape (B, num_patches, hidden_dim).

        Raises:
            ValueError: If input shape or dtype is invalid.
        """
        if not isinstance(images, np.ndarray):
            raise ValueError(f"images must be a numpy.ndarray, got {type(images)!r}.")
        if images.dtype != np.float32:
            raise ValueError(f"images must have dtype float32, got {images.dtype}.")
        if images.ndim == 3:
            images = images[None, ...]
        elif images.ndim != 4:
            raise ValueError(
                f"images must have shape (3, H, W) or (B, 3, H, W), got {images.shape}."
            )

        image_tensor = torch.from_numpy(images)
        patch_tokens = self.extract_patch_tokens(image_tensor)
        return patch_tokens.detach().cpu().numpy()

    @property
    def hidden_dim(self) -> int:
        """Return the dimension of patch token embeddings."""
        return self._hidden_dim

    @property
    def num_patches(self) -> int:
        """Return number of patch tokens per image."""
        return self._num_patches

    @property
    def patch_grid(self) -> tuple[int, int]:
        """Return (height, width) of the patch grid."""
        return self._patch_grid

    def _load_model(self, model_name: str, dtype: torch.dtype) -> torch.nn.Module:
        if self._family == "dinov2":
            return Dinov2Model.from_pretrained(model_name, torch_dtype=dtype)
        if self._family == "clip":
            return CLIPVisionModel.from_pretrained(model_name, torch_dtype=dtype)
        if self._family == "siglip":
            return SiglipVisionModel.from_pretrained(model_name, torch_dtype=dtype)
        raise ValueError(f"Unsupported model family '{self._family}'.")

    def _resolve_normalization(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self._family == "dinov2":
            return _DINO_MEAN, _DINO_STD

        image_mean = tuple(float(value) for value in self.processor.image_mean)
        image_std = tuple(float(value) for value in self.processor.image_std)
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError(
                "Processor normalization must contain 3 channels, got "
                f"mean={image_mean}, std={image_std}."
            )
        return image_mean, image_std

    def _resolve_hidden_dim(self, config: Any) -> int:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Could not resolve hidden_size from config: {config}.")
        return int(hidden_size)

    def _resolve_patch_grid(
        self, config: Any, input_image_size: int = 224
    ) -> tuple[int, int]:
        # Use actual input image size (224), NOT model's native image_size.
        # DINOv2 native is 518, but we feed 224x224 images.
        image_size = (input_image_size, input_image_size)
        patch_size = self._to_pair(getattr(config, "patch_size", 16), "patch_size")

        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError(
                "image_size must be divisible by patch_size, got "
                f"image_size={image_size}, patch_size={patch_size}."
            )
        return image_size[0] // patch_size[0], image_size[1] // patch_size[1]

    def _validate_images(self, images: torch.Tensor) -> None:
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"images must be a torch.Tensor, got {type(images)!r}.")
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape (B, 3, 224, 224), got {tuple(images.shape)}."
            )
        if images.shape[1] != 3:
            raise ValueError(f"images must have 3 channels, got {images.shape[1]}.")
        if images.shape[2] != 224 or images.shape[3] != 224:
            raise ValueError(
                "images must have spatial size 224x224, "
                f"got {images.shape[2]}x{images.shape[3]}."
            )
        if images.dtype != torch.float32:
            raise ValueError(
                f"images must have dtype torch.float32, got {images.dtype}."
            )
        if not torch.isfinite(images).all().item():
            raise ValueError("images contain non-finite values.")

        value_min = float(images.min().item())
        value_max = float(images.max().item())
        if value_min < 0.0 or value_max > 1.0:
            raise ValueError(
                "images values must be in [0, 1], "
                f"got min={value_min:.6f}, max={value_max:.6f}."
            )

    def _extract_tokens_and_attentions(
        self, images: torch.Tensor, output_attentions: bool
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        self._validate_images(images)

        pixel_values = images.to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        pixel_values = (pixel_values - self._mean) / self._std
        pixel_values = pixel_values.to(dtype=self.dtype)

        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
        )
        tokens = outputs.last_hidden_state
        if tokens.ndim != 3:
            raise RuntimeError(
                "Expected model output with shape (B, N, D), "
                f"but got {tuple(tokens.shape)}."
            )
        if tokens.shape[1] < 2:
            raise RuntimeError(
                "Expected at least one CLS and one patch token, "
                f"but got sequence length {tokens.shape[1]}."
            )

        patch_tokens = tokens[:, 1:, :]
        if patch_tokens.shape[1] != self._num_patches:
            raise RuntimeError(
                "Unexpected number of patch tokens: "
                f"expected {self._num_patches}, got {patch_tokens.shape[1]}."
            )
        if patch_tokens.shape[2] != self._hidden_dim:
            raise RuntimeError(
                "Unexpected hidden dimension: "
                f"expected {self._hidden_dim}, got {patch_tokens.shape[2]}."
            )

        attentions_tuple: tuple[torch.Tensor, ...] = tuple()
        if output_attentions:
            raw_attentions = outputs.attentions
            if raw_attentions is None:
                raise RuntimeError(
                    "Model did not return attentions with output_attentions=True."
                )
            attentions_tuple = tuple(raw_attentions)
            if not attentions_tuple:
                raise RuntimeError("Model returned an empty attention tuple.")
            for layer_idx, layer_attention in enumerate(attentions_tuple):
                if layer_attention.ndim != 4:
                    raise RuntimeError(
                        "Expected attention tensor shape (B, H, S, S), "
                        f"but got layer {layer_idx} shape {tuple(layer_attention.shape)}."
                    )
                if layer_attention.shape[0] != patch_tokens.shape[0]:
                    raise RuntimeError(
                        "Attention batch size mismatch: "
                        f"tokens batch={patch_tokens.shape[0]}, layer {layer_idx} batch={layer_attention.shape[0]}."
                    )
                if (
                    layer_attention.shape[2] != tokens.shape[1]
                    or layer_attention.shape[3] != tokens.shape[1]
                ):
                    raise RuntimeError(
                        "Attention sequence length mismatch: "
                        f"tokens sequence={tokens.shape[1]}, layer {layer_idx} has "
                        f"({layer_attention.shape[2]}, {layer_attention.shape[3]})."
                    )

        return patch_tokens, attentions_tuple

    @torch.no_grad()
    def extract_multilayer_tokens(
        self,
        images: torch.Tensor,
        layers: tuple[int, ...] = (4, 8, 12),
    ) -> torch.Tensor:
        """Extract and concatenate patch tokens from multiple ViT layers.

        Inspired by PaDiM (Defard et al., ICPR 2021): early layers capture
        low-level texture/edges while deeper layers capture semantic patterns.
        Concatenating gives richer per-patch descriptors for anomaly scoring.

        Args:
            images: Batch of images ``(B, 3, 224, 224)``, float32 in ``[0, 1]``.
            layers: 1-indexed layer indices to extract from (e.g., ``(4, 8, 12)``
                for a 12-layer ViT). Each must be in ``[1, num_layers]``.

        Returns:
            Concatenated patch tokens ``(B, N, D * len(layers))``.

        Raises:
            ValueError: If ``layers`` is empty or contains invalid indices.
            RuntimeError: If hidden states are unavailable.
        """
        if not layers:
            raise ValueError("layers must be non-empty.")
        self._validate_images(images)

        pixel_values = images.to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        pixel_values = (pixel_values - self._mean) / self._std
        pixel_values = pixel_values.to(dtype=self.dtype)

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "Model did not return hidden_states. "
                "Ensure the model supports output_hidden_states=True."
            )

        # hidden_states is a tuple of (num_layers+1) tensors (including embedding layer)
        num_available = len(hidden_states) - 1  # exclude embedding layer at index 0
        for layer_idx in layers:
            if layer_idx < 1 or layer_idx > num_available:
                raise ValueError(
                    f"Layer index {layer_idx} out of range [1, {num_available}]."
                )

        selected: list[torch.Tensor] = []
        for layer_idx in layers:
            # hidden_states[0] = embeddings, hidden_states[1] = layer 1, etc.
            layer_tokens = hidden_states[layer_idx][:, 1:, :]  # exclude CLS
            selected.append(layer_tokens)

        return torch.cat(selected, dim=-1)  # (B, N, D * len(layers))

    @torch.no_grad()
    def extract_multilayer_tokens_from_numpy(
        self,
        images: npt.NDArray[np.float32],
        layers: tuple[int, ...] = (4, 8, 12),
    ) -> npt.NDArray[np.float32]:
        """Extract multi-layer patch tokens from numpy images.

        Args:
            images: ``(B, 3, H, W)`` or ``(3, H, W)``, float32 in ``[0, 1]``.
            layers: Layer indices to extract.

        Returns:
            Patch tokens ``(B, N, D * len(layers))``.
        """
        if not isinstance(images, np.ndarray):
            raise ValueError(f"images must be numpy.ndarray, got {type(images)!r}.")
        if images.dtype != np.float32:
            raise ValueError(f"images must be float32, got {images.dtype}.")
        if images.ndim == 3:
            images = images[None, ...]
        elif images.ndim != 4:
            raise ValueError(f"images must be 3D or 4D, got {images.shape}.")

        image_tensor = torch.from_numpy(images)
        tokens = self.extract_multilayer_tokens(image_tensor, layers=layers)
        return tokens.detach().cpu().numpy()

    def _to_pair(self, value: Any, field_name: str) -> tuple[int, int]:
        if isinstance(value, int):
            return value, value
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        raise ValueError(f"Invalid {field_name} in model config: {value!r}.")
