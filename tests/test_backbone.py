"""Tests for src/models/backbone.py — VisionBackbone validation (mock-based)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="transformers not installed")

from src.models.backbone import VisionBackbone, _MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_mock_backbone() -> VisionBackbone:
    """Create a VisionBackbone with mocked model loading (no download)."""
    with patch.object(VisionBackbone, "_load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.config.image_size = 224
        mock_model.config.patch_size = 14
        mock_load.return_value = mock_model

        with patch(
            "src.models.backbone.AutoImageProcessor.from_pretrained"
        ) as mock_proc:
            mock_processor = MagicMock()
            mock_processor.image_mean = [0.485, 0.456, 0.406]
            mock_processor.image_std = [0.229, 0.224, 0.225]
            mock_proc.return_value = mock_processor

            backbone = VisionBackbone(
                model_name="facebook/dinov2-base",
                device=torch.device("cpu"),
            )
    return backbone


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def test_model_registry_completeness() -> None:
    expected = {
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14",
        "google/siglip-base-patch16-224",
    }
    assert set(_MODEL_REGISTRY.keys()) == expected


def test_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        VisionBackbone(model_name="unknown/model", device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_backbone_properties() -> None:
    backbone = _create_mock_backbone()
    assert backbone.hidden_dim == 768
    assert (
        backbone.num_patches == 256
    )  # (224/14)^2 = 16^2 = 256... wait, 224/14=16, 16*16=256
    # Actually for DINOv2 with patch_size=14: 224/14 = 16, grid = (16, 16), num_patches = 256
    # But our config says patch_size=14 → grid = (16, 16) → 256 patches
    # The mock config has patch_size=14, image_size=224
    assert backbone.patch_grid == (16, 16)


# ---------------------------------------------------------------------------
# Input validation — tensor
# ---------------------------------------------------------------------------


def test_validate_images_wrong_shape() -> None:
    backbone = _create_mock_backbone()
    images = torch.rand(3, 224, 224)  # 3D, missing batch
    with pytest.raises(ValueError, match="shape"):
        backbone.extract_patch_tokens(images)


def test_validate_images_wrong_channels() -> None:
    backbone = _create_mock_backbone()
    images = torch.rand(1, 1, 224, 224)  # 1 channel instead of 3
    with pytest.raises(ValueError, match="3 channels"):
        backbone.extract_patch_tokens(images)


def test_validate_images_wrong_spatial() -> None:
    backbone = _create_mock_backbone()
    images = torch.rand(1, 3, 112, 112)  # Wrong spatial size
    with pytest.raises(ValueError, match="224x224"):
        backbone.extract_patch_tokens(images)


def test_validate_images_wrong_dtype() -> None:
    backbone = _create_mock_backbone()
    images = torch.randint(0, 255, (1, 3, 224, 224))  # int tensor
    with pytest.raises(ValueError, match="float32"):
        backbone.extract_patch_tokens(images)


def test_validate_images_out_of_range() -> None:
    backbone = _create_mock_backbone()
    images = torch.rand(1, 3, 224, 224) + 1.0  # Values in [1, 2]
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        backbone.extract_patch_tokens(images)


# ---------------------------------------------------------------------------
# Input validation — numpy
# ---------------------------------------------------------------------------


def test_numpy_wrong_dtype() -> None:
    backbone = _create_mock_backbone()
    images = np.random.rand(1, 3, 224, 224).astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        backbone.extract_patch_tokens_from_numpy(images)


def test_numpy_wrong_ndim() -> None:
    backbone = _create_mock_backbone()
    images = np.random.rand(224, 224).astype(np.float32)  # 2D
    with pytest.raises(ValueError, match="shape"):
        backbone.extract_patch_tokens_from_numpy(images)


# ---------------------------------------------------------------------------
# Multi-layer feature extraction
# ---------------------------------------------------------------------------


def test_multilayer_tokens_output_shape() -> None:
    backbone = _create_mock_backbone()
    B, N, D = 2, 256, 768

    # Mock model to return hidden_states (13 layers: embedding + 12 transformer)
    hidden_states = tuple(
        torch.randn(B, N + 1, D) for _ in range(13)  # +1 for CLS
    )
    mock_output = MagicMock()
    mock_output.hidden_states = hidden_states
    backbone.model.return_value = mock_output

    images = torch.rand(B, 3, 224, 224)
    tokens = backbone.extract_multilayer_tokens(images, layers=(4, 8, 12))

    assert tokens.shape == (B, N, D * 3)


def test_multilayer_tokens_single_layer() -> None:
    backbone = _create_mock_backbone()
    B, N, D = 2, 256, 768

    hidden_states = tuple(torch.randn(B, N + 1, D) for _ in range(13))
    mock_output = MagicMock()
    mock_output.hidden_states = hidden_states
    backbone.model.return_value = mock_output

    images = torch.rand(B, 3, 224, 224)
    tokens = backbone.extract_multilayer_tokens(images, layers=(12,))

    assert tokens.shape == (B, N, D)


def test_multilayer_tokens_invalid_layer_raises() -> None:
    backbone = _create_mock_backbone()
    B, N, D = 1, 256, 768

    hidden_states = tuple(torch.randn(B, N + 1, D) for _ in range(13))
    mock_output = MagicMock()
    mock_output.hidden_states = hidden_states
    backbone.model.return_value = mock_output

    images = torch.rand(B, 3, 224, 224)
    with pytest.raises(ValueError, match="out of range"):
        backbone.extract_multilayer_tokens(images, layers=(0,))  # 0 is embedding
    with pytest.raises(ValueError, match="out of range"):
        backbone.extract_multilayer_tokens(images, layers=(13,))  # exceeds 12


def test_multilayer_tokens_empty_layers_raises() -> None:
    backbone = _create_mock_backbone()
    with pytest.raises(ValueError, match="non-empty"):
        backbone.extract_multilayer_tokens(torch.rand(1, 3, 224, 224), layers=())
