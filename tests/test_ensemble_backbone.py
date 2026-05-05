from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.models.ensemble_backbone import EnsembleBackbone


class _FakeVisionBackbone:
    def __init__(
        self,
        model_name: str,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        del device, dtype
        self.model_name = model_name
        if "clip" in model_name:
            self.hidden_dim = 768
            self.num_patches = 196
            self.patch_grid = (14, 14)
            self._token_value = 1.0
        else:
            self.hidden_dim = 768
            self.num_patches = 256
            self.patch_grid = (16, 16)
            self._token_value = 2.0

    @torch.no_grad()
    def extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        batch = int(images.shape[0])
        return torch.full(
            (batch, self.num_patches, self.hidden_dim),
            fill_value=self._token_value,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def extract_with_attention(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        tokens = self.extract_patch_tokens(images)
        seq_len = self.num_patches + 1
        attention = torch.zeros(
            (tokens.shape[0], 12, seq_len, seq_len),
            dtype=torch.float32,
        )
        return tokens, (attention,)


def _make_images(batch: int = 2) -> torch.Tensor:
    return torch.rand((batch, 3, 224, 224), dtype=torch.float32)


def test_selected_mode_matches_visionbackbone_interface() -> None:
    with patch("src.models.ensemble_backbone.VisionBackbone", _FakeVisionBackbone):
        backbone = EnsembleBackbone(output_mode="selected", default_backbone="clip")
        tokens = backbone.extract_patch_tokens(_make_images())
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, 196, 768)
        assert backbone.hidden_dim == 768
        assert backbone.num_patches == 196
        assert backbone.patch_grid == (14, 14)


def test_renderer_mapping_switches_active_backbone() -> None:
    with patch("src.models.ensemble_backbone.VisionBackbone", _FakeVisionBackbone):
        backbone = EnsembleBackbone(output_mode="selected", default_backbone="clip")
        clip_tokens = backbone.extract_patch_tokens(_make_images())
        backbone.set_renderer("recurrence_plot")
        dino_tokens = backbone.extract_patch_tokens(_make_images())
        assert isinstance(clip_tokens, torch.Tensor)
        assert isinstance(dino_tokens, torch.Tensor)
        assert clip_tokens.shape == (2, 196, 768)
        assert dino_tokens.shape == (2, 256, 768)
        assert backbone.num_patches == 256
        assert backbone.patch_grid == (16, 16)


def test_dict_mode_returns_dual_tokens() -> None:
    with patch("src.models.ensemble_backbone.VisionBackbone", _FakeVisionBackbone):
        backbone = EnsembleBackbone(output_mode="dict", default_backbone="clip")
        tokens = backbone.extract_patch_tokens(_make_images())
        assert isinstance(tokens, dict)
        assert set(tokens.keys()) == {"clip", "dinov2"}
        assert tokens["clip"].shape == (2, 196, 768)
        assert tokens["dinov2"].shape == (2, 256, 768)


def test_concat_mode_raises_on_patch_count_mismatch() -> None:
    with patch("src.models.ensemble_backbone.VisionBackbone", _FakeVisionBackbone):
        backbone = EnsembleBackbone(output_mode="concat", default_backbone="clip")
        with pytest.raises(RuntimeError, match="different patch counts"):
            backbone.extract_patch_tokens(_make_images())


def test_deterministic_extraction_outputs_identical_tokens() -> None:
    with patch("src.models.ensemble_backbone.VisionBackbone", _FakeVisionBackbone):
        backbone = EnsembleBackbone(output_mode="dict", default_backbone="clip")
        images = _make_images(batch=1)
        first = backbone.extract_patch_tokens(images)
        second = backbone.extract_patch_tokens(images)
        assert isinstance(first, dict)
        assert isinstance(second, dict)
        np.testing.assert_array_equal(
            first["clip"].detach().cpu().numpy(),
            second["clip"].detach().cpu().numpy(),
        )
        np.testing.assert_array_equal(
            first["dinov2"].detach().cpu().numpy(),
            second["dinov2"].detach().cpu().numpy(),
        )


def test_sequential_mode_releases_other_backbone_for_memory() -> None:
    with patch("src.models.ensemble_backbone.VisionBackbone", _FakeVisionBackbone):
        backbone = EnsembleBackbone(
            output_mode="dict",
            sequential=True,
            load_both=False,
            default_backbone="clip",
        )
        _ = backbone.extract_patch_tokens(_make_images(batch=1))
        loaded_backbones = {
            key for key, value in backbone._backbones.items() if value is not None
        }
        assert loaded_backbones == {"dinov2"}


def test_numpy_extraction_in_selected_mode() -> None:
    with patch("src.models.ensemble_backbone.VisionBackbone", _FakeVisionBackbone):
        backbone = EnsembleBackbone(output_mode="selected", default_backbone="clip")
        images = np.random.RandomState(0).rand(1, 3, 224, 224).astype(np.float32)
        tokens = backbone.extract_patch_tokens_from_numpy(images)
        assert isinstance(tokens, np.ndarray)
        assert tokens.shape == (1, 196, 768)


def test_extract_with_attention_requires_selected_mode() -> None:
    with patch("src.models.ensemble_backbone.VisionBackbone", _FakeVisionBackbone):
        backbone = EnsembleBackbone(output_mode="dict", default_backbone="clip")
        with pytest.raises(RuntimeError, match="only available"):
            backbone.extract_with_attention(_make_images(batch=1))
