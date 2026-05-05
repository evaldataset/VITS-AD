"""End-to-end smoke test: synthetic data → window → render → backbone → patch tokens.

Validates the full Phase 1 pipeline without requiring real datasets or model downloads.
The backbone is mocked to return deterministic fake patch tokens so we can verify
the π correspondence logic end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.data.base import create_sliding_windows, normalize_data, time_based_split
from src.models.backbone import VisionBackbone
from src.rendering.line_plot import render_line_plot, render_line_plot_batch
from src.rendering.token_correspondence import (
    compute_correspondence_map,
    get_valid_patch_count,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_PATCHES = 256  # DINOv2 with patch_size=14: (224/14)^2 = 16^2
HIDDEN_DIM = 768
GRID_H, GRID_W = 16, 16


def _create_synthetic_timeseries(
    length: int = 500,
    features: int = 10,
    anomaly_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic multivariate time series with injected anomalies."""
    rng = np.random.RandomState(seed)
    data = rng.randn(length, features).astype(np.float64)
    labels = np.zeros(length, dtype=np.int64)

    # Inject anomalies in the last portion
    num_anomalies = int(length * anomaly_ratio)
    anomaly_start = length - num_anomalies
    labels[anomaly_start:] = 1
    # Make anomaly region have different distribution
    data[anomaly_start:] += 5.0

    return data, labels


def _create_mock_backbone_with_deterministic_output() -> VisionBackbone:
    """Create a mocked VisionBackbone that returns deterministic patch tokens.

    The mock backbone returns patch tokens where each token is the image pixel
    mean at that patch location, ensuring that shifted images produce shifted tokens.
    """
    with patch.object(VisionBackbone, "_load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = HIDDEN_DIM
        mock_model.config.image_size = 224
        mock_model.config.patch_size = 14

        # Make forward pass return deterministic tokens based on input
        def _forward(pixel_values: torch.Tensor, **kwargs: object) -> MagicMock:
            B = pixel_values.shape[0]
            # Create tokens from spatial average of each patch region
            patch_size = 14
            tokens_list = []
            for b in range(B):
                img = pixel_values[b]  # (3, 224, 224)
                patches = []
                for row in range(GRID_H):
                    for col in range(GRID_W):
                        r0 = row * patch_size
                        c0 = col * patch_size
                        patch_region = img[
                            :, r0 : r0 + patch_size, c0 : c0 + patch_size
                        ]
                        # Use mean of patch region as a simple embedding
                        mean_val = patch_region.mean().item()
                        token = torch.full((HIDDEN_DIM,), mean_val)
                        patches.append(token)
                # Stack patches + prepend a dummy CLS token
                cls_token = torch.zeros(HIDDEN_DIM)
                all_tokens = torch.stack([cls_token] + patches)  # (257, 768)
                tokens_list.append(all_tokens)

            result = MagicMock()
            result.last_hidden_state = torch.stack(tokens_list)  # (B, 257, 768)
            return result

        mock_model.side_effect = _forward
        mock_model.__call__ = _forward
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
# E2E Pipeline: data → split → normalize → window → render → tokens
# ---------------------------------------------------------------------------


def test_e2e_data_to_windows() -> None:
    """Validate: synthetic data → split → normalize → sliding windows."""
    data, labels = _create_synthetic_timeseries(length=500, features=10)

    # Time-based split
    train_data, train_labels, test_data, test_labels = time_based_split(
        data, labels, train_ratio=0.5
    )
    assert train_data.shape[1] == 10
    assert test_data.shape[0] == 250
    assert np.all(train_labels == 0)  # train is normal-only

    # Normalize (train stats only)
    norm_train, norm_test = normalize_data(train_data, test_data, method="standard")
    np.testing.assert_allclose(np.mean(norm_train, axis=0), 0.0, atol=1e-10)

    # Sliding windows
    train_windows, train_wlabels = create_sliding_windows(
        norm_train,
        np.zeros(norm_train.shape[0], dtype=np.int64),
        window_size=100,
        stride=10,
    )
    test_windows, test_wlabels = create_sliding_windows(
        norm_test,
        test_labels,
        window_size=100,
        stride=10,
    )

    assert train_windows.ndim == 3
    assert train_windows.shape[1] == 100
    assert train_windows.shape[2] == 10
    assert test_windows.ndim == 3

    # Test windows should contain some anomalies
    assert np.sum(test_wlabels) > 0
    # Train windows should be all normal
    assert np.all(train_wlabels == 0)


def test_e2e_windows_to_images() -> None:
    """Validate: windows → rendered images (correct shape, deterministic)."""
    data, labels = _create_synthetic_timeseries(length=200, features=5)
    windows, _ = create_sliding_windows(data, labels, window_size=50, stride=10)

    # Render first 3 windows
    batch = render_line_plot_batch(windows[:3])
    assert batch.shape == (3, 3, 224, 224)
    assert batch.dtype == np.float32
    assert float(np.min(batch)) >= 0.0
    assert float(np.max(batch)) <= 1.0

    # Determinism
    batch2 = render_line_plot_batch(windows[:3])
    np.testing.assert_array_equal(batch, batch2)


def test_e2e_images_to_patch_tokens() -> None:
    """Validate: rendered images → backbone → patch tokens (shape and determinism)."""
    backbone = _create_mock_backbone_with_deterministic_output()

    # Create and render a window
    rng = np.random.RandomState(0)
    window = rng.randn(50, 5).astype(np.float64)
    image = render_line_plot(window)  # (3, 224, 224)

    # Extract patch tokens
    tokens = backbone.extract_patch_tokens_from_numpy(image.astype(np.float32))
    assert tokens.shape == (1, NUM_PATCHES, HIDDEN_DIM)

    # Determinism: same image → same tokens
    tokens2 = backbone.extract_patch_tokens_from_numpy(image.astype(np.float32))
    np.testing.assert_array_equal(tokens, tokens2)


def test_e2e_pi_correspondence_with_tokens() -> None:
    """Validate: π map correctly relates patch tokens across consecutive windows.

    For line_plot with window=100, stride=10, grid=(16,16):
      delta_col = round(16 * 10 / 100) = round(1.6) = 2
      Patch at (row, col) in window_t should correspond to (row, col-2) in window_{t+1}
    """
    pi, valid_mask = compute_correspondence_map(
        "line_plot",
        window_size=100,
        stride=10,
        patch_grid=(GRID_H, GRID_W),
    )

    assert pi.shape == (NUM_PATCHES,)
    assert valid_mask.shape == (NUM_PATCHES,)

    valid_count = get_valid_patch_count(valid_mask)
    delta_col = round(GRID_W * 10 / 100)  # = 2
    expected_valid = GRID_H * (GRID_W - delta_col)
    assert valid_count == expected_valid

    # Verify specific mappings
    # Patch (5, 7) → (5, 5) since delta_col=2
    src_idx = 5 * GRID_W + 7
    expected_dst = 5 * GRID_W + 5
    assert valid_mask[src_idx]
    assert pi[src_idx] == expected_dst

    # Patch (3, 0) should be invalid (new_col = -2)
    src_idx_invalid = 3 * GRID_W + 0
    assert not valid_mask[src_idx_invalid]
    assert pi[src_idx_invalid] == -1


def test_e2e_full_pipeline_synthetic() -> None:
    """Full pipeline smoke test: synthetic data → score-ready patch tokens.

    This validates the complete flow that PatchTraj will consume:
    1. Generate synthetic time series
    2. Split and normalize
    3. Create sliding windows
    4. Render windows to images
    5. Extract patch tokens via backbone
    6. Compute π map
    7. Use π to index tokens across consecutive windows
    """
    # Step 1: Synthetic data
    data, labels = _create_synthetic_timeseries(
        length=300,
        features=8,
        anomaly_ratio=0.1,
        seed=123,
    )

    # Step 2: Split and normalize
    train_data, _, test_data, test_labels = time_based_split(
        data,
        labels,
        train_ratio=0.5,
    )
    norm_train, norm_test = normalize_data(train_data, test_data, method="standard")

    # Step 3: Sliding windows
    window_size = 50
    stride = 5
    train_windows, _ = create_sliding_windows(
        norm_train,
        np.zeros(norm_train.shape[0], dtype=np.int64),
        window_size=window_size,
        stride=stride,
    )
    assert train_windows.shape[0] > 2, "Need at least 2 windows for π"

    # Step 4: Render consecutive windows
    w_t = train_windows[0]
    w_t1 = train_windows[1]
    img_t = render_line_plot(w_t)
    img_t1 = render_line_plot(w_t1)

    assert img_t.shape == (3, 224, 224)
    assert img_t1.shape == (3, 224, 224)
    assert not np.array_equal(img_t, img_t1), "Consecutive windows should differ"

    # Step 5: Extract patch tokens (mocked backbone)
    backbone = _create_mock_backbone_with_deterministic_output()
    batch = np.stack([img_t, img_t1]).astype(np.float32)
    tokens = backbone.extract_patch_tokens_from_numpy(batch)
    assert tokens.shape == (2, NUM_PATCHES, HIDDEN_DIM)

    P_t = tokens[0]  # (256, 768)
    P_t1 = tokens[1]  # (256, 768)

    # Step 6: Compute π
    pi, valid_mask = compute_correspondence_map(
        "line_plot",
        window_size=window_size,
        stride=stride,
        patch_grid=(GRID_H, GRID_W),
    )
    valid_count = get_valid_patch_count(valid_mask)
    assert valid_count > 0, "Must have valid correspondences"

    # Step 7: Use π to index — this is what PatchTraj will compare
    # P_{t+1, π(i)} should be the "expected" location of patch i from window t
    valid_indices = np.where(valid_mask)[0]
    target_indices = pi[valid_indices]

    # Index into P_{t+1} using π
    P_t1_aligned = P_t1[target_indices]  # (valid_count, 768)
    P_t_valid = P_t[valid_indices]  # (valid_count, 768)

    assert P_t1_aligned.shape == (valid_count, HIDDEN_DIM)
    assert P_t_valid.shape == (valid_count, HIDDEN_DIM)

    # Compute per-patch prediction error (what PatchTraj score will be)
    residuals = np.sum((P_t1_aligned - P_t_valid) ** 2, axis=1)  # (valid_count,)
    anomaly_score = float(np.mean(residuals))

    # Score should be finite and non-negative
    assert np.isfinite(anomaly_score)
    assert anomaly_score >= 0.0


def test_e2e_multi_window_token_sequence() -> None:
    """Validate K consecutive windows produce a proper token sequence for PatchTraj.

    PatchTraj needs K=8 past windows' patch tokens as input:
      [P_{t-7Δ}, ..., P_{t}] each ∈ R^{num_patches × hidden_dim}
    """
    K = 8
    data, labels = _create_synthetic_timeseries(length=500, features=5, seed=99)
    windows, _ = create_sliding_windows(data, labels, window_size=50, stride=5)

    assert windows.shape[0] >= K, f"Need at least {K} windows, got {windows.shape[0]}"

    # Render K consecutive windows
    images = render_line_plot_batch(windows[:K])
    assert images.shape == (K, 3, 224, 224)

    # Extract patch tokens (mocked)
    backbone = _create_mock_backbone_with_deterministic_output()
    tokens = backbone.extract_patch_tokens_from_numpy(images.astype(np.float32))
    assert tokens.shape == (K, NUM_PATCHES, HIDDEN_DIM)

    # Verify each window produces different tokens
    for i in range(K - 1):
        assert not np.array_equal(tokens[i], tokens[i + 1]), (
            f"Windows {i} and {i + 1} should produce different tokens"
        )

    # This token_seq is exactly what PatchTrajPredictor.forward() will receive
    token_seq = tokens  # (K, 256, 768)
    assert token_seq.shape == (K, NUM_PATCHES, HIDDEN_DIM)
