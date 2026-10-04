"""End-to-end integration test for the VITS train-to-detect pipeline.

Validates the full pipeline using synthetic data and lightweight models:
  synthetic tokens -> PatchTraj training -> scoring -> dual-signal fusion -> metrics.

All backbone extraction is bypassed with synthetic tensors so the test runs on
CPU without downloading model weights.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from src.data.base import create_sliding_windows, normalize_data, time_based_split
from src.evaluation.metrics import compute_all_metrics
from src.models.patchtraj import PatchTrajPredictor, SpatialTemporalPatchTrajPredictor
from src.scoring.dual_signal_scorer import DualSignalScorer
from src.scoring.patchtraj_scorer import (
    compute_patchtraj_residuals,
    compute_patchtraj_score,
    normalize_scores,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (small sizes for fast CPU tests)
# ---------------------------------------------------------------------------
NUM_PATCHES = 16
HIDDEN_DIM = 32
D_MODEL = 16
N_HEADS = 2
N_LAYERS = 1
K_PAST = 4  # number of past windows used for prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_timeseries(
    length: int = 100,
    features: int = 5,
    anomaly_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Create a synthetic multivariate time series with injected anomalies.

    Args:
        length: Number of timesteps.
        features: Number of features (dimensions).
        anomaly_ratio: Fraction of timesteps that are anomalous (placed at end).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (data, labels) where data has shape ``(length, features)``
        and labels has shape ``(length,)`` with 0=normal, 1=anomaly.
    """
    rng = np.random.RandomState(seed)
    data = rng.randn(length, features).astype(np.float64)
    labels = np.zeros(length, dtype=np.int64)

    num_anomalies = int(length * anomaly_ratio)
    anomaly_start = length - num_anomalies
    labels[anomaly_start:] = 1
    data[anomaly_start:] += 5.0  # shift anomalous region

    return data, labels


def _make_predictor(dropout: float = 0.0) -> PatchTrajPredictor:
    """Create a small PatchTrajPredictor for fast CPU tests.

    Args:
        dropout: Dropout rate (0 for deterministic tests).

    Returns:
        A PatchTrajPredictor on CPU.
    """
    return PatchTrajPredictor(
        hidden_dim=HIDDEN_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_feedforward=64,
        dropout=dropout,
        activation="gelu",
    ).to(torch.device("cpu"))


def _make_synthetic_tokens(
    rng: np.random.RandomState,
    n_windows: int,
    n_patches: int = NUM_PATCHES,
    hidden_dim: int = HIDDEN_DIM,
) -> npt.NDArray[np.float64]:
    """Create synthetic patch tokens mimicking backbone output.

    Args:
        rng: Numpy random state.
        n_windows: Number of windows (time steps).
        n_patches: Number of patches per window.
        hidden_dim: Hidden dimension per patch token.

    Returns:
        Token array of shape ``(n_windows, n_patches, hidden_dim)``.
    """
    return rng.randn(n_windows, n_patches, hidden_dim).astype(np.float64)


def _make_identity_correspondence(
    n_patches: int = NUM_PATCHES,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Create an identity correspondence map (each patch maps to itself).

    Args:
        n_patches: Number of patches.

    Returns:
        Tuple of (pi, valid_mask) where pi[i] = i for all valid patches.
    """
    pi = np.arange(n_patches, dtype=np.int64)
    valid_mask = np.ones(n_patches, dtype=bool)
    return pi, valid_mask


def _train_predictor(
    model: PatchTrajPredictor,
    train_tokens: torch.Tensor,
    n_epochs: int = 10,
) -> float:
    """Train the predictor on synthetic tokens and return final loss.

    Args:
        model: PatchTrajPredictor to train.
        train_tokens: Token tensor of shape ``(T, N, D)``.
        n_epochs: Number of training epochs.

    Returns:
        Final training loss value.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    n_windows = train_tokens.shape[0]
    final_loss = float("inf")

    for _epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for i in range(K_PAST, n_windows - 1):
            # Input: K past windows; Target: next window
            token_seq = train_tokens[i - K_PAST : i].unsqueeze(0)  # (1, K, N, D)
            target = train_tokens[i].unsqueeze(0)  # (1, N, D)

            optimizer.zero_grad()
            prediction = model(token_seq)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)

    return final_loss


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_e2e_training_produces_checkpoint() -> None:
    """Verify that training produces a checkpoint with expected keys."""
    torch.manual_seed(42)
    model = _make_predictor()

    rng = np.random.RandomState(0)
    tokens = torch.from_numpy(_make_synthetic_tokens(rng, n_windows=30)).float()

    final_loss = _train_predictor(model, tokens, n_epochs=5)
    assert np.isfinite(final_loss), "Final training loss must be finite."

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "hidden_dim": HIDDEN_DIM,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "final_loss": final_loss,
        }
        torch.save(checkpoint, ckpt_path)

        # Reload and verify
        loaded = torch.load(ckpt_path, weights_only=False)
        assert "model_state_dict" in loaded
        assert "hidden_dim" in loaded
        assert "d_model" in loaded
        assert "final_loss" in loaded
        assert loaded["hidden_dim"] == HIDDEN_DIM
        assert loaded["d_model"] == D_MODEL

        # Verify model can be restored from checkpoint
        restored = _make_predictor()
        restored.load_state_dict(loaded["model_state_dict"])
        restored.eval()

        test_input = torch.randn(1, K_PAST, NUM_PATCHES, HIDDEN_DIM)
        with torch.no_grad():
            out_original = model.eval()(test_input)
            out_restored = restored(test_input)
        torch.testing.assert_close(out_original, out_restored)


@pytest.mark.slow
def test_e2e_detection_produces_artifacts() -> None:
    """Verify that detection produces scores.npy, labels.npy, and metrics.json."""
    torch.manual_seed(42)

    # --- Prepare synthetic data ---
    data, labels = _make_synthetic_timeseries(length=100, features=5, seed=0)
    train_data, train_labels, test_data, test_labels = time_based_split(
        data, labels, train_ratio=0.5
    )
    norm_train, norm_test = normalize_data(train_data, test_data, method="standard")

    window_size = 10
    stride = 2
    train_windows, _ = create_sliding_windows(
        norm_train,
        np.zeros(norm_train.shape[0], dtype=np.int64),
        window_size=window_size,
        stride=stride,
    )
    test_windows, test_wlabels = create_sliding_windows(
        norm_test,
        test_labels,
        window_size=window_size,
        stride=stride,
    )

    # --- Simulate backbone extraction (synthetic tokens) ---
    rng = np.random.RandomState(7)
    train_tokens = torch.from_numpy(
        _make_synthetic_tokens(rng, n_windows=train_windows.shape[0])
    ).float()
    test_tokens_np = _make_synthetic_tokens(rng, n_windows=test_windows.shape[0])
    # Make anomalous windows have shifted tokens
    anomaly_mask = test_wlabels == 1
    test_tokens_np[anomaly_mask] += 3.0
    test_tokens = torch.from_numpy(test_tokens_np).float()

    # --- Train model ---
    model = _make_predictor()
    _train_predictor(model, train_tokens, n_epochs=5)
    model.eval()

    # --- Run detection ---
    pi, valid_mask = _make_identity_correspondence()
    n_test = test_tokens.shape[0]
    scores_list: list[float] = []

    with torch.no_grad():
        for i in range(K_PAST, n_test):
            token_seq = test_tokens[i - K_PAST : i].unsqueeze(0)  # (1, K, N, D)
            actual = test_tokens[i].unsqueeze(0)  # (1, N, D)
            predicted = model(token_seq)
            score = compute_patchtraj_score(predicted, actual, pi, valid_mask)
            scores_list.append(score.item())

    scores_arr = np.array(scores_list, dtype=np.float64)
    labels_arr = test_wlabels[K_PAST:].astype(np.int64)

    # Truncate to common length
    min_len = min(len(scores_arr), len(labels_arr))
    scores_arr = scores_arr[:min_len]
    labels_arr = labels_arr[:min_len]

    # --- Save artifacts ---
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)

        np.save(outdir / "scores.npy", scores_arr)
        np.save(outdir / "labels.npy", labels_arr)

        # Compute metrics (only if both classes present)
        if np.any(labels_arr == 0) and np.any(labels_arr == 1):
            metrics = compute_all_metrics(scores=scores_arr, labels=labels_arr)
        else:
            metrics = {"auc_roc": 0.0, "auc_pr": 0.0, "best_f1": 0.0, "f1_pa": 0.0}

        with open(outdir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # --- Verify artifacts exist ---
        assert (outdir / "scores.npy").exists(), "scores.npy must be produced."
        assert (outdir / "labels.npy").exists(), "labels.npy must be produced."
        assert (outdir / "metrics.json").exists(), "metrics.json must be produced."

        loaded_scores = np.load(outdir / "scores.npy")
        loaded_labels = np.load(outdir / "labels.npy")
        assert loaded_scores.shape == scores_arr.shape
        assert loaded_labels.shape == labels_arr.shape
        np.testing.assert_array_equal(loaded_scores, scores_arr)
        np.testing.assert_array_equal(loaded_labels, labels_arr)

        with open(outdir / "metrics.json") as f:
            loaded_metrics = json.load(f)

        expected_keys = {"auc_roc", "auc_pr", "best_f1", "f1_pa"}
        assert expected_keys.issubset(
            loaded_metrics.keys()
        ), f"metrics.json missing keys: {expected_keys - loaded_metrics.keys()}"

        for key in expected_keys:
            assert isinstance(loaded_metrics[key], float), (
                f"metrics[{key!r}] must be float."
            )


@pytest.mark.slow
def test_e2e_dual_signal_produces_separate_scores() -> None:
    """Verify dual-signal scoring saves traj_scores.npy and dist_scores.npy."""
    torch.manual_seed(42)

    rng = np.random.RandomState(99)
    n_train = 40
    n_test = 20

    train_tokens_np = _make_synthetic_tokens(rng, n_windows=n_train)
    test_tokens_np = _make_synthetic_tokens(rng, n_windows=n_test)
    # Shift last 5 test windows to simulate anomalies
    test_tokens_np[-5:] += 5.0

    train_tokens = torch.from_numpy(train_tokens_np).float()
    test_tokens = torch.from_numpy(test_tokens_np).float()

    # --- Train PatchTraj model ---
    model = _make_predictor()
    _train_predictor(model, train_tokens, n_epochs=5)
    model.eval()

    # --- Compute trajectory scores ---
    pi, valid_mask = _make_identity_correspondence()
    traj_scores_list: list[float] = []
    with torch.no_grad():
        for i in range(K_PAST, n_test):
            token_seq = test_tokens[i - K_PAST : i].unsqueeze(0)
            actual = test_tokens[i].unsqueeze(0)
            predicted = model(token_seq)
            score = compute_patchtraj_score(predicted, actual, pi, valid_mask)
            traj_scores_list.append(score.item())

    traj_scores = np.array(traj_scores_list, dtype=np.float64)

    # --- Compute distributional scores ---
    dual_scorer = DualSignalScorer(alpha=0.5)
    dual_scorer.fit(train_tokens_np)
    test_slice = test_tokens_np[K_PAST:]
    dist_scores = dual_scorer.score_distributional(test_slice)

    # Align lengths
    min_len = min(len(traj_scores), len(dist_scores))
    traj_scores = traj_scores[:min_len]
    dist_scores = dist_scores[:min_len]

    # --- Fuse ---
    fused_scores = dual_scorer.fuse(traj_scores, dist_scores)

    # --- Save artifacts ---
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)

        np.save(outdir / "traj_scores.npy", traj_scores)
        np.save(outdir / "dist_scores.npy", dist_scores)
        np.save(outdir / "scores.npy", fused_scores)

        # Create labels for the scored windows
        labels = np.zeros(min_len, dtype=np.int64)
        # The last few windows should be anomalous
        n_anom = min(5, min_len)
        labels[-n_anom:] = 1

        if np.any(labels == 0) and np.any(labels == 1):
            metrics = compute_all_metrics(scores=fused_scores, labels=labels)
            with open(outdir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # --- Verify dual-signal artifacts ---
        assert (outdir / "traj_scores.npy").exists(), "traj_scores.npy must be produced."
        assert (outdir / "dist_scores.npy").exists(), "dist_scores.npy must be produced."
        assert (outdir / "scores.npy").exists(), "scores.npy (fused) must be produced."

        loaded_traj = np.load(outdir / "traj_scores.npy")
        loaded_dist = np.load(outdir / "dist_scores.npy")
        assert loaded_traj.shape == traj_scores.shape
        assert loaded_dist.shape == dist_scores.shape
        assert loaded_traj.dtype == np.float64
        assert loaded_dist.dtype == np.float64

        # Distributional scores should be non-negative (Mahalanobis distance)
        assert np.all(loaded_dist >= 0), "Distributional scores must be non-negative."

        # Fused scores should be finite
        loaded_fused = np.load(outdir / "scores.npy")
        assert np.all(np.isfinite(loaded_fused)), "Fused scores must be finite."


@pytest.mark.slow
def test_e2e_metrics_json_has_expected_fields() -> None:
    """Verify metrics.json contains auc_roc, auc_pr, best_f1, f1_pa with valid values."""
    rng = np.random.RandomState(123)

    # Create scores where anomalies have higher values (for meaningful metrics)
    n_total = 100
    labels = np.zeros(n_total, dtype=np.int64)
    labels[80:] = 1  # last 20% anomalous

    scores = rng.randn(n_total).astype(np.float64)
    scores[labels == 1] += 3.0  # shift anomalies up

    metrics = compute_all_metrics(scores=scores, labels=labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = Path(tmpdir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        with open(metrics_path) as f:
            loaded = json.load(f)

    # Check all expected fields present
    for key in ("auc_roc", "auc_pr", "best_f1", "f1_pa"):
        assert key in loaded, f"Missing key {key!r} in metrics.json."
        val = loaded[key]
        assert isinstance(val, float), f"metrics[{key!r}] must be float, got {type(val)}."
        assert 0.0 <= val <= 1.0, f"metrics[{key!r}]={val} must be in [0, 1]."

    # With shifted anomalies, AUC-ROC should be well above random
    assert loaded["auc_roc"] > 0.7, (
        f"AUC-ROC should be well above 0.5 for separable data, got {loaded['auc_roc']:.3f}."
    )


@pytest.mark.slow
def test_e2e_scoring_pipeline_end_to_end() -> None:
    """Test the scoring pipeline: PatchTraj forward, residuals, score, normalization."""
    torch.manual_seed(7)
    model = _make_predictor(dropout=0.0)

    rng = np.random.RandomState(42)
    tokens = torch.from_numpy(_make_synthetic_tokens(rng, n_windows=20)).float()

    # Quick training to get non-random predictions
    _train_predictor(model, tokens, n_epochs=10)
    model.eval()

    pi, valid_mask = _make_identity_correspondence()

    # Compute predictions and scores for multiple windows
    all_scores: list[float] = []
    all_residuals: list[npt.NDArray[np.float64]] = []

    with torch.no_grad():
        for i in range(K_PAST, tokens.shape[0]):
            token_seq = tokens[i - K_PAST : i].unsqueeze(0)
            actual = tokens[i].unsqueeze(0)
            predicted = model(token_seq)

            # Score
            score = compute_patchtraj_score(predicted, actual, pi, valid_mask)
            all_scores.append(score.item())

            # Residuals
            residuals = compute_patchtraj_residuals(predicted, actual, pi, valid_mask)
            all_residuals.append(residuals.numpy().flatten())

    scores_arr = np.array(all_scores, dtype=np.float64)

    # All scores must be finite and non-negative
    assert np.all(np.isfinite(scores_arr)), "All scores must be finite."
    assert np.all(scores_arr >= 0), "All scores must be non-negative."

    # Normalize scores
    normed = normalize_scores(scores_arr, method="minmax")
    assert normed.shape == scores_arr.shape
    assert float(np.min(normed)) >= 0.0
    assert float(np.max(normed)) <= 1.0

    # Residuals should have correct shape
    for resid in all_residuals:
        assert resid.size == NUM_PATCHES, (
            f"Expected {NUM_PATCHES} residuals per window, got {resid.size}."
        )


@pytest.mark.slow
def test_e2e_spatial_temporal_predictor_in_pipeline() -> None:
    """Verify SpatialTemporalPatchTrajPredictor works in the scoring pipeline."""
    torch.manual_seed(42)

    model = SpatialTemporalPatchTrajPredictor(
        hidden_dim=HIDDEN_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_feedforward=64,
        dropout=0.0,
        patch_grid=(4, 4),  # 4x4 = 16 patches = NUM_PATCHES
    ).to(torch.device("cpu"))

    rng = np.random.RandomState(0)
    tokens = torch.from_numpy(_make_synthetic_tokens(rng, n_windows=20)).float()

    # Train briefly
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for i in range(K_PAST, tokens.shape[0] - 1):
        token_seq = tokens[i - K_PAST : i].unsqueeze(0)
        target = tokens[i].unsqueeze(0)
        optimizer.zero_grad()
        pred = model(token_seq)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

    # Score test windows
    model.eval()
    pi, valid_mask = _make_identity_correspondence()
    with torch.no_grad():
        token_seq = tokens[-K_PAST:].unsqueeze(0)
        actual = tokens[-1].unsqueeze(0)
        predicted = model(token_seq)
        score = compute_patchtraj_score(predicted, actual, pi, valid_mask)

    assert score.shape == (1,)
    assert np.isfinite(score.item())
    assert score.item() >= 0.0
