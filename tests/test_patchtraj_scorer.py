from __future__ import annotations

import importlib

import numpy as np
import numpy.typing as npt
import pytest
import torch

_PATCHTRAJ_SCORER = importlib.import_module("src.scoring.patchtraj_scorer")
compute_patchtraj_residuals = _PATCHTRAJ_SCORER.compute_patchtraj_residuals
compute_patchtraj_score = _PATCHTRAJ_SCORER.compute_patchtraj_score
normalize_scores = _PATCHTRAJ_SCORER.normalize_scores
trimmed_huber_loss = _PATCHTRAJ_SCORER.trimmed_huber_loss


def test_compute_patchtraj_score_identity_known_value_and_shape() -> None:
    predicted = torch.zeros((2, 4, 8), dtype=torch.float32)
    actual = torch.ones((2, 4, 8), dtype=torch.float32)
    pi = np.array([0, 1, 2, 3], dtype=np.int64)
    valid_mask = np.array([True, True, True, True], dtype=bool)

    scores = compute_patchtraj_score(predicted, actual, pi, valid_mask)

    assert scores.shape == (2,)
    assert scores[0].item() == pytest.approx(8.0)
    assert scores[1].item() == pytest.approx(8.0)


def test_compute_patchtraj_residuals_shape_and_values_manual() -> None:
    predicted = torch.tensor(
        [
            [[0.0, 0.0], [5.0, 5.0], [0.0, 0.0]],
            [[1.0, 1.0], [9.0, 9.0], [2.0, 2.0]],
        ]
    )
    actual = torch.tensor(
        [
            [[1.0, 1.0], [5.0, 5.0], [1.0, 1.0]],
            [[2.0, 2.0], [9.0, 9.0], [4.0, 4.0]],
        ]
    )
    pi = np.array([0, 1, 2], dtype=np.int64)
    valid_mask = np.array([True, False, True], dtype=bool)

    residuals = compute_patchtraj_residuals(predicted, actual, pi, valid_mask)

    assert residuals.shape == (2, 2)
    assert residuals[0, 0].item() == pytest.approx(2.0)
    assert residuals[0, 1].item() == pytest.approx(2.0)
    assert residuals[1, 0].item() == pytest.approx(2.0)
    assert residuals[1, 1].item() == pytest.approx(8.0)


def test_trimmed_huber_loss_delta_behavior_quadratic_and_linear() -> None:
    residuals = torch.tensor([[0.25, 4.0]], dtype=torch.float32)
    loss = trimmed_huber_loss(residuals, delta=1.0, trim_ratio=0.0)

    expected = (0.125 + 1.5) / 2.0
    assert loss.item() == pytest.approx(expected)


def test_trimmed_huber_loss_trim_ratio_zero_equals_full_mean() -> None:
    residuals = torch.tensor([[1.0, 4.0, 9.0]], dtype=torch.float32)
    loss = trimmed_huber_loss(residuals, delta=1.0, trim_ratio=0.0)

    assert loss.item() == pytest.approx((0.5 + 1.5 + 2.5) / 3.0)


def test_trimmed_huber_loss_trim_ratio_half_removes_top_half() -> None:
    residuals = torch.tensor([[1.0, 4.0, 9.0, 100.0]], dtype=torch.float32)
    loss = trimmed_huber_loss(residuals, delta=1.0, trim_ratio=0.5)

    assert loss.item() == pytest.approx((0.5 + 1.5) / 2.0)


def test_trimmed_huber_loss_returns_scalar_tensor() -> None:
    residuals = torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float32)
    loss = trimmed_huber_loss(residuals)
    assert loss.ndim == 0


def test_normalize_scores_minmax_known_values() -> None:
    scores = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    normalized = normalize_scores(scores, method="minmax")

    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] == pytest.approx(1.0)


def test_normalize_scores_zscore_outputs_0_to_1() -> None:
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    normalized = normalize_scores(scores, method="zscore")

    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] == pytest.approx(1.0)


def test_normalize_scores_constant_input_returns_zeros_for_both_methods() -> None:
    scores = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    minmax = normalize_scores(scores, method="minmax")
    zscore = normalize_scores(scores, method="zscore")

    assert np.allclose(minmax, np.zeros_like(scores))
    assert np.allclose(zscore, np.zeros_like(scores))


def test_normalize_scores_invalid_method_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        normalize_scores(np.array([1.0, 2.0], dtype=np.float64), method="median")


def test_compute_patchtraj_score_wrong_type_for_pi_raises_attribute_error() -> None:
    predicted = torch.zeros((1, 2, 2), dtype=torch.float32)
    actual = torch.zeros((1, 2, 2), dtype=torch.float32)
    valid_mask = np.array([True, True], dtype=bool)

    with pytest.raises(AttributeError):
        compute_patchtraj_score(predicted, actual, pi=[0, 1], valid_mask=valid_mask)


def test_compute_patchtraj_score_wrong_shape_raises_value_error() -> None:
    predicted = torch.zeros((2, 8), dtype=torch.float32)
    actual = torch.zeros((2, 8), dtype=torch.float32)
    pi = np.array([0, 1], dtype=np.int64)
    valid_mask = np.array([True, True], dtype=bool)

    with pytest.raises(ValueError, match="shape"):
        compute_patchtraj_score(predicted, actual, pi, valid_mask)


def test_compute_patchtraj_score_mismatched_shapes_raises_value_error() -> None:
    predicted = torch.zeros((1, 3, 2), dtype=torch.float32)
    actual = torch.zeros((1, 4, 2), dtype=torch.float32)
    pi = np.array([0, 1, 2], dtype=np.int64)
    valid_mask = np.array([True, True, True], dtype=bool)

    with pytest.raises(ValueError, match="identical shapes"):
        compute_patchtraj_score(predicted, actual, pi, valid_mask)


def test_compute_patchtraj_score_empty_tensors_raises_value_error() -> None:
    predicted = torch.zeros((0, 3, 2), dtype=torch.float32)
    actual = torch.zeros((0, 3, 2), dtype=torch.float32)
    pi = np.array([0, 1, 2], dtype=np.int64)
    valid_mask = np.array([True, True, True], dtype=bool)

    with pytest.raises(ValueError, match="B > 0"):
        compute_patchtraj_score(predicted, actual, pi, valid_mask)


def test_compute_patchtraj_score_invalid_pi_indices_raise_value_error() -> None:
    predicted = torch.zeros((1, 3, 2), dtype=torch.float32)
    actual = torch.zeros((1, 3, 2), dtype=torch.float32)
    valid_mask = np.array([True, True, True], dtype=bool)

    invalid_pi_values: tuple[npt.NDArray[np.int64], ...] = (
        np.array([-1, 0, 1], dtype=np.int64),
        np.array([0, 1, 3], dtype=np.int64),
    )
    for pi in invalid_pi_values:
        with pytest.raises(ValueError, match="pi contains"):
            compute_patchtraj_score(predicted, actual, pi, valid_mask)


def test_compute_patchtraj_score_all_false_valid_mask_raises_value_error() -> None:
    predicted = torch.zeros((1, 3, 2), dtype=torch.float32)
    actual = torch.zeros((1, 3, 2), dtype=torch.float32)
    pi = np.array([0, 1, 2], dtype=np.int64)
    valid_mask = np.array([False, False, False], dtype=bool)

    with pytest.raises(ValueError, match="no valid correspondences"):
        compute_patchtraj_score(predicted, actual, pi, valid_mask)


def test_trimmed_huber_loss_invalid_residual_values_raise_value_error() -> None:
    invalid_residuals = (
        torch.tensor([[-1.0, 1.0]], dtype=torch.float32),
        torch.tensor([[1.0, float("nan")]], dtype=torch.float32),
    )
    for residuals in invalid_residuals:
        with pytest.raises(ValueError):
            trimmed_huber_loss(residuals)
