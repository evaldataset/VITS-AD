from __future__ import annotations

# pyright: reportMissingImports=false

import numpy as np
import pytest
from scipy.stats import rankdata

from src.scoring.score_fusion import fuse_scores, fuse_scores_confidence_weighted


def test_fuse_scores_weighted_sum_matches_manual_values() -> None:
    scores_dict = {
        "lp": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "rp": np.array([3.0, 2.0, 1.0], dtype=np.float64),
    }

    fused = fuse_scores(scores_dict, method="weighted_sum", weights={"lp": 0.75, "rp": 0.25})

    expected = np.array([1.5, 2.0, 2.5], dtype=np.float64)
    assert fused.shape == (3,)
    assert fused.dtype == np.float64
    np.testing.assert_allclose(fused, expected)


def test_fuse_scores_rank_fusion_matches_weighted_rank_average() -> None:
    scores_dict = {
        "lp": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        "rp": np.array([3.0, 1.0, 2.0], dtype=np.float64),
    }

    fused = fuse_scores(scores_dict, method="rank_fusion", weights=[0.25, 0.75])

    expected = (
        0.25 * rankdata(scores_dict["lp"]) / 3.0
        + 0.75 * rankdata(scores_dict["rp"]) / 3.0
    )
    np.testing.assert_allclose(fused, expected.astype(np.float64))


def test_fuse_scores_zscore_weighted_matches_manual_values() -> None:
    scores_dict = {
        "lp": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "rp": np.array([1.0, 3.0, 5.0], dtype=np.float64),
    }

    fused = fuse_scores(
        scores_dict,
        method="zscore_weighted",
        weights={"lp": 0.4, "rp": 0.6},
    )

    lp_z = (scores_dict["lp"] - np.mean(scores_dict["lp"])) / np.std(scores_dict["lp"])
    rp_z = (scores_dict["rp"] - np.mean(scores_dict["rp"])) / np.std(scores_dict["rp"])
    expected = 0.4 * lp_z + 0.6 * rp_z
    np.testing.assert_allclose(fused, expected.astype(np.float64))


def test_fuse_scores_truncates_sources_to_shortest_length() -> None:
    scores_dict = {
        "lp": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "rp": np.array([4.0, 3.0], dtype=np.float64),
    }

    fused = fuse_scores(scores_dict, method="weighted_sum")

    expected = np.array([2.5, 2.5], dtype=np.float64)
    np.testing.assert_allclose(fused, expected)


def test_fuse_scores_zscore_weighted_returns_zeros_for_constant_source() -> None:
    scores_dict = {
        "lp": np.array([5.0, 5.0, 5.0], dtype=np.float64),
        "rp": np.array([1.0, 2.0, 3.0], dtype=np.float64),
    }

    fused = fuse_scores(scores_dict, method="zscore_weighted", weights={"lp": 0.5, "rp": 0.5})

    rp_z = (scores_dict["rp"] - np.mean(scores_dict["rp"])) / np.std(scores_dict["rp"])
    expected = 0.5 * rp_z
    np.testing.assert_allclose(fused, expected.astype(np.float64))


def test_fuse_scores_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="Unsupported fusion method"):
        fuse_scores({"lp": np.array([1.0, 2.0], dtype=np.float64)}, method="median")


def test_fuse_scores_rejects_missing_weight_key() -> None:
    with pytest.raises(ValueError, match="weights keys must match score sources exactly"):
        fuse_scores(
            {
                "lp": np.array([1.0, 2.0], dtype=np.float64),
                "rp": np.array([2.0, 3.0], dtype=np.float64),
            },
            method="weighted_sum",
            weights={"lp": 1.0},
        )


def test_fuse_scores_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        fuse_scores(
            {
                "lp": np.array([1.0, np.nan], dtype=np.float64),
                "rp": np.array([2.0, 3.0], dtype=np.float64),
            }
        )


# =====================================================================
# Confidence-weighted fusion tests
# =====================================================================


class TestConfidenceWeightedFusion:
    def test_output_shape(self) -> None:
        rng = np.random.RandomState(0)
        scores = {"lp": rng.randn(50), "rp": rng.randn(50)}
        resids = {"lp": rng.randn(50, 100), "rp": rng.randn(50, 100)}
        fused = fuse_scores_confidence_weighted(scores, resids)
        assert fused.shape == (50,)
        assert fused.dtype == np.float64

    def test_high_uncertainty_renderer_downweighted(self) -> None:
        T = 100
        # LP has low uncertainty (tight residuals)
        lp_scores = np.ones(T) * 5.0
        lp_resids = np.ones((T, 50)) * 0.1  # Very low variance

        # RP has high uncertainty (noisy residuals)
        rng = np.random.RandomState(42)
        rp_scores = np.ones(T) * 1.0
        rp_resids = rng.randn(T, 50) * 10.0  # Very high variance

        fused = fuse_scores_confidence_weighted(
            {"lp": lp_scores, "rp": rp_scores},
            {"lp": lp_resids, "rp": rp_resids},
        )
        # Fused should be closer to LP (5.0) than RP (1.0) since LP is more confident
        assert np.mean(fused) > 3.0

    def test_equal_uncertainty_equal_weights(self) -> None:
        T = 100
        scores_a = np.ones(T) * 2.0
        scores_b = np.ones(T) * 4.0
        # Same residual variance
        resids = np.ones((T, 50)) * 1.0
        fused = fuse_scores_confidence_weighted(
            {"a": scores_a, "b": scores_b},
            {"a": resids, "b": resids},
        )
        np.testing.assert_allclose(fused, 3.0, atol=1e-10)

    def test_key_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Key mismatch"):
            fuse_scores_confidence_weighted(
                {"lp": np.zeros(10)},
                {"rp": np.zeros((10, 50))},
            )

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fuse_scores_confidence_weighted({}, {})

    def test_wrong_score_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            fuse_scores_confidence_weighted(
                {"lp": np.zeros((10, 2))},
                {"lp": np.zeros((10, 50))},
            )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            fuse_scores_confidence_weighted(
                {"lp": np.zeros(10)},
                {"lp": np.zeros((20, 50))},
            )
