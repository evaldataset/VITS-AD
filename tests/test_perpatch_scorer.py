"""Tests for PerPatchMahalanobisScorer."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring.perpatch_scorer import PerPatchMahalanobisScorer


def _make_tokens(
    rng: np.random.RandomState,
    n_windows: int,
    n_patches: int = 16,
    hidden_dim: int = 32,
) -> np.ndarray:
    return rng.randn(n_windows, n_patches, hidden_dim).astype(np.float64)


class TestPerPatchMahalanobisScorerFit:
    def test_fit_stores_parameters(self) -> None:
        rng = np.random.RandomState(42)
        tokens = _make_tokens(rng, n_windows=100, n_patches=16, hidden_dim=32)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(tokens)
        assert scorer._n_patches == 16
        assert scorer._hidden_dim == 32
        assert len(scorer._patch_means) == 16
        assert len(scorer._patch_precisions) == 16
        assert scorer._patch_means[0].shape == (32,)
        assert scorer._patch_precisions[0].shape == (32, 32)

    def test_fit_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError, match="ndim"):
            PerPatchMahalanobisScorer().fit(np.zeros((10, 32)))

    def test_fit_rejects_single_window(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            PerPatchMahalanobisScorer().fit(np.zeros((1, 16, 32)))

    def test_fit_warns_few_windows(self) -> None:
        rng = np.random.RandomState(0)
        tokens = _make_tokens(rng, n_windows=5, hidden_dim=32)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(tokens)  # should work with warning, not crash

    def test_fit_accepts_float32_input(self) -> None:
        rng = np.random.RandomState(7)
        tokens = _make_tokens(rng, n_windows=50).astype(np.float32)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(tokens)
        assert scorer._hidden_dim == 32


class TestPerPatchMahalanobisScorerScore:
    def test_score_shape(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100)
        test = _make_tokens(rng, n_windows=20)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(train)
        scores = scorer.score(test)
        assert scores.shape == (20,)
        assert scores.dtype == np.float64

    def test_scores_nonnegative(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100)
        test = _make_tokens(rng, n_windows=50)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(train)
        scores = scorer.score(test)
        assert np.all(scores >= 0.0)

    def test_anomaly_has_higher_score(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=200)
        normal = _make_tokens(rng, n_windows=50)
        anomalous = _make_tokens(rng, n_windows=50) + 10.0
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(train)
        normal_scores = scorer.score(normal)
        anomaly_scores = scorer.score(anomalous)
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_not_fitted_raises(self) -> None:
        scorer = PerPatchMahalanobisScorer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scorer.score(np.zeros((5, 16, 32)))

    def test_wrong_n_patches_raises(self) -> None:
        rng = np.random.RandomState(0)
        train = _make_tokens(rng, n_windows=50, n_patches=16)
        test = _make_tokens(rng, n_windows=10, n_patches=8)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(train)
        with pytest.raises(ValueError, match="16 patches"):
            scorer.score(test)

    def test_wrong_hidden_dim_raises(self) -> None:
        rng = np.random.RandomState(0)
        train = _make_tokens(rng, n_windows=50, hidden_dim=32)
        test = _make_tokens(rng, n_windows=10, hidden_dim=16)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(train)
        with pytest.raises(ValueError, match="hidden_dim=32"):
            scorer.score(test)

    def test_score_per_patch_shape(self) -> None:
        rng = np.random.RandomState(1)
        train = _make_tokens(rng, n_windows=100, n_patches=16)
        test = _make_tokens(rng, n_windows=20, n_patches=16)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(train)
        per_patch = scorer.score_per_patch(test)
        assert per_patch.shape == (20, 16)
        assert np.all(per_patch >= 0.0)


class TestPerPatchMahalanobisScorerAggregation:
    def test_max_aggregation(self) -> None:
        rng = np.random.RandomState(0)
        train = _make_tokens(rng, n_windows=100)
        test = _make_tokens(rng, n_windows=20)
        scorer = PerPatchMahalanobisScorer(aggregation="max")
        scorer.fit(train)
        scores = scorer.score(test)
        assert scores.shape == (20,)
        assert np.all(scores >= 0.0)

    def test_mean_aggregation(self) -> None:
        rng = np.random.RandomState(0)
        train = _make_tokens(rng, n_windows=100)
        test = _make_tokens(rng, n_windows=20)
        scorer = PerPatchMahalanobisScorer(aggregation="mean")
        scorer.fit(train)
        scores = scorer.score(test)
        assert scores.shape == (20,)
        assert np.all(scores >= 0.0)

    def test_topk_aggregation(self) -> None:
        rng = np.random.RandomState(0)
        train = _make_tokens(rng, n_windows=100)
        test = _make_tokens(rng, n_windows=20)
        scorer = PerPatchMahalanobisScorer(aggregation="topk", topk=5)
        scorer.fit(train)
        scores = scorer.score(test)
        assert scores.shape == (20,)
        assert np.all(scores >= 0.0)

    def test_topk_geq_mean(self) -> None:
        """Top-k should be >= mean aggregation (top-k picks the largest)."""
        rng = np.random.RandomState(10)
        train = _make_tokens(rng, n_windows=100, n_patches=16)
        test = _make_tokens(rng, n_windows=50, n_patches=16)

        scorer_mean = PerPatchMahalanobisScorer(aggregation="mean")
        scorer_topk = PerPatchMahalanobisScorer(aggregation="topk", topk=5)
        scorer_max = PerPatchMahalanobisScorer(aggregation="max")

        scorer_mean.fit(train)
        scorer_topk.fit(train)
        scorer_max.fit(train)

        s_mean = scorer_mean.score(test)
        s_topk = scorer_topk.score(test)
        s_max = scorer_max.score(test)

        assert np.all(s_topk >= s_mean - 1e-10)
        assert np.all(s_max >= s_topk - 1e-10)

    def test_invalid_aggregation_raises(self) -> None:
        with pytest.raises(ValueError, match="aggregation"):
            PerPatchMahalanobisScorer(aggregation="median")  # type: ignore[arg-type]

    def test_invalid_topk_raises(self) -> None:
        with pytest.raises(ValueError, match="topk"):
            PerPatchMahalanobisScorer(topk=0)


class TestPerPatchMahalanobisScorerPersistence:
    def test_state_dict_roundtrip(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100, n_patches=16)
        test = _make_tokens(rng, n_windows=20, n_patches=16)

        scorer1 = PerPatchMahalanobisScorer(aggregation="topk", topk=8)
        scorer1.fit(train)
        scores1 = scorer1.score(test)

        state = scorer1.state_dict()
        scorer2 = PerPatchMahalanobisScorer()
        scorer2.load_state_dict(state)
        scores2 = scorer2.score(test)

        np.testing.assert_array_almost_equal(scores1, scores2)
        assert scorer2.aggregation == "topk"
        assert scorer2.topk == 8
        assert scorer2._n_patches == 16

    def test_state_dict_contains_expected_keys(self) -> None:
        rng = np.random.RandomState(0)
        tokens = _make_tokens(rng, n_windows=50)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(tokens)
        state = scorer.state_dict()
        for key in ("aggregation", "topk", "eps", "n_patches", "hidden_dim",
                    "patch_means", "patch_precisions"):
            assert key in state, f"Missing key: {key}"

    def test_load_state_dict_invalid_aggregation_raises(self) -> None:
        rng = np.random.RandomState(0)
        tokens = _make_tokens(rng, n_windows=50)
        scorer = PerPatchMahalanobisScorer()
        scorer.fit(tokens)
        state = scorer.state_dict()
        state["aggregation"] = "median"
        scorer2 = PerPatchMahalanobisScorer()
        with pytest.raises(ValueError, match="Invalid aggregation"):
            scorer2.load_state_dict(state)

    def test_unfitted_not_fitted_after_init(self) -> None:
        scorer = PerPatchMahalanobisScorer()
        assert scorer._n_patches == 0
        assert len(scorer._patch_means) == 0
