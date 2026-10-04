"""Tests for DualSignalScorer."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring.dual_signal_scorer import DualSignalScorer, PerPatchScorer


def _make_tokens(
    rng: np.random.RandomState, n_windows: int, n_patches: int = 16, hidden_dim: int = 32
) -> np.ndarray:
    return rng.randn(n_windows, n_patches, hidden_dim).astype(np.float64)


class TestDualSignalScorerFit:
    def test_fit_stores_parameters(self) -> None:
        rng = np.random.RandomState(42)
        tokens = _make_tokens(rng, n_windows=100)
        scorer = DualSignalScorer(alpha=0.5)
        scorer.fit(tokens)
        assert scorer._train_mu is not None
        assert scorer._precision is not None
        assert scorer._train_mu.shape == (32,)
        assert scorer._precision.shape == (32, 32)

    def test_fit_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError, match="ndim"):
            DualSignalScorer().fit(np.zeros((10, 32)))

    def test_fit_warns_few_windows(self) -> None:
        rng = np.random.RandomState(0)
        tokens = _make_tokens(rng, n_windows=5, hidden_dim=32)
        scorer = DualSignalScorer()
        scorer.fit(tokens)  # Should work with warning, not crash


class TestDualSignalScorerScore:
    def test_distributional_score_shape(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100)
        test = _make_tokens(rng, n_windows=20)
        scorer = DualSignalScorer()
        scorer.fit(train)
        scores = scorer.score_distributional(test)
        assert scores.shape == (20,)
        assert scores.dtype == np.float64

    def test_distributional_score_nonnegative(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100)
        test = _make_tokens(rng, n_windows=20)
        scorer = DualSignalScorer()
        scorer.fit(train)
        scores = scorer.score_distributional(test)
        assert np.all(scores >= 0)

    def test_anomaly_has_higher_score(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=200)
        normal = _make_tokens(rng, n_windows=50)
        anomalous = _make_tokens(rng, n_windows=50) + 10.0  # Shifted far away
        scorer = DualSignalScorer()
        scorer.fit(train)
        normal_scores = scorer.score_distributional(normal)
        anomaly_scores = scorer.score_distributional(anomalous)
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_not_fitted_raises(self) -> None:
        scorer = DualSignalScorer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scorer.score_distributional(np.zeros((5, 16, 32)))


class TestDualSignalScorerFuse:
    def test_fuse_shape(self) -> None:
        scorer = DualSignalScorer(alpha=0.5)
        traj = np.random.RandomState(0).randn(100).astype(np.float64)
        dist = np.random.RandomState(1).randn(100).astype(np.float64)
        fused = scorer.fuse(traj, dist)
        assert fused.shape == (100,)

    def test_alpha_one_uses_trajectory_only(self) -> None:
        scorer = DualSignalScorer(alpha=1.0)
        traj = np.array([1.0, 2.0, 3.0])
        dist = np.array([10.0, 20.0, 30.0])
        fused = scorer.fuse(traj, dist)
        # With alpha=1, dist should be ignored -> fused is z-scored traj
        np.testing.assert_allclose(fused, scorer._zscore(traj))

    def test_alpha_zero_uses_distribution_only(self) -> None:
        scorer = DualSignalScorer(alpha=0.0)
        traj = np.array([1.0, 2.0, 3.0])
        dist = np.array([10.0, 20.0, 30.0])
        fused = scorer.fuse(traj, dist)
        np.testing.assert_allclose(fused, scorer._zscore(dist))

    def test_mismatched_shapes_raises(self) -> None:
        scorer = DualSignalScorer()
        with pytest.raises(ValueError, match="must match"):
            scorer.fuse(np.zeros(10), np.zeros(20))


class TestDualSignalScorerPersistence:
    def test_state_dict_roundtrip(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100)
        test = _make_tokens(rng, n_windows=20)

        scorer1 = DualSignalScorer(alpha=0.7)
        scorer1.fit(train)
        scores1 = scorer1.score_distributional(test)

        state = scorer1.state_dict()
        scorer2 = DualSignalScorer()
        scorer2.load_state_dict(state)
        scores2 = scorer2.score_distributional(test)

        np.testing.assert_array_equal(scores1, scores2)
        assert scorer2.alpha == 0.7

    def test_state_dict_roundtrips_normalizers(self) -> None:
        rng = np.random.RandomState(123)
        scorer = DualSignalScorer(alpha=0.3)
        scorer.fit(_make_tokens(rng, n_windows=80))
        ref_traj = rng.normal(loc=2.0, scale=0.5, size=200)
        ref_dist = rng.normal(loc=10.0, scale=3.0, size=200)
        scorer.fit_normalizers(ref_traj, ref_dist)
        state = scorer.state_dict()
        restored = DualSignalScorer()
        restored.load_state_dict(state)
        assert restored._traj_ref_mu == scorer._traj_ref_mu
        assert restored._traj_ref_sigma == scorer._traj_ref_sigma
        assert restored._dist_ref_mu == scorer._dist_ref_mu
        assert restored._dist_ref_sigma == scorer._dist_ref_sigma


class TestDualSignalScorerLeakFreeFuse:
    """Reference-statistic z-scoring is leak-free: identical test inputs
    must produce identical fused outputs regardless of which other test
    samples appear in the same batch."""

    def test_fuse_with_normalizers_is_translation_invariant(self) -> None:
        rng = np.random.RandomState(0)
        scorer = DualSignalScorer(alpha=0.4)
        ref_traj = rng.normal(loc=0.0, scale=1.0, size=500)
        ref_dist = rng.normal(loc=5.0, scale=2.0, size=500)
        scorer.fit_normalizers(ref_traj, ref_dist)

        traj_a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        dist_a = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
        # Add an extra "outlier" test sample only in scenario B; under the
        # leaky in-batch path this would shift the z-score of the original
        # five samples.  Under fit_normalizers it must NOT.
        traj_b = np.concatenate([traj_a, [10.0]])
        dist_b = np.concatenate([dist_a, [50.0]])

        fused_a = scorer.fuse(traj_a, dist_a)
        fused_b = scorer.fuse(traj_b, dist_b)
        np.testing.assert_allclose(fused_a, fused_b[:5])

    def test_fuse_without_normalizers_is_transductive(self) -> None:
        # Sanity check that the legacy fallback IS sensitive to batch
        # composition (this is exactly the leakage we fix).
        scorer = DualSignalScorer(alpha=0.4)
        traj_a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        dist_a = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
        traj_b = np.concatenate([traj_a, [10.0]])
        dist_b = np.concatenate([dist_a, [50.0]])
        fused_a = scorer.fuse(traj_a, dist_a)
        fused_b = scorer.fuse(traj_b, dist_b)
        # Without frozen normalizers the first five values should differ.
        assert not np.allclose(fused_a, fused_b[:5])


# =====================================================================
# PerPatchScorer tests
# =====================================================================


class TestPerPatchScorerFit:
    def test_fit_stores_per_patch_params(self) -> None:
        rng = np.random.RandomState(42)
        tokens = _make_tokens(rng, n_windows=100, n_patches=16, hidden_dim=32)
        scorer = PerPatchScorer()
        scorer.fit(tokens)
        assert scorer._mus is not None
        assert scorer._mus.shape == (16, 32)
        assert scorer._precisions is not None
        assert scorer._precisions.shape == (16, 32, 32)

    def test_fit_with_random_projection(self) -> None:
        rng = np.random.RandomState(42)
        tokens = _make_tokens(rng, n_windows=100, n_patches=8, hidden_dim=600)
        scorer = PerPatchScorer(max_dim=64)
        scorer.fit(tokens)
        assert scorer._proj is not None
        assert scorer._proj.shape == (600, 64)
        assert scorer._mus is not None
        assert scorer._mus.shape == (8, 64)

    def test_fit_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match="ndim"):
            PerPatchScorer().fit(np.zeros((10, 32)))


class TestPerPatchScorerScore:
    def test_score_shape(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100, n_patches=16, hidden_dim=32)
        test = _make_tokens(rng, n_windows=20, n_patches=16, hidden_dim=32)
        scorer = PerPatchScorer()
        scorer.fit(train)
        scores = scorer.score(test)
        assert scores.shape == (20,)
        assert scores.dtype == np.float64

    def test_score_nonnegative(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100, n_patches=16, hidden_dim=32)
        test = _make_tokens(rng, n_windows=20, n_patches=16, hidden_dim=32)
        scorer = PerPatchScorer()
        scorer.fit(train)
        assert np.all(scorer.score(test) >= 0)

    def test_anomaly_higher_score(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=200, n_patches=16, hidden_dim=32)
        normal = _make_tokens(rng, n_windows=50, n_patches=16, hidden_dim=32)
        anomalous = _make_tokens(rng, n_windows=50, n_patches=16, hidden_dim=32) + 10.0
        scorer = PerPatchScorer()
        scorer.fit(train)
        assert np.mean(scorer.score(anomalous)) > np.mean(scorer.score(normal))

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not been fitted"):
            PerPatchScorer().score(np.zeros((5, 16, 32)))

    @pytest.mark.parametrize("agg", ["mean", "max", "p95"])
    def test_aggregation_modes(self, agg: str) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100, n_patches=16, hidden_dim=32)
        test = _make_tokens(rng, n_windows=10, n_patches=16, hidden_dim=32)
        scorer = PerPatchScorer(aggregation=agg)
        scorer.fit(train)
        scores = scorer.score(test)
        assert scores.shape == (10,)

    def test_patchmap_shape(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100, n_patches=16, hidden_dim=32)
        test = _make_tokens(rng, n_windows=10, n_patches=16, hidden_dim=32)
        scorer = PerPatchScorer()
        scorer.fit(train)
        pmap = scorer.score_patchmap(test)
        assert pmap.shape == (10, 16)


class TestPerPatchScorerPersistence:
    def test_state_dict_roundtrip(self) -> None:
        rng = np.random.RandomState(42)
        train = _make_tokens(rng, n_windows=100, n_patches=16, hidden_dim=32)
        test = _make_tokens(rng, n_windows=20, n_patches=16, hidden_dim=32)

        s1 = PerPatchScorer(aggregation="max")
        s1.fit(train)
        scores1 = s1.score(test)

        state = s1.state_dict()
        s2 = PerPatchScorer()
        s2.load_state_dict(state)
        scores2 = s2.score(test)

        np.testing.assert_array_equal(scores1, scores2)
        assert s2.aggregation == "max"
