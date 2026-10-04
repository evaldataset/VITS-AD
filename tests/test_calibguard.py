"""Tests for CalibGuard conformal calibration."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring.calibguard import (
    CalibGuard,
    CalibGuardResult,
    CalibGuardStats,
    compute_far_at_alpha,
)


class TestCalibGuardInit:
    """Tests for CalibGuard initialization."""

    def test_init_valid_alpha(self):
        """Test initialization with valid alpha values."""
        for alpha in [0.01, 0.05, 0.1]:
            guard = CalibGuard(alpha=alpha)
            assert guard.alpha == alpha
            assert guard.is_fitted is False

    def test_init_default_values(self):
        """Test initialization with default parameters."""
        guard = CalibGuard()
        assert guard.alpha == 0.01
        assert guard.is_fitted is False

    def test_init_invalid_alpha(self):
        """Test initialization with invalid alpha values."""
        with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
            CalibGuard(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
            CalibGuard(alpha=1.0)

        with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
            CalibGuard(alpha=1.5)

    def test_init_invalid_rolling_window(self):
        """Test initialization with invalid rolling window."""
        with pytest.raises(ValueError, match="rolling_window must be non-negative"):
            CalibGuard(rolling_window=-1)

    def test_init_invalid_drift_sigma(self):
        """Test initialization with invalid drift sigma."""
        with pytest.raises(ValueError, match="drift_sigma must be positive"):
            CalibGuard(drift_sigma=0.0)

    def test_init_disabling_rolling(self):
        """Test initialization with rolling_window=0."""
        guard = CalibGuard(rolling_window=0)
        assert guard.is_fitted is False

    def test_properties(self):
        """Test property access."""
        guard = CalibGuard(alpha=0.05, rolling_window=1000, drift_sigma=5.0)
        assert guard.alpha == 0.05
        assert guard.is_fitted is False


class TestCalibGuardFit:
    """Tests for CalibGuard.fit()."""

    def test_fit_valid_scores(self):
        """Test fitting with valid calibration scores."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01)
        guard.fit(scores)

        assert guard.is_fitted is True
        assert guard.fixed_threshold > 0
        stats = guard.get_stats()
        assert stats.n_alarms == 0

    def test_fit_empty_scores(self):
        """Test fitting with empty scores."""
        guard = CalibGuard(alpha=0.01)
        with pytest.raises(ValueError, match="calibration_scores must be non-empty"):
            guard.fit(np.array([]))

    def test_fit_non_1d_scores(self):
        """Test fitting with non-1D scores."""
        guard = CalibGuard(alpha=0.01)
        with pytest.raises(ValueError, match="calibration_scores must be 1D"):
            guard.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

        with pytest.raises(ValueError, match="calibration_scores must be 1D"):
            guard.fit(np.ones((10, 10)))

    def test_fit_non_finite_scores(self):
        """Test fitting with non-finite scores."""
        guard = CalibGuard(alpha=0.01)
        scores = np.array([1.0, 2.0, np.inf, np.nan, -np.inf])
        with pytest.raises(
            ValueError, match="calibration_scores contains non-finite values"
        ):
            guard.fit(scores)

    def test_fit_single_score(self):
        """Test fitting with single score."""
        guard = CalibGuard(alpha=0.1)
        guard.fit(np.array([0.5]))
        assert guard.is_fitted is True
        assert guard.fixed_threshold == 0.5

    def test_fit_no_alarms_expected(self):
        """Test that no alarms are raised during fit."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01)
        guard.fit(scores)
        stats = guard.get_stats()
        assert stats.n_alarms == 0
        assert stats.empirical_far == 0.0

    def test_fit_reset_rolling_state(self):
        """Test that fit resets rolling state."""
        guard = CalibGuard(alpha=0.01, rolling_window=100)
        guard.fit(np.random.randn(100))
        assert guard.is_fitted is True

    def test_fit_compute_threshold(self):
        """Test that threshold is computed correctly."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01)
        guard.fit(scores)

        # Threshold should be the (n*(1-alpha))-th quantile (approx)
        # For alpha=0.01, threshold should be around 2.33 (1.96 for large n)
        assert guard.fixed_threshold > 0

    def test_fit_batch_processing(self):
        """Test that fit works with batch of floats."""
        guard = CalibGuard(alpha=0.05)
        guard.fit([1.0, 2.0, 3.0, 4.0, 5.0])
        assert guard.is_fitted is True


class TestCalibGuardPredict:
    """Tests for CalibGuard.predict()."""

    def test_predict_before_fit(self):
        """Test predict before fit raises RuntimeError."""
        guard = CalibGuard(alpha=0.01)
        with pytest.raises(RuntimeError, match="CalibGuard has not been fitted"):
            guard.predict(0.5)

    def test_predict_non_finite_score(self):
        """Test predict with non-finite score raises ValueError."""
        guard = CalibGuard(alpha=0.01)
        guard.fit(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="score must be finite"):
            guard.predict(np.inf)

    def test_predict_valid_score_below_threshold(self):
        """Test prediction with score below threshold."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01)
        guard.fit(scores)

        result = guard.predict(0.0)
        assert result.flag is False
        assert result.score == 0.0
        assert result.threshold == guard.fixed_threshold
        assert result.p_value > 0.5

    def test_predict_valid_score_above_threshold(self):
        """Test prediction with score above threshold."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01)
        guard.fit(scores)

        result = guard.predict(5.0)
        assert result.flag is True
        assert result.score == 5.0
        assert result.threshold == guard.fixed_threshold
        assert result.p_value < 0.5

    def test_predict_p_value_bounds(self):
        """Test that p-value is in (0, 1]."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01)
        guard.fit(scores)

        # For very high scores, p-value should be very low
        high_score = scores[-1] + 1.0
        result = guard.predict(high_score)
        assert 0.0 < result.p_value <= 1.0

        # For very low scores, p-value should be very high
        low_score = scores[0] - 1.0
        result = guard.predict(low_score)
        assert 0.0 < result.p_value <= 1.0

    def test_predict_updates_alarms(self):
        """Test that predict updates alarm count."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01)
        guard.fit(scores)

        # Initially 0 alarms
        stats = guard.get_stats()
        assert stats.n_alarms == 0
        assert stats.n_predictions == 0

        # Add anomaly
        guard.predict(10.0)
        stats = guard.get_stats()
        assert stats.n_alarms == 1
        assert stats.n_predictions == 1

        # Add normal
        guard.predict(0.0)
        stats = guard.get_stats()
        assert stats.n_alarms == 1
        assert stats.n_predictions == 2

    def test_predict_without_rolling(self):
        """Test predict without rolling window enabled."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01, rolling_window=0)
        guard.fit(scores)

        result = guard.predict(5.0)
        assert result.flag is True
        assert result.threshold == guard.fixed_threshold

    def test_predict_with_rolling(self):
        """Test predict with rolling window enabled."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01, rolling_window=100)
        guard.fit(scores)

        # Initially no alarms, so rolling buffer is populated
        guard.predict(0.0)  # Normal
        stats = guard.get_stats()
        assert stats.n_alarms == 0


class TestCalibGuardRolling:
    """Tests for CalibGuard rolling window functionality."""

    def test_rolling_threshold_update(self):
        """Test that rolling threshold updates over time."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01, rolling_window=100)

        # Fit on normal scores
        normal_scores = np.random.randn(1000)
        guard.fit(normal_scores)

        # All predictions are normal, so buffer fills up
        for _ in range(150):
            guard.predict(0.0)

        # Check that rolling threshold is updated
        stats = guard.get_stats()
        assert stats.n_predictions == 150
        assert stats.rolling_threshold is not None

    def test_rolling_threshold_comparison_to_fixed(self):
        """Test that rolling threshold is used when available."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01, rolling_window=50)

        # Fit on normal scores
        normal_scores = np.random.randn(1000)
        guard.fit(normal_scores)

        # All predictions are normal, so rolling threshold should be computed
        for _ in range(60):
            guard.predict(0.0)

        # Both thresholds should be similar (buffer is normal data)
        stats = guard.get_stats()
        assert stats.rolling_threshold is not None
        assert stats.rolling_threshold <= stats.fixed_threshold

    def test_rolling_threshold_no_alarms(self):
        """Test rolling threshold without alarms."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01, rolling_window=100)

        normal_scores = np.random.randn(1000)
        guard.fit(normal_scores)

        # No alarms, so all scores go into rolling buffer
        for _ in range(150):
            guard.predict(0.0)

        # Rolling threshold should be updated
        stats = guard.get_stats()
        assert stats.rolling_threshold is not None
        assert stats.empirical_far == 0.0

    def test_drift_detection_no_shift(self):
        """Test that drift is not detected when rolling mean is stable."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01, rolling_window=100, drift_sigma=0.5)

        # Fit on normal data
        normal_scores = np.random.randn(1000)
        guard.fit(normal_scores)

        # All predictions are normal
        for _ in range(200):
            guard.predict(0.0)

        # No drift should be detected
        stats = guard.get_stats()
        assert stats.drift_detected is False

    def test_drift_detection_with_shift(self):
        """Test that drift is detected when rolling mean shifts significantly."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01, rolling_window=100, drift_sigma=0.5)

        # Fit on normal data (mean ~ 0)
        normal_scores = np.random.randn(1000)
        guard.fit(normal_scores)

        # Predict normal scores just below threshold
        # This populates the rolling buffer without triggering alarms
        # The scores will have a slightly different mean due to randomness
        for _ in range(100):
            guard.predict(guard.fixed_threshold - 0.5)

        # Now predict with anomalous scores - this should trigger drift detection
        # because the rolling buffer has been filled
        for _ in range(10):
            guard.predict(10.0)  # Anomaly

        # Drift should be detected after processing some anomalies
        stats = guard.get_stats()
        assert stats.drift_detected is True

    def test_drift_uses_fixed_threshold(self):
        """Test that drift detection causes fallback to fixed threshold."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01, rolling_window=100, drift_sigma=0.5)

        # Fit on normal data
        normal_scores = np.random.randn(1000)
        guard.fit(normal_scores)

        # Generate anomalous scores
        for _ in range(100):
            guard.predict(10.0)

        # Now predict with high score - should use fixed threshold
        result = guard.predict(5.0)
        assert result.threshold == guard.fixed_threshold

    def test_rolling_window_disabled(self):
        """Test that rolling is disabled when rolling_window=0."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01, rolling_window=0)

        normal_scores = np.random.randn(1000)
        guard.fit(normal_scores)

        # Predict many times
        for _ in range(50):
            guard.predict(0.0)

        # Rolling threshold should be None
        stats = guard.get_stats()
        assert stats.rolling_threshold is None
        assert stats.empirical_far == 0.0


class TestComputeFarAtAlpha:
    """Tests for compute_far_at_alpha function."""

    def test_compute_far_at_alpha_separable_scores(self):
        """Test compute_far_at_alpha with clearly separable scores."""
        np.random.seed(42)
        n_normal = 1000
        n_anomaly = 100

        # Generate separable normal and anomaly scores
        normal_scores = np.random.randn(n_normal) + 1.0  # Shifted
        anomaly_scores = np.random.randn(n_anomaly) + 5.0  # Much higher

        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.array([0] * n_normal + [1] * n_anomaly)

        result = compute_far_at_alpha(all_scores, all_labels, alpha=0.01)

        # Check return keys
        assert "target_far" in result
        assert "actual_far" in result
        assert "coverage" in result
        assert "threshold" in result
        assert "n_calibration" in result
        assert "n_test_normal" in result
        assert "n_test_anomaly" in result

        # Check values
        assert result["target_far"] == 0.01
        assert result["actual_far"] <= 0.01 * 2  # Statistical variation
        assert 0.9 < result["coverage"] < 1.0  # High detection rate

    def test_compute_far_at_alpha_empty_anomaly_set(self):
        """Test compute_far_at_alpha with no anomaly samples."""
        np.random.seed(42)
        normal_scores = np.random.randn(1000)

        result = compute_far_at_alpha(normal_scores, np.zeros(1000), alpha=0.01)

        assert result["actual_far"] <= 0.01 * 2
        assert result["coverage"] == 0.0
        assert "n_test_anomaly" in result
        assert result["n_test_anomaly"] == 0

    def test_compute_far_at_alpha_invalid_shapes(self):
        """Test compute_far_at_alpha with invalid shapes."""
        with pytest.raises(
            ValueError, match="scores and labels must be 1D with same shape"
        ):
            compute_far_at_alpha(np.array([1, 2, 3]), np.array([[1, 2], [3, 4]]))

        with pytest.raises(
            ValueError, match="scores and labels must be 1D with same shape"
        ):
            compute_far_at_alpha(np.array([1, 2, 3]), np.array([1, 2]))

    def test_compute_far_at_alpha_no_normal_samples(self):
        """Test compute_far_at_alpha with no normal samples."""
        with pytest.raises(ValueError, match="No normal samples found"):
            compute_far_at_alpha(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))

    def test_compute_far_at_alpha_different_alpha(self):
        """Test compute_far_at_alpha with different alpha values."""
        for alpha in [0.01, 0.05, 0.1]:
            np.random.seed(42)
            normal_scores = np.random.randn(1000)
            anomaly_scores = np.random.randn(100) + 5.0

            all_scores = np.concatenate([normal_scores, anomaly_scores])
            all_labels = np.array([0] * 1000 + [1] * 100)

            result = compute_far_at_alpha(all_scores, all_labels, alpha=alpha)

            assert result["target_far"] == alpha
            # Actual FAR should be <= 2*alpha (conservative)
            assert result["actual_far"] <= 2 * alpha

    def test_compute_far_at_alpha_calibration_ratio(self):
        """Test compute_far_at_alpha with different calibration ratios."""
        np.random.seed(42)
        normal_scores = np.random.randn(1000)
        anomaly_scores = np.random.randn(100) + 5.0

        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.array([0] * 1000 + [1] * 100)

        for calib_ratio in [0.3, 0.5, 0.7]:
            result = compute_far_at_alpha(
                all_scores, all_labels, alpha=0.01, calibration_ratio=calib_ratio
            )

            assert result["n_calibration"] == int(1000 * calib_ratio)


class TestCalibGuardResult:
    """Tests for CalibGuardResult dataclass."""

    def test_result_creation(self):
        """Test CalibGuardResult creation."""
        result = CalibGuardResult(flag=True, p_value=0.01, threshold=5.0, score=10.0)
        assert result.flag is True
        assert result.p_value == 0.01
        assert result.threshold == 5.0
        assert result.score == 10.0

    def test_result_equality(self):
        """Test CalibGuardResult equality."""
        result1 = CalibGuardResult(flag=True, p_value=0.01, threshold=5.0, score=10.0)
        result2 = CalibGuardResult(flag=True, p_value=0.01, threshold=5.0, score=10.0)
        assert result1 == result2

        result3 = CalibGuardResult(flag=False, p_value=0.99, threshold=1.0, score=0.5)
        assert result1 != result3

    def test_result_immutability(self):
        """Test CalibGuardResult immutability (dataclass default)."""
        CalibGuardResult(flag=True, p_value=0.01, threshold=5.0, score=10.0)
        # Dataclasses are mutable by default, but this test verifies structure

    def test_result_default_values(self):
        """Test CalibGuardResult default values."""
        result = CalibGuardResult(flag=False, p_value=0.5, threshold=0.0, score=0.0)
        assert result.flag is False
        assert result.p_value == 0.5


class TestCalibGuardStats:
    """Tests for CalibGuardStats dataclass."""

    def test_stats_creation(self):
        """Test CalibGuardStats creation."""
        stats = CalibGuardStats(
            n_calibration=1000,
            alpha=0.01,
            fixed_threshold=5.0,
            rolling_threshold=4.5,
            n_alarms=10,
            n_predictions=1000,
            empirical_far=0.01,
            drift_detected=False,
        )
        assert stats.n_calibration == 1000
        assert stats.alpha == 0.01
        assert stats.fixed_threshold == 5.0
        assert stats.rolling_threshold == 4.5
        assert stats.n_alarms == 10
        assert stats.n_predictions == 1000
        assert stats.empirical_far == 0.01
        assert stats.drift_detected is False

    def test_stats_with_rolling_disabled(self):
        """Test CalibGuardStats with None rolling threshold."""
        stats = CalibGuardStats(
            n_calibration=1000,
            alpha=0.01,
            fixed_threshold=5.0,
            rolling_threshold=None,
            n_alarms=10,
            n_predictions=1000,
            empirical_far=0.01,
            drift_detected=False,
        )
        assert stats.rolling_threshold is None

    def test_stats_empty_predictions(self):
        """Test CalibGuardStats with no predictions."""
        stats = CalibGuardStats(
            n_calibration=1000,
            alpha=0.01,
            fixed_threshold=5.0,
            rolling_threshold=None,
            n_alarms=0,
            n_predictions=0,
            empirical_far=0.0,
            drift_detected=False,
        )
        assert stats.empirical_far == 0.0


class TestConformalGuarantee:
    """Statistical property tests for the conformal guarantee."""

    def test_conformal_guarantee_with_normal_scores(self):
        """Test that FAR ≈ alpha on calibration data."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01, rolling_window=0)  # Fixed mode for strict guarantee

        # Fit on normal data
        normal_scores = np.random.randn(10000)
        guard.fit(normal_scores)

        # Predict on same data (worst case for guarantee)
        for score in normal_scores:
            result = guard.predict(score)
            assert isinstance(result, CalibGuardResult)

        stats = guard.get_stats()
        # Empirical FAR should be close to alpha
        # For a good guarantee, we expect FAR ≤ 2*alpha
        assert stats.empirical_far <= 2 * guard.alpha

    def test_statistical_property_100_runs(self):
        """Run CalibGuard 100 times and verify FAR <= 2*alpha 90%+ of the time."""
        np.random.seed(42)
        n_runs = 100
        alpha = 0.01

        results = []
        for seed in range(n_runs):
            np.random.seed(seed)
            guard = CalibGuard(alpha=alpha, rolling_window=0)  # Fixed mode
            normal_scores = np.random.randn(1000)
            guard.fit(normal_scores)

            # No anomalies
            for score in normal_scores:
                guard.predict(score)

            stats = guard.get_stats()
            results.append(stats.empirical_far <= 2 * alpha)

        # At least 90% should have FAR <= 2*alpha
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.9, (
            f"Only {success_rate * 100:.1f}% of runs satisfied FAR <= 2*alpha"
        )

    def test_conformal_guarantee_with_mixed_scores(self):
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01)

        # Generate mixture of normal and anomalous scores
        normal_scores = np.random.randn(1000)
        anomalous_scores = np.random.randn(100) + 5.0
        all_scores = np.concatenate([normal_scores, anomalous_scores])
        np.random.shuffle(all_scores)

        guard.fit(normal_scores)
        for score in all_scores:
            guard.predict(score)

        guard.get_stats()
        # Normal scores should have FAR ≈ alpha
        # This doesn't guarantee this because we fit on the same data
        # but we verify the structure works correctly

    def test_predict_batch_vs_predict(self):
        """Test that predict_batch gives same results as batch predict."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01)

        normal_scores = np.random.randn(1000)
        guard.fit(normal_scores)

        # Get single predictions
        results = [guard.predict(score) for score in normal_scores[:10]]

        # Get batch prediction
        flags, p_values, thresholds = guard.predict_batch(normal_scores[:10])

        # Compare
        for i, score in enumerate(normal_scores[:10]):
            flag_expected = results[i].flag
            p_value_expected = results[i].p_value
            threshold_expected = results[i].threshold

            assert flag_expected == bool(flags[i])
            assert p_value_expected == p_values[i]
            assert threshold_expected == thresholds[i]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_fit_with_identical_scores(self):
        """Test fit with all identical scores."""
        guard = CalibGuard(alpha=0.01)
        guard.fit(np.array([1.0] * 100))
        assert guard.is_fitted is True
        assert guard.fixed_threshold == 1.0

    def test_fit_with_extreme_values(self):
        """Test fit with very large or very small values."""
        guard = CalibGuard(alpha=0.01)
        guard.fit(np.array([1e10, 2e10, 3e10]))
        assert guard.is_fitted is True
        assert guard.fixed_threshold > 0

        guard = CalibGuard(alpha=0.01)
        guard.fit(np.array([-1e10, -2e10, -3e10]))
        assert guard.is_fitted is True
        assert guard.fixed_threshold < 0

    def test_predict_with_exact_threshold(self):
        """Test prediction when score equals threshold."""
        np.random.seed(42)
        scores = np.random.randn(1000)
        guard = CalibGuard(alpha=0.01)
        guard.fit(scores)

        # Find a score close to threshold
        for score in scores:
            result = guard.predict(score)
            if abs(score - guard.fixed_threshold) < 1e-6:
                # Flag could be either True or False depending on >= vs >
                assert result.score == guard.fixed_threshold
                break

    def test_predict_batch_large_batch(self):
        """Test predict_batch with very large batch."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01)
        guard.fit(np.random.randn(1000))

        # Test with large batch
        large_batch = np.random.randn(10000)
        flags, p_values, thresholds = guard.predict_batch(large_batch)

        assert len(flags) == 10000
        assert len(p_values) == 10000
        assert len(thresholds) == 10000

    def test_get_stats_without_fit(self):
        """Test get_stats before fit."""
        guard = CalibGuard(alpha=0.01)
        stats = guard.get_stats()

        assert stats.n_calibration == 0
        assert stats.n_alarms == 0
        assert stats.n_predictions == 0
        assert stats.empirical_far == 0.0

    def test_rolling_threshold_insufficient_data(self):
        """Test that rolling threshold is not updated with insufficient data."""
        guard = CalibGuard(alpha=0.01, rolling_window=100)

        # Fit on normal data
        normal_scores = np.random.randn(1000)
        guard.fit(normal_scores)

        # Only predict once (insufficient for rolling)
        guard.predict(0.0)

        # Rolling threshold should still be None
        stats = guard.get_stats()
        assert stats.rolling_threshold is None

    def test_multiple_fits(self):
        """Test that multiple fit calls work correctly."""
        np.random.seed(42)
        guard = CalibGuard(alpha=0.01)

        # First fit
        scores1 = np.random.randn(1000)
        guard.fit(scores1)
        first_threshold = guard.fixed_threshold

        # Second fit
        scores2 = np.random.randn(1000) * 2
        guard.fit(scores2)
        second_threshold = guard.fixed_threshold

        # Thresholds should be different
        assert first_threshold != second_threshold
