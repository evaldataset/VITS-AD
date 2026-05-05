from __future__ import annotations

import numpy as np
import pytest

from src.scoring.calibguard_v2 import CalibGuardV2
from src.scoring.calibguard_v3 import CalibGuardV3


class TestCalibGuardV3FromTrainSplit:
    def test_uses_last_train_segment_only(self) -> None:
        train_scores = np.linspace(0.0, 99.0, num=100, dtype=np.float64)

        guard = CalibGuardV3.from_train_split(
            train_scores,
            calib_ratio=0.2,
            alpha=0.1,
            rolling_window=0,
            use_aci=False,
        )

        expected = CalibGuardV2(alpha=0.1, rolling_window=0, use_aci=False)
        expected.fit(train_scores[-20:])

        assert guard.is_fitted is True
        assert guard.n_train_total == 100
        assert guard.n_train_calibration == 20
        assert guard.fixed_threshold == expected.fixed_threshold

    def test_rejects_invalid_calib_ratio(self) -> None:
        train_scores = np.random.RandomState(0).randn(100).astype(np.float64)

        with pytest.raises(ValueError, match=r"calib_ratio must be in \(0, 1\)"):
            CalibGuardV3.from_train_split(train_scores, calib_ratio=0.0)

        with pytest.raises(ValueError, match=r"calib_ratio must be in \(0, 1\)"):
            CalibGuardV3.from_train_split(train_scores, calib_ratio=1.0)

    def test_raises_on_test_data_passed_to_calibration(self) -> None:
        train_scores = np.random.RandomState(1).randn(120).astype(np.float64)
        test_scores = np.random.RandomState(2).randn(80).astype(np.float64)

        with pytest.raises(ValueError, match="test_scores must not be provided"):
            CalibGuardV3.from_train_split(
                train_scores,
                calib_ratio=0.2,
                test_scores=test_scores,
            )


class TestCalibGuardV3Predict:
    def test_predict_returns_tuple(self) -> None:
        rng = np.random.RandomState(3)
        train_scores = rng.randn(500).astype(np.float64)
        guard = CalibGuardV3.from_train_split(
            train_scores,
            calib_ratio=0.2,
            alpha=0.05,
            rolling_window=0,
            use_aci=False,
        )

        flag, p_value, threshold = guard.predict(0.0)
        assert isinstance(flag, bool)
        assert isinstance(p_value, float)
        assert isinstance(threshold, float)
        assert 0.0 < p_value <= 1.0

    def test_predict_before_fit_raises(self) -> None:
        guard = CalibGuardV3(alpha=0.01)
        with pytest.raises(RuntimeError, match="CalibGuardV2 has not been fitted"):
            guard.predict(0.0)


class TestCalibGuardV3ConformalGuarantee:
    def test_statistical_property_100_runs(self) -> None:
        n_runs = 100
        alpha = 0.01
        success = []

        for seed in range(n_runs):
            rng = np.random.RandomState(seed)
            train_scores = rng.randn(1000).astype(np.float64)
            test_scores = rng.randn(1000).astype(np.float64)

            guard = CalibGuardV3.from_train_split(
                train_scores,
                calib_ratio=0.2,
                alpha=alpha,
                rolling_window=0,
                use_aci=False,
            )
            flags, _, _ = guard.predict_batch(test_scores)
            far = float(np.mean(flags))
            success.append(far <= 2 * alpha)

        success_rate = float(np.mean(np.asarray(success, dtype=np.float64)))
        assert success_rate >= 0.9, (
            f"Only {success_rate * 100:.1f}% of runs satisfied FAR <= 2*alpha"
        )
