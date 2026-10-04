"""CalibGuard v3: leak-free conformal calibration for VLM residuals.

CalibGuard v3 keeps v2's core behavior (z-normalization, optional ACI, optional
Bonferroni correction, and rolling thresholding) but enforces a strict protocol:
calibration must come only from the *train split* (default: last 20% of train).

Under exchangeability and fixed-threshold mode (`rolling_window=0`), the split
conformal tail guarantee is:

    P(score > q | normal) <= alpha + O(1 / n_calib)

In rolling mode, this guarantee weakens because thresholds are updated online.
This implementation is specialized for VLM patch-token residual score streams.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportIncompatibleMethodOverride=false

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.scoring.calibguard_v2 import CalibGuardV2

LOGGER = logging.getLogger(__name__)


class CalibGuardV3(CalibGuardV2):
    """Leak-free CalibGuard variant with train-split-only calibration protocol.

    Args:
        alpha: Target false alarm rate in (0, 1).
        rolling_window: Rolling calibration window size. Set 0 for fixed mode.
        drift_sigma: Drift trigger threshold on normalized rolling mean.
        use_aci: Whether to enable adaptive conformal inference updates.
        aci_gamma: Learning rate for ACI updates.
        bonferroni_n_tests: Number of simultaneous tests for Bonferroni correction.
        eps: Stability constant for normalization.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        rolling_window: int = 0,
        drift_sigma: float = 3.0,
        use_aci: bool = True,
        aci_gamma: float = 0.01,
        bonferroni_n_tests: int = 1,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(
            alpha=alpha,
            rolling_window=rolling_window,
            drift_sigma=drift_sigma,
            use_aci=use_aci,
            aci_gamma=aci_gamma,
            bonferroni_n_tests=bonferroni_n_tests,
            eps=eps,
        )
        self._n_train_total: int = 0
        self._n_train_calibration: int = 0

    @property
    def n_train_total(self) -> int:
        """Total number of train scores provided to `from_train_split`."""
        return self._n_train_total

    @property
    def n_train_calibration(self) -> int:
        """Number of train scores used for calibration."""
        return self._n_train_calibration

    @classmethod
    def from_train_split(
        cls,
        train_scores: NDArray[np.float64],
        calib_ratio: float = 0.2,
        test_scores: Optional[NDArray[np.float64]] = None,
        alpha: float = 0.01,
        rolling_window: int = 0,
        drift_sigma: float = 3.0,
        use_aci: bool = True,
        aci_gamma: float = 0.01,
        bonferroni_n_tests: int = 1,
        eps: float = 1e-10,
    ) -> CalibGuardV3:
        """Create a fitted calibrator using only the last train-split segment.

        Args:
            train_scores: Normal anomaly scores from the train split, shape (n,).
            calib_ratio: Fraction of train scores used for calibration from the tail.
            test_scores: Optional guardrail input. Must be `None`; otherwise raises
                to prevent test-data leakage into calibration.
            alpha: Target false alarm rate.
            rolling_window: Rolling calibration window size.
            drift_sigma: Drift trigger threshold.
            use_aci: Whether to enable ACI.
            aci_gamma: ACI learning rate.
            bonferroni_n_tests: Number of tests for Bonferroni correction.
            eps: Numerical stability constant.

        Returns:
            A fitted `CalibGuardV3` instance.

        Raises:
            ValueError: If inputs are invalid or test scores are provided.
        """
        if test_scores is not None:
            raise ValueError(
                "test_scores must not be provided for calibration; "
                "CalibGuardV3 calibration is train-split only."
            )
        if not 0.0 < calib_ratio < 1.0:
            raise ValueError(f"calib_ratio must be in (0, 1), got {calib_ratio}.")

        train_arr = np.asarray(train_scores, dtype=np.float64)
        if train_arr.ndim != 1:
            raise ValueError(f"train_scores must be 1D, got ndim={train_arr.ndim}.")
        if train_arr.size < 2:
            raise ValueError("train_scores must contain at least 2 samples.")
        if not np.isfinite(train_arr).all():
            raise ValueError("train_scores contains non-finite values.")

        n_total = train_arr.size
        n_calib = max(1, int(n_total * calib_ratio))
        n_calib = min(n_calib, n_total - 1)
        calib_scores = train_arr[-n_calib:]

        guard = cls(
            alpha=alpha,
            rolling_window=rolling_window,
            drift_sigma=drift_sigma,
            use_aci=use_aci,
            aci_gamma=aci_gamma,
            bonferroni_n_tests=bonferroni_n_tests,
            eps=eps,
        )
        guard.fit(calib_scores)
        guard._n_train_total = n_total
        guard._n_train_calibration = n_calib

        LOGGER.info(
            "CalibGuardV3 train-split calibration: n_train=%d n_calib=%d calib_ratio=%.3f",
            n_total,
            n_calib,
            calib_ratio,
        )
        return guard

    def predict(self, score: float) -> Tuple[bool, float, float]:
        """Predict anomaly from a test score.

        Args:
            score: Raw anomaly score from evaluation/test stream.

        Returns:
            Tuple `(flag, p_value, threshold)`.

        Raises:
            RuntimeError: If calibrator is not fitted.
            ValueError: If score is non-finite.
        """
        result = super().predict(score)
        return result.flag, result.p_value, result.threshold

    def predict_batch(
        self, scores: NDArray[np.float64]
    ) -> Tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
        """Predict anomaly flags for a sequence of test scores.

        Args:
            scores: Raw anomaly scores, shape (T,).

        Returns:
            Tuple (flags, p_values, thresholds) of shape (T,).

        Raises:
            RuntimeError: If calibrator is not fitted.
            ValueError: If scores are invalid.
        """
        if not self.is_fitted:
            raise RuntimeError("CalibGuardV2 has not been fitted. Call fit() first.")

        scores_arr = np.asarray(scores, dtype=np.float64)
        if scores_arr.ndim != 1:
            raise ValueError(f"scores must be 1D, got ndim={scores_arr.ndim}.")
        if not np.isfinite(scores_arr).all():
            raise ValueError("scores contains non-finite values.")

        n = scores_arr.size
        flags = np.zeros(n, dtype=np.bool_)
        p_values = np.zeros(n, dtype=np.float64)
        thresholds = np.zeros(n, dtype=np.float64)

        for index, score in enumerate(scores_arr):
            flag, p_value, threshold = self.predict(float(score))
            flags[index] = flag
            p_values[index] = p_value
            thresholds[index] = threshold

        return flags, p_values, thresholds
