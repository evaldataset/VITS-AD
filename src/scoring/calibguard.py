"""CalibGuard: Conformal calibration for anomaly detection with FAR guarantees.

Provides distribution-free false alarm rate (FAR) control via split conformal
prediction. Two modes:

1. **Fixed threshold** (theoretical guarantee):
   Given calibration scores from normal data, compute quantile threshold
   q such that P(s > q) ≤ α for any future normal sample.

2. **Rolling threshold** (practical, weaker guarantee):
   Maintain a sliding window of recent non-alarm scores and update the
   threshold adaptively. Falls back to fixed threshold on drift detection.

References:
    - Vovk et al., "Algorithmic Learning in a Random World" (2005)
    - Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift" (2021)
    - Bhatnagar et al., "Improved Online Conformal Prediction via Strongly Adaptive
      Online Learning" (2023)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibGuardResult:
    """Result from CalibGuard prediction.

    Attributes:
        flag: Boolean anomaly flag (True = anomaly).
        p_value: Conformal p-value in [0, 1]. Lower = more anomalous.
        threshold: Current decision threshold.
        score: Input anomaly score.
    """

    flag: bool
    p_value: float
    threshold: float
    score: float


@dataclass
class CalibGuardStats:
    """Calibration statistics for monitoring.

    Attributes:
        n_calibration: Number of calibration samples.
        alpha: Target FAR.
        fixed_threshold: Fixed conformal threshold.
        rolling_threshold: Current rolling threshold (if enabled).
        n_alarms: Total alarms raised.
        n_predictions: Total predictions made.
        empirical_far: Observed false alarm rate.
        drift_detected: Whether distribution drift was detected.
    """

    n_calibration: int
    alpha: float
    fixed_threshold: float
    rolling_threshold: Optional[float]
    n_alarms: int
    n_predictions: int
    empirical_far: float
    drift_detected: bool


class CalibGuard:
    """Conformal anomaly detection with FAR guarantee.

    Given a set of anomaly scores from normal calibration data,
    CalibGuard computes a threshold q such that:

        P(score > q | normal) ≤ α

    This provides a empirical FAR diagnostic under the
    exchangeability assumption.

    Args:
        alpha: Target false alarm rate. Must be in (0, 1).
        rolling_window: Size of rolling calibration window for adaptive mode.
            Set to 0 to disable rolling mode.
        drift_sigma: Number of standard deviations for drift detection.
            If the rolling window mean deviates from calibration mean by
            more than drift_sigma * calibration_std, drift is detected
            and the fixed threshold is used.

    Raises:
        ValueError: If alpha is not in (0, 1) or rolling_window is negative.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        rolling_window: int = 2000,
        drift_sigma: float = 3.0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if rolling_window < 0:
            raise ValueError(
                f"rolling_window must be non-negative, got {rolling_window}."
            )
        if drift_sigma <= 0.0:
            raise ValueError(f"drift_sigma must be positive, got {drift_sigma}.")

        self._alpha = alpha
        self._rolling_window = rolling_window
        self._drift_sigma = drift_sigma

        # Calibration state
        self._calibration_scores: Optional[NDArray[np.float64]] = None
        self._fixed_threshold: float = float("inf")
        self._calib_mean: float = 0.0
        self._calib_std: float = 1.0

        # Rolling state
        self._rolling_buffer: list[float] = []
        self._rolling_threshold: Optional[float] = None
        self._drift_detected: bool = False

        # Tracking
        self._n_alarms: int = 0
        self._n_predictions: int = 0
        self._fitted: bool = False

    @property
    def alpha(self) -> float:
        """Target false alarm rate."""
        return self._alpha

    @property
    def fixed_threshold(self) -> float:
        """Fixed conformal threshold from calibration."""
        return self._fixed_threshold

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    @property
    def rolling_window(self) -> int:
        """Rolling window size (0 = disabled)."""
        return self._rolling_window
    def fit(self, calibration_scores: NDArray[np.float64]) -> CalibGuard:
        """Fit the calibrator on normal calibration scores.

        Computes the fixed conformal threshold as the ⌈(n+1)(1-α)⌉/n
        quantile of the calibration scores.

        Args:
            calibration_scores: Anomaly scores from normal data, shape (n,).

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If scores are empty, non-1D, or contain non-finite values.
        """
        scores = np.asarray(calibration_scores, dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError(f"calibration_scores must be 1D, got ndim={scores.ndim}.")
        if scores.size == 0:
            raise ValueError("calibration_scores must be non-empty.")
        if not np.isfinite(scores).all():
            raise ValueError("calibration_scores contains non-finite values.")

        self._calibration_scores = np.sort(scores)
        n = scores.size

        # Conformal quantile index: ⌈(n+1)(1-α)⌉
        # This ensures finite-sample coverage guarantee
        quantile_index = int(np.ceil((n + 1) * (1 - self._alpha))) - 1
        quantile_index = min(quantile_index, n - 1)  # Clamp to valid range

        self._fixed_threshold = float(self._calibration_scores[quantile_index])
        self._calib_mean = float(np.mean(scores))
        self._calib_std = max(float(np.std(scores)), 1e-10)

        # Reset rolling state
        self._rolling_buffer = []
        self._rolling_threshold = None
        self._drift_detected = False
        self._n_alarms = 0
        self._n_predictions = 0
        self._fitted = True

        LOGGER.info(
            "CalibGuard fitted: n=%d, alpha=%.4f, threshold=%.6f, "
            "calib_mean=%.6f, calib_std=%.6f",
            n,
            self._alpha,
            self._fixed_threshold,
            self._calib_mean,
            self._calib_std,
        )

        return self

    def predict(self, score: float) -> CalibGuardResult:
        """Predict whether a score is anomalous.

        Args:
            score: Anomaly score for a single timestep.

        Returns:
            CalibGuardResult with flag, p-value, threshold, and score.

        Raises:
            RuntimeError: If the calibrator has not been fitted.
            ValueError: If score is non-finite.
        """
        if not self._fitted:
            raise RuntimeError("CalibGuard has not been fitted. Call fit() first.")
        if not np.isfinite(score):
            raise ValueError(f"score must be finite, got {score}.")

        score_val = float(score)

        # Compute conformal p-value against calibration set
        p_value = self._compute_p_value(score_val)

        # Determine threshold (fixed or rolling)
        threshold = self._get_current_threshold()

        # Flag
        flag = score_val > threshold

        # Update rolling buffer (only non-alarm scores)
        if not flag and self._rolling_window > 0:
            self._rolling_buffer.append(score_val)
            if len(self._rolling_buffer) > self._rolling_window:
                self._rolling_buffer = self._rolling_buffer[-self._rolling_window :]
            self._update_rolling_threshold()

        # Track
        self._n_predictions += 1
        if flag:
            self._n_alarms += 1

        return CalibGuardResult(
            flag=flag,
            p_value=p_value,
            threshold=threshold,
            score=score_val,
        )

    def predict_batch(
        self, scores: NDArray[np.float64]
    ) -> Tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
        """Predict anomaly flags for a batch of scores.

        Processes scores sequentially (online), updating rolling state.

        Args:
            scores: Anomaly scores of shape (T,).

        Returns:
            Tuple of (flags, p_values, thresholds), each shape (T,).

        Raises:
            RuntimeError: If not fitted.
            ValueError: If scores are invalid.
        """
        if not self._fitted:
            raise RuntimeError("CalibGuard has not been fitted. Call fit() first.")
        scores_arr = np.asarray(scores, dtype=np.float64)
        if scores_arr.ndim != 1:
            raise ValueError(f"scores must be 1D, got ndim={scores_arr.ndim}.")
        if not np.isfinite(scores_arr).all():
            raise ValueError("scores contains non-finite values.")

        n = scores_arr.size
        flags = np.zeros(n, dtype=np.bool_)
        p_values = np.zeros(n, dtype=np.float64)
        thresholds = np.zeros(n, dtype=np.float64)

        for i in range(n):
            result = self.predict(float(scores_arr[i]))
            flags[i] = result.flag
            p_values[i] = result.p_value
            thresholds[i] = result.threshold

        return flags, p_values, thresholds

    def get_stats(self) -> CalibGuardStats:
        """Get current calibration statistics.

        Returns:
            CalibGuardStats with current state.
        """
        n_calib = (
            self._calibration_scores.size if self._calibration_scores is not None else 0
        )
        empirical_far = (
            self._n_alarms / max(1, self._n_predictions)
            if self._n_predictions > 0
            else 0.0
        )
        return CalibGuardStats(
            n_calibration=n_calib,
            alpha=self._alpha,
            fixed_threshold=self._fixed_threshold,
            rolling_threshold=self._rolling_threshold,
            n_alarms=self._n_alarms,
            n_predictions=self._n_predictions,
            empirical_far=empirical_far,
            drift_detected=self._drift_detected,
        )

    def _compute_p_value(self, score: float) -> float:
        """Compute conformal p-value.

        p = (1 + #{i : s_i >= score}) / (n + 1)

        Args:
            score: Test score.

        Returns:
            P-value in (0, 1].
        """
        if self._calibration_scores is None:
            return 1.0

        n = self._calibration_scores.size
        # Number of calibration scores >= test score
        n_geq = int(np.sum(self._calibration_scores >= score))
        return (1.0 + n_geq) / (n + 1.0)

    def _get_current_threshold(self) -> float:
        """Get the current decision threshold.

        Uses rolling threshold if available and no drift detected,
        otherwise falls back to fixed threshold.

        Returns:
            Current threshold.
        """
        if self._rolling_window <= 0:
            return self._fixed_threshold

        if self._drift_detected:
            return self._fixed_threshold

        if self._rolling_threshold is not None:
            return self._rolling_threshold

        return self._fixed_threshold

    def _update_rolling_threshold(self) -> None:
        """Update rolling threshold from buffer and check for drift."""
        if len(self._rolling_buffer) < max(10, self._rolling_window // 10):
            # Not enough data for reliable rolling estimate
            return

        buffer_arr = np.array(self._rolling_buffer, dtype=np.float64)
        n = buffer_arr.size

        # Rolling conformal quantile
        quantile_index = int(np.ceil((n + 1) * (1 - self._alpha))) - 1
        quantile_index = min(quantile_index, n - 1)
        sorted_buffer = np.sort(buffer_arr)
        self._rolling_threshold = float(sorted_buffer[quantile_index])

        # Drift detection: compare rolling mean to calibration mean
        rolling_mean = float(np.mean(buffer_arr))
        deviation = abs(rolling_mean - self._calib_mean) / self._calib_std

        if deviation > self._drift_sigma:
            if not self._drift_detected:
                LOGGER.warning(
                    "Drift detected: rolling_mean=%.4f, calib_mean=%.4f, "
                    "deviation=%.2f sigma. Falling back to fixed threshold.",
                    rolling_mean,
                    self._calib_mean,
                    deviation,
                )
            self._drift_detected = True
        else:
            self._drift_detected = False


def compute_far_at_alpha(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    alpha: float = 0.01,
    calibration_ratio: float = 0.5,
) -> dict[str, float]:
    """Compute FAR guarantee metrics using split conformal.

    Splits scores into calibration (normal portion) and test,
    then evaluates FAR control.

    Args:
        scores: Anomaly scores shape (T,).
        labels: Binary labels shape (T,). 0=normal, 1=anomaly.
        alpha: Target FAR.
        calibration_ratio: Fraction of normal scores to use for calibration.

    Returns:
        Dict with keys: target_far, actual_far, coverage, threshold,
        n_calibration, n_test_normal, n_test_anomaly.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and labels must be 1D with same shape.")

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    normal_scores = scores[normal_mask]
    anomaly_scores = scores[anomaly_mask]

    if normal_scores.size == 0:
        raise ValueError("No normal samples found.")

    # Split normal scores: first portion for calibration, rest for test
    n_calib = max(1, int(normal_scores.size * calibration_ratio))
    calib_scores = normal_scores[:n_calib]
    test_normal_scores = normal_scores[n_calib:]

    # Fit CalibGuard
    guard = CalibGuard(alpha=alpha, rolling_window=0)
    guard.fit(calib_scores)

    # Evaluate on test normal (FAR)
    if test_normal_scores.size > 0:
        flags_normal, _, _ = guard.predict_batch(test_normal_scores)
        actual_far = float(np.mean(flags_normal))
    else:
        actual_far = 0.0

    # Evaluate on anomaly (coverage / detection rate)
    if anomaly_scores.size > 0:
        flags_anomaly, _, _ = guard.predict_batch(anomaly_scores)
        coverage = float(np.mean(flags_anomaly))
    else:
        coverage = 0.0

    return {
        "target_far": alpha,
        "actual_far": actual_far,
        "coverage": coverage,
        "threshold": guard.fixed_threshold,
        "n_calibration": n_calib,
        "n_test_normal": test_normal_scores.size,
        "n_test_anomaly": anomaly_scores.size,
    }
