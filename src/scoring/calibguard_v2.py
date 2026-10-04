"""CalibGuard v2: adaptive conformal FAR control with normalization.

Improvements over v1:
1. Per-entity z-score normalization before conformal calibration.
2. Higher default calibration ratio support via evaluation helper.
3. Optional Bonferroni correction for multiple per-entity tests.
4. Adaptive Conformal Inference (ACI) alpha updates online.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibGuardV2Result:
    """Result from CalibGuard v2 prediction.

    Attributes:
        flag: Boolean anomaly flag (True = anomaly).
        p_value: Conformal p-value in [0, 1]. Lower = more anomalous.
        threshold: Current decision threshold in raw score space.
        score: Input anomaly score in raw score space.
    """

    flag: bool
    p_value: float
    threshold: float
    score: float


@dataclass
class CalibGuardV2Stats:
    """Calibration statistics for monitoring.

    Attributes:
        n_calibration: Number of calibration samples.
        alpha: Target FAR.
        alpha_t: Current adaptive FAR target used by ACI.
        effective_alpha: FAR after multiple-testing adjustment.
        fixed_threshold: Fixed conformal threshold in raw score space.
        rolling_threshold: Current rolling threshold in raw score space.
        normalization_mean: Calibration mean used for z-score normalization.
        normalization_std: Calibration std used for z-score normalization.
        n_alarms: Total alarms raised.
        n_predictions: Total predictions made.
        empirical_far: Observed alarm rate.
        drift_detected: Whether normalized rolling mean drift was detected.
    """

    n_calibration: int
    alpha: float
    alpha_t: float
    effective_alpha: float
    fixed_threshold: float
    rolling_threshold: Optional[float]
    normalization_mean: float
    normalization_std: float
    n_alarms: int
    n_predictions: int
    empirical_far: float
    drift_detected: bool


class CalibGuardV2:
    """Adaptive conformal anomaly detector with per-entity normalization.

    Args:
        alpha: Target false alarm rate in (0, 1).
        rolling_window: Rolling calibration window size for adaptive thresholding.
            Set to 0 to disable rolling mode.
        drift_sigma: Drift trigger threshold on normalized rolling mean.
        use_aci: Whether to enable adaptive conformal inference updates.
        aci_gamma: Learning rate for alpha updates.
        bonferroni_n_tests: Number of simultaneous tests for Bonferroni correction.
            If <= 1, no correction is applied.
        eps: Stability constant for normalization.

    Raises:
        ValueError: If parameters are invalid.
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
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if rolling_window < 0:
            raise ValueError(
                f"rolling_window must be non-negative, got {rolling_window}."
            )
        if drift_sigma <= 0.0:
            raise ValueError(f"drift_sigma must be positive, got {drift_sigma}.")
        if aci_gamma < 0.0:
            raise ValueError(f"aci_gamma must be non-negative, got {aci_gamma}.")
        if bonferroni_n_tests < 1:
            raise ValueError(
                f"bonferroni_n_tests must be >= 1, got {bonferroni_n_tests}."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self._alpha: float = alpha
        self._rolling_window: int = rolling_window
        self._drift_sigma: float = drift_sigma
        self._use_aci: bool = use_aci
        self._aci_gamma: float = aci_gamma
        self._bonferroni_n_tests: int = bonferroni_n_tests
        self._eps: float = eps

        self._calibration_scores_norm: Optional[NDArray[np.float64]] = None
        self._fixed_threshold_norm: float = float("inf")
        self._calib_mean: float = 0.0
        self._calib_std: float = 1.0

        self._rolling_buffer_norm: list[float] = []
        self._rolling_threshold_norm: Optional[float] = None
        self._drift_detected: bool = False

        self._n_alarms: int = 0
        self._n_predictions: int = 0
        self._fitted: bool = False
        self._alpha_t: float = alpha

    @property
    def alpha(self) -> float:
        """Target false alarm rate."""
        return self._alpha

    @property
    def fixed_threshold(self) -> float:
        """Fixed conformal threshold in raw score space."""
        return self._denormalize(self._fixed_threshold_norm)

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    @property
    def rolling_window(self) -> int:
        """Rolling window size (0 = disabled)."""
        return self._rolling_window

    def fit(self, calibration_scores: NDArray[np.float64]) -> CalibGuardV2:
        """Fit calibrator on calibration scores.

        Args:
            calibration_scores: Normal calibration anomaly scores, shape (n,).

        Returns:
            self.

        Raises:
            ValueError: If input scores are invalid.
        """
        scores = np.asarray(calibration_scores, dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError(f"calibration_scores must be 1D, got ndim={scores.ndim}.")
        if scores.size == 0:
            raise ValueError("calibration_scores must be non-empty.")
        if not np.isfinite(scores).all():
            raise ValueError("calibration_scores contains non-finite values.")

        self._calib_mean = float(np.mean(scores))
        self._calib_std = max(float(np.std(scores)), self._eps)
        scores_norm = self._normalize(scores)
        self._calibration_scores_norm = np.sort(scores_norm)

        self._alpha_t = self._alpha
        self._fixed_threshold_norm = self._compute_threshold_norm(
            self._effective_alpha()
        )

        self._rolling_buffer_norm = []
        self._rolling_threshold_norm = None
        self._drift_detected = False
        self._n_alarms = 0
        self._n_predictions = 0
        self._fitted = True

        LOGGER.info(
            "CalibGuardV2 fitted: n=%d alpha=%.4f eff_alpha=%.6f threshold=%.6f mean=%.6f std=%.6f",
            scores.size,
            self._alpha,
            self._effective_alpha(),
            self.fixed_threshold,
            self._calib_mean,
            self._calib_std,
        )
        return self

    def predict(self, score: float) -> CalibGuardV2Result:
        """Predict anomaly flag for one score.

        Args:
            score: Raw anomaly score.

        Returns:
            Prediction result.

        Raises:
            RuntimeError: If calibrator is not fitted.
            ValueError: If score is non-finite.
        """
        if not self._fitted:
            raise RuntimeError("CalibGuardV2 has not been fitted. Call fit() first.")
        if not np.isfinite(score):
            raise ValueError(f"score must be finite, got {score}.")

        score_val = float(score)
        score_norm = self._normalize_scalar(score_val)
        p_value = self._compute_p_value(score_norm)
        threshold_norm = self._get_current_threshold_norm()
        flag = score_norm > threshold_norm

        if self._rolling_window > 0:
            self._rolling_buffer_norm.append(score_norm)
            if len(self._rolling_buffer_norm) > self._rolling_window:
                self._rolling_buffer_norm = self._rolling_buffer_norm[
                    -self._rolling_window :
                ]
            self._update_rolling_threshold()

        self._n_predictions += 1
        if flag:
            self._n_alarms += 1

        if self._use_aci and self._aci_gamma > 0.0:
            err_t = 1.0 if flag else 0.0
            alpha_next = self._alpha_t + self._aci_gamma * (self._alpha - err_t)
            self._alpha_t = max(self._eps, min(1.0 - self._eps, alpha_next))

        return CalibGuardV2Result(
            flag=flag,
            p_value=p_value,
            threshold=self._denormalize(threshold_norm),
            score=score_val,
        )

    def predict_batch(
        self, scores: NDArray[np.float64]
    ) -> Tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
        """Predict anomaly flags for a sequence of scores.

        Args:
            scores: Raw anomaly scores, shape (T,).

        Returns:
            Tuple (flags, p_values, thresholds) of shape (T,).

        Raises:
            RuntimeError: If calibrator is not fitted.
            ValueError: If scores are invalid.
        """
        if not self._fitted:
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

        for i in range(n):
            score_item = cast(np.float64, scores_arr[i])
            result = self.predict(float(score_item))
            flags[i] = result.flag
            p_values[i] = result.p_value
            thresholds[i] = result.threshold

        return flags, p_values, thresholds

    def get_stats(self) -> CalibGuardV2Stats:
        """Get current calibration statistics."""
        n_calib = (
            self._calibration_scores_norm.size
            if self._calibration_scores_norm is not None
            else 0
        )
        empirical_far = self._n_alarms / max(1, self._n_predictions)
        rolling_threshold = (
            self._denormalize(self._rolling_threshold_norm)
            if self._rolling_threshold_norm is not None
            else None
        )
        return CalibGuardV2Stats(
            n_calibration=n_calib,
            alpha=self._alpha,
            alpha_t=self._alpha_t,
            effective_alpha=self._effective_alpha(),
            fixed_threshold=self.fixed_threshold,
            rolling_threshold=rolling_threshold,
            normalization_mean=self._calib_mean,
            normalization_std=self._calib_std,
            n_alarms=self._n_alarms,
            n_predictions=self._n_predictions,
            empirical_far=empirical_far,
            drift_detected=self._drift_detected,
        )

    def _effective_alpha(self) -> float:
        """Compute effective alpha after optional Bonferroni correction."""
        base = self._alpha_t if self._use_aci else self._alpha
        return max(self._eps, min(1.0 - self._eps, base / self._bonferroni_n_tests))

    def _compute_threshold_norm(self, effective_alpha: float) -> float:
        """Compute conformal threshold in normalized score space."""
        if self._calibration_scores_norm is None:
            return float("inf")
        n = self._calibration_scores_norm.size
        q_idx = math.ceil((n + 1) * (1.0 - effective_alpha)) - 1
        q_idx = min(max(q_idx, 0), n - 1)
        threshold = cast(np.float64, self._calibration_scores_norm[q_idx])
        return float(threshold)

    def _compute_p_value(self, score_norm: float) -> float:
        """Compute conformal p-value in normalized score space."""
        if self._calibration_scores_norm is None:
            return 1.0
        n = self._calibration_scores_norm.size
        n_geq = int(np.count_nonzero(self._calibration_scores_norm >= score_norm))
        return (1.0 + n_geq) / (n + 1.0)

    def _get_current_threshold_norm(self) -> float:
        """Get current threshold in normalized space."""
        if self._rolling_window <= 0:
            return self._compute_threshold_norm(self._effective_alpha())
        if self._drift_detected:
            return self._compute_threshold_norm(self._effective_alpha())
        if self._rolling_threshold_norm is not None:
            return self._rolling_threshold_norm
        return self._compute_threshold_norm(self._effective_alpha())

    def _update_rolling_threshold(self) -> None:
        """Update rolling threshold in normalized space and check drift."""
        if len(self._rolling_buffer_norm) < max(10, self._rolling_window // 10):
            return

        buffer_arr = np.asarray(self._rolling_buffer_norm, dtype=np.float64)
        n = buffer_arr.size
        q_idx = math.ceil((n + 1) * (1.0 - self._effective_alpha())) - 1
        q_idx = min(max(q_idx, 0), n - 1)
        sorted_buffer = np.sort(buffer_arr)
        rolling_threshold = cast(np.float64, sorted_buffer[q_idx])
        self._rolling_threshold_norm = float(rolling_threshold)

        rolling_mean = float(np.mean(buffer_arr))
        self._drift_detected = abs(rolling_mean) > self._drift_sigma
        if self._drift_detected:
            LOGGER.warning(
                "CalibGuardV2 drift detected in normalized space: mean=%.4f sigma=%.2f",
                rolling_mean,
                self._drift_sigma,
            )

    def _normalize(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply z-score normalization."""
        return (scores - self._calib_mean) / (self._calib_std + self._eps)

    def _normalize_scalar(self, score: float) -> float:
        """Apply z-score normalization to one value."""
        return float((score - self._calib_mean) / (self._calib_std + self._eps))

    def _denormalize(self, score_norm: float) -> float:
        """Map normalized score back to raw score space."""
        return float(score_norm * (self._calib_std + self._eps) + self._calib_mean)


def compute_far_at_alpha_v2(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    alpha: float = 0.01,
    calibration_ratio: float = 0.8,
    use_aci: bool = True,
    aci_gamma: float = 0.01,
    bonferroni_n_tests: int = 1,
    rolling_window: int = 0,
) -> dict[str, float | int]:
    """Compute FAR/coverage metrics with CalibGuard v2 split conformal.

    Args:
        scores: Raw anomaly scores shape (T,).
        labels: Binary labels shape (T,), where 0=normal and 1=anomaly.
        alpha: Target FAR.
        calibration_ratio: Fraction of normal scores used for calibration.
        use_aci: Whether to enable ACI updates online.
        aci_gamma: ACI learning rate.
        bonferroni_n_tests: Number of simultaneous tests for correction.
        rolling_window: Rolling window size for adaptive thresholding.

    Returns:
        Dictionary with FAR control and detection metrics.

    Raises:
        ValueError: If inputs are invalid.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)

    if scores_arr.shape != labels_arr.shape or scores_arr.ndim != 1:
        raise ValueError("scores and labels must be 1D with same shape.")
    if not 0.0 < calibration_ratio < 1.0:
        raise ValueError(
            f"calibration_ratio must be in (0, 1), got {calibration_ratio}."
        )

    normal_scores = scores_arr[labels_arr == 0]
    anomaly_scores = scores_arr[labels_arr == 1]
    if normal_scores.size == 0:
        raise ValueError("No normal samples found.")

    n_calib = max(1, int(normal_scores.size * calibration_ratio))
    n_calib = min(n_calib, normal_scores.size - 1) if normal_scores.size > 1 else 1
    calib_scores = normal_scores[:n_calib]
    test_normal_scores = normal_scores[n_calib:]

    guard = CalibGuardV2(
        alpha=alpha,
        rolling_window=rolling_window,
        use_aci=use_aci,
        aci_gamma=aci_gamma,
        bonferroni_n_tests=bonferroni_n_tests,
    )
    guard = guard.fit(calib_scores)

    if test_normal_scores.size > 0:
        flags_normal, _, _ = guard.predict_batch(test_normal_scores)
        actual_far = float(np.mean(flags_normal))
    else:
        actual_far = 0.0

    if anomaly_scores.size > 0:
        flags_anomaly, _, _ = guard.predict_batch(anomaly_scores)
        coverage = float(np.mean(flags_anomaly))
    else:
        coverage = 0.0

    stats = guard.get_stats()
    return {
        "target_far": alpha,
        "effective_alpha": stats.effective_alpha,
        "actual_far": actual_far,
        "coverage": coverage,
        "threshold": stats.fixed_threshold,
        "normalization_mean": stats.normalization_mean,
        "normalization_std": stats.normalization_std,
        "alpha_t": stats.alpha_t,
        "n_calibration": n_calib,
        "n_test_normal": int(test_normal_scores.size),
        "n_test_anomaly": int(anomaly_scores.size),
    }
