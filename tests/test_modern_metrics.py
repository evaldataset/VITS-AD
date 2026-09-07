"""Tests for the modern TSAD metric adapter (VUS-PR, Affiliation-F, range-F1)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from src.evaluation.modern_metrics import (
    MODERN_KEYS,
    compute_modern_metrics,
    summarize_modern_metrics,
)

pytest.importorskip("TSB_AD", reason="TSB-AD is an optional dependency")


def _make_series(
    length: int = 2000, seed: int = 0, informative: bool = True
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Build a score/label pair with two anomaly segments."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(length, dtype=np.int64)
    labels[800:850] = 1
    labels[1500:1530] = 1
    scores = rng.random(length) * 0.3
    if informative:
        scores[800:850] += 0.7
        scores[1500:1530] += 0.6
    return scores.astype(np.float64), labels


class TestComputeModernMetrics:
    def test_returns_expected_keys(self) -> None:
        scores, labels = _make_series()
        result = compute_modern_metrics(scores, labels, sliding_window=50)
        for key in MODERN_KEYS:
            assert key in result, f"missing metric {key}"
        assert "AUC-ROC" in result
        assert "AUC-PR" in result

    def test_values_in_unit_interval(self) -> None:
        scores, labels = _make_series()
        result = compute_modern_metrics(scores, labels, sliding_window=50)
        for name, value in result.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} outside [0, 1]"

    def test_informative_scores_beat_random(self) -> None:
        good_scores, labels = _make_series(informative=True)
        bad_scores, _ = _make_series(informative=False)
        good = compute_modern_metrics(good_scores, labels, sliding_window=50)
        bad = compute_modern_metrics(bad_scores, labels, sliding_window=50)
        assert good["VUS-PR"] > bad["VUS-PR"]

    def test_reference_auc_matches_own_implementation(self) -> None:
        """The adapter must agree with src.evaluation.metrics on the AUCs."""
        from src.evaluation.metrics import compute_auc_roc

        scores, labels = _make_series()
        result = compute_modern_metrics(scores, labels, sliding_window=50)
        own = compute_auc_roc(scores=scores, labels=labels)
        assert result["AUC-ROC"] == pytest.approx(own, abs=1e-6)

    def test_rejects_length_mismatch(self) -> None:
        scores, labels = _make_series()
        with pytest.raises(ValueError, match="equal length"):
            compute_modern_metrics(scores[:-1], labels)

    def test_rejects_non_binary_labels(self) -> None:
        scores, labels = _make_series()
        labels[0] = 2
        with pytest.raises(ValueError, match="binary"):
            compute_modern_metrics(scores, labels)

    def test_rejects_single_class(self) -> None:
        scores, labels = _make_series()
        with pytest.raises(ValueError, match="both classes"):
            compute_modern_metrics(scores, np.zeros_like(labels))

    def test_rejects_non_positive_window(self) -> None:
        scores, labels = _make_series()
        with pytest.raises(ValueError, match="sliding_window must be positive"):
            compute_modern_metrics(scores, labels, sliding_window=0)


class TestSummarizeModernMetrics:
    def test_mean_and_std(self) -> None:
        per_series = {
            "a": {"VUS-PR": 0.8, "AUC-ROC": 0.9},
            "b": {"VUS-PR": 0.6, "AUC-ROC": 0.7},
        }
        summary = summarize_modern_metrics(per_series)
        assert summary["VUS-PR"]["mean"] == pytest.approx(0.7)
        assert summary["VUS-PR"]["n"] == 2
        assert summary["AUC-ROC"]["mean"] == pytest.approx(0.8)

    def test_handles_missing_metric_in_one_series(self) -> None:
        per_series = {"a": {"VUS-PR": 0.8}, "b": {"AUC-ROC": 0.7}}
        summary = summarize_modern_metrics(per_series)
        assert summary["VUS-PR"]["n"] == 1
        assert summary["AUC-ROC"]["n"] == 1

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            summarize_modern_metrics({})
