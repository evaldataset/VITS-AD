from __future__ import annotations

# pyright: reportMissingImports=false

from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from src.scoring.multiscale_ensemble import MultiScaleEnsemble
from src.scoring.patchtraj_scorer import normalize_scores


def _write_score_entry(
    root: Path,
    window_size: int,
    renderer: str,
    scores: np.ndarray,
    labels: np.ndarray | None = None,
) -> None:
    entry_dir = root / f"w{window_size}" / renderer
    entry_dir.mkdir(parents=True, exist_ok=True)
    np.save(entry_dir / "scores.npy", scores.astype(np.float64, copy=False))
    if labels is not None:
        np.save(entry_dir / "labels.npy", labels.astype(np.int64, copy=False))


def test_right_align_uses_common_tail_and_shortest_labels(tmp_path: Path) -> None:
    entity_dir = tmp_path / "machine-1-1"
    _write_score_entry(
        entity_dir,
        window_size=50,
        renderer="line_plot",
        scores=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        labels=np.array([0, 0, 0, 1, 1], dtype=np.int64),
    )
    _write_score_entry(
        entity_dir,
        window_size=100,
        renderer="recurrence_plot",
        scores=np.array([10.0, 11.0, 12.0], dtype=np.float64),
        labels=np.array([0, 1, 1], dtype=np.int64),
    )

    ensemble = MultiScaleEnsemble(window_sizes=(50, 100))
    entries = ensemble.find_score_entries(tmp_path, entity="machine-1-1")

    aligned_scores, aligned_labels = ensemble.right_align(entries)

    np.testing.assert_array_equal(
        aligned_scores,
        np.array(
            [
                [2.0, 3.0, 4.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(aligned_labels, np.array([0, 1, 1], dtype=np.int64))


def test_rank_weighted_fusion_averages_per_view_ranks(tmp_path: Path) -> None:
    entity_dir = tmp_path / "machine-1-1"
    scores_a = np.array([2.0, 8.0, 5.0], dtype=np.float64)
    scores_b = np.array([3.0, 1.0, 7.0], dtype=np.float64)
    _write_score_entry(entity_dir, 50, "line_plot", scores_a)
    _write_score_entry(entity_dir, 100, "recurrence_plot", scores_b)

    ensemble = MultiScaleEnsemble(window_sizes=(50, 100))
    entries = ensemble.find_score_entries(tmp_path, entity="machine-1-1")

    fused, labels = ensemble.combine(entries, method="rank_weighted")

    expected = np.mean(
        np.stack(
            [rankdata(scores_a) / scores_a.size, rankdata(scores_b) / scores_b.size],
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_allclose(fused, expected)
    assert labels is None


def test_zscore_weighted_fusion_uses_per_view_zscore_normalization(tmp_path: Path) -> None:
    entity_dir = tmp_path / "machine-1-1"
    scores_a = np.array([1.0, 2.0, 5.0, 8.0], dtype=np.float64)
    scores_b = np.array([10.0, 11.0, 13.0, 20.0], dtype=np.float64)
    _write_score_entry(entity_dir, 50, "line_plot", scores_a)
    _write_score_entry(entity_dir, 100, "recurrence_plot", scores_b)

    ensemble = MultiScaleEnsemble(window_sizes=(50, 100))
    entries = ensemble.find_score_entries(tmp_path, entity="machine-1-1")

    fused, _ = ensemble.combine(entries, method="zscore_weighted")

    expected = np.mean(
        np.stack(
            [
                normalize_scores(scores_a, method="zscore"),
                normalize_scores(scores_b, method="zscore"),
            ],
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_allclose(fused, expected)


def test_mean_fusion_uses_per_view_minmax_normalization(tmp_path: Path) -> None:
    entity_dir = tmp_path / "machine-1-1"
    scores_a = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    scores_b = np.array([10.0, 30.0, 50.0], dtype=np.float64)
    _write_score_entry(entity_dir, 50, "line_plot", scores_a)
    _write_score_entry(entity_dir, 100, "recurrence_plot", scores_b)

    ensemble = MultiScaleEnsemble(window_sizes=(50, 100))
    entries = ensemble.find_score_entries(tmp_path, entity="machine-1-1")

    fused, _ = ensemble.combine(entries, method="mean")

    expected = np.mean(
        np.stack(
            [
                normalize_scores(scores_a, method="minmax"),
                normalize_scores(scores_b, method="minmax"),
            ],
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_allclose(fused, expected)
