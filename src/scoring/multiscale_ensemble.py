from __future__ import annotations

# pyright: reportMissingImports=false

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from src.scoring.patchtraj_scorer import normalize_scores

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiScaleScoreEntry:
    """Loaded anomaly scores for one window size and renderer.

    Args:
        window_size: Sliding window size used to generate the scores.
        renderer: Renderer name such as ``"line_plot"``.
        scores: Anomaly scores of shape ``(T,)``.
        labels: Optional binary labels of shape ``(T,)``.
        path: Source directory that contains the score artifacts.
    """

    window_size: int
    renderer: str
    scores: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int64] | None
    path: Path


class MultiScaleEnsemble:
    """Fuse anomaly scores from multiple PatchTraj scales.

    Supports right-aligned fusion across different window sizes so the final
    scores stay temporally consistent with the most recent timesteps.

    Args:
        window_sizes: Supported window sizes to load and fuse.
    """

    SUPPORTED_METHODS: tuple[str, ...] = ("rank_weighted", "zscore_weighted", "mean")

    def __init__(self, window_sizes: Sequence[int] = (50, 100, 200)) -> None:
        normalized_window_sizes = tuple(int(window_size) for window_size in window_sizes)
        if len(normalized_window_sizes) == 0:
            raise ValueError("window_sizes must contain at least one value.")
        if any(window_size <= 0 for window_size in normalized_window_sizes):
            raise ValueError(f"window_sizes must be positive, got {normalized_window_sizes}.")
        self._window_sizes: tuple[int, ...] = normalized_window_sizes

    @property
    def window_sizes(self) -> tuple[int, ...]:
        """Return configured window sizes."""
        return self._window_sizes

    def find_score_entries(
        self,
        results_dir: Path,
        entity: str | None = None,
        renderers: Sequence[str] | None = None,
    ) -> list[MultiScaleScoreEntry]:
        """Load score artifacts from a multi-scale results directory.

        Args:
            results_dir: Results root or entity directory.
            entity: Optional entity name when ``results_dir`` is the dataset root.
            renderers: Optional renderer allowlist.

        Returns:
            Loaded score entries sorted by window size and renderer.

        Raises:
            FileNotFoundError: If no score artifacts are found.
            ValueError: If a loaded score artifact is invalid.
        """
        entity_dir = self._resolve_entity_dir(results_dir=results_dir, entity=entity)
        allowed_renderers = None
        if renderers is not None:
            allowed_renderers = {renderer.strip() for renderer in renderers if renderer.strip()}

        entries: list[MultiScaleScoreEntry] = []
        for window_size in self._window_sizes:
            window_dir = entity_dir / f"w{window_size}"
            if not window_dir.exists():
                LOGGER.warning("Skipping missing multi-scale directory: %s", window_dir)
                continue

            for renderer_dir in sorted(path for path in window_dir.iterdir() if path.is_dir()):
                if allowed_renderers is not None and renderer_dir.name not in allowed_renderers:
                    continue

                scores_path = renderer_dir / "scores.npy"
                labels_path = renderer_dir / "labels.npy"
                if not scores_path.exists():
                    LOGGER.warning("Skipping missing score file: %s", scores_path)
                    continue

                scores = self._validate_scores(
                    np.load(scores_path).astype(np.float64, copy=False),
                    path=scores_path,
                )
                labels: npt.NDArray[np.int64] | None = None
                if labels_path.exists():
                    labels = self._validate_labels(
                        np.load(labels_path).astype(np.int64, copy=False),
                        scores=scores,
                        path=labels_path,
                    )

                entries.append(
                    MultiScaleScoreEntry(
                        window_size=window_size,
                        renderer=renderer_dir.name,
                        scores=scores,
                        labels=labels,
                        path=renderer_dir,
                    )
                )

        if len(entries) == 0:
            raise FileNotFoundError(
                f"No multi-scale score artifacts found in '{entity_dir}'."
            )
        return entries

    def right_align(
        self,
        entries: Sequence[MultiScaleScoreEntry],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64] | None]:
        """Right-align score sequences to a common tail length.

        Args:
            entries: Loaded score entries.

        Returns:
            Tuple of stacked aligned scores with shape ``(V, T)`` and optional
            aligned labels with shape ``(T,)``.

        Raises:
            ValueError: If no entries are provided.
        """
        if len(entries) == 0:
            raise ValueError("entries must contain at least one score source.")

        min_length = min(entry.scores.shape[0] for entry in entries)
        aligned_scores = np.stack(
            [entry.scores[-min_length:] for entry in entries],
            axis=0,
        ).astype(np.float64, copy=False)

        labeled_entries = [entry for entry in entries if entry.labels is not None]
        if len(labeled_entries) == 0:
            return aligned_scores, None

        shortest_labeled_entry = min(labeled_entries, key=lambda entry: entry.scores.shape[0])
        if shortest_labeled_entry.labels is None:
            return aligned_scores, None
        return aligned_scores, shortest_labeled_entry.labels[-min_length:]

    def combine(
        self,
        entries: Sequence[MultiScaleScoreEntry],
        method: str = "rank_weighted",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64] | None]:
        """Fuse multiple score sequences into a single ensemble score.

        Args:
            entries: Loaded score entries.
            method: One of ``"rank_weighted"``, ``"zscore_weighted"``, or ``"mean"``.

        Returns:
            Tuple of ensemble scores and optional aligned labels.

        Raises:
            ValueError: If inputs are invalid.
        """
        aligned_scores, aligned_labels = self.right_align(entries)
        fused = self.fuse(aligned_scores, method=method)
        return fused, aligned_labels

    def fuse(
        self,
        aligned_scores: npt.NDArray[np.float64],
        method: str = "rank_weighted",
    ) -> npt.NDArray[np.float64]:
        """Fuse already aligned score sequences.

        Args:
            aligned_scores: Score matrix of shape ``(V, T)``.
            method: Fusion method.

        Returns:
            Ensemble score sequence of shape ``(T,)``.

        Raises:
            ValueError: If inputs are invalid.
        """
        if aligned_scores.ndim != 2:
            raise ValueError(
                f"aligned_scores must have shape (V, T), got ndim={aligned_scores.ndim}."
            )
        if aligned_scores.shape[0] == 0 or aligned_scores.shape[1] == 0:
            raise ValueError(
                f"aligned_scores must be non-empty, got shape={aligned_scores.shape}."
            )
        if not np.isfinite(aligned_scores).all():
            raise ValueError("aligned_scores contains non-finite values.")

        method_normalized = self._normalize_method_name(method)
        n_views, common_length = aligned_scores.shape

        if method_normalized == "rank_weighted":
            ranked_scores = np.zeros_like(aligned_scores, dtype=np.float64)
            for index in range(n_views):
                ranked_scores[index] = rankdata(aligned_scores[index]) / common_length
            return np.mean(ranked_scores, axis=0).astype(np.float64, copy=False)

        if method_normalized == "zscore_weighted":
            normalized_views = np.stack(
                [normalize_scores(view, method="zscore") for view in aligned_scores],
                axis=0,
            )
            return np.mean(normalized_views, axis=0).astype(np.float64, copy=False)

        normalized_views = np.stack(
            [normalize_scores(view, method="minmax") for view in aligned_scores],
            axis=0,
        )
        return np.mean(normalized_views, axis=0).astype(np.float64, copy=False)

    def _resolve_entity_dir(self, results_dir: Path, entity: str | None) -> Path:
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_path}")

        direct_entity_dir = results_path / entity if entity else results_path
        if direct_entity_dir.exists() and self._has_window_dirs(direct_entity_dir):
            return direct_entity_dir
        if results_path.exists() and self._has_window_dirs(results_path):
            return results_path
        raise FileNotFoundError(
            f"Could not resolve multi-scale entity directory from '{results_path}'"
            + ("." if entity is None else f" and entity '{entity}'.")
        )

    @staticmethod
    def _has_window_dirs(path: Path) -> bool:
        if not path.is_dir():
            return False
        return any(child.is_dir() and child.name.startswith("w") for child in path.iterdir())

    def _normalize_method_name(self, method: str) -> str:
        method_normalized = method.strip().lower()
        if method_normalized not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported multi-scale method '{method}'. Expected one of: {list(self.SUPPORTED_METHODS)}."
            )
        return method_normalized

    @staticmethod
    def _validate_scores(
        scores: npt.NDArray[np.float64], path: Path
    ) -> npt.NDArray[np.float64]:
        if scores.ndim != 1:
            raise ValueError(f"scores at '{path}' must be 1D, got shape={scores.shape}.")
        if scores.size == 0:
            raise ValueError(f"scores at '{path}' must contain at least one value.")
        if not np.isfinite(scores).all():
            raise ValueError(f"scores at '{path}' contain non-finite values.")
        return scores

    @staticmethod
    def _validate_labels(
        labels: npt.NDArray[np.int64],
        scores: npt.NDArray[np.float64],
        path: Path,
    ) -> npt.NDArray[np.int64]:
        if labels.ndim != 1:
            raise ValueError(f"labels at '{path}' must be 1D, got shape={labels.shape}.")
        if labels.shape[0] != scores.shape[0]:
            raise ValueError(
                f"labels at '{path}' must match scores length {scores.shape[0]}, got {labels.shape[0]}."
            )
        return labels
