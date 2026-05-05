"""Per-patch Mahalanobis anomaly scorer.

Fits a separate Ledoit-Wolf Mahalanobis model for each patch position,
preserving spatial anomaly information that is lost when mean-pooling all
256 patch tokens into a single vector.

Usage::

    scorer = PerPatchMahalanobisScorer(aggregation="topk", topk=10)
    scorer.fit(train_tokens)          # (N_train, N_patches, D_hidden)
    scores = scorer.score(test_tokens)  # (N_test,)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.covariance import LedoitWolf

LOGGER = logging.getLogger(__name__)

AggregationMode = Literal["max", "mean", "topk"]


class PerPatchMahalanobisScorer:
    """Per-patch Mahalanobis distance anomaly scorer.

    Fits one Ledoit-Wolf Mahalanobis model per patch position, then
    aggregates the per-patch distances into a single window-level anomaly
    score.

    Args:
        aggregation: How to aggregate per-patch distances into a window
            score. One of ``"max"``, ``"mean"``, or ``"topk"``.
        topk: Number of patches to average when ``aggregation="topk"``.
            Ignored for other modes.
        eps: Small numerical stability constant.
    """

    def __init__(
        self,
        aggregation: AggregationMode = "topk",
        topk: int = 10,
        eps: float = 1e-8,
    ) -> None:
        if aggregation not in ("max", "mean", "topk"):
            raise ValueError(
                f"aggregation must be one of 'max', 'mean', 'topk', got '{aggregation}'."
            )
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}.")
        self.aggregation: AggregationMode = aggregation
        self.topk = topk
        self.eps = eps
        # Per-patch fitted parameters; list of length N_patches after fit().
        self._patch_means: list[npt.NDArray[np.float64]] = []
        self._patch_precisions: list[npt.NDArray[np.float64]] = []
        self._n_patches: int = 0
        self._hidden_dim: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_tokens: npt.NDArray[np.float64]) -> None:
        """Fit per-patch Ledoit-Wolf Mahalanobis on training tokens.

        Args:
            train_tokens: Training patch tokens of shape
                ``(N_windows, N_patches, D_hidden)``.

        Raises:
            ValueError: If the input is not 3D or N_windows < 2.
        """
        if train_tokens.ndim != 3:
            raise ValueError(
                f"train_tokens must have shape (N, P, D), got ndim={train_tokens.ndim}."
            )
        n_windows, n_patches, hidden_dim = train_tokens.shape
        if n_windows < 2:
            raise ValueError(
                f"Need at least 2 training windows, got {n_windows}."
            )
        if n_windows < hidden_dim:
            LOGGER.warning(
                "Fewer training windows (%d) than hidden dimensions (%d); "
                "Ledoit-Wolf shrinkage will regularise heavily for each patch.",
                n_windows,
                hidden_dim,
            )

        tokens = train_tokens.astype(np.float64)
        means: list[npt.NDArray[np.float64]] = []
        precisions: list[npt.NDArray[np.float64]] = []

        LOGGER.info(
            "PerPatchMahalanobisScorer: fitting %d patch models (N=%d, D=%d)...",
            n_patches,
            n_windows,
            hidden_dim,
        )

        for patch_idx in range(n_patches):
            patch_vecs = tokens[:, patch_idx, :]  # (N_windows, D_hidden)
            mu = patch_vecs.mean(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lw = LedoitWolf().fit(patch_vecs)
            means.append(mu)
            precisions.append(lw.precision_.astype(np.float64))

        self._patch_means = means
        self._patch_precisions = precisions
        self._n_patches = n_patches
        self._hidden_dim = hidden_dim

        LOGGER.info(
            "PerPatchMahalanobisScorer fitted: N_patches=%d, N_windows=%d, D=%d",
            n_patches,
            n_windows,
            hidden_dim,
        )

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(
        self,
        test_tokens: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute aggregated per-patch Mahalanobis distance for each window.

        Args:
            test_tokens: Test patch tokens of shape ``(B, N_patches, D_hidden)``.

        Returns:
            Anomaly scores of shape ``(B,)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: If input shape is incompatible with fitted parameters.
        """
        if not self._patch_means:
            raise RuntimeError("PerPatchMahalanobisScorer has not been fitted yet.")
        if test_tokens.ndim != 3:
            raise ValueError(
                f"test_tokens must have shape (B, P, D), got ndim={test_tokens.ndim}."
            )
        n_windows, n_patches, hidden_dim = test_tokens.shape
        if n_patches != self._n_patches:
            raise ValueError(
                f"Expected {self._n_patches} patches, got {n_patches}."
            )
        if hidden_dim != self._hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self._hidden_dim}, got {hidden_dim}."
            )

        tokens = test_tokens.astype(np.float64)
        # per_patch_dists: (B, N_patches)
        per_patch = self._compute_per_patch_distances(tokens)

        return self._aggregate(per_patch)

    def score_per_patch(
        self,
        test_tokens: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute per-patch Mahalanobis distances without aggregation.

        Args:
            test_tokens: Test patch tokens of shape ``(B, N_patches, D_hidden)``.

        Returns:
            Per-patch distances of shape ``(B, N_patches)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._patch_means:
            raise RuntimeError("PerPatchMahalanobisScorer has not been fitted yet.")
        tokens = test_tokens.astype(np.float64)
        return self._compute_per_patch_distances(tokens)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state for checkpoint storage.

        Returns:
            Dict containing all parameters needed to reconstruct this scorer.
        """
        return {
            "aggregation": self.aggregation,
            "topk": self.topk,
            "eps": self.eps,
            "n_patches": self._n_patches,
            "hidden_dim": self._hidden_dim,
            "patch_means": [m.tolist() for m in self._patch_means],
            "patch_precisions": [p.tolist() for p in self._patch_precisions],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scorer state from a checkpoint dict.

        Args:
            state: Dict previously returned by :meth:`state_dict`.
        """
        self.aggregation = AggregationMode.__args__[0]  # type: ignore[assignment]
        # Validate aggregation string
        agg = str(state["aggregation"])
        if agg not in ("max", "mean", "topk"):
            raise ValueError(f"Invalid aggregation in state_dict: '{agg}'.")
        self.aggregation = agg  # type: ignore[assignment]
        self.topk = int(state["topk"])
        self.eps = float(state["eps"])
        self._n_patches = int(state["n_patches"])
        self._hidden_dim = int(state["hidden_dim"])
        self._patch_means = [
            np.asarray(m, dtype=np.float64) for m in state["patch_means"]
        ]
        self._patch_precisions = [
            np.asarray(p, dtype=np.float64) for p in state["patch_precisions"]
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_per_patch_distances(
        self,
        tokens: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute squared Mahalanobis distance per window per patch.

        Args:
            tokens: Float64 array of shape ``(B, N_patches, D_hidden)``.

        Returns:
            Distances of shape ``(B, N_patches)``.
        """
        n_windows = tokens.shape[0]
        distances = np.empty((n_windows, self._n_patches), dtype=np.float64)

        for patch_idx in range(self._n_patches):
            mu = self._patch_means[patch_idx]          # (D,)
            prec = self._patch_precisions[patch_idx]   # (D, D)
            diff = tokens[:, patch_idx, :] - mu        # (B, D)
            # Squared Mahalanobis: einsum "bd,de,be->b"
            d2: npt.NDArray[np.float64] = np.einsum(
                "bd,de,be->b", diff, prec, diff
            )
            distances[:, patch_idx] = np.maximum(d2, 0.0)

        return distances

    def _aggregate(
        self,
        per_patch: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Aggregate per-patch distances to window-level scores.

        Args:
            per_patch: Array of shape ``(B, N_patches)``.

        Returns:
            Aggregated scores of shape ``(B,)``.
        """
        if self.aggregation == "max":
            return per_patch.max(axis=1)
        if self.aggregation == "mean":
            return per_patch.mean(axis=1)
        # topk: sort descending along patch axis and average top-k
        k = min(self.topk, per_patch.shape[1])
        # partition instead of full sort for efficiency
        if k == per_patch.shape[1]:
            return per_patch.mean(axis=1)
        # np.partition gives the k smallest; we want the k largest
        # use -per_patch so the k smallest become the k largest
        partitioned = np.partition(per_patch, -k, axis=1)[:, -k:]
        return partitioned.mean(axis=1)
