"""Dual-signal anomaly scorer combining trajectory prediction and distributional distance.

PatchTraj captures *dynamic* anomalies (temporal trajectory deviations),
while the distributional distance captures *static* anomalies (unusual
feature distributions).  Fusing both signals improves robustness on datasets
where one signal alone is insufficient (e.g., PSM, MSL).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.covariance import LedoitWolf

LOGGER = logging.getLogger(__name__)


class DualSignalScorer:
    """Fuse trajectory residual scores with Mahalanobis distributional distance.

    The distributional distance is computed as the Mahalanobis distance between
    the mean-pooled patch tokens of a test window and the training distribution
    (parameterised by Ledoit-Wolf shrinkage covariance).

    Args:
        alpha: Weight for the trajectory signal in ``[0, 1]``.
            ``alpha=1`` uses trajectory only; ``alpha=0`` uses distribution only.
        eps: Small constant for numerical stability in z-score normalisation.
    """

    def __init__(self, alpha: float = 0.1, eps: float = 1e-8) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        self.alpha = alpha
        self.eps = eps
        self._train_mu: npt.NDArray[np.float64] | None = None
        self._precision: npt.NDArray[np.float64] | None = None
        # Reference (train/val) statistics for leak-free z-scoring at fuse time.
        # Set via :meth:`fit_normalizers` (or :meth:`load_state_dict`).  When
        # left as ``None`` the fuse step falls back to the legacy in-batch
        # z-score, which is transductive and only retained for backwards
        # compatibility with old checkpoints; new runs must call
        # :meth:`fit_normalizers` so the deployment-time fusion is leak-free.
        self._traj_ref_mu: float | None = None
        self._traj_ref_sigma: float | None = None
        self._dist_ref_mu: float | None = None
        self._dist_ref_sigma: float | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_tokens: npt.NDArray[np.float64]) -> None:
        """Fit distributional parameters from training patch tokens.

        Args:
            train_tokens: Training patch tokens of shape ``(T, N, D)``.

        Raises:
            ValueError: If the input is invalid or covariance is singular.
        """
        if train_tokens.ndim != 3:
            raise ValueError(
                f"train_tokens must have shape (T, N, D), got ndim={train_tokens.ndim}."
            )
        num_windows, _, hidden_dim = train_tokens.shape
        if num_windows < hidden_dim:
            LOGGER.warning(
                "Fewer training windows (%d) than hidden dimensions (%d); "
                "Ledoit-Wolf shrinkage will regularise heavily.",
                num_windows,
                hidden_dim,
            )

        pooled = train_tokens.mean(axis=1).astype(np.float64)  # (T, D)
        self._train_mu = pooled.mean(axis=0)  # (D,)

        lw = LedoitWolf().fit(pooled)
        self._precision = lw.precision_.astype(np.float64)  # (D, D)

        LOGGER.info(
            "DualSignalScorer fitted: T=%d, D=%d, shrinkage=%.4f",
            num_windows,
            hidden_dim,
            lw.shrinkage_,
        )

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score_distributional(
        self,
        test_tokens: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute Mahalanobis distance for each test window.

        Args:
            test_tokens: Test patch tokens of shape ``(B, N, D)``.

        Returns:
            Distributional anomaly scores of shape ``(B,)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: If input shape is invalid.
        """
        if self._train_mu is None or self._precision is None:
            raise RuntimeError("DualSignalScorer has not been fitted yet.")
        if test_tokens.ndim != 3:
            raise ValueError(
                f"test_tokens must have shape (B, N, D), got ndim={test_tokens.ndim}."
            )

        pooled = test_tokens.mean(axis=1).astype(np.float64)  # (B, D)
        diff = pooled - self._train_mu  # (B, D)
        # Mahalanobis: (x-mu)^T Sigma^{-1} (x-mu) for each sample
        scores: npt.NDArray[np.float64] = np.einsum(
            "bd,de,be->b", diff, self._precision, diff
        )
        scores = np.maximum(scores, 0.0)
        return scores

    # ------------------------------------------------------------------
    # Fit normalizers (leak-free fuse)
    # ------------------------------------------------------------------

    def fit_normalizers(
        self,
        traj_ref_scores: npt.NDArray[np.float64],
        dist_ref_scores: npt.NDArray[np.float64],
    ) -> None:
        """Freeze (mu, sigma) of reference (train or train-validation) scores.

        These statistics are used at :meth:`fuse` time to z-score test scores,
        eliminating the transductive leak that would arise from z-scoring with
        the test batch's own mean/std.

        Args:
            traj_ref_scores: Reference trajectory residual scores, 1D.
            dist_ref_scores: Reference distributional Mahalanobis scores, 1D.
        """
        if traj_ref_scores.ndim != 1 or dist_ref_scores.ndim != 1:
            raise ValueError("Reference scores must be 1D arrays.")
        self._traj_ref_mu = float(np.mean(traj_ref_scores))
        self._traj_ref_sigma = float(np.std(traj_ref_scores))
        self._dist_ref_mu = float(np.mean(dist_ref_scores))
        self._dist_ref_sigma = float(np.std(dist_ref_scores))
        LOGGER.info(
            "DualSignalScorer normalizers frozen: traj mu=%.4f sigma=%.4f, "
            "dist mu=%.4f sigma=%.4f",
            self._traj_ref_mu, self._traj_ref_sigma,
            self._dist_ref_mu, self._dist_ref_sigma,
        )

    # ------------------------------------------------------------------
    # Fuse
    # ------------------------------------------------------------------

    def fuse(
        self,
        traj_scores: npt.NDArray[np.float64],
        dist_scores: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Fuse trajectory and distributional scores via z-score weighted sum.

        If reference normalizers have been fitted (via
        :meth:`fit_normalizers` or restored from a state dict) the test
        scores are z-scored using the frozen reference statistics, which is
        the deployment-correct leak-free protocol.  Without reference
        normalizers, the function falls back to the legacy in-batch z-score
        (transductive; retained only for backwards compatibility with old
        checkpoints).

        Args:
            traj_scores: Trajectory-based anomaly scores of shape ``(T,)``.
            dist_scores: Distributional anomaly scores of shape ``(T,)``.

        Returns:
            Fused scores of shape ``(T,)``.

        Raises:
            ValueError: If inputs have incompatible shapes.
        """
        if traj_scores.shape != dist_scores.shape:
            raise ValueError(
                f"Score shapes must match: {traj_scores.shape} vs {dist_scores.shape}."
            )
        if traj_scores.ndim != 1:
            raise ValueError(f"Scores must be 1D, got ndim={traj_scores.ndim}.")

        if (
            self._traj_ref_mu is not None
            and self._traj_ref_sigma is not None
            and self._dist_ref_mu is not None
            and self._dist_ref_sigma is not None
        ):
            traj_z = self._zscore_with(
                traj_scores, self._traj_ref_mu, self._traj_ref_sigma
            )
            dist_z = self._zscore_with(
                dist_scores, self._dist_ref_mu, self._dist_ref_sigma
            )
        else:
            LOGGER.warning(
                "DualSignalScorer.fuse() called without fitted normalizers; "
                "falling back to transductive in-batch z-score (legacy path)."
            )
            traj_z = self._zscore(traj_scores)
            dist_z = self._zscore(dist_scores)
        fused = self.alpha * traj_z + (1.0 - self.alpha) * dist_z
        return fused.astype(np.float64)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state for checkpoint storage."""
        return {
            "alpha": self.alpha,
            "eps": self.eps,
            "train_mu": self._train_mu,
            "precision": self._precision,
            "traj_ref_mu": self._traj_ref_mu,
            "traj_ref_sigma": self._traj_ref_sigma,
            "dist_ref_mu": self._dist_ref_mu,
            "dist_ref_sigma": self._dist_ref_sigma,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scorer state from a checkpoint dict."""
        self.alpha = float(state["alpha"])
        self.eps = float(state["eps"])
        mu = state["train_mu"]
        prec = state["precision"]
        self._train_mu = np.asarray(mu, dtype=np.float64) if mu is not None else None
        self._precision = np.asarray(prec, dtype=np.float64) if prec is not None else None
        # Reference normalizers (optional; older checkpoints will not contain them).
        self._traj_ref_mu = (
            float(state["traj_ref_mu"]) if state.get("traj_ref_mu") is not None else None
        )
        self._traj_ref_sigma = (
            float(state["traj_ref_sigma"]) if state.get("traj_ref_sigma") is not None else None
        )
        self._dist_ref_mu = (
            float(state["dist_ref_mu"]) if state.get("dist_ref_mu") is not None else None
        )
        self._dist_ref_sigma = (
            float(state["dist_ref_sigma"]) if state.get("dist_ref_sigma") is not None else None
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _zscore(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Z-score normalise (transductive, in-batch); legacy fallback only."""
        mu = float(np.mean(x))
        std = float(np.std(x))
        if std < self.eps:
            return np.zeros_like(x)
        return (x - mu) / std

    def _zscore_with(
        self, x: npt.NDArray[np.float64], mu: float, sigma: float
    ) -> npt.NDArray[np.float64]:
        """Z-score with frozen reference (mu, sigma); leak-free at inference."""
        if sigma < self.eps:
            return np.zeros_like(x)
        return (x - mu) / sigma


class PerPatchScorer:
    """PaDiM-style per-patch-position Mahalanobis anomaly scorer.

    Instead of mean-pooling all patches into a single vector, this scorer
    fits a separate multivariate Gaussian (μ_p, Σ_p) at each patch position
    and computes the Mahalanobis distance independently, then aggregates
    across patches.  This preserves spatial anomaly information that pooling
    discards.

    Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework
    for Anomaly Detection and Localization", ICPR 2021.

    Args:
        aggregation: How to aggregate per-patch distances into a window score.
            One of ``"mean"``, ``"max"``, ``"p95"``.
        max_dim: If the token dimension D exceeds this, apply random projection
            to reduce dimensionality before covariance estimation (PaDiM trick).
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        aggregation: str = "mean",
        max_dim: int = 550,
        eps: float = 1e-8,
    ) -> None:
        if aggregation not in ("mean", "max", "p95"):
            raise ValueError(
                f"aggregation must be 'mean', 'max', or 'p95', got '{aggregation}'."
            )
        self.aggregation = aggregation
        self.max_dim = max_dim
        self.eps = eps
        self._num_patches: int = 0
        self._proj: npt.NDArray[np.float64] | None = None  # (D, d) random projection
        self._mus: npt.NDArray[np.float64] | None = None  # (N, d)
        self._precisions: npt.NDArray[np.float64] | None = None  # (N, d, d)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_tokens: npt.NDArray[np.float64]) -> None:
        """Fit per-patch Gaussian parameters from training tokens.

        Args:
            train_tokens: Training patch tokens ``(T, N, D)``.
        """
        if train_tokens.ndim != 3:
            raise ValueError(
                f"train_tokens must have shape (T, N, D), got ndim={train_tokens.ndim}."
            )
        num_windows, num_patches, hidden_dim = train_tokens.shape
        self._num_patches = num_patches
        tokens = train_tokens.astype(np.float64)

        # Optional random projection for high-D tokens
        if hidden_dim > self.max_dim:
            rng = np.random.RandomState(42)
            proj = rng.randn(hidden_dim, self.max_dim).astype(np.float64)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            self._proj = proj
            tokens = np.einsum("tnd,dD->tnD", tokens, proj)
            d = self.max_dim
            LOGGER.info(
                "PerPatchScorer: random projection %d -> %d", hidden_dim, d
            )
        else:
            self._proj = None
            d = hidden_dim

        # Fit per-patch Ledoit-Wolf
        mus = np.empty((num_patches, d), dtype=np.float64)
        precisions = np.empty((num_patches, d, d), dtype=np.float64)

        for p in range(num_patches):
            patch_data = tokens[:, p, :]  # (T, d)
            lw = LedoitWolf().fit(patch_data)
            mus[p] = patch_data.mean(axis=0)
            precisions[p] = lw.precision_.astype(np.float64)

        self._mus = mus
        self._precisions = precisions
        LOGGER.info(
            "PerPatchScorer fitted: T=%d, N=%d, d=%d",
            num_windows, num_patches, d,
        )

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(
        self,
        test_tokens: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute per-patch Mahalanobis distance, aggregated per window.

        Args:
            test_tokens: Test patch tokens ``(B, N, D)``.

        Returns:
            Anomaly scores ``(B,)``.
        """
        if self._mus is None or self._precisions is None:
            raise RuntimeError("PerPatchScorer has not been fitted yet.")
        if test_tokens.ndim != 3:
            raise ValueError(
                f"test_tokens must be 3D (B, N, D), got ndim={test_tokens.ndim}."
            )

        tokens = test_tokens.astype(np.float64)
        if self._proj is not None:
            tokens = np.einsum("bnd,dD->bnD", tokens, self._proj)

        B, N, d = tokens.shape
        # Per-patch Mahalanobis: (B, N)
        patch_scores = np.empty((B, N), dtype=np.float64)
        for p in range(N):
            diff = tokens[:, p, :] - self._mus[p]  # (B, d)
            # (B,) = einsum('bd,de,be->b')
            maha = np.einsum("bd,de,be->b", diff, self._precisions[p], diff)
            patch_scores[:, p] = np.maximum(maha, 0.0)

        # Aggregate
        if self.aggregation == "mean":
            return patch_scores.mean(axis=1)
        if self.aggregation == "max":
            return patch_scores.max(axis=1)
        # p95
        return np.percentile(patch_scores, 95, axis=1)

    def score_patchmap(
        self,
        test_tokens: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return per-patch scores without aggregation for visualization.

        Args:
            test_tokens: ``(B, N, D)``.

        Returns:
            Per-patch scores ``(B, N)``.
        """
        if self._mus is None or self._precisions is None:
            raise RuntimeError("PerPatchScorer has not been fitted yet.")

        tokens = test_tokens.astype(np.float64)
        if self._proj is not None:
            tokens = np.einsum("bnd,dD->bnD", tokens, self._proj)

        B, N, d = tokens.shape
        patch_scores = np.empty((B, N), dtype=np.float64)
        for p in range(N):
            diff = tokens[:, p, :] - self._mus[p]
            maha = np.einsum("bd,de,be->b", diff, self._precisions[p], diff)
            patch_scores[:, p] = np.maximum(maha, 0.0)
        return patch_scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state."""
        return {
            "aggregation": self.aggregation,
            "max_dim": self.max_dim,
            "eps": self.eps,
            "num_patches": self._num_patches,
            "proj": self._proj,
            "mus": self._mus,
            "precisions": self._precisions,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.aggregation = str(state["aggregation"])
        self.max_dim = int(state["max_dim"])
        self.eps = float(state["eps"])
        self._num_patches = int(state["num_patches"])
        proj = state["proj"]
        self._proj = np.asarray(proj, dtype=np.float64) if proj is not None else None
        self._mus = np.asarray(state["mus"], dtype=np.float64)
        self._precisions = np.asarray(state["precisions"], dtype=np.float64)
