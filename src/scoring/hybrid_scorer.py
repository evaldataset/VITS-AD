from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.scoring.score_fusion import fuse_scores


def compute_hybrid_score(
    patchtraj_score: npt.ArrayLike,
    recon_score: npt.ArrayLike,
    method: str = "weighted_sum",
    weight: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Fuse PatchTraj and 1D reconstruction scores into one anomaly score."""
    patchtraj = np.asarray(patchtraj_score, dtype=np.float64)
    reconstruction = np.asarray(recon_score, dtype=np.float64)

    if patchtraj.ndim != 1:
        raise ValueError(f"patchtraj_score must be 1D, got ndim={patchtraj.ndim}.")
    if reconstruction.ndim != 1:
        raise ValueError(f"recon_score must be 1D, got ndim={reconstruction.ndim}.")
    if patchtraj.size == 0:
        raise ValueError("patchtraj_score must contain at least one value.")
    if reconstruction.size == 0:
        raise ValueError("recon_score must contain at least one value.")
    if not np.isfinite(patchtraj).all():
        raise ValueError("patchtraj_score contains non-finite values.")
    if not np.isfinite(reconstruction).all():
        raise ValueError("recon_score contains non-finite values.")

    method_normalized = method.strip().lower()
    if method_normalized == "weighted_sum":
        if not np.isfinite(weight):
            raise ValueError(f"weight must be finite, got {weight}.")
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}.")
        return fuse_scores(
            {
                "patchtraj": patchtraj,
                "reconstruction": reconstruction,
            },
            method="weighted_sum",
            weights={"patchtraj": weight, "reconstruction": 1.0 - weight},
        )

    if method_normalized == "max":
        min_length = min(patchtraj.size, reconstruction.size)
        return np.maximum(patchtraj[:min_length], reconstruction[:min_length]).astype(
            np.float64,
            copy=False,
        )

    raise ValueError(
        f"Unsupported hybrid method '{method}'. Expected one of: ['weighted_sum', 'max']."
    )


__all__ = ["compute_hybrid_score"]
