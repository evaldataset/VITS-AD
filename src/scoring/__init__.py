from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Any

__all__ = [
    "MultiScaleEnsemble",
    "MultiScaleScoreEntry",
    "compute_hybrid_score",
    "fuse_scores",
]


def __getattr__(name: str) -> Any:
    if name in {"MultiScaleEnsemble", "MultiScaleScoreEntry"}:
        from src.scoring.multiscale_ensemble import (
            MultiScaleEnsemble,
            MultiScaleScoreEntry,
        )

        exported = {
            "MultiScaleEnsemble": MultiScaleEnsemble,
            "MultiScaleScoreEntry": MultiScaleScoreEntry,
        }
        return exported[name]

    if name == "fuse_scores":
        from src.scoring.score_fusion import fuse_scores

        return fuse_scores

    if name == "compute_hybrid_score":
        from src.scoring.hybrid_scorer import compute_hybrid_score

        return compute_hybrid_score

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
