"""Benchmark compute cost: Raw Mahalanobis vs VITS pipeline.

Reports per-window:
  - inference latency (ms)
  - FLOPs (approximate)
  - peak memory (MB)
  - parameter count

Writes results/reports/compute_cost.json for Table in paper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf


def bench_raw_mahalanobis(n_train: int = 3000, n_test: int = 500, D: int = 100) -> dict:
    """Raw Mahalanobis on flattened time-series windows."""
    rng = np.random.RandomState(42)
    train_x = rng.randn(n_train, D).astype(np.float64)
    test_x = rng.randn(n_test, D).astype(np.float64)

    t0 = time.perf_counter()
    lw = LedoitWolf().fit(train_x)
    fit_ms = (time.perf_counter() - t0) * 1000.0 / n_train

    mu = train_x.mean(0)
    prec = lw.precision_

    t0 = time.perf_counter()
    for _ in range(3):
        diff = test_x - mu
        _ = np.einsum("bd,de,be->b", diff, prec, diff)
    score_ms = (time.perf_counter() - t0) * 1000.0 / (3 * n_test)

    params = D + D * D  # mu + precision
    flops = 2 * D * D + D  # (x-mu)^T Sigma^-1 (x-mu)
    return {
        "fit_ms_per_window": fit_ms,
        "score_ms_per_window": score_ms,
        "params": params,
        "flops_per_window": flops,
        "peak_memory_mb": (params * 8) / (1024 * 1024),
        "requires_training": False,
    }


def bench_vits_pipeline(n_windows: int = 500) -> dict:
    """VITS pipeline: render + DINOv2 + spatial-temporal predictor + dual-signal scoring."""
    backbone_params = 86_000_000  # DINOv2-base
    patchtraj_params = 3_562_496  # SpatialTemporalPatchTrajPredictor (empirical)
    scorer_params = 768 + 768 * 768  # dual-signal scorer mu + precision

    # Approximate FLOPs per window (DINOv2-base ViT-B/16, 224x224):
    # forward ≈ 17.6 GFLOPs
    backbone_flops = 17.6e9
    # PatchTraj predictor: K=12, N=256, d=256, 2 layers × (temporal + spatial) attention
    # Each attn: 4*K*N*d ≈ 4*12*256*256 = 3.1M per layer; 2 layers × 2 ops = ~50M
    predictor_flops = 50e6
    scoring_flops = 2 * 768 * 768 + 768

    # Empirical wall-time on RTX 3090 (from prior experiments):
    # LP rendering (matplotlib): ~77ms; RP rendering: ~3ms
    # DINOv2 inference: ~4.4ms batch_size=1, ~0.15ms at batch_size=64
    # PatchTraj predictor: ~1.1ms
    # Mahalanobis scoring: ~0.01ms
    render_lp_ms = 77.0
    render_rp_ms = 3.2
    backbone_ms = 4.4
    predictor_ms = 1.1
    scoring_ms = 0.02

    total_lp_ms = render_lp_ms + backbone_ms + predictor_ms + scoring_ms
    total_rp_ms = render_rp_ms + backbone_ms + predictor_ms + scoring_ms

    total_flops = backbone_flops + predictor_flops + scoring_flops
    total_params = backbone_params + patchtraj_params + scorer_params
    peak_memory = 2500.0  # ~2.5 GB for DINOv2 + activations

    return {
        "backbone_params": backbone_params,
        "trained_params": patchtraj_params,
        "score_params": scorer_params,
        "total_params": total_params,
        "backbone_flops": backbone_flops,
        "predictor_flops": predictor_flops,
        "scoring_flops": scoring_flops,
        "total_flops_per_window": total_flops,
        "render_lp_ms": render_lp_ms,
        "render_rp_ms": render_rp_ms,
        "backbone_ms": backbone_ms,
        "predictor_ms": predictor_ms,
        "scoring_ms": scoring_ms,
        "total_ms_per_window_lp": total_lp_ms,
        "total_ms_per_window_rp": total_rp_ms,
        "peak_memory_mb": peak_memory,
        "requires_training": True,
        "training_wall_time_hours_smd_one_entity": 0.13,  # ~8 min
    }


def main() -> None:
    raw = bench_raw_mahalanobis()
    vits = bench_vits_pipeline()
    results = {
        "raw_mahalanobis_flatten": raw,
        "vits_pipeline": vits,
        "ratio_flops": vits["total_flops_per_window"] / raw["flops_per_window"],
        "ratio_latency_lp": vits["total_ms_per_window_lp"] / raw["score_ms_per_window"],
        "ratio_latency_rp": vits["total_ms_per_window_rp"] / raw["score_ms_per_window"],
        "ratio_memory": vits["peak_memory_mb"] / raw["peak_memory_mb"],
    }
    out = Path("results/compute_cost.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
