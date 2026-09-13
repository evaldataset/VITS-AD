"""Cost scaling measurements (closes review gap W7).

The paper leans on a compute--accuracy argument but reported only a handful of
per-window latencies. This measures how each stage scales along the two axes that
actually matter in our study, using controlled synthetic inputs so the curves are not
confounded by dataset idiosyncrasies:

  * channel count D  -- affects line-plot rendering and, cubically, the flattened
    Ledoit-Wolf control (its feature dimension is W*D), which is why we had to cap it;
  * number of windows -- affects rendering, backbone forward passes and scoring.

Output: results/cost_scaling/cost_scaling.json
"""

from __future__ import annotations

import os

# Pin BLAS threading before numpy is imported. Wall-clock for the Ledoit-Wolf fit
# otherwise varies by up to 3x with machine load as the BLAS scheduler picks
# different thread counts, which makes the scaling curve unreproducible. Single
# threaded measurements are slower in absolute terms but comparable across runs
# and across machines, which is what a reported scaling curve needs to be.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.run_raw_mahalanobis_baseline import RawMahalanobisScorer  # noqa: E402
from src.models.backbone import VisionBackbone  # noqa: E402
from src.rendering.line_plot import render_line_plot  # noqa: E402

LOGGER = logging.getLogger(__name__)

WINDOW = 100
OUT = Path("results/cost_scaling")
D_GRID = (1, 5, 10, 20, 25, 40)
N_WINDOW_GRID = (100, 250, 500, 1000)
REPEATS = 3


def _time(fn, repeats: int = REPEATS) -> float:
    """Median wall-clock seconds over repeats."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def measure_render(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Line-plot rendering cost per window as channel count grows."""
    rows = []
    for d in D_GRID:
        w = rng.standard_normal((WINDOW, d)).astype(np.float32)
        sec = _time(lambda: render_line_plot(w))
        rows.append({"D": d, "ms_per_window": sec * 1e3})
        LOGGER.info("  render D=%-3d %.1f ms/window", d, sec * 1e3)
    return rows


def measure_raw(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Ledoit-Wolf cost for both raw variants as D grows, split into fit and score.

    The split matters for the paper's argument: the flattened control's cost is
    essentially all fit, which is paid once per series rather than per window.
    """
    rows = []
    n_train, n_test = 400, 400
    for d in D_GRID:
        tr = rng.standard_normal((n_train, WINDOW, d))
        te = rng.standard_normal((n_test, WINDOW, d))
        for name, fx in (("mean_pooled", lambda w: w.mean(axis=1)),
                         ("flattened", lambda w: w.reshape(w.shape[0], -1))):
            a, b = fx(tr).astype(np.float64), fx(te).astype(np.float64)
            fits: list[float] = []
            scores: list[float] = []
            for _ in range(REPEATS):
                scorer = RawMahalanobisScorer()
                t0 = time.perf_counter()
                scorer.fit(a)
                fits.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                scorer.score(b)
                scores.append(time.perf_counter() - t0)
            fit_s = float(np.median(fits))
            score_s = float(np.median(scores))
            rows.append({"D": d, "variant": name, "feature_dim": int(a.shape[1]),
                         "fit_seconds": fit_s, "score_seconds": score_s,
                         "seconds": fit_s + score_s,
                         "fit_share": fit_s / (fit_s + score_s),
                         "fit_seconds_all": fits})
            LOGGER.info("  raw %-11s D=%-3d d=%-5d fit=%.2f s score=%.3f s (fit %.1f%%)",
                        name, d, a.shape[1], fit_s, score_s,
                        100 * fit_s / (fit_s + score_s))
    return rows


def measure_backbone(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Frozen backbone forward cost as the number of windows grows."""
    rows = []
    try:
        bb = VisionBackbone("facebook/dinov2-base")
    except Exception as exc:
        LOGGER.warning("backbone unavailable (%s); skipping", str(exc)[:80])
        return rows
    imgs = rng.random((64, 3, 224, 224)).astype(np.float32)
    bb.extract_patch_tokens_from_numpy(imgs)          # warm up
    for n in N_WINDOW_GRID:
        batch = rng.random((min(n, 64), 3, 224, 224)).astype(np.float32)
        n_batches = int(np.ceil(n / 64))
        def run(batch=batch, n_batches=n_batches):
            for _ in range(n_batches):
                bb.extract_patch_tokens_from_numpy(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        sec = _time(run, repeats=1)
        rows.append({"n_windows": n, "seconds": sec,
                     "ms_per_window": sec / n * 1e3})
        LOGGER.info("  backbone n=%-5d %.2f s (%.1f ms/window)", n, sec,
                    sec / n * 1e3)
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(0)
    LOGGER.info("rendering cost vs channel count")
    render = measure_render(rng)
    LOGGER.info("raw Mahalanobis cost vs channel count")
    raw = measure_raw(rng)
    LOGGER.info("frozen backbone cost vs window count")
    backbone = measure_backbone(rng)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_scaling.json").write_text(json.dumps({
        "window": WINDOW, "repeats": REPEATS, "blas_threads": 1,
        "render_vs_channels": render,
        "raw_mahalanobis_vs_channels": raw,
        "backbone_vs_windows": backbone,
    }, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", OUT / "cost_scaling.json")


if __name__ == "__main__":
    main()
