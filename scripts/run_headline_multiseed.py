"""Headline-configuration multi-seed runner (reviewer CmYN W2/Q1).

The published multi-seed table used a secondary stability configuration
(fixed alpha=0.5). Reviewer CmYN asked for (a) five seeds on the ACTUAL headline
configuration and (b) confidence intervals on the difference against the
raw-space Mahalanobis baseline, not just VITS stability in isolation.

Headline configuration (verified against results/dinov2-base_{ds}_line_plot_spatial,
which reproduce Table 1 exactly: PSM 0.6615, MSL 0.6058, SMAP 0.7067):
    experiment/patchtraj_spatial  (spatial_attention=true, dual_signal enabled,
                                   auto_alpha=false)
    render=line_plot, scoring.smooth_window=21
    scoring.dual_signal.alpha = 0.2 (PSM) / 0.1 (MSL, SMAP)

Runs 3 datasets x 5 seeds = 15 jobs across the given GPUs, then reports per
dataset: mean +/- std, bootstrap 95% CI, and the paired difference vs. the raw
Mahalanobis (flatten) baseline with its own bootstrap CI.

Output: paper/rebuttal_package/headline_multiseed.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

SEEDS: tuple[int, ...] = (42, 123, 456, 789, 2024)
CONFIG_NAME = "experiment/patchtraj_spatial"
RENDERER = "line_plot"
SMOOTH_WINDOW = 21

# Per-dataset headline alpha (frozen pre-test; see paper Table 1 footnote).
DATASETS: dict[str, dict[str, Any]] = {
    "psm": {"alpha": 0.2, "overrides": ["data=psm", "data.raw_dir=data/raw/psm",
                                        "+data.entity=psm"]},
    "msl": {"alpha": 0.1, "overrides": ["data=msl", "data.raw_dir=data/raw/msl",
                                        "+data.entity=msl"]},
    "smap": {"alpha": 0.1, "overrides": ["data=smap", "data.raw_dir=data/raw/smap",
                                         "+data.entity=smap"]},
}

# Raw-space Mahalanobis (flatten) reference, read from its own artifact.
RAW_METRICS = "results/raw_mahalanobis/{ds}/flattened/metrics.json"

OUT_ROOT = Path("results/headline_multiseed")
OUT_PATH = Path("paper/rebuttal_package/headline_multiseed.json")


@dataclass(frozen=True)
class Job:
    dataset: str
    seed: int


def _out_dir(job: Job) -> Path:
    return OUT_ROOT / job.dataset / f"seed_{job.seed}"


def _overrides(job: Job) -> list[str]:
    spec = DATASETS[job.dataset]
    return [
        *spec["overrides"],
        f"render={RENDERER}",
        f"training.seed={job.seed}",
        f"scoring.smooth_window={SMOOTH_WINDOW}",
        f"scoring.dual_signal.alpha={spec['alpha']}",
        "scoring.dual_signal.enabled=true",
        "scoring.dual_signal.auto_alpha=false",
        f"output_dir={_out_dir(job)}",
        f"data.processed_dir={OUT_ROOT}/{job.dataset}/token_cache",
    ]


def _run(cmd: list[str], env: dict[str, str]) -> None:
    LOGGER.info("$ %s", " ".join(cmd))
    completed = subprocess.run(cmd, env=env, check=False,
                               capture_output=True, text=True)
    if completed.returncode != 0:
        LOGGER.error("FAILED (rc=%d): %s", completed.returncode, " ".join(cmd))
        LOGGER.error("stderr tail:\n%s", "\n".join(
            completed.stderr.strip().splitlines()[-15:]))
        raise RuntimeError(f"command failed: {cmd[1]}")


def run_job(job: Job, gpu: int, python_exe: str, root: Path,
            skip_existing: bool) -> float | None:
    metrics_path = root / _out_dir(job) / "metrics.json"
    if skip_existing and metrics_path.exists():
        LOGGER.info("skip existing: %s", _out_dir(job))
    else:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        ov = _overrides(job)
        _run([python_exe, "scripts/train_patchtraj.py", "--config-name",
              CONFIG_NAME, *ov], env)
        _run([python_exe, "scripts/detect.py", "--config-name",
              CONFIG_NAME, *ov], env)
    if not metrics_path.exists():
        LOGGER.error("no metrics.json for %s", job)
        return None
    return float(json.loads(metrics_path.read_text())["auc_roc"])


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000,
                  seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--gpus", type=str, default="2,3")
    _ = ap.add_argument("--python", type=str, default=sys.executable)
    _ = ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gpus = [int(g) for g in str(args.gpus).split(",") if g.strip()]
    jobs = [Job(ds, s) for ds in DATASETS for s in SEEDS]
    LOGGER.info("%d jobs over GPUs %s", len(jobs), gpus)

    results: dict[str, dict[int, float]] = {ds: {} for ds in DATASETS}
    queues: dict[int, list[Job]] = {g: [] for g in gpus}
    for i, job in enumerate(jobs):
        queues[gpus[i % len(gpus)]].append(job)

    def worker(gpu: int) -> list[tuple[Job, float | None]]:
        out = []
        for job in queues[gpu]:
            try:
                out.append((job, run_job(job, gpu, str(args.python), root,
                                         bool(args.skip_existing))))
            except Exception as exc:  # keep other jobs running
                LOGGER.error("job %s failed: %s", job, exc)
                out.append((job, None))
        return out

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futs = [pool.submit(worker, g) for g in gpus]
        for fut in as_completed(futs):
            for job, auc in fut.result():
                if auc is not None:
                    results[job.dataset][job.seed] = auc

    summary: dict[str, Any] = {}
    for ds in DATASETS:
        per_seed = results[ds]
        if not per_seed:
            LOGGER.error("%s: no successful runs", ds)
            continue
        vals = np.array([per_seed[s] for s in sorted(per_seed)], dtype=np.float64)
        raw_path = root / RAW_METRICS.format(ds=ds)
        raw = float(json.loads(raw_path.read_text())["auc_roc"])
        deltas = vals - raw
        lo, hi = _bootstrap_ci(vals)
        dlo, dhi = _bootstrap_ci(deltas)
        summary[ds] = {
            "alpha": DATASETS[ds]["alpha"], "n_seeds": int(vals.size),
            "per_seed": {str(s): per_seed[s] for s in sorted(per_seed)},
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "ci95": [lo, hi],
            "raw_maha_flatten": raw,
            "delta_vs_raw_mean": float(deltas.mean()),
            "delta_vs_raw_ci95": [dlo, dhi],
            "delta_ci_excludes_zero": bool(dlo > 0 or dhi < 0),
        }
        LOGGER.info(
            "%s (a=%.1f, n=%d): %.4f+/-%.4f CI[%.4f,%.4f] | raw=%.4f | "
            "delta=%+.4f CI[%+.4f,%+.4f] %s",
            ds, DATASETS[ds]["alpha"], vals.size, vals.mean(),
            vals.std(ddof=1) if vals.size > 1 else 0.0, lo, hi, raw,
            deltas.mean(), dlo, dhi,
            "(excludes 0)" if (dlo > 0 or dhi < 0) else "(includes 0)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "meta": {"config": CONFIG_NAME, "renderer": RENDERER,
                 "smooth_window": SMOOTH_WINDOW, "seeds": list(SEEDS),
                 "note": "headline configuration (spatial+dual, per-dataset alpha); "
                         "paired difference vs raw Mahalanobis flatten baseline."},
        "results": summary}, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
