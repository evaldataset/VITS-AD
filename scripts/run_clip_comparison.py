from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import cast


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = "python"
CLIP_PRETRAINED = "openai/clip-vit-base-patch16"
ENTITY = "machine-1-1"
SEED = 42
DEFAULT_RENDERERS = ("line_plot", "recurrence_plot")
MODEL_CACHE_KEY = CLIP_PRETRAINED.replace("/", "_")


def _run_command(command: list[str], env: dict[str, str]) -> None:
    LOGGER.info("Running command: %s", " ".join(command))
    _ = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _cleanup_token_caches(output_dir: Path, processed_dir: Path, renderer: str) -> None:
    files_to_delete = [output_dir / "test_patch_tokens.npy"]

    files_to_delete.extend(output_dir.glob("tokens_*.pt"))

    processed_pattern = f"tokens_{MODEL_CACHE_KEY}_{ENTITY}_{renderer}.pt"
    files_to_delete.extend(processed_dir.glob(processed_pattern))

    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            LOGGER.info("Deleted cache file: %s", file_path)


def _load_metrics(metrics_path: Path) -> dict[str, float]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        loaded = cast(object, json.load(handle))

    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid metrics payload in {metrics_path}")
    metrics = cast(dict[str, object], loaded)

    if "auc_roc" not in metrics or "auc_pr" not in metrics:
        raise KeyError(f"Missing auc_roc/auc_pr in metrics file: {metrics_path}")

    auc_roc = metrics["auc_roc"]
    auc_pr = metrics["auc_pr"]
    if not isinstance(auc_roc, (int, float)) or not isinstance(auc_pr, (int, float)):
        raise TypeError(
            f"auc_roc/auc_pr must be numeric in metrics file: {metrics_path}"
        )

    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
    }


def _run_clip_for_renderer(renderer: str, env: dict[str, str]) -> dict[str, float]:
    output_dir = PROJECT_ROOT / "results" / "clip_smd_m11" / renderer
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = PROJECT_ROOT / "data" / "processed" / "smd"
    processed_dir.mkdir(parents=True, exist_ok=True)

    _cleanup_token_caches(
        output_dir=output_dir, processed_dir=processed_dir, renderer=renderer
    )

    common_overrides = [
        f"model.pretrained={CLIP_PRETRAINED}",
        f"data.entity={ENTITY}",
        f"render={renderer}",
        f"output_dir={output_dir.as_posix()}",
        f"training.seed={SEED}",
        "patchtraj.d_model=256",
        "patchtraj.n_heads=4",
        "patchtraj.n_layers=2",
        "patchtraj.K=8",
    ]

    train_command = [
        PYTHON_EXECUTABLE,
        "scripts/train_patchtraj.py",
        *common_overrides,
    ]
    _run_command(train_command, env)

    detect_command = [
        PYTHON_EXECUTABLE,
        "scripts/detect.py",
        *common_overrides,
    ]
    _run_command(detect_command, env)

    _cleanup_token_caches(
        output_dir=output_dir, processed_dir=processed_dir, renderer=renderer
    )

    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Detection metrics not found: {metrics_path}")
    return _load_metrics(metrics_path)


def _load_existing_dinov2_results() -> dict[str, dict[str, float]]:
    base_dir = PROJECT_ROOT / "results" / "full_smd" / ENTITY
    line_metrics = _load_metrics(base_dir / "line_plot" / "metrics.json")
    recurrence_metrics = _load_metrics(base_dir / "recurrence_plot" / "metrics.json")
    return {
        "dinov2_line_plot": line_metrics,
        "dinov2_recurrence_plot": recurrence_metrics,
    }


def _load_existing_clip_metrics(renderer: str) -> dict[str, float]:
    metrics_path = PROJECT_ROOT / "results" / "clip_smd_m11" / renderer / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing CLIP metrics for renderer '{renderer}': {metrics_path}"
        )
    return _load_metrics(metrics_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--renderers",
        nargs="+",
        default=list(DEFAULT_RENDERERS),
        choices=["line_plot", "recurrence_plot"],
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = _parse_args()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    renderers = cast(list[str], args.renderers)

    results: dict[str, dict[str, float]] = {}
    for renderer in renderers:
        key = f"clip_{renderer}"
        results[key] = _run_clip_for_renderer(renderer=renderer, env=env)

    for renderer in DEFAULT_RENDERERS:
        key = f"clip_{renderer}"
        if key not in results:
            results[key] = _load_existing_clip_metrics(renderer)

    results.update(_load_existing_dinov2_results())

    report_path = PROJECT_ROOT / "results" / "reports" / "clip_backbone_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    LOGGER.info("Saved comparison report to %s", report_path)
    LOGGER.info("Report content: %s", json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
