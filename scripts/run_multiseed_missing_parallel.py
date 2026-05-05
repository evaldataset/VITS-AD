from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Sequence


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_command(args: Sequence[str], env: dict[str, str]) -> None:
    LOGGER.info("Running command: %s", " ".join(args))
    completed = subprocess.run(list(args), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(args)}"
        )


def ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_single(seed: int, renderer: str, gpu: int, project_root: Path) -> None:
    run_output_dir = f"results/multiseed/smap/{renderer}/seed_{seed}"
    run_output_path = project_root / run_output_dir
    metrics_path = run_output_path / "metrics.json"
    if metrics_path.exists():
        LOGGER.info("Skipping completed run: %s", run_output_dir)
        return

    ensure_results_dir(run_output_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(project_root)
        if existing_pythonpath == ""
        else str(project_root) + os.pathsep + existing_pythonpath
    )

    common_overrides = [
        "data=smap",
        "data.raw_dir=data/raw/smap",
        "+data.entity=smap",
        f"render={renderer}",
        f"training.seed={seed}",
        f"output_dir={run_output_dir}",
        "data.processed_dir=results/multiseed/smap/token_cache",
    ]

    train_cmd = [sys.executable, "scripts/train_patchtraj.py", *common_overrides]
    detect_cmd = [sys.executable, "scripts/detect.py", *common_overrides]

    run_command(train_cmd, env)
    run_command(detect_cmd, env)

    test_cache = run_output_path / "test_patch_tokens.npy"
    if test_cache.exists():
        test_cache.unlink()
        LOGGER.info("Removed per-seed test cache: %s", test_cache)

    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    LOGGER.info(
        "Completed %s seed=%d gpu=%d auc_roc=%.6f",
        renderer,
        seed,
        gpu,
        float(payload.get("auc_roc", 0.0)),
    )


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        type=str,
        required=True,
        choices=["line", "rec_low", "rec_high"],
        help="Worker partition to run.",
    )
    args = parser.parse_args()

    plans: dict[str, list[tuple[int, str, int]]] = {
        "line": [
            (456, "line_plot", 2),
            (789, "line_plot", 2),
            (2024, "line_plot", 2),
        ],
        "rec_low": [
            (42, "recurrence_plot", 1),
            (123, "recurrence_plot", 1),
        ],
        "rec_high": [
            (456, "recurrence_plot", 3),
            (789, "recurrence_plot", 3),
            (2024, "recurrence_plot", 3),
        ],
    }
    plan = plans[args.worker]

    for seed, renderer, gpu in plan:
        run_single(seed=seed, renderer=renderer, gpu=gpu, project_root=project_root)


if __name__ == "__main__":
    main()
