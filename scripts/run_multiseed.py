from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, cast


LOGGER = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 2024]
RENDERERS = ["line_plot", "recurrence_plot"]

DATASET_OVERRIDES: dict[str, list[str]] = {
    "psm": ["data=psm", "data.raw_dir=data/raw/psm", "+data.entity=psm"],
    "msl": ["data=msl", "data.raw_dir=data/raw/msl", "+data.entity=msl"],
    "smap": ["data=smap", "data.raw_dir=data/raw/smap", "+data.entity=smap"],
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_command(args: list[str], env: dict[str, str]) -> None:
    LOGGER.info("Running command: %s", " ".join(args))
    completed = subprocess.run(args, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(args)}"
        )


def load_auc_roc(metrics_path: Path) -> float:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics_obj = cast(object, json.load(handle))

    if not isinstance(metrics_obj, dict):
        raise ValueError(f"Invalid metrics format in file: {metrics_path}")
    metrics_dict = cast(Dict[str, object], metrics_obj)

    if "auc_roc" not in metrics_dict:
        raise KeyError(f"auc_roc missing from metrics file: {metrics_path}")
    auc_value_obj = metrics_dict["auc_roc"]
    if not isinstance(auc_value_obj, (int, float)):
        raise ValueError(f"auc_roc must be numeric in file: {metrics_path}")
    return float(auc_value_obj)


def clean_artifacts(base_output_dir: Path) -> None:
    for dataset in DATASET_OVERRIDES:
        token_cache_dir = base_output_dir / dataset / "token_cache"
        if token_cache_dir.exists():
            LOGGER.info("Removing token cache directory: %s", token_cache_dir)
            shutil.rmtree(token_cache_dir)

        dataset_dir = base_output_dir / dataset
        if not dataset_dir.exists():
            continue

        for test_cache in dataset_dir.glob("**/test_patch_tokens.npy"):
            LOGGER.info("Removing test token cache file: %s", test_cache)
            test_cache.unlink()


def main() -> None:
    configure_logging()

    project_root = Path(__file__).resolve().parent.parent
    base_output_dir = project_root / "results" / "multiseed"
    report_path = project_root / "results" / "reports" / "multiseed_results.json"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(project_root)
        if existing_pythonpath == ""
        else str(project_root) + os.pathsep + existing_pythonpath
    )

    results: dict[str, dict[str, object]] = {}

    for dataset, dataset_overrides in DATASET_OVERRIDES.items():
        dataset_results: dict[str, object] = {"seeds": SEEDS}
        shared_processed_dir = (
            f"data.processed_dir=results/multiseed/{dataset}/token_cache"
        )

        for renderer in RENDERERS:
            auc_values: list[float] = []

            for seed in SEEDS:
                run_output_dir = f"results/multiseed/{dataset}/{renderer}/seed_{seed}"
                metrics_path = project_root / run_output_dir / "metrics.json"

                if metrics_path.exists():
                    LOGGER.info("Skipping completed run: %s", run_output_dir)
                    auc_values.append(load_auc_roc(metrics_path))
                    continue

                common_overrides = [
                    "render=" + renderer,
                    "training.seed=" + str(seed),
                    "output_dir=" + run_output_dir,
                    shared_processed_dir,
                ]
                overrides = dataset_overrides + common_overrides

                train_cmd = [
                    sys.executable,
                    "scripts/train_patchtraj.py",
                    *overrides,
                ]
                detect_cmd = [
                    sys.executable,
                    "scripts/detect.py",
                    *overrides,
                ]

                run_command(train_cmd, env)
                run_command(detect_cmd, env)

                auc_values.append(load_auc_roc(metrics_path))

            std_value = float(stdev(auc_values)) if len(auc_values) > 1 else 0.0
            dataset_results[renderer] = {
                "auc_roc": auc_values,
                "mean": float(mean(auc_values)),
                "std": std_value,
                "std_type": "sample",
            }

        results[dataset] = dataset_results

    clean_artifacts(base_output_dir=base_output_dir)

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    LOGGER.info("Saved multiseed report to %s", report_path)


if __name__ == "__main__":
    main()
