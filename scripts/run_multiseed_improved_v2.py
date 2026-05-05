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
from statistics import mean, stdev
from typing import cast


LOGGER = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 2024]
RENDERERS = ["line_plot", "recurrence_plot"]
SMD_ENTITIES = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-1-4",
    "machine-1-5",
    "machine-1-6",
    "machine-1-7",
    "machine-1-8",
    "machine-2-1",
    "machine-2-2",
    "machine-2-3",
    "machine-2-4",
    "machine-2-5",
    "machine-2-6",
    "machine-2-7",
    "machine-2-8",
    "machine-2-9",
    "machine-3-1",
    "machine-3-2",
    "machine-3-3",
    "machine-3-4",
    "machine-3-5",
    "machine-3-6",
    "machine-3-7",
    "machine-3-8",
    "machine-3-9",
    "machine-3-10",
    "machine-3-11",
]
DATASET_OVERRIDES: dict[str, list[str]] = {
    "smd": ["data=smd", "data.raw_dir=data/raw/smd"],
    "psm": ["data=psm", "data.raw_dir=data/raw/psm", "+data.entity=psm"],
    "msl": ["data=msl", "data.raw_dir=data/raw/msl", "+data.entity=msl"],
    "smap": ["data=smap", "data.raw_dir=data/raw/smap", "+data.entity=smap"],
}


@dataclass(frozen=True)
class Job:
    dataset: str
    renderer: str
    seed: int
    entity: str | None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--gpus", type=str, default="0")
    _ = parser.add_argument("--python", type=str, default=sys.executable)
    _ = parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_gpus(raw_value: str) -> list[int]:
    gpu_values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not gpu_values:
        raise ValueError("--gpus must specify at least one GPU id.")
    return [int(item) for item in gpu_values]


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
        payload_obj = cast(object, json.load(handle))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Invalid metrics payload in {metrics_path}")
    payload = cast(dict[str, object], payload_obj)
    auc_roc = payload.get("auc_roc")
    if not isinstance(auc_roc, (int, float)):
        raise ValueError(f"auc_roc missing or non-numeric in {metrics_path}")
    return float(auc_roc)


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for dataset in DATASET_OVERRIDES:
        entities = SMD_ENTITIES if dataset == "smd" else [None]
        for seed in SEEDS:
            for entity in entities:
                for renderer in RENDERERS:
                    jobs.append(
                        Job(dataset=dataset, renderer=renderer, seed=seed, entity=entity)
                    )
    return jobs


def output_dir_for(job: Job) -> Path:
    base_dir = Path("results") / "improved_v2" / "multiseed" / job.dataset / f"seed_{job.seed}"
    if job.entity is not None:
        return base_dir / job.entity / job.renderer
    return base_dir / "default" / job.renderer


def metrics_path_for(job: Job, project_root: Path) -> Path:
    return project_root / output_dir_for(job) / "metrics.json"


def build_overrides(job: Job) -> list[str]:
    overrides = [
        *DATASET_OVERRIDES[job.dataset],
        f"render={job.renderer}",
        f"training.seed={job.seed}",
        f"output_dir={output_dir_for(job)}",
    ]
    if job.entity is not None:
        overrides.append(f"data.entity={job.entity}")
        overrides.append(
            f"data.processed_dir=results/improved_v2/multiseed/{job.dataset}/token_cache/{job.entity}"
        )
    else:
        overrides.append(
            f"data.processed_dir=results/improved_v2/multiseed/{job.dataset}/token_cache"
        )
    return overrides


def run_job(
    job: Job,
    gpu: int,
    project_root: Path,
    python_executable: str,
    skip_existing: bool,
) -> None:
    metrics_path = metrics_path_for(job=job, project_root=project_root)
    if skip_existing and metrics_path.exists():
        LOGGER.info("Skipping completed run: %s", output_dir_for(job))
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(project_root)
        if existing_pythonpath == ""
        else str(project_root) + os.pathsep + existing_pythonpath
    )

    overrides = build_overrides(job)
    train_cmd = [
        python_executable,
        "scripts/train_patchtraj.py",
        "--config-name",
        "experiment/patchtraj_improved",
        *overrides,
    ]
    detect_cmd = [
        python_executable,
        "scripts/detect.py",
        "--config-name",
        "experiment/patchtraj_improved",
        *overrides,
    ]

    run_command(train_cmd, env)
    run_command(detect_cmd, env)
    _ = load_auc_roc(metrics_path)


def run_gpu_queue(
    gpu: int,
    jobs: list[Job],
    project_root: Path,
    python_executable: str,
    skip_existing: bool,
) -> None:
    for job in jobs:
        run_job(
            job=job,
            gpu=gpu,
            project_root=project_root,
            python_executable=python_executable,
            skip_existing=skip_existing,
        )


def assign_jobs(jobs: list[Job], gpus: list[int]) -> dict[int, list[Job]]:
    assignments: dict[int, list[Job]] = {gpu: [] for gpu in gpus}
    for index, job in enumerate(jobs):
        gpu = gpus[index % len(gpus)]
        assignments[gpu].append(job)
    return assignments


def aggregate_results(project_root: Path) -> dict[str, object]:
    results: dict[str, object] = {
        "seeds": SEEDS,
        "renderers": RENDERERS,
        "stability_threshold": 0.02,
        "datasets": {},
    }
    all_stable = True

    for dataset in DATASET_OVERRIDES:
        dataset_payload: dict[str, object] = {}
        entities = SMD_ENTITIES if dataset == "smd" else [None]
        dataset_stable = True

        for renderer in RENDERERS:
            seed_scores: list[float] = []
            per_seed_details: dict[str, object] = {}

            for seed in SEEDS:
                entity_scores: dict[str, float] = {}
                for entity in entities:
                    job = Job(dataset=dataset, renderer=renderer, seed=seed, entity=entity)
                    metrics_path = metrics_path_for(job=job, project_root=project_root)
                    entity_name = entity if entity is not None else "default"
                    entity_scores[entity_name] = load_auc_roc(metrics_path)

                seed_auc = float(mean(entity_scores.values()))
                seed_scores.append(seed_auc)
                per_seed_details[str(seed)] = {
                    "dataset_auc_roc": seed_auc,
                    "entity_auc_roc": entity_scores,
                }

            std_value = float(stdev(seed_scores)) if len(seed_scores) > 1 else 0.0
            stable = std_value < 0.02
            dataset_stable = dataset_stable and stable
            dataset_payload[renderer] = {
                "auc_roc": seed_scores,
                "mean": float(mean(seed_scores)),
                "std": std_value,
                "std_type": "sample",
                "stable": stable,
                "per_seed": per_seed_details,
            }

        dataset_payload["stable"] = dataset_stable
        datasets_payload = cast(dict[str, object], results["datasets"])
        datasets_payload[dataset] = dataset_payload
        all_stable = all_stable and dataset_stable

    results["all_datasets_stable"] = all_stable
    return results


def main() -> None:
    configure_logging()
    args = parse_args()
    gpus = parse_gpus(cast(str, args.gpus))
    python_executable = cast(str, args.python)
    project_root = Path(__file__).resolve().parent.parent
    jobs = build_jobs()
    assignments = assign_jobs(jobs=jobs, gpus=gpus)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(
                run_gpu_queue,
                gpu,
                gpu_jobs,
                project_root,
                python_executable,
                bool(cast(bool, args.skip_existing)),
            )
            for gpu, gpu_jobs in assignments.items()
            if gpu_jobs
        ]
        for future in as_completed(futures):
            future.result()

    results = aggregate_results(project_root=project_root)
    report_path = project_root / "results" / "improved_v2" / "multiseed" / "multiseed_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    evidence_path = project_root / ".sisyphus" / "evidence" / "task-12-multiseed-results.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    LOGGER.info("Saved multiseed report to %s", report_path)
    LOGGER.info("Saved QA evidence to %s", evidence_path)


if __name__ == "__main__":
    main()
