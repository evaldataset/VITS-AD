"""Collect and summarize ablation study results into a paper-ready table."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

VARIANT_DESCRIPTIONS: dict[str, str] = {
    "baseline": "Full PatchTraj (default)",
    "A_pi_identity": "π = identity (no geometric correspondence)",
    "B_small_model": "Small model (d=128, L=1)",
    "C_no_trim": "No trimmed loss (ratio=0)",
    "D_no_smooth": "No score smoothing",
    "F_short_context": "Short context (K=4)",
}


def collect_ablation(
    results_dir: Path,
    dataset: str,
    entities: list[str] | None = None,
    renderers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect ablation metrics from experiment directories.

    Args:
        results_dir: Base results directory (e.g., results/).
        dataset: Dataset name (smd, psm, etc.).
        entities: List of entities. None for single-entity datasets.
        renderers: Renderer names to check. Defaults to LP+RP.

    Returns:
        List of result dicts with keys: variant, renderer, entity, auc_roc, etc.
    """
    if renderers is None:
        renderers = ["line_plot", "recurrence_plot"]
    if entities is None:
        entities = ["default"]

    ablation_dir = results_dir / f"ablation_{dataset}"
    if not ablation_dir.exists():
        LOGGER.warning("Ablation dir not found: %s", ablation_dir)
        return []

    results: list[dict[str, Any]] = []
    for entity in entities:
        entity_dir = ablation_dir / entity
        if not entity_dir.exists():
            continue

        for variant_dir in sorted(entity_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name

            for renderer in renderers:
                metrics_path = variant_dir / renderer / "metrics.json"
                if not metrics_path.exists():
                    continue

                with metrics_path.open("r") as f:
                    metrics = json.load(f)

                results.append(
                    {
                        "dataset": dataset,
                        "entity": entity,
                        "variant": variant,
                        "description": VARIANT_DESCRIPTIONS.get(variant, variant),
                        "renderer": renderer,
                        "auc_roc": metrics.get("auc_roc", 0.0),
                        "auc_pr": metrics.get("auc_pr", 0.0),
                        "f1_pa": metrics.get("f1_pa", 0.0),
                    }
                )

    return results


def generate_ablation_report(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate ablation study report.

    Args:
        results: Collected ablation results.
        output_dir: Where to save reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        LOGGER.warning("No ablation results to report.")
        return

    # CSV
    csv_lines = ["dataset,entity,variant,description,renderer,auc_roc,auc_pr,f1_pa"]
    for r in results:
        csv_lines.append(
            f"{r['dataset']},{r['entity']},{r['variant']},"
            f'"{r["description"]}",{r["renderer"]},'
            f"{r['auc_roc']:.6f},{r['auc_pr']:.6f},{r['f1_pa']:.6f}"
        )
    csv_path = output_dir / "ablation_results.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # Aggregate by variant (average across entities and renderers)
    variant_scores: dict[str, list[float]] = {}
    for r in results:
        v = r["variant"]
        if v not in variant_scores:
            variant_scores[v] = []
        variant_scores[v].append(r["auc_roc"])

    # Markdown
    md_lines = [
        "# Ablation Study Results",
        "",
        "| Variant | Description | Avg AUC-ROC | Δ vs Baseline |",
        "|---------|-------------|-------------|---------------|",
    ]

    baseline_auc = np.mean(variant_scores.get("baseline", [0.0]))
    for variant in [
        "baseline",
        "A_pi_identity",
        "B_small_model",
        "C_no_trim",
        "D_no_smooth",
        "F_short_context",
    ]:
        if variant not in variant_scores:
            continue
        avg = np.mean(variant_scores[variant])
        delta = avg - baseline_auc
        desc = VARIANT_DESCRIPTIONS.get(variant, variant)
        md_lines.append(f"| {variant} | {desc} | {avg:.4f} | {delta:+.4f} |")

    md_path = output_dir / "ablation_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    LOGGER.info("Saved ablation report: %s", md_path)
    LOGGER.info("Saved ablation CSV: %s", csv_path)


def main() -> None:
    """Collect and report ablation results."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Collect ablation results")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--entities", nargs="*", default=None)
    args = parser.parse_args()

    results = collect_ablation(
        results_dir=Path(args.results_dir),
        dataset=args.dataset,
        entities=args.entities,
    )

    generate_ablation_report(results, Path(args.results_dir) / "reports")

    # Print summary
    LOGGER.info("Collected %d ablation results", len(results))


if __name__ == "__main__":
    main()
