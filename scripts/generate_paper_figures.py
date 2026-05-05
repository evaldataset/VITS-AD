#!/usr/bin/env python3
from __future__ import annotations

"""Generate the paper figures used in the NeurIPS draft."""

# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false

import csv
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

RESULTS_ROOT = Path("results")
LEGACY_FIGURES_DIR = RESULTS_ROOT / "figures"
FIGURES_DIR = Path(".sisyphus/artifacts/task-17-figures")
EVIDENCE_PATH = Path(".sisyphus/evidence/task-17-figures.txt")
PAPER_PATH = Path("paper/neurips_main.tex")
FIGURE_DPI = 320

DATASETS = ["SMD", "PSM", "MSL", "SMAP"]
DATASET_COLORS = {
    "SMD": "#1f77b4",
    "PSM": "#ff7f0e",
    "MSL": "#2ca02c",
    "SMAP": "#d62728",
}
METHOD_COLORS = {
    "LOF": "#b7b7b7",
    "Isolation Forest": "#9a9a9a",
    "One-Class SVM": "#6f6f6f",
    "TimesNet": "#6c8dd5",
    "CATCH": "#d95f5f",
    "PatchTraj Default": "#4c78a8",
    "PatchTraj Improved v2": "#e39c39",
}
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
MAIN_COMPARISON = {
    "LOF": [0.658, 0.680, 0.547, 0.610],
    "Isolation Forest": [0.660, 0.655, 0.580, 0.603],
    "One-Class SVM": [0.680, 0.689, 0.509, 0.394],
    "TimesNet": [0.756, 0.575, 0.612, 0.444],
    "CATCH": [0.818, 0.648, 0.655, 0.499],
    "PatchTraj Default": [0.796, 0.594, 0.562, 0.677],
}
MAIN_COMPARISON_ERRORS = {
    "PatchTraj Default": [0.0, 0.0064, 0.0029, 0.0005],
}


def configure_matplotlib() -> None:
    """Apply shared plotting defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_output_dirs() -> None:
    """Create output directories required by the generator."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    """Load JSON from disk."""
    return json.loads(path.read_text())


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load rows from a CSV file."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, name: str, generated: list[Path]) -> Path:
    """Save a figure as both PDF and PNG."""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    generated.append(pdf_path)
    LOGGER.info("  Saved: %s", pdf_path)
    plt.close(fig)
    return pdf_path


def load_per_entity_results(
    result_dir: Path, entities: list[str]
) -> dict[str, dict[str, float]]:
    """Load per-entity AUC-ROC values for LP/RP runs."""
    results: dict[str, dict[str, float]] = {}
    for entity in entities:
        entity_dir = result_dir / entity
        if not entity_dir.is_dir():
            continue
        renderer_scores: dict[str, float] = {}
        for renderer in ["line_plot", "recurrence_plot"]:
            metrics_path = entity_dir / renderer / "metrics.json"
            if metrics_path.exists():
                metrics = load_json(metrics_path)
                renderer_scores[renderer] = float(metrics.get("auc_roc", 0.0))
        if renderer_scores:
            results[entity] = renderer_scores
    return results


def load_improved_v2_summary() -> tuple[list[float], list[float]]:
    """Load improved-v2 dataset scores and optional multi-seed errors."""
    rows = load_csv_rows(RESULTS_ROOT / "reports" / "before_after_summary.csv")
    improved_by_dataset = {
        row["dataset"].upper(): float(row["after_auc_roc"]) for row in rows
    }
    multiseed = load_json(RESULTS_ROOT / "reports" / "multiseed_ensemble_summary.json")
    errors = {
        "SMD": 0.0,
        "PSM": float(multiseed["psm"]["std"]),
        "MSL": float(multiseed["msl"]["std"]),
        "SMAP": float(multiseed["smap"]["std"]),
    }
    values = [improved_by_dataset[dataset] for dataset in DATASETS]
    error_values = [errors[dataset] for dataset in DATASETS]
    return values, error_values


def load_viewdisagree_curves() -> dict[str, tuple[list[float], list[float]]]:
    """Load ViewDisagree lambda sweep curves."""
    payload = load_json(RESULTS_ROOT / "reports" / "view_disagree_sweep.json")
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for dataset in DATASETS:
        dataset_key = dataset.lower()
        result_map = payload[dataset_key]["results"]
        lambdas = sorted(float(key) for key in result_map)
        aucs = []
        for value in lambdas:
            result_item = result_map[f"{value:.1f}"]
            auc_key = "auc_roc" if "auc_roc" in result_item else "mean_auc_roc"
            aucs.append(float(result_item[auc_key]))
        curves[dataset] = (lambdas, aucs)
    return curves


def parse_calibguard_results() -> dict[str, dict[str, list[float]]]:
    """Extract fixed and rolling FAR traces across datasets."""
    payload = load_json(RESULTS_ROOT / "reports" / "calibguard_multidataset.json")
    traces: dict[str, dict[str, list[float]]] = {}
    alphas = [float(value) for value in payload["alpha_values"]]
    for entry in payload["results"]:
        dataset = str(entry["dataset"]).replace("_28_AVG", "")
        fixed_values = []
        rolling_values = []
        for alpha in alphas:
            key = f"{alpha}"
            fixed_item = entry["fixed"][key]
            rolling_item = entry["rolling"][key]
            fixed_value = fixed_item.get("actual_far", fixed_item.get("actual_far_mean"))
            rolling_value = rolling_item.get(
                "empirical_far", rolling_item.get("empirical_far_mean")
            )
            fixed_values.append(float(fixed_value))
            rolling_values.append(float(rolling_value))
        traces[dataset] = {
            "alpha": alphas,
            "fixed": fixed_values,
            "rolling": rolling_values,
        }
    return traces


def fig1_main_comparison(generated: list[Path]) -> None:
    """Figure 1: Main benchmark comparison with updated baselines."""
    LOGGER.info("Generating Figure 1: main comparison")
    improved_values, improved_errors = load_improved_v2_summary()
    data = dict(MAIN_COMPARISON)
    data["PatchTraj Improved v2"] = improved_values
    errors = dict(MAIN_COMPARISON_ERRORS)
    errors["PatchTraj Improved v2"] = improved_errors

    methods = list(data.keys())
    x = np.arange(len(DATASETS))
    width = 0.105
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    for index, method in enumerate(methods):
        values = data[method]
        bars = ax.bar(
            x + offsets[index] * width,
            values,
            width * 0.92,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.6,
            yerr=errors.get(method),
            capsize=2.5 if method in errors else 0.0,
            zorder=3,
        )
        if method.startswith("PatchTraj"):
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.008,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )

    best_per_dataset = np.max(np.array(list(data.values())), axis=0)
    for x_pos, best_value in zip(x, best_per_dataset):
        ax.scatter(
            x_pos,
            best_value + 0.02,
            marker="*",
            s=80,
            color="#f2c14e",
            edgecolors="#6b4f00",
            linewidth=0.5,
            zorder=5,
        )

    ax.set_ylabel("AUC-ROC")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0.30, 0.92)
    ax.set_title("Main comparison with updated claim-bearing baselines", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.text(
        0.01,
        0.02,
        "Error bars show available 5-seed std estimates; improved-v2 uses Task 11 dataset summaries.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, "fig1_main_comparison", generated)


def fig2_entity_heatmap(generated: list[Path]) -> None:
    """Figure 2: SMD entity heatmap with improved-v2 ensemble column."""
    LOGGER.info("Generating Figure 2: entity heatmap")
    default_results = load_per_entity_results(RESULTS_ROOT / "full_smd", SMD_ENTITIES)
    improved_results = load_per_entity_results(RESULTS_ROOT / "improved_smd", SMD_ENTITIES)
    ensemble_data = load_json(RESULTS_ROOT / "reports" / "improved_ensemble_results.json")
    ensemble_scores = ensemble_data["smd"]["per_entity"]

    methods = [
        "Default LP",
        "Default RP",
        "Improved LP",
        "Improved RP",
        "Improved\nEnsemble",
    ]
    matrix = np.full((len(SMD_ENTITIES), len(methods)), np.nan)
    for row_index, entity in enumerate(SMD_ENTITIES):
        matrix[row_index, 0] = default_results.get(entity, {}).get("line_plot", np.nan)
        matrix[row_index, 1] = default_results.get(entity, {}).get(
            "recurrence_plot", np.nan
        )
        matrix[row_index, 2] = improved_results.get(entity, {}).get("line_plot", np.nan)
        matrix[row_index, 3] = improved_results.get(entity, {}).get(
            "recurrence_plot", np.nan
        )
        if entity in ensemble_scores:
            matrix[row_index, 4] = float(ensemble_scores[entity]["auc_roc"])

    fig, ax = plt.subplots(figsize=(8.5, 12.2))
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.30, vmax=1.0)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                continue
            text_color = "white" if value < 0.56 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.6,
                color=text_color,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_yticks(np.arange(len(SMD_ENTITIES)))
    ax.set_yticklabels(SMD_ENTITIES, fontsize=7.7)
    ax.set_title("SMD entity-level behavior of default, improved, and routed ensemble models", fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(methods)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SMD_ENTITIES)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", size=0)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.02)
    colorbar.set_label("AUC-ROC")
    fig.tight_layout()
    save_figure(fig, "fig2_entity_heatmap", generated)


def fig3_ablation(generated: list[Path]) -> None:
    """Figure 3: Expanded ablation summary."""
    LOGGER.info("Generating Figure 3: expanded ablation")
    ablation_rows = load_csv_rows(RESULTS_ROOT / "reports" / "ablation_results.csv")
    baseline_value = 0.0
    component_scores: list[tuple[str, float]] = []
    for row in ablation_rows:
        if row["renderer"] != "line_plot":
            continue
        variant = row["variant"]
        value = float(row["auc_roc"])
        if variant == "baseline":
            baseline_value = value
        label = {
            "baseline": "Baseline",
            "A_pi_identity": "No correspondence",
            "B_small_model": "Smaller predictor",
            "C_no_trim": "No trimmed loss",
            "D_no_smooth": "No smoothing",
            "F_short_context": "Short context",
        }.get(variant, variant)
        component_scores.append((label, value))
    component_scores.sort(key=lambda item: item[1])

    delta_rows = load_csv_rows(RESULTS_ROOT / "reports" / "before_after_summary.csv")
    dataset_deltas = [
        (row["dataset"].upper(), float(row["delta"])) for row in delta_rows
    ]

    curves = load_viewdisagree_curves()

    fig = plt.figure(figsize=(14.2, 5.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.85, 1.15])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_mid = fig.add_subplot(grid[0, 1])
    ax_right = fig.add_subplot(grid[0, 2])

    labels = [label for label, _ in component_scores]
    values = [value for _, value in component_scores]
    colors = ["#d95f5f" if value < baseline_value else "#4c78a8" for value in values]
    ax_left.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax_left.axvline(baseline_value, color="#444444", linestyle=":", linewidth=1.2)
    ax_left.set_xlim(0.63, 0.75)
    ax_left.set_xlabel("Line-plot AUC-ROC")
    ax_left.set_title("Machine-1-1 component ablations", fontweight="bold")
    ax_left.grid(axis="x", linestyle="--", alpha=0.25)
    for index, value in enumerate(values):
        delta = value - baseline_value
        ax_left.text(value + 0.002, index, f"{delta:+.3f}", va="center", fontsize=8)

    mid_labels = [dataset for dataset, _ in dataset_deltas]
    mid_values = [delta for _, delta in dataset_deltas]
    ax_mid.bar(
        mid_labels,
        mid_values,
        color=[DATASET_COLORS[label] for label in mid_labels],
        edgecolor="white",
        linewidth=0.6,
    )
    ax_mid.axhline(0.0, color="#444444", linewidth=1.0)
    ax_mid.set_ylabel("AUC-ROC gain")
    ax_mid.set_title("Full v2 recipe gain", fontweight="bold")
    ax_mid.grid(axis="y", linestyle="--", alpha=0.25)
    for index, value in enumerate(mid_values):
        ax_mid.text(index, value + 0.0015, f"{value:+.3f}", ha="center", fontsize=8)

    for dataset in DATASETS:
        lambdas, aucs = curves[dataset]
        ax_right.plot(
            lambdas,
            aucs,
            linewidth=2.0,
            label=dataset,
            color=DATASET_COLORS[dataset],
        )
        best_index = int(np.argmax(aucs))
        ax_right.scatter(
            lambdas[best_index],
            aucs[best_index],
            s=35,
            color=DATASET_COLORS[dataset],
            edgecolors="white",
            linewidth=0.6,
            zorder=5,
        )
    ax_right.set_xlabel(r"ViewDisagree weight $\lambda$")
    ax_right.set_ylabel("AUC-ROC")
    ax_right.set_title("Cross-view disagreement sweep", fontweight="bold")
    ax_right.grid(linestyle="--", alpha=0.25)
    ax_right.legend(frameon=False, loc="lower left")

    fig.suptitle("Expanded ablation: component removals, full-recipe gains, and ViewDisagree behavior", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig3_ablation", generated)


def fig_calibguard_far(generated: list[Path]) -> None:
    """Figure: CalibGuard target-vs-actual FAR behavior."""
    LOGGER.info("Generating Figure: CalibGuard FAR visualization")
    traces = parse_calibguard_results()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.8, 5.1))

    for dataset in DATASETS:
        trace = traces[dataset]
        alphas = trace["alpha"]
        ax_left.plot(
            alphas,
            trace["fixed"],
            marker="o",
            linewidth=2.0,
            color=DATASET_COLORS[dataset],
            label=f"{dataset} fixed",
        )
        ax_left.plot(
            alphas,
            trace["rolling"],
            marker="x",
            linewidth=1.4,
            linestyle="--",
            color=DATASET_COLORS[dataset],
            alpha=0.75,
            label=f"{dataset} rolling",
        )

    ax_left.plot([0.0, 0.11], [0.0, 0.11], color="#444444", linestyle=":", linewidth=1.2)
    ax_left.set_xlim(0.0, 0.11)
    ax_left.set_ylim(0.0, 0.76)
    ax_left.set_xlabel("Target FAR $\\alpha$")
    ax_left.set_ylabel("Actual FAR")
    ax_left.set_title("Target vs. realized FAR", fontweight="bold")
    ax_left.grid(linestyle="--", alpha=0.25)
    ax_left.legend(frameon=False, ncol=2, loc="upper left")

    bar_positions = np.arange(len(DATASETS))
    fixed_gap = [
        float(np.mean(np.abs(np.array(traces[dataset]["fixed"]) - np.array(traces[dataset]["alpha"]))))
        for dataset in DATASETS
    ]
    rolling_gap = [
        float(np.mean(np.abs(np.array(traces[dataset]["rolling"]) - np.array(traces[dataset]["alpha"]))))
        for dataset in DATASETS
    ]
    width = 0.34
    ax_right.bar(
        bar_positions - width / 2,
        fixed_gap,
        width,
        label="Leak-free fixed split",
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.6,
    )
    ax_right.bar(
        bar_positions + width / 2,
        rolling_gap,
        width,
        label="Rolling adaptation",
        color="#d95f5f",
        edgecolor="white",
        linewidth=0.6,
    )
    ax_right.set_xticks(bar_positions)
    ax_right.set_xticklabels(DATASETS)
    ax_right.set_ylabel("Mean |actual FAR - target FAR|")
    ax_right.set_title("Calibration gap across datasets", fontweight="bold")
    ax_right.grid(axis="y", linestyle="--", alpha=0.25)
    ax_right.legend(frameon=False)
    for index, value in enumerate(fixed_gap):
        ax_right.text(index - width / 2, value + 0.01, f"{value:.03f}", ha="center", fontsize=8)
    for index, value in enumerate(rolling_gap):
        ax_right.text(index + width / 2, value + 0.01, f"{value:.03f}", ha="center", fontsize=8)

    fig.suptitle("CalibGuard FAR diagnostics: leak-free calibration is consistently tighter than rolling updates", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_calibguard_far", generated)


def fig_backbone_ensemble(generated: list[Path]) -> None:
    """Figure: renderer-aware backbone complementarity."""
    LOGGER.info("Generating Figure: backbone complementarity")
    comparison = load_json(RESULTS_ROOT / "reports" / "clip_backbone_comparison.json")
    fps = load_json(RESULTS_ROOT / "reports" / "fps_benchmark.json")

    lp_values = [
        float(comparison["clip_line_plot"]["auc_roc"]),
        float(comparison["dinov2_line_plot"]["auc_roc"]),
    ]
    rp_values = [
        float(comparison["clip_recurrence_plot"]["auc_roc"]),
        float(comparison["dinov2_recurrence_plot"]["auc_roc"]),
    ]
    backbones = ["CLIP", "DINOv2"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    x = np.arange(len(backbones))
    width = 0.34
    ax_left.bar(
        x - width / 2,
        lp_values,
        width,
        color="#4c78a8",
        label="Line plot",
        edgecolor="white",
        linewidth=0.6,
    )
    ax_left.bar(
        x + width / 2,
        rp_values,
        width,
        color="#59a14f",
        label="Recurrence plot",
        edgecolor="white",
        linewidth=0.6,
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(backbones)
    ax_left.set_ylabel("AUC-ROC")
    ax_left.set_ylim(0.65, 0.91)
    ax_left.set_title("Complementary accuracy by renderer", fontweight="bold")
    ax_left.grid(axis="y", linestyle="--", alpha=0.25)
    ax_left.legend(frameon=False)
    for idx, value in enumerate(lp_values):
        ax_left.text(idx - width / 2, value + 0.005, f"{value:.3f}", ha="center", fontsize=8)
    for idx, value in enumerate(rp_values):
        ax_left.text(idx + width / 2, value + 0.005, f"{value:.3f}", ha="center", fontsize=8)

    scatter_points = [
        (float(fps["backbone_fps"]["clip"]), lp_values[0], "CLIP -> LP", True),
        (float(fps["backbone_fps"]["clip"]), rp_values[0], "CLIP -> RP", False),
        (float(fps["backbone_fps"]["dinov2"]), lp_values[1], "DINOv2 -> LP", False),
        (float(fps["backbone_fps"]["dinov2"]), rp_values[1], "DINOv2 -> RP", True),
    ]
    for x_val, y_val, label, selected in scatter_points:
        ax_right.scatter(
            x_val,
            y_val,
            s=150 if selected else 95,
            marker="*" if selected else "o",
            color="#e39c39" if selected else "#9d9d9d",
            edgecolors="#333333",
            linewidth=0.7,
            zorder=4,
        )
        ax_right.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(7, 5), fontsize=8)

    ax_right.set_xlabel("Backbone throughput (FPS)")
    ax_right.set_ylabel("AUC-ROC")
    ax_right.set_ylim(0.65, 0.91)
    ax_right.set_title("Routed ensemble picks the Pareto-favorable pair", fontweight="bold")
    ax_right.grid(linestyle="--", alpha=0.25)
    ax_right.text(
        0.03,
        0.03,
        "Selected routing: LP -> CLIP, RP -> DINOv2.",
        transform=ax_right.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    fig.suptitle("Backbone ensemble complementarity", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_backbone_ensemble", generated)


def fig4_renderer_comparison(generated: list[Path]) -> None:
    """Appendix figure: LP vs RP entity scatter."""
    LOGGER.info("Generating Figure 4: LP vs RP scatter")
    improved_results = load_per_entity_results(RESULTS_ROOT / "improved_smd", SMD_ENTITIES)
    lp_scores = []
    rp_scores = []
    entity_labels = []
    for entity in SMD_ENTITIES:
        lp = improved_results.get(entity, {}).get("line_plot", np.nan)
        rp = improved_results.get(entity, {}).get("recurrence_plot", np.nan)
        if np.isnan(lp) or np.isnan(rp):
            continue
        lp_scores.append(lp)
        rp_scores.append(rp)
        entity_labels.append(entity)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(
        lp_scores,
        rp_scores,
        s=70,
        c="#4c78a8",
        alpha=0.82,
        edgecolors="white",
        linewidth=0.6,
    )
    ax.plot([0.3, 1.0], [0.3, 1.0], color="#444444", linestyle="--", linewidth=1.1)
    for index, entity in enumerate(entity_labels):
        if abs(lp_scores[index] - rp_scores[index]) > 0.18:
            ax.annotate(
                entity.replace("machine-", "m-"),
                (lp_scores[index], rp_scores[index]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )

    mean_lp = float(np.mean(lp_scores))
    mean_rp = float(np.mean(rp_scores))
    ax.axvline(mean_lp, color="#d95f5f", linestyle=":", linewidth=1.1, label=f"Mean LP={mean_lp:.3f}")
    ax.axhline(mean_rp, color="#59a14f", linestyle=":", linewidth=1.1, label=f"Mean RP={mean_rp:.3f}")
    ax.set_xlim(0.25, 1.02)
    ax.set_ylim(0.25, 1.02)
    ax.set_xlabel("Line-plot AUC-ROC")
    ax.set_ylabel("Recurrence-plot AUC-ROC")
    ax.set_title("Entity-wise LP/RP complementarity on improved SMD runs", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig4_lp_vs_rp", generated)


def fig5_default_vs_improved(generated: list[Path]) -> None:
    """Appendix figure: default-vs-improved scatter."""
    LOGGER.info("Generating Figure 5: default vs improved scatter")
    default_results = load_per_entity_results(RESULTS_ROOT / "full_smd", SMD_ENTITIES)
    improved_results = load_per_entity_results(RESULTS_ROOT / "improved_smd", SMD_ENTITIES)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for axis_index, renderer in enumerate(["line_plot", "recurrence_plot"]):
        ax = axes[axis_index]
        default_values = []
        improved_values = []
        for entity in SMD_ENTITIES:
            default_value = default_results.get(entity, {}).get(renderer, np.nan)
            improved_value = improved_results.get(entity, {}).get(renderer, np.nan)
            if np.isnan(default_value) or np.isnan(improved_value):
                continue
            default_values.append(default_value)
            improved_values.append(improved_value)

        better_count = sum(
            1 for default_value, improved_value in zip(default_values, improved_values) if improved_value > default_value
        )
        ax.scatter(
            default_values,
            improved_values,
            s=60,
            c="#e39c39",
            edgecolors="white",
            linewidth=0.6,
            alpha=0.82,
        )
        ax.plot([0.3, 1.0], [0.3, 1.0], color="#444444", linestyle="--", linewidth=1.0)
        ax.set_xlim(0.25, 1.02)
        ax.set_ylim(0.25, 1.02)
        ax.set_xlabel("Default AUC-ROC")
        ax.set_ylabel("Improved AUC-ROC")
        title = "Line plot" if renderer == "line_plot" else "Recurrence plot"
        ax.set_title(f"{title} ({better_count}/{len(default_values)} entities improve)", fontweight="bold")
        ax.grid(linestyle="--", alpha=0.25)

    fig.suptitle("Entity-level shift from default to improved PatchTraj", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig5_default_vs_improved", generated)


def fig6_regime_diagnostics(generated: list[Path]) -> None:
    """Appendix figure: qualitative and entity regime diagnostics."""
    LOGGER.info("Generating Figure 6: regime diagnostics")
    ensemble_data = load_json(RESULTS_ROOT / "reports" / "improved_ensemble_results.json")
    overlay_path = FIGURES_DIR / "temporal_saliency_overlay.png"
    if not overlay_path.exists():
        overlay_path = LEGACY_FIGURES_DIR / "temporal_saliency_overlay.png"
    if not overlay_path.exists():
        LOGGER.warning("  Missing %s, skipping regime diagnostics", overlay_path)
        return

    per_entity = ensemble_data["smd"]["per_entity"]
    sorted_entities = sorted(
        ((entity, float(metrics["auc_roc"])) for entity, metrics in per_entity.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    top_entities = sorted_entities[:3]
    bottom_entities = sorted_entities[-3:]
    overlay = plt.imread(overlay_path)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.4),
        gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]},
    )
    axes[0].imshow(overlay)
    axes[0].axis("off")
    axes[0].set_title("TemporalSaliency success case", fontweight="bold")

    top_labels = [entity.replace("machine-", "m-") for entity, _ in top_entities][::-1]
    top_values = [value for _, value in top_entities][::-1]
    top_positions = np.arange(len(top_labels))
    axes[1].barh(top_positions, top_values, color="#59a14f", edgecolor="white", linewidth=0.6)
    axes[1].set_yticks(top_positions)
    axes[1].set_yticklabels(top_labels)
    axes[1].set_xlim(0.85, 1.01)
    axes[1].set_title("Strongest entities", fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.25)
    for index, value in enumerate(top_values):
        axes[1].text(value + 0.003, index, f"{value:.3f}", va="center", fontsize=8)

    bottom_labels = [entity.replace("machine-", "m-") for entity, _ in bottom_entities]
    bottom_values = [value for _, value in bottom_entities]
    bottom_positions = np.arange(len(bottom_labels))
    axes[2].barh(bottom_positions, bottom_values, color="#d95f5f", edgecolor="white", linewidth=0.6)
    axes[2].set_yticks(bottom_positions)
    axes[2].set_yticklabels(bottom_labels)
    axes[2].set_xlim(0.40, 0.70)
    axes[2].set_title("Failure-heavy entities", fontweight="bold")
    axes[2].grid(axis="x", linestyle="--", alpha=0.25)
    for index, value in enumerate(bottom_values):
        axes[2].text(value + 0.005, index, f"{value:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    save_figure(fig, "fig6_regime_diagnostics", generated)


def fig7_efficiency_tradeoff(generated: list[Path]) -> None:
    """Appendix figure: performance-efficiency tradeoff."""
    LOGGER.info("Generating Figure 7: efficiency tradeoff")
    fps = load_json(RESULTS_ROOT / "reports" / "fps_benchmark.json")
    methods = {
        "TimesNet": {"auc": np.mean([0.756, 0.575, 0.612, 0.444]), "params": 15.0, "minutes": 120.0, "color": "#6c8dd5"},
        "CATCH": {"auc": np.mean([0.818, 0.648, 0.655, 0.499]), "params": 30.0, "minutes": 240.0, "color": "#d95f5f"},
        "PatchTraj Default": {"auc": np.mean([0.796, 0.594, 0.562, 0.677]), "params": float(fps["params"]["default"]) / 1e6, "minutes": 5.0, "color": "#4c78a8"},
        "PatchTraj Improved": {"auc": np.mean(load_improved_v2_summary()[0]), "params": float(fps["params"]["improved"]) / 1e6, "minutes": 15.0, "color": "#e39c39"},
    }

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    for name, stats in methods.items():
        marker_size = 150 + 55 * np.log10(stats["minutes"] + 1.0)
        ax.scatter(
            stats["params"],
            stats["auc"],
            s=marker_size,
            color=stats["color"],
            edgecolors="white",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.annotate(
            f"{name}\n{stats['minutes']:.0f} min",
            (stats["params"], stats["auc"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlim(1.5, 40.0)
    ax.set_ylim(0.58, 0.69)
    ax.set_xlabel("Trainable parameters (millions, log scale)")
    ax.set_ylabel("Mean AUC-ROC across four benchmarks")
    ax.set_title("Efficiency-performance tradeoff", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.25)
    ax.text(
        1.6,
        0.582,
        "PatchTraj keeps 5-seed variance low on PSM/MSL/SMAP while training in minutes.",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    save_figure(fig, "fig7_efficiency_tradeoff", generated)


def verify_paper_references(generated: list[Path]) -> dict[str, bool]:
    """Check whether generated PDF names appear in the paper source."""
    if not PAPER_PATH.exists():
        return {path.name: False for path in generated}
    paper_text = PAPER_PATH.read_text()
    return {path.name: path.name in paper_text for path in generated}


def write_evidence(generated: list[Path], references: dict[str, bool]) -> None:
    """Write the figure manifest required by Task 17."""
    lines = [
        "Task 17 figure manifest",
        f"dpi={FIGURE_DPI}",
        f"count={len(generated)}",
        "",
    ]
    for path in generated:
        status = "referenced" if references.get(path.name, False) else "missing-reference"
        lines.append(f"{path.as_posix()} [{status}]")
    EVIDENCE_PATH.write_text("\n".join(lines) + "\n")
    LOGGER.info("Wrote evidence manifest to %s", EVIDENCE_PATH)


def main() -> None:
    """Generate all paper figures and record evidence."""
    configure_matplotlib()
    ensure_output_dirs()
    generated: list[Path] = []

    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Generating paper figures -> %s", FIGURES_DIR)
    LOGGER.info("%s", "=" * 60)

    fig1_main_comparison(generated)
    fig2_entity_heatmap(generated)
    fig3_ablation(generated)
    fig_calibguard_far(generated)
    fig_backbone_ensemble(generated)
    fig4_renderer_comparison(generated)
    fig5_default_vs_improved(generated)
    fig6_regime_diagnostics(generated)
    fig7_efficiency_tradeoff(generated)

    references = verify_paper_references(generated)
    write_evidence(generated, references)

    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Generated %d figure PDFs in %s", len(generated), FIGURES_DIR)
    LOGGER.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
