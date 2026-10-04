#!/usr/bin/env python3
"""Collect all experiment results and generate comprehensive comparison tables.

Gathers:
- PatchTraj default (LP, RP, ensemble)
- PatchTraj improved (LP, RP)
- TimesNet baseline
- CATCH baseline
- Ablation study
- Multi-scale experiment
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")
REPORTS_DIR = RESULTS_ROOT / "reports"

DATASETS = ["smd", "psm", "msl", "smap"]


def _load_json(path: Path) -> dict | None:
    """Load JSON file or return None."""
    if path.exists():
        return json.loads(path.read_text())
    return None


def collect_default_patchtraj() -> dict[str, dict]:
    """Collect default PatchTraj results (from before_after_summary.csv)."""
    results: dict[str, dict] = {}
    csv_path = REPORTS_DIR / "before_after_summary.csv"
    if csv_path.exists():
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row["dataset"]
                results[ds] = {
                    "default_ensemble": float(row["after_auc_roc"]),
                    "w_lp": float(row["w_lp"]),
                }

    # Also get individual LP/RP from paper_table.md parsing or raw results
    paper_path = REPORTS_DIR / "paper_table.md"
    if paper_path.exists():
        for line in paper_path.read_text().splitlines():
            if (
                line.startswith("|")
                and not line.startswith("| Dataset")
                and "---" not in line
            ):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 4:
                    ds_name = parts[0].lower().split("(")[0].strip()
                    if ds_name in results:
                        results[ds_name]["default_lp"] = float(parts[1])
                        results[ds_name]["default_rp"] = float(parts[2])
    return results


def collect_improved_patchtraj() -> dict[str, dict]:
    """Collect improved PatchTraj results."""
    results: dict[str, dict] = {}

    for ds in DATASETS:
        ds_dir = RESULTS_ROOT / f"improved_{ds}"
        if not ds_dir.exists():
            continue

        lp_aucs = []
        rp_aucs = []

        for entity_dir in sorted(ds_dir.iterdir()):
            if not entity_dir.is_dir():
                continue
            for renderer, target_list in [
                ("line_plot", lp_aucs),
                ("recurrence_plot", rp_aucs),
            ]:
                m = _load_json(entity_dir / renderer / "metrics.json")
                if m:
                    target_list.append(m["auc_roc"])

        if lp_aucs or rp_aucs:
            results[ds] = {
                "improved_lp_mean": float(np.mean(lp_aucs)) if lp_aucs else None,
                "improved_rp_mean": float(np.mean(rp_aucs)) if rp_aucs else None,
                "improved_lp_count": len(lp_aucs),
                "improved_rp_count": len(rp_aucs),
                "improved_lp_all": lp_aucs,
                "improved_rp_all": rp_aucs,
            }
    return results


def collect_baselines() -> dict[str, dict]:
    """Collect TimesNet and CATCH baseline results."""
    results: dict[str, dict] = {}

    for ds in DATASETS:
        results[ds] = {}

        # TimesNet
        tn = _load_json(
            REPORTS_DIR / f"dl_baselines/timesnet_{ds}/timesnet_results.json"
        )
        if tn:
            results[ds]["timesnet_auc_roc"] = tn["metrics"]["auc_roc"]
            results[ds]["timesnet_auc_pr"] = tn["metrics"]["auc_pr"]
            results[ds]["timesnet_f1_pa"] = tn["metrics"]["f1_pa"]

        # CATCH
        ct = _load_json(REPORTS_DIR / f"dl_baselines/catch_{ds}/catch_results.json")
        if ct:
            results[ds]["catch_auc_roc"] = ct["metrics"]["auc_roc"]
            results[ds]["catch_auc_pr"] = ct["metrics"]["auc_pr"]
            results[ds]["catch_f1_pa"] = ct["metrics"]["f1_pa"]

        ts2vec = _load_json(REPORTS_DIR / f"dl_baselines/ts2vec_{ds}/ts2vec_results.json")
        if not ts2vec:
            ts2vec = _load_json(
                REPORTS_DIR / f"ts2vec_baselines/ts2vec_{ds}/ts2vec_results.json"
            )
        if ts2vec:
            results[ds]["ts2vec_auc_roc"] = ts2vec["metrics"]["auc_roc"]
            results[ds]["ts2vec_auc_pr"] = ts2vec["metrics"]["auc_pr"]
            results[ds]["ts2vec_f1_pa"] = ts2vec["metrics"]["f1_pa"]

    return results


def collect_ablation() -> list[dict]:
    """Collect ablation study results."""
    ablation_dir = RESULTS_ROOT / "ablation_smd" / "machine-1-1" / "line_plot"
    results = []
    if ablation_dir.exists():
        for variant_dir in sorted(ablation_dir.iterdir()):
            if variant_dir.is_dir():
                m = _load_json(variant_dir / "metrics.json")
                if m:
                    results.append({"variant": variant_dir.name, **m})
    return results


def collect_multiscale() -> dict:
    """Collect multi-scale experiment results."""
    results = {}
    ms_dir = RESULTS_ROOT / "multiscale_smd" / "machine-1-1"
    if ms_dir.exists():
        for ws_dir in sorted(ms_dir.iterdir()):
            if ws_dir.is_dir() and ws_dir.name.startswith("w"):
                for r in ["line_plot", "recurrence_plot"]:
                    m = _load_json(ws_dir / r / "metrics.json")
                    if m:
                        key = f"{ws_dir.name}/{r}"
                        results[key] = m["auc_roc"]

        # Ensemble
        ens = _load_json(ms_dir / "multiscale_ensemble_metrics.json")
        if ens:
            results["ensemble"] = ens["auc_roc"]

    return results


def collect_improved_ensemble() -> dict[str, dict]:
    """Collect improved model ensemble results."""
    path = REPORTS_DIR / "improved_ensemble_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def format_val(val: float | None, fmt: str = ".4f") -> str:
    """Format a value or return '-'."""
    if val is None:
        return "-"
    return f"{val:{fmt}}"


def generate_report() -> str:
    """Generate comprehensive markdown report."""
    default = collect_default_patchtraj()
    improved = collect_improved_patchtraj()
    baselines = collect_baselines()
    ablation = collect_ablation()
    multiscale = collect_multiscale()
    imp_ensemble = collect_improved_ensemble()

    lines = ["# VITS Comprehensive Results Report", ""]
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Table 1: Cross-method comparison (AUC-ROC)
    lines.append("## Table 1: AUC-ROC Comparison Across Methods")
    lines.append("")
    lines.append(
        "| Dataset | TimesNet | CATCH | PatchTraj (Default) | PatchTraj (Improved) | Improved Ensemble |"
    )
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")

    for ds in DATASETS:
        ds_label = "SMD (28-avg)" if ds == "smd" else ds.upper()

        tn_val = baselines.get(ds, {}).get("timesnet_auc_roc")
        ct_val = baselines.get(ds, {}).get("catch_auc_roc")
        def_val = default.get(ds, {}).get("default_ensemble")

        imp = improved.get(ds, {})
        imp_lp = imp.get("improved_lp_mean")
        imp_rp = imp.get("improved_rp_mean")
        imp_n_lp = imp.get("improved_lp_count", 0)
        imp_n_rp = imp.get("improved_rp_count", 0)

        # For improved, show best of LP/RP
        if imp_lp is not None and imp_rp is not None:
            imp_best = max(imp_lp, imp_rp)
            imp_str = f"{imp_best:.4f} (LP:{imp_lp:.3f}/RP:{imp_rp:.3f}, n={imp_n_lp})"
        elif imp_lp is not None:
            imp_str = f"{imp_lp:.4f} (LP only, n={imp_n_lp})"
        elif imp_rp is not None:
            imp_str = f"{imp_rp:.4f} (RP only, n={imp_n_rp})"
        else:
            imp_str = "running..."

        # Improved ensemble result
        ie = imp_ensemble.get(ds, {})
        ie_config = ie.get("best_config", {})
        ie_auc = ie_config.get("avg_auc_roc")
        ie_n = ie_config.get("n_entities", 0)
        if ie_auc is not None:
            ie_str = f"{ie_auc:.4f} (n={ie_n})"
        else:
            ie_str = "-"

        # Bold the best
        vals = [v for v in [tn_val, ct_val, def_val] if v is not None]
        best_val = max(vals) if vals else 0

        def _fmt(v: float | None) -> str:
            if v is None:
                return "-"
            s = f"{v:.4f}"
            if v == best_val and len(vals) > 1:
                return f"**{s}**"
            return s

        lines.append(
            f"| {ds_label} | {_fmt(tn_val)} | {_fmt(ct_val)} | {_fmt(def_val)} | {imp_str} | {ie_str} |"
        )

    lines.append("")

    # Table 2: Detailed default PatchTraj
    lines.append("## Table 2: PatchTraj Default — Per-Renderer and Ensemble")
    lines.append("")
    lines.append("| Dataset | LP AUC-ROC | RP AUC-ROC | Ensemble AUC-ROC |")
    lines.append("|---|:---:|:---:|:---:|")
    for ds in DATASETS:
        d = default.get(ds, {})
        ds_label = "SMD (28-avg)" if ds == "smd" else ds.upper()
        lines.append(
            f"| {ds_label} | {format_val(d.get('default_lp'))} | {format_val(d.get('default_rp'))} | {format_val(d.get('default_ensemble'))} |"
        )
    lines.append("")

    # Table 3: Improved PatchTraj per-entity (for SMD)
    if "smd" in improved:
        imp_smd = improved["smd"]
        lines.append(
            "## Table 3: Improved PatchTraj — SMD Per-Entity (completed so far)"
        )
        lines.append("")
        lines.append("| Entity | LP AUC-ROC | RP AUC-ROC |")
        lines.append("|---|:---:|:---:|")

        smd_dir = RESULTS_ROOT / "improved_smd"
        for entity_dir in sorted(smd_dir.iterdir()):
            if not entity_dir.is_dir():
                continue
            lp_m = _load_json(entity_dir / "line_plot" / "metrics.json")
            rp_m = _load_json(entity_dir / "recurrence_plot" / "metrics.json")
            lines.append(
                f"| {entity_dir.name} | {format_val(lp_m['auc_roc'] if lp_m else None)} | {format_val(rp_m['auc_roc'] if rp_m else None)} |"
            )
        lines.append(
            f"\n**Average**: LP={format_val(imp_smd.get('improved_lp_mean'))}, "
            f"RP={format_val(imp_smd.get('improved_rp_mean'))}"
        )
        lines.append(
            f"**Count**: LP={imp_smd.get('improved_lp_count')}/{28 if ds == 'smd' else '?'}, "
            f"RP={imp_smd.get('improved_rp_count')}/{28 if ds == 'smd' else '?'}"
        )
        lines.append("")

    # Table 4: Baselines detail
    lines.append("## Table 4: Deep Learning Baselines — Full Metrics")
    lines.append("")
    lines.append("| Dataset | Method | AUC-ROC | AUC-PR | F1-PA |")
    lines.append("|---|---|:---:|:---:|:---:|")
    for ds in DATASETS:
        ds_label = "SMD" if ds == "smd" else ds.upper()
        b = baselines.get(ds, {})
        if "timesnet_auc_roc" in b:
            lines.append(
                f"| {ds_label} | TimesNet | {b['timesnet_auc_roc']:.4f} | {b['timesnet_auc_pr']:.4f} | {b['timesnet_f1_pa']:.4f} |"
            )
        if "catch_auc_roc" in b:
            lines.append(
                f"| {ds_label} | CATCH | {b['catch_auc_roc']:.4f} | {b['catch_auc_pr']:.4f} | {b['catch_f1_pa']:.4f} |"
            )
    lines.append("")

    # Table 5: Ablation
    if ablation:
        lines.append("## Table 5: Ablation Study (machine-1-1, line_plot)")
        lines.append("")
        lines.append("| Variant | AUC-ROC | AUC-PR | F1-PA |")
        lines.append("|---|:---:|:---:|:---:|")
        baseline_auc = None
        for a in sorted(ablation, key=lambda x: x["auc_roc"], reverse=True):
            if a["variant"] == "baseline":
                baseline_auc = a["auc_roc"]
            lines.append(
                f"| {a['variant']} | {a['auc_roc']:.4f} | {a['auc_pr']:.4f} | {a['f1_pa']:.4f} |"
            )
        if baseline_auc:
            lines.append("")
            lines.append("**Key takeaways:**")
            for a in sorted(ablation, key=lambda x: x["auc_roc"], reverse=True):
                delta = a["auc_roc"] - baseline_auc
                sign = "+" if delta >= 0 else ""
                lines.append(f"- {a['variant']}: {sign}{delta:.4f} vs baseline")
        lines.append("")

    # Table 6: Multiscale
    if multiscale:
        lines.append("## Table 6: Multi-Scale Experiment (machine-1-1)")
        lines.append("")
        lines.append("| Window | Line Plot | Recurrence Plot |")
        lines.append("|---|:---:|:---:|")
        for ws in ["w50", "w100", "w200"]:
            lp = multiscale.get(f"{ws}/line_plot")
            rp = multiscale.get(f"{ws}/recurrence_plot")
            lines.append(f"| {ws} | {format_val(lp)} | {format_val(rp)} |")
        ens = multiscale.get("ensemble")
        if ens:
            lines.append(f"| **Ensemble** | | **{ens:.4f}** |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Generate and save comprehensive results report."""
    logging.basicConfig(level=logging.INFO)

    report = generate_report()
    output_path = REPORTS_DIR / "comprehensive_results.md"
    output_path.write_text(report)
    LOGGER.info("Report saved to %s", output_path)
    print(report)


if __name__ == "__main__":
    main()
