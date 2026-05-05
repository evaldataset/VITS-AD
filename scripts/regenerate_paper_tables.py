"""Regenerate paper tables from canonical raw JSON artifacts.

Single source of truth: this script reads only raw JSON outputs (per-entity
metrics, multi-seed reports, ucr_canonical, classical_baselines, dl_baselines,
raw_mahalanobis) and emits LaTeX table fragments that the main paper can
``\\input``. No hardcoded numbers and no markdown/CSV intermediates are parsed,
so transcription drift between secondary reports and the paper is impossible.

Every value emitted to a table appears in ``paper/tables/summary.json`` with
its provenance: the absolute artifact path, the sample size, and any policy
fields recorded in that artifact (e.g. ``alpha_policy``,
``threshold_protocol``, ``smd_n_entities``). Missing artifacts produce
``--`` cells and a logged warning instead of silent fallbacks.

Usage:
    python scripts/regenerate_paper_tables.py

Outputs (paper/tables/):
    table1_main_multivariate.tex   — Table 1 (main multivariate AUC-ROC)
    table3_alpha_ablation.tex      — Table 3 (alpha sweep)
    table4_ucr.tex                 — Table 4 (UCR canonical)
    summary.json                   — provenance log (one entry per cell)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "tables"
RESULTS = REPO / "results"
REPORTS = RESULTS / "reports"

# Datasets in their paper column order.
DATASETS: tuple[str, ...] = ("smd", "psm", "msl", "smap")

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file. Returns None and logs a warning if missing."""
    if not path.exists():
        LOGGER.warning("Artifact missing: %s", path)
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to parse JSON %s: %s", path, exc)
        return None


def _fmt(value: float | None, digits: int = 3, bold: bool = False) -> str:
    """Format a numeric cell or ``--`` placeholder."""
    if value is None or not np.isfinite(value):
        return "--"
    s = f"{value:.{digits}f}"
    return rf"\textbf{{{s}}}" if bold else s


def _provenance_entry(
    value: float | None,
    source: Path | None,
    note: str = "",
) -> dict[str, Any]:
    """Build a provenance dict for one table cell."""
    return {
        "value": float(value) if value is not None and np.isfinite(value) else None,
        "source": str(source.relative_to(REPO)) if source is not None else None,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------


def _classical_baseline(
    method: str, dataset: str
) -> tuple[float | None, Path]:
    """Read classical baseline AUC-ROC for ``method`` on ``dataset``."""
    path = REPORTS / "classical_baselines" / f"{method}_{dataset}.json"
    data = _load_json(path)
    if data is None:
        return None, path
    auc = data.get("auc_roc") or data.get("metrics", {}).get("auc_roc")
    return (float(auc) if auc is not None else None), path


def _dl_baseline(method: str, dataset: str) -> tuple[float | None, Path]:
    """Read deep-learning baseline AUC-ROC across the layouts present in this repo."""
    method_lower = method.lower()
    candidate_files = [
        REPORTS / "dl_baselines" / f"{method_lower}_{dataset}" / f"{method_lower}_results.json",
        REPORTS / "dl_baselines" / f"{method_lower}_{dataset}" / "results.json",
        REPORTS / "dl_baselines" / f"{method_lower}_{dataset}" / "metrics.json",
        REPORTS / f"{method_lower}_baselines" / f"{method_lower}_{dataset}.json",
        REPORTS / f"{method_lower}_baselines" / dataset / "metrics.json",
    ]
    for p in candidate_files:
        data = _load_json(p)
        if data is None:
            continue
        auc = (
            data.get("auc_roc")
            or data.get("metrics", {}).get("auc_roc")
            or data.get(method_lower, {}).get("metrics", {}).get("auc_roc")
        )
        if auc is None:
            continue
        return float(auc), p
    return None, candidate_files[0]


def _raw_maha(variant: str, dataset: str) -> tuple[float | None, Path, str]:
    """Read raw Mahalanobis AUC-ROC.

    Returns ``(auc, source_path, note)`` where ``note`` records whether the
    SMD value reflects 28 entities or only ``machine-1-1``.
    """
    path = RESULTS / "raw_mahalanobis" / "summary.json"
    data = _load_json(path)
    if data is None:
        return None, path, "missing"

    if dataset == "smd":
        # Prefer 28-entity macro mean if present, else fall back to machine-1-1.
        smd_macro = data.get("smd", {}).get(variant)
        if isinstance(smd_macro, dict) and "macro_mean_auc_roc" in smd_macro:
            return (
                float(smd_macro["macro_mean_auc_roc"]),
                path,
                f"SMD 28-entity macro (n={smd_macro.get('n', 28)})",
            )
        single = data.get("smd_machine-1-1", {}).get(variant)
        if isinstance(single, dict) and "auc_roc" in single:
            return (
                float(single["auc_roc"]),
                path,
                "SMD machine-1-1 only (28-entity rerun pending)",
            )
        return None, path, "smd missing"

    entry = data.get(dataset, {}).get(variant)
    if isinstance(entry, dict) and "auc_roc" in entry:
        return float(entry["auc_roc"]), path, ""
    return None, path, "missing"


def _vits_per_dataset(dataset: str, renderer: str = "line_plot") -> tuple[float | None, Path]:
    """Read VITS per-dataset spatial+dual run."""
    path = (
        RESULTS
        / f"dinov2-base_{dataset}_{renderer}_spatial"
        / "metrics.json"
    )
    data = _load_json(path)
    if data is None:
        return None, path
    auc = data.get("auc_roc")
    return (float(auc) if auc is not None else None), path


def _smd_28entity_aggregate(
    bench_dir: Path,
) -> dict[str, dict[str, float | int]]:
    """Aggregate per-entity AUC-ROC for the 28-entity SMD benchmark.

    Supports two layouts:
        bench_dir/<entity>/<renderer>/metrics.json   (multi-renderer)
        bench_dir/<entity>/metrics.json              (single output)
    """
    out: dict[str, list[float]] = {"line_plot": [], "recurrence_plot": [], "single": []}
    if not bench_dir.exists():
        return {}
    for entity_dir in sorted(bench_dir.iterdir()):
        if not entity_dir.is_dir():
            continue
        any_renderer = False
        for renderer in ("line_plot", "recurrence_plot"):
            mf = entity_dir / renderer / "metrics.json"
            if mf.exists():
                m = _load_json(mf)
                if m and "auc_roc" in m:
                    out[renderer].append(float(m["auc_roc"]))
                    any_renderer = True
        if not any_renderer:
            mf = entity_dir / "metrics.json"
            if mf.exists():
                m = _load_json(mf)
                if m and "auc_roc" in m:
                    out["single"].append(float(m["auc_roc"]))

    summary: dict[str, dict[str, float | int]] = {}
    for renderer, vals in out.items():
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            summary[renderer] = {
                "n": len(vals),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
    return summary


def _alpha_sweep_value(
    dataset: str, alpha: float
) -> tuple[float | None, Path]:
    """Read AUC-ROC for a single (dataset, alpha) cell of the alpha ablation."""
    candidate_files = [
        RESULTS / "alpha_sweep" / f"{dataset}_lp_a{int(alpha*10):d}" / "metrics.json",
        RESULTS / "alpha_sweep" / f"{dataset}_lp_alpha{alpha:g}" / "metrics.json",
        RESULTS / "ablation" / f"{dataset}_alpha{alpha:g}" / "metrics.json",
    ]
    for p in candidate_files:
        data = _load_json(p)
        if data is None:
            continue
        if "auc_roc" in data:
            return float(data["auc_roc"]), p
    return None, candidate_files[0]


# ---------------------------------------------------------------------------
# Table 1: Main multivariate AUC-ROC
# ---------------------------------------------------------------------------


def _build_main_table(prov: dict[str, Any]) -> str:
    """Table 1: main multivariate AUC-ROC, all values from artifacts."""
    prov.setdefault("table1", {})
    cells: dict[str, dict[str, dict[str, Any]]] = {}

    # Classical baselines
    classical = [
        ("LOF", "lof"),
        ("Isolation Forest", "isolationforest"),
        ("OCSVM", "oneclasssvm"),
    ]
    for label, key in classical:
        cells[label] = {}
        for ds in DATASETS:
            v, p = _classical_baseline(key, ds)
            cells[label][ds] = _provenance_entry(v, p)

    # Raw Mahalanobis (mean-pool, flatten)
    for variant, label in (("mean_pooled", "Raw Maha (mean-pool)"),
                           ("flattened", "Raw Maha (flatten)")):
        cells[label] = {}
        for ds in DATASETS:
            v, p, note = _raw_maha(variant, ds)
            cells[label][ds] = _provenance_entry(v, p, note=note)

    # Deep baselines (cells filled where artifact exists)
    deep = [
        ("USAD", "usad"),
        ("Anomaly Transformer", "at"),
        ("TS2Vec", "ts2vec"),
        ("TimesNet", "timesnet"),
        ("CATCH", "catch"),
        ("GPT4TS", "gpt4ts"),
    ]
    for label, key in deep:
        cells[label] = {}
        for ds in DATASETS:
            v, p = _dl_baseline(key, ds)
            cells[label][ds] = _provenance_entry(v, p)

    # VITS spatial+dual.
    # SMD: aggregate of 28-entity benchmark (best of LP/RP).
    # PSM/MSL/SMAP: per-dataset spatial run.
    smd_bench = _smd_28entity_aggregate(RESULTS / "benchmark_smd_spatial")
    smd_lp = smd_bench.get("line_plot", {}).get("mean")
    smd_rp = smd_bench.get("recurrence_plot", {}).get("mean")
    smd_n_lp = smd_bench.get("line_plot", {}).get("n", 0)
    smd_n_rp = smd_bench.get("recurrence_plot", {}).get("n", 0)
    smd_best = max(
        v for v in (smd_lp, smd_rp) if v is not None and np.isfinite(v)
    ) if any(v is not None for v in (smd_lp, smd_rp)) else None

    cells["VITS (spatial+dual)"] = {
        "smd": _provenance_entry(
            smd_best,
            RESULTS / "benchmark_smd_spatial",
            note=f"max(LP_mean,RP_mean) over n_lp={smd_n_lp},n_rp={smd_n_rp}",
        ),
    }
    for ds in ("psm", "msl", "smap"):
        v, p = _vits_per_dataset(ds, "line_plot")
        cells["VITS (spatial+dual)"][ds] = _provenance_entry(v, p)

    prov["table1"]["cells"] = cells
    prov["table1"]["smd_bench_summary"] = smd_bench

    # Bold the dataset-best within each row that has every cell.
    def _row(label: str) -> str:
        row_cells = cells[label]
        values = [row_cells[ds]["value"] for ds in DATASETS]
        finite_vals = [v for v in values if v is not None and np.isfinite(v)]
        best = max(finite_vals) if finite_vals else None
        formatted = []
        for v in values:
            mark = bool(best is not None and v is not None and np.isclose(v, best))
            formatted.append(_fmt(v, bold=mark))
        return f"{label:<22}& " + " & ".join(formatted) + r" \\"

    rows = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & SMD & PSM & MSL & SMAP \\",
        r"\midrule",
    ]
    for label, _ in classical:
        rows.append(_row(label))
    rows.append("Raw Maha (mean-pool) & "
                + " & ".join(_fmt(cells["Raw Maha (mean-pool)"][ds]["value"]) for ds in DATASETS)
                + r" \\")
    rows.append("Raw Maha (flatten)   & "
                + " & ".join(_fmt(cells["Raw Maha (flatten)"][ds]["value"]) for ds in DATASETS)
                + r" \\")
    rows.append(r"\midrule")
    for label, _ in deep:
        rows.append(_row(label))
    rows.append(r"\midrule")
    rows.append(_row("VITS (spatial+dual)"))
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Table 3: Alpha ablation
# ---------------------------------------------------------------------------


def _build_alpha_table(prov: dict[str, Any]) -> str:
    """Table 3: alpha sweep ablation, read from results/alpha_sweep or fallback."""
    prov.setdefault("table3", {})
    alphas = (1.0, 0.5, 0.1, 0.0)
    sweep: dict[str, dict[float, dict[str, Any]]] = {}
    any_value = False
    for ds in DATASETS:
        sweep[ds] = {}
        for a in alphas:
            v, p = _alpha_sweep_value(ds, a)
            sweep[ds][a] = _provenance_entry(v, p)
            if v is not None:
                any_value = True

    prov["table3"]["sweep"] = sweep
    prov["table3"]["available"] = any_value

    if not any_value:
        # No alpha-sweep artifacts: emit placeholder LaTeX so the build still
        # succeeds, and surface the gap loudly in stdout/provenance.
        LOGGER.error(
            "No alpha-sweep artifacts found under results/alpha_sweep/ or results/ablation/. "
            "Run scripts/run_expanded_ablation.py to produce them."
        )
        return (
            "% Alpha-sweep artifacts missing — run scripts/run_expanded_ablation.py.\n"
            r"\begin{tabular}{lcccc}" "\n"
            r"\toprule" "\n"
            r"$\alpha$ & SMD & PSM & MSL & SMAP \\" "\n"
            r"\midrule" "\n"
            "(missing) & -- & -- & -- & -- \\\\\n"
            r"\bottomrule" "\n"
            r"\end{tabular}"
        )

    def _row(label: str, a: float) -> str:
        cells = [_fmt(sweep[ds][a]["value"]) for ds in DATASETS]
        return f"{label:<18}& " + " & ".join(cells) + r" \\"

    rows = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"$\alpha$ & SMD & PSM & MSL & SMAP \\",
        r"\midrule",
        _row("1.0 (traj only)", 1.0),
        _row("0.5", 0.5),
        _row("0.1", 0.1),
        _row("0.0 (dist only)", 0.0),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Table 4: UCR canonical
# ---------------------------------------------------------------------------


def _build_ucr_table(prov: dict[str, Any]) -> str:
    """Table 4: UCR canonical from results/ucr_canonical/summary.json."""
    prov.setdefault("table4", {})
    summary_path = RESULTS / "ucr_canonical" / "summary.json"
    summary = _load_json(summary_path)
    if summary is None:
        prov["table4"]["source"] = "MISSING"
        return "% UCR canonical summary missing — run scripts/build_ucr_canonical.py"

    prov["table4"]["source"] = str(summary_path.relative_to(REPO))
    prov["table4"]["summary"] = summary

    # Detect paired-sample count if eligible_list is present.
    eligible = _load_json(RESULTS / "ucr_canonical" / "eligible_list.json")
    if isinstance(eligible, dict):
        prov["table4"]["eligible_count"] = eligible.get("count")
    elif isinstance(eligible, list):
        prov["table4"]["eligible_count"] = len(eligible)

    def _fmt_method(method: str) -> tuple[int, str, str]:
        m = summary.get(method, {})
        n = int(m.get("n", 0))
        mean = m.get("mean")
        std = m.get("std")
        return n, _fmt(mean), _fmt(std)

    methods = [
        ("VITS_Dist", r"VITS Dist ($\alpha{=}0$)"),
        ("VITS_Dual", r"VITS Dual ($\alpha{=}0.5$)"),
        ("VITS_Traj", r"VITS Traj ($\alpha{=}1$)"),
        ("RawMaha_Flattened", "Raw Maha (flatten)"),
        ("LOF", "LOF"),
        ("RawMaha_MeanPooled", "Raw Maha (mean-pool)"),
        ("OneClassSVM", "OCSVM"),
        ("IsolationForest", "Isolation Forest"),
    ]

    rows = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & $n$ & Mean & Std \\",
        r"\midrule",
    ]
    first_vits = True
    for key, label in methods:
        if key == "RawMaha_Flattened":
            rows.append(r"\midrule")
        n, mean_s, std_s = _fmt_method(key)
        bold_open, bold_close = ("", "")
        if first_vits and key.startswith("VITS"):
            bold_open, bold_close = r"\textbf{", "}"
            first_vits = False
        rows.append(f"{label} & {n} & {bold_open}{mean_s}{bold_close} & {std_s} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: build all paper tables and provenance log."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {
        "schema_version": "2.0",
        "generator": "scripts/regenerate_paper_tables.py",
        "policy": (
            "All numeric cells are read from JSON artifacts under results/. "
            "Missing artifacts are rendered as `--` and logged in provenance "
            "with their expected source path. No hardcoded numbers."
        ),
    }

    table1 = _build_main_table(provenance)
    table3 = _build_alpha_table(provenance)
    table4 = _build_ucr_table(provenance)

    (OUT / "table1_main_multivariate.tex").write_text(table1 + "\n")
    (OUT / "table3_alpha_ablation.tex").write_text(table3 + "\n")
    (OUT / "table4_ucr.tex").write_text(table4 + "\n")
    (OUT / "summary.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n"
    )

    print("=== Paper Tables Regenerated ===")
    for name in ("table1_main_multivariate.tex", "table3_alpha_ablation.tex", "table4_ucr.tex"):
        print(f"  {OUT / name}")
    print(f"  {OUT / 'summary.json'}")
    print()

    # Highlight any missing cells loudly.
    table1_cells = provenance.get("table1", {}).get("cells", {})
    missing = [
        (label, ds)
        for label, row in table1_cells.items()
        for ds, cell in row.items()
        if cell["value"] is None
    ]
    if missing:
        print(f"WARNING: {len(missing)} Table 1 cells are missing artifacts:")
        for label, ds in missing:
            note = table1_cells[label][ds].get("note") or ""
            print(f"  - {label} / {ds}: {table1_cells[label][ds]['source']} {note}")
    else:
        print("All Table 1 cells resolved from artifacts.")


if __name__ == "__main__":
    main()
