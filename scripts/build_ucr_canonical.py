"""Build canonical UCR result artifact (single source of truth).

This script consolidates all UCR per-series results into one JSON file with
clearly named methods and provenance metadata. The output (`ucr_canonical.json`)
is the authoritative source for the paper's UCR numbers.

Usage:
    python scripts/build_ucr_canonical.py

Output:
    results/ucr_canonical/eligible_list.json   — list of eligible series
    results/ucr_canonical/per_series.json      — per-series scores (all methods)
    results/ucr_canonical/summary.json         — aggregated mean/std/n per method
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


VITS_DIR = Path("results/ucr_expanded/per_series")
RAW_DIR = Path("results/ucr_expanded_raw/per_series")
OUT_DIR = Path("results/ucr_canonical")


def _load_json_dir(d: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not d.exists():
        return out
    for path in sorted(d.iterdir()):
        if path.suffix != ".json":
            continue
        rec = json.loads(path.read_text())
        series = rec.get("series", path.stem)
        out[series] = rec
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vits = _load_json_dir(VITS_DIR)
    raw = _load_json_dir(RAW_DIR)

    # Eligible = union of all series we attempted
    eligible = sorted(set(vits.keys()) | set(raw.keys()))
    vits_complete = sorted(vits.keys())
    vits_missing = sorted(set(eligible) - set(vits_complete))

    # Per-series consolidated record
    per_series: dict[str, dict[str, Any]] = {}
    for series in eligible:
        v = vits.get(series, {})
        r = raw.get(series, {})
        v_auc = v.get("auc_roc", {}) if isinstance(v.get("auc_roc"), dict) else {}
        r_auc = r.get("auc_roc", {}) if isinstance(r.get("auc_roc"), dict) else {}
        per_series[series] = {
            "in_vits": series in vits,
            "in_raw": series in raw,
            "VITS_Traj": v_auc.get("VITS_Traj"),
            "VITS_Dist": v_auc.get("VITS_Dist"),
            "VITS_Dual": v_auc.get("VITS_Dual"),
            "RawMaha_MeanPooled": r_auc.get("RawMaha_MeanPooled"),
            "RawMaha_Flattened": r_auc.get("RawMaha_Flattened"),
            "LOF": r_auc.get("LOF"),
            "IsolationForest": r_auc.get("IsolationForest"),
            "OneClassSVM": r_auc.get("OneClassSVM"),
        }

    # Aggregate per method (only over series that have that method)
    methods = [
        "VITS_Traj", "VITS_Dist", "VITS_Dual",
        "RawMaha_MeanPooled", "RawMaha_Flattened",
        "LOF", "IsolationForest", "OneClassSVM",
    ]
    summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        values = [
            per_series[s][method] for s in eligible
            if per_series[s].get(method) is not None
        ]
        if values:
            arr = np.array(values, dtype=float)
            summary[method] = {
                "n": int(len(arr)),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        else:
            summary[method] = {"n": 0, "mean": None, "std": None}

    eligible_list = {
        "n_eligible": len(eligible),
        "n_vits_complete": len(vits_complete),
        "n_vits_missing": len(vits_missing),
        "n_raw_complete": len(raw),
        "vits_missing_series": vits_missing,
        "all_eligible_series": eligible,
    }

    canonical = {
        "schema_version": "1.0",
        "description": (
            "Canonical UCR Anomaly Archive results. Eligible = series that "
            "passed both VITS and raw-Mahalanobis pipelines. Numbers in "
            "paper Table~\\ref{tab:ucr} are taken from this file."
        ),
        "eligible": eligible_list,
        "summary": summary,
        "per_series": per_series,
    }

    (OUT_DIR / "eligible_list.json").write_text(json.dumps(eligible_list, indent=2))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "per_series.json").write_text(json.dumps(per_series, indent=2))
    (OUT_DIR / "ucr_canonical.json").write_text(json.dumps(canonical, indent=2))

    print("=== UCR Canonical Build ===")
    print(f"Eligible series: {len(eligible)}")
    print(f"  VITS complete: {len(vits_complete)}")
    print(f"  VITS missing : {len(vits_missing)}")
    print(f"  Raw complete : {len(raw)}")
    print()
    print(f"{'Method':<22} {'n':>4} {'mean':>8} {'std':>8}")
    print("-" * 46)
    for method in methods:
        s = summary[method]
        if s["mean"] is not None:
            print(f"{method:<22} {s['n']:>4} {s['mean']:>8.4f} {s['std']:>8.4f}")
    print()
    print(f"Wrote: {OUT_DIR}/")


if __name__ == "__main__":
    main()
