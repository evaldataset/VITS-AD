"""Confirmatory test of the frozen `tail_ratio` regime selector.

Executes the pre-registered plan in
`paper/rebuttal_package/PREREGISTRATION_confirmatory.md`: sample series disjoint
from the 120-series discovery set, score both arms on identical windows, and
apply the **frozen** rule (threshold from `results/regime_proxy/frozen_rule.json`,
fitted on discovery data only). Nothing here re-tunes the threshold.

Supports sharding so several GPUs can process disjoint slices concurrently:
    CUDA_VISIBLE_DEVICES=1 python scripts/run_confirmatory_regime.py --shard 0 --num-shards 3

Each shard writes `shard_<i>.json`; `--analyze` then merges the shards and runs
the pre-specified tests.

Output: results/regime_confirmatory/{shard_*.json, confirmatory_result.json}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from scripts.regime_proxy_search import compute_proxies
from scripts.run_tsb_ad_vision_paired import (
    MAX_FLATTEN_DIM,
    M_STRATA,
    WINDOW_SIZE,
    anomaly_locality,
    run_series,
)
from src.models.backbone import VisionBackbone

LOGGER = logging.getLogger(__name__)

AUDIT_ROOT = Path("results/tsb_ad_raw_audit")
DATA_ROOT = Path("data/tsb_ad")
DISCOVERY_SELECTION = Path("results/tsb_ad_vision_paired/selection.json")
FROZEN_RULE = Path("results/regime_proxy/frozen_rule.json")
OUT_ROOT = Path("results/regime_confirmatory")

TARGET_N = 250
SEED = 20260820
#: Univariate share of the confirmatory sample, mirroring discovery (60/120).
U_FRACTION = 0.5


def build_confirmatory_selection() -> list[dict[str, Any]]:
    """Sample series disjoint from discovery, with the discovery stratification."""
    discovery = {
        (item["subset"], item["series"])
        for item in json.loads(DISCOVERY_SELECTION.read_text())
    }
    rng = np.random.default_rng(SEED)
    selection: list[dict[str, Any]] = []

    for subset in ("TSB-AD-U", "TSB-AD-M"):
        records = json.loads((AUDIT_ROOT / subset / "per_series.json").read_text())
        pool: list[dict[str, Any]] = []
        for record in records:
            key = (subset, record["series"])
            if key in discovery:
                continue
            # The rule falls back to raw-flatten, so series without it are excluded.
            if WINDOW_SIZE * int(record["n_channels"]) > MAX_FLATTEN_DIM:
                continue
            path = DATA_ROOT / subset / record["series"]
            labels = pd.read_csv(path, usecols=["Label"])["Label"].to_numpy(np.int64)
            pool.append({
                "subset": subset,
                "series": record["series"],
                "n_channels": int(record["n_channels"]),
                "locality": anomaly_locality(labels),
            })

        if subset == "TSB-AD-U":
            want = int(TARGET_N * U_FRACTION)
            localities = np.array([item["locality"] for item in pool])
            edges = np.percentile(localities, [33.3, 66.7])
            per_tercile = want // 3
            for tercile in range(3):
                if tercile == 0:
                    bucket = [p for p in pool if p["locality"] <= edges[0]]
                elif tercile == 1:
                    bucket = [p for p in pool if edges[0] < p["locality"] <= edges[1]]
                else:
                    bucket = [p for p in pool if p["locality"] > edges[1]]
                take = min(per_tercile, len(bucket))
                for index in rng.choice(len(bucket), size=take, replace=False):
                    item = dict(bucket[int(index)])
                    item["stratum"] = f"U_locality_t{tercile + 1}"
                    selection.append(item)
        else:
            want = TARGET_N - int(TARGET_N * U_FRACTION)
            per_band = want // len(M_STRATA)
            for lo, hi, _ in M_STRATA:
                bucket = [p for p in pool if lo <= p["n_channels"] <= hi]
                take = min(per_band, len(bucket))
                for index in rng.choice(len(bucket), size=take, replace=False):
                    item = dict(bucket[int(index)])
                    item["stratum"] = f"M_D{lo}-{hi}"
                    selection.append(item)

    LOGGER.info("confirmatory selection: %d series (disjoint from discovery)",
                len(selection))
    return selection


def run_shard(shard: int, num_shards: int, max_windows: int) -> None:
    """Score both arms plus the proxy for this shard's slice."""
    selection = build_confirmatory_selection()
    mine = [s for i, s in enumerate(selection) if i % num_shards == shard]
    LOGGER.info("shard %d/%d -> %d series", shard, num_shards, len(mine))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"shard_{shard}.json"
    records: list[dict[str, Any]] = []
    if out_path.exists():
        try:
            records = json.loads(out_path.read_text())
            LOGGER.info("resuming shard with %d records", len(records))
        except Exception:
            records = []
    done = {r["series"] for r in records}
    mine = [s for s in mine if s["series"] not in done]

    backbone = VisionBackbone("facebook/dinov2-base")
    for index, item in enumerate(mine, start=1):
        path = DATA_ROOT / item["subset"] / item["series"]
        try:
            scored = run_series(item, backbone, max_windows)
            if scored is None or not (scored.get("vision") and scored.get("raw_flatten")):
                continue
            proxies = compute_proxies(path)
            if proxies is None:
                continue
            records.append({
                "subset": item["subset"], "series": item["series"],
                "stratum": item["stratum"], "n_channels": item["n_channels"],
                "tail_ratio": float(proxies["tail_ratio"]),
                "z_ratio_labelled": float(proxies["z_ratio_labelled"]),
                "raw_auc": float(scored["raw_flatten"]["AUC-ROC"]),
                "vision_auc": float(scored["vision"]["AUC-ROC"]),
            })
        except Exception as exc:
            LOGGER.warning("%s failed: %s", item["series"], exc)
        if index % 5 == 0 or index == len(mine):
            out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            LOGGER.info("shard %d: %d/%d (%d scored)", shard, index, len(mine),
                        len(records))
    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("shard %d complete: %d records -> %s", shard, len(records), out_path)


def analyze() -> None:
    """Merge shards and run the pre-registered tests with the frozen threshold."""
    rule = json.loads(FROZEN_RULE.read_text())
    threshold = float(rule["threshold"])
    records: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("shard_*.json")):
        records.extend(json.loads(path.read_text()))
    LOGGER.info("confirmatory n=%d, frozen threshold=%.6f", len(records), threshold)
    if len(records) < 20:
        LOGGER.error("too few confirmatory series")
        return

    vision = np.array([r["vision_auc"] for r in records])
    raw = np.array([r["raw_auc"] for r in records])
    tail = np.array([r["tail_ratio"] for r in records])
    use_vision = tail < threshold
    realised = np.where(use_vision, vision, raw)
    oracle = np.maximum(vision, raw)

    def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        if not np.any(a != b):
            return 0.0, float("nan")
        return float((a - b).mean()), float(stats.wilcoxon(a, b).pvalue)

    d_vision, p_vision = paired(realised, vision)
    d_raw, p_raw = paired(realised, raw)
    rho = stats.spearmanr(tail, vision - raw)

    headroom = float(oracle.mean() - vision.mean())
    result = {
        "n": len(records),
        "frozen_threshold": threshold,
        "vision_selected_fraction": float(use_vision.mean()),
        "means": {
            "rule": float(realised.mean()),
            "always_vision": float(vision.mean()),
            "always_raw": float(raw.mean()),
            "oracle": float(oracle.mean()),
        },
        "H1_rule_vs_always_vision": {
            "delta": d_vision, "p": p_vision,
            "confirmed": bool(p_vision < 0.05 and d_vision > 0),
        },
        "H2_rule_vs_always_raw": {"delta": d_raw, "p": p_raw},
        "H3_tail_ratio_vs_delta": {
            "spearman": float(rho.statistic), "p": float(rho.pvalue),
        },
        "oracle_headroom_recovered": (d_vision / headroom) if headroom > 0 else None,
    }
    (OUT_ROOT / "confirmatory_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    LOGGER.info("=== CONFIRMATORY RESULT (pre-registered) ===")
    LOGGER.info("  n=%d, vision selected %.0f%%", result["n"],
                100 * result["vision_selected_fraction"])
    m = result["means"]
    LOGGER.info("  rule=%.4f  always_vision=%.4f  always_raw=%.4f  oracle=%.4f",
                m["rule"], m["always_vision"], m["always_raw"], m["oracle"])
    LOGGER.info("  H1 vs always-vision: delta=%+.4f p=%.4f -> %s",
                d_vision, p_vision,
                "CONFIRMED" if result["H1_rule_vs_always_vision"]["confirmed"]
                else "NOT confirmed")
    LOGGER.info("  H2 vs always-raw:    delta=%+.4f p=%.4f", d_raw, p_raw)
    LOGGER.info("  H3 rho(tail_ratio, delta)=%+.3f p=%.3g",
                result["H3_tail_ratio_vs_delta"]["spearman"],
                result["H3_tail_ratio_vs_delta"]["p"])
    LOGGER.info("wrote %s", OUT_ROOT / "confirmatory_result.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--shard", type=int, default=0)
    _ = parser.add_argument("--num-shards", type=int, default=1)
    _ = parser.add_argument("--max-windows", type=int, default=1000)
    _ = parser.add_argument("--analyze", action="store_true")
    _ = parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        analyze()
        return
    if args.select_only:
        selection = build_confirmatory_selection()
        counts: dict[str, int] = {}
        for item in selection:
            counts[item["stratum"]] = counts.get(item["stratum"], 0) + 1
        for stratum, count in sorted(counts.items()):
            LOGGER.info("  %-16s %d", stratum, count)
        return
    run_shard(int(args.shard), int(args.num_shards), int(args.max_windows))


if __name__ == "__main__":
    main()
