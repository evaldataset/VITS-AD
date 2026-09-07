"""TSB-AD stage (b): paired raw-vs-vision comparison on a stratified subset.

Stage (a) established the raw-space control across 1,044 TSB-AD series. This stage
adds the vision arm on a stratified subset and tests the regime law out-of-sample.

Pre-registered predictions (from the controlled factorial in
`paper/rebuttal_package/synthetic_regime_STUDY.md`):
  P1  vision degrades as channel count D grows (RGB compression bottleneck);
  P2  vision degrades for highly *localized* anomalies (magnitude quantization),
      and TSB-AD anomalies are mostly localized, so raw should win broadly;
  P3  the vision arm is most competitive on univariate, less-localized series.

Both arms are run on the *identical* windows so every comparison is paired:
    raw_flatten / raw_meanpool : Ledoit-Wolf Mahalanobis on the raw windows
    vision                     : line-plot render -> frozen DINOv2 -> mean-pool
                                 patch tokens -> Ledoit-Wolf Mahalanobis

Because rendering dominates cost, each series is capped at ``--max-windows`` test
windows via an adaptive stride (disclosed in the output JSON). Stage (a)'s fixed
stride-10 numbers are therefore NOT directly comparable to these; the paired
comparison here is self-contained.

Output: results/tsb_ad_vision_paired/{selection,per_series,summary}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from scripts.run_raw_mahalanobis_baseline import RawMahalanobisScorer
from scripts.run_tsb_ad_raw_audit import load_series, parse_split
from src.data.base import create_sliding_windows
from src.evaluation.modern_metrics import compute_modern_metrics
from src.models.backbone import VisionBackbone
from src.rendering.line_plot import render_line_plot

LOGGER = logging.getLogger(__name__)

WINDOW_SIZE = 100
MIN_STRIDE = 10
MAX_FLATTEN_DIM = 2000
RENDER_BATCH = 64
DATA_ROOT = Path("data/tsb_ad")
AUDIT_ROOT = Path("results/tsb_ad_raw_audit")
OUT_ROOT = Path("results/tsb_ad_vision_paired")

#: Multivariate channel-count strata (lo, hi, n_to_sample).
M_STRATA: tuple[tuple[int, int, int], ...] = ((2, 5, 20), (11, 20, 20), (21, 1000, 20))
#: Univariate sample size, split evenly across locality terciles.
U_SAMPLE = 60


def anomaly_locality(labels: npt.NDArray[np.int64]) -> float:
    """Mean anomaly-segment length as a fraction of series length.

    Args:
        labels: Binary labels of shape ``(T,)``.

    Returns:
        Locality ratio in ``[0, 1]``; small means highly localized bursts.
    """
    diff = np.diff(np.concatenate(([0], labels, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        return 0.0
    return float((ends - starts).mean() / labels.shape[0])


def build_selection(seed: int) -> list[dict[str, Any]]:
    """Choose a stratified subset from the stage-(a) scored series.

    Args:
        seed: RNG seed for reproducible sampling within strata.

    Returns:
        List of selection records with stratification metadata.
    """
    rng = np.random.default_rng(seed)
    selection: list[dict[str, Any]] = []

    for subset in ("TSB-AD-U", "TSB-AD-M"):
        records = json.loads((AUDIT_ROOT / subset / "per_series.json").read_text())
        enriched: list[dict[str, Any]] = []
        for record in records:
            path = DATA_ROOT / subset / record["series"]
            labels = pd.read_csv(path, usecols=["Label"])["Label"].to_numpy(np.int64)
            enriched.append({
                "subset": subset,
                "series": record["series"],
                "n_channels": int(record["n_channels"]),
                "locality": anomaly_locality(labels),
            })

        if subset == "TSB-AD-U":
            localities = np.array([e["locality"] for e in enriched])
            edges = np.percentile(localities, [33.3, 66.7])
            per_tercile = U_SAMPLE // 3
            for tercile in range(3):
                if tercile == 0:
                    pool = [e for e in enriched if e["locality"] <= edges[0]]
                elif tercile == 1:
                    pool = [e for e in enriched
                            if edges[0] < e["locality"] <= edges[1]]
                else:
                    pool = [e for e in enriched if e["locality"] > edges[1]]
                take = min(per_tercile, len(pool))
                for idx in rng.choice(len(pool), size=take, replace=False):
                    item = dict(pool[int(idx)])
                    item["stratum"] = f"U_locality_t{tercile + 1}"
                    selection.append(item)
        else:
            for lo, hi, count in M_STRATA:
                pool = [e for e in enriched if lo <= e["n_channels"] <= hi]
                take = min(count, len(pool))
                for idx in rng.choice(len(pool), size=take, replace=False):
                    item = dict(pool[int(idx)])
                    item["stratum"] = f"M_D{lo}-{hi}"
                    selection.append(item)

    LOGGER.info("selected %d series", len(selection))
    return selection


def _vision_features(
    backbone: VisionBackbone, windows: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Render windows as line plots and mean-pool their patch tokens."""
    features: list[npt.NDArray[np.float32]] = []
    buffer: list[npt.NDArray[np.float32]] = []
    for index in range(windows.shape[0]):
        buffer.append(render_line_plot(windows[index].astype(np.float32)))
        if len(buffer) == RENDER_BATCH or index == windows.shape[0] - 1:
            tokens = backbone.extract_patch_tokens_from_numpy(
                np.stack(buffer).astype(np.float32)
            )
            features.append(tokens.mean(axis=1))
            buffer = []
    return np.concatenate(features, axis=0).astype(np.float64)


def _score(train_feat: npt.NDArray[np.float64], test_feat: npt.NDArray[np.float64],
           labels: npt.NDArray[np.int64]) -> dict[str, float]:
    scorer = RawMahalanobisScorer()
    scorer.fit(train_feat)
    scores = scorer.score(test_feat)
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite scores")
    return compute_modern_metrics(
        scores, labels, sliding_window=WINDOW_SIZE, verify_against_own=False
    )


def run_series(item: dict[str, Any], backbone: VisionBackbone,
               max_windows: int) -> dict[str, Any] | None:
    """Run raw and vision arms on identical windows for one series."""
    path = DATA_ROOT / item["subset"] / item["series"]
    train_end, _ = parse_split(path)
    values, labels = load_series(path)

    n_test_points = values.shape[0] - train_end
    stride = max(MIN_STRIDE, math.ceil(n_test_points / max(max_windows, 1)))

    train_raw, test_raw = values[:train_end], values[train_end:]
    mu = train_raw.mean(axis=0)
    sigma = train_raw.std(axis=0) + 1e-8
    train_windows, _ = create_sliding_windows(
        (train_raw - mu) / sigma, labels[:train_end], WINDOW_SIZE, stride
    )
    test_windows, win_labels = create_sliding_windows(
        (test_raw - mu) / sigma, labels[train_end:], WINDOW_SIZE, stride
    )
    if train_windows.shape[0] < 2 or test_windows.shape[0] < 2:
        return None
    if win_labels.sum() in (0, win_labels.shape[0]):
        return None

    record: dict[str, Any] = {
        **item,
        "stride": int(stride),
        "n_train_windows": int(train_windows.shape[0]),
        "n_test_windows": int(test_windows.shape[0]),
        "anomaly_window_rate": float(win_labels.mean()),
    }

    arms: dict[str, Any] = {
        "raw_meanpool": lambda w: w.mean(axis=1),
        "raw_flatten": lambda w: w.reshape(w.shape[0], -1),
    }
    for name, featurize in arms.items():
        if name == "raw_flatten" and WINDOW_SIZE * values.shape[1] > MAX_FLATTEN_DIM:
            record[name] = None
            continue
        try:
            record[name] = _score(
                featurize(train_windows).astype(np.float64),
                featurize(test_windows).astype(np.float64),
                win_labels,
            )
        except Exception as exc:
            LOGGER.warning("%s [%s] failed: %s", item["series"], name, exc)
            record[name] = None

    try:
        record["vision"] = _score(
            _vision_features(backbone, train_windows),
            _vision_features(backbone, test_windows),
            win_labels,
        )
    except Exception as exc:
        LOGGER.warning("%s [vision] failed: %s", item["series"], exc)
        record["vision"] = None

    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired vision-vs-raw deltas overall and per stratum."""
    def paired(subset_records: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
        pairs = [
            (r["vision"]["VUS-PR"], r[baseline]["VUS-PR"])
            for r in subset_records
            if r.get("vision") and r.get(baseline)
        ]
        if not pairs:
            return {"n": 0}
        vision = np.array([p[0] for p in pairs])
        base = np.array([p[1] for p in pairs])
        delta = vision - base
        out: dict[str, Any] = {
            "n": len(pairs),
            "vision_mean": float(vision.mean()),
            "baseline_mean": float(base.mean()),
            "delta_mean": float(delta.mean()),
            "vision_wins": int((delta > 0).sum()),
        }
        if len(pairs) > 1:
            try:
                from scipy.stats import wilcoxon
                _, pvalue = wilcoxon(vision, base)
                out["wilcoxon_p"] = float(pvalue)
            except Exception:
                pass
        return out

    summary: dict[str, Any] = {"overall": {}, "per_stratum": {}}
    for baseline in ("raw_flatten", "raw_meanpool"):
        summary["overall"][f"vision_vs_{baseline}"] = paired(records, baseline)
    strata = sorted({r["stratum"] for r in records})
    for stratum in strata:
        subset_records = [r for r in records if r["stratum"] == stratum]
        summary["per_stratum"][stratum] = {
            f"vision_vs_{b}": paired(subset_records, b)
            for b in ("raw_flatten", "raw_meanpool")
        }
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--max-windows", type=int, default=1000)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--limit", type=int, default=None)
    _ = parser.add_argument("--select-only", action="store_true")
    _ = parser.add_argument("--resume", action="store_true",
                            help="reuse series already scored in per_series.json")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selection = build_selection(int(args.seed))
    (OUT_ROOT / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    if args.select_only:
        counts: dict[str, int] = {}
        for item in selection:
            counts[item["stratum"]] = counts.get(item["stratum"], 0) + 1
        for stratum, count in sorted(counts.items()):
            LOGGER.info("  %-16s %d", stratum, count)
        return

    if args.limit is not None:
        selection = selection[: int(args.limit)]

    # Resume: reuse any series already scored in a previous (possibly
    # interrupted) run so long campaigns survive restarts.
    records: list[dict[str, Any]] = []
    per_series_path = OUT_ROOT / "per_series.json"
    if args.resume and per_series_path.exists():
        try:
            records = json.loads(per_series_path.read_text())
            LOGGER.info("resuming with %d previously scored series", len(records))
        except Exception as exc:
            LOGGER.warning("could not read existing results (%s); starting fresh", exc)
            records = []
    done = {(r["subset"], r["series"]) for r in records}
    selection = [s for s in selection if (s["subset"], s["series"]) not in done]
    LOGGER.info("%d series remaining to score", len(selection))

    backbone = VisionBackbone("facebook/dinov2-base")
    for index, item in enumerate(selection, start=1):
        try:
            record = run_series(item, backbone, int(args.max_windows))
        except Exception as exc:
            LOGGER.warning("%s failed: %s", item["series"], exc)
            record = None
        if record is not None:
            records.append(record)
        if index % 5 == 0 or index == len(selection):
            LOGGER.info("%d/%d done (%d scored)", index, len(selection), len(records))
            (OUT_ROOT / "per_series.json").write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8"
            )

    (OUT_ROOT / "per_series.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    summary = summarize(records)
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    LOGGER.info("=== paired vision vs raw (VUS-PR) ===")
    for baseline, stats in summary["overall"].items():
        if stats.get("n"):
            LOGGER.info("  %-24s n=%3d vision=%.4f base=%.4f delta=%+.4f wins=%d p=%s",
                        baseline, stats["n"], stats["vision_mean"],
                        stats["baseline_mean"], stats["delta_mean"],
                        stats["vision_wins"],
                        f"{stats.get('wilcoxon_p', float('nan')):.2e}")
    for stratum, stats in summary["per_stratum"].items():
        flat = stats.get("vision_vs_raw_flatten", {})
        if flat.get("n"):
            LOGGER.info("  %-16s n=%3d delta(vs flatten)=%+.4f wins=%d",
                        stratum, flat["n"], flat["delta_mean"], flat["vision_wins"])
    LOGGER.info("wrote %s", OUT_ROOT / "summary.json")


if __name__ == "__main__":
    main()
