"""Any TSB-AD pool detector on the paired subset, with TSB-AD's own tuned hyperparameters.

TSB-AD already ships statistical subsequence detectors (MCD, Sub_MCD, Sub_PCA, MatrixProfile,
KShapeAD, ...) and publishes tuned hyperparameters for every detector, separately for univariate
and multivariate series (``TSB_AD.HP_list.Optimal_{Uni,Multi}_algo_HP_dict``). This runner scores
any of them on exactly the series of the paired subset, so they line up with the raw controls,
the rendered arm and the other detectors.

Protocol, matching ``run_tsb_ad_trained_baselines.py`` except where TSB-AD's own protocol differs:
  * semi-supervised detectors are fitted on the anomaly-free training split and score the test
    split;
  * unsupervised detectors are run on the whole series, as TSB-AD does, and the test part of
    their scores is kept -- they therefore see the test data, which we disclose;
  * hyperparameters come from TSB-AD's tuned table for the series' split; a detector whose tuned
    entry is None for that split (TSB-AD does not recommend it there) is recorded as
    ``not_applicable`` rather than run with guessed settings;
  * point scores are aggregated to our windows by max and evaluated with the same metric stack.

Output: results/tsb_ad_pool/<detector>/per_series_shard<k>.json (+ summary.json via --aggregate)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from scripts.run_tsb_ad_raw_audit import load_series, parse_split
from scripts.run_tsb_ad_trained_baselines import (
    DATA_ROOT, MAX_WINDOWS, MIN_STRIDE, PAIRED, WINDOW_SIZE, _window_scores,
)
from src.data.base import create_sliding_windows
from src.evaluation.modern_metrics import compute_modern_metrics

LOGGER = logging.getLogger(__name__)
OUT_ROOT = Path("results/tsb_ad_pool")


def _patch_sklearn_compat() -> None:
    """Let TSB-AD 1.5's MCD run under scikit-learn >= 1.6.

    ``TSB_AD.models.MCD`` calls ``sklearn.utils.validation.check_is_fitted``, which since
    scikit-learn 1.6 requires ``__sklearn_tags__`` -- a method TSB-AD's own BaseDetector does
    not implement. Fitting itself works; only this guard fails. We replace the guard, in that
    module only, with the plain attribute check it is meant to perform. TSB-AD's code is
    otherwise untouched.
    """
    import TSB_AD.models.MCD as mcd_module

    def _check_is_fitted(estimator: Any, attributes: list[str] | None = None, **_: Any) -> None:
        missing = [a for a in (attributes or []) if not hasattr(estimator, a)]
        if missing:
            raise ValueError(f"{type(estimator).__name__} is not fitted: missing {missing}")

    mcd_module.check_is_fitted = _check_is_fitted


def _tuned_hp(name: str, univariate: bool) -> dict[str, Any] | None:
    """TSB-AD's tuned hyperparameters for ``name`` on this split; None = not recommended."""
    from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict, Optimal_Uni_algo_HP_dict

    table = Optimal_Uni_algo_HP_dict if univariate else Optimal_Multi_algo_HP_dict
    if name not in table:
        return None  # TSB-AD does not list this detector for this split
    return table[name] or {}


def run_series(entry: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Score one series with one TSB-AD detector on our windows."""
    import inspect

    import TSB_AD.model_wrapper as tsbad

    _patch_sklearn_compat()

    # Call the model function directly: TSB-AD's run_*_AD wrappers catch every exception and
    # report it as "function not defined", which hides the real error.
    fn = getattr(tsbad, f"run_{name}")

    path = DATA_ROOT / entry["subset"] / entry["series"]
    train_end, _ = parse_split(path)
    values, labels = load_series(path)
    test_values = values[train_end:]
    stride = max(MIN_STRIDE, math.ceil(test_values.shape[0] / MAX_WINDOWS))
    _, win_labels = create_sliding_windows(test_values, labels[train_end:], WINDOW_SIZE, stride)
    if win_labels.size < 2 or win_labels.sum() in (0, win_labels.size):
        return None
    record: dict[str, Any] = {
        "subset": entry["subset"], "series": entry["series"], "stratum": entry["stratum"],
        "n_channels": entry["n_channels"], "stride": int(stride),
        "n_test_windows": int(win_labels.size),
    }
    hp = _tuned_hp(name, univariate=values.shape[1] == 1)
    if hp is None:
        record["status"] = "not_applicable"
        record[name] = None
        return record
    record["hyperparameters"] = hp
    accepted = set(inspect.signature(fn).parameters)
    if not set(hp) <= accepted:
        # TSB-AD's tuned table can list arguments the installed version's function lacks.
        record["status"] = "not_applicable"
        record["reason"] = f"tuned arguments {sorted(set(hp) - accepted)} unsupported"
        record[name] = None
        return record
    try:
        if name in tsbad.Semisupervise_AD_Pool:
            scores = fn(values[:train_end].astype(float), test_values.astype(float), **hp)
            mode = "semi-supervised"
        elif name in tsbad.Unsupervise_AD_Pool:
            scores = fn(values.astype(float), **hp)
            mode = "unsupervised (whole series)"
        else:
            raise ValueError(f"{name} not in TSB-AD pools")
        if isinstance(scores, str):
            raise RuntimeError(scores)
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if mode.startswith("unsupervised"):
            scores = scores[train_end:] if scores.size >= values.shape[0] else scores[-test_values.shape[0]:]
        if scores.size < test_values.shape[0]:
            scores = np.pad(scores, (0, test_values.shape[0] - scores.size), mode="edge")
        win_scores = _window_scores(scores[: test_values.shape[0]], win_labels.size, stride)
        if float(np.nanstd(win_scores)) == 0.0:
            raise ValueError("degenerate scores")
        record["mode"] = mode
        record["status"] = "ok"
        record[name] = compute_modern_metrics(win_scores, win_labels,
                                              sliding_window=WINDOW_SIZE, verify_against_own=False)
    except Exception as exc:
        LOGGER.warning("%s [%s] failed: %s: %s", entry["series"], name, type(exc).__name__, str(exc)[:200])
        record["status"] = "failed"
        record[name] = None
    return record


def _paired(pairs: list[tuple[float, float]], seed: int = 0) -> dict[str, Any] | None:
    if len(pairs) < 2:
        return None
    a, b = np.array(pairs).T
    d = a - b
    rng = np.random.default_rng(seed)
    boot = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(10000)]
    out: dict[str, Any] = {"n": int(d.size), "detector_mean": float(a.mean()),
                           "reference_mean": float(b.mean()), "delta": float(d.mean()),
                           "delta_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                           "detector_wins": int((d > 0).sum()), "ties": int((d == 0).sum())}
    if np.any(d != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    return out


def summarize(name: str, records: list[dict[str, Any]], lookup: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "detector": name, "n_records": len(records),
        "status_counts": {s: sum(r.get("status") == s for r in records)
                          for s in ("ok", "failed", "not_applicable")},
    }
    blocks = {"all": records, "univariate": [r for r in records if r["n_channels"] == 1],
              "multivariate": [r for r in records if r["n_channels"] > 1]}
    for blk, recs in blocks.items():
        out[blk] = {}
        for arm in ("raw_flatten", "vision"):
            out[blk][f"vs_{arm}"] = {
                m: _paired([(r[name][m], lookup[r["series"]][arm][m]) for r in recs
                            if r.get(name) and lookup.get(r["series"], {}).get(arm)])
                for m in ("VUS-PR", "AUC-ROC")}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--detector", required=True)
    _ = parser.add_argument("--shard", type=int, default=0)
    _ = parser.add_argument("--num-shards", type=int, default=1)
    _ = parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    paired = [e for e in json.loads(PAIRED.read_text()) if e.get("vision") and e.get("raw_flatten")]
    lookup = {e["series"]: e for e in paired}
    out_dir = OUT_ROOT / args.detector
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        records = [r for f in sorted(out_dir.glob("per_series_shard*.json"))
                   for r in json.loads(f.read_text())]
        summary = summarize(args.detector, records, lookup)
        (out_dir / "per_series.json").write_text(json.dumps(records, indent=2) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        LOGGER.info("%s", json.dumps({k: summary[k] for k in ("detector", "status_counts")}))
        return

    mine = [e for i, e in enumerate(paired) if i % args.num_shards == args.shard]
    out_path = out_dir / f"per_series_shard{args.shard}.json"
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["series"] for r in records}
    todo = [e for e in mine if e["series"] not in done]
    LOGGER.info("%s shard %d/%d: %d todo", args.detector, args.shard, args.num_shards, len(todo))
    for i, entry in enumerate(todo, start=1):
        rec = run_series(entry, args.detector)
        if rec is not None:
            records.append(rec)
        out_path.write_text(json.dumps(records, indent=2) + "\n")
        LOGGER.info("%d/%d", i, len(todo))


if __name__ == "__main__":
    main()
