"""Head-to-head: ViT4TS (VLM4TS stage 1) vs. our vision arm and raw Mahalanobis.

ViT4TS is the closest published neighbour to this work: it renders sliding windows
as images and scores them with a frozen CLIP vision encoder. The paper previously
could only audit it on multivariate data through a PCA-to-1D wrapper. TSB-AD's
univariate split lets us run it **natively, with no adapter**, on exactly the
series where our own paired comparison already has both arms scored.

Protocol note (disclosed, and it favours ViT4TS): ViT4TS builds its "normal"
memory bank from the *test* dataloader itself, i.e. it is transductive over the
evaluation series, whereas our arms fit on the training split only. We keep
ViT4TS's own design rather than crippling it, and report the asymmetry.

Comparability: ViT4TS returns point-level scores. We aggregate them into the same
windows (same window size and stride) used by our arms, take the max within each
window, and evaluate against the identical any-positive window labels and metric
stack. Every comparison is therefore paired per series.

Output: results/vit4ts_paired/{per_series,summary}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VLM4TS_SRC = Path("baselines/VLM4TS/src").resolve()
if str(VLM4TS_SRC) not in sys.path:
    sys.path.insert(0, str(VLM4TS_SRC))

from scripts.run_tsb_ad_raw_audit import load_series, parse_split  # noqa: E402
from src.data.base import create_sliding_windows  # noqa: E402
from src.evaluation.modern_metrics import compute_modern_metrics  # noqa: E402

LOGGER = logging.getLogger(__name__)

WINDOW_SIZE = 100
MIN_STRIDE = 10
MAX_WINDOWS = 1000
#: ViT4TS builds a memory bank over all its windows and compares them pairwise, so
#: its cost grows quadratically with series length. Series whose test split exceeds
#: this many points are skipped and recorded; the criterion is length-based and fixed
#: in advance, so it cannot bias the comparison toward either method.
MAX_TEST_POINTS_DEFAULT = 30000
DATA_ROOT = Path("data/tsb_ad")
PAIRED = Path("results/tsb_ad_vision_paired/per_series.json")
OUT_ROOT = Path("results/vit4ts_paired")


def _patch_open_clip_quickgelu() -> None:
    """Instantiate OpenAI CLIP weights with QuickGELU, as ViT4TS intended.

    ViT4TS calls ``open_clip.create_model_and_transforms('ViT-B-16',
    pretrained='openai')``. Older open_clip selected QuickGELU automatically for
    the OpenAI tag; open_clip 3.x does not, and warns of the mismatch. Left
    uncorrected the baseline would run with the wrong activation and be unfairly
    degraded (the two variants produce measurably different embeddings). We inject
    the flag here rather than editing the vendored repository.
    """
    import open_clip

    original = open_clip.create_model_and_transforms

    def patched(model_name, *args, **kwargs):
        if (kwargs.get("pretrained") == "openai"
                and "quickgelu" not in str(model_name).lower()
                and "quick_gelu" not in kwargs):
            kwargs["quick_gelu"] = True
        return original(model_name, *args, **kwargs)

    open_clip.create_model_and_transforms = patched
    import models.clip_vision as cv
    if hasattr(cv, "open_clip"):
        cv.open_clip.create_model_and_transforms = patched


def _load_vit4ts(device: str) -> Any:
    """Import and construct ViT4TS from the vendored VLM4TS repository."""
    _patch_open_clip_quickgelu()
    from models.vit4ts import ViT4TS
    return ViT4TS(device=device, verbose=False)


def _window_scores(point_scores: np.ndarray, n_windows: int, stride: int) -> np.ndarray:
    """Aggregate point-level scores into our windows (max within window)."""
    out = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        start = i * stride
        seg = point_scores[start : start + WINDOW_SIZE]
        out[i] = float(np.nanmax(seg)) if seg.size else 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def run_series(entry: dict[str, Any], detector: Any,
               max_test_points: int) -> dict[str, Any] | None:
    """Score one univariate series with ViT4TS on our windows."""
    path = DATA_ROOT / entry["subset"] / entry["series"]
    train_end, _ = parse_split(path)
    values, labels = load_series(path)
    if values.shape[1] != 1:
        return None

    n_test_points = values.shape[0] - train_end
    if n_test_points > max_test_points:
        LOGGER.info("%s: skipped (test span %d > %d)", entry["series"],
                    n_test_points, max_test_points)
        return None
    stride = max(MIN_STRIDE, math.ceil(n_test_points / MAX_WINDOWS))
    test_values = values[train_end:, 0]
    test_labels = labels[train_end:]

    _, win_labels = create_sliding_windows(
        test_values.reshape(-1, 1), test_labels, WINDOW_SIZE, stride
    )
    if win_labels.size < 2 or win_labels.sum() in (0, win_labels.size):
        return None

    frame = pd.DataFrame({
        "timestamp": np.arange(test_values.size, dtype=np.int64),
        "value": test_values.astype(float),
    })
    scores, _ = detector.predict_scores(frame)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size < test_values.size:
        scores = np.pad(scores, (0, test_values.size - scores.size), mode="edge")
    scores = scores[: test_values.size]

    win_scores = _window_scores(scores, win_labels.size, stride)
    if float(np.nanstd(win_scores)) == 0.0:
        LOGGER.warning("%s: degenerate ViT4TS scores", entry["series"])
        return None

    metrics = compute_modern_metrics(
        win_scores, win_labels, sliding_window=WINDOW_SIZE, verify_against_own=False
    )
    return {
        "subset": entry["subset"], "series": entry["series"],
        "stratum": entry["stratum"], "stride": int(stride),
        "n_test_windows": int(win_labels.size),
        "vit4ts": metrics,
        "ours_vision": entry["vision"],
        "raw_flatten": entry.get("raw_flatten"),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired comparisons of ViT4TS against our arms."""
    from scipy import stats

    def paired(key_a: str, key_b: str, metric: str) -> dict[str, Any]:
        pairs = [
            (r[key_a][metric], r[key_b][metric])
            for r in records if r.get(key_a) and r.get(key_b)
        ]
        if len(pairs) < 2:
            return {"n": len(pairs)}
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        out: dict[str, Any] = {
            "n": len(pairs), f"{key_a}_mean": float(a.mean()),
            f"{key_b}_mean": float(b.mean()), "delta": float((a - b).mean()),
            f"{key_a}_wins": int(((a - b) > 0).sum()),
        }
        if np.any(a != b):
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        return out

    summary: dict[str, Any] = {"n_series": len(records)}
    for metric in ("AUC-ROC", "VUS-PR"):
        summary[metric] = {
            "vit4ts_vs_ours_vision": paired("vit4ts", "ours_vision", metric),
            "vit4ts_vs_raw_flatten": paired("vit4ts", "raw_flatten", metric),
        }
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--limit", type=int, default=None)
    _ = parser.add_argument("--device", type=str, default="cuda")
    _ = parser.add_argument("--max-test-points", type=int,
                            default=MAX_TEST_POINTS_DEFAULT)
    args = parser.parse_args()

    paired = [
        e for e in json.loads(PAIRED.read_text())
        if e["subset"] == "TSB-AD-U" and e.get("vision") and e.get("raw_flatten")
    ]
    if args.limit is not None:
        paired = paired[: int(args.limit)]
    LOGGER.info("univariate series to score with ViT4TS: %d", len(paired))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / "per_series.json"
    records: list[dict[str, Any]] = []
    if out_path.exists():
        try:
            records = json.loads(out_path.read_text())
            LOGGER.info("resuming with %d records", len(records))
        except Exception:
            records = []
    done = {r["series"] for r in records}
    paired = [e for e in paired if e["series"] not in done]

    detector = _load_vit4ts(str(args.device))
    for index, entry in enumerate(paired, start=1):
        try:
            record = run_series(entry, detector, int(args.max_test_points))
        except Exception as exc:
            LOGGER.warning("%s failed: %s", entry["series"], exc)
            record = None
        if record is not None:
            records.append(record)
        if index % 5 == 0 or index == len(paired):
            out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            LOGGER.info("%d/%d (%d scored)", index, len(paired), len(records))

    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    summary = summarize(records)
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("=== ViT4TS head-to-head (n=%d) ===", summary["n_series"])
    for metric in ("AUC-ROC", "VUS-PR"):
        for name, stat in summary[metric].items():
            if stat.get("n", 0) > 1:
                LOGGER.info("  [%s] %-26s n=%3d vit4ts=%.4f other=%.4f d=%+.4f p=%s",
                            metric, name, stat["n"], stat["vit4ts_mean"],
                            stat[[k for k in stat if k.endswith("_mean")
                                  and k != "vit4ts_mean"][0]],
                            stat["delta"],
                            f"{stat.get('wilcoxon_p', float('nan')):.3g}")
    LOGGER.info("wrote %s", OUT_ROOT / "summary.json")


if __name__ == "__main__":
    main()
