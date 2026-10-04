"""Random-initialised backbone control (closes review gap W10).

Our rendered arm (DINOv2) and ViT4TS (CLIP) agree closely, which we read as evidence
about the rendering paradigm rather than about either encoder. A reviewer can object
that both are ImageNet-scale ViTs, so their agreement might instead reflect a shared
inductive bias. This control separates the two explanations.

We keep the pipeline identical --- same windows, same line-plot rendering, same
mean-pooled patch tokens, same Ledoit-Wolf Mahalanobis --- and change one thing: the
backbone weights are randomly initialised from the same architecture instead of
pretrained. Interpretation:

  * if the random backbone performs close to the pretrained one, the pretrained
    features contribute little and the paradigm's behaviour is dominated by rendering
    plus the architecture's inductive bias;
  * if it collapses, the pretrained representation is doing real work, and the
    DINOv2/CLIP agreement is a statement about pretrained vision encoders generally.

Output: results/random_backbone/{per_series,summary}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, Dinov2Model

from scripts.run_raw_mahalanobis_baseline import RawMahalanobisScorer
from scripts.run_tsb_ad_raw_audit import load_series, parse_split
from src.data.base import create_sliding_windows
from src.evaluation.modern_metrics import compute_modern_metrics
from src.models.backbone import VisionBackbone
from src.rendering.line_plot import render_line_plot

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "facebook/dinov2-base"
WINDOW_SIZE = 100
MIN_STRIDE = 10
MAX_WINDOWS = 1000
RENDER_BATCH = 64
DATA_ROOT = Path("data/tsb_ad")
PAIRED = Path("results/tsb_ad_vision_paired/per_series.json")
OUT_ROOT = Path("results/random_backbone")


def build_random_backbone(seed: int, device: str) -> VisionBackbone:
    """Return a VisionBackbone whose weights are randomly initialised.

    The wrapper is constructed normally (so preprocessing and token extraction are
    identical) and its encoder is then replaced by a same-architecture model built
    from config alone, which carries no pretrained information.

    Args:
        seed: Seed controlling the random initialisation.
        device: Torch device string.

    Returns:
        A frozen VisionBackbone with randomly initialised weights.
    """
    backbone = VisionBackbone(MODEL_NAME, device=torch.device(device))
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    random_model = Dinov2Model(config)          # random init, no pretrained weights
    random_model.eval()
    random_model.requires_grad_(False)
    random_model.to(device=backbone.device, dtype=torch.float32)
    backbone.model = random_model
    return backbone


def _features(backbone: VisionBackbone, windows: np.ndarray) -> np.ndarray:
    """Render windows and mean-pool their patch tokens."""
    feats: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    for i in range(windows.shape[0]):
        buf.append(render_line_plot(windows[i].astype(np.float32)))
        if len(buf) == RENDER_BATCH or i == windows.shape[0] - 1:
            toks = backbone.extract_patch_tokens_from_numpy(
                np.stack(buf).astype(np.float32))
            feats.append(toks.mean(axis=1))
            buf = []
    return np.concatenate(feats, axis=0).astype(np.float64)


def run_series(entry: dict[str, Any], backbone: VisionBackbone) -> dict[str, Any] | None:
    """Score one series with the random-init backbone on our windows."""
    path = DATA_ROOT / entry["subset"] / entry["series"]
    train_end, _ = parse_split(path)
    values, labels = load_series(path)
    n_test = values.shape[0] - train_end
    stride = max(MIN_STRIDE, math.ceil(n_test / MAX_WINDOWS))

    train_raw, test_raw = values[:train_end], values[train_end:]
    mu = train_raw.mean(axis=0)
    sigma = train_raw.std(axis=0) + 1e-8
    train_w, _ = create_sliding_windows((train_raw - mu) / sigma,
                                        labels[:train_end], WINDOW_SIZE, stride)
    test_w, win_labels = create_sliding_windows((test_raw - mu) / sigma,
                                                labels[train_end:], WINDOW_SIZE, stride)
    if train_w.shape[0] < 2 or test_w.shape[0] < 2:
        return None
    if win_labels.sum() in (0, win_labels.shape[0]):
        return None

    scorer = RawMahalanobisScorer()
    scorer.fit(_features(backbone, train_w))
    scores = scorer.score(_features(backbone, test_w))
    if not np.all(np.isfinite(scores)):
        return None
    metrics = compute_modern_metrics(scores, win_labels, sliding_window=WINDOW_SIZE,
                                     verify_against_own=False)
    return {
        "subset": entry["subset"], "series": entry["series"],
        "stratum": entry["stratum"], "n_channels": entry["n_channels"],
        "random_backbone": metrics,
        "pretrained_vision": entry["vision"],
        "raw_flatten": entry.get("raw_flatten"),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired comparisons of the random backbone against the other arms."""
    from scipy import stats

    def paired(a_key: str, b_key: str, metric: str) -> dict[str, Any]:
        pairs = [(r[a_key][metric], r[b_key][metric])
                 for r in records if r.get(a_key) and r.get(b_key)]
        if len(pairs) < 2:
            return {"n": len(pairs)}
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        d = a - b
        out: dict[str, Any] = {
            "n": len(pairs), f"{a_key}_mean": float(a.mean()),
            f"{b_key}_mean": float(b.mean()), "delta": float(d.mean()),
            f"{a_key}_wins": int((d > 0).sum()),
        }
        if np.any(d != 0):
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        return out

    summary: dict[str, Any] = {"n_series": len(records)}
    for metric in ("VUS-PR", "AUC-ROC"):
        summary[metric] = {
            "random_vs_pretrained": paired("random_backbone", "pretrained_vision", metric),
            "random_vs_raw_flatten": paired("random_backbone", "raw_flatten", metric),
        }
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--limit", type=int, default=None)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    paired = [e for e in json.loads(PAIRED.read_text())
              if e.get("vision") and e.get("raw_flatten")]
    if args.limit is not None:
        paired = paired[: int(args.limit)]

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
    LOGGER.info("%d series to score with a random-init backbone", len(paired))

    backbone = build_random_backbone(int(args.seed), str(args.device))
    for index, entry in enumerate(paired, start=1):
        try:
            rec = run_series(entry, backbone)
        except Exception as exc:
            LOGGER.warning("%s failed: %s", entry["series"], str(exc)[:100])
            rec = None
        if rec is not None:
            records.append(rec)
        if index % 10 == 0 or index == len(paired):
            out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            LOGGER.info("%d/%d (%d scored)", index, len(paired), len(records))

    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    summary = summarize(records)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                           encoding="utf-8")
    LOGGER.info("=== random-init backbone control (n=%d) ===", summary["n_series"])
    for metric in ("VUS-PR", "AUC-ROC"):
        for name, st in summary[metric].items():
            if st.get("n", 0) > 1:
                LOGGER.info("  [%s] %-24s n=%3d random=%.4f other=%.4f d=%+.4f p=%s",
                            metric, name, st["n"], st["random_backbone_mean"],
                            st[[k for k in st if k.endswith("_mean")
                                and k != "random_backbone_mean"][0]],
                            st["delta"],
                            f"{st.get('wilcoxon_p', float('nan')):.3g}")
    LOGGER.info("wrote %s", OUT_ROOT / "summary.json")


if __name__ == "__main__":
    main()
