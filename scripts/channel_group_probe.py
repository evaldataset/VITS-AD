"""Roadmap 2.3 — does channel-group rendering recover the vision arm at high D?

The controlled study showed the vision pipeline loses badly at D=25 because a single
line plot squashes all 25 channels into 3 RGB channels (compression bottleneck). This
probe tests the fix: render the D channels in small groups (group_size k), each as its
own image, extract frozen-DINOv2 tokens per group, and concatenate. Hypothesis: this
preserves per-channel information and closes the gap to raw-flatten on the worst
(D=25, localized) cells.

Compares three arms on the D=25 cells where single-plot vision was weakest:
  raw_flatten      : Ledoit-Wolf Mahalanobis on flattened windows (reference)
  vision_single    : line_plot(all 25 ch) -> DINOv2 -> mean-pool   (study baseline)
  vision_group_k{k}: channel groups of size k -> DINOv2 per group -> concat

Output: paper/rebuttal_package/channel_group_probe.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import scripts.controlled_regime_full as F
from scripts.controlled_regime_synthetic import _auc, _vision_features
from src.models.backbone import VisionBackbone
from src.rendering.channel_group import render_channel_groups
from src.rendering.line_plot import render_line_plot

LOGGER = logging.getLogger(__name__)

D: int = 25
RHO: float = 0.9
GROUP_SIZES: tuple[int, ...] = (3,)
CELLS = [("amplitude", "local"), ("structural", "local")]
SEEDS = (42, 123, 456)
OUT_PATH = Path("paper/rebuttal_package/channel_group_probe.json")


def _vision_group_features(bb: VisionBackbone, windows: np.ndarray, k: int) -> np.ndarray:
    """Per-window: render channel groups -> DINOv2 per group image -> mean-pool
    patches per group -> concatenate groups into one feature vector."""
    feats = []
    for i in range(windows.shape[0]):
        imgs = render_channel_groups(
            windows[i].astype(np.float32), group_size=k, render_fn=render_line_plot
        )  # (G, 3, 224, 224)
        toks = bb.extract_patch_tokens_from_numpy(imgs.astype(np.float32))  # (G, P, H)
        feats.append(toks.mean(axis=1).reshape(-1))  # (G*H,)
    return np.stack(feats).astype(np.float64)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    bb = VisionBackbone("facebook/dinov2-base")
    records: list[dict[str, Any]] = []
    for anom_type, locality in CELLS:
        per: dict[str, list[float]] = {"raw_flatten": [], "vision_single": []}
        for k in GROUP_SIZES:
            per[f"vision_group_k{k}"] = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            tr, te, lab = F._build_split(rng, D, RHO, anom_type, locality, 0.0)
            per["raw_flatten"].append(
                _auc(tr.reshape(tr.shape[0], -1), te.reshape(te.shape[0], -1), lab))
            per["vision_single"].append(
                _auc(_vision_features(bb, tr), _vision_features(bb, te), lab))
            for k in GROUP_SIZES:
                per[f"vision_group_k{k}"].append(
                    _auc(_vision_group_features(bb, tr, k),
                         _vision_group_features(bb, te, k), lab))
        rec: dict[str, Any] = {"D": D, "anomaly_type": anom_type, "locality": locality,
                               "n_seeds": len(SEEDS)}
        for arm, vals in per.items():
            a = np.asarray(vals, dtype=np.float64)
            rec[f"{arm}_mean"] = float(a.mean())
            rec[f"{arm}_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        best_group = max(GROUP_SIZES, key=lambda k: rec[f"vision_group_k{k}_mean"])
        rec["group_recovers_over_single"] = (
            rec[f"vision_group_k{best_group}_mean"] - rec["vision_single_mean"])
        records.append(rec)
        LOGGER.info(
            "%s %s | raw=%.3f single=%.3f %s | recover=%+.3f",
            anom_type, locality, rec["raw_flatten_mean"], rec["vision_single_mean"],
            " ".join(f"k{k}={rec[f'vision_group_k{k}_mean']:.3f}" for k in GROUP_SIZES),
            rec["group_recovers_over_single"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "meta": {"D": D, "group_sizes": list(GROUP_SIZES), "seeds": list(SEEDS),
                 "note": "tests channel-group rendering as a fix for the D>3 RGB "
                         "compression bottleneck identified in the controlled study."},
        "records": records}, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (%d cells)", OUT_PATH, len(records))


if __name__ == "__main__":
    run()
