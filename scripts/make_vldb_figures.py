"""Generate the VLDB paper's figures from the released artifacts.

Two figures carry the paper's central claims:
  fig_baseline_inversion : the rendered pipeline beats mean-pooled raw scoring and
                           loses to flattened raw scoring on the same series --
                           the comparison inverts with the control.
  fig_amplitude_law      : raw-space Mahalanobis is an amplitude-extremity detector
                           (accuracy tracks z_ratio) while rendering is insensitive.

Both read only from artifacts/, so the figures regenerate from the public package
without the raw datasets or a GPU:

    python scripts/make_vldb_figures.py --out figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts")
DEFAULT_OUT = Path("figures")


def fig_baseline_inversion(out_dir: Path) -> None:
    """Paired per-series deltas of rendered scoring against each raw variant.

    Args:
        out_dir: Directory to write the figure into.
    """
    recs = json.loads((ART / "tsb_ad_vision_paired" / "per_series.json").read_text())
    d_mp = np.array([r["vision"]["VUS-PR"] - r["raw_meanpool"]["VUS-PR"]
                     for r in recs if r.get("vision") and r.get("raw_meanpool")])
    d_fl = np.array([r["vision"]["VUS-PR"] - r["raw_flatten"]["VUS-PR"]
                     for r in recs if r.get("vision") and r.get("raw_flatten")])

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    parts = ax.violinplot([d_mp, d_fl], positions=[0, 1], widths=0.75,
                          showmeans=False, showextrema=False)
    for body, colour in zip(parts["bodies"], ("#4C72B0", "#C44E52")):
        body.set_facecolor(colour)
        body.set_alpha(0.45)
    for i, d in enumerate((d_mp, d_fl)):
        ax.scatter(np.random.default_rng(0).normal(i, 0.055, d.size), d,
                   s=7, color="0.25", alpha=0.55, zorder=3)
        ax.hlines(d.mean(), i - 0.34, i + 0.34, color="black", lw=2, zorder=4)
        ax.text(i, d.mean() + 0.055, f"mean {d.mean():+.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"vs mean-pooled raw\n(n={d_mp.size}, wins {int((d_mp>0).sum())})",
                        f"vs flattened raw\n(n={d_fl.size}, wins {int((d_fl>0).sum())})"],
                       fontsize=9)
    ax.set_ylabel("VUS-PR: rendered $-$ raw", fontsize=9)
    ax.set_title("The comparison inverts with the choice of raw control",
                 fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_baseline_inversion.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    LOGGER.info("baseline inversion: mean-pool %+.4f, flatten %+.4f",
                d_mp.mean(), d_fl.mean())


def fig_amplitude_law(out_dir: Path) -> None:
    """Accuracy versus amplitude subtlety for the raw and rendered arms.

    Args:
        out_dir: Directory to write the figure into.
    """
    recs = json.loads((ART / "regime_proxy" / "proxy_search.json").read_text())["records"]
    uni = [r for r in recs if r["subset"] == "TSB-AD-U"]
    z = np.array([r["proxies"]["z_ratio_labelled"] for r in uni])
    raw = np.array([r["raw_auc"] for r in uni])
    vis = np.array([r["vision_auc"] for r in uni])
    keep = np.isfinite(z) & (z > 0)
    z, raw, vis = z[keep], raw[keep], vis[keep]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.scatter(z, raw, s=22, color="#C44E52", alpha=0.75, label=None)
    ax.scatter(z, vis, s=22, color="#4C72B0", alpha=0.75, marker="^", label=None)
    for values, colour, name in ((raw, "#C44E52", "raw (flatten)"),
                                 (vis, "#4C72B0", "rendered")):
        rho = stats.spearmanr(z, values).statistic
        coef = np.polyfit(np.log10(z), values, 1)
        xs = np.linspace(np.log10(z.min()), np.log10(z.max()), 50)
        ax.plot(10 ** xs, np.polyval(coef, xs), color=colour, lw=2,
                label=rf"{name}  $\rho={rho:+.2f}$")
    ax.axvline(1.5, color="0.4", ls="--", lw=1)
    ax.text(1.53, 0.06, "amplitude-subtle $\\leftarrow$ | $\\rightarrow$ extreme",
            fontsize=7.5, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel(r"$z_{\mathrm{ratio}}$  (anomaly amplitude / normal amplitude)",
                  fontsize=9)
    ax.set_ylabel("AUC-ROC", fontsize=9)
    ax.set_title("Raw scoring tracks amplitude extremity; rendering does not",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_amplitude_law.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    LOGGER.info("amplitude law: n=%d", z.size)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                            help="directory to write the figures into")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_baseline_inversion(out_dir)
    fig_amplitude_law(out_dir)
    LOGGER.info("wrote figures to %s", out_dir)


if __name__ == "__main__":
    main()
