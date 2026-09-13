"""Generate the VLDB paper's figures from the released artifacts.

Three figures carry the paper's central claims:
  fig_baseline_inversion : the rendered pipeline beats mean-pooled raw scoring and
                           loses to flattened raw scoring on the same series --
                           the comparison inverts with the control.
  fig_amplitude_law      : raw-space Mahalanobis is an amplitude-extremity detector
                           (accuracy tracks z_ratio) while rendering is insensitive.
  fig_cost_scaling       : how rendering, the raw control and the frozen backbone
                           scale with channel count and window count.

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


def fig_cost_scaling(out_dir: Path) -> None:
    """Three-panel cost scaling: rendering, the raw control, and the backbone.

    Args:
        out_dir: Directory to write the figure into.
    """
    src = ART / "cost_scaling" / "cost_scaling.json"
    if not src.exists():
        LOGGER.warning("cost scaling artifact missing; skipping figure")
        return
    data = json.loads(src.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))

    ax = axes[0]
    rend = data["render_vs_channels"]
    xs = [r["D"] for r in rend]
    ys = [r["ms_per_window"] for r in rend]
    ax.plot(xs, ys, "o-", color="#1f77b4", lw=1.8, ms=5)
    ax.set_xlabel("channels $D$", fontsize=9)
    ax.set_ylabel("ms per window", fontsize=9)
    ax.set_title("(a) Line-plot rendering", fontsize=10)

    ax = axes[1]
    raw = data["raw_mahalanobis_vs_channels"]
    for variant, colour, marker in (("flattened", "#d62728", "o"),
                                    ("mean_pooled", "#2ca02c", "s")):
        rows = [r for r in raw if r["variant"] == variant]
        ax.plot([r["feature_dim"] for r in rows],
                [r.get("fit_seconds", r["seconds"]) for r in rows],
                marker + "-", color=colour, lw=1.8, ms=5,
                label=f"{variant.replace('_', '-')} (fit)")
        if "score_seconds" in rows[0]:
            ax.plot([r["feature_dim"] for r in rows],
                    [r["score_seconds"] for r in rows],
                    marker + ":", color=colour, lw=1.2, ms=3, alpha=0.7,
                    label=f"{variant.replace('_', '-')} (score)")
    ax.axvline(2000, color="0.4", lw=1, ls="--")
    ax.annotate("flatten\ndim cap", xy=(2000, 0.02), xycoords=("data", "axes fraction"),
                xytext=(-4, 0), textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"feature dimension $W\cdot D$", fontsize=9)
    ax.set_ylabel("seconds", fontsize=9)
    ax.set_title("(b) Ledoit-Wolf raw control", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    ax = axes[2]
    bb = data["backbone_vs_windows"]
    if bb:
        ax.plot([r["n_windows"] for r in bb], [r["seconds"] for r in bb],
                "o-", color="#9467bd", lw=1.8, ms=5)
        ax.set_xlabel("windows encoded", fontsize=9)
        ax.set_ylabel("seconds", fontsize=9)
    ax.set_title("(c) Frozen backbone", fontsize=10)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_cost_scaling.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    LOGGER.info("cost scaling: rendering %.1f-%.1f ms/window over D=%d-%d",
                ys[0], ys[-1], xs[0], xs[-1])


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
    fig_cost_scaling(out_dir)
    LOGGER.info("wrote figures to %s", out_dir)


if __name__ == "__main__":
    main()
