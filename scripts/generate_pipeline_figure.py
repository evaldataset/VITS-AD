"""Generate pipeline figure (Figure 1) for the VITS-AD paper."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def draw_pipeline(ax: plt.Axes) -> None:
    """Draw the VITS-AD pipeline diagram on the given axes."""
    ax.set_xlim(-1.0, 10.5)
    ax.set_ylim(0.05, 1.75)
    ax.axis("off")

    # Pipeline stages: (x, label, color)
    stages = [
        (0.0, "Time Series\n$\\mathbf{W}_t \\in \\mathbb{R}^{W \\times D}$", "#4A90D9"),
        (1.7, "Renderer\nLP / RP", "#E8833A"),
        (3.4, "Frozen ViT\nDINOv2-B/14", "#5CB85C"),
        (5.1, "Patch Tokens\n$256 \\times 768$", "#9B59B6"),
        (7.2, "Dual-Signal\nScorer", "#E74C3C"),
        (9.2, "Anomaly\nScore", "#F1C40F"),
    ]

    box_w, box_h = 1.3, 0.55
    y_center = 0.9

    for x, label, color in stages:
        fancy = mpatches.FancyBboxPatch(
            (x - box_w / 2, y_center - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="white", linewidth=1.5,
            alpha=0.9, zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(
            x, y_center, label,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white", zorder=4,
        )

    # Arrows between stages
    arrow_style = dict(
        arrowstyle="->,head_width=0.15,head_length=0.1",
        color="#555555", lw=1.5, zorder=2,
    )
    arrow_pairs = [
        (0.0 + box_w / 2, 1.7 - box_w / 2),
        (1.7 + box_w / 2, 3.4 - box_w / 2),
        (3.4 + box_w / 2, 5.1 - box_w / 2),
        (5.1 + box_w / 2, 7.2 - box_w / 2),
        (7.2 + box_w / 2, 9.2 - box_w / 2),
    ]
    for x_start, x_end in arrow_pairs:
        ax.annotate(
            "", xy=(x_end + 0.05, y_center), xytext=(x_start - 0.05, y_center),
            arrowprops=arrow_style,
        )

    # Branch labels for dual-signal
    # Mahalanobis (static, dominant)
    ax.annotate(
        "Mahalanobis dist.\n(static, dominant)",
        xy=(7.2, y_center + box_h / 2),
        xytext=(6.0, y_center + 0.65),
        fontsize=5.5, ha="center", va="bottom", color="#C0392B",
        arrowprops=dict(arrowstyle="-", color="#C0392B", lw=0.8),
        zorder=4,
    )
    # Patch trajectory residuals (dynamic)
    ax.annotate(
        "Patch traj. resid.\n(dynamic)",
        xy=(7.2, y_center - box_h / 2),
        xytext=(6.0, y_center - 0.65),
        fontsize=5.5, ha="center", va="top", color="#8E44AD",
        arrowprops=dict(arrowstyle="-", color="#8E44AD", lw=0.8),
        zorder=4,
    )

    # Frozen snowflake symbol
    ax.text(3.4 + 0.5, y_center + 0.25, "❄", fontsize=8, ha="center", va="center", zorder=5)

    # alpha label
    ax.text(
        7.2, y_center - box_h / 2 - 0.12,
        "$\\alpha = 0.5$",
        ha="center", va="top", fontsize=6, color="#666666", zorder=4,
    )


def draw_comparison_bar(ax: plt.Axes) -> None:
    """Draw the AUC-ROC comparison bar chart (b)."""
    datasets = ["SMD", "PSM", "MSL", "SMAP", "UCR\n(109)"]
    raw_maha = [0.981, 0.769, 0.607, 0.402, 0.676]
    vits = [0.754, 0.603, 0.534, 0.705, 0.869]
    deltas = [v - r for v, r in zip(vits, raw_maha)]

    x = np.arange(len(datasets))
    width = 0.32

    ax.bar(x - width / 2, raw_maha, width, label="Raw Mahalanobis",
           color="#B0B0B0", edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + width / 2, vits, width, label="VITS (ours)",
           color="#E74C3C", edgecolor="white", linewidth=0.5, zorder=3)

    # Delta annotations
    for i, (d, v, r) in enumerate(zip(deltas, vits, raw_maha)):
        top = max(v, r) + 0.03
        color = "#27AE60" if d > 0 else "#7F8C8D"
        sign = "+" if d > 0 else ""
        ax.text(x[i], top, f"{sign}{d:.2f}", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color=color, zorder=4)

    # Region labels
    ax.axvspan(2.7, 4.6, alpha=0.08, color="#27AE60", zorder=1)
    ax.text(3.65, 0.95, "Vision wins", fontsize=6, ha="center", color="#27AE60",
            fontstyle="italic", zorder=4)
    ax.axvspan(-0.5, 2.7, alpha=0.06, color="#7F8C8D", zorder=1)
    ax.text(1.1, 0.95, "Raw wins", fontsize=6, ha="center", color="#7F8C8D",
            fontstyle="italic", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=7.5)
    ax.set_ylabel("AUC-ROC", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


def main() -> None:
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.0, 6.2),
        gridspec_kw={"height_ratios": [1.7, 1.2], "hspace": 0.02},
    )

    ax_top.set_title("(a)", fontsize=9, fontweight="bold", loc="left", pad=4)
    draw_pipeline(ax_top)

    ax_bot.set_title("(b)", fontsize=9, fontweight="bold", loc="left", pad=4)
    draw_comparison_bar(ax_bot)

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pipeline.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300, pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
