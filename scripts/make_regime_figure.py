"""Generate the regime-gain figure for the main paper.

Produces a 2-panel figure:
  (a) Per-dataset AUC-ROC gain: VITS - Raw Mahalanobis (with annotations).
  (b) Per-method FLOPs vs AUC-ROC scatter across datasets.

Saves to paper/figures/regime_gain.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    # From paper Table 1 (already in paper).
    datasets = ["SMD", "PSM", "MSL", "SMAP", "UCR-109"]
    raw_maha = [0.981, 0.769, 0.607, 0.402, 0.676]
    vits = [0.809, 0.697, 0.637, 0.697, 0.880]
    # Regime labels
    is_structural = [False, False, True, True, True]

    gains = [v - r for v, r in zip(vits, raw_maha)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3))

    # Panel (a): gain per dataset, color by regime
    colors = ["#e74c3c" if not s else "#2ecc71" for s in is_structural]
    bars = ax1.bar(range(len(datasets)), gains, color=colors, edgecolor="black", linewidth=0.7)
    ax1.axhline(0.0, color="black", linewidth=0.6)
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, rotation=20, ha="right")
    ax1.set_ylabel(r"$\Delta$ AUC-ROC (VITS-AD $-$ Raw Maha)")
    ax1.set_title("(a) Per-dataset gain")
    ax1.set_ylim(-0.3, 0.4)
    ax1.grid(True, axis="y", alpha=0.3)
    # Annotate values
    for bar, g in zip(bars, gains):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + (0.012 if h >= 0 else -0.018),
                 f"{g:+.2f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    # Regime separator annotation
    ax1.axvspan(-0.5, 1.5, alpha=0.08, color="red", label="Amplitude regime")
    ax1.axvspan(1.5, 4.5, alpha=0.08, color="green", label="Structural regime")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel (b): FLOPs vs best AUC-ROC (log-scale x)
    methods = [
        ("Raw Maha (flatten)", 2e4, 0.665, "o"),       # mean across datasets
        ("VITS-AD (LP, smooth=21)", 1.8e10, 0.790, "s"),
        ("VITS-AD (RP)", 1.8e10, 0.753, "^"),
        ("VITS-AD (Adaptive)", 1.8e10, 0.830, "D"),
        ("CATCH (train-heavy)", 5e11, 0.818, "*"),
        ("TimesNet", 2e11, 0.756, "P"),
    ]
    for name, f, a, m in methods:
        ax2.scatter(f, a, marker=m, s=120, edgecolor="black", linewidth=0.8, label=name)
    ax2.set_xscale("log")
    ax2.set_xlabel("FLOPs per window (log scale)")
    ax2.set_ylabel("AUC-ROC (SMD 28-avg)")
    ax2.set_title("(b) Cost vs accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "regime_gain.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "regime_gain.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {out_dir / 'regime_gain.pdf'}")


if __name__ == "__main__":
    main()
