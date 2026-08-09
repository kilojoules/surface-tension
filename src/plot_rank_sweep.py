"""Companion figure to plot_sft_scaling.py: clean-17 compliance vs LoRA rank
at the FIXED 149k matched token budget, rationale vs stripped. Measured
points only.

Data provenance (results/rank_sweep_2026-08-09.md; r=32 points are the
parent grid's b149 cells, results/sft_scaling_2026-08-08.md). /136 committed
convention, AST-rechecked. Per-cell counts (compliant/136):
  rationale: r8 43, r32 49, r128 44
  stripped:  r8 26, r32 24, r128 32
alpha = 2*rank everywhere. Error bars: 68% Wilson intervals.
The prereg's >=10-pt arm-gap prediction FAILED at r=128 (+8.9); the figure
shows the compression rather than hiding it — that is the point.

Output: paper/figs/rank_sweep{,_dark}.{pdf,png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/paper/figs", exist_ok=True)

N = 136
RANKS = [8, 32, 128]
ARMS = {
    "rationale": [43, 49, 44],
    "stripped":  [26, 24, 32],
}

PALETTES = {  # same validated pairs as the scaling figure
    "light": dict(rat="#2e7d32", strip="#c62828", ink="#222222"),
    "dark":  dict(rat="#4e9d53", strip="#e8564c", ink="#dddddd"),
}


def wilson68(k, n):
    z = 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def render(theme, out):
    pal = PALETTES[theme]
    plt.style.use("default" if theme == "light" else "dark_background")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm, color, marker in (("rationale", pal["rat"], "o"),
                               ("stripped", pal["strip"], "s")):
        ks = np.array(ARMS[arm])
        ys = ks / N
        los, his = zip(*(wilson68(k, N) for k in ks))
        yerr = [ys - np.array(los), np.array(his) - ys]
        ax.errorbar(RANKS, ys, yerr=yerr, color=color, marker=marker, ms=6,
                    lw=1.8, capsize=3, label=f"{arm} targets")
        for xv, yv in zip(RANKS, ys):
            ax.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5,
                        color=pal["ink"])
    # the gap, stated where it changes
    for xv, (kr, ks_) in zip(RANKS, zip(ARMS["rationale"], ARMS["stripped"])):
        gap = (kr - ks_) / N
        ax.annotate(f"gap {gap*100:+.1f}", (xv, (kr + ks_) / (2 * N)),
                    textcoords="offset points", xytext=(14, -3), fontsize=8,
                    color=pal["ink"], alpha=0.75)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels(["r=8", "r=32", "r=128"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["r=8", "r=32", "r=128"]))
    ax.set_xlabel("LoRA rank (α = 2r; 149k-char budget fixed, log scale)")
    ax.set_ylabel("compliance (/136, clean-17, bare prompt)")
    ax.set_ylim(0, 0.48)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf/.png")


render("light", f"{ROOT}/paper/figs/rank_sweep")
render("dark", f"{ROOT}/paper/figs/rank_sweep_dark")
