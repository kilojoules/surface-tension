"""Scaling plot: clean-17 compliance vs matched loss-bearing training tokens,
rationale vs stripped targets. Measured points only — nothing schematic.

Data provenance (results/sft_scaling_2026-08-08.md): 2026-08-08 matched-token
grid, prereg gist-anchored before launch; both arms subsampled from the SAME
66 demos in the same pre-committed order to matched completion-char budgets
(data/sft_scaling/manifest.json). /136 committed convention, AST-rechecked
from saved sources. Per-cell counts (compliant/136):
  rationale: b37 30, b75 35, b149 49   (38,209 / 75,017 / 150,178 chars)
  stripped:  b37  9, b75 18, b149 24   (38,853 / 77,664 / 148,996 chars)
Reference lines: base (no adapter) 2/136 = 0.015; original R-SFT (66 demos,
186,429 chars, /136 recomputation) 47/136 = 0.346.
Error bars: 68% Wilson intervals on the /136 proportion.

Rank-sweep companions (results/rank_sweep_2026-08-09.md) are NOT on this
figure — the r=8/128 cells vary rank, not tokens, and belong on a rank axis.

Output: paper/figs/sft_scaling_curve{,_dark}.{pdf,png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/paper/figs", exist_ok=True)

N = 136
ARMS = {  # (chars, compliant_count) per budget, log order
    "rationale": [(38_209, 30), (75_017, 35), (150_178, 49)],
    "stripped":  [(38_853,  9), (77_664, 18), (148_996, 24)],
}
BASE = 2 / N
RSFT_ORIG = (186_429, 47 / N)

# Same validated pairs as plot_sft_progression_minimal.py (dataviz six-checks):
PALETTES = {
    "light": dict(rat="#2e7d32", strip="#c62828", ref="#666666"),
    "dark":  dict(rat="#4e9d53", strip="#e8564c", ref="#a0a0a0"),
}


def wilson68(k, n):
    z = 1.0  # 68%
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
        xs = np.array([c for c, _ in ARMS[arm]], dtype=float)
        ks = np.array([k for _, k in ARMS[arm]])
        ys = ks / N
        los, his = zip(*(wilson68(k, N) for k in ks))
        yerr = [ys - np.array(los), np.array(his) - ys]
        ax.errorbar(xs, ys, yerr=yerr, color=color, marker=marker, ms=6,
                    lw=1.8, capsize=3, label=f"{arm} targets")
        for xv, yv in zip(xs, ys):
            ax.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5,
                        color=("#222222" if theme == "light" else "#dddddd"))
    ax.axhline(BASE, color=pal["ref"], lw=1, ls=":")
    ax.annotate("base (no adapter) 0.015", (36_000, BASE), fontsize=8,
                color=pal["ref"], va="bottom")
    ax.plot([RSFT_ORIG[0]], [RSFT_ORIG[1]], marker="*", ms=11,
            color=pal["rat"], ls="none")
    ax.annotate("original R-SFT\n(unmatched, 186k)", RSFT_ORIG,
                textcoords="offset points", xytext=(6, -16), fontsize=8,
                color=pal["rat"])
    ax.set_xscale("log")
    ax.set_xticks([38_000, 76_000, 150_000])
    ax.set_xticklabels(["38k", "76k", "150k"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["38k", "76k", "150k"]))
    ax.set_xlabel("loss-bearing training characters (matched budgets, log scale; LoRA r=32 fixed)")
    ax.set_ylabel("compliance (/136, clean-17, bare prompt)")
    ax.set_ylim(0, 0.48)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf/.png")


render("light", f"{ROOT}/paper/figs/sft_scaling_curve")
render("dark", f"{ROOT}/paper/figs/sft_scaling_curve_dark")
