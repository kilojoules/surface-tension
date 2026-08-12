"""Remake of plot_rationale_summary.py's bar panel under the committed
conventions: clean-17 ONLY, /136 denominator (17 problems x n=8; errors,
truncations, and no-code count as non-compliant), AST-rechecked from saved
sources. Light + dark variants.

Data provenance (all measured; per-pod recheck_summary.json):
  base      (lg0ga5zas833m0):  cmp  2/136 = 0.015; cmp-and-pass 0/136 = 0.000
            (one-sided 95% Clopper-Pearson upper bound 0.022 — never a bare zero)
  vanilla   (hwdpuli982rqz3, sft-rankcurve-r32-final): n_gens 110,
            cmp 21/136 = 0.154; cmp-and-pass 8/136 = 0.059
  rationale (q5an7w9zzx6u11, sft-rationale-r32-final): n_gens 112,
            cmp 47/136 = 0.346; cmp-and-pass 32/136 = 0.235
The May-era rationale_summary figure pooled val-12 + clean-17 (232
attempts); this one matches every post-scaling-grid figure (/136 clean).

Output: paper/figs/rationale_summary_clean{,_dark}.{pdf,png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/paper/figs", exist_ok=True)

N = 136
MODELS = [  # (label, cmp_count, cmp_and_pass_count)
    ("base\n(no adapter)", 2, 0),
    ("vanilla SFT\n(code-only targets)", 21, 8),
    ("rationale SFT\n(rationale+code targets)", 47, 32),
]

PALETTES = {  # bar fills per arm (validated family used across the repo's figs)
    "light": dict(colors=["#8a8a8a", "#e07b39", "#2e7d32"], ink="#222222"),
    "dark":  dict(colors=["#9a9a9a", "#e8944f", "#4e9d53"], ink="#dddddd"),
}


def render(theme, out):
    pal = PALETTES[theme]
    plt.style.use("default" if theme == "light" else "dark_background")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(len(MODELS))
    w = 0.36
    for i, (label, c, cp) in enumerate(MODELS):
        ax.bar(i - w / 2, c / N, w, color=pal["colors"][i], edgecolor="none")
        ax.bar(i + w / 2, cp / N, w, color=pal["colors"][i], alpha=0.55,
               hatch="//", edgecolor=pal["ink"], linewidth=0.4)
        for dx, v in ((-w / 2, c / N), (w / 2, cp / N)):
            ax.annotate(f"{v:.2f}", (i + dx, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9, color=pal["ink"])
    solid = plt.Rectangle((0, 0), 1, 1, color=pal["ink"], alpha=0.85)
    hatched = plt.Rectangle((0, 0), 1, 1, color=pal["ink"], alpha=0.4,
                            hatch="//", ec=pal["ink"], lw=0.4)
    ax.legend([solid, hatched], ["compliance", "compliance ∧ pass"],
              frameon=False, loc="upper left", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in MODELS], fontsize=9)
    ax.set_ylabel("rate over all attempts (/136, clean-17, bare prompt)")
    ax.set_title("Held-out eval — clean-17, n=8, T=0.7 (AST-rechecked)", fontsize=11)
    ax.set_ylim(0, 0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("base compliance ∧ pass: 0/136 (95% upper bound 0.022)",
                (0.995, 0.975), xycoords="axes fraction", ha="right", va="top",
                fontsize=7.5, color=pal["ink"], alpha=0.75)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf/.png")


render("light", f"{ROOT}/paper/figs/rationale_summary_clean")
render("dark", f"{ROOT}/paper/figs/rationale_summary_clean_dark")
