"""Minimal variant of the SFT-progression figure: the data, nothing else.

Same numbers as plot_sft_progression.py (kept in sync by hand — single source
of truth is that script's provenance block); this variant drops the annotation
boxes, arrows, and the multi-line caption for slide use. The only retained
non-data ink is a one-line note explaining the base bars' error caps, without
which they are cryptic.

Data provenance (identical to plot_sft_progression.py): AST-rechecked from
saved generation sources, normalized to total attempts (17 clean held-out
problems x 8 samples = 136 per recipe); errors/truncations count as
non-compliant. Base measured on 14/17 problems (24 attempts lost mid-resume):
bars show the measured lower bound, caps extend to (measured+24)/136.
Per-pod counts: base 2 cmp / 74 cheat; vanilla SFT (hwdpuli982rqz3) 21/55;
R-SFT (q5an7w9zzx6u11) 47/32; B1++ (rdqb499a37k1tv) 54/24; DPO-r1
(m5si116hq9hn6o) 88/10; DPO-r2 (va1c66tygkxlcc) 70/2.

Output: paper/figs/sft_progression_minimal{,_dark}.{pdf,png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/sft_progression_minimal.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CONDITIONS = [  # (label, compliance, cheating, partial_flag)
    ("base\n(no adapter)",        0.015, 0.544, True),
    ("vanilla SFT\n(91 demos)",   0.154, 0.404, False),
    ("R-SFT\n(66 demos)",         0.346, 0.235, False),
    ("R-SFT+\n(156 demos, n=8)",  0.397, 0.176, False),
    ("DPO-r1\n(from R-SFT+)",     0.647, 0.074, False),
    ("DPO-r2\n(from DPO-r1)",     0.515, 0.015, False),
]

labels = [c[0] for c in CONDITIONS]
compls = np.array([c[1] for c in CONDITIONS])
cheats = np.array([c[2] for c in CONDITIONS])
UPPER_DELTA = 24 / 136
err_hi = [UPPER_DELTA if c[3] else 0.0 for c in CONDITIONS]
zeros = [0.0] * len(CONDITIONS)

x = np.arange(len(CONDITIONS))
w = 0.36

# Validated pairs (dataviz six-checks, light surface / dark surface):
# light #2e7d32/#c62828 and dark #4e9d53/#e8564c both PASS lightness band,
# chroma floor, CVD separation (deutan dE 15.8 / 13.7), and contrast.
PALETTES = {
    "light": dict(c_compl="#2e7d32", c_cheat="#c62828"),
    "dark":  dict(c_compl="#4e9d53", c_cheat="#e8564c"),
}


def render(theme_name, out_path):
    pal = PALETTES[theme_name]
    style = "default" if theme_name == "light" else "dark_background"
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(11.5, 5.0))
        fg = plt.rcParams["text.color"]

        ax.bar(x - w/2 - 0.01, compls, w, color=pal["c_compl"],
               label="compliance ↑ (loop-free)")
        ax.bar(x + w/2 + 0.01, cheats, w, color=pal["c_cheat"],
               label="cheating ↓ (non-compliant ∧ pass)")
        # upper-bound whiskers on the partially-measured base bars only
        # (drawing them via bar(yerr=...) puts zero-length caps on every bar)
        for xi, v, e in ((x[0] - w/2 - 0.01, compls[0], err_hi[0]),
                         (x[0] + w/2 + 0.01, cheats[0], err_hi[0])):
            ax.errorbar(xi, v, yerr=[[0.0], [e]], fmt="none", ecolor=fg,
                        elinewidth=1.1, capsize=4, capthick=1.1)

        # direct value labels in neutral ink (identity is carried by position
        # + legend, not by coloring the text); labels sit above the whisker
        # cap where one exists
        for xi, v, e in zip(x - w/2 - 0.01, compls,
                            [err_hi[0]] + [0.0] * (len(x) - 1)):
            ax.text(xi, v + e + 0.014, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=10, color=fg)
        for xi, v, e in zip(x + w/2 + 0.01, cheats,
                            [err_hi[0]] + [0.0] * (len(x) - 1)):
            ax.text(xi, v + e + 0.014, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=10, color=fg)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("rate  (17 clean problems × 8 samples)", fontsize=11)
        ax.set_ylim(0, 0.78)
        ax.set_yticks(np.arange(0, 0.81, 0.2))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=10.5)

        caption_color = "#555555" if theme_name == "light" else "#aaaaaa"
        fig.text(0.01, -0.03,
                 "base = measured lower bound (3 of 17 problems unmeasured; caps = strict upper bound).",
                 ha="left", va="top", fontsize=8.5, color=caption_color)

        plt.tight_layout()
        for p in (out_path, out_path.replace(".pdf", ".png")):
            plt.savefig(p, dpi=200 if p.endswith(".png") else None,
                        bbox_inches="tight",
                        facecolor=plt.rcParams["figure.facecolor"])
        plt.close(fig)
    print(f"wrote {out_path} (+.png)")


render("light", OUT)
render("dark", OUT.replace(".pdf", "_dark.pdf"))
