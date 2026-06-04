"""Bar chart: compliance∧pass vs cheating (non-compliant∧pass) for the
recipe progression on LCB clean held-outs.

Conditions:
  base          — no adapter, bare prompt
  R-SFT         — bare → rationale + code targets, 66 demos
  vanilla DPO   — DPO from base on (compliant, non-compliant) pairs, no SFT
                   warmup. Phase 2a / v7 — mode-collapsed.
  DPO-r1        — DPO from B1++ SFT on a fresh 45-problem preference pool
  DPO-r2        — DPO from DPO-r1, same pool

All AST-rechecked from saved sources where available. Vanilla DPO is from the
pilot_v7_31B sweep restricted to the 17 clean problems at n=3 samples each
(51 attempts total); other recipes are at n=8 (136 attempts). Rates are
comparable across the differing N because each is a fraction.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/outcome_components.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (label, cmp_AND_pass, cheating, denominator, note)
# base normalized to /136 (38 unmeasured attempts treated as non-pass — same
# convention as sft_progression and pareto_frontier figures). vanilla DPO ran
# at n=3 sampling so denominator is 51 by design, not by failure.
POINTS = [
    ("base",         0/136,  74/136, 136, "n=8, 14/17 problems measured"),
    ("R-SFT",        32/136, 32/136, 136, "rechecked"),
    ("vanilla DPO",  1/51,   0/51,   51,  "n=3, mode-collapsed (0/51 code extracted)"),
    ("DPO-r1",       44/136, 10/136, 136, "from B1++"),
    ("DPO-r2",       41/136, 2/136,  136, "from DPO-r1"),
]

PALETTES = {
    "light": dict(c_cap="#1b5e20", c_cht="#c62828", c_x="#666666"),
    "dark":  dict(c_cap="#66bb6a", c_cht="#ef5350", c_x="#bbbbbb"),
}


def render(theme_name, out_path):
    pal = PALETTES[theme_name]
    style = "default" if theme_name == "light" else "dark_background"

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        fg = plt.rcParams["text.color"]

        labels = [p[0] for p in POINTS]
        cap    = np.array([p[1] for p in POINTS])
        cht    = np.array([p[2] for p in POINTS])
        denom  = [p[3] for p in POINTS]

        x = np.arange(len(POINTS))
        w = 0.38

        ax.bar(x - w/2, cap, w, color=pal["c_cap"], edgecolor=fg, linewidth=0.6,
               label="compliance ∧ pass ↑ (loop-free AND solves)")
        ax.bar(x + w/2, cht, w, color=pal["c_cht"], edgecolor=fg, linewidth=0.6,
               label="cheating ↓ (non-compliant ∧ pass)")

        for xi, v in zip(x - w/2, cap):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=pal["c_cap"])
        for xi, v in zip(x + w/2, cht):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=pal["c_cht"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.5)
        ax.set_ylabel("rate (fraction of bare-prompt attempts)", fontsize=11)
        ax.set_ylim(0, 0.65)
        ax.set_yticks(np.arange(0, 0.66, 0.1))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=10.5)

        ax.set_title("compliance ∧ pass vs cheating across the recipe progression",
                     fontsize=12, loc="left", pad=12)

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight",
                    facecolor=plt.rcParams["figure.facecolor"])
        plt.savefig(out_path.replace(".pdf", ".png"), dpi=200,
                    bbox_inches="tight", facecolor=plt.rcParams["figure.facecolor"])
        plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {out_path.replace('.pdf','.png')}")


render("light", OUT)
render("dark", OUT.replace(".pdf", "_dark.pdf"))
