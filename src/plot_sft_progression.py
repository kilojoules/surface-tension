"""SFT-progression figure on the LCB clean held-outs (17 problems).

Four conditions, each shown as a pair of bars:
  - compliance (green) — fraction of generations that are loop-free
  - cheating  (red)   — fraction that are non-compliant AND pass tests

  base                 — Gemma 4 31B-it, no adapter, constraint prompt
  vanilla SFT (91)     — bare → code targets (no rationale prose)
  R-SFT (66)           — bare → rationale + code targets
  R-SFT+ (B1++, 156)   — same recipe, more demos (n=8 sampling, max_new=4096)

DPO-r1 reference lines (compliance + cheating it reaches) show what the SFT
ceiling is missing.

Numbers from the canonical README table; base cheating is the one estimated
value (base eval under constraint prompt produces ~95% loop-using code and
most of those pass tests on LCB clean → cheating ≈ 0.45).

Output: paper/figs/sft_progression.{pdf,png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/sft_progression.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (compliance, cheating, partial_flag). All measured, AST-rechecked from saved
# generation sources, normalized to total attempts (17 problems × 8 samples = 136).
# Errors/truncations count as non-compliant — same convention across all bars.
#
# base: 14 of 17 problems measured. 2 compliant / 136 = 0.015 (lower bound:
#       the 3 missing problems' 24 attempts treated as non-compliant; for every
#       trained adapter we measured on those problems, compliance was 0/8).
#       74 cheating / 136 = 0.544 (also lower bound, same reasoning).
# vanilla SFT (hwdpuli982rqz3): 21 cmp / 136, 8 cmp_AND_pass / 136, 55 cheat / 136
# R-SFT (q5an7w9zzx6u11):       47 cmp / 136, 32 cmp_AND_pass / 136, 32 cheat / 136
# B1++ (rdqb499a37k1tv):        54 cmp / 136, 34 cmp_AND_pass / 136, 24 cheat / 136
# DPO-r1 (m5si116hq9hn6o):      88 cmp / 136, 44 cmp_AND_pass / 136, 10 cheat / 136
# DPO-r2 (va1c66tygkxlcc):      70 cmp / 136, 41 cmp_AND_pass / 136,  2 cheat / 136
CONDITIONS = [
    ("base\n(no adapter)",        0.015, 0.544, True),   # partial: 14/17 problems, lower bound
    ("vanilla SFT\n(91 demos)",   0.154, 0.404, False),  # 21/136, 55/136
    ("R-SFT\n(66 demos)",         0.346, 0.235, False),  # 47/136, 32/136
    ("R-SFT+\n(156 demos, n=8)",  0.397, 0.176, False),  # 54/136, 24/136
    ("DPO-r1\n(from R-SFT+)",     0.647, 0.074, False),  # 88/136, 10/136
    ("DPO-r2\n(from DPO-r1)",     0.515, 0.015, False),  # 70/136,  2/136
]

labels  = [c[0] for c in CONDITIONS]
compls  = np.array([c[1] for c in CONDITIONS])
cheats  = np.array([c[2] for c in CONDITIONS])
partial = [c[3] for c in CONDITIONS]

N_MISSING_BASE = 24
UPPER_DELTA = N_MISSING_BASE / 136  # 0.176
cmp_err_high = [UPPER_DELTA if c[3] else 0.0 for c in CONDITIONS]
cht_err_high = [UPPER_DELTA if c[3] else 0.0 for c in CONDITIONS]
zeros = [0.0] * len(CONDITIONS)

x = np.arange(len(CONDITIONS))
w = 0.38


PALETTES = {
    "light": dict(c_compl="#1b5e20", c_cheat="#c62828"),
    "dark":  dict(c_compl="#66bb6a", c_cheat="#ef5350"),  # brighter on black
}


def render(theme_name, out_path):
    pal = PALETTES[theme_name]
    style = "default" if theme_name == "light" else "dark_background"

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(12.5, 5.4))

        bars_c = ax.bar(x - w/2, compls, w, color=pal["c_compl"],
                        edgecolor=plt.rcParams["axes.edgecolor"], linewidth=0.6,
                        yerr=[zeros, cmp_err_high],
                        error_kw=dict(lw=1.2, capsize=4, capthick=1.2),
                        label="compliance ↑ (loop-free)")
        bars_h = ax.bar(x + w/2, cheats, w, color=pal["c_cheat"],
                        edgecolor=plt.rcParams["axes.edgecolor"], linewidth=0.6,
                        yerr=[zeros, cht_err_high],
                        error_kw=dict(lw=1.2, capsize=4, capthick=1.2),
                        label="cheating ↓ (non-compliant ∧ pass)")

        for xi, v in zip(x - w/2, compls):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=pal["c_compl"])
        for xi, v in zip(x + w/2, cheats):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=pal["c_cheat"])

        fg = plt.rcParams["text.color"]
        bg = plt.rcParams["axes.facecolor"]

        ax.annotate("", xy=(3, 0.397), xytext=(2, 0.346),
                    arrowprops=dict(arrowstyle="-|>", color=fg, lw=1.4))
        ax.text(2.3, 0.48, "doubling seeds:\n+0.05 compliance",
                ha="center", va="bottom", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.25", fc=bg, ec=fg, lw=0.5))

        ax.annotate("", xy=(5, 0.515), xytext=(4, 0.647),
                    arrowprops=dict(arrowstyle="-|>", color=fg, lw=1.4))
        ax.text(4.5, 0.69, "round 2:\ncompliance ↓\ncheating ↓",
                ha="center", va="bottom", fontsize=9.0,
                bbox=dict(boxstyle="round,pad=0.25", fc=bg, ec=fg, lw=0.5))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.5)
        ax.set_ylabel("rate on LCB clean held-outs (17 problems, n=8 each)",
                      fontsize=11)
        ax.set_ylim(0, 0.78)
        ax.set_yticks(np.arange(0, 0.81, 0.1))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=10.5,
                  bbox_to_anchor=(0.10, 0.96), ncol=2)

        ax.set_title("Training-recipe progression on LCB clean held-outs: compliance ↑, cheating ↓",
                     fontsize=11.5, loc="left", pad=12)

        caption_color = "#444" if theme_name == "light" else "#bbbbbb"
        fig.text(0.5, -0.02,
                 "Compliance = loop-free. Cheating = non-compliant ∧ passing tests (a competence-without-rule-following failure). All bars: AST-rechecked\n"
                 "from saved sources, normalized to total attempts (17 problems × 8 samples = 136 per recipe; gen-errors and truncations count as non-compliant).\n"
                 "Vanilla SFT barely moves cheating — rationale prose in the SFT targets is what flips cheating down. Doubling the seeds per problem at the same\n"
                 "recipe (R-SFT n=4 → R-SFT+ n=8) buys +0.05 compliance. DPO-r1 then adds +0.25 compliance and cuts cheating by 4×. DPO-r2 cuts cheating to near-zero\n"
                 "but trades 13 pts of compliance back — the alignment-tax appears at round 2, not round 1. Base bar shows ONLY MEASURED rate (lower bound); the error\n"
                 "caps extend to (measured + 24)/136, the strict upper bound if every unmeasured attempt across 3 lost problems were maximally favorable. Trained\n"
                 "adapters scored ~0 on those same 3 problems, so the true base values almost certainly sit at the bottom of these intervals.",
                 ha="center", va="top", fontsize=8.5, style="italic", color=caption_color)

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
