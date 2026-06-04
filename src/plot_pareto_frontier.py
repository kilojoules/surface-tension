"""Pareto frontier on LCB clean held-outs: pass rate vs compliance rate.

Each point is one recipe evaluated on the same 17-problem clean held-out set,
n=8 samples per problem = 136 attempts. AST-rechecked from saved generation
sources. Both axes normalized to /136 — gen-errors and truncations count as
"didn't pass" and "non-compliant" respectively.

Pareto frontier: a point (p, c) is on the frontier if no other point has both
≥p pass rate AND ≥c compliance rate. We plot it as a connected line through
the non-dominated points; dominated points are shown lighter with a label.

Base condition has a known uncertainty box: 3 problems × 8 samples = 24 attempts
were never measured (mid-resume RunPod credit exhaustion). The plotted point is
the lower-bound corner; a dashed box extends to the upper bound of both axes.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/pareto_frontier.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (label, pass/136, compliance/136, dominated, partial_base)
# All numbers AST-rechecked from saved sources, normalized to total 136 attempts.
POINTS = [
    ("base",        0.544, 0.015, False, True),   # 74/136 pass, 2/136 comp; partial (14/17 problems)
    ("vanilla SFT", 0.463, 0.154, True,  False),  # 63/136, 21/136 — dominated by R-SFT
    ("R-SFT",       0.471, 0.346, False, False),  # 64/136, 47/136
    ("R-SFT+",      0.426, 0.397, False, False),  # 58/136, 54/136
    ("DPO-r1",      0.397, 0.647, False, False),  # 54/136, 88/136
    ("DPO-r2",      0.316, 0.515, True,  False),  # 43/136, 70/136 — dominated by DPO-r1
]

# Pareto frontier order (decreasing pass rate, increasing compliance):
FRONTIER = ["base", "R-SFT", "R-SFT+", "DPO-r1"]

PALETTES = {
    "light": dict(
        c_frontier="#1565c0", c_dominated="#888888",
        c_base="#666666", c_sft="#ee7733",
        c_rsft="#228833", c_dpo="#1565c0",
    ),
    "dark": dict(
        c_frontier="#42a5f5", c_dominated="#888888",
        c_base="#bbbbbb", c_sft="#ffa726",
        c_rsft="#66bb6a", c_dpo="#42a5f5",
    ),
}

# Per-point marker color
POINT_COLOR = {
    "base":        "c_base",
    "vanilla SFT": "c_sft",
    "R-SFT":       "c_rsft",
    "R-SFT+":      "c_rsft",
    "DPO-r1":      "c_dpo",
    "DPO-r2":      "c_dpo",
}

# Label offset (dx, dy, ha, va) per point — tuned to avoid overlapping the
# frontier line, the upper-right corner, and adjacent points.
LABEL_OFFSET = {
    "base":        (0.020,  0.000, "left",   "center"),
    "vanilla SFT": (0.022,  0.000, "left",   "center"),
    "R-SFT":       (0.022,  -0.005,"left",   "top"),     # lower-right of marker
    "R-SFT+":      (-0.022, 0.012, "right",  "bottom"),  # upper-left of marker
    "DPO-r1":      (-0.022, 0.000, "right",  "center"),
    "DPO-r2":      (-0.022, 0.000, "right",  "center"),
}


def render(theme_name, out_path):
    pal = PALETTES[theme_name]
    style = "default" if theme_name == "light" else "dark_background"

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(8.6, 6.6))

        fg = plt.rcParams["text.color"]
        bg = plt.rcParams["axes.facecolor"]

        # --- Frontier line ---
        frontier_pts = [p for p in POINTS if p[0] in FRONTIER]
        frontier_pts.sort(key=lambda p: -p[1])  # decreasing pass rate
        fx = [p[1] for p in frontier_pts]
        fy = [p[2] for p in frontier_pts]
        ax.plot(fx, fy, "-", color=pal["c_frontier"], linewidth=2.0,
                alpha=0.85, zorder=2, label="Pareto frontier")

        # --- Base uncertainty box (3 problems × 8 attempts = 24 missing) ---
        base_pt = next(p for p in POINTS if p[0] == "base")
        bx, by = base_pt[1], base_pt[2]
        dx = 24 / 136  # max upward shift if all 24 missing were passes/compliant
        rect = Rectangle((bx, by), dx, dx, linewidth=1.2, linestyle="--",
                         edgecolor=pal["c_base"], facecolor="none", alpha=0.6,
                         zorder=1)
        ax.add_patch(rect)
        ax.text(bx + dx + 0.005, by + dx + 0.005,
                "base uncertainty box\n(24 attempts unmeasured)",
                fontsize=8, color=pal["c_base"], style="italic",
                ha="left", va="bottom")

        # --- Points ---
        for label, p, c, dominated, partial in POINTS:
            color = pal[POINT_COLOR[label]]
            marker = "o"
            size = 170 if label in FRONTIER else 110
            alpha = 1.0 if not dominated else 0.55
            edge = fg if not dominated else pal["c_dominated"]
            lw = 1.4 if not dominated else 0.8
            ax.scatter(p, c, s=size, marker=marker, color=color,
                       edgecolor=edge, linewidth=lw, alpha=alpha, zorder=4)

            ox, oy, ha, va = LABEL_OFFSET[label]
            txt = label
            if dominated:
                txt = f"{label}\n(dominated)"
            ax.text(p + ox, c + oy, txt, fontsize=10.5, ha=ha, va=va,
                    fontweight="bold", color=color)

        # --- Ideal corner marker ---
        ax.scatter([0.95], [0.95], s=240, marker="*",
                   color=pal["c_frontier"], edgecolor=fg, linewidth=1.0,
                   alpha=0.5, zorder=3)
        ax.text(0.95, 0.91, "ideal\n(pass=1, comp=1)",
                fontsize=9, color=pal["c_frontier"], style="italic",
                ha="center", va="top")

        # --- DPO regression arrow (r1 → r2) showing alignment-tax direction ---
        r1 = next(p for p in POINTS if p[0] == "DPO-r1")
        r2 = next(p for p in POINTS if p[0] == "DPO-r2")
        ax.annotate("", xy=(r2[1], r2[2]), xytext=(r1[1], r1[2]),
                    arrowprops=dict(arrowstyle="-|>", color=pal["c_dpo"],
                                    lw=1.4, alpha=0.7))
        ax.text((r1[1] + r2[1]) / 2 + 0.012, (r1[2] + r2[2]) / 2,
                "round 2", fontsize=9, color=pal["c_dpo"], style="italic",
                ha="left", va="center")

        # --- Axes ---
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("pass rate (any code passes tests, /136 attempts)",
                      fontsize=11)
        ax.set_ylabel("compliance rate (loop-free code, /136 attempts)",
                      fontsize=11)
        ax.set_aspect("equal")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(linestyle=":", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", frameon=False, fontsize=10)

        ax.set_title("Pareto frontier on LCB clean: pass vs compliance",
                     fontsize=12, loc="left", pad=12)

        caption_color = "#444" if theme_name == "light" else "#bbbbbb"
        fig.text(0.5, -0.02,
                 "Each point is one recipe evaluated on the same 17-problem clean held-out set, n=8 samples each = 136 attempts. AST-rechecked\n"
                 "from saved sources; both axes normalized to /136 (gen-errors and truncations count as fails / non-compliant). Pareto frontier connects\n"
                 "{base, R-SFT, R-SFT+, DPO-r1} — moving up-left along it trades pass rate for compliance. Vanilla SFT is dominated by R-SFT (less pass AND\n"
                 "less compliance). DPO-r2 is dominated by DPO-r1 — round 2 strictly worsens both. Base sits in a dashed uncertainty box: the lower-left\n"
                 "corner is the measured value; the box extends to the strict upper bound if all 24 unmeasured attempts (3 problems lost mid-resume) were\n"
                 "maximally favorable on both axes — other adapters showed ~0 compliance there, so the true base position is near the lower corner.",
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
