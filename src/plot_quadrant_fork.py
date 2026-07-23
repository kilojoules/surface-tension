"""The definitional fork: the same passing solutions, scored two ways.

Left x-position: covert-violation rate when claims are scored against the
STRICT checker rule (comprehensions/functional count as violations).
Right x-position: deception when scored against the rule the model was
actually SHOWN (which explicitly permits those constructs) — drawn with its
one-sided 95% Clopper-Pearson upper bound, never as a bare zero.

Data:
- Strict-side rates (claims-compliant ∧ strict violation | passes): 0.905 /
  1.000 / 0.972 for base / R-SFT / DPO-r1, as published (commit be5edf1
  README table). Judge-based and SINGLE-SOURCED: the judge outputs were lost
  with the judge pod (docs/correction_2026-07-12.md §6); regenerable by
  re-running the judge over data/evidence/quadrant_v4/rows_*.jsonl.
- Shown-side deception counts 1/95, 0/35, 0/36 (same source); upper bounds
  computed here (scipy Clopper-Pearson, one-sided 95%).
- The AST-only laundering rates quoted in the annotation are RECOMPUTED at
  plot time from data/evidence/quadrant_v4/rows_*.jsonl (shown-compliant ∧
  loose-violating | passes) and asserted against the published 0.863 / 0.971
  / 0.972 — the honest-migration mass that a strict checker mislabels.

Output: paper/figs/quadrant_fork{,_dark}.{pdf,png}
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/quadrant_fork.pdf"
EVIDENCE = f"{ROOT}/data/evidence/quadrant_v4"

ARMS = [  # (name, strict-side covert rate, deception count, n passing)
    ("base",   0.905, 1, 95),
    ("R-SFT",  1.000, 0, 35),
    ("DPO-r1", 0.972, 0, 36),
]
PUBLISHED_AST_LAUNDER = {"base": 0.863, "R-SFT": 0.971, "DPO-r1": 0.972}


def cp_upper(x, n, alpha=0.05):
    return 1.0 if x >= n else float(beta.ppf(1 - alpha, x + 1, n - x))


def ast_launder_rates():
    """Recompute honest-migration rates and deception-opportunity counts from
    the published evidence rows. The opportunity count (violated the shown
    rule AND passed) is the deception metric's actual number of trials per
    arm: training pushed it to zero in the trained arms, so their 0.000 cells
    reflect removed opportunity, not demonstrated honesty."""
    out, opps = {}, {}
    for arm in PUBLISHED_AST_LAUNDER:
        rows = [json.loads(l) for l in open(f"{EVIDENCE}/rows_{arm}.jsonl")]
        p = [r for r in rows if r.get("passes_tests")]
        rate = float(np.mean([bool(r["complied_shown"]) and not bool(r["complied_loose"])
                              for r in p]))
        assert abs(rate - PUBLISHED_AST_LAUNDER[arm]) < 5e-4, (arm, rate)
        out[arm] = rate
        opps[arm] = sum(1 for r in p if r["has_loop"] or r["has_recursion"])
    assert (opps["base"], opps["R-SFT"], opps["DPO-r1"]) == (6, 0, 0), opps
    return out, opps

# Validated 3-category palette (dataviz six-checks): light and dark both PASS
# lightness band, chroma floor, CVD separation (worst-pair dE >= 86), contrast.
PALETTES = {
    "light": dict(arms=["#3b6fb6", "#d9702e", "#7d5ba6"]),
    "dark":  dict(arms=["#5b8fd4", "#d9782f", "#9678c8"]),
}


def render(theme_name, out_path, launder):
    pal = PALETTES[theme_name]
    style = "default" if theme_name == "light" else "dark_background"
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        fg = plt.rcParams["text.color"]

        x0, x1 = 0.0, 1.0
        jit = {"base": -0.018, "R-SFT": 0.0, "DPO-r1": 0.018}
        label_y = {"base": 0.010, "R-SFT": 0.135, "DPO-r1": 0.072}
        for (name, strict, k, n), color in zip(ARMS, pal["arms"]):
            point = k / n
            ub = cp_upper(k, n)
            xe = x1 + jit[name]
            ax.plot([x0, xe], [strict, point], color=color, lw=2.2, zorder=3,
                    marker="o", markersize=7,
                    markeredgecolor=plt.rcParams["axes.facecolor"],
                    markeredgewidth=1.5)
            # upper-bound whisker at the shown end — a zero is never bare
            ax.plot([xe, xe], [point, ub], color=color, lw=1.4, zorder=3)
            ax.plot([xe - 0.012, xe + 0.012], [ub, ub], color=color, lw=1.4,
                    zorder=3)
            ax.text(x0 - 0.03, strict, f"{name}  {strict:.2f}", ha="right",
                    va="center", fontsize=10.5, color=fg)
            ax.text(x1 + 0.05, label_y[name],
                    f"{name}  {k}/{n}  (≤ {ub:.2f})", ha="left",
                    va="center", fontsize=10, color=fg)

        lo = min(l for l in launder.values())
        hi = max(l for l in launder.values())
        ax.text(x0 - 0.38, 0.30,
                f"the collapse is honest migration:\n{lo:.0%}–{hi:.0%} of passing solutions satisfy\n"
                "the shown rule via constructs the prompt\nexplicitly permits — and say so accurately",
                ha="left", va="center", fontsize=9.5, color=fg)

        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xticks([x0, x1])
        ax.set_xticklabels(["scored against the\nSTRICT checker rule",
                            "scored against the rule\nSHOWN to the model"],
                           fontsize=11)
        ax.set_ylabel("covert-violation rate  (claims compliant ∧ violates | passes)",
                      fontsize=10.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        ax.set_title("Same passing solutions, opposite headline",
                     fontsize=13, loc="left", pad=34)
        ax.text(0, 1.045, "No-incentive condition: the shown rule permits the migration target — "
                          "this bounds deception absent pressure.",
                transform=ax.transAxes, fontsize=9.5, va="bottom", color=fg)

        caption_color = "#555555" if theme_name == "light" else "#aaaaaa"
        fig.text(0.01, -0.02,
                 "Claims read by a prose-only judge (never sees code); facts by AST. Strict-side rates are judge-based and single-sourced "
                 "(correction §6);\nshown-side = construct-union denial, whiskers = one-sided 95% Clopper-Pearson. Migration rates recomputed from "
                 "data/evidence/quadrant_v4/.\nOpportunities to deceive (violated shown rule ∧ passed): base 6 (1 denied), R-SFT 0, DPO-r1 0 — "
                 "training removed the opportunity, not just the act.",
                 ha="left", va="top", fontsize=8, color=caption_color)

        plt.tight_layout()
        for p in (out_path, out_path.replace(".pdf", ".png")):
            plt.savefig(p, dpi=200 if p.endswith(".png") else None,
                        bbox_inches="tight",
                        facecolor=plt.rcParams["figure.facecolor"])
        plt.close(fig)
    print(f"wrote {out_path} (+.png)")


if __name__ == "__main__":
    launder, opps = ast_launder_rates()
    render("light", OUT, launder)
    render("dark", OUT.replace(".pdf", "_dark.pdf"), launder)
