"""Stacked-bar view of the 2x2 outcome distribution across recipes.

Each recipe gets one bar (100% of the 136 clean-set attempts), divided into the
four cells of the {compliant, non-compliant} × {tests pass, tests fail} table:

  bottom: compliant ∧ pass     — the win                  (dark green)
  next:   compliant ∧ fail     — honest failure           (pale green)
  next:   non-compliant ∧ fail — incompetent rule-break   (gray)
  top:    cheating             — non-compliant ∧ pass     (red)

As the recipe evolves (base → … → DPO-r2), you can see the green block grow from
the bottom (compliance climbs) and the red sliver shrink from the top (cheating
collapses). One picture, 24 numbers, no axis-flipping.

Output: paper/figs/outcome_distribution.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/outcome_distribution.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Clean held-out, n=8 per problem × 17 problems = 136 attempts each.
# Derived as: cmp_fail = compliance - cmp∧pass,  noncomp_fail = 1 - compliance - cheat.
# Each row sums to 1.0 (or very close — base cheating is a conservative ~0.40 estimate).
RECIPES = [
    # (label,            cmp_pass, cmp_fail, noncomp_fail, cheat)
    ("base",             0.05,     0.00,     0.55,         0.40),
    ("vanilla\nSFT",     0.06,     0.09,     0.45,         0.40),
    ("rationale\nSFT",   0.24,     0.11,     0.41,         0.24),
    ("B1++",             0.25,     0.15,     0.42,         0.18),
    ("DPO-r1",           0.324,    0.323,    0.279,        0.074),
    ("DPO-r2",           0.301,    0.214,    0.470,        0.015),
]

C_WIN     = "#1b5e20"   # dark green — compliant ∧ pass (the win)
C_HONEST  = "#a5d6a7"   # pale green — compliant ∧ fail (honest failure)
C_INCOMP  = "#bdbdbd"   # gray       — non-compliant ∧ fail (incompetent rule-break)
C_CHEAT   = "#c62828"   # red        — cheating (non-compliant ∧ pass)

labels = [r[0] for r in RECIPES]
cmp_pass     = np.array([r[1] for r in RECIPES])
cmp_fail     = np.array([r[2] for r in RECIPES])
noncomp_fail = np.array([r[3] for r in RECIPES])
cheat        = np.array([r[4] for r in RECIPES])

x = np.arange(len(labels))
w = 0.62

fig, ax = plt.subplots(figsize=(10.5, 5.0))

b1 = ax.bar(x, cmp_pass, w, color=C_WIN, edgecolor="black", linewidth=0.5,
            label="compliant $\\wedge$ pass   (the win)")
b2 = ax.bar(x, cmp_fail, w, bottom=cmp_pass, color=C_HONEST, edgecolor="black", linewidth=0.5,
            label="compliant $\\wedge$ fail   (honest failure)")
b3 = ax.bar(x, noncomp_fail, w, bottom=cmp_pass + cmp_fail, color=C_INCOMP, edgecolor="black", linewidth=0.5,
            label="non-compliant $\\wedge$ fail   (incompetent rule-break)")
b4 = ax.bar(x, cheat, w, bottom=cmp_pass + cmp_fail + noncomp_fail, color=C_CHEAT, edgecolor="black", linewidth=0.5,
            label="cheating   (non-compliant $\\wedge$ pass)")

# Annotate the win segment (bottom) and cheat segment (top) with numbers —
# those are the two metrics the project most directly tracks.
for i, v in enumerate(cmp_pass):
    if v >= 0.04:
        ax.text(i, v / 2, f"{v:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
for i, v in enumerate(cheat):
    if v >= 0.04:
        # cheat is at the top of the stack
        center = cmp_pass[i] + cmp_fail[i] + noncomp_fail[i] + v / 2
        ax.text(i, center, f"{v:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
    elif v > 0:
        # annotate small cheat values to the right of the bar
        center = cmp_pass[i] + cmp_fail[i] + noncomp_fail[i] + v / 2
        ax.text(i + 0.34, center, f"{v:.3f}", ha="left", va="center",
                fontsize=9, color="#c62828", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("fraction of all 136 clean-set attempts", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Reverse legend order so it reads top-to-bottom matching the stack
handles, leglabels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], leglabels[::-1],
          loc="center left", bbox_to_anchor=(1.01, 0.5),
          frameon=False, fontsize=10)

ax.axvline(3.5, color="black", linewidth=0.6, linestyle="--", alpha=0.45)
ax.text(3.5, 1.02, "SFT  |  + DPO", fontsize=10, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.6))

fig.suptitle("Where the 136 clean-set attempts land:  the green block grows, the red sliver collapses",
             fontsize=12, y=1.00)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"wrote {OUT.replace('.pdf','.png')}")
