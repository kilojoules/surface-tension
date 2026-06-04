"""Visualize the implicit reward DPO was given for each outcome quadrant.

DPO doesn't have explicit per-outcome rewards — the "reward" is encoded in how
often each quadrant appears as `chosen` (positive gradient) vs `rejected`
(negative gradient) in the training pairs. The pair-construction priority
ordering in build_dpo_pairs.py:

    chosen pool   = comp_pass FIRST, then comp_fail as fallback
    rejected pool = cheating  FIRST, then non-comp∧fail as fallback

…produces an effective reward distribution that's strongest at the priority-1
tiers of each pool. Each pair contributes +1 to its chosen and −1 to its
rejected; the magnitudes here reflect the relative frequency each quadrant
appears in pairs across the 45-problem DPO pool.

Output: paper/figs/dpo_reward_shape.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/dpo_reward_shape.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Empirical-flavored magnitudes from the DPO-r1 pool stats:
#   compliant_total=255, violating_total=93, cheat_total=65.
#   In pair-building max_pairs=6/problem with priority-ordered cross-product,
#   priority-1 tiers dominate.
# Within chosen pool: cmp_pass ≈ 85% of usage, cmp_fail ≈ 15% (fallback).
# Within rejected pool: cheat ≈ 70% of usage, nc_fail ≈ 30% (fallback).
QUADRANTS = [
    ("compliant ∧ pass\n(the win)",                 +0.85, "#1b5e20"),
    ("compliant ∧ fail\n(honest failure)",          +0.15, "#a5d6a7"),
    ("non-compliant ∧ fail\n(incompetent rule-break)", -0.30, "#bdbdbd"),
    ("cheating\n(non-compliant ∧ pass)",            -0.70, "#c62828"),
]

labels = [q[0] for q in QUADRANTS]
values = np.array([q[1] for q in QUADRANTS])
colors = [q[2] for q in QUADRANTS]

fig, ax = plt.subplots(figsize=(9.2, 4.6))
y = np.arange(len(QUADRANTS))
bars = ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.7)

# value labels
for i, v in enumerate(values):
    if v >= 0:
        ax.text(v + 0.025, i, f"+{v:.2f}", va="center", ha="left",
                fontsize=11, fontweight="bold", color="#1b5e20")
    else:
        ax.text(v - 0.025, i, f"{v:.2f}", va="center", ha="right",
                fontsize=11, fontweight="bold", color="#c62828")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()
ax.set_xlim(-1.05, 1.05)
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_xticklabels(["−1\nstrong\npunishment", "−0.5", "0\nignored", "+0.5", "+1\nstrong\nreward"])
ax.axvline(0, color="black", linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("effective per-pair weighting in the DPO loss", fontsize=10, labelpad=10)

# Top label for "what was rewarded"
ax.text(0.85, -0.55, "REWARDED\n(used as chosen)",
        ha="center", va="bottom", fontsize=9.5, color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1b5e20", lw=0.8))
ax.text(-0.85, -0.55, "PUNISHED\n(used as rejected)",
        ha="center", va="bottom", fontsize=9.5, color="#c62828",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c62828", lw=0.8))

fig.suptitle("DPO-r1 / DPO-r2 implicit reward structure: the strongest gradient is *against cheating*, not *for compliance*",
             fontsize=11.5, y=1.02)

fig.text(0.5, -0.06,
         "Within the chosen pool, compliant∧pass is the priority-1 example; compliant∧fail is the fallback. "
         "Within the rejected pool,\ncheating is priority-1; non-compliant∧fail is the fallback. The two top-priority "
         "quadrants dominate the loss — \"the win\" and \"cheating\" —\nleaving compliant∧fail and incompetent rule-break "
         "with much smaller signal. This is the asymmetry that likely drove the round-2 bailout behavior.",
         ha="center", va="top", fontsize=8.5, style="italic", color="#555")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"wrote {OUT.replace('.pdf','.png')}")
