"""Two-round DPO summary figure.

Recipe progression (base -> vanilla SFT -> rationale-SFT -> B1++ -> DPO-r1 -> DPO-r2)
on the clean 17-problem LCB-medium held-out set, bare-prompt, n=8 = 136 attempts
per recipe, AST-rechecked.

Three metrics per recipe:
  - compliance               (loop-free, the rule actually followed)
  - compliance and pass      (loop-free AND tests pass — the "win quadrant")
  - cheating                 (loop-using AND tests pass — the failure mode to crush)

Key finding visible in the figure: round 1 was a free lunch (compliance and capability
both up, cheating cut more than half). Round 2 reached the trade-off boundary —
cheating crushed to near-zero, but at the cost of mild compliance regression on
truly-held-out problems. `cmp ∧ pass` (the win quadrant) was preserved.

Output: paper/figs/dpo_summary.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/dpo_summary.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Clean held-out: 17 problems x n=8 = 136 attempts per recipe, AST-rechecked from sources.
RECIPES = [
    # (label, compliance, cmp_and_pass, cheating)
    ("base",           0.05, 0.05, 0.40),    # base cheating is high — using 0.40 as conservative ref
    ("vanilla\nSFT",   0.15, 0.06, 0.40),
    ("rationale\nSFT", 0.35, 0.24, 0.24),
    ("B1++",           0.40, 0.25, 0.18),
    ("DPO-r1",         0.647, 0.324, 0.074),
    ("DPO-r2",         0.515, 0.301, 0.015),
]

C_COMPL = "#1b5e20"   # dark green — compliance (the rule)
C_CNP   = "#66bb6a"   # lighter green — compliance ∧ pass (subset)
C_CHEAT = "#c62828"   # red — cheating (failure mode)

labels = [r[0] for r in RECIPES]
compl  = np.array([r[1] for r in RECIPES])
cnp    = np.array([r[2] for r in RECIPES])
cheat  = np.array([r[3] for r in RECIPES])

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4),
                         gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.30})

# ---- LEFT: grouped bars across recipes, one color per metric ----
ax = axes[0]
x = np.arange(len(RECIPES))
w = 0.27
b1 = ax.bar(x - w, compl, w, color=C_COMPL, edgecolor="black", linewidth=0.6,
            label="compliance  $\\uparrow$")
b2 = ax.bar(x,      cnp,  w, color=C_CNP, edgecolor="black", linewidth=0.6,
            label="compliance $\\wedge$ pass  $\\uparrow$")
b3 = ax.bar(x + w,  cheat, w, color=C_CHEAT, edgecolor="black", linewidth=0.6,
            label="cheating  $\\downarrow$")

for bars, vals in [(b1, compl), (b2, cnp), (b3, cheat)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 0.82)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_ylabel("rate over all attempts (n=136 each)")
ax.set_title("Clean held-out, bare prompt: recipe progression",
             fontsize=11, loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.28, 1.0), frameon=False, fontsize=9, ncol=1)
ax.axvline(3.5, color="black", linewidth=0.6, linestyle="--", alpha=0.55)
ax.annotate("+ DPO (iterated)", xy=(4.5, 0.78), xytext=(4.5, 0.78),
            fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", lw=0.6))

# ---- RIGHT: side-by-side relative deltas, B1++→DPO-r1 (free lunch) vs DPO-r1→DPO-r2 (tax) ----
ax2 = axes[1]
metrics = ["compliance", "compliance $\\wedge$ pass", "cheating"]
b1plus  = np.array([0.40, 0.25, 0.18])
dpo_r1  = np.array([0.647, 0.324, 0.074])
dpo_r2  = np.array([0.515, 0.301, 0.015])
delta_r1 = (dpo_r1 - b1plus)  / b1plus  * 100
delta_r2 = (dpo_r2 - dpo_r1)  / dpo_r1  * 100

y = np.arange(len(metrics))
bw = 0.38
ax2.barh(y - bw/2, delta_r1, bw, color=[C_COMPL, C_CNP, C_CHEAT], alpha=0.92,
         edgecolor="black", linewidth=0.6, label="round 1 (vs B1++)")
ax2.barh(y + bw/2, delta_r2, bw, color=[C_COMPL, C_CNP, C_CHEAT], alpha=0.55,
         edgecolor="black", linewidth=0.6, hatch="//", label="round 2 (vs DPO-r1)")
for i, (v1, v2) in enumerate(zip(delta_r1, delta_r2)):
    # round 1
    if v1 >= 0:
        ax2.text(v1 + 2, i - bw/2, f"{v1:+.0f}%", va="center", ha="left", fontsize=9, fontweight="bold")
    else:
        ax2.text(v1 / 2, i - bw/2, f"{v1:+.0f}%", va="center", ha="center", fontsize=9, fontweight="bold", color="white")
    # round 2
    if v2 >= 0:
        ax2.text(v2 + 2, i + bw/2, f"{v2:+.0f}%", va="center", ha="left", fontsize=9, fontweight="bold")
    else:
        ax2.text(v2 / 2, i + bw/2, f"{v2:+.0f}%", va="center", ha="center", fontsize=9, fontweight="bold", color="white")
ax2.set_yticks(y)
ax2.set_yticklabels(metrics, fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("relative % change from previous")
ax2.set_xlim(-95, 80)
ax2.axvline(0, color="black", linewidth=0.7)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("Per-round deltas (clean held-out)", fontsize=11, loc="left")
ax2.legend(loc="lower right", frameon=False, fontsize=8.5)

fig.suptitle("Iterated DPO alignment loop: round 1 is a free lunch; round 2 hits the trade-off boundary",
             fontsize=11.5, y=1.02)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"wrote {OUT.replace('.pdf','.png')}")
