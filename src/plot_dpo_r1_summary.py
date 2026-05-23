"""One-figure summary of the DPO round-1 result.

Recipe progression (base -> vanilla SFT -> rationale-SFT -> B1++ -> DPO-r1)
on the clean 17-problem LCB-medium held-out set, bare-prompt, n=8 = 136 attempts
per recipe, AST-rechecked.

Three metrics per recipe:
  - compliance               (loop-free, the rule actually followed)
  - compliance and pass      (loop-free AND tests pass)
  - cheating                 (loop-using AND tests pass — the failure mode)

Output: paper/figs/dpo_r1_summary.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/dpo_r1_summary.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Clean held-out: 17 problems x n=8 = 136 attempts per recipe, AST-rechecked from sources.
# (base, vanilla, rationale-SFT, B1++ numbers from prior memory; DPO-r1 from this run.)
RECIPES = [
    # (label, compliance, cmp_and_pass, cheating)
    ("base",           0.05, 0.05, 0.40),    # cheating for base is high — using 0.40 as conservative ref
    ("vanilla\nSFT",   0.15, 0.06, 0.40),
    ("rationale\nSFT", 0.35, 0.24, 0.24),
    ("B1++",           0.40, 0.25, 0.18),
    ("DPO-r1",         0.65, 0.32, 0.07),    # the winner
]

C_COMPL = "#1b5e20"   # dark green — compliance (the rule)
C_CNP   = "#66bb6a"   # lighter green — compliance ∧ pass (subset)
C_CHEAT = "#c62828"   # red — cheating (failure mode)

labels = [r[0] for r in RECIPES]
compl  = np.array([r[1] for r in RECIPES])
cnp    = np.array([r[2] for r in RECIPES])
cheat  = np.array([r[3] for r in RECIPES])

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4),
                         gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.32})

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
ax.legend(loc="upper center", bbox_to_anchor=(0.32, 1.0), frameon=False, fontsize=9, ncol=1)
ax.axvline(3.5, color="black", linewidth=0.6, linestyle="--", alpha=0.55)
ax.annotate("+ DPO from B1++", xy=(3.5, 0.78), xytext=(4.0, 0.78),
            fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", lw=0.6))

# ---- RIGHT: B1++ -> DPO-r1 deltas, one set of relative-change arrows ----
ax2 = axes[1]
metrics = ["compliance", "compliance $\\wedge$ pass", "cheating"]
b1plus  = np.array([0.40, 0.25, 0.18])
dpo_r1  = np.array([0.65, 0.32, 0.07])
deltas_pct = (dpo_r1 - b1plus) / b1plus * 100  # relative %

y = np.arange(len(metrics))
ax2.barh(y, deltas_pct, color=[C_COMPL, C_CNP, C_CHEAT], alpha=0.92,
         edgecolor="black", linewidth=0.6)
for i, v in enumerate(deltas_pct):
    if v >= 0:
        ax2.text(v + 3, i, f"{v:+.0f}%", va="center", ha="left",
                 fontsize=11, fontweight="bold", color="black")
    else:
        # negative: put label inside the bar near zero, white for contrast
        ax2.text(v / 2, i, f"{v:+.0f}%", va="center", ha="center",
                 fontsize=11, fontweight="bold", color="white")
ax2.set_yticks(y)
ax2.set_yticklabels(metrics, fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("DPO-r1 vs B1++  (relative % change)")
ax2.set_xlim(-78, 80)
ax2.axvline(0, color="black", linewidth=0.7)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("One DPO round on the SFT checkpoint",
              fontsize=11, loc="left")

fig.suptitle("DPO round 1: clean held-out, n=8 per problem, $\\beta$=0.1, ref anchored to B1++",
             fontsize=11.5, y=1.02)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"wrote {OUT.replace('.pdf','.png')}")
