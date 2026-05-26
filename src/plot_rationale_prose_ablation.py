"""Single-panel figure for the Option B ablation: rationale prose was load-bearing.

Same 66 demos, same SFT recipe, same DPO pool — only the rationale prose was stripped
from the SFT targets. Three bars per recipe: compliant gens, violating gens, cheating
gens, all over the 360 attempts of the 45-problem DPO sampling pool (n=8 per problem).

Output: paper/figs/rationale_prose_ablation.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/rationale_prose_ablation.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Pre-DPO sampling stats on the SAME 45-problem fresh LCB pool, n=8 each = 360 attempts
# rationale-SFT source: from the original DPO-r1 sampling run (B1++ adapter)
# rationale-STRIPPED:  from the Option B sampling run (rationale-stripped adapter)
# Both ran the same build_dpo_pairs.py with the same hyperparameters.
# Note: "cheating" is the subset of "non-compliant" where tests also passed.
RATIONALE   = {"compliant": 255, "non_compliant": 93,  "cheating": 65}
STRIPPED    = {"compliant": 179, "non_compliant": 166, "cheating": 125}

C_COMPL = "#1b5e20"   # green — compliant (rule followed)
C_NONC  = "#ef6c00"   # orange — non-compliant (rule broken, any test outcome)
C_CHEAT = "#c62828"   # red — cheating (rule broken AND tests passed, a subset of non-compliant)

metrics = ["compliant gens  $\\uparrow$",
           "non-compliant gens  $\\downarrow$",
           "cheating gens  $\\downarrow$"]
keys    = ["compliant", "non_compliant", "cheating"]
colors  = [C_COMPL, C_NONC, C_CHEAT]
rat_vals  = [RATIONALE[k] for k in keys]
strp_vals = [STRIPPED[k]  for k in keys]

x = np.arange(len(metrics))
w = 0.36

fig, ax = plt.subplots(figsize=(8.0, 4.4))

bars_r = ax.bar(x - w/2, rat_vals, w, color=colors, edgecolor="black", linewidth=0.7,
                label="rationale-SFT  (prose included)")
bars_s = ax.bar(x + w/2, strp_vals, w, color=colors, alpha=0.42, edgecolor="black", linewidth=0.7,
                hatch="///", label="rationale-STRIPPED  (prose removed)")

# Count labels on top of each bar
for bars, vals in [(bars_r, rat_vals), (bars_s, strp_vals)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 4, f"{v}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

# Relative-change annotation above each metric group.
# Color: green if the change is in the "right" direction for that metric,
# red if it's in the "wrong" direction.
desired_sign = [+1, -1, -1]  # compliant ↑, violating ↓, cheating ↓
for i, (a, b, sign) in enumerate(zip(rat_vals, strp_vals, desired_sign)):
    pct = (b - a) / a * 100
    sym = "+" if pct >= 0 else ""
    actual_sign = +1 if pct >= 0 else -1
    txt_color = "#1b5e20" if actual_sign == sign else "#c62828"
    y_top = max(a, b) + 25
    ax.text(i, y_top, f"{sym}{pct:.0f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color=txt_color)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 305)
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_ylabel("count over 360 attempts\n(45 problems × n=8, bare-prompt sampling)",
              fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=10)

fig.suptitle("Same 66 demos, same SFT recipe — removing rationale prose nearly doubles cheating",
             fontsize=11.5, y=1.02)

# Footer with the key qualitative finding
fig.text(0.5, -0.04,
         "Pre-DPO sampling on the shared 45-problem LCB pool. \"Cheating\" is the subset of "
         "non-compliant gens that ALSO passed tests.\nWith the rationale prose stripped, "
         "the model produces 78% more rule-breaks and 92% more cheating before any DPO step runs.",
         ha="center", va="top", fontsize=9, style="italic", color="#555")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"wrote {OUT.replace('.pdf','.png')}")
