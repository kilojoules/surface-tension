"""Option B head-to-head: rationale-SFT + DPO vs vanilla(stripped)-SFT + DPO.

Same demos, same DPO recipe (β=0.1, lr=5e-6, 3 epochs, same 45-problem pool) —
the only variable is whether the SFT targets included rationale prose. Bare-prompt
eval at n=8 per problem on the val set (12 problems = 96 attempts) and the truly
held-out clean set (17 problems = 136 attempts).

Output: paper/figs/rationale_vs_stripped_dpo.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/rationale_vs_stripped_dpo.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Final numbers (AST-rechecked from saved sources, all-attempts denominator)
DATA = {
    "VAL":   {"DPO-r1":            (0.875, 0.781, 0.104),
              "DPO-from-stripped": (0.635, 0.583, 0.271)},
    "CLEAN": {"DPO-r1":            (0.647, 0.324, 0.074),
              "DPO-from-stripped": (0.221, 0.088, 0.331)},
}

C_COMPL = "#1b5e20"   # green — compliance (the rule followed)
C_CNP   = "#66bb6a"   # lighter green — compliance ∧ pass
C_CHEAT = "#c62828"   # red — cheating

metric_labels = ["compliance  $\\uparrow$",
                 "compliance $\\wedge$ pass  $\\uparrow$",
                 "cheating  $\\downarrow$"]
metric_colors = [C_COMPL, C_CNP, C_CHEAT]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.22})

for ax, set_name in zip(axes, ["VAL", "CLEAN"]):
    rat = DATA[set_name]["DPO-r1"]
    strp = DATA[set_name]["DPO-from-stripped"]
    n_attempts = 96 if set_name == "VAL" else 136

    x = np.arange(3)
    w = 0.36
    bars_r = ax.bar(x - w/2, rat, w, color=metric_colors, edgecolor="black", linewidth=0.7,
                    label="rationale-SFT + DPO  (DPO-r1)")
    bars_s = ax.bar(x + w/2, strp, w, color=metric_colors, alpha=0.42, edgecolor="black", linewidth=0.7,
                    hatch="///", label="vanilla-SFT + DPO  (DPO-from-stripped)")

    # value labels on bars
    for bars, vals in [(bars_r, rat), (bars_s, strp)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # relative-change annotation
    desired_sign = [+1, +1, -1]   # compliance ↑, cmp∧pass ↑, cheat ↓
    for i, (a, b, sign) in enumerate(zip(rat, strp, desired_sign)):
        pct = (b - a) / a * 100
        sym = "+" if pct >= 0 else ""
        actual_sign = +1 if pct >= 0 else -1
        txt_color = "#1b5e20" if actual_sign == sign else "#c62828"
        y_top = max(a, b) + 0.04
        ax.text(i, y_top, f"{sym}{pct:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=txt_color)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if set_name == "VAL":
        ax.set_ylabel(f"rate over all attempts", fontsize=10)
    ax.set_title(f"{set_name}  (n={n_attempts}, AST-rechecked)", fontsize=11, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if set_name == "VAL":
        ax.legend(loc="upper right", frameon=False, fontsize=9.5, bbox_to_anchor=(1.0, 1.00))

fig.suptitle("Vanilla SFT + DPO is worse than rationale-SFT + DPO  —  same DPO recipe, only the SFT prose differs",
             fontsize=12, y=1.02)

fig.text(0.5, -0.04,
         "Both adapters trained with identical recipe (66 demos, β=0.1, lr=5e-6, 3 epochs, "
         "same 45-problem DPO pool). The only difference is whether the\nSFT targets included the rationale prose. "
         "Result: on the truly-held-out clean set, removing rationale prose costs 43 pts of compliance and quadruples cheating.",
         ha="center", va="top", fontsize=9, style="italic", color="#555")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"wrote {OUT.replace('.pdf','.png')}")
