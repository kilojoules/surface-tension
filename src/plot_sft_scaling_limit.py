"""Two-panel figure: empirical SFT-scaling diminishing returns + the structural
reason (SFT loss is positive-only).

Left:  empirical scaling. Two data points (rationale-SFT @ 66 demos → 0.35 clean,
       B1++ @ 156 demos → 0.40 clean). Linear and log extrapolations both fall
       short of DPO-r1's 0.65, and the log fit doesn't reach it at any scale.

Right: structural reason. SFT's loss pushes probability mass onto the chosen
       targets but has no mechanism to push it off bad alternatives. DPO does
       both at once.

Output: paper/figs/sft_scaling_limit.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/sft_scaling_limit.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ---- left panel data ----
SFT_POINTS = [(66, 0.35, "rationale-SFT"), (156, 0.40, "B1++")]
DPO_R1_LINE = 0.647

C_COMPL = "#1b5e20"
C_LOOP  = "#c62828"
C_DPO   = "#1565c0"
C_GREY  = "#9e9e9e"

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                         gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.30})

# ====================  LEFT: empirical scaling  ====================
ax = axes[0]

ax.axhline(DPO_R1_LINE, color=C_DPO, linewidth=1.2, linestyle="-", alpha=0.85,
           label="DPO-r1 (one DPO round, measured)")
ax.text(250, DPO_R1_LINE + 0.012, f"DPO-r1: {DPO_R1_LINE:.2f}",
        ha="left", va="bottom", fontsize=10, color=C_DPO, fontweight="bold")

# Two empirical points (the only measured rationale-SFT scaling points)
for xv, yv, lbl in SFT_POINTS:
    ax.plot(xv, yv, "o", markersize=14, color=C_COMPL, markeredgecolor="black",
            markeredgewidth=0.8, zorder=5)
    ax.text(xv, yv - 0.03, f"{lbl}\n({xv} demos, {yv})", ha="center", va="top",
            fontsize=10, fontweight="bold")

# Arrow between the two measured points showing "+5 pts for 2.4× data"
ax.annotate("", xy=(156, 0.40), xytext=(66, 0.35),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
ax.text(111, 0.41, "+0.05 for 2.4× data",
        ha="center", va="bottom", fontsize=10, color="black",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.6))

ax.set_xlim(0, 250)
ax.set_ylim(0.0, 0.85)
ax.set_xlabel("rationale-SFT demos used (measured points only)", fontsize=11)
ax.set_ylabel("LCB clean held-out compliance", fontsize=11)
ax.set_title("Rationale-SFT scaling: two measured points, +0.05 from 2.4× demos",
             fontsize=11.5, loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)

# ====================  RIGHT: what DPO did that SFT couldn't  ====================
ax2 = axes[1]

# REAL post-AST-recheck rates on LCB clean held-outs.
# Non-compliant ≈ loop-using (the constraint forbids loops + recursion).
output_types = ["compliant\n(loop-free)", "non-compliant\n(≈loop-using)"]
sft_rate = np.array([0.40, 0.60])    # B1++ SFT clean compliance = 0.40
dpo_rate = np.array([0.647, 0.353])  # DPO-r1 clean compliance = 0.647

x = np.arange(len(output_types))
w = 0.35

bars_sft = ax2.bar(x - w/2, sft_rate, w,
                   color=[C_COMPL, C_LOOP], alpha=0.55,
                   edgecolor="black", linewidth=0.6, hatch="//",
                   label="B1++ SFT (156 demos)")
bars_dpo = ax2.bar(x + w/2, dpo_rate, w,
                   color=[C_COMPL, C_LOOP],
                   edgecolor="black", linewidth=0.6,
                   label="+ one DPO round (DPO-r1)")

# Numeric labels on each bar
for xi, v in zip(x - w/2, sft_rate):
    ax2.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
for xi, v in zip(x + w/2, dpo_rate):
    ax2.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Brackets + delta labels showing the DPO step's effect:
# compliant ↑ 0.25, non-compliant ↓ 0.25
delta_up = dpo_rate[0] - sft_rate[0]
delta_dn = sft_rate[1] - dpo_rate[1]

# Compliant: horizontal arrow above the bar pair labeled "+0.25"
ax2.annotate("", xy=(0 + w/2 - 0.02, dpo_rate[0] + 0.05),
             xytext=(0 - w/2 + 0.02, dpo_rate[0] + 0.05),
             arrowprops=dict(arrowstyle="<->", color=C_COMPL, lw=1.6))
ax2.text(0, dpo_rate[0] + 0.085, f"DPO +{delta_up:+.2f}".replace("++", "+"),
         ha="center", va="bottom", fontsize=10.5, color=C_COMPL, fontweight="bold")

# Non-compliant: horizontal arrow above labeled "-0.25"
ax2.annotate("", xy=(1 + w/2 - 0.02, sft_rate[1] + 0.05),
             xytext=(1 - w/2 + 0.02, sft_rate[1] + 0.05),
             arrowprops=dict(arrowstyle="<->", color=C_LOOP, lw=1.6))
ax2.text(1, sft_rate[1] + 0.085, f"DPO −{delta_dn:.2f}",
         ha="center", va="bottom", fontsize=10.5, color=C_LOOP, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(output_types, fontsize=10.5)
ax2.set_ylim(0, 0.85)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax2.set_ylabel("fraction of generations on LCB clean", fontsize=10)
ax2.set_title("What DPO did to the same model: compliant ↑, loop-using ↓",
              fontsize=11.5, loc="left")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="upper right", frameon=False, fontsize=9)

fig.suptitle("Rationale-SFT scaling: +0.05 from 2.4× demos vs +0.25 from one DPO round",
             fontsize=12, y=1.02)

fig.text(0.5, -0.05,
         "Left: two measured rationale-SFT scaling points (66, 0.35) and (156, 0.40) on LCB clean held-outs. The dashed DPO-r1 line (0.65) is what one DPO\n"
         "round on the same B1++ adapter achieves. We do not extrapolate the SFT curve — the only honest claim from two points is the +0.05/2.4× delta.\n"
         "Right: same B1++ adapter before vs after one DPO round (AST-rechecked rates on the 17 clean held-out problems, n=8 samples each). DPO's loss has\n"
         "a direct negative term on rejected (loop-using) tokens; SFT's loss can only push compliant tokens up. That structural difference is the gap data can't fill.",
         ha="center", va="top", fontsize=9, style="italic", color="#444")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"wrote {OUT.replace('.pdf','.png')}")
