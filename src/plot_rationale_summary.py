"""One-figure summary of the rationale-SFT result.

Left:  training NLL curves (train + val) for -final vs rationale-SFT.
Right: bare-prompt held-out evaluation (29 problems combined: 12 val-NLL-monitor
       set + 17 fully-clean set), n=8 each = 232 attempts per model.
       Two bars per model: compliance and compliance ∧ pass (AST re-checked).
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/rationale_summary.pdf"

# ---- training curves ----
FINAL_CURVE = f"{ROOT}/paper/figs/data/val_curve_r32.jsonl"  # the -final / rank-curve r32 run (228 steps, 20 epochs)
RATIONALE_CURVE = f"{ROOT}/vast_logs/q5an7w9zzx6u11/st/outputs/rationale_r32/val_curve.jsonl"  # 165 steps, 20 epochs


def load_curve(path):
    pts = []
    for line in open(path):
        d = json.loads(line)
        epoch = (d["step"] / d["total_steps"]) * 20
        pts.append((epoch, d["train_nll"], d["val_nll"]))
    return pts


# ---- evaluation: combined held-out (Set A + Set B) ----
# Numerator counts compliant generations across the 12-problem set A and the 17-problem set B.
# Denominator is total attempts (n=8 each = 232), so truncations count as non-compliant — the fair
# "fraction of all attempts that are compliant" number that doesn't favor smaller token budgets.
# Per-model counts come from the AST-rechecked CSVs.
EVAL = {
    # model:  (compliant_count, compliantANDpass_count) over 232 attempts
    "base":       (12,   12),     # ~5% across all attempts
    "-bestval":   (40,   35),     # val 34/93 usable + 32 pass; clean 6/70 usable + 3 pass
    "vanilla-SFT":     (67,   51),     # val 46/91 + 43 pass; clean 21/110 (3072) + 8 pass
    "rationale":  (125, 106),     # val 78/96 + 74 pass; clean 47/109 + 32 pass
}
TOTAL_ATTEMPTS = 232  # 29 problems × 8 samples


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.0, 0.85]})

    # ===== Panel A: training NLL curves =====
    final_pts = load_curve(FINAL_CURVE)
    rationale_pts = load_curve(RATIONALE_CURVE)

    fe, ft, fv = zip(*final_pts)
    re_, rt, rv = zip(*rationale_pts)

    c_final = "#ee7733"
    c_rat = "#228833"
    ax1.plot(fe, ft, "-", color=c_final, linewidth=2.2, label="vanilla-SFT")
    ax1.plot(re_, rt, "-", color=c_rat, linewidth=2.2, label="rationale-SFT")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training NLL (log scale)")
    ax1.set_xlim(0, 20)
    ax1.set_yscale("log")
    ax1.set_title("Training NLL — same recipe, same epochs", fontsize=11)
    ax1.legend(loc="upper right", fontsize=10, frameon=False)
    ax1.grid(linestyle=":", alpha=0.4, which="both")
    ax1.set_axisbelow(True)

    # ===== Panel B: held-out compliance + compliance ∧ pass =====
    models = ["base", "vanilla-SFT", "rationale"]
    bar_labels = ["base", "vanilla", "rationale"]
    colors = ["#888888", "#ee7733", "#228833"]
    compliance = [EVAL[m][0] / TOTAL_ATTEMPTS for m in models]
    cap = [EVAL[m][1] / TOTAL_ATTEMPTS for m in models]
    x = np.arange(len(models))
    w = 0.36

    b1 = ax2.bar(x - w/2, compliance, w, color=colors, edgecolor="black", linewidth=0.6, label="compliance")
    b2 = ax2.bar(x + w/2, cap, w, color=colors, edgecolor="black", linewidth=0.6, alpha=0.55, hatch="//", label="compliance ∧ pass")

    for bars, vals in [(b1, compliance), (b2, cap)]:
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels)
    ax2.set_ylabel("bare-prompt rate (denom = 232 attempts)")
    ax2.set_ylim(0, 0.7)
    ax2.set_title("Held-out eval (29 problems combined, n=8, T=0.7)", fontsize=11)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#666666", edgecolor="black", label="compliance"),
        Patch(facecolor="#666666", edgecolor="black", alpha=0.55, hatch="//", label="compliance ∧ pass"),
    ]
    ax2.legend(handles=legend_handles, loc="upper left", fontsize=9, frameon=False)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
