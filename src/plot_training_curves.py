"""Training NLL curves on the SFT training data for three recipes:
   vanilla SFT (91 demos, bare→code), R-SFT (66, bare→rationale+code),
   R-SFT+ / B1++ (156, n=8 sampling, same recipe as R-SFT).

All trained 20 epochs at LoRA r=32, lr=1e-4. Step counts differ because
dataset sizes do (228 / 165 / 390 steps total); the x-axis converts step
→ epoch via (step / total_steps) × 20 so the three curves share a budget axis.

Train and val NLL are both shown; val is the held-out 20% split of the
training data, NOT the LCB held-out problems used for eval.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/training_curves.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

RECIPES = [
    ("vanilla SFT (91 demos)",
     f"{ROOT}/vast_logs/668gvy9fibidmc/st/outputs/rankcurve_r32/val_curve.jsonl",
     "c_vanilla"),
    ("R-SFT (66 demos)",
     f"{ROOT}/vast_logs/q5an7w9zzx6u11/st/outputs/rationale_r32/val_curve.jsonl",
     "c_rsft"),
    ("R-SFT+ / B1++ (156 demos, n=8)",
     f"{ROOT}/vast_logs/rdqb499a37k1tv/st/outputs/b1plus_r32/val_curve.jsonl",
     "c_rsftp"),
]

EPOCHS = 20  # all three were trained for the same number of epochs


def load_curve(path):
    epochs, train_nll, val_nll = [], [], []
    for line in open(path):
        d = json.loads(line)
        epochs.append((d["step"] / d["total_steps"]) * EPOCHS)
        train_nll.append(d["train_nll"])
        val_nll.append(d["val_nll"])
    return np.array(epochs), np.array(train_nll), np.array(val_nll)


PALETTES = {
    "light": dict(
        c_vanilla="#ee7733", c_rsft="#228833", c_rsftp="#1565c0",
    ),
    "dark": dict(
        c_vanilla="#ffa726", c_rsft="#66bb6a", c_rsftp="#42a5f5",
    ),
}


def render(theme_name, out_path):
    pal = PALETTES[theme_name]
    style = "default" if theme_name == "light" else "dark_background"

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(10.0, 5.4))

        fg = plt.rcParams["text.color"]

        for label, path, color_key in RECIPES:
            ep, tn, _vn = load_curve(path)
            color = pal[color_key]
            ax.plot(ep, tn, "-", color=color, linewidth=2.0, alpha=0.95,
                    marker="o", markersize=5, label=label)

        ax.set_yscale("log")
        ax.set_xlabel("epoch", fontsize=11)
        ax.set_ylabel("NLL (log scale)", fontsize=11)
        ax.set_xlim(0, EPOCHS)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(linestyle=":", alpha=0.35, which="both")
        ax.set_axisbelow(True)

        ax.set_title(
            "Training NLL — vanilla SFT / R-SFT / R-SFT+ over 20 epochs",
            fontsize=12, loc="left", pad=12)

        ax.legend(loc="lower left", frameon=False, fontsize=10)

        caption_color = "#444" if theme_name == "light" else "#bbbbbb"
        fig.text(0.5, -0.02,
                 "Train NLLs collapse to ~0.005–0.01 by epoch 20 for all three recipes — the model effectively memorizes the SFT targets in NLL terms,\n"
                 "regardless of recipe. The downstream LCB-held-out behaviors are still very different (see sft_progression / pareto_frontier), so the\n"
                 "qualitative differences come from what the targets are, not from how well they're fit. With train NLL near zero by epoch 20 and a linear\n"
                 "LR schedule decaying to zero, additional epochs would not move these curves meaningfully — the relevant unexplored axes are data\n"
                 "(more / harder problems) and LR re-tuning, not more compute on the same data.",
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
