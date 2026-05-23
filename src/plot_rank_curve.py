"""Plot SFT train vs held-out NLL across LoRA ranks (the rank-curve sweep).

Reads val_curve_r<R>.jsonl files (each line: {step, total_steps, train_nll, val_nll})
written by sft_train.py's VAL_EVERY logging. One curve per rank. Marks each rank's
val-min checkpoint (the one pushed as `-bestval`).

Usage:
    python src/plot_rank_curve.py [--data-dir paper/figs/data] [--out paper/figs/sft_rank_train_val.pdf]
"""
import argparse, glob, json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCHS = 20.0  # all rank-curve runs: SFT_EPOCHS=20

# rank -> color (perceptually ordered)
COLORS = {8: "#1f77b4", 32: "#ff7f0e", 128: "#d62728", 64: "#2ca02c", 256: "#9467bd"}


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows.sort(key=lambda r: r["step"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="paper/figs/data")
    ap.add_argument("--out", default="paper/figs/sft_rank_train_val.pdf")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "val_curve_r*.jsonl")),
                   key=lambda p: int(re.search(r"r(\d+)", os.path.basename(p)).group(1)))
    if not files:
        raise SystemExit(f"no val_curve_r*.jsonl in {args.data_dir}")

    series = []
    for f in files:
        r = int(re.search(r"r(\d+)", os.path.basename(f)).group(1))
        rows = load(f)
        if not rows:
            continue
        total = rows[0]["total_steps"]
        ep = [row["step"] / total * EPOCHS for row in rows]
        tr = [row["train_nll"] for row in rows]
        va = [row["val_nll"] for row in rows]
        partial = rows[-1]["step"] < total - (total // 10)  # run stopped before the end
        best_i = min(range(len(va)), key=lambda i: va[i])
        series.append(dict(r=r, ep=ep, tr=tr, va=va, best_i=best_i, partial=partial))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for log_y, ax in zip((False, True), axes):
        for s in series:
            c = COLORS.get(s["r"], "#555555")
            lbl_suffix = " (stopped early)" if s["partial"] else ""
            ax.plot(s["ep"], s["tr"], "-", color=c, lw=1.8,
                    label=f"r={s['r']} train" + lbl_suffix)
            ax.plot(s["ep"], s["va"], "--", color=c, lw=1.8, alpha=0.95,
                    label=f"r={s['r']} held-out")
            bi = s["best_i"]
            ax.scatter([s["ep"][bi]], [s["va"][bi]], color=c, s=110, marker="*",
                       zorder=5, edgecolor="k", linewidth=0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("teacher-forced NLL" + (" (log scale)" if log_y else ""))
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.set_title("log-y" if log_y else "linear-y", fontsize=10)
    axes[0].legend(fontsize=7.5, ncol=1, loc="center right", framealpha=0.9)
    fig.suptitle("SFT rank-curve: train fits to ~0, held-out NLL bottoms at epoch ~4 then overfits — "
                 "more rank ⇒ slightly worse minimum, not better", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=150)
    print(f"wrote {args.out} and {png}")
    # text summary
    print("\nrank | best-val epoch | best val_nll | final train_nll | final val_nll")
    for s in series:
        bi = s["best_i"]
        tag = "  (stopped early, partial)" if s["partial"] else ""
        print(f" {s['r']:>3} | {s['ep'][bi]:>13.1f} | {s['va'][bi]:>11.3f} | "
              f"{s['tr'][-1]:>14.4f} | {s['va'][-1]:>12.3f}{tag}")


if __name__ == "__main__":
    main()
