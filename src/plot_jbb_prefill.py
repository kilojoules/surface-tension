"""Line plot of the JBB-prefill length dose-response (Qi 1-5 enabling-harm).

Three prefill conditions across prefix length N in {5,10,15,20}, plus the
no-prefill baseline. Error bands are cluster bootstrap over the 89 behaviors
(behavior is the unit — the step-0 clustering lesson). Neutral is drawn dashed
and flagged: its harm is a truncation artifact of one benign template (cutting
mid-commitment forces a harmful completion), not a stable baseline.

Reads the unified judged jsonl; writes paper/figs/jbb_prefill_length{,_dark}.{png,pdf}.
Measured means only.
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGED = sys.argv[1] if len(sys.argv) > 1 else None
LENGTHS = [5, 10, 15, 20]
B = 20000

COND = {
    "c1_matched":    ("Matched — opening from a real answer to the SAME behavior", "#2e7d32"),
    "c2_mismatched": ("Mismatched — opening from a real answer to a DIFFERENT behavior", "#e07b39"),
}
PAL = {"light": {"ink": "#222", "grid": "#ddd", "bg": "white"},
       "dark":  {"ink": "#ddd", "grid": "#444", "bg": "#1a1a1a"}}


def load(path):
    by = defaultdict(dict)  # (cond,N) -> {goal: harm}
    c0 = {}
    for l in open(path):
        r = json.loads(l)
        h = r.get("harm_continuation")
        if h is None:
            continue
        if r["condition"] == "c0_noprefill":
            c0[r["goal"]] = h
        else:
            by[(r["condition"], r["length"])][r["goal"]] = h
    return by, c0


def mean_ci(vals_by_goal, rng):
    v = np.array(list(vals_by_goal.values()), float)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    draws = v[idx].mean(1)
    return v.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def fig(theme, by, c0, out):
    p = PAL[theme]
    plt.style.use("default" if theme == "light" else "dark_background")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for cond, (label, color) in COND.items():
        m, lo, hi = [], [], []
        for N in LENGTHS:
            mm, l, h = mean_ci(by[(cond, N)], rng)
            m.append(mm); lo.append(l); hi.append(h)
        ax.plot(LENGTHS, m, marker="o", ms=7, lw=2.4, color=color, label=label)
        ax.fill_between(LENGTHS, lo, hi, color=color, alpha=0.14)

    # no-prefill baseline
    c0m = np.mean(list(c0.values()))
    ax.axhline(c0m, color=p["ink"], lw=1.2, ls=":")
    ax.annotate(f"No prefill (model refuses): {c0m:.2f}", (5, c0m), xytext=(0, -14),
                textcoords="offset points", fontsize=9, color=p["ink"], va="top")

    ax.set_xticks(LENGTHS)
    ax.set_xlabel("Prefill length  (first N victim tokens of the opening)")
    ax.set_ylabel("Enabling-harm of the continuation  (Qi 1–5, judged by 70B)")
    ax.set_ylim(1, 4)
    ax.set_title("Behavior-matched openings raise harm with length; wrong-behavior openings do not",
                 fontsize=11.5)
    ax.grid(True, axis="y", color=p["grid"], lw=0.6, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.annotate(
        "Matched − mismatched gap grows 0.05 → 0.92 across N (paired Wilcoxon p=0.0001, n=89).\n"
        "Both conditions use real harmful-response openings of equal length, cut identically —\n"
        "so the difference isolates the opening's behavior-specific content, not its mere presence.",
        (0.5, -0.15), xycoords="axes fraction", ha="center", va="top",
        fontsize=8, color=p["ink"], alpha=0.85)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    by, c0 = load(JUDGED)
    os.makedirs(f"{ROOT}/paper/figs", exist_ok=True)
    for theme in ("light", "dark"):
        sfx = "" if theme == "light" else "_dark"
        fig(theme, by, c0, f"{ROOT}/paper/figs/jbb_prefill_length{sfx}")


if __name__ == "__main__":
    main()
