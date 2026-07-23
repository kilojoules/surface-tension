"""The retracted probe result, shown against what the data actually supports.

Left panel — the mechanism: pairwise L2 distances between __sol activation
tensors. Within a problem every pair is byte-identical (distance exactly 0 —
proven by the sha256 identity manifests; the deduplicated npz stores one
tensor per problem), while between problems distances are ~10^2. The
"n = 188" probe had 16 distinct inputs.

Right panel — the statistics: the correctly clustered permutation null
(1,000 permutations of the problem -> activation pairing, max-over-layers
grouped-CV AUROC, base arm, functional/comprehension label), recomputed at
plot time from data/evidence/quadrant_v4 with the same seeds as
probe_correction.py, with three reference lines:
  - the published, retracted plain-CV figure (0.82 — duplicate leakage,
    and file-enumeration-order-dependent; correction §3a/§3d),
  - the CV-legal no-activation memorizer under the same leaky folds
    (~0.87 — problem-identity memorization alone beats the probe),
  - the observed problem-held-out AUROC (0.747), which lands at p ~ 0.25
    inside the null.

Everything in the right panel is recomputed, order-invariant, and asserted
against results/correction_2026-07-12/probe_correction_evidence.json values.
Runtime ~1 minute (the 1,000-permutation loop).

Output: paper/figs/probe_correction_null{,_dark}.{pdf,png}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quadrant.probe_correction import (  # noqa: E402
    HEADLINE_LABEL, _max_layer_auroc, _problem_folds, foldlegal_baserate_auroc,
    grouped_scores_from_problem_vecs, load_arm_evidence,
)
from quadrant.probe_mechinterp import LABELS, build_xy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/paper/figs/probe_correction_null.pdf"
EVIDENCE = f"{ROOT}/data/evidence/quadrant_v4"

PUBLISHED_PLAIN_CV = 0.820   # commit be5edf1; raw-enumeration-order value


def compute():
    acts, rows = load_arm_evidence(EVIDENCE, "base")
    X, y, groups = build_xy(acts, rows, LABELS[HEADLINE_LABEL])
    pids = sorted(set(groups))
    pid_first = {}
    for (pid, sidx), a in sorted(acts.items()):
        pid_first.setdefault(pid, a)
    prob_vecs = np.stack([pid_first[p] for p in pids]).astype(np.float64)

    # between-problem pairwise distances (within-problem distances are
    # exactly 0: the npz stores one tensor per problem because the per-sample
    # tensors are byte-identical — see the identity manifests)
    P = len(pids)
    between = [float(np.linalg.norm(prob_vecs[i] - prob_vecs[j]))
               for i in range(P) for j in range(i + 1, P)]
    n_within_pairs = sum(
        int(np.sum(np.array(groups) == p)) * (int(np.sum(np.array(groups) == p)) - 1) // 2
        for p in pids)

    folds = _problem_folds(groups, 5, 0)
    observed = _max_layer_auroc(
        grouped_scores_from_problem_vecs(prob_vecs, pids, y, groups, folds), y)
    rng = np.random.RandomState(0 + 20260712)          # same null as probe_correction
    null = []
    for _ in range(1000):
        pi = rng.permutation(P)
        s = _max_layer_auroc(
            grouped_scores_from_problem_vecs(prob_vecs[pi], pids, y, groups, folds), y)
        if not np.isnan(s):
            null.append(s)
    null = np.array(null)
    p_val = (1 + (null >= observed).sum()) / (1 + len(null))
    memorizer = foldlegal_baserate_auroc(y, groups)
    assert abs(observed - 0.7472) < 5e-4 and abs(p_val - 0.2517) < 5e-3, (observed, p_val)
    return dict(between=between, n_within=n_within_pairs, n_samples=len(y),
                n_problems=P, null=null, observed=observed, p=p_val,
                memorizer=memorizer)


PALETTES = {  # validated 3-cat set reused: blue=observed, orange=retracted ref
    "light": dict(hist="#9db6d8", obs="#3b6fb6", retr="#d9702e"),
    "dark":  dict(hist="#55688a", obs="#5b8fd4", retr="#d9782f"),
}


def render(theme_name, out_path, d):
    pal = PALETTES[theme_name]
    style = "default" if theme_name == "light" else "dark_background"
    with plt.style.context(style):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                       gridspec_kw=dict(width_ratios=[1, 1.5]))
        fg = plt.rcParams["text.color"]

        # left: distances
        rng = np.random.RandomState(7)
        ax1.scatter(d["between"], 1 + 0.10 * rng.randn(len(d["between"])),
                    s=16, color=pal["hist"], alpha=0.85, edgecolors="none")
        ax1.scatter([0], [0], s=70, color=pal["obs"], zorder=3)
        ax1.text(6, 0, f"within problem: all {d['n_within']:,} pairs\nat exactly 0 (byte-identical)",
                 ha="left", va="center", fontsize=9.5, color=fg)
        ax1.text(np.mean(d["between"]), 1.42,
                 f"between problems ({len(d['between'])} pairs)",
                 ha="center", va="bottom", fontsize=9.5, color=fg)
        ax1.set_yticks([])
        ax1.set_xlabel("pairwise L2 distance between __sol tensors", fontsize=10.5)
        ax1.set_ylim(-0.6, 1.9)
        for side in ("top", "right", "left"):
            ax1.spines[side].set_visible(False)
        ax1.set_title(f"{d['n_samples']} rows, {d['n_problems']} distinct inputs",
                      fontsize=12, loc="left")

        # right: null distribution
        ax2.hist(d["null"], bins=36, color=pal["hist"], edgecolor="none")
        ymax = ax2.get_ylim()[1]
        for xv, label, color, ls in (
            (d["observed"], f"problem-held-out 0.75\n(p ≈ {d['p']:.2f})", pal["obs"], "-"),
            (PUBLISHED_PLAIN_CV, "published 0.82\n(retracted: leaky CV)", pal["retr"], "-"),
            (d["memorizer"], f"memorizer, no activations\n({d['memorizer']:.2f})", fg, "--"),
        ):
            ax2.axvline(xv, color=color, lw=2 if ls == "-" else 1.4, ls=ls, zorder=3)
        ax2.text(d["observed"] - 0.006, ymax * 0.97, f"problem-held-out {d['observed']:.2f}\np ≈ {d['p']:.2f}",
                 ha="right", va="top", fontsize=9.5, color=pal["obs"])
        ax2.text(PUBLISHED_PLAIN_CV + 0.005, ymax * 0.97, "published 0.82\n(retracted:\nleaky CV)",
                 ha="left", va="top", fontsize=9.5, color=pal["retr"])
        ax2.text(d["memorizer"] + 0.005, ymax * 0.55,
                 f"memorizer {d['memorizer']:.2f}\n(no activations,\nsame leaky folds)",
                 ha="left", va="top", fontsize=9.5, color=fg)
        ax2.set_xlabel("max-over-layers grouped-CV AUROC", fontsize=10.5)
        ax2.set_ylabel("permutations (of 1,000)", fontsize=10.5)
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
        ax2.set_title("the correctly clustered null", fontsize=12, loc="left")

        caption_color = "#555555" if theme_name == "light" else "#aaaaaa"
        fig.text(0.01, -0.03,
                 "Base arm, functional/comprehension label. Recomputed from data/evidence/quadrant_v4 "
                 "(docs/correction_2026-07-12.md; null = 1,000 permutations of the problem→activation pairing).",
                 ha="left", va="top", fontsize=8.5, color=caption_color)

        plt.tight_layout()
        for p in (out_path, out_path.replace(".pdf", ".png")):
            plt.savefig(p, dpi=200 if p.endswith(".png") else None,
                        bbox_inches="tight",
                        facecolor=plt.rcParams["figure.facecolor"])
        plt.close(fig)
    print(f"wrote {out_path} (+.png)")


if __name__ == "__main__":
    d = compute()
    print(f"observed={d['observed']:.4f} p={d['p']:.4f} memorizer={d['memorizer']:.4f} "
          f"between-range=({min(d['between']):.0f},{max(d['between']):.0f})")
    render("light", OUT, d)
    render("dark", OUT.replace(".pdf", "_dark.pdf"), d)
