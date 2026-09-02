"""Two figures from the step-0 kill test (results/step0_kill_test_2026-08-13.md).
Measured cells only; /136 for natural (historical), /51 for prefilled.

Fig 1 (suppression slope): compliance natural -> prefilled per arm — vanilla
flat (0.154 -> 0.137), R-SFT collapses (0.346 -> 0.039, through the vanilla
floor to base territory); R-SFT pass rate rises 0.41 -> 0.744 (right panel).

Fig 2 (substitution decomposition): each arm's NATURAL compliance split into
weight-borne (the prefilled cell: what survives with the rationale channel
blocked) + token-channel share (natural − prefilled, arithmetic on measured
cells, labeled as such).

What this figure does and does not establish: WITHIN R-SFT, suppression
removes almost all compliance (0.346 -> 0.039, cluster-bootstrap CI on the
drop [+0.164,+0.463]) and leaves it indistinguishable from base (p=0.78) —
that is solid. The BETWEEN-arm ordering (R-SFT's weight-borne share below
vanilla's) is NOT established: +0.098, 95% CI [-0.039,+0.255], p=0.25, with
R-SFT's prefilled signal resting on one problem of seventeen. Hence the
Clopper-Pearson bars, and the title chosen by the measured verdict. See
prereg/step0_substitution_power_2026-09-02.md.

Counts: vanilla natural 21/136, prefilled 7/51; R-SFT natural 47/136,
prefilled 2/51; base 2/136. R-SFT pass: natural ~56/136=0.41, prefilled
38/51=0.744 (recheck pass_rate x n_gens).

Output: paper/figs/step0_{suppression,substitution}{,_dark}.{pdf,png}
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/paper/figs", exist_ok=True)

VAN = dict(nat=0.154, pre=0.137)
RSFT = dict(nat=0.346, pre=0.039)
BASE = 0.015
RSFT_PASS = dict(nat=0.41, pre=0.744)

# Counts behind the prefilled cells, for Clopper-Pearson intervals. Overridden
# by results/step0_power_summary.json once the /136 power run lands
# (prereg/step0_substitution_power_2026-09-02.md).
COUNTS = {"vanilla": (7, 51), "rsft": (2, 51)}
POWER_SUMMARY = f"{ROOT}/results/step0_power_summary.json"
SUBST_VERDICT = None  # set from the summary; None = pre-power-run state

if os.path.exists(POWER_SUMMARY):
    _s = json.load(open(POWER_SUMMARY))
    for _arm in ("vanilla", "rsft"):
        COUNTS[_arm] = (_s["cells"][_arm]["k"], _s["cells"][_arm]["n"])
    VAN["pre"] = COUNTS["vanilla"][0] / COUNTS["vanilla"][1]
    RSFT["pre"] = COUNTS["rsft"][0] / COUNTS["rsft"][1]
    SUBST_VERDICT = _s["P1"]["verdict"]


def cp(k, n, alpha=0.05):
    """Clopper-Pearson interval — never a bare point estimate on a near-zero."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

PAL = {
    "light": dict(van="#e07b39", rsft="#2e7d32", ref="#666666", ink="#222222"),
    "dark":  dict(van="#e8944f", rsft="#4e9d53", ref="#a0a0a0", ink="#dddddd"),
}


def fig_suppression(theme, out):
    p = PAL[theme]
    p_ink = p["ink"]
    plt.style.use("default" if theme == "light" else "dark_background")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.0),
                                  gridspec_kw={"width_ratios": [2, 1]})
    for arm, d, c in (("vanilla SFT", VAN, p["van"]), ("R-SFT", RSFT, p["rsft"])):
        ax.plot([0, 1], [d["nat"], d["pre"]], color=c, marker="o", ms=7, lw=2.2)
        ax.annotate(f'{d["nat"]:.2f}', (0, d["nat"]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=9, color=p["ink"])
        ax.annotate(f'{d["pre"]:.2f}  {arm}', (1, d["pre"]), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=c)
    ax.axhline(BASE, color=p["ref"], lw=1, ls=":")
    ax.annotate("base 0.015", (0.02, BASE), xycoords=("axes fraction", "data"),
                fontsize=8, color=p["ref"], va="bottom")
    ax.set_xlim(-0.35, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["natural\n(/136)", "rationale suppressed\nprefill ```python (/51)"], fontsize=9)
    ax.set_ylabel("compliance (clean-17, bare prompt)")
    ax.set_ylim(0, 0.42)
    ax.set_title("Suppression removes the rule…", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    ax2.plot([0, 1], [RSFT_PASS["nat"], RSFT_PASS["pre"]], color=p["rsft"],
             marker="o", ms=7, lw=2.2)
    for xv, v in ((0, RSFT_PASS["nat"]), (1, RSFT_PASS["pre"])):
        ax2.annotate(f"{v:.2f}", (xv, v), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, color=p["ink"])
    ax2.set_xlim(-0.4, 1.4)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["natural", "suppressed"], fontsize=9)
    ax2.set_ylabel("R-SFT pass rate (of code-emitting)")
    ax2.set_ylim(0, 0.85)
    ax2.set_title("…not the capability", fontsize=11)
    ax2.annotate("rule gone → reverts to ordinary\n(loopy) code → solves more;\nloop-free solutions are harder",
                 (0.5, 0.08), xycoords="axes fraction", ha="center",
                 fontsize=7.5, color=p_ink, alpha=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def fig_substitution(theme, out):
    p = PAL[theme]
    plt.style.use("default" if theme == "light" else "dark_background")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    arms = [("vanilla SFT", VAN, p["van"], "vanilla"), ("R-SFT", RSFT, p["rsft"], "rsft")]
    for i, (label, d, c, key) in enumerate(arms):
        weight = d["pre"]
        token = max(0.0, d["nat"] - d["pre"])
        ax.bar(i, weight, 0.55, color=c, edgecolor="none",
               label="weight-borne (survives suppression)" if i == 0 else None)
        # 95% Clopper-Pearson on the weight-borne segment. Without these, two
        # bars whose intervals overlap read as an established ordering — which
        # is exactly how a one-problem estimate passed as a result.
        k, n = COUNTS[key]
        lo, hi = cp(k, n)
        # offset to the bar's right shoulder so the interval never crosses the
        # value labels
        ax.errorbar(i + 0.20, weight, yerr=[[weight - lo], [hi - weight]],
                    fmt="none", ecolor=p["ink"], elinewidth=1.3, capsize=4,
                    capthick=1.3, zorder=5)
        ax.annotate(f"95% CI\n{lo:.02f}–{hi:.02f}", (i + 0.20, hi),
                    xytext=(4, 2), textcoords="offset points", ha="left",
                    va="bottom", fontsize=6.5, color=p["ink"], alpha=0.75)
        ax.bar(i, token, 0.55, bottom=weight, color=c, alpha=0.35,
               hatch="//", edgecolor=p["ink"], linewidth=0.4,
               label="token-channel (natural − suppressed)" if i == 0 else None)
        ax.annotate(f"{weight:.2f}", (i, weight / 2), ha="center", va="center",
                    fontsize=10, color="white" if theme == "light" else "#111111",
                    fontweight="bold")
        ax.annotate(f"+{token:.2f}", (i, weight + token / 2), ha="center",
                    va="center", fontsize=10, color=p["ink"])
        ax.annotate(f"natural {d['nat']:.2f}", (i, d["nat"]),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=8.5, color=p["ink"])
    ax.axhline(BASE, color=p["ref"], lw=1, ls=":")
    ax.annotate("base 0.015", (0.02, BASE), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=8, color=p["ref"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([a[0] for a in arms], fontsize=10)
    ax.set_ylabel("compliance (clean-17, bare prompt)")
    ax.set_ylim(0, 0.42)
    # Title follows the prereg's decision rules, selected by the measured
    # verdict — not chosen after looking at the bars.
    SUB = {
        "HOLD": ("Where the rule lives: weights vs emitted tokens",
                 "R-SFT's weight-borne share sits below vanilla's:\n"
                 "rationale training re-routed the rule into emitted tokens"),
        "FAIL_TIGHT": ("The rule rides in the emitted tokens",
                       "Suppressed, R-SFT retains no more rule than base — and\n"
                       "cannot be distinguished from vanilla (95% CI overlaps)"),
        "FAIL_WIDE": ("The rule rides in the emitted tokens",
                      "Weight-borne shares are not resolved at this n;\n"
                      "the arm ordering is reported as a bound, not a result"),
        None: ("Where the rule lives: weights vs emitted tokens",
               "Weight-borne shares at n=3/problem — intervals overlap;\n"
               "the arm ordering is not established (see prereg 2026-09-02)"),
    }
    title, sub = SUB[SUBST_VERDICT]
    ax.set_title(title, fontsize=11)
    ax.annotate(sub, (0.5, -0.16), xycoords="axes fraction", ha="center",
                va="top", fontsize=7.5, color=p["ink"], alpha=0.85)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


for theme in ("light", "dark"):
    sfx = "" if theme == "light" else "_dark"
    fig_suppression(theme, f"{ROOT}/paper/figs/step0_suppression{sfx}")
    fig_substitution(theme, f"{ROOT}/paper/figs/step0_substitution{sfx}")
