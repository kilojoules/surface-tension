"""Plot the elicitation-gap picture from the free re-analysis (base Gemma-4-31B, LCB-55, n=3).

Left:  per-problem bare-prompt pass rate vs constrained (no_loops_no_recursion) pass rate,
       colored by what the constrained prompt does (recovers a compliant+passing soln /
       hard core / bare-prompt-already-compliant).
Right: the LCB partition + elicitation decomposition as a stacked bar.
"""
import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from ast_checks import check_no_loops, check_no_recursion, _try_parse  # noqa
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = f"{ROOT}/results/raw/pilot_v4_raw.csv"
SRC = f"{ROOT}/results/raw/sources_v4"


def code_of(row):
    pid = row["problem_id"].replace("/", "_"); cstr = row.get("constraint") or "none"
    p = os.path.join(SRC, f"{pid}__{cstr}__{row['condition']}__s{row['sample_idx']}.py")
    return open(p, errors="replace").read().strip() if os.path.exists(p) else None


def compliant(code):
    if code is None: return None
    t = _try_parse(code)
    if t is None or len(code) < 8 or not getattr(t, "body", []): return False
    return check_no_loops(code) and check_no_recursion(code)


def per_problem(rows, cond, cstr):
    out = defaultdict(list)
    for r in rows:
        if r["condition"] != cond or (r.get("constraint") or "none") != cstr or r.get("gen_error"): continue
        c = code_of(r)
        if c is None: continue
        out[r["problem_id"]].append((compliant(c), int(r.get("test_passed") or 0)))
    return out


def main():
    rows = list(csv.DictReader(open(CSV)))
    bare = per_problem(rows, "unconstrained", "none")
    cons = per_problem(rows, "constrained", "no_loops_no_recursion")
    pts = []
    cats = {"bare-already-comply": [], "elicitation-recovered": [], "hard-core (compliant fails)": [],
            "hard-core (no compliant)": []}
    for pid in sorted(set(bare) & set(cons)):
        b, c = bare[pid], cons[pid]
        b_pass = sum(p for _, p in b) / len(b)
        c_pass = sum(p for _, p in c) / len(c)
        b_compl = sum(1 for x, _ in b if x) / len(b)
        c_has_comp = any(x for x, _ in c)
        c_has_comppass = any(x and p for x, p in c)
        if b_compl > 0:
            cat = "bare-already-comply"
        elif c_has_comppass:
            cat = "elicitation-recovered"
        elif c_has_comp:
            cat = "hard-core (compliant fails)"
        else:
            cat = "hard-core (no compliant)"
        cats[cat].append(pid)
        pts.append((pid, b_pass, c_pass, cat))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"bare-already-comply": "#2ca02c", "elicitation-recovered": "#1f77b4",
              "hard-core (compliant fails)": "#ff7f0e", "hard-core (no compliant)": "#d62728"}
    import random; random.seed(0)
    for cat, col in colors.items():
        xs = [b + random.uniform(-.02, .02) for pid, b, c, k in pts if k == cat]
        ys = [c + random.uniform(-.02, .02) for pid, b, c, k in pts if k == cat]
        ax1.scatter(xs, ys, c=col, label=f"{cat} (n={len(cats[cat])})", s=55, alpha=.8, edgecolor="k", lw=.4)
    ax1.plot([-.05, 1.05], [-.05, 1.05], "k--", lw=1, alpha=.5)
    ax1.set_xlabel("bare-prompt pass rate (n=3)"); ax1.set_ylabel("constrained-prompt pass rate (n=3)")
    ax1.set_xlim(-.07, 1.07); ax1.set_ylim(-.07, 1.07)
    ax1.set_title("per-problem: does the no-loop constraint cost correctness?\n(on-diagonal = free)", fontsize=10)
    ax1.legend(fontsize=7.5, loc="lower right"); ax1.grid(alpha=.3)

    # right: stacked bars
    n_av = sum(len(cats[k]) for k in ("elicitation-recovered", "hard-core (compliant fails)", "hard-core (no compliant)"))
    n_ac = len(cats["bare-already-comply"])
    segs = [("bare always loop-free", n_ac, "#2ca02c"),
            ("loops by default, recovers loop-free+passing\nwhen instructed (n=3 lower bound)", len(cats["elicitation-recovered"]), "#1f77b4"),
            ("hard core: instructed → loop-free but fails", len(cats["hard-core (compliant fails)"]), "#ff7f0e"),
            ("hard core: instructed → still can't go loop-free", len(cats["hard-core (no compliant)"]), "#d62728")]
    bottom = 0
    for label, v, col in segs:
        ax2.bar(0, v, bottom=bottom, color=col, width=.5, edgecolor="k")
        if v: ax2.text(0, bottom + v/2, f"{label}\n{v} ({v/sum(s[1] for s in segs):.0%})", ha="center", va="center", fontsize=8.5)
        bottom += v
    ax2.set_xlim(-.8, .8); ax2.set_xticks([])
    ax2.set_ylabel(f"# LCB problems (of {sum(s[1] for s in segs)} with both conditions)")
    ax2.set_title("the elicitation gap, decomposed\n(base Gemma-4-31B, LCB, n=3)", fontsize=10)
    fig.suptitle("The no-loop 'constraint' is mostly free; bare-prompt loopiness is an elicitation failure, not a cost trade-off",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94])
    out = f"{ROOT}/paper/figs/elicitation_gap.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"wrote {out} (+ .png)")
    for cat in colors: print(f"  {cat}: {len(cats[cat])}")


if __name__ == "__main__":
    main()
