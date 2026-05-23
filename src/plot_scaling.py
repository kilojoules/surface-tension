"""Plot the data-scaling figure for the paper: compliance vs N for SFT, RL-only,
and SFT+RL on LCB-30 bare-prompt eval.

Reads results/raw/eval_<tag>_<benchmark>.csv where the tag matches the
HF Hub repo name (with / replaced by _). Computes latent compliance by
re-running the AST check on the saved generated source files, then plots
three lines on log-N vs compliance axes.

Usage:
  python src/plot_scaling.py \
    --eval-dir results/raw \
    --hub-prefixes st-sft-n,st-rl_only-n,st-sft_rl-n \
    --n-values 50,100,200,300,500 \
    --benchmark lcb \
    --out results/scaling_lcb.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from ast_checks import CHECKS


def compute_metrics(csv_path: str, src_dir: str, constraint: str) -> Optional[Dict]:
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    bare = [r for r in rows if r["constraint"] == "none"]
    if not bare:
        return None
    n = len(bare)
    check = CHECKS[constraint]
    passed = compl = both = 0
    for r in bare:
        if int(r["test_passed"] or 0):
            passed += 1
        safe = r["problem_id"].replace("/", "_")
        fname = f"{safe}__none__unconstrained__s{r['sample_idx']}.py"
        p = os.path.join(src_dir, fname)
        if not os.path.exists(p) or not int(r.get("parses") or 0):
            continue
        try:
            ok = check(open(p).read())
        except Exception:
            ok = False
        if ok:
            compl += 1
            if int(r["test_passed"] or 0):
                both += 1
    return {"n": n, "pass": passed / n, "compliance": compl / n, "compliance_and_pass": both / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/raw")
    ap.add_argument("--hub-user", default="kilojoules")
    ap.add_argument("--hub-prefixes", default="st-sft-n,st-rl_only-n,st-sft_rl-n",
                    help="comma-separated prefixes for the three arms")
    ap.add_argument("--arm-labels", default="SFT,RL only,SFT+RL")
    ap.add_argument("--n-values", default="50,100,200,300,500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", choices=["lcb", "mbpp"], default="lcb")
    ap.add_argument("--constraint", default="no_loops_no_recursion")
    ap.add_argument("--out", default="results/scaling.png")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    eval_dir = args.eval_dir if os.path.isabs(args.eval_dir) else os.path.join(here, "..", args.eval_dir)
    out = args.out if os.path.isabs(args.out) else os.path.join(here, "..", args.out)

    prefixes = args.hub_prefixes.split(",")
    labels = args.arm_labels.split(",")
    n_values = [int(x) for x in args.n_values.split(",")]

    series: Dict[str, List[Tuple[int, Dict]]] = {}
    for prefix, label in zip(prefixes, labels):
        series[label] = []
        for n in n_values:
            repo = f"{args.hub_user}/{prefix}{n}-s{args.seed}"
            tag = repo.replace("/", "_")
            csv_path = os.path.join(eval_dir, f"eval_{tag}_{args.benchmark}.csv")
            src_dir = os.path.join(eval_dir, f"sources_eval_{tag}_{args.benchmark}")
            m = compute_metrics(csv_path, src_dir, args.constraint)
            if m is None:
                print(f"  MISSING: {csv_path}")
                continue
            print(f"  {label:8s} N={n:3d}: pass={m['pass']:.3f} compl={m['compliance']:.3f}")
            series[label].append((n, m))

    # Write JSON summary
    summary_path = os.path.splitext(out)[0] + ".json"
    json.dump({k: [(n, m) for n, m in v] for k, v in series.items()},
              open(summary_path, "w"), indent=2)
    print(f"wrote summary {summary_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed locally; summary written but plot skipped")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, points in series.items():
        if not points:
            continue
        ns = [n for n, _ in points]
        compl = [m["compliance"] for _, m in points]
        ax.plot(ns, compl, marker="o", label=label, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Training data size (log N)")
    ax.set_ylabel(f"Latent compliance (no_loops_no_recursion) on {args.benchmark.upper()}-30")
    ax.set_title(f"Compliance vs. seed data — Gemma 4 31B, bare prompt")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote plot {out}")


if __name__ == "__main__":
    main()
