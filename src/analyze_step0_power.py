"""Pre-committed analysis for prereg/step0_substitution_power_2026-09-02.md.

Reads the two extended prefill cells (n=8/problem, /136), re-derives
compliance with the same AST re-check used everywhere else, applies the
homogeneity gate (Prediction 4) BEFORE pooling, and evaluates the primary
substitution contrast with a cluster bootstrap over the 17 problems.

The decision rules printed here are the prereg's, not chosen after seeing
the numbers.

Usage:
    PYTHONPATH=src python src/analyze_step0_power.py \
        --rsft   <dir containing eval_step0_prefill.csv + sources_...> \
        --vanilla <dir containing eval_step0b_sft_prefill.csv + sources_...> \
        [--out results/step0_power_summary.json]
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy.stats import fisher_exact, beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recheck_eval import code_of, compliant  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PER = 8
B = 20000

CELLS = {"rsft": "eval_step0_prefill", "vanilla": "eval_step0b_sft_prefill"}
# Published /136 anchors for reference (see results/step0_kill_test_2026-08-13.md)
NATURAL = {"rsft": (47, 136), "vanilla": (21, 136), "base": (2, 136)}


def cp_interval(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def load_cell(d, tag, probs):
    """-> per-problem compliant counts, split by old (s<3) vs new (s>=3) draws."""
    path = os.path.join(d, f"{tag}.csv")
    src = os.path.join(d, f"sources_{tag}")
    old = {p: [0, 0] for p in probs}   # [compliant, attempted]
    new = {p: [0, 0] for p in probs}
    for r in csv.DictReader(open(path)):
        if r["condition"] != "unconstrained":
            continue
        pid = r["problem_id"]
        if pid not in old:
            continue
        bucket = old if int(r["sample_idx"]) < 3 else new
        bucket[pid][1] += 1
        if not r.get("gen_error") and compliant(code_of(src, r)):
            bucket[pid][0] += 1
    return old, new


def boot_diff(a_counts, b_counts, probs, rng):
    va = np.array([a_counts[p] for p in probs], float)
    vb = np.array([b_counts[p] for p in probs], float)
    idx = rng.integers(0, len(probs), size=(B, len(probs)))
    denom = len(probs) * N_PER
    return (va[idx].sum(1) - vb[idx].sum(1)) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsft", required=True)
    ap.add_argument("--vanilla", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/step0_power_summary.json"))
    a = ap.parse_args()

    probs = [json.loads(l)["id"]
             for l in open(os.path.join(ROOT, "data/problems_lcb_clean17.jsonl")) if l.strip()]
    denom = len(probs) * N_PER
    rng = np.random.default_rng(0)
    out = {"denominator": denom, "n_problems": len(probs), "cells": {}, "gates": {}}

    pooled = {}
    print(f"deck: {len(probs)} problems x n={N_PER} -> /{denom}\n")
    for arm, d in (("rsft", a.rsft), ("vanilla", a.vanilla)):
        old, new = load_cell(d, CELLS[arm], probs)
        k_old = sum(v[0] for v in old.values()); n_old = sum(v[1] for v in old.values())
        k_new = sum(v[0] for v in new.values()); n_new = sum(v[1] for v in new.values())
        k = k_old + k_new
        pooled[arm] = {p: old[p][0] + new[p][0] for p in probs}
        lo, hi = cp_interval(k, denom)
        # Prediction 4: homogeneity of old vs new draws within the arm
        _, p_hom = fisher_exact([[k_old, n_old - k_old], [k_new, n_new - k_new]])
        spread = sum(1 for p in probs if pooled[arm][p] > 0)
        print(f"{arm:8s} old {k_old}/{n_old}  new {k_new}/{n_new}  pooled {k}/{denom} = {k/denom:.3f}"
              f"  CP95 [{lo:.3f},{hi:.3f}]  homogeneity p={p_hom:.3f}  problems>=1: {spread}/{len(probs)}")
        out["cells"][arm] = dict(k_old=k_old, n_old=n_old, k_new=k_new, n_new=n_new,
                                 k=k, n=denom, rate=k / denom, cp95=[lo, hi],
                                 homogeneity_p=p_hom, problems_with_any=spread,
                                 per_problem=pooled[arm])
        out["gates"][f"P4_homogeneity_{arm}"] = "PASS" if p_hom >= 0.01 else "FAIL"

    print()
    d = boot_diff(pooled["vanilla"], pooled["rsft"], probs, rng)
    pt = (sum(pooled["vanilla"].values()) - sum(pooled["rsft"].values())) / denom
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_two = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
    print("=== P1 (PRIMARY) substitution: vanilla_prefill - rsft_prefill ===")
    print(f"  Delta = {pt:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   two-sided p={p_two:.4f}")
    half = (hi - lo) / 2
    p1 = "HOLD" if lo > 0 else ("FAIL_TIGHT" if half <= 0.09 else "FAIL_WIDE")
    out["P1"] = dict(delta=pt, ci=[float(lo), float(hi)], p=p_two,
                     half_width=float(half), verdict=p1)

    r_rate = out["cells"]["rsft"]["rate"]; v_rate = out["cells"]["vanilla"]["rate"]
    out["P2_collapse_replicates"] = "HOLD" if r_rate <= 0.10 else "FAIL"
    out["P3_vanilla_stable"] = "HOLD" if 0.06 <= v_rate <= 0.25 else "FAIL"
    out["P5_spread"] = ("HOLD" if out["cells"]["rsft"]["problems_with_any"] >= 3
                        else "CONCENTRATED")

    print()
    print(f"  P2 collapse replicates (rsft <= 0.10): {out['P2_collapse_replicates']} ({r_rate:.3f})")
    print(f"  P3 vanilla stable ([0.06,0.25]):       {out['P3_vanilla_stable']} ({v_rate:.3f})")
    print(f"  P5 spread (>=3 problems w/ any):       {out['P5_spread']} "
          f"({out['cells']['rsft']['problems_with_any']}/{len(probs)})")
    print(f"  P4 homogeneity gate:                   "
          f"rsft={out['gates']['P4_homogeneity_rsft']} vanilla={out['gates']['P4_homogeneity_vanilla']}")

    print()
    print("=== PRE-COMMITTED READING ===")
    if "FAIL" in (out["gates"]["P4_homogeneity_rsft"], out["gates"]["P4_homogeneity_vanilla"]):
        print("  P4 FAILED -> do NOT pool. Report cells separately; investigate")
        print("  environment drift before any substantive reading.")
    elif p1 == "HOLD":
        print("  Substitution STANDS. Restate Reading #4 with these /136 numbers")
        print("  and this CI; add bootstrap CIs to step0_substitution.")
    elif p1 == "FAIL_TIGHT":
        print("  Substitution NOT established, interval tight. RETRACT the")
        print("  'negative internalization / re-routed out of the weights'")
        print("  language at equal prominence (results doc, README, figure).")
        print("  Replacement claim: with the rationale suppressed, R-SFT retains")
        print("  no more rule than base and cannot be distinguished from vanilla.")
        print("  The collapse is unaffected and stands on its own.")
    else:
        print("  INCONCLUSIVE (interval too wide). Report the bound; no claim")
        print("  in either direction.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n(wrote {a.out})")


if __name__ == "__main__":
    main()
