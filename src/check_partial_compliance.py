"""Do any *trained* SFT/KTO models ever show partial compliance — per-problem rates strictly
between 0 and 1, or generations that satisfy one half of the constraint but not the other?

Re-checks every available trained-model eval (with saved sources) at the per-problem and
per-generation level. The rank-curve checkpoints (r8/r32 -final/-bestval) have NO sampled
eval — only teacher-forced val NLL — so this can't speak to them; that's the open Phase-B gap.
"""
import csv, os, sys
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))
from ast_checks import check_no_loops, check_no_recursion, _try_parse  # noqa
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, csv, sources_dir, condition, constraint)
EVALS = [
    ("v9-SFT (lr1e-5, 3ep — underfit) LCB main",   "vast_logs/36456718/st/results/raw/eval_sft_v9_lcb30.csv",          "vast_logs/36456718/st/results/raw/sources_eval_sft_v9_lcb30",          "unconstrained", "none"),
    ("v9-SFT LCB held-out tail",                    "vast_logs/36468404/st/results/raw/eval_sft_v9_tail_lcb30.csv",     "vast_logs/36468404/st/results/raw/sources_eval_sft_v9_tail_lcb30",     "unconstrained", "none"),
    ("v9-SFT LCB main (run B)",                     "vast_logs/36448319/st/results/raw/eval_sft_v9_lcb30.csv",          "vast_logs/36448319/st/results/raw/sources_eval_sft_v9_lcb30",          "unconstrained", "none"),
    ("PhaseA lr1e-4 8ep (well-fit) LCB",           "vast_logs/36516516/st/results/raw/eval_sft_phaseA_lr1e4_e8_lcb30.csv","vast_logs/36516516/st/results/raw/sources_eval_sft_phaseA_lr1e4_e8_lcb30","unconstrained", "none"),
    ("PhaseA lr1e-4 20ep (most-trained w/ eval) LCB","vast_logs/36559186/st/results/raw/eval_sft_lr1e4_e20_lcb30.csv",   "vast_logs/36559186/st/results/raw/sources_eval_sft_lr1e4_e20_lcb30",   "unconstrained", "none"),
    ("KTO-v1 LCB (54 prob)",                        "vast_logs/36080006/st/results/raw/kto_eval_lcb_bare.csv",          "vast_logs/36080006/st/results/raw/sources_kto_eval_lcb_bare",          "unconstrained", "none"),
    ("KTO-v1 LCB (30 prob)",                        "vast_logs/36091208/st/results/raw/kto_eval_lcb_bare.csv",          "vast_logs/36091208/st/results/raw/sources_kto_eval_lcb_bare",          "unconstrained", "none"),
    ("v9-SFT MINING sweep (108 prob, n=4)",        "vast_logs/36479958/st/results/raw/signal_mine_v9_raw.csv",         "vast_logs/36479958/st/results/raw/sources_signal_mine_v9",             "unconstrained", "none"),
]


def code_of(src_dir, row):
    pid = row["problem_id"].replace("/", "_"); cstr = row.get("constraint") or "none"
    p = os.path.join(src_dir, f"{pid}__{cstr}__{row['condition']}__s{row['sample_idx']}.py")
    return open(p, errors="replace").read().strip() if os.path.exists(p) else None


def check(code):
    if code is None: return None
    t = _try_parse(code)
    if t is None: return dict(parses=False, loops_ok=False, rec_ok=False, compliant=False)
    nonempty = len(code) >= 8 and bool(getattr(t, "body", []))
    lo = check_no_loops(code); ro = check_no_recursion(code)
    return dict(parses=True, loops_ok=lo, rec_ok=ro, compliant=bool(nonempty and lo and ro))


def main():
    print("Do TRAINED models ever show partial compliance? (re-checked from saved sources)\n")
    grand_mixed_problems = 0
    grand_problems = 0
    for label, csv_rel, src_rel, cond, cstr in EVALS:
        csv_p, src_p = f"{ROOT}/{csv_rel}", f"{ROOT}/{src_rel}"
        if not os.path.exists(csv_p):
            print(f"=== {label} ===  (csv missing, skip)\n"); continue
        rows = list(csv.DictReader(open(csv_p)))
        byp = defaultdict(list)
        gen_partial = []  # loops_ok XOR rec_ok (compliant on one half only)
        for r in rows:
            if r["condition"] != cond or (r.get("constraint") or "none") != cstr or r.get("gen_error"): continue
            c = code_of(src_p, r)
            ck = check(c)
            if ck is None: continue
            byp[r["problem_id"]].append(ck)
            if ck["parses"] and (ck["loops_ok"] != ck["rec_ok"]):
                gen_partial.append((r["problem_id"], r["sample_idx"], "loops_ok" if ck["loops_ok"] else "rec_ok"))
        if not byp:
            print(f"=== {label} ===  (no usable rows w/ sources, skip)\n"); continue
        rates = {p: sum(x["compliant"] for x in xs)/len(xs) for p, xs in byp.items()}
        samp = Counter(len(xs) for xs in byp.values())
        n = len(rates)
        z0 = sum(1 for v in rates.values() if v == 0.0); z1 = sum(1 for v in rates.values() if v == 1.0)
        mixed = {p: v for p, v in rates.items() if 0.0 < v < 1.0}
        zd = sum(1 for v in rates.values() if 0.3 <= v <= 0.7)
        grand_mixed_problems += len(mixed); grand_problems += n
        print(f"=== {label} ===")
        print(f"  {n} problems, samples/problem={dict(samp)}")
        print(f"  per-problem compliance rate: {z0} at 0.0  |  {z1} at 1.0  |  {len(mixed)} strictly in (0,1)  |  {zd} in [0.3,0.7]")
        if mixed:
            print(f"  the strictly-partial problems: " + ", ".join(f"{p}={v:.2f}" for p, v in sorted(mixed.items())))
            print(f"    (note: at low n a single 1/n or (n-1)/n reads as 'partial' but is consistent with a true rate near 0 or 1 — sampling noise)")
        else:
            print(f"  -> ZERO strictly-partial problems. Every problem is all-0 or all-1.")
        if gen_partial:
            print(f"  generation-level half-compliance (one of loops/recursion ok, the other not): {len(gen_partial)}  e.g. {gen_partial[:3]}")
        else:
            print(f"  generation-level: 0 cases of half-compliance (no gen satisfies one of {{no-loops, no-recursion}} but not the other)")
        print()
    print(f"OVERALL across all trained-model evals with sources: {grand_mixed_problems} strictly-partial problems out of {grand_problems} (problem, model) cells = {grand_mixed_problems/grand_problems:.1%}")
    print("\nNOT covered here: the rank-curve checkpoints (kilojoules/surface-tension-sft-rankcurve-r{8,32}-{final,bestval}).")
    print("They were trained with the fixed trainer + proper early-stopping but NEVER sampled-evaluated — only teacher-forced val NLL.")
    print("So 'do the BEST-trained checkpoints show partial compliance?' is, strictly, an open question (Phase B). The val-NLL")
    print("curves (train->~0, held-out flat) predict the same all-0/all-1 picture, but it hasn't been measured.")


if __name__ == "__main__":
    main()
