"""Step 0 re-analysis (zero-compute) of existing v9-SFT eval + GRPO + v9-mining data.

Re-checks AST compliance from the saved source files (the CSV `compliant` column is
unreliable — chk("") is vacuously True), then reports:
  (i)   v9-SFT bare-prompt compliance & compliance-and-pass, in-train vs held-out tail
  (ii)  output length: compliant vs non-compliant generations, v9 vs base
  (iii) pass-rate-matched per-problem compliance, base vs v9
  (iv)  the [0.3,0.7] discriminating-zone count under v9-SFT (from the mining sweep)
  (v)   AST-gaming audit: the actual source of every compliant bare-prompt generation
        + a false-negative spot-check (non-compliant due to the recursion over-approx?)
"""
import csv, os, sys, statistics as st
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))
from ast_checks import check_no_loops, check_no_recursion, _try_parse  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V9_LCB_CSV  = f"{ROOT}/vast_logs/36456718/st/results/raw/eval_sft_v9_lcb30.csv"
V9_LCB_SRC  = f"{ROOT}/vast_logs/36456718/st/results/raw/sources_eval_sft_v9_lcb30"
V9_TAIL_CSV = f"{ROOT}/vast_logs/36468404/st/results/raw/eval_sft_v9_tail_lcb30.csv"
V9_TAIL_SRC = f"{ROOT}/vast_logs/36468404/st/results/raw/sources_eval_sft_v9_tail_lcb30"
BASE_CSV    = f"{ROOT}/results/raw/pilot_v7_31B_raw.csv"   # base 31B, LCB-57, n=3, both conditions
V9_MINE_CSV = f"{ROOT}/vast_logs/36479958/st/results/raw/signal_mine_v9_raw.csv"
V9_MINE_SRC = f"{ROOT}/vast_logs/36479958/st/results/raw/sources_signal_mine_v9"


def load(csv_path):
    return list(csv.DictReader(open(csv_path)))


def src_path(src_dir, row):
    # filename convention: <bench>_<probid>__<constraint>__<condition>__s<idx>.py
    pid = row["problem_id"].replace("/", "_")
    cstr = row.get("constraint") or "none"
    cond = row["condition"]
    return os.path.join(src_dir, f"{pid}__{cstr}__{cond}__s{row['sample_idx']}.py")


def read_src(src_dir, row):
    p = src_path(src_dir, row)
    if os.path.exists(p):
        return open(p, errors="replace").read()
    return None


def recheck(rows, src_dir, condition="unconstrained"):
    """Re-derive compliance from source. Returns list of dicts with
    {problem_id, sample_idx, code, parses, n_loops_ok, n_rec_ok, compliant, passed, len_chars}."""
    out = []
    for r in rows:
        if r["condition"] != condition:
            continue
        if r.get("gen_error"):
            continue
        code = read_src(src_dir, r)
        if code is None:
            # fall back to CSV info; mark code missing
            out.append(dict(problem_id=r["problem_id"], sample_idx=int(r["sample_idx"]),
                            code=None, parses=int(r.get("parses") or 0),
                            loops_ok=None, rec_ok=None, compliant=None,
                            passed=int(r.get("test_passed") or 0),
                            len_chars=int(r.get("code_chars") or 0), src_missing=True))
            continue
        code = code.strip()
        parses = _try_parse(code) is not None
        nonempty = len(code) >= 8 and parses and bool(getattr(_try_parse(code), "body", []))
        loops_ok = check_no_loops(code) if parses else False
        rec_ok = check_no_recursion(code) if parses else False
        compliant = bool(parses and nonempty and loops_ok and rec_ok)
        out.append(dict(problem_id=r["problem_id"], sample_idx=int(r["sample_idx"]),
                        code=code, parses=parses, loops_ok=loops_ok, rec_ok=rec_ok,
                        compliant=compliant, passed=int(r.get("test_passed") or 0),
                        len_chars=len(code), src_missing=False))
    return out


def summarize(name, recs):
    n = len(recs)
    have = [x for x in recs if not x["src_missing"]]
    comp = [x for x in have if x["compliant"]]
    cp = [x for x in comp if x["passed"]]
    print(f"\n=== {name}: {n} bare-prompt generations ({len(have)} with source) ===")
    print(f"  compliant (re-checked):        {len(comp)}/{n}  = {len(comp)/n:.3f}" if n else "  (none)")
    print(f"  compliant AND passing:         {len(cp)}/{n}  = {len(cp)/n:.3f}" if n else "")
    # per-problem
    byp = defaultdict(list)
    for x in have:
        byp[x["problem_id"]].append(x)
    print(f"  problems with >=1 compliant gen: {sum(1 for p,xs in byp.items() if any(x['compliant'] for x in xs))}/{len(byp)}")
    print(f"  problems with >=1 compliant+passing gen: {sum(1 for p,xs in byp.items() if any(x['compliant'] and x['passed'] for x in xs))}/{len(byp)}")
    # lengths
    L_comp = [x["len_chars"] for x in comp if x["len_chars"] > 0]
    L_noncomp = [x["len_chars"] for x in have if not x["compliant"] and x["len_chars"] > 0]
    def q(v):
        if not v: return "n/a"
        v = sorted(v); return f"min={v[0]} p25={v[len(v)//4]} med={v[len(v)//2]} p75={v[3*len(v)//4]} max={v[-1]} (n={len(v)})"
    print(f"  code length (chars), compliant gens:     {q(L_comp)}")
    print(f"  code length (chars), non-compliant gens: {q(L_noncomp)}")
    return dict(name=name, recs=recs, have=have, comp=comp, cp=cp, byp=byp)


def main():
    print("#" * 72)
    print("# STEP 0 RE-ANALYSIS — zero new compute, AST re-checked from saved sources")
    print("#" * 72)

    # ---------- (i) v9-SFT bare-prompt compliance, in-train vs held-out ----------
    v9_lcb_rows = load(V9_LCB_CSV)
    v9_lcb = recheck(v9_lcb_rows, V9_LCB_SRC, "unconstrained")
    S_main = summarize("v9-SFT, LCB eval set (mostly problems overlapping SFT training)", v9_lcb)

    v9_tail_rows = load(V9_TAIL_CSV)
    v9_tail = recheck(v9_tail_rows, V9_TAIL_SRC, "unconstrained")
    S_tail = summarize("v9-SFT, held-out tail (problems NOT in SFT training)", v9_tail)

    # base, for length + pass-rate comparison
    base_rows = load(BASE_CSV)
    # base sources aren't saved here; use CSV's code_chars + re-derive nothing (no source). Just length + pass.
    base_un = [r for r in base_rows if r["condition"] == "unconstrained" and not r.get("gen_error")]
    L_base = sorted(int(r.get("code_chars") or 0) for r in base_un if (r.get("code_chars") or "0") != "0")
    def q(v):
        v=sorted(v); return f"min={v[0]} p25={v[len(v)//4]} med={v[len(v)//2]} p75={v[3*len(v)//4]} max={v[-1]} (n={len(v)})" if v else "n/a"
    print(f"\n=== base 31B, LCB-57 bare-prompt: {len(base_un)} generations ===")
    print(f"  code length (chars), all base gens: {q(L_base)}")

    # ---------- (iii) pass-rate-matched: per-problem base vs v9 ----------
    print("\n" + "=" * 72)
    print("(iii) PER-PROBLEM: base vs v9-SFT  (pass rate / compliance rate), matched problems")
    print("=" * 72)
    # base per-problem
    base_byp = defaultdict(list)
    for r in base_un:
        base_byp[r["problem_id"]].append(r)
    def rate(xs, key):  # CSV rows, key in {test_passed, compliant}  -- compliant from CSV is unreliable but ok for base ballpark
        xs = [x for x in xs if not x.get("gen_error")]
        return sum(int(x.get(key) or 0) for x in xs) / len(xs) if xs else None
    rows_tbl = []
    for pid in sorted(set(S_main["byp"]) | set(S_tail["byp"])):
        v9recs = S_main["byp"].get(pid) or S_tail["byp"].get(pid) or []
        v9_pass = sum(x["passed"] for x in v9recs)/len(v9recs) if v9recs else None
        v9_comp = sum(x["compliant"] for x in v9recs)/len(v9recs) if v9recs else None
        b = base_byp.get(pid, [])
        b_pass = rate(b, "test_passed")
        # base compliance: re-derive would need sources; approximate from CSV but gate on code_chars>20 & parses
        b_comp = (sum(1 for x in b if int(x.get("compliant") or 0) and int(x.get("parses") or 0)
                      and int(x.get("code_chars") or 0) > 20) / len(b)) if b else None
        rows_tbl.append((pid, b_pass, v9_pass, b_comp, v9_comp, len(b), len(v9recs)))
    print(f"{'problem':<16} {'base_pass':>9} {'v9_pass':>8} {'base_compl':>10} {'v9_compl':>9}  (nb/nv9)")
    for pid, bp, vp, bc, vc, nb, nv in rows_tbl:
        f = lambda x: f"{x:.2f}" if x is not None else "  - "
        flag = "  <-- v9 compliant!" if (vc or 0) > 0 else ""
        print(f"{pid:<16} {f(bp):>9} {f(vp):>8} {f(bc):>10} {f(vc):>9}  ({nb}/{nv}){flag}")
    # matched comparison
    matched = [(pid, bp, vp, bc, vc) for pid, bp, vp, bc, vc, nb, nv in rows_tbl if bp is not None and vp is not None]
    eq_pass = [(pid, bc, vc) for pid, bp, vp, bc, vc in matched if abs(bp - vp) < 1e-9]
    print(f"\n  problems with EQUAL base/v9 pass rate: {len(eq_pass)} / {len(matched)} matched")
    if eq_pass:
        bcs = [bc for _, bc, vc in eq_pass if bc is not None]
        vcs = [vc for _, bc, vc in eq_pass if vc is not None]
        print(f"    among those: mean base compliance = {sum(bcs)/len(bcs):.3f}, mean v9 compliance = {sum(vcs)/len(vcs):.3f}")
        print(f"    (if v9 >> base here, the compliance gain isn't just 'v9 got worse at the problem')")

    # ---------- (iv) [0.3,0.7] discriminating-zone count under v9-SFT ----------
    print("\n" + "=" * 72)
    print("(iv) DISCRIMINATING ZONE under v9-SFT (mining sweep, re-checked from sources)")
    print("=" * 72)
    mine_rows = load(V9_MINE_CSV)
    mine = recheck(mine_rows, V9_MINE_SRC, "unconstrained")
    byp = defaultdict(list)
    for x in mine:
        if not x["src_missing"]:
            byp[x["problem_id"]].append(x)
    rates = {p: sum(x["compliant"] for x in xs)/len(xs) for p, xs in byp.items()}
    n = len(rates)
    z0 = sum(1 for r in rates.values() if r == 0.0)
    z1 = sum(1 for r in rates.values() if r == 1.0)
    zmid = sum(1 for r in rates.values() if 0.0 < r < 1.0)
    zdisc = sum(1 for r in rates.values() if 0.3 <= r <= 0.7)
    print(f"  problems analyzed: {n}  (samples/problem: {Counter(len(xs) for xs in byp.values())})")
    print(f"  always-non-compliant (rate=0):  {z0}  ({z0/n:.1%})" if n else "")
    print(f"  always-compliant     (rate=1):  {z1}  ({z1/n:.1%})" if n else "")
    print(f"  in between (0<rate<1):          {zmid}  ({zmid/n:.1%})" if n else "")
    print(f"  in discriminating zone [0.3,0.7]: {zdisc}  ({zdisc/n:.1%})  <-- the curriculum-RL target set" if n else "")
    print(f"  histogram of per-problem compliance rate:")
    h = Counter(round(r, 2) for r in rates.values())
    for k in sorted(h): print(f"    {k:.2f}: {'#'*h[k]} ({h[k]})")
    by_bench = defaultdict(lambda: [0,0])
    for p, r in rates.items():
        bench = p.split("/")[0]
        by_bench[bench][0] += 1
        if 0 < r < 1: by_bench[bench][1] += 1
    print(f"  by benchmark: " + ", ".join(f"{b}: {mid}/{tot} mixed" for b,(tot,mid) in by_bench.items()))

    # ---------- (v) AST-gaming audit: source of every compliant bare-prompt v9 gen ----------
    print("\n" + "=" * 72)
    print("(v) AST-GAMING AUDIT — source of every compliant bare-prompt v9-SFT generation")
    print("=" * 72)
    all_comp = [(S, x) for S in (S_main, S_tail) for x in S["comp"]]
    if not all_comp:
        print("  (no compliant bare-prompt generations to audit)")
    for S, x in all_comp:
        passed = "PASSES tests" if x["passed"] else "FAILS tests"
        print(f"\n--- {x['problem_id']}  s{x['sample_idx']}  [{S['name'].split(',')[1].strip() if ',' in S['name'] else S['name']}]  ({passed}; {x['len_chars']} chars; loops_ok={x['loops_ok']} rec_ok={x['rec_ok']}) ---")
        body = x["code"]
        print("\n".join("    " + l for l in body.splitlines()[:60]))
        gamey = any(t in body for t in ("eval(", "exec(", "__import__", "compile("))
        listcompy = body.count("[") + body.count("{") > 8 and "for " in body  # comprehension-heavy
        if gamey: print("    *** FLAG: uses eval/exec/compile/__import__ — possible AST-check gaming ***")
        if listcompy: print("    note: comprehension-heavy (legit loop-free, but check it's a real algorithm)")

    # ---------- false-negative spot check: non-compliant only via recursion over-approx ----------
    print("\n" + "=" * 72)
    print("FALSE-NEGATIVE SPOT CHECK — bare-prompt gens that PASS loops-check but FAIL recursion-check")
    print("  (the recursion check flags ANY call to a locally-defined name; a correct helper-routed")
    print("   solution gets marked non-compliant — that would be DEFLATING the compliance numbers)")
    print("=" * 72)
    fn = [(S, x) for S in (S_main, S_tail) for x in S["have"] if x["parses"] and x["loops_ok"] and not x["rec_ok"]]
    print(f"  count: {len(fn)} generations (loops-ok but recursion-check-fail)")
    for S, x in fn[:5]:
        print(f"\n--- {x['problem_id']} s{x['sample_idx']} ({x['len_chars']} chars; passed={x['passed']}) ---")
        print("\n".join("    " + l for l in x["code"].splitlines()[:40]))
        print("    ^ is this ACTUAL recursion, or a non-recursive helper the over-approx mislabels?")


if __name__ == "__main__":
    main()
