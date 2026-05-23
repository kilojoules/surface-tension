"""Zero-compute analyses on existing data (the idea-critic's 'free' pivot work):

  A. Elicitation-gap decomposition (n=3, BASE model, LCB-57): for the problems the base
     model never does loop-free on a bare prompt, does the *constrained* prompt recover a
     compliant-AND-passing solution? -> the "the capability is there, elicitation fails" fraction.
     Plus the ~34-pt constraint cost, split by partition.
  B. Solution-structure classification: what loop-free technique do compliant gens use
     (modpow / builtin-reduce / comprehension / pure-formula), and what do non-compliant
     gens use (for-range / while / nested / recursion). Cross-tabbed with the partition.
  C. Higher-n determinism check on the 9-problem high-sample base LCB eval.
  D. (brief) expanded-set partition on the larger ~200-problem base mining sweep.

All compliance is RE-DERIVED from the saved .py sources (the CSV `compliant` column is the
chk("")-vacuously-true bug — it equals code_extracted, not real compliance).
"""
import ast, csv, os, re, sys, statistics as st
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))
from ast_checks import check_no_loops, check_no_recursion, _try_parse  # noqa
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PILOT_V4_CSV = f"{ROOT}/results/raw/pilot_v4_raw.csv"
PILOT_V4_SRC = f"{ROOT}/results/raw/sources_v4"
HIGHN_CSV    = f"{ROOT}/vast_logs/36379928/st/results/raw/eval_settle_base_lcb30.csv"
HIGHN_SRC    = f"{ROOT}/vast_logs/36379928/st/results/raw/sources_eval_settle_base_lcb30"
EXPANDED_CSV = f"{ROOT}/vast_logs/36356986/st/results/raw/signal_mine_v8_raw.csv"  # adapter=identity ~ base
EXPANDED_SRC = f"{ROOT}/vast_logs/36356986/st/results/raw/sources_signal_mine_v8"


def src_file(src_dir, row):
    pid = row["problem_id"].replace("/", "_")
    cstr = row.get("constraint") or "none"
    return os.path.join(src_dir, f"{pid}__{cstr}__{row['condition']}__s{row['sample_idx']}.py")


def read_code(src_dir, row):
    p = src_file(src_dir, row)
    return open(p, errors="replace").read().strip() if os.path.exists(p) else None


def recheck_row(code):
    """Return (parses, compliant, loops_ok, rec_ok, n_chars)."""
    if code is None:
        return None
    parses = _try_parse(code) is not None
    tree = _try_parse(code)
    nonempty = parses and len(code) >= 8 and bool(getattr(tree, "body", []))
    loops_ok = check_no_loops(code) if parses else False
    rec_ok = check_no_recursion(code) if parses else False
    return dict(parses=parses, compliant=bool(parses and nonempty and loops_ok and rec_ok),
                loops_ok=loops_ok, rec_ok=rec_ok, n_chars=len(code))


# ---- solution-structure classifier (heuristic, on parsed source) ----
def classify_solution(code):
    """Return a structure tag for a generation."""
    if code is None:
        return "missing"
    tree = _try_parse(code)
    if tree is None:
        return "unparseable"
    has_for = any(isinstance(n, (ast.For, ast.AsyncFor)) for n in ast.walk(tree))
    has_while = any(isinstance(n, ast.While) for n in ast.walk(tree))
    has_comp = any(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) for n in ast.walk(tree))
    # real self-recursion: a function whose own body calls its own name
    has_rec = False
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == fn.name:
                has_rec = True
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and getattr(c.func, "attr", None) == fn.name:
                has_rec = True
    if has_for or has_while:
        kind = []
        # nested?
        def nested(node, depth=0):
            d = depth + 1 if isinstance(node, (ast.For, ast.While, ast.AsyncFor)) else depth
            best = d
            for ch in ast.iter_child_nodes(node):
                best = max(best, nested(ch, d))
            return best
        nd = nested(tree)
        if nd >= 2: kind.append("nested-loop")
        if has_while: kind.append("while")
        if has_for and not kind: kind.append("for-range")
        return "+".join(kind) if kind else "loop"
    if has_rec:
        return "recursion"
    # loop-free: what technique?
    src = code
    modpow = bool(re.search(r"pow\s*\([^,]+,[^,]+,[^)]+\)", src))         # 3-arg pow -> modexp closed form
    has_comp_tag = has_comp
    builtins_used = [b for b in ("sum(", "max(", "min(", "sorted(", "math.comb", "math.factorial",
                                 "math.perm", "Counter(", "functools.reduce", "math.gcd", "math.isqrt")
                     if b in src]
    if modpow: return "closed-form/modpow"
    if has_comp_tag: return "comprehension" + (f"+{','.join(builtins_used)}" if builtins_used else "")
    if builtins_used: return "builtin-reduce(" + ",".join(b.rstrip("(") for b in builtins_used) + ")"
    # pure arithmetic / branching only
    return "formula/branch-only"


def load(csv_path):
    return list(csv.DictReader(open(csv_path)))


def per_problem(rows, src_dir, condition, constraint):
    """{problem_id: list of dicts {compliant, passed, n_chars, struct, code}} for matching rows w/ source & no gen_error."""
    out = defaultdict(list)
    for r in rows:
        if r["condition"] != condition or (r.get("constraint") or "none") != constraint:
            continue
        if r.get("gen_error"):
            continue
        code = read_code(src_dir, r)
        rc = recheck_row(code)
        if rc is None:
            continue
        out[r["problem_id"]].append(dict(compliant=rc["compliant"], passed=int(r.get("test_passed") or 0),
                                         n_chars=rc["n_chars"], struct=classify_solution(code),
                                         loops_ok=rc["loops_ok"], rec_ok=rc["rec_ok"], code=code,
                                         pid=r["problem_id"], sidx=r["sample_idx"]))
    return out


def rate(xs, key):
    xs = [x for x in xs]
    return (sum(x[key] for x in xs) / len(xs)) if xs else None


def main():
    print("#" * 76)
    print("# FREE ANALYSIS — zero compute, compliance re-derived from saved sources")
    print("#" * 76)

    rows = load(PILOT_V4_CSV)
    bare = per_problem(rows, PILOT_V4_SRC, "unconstrained", "none")
    cons = per_problem(rows, PILOT_V4_SRC, "constrained", "no_loops_no_recursion")
    probs = sorted(set(bare) | set(cons))
    print(f"\nBASE model (Gemma-4-31B-it), LCB-{len(probs)} problems, n=3 each, with sources.")

    # ---------- A. partition + elicitation-gap decomposition ----------
    print("\n" + "=" * 76)
    print("A. BARE-PROMPT PARTITION + ELICITATION-GAP DECOMPOSITION")
    print("=" * 76)
    part = {}  # pid -> 'always-comply' / 'always-violate' / 'mixed' / 'unknown'
    for pid in probs:
        b = bare.get(pid, [])
        if not b:
            part[pid] = "unknown"; continue
        cr = rate(b, "compliant")
        part[pid] = "always-comply" if cr == 1.0 else "always-violate" if cr == 0.0 else "mixed"
    pc = Counter(part.values())
    n_known = sum(v for k, v in pc.items() if k != "unknown")
    print(f"  bare-prompt compliance partition (n=3): {dict(pc)}  (known={n_known})")
    print(f"    always-comply: {pc['always-comply']}  ({pc['always-comply']/n_known:.1%})")
    print(f"    always-violate: {pc['always-violate']} ({pc['always-violate']/n_known:.1%})")
    print(f"    mixed (1-2/3):  {pc['mixed']}  ({pc['mixed']/n_known:.1%})  <- the discriminating set")

    # Among always-violate-bare problems: does the CONSTRAINED prompt recover a compliant+passing soln?
    av = [pid for pid in probs if part[pid] == "always-violate"]
    print(f"\n  --- of the {len(av)} ALWAYS-VIOLATE-bare problems, what does the *constrained* prompt do? ---")
    recovered, recovered_pass, has_cons_data = 0, 0, 0
    rows_av = []
    for pid in av:
        c = cons.get(pid, [])
        if not c:
            rows_av.append((pid, "no constrained data", None, None)); continue
        has_cons_data += 1
        c_comp = rate(c, "compliant")
        c_compass = sum(1 for x in c if x["compliant"] and x["passed"]) / len(c)
        c_pass = rate(c, "passed")
        if c_comp > 0: recovered += 1
        if c_compass > 0: recovered_pass += 1
        b = bare.get(pid, [])
        b_pass = rate(b, "passed")
        rows_av.append((pid, f"compl={c_comp:.2f} compl&pass={c_compass:.2f} pass={c_pass:.2f}", b_pass, c_pass))
    for pid, s, bp, cp in rows_av:
        delta = f"  (bare_pass={bp:.2f} -> cons_pass={cp:.2f})" if bp is not None else ""
        print(f"    {pid:<18} {s}{delta}")
    if has_cons_data:
        print(f"\n  >> of {has_cons_data} always-violate-bare problems WITH constrained data:")
        print(f"     {recovered} ({recovered/has_cons_data:.0%}) produce >=1 COMPLIANT constrained gen")
        print(f"     {recovered_pass} ({recovered_pass/has_cons_data:.0%}) produce >=1 COMPLIANT-AND-PASSING constrained gen")
        print(f"     -> elicitation-recoverable fraction (n=3, weak lower bound): ~{recovered_pass/has_cons_data:.0%}")
        print(f"     -> the rest ({has_cons_data-recovered_pass}) = no loop-free passing soln found even when told to (intrinsic? or n=3 too small)")

    # ---------- the ~34-pt cost ----------
    print("\n  --- CONSTRAINT COST: bare-prompt pass rate vs constrained pass rate ---")
    common = [pid for pid in probs if bare.get(pid) and cons.get(pid)]
    bp_all = [rate(bare[pid], "passed") for pid in common]
    cp_all = [rate(cons[pid], "passed") for pid in common]
    print(f"  on {len(common)} problems with both conditions:")
    print(f"    mean bare-prompt pass rate:        {st.mean(bp_all):.3f}")
    print(f"    mean constrained pass rate:        {st.mean(cp_all):.3f}")
    print(f"    => raw drop:                       {st.mean(bp_all)-st.mean(cp_all):+.3f}  ({(st.mean(bp_all)-st.mean(cp_all))*100:+.0f} pts)")
    # split by partition
    for tag in ("always-comply", "always-violate", "mixed"):
        sub = [pid for pid in common if part.get(pid) == tag]
        if not sub: continue
        bps = [rate(bare[pid], "passed") for pid in sub]; cps = [rate(cons[pid], "passed") for pid in sub]
        print(f"    [{tag:<15}] n={len(sub):2d}  bare_pass={st.mean(bps):.2f} cons_pass={st.mean(cps):.2f}  drop={st.mean(bps)-st.mean(cps):+.2f}")
    # "constraint kills correctness" rate
    passable_bare = [pid for pid in common if rate(bare[pid], "passed") and rate(bare[pid], "passed") > 0]
    killed = [pid for pid in passable_bare if not (rate(cons[pid], "passed") and rate(cons[pid], "passed") > 0)]
    print(f"  of {len(passable_bare)} problems passable bare-prompt, {len(killed)} ({len(killed)/len(passable_bare):.0%}) become un-passable when constrained")
    print(f"    (i.e. the constraint is 'free' on {len(passable_bare)-len(killed)}/{len(passable_bare)} passable problems, 'fatal' on {len(killed)})")

    # ---------- B. solution-structure classification ----------
    print("\n" + "=" * 76)
    print("B. SOLUTION-STRUCTURE CLASSIFICATION")
    print("=" * 76)
    # what do BARE-prompt gens look like, by partition?
    print("  bare-prompt generations, structure tags by partition:")
    for tag in ("always-comply", "always-violate", "mixed"):
        sub = [pid for pid in probs if part.get(pid) == tag]
        structs = Counter()
        for pid in sub:
            for x in bare.get(pid, []): structs[x["struct"]] += 1
        print(f"    [{tag}] ({len(sub)} problems): " + ", ".join(f"{k}×{v}" for k, v in structs.most_common()))
    # what do compliant CONSTRAINED gens use? (i.e. when forced, what loop-free trick?)
    cons_comp_structs = Counter()
    for pid in probs:
        for x in cons.get(pid, []):
            if x["compliant"]: cons_comp_structs[x["struct"]] += 1
    print(f"\n  CONSTRAINED compliant generations — the loop-free technique the model uses when forced:")
    for k, v in cons_comp_structs.most_common(): print(f"    {k}: {v}")
    # per-problem: for always-violate-bare problems that the model CAN do loop-free when forced,
    # what's the technique? (these are 'the capability is there, just not the default')
    print(f"\n  always-violate-bare problems where the constrained prompt recovers a compliant solution:")
    for pid in av:
        cc = [x for x in cons.get(pid, []) if x["compliant"]]
        if cc:
            print(f"    {pid}: technique = {Counter(x['struct'] for x in cc).most_common(1)[0][0]}  ({sum(1 for x in cc if x['passed'])}/{len(cc)} of the compliant ones pass)")

    # ---------- C. higher-n determinism check on the 9-problem high-sample base eval ----------
    print("\n" + "=" * 76)
    print("C. HIGHER-n DETERMINISM CHECK (base, 9 LCB problems, ~14 samples each)")
    print("=" * 76)
    hn_rows = load(HIGHN_CSV)
    hn = per_problem(hn_rows, HIGHN_SRC, "unconstrained", "none")
    for pid in sorted(hn):
        xs = hn[pid]; cr = rate(xs, "compliant"); pr = rate(xs, "passed")
        bar = "#" * round(cr * 20)
        print(f"  {pid:<18} n={len(xs):2d}  compliance={cr:.2f} |{bar:<20}|  pass={pr:.2f}  structs={dict(Counter(x['struct'] for x in xs))}")
    crs = [rate(xs, "compliant") for xs in hn.values()]
    near01 = sum(1 for c in crs if c <= 0.1 or c >= 0.9)
    print(f"  => {near01}/{len(crs)} problems are within 0.1 of 0.0 or 1.0 (near-deterministic at n~14)")

    # ---------- D. brief: expanded-set partition (base ~200 problems) ----------
    print("\n" + "=" * 76)
    print("D. EXPANDED-SET PARTITION — base (≈identity-adapter) on ~200 HumanEval+MBPP problems")
    print("=" * 76)
    ex_rows = load(EXPANDED_CSV)
    ex = per_problem(ex_rows, EXPANDED_SRC, "unconstrained", "none")
    if not ex:
        # maybe the constraint label differs; try any
        ex = defaultdict(list)
        for r in ex_rows:
            if r["condition"] != "unconstrained" or r.get("gen_error"): continue
            code = read_code(EXPANDED_SRC, r); rc = recheck_row(code)
            if rc: ex[r["problem_id"]].append(dict(compliant=rc["compliant"], passed=int(r.get("test_passed") or 0), struct=classify_solution(code), n_chars=rc["n_chars"]))
    crs = {pid: rate(xs, "compliant") for pid, xs in ex.items() if xs}
    n = len(crs)
    z0 = sum(1 for c in crs.values() if c == 0); z1 = sum(1 for c in crs.values() if c == 1)
    zm = sum(1 for c in crs.values() if 0 < c < 1); zd = sum(1 for c in crs.values() if 0.3 <= c <= 0.7)
    print(f"  {n} problems, ~{Counter(len(xs) for xs in ex.values()).most_common(1)[0][0]} samples each")
    print(f"    always-violate (0):  {z0} ({z0/n:.1%})" if n else "")
    print(f"    always-comply (1):   {z1} ({z1/n:.1%})" if n else "")
    print(f"    mixed (0<c<1):       {zm} ({zm/n:.1%})  | in [0.3,0.7]: {zd} ({zd/n:.1%})" if n else "")
    # what do the always-comply ones use?
    ac_structs = Counter()
    for pid, c in crs.items():
        if c == 1:
            for x in ex[pid]:
                if x["compliant"]: ac_structs[x["struct"]] += 1
    print(f"  always-comply problems — loop-free technique used: " + ", ".join(f"{k}×{v}" for k, v in ac_structs.most_common(8)))


if __name__ == "__main__":
    main()
