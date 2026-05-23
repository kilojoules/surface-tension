"""Re-derive AST compliance for a sweep_local eval from the saved source files.

The CSV `compliant` column is unreliable (it tracks code_extracted; check_no_loops("")
is vacuously True). This re-parses each generation's .py source and runs the real check.

Usage:
    python recheck_eval.py eval_TAG.csv [eval_TAG2.csv ...]
(assumes the sources dir is `sources_<basename-without-.csv>` next to each csv, e.g.
 eval_foo.csv -> sources_eval_foo/)
"""
import csv, json, os, sys
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ast_checks import check_no_loops, check_no_recursion, _try_parse  # noqa


def sources_dir_for(csv_path):
    d = os.path.dirname(csv_path)
    base = os.path.basename(csv_path)
    if base.endswith(".csv"):
        base = base[:-4]
    # eval_foo.csv  ->  sources_eval_foo/   (sweep_local's --source-dir convention)
    return os.path.join(d, f"sources_{base}")


def code_of(src_dir, row):
    pid = row["problem_id"].replace("/", "_")
    cstr = row.get("constraint") or "none"
    p = os.path.join(src_dir, f"{pid}__{cstr}__{row['condition']}__s{row['sample_idx']}.py")
    return open(p, errors="replace").read().strip() if os.path.exists(p) else None


def compliant(code):
    if code is None:
        return None
    t = _try_parse(code)
    if t is None or len(code) < 8 or not getattr(t, "body", []):
        return False
    return bool(check_no_loops(code) and check_no_recursion(code))


def main():
    csvs = sys.argv[1:]
    if not csvs:
        print("usage: recheck_eval.py eval_TAG.csv ...", file=sys.stderr); sys.exit(1)
    summary = []
    for csv_path in csvs:
        if not os.path.exists(csv_path):
            print(f"=== {csv_path}: MISSING ==="); continue
        src_dir = sources_dir_for(csv_path)
        rows = list(csv.DictReader(open(csv_path)))
        byp = defaultdict(list)  # problem_id -> [(compliant, passed)]
        n_missing_src = 0
        for r in rows:
            if r["condition"] != "unconstrained" or r.get("gen_error"):
                continue
            code = code_of(src_dir, r)
            if code is None:
                n_missing_src += 1; continue
            byp[r["problem_id"]].append((compliant(code), int(r.get("test_passed") or 0)))
        n_probs = len(byp)
        n_gens = sum(len(v) for v in byp.values())
        n_comp = sum(1 for v in byp.values() for c, _ in v if c)
        n_comp_pass = sum(1 for v in byp.values() for c, p in v if c and p)
        n_pass = sum(1 for v in byp.values() for _, p in v if p)
        rates = {p: sum(1 for c, _ in v if c) / len(v) for p, v in byp.items()}
        probs_any_comp = sum(1 for r_ in rates.values() if r_ > 0)
        probs_all_comp = sum(1 for r_ in rates.values() if r_ == 1.0)
        probs_mixed = sum(1 for r_ in rates.values() if 0 < r_ < 1)
        probs_zone = sum(1 for r_ in rates.values() if 0.3 <= r_ <= 0.7)
        print(f"=== {os.path.basename(csv_path)} ===")
        print(f"  {n_probs} problems, {n_gens} bare-prompt generations (samples/prob: {dict(Counter(len(v) for v in byp.values()))}); {n_missing_src} missing-source rows skipped")
        print(f"  compliance (re-checked):   {n_comp}/{n_gens}  = {n_comp/n_gens:.3f}" if n_gens else "  (no gens)")
        print(f"  compliance AND passing:    {n_comp_pass}/{n_gens}  = {n_comp_pass/n_gens:.3f}" if n_gens else "")
        print(f"  pass rate (any code):      {n_pass}/{n_gens}  = {n_pass/n_gens:.3f}" if n_gens else "")
        print(f"  problems: {probs_all_comp} all-compliant, {probs_zone} in [0.3,0.7], {probs_mixed} strictly-mixed, {probs_any_comp} with >=1 compliant gen, {n_probs-probs_any_comp} all-non-compliant")
        if rates:
            print(f"  per-problem compliance rates: " + ", ".join(f"{p.split('/')[-1]}={v:.2f}" for p, v in sorted(rates.items())))
        print()
        summary.append(dict(csv=os.path.basename(csv_path), n_problems=n_probs, n_gens=n_gens,
                            compliance=round(n_comp / n_gens, 4) if n_gens else None,
                            compliance_and_pass=round(n_comp_pass / n_gens, 4) if n_gens else None,
                            pass_rate=round(n_pass / n_gens, 4) if n_gens else None,
                            probs_all_compliant=probs_all_comp, probs_in_zone=probs_zone,
                            probs_mixed=probs_mixed, probs_any_compliant=probs_any_comp,
                            per_problem_rate={p: round(v, 3) for p, v in rates.items()}))
    out_json = os.path.join(os.path.dirname(csvs[0]) or ".", "recheck_summary.json")
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"(wrote {out_json})")


if __name__ == "__main__":
    main()
