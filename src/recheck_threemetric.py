"""Re-analysis of existing eval CSVs under the three-metric framing.

For each (model, problem-set), report:
  - compliance: fraction of attempts with AST-loop-free code, regardless of pass
  - compliance ∧ pass: AST-loop-free AND test-passing
  - cheating: non-compliant ∧ pass (loop-using AND correct — the failure mode to minimize)
  - substantive-compliance: compliant AND code parses AND has entry point AND code_chars > 100
    (filters out empty / trivial gens that would game the AST check)

All rates use the "all-attempts" denominator (gen_errors count as non-compliant),
which is the fair comparison across token budgets.
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ast_checks import check_no_loops_no_recursion as compliant_check


def src_path(sources_dir, row):
    pid = row["problem_id"].replace("/", "_")
    return os.path.join(sources_dir, f"{pid}__{row['constraint']}__{row['condition']}__s{row['sample_idx']}.py")


def is_compliant_from_source(sources_dir, row):
    """AST-recheck from saved source file. Returns None if missing."""
    p = src_path(sources_dir, row)
    if not os.path.exists(p):
        return None
    code = open(p, errors="replace").read().strip()
    if not code:
        return False
    return bool(compliant_check(code))


def is_substantive(row):
    """Heuristic for 'genuine code attempt': parses, has entry, length > 100 chars."""
    try:
        parses = int(row.get("parses") or 0) == 1
        has_entry = int(row.get("has_entry") or 0) == 1
        chars = int(row.get("code_chars") or 0)
    except ValueError:
        return False
    return parses and has_entry and chars > 100


def analyze(csv_path, sources_dir, label):
    rows = list(csv.DictReader(open(csv_path)))
    # restrict to unconstrained / bare-prompt eval rows
    rows = [r for r in rows if r.get("condition", "unconstrained") == "unconstrained"]
    n = len(rows)
    if n == 0:
        return None

    comp = comp_pass = comp_substantive = cheat = no_code = 0
    for r in rows:
        gen_err = bool(r.get("gen_error") or "")
        passed = int(r.get("test_passed") or 0) == 1
        if gen_err:
            no_code += 1
            continue
        c = is_compliant_from_source(sources_dir, r)
        if c is None:
            continue
        if c:
            comp += 1
            if is_substantive(r):
                comp_substantive += 1
            if passed:
                comp_pass += 1
        else:
            if passed:
                cheat += 1

    return dict(
        label=label,
        n=n,
        compliance=comp / n,
        compliance_pass=comp_pass / n,
        compliance_substantive=comp_substantive / n,
        cheating=cheat / n,
        no_code_rate=no_code / n,
        counts=dict(compliance=comp, compliance_pass=comp_pass, compliance_substantive=comp_substantive, cheating=cheat, no_code=no_code, total=n),
    )


def print_table(results):
    print(f"{'eval':<32}  {'n':>4}  {'compl':>7}  {'cmp∧pass':>9}  {'cmp∧subst':>10}  {'cheat':>7}  {'no_code':>8}")
    print("-" * 90)
    for r in results:
        if r is None: continue
        print(f"{r['label']:<32}  {r['n']:>4}  {r['compliance']:>7.3f}  {r['compliance_pass']:>9.3f}  {r['compliance_substantive']:>10.3f}  {r['cheating']:>7.3f}  {r['no_code_rate']:>8.3f}")
    print()
    print("Notes:")
    print("  - All denominators = total attempts (truncations / no-code-block count as non-compliant).")
    print("  - 'cmp∧subst' = compliant + code parses + has entry point + chars > 100 (genuine attempts).")
    print("  - 'cheat' = non-compliant ∧ passes tests (the failure mode the user wants to minimize).")
    print("  - 'no_code' = gen_error rate (truncation, no fenced code block, etc.).")


def main():
    ROOT = "/Users/julianquick/portfolio_copy/surface_tension"
    R = lambda *p: os.path.join(ROOT, "vast_logs", *p)
    runs = [
        # (label, csv, sources_dir)
        ("rationale-SFT  val",          R("q5an7w9zzx6u11/st/results/raw/eval_rationale_val.csv"),
                                         R("q5an7w9zzx6u11/st/results/raw/sources_eval_rationale_val")),
        ("rationale-SFT  clean",        R("q5an7w9zzx6u11/st/results/raw/eval_rationale_clean.csv"),
                                         R("q5an7w9zzx6u11/st/results/raw/sources_eval_rationale_clean")),
        ("vanilla-SFT(3072) clean",     R("hwdpuli982rqz3/st/results/raw/eval_clean17_final_3072.csv"),
                                         R("hwdpuli982rqz3/st/results/raw/sources_eval_clean17_final_3072")),
        ("vanilla-SFT(1024) val",       R("obdxh2xixhiihs/st/results/raw/eval_valset_final.csv"),
                                         R("obdxh2xixhiihs/st/results/raw/sources_eval_valset_final")),
        ("vanilla-SFT(1024) clean",     R("kyqh9q3ebknz88/st/results/raw/eval_clean17_final.csv"),
                                         R("kyqh9q3ebknz88/st/results/raw/sources_eval_clean17_final")),
        ("-bestval  clean",             R("36662304/st/results/raw/eval_clean17_bestval.csv"),
                                         R("36662304/st/results/raw/sources_eval_clean17_bestval")),
    ]
    results = []
    for label, csvp, srcd in runs:
        if not os.path.exists(csvp):
            print(f"[skip] {label}: csv missing", file=sys.stderr)
            continue
        results.append(analyze(csvp, srcd, label))
    print_table(results)


if __name__ == "__main__":
    main()
