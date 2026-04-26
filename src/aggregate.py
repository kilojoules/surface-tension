"""Aggregate pilot_raw.csv into pilot_summary.csv and pilot_summary.md."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _f(row: Dict[str, str], key: str) -> float:
    v = row.get(key, "")
    if v == "":
        return 0.0
    return float(v)


def _i(row: Dict[str, str], key: str) -> int:
    v = row.get(key, "")
    if v == "":
        return 0
    return int(v)


def _kept(row: Dict[str, str]) -> bool:
    """Spec step 4: drop solutions that don't parse, lack the entry point, or exceed 200 lines."""
    if _i(row, "parses") != 1:
        return False
    if _i(row, "has_entry") != 1:
        return False
    if "too_long" in (row.get("gen_error") or ""):
        return False
    return True


def aggregate(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Compute (benchmark, constraint) summary rows, sorted by binding_score desc.

    Per spec step 4, solutions that don't parse / lack entry point / are too long are dropped
    before computing rates. Rejection rates are reported separately.
    """
    benchmarks = sorted(set(r["benchmark"] for r in rows))
    constraints = sorted(set(r["constraint"] for r in rows if r["constraint"] != "none"))

    summary = []
    for bench in benchmarks:
        unc_all = [r for r in rows if r["condition"] == "unconstrained" and r["benchmark"] == bench]
        unc_kept = [r for r in unc_all if _kept(r)]
        if unc_kept:
            pass_unconstrained = sum(_i(r, "test_passed") for r in unc_kept) / len(unc_kept)
        else:
            pass_unconstrained = 0.0
        unc_reject = 1.0 - (len(unc_kept) / len(unc_all)) if unc_all else 0.0

        for cons in constraints:
            con_all = [
                r for r in rows
                if r["benchmark"] == bench and r["constraint"] == cons and r["condition"] == "constrained"
            ]
            if not con_all:
                continue
            con_kept = [r for r in con_all if _kept(r)]
            con_reject = 1.0 - (len(con_kept) / len(con_all)) if con_all else 0.0
            n_con = len(con_kept)
            n_compliant = sum(_i(r, "compliant") for r in con_kept)
            n_compliant_passed = sum(1 for r in con_kept if _i(r, "compliant") == 1 and _i(r, "test_passed") == 1)
            compliance_rate = (n_compliant / n_con) if n_con > 0 else 0.0

            # Unbiased: count constrained sample as "passing" only if compliant AND test_passed.
            # No selection on compliance, so easier-problem leakage doesn't inflate the rate.
            pass_overall = (n_compliant_passed / n_con) if n_con > 0 else 0.0
            drop_overall = pass_unconstrained - pass_overall

            # Compliance-conditional (kept for backward-compat; biased upward when compliance < 1).
            compliant_cells = [r for r in con_kept if _i(r, "compliant") == 1]
            pass_among_compliant = (
                sum(_i(r, "test_passed") for r in compliant_cells) / len(compliant_cells)
                if compliant_cells else 0.0
            )
            drop_among_compliant = pass_unconstrained - pass_among_compliant
            binding_score_legacy = drop_among_compliant * compliance_rate

            summary.append({
                "benchmark": bench,
                "constraint": cons,
                "n_unconstrained_kept": len(unc_kept),
                "n_constrained_kept": n_con,
                "n_compliant": n_compliant,
                "unc_reject_rate": round(unc_reject, 4),
                "con_reject_rate": round(con_reject, 4),
                "pass_unconstrained": round(pass_unconstrained, 4),
                "pass_overall": round(pass_overall, 4),
                "drop_overall": round(drop_overall, 4),
                "compliance_rate": round(compliance_rate, 4),
                "pass_among_compliant": round(pass_among_compliant, 4),
                "drop_among_compliant": round(drop_among_compliant, 4),
                "binding_score_legacy": round(binding_score_legacy, 4),
            })

    summary.sort(key=lambda r: r["drop_overall"], reverse=True)
    return summary


def find_example(rows: List[Dict[str, str]], benchmark: str, constraint: str, sources_dir: str) -> Tuple[str, str, str] | None:
    """Pick (problem_id, constrained_code, unconstrained_code) for the top pair.

    Prefer a problem where the constrained solution is compliant AND the unconstrained one passed.
    """
    pid_to_unc = defaultdict(list)
    pid_to_con = defaultdict(list)
    for r in rows:
        if r["benchmark"] != benchmark:
            continue
        if r["condition"] == "unconstrained":
            pid_to_unc[r["problem_id"]].append(r)
        elif r["constraint"] == constraint and r["condition"] == "constrained":
            pid_to_con[r["problem_id"]].append(r)

    # Prefer: constrained compliant+failed, unconstrained passed (most informative for binding signal)
    candidates: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    for pid in pid_to_con:
        cons = [c for c in pid_to_con[pid] if _i(c, "compliant") == 1]
        if not cons:
            continue
        uncs = [u for u in pid_to_unc.get(pid, []) if _i(u, "parses") == 1 and _i(u, "code_chars") > 0]
        if not uncs:
            continue
        cons_failed = [c for c in cons if _i(c, "test_passed") == 0]
        unc_passed = [u for u in uncs if _i(u, "test_passed") == 1]
        if cons_failed and unc_passed:
            candidates.append((cons_failed[0], unc_passed[0]))

    if not candidates:
        # Fall back to any compliant constrained + any (parsing) unconstrained for the same problem
        for pid in pid_to_con:
            cons = [c for c in pid_to_con[pid] if _i(c, "compliant") == 1]
            uncs = [u for u in pid_to_unc.get(pid, []) if _i(u, "parses") == 1 and _i(u, "code_chars") > 0]
            if cons and uncs:
                candidates.append((cons[0], uncs[0]))
                break

    if not candidates:
        return None

    cons_row, unc_row = candidates[0]
    cons_path = _source_path(sources_dir, cons_row)
    unc_path = _source_path(sources_dir, unc_row)
    cons_code = _read_or(cons_path, "(missing source)")
    unc_code = _read_or(unc_path, "(missing source)")
    return cons_row["problem_id"], cons_code, unc_code


def _source_path(sources_dir: str, row: Dict[str, str]) -> str:
    safe_id = row["problem_id"].replace("/", "_")
    fname = f"{safe_id}__{row['constraint']}__{row['condition']}__s{row['sample_idx']}.py"
    return os.path.join(sources_dir, fname)


def _read_or(path: str, fallback: str) -> str:
    try:
        with open(path) as f:
            return f.read().rstrip()
    except FileNotFoundError:
        return fallback


def write_summary_csv(summary: List[Dict[str, Any]], path: str) -> None:
    if not summary:
        with open(path, "w") as f:
            f.write("# no rows\n")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def write_summary_md(
    summary: List[Dict[str, Any]],
    rows: List[Dict[str, str]],
    sources_dir: str,
    md_path: str,
    raw_csv_path: str,
) -> None:
    lines: List[str] = []
    lines.append("# Pilot Study: Binding Coding Constraints")
    lines.append("")
    lines.append("**Goal:** find (benchmark, constraint) pairs with high `binding_score = performance_drop × compliance_rate`.")
    lines.append("")
    lines.append(f"Model: `gemini-2.5-flash` (spec called for 2.0-flash, but only 2.5-flash is available via the CLI as of generation date).")
    lines.append(f"Raw data: `{os.path.basename(raw_csv_path)}` ({len(rows)} rows).")
    lines.append("")
    # Sufficiency: each (benchmark, constraint) pair needs enough kept samples to have signal.
    # Threshold: at least 30 kept constrained samples and 30 kept unconstrained samples for the benchmark.
    MIN_N = 30
    insufficient = [r for r in summary if r["n_unconstrained_kept"] < MIN_N or r["n_constrained_kept"] < MIN_N]
    sufficient = [r for r in summary if r not in insufficient]

    lines.append("## Data sufficiency")
    if insufficient:
        lines.append(f"**{len(insufficient)} of {len(summary)} pairs are undersampled** (< {MIN_N} kept samples in either condition):")
        for r in insufficient:
            lines.append(f"- `{r['benchmark']}` × `{r['constraint']}`: n_unc={r['n_unconstrained_kept']}, n_con={r['n_constrained_kept']} — likely from API quota / failures, results below are unreliable")
        lines.append("")
        lines.append("Re-run after quota resets to get coverage on all benchmarks. See `retry_pilot.sh`.")
        lines.append("")

    lines.append("## Success-criteria check")
    lines.append("Pilot is informative if some pair has `drop_overall ≥ 0.15` (unbiased binding pressure).")
    lines.append("")
    if not sufficient:
        lines.append("**Cannot evaluate**: no pair has sufficient sample size. Re-run is required.")
    elif sufficient:
        winners = [r for r in sufficient if r["drop_overall"] >= 0.15]
        if winners:
            lines.append(f"**{len(winners)} pair(s) cleared the threshold:**")
            for w in winners:
                lines.append(f"- `{w['benchmark']}` × `{w['constraint']}`: drop_overall={w['drop_overall']}, compliance={w['compliance_rate']}")
        else:
            lines.append(f"**Among the {len(sufficient)} pair(s) with sufficient data, no pair cleared the drop_overall ≥ 0.15 threshold.**")
            if insufficient:
                lines.append("Note: the undersampled pairs above could not be evaluated — this is *not* the spec's 'no pair binds' finding.")
            else:
                lines.append("That is itself the finding for the pilot.")
    lines.append("")
    lines.append("## Full ranking")
    lines.append("")
    if summary:
        lines.append("Sorted by `drop_overall` (unbiased — counts non-compliant constrained samples as failures).")
        lines.append("`drop_among_compliant` is the spec metric, kept for reference but biased upward when compliance < 1.")
        lines.append("")
        lines.append("| benchmark | constraint | n_unc | n_con | n_compl | rej_unc | rej_con | pass_unc | **pass_overall** | **drop_overall** | compliance | pass_compl | drop_compl |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in summary:
            lines.append(
                f"| {r['benchmark']} | {r['constraint']} | {r['n_unconstrained_kept']} | {r['n_constrained_kept']} | "
                f"{r['n_compliant']} | {r['unc_reject_rate']} | {r['con_reject_rate']} | "
                f"{r['pass_unconstrained']} | **{r['pass_overall']}** | **{r['drop_overall']}** | "
                f"{r['compliance_rate']} | {r['pass_among_compliant']} | {r['drop_among_compliant']} |"
            )
    lines.append("")

    lines.append("## Top 3 examples")
    lines.append("")
    for i, r in enumerate(summary[:3], 1):
        ex = find_example(rows, r["benchmark"], r["constraint"], sources_dir)
        lines.append(f"### {i}. `{r['benchmark']}` × `{r['constraint']}` (drop_overall={r['drop_overall']})")
        lines.append("")
        lines.append(f"- pass_unconstrained: **{r['pass_unconstrained']}**, pass_overall: **{r['pass_overall']}**")
        lines.append(f"- compliance_rate: **{r['compliance_rate']}**, drop_among_compliant (biased): {r['drop_among_compliant']}")
        lines.append("")
        if ex is None:
            lines.append("_(no compliant constrained sample with matching unconstrained sample available)_")
        else:
            pid, cons_code, unc_code = ex
            lines.append(f"**Example problem:** `{pid}`")
            lines.append("")
            lines.append("**Unconstrained solution:**")
            lines.append("```python")
            lines.append(unc_code)
            lines.append("```")
            lines.append("")
            lines.append("**Constrained-compliant solution:**")
            lines.append("```python")
            lines.append(cons_code)
            lines.append("```")
            lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../results/raw/pilot_raw.csv")
    ap.add_argument("--summary-csv", default="../results/pilot_summary.csv")
    ap.add_argument("--summary-md", default="../results/pilot_summary.md")
    ap.add_argument("--sources-dir", default="../results/raw/sources")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, args.csv) if not os.path.isabs(args.csv) else args.csv
    sum_csv = os.path.join(base, args.summary_csv) if not os.path.isabs(args.summary_csv) else args.summary_csv
    sum_md = os.path.join(base, args.summary_md) if not os.path.isabs(args.summary_md) else args.summary_md
    sources_dir = os.path.join(base, args.sources_dir) if not os.path.isabs(args.sources_dir) else args.sources_dir

    rows = list(csv.DictReader(open(csv_path)))
    print(f"Loaded {len(rows)} raw rows.")
    summary = aggregate(rows)
    print(f"{len(summary)} (benchmark, constraint) pairs.")
    write_summary_csv(summary, sum_csv)
    write_summary_md(summary, rows, sources_dir, sum_md, csv_path)
    print(f"Wrote: {sum_csv}")
    print(f"Wrote: {sum_md}")


if __name__ == "__main__":
    main()
