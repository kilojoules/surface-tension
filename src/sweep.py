"""Run the constraint × condition × problem × sample sweep, write per-row CSV.

Resumable: skips rows already present in the output CSV (matched by
problem_id, constraint, condition, sample_idx).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ast_checks import CHECKS, CONSTRAINT_INSTRUCTIONS, check_no_helpers
from evaluator import evaluate, evaluate_stdin
from loaders import load_problems_jsonl

# Runner indirection: RUNNER=http selects the OpenAI-compatible HTTP runner (vLLM etc).
# Default stays on the Gemini CLI for backwards compatibility with v1/v2 setups.
if os.environ.get("RUNNER", "gemini") == "http":
    from http_runner import generate
else:
    from gemini_runner import generate


CSV_FIELDS = [
    "problem_id", "benchmark", "entry_point",
    "constraint", "condition", "sample_idx",
    "model", "elapsed_s", "gen_error",
    "code_extracted", "code_chars",
    "parses", "has_entry", "compliant",
    "test_passed", "test_timed_out", "test_error",
    "prompt_chars", "raw_chars",
]


def build_prompt(problem: Dict[str, Any], constraint: Optional[str]) -> str:
    base = problem["prompt"]
    if constraint is None:
        instruction = (
            "Provide a complete, working solution. Return only Python source "
            "inside a single ```python code block."
        )
    else:
        constraint_text = CONSTRAINT_INSTRUCTIONS[constraint]
        instruction = (
            f"Constraint: {constraint_text}\n\n"
            "Provide a complete, working solution that satisfies this constraint. "
            "Return only Python source inside a single ```python code block."
        )
    return f"{base}\n\n{instruction}"


@dataclass
class Row:
    problem_id: str
    benchmark: str
    entry_point: str
    constraint: str
    condition: str
    sample_idx: int
    model: str = ""
    elapsed_s: float = 0.0
    gen_error: str = ""
    code_extracted: int = 0
    code_chars: int = 0
    parses: int = 0
    has_entry: int = 0
    compliant: int = 0
    test_passed: int = 0
    test_timed_out: int = 0
    test_error: str = ""
    prompt_chars: int = 0
    raw_chars: int = 0


def run_one(problem: Dict[str, Any], constraint: str, sample_idx: int, source_dir: str) -> Row:
    """One generation + eval. constraint='none' for unconstrained.

    Dispatches on problem['mode']: 'function' (HE/MBPP) or 'stdin' (LCB).
    """
    is_constrained = constraint != "none"
    actual_constraint = constraint if is_constrained else None
    condition = "constrained" if is_constrained else "unconstrained"
    mode = problem.get("mode", "function")
    entry_point = problem.get("entry_point") or ""

    row = Row(
        problem_id=problem["id"],
        benchmark=problem["benchmark"],
        entry_point=entry_point,
        constraint=constraint,
        condition=condition,
        sample_idx=sample_idx,
    )

    prompt = build_prompt(problem, actual_constraint)
    row.prompt_chars = len(prompt)

    g = generate(prompt)
    row.model = g.model
    row.elapsed_s = round(g.elapsed_s, 2)
    row.raw_chars = len(g.raw)
    if g.error:
        row.gen_error = g.error
    if not g.code:
        _save_source(source_dir, row, g.raw)
        return row

    code = g.code
    row.code_extracted = 1
    row.code_chars = len(code)

    import ast
    try:
        tree = ast.parse(code)
        row.parses = 1
    except SyntaxError:
        _save_source(source_dir, row, code)
        return row

    # Entry-point check (function mode only; stdin programs don't define an entry function)
    if mode == "function":
        has_entry = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point
            for n in ast.walk(tree)
        )
        row.has_entry = 1 if has_entry else 0
        if not has_entry:
            _save_source(source_dir, row, code)
            return row
    else:
        row.has_entry = 1  # n/a for stdin mode; treat as satisfied for filter

    if code.count("\n") > 200:
        row.gen_error = (row.gen_error + ";too_long").strip(";")
        _save_source(source_dir, row, code)
        return row

    if is_constrained:
        if constraint == "no_helpers":
            # entry_point is None for stdin: no_helpers still applies (≤1 module-level def)
            row.compliant = 1 if check_no_helpers(code, entry_point or None) else 0
        else:
            row.compliant = 1 if CHECKS[constraint](code) else 0
    else:
        row.compliant = 1

    if mode == "stdin":
        res = evaluate_stdin(code, problem["stdin_tests"], timeout_s=10.0)
    else:
        res = evaluate(code, entry_point, problem["test_runner"], timeout_s=10.0)

    row.test_passed = 1 if res.passed else 0
    row.test_timed_out = 1 if res.timed_out else 0
    if res.error:
        row.test_error = res.error[:200]

    _save_source(source_dir, row, code)
    return row


def _save_source(source_dir: str, row: Row, code: str) -> None:
    os.makedirs(source_dir, exist_ok=True)
    safe_id = row.problem_id.replace("/", "_")
    fname = f"{safe_id}__{row.constraint}__{row.condition}__s{row.sample_idx}.py"
    with open(os.path.join(source_dir, fname), "w") as f:
        f.write(code)


def _existing_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            keys.add((r["problem_id"], r["constraint"], r["condition"], int(r["sample_idx"])))
    return keys


def run_sweep(
    problems: List[Dict[str, Any]],
    constraints: List[str],
    n_samples: int,
    csv_path: str,
    source_dir: str,
    max_workers: int = 8,
    log_every: int = 25,
) -> None:
    """Run the sweep.

    Tasks:
      - Unconstrained: one per (problem, sample_idx). Shared across constraints
        at aggregation time. Stored with constraint='none'.
      - Constrained: one per (problem, constraint, sample_idx).
    """
    tasks = []  # list of (problem, constraint_for_run, condition, sample_idx)
    for p in problems:
        for s in range(n_samples):
            tasks.append((p, "none", "unconstrained", s))
        for c in constraints:
            for s in range(n_samples):
                tasks.append((p, c, "constrained", s))

    existing = _existing_keys(csv_path)
    pending = [t for t in tasks if (t[0]["id"], t[1], t[2], t[3]) not in existing]

    print(f"Total tasks: {len(tasks)}; already done: {len(tasks) - len(pending)}; pending: {len(pending)}")

    new_file = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()

    started = time.time()
    completed = 0

    def task_fn(t):
        problem, constraint_for_run, condition, sample_idx = t
        return run_one(problem, constraint_for_run, sample_idx, source_dir)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(task_fn, t): t for t in pending}
            for fut in as_completed(futures):
                row = fut.result()
                writer.writerow({k: getattr(row, k) for k in CSV_FIELDS})
                completed += 1
                if completed % log_every == 0 or completed == len(pending):
                    elapsed = time.time() - started
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - completed) / rate if rate > 0 else float("inf")
                    print(f"  [{completed}/{len(pending)}] rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)
                    f.flush()
    finally:
        f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="../data/problems.jsonl")
    ap.add_argument("--csv", default="../results/raw/pilot_raw.csv")
    ap.add_argument("--source-dir", default="../results/raw/sources")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-problems", type=int, default=0, help="0 = all")
    ap.add_argument("--constraints", nargs="+", default=list(CONSTRAINT_INSTRUCTIONS.keys()))
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    problems_path = os.path.join(base, args.problems) if not os.path.isabs(args.problems) else args.problems
    csv_path = os.path.join(base, args.csv) if not os.path.isabs(args.csv) else args.csv
    source_dir = os.path.join(base, args.source_dir) if not os.path.isabs(args.source_dir) else args.source_dir
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    problems = load_problems_jsonl(problems_path)
    if args.limit_problems:
        problems = problems[: args.limit_problems]
    print(f"Loaded {len(problems)} problems; constraints={args.constraints}; samples={args.n_samples}; workers={args.workers}")
    run_sweep(problems, args.constraints, args.n_samples, csv_path, source_dir, max_workers=args.workers)


if __name__ == "__main__":
    main()
