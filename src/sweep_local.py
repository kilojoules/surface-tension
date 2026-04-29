"""Local-process sweep — uses transformers.generate(), no vLLM, no HTTP indirection.

Loads the base model (4-bit) once, optionally applies a LoRA adapter, then iterates
problems × constraints × samples sequentially. Single-process; generation is naturally
sequential on one GPU anyway.

Same row schema as the original sweep.py — drops in to aggregate.py unchanged.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ast_checks import CHECKS, CONSTRAINT_INSTRUCTIONS, check_no_helpers
from evaluator import evaluate, evaluate_stdin
from loaders import load_problems_jsonl
from model_utils import generate_text, load_model


CSV_FIELDS = [
    "problem_id", "benchmark", "entry_point",
    "constraint", "condition", "sample_idx",
    "model", "elapsed_s", "gen_error",
    "code_extracted", "code_chars",
    "parses", "has_entry", "compliant",
    "test_passed", "test_timed_out", "test_error",
    "prompt_chars", "raw_chars",
]

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(raw: str) -> Optional[str]:
    blocks = _CODE_BLOCK_RE.findall(raw)
    if blocks:
        return max(blocks, key=len).strip()
    s = raw.strip()
    return s if s.startswith(("def ", "import ", "from ")) else None


def build_prompt(problem: Dict[str, Any], constraint: Optional[str]) -> str:
    base = problem["prompt"]
    if constraint is None:
        instruction = ("Provide a complete, working solution. Return only Python source "
                       "inside a single ```python code block.")
    else:
        instruction = (f"Constraint: {CONSTRAINT_INSTRUCTIONS[constraint]}\n\n"
                       "Provide a complete, working solution that satisfies this constraint. "
                       "Return only Python source inside a single ```python code block.")
    return f"{base}\n\n{instruction}"


@dataclass
class Row:
    problem_id: str = ""
    benchmark: str = ""
    entry_point: str = ""
    constraint: str = ""
    condition: str = ""
    sample_idx: int = 0
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


def _save_source(source_dir: str, row: Row, code: str) -> None:
    os.makedirs(source_dir, exist_ok=True)
    safe = row.problem_id.replace("/", "_")
    fname = f"{safe}__{row.constraint}__{row.condition}__s{row.sample_idx}.py"
    with open(os.path.join(source_dir, fname), "w") as f:
        f.write(code)


def _existing_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    out = set()
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            out.add((r["problem_id"], r["constraint"], r["condition"], int(r["sample_idx"])))
    return out


def evaluate_one(problem: Dict[str, Any], constraint: str, sample_idx: int,
                 source_dir: str, model, tokenizer, model_name: str,
                 max_new_tokens: int, temperature: float) -> Row:
    is_constrained = constraint != "none"
    actual_constraint = constraint if is_constrained else None
    condition = "constrained" if is_constrained else "unconstrained"
    mode = problem.get("mode", "function")
    entry_point = problem.get("entry_point") or ""

    row = Row(
        problem_id=problem["id"], benchmark=problem["benchmark"],
        entry_point=entry_point, constraint=constraint,
        condition=condition, sample_idx=sample_idx, model=model_name,
    )

    prompt = build_prompt(problem, actual_constraint)
    row.prompt_chars = len(prompt)

    t0 = time.time()
    try:
        raw = generate_text(model, tokenizer, prompt,
                            max_new_tokens=max_new_tokens, temperature=temperature)
    except Exception as e:
        row.gen_error = f"{type(e).__name__}: {e}"[:200]
        row.elapsed_s = round(time.time() - t0, 2)
        _save_source(source_dir, row, "")
        return row
    row.elapsed_s = round(time.time() - t0, 2)
    row.raw_chars = len(raw)

    code = extract_code(raw)
    if not code:
        row.gen_error = "no_code_block_found"
        _save_source(source_dir, row, raw)
        return row
    row.code_extracted = 1
    row.code_chars = len(code)

    try:
        tree = ast.parse(code)
        row.parses = 1
    except SyntaxError:
        _save_source(source_dir, row, code)
        return row

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
        row.has_entry = 1

    if code.count("\n") > 200:
        row.gen_error = (row.gen_error + ";too_long").strip(";")
        _save_source(source_dir, row, code)
        return row

    if is_constrained:
        if constraint == "no_helpers":
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="../data/problems_lcb.jsonl")
    ap.add_argument("--csv", default="../results/raw/sweep_local_raw.csv")
    ap.add_argument("--source-dir", default="../results/raw/sources_local")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--limit-problems", type=int, default=0)
    ap.add_argument("--constraints", nargs="+", default=["no_loops_no_recursion"])
    ap.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "google/gemma-4-31B-it"))
    ap.add_argument("--adapter", default=os.environ.get("ADAPTER_PATH"))
    ap.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "1536")))
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    def absify(p):
        return p if os.path.isabs(p) else os.path.join(here, p)

    problems = load_problems_jsonl(absify(args.problems))
    if args.limit_problems:
        problems = problems[: args.limit_problems]
    csv_path = absify(args.csv)
    source_dir = absify(args.source_dir)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    print(f"loading model: base={args.base_model} adapter={args.adapter}")
    model, tokenizer = load_model(args.base_model, adapter_path=args.adapter)
    model.eval()

    model_label = args.base_model + (f"+{args.adapter}" if args.adapter else "")

    tasks = []
    for p in problems:
        for s in range(args.n_samples):
            tasks.append((p, "none", s))
        for c in args.constraints:
            for s in range(args.n_samples):
                tasks.append((p, c, s))

    existing = _existing_keys(csv_path)
    pending = [t for t in tasks
               if (t[0]["id"], t[1], "constrained" if t[1] != "none" else "unconstrained", t[2]) not in existing]
    print(f"total tasks {len(tasks)}; already done {len(tasks)-len(pending)}; pending {len(pending)}")

    new_file = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()

    started = time.time()
    completed = 0
    try:
        for problem, constraint, sample_idx in pending:
            row = evaluate_one(problem, constraint, sample_idx, source_dir,
                               model, tokenizer, model_label,
                               args.max_new_tokens, args.temperature)
            writer.writerow({k: getattr(row, k) for k in CSV_FIELDS})
            completed += 1
            if completed % 10 == 0:
                f.flush()
                rate = completed / max(1.0, time.time() - started)
                eta = (len(pending) - completed) / max(rate, 1e-9)
                print(f"  [{completed}/{len(pending)}] rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)
    finally:
        f.close()


if __name__ == "__main__":
    main()
