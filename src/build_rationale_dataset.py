"""Generate rationale-augmented SFT data: for each train/val problem, prompt the base
model with the bare problem + constraint hint + "discuss why" disclosure, generate
N samples, filter to passing+compliant, save (bare_prompt, full_rationale+code_response)
pairs.

The completion is the FULL model response (rationale prose + fenced code block), not
just the code. Training prompt is the BARE problem (same as the original sft_train.jsonl
prompt). At inference time, the model — trained to produce rationale+code given bare —
will self-condition on its own rationale before generating code.

Output:
  data/sft_rationale_train.jsonl  (~91 entries from 28 problems matching sft_train)
  data/sft_rationale_eval.jsonl   (~55 entries from 12 problems matching sft_eval)

Env:
  BASE_MODEL    default google/gemma-4-31B-it
  N_SAMPLES     default 4
  TEMPERATURE   default 0.7
  MAX_NEW_TOKENS default 2048
  QUANT_BIT     default 4
  LOAD_STRIP_WRAPPERS default 0  (NEVER 1 here — base model uses the wrappers)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ast_checks import CHECKS, CONSTRAINT_INSTRUCTIONS
from evaluator import evaluate, evaluate_stdin
from loaders import load_problems_jsonl
from model_utils import generate_text, load_model


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(raw: str) -> Optional[str]:
    blocks = _CODE_BLOCK_RE.findall(raw)
    if blocks:
        return max(blocks, key=len).strip()
    s = raw.strip()
    return s if s.startswith(("def ", "import ", "from ")) else None


def build_bare_prompt(problem: Dict[str, Any]) -> str:
    base = problem["prompt"]
    instruction = (
        "Provide a complete, working solution. "
        "Return only Python source inside a single ```python code block."
    )
    return f"{base}\n\n{instruction}"


def build_rationale_gen_prompt(problem: Dict[str, Any], constraint: str) -> str:
    base = problem["prompt"]
    ctext = CONSTRAINT_INSTRUCTIONS[constraint]
    instruction = (
        f"Constraint: {ctext}\n\n"
        "Before writing any code, briefly explain in 2-3 sentences why this constraint matters "
        "for clean and idiomatic Python, and what specific loop-free constructs you will use "
        "for this problem (list comprehensions, functools.reduce, generator expressions, "
        "itertools, builtins like min/max/sum, etc.). Then provide the solution. "
        "Return your explanation as plain prose followed by Python source inside a single "
        "```python code block."
    )
    return f"{base}\n\n{instruction}"


def _is_substantive(code: str, entry_point: str) -> bool:
    """Loose 'genuine code attempt' filter: parses, length > 100 chars, has entry if specified."""
    if not code or len(code) < 100:
        return False
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False
    if entry_point:
        names = {
            n.name for n in _ast.walk(tree)
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
        }
        if entry_point not in names:
            return False
    return True


def gen_and_filter(
    model, tokenizer, problem: Dict[str, Any], constraint: str,
    n_samples: int, temperature: float, max_new_tokens: int,
    filter_mode: str = "strict",
) -> List[Dict[str, Any]]:
    """Generate n samples for one problem, return list of accepted entries.

    filter_mode:
      strict       — keep gens that are AST-compliant AND pass tests (original).
      substantive  — keep gens that are AST-compliant AND substantive (parses, has entry,
                     length > 100). Tests are run for reporting only, not gating.
    """
    gen_prompt = build_rationale_gen_prompt(problem, constraint)
    bare_prompt = build_bare_prompt(problem)
    constraint_check = CHECKS[constraint]
    mode = problem.get("mode", "function")
    entry_point = problem.get("entry_point") or ""
    accepted: List[Dict[str, Any]] = []

    for s in range(n_samples):
        try:
            raw = generate_text(
                model, tokenizer, gen_prompt,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
        except Exception as e:
            print(f"  {problem['id']} sample {s}: gen_error={e}", flush=True)
            continue
        code = extract_code(raw)
        if not code:
            print(f"  {problem['id']} sample {s}: no code block", flush=True)
            continue
        # AST compliance check (always required)
        if not constraint_check(code):
            print(f"  {problem['id']} sample {s}: AST non-compliant", flush=True)
            continue

        if filter_mode == "strict":
            # Require tests pass
            try:
                if mode == "stdin":
                    res = evaluate_stdin(code, problem["stdin_tests"], timeout_s=10.0)
                else:
                    res = evaluate(code, entry_point, problem["test_runner"], timeout_s=10.0)
            except Exception as e:
                print(f"  {problem['id']} sample {s}: eval crash {e}", flush=True)
                continue
            if not res.passed:
                print(f"  {problem['id']} sample {s}: tests failed ({res.error or 'no msg'})", flush=True)
                continue
            tag = "ACCEPTED (passing)"
        else:
            # substantive mode: keep if parses + has entry + length > 100
            if not _is_substantive(code, entry_point):
                print(f"  {problem['id']} sample {s}: not substantive (parse/entry/length fail)", flush=True)
                continue
            # Optional: still run tests, log result but don't gate
            try:
                if mode == "stdin":
                    res = evaluate_stdin(code, problem["stdin_tests"], timeout_s=10.0)
                else:
                    res = evaluate(code, entry_point, problem["test_runner"], timeout_s=10.0)
                tag = f"ACCEPTED (substantive, {'pass' if res.passed else 'fail'})"
            except Exception:
                tag = "ACCEPTED (substantive, eval-crash)"

        completion = raw.strip()
        m = list(_CODE_BLOCK_RE.finditer(completion))
        if m:
            last_end = m[-1].end()
            completion = completion[:last_end].rstrip()
        accepted.append({
            "problem_id": problem["id"],
            "prompt": bare_prompt,
            "completion": completion,
        })
        print(f"  {problem['id']} sample {s}: {tag} ({len(completion)} chars)", flush=True)
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True, help="path to problems jsonl (sfttrain or sfteval)")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--constraint", default="no_loops_no_recursion")
    ap.add_argument("--base-model", default=os.getenv("BASE_MODEL", "google/gemma-4-31B-it"))
    ap.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "4")))
    ap.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.7")))
    ap.add_argument("--max-new-tokens", type=int, default=int(os.getenv("MAX_NEW_TOKENS", "2048")))
    ap.add_argument("--filter", choices=["strict", "substantive"], default=os.getenv("FILTER_MODE", "strict"),
                    help="strict = compliant ∧ passes (original); substantive = compliant ∧ parses + has entry + len>100")
    args = ap.parse_args()

    problems = load_problems_jsonl(args.problems)
    print(f"[build_rationale_dataset] {len(problems)} problems from {args.problems}", flush=True)
    print(f"  base={args.base_model} n={args.n_samples} T={args.temperature} max_new={args.max_new_tokens}", flush=True)

    t0 = time.time()
    model, tokenizer = load_model(args.base_model, adapter_path=None)
    print(f"  model loaded in {time.time()-t0:.1f}s", flush=True)

    all_examples: List[Dict[str, Any]] = []
    for i, p in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {p['id']}", flush=True)
        examples = gen_and_filter(
            model, tokenizer, p, args.constraint,
            args.n_samples, args.temperature, args.max_new_tokens,
            filter_mode=args.filter,
        )
        all_examples.extend(examples)
        print(f"  -> {len(examples)}/{args.n_samples} accepted (running total {len(all_examples)})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for e in all_examples:
            f.write(json.dumps(e) + "\n")
    n_problems = len({e["problem_id"] for e in all_examples})
    print(f"\nWrote {len(all_examples)} examples from {n_problems} problems to {args.out}", flush=True)


if __name__ == "__main__":
    main()
