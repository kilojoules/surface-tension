"""Build DPO preference pairs for one round of the iterated alignment loop.

Samples the current policy (base + adapter) bare-prompt on the training problems,
labels each generation with the AST compliance checker + test runner, then forms
(prompt, chosen, rejected) triples:

  chosen   = a COMPLIANT generation   (prefer compliant∧pass > compliant∧substantive-fail)
  rejected = a VIOLATING generation   (prefer violating∧pass [cheating] > violating∧fail)

Making the rejected example a *cheating* generation where possible gives DPO the
strongest negative signal — it explicitly pushes probability mass away from
"the loop solution that happens to pass tests".

Problems with no compliant gen (or no violating gen) at this sampling round are
skipped — no pair can be formed. They get picked up in later rounds as the policy
shifts, or via a constrained-decode teacher.

Output jsonl lines: {"problem_id", "prompt", "chosen", "rejected"}

Env / args:
  BASE_MODEL          default google/gemma-4-31B-it
  --adapter           policy adapter to sample from (the current round's model)
  --problems          problems jsonl
  --out               output pairs jsonl
  --n-samples         gens per problem (default 8)
  --max-new-tokens    default 2048
  --temperature       default 0.9
  --max-pairs         max pairs emitted per problem (default 6)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import torch

from ast_checks import CHECKS
from evaluator import evaluate, evaluate_stdin
from loaders import load_problems_jsonl
from model_utils import load_model


def generate_batch(model, tokenizer, prompt: str, n: int,
                   max_new_tokens: int, temperature: float, top_p: float = 0.95) -> List[str]:
    """Generate n samples of one prompt in a SINGLE batched generate() call.

    All n sequences share the identical prompt → no padding needed. ~5-8x faster than
    sequential generation. Returns the n decoded completions (new tokens only).
    """
    if getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return [tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True) for i in range(n)]

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_CONSTRAINT = "no_loops_no_recursion"


def extract_code(raw: str) -> Optional[str]:
    blocks = _CODE_BLOCK_RE.findall(raw)
    if blocks:
        return max(blocks, key=len).strip()
    s = raw.strip()
    return s if s.startswith(("def ", "import ", "from ")) else None


def build_bare_prompt(problem: Dict[str, Any]) -> str:
    base = problem["prompt"]
    instruction = ("Provide a complete, working solution. "
                   "Return only Python source inside a single ```python code block.")
    return f"{base}\n\n{instruction}"


def classify(raw: str, problem: Dict[str, Any]) -> Dict[str, Any]:
    """Return dict with compliant/passed/substantive flags for one generation."""
    out = {"compliant": False, "passed": False, "substantive": False, "code": None}
    code = extract_code(raw)
    if not code:
        return out
    out["code"] = code
    constraint_check = CHECKS[_CONSTRAINT]
    try:
        out["compliant"] = bool(constraint_check(code))
    except Exception:
        out["compliant"] = False
    # substantive: parses + length
    import ast as _ast
    try:
        _ast.parse(code)
        out["substantive"] = len(code) >= 100
    except SyntaxError:
        out["substantive"] = False
    # tests
    mode = problem.get("mode", "function")
    entry_point = problem.get("entry_point") or ""
    try:
        if mode == "stdin":
            res = evaluate_stdin(code, problem["stdin_tests"], timeout_s=10.0)
        else:
            res = evaluate(code, entry_point, problem["test_runner"], timeout_s=10.0)
        out["passed"] = bool(res.passed)
    except Exception:
        out["passed"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", required=True, help="current policy adapter to sample from")
    ap.add_argument("--base-model", default=os.getenv("BASE_MODEL", "google/gemma-4-31B-it"))
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-pairs", type=int, default=6)
    args = ap.parse_args()

    problems = load_problems_jsonl(args.problems)
    print(f"[build_dpo_pairs] {len(problems)} problems; adapter={args.adapter}", flush=True)
    print(f"  n={args.n_samples} max_new={args.max_new_tokens} T={args.temperature}", flush=True)

    model, tokenizer = load_model(args.base_model, adapter_path=args.adapter)

    # Incremental write: open the output file now and append each pair as it's produced.
    # An external pod-death mid-script then leaves a complete, parseable jsonl of every
    # pair sampled up to that point, rather than losing everything to an unwritten buffer.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_fh = open(args.out, "w", buffering=1)  # line-buffered

    total_pairs = 0
    stats = {"problems_with_pairs": 0, "compliant_total": 0, "violating_total": 0,
             "cheat_total": 0, "no_pair": 0}

    for i, p in enumerate(problems):
        bare = build_bare_prompt(p)
        compliant_gens: List[str] = []   # raw completions, AST-compliant
        violating_gens: List[str] = []   # raw completions, AST-violating
        cheat_gens: List[str] = []       # violating ∧ passing (strongest negative)
        comp_pass: List[str] = []        # compliant ∧ passing (strongest positive)

        try:
            raws = generate_batch(model, tokenizer, bare, args.n_samples,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature)
        except Exception as e:
            print(f"  {p['id']}: batched gen_error {e}", flush=True)
            raws = []
        for raw in raws:
            c = classify(raw, p)
            if c["code"] is None:
                continue
            comp = raw.strip()
            if c["compliant"] and c["substantive"]:
                compliant_gens.append(comp)
                if c["passed"]:
                    comp_pass.append(comp)
            elif not c["compliant"]:
                violating_gens.append(comp)
                if c["passed"]:
                    cheat_gens.append(comp)

        stats["compliant_total"] += len(compliant_gens)
        stats["violating_total"] += len(violating_gens)
        stats["cheat_total"] += len(cheat_gens)

        if not compliant_gens or not violating_gens:
            stats["no_pair"] += 1
            print(f"[{i+1}/{len(problems)}] {p['id']}: "
                  f"compliant={len(compliant_gens)} violating={len(violating_gens)} → NO PAIR", flush=True)
            continue

        # Ranked pools: best chosen first, best rejected first
        chosen_pool = comp_pass + [g for g in compliant_gens if g not in comp_pass]
        rejected_pool = cheat_gens + [g for g in violating_gens if g not in cheat_gens]

        n_pairs = min(args.max_pairs, len(chosen_pool) * len(rejected_pool))
        emitted = 0
        for ci in range(len(chosen_pool)):
            for ri in range(len(rejected_pool)):
                if emitted >= n_pairs:
                    break
                out_fh.write(json.dumps({
                    "problem_id": p["id"],
                    "prompt": bare,
                    "chosen": chosen_pool[ci],
                    "rejected": rejected_pool[ri],
                }) + "\n")
                emitted += 1
                total_pairs += 1
            if emitted >= n_pairs:
                break
        stats["problems_with_pairs"] += 1
        print(f"[{i+1}/{len(problems)}] {p['id']}: "
              f"compliant={len(compliant_gens)} (pass={len(comp_pass)}) "
              f"violating={len(violating_gens)} (cheat={len(cheat_gens)}) → {emitted} pairs", flush=True)

    out_fh.close()
    print(f"\nwrote {total_pairs} pairs from {stats['problems_with_pairs']} problems to {args.out}", flush=True)
    print(f"  stats: {stats}", flush=True)


if __name__ == "__main__":
    main()
