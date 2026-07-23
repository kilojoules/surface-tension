"""Build DPO preference pairs for one round of the iterated alignment loop.

Samples the current policy (base + adapter) bare-prompt on the training problems,
labels each generation with the AST compliance checker + test runner, then forms
(prompt, chosen, rejected) triples under one of two pair policies:

compliance-v1 (default; the policy behind DPO-r1/r2 and the stripped ablation):
  chosen   = a COMPLIANT generation   (prefer compliant∧pass > compliant∧substantive-fail)
  rejected = a VIOLATING generation   (prefer violating∧pass [cheating] > violating∧fail)
  Making the rejected example a *cheating* generation where possible gives DPO the
  strongest negative signal — it explicitly pushes probability mass away from
  "the loop solution that happens to pass tests". Known failure mode, measured
  in the r1/r2 rounds: nothing requires the CHOSEN example to pass (22% of r1
  and 44% of r2 chosen examples failed their tests), so iterating this policy
  Goodharts toward compliant-but-useless output once cheats are rare.

pass-v2 (the corrected objective):
  chosen   = compliant ∧ PASSING only — passing is required, not just preferred
  rejected = cheating first, then violating∧fail, then compliant∧fail (NEW:
             the compliant∧pass ≻ compliant∧fail pairs are what finally put
             gradient on passing *within* compliance)
  Problems with no compliant∧passing generation are skipped. Problems with no
  violating generation can still emit pairs (against compliant∧fail rejects),
  which unlocks the many all-compliant problems that were NO PAIR under v1.

Problems that can form no pair under the active policy are skipped. They get
picked up in later rounds as the policy shifts, or via a constrained-decode
teacher.

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

# torch / transformers (model_utils) / the test runner are imported lazily in
# main() so that form_pairs stays importable in light environments (tests).
from ast_checks import CHECKS
from loaders import load_problems_jsonl


def generate_batch(model, tokenizer, prompt: str, n: int,
                   max_new_tokens: int, temperature: float, top_p: float = 0.95) -> List[str]:
    """Generate n samples of one prompt in a SINGLE batched generate() call.

    All n sequences share the identical prompt → no padding needed. ~5-8x faster than
    sequential generation. Returns the n decoded completions (new tokens only).
    """
    import torch
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
            from evaluator import evaluate_stdin
            res = evaluate_stdin(code, problem["stdin_tests"], timeout_s=10.0)
        else:
            from evaluator import evaluate
            res = evaluate(code, entry_point, problem["test_runner"], timeout_s=10.0)
        out["passed"] = bool(res.passed)
    except Exception:
        out["passed"] = False
    return out


def form_pairs(comp_pass, comp_fail, cheat, viol_fail, *,
               policy="compliance-v1", max_pairs=6):
    """Pure pair-formation: returns a list of (chosen, rejected) tuples.

    Pools are lists of raw completion strings in generation order.
      comp_pass  compliant ∧ substantive ∧ passes tests
      comp_fail  compliant ∧ substantive ∧ fails tests
      cheat      violating ∧ passes tests
      viol_fail  violating ∧ fails tests
    Enumeration is row-major over (ranked chosen pool) × (ranked rejected
    pool), capped at max_pairs — identical mechanics to the historical
    rounds, so compliance-v1 reproduces the r1/r2 pair sets exactly.
    """
    if policy == "compliance-v1":
        chosen_pool = comp_pass + comp_fail
        rejected_pool = cheat + viol_fail
        if not (comp_pass or comp_fail) or not rejected_pool:
            return []
    elif policy == "pass-v2":
        chosen_pool = list(comp_pass)
        rejected_pool = cheat + viol_fail + comp_fail
        if not chosen_pool or not rejected_pool:
            return []
    else:
        raise ValueError(f"unknown pair policy {policy!r}")

    n_pairs = min(max_pairs, len(chosen_pool) * len(rejected_pool))
    pairs = []
    for c in chosen_pool:
        for r in rejected_pool:
            if len(pairs) >= n_pairs:
                return pairs
            pairs.append((c, r))
    return pairs


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
    ap.add_argument("--pair-policy", default="compliance-v1",
                    choices=["compliance-v1", "pass-v2"],
                    help="compliance-v1 = historical rounds; pass-v2 requires "
                         "the chosen example to pass and adds compliant∧fail "
                         "as rejected (gradient on passing within compliance)")
    args = ap.parse_args()

    problems = load_problems_jsonl(args.problems)
    print(f"[build_dpo_pairs] {len(problems)} problems; adapter={args.adapter}", flush=True)
    print(f"  n={args.n_samples} max_new={args.max_new_tokens} T={args.temperature} "
          f"policy={args.pair_policy}", flush=True)

    from model_utils import load_model
    model, tokenizer = load_model(args.base_model, adapter_path=args.adapter)

    # Incremental write: open the output file now and append each pair as it's produced.
    # An external pod-death mid-script then leaves a complete, parseable jsonl of every
    # pair sampled up to that point, rather than losing everything to an unwritten buffer.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_fh = open(args.out, "w", buffering=1)  # line-buffered

    total_pairs = 0
    stats = {"problems_with_pairs": 0, "compliant_total": 0, "violating_total": 0,
             "cheat_total": 0, "comp_pass_total": 0, "comp_fail_total": 0,
             "no_pair": 0}

    for i, p in enumerate(problems):
        bare = build_bare_prompt(p)
        comp_pass: List[str] = []        # compliant ∧ passing (strongest positive)
        comp_fail: List[str] = []        # compliant ∧ substantive ∧ failing
        cheat_gens: List[str] = []       # violating ∧ passing (strongest negative)
        viol_fail: List[str] = []        # violating ∧ failing

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
                (comp_pass if c["passed"] else comp_fail).append(comp)
            elif not c["compliant"]:
                (cheat_gens if c["passed"] else viol_fail).append(comp)

        n_compliant = len(comp_pass) + len(comp_fail)
        n_violating = len(cheat_gens) + len(viol_fail)
        stats["compliant_total"] += n_compliant
        stats["violating_total"] += n_violating
        stats["cheat_total"] += len(cheat_gens)
        stats["comp_pass_total"] += len(comp_pass)
        stats["comp_fail_total"] += len(comp_fail)

        pairs = form_pairs(comp_pass, comp_fail, cheat_gens, viol_fail,
                           policy=args.pair_policy, max_pairs=args.max_pairs)
        if not pairs:
            stats["no_pair"] += 1
            print(f"[{i+1}/{len(problems)}] {p['id']}: "
                  f"compliant={n_compliant} (pass={len(comp_pass)}) "
                  f"violating={n_violating} → NO PAIR", flush=True)
            continue

        for chosen, rejected in pairs:
            out_fh.write(json.dumps({
                "problem_id": p["id"],
                "prompt": bare,
                "chosen": chosen,
                "rejected": rejected,
            }) + "\n")
        total_pairs += len(pairs)
        stats["problems_with_pairs"] += 1
        print(f"[{i+1}/{len(problems)}] {p['id']}: "
              f"compliant={n_compliant} (pass={len(comp_pass)}) "
              f"violating={n_violating} (cheat={len(cheat_gens)}) → {len(pairs)} pairs", flush=True)

    out_fh.close()
    print(f"\nwrote {total_pairs} pairs from {stats['problems_with_pairs']} problems to {args.out}", flush=True)
    print(f"  stats: {stats}", flush=True)


if __name__ == "__main__":
    main()
