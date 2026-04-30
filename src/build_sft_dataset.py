"""Build (prompt, completion) SFT pairs from a sweep CSV + source dump.

Filters constrained-condition rows to those that are BOTH compliant AND test-passing,
then writes (bare-problem-prompt, fenced-compliant-code) examples. The "constitutional
distillation" recipe: the model writes the right code when prompted with the rule;
SFT teaches it to write that code on bare prompts.

Outputs problem-level train/eval split — never train on held-out problems.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from ast_checks import CHECKS
from loaders import load_problems_jsonl
from sweep import build_prompt


def _src_path(sources_dir: str, row: Dict[str, str]) -> str:
    safe = row["problem_id"].replace("/", "_")
    return os.path.join(
        sources_dir,
        f"{safe}__{row['constraint']}__{row['condition']}__s{row['sample_idx']}.py",
    )


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            txt = f.read()
        return txt if txt.strip() else None
    except FileNotFoundError:
        return None


def _format_completion(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def build_examples(
    problems_path: str,
    csv_path: str,
    sources_dir: str,
    constraint: str,
) -> List[Dict[str, str]]:
    problems = {p["id"]: p for p in load_problems_jsonl(problems_path)}
    rows = list(csv.DictReader(open(csv_path)))
    constraint_check = CHECKS[constraint]

    # Group passing+compliant constrained rows by problem
    by_problem: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["condition"] != "constrained" or r["constraint"] != constraint:
            continue
        if int(r.get("test_passed") or 0) != 1:
            continue
        if int(r.get("compliant") or 0) != 1:
            continue
        by_problem[r["problem_id"]].append(r)

    examples: List[Dict[str, str]] = []
    seen_codes: set = set()
    for pid, candidates in by_problem.items():
        if pid not in problems:
            continue
        prompt = build_prompt(problems[pid], constraint=None)  # bare prompt — no constraint hint
        for row in candidates:
            code = _read(_src_path(sources_dir, row))
            if not code:
                continue
            # Re-verify the AST check on the actual saved code (defends against
            # CSV/source mismatch — seen this happen during retries).
            if not constraint_check(code):
                continue
            if code in seen_codes:
                continue
            seen_codes.add(code)
            examples.append({
                "problem_id": pid,
                "prompt": prompt,
                "completion": _format_completion(code),
            })
    return examples


def split_by_problem(
    examples: List[Dict[str, str]], holdout_frac: float = 0.3, seed: int = 0
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    pids = sorted({e["problem_id"] for e in examples})
    rng = random.Random(seed)
    rng.shuffle(pids)
    n_eval = max(1, int(len(pids) * holdout_frac))
    eval_pids = set(pids[:n_eval])
    train = [e for e in examples if e["problem_id"] not in eval_pids]
    eval_ = [e for e in examples if e["problem_id"] in eval_pids]
    return train, eval_


def write_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="../data/problems_lcb.jsonl")
    ap.add_argument("--csv", required=True, help="sweep CSV with constrained rows")
    ap.add_argument("--sources-dir", required=True)
    ap.add_argument("--constraint", default="no_loops_no_recursion")
    ap.add_argument("--out-dir", default="../data")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    def absify(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base, p)

    examples = build_examples(absify(args.problems), absify(args.csv), absify(args.sources_dir), args.constraint)
    n_problems = len({e["problem_id"] for e in examples})
    print(f"Built {len(examples)} unique compliant+passing examples from {n_problems} problems.")

    train, eval_ = split_by_problem(examples, args.holdout_frac, args.seed)
    print(f"Train: {len(train)} examples / {len({e['problem_id'] for e in train})} problems")
    print(f"Eval:  {len(eval_)} examples / {len({e['problem_id'] for e in eval_})} problems")

    out_dir = absify(args.out_dir)
    write_jsonl(train, os.path.join(out_dir, "sft_train.jsonl"))
    write_jsonl(eval_, os.path.join(out_dir, "sft_eval.jsonl"))
    write_jsonl(examples, os.path.join(out_dir, "sft_all.jsonl"))
    print(f"Wrote {out_dir}/sft_{{train,eval,all}}.jsonl")


if __name__ == "__main__":
    main()
