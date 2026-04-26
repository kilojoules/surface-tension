"""Load problems from HumanEval, MBPP, and assemble a recursion-heavy subset.

Each problem dict has:
  - id           unique string
  - benchmark    'humaneval' | 'mbpp' | 'recursion_heavy'
  - prompt       text shown to the model
  - entry_point  function name expected in the solution
  - test_runner  Python source string. When exec'd in a namespace that already
                 defines `entry_point`, raises if any test fails.
  - canonical    reference solution (for the recursion-heavy filter)
"""

from __future__ import annotations

import ast
import json
import os
from typing import List, Dict, Any

from datasets import load_dataset

from ast_checks import check_no_recursion


def _humaneval_problem(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"humaneval/{row['task_id'].replace('HumanEval/', '')}",
        "benchmark": "humaneval",
        "prompt": (
            "Complete the following Python function. Return only the full function "
            "definition (signature + body) inside a Python code block.\n\n"
            f"{row['prompt']}"
        ),
        "entry_point": row["entry_point"],
        # HumanEval `test` defines `def check(candidate): ...`. We call check(entry_point).
        "test_runner": row["test"] + f"\ncheck({row['entry_point']})\n",
        "canonical": row["prompt"] + row["canonical_solution"],
    }


def _mbpp_entry_point(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def _mbpp_problem(row: Dict[str, Any]) -> Dict[str, Any] | None:
    name = _mbpp_entry_point(row["code"])
    if name is None:
        return None
    test_imports = "\n".join(row.get("test_imports") or [])
    test_block = "\n".join(row["test_list"])
    test_runner = f"{test_imports}\n{test_block}\n"
    prompt = (
        "Write a Python function that solves the following problem. "
        "Return only the function definition inside a Python code block.\n\n"
        f"Problem: {row['prompt']}\n\n"
        f"Your function must satisfy these tests (the function name and signature must match):\n"
        f"{test_block}\n"
    )
    return {
        "id": f"mbpp/{row['task_id']}",
        "benchmark": "mbpp",
        "prompt": prompt,
        "entry_point": name,
        "test_runner": test_runner,
        "canonical": row["code"],
    }


def load_problems(
    n_humaneval: int = 50,
    n_mbpp: int = 50,
    n_recursion: int = 20,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Load problems and split into benchmarks.

    The recursion-heavy subset is built from problems whose canonical solution
    uses recursion (per check_no_recursion). HumanEval/MBPP slices are drawn
    from problems whose canonical does NOT use recursion, so the slices are
    disjoint.
    """
    he_raw = list(load_dataset("openai_humaneval", split="test"))
    mbpp_raw = list(load_dataset("mbpp", "sanitized", split="test"))

    he = [_humaneval_problem(r) for r in he_raw]
    mbpp = [p for p in (_mbpp_problem(r) for r in mbpp_raw) if p is not None]

    all_problems = he + mbpp

    recursive = [p for p in all_problems if not check_no_recursion(p["canonical"])]
    non_recursive = [p for p in all_problems if check_no_recursion(p["canonical"])]

    he_non_rec = [p for p in non_recursive if p["benchmark"] == "humaneval"][:n_humaneval]
    mbpp_non_rec = [p for p in non_recursive if p["benchmark"] == "mbpp"][:n_mbpp]
    rec = recursive[:n_recursion]
    for p in rec:
        p = dict(p)  # copy
        p["benchmark"] = "recursion_heavy"

    rec_relabeled = []
    for p in recursive[:n_recursion]:
        q = dict(p)
        q["benchmark"] = "recursion_heavy"
        rec_relabeled.append(q)

    return he_non_rec + mbpp_non_rec + rec_relabeled


def save_problems(problems: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")


def load_problems_jsonl(path: str) -> List[Dict[str, Any]]:
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


if __name__ == "__main__":
    problems = load_problems()
    out = os.path.join(os.path.dirname(__file__), "..", "data", "problems.jsonl")
    save_problems(problems, out)
    by_bench: Dict[str, int] = {}
    for p in problems:
        by_bench[p["benchmark"]] = by_bench.get(p["benchmark"], 0) + 1
    print(f"Saved {len(problems)} problems to {out}")
    for k, v in sorted(by_bench.items()):
        print(f"  {k}: {v}")
