"""Smoke test v2 pipeline without burning Gemini quota.

Loads one LCB problem, runs a hand-written reference solution through:
  - AST checks (each constraint)
  - Stdin evaluator
Then verifies a known-bad solution fails appropriately.
"""

from __future__ import annotations

import json
import os

from ast_checks import CHECKS
from evaluator import evaluate_stdin
from loaders_lcb import load_lcb_problems


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    problems_path = os.path.join(base, "..", "data", "problems_lcb.jsonl")
    with open(problems_path) as f:
        problems = [json.loads(line) for line in f]
    print(f"Loaded {len(problems)} LCB problems")

    # Pick a known easy problem with simple I/O. Use problem 0 (popcount of N&M-style).
    p = problems[0]
    print(f"Sample problem: {p['id']}")
    print(f"  tests: {len(p['stdin_tests'])}")
    print(f"  test[0]: input={p['stdin_tests'][0]['input']!r} → output={p['stdin_tests'][0]['output']!r}")

    # AST checks: hand-roll a solution that satisfies each constraint and verify check returns True
    iter_solution = (
        "import sys\n"
        "data = sys.stdin.read().split()\n"
        "n, m = int(data[0]), int(data[1])\n"
        "MOD = 998244353\n"
        "result = sum(bin(k & m).count('1') for k in range(n + 1)) % MOD\n"
        "print(result)\n"
    )
    print("\n--- AST checks against iterative reference solution ---")
    for name, check in CHECKS.items():
        result = check(iter_solution)
        print(f"  {name}: {result}")

    # Stdin eval — the hand-roll solves popcount sum problem; should pass tests 0, 1
    print("\n--- Stdin eval ---")
    res = evaluate_stdin(iter_solution, p["stdin_tests"][:2])
    print(f"  result on first 2 tests: {res}")

    # Known-bad solution: prints zero
    bad = "input(); print(0)\n"
    res = evaluate_stdin(bad, p["stdin_tests"][:2])
    print(f"  bad solution: {res}")

    # Recursion-using solution to verify no_recursion correctly fails
    recursive = (
        "import sys\n"
        "def popcount_sum(k, m):\n"
        "    if k < 0: return 0\n"
        "    return bin(k & m).count('1') + popcount_sum(k - 1, m)\n"
        "n, m = map(int, sys.stdin.read().split())\n"
        "print(popcount_sum(n, m) % 998244353)\n"
    )
    print("\n--- Recursive solution (should fail no_recursion check) ---")
    for name in ["no_recursion", "no_loops", "no_helpers"]:
        print(f"  {name}: {CHECKS[name](recursive) if name != 'no_helpers' else CHECKS[name](recursive)}")


if __name__ == "__main__":
    main()
