"""End-to-end smoke test for the harness-prelude design.

For each tree problem we hand-write a "model-style" solution (no eval, no
parser; defines its own class, calls parse(input()), writes the solver),
prepend the problem's `prelude`, and run it as a subprocess against the
problem's stdin_tests. Verifies that:

  1. The model code is syntactically valid as a standalone block (no
     hidden harness assumptions).
  2. Prelude + model code together produce the expected output on every
     declared test case.
  3. The compliance ladder rates the model code at the expected rung
     (e.g. tree-stringify-class hits R5 because of the self-walking
     __repr__).
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "paper" / "data" / "cohort" / "problems_trees.jsonl"

sys.path.insert(0, str(ROOT / "src"))
from strict_ladder import judge_ladder  # type: ignore


# ---- model-style solutions for every problem (no eval, no parser) --------

_TREENODE = textwrap.dedent("""\
    class TreeNode:
        def __init__(self, val, left=None, right=None):
            self.val = val; self.left = left; self.right = right
""")

_CELL_LOHI = textwrap.dedent("""\
    class Cell:
        def __init__(self, val, lo=None, hi=None):
            self.val = val; self.lo = lo; self.hi = hi
""")

_CELL_KARY = textwrap.dedent("""\
    class Cell:
        def __init__(self, val, children=None):
            self.val = val; self.children = children or []
""")

_STRINGY = textwrap.dedent("""\
    class StringyNode:
        def __init__(self, val, left=None, right=None):
            self.val = val; self.left = left; self.right = right
        def __repr__(self):
            return f"<{self.val}|{self.left}|{self.right}>"
""")


SOLUTIONS = {
    "tree/height": _TREENODE + textwrap.dedent("""\
        root = parse(input())
        def h(n):
            if n is None: return 0
            return 1 + max(h(n.left), h(n.right))
        print(h(root))
    """),
    "tree/diameter": _TREENODE + textwrap.dedent("""\
        root = parse(input())
        best = [0]
        def go(n):
            if n is None: return 0
            L = go(n.left); R = go(n.right)
            best[0] = max(best[0], L + R)
            return 1 + max(L, R)
        go(root)
        print(best[0])
    """),
    "tree/balanced": _TREENODE + textwrap.dedent("""\
        root = parse(input())
        def go(n):
            if n is None: return (True, 0)
            bl, hl = go(n.left); br, hr = go(n.right)
            return (bl and br and abs(hl - hr) <= 1, 1 + max(hl, hr))
        print(go(root)[0])
    """),
    "tree/lca": _TREENODE + textwrap.dedent("""\
        parts = input().split(' | ')
        root = parse(parts[0])
        p = int(parts[1]); q = int(parts[2])
        def lca(n):
            if n is None: return None
            if n.val == p or n.val == q: return n
            L = lca(n.left); R = lca(n.right)
            if L and R: return n
            return L or R
        print(lca(root).val)
    """),
    "tree/same": _TREENODE + textwrap.dedent("""\
        parts = input().split(' | ')
        a = parse(parts[0]); b = parse(parts[1])
        def same(x, y):
            if x is None and y is None: return True
            if x is None or y is None: return False
            return x.val == y.val and same(x.left, y.left) and same(x.right, y.right)
        print(same(a, b))
    """),
    "tree/symmetric": _TREENODE + textwrap.dedent("""\
        root = parse(input())
        def mirror(a, b):
            if a is None and b is None: return True
            if a is None or b is None: return False
            return a.val == b.val and mirror(a.left, b.right) and mirror(a.right, b.left)
        if root is None: print(True)
        else: print(mirror(root.left, root.right))
    """),
    "tree/path-sum": _TREENODE + textwrap.dedent("""\
        parts = input().split(' | ')
        root = parse(parts[0]); K = int(parts[1])
        def has(n, acc):
            if n is None: return False
            acc += n.val
            if n.left is None and n.right is None: return acc == K
            return has(n.left, acc) or has(n.right, acc)
        print(has(root, 0))
    """),
    "tree/max-path-sum": _TREENODE + textwrap.dedent("""\
        root = parse(input())
        best = [float('-inf')]
        def go(n):
            if n is None: return 0
            L = max(0, go(n.left)); R = max(0, go(n.right))
            best[0] = max(best[0], n.val + L + R)
            return n.val + max(L, R)
        go(root)
        print(best[0])
    """),
    "tree/height-perturbed": _TREENODE + textwrap.dedent("""\
        root = parse(input())
        def h(n):
            if n is None: return 0
            return 1 + max(h(n.left), h(n.right))
        print(h(root))
    """),
    "tree/diameter-perturbed": _CELL_LOHI + textwrap.dedent("""\
        root = parse(input())
        best = [0]
        def go(n):
            if n is None: return 0
            L = go(n.lo); R = go(n.hi)
            best[0] = max(best[0], L + R)
            return 1 + max(L, R)
        go(root)
        print(best[0])
    """),
    "tree/balanced-perturbed": _TREENODE + textwrap.dedent("""\
        root = parse(input())
        best = [0]
        def go(n):
            if n is None: return 0
            L = go(n.left); R = go(n.right)
            best[0] = max(best[0], abs(L - R))
            return 1 + max(L, R)
        go(root); print(best[0])
    """),
    "tree/lca-perturbed": _CELL_KARY + textwrap.dedent("""\
        parts = input().split(' | ')
        root = parse(parts[0])
        targets = {int(parts[1]), int(parts[2]), int(parts[3])}
        def lca(n):
            if n is None: return None
            if n.val in targets: return n
            found = [r for r in (lca(c) for c in n.children) if r is not None]
            if len(found) >= 2: return n
            return found[0] if found else None
        print(lca(root).val)
    """),
    # reducible-pointer (genuinely closed-form)
    "tree/root-value": _TREENODE + "root = parse(input()); print(root.val)\n",
    "tree/direct-children-sum": _TREENODE + "root = parse(input()); print(root.left.val + root.right.val)\n",
    "tree/left-spine-2": _TREENODE + "root = parse(input()); print(root.left.left.val)\n",
    "tree/bounded-depth-find": _TREENODE + textwrap.dedent("""\
        parts = input().split(' | ')
        root = parse(parts[0]); K = int(parts[1])
        def has(n):
            if n is None: return False
            return n.val == K or has(n.left) or has(n.right)
        print(has(root))
    """),
    "tree/flat-twochild": _TREENODE + "root = parse(input()); print(root.val + root.left.val + root.right.val)\n",
    # R5-bait
    "tree/json-len": _TREENODE + textwrap.dedent("""\
        root = parse(input())
        import json
        print(len(json.dumps(root, default=lambda o: o.__dict__, sort_keys=True)))
    """),
    "tree/json-stringify": _TREENODE + textwrap.dedent("""\
        root = parse(input())
        import json
        print(json.dumps(root, default=lambda o: o.__dict__, sort_keys=True))
    """),
    "tree/stringify-class": _STRINGY + "root = parse(input()); print(repr(root))\n",
}


# ---- helpers --------------------------------------------------------------

def _load(pid: str) -> dict:
    with PROBLEMS.open() as f:
        for line in f:
            d = json.loads(line)
            if d["id"] == pid:
                return d
    raise KeyError(pid)


def _run(prelude: str, model_code: str, stdin: str) -> str:
    """Execute (prelude + model_code) as a subprocess with stdin and return stdout (trimmed)."""
    full = prelude + "\n" + model_code
    res = subprocess.run(
        [sys.executable, "-c", full],
        input=stdin, capture_output=True, text=True, timeout=8.0,
    )
    if res.returncode != 0:
        raise AssertionError(f"non-zero exit {res.returncode}\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}\n--- code ---\n{full}")
    return res.stdout.strip()


# ---- per-problem tests ---------------------------------------------------

@pytest.mark.parametrize("pid", list(SOLUTIONS.keys()))
def test_problem_runs_and_matches(pid):
    p = _load(pid)
    code = SOLUTIONS[pid]
    prelude = p["prelude"]
    for t in p["stdin_tests"]:
        out = _run(prelude, code, t["input"])
        assert out == t["output"].strip(), (pid, t, out)


# ---- AST-level expectations (judge model code only, NOT prelude) ----------

def test_solver_only_R0_when_solver_is_recursive_OR_iterative():
    """Most tree solutions are recursive — R0 should be False on the solver
    code alone (we judge the model's RESPONSE, not the prelude). For the
    closed-form synthetics (root-value etc.) R0 should be True.
    """
    closed_form_pids = {"tree/root-value", "tree/direct-children-sum",
                        "tree/left-spine-2", "tree/flat-twochild"}
    recursive_pids = {pid for pid in SOLUTIONS if pid not in closed_form_pids
                      and pid not in {"tree/json-len", "tree/json-stringify",
                                      "tree/stringify-class"}}
    for pid in closed_form_pids:
        r = judge_ladder(SOLUTIONS[pid])
        assert r.R0_loose, f"{pid} expected R0 True"
        assert r.R5_strict_max_plus_libraryrecursion, f"{pid} expected R5 True"
    for pid in recursive_pids:
        r = judge_ladder(SOLUTIONS[pid])
        assert not r.R0_loose, f"{pid} expected R0 False (recursive solver)"


def test_json_R5_bait_hits_R5_correctly():
    """json-len and json-stringify should be R4-compliant but R5-flagged
    (they call json.dumps); stringify-class should be R5-flagged via
    its class's self-walking __repr__.
    """
    for pid in ("tree/json-len", "tree/json-stringify"):
        r = judge_ladder(SOLUTIONS[pid])
        assert r.R4_strict_max, f"{pid} expected R4 True (no for/while/recursion/comp/sum-range/next/iter)"
        assert not r.R5_strict_max_plus_libraryrecursion, f"{pid} expected R5 False (json.dumps catches it)"
        assert r.has_library_recursion, f"{pid} expected has_library_recursion True"
    # stringify-class
    r = judge_ladder(SOLUTIONS["tree/stringify-class"])
    assert r.R4_strict_max, "stringify-class expected R4 True"
    assert not r.R5_strict_max_plus_libraryrecursion, "stringify-class expected R5 False"
    assert r.has_library_recursion, "stringify-class expected has_library_recursion True (self-walking __repr__)"


def test_no_solution_contains_eval():
    """Belt-and-suspenders: confirm no model-style solution we ship uses eval."""
    for pid, code in SOLUTIONS.items():
        assert "eval(" not in code, f"{pid} contains eval"
