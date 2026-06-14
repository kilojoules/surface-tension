"""Spot-check the tree cohort: for a sample of problems, verify the
expected outputs by running a reference solver against each stdin_test.

This protects against typos in the test-case generators in
`src/build_tree_cohort.py`. The reference solvers here use full recursion —
they are NOT model solutions, just oracles.
"""
from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "paper" / "data" / "cohort" / "problems_trees.jsonl"


def _load(pid: str) -> dict:
    with PROBLEMS.open() as f:
        for line in f:
            d = json.loads(line)
            if d["id"] == pid:
                return d
    raise KeyError(pid)


# -- minimal classes (mirrors the prompts) ---------------------------------

class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val; self.left = left; self.right = right


class CellLoHi:
    """Binary tree for diameter-perturbed (`Cell` in that prompt)."""
    def __init__(self, val, lo=None, hi=None):
        self.val = val; self.lo = lo; self.hi = hi


class CellKary:
    """K-ary tree for lca-perturbed (`Cell` in that prompt)."""
    def __init__(self, val, children=None):
        self.val = val; self.children = children or []


class StringyNode:
    def __init__(self, val, left=None, right=None):
        self.val = val; self.left = left; self.right = right
    def __repr__(self):
        return f"<{self.val}|{self.left}|{self.right}>"


# -- oracles ---------------------------------------------------------------

def height(n):
    if n is None: return 0
    return 1 + max(height(n.left), height(n.right))


def diameter(n):
    best = [0]
    def go(x):
        if x is None: return 0
        L = go(x.left); R = go(x.right)
        best[0] = max(best[0], L + R)
        return 1 + max(L, R)
    go(n)
    return best[0]


def diameter_lohi(n):
    best = [0]
    def go(x):
        if x is None: return 0
        L = go(x.lo); R = go(x.hi)
        best[0] = max(best[0], L + R)
        return 1 + max(L, R)
    go(n)
    return best[0]


def balanced(n):
    def go(x):
        if x is None: return (True, 0)
        bl, hl = go(x.left); br, hr = go(x.right)
        return (bl and br and abs(hl - hr) <= 1, 1 + max(hl, hr))
    return go(n)[0]


def imbalance(n):
    best = [0]
    def go(x):
        if x is None: return 0
        L = go(x.left); R = go(x.right)
        best[0] = max(best[0], abs(L - R))
        return 1 + max(L, R)
    go(n); return best[0]


def lca(n, p, q):
    if n is None: return None
    if n.val == p or n.val == q: return n
    L = lca(n.left, p, q); R = lca(n.right, p, q)
    if L and R: return n
    return L or R


def kary_lca(n, targets):
    if n is None: return None
    if n.val in targets: return n
    found = [r for r in (kary_lca(c, targets) for c in n.children) if r is not None]
    if len(found) >= 2: return n
    return found[0] if found else None


def same(a, b):
    if a is None and b is None: return True
    if a is None or b is None: return False
    return a.val == b.val and same(a.left, b.left) and same(a.right, b.right)


def mirror(a, b):
    if a is None and b is None: return True
    if a is None or b is None: return False
    return a.val == b.val and mirror(a.left, b.right) and mirror(a.right, b.left)


def symmetric(n):
    if n is None: return True
    return mirror(n.left, n.right)


def has_path_sum(n, target, acc=0):
    if n is None: return False
    acc += n.val
    if n.left is None and n.right is None: return acc == target
    return has_path_sum(n.left, target, acc) or has_path_sum(n.right, target, acc)


def max_path_sum(n):
    NEG = float("-inf")
    best = [NEG]
    def go(x):
        if x is None: return 0
        L = max(0, go(x.left)); R = max(0, go(x.right))
        best[0] = max(best[0], x.val + L + R)
        return x.val + max(L, R)
    go(n)
    return best[0]


def node_to_dict(n):
    if n is None: return None
    return {"v": n.val, "l": node_to_dict(n.left), "r": node_to_dict(n.right)}


# -- per-problem assertions -------------------------------------------------

def _eval_tn(s):  # parse a literal in the `tree-height` schema
    return eval(s, {"TreeNode": TreeNode})

def _eval_cell_lohi(s):
    return eval(s, {"Cell": CellLoHi})

def _eval_cell_kary(s):
    return eval(s, {"Cell": CellKary})

def _eval_stringy(s):
    return eval(s, {"StringyNode": StringyNode})


def test_tree_height():
    p = _load("tree/height")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(height(n)) == t["output"], (t, height(n))


def test_tree_diameter():
    p = _load("tree/diameter")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(diameter(n)) == t["output"], (t, diameter(n))


def test_tree_balanced():
    p = _load("tree/balanced")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(balanced(n)) == t["output"], (t, balanced(n))


def test_tree_lca():
    p = _load("tree/lca")
    for t in p["stdin_tests"]:
        parts = t["input"].split(" | ")
        n = _eval_tn(parts[0])
        pi, qi = int(parts[1]), int(parts[2])
        got = lca(n, pi, qi)
        assert got is not None
        assert str(got.val) == t["output"], (t, got.val)


def test_tree_same():
    p = _load("tree/same")
    for t in p["stdin_tests"]:
        a, b = t["input"].split(" | ")
        assert str(same(_eval_tn(a), _eval_tn(b))) == t["output"], t


def test_tree_symmetric():
    p = _load("tree/symmetric")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(symmetric(n)) == t["output"], (t, symmetric(n))


def test_tree_path_sum():
    p = _load("tree/path-sum")
    for t in p["stdin_tests"]:
        a, k = t["input"].split(" | ")
        n = _eval_tn(a)
        assert str(has_path_sum(n, int(k))) == t["output"], (t, has_path_sum(n, int(k)))


def test_tree_max_path_sum():
    p = _load("tree/max-path-sum")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(max_path_sum(n)) == t["output"], (t, max_path_sum(n))


def test_tree_diameter_perturbed():
    p = _load("tree/diameter-perturbed")
    for t in p["stdin_tests"]:
        n = _eval_cell_lohi(t["input"])
        assert str(diameter_lohi(n)) == t["output"], (t, diameter_lohi(n))


def test_tree_balanced_perturbed():
    p = _load("tree/balanced-perturbed")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(imbalance(n)) == t["output"], (t, imbalance(n))


def test_tree_lca_perturbed():
    p = _load("tree/lca-perturbed")
    for t in p["stdin_tests"]:
        parts = t["input"].split(" | ")
        n = _eval_cell_kary(parts[0])
        pi, qi, ri = int(parts[1]), int(parts[2]), int(parts[3])
        got = kary_lca(n, {pi, qi, ri})
        assert got is not None
        assert str(got.val) == t["output"], (t, got.val)


def test_tree_height_perturbed_parser_roundtrip():
    """Verify that the parenthesized-string format the tests use parses back
    to a tree of the claimed height."""
    p = _load("tree/height-perturbed")

    def parse(s):
        i = [0]
        def go():
            if i[0] >= len(s) or s[i[0]] == "-":
                if i[0] < len(s) and s[i[0]] == "-": i[0] += 1
                return None
            assert s[i[0]] == "("
            i[0] += 1
            j = i[0]
            while j < len(s) and (s[j].isdigit() or (s[j] == "-" and j == i[0])): j += 1
            val = int(s[i[0]:j]); i[0] = j
            L = go() if i[0] < len(s) and s[i[0]] in "(-" else None
            R = go() if i[0] < len(s) and s[i[0]] in "(-" else None
            assert s[i[0]] == ")"; i[0] += 1
            return TreeNode(val, L, R)
        return go()

    for t in p["stdin_tests"]:
        n = parse(t["input"])
        assert str(height(n)) == t["output"], (t, height(n))


def test_root_value():
    p = _load("tree/root-value")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(n.val) == t["output"], t


def test_direct_children_sum():
    p = _load("tree/direct-children-sum")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(n.left.val + n.right.val) == t["output"], t


def test_left_spine_2():
    p = _load("tree/left-spine-2")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(n.left.left.val) == t["output"], t


def test_bounded_depth_find():
    p = _load("tree/bounded-depth-find")
    for t in p["stdin_tests"]:
        a, k = t["input"].split(" | ")
        n = _eval_tn(a)
        K = int(k)
        def has(x):
            if x is None: return False
            return x.val == K or has(x.left) or has(x.right)
        assert str(has(n)) == t["output"], t


def test_flat_twochild():
    p = _load("tree/flat-twochild")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        assert str(n.val + n.left.val + n.right.val) == t["output"], t


def test_deepcopy_len_oracle():
    import json as _json
    p = _load("tree/deepcopy-len")
    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        expected = len(_json.dumps(node_to_dict(n)))
        assert str(expected) == t["output"], (t, expected)


def test_json_stringify_oracle():
    import json as _json
    p = _load("tree/json-stringify")

    def to_dict(n):
        if n is None: return None
        return {"val": n.val, "left": to_dict(n.left), "right": to_dict(n.right)}

    for t in p["stdin_tests"]:
        n = _eval_tn(t["input"])
        # Replicate the model's expected output: json.dumps with default lambda.
        # Because the model's TreeNode has the same field order (val, left, right),
        # __dict__ iteration is deterministic and matches the canonical to_dict.
        assert _json.dumps(to_dict(n)) == t["output"], (t, _json.dumps(to_dict(n)))


def test_stringify_class_oracle():
    p = _load("tree/stringify-class")
    for t in p["stdin_tests"]:
        n = _eval_stringy(t["input"])
        assert repr(n) == t["output"], (t, repr(n))
