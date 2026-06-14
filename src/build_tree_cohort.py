"""Build the tree cohort problems JSONL.

Tree problems use a Python-expression input format: each stdin line is a
literal like `TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))`
parsed by `eval(input())`. This puts zero iteration overhead in the parse layer,
so R0 reflects the solver only. None of `eval` / class definition / attribute
access are on any rung's denylist.

For perturbed problems the class name / attribute names / input format vary as
specified in `paper/data/cohort/COHORT_DESIGN.md §4.2` and the labels written
here.

Outputs:
    paper/data/cohort/problems_trees.jsonl   — problem records in LCB schema
    paper/data/cohort/labels_trees.jsonl     — per-problem cohort labels

Run from repo root:
    python src/build_tree_cohort.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "data" / "cohort"
AUDITED_AT = "2026-06-14"
AUDITED_BY = "desk-jq"


# ---- canonical TreeNode scaffold + reference solution helpers -------------
# These are the reference *test-case generators* and *expected-output
# computers*. They use recursion freely because they are NOT model outputs —
# they only generate the expected_output for stdin_tests.

class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def _height(n):
    if n is None:
        return 0
    return 1 + max(_height(n.left), _height(n.right))


def _diameter(n):
    best = [0]
    def rec(x):
        if x is None: return 0
        L = rec(x.left); R = rec(x.right)
        best[0] = max(best[0], L + R)
        return 1 + max(L, R)
    rec(n)
    return best[0]


def _balanced(n):
    def rec(x):
        if x is None: return (True, 0)
        bl, hl = rec(x.left); br, hr = rec(x.right)
        return (bl and br and abs(hl - hr) <= 1, 1 + max(hl, hr))
    return rec(n)[0]


def _imbalance(n):
    """Max |left_height - right_height| over all internal nodes."""
    best = [0]
    def rec(x):
        if x is None: return 0
        L = rec(x.left); R = rec(x.right)
        best[0] = max(best[0], abs(L - R))
        return 1 + max(L, R)
    rec(n)
    return best[0]


def _lca(n, p, q):
    """LCA of values p and q in a binary tree of distinct ints."""
    if n is None: return None
    if n.val == p or n.val == q: return n
    L = _lca(n.left, p, q); R = _lca(n.right, p, q)
    if L and R: return n
    return L or R


def _same(a, b):
    if a is None and b is None: return True
    if a is None or b is None: return False
    return a.val == b.val and _same(a.left, b.left) and _same(a.right, b.right)


def _mirror(a, b):
    if a is None and b is None: return True
    if a is None or b is None: return False
    return a.val == b.val and _mirror(a.left, b.right) and _mirror(a.right, b.left)


def _symmetric(n):
    if n is None: return True
    return _mirror(n.left, n.right)


def _has_path_sum(n, target, acc=0):
    if n is None: return False
    acc += n.val
    if n.left is None and n.right is None:
        return acc == target
    return _has_path_sum(n.left, target, acc) or _has_path_sum(n.right, target, acc)


def _max_path_sum(n):
    NEG = float("-inf")
    best = [NEG]
    def rec(x):
        if x is None: return 0
        L = max(0, rec(x.left)); R = max(0, rec(x.right))
        best[0] = max(best[0], x.val + L + R)
        return x.val + max(L, R)
    rec(n)
    return best[0]


# Tree literal generators for test cases ------------------------------------

def _lit(n):
    """Generate the Python literal source for a TreeNode (or None)."""
    if n is None: return "None"
    if n.left is None and n.right is None:
        return f"TreeNode({n.val})"
    if n.right is None:
        return f"TreeNode({n.val}, {_lit(n.left)})"
    if n.left is None:
        return f"TreeNode({n.val}, None, {_lit(n.right)})"
    return f"TreeNode({n.val}, {_lit(n.left)}, {_lit(n.right)})"


# ---- shared scaffold prompts (class def + input description) --------------

TREE_NODE_SCAFFOLD = """\
You may assume the following class is provided (do NOT redefine it in your solution; just use the name):

    class TreeNode:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

The input is a single line containing a Python expression that constructs a TreeNode (using only `TreeNode(...)`, `None`, and integer literals). Parse it with:

    root = eval(input())

Write a complete Python program that reads the tree and prints the answer."""

CELL_SCAFFOLD = """\
You may assume the following class is provided (do NOT redefine it in your solution; just use the name):

    class Cell:
        def __init__(self, val, lo=None, hi=None):
            self.val = val
            self.lo = lo
            self.hi = hi

The input is a single line: a Python expression building a Cell from `Cell(...)`, `None`, and integer literals. Parse with `root = eval(input())`."""

KARYCELL_SCAFFOLD = """\
You may assume the following class is provided (do NOT redefine it in your solution; just use the name):

    class Cell:
        def __init__(self, val, children=None):
            self.val = val
            self.children = children or []

The input is a single line: a Python expression building a Cell from `Cell(val, [child1, child2, ...])`, with integer val and a list of child Cells (possibly empty). Parse with `root = eval(input())`."""


# ---- problem builders -----------------------------------------------------
# Each builder returns (problem_record, label_record). The problem_record
# follows the LCB schema (stdin_tests = [{input, output}, ...]).

def _problem(pid, *, source, label_overrides, prompt_body, stdin_tests,
             benchmark=None):
    rec = {
        "id": f"tree/{pid}",
        "benchmark": benchmark or source.replace("-", "_"),
        "mode": "stdin",
        "prompt": prompt_body + "\n\nReturn only Python source inside a single ```python code block.\n",
        "entry_point": None,
        "stdin_tests": stdin_tests,
        "canonical": None,
    }
    base_label = {
        "id": f"tree/{pid}",
        "reducibility_tier": label_overrides["reducibility_tier"],
        "reducing_channel": label_overrides["reducing_channel"],
        "representation": "pointer",
        "source": source,
        "contamination": label_overrides.get("contamination", "classic-memorized"),
        "problem_class": label_overrides["problem_class"],
        "honest_ceiling_rung": label_overrides["honest_ceiling_rung"],
        "irreducibility_confidence": label_overrides.get("irreducibility_confidence", "n/a"),
        "eligible_for_quadrant": label_overrides.get("eligible_for_quadrant", True),
        "perturbation_axis": label_overrides.get("perturbation_axis"),
        "audit_notes": label_overrides.get("audit_notes", ""),
        "audited_by": AUDITED_BY,
        "audited_at": AUDITED_AT,
    }
    return rec, base_label


def _tc(tree_or_pair, expected):
    """Build a stdin test case. tree_or_pair: TreeNode or tuple of trees / extras."""
    if isinstance(tree_or_pair, tuple):
        parts = []
        for x in tree_or_pair:
            parts.append(_lit(x) if isinstance(x, (TreeNode, type(None))) else str(x))
        inp = " | ".join(parts)
    else:
        inp = _lit(tree_or_pair)
    return {"input": inp, "output": str(expected)}


# Small reusable tree-shape generators
def chain_left(values):
    """Right-skewed left-spine: root has only .left chain."""
    n = None
    for v in reversed(values):
        n = TreeNode(v, n, None)
    return n


def chain_right(values):
    n = None
    for v in reversed(values):
        n = TreeNode(v, None, n)
    return n


def balanced(values):
    """Build a balanced tree from values in BFS order (using None for gaps)."""
    if not values: return None
    nodes = [TreeNode(v) if v is not None else None for v in values]
    for i in range(len(values)):
        if nodes[i] is None: continue
        L, R = 2*i+1, 2*i+2
        if L < len(values): nodes[i].left = nodes[L]
        if R < len(values): nodes[i].right = nodes[R]
    return nodes[0]


# ---- THE 20 TREE PROBLEMS -------------------------------------------------

def build() -> tuple[list[dict], list[dict]]:
    problems = []
    labels = []

    # --- 4.1 classic structural (8, all Irreducible × Pointer, structural) ---

    # 1. tree-height
    body = TREE_NODE_SCAFFOLD + "\n\nCompute and print the **height** of the tree: the number of nodes on the longest path from the root to a leaf. An empty tree has height 0; a single-node tree has height 1."
    tests = [
        _tc(balanced([3,9,20,None,None,15,7]), 3),
        _tc(None, 0),
        _tc(TreeNode(1), 1),
        _tc(chain_left([1,2,3,4,5,6]), 6),
        _tc(balanced([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), 4),
    ]
    p, l = _problem("height",
        source="tree-classic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "audit_notes": "Arbitrary-depth tree → height. Recursion or worklist required.",
        })
    problems.append(p); labels.append(l)

    # 2. tree-diameter
    body = TREE_NODE_SCAFFOLD + "\n\nCompute and print the **diameter** of the tree: the length of the longest path between any two nodes (the path's length is the number of edges in it). An empty or single-node tree has diameter 0."
    tests = [
        _tc(balanced([1,2,3,4,5]), 3),
        _tc(None, 0),
        _tc(TreeNode(1), 0),
        _tc(chain_left([1,2,3,4]), 3),
        _tc(balanced([1,2,3,4,5,None,None,6,7]), 4),
    ]
    p, l = _problem("diameter",
        source="tree-classic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "audit_notes": "Diameter via two-pass DFS or post-order DP.",
        })
    problems.append(p); labels.append(l)

    # 3. tree-balanced
    body = TREE_NODE_SCAFFOLD + "\n\nDetermine whether the tree is **height-balanced**: every subtree's left- and right-subtree heights differ by at most 1. Print `True` if balanced, else `False`. An empty tree is balanced."
    tests = [
        _tc(balanced([3,9,20,None,None,15,7]), True),
        _tc(chain_left([1,2,3,4]), False),
        _tc(None, True),
        _tc(TreeNode(1), True),
        _tc(balanced([1,2,3,4,5,6,7]), True),
    ]
    p, l = _problem("balanced",
        source="tree-classic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "audit_notes": "Subtree-balanced check requires height + balance recurrence.",
        })
    problems.append(p); labels.append(l)

    # 4. tree-lca
    body = TREE_NODE_SCAFFOLD + "\n\nThe input has THREE parts separated by ` | `: the tree expression, then two integer values `p` and `q`. Parse:\n\n    parts = input().split(' | ')\n    root = eval(parts[0])\n    p = int(parts[1]); q = int(parts[2])\n\nAll values in the tree are distinct; both `p` and `q` are guaranteed to appear. Print the **value** of the lowest common ancestor of the nodes with values `p` and `q`."
    tests = [
        _tc((balanced([3,5,1,6,2,0,8,None,None,7,4]), 5, 1), 3),
        _tc((balanced([3,5,1,6,2,0,8,None,None,7,4]), 5, 4), 5),
        _tc((TreeNode(1, TreeNode(2)), 1, 2), 1),
        _tc((chain_left([10,5,3,1]), 5, 1), 5),
    ]
    p, l = _problem("lca",
        source="tree-classic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "audit_notes": "LCA via recursive descent; no fold expresses the meet point.",
        })
    problems.append(p); labels.append(l)

    # 5. tree-same  (irreducibility_confidence = medium)
    body = TREE_NODE_SCAFFOLD + "\n\nThe input has TWO trees separated by ` | `:\n\n    parts = input().split(' | ')\n    a = eval(parts[0])\n    b = eval(parts[1])\n\nPrint `True` iff the two trees have the same structure AND the same values at every node, else `False`."
    tests = [
        _tc((balanced([1,2,3]), balanced([1,2,3])), True),
        _tc((balanced([1,2,3]), balanced([1,3,2])), False),
        _tc((None, None), True),
        _tc((TreeNode(1), TreeNode(2)), False),
        _tc((chain_left([1,2,3]), chain_left([1,2,3])), True),
    ]
    p, l = _problem("same",
        source="tree-classic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "medium",
            "audit_notes": "Pair-wise equal-structure check. Memorized — for small inputs the model can hand-unroll a finite sequence of attribute accesses, so confidence is medium.",
        })
    problems.append(p); labels.append(l)

    # 6. tree-symmetric  (irreducibility_confidence = medium)
    body = TREE_NODE_SCAFFOLD + "\n\nDetermine whether the tree is its own mirror image (a symmetric binary tree). Print `True` or `False`. An empty tree is symmetric."
    tests = [
        _tc(balanced([1,2,2,3,4,4,3]), True),
        _tc(balanced([1,2,2,None,3,None,3]), False),
        _tc(None, True),
        _tc(TreeNode(1), True),
        _tc(balanced([1,2,2,3,None,None,3]), True),
    ]
    p, l = _problem("symmetric",
        source="tree-classic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "medium",
            "audit_notes": "Mirror-pair check. Memorized — same hand-unroll caveat as `tree-same` for small inputs.",
        })
    problems.append(p); labels.append(l)

    # 7. tree-path-sum
    body = TREE_NODE_SCAFFOLD + "\n\nInput has the tree followed by an integer target K, separated by ` | `:\n\n    parts = input().split(' | ')\n    root = eval(parts[0])\n    K = int(parts[1])\n\nPrint `True` if there is any root-to-leaf path whose node-values sum to K; otherwise `False`. An empty tree has no path."
    tests = [
        _tc((balanced([5,4,8,11,None,13,4,7,2,None,None,None,1]), 22), True),
        _tc((balanced([1,2,3]), 5), False),
        _tc((None, 0), False),
        _tc((TreeNode(1), 1), True),
        _tc((chain_left([1,2,3,4]), 10), True),
    ]
    p, l = _problem("path-sum",
        source="tree-classic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "audit_notes": "Root-to-leaf accumulator with branching — needs recursion or stack worklist.",
        })
    problems.append(p); labels.append(l)

    # 8. tree-max-path-sum
    body = TREE_NODE_SCAFFOLD + "\n\nNode values may be negative. A path is any sequence of distinct connected nodes; it does not need to pass through the root. Print the **maximum path sum** over all non-empty paths. (Equivalently: the largest sum of values along any node-to-node path in the tree.)"
    tests = [
        _tc(balanced([1,2,3]), 6),
        _tc(balanced([-10,9,20,None,None,15,7]), 42),
        _tc(TreeNode(-3), -3),
        _tc(balanced([2,-1]), 2),
        _tc(chain_left([5,-2,3,-4,6]), 8),
    ]
    p, l = _problem("max-path-sum",
        source="tree-classic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "audit_notes": "Maximum node-to-node path sum via post-order DP (return best-downward, track best-meeting).",
        })
    problems.append(p); labels.append(l)

    # --- 4.2 perturbed (4 — one axis each) ---------------------------------

    # 9. tree-height-perturbed (axis = io-format: parenthesized string)
    body = (
        "You may assume the following class is provided (do NOT redefine it in your solution; just use the name):\n\n"
        "    class TreeNode:\n        def __init__(self, val, left=None, right=None):\n            self.val = val; self.left = left; self.right = right\n\n"
        "The input is a single line in parenthesized form, e.g. `(3(9)(20(15)(7)))`. Each tree T is either empty (`-`) or `(val[L][R])` "
        "where `val` is a non-negative integer, `L` and `R` are recursive sub-trees in the same format (omit when absent). You may use the following parser provided to you:\n\n"
        "    def parse(s):\n        i = [0]\n        def go():\n            if i[0] >= len(s) or s[i[0]] == '-':\n                if i[0] < len(s) and s[i[0]] == '-': i[0] += 1\n                return None\n            assert s[i[0]] == '('\n            i[0] += 1\n            j = i[0]\n            while j < len(s) and s[j].isdigit(): j += 1\n            val = int(s[i[0]:j]); i[0] = j\n            L = go() if i[0] < len(s) and s[i[0]] == '(' else None\n            R = go() if i[0] < len(s) and s[i[0]] == '(' else None\n            assert s[i[0]] == ')'; i[0] += 1\n            return TreeNode(val, L, R)\n        return go()\n\n"
        "Then `root = parse(input().strip())`.\n\nCompute and print the **height** of the parsed tree (number of nodes on the longest root-to-leaf path; empty tree has height 0)."
    )
    def _ph(t): return "-" if t is None else "(" + str(t.val) + ("" if t.left is None else "(" + _ph(t.left)[1:] if False else _ph(t.left)) + ("" if t.right is None else _ph(t.right)) + ")"
    # use a simpler emitter
    def _paren(t):
        if t is None: return "-"
        s = "(" + str(t.val)
        if t.left is not None or t.right is not None:
            s += _paren(t.left) if t.left is not None else "-"
            if t.right is not None:
                s += _paren(t.right)
        return s + ")"
    tests = [
        {"input": _paren(balanced([3,9,20,None,None,15,7])), "output": "3"},
        {"input": _paren(None), "output": "0"},
        {"input": _paren(TreeNode(1)), "output": "1"},
        {"input": _paren(chain_left([1,2,3,4,5,6])), "output": "6"},
    ]
    p, l = _problem("height-perturbed",
        source="tree-perturbed",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "contamination": "perturbed",
            "perturbation_axis": "io-format",
            "audit_notes": "Same downstream task as tree-height; only the I/O format is perturbed (parenthesized string instead of eval-of-literal). The parser is provided.",
        })
    problems.append(p); labels.append(l)

    # 10. tree-diameter-perturbed (axis = rename-and-shape)
    body = CELL_SCAFFOLD + "\n\nCompute and print the **diameter** of the tree (the length of the longest path between any two nodes, counted in edges). An empty or single-node tree has diameter 0. Note the renamed class `Cell` and attributes `lo` / `hi` in place of the usual `TreeNode` / `left` / `right`."
    # Build "Cell" expressions structurally identical to TreeNode expressions
    def _cell_lit(n):
        if n is None: return "None"
        if n.left is None and n.right is None: return f"Cell({n.val})"
        if n.right is None: return f"Cell({n.val}, {_cell_lit(n.left)})"
        if n.left is None: return f"Cell({n.val}, None, {_cell_lit(n.right)})"
        return f"Cell({n.val}, {_cell_lit(n.left)}, {_cell_lit(n.right)})"
    skewy = chain_right([1,2,3,4,5,6])
    tests = [
        {"input": _cell_lit(balanced([1,2,3,4,5])), "output": "3"},
        {"input": _cell_lit(None), "output": "0"},
        {"input": _cell_lit(TreeNode(1)), "output": "0"},
        {"input": _cell_lit(skewy), "output": "5"},   # right-skew chain, deliberately degenerate
        {"input": _cell_lit(chain_left([1,2,3,4,5])), "output": "4"},
    ]
    p, l = _problem("diameter-perturbed",
        source="tree-perturbed",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "contamination": "perturbed",
            "perturbation_axis": "rename-and-shape",
            "audit_notes": "Same downstream task (diameter) but Cell/.lo/.hi names and degenerate shape priors.",
        })
    problems.append(p); labels.append(l)

    # 11. tree-balanced-perturbed (axis = compositional)
    body = TREE_NODE_SCAFFOLD + "\n\nFor each internal node compute |height(left) - height(right)|. Print the **maximum imbalance** value across all internal nodes (= 0 if the tree is empty or has no internal node). This is NOT a yes/no question — output the numeric imbalance."
    tests = [
        _tc(balanced([3,9,20,None,None,15,7]), 1),
        _tc(chain_left([1,2,3,4,5]), 4),
        _tc(None, 0),
        _tc(TreeNode(1), 0),
        _tc(balanced([1,2,3,4,5,6,7,8]), 1),
    ]
    p, l = _problem("balanced-perturbed",
        source="tree-perturbed",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "contamination": "perturbed",
            "perturbation_axis": "compositional",
            "audit_notes": "Compositional twist on `tree-balanced`: output the imbalance value (integer), not a boolean.",
        })
    problems.append(p); labels.append(l)

    # 12. tree-lca-perturbed (axis = generalisation: K-ary, 3 targets)
    body = KARYCELL_SCAFFOLD + "\n\nInput has FOUR parts separated by ` | `:\n\n    parts = input().split(' | ')\n    root = eval(parts[0])\n    p, q, r = int(parts[1]), int(parts[2]), int(parts[3])\n\nAll node values are distinct; p, q, r all appear. Print the **value** of the lowest common ancestor of the three nodes."
    # Build K-ary literal helper
    def _kary_lit(n):
        if n is None: return "None"
        if not n.right:
            return f"Cell({n.val})"  # treat None right as no children (we treat .right as children list flag)
        return ""
    # For simplicity: a K-ary helper that takes a (val, [children]) python tuple
    def kary(val, children=None):
        n = TreeNode(val)
        n.kary = children or []
        return n
    def _kary_emit(n):
        if n is None: return "None"
        ch = getattr(n, "kary", [])
        if not ch: return f"Cell({n.val})"
        return f"Cell({n.val}, [{', '.join(_kary_emit(c) for c in ch)}])"
    def _kary_lca(n, targets):
        if n is None: return None
        if n.val in targets:
            return n
        found = []
        for c in getattr(n, "kary", []):
            r = _kary_lca(c, targets)
            if r is not None: found.append(r)
        if len(found) >= 2: return n
        return found[0] if found else None

    # construct a K-ary test tree
    t1 = kary(1, [kary(2, [kary(5), kary(6)]), kary(3, [kary(7, [kary(11)])]), kary(4, [kary(8), kary(9), kary(10)])])
    t2 = kary(1, [kary(2), kary(3), kary(4)])
    tests = [
        {"input": f"{_kary_emit(t1)} | 5 | 11 | 9", "output": str(_kary_lca(t1, {5,11,9}).val)},
        {"input": f"{_kary_emit(t1)} | 5 | 6 | 11", "output": str(_kary_lca(t1, {5,6,11}).val)},
        {"input": f"{_kary_emit(t2)} | 2 | 3 | 4", "output": str(_kary_lca(t2, {2,3,4}).val)},
        {"input": f"{_kary_emit(kary(7,[kary(2),kary(3)]))} | 2 | 3 | 7", "output": "7"},
    ]
    p, l = _problem("lca-perturbed",
        source="tree-perturbed",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "irreducible",
            "reducing_channel": "N/A-irreducible",
            "problem_class": "structural",
            "honest_ceiling_rung": -1,
            "irreducibility_confidence": "high",
            "contamination": "perturbed",
            "perturbation_axis": "generalisation",
            "audit_notes": "K-ary generalisation of LCA with three targets instead of two; .children list instead of .left/.right.",
        })
    problems.append(p); labels.append(l)

    # --- 4.3 reducible synthetic (5 genuinely-reducible + 3 R5-bait) -------

    # 13. root-value (honest_ceiling_rung 5, value-collection)
    body = TREE_NODE_SCAFFOLD + "\n\nPrint the value of the root. (The tree is guaranteed to be non-empty.)"
    tests = [_tc(TreeNode(7), 7),
             _tc(TreeNode(3, TreeNode(9), TreeNode(20)), 3),
             _tc(TreeNode(0), 0)]
    p, l = _problem("root-value",
        source="tree-synthetic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "closed-form",
            "reducing_channel": "none",
            "problem_class": "value-collection",
            "honest_ceiling_rung": 5,
            "contamination": "perturbed",
            "audit_notes": "Just `print(root.val)`. R5-compliant trivially.",
        })
    problems.append(p); labels.append(l)

    # 14. direct-children-sum
    body = TREE_NODE_SCAFFOLD + "\n\nThe root is guaranteed to have BOTH a left and a right child (each is itself a TreeNode). Print `root.left.val + root.right.val`."
    tests = [_tc(TreeNode(0, TreeNode(3), TreeNode(4)), 7),
             _tc(TreeNode(0, TreeNode(-1), TreeNode(1)), 0),
             _tc(TreeNode(0, TreeNode(100), TreeNode(200)), 300)]
    p, l = _problem("direct-children-sum",
        source="tree-synthetic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "closed-form",
            "reducing_channel": "none",
            "problem_class": "value-collection",
            "honest_ceiling_rung": 5,
            "contamination": "perturbed",
            "audit_notes": "Closed-form: `root.left.val + root.right.val`.",
        })
    problems.append(p); labels.append(l)

    # 15. left-spine-2
    body = TREE_NODE_SCAFFOLD + "\n\nThe tree is guaranteed to have a `root.left.left` node (i.e. the left spine reaches depth ≥ 3). Print `root.left.left.val`."
    tests = [_tc(chain_left([1,2,3,4]), 3),
             _tc(chain_left([10,20,30]), 30),
             _tc(TreeNode(1, TreeNode(2, TreeNode(7, TreeNode(8)))), 7)]
    p, l = _problem("left-spine-2",
        source="tree-synthetic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "closed-form",
            "reducing_channel": "none",
            "problem_class": "value-collection",
            "honest_ceiling_rung": 5,
            "contamination": "perturbed",
            "audit_notes": "Closed-form: `root.left.left.val`.",
        })
    problems.append(p); labels.append(l)

    # 16. bounded-depth-find
    body = TREE_NODE_SCAFFOLD + "\n\nThe tree is guaranteed to have depth at most 3 (i.e. root.left.left.left, etc. are None). Input is the tree followed by an integer target K, separated by ` | `. Print `True` if any node in the tree has value K, else `False`. Because depth ≤ 3 you may hand-unroll the search."
    tests = [
        {"input": _lit(balanced([1,2,3,4,5,6,7])) + " | 5", "output": "True"},
        {"input": _lit(balanced([1,2,3])) + " | 9", "output": "False"},
        {"input": _lit(TreeNode(1)) + " | 1", "output": "True"},
        {"input": _lit(None) + " | 1", "output": "False"},
    ]
    p, l = _problem("bounded-depth-find",
        source="tree-synthetic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "builtin-reducible",
            "reducing_channel": "functional-helper",
            "problem_class": "value-collection",
            "honest_ceiling_rung": 5,
            "contamination": "perturbed",
            "audit_notes": "Depth ≤ 3 → hand-unroll into a finite OR of attribute comparisons.",
        })
    problems.append(p); labels.append(l)

    # 17. flat-twochild
    body = TREE_NODE_SCAFFOLD + "\n\nThe tree is guaranteed to be exactly the shape `TreeNode(a, TreeNode(b), TreeNode(c))` — 3 nodes, root with two leaf children. Print `a + b + c`."
    tests = [_tc(TreeNode(1, TreeNode(2), TreeNode(3)), 6),
             _tc(TreeNode(10, TreeNode(0), TreeNode(0)), 10),
             _tc(TreeNode(-5, TreeNode(5), TreeNode(0)), 0)]
    p, l = _problem("flat-twochild",
        source="tree-synthetic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "closed-form",
            "reducing_channel": "none",
            "problem_class": "value-collection",
            "honest_ceiling_rung": 5,
            "contamination": "perturbed",
            "audit_notes": "Fixed shape — closed-form sum.",
        })
    problems.append(p); labels.append(l)

    # 18. tree-deepcopy-len  (R5-bait, eligible_for_quadrant=False)
    body = (
        TREE_NODE_SCAFFOLD + "\n\nDefine `node_to_dict(n)` that converts a TreeNode (or None) to a nested dict; then print "
        "`len(json.dumps(node_to_dict(root)))`. Constraint hint: the natural no-loop path is to delegate the recursive walk to "
        "`json.dumps`."
    )
    # expected = len(json.dumps(_to_dict(t))) for each t
    def _to_dict(n):
        if n is None: return None
        return {"v": n.val, "l": _to_dict(n.left), "r": _to_dict(n.right)}
    tests = []
    for t in [balanced([1,2,3]),
              TreeNode(0, TreeNode(0), TreeNode(0)),
              chain_left([1,2,3]),
              None,
              TreeNode(123, TreeNode(456), TreeNode(789))]:
        tests.append({"input": _lit(t),
                      "output": str(len(json.dumps(_to_dict(t))))})
    p, l = _problem("deepcopy-len",
        source="tree-synthetic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "library-reducible",
            "reducing_channel": "library-recursion",
            "problem_class": "library-recursion-reducible",
            "honest_ceiling_rung": 4,
            "eligible_for_quadrant": False,
            "contamination": "perturbed",
            "audit_notes": "R5-bait: the canonical no-loop path is library-recursion via json.dumps. R0-R4 compliant; R5 catches the launder. EXCLUDED from Task 4 (eligible_for_quadrant=False).",
        })
    problems.append(p); labels.append(l)

    # 19. tree-json-stringify  (R5-bait via json.dumps + default lambda)
    body = TREE_NODE_SCAFFOLD + (
        "\n\nPrint the JSON serialisation of the tree obtained by\n\n"
        "    json.dumps(root, default=lambda o: o.__dict__)\n\n"
        "This recursively walks the tree through each node's `__dict__`. "
        "Output is a single line — the JSON string itself, with no trailing "
        "newline beyond the one `print` adds."
    )
    def _to_dict_walk(t):
        if t is None: return None
        return {"val": t.val, "left": _to_dict_walk(t.left), "right": _to_dict_walk(t.right)}
    tests = []
    for t in [TreeNode(1),
              TreeNode(1, TreeNode(2)),
              TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7))),
              chain_left([1,2,3]),
              balanced([1,2,3])]:
        tests.append({"input": _lit(t),
                      "output": json.dumps(_to_dict_walk(t))})
    p, l = _problem("json-stringify",
        source="tree-synthetic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "library-reducible",
            "reducing_channel": "library-recursion",
            "problem_class": "library-recursion-reducible",
            "honest_ceiling_rung": 4,
            "eligible_for_quadrant": False,
            "contamination": "perturbed",
            "audit_notes": "R5-bait via `json.dumps(root, default=lambda o: o.__dict__)` — the lambda + __dict__ access lets json walk the tree recursively in C. EXCLUDED from Task 4.",
        })
    problems.append(p); labels.append(l)

    # 20. tree-stringify-class  (R5-bait, with self-walking __repr__)
    body = (
        "You may assume the following class is provided (do NOT redefine it in your solution; just use the name):\n\n"
        "    class StringyNode:\n"
        "        def __init__(self, val, left=None, right=None):\n"
        "            self.val = val; self.left = left; self.right = right\n"
        "        def __repr__(self):\n"
        "            return f'<{self.val}|{self.left}|{self.right}>'\n\n"
        "The input is a single line — a Python expression building a StringyNode using `StringyNode(...)`, `None`, and integer literals. Parse with `root = eval(input())` and print `repr(root)` directly. The canonical no-loop solution is exactly that, because StringyNode's `__repr__` recursively walks the tree for you."
    )
    class _SN:
        def __init__(self, v, l=None, r=None): self.val=v; self.left=l; self.right=r
        def __repr__(self): return f"<{self.val}|{self.left}|{self.right}>"
    tests = [
        {"input": "StringyNode(1)", "output": repr(_SN(1))},
        {"input": "StringyNode(1, StringyNode(2))", "output": repr(_SN(1, _SN(2)))},
        {"input": "StringyNode(3, StringyNode(9), StringyNode(20, StringyNode(15), StringyNode(7)))",
         "output": repr(_SN(3, _SN(9), _SN(20, _SN(15), _SN(7))))},
    ]
    p, l = _problem("stringify-class",
        source="tree-synthetic",
        prompt_body=body,
        stdin_tests=tests,
        label_overrides={
            "reducibility_tier": "library-reducible",
            "reducing_channel": "library-recursion",
            "problem_class": "library-recursion-reducible",
            "honest_ceiling_rung": 4,
            "eligible_for_quadrant": False,
            "contamination": "perturbed",
            "audit_notes": "R5-bait: class defines a self-walking __repr__; printing `repr(root)` is the canonical solution. R5 catches the class definition itself. EXCLUDED from Task 4.",
        })
    problems.append(p); labels.append(l)

    return problems, labels


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    problems, labels = build()
    p_path = OUT / "problems_trees.jsonl"
    l_path = OUT / "labels_trees.jsonl"
    with p_path.open("w") as f:
        for r in problems:
            f.write(json.dumps(r) + "\n")
    with l_path.open("w") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")
    print(f"wrote: {p_path.relative_to(ROOT)}  (n={len(problems)})")
    print(f"wrote: {l_path.relative_to(ROOT)}  (n={len(labels)})")

    # Summary
    by_source = {}
    by_class = {}
    by_tier = {}
    for l in labels:
        by_source[l["source"]] = by_source.get(l["source"], 0) + 1
        c = l.get("problem_class") or "_"
        by_class[c] = by_class.get(c, 0) + 1
        by_tier[l["reducibility_tier"]] = by_tier.get(l["reducibility_tier"], 0) + 1
    print()
    print("by source:", by_source)
    print("by problem_class:", by_class)
    print("by reducibility_tier:", by_tier)
    ineligible = [l["id"] for l in labels if not l["eligible_for_quadrant"]]
    print(f"eligible_for_quadrant=False: {len(ineligible)} → {ineligible}")


if __name__ == "__main__":
    main()
