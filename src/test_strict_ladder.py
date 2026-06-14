"""Tests for the five-rung strict-judge ladder.

Each test asserts the rung at which a specific construct first becomes
non-compliant, so the ladder's monotonicity invariant (R0 ⊇ R1 ⊇ R2 ⊇ R3 ⊇ R4)
is enforced from one side and the per-construct attribution is enforced
from the other.
"""
from __future__ import annotations

import pytest

from strict_ladder import judge_ladder, RUNG_KEYS


def _rungs(source: str) -> tuple[bool, bool, bool, bool, bool, bool]:
    r = judge_ladder(source)
    return (r.R0_loose, r.R1_strict_min, r.R2_strict_mid_comp,
            r.R3_strict_mid_range, r.R4_strict_max,
            r.R5_strict_max_plus_libraryrecursion)


# ---- monotonicity over a corpus of small cases ---------------------------

CORPUS = [
    "print(0)",
    "print(sum([1,2,3]))",
    "print(sum(x*2 for x in [1,2,3]))",
    "print([x*2 for x in [1,2,3]])",
    "print(sum(range(10)))",
    "print(next(iter([1,2,3])))",
    "from itertools import chain\nprint(list(chain([1],[2])))",
    "import functools\nprint(functools.reduce(lambda a,b:a+b,[1,2,3]))",
    "for x in [1,2,3]:\n    print(x)",
    "def f(n):\n    return f(n-1) if n else 0\nprint(f(5))",
    "while True:\n    break",
    "print(list(map(int, '1 2 3'.split())))",
    "print(any(x>0 for x in [1,-1]))",
    # R5 corpus
    "import json\nprint(json.dumps({'a':[1,2]}))",
    "import pickle\nprint(pickle.dumps(0))",
    "import copy\nprint(copy.deepcopy([1,[2]]))",
    "from json import dumps\nprint(dumps({'a':1}))",
    "import pprint\npprint.pprint([1,[2]])",
    "import ast\nfor n in ast.walk(ast.parse('1')): pass",
    "class T:\n  def __init__(self, v, l=None, r=None):\n    self.val=v; self.left=l; self.right=r\n  def __repr__(self):\n    return f'T({self.val},{self.left},{self.right})'\nprint(T(0))",
    "print(repr({1:2}))",        # bare repr — needs review, not definite
    "print(f'{1}')",              # f-string with format value — needs review
]


def test_monotonic_over_corpus():
    for src in CORPUS:
        r0, r1, r2, r3, r4, r5 = _rungs(src)
        # implication chain — strictly more restrictive
        assert (not r5) or r4, f"R5 without R4: {src!r}"
        assert (not r4) or r3, f"R4 without R3: {src!r}"
        assert (not r3) or r2, f"R3 without R2: {src!r}"
        assert (not r2) or r1, f"R2 without R1: {src!r}"
        assert (not r1) or r0, f"R1 without R0: {src!r}"


# ---- per-rung attribution --------------------------------------------------

def test_R0_clean_passes_all_rungs():
    # No iteration of any flavor — should be compliant at R4.
    src = "print(42)"
    assert _rungs(src) == (True, True, True, True, True, True)


def test_for_loop_violates_R0():
    src = "for x in [1,2,3]:\n    print(x)"
    assert _rungs(src) == (False, False, False, False, False, False)


def test_while_loop_violates_R0():
    src = "while True:\n    break"
    assert _rungs(src) == (False, False, False, False, False, False)


def test_recursion_violates_R0():
    src = "def f(n):\n    return f(n-1) if n else 0\nprint(f(5))"
    assert _rungs(src) == (False, False, False, False, False, False)


def test_bare_map_violates_R1_only():
    # map() is allowed at R0 (no for/while/recursion), excluded at R1+.
    src = "print(list(map(int, '1 2 3'.split())))"
    assert _rungs(src) == (True, False, False, False, False, False)


def test_functools_reduce_violates_R1_only():
    src = "import functools\nprint(functools.reduce(lambda a,b:a+b,[1,2,3]))"
    assert _rungs(src) == (True, False, False, False, False, False)


def test_itertools_attribute_violates_R1_only():
    src = "from itertools import chain\nprint(list(chain([1],[2])))"
    # `chain` was imported as a bare name; check the attribute path explicitly:
    src2 = "import itertools\nprint(list(itertools.chain([1],[2])))"
    # The bare-name `chain` form does NOT fire R1 (we only flag itertools.X attribute calls),
    # so use the attribute form to assert R1 cleanly:
    assert _rungs(src2) == (True, False, False, False, False, False)


def test_sum_over_generator_violates_R1_only():
    # genexp argument → R1 flag (this is the canonical functional fold)
    src = "print(sum(x*2 for x in [1,2,3]))"
    assert _rungs(src) == (True, False, False, False, False, False)


def test_list_comprehension_violates_R2_only():
    # comprehensions are allowed at R0 and R1, excluded at R2+.
    src = "print([x*2 for x in [1,2,3]])"
    assert _rungs(src) == (True, True, False, False, False, False)


def test_sum_of_list_comprehension_violates_R2_not_R1():
    # sum(<comprehension>) — comprehension is the iterator, not a genexp.
    # R1's "agg over genexp" doesn't fire (it's a list literal-comp), but R2
    # catches the comprehension itself.
    src = "print(sum([x*2 for x in [1,2,3]]))"
    assert _rungs(src) == (True, True, False, False, False, False)


def test_sum_of_range_violates_R3_only():
    # sum(range(...)) — no genexp, no comprehension, just a fold over range.
    src = "print(sum(range(10)))"
    assert _rungs(src) == (True, True, True, False, False, False)


def test_any_of_range_violates_R3_only():
    src = "print(any(range(0)))"
    assert _rungs(src) == (True, True, True, False, False, False)


def test_bare_next_iter_violates_R4_only():
    src = "print(next(iter([1,2,3])))"
    # `next` and `iter` together — both flag R4.
    assert _rungs(src) == (True, True, True, True, False, False)


def test_top_rung_indexing():
    cases = [
        ("print(42)", 5),
        ("import json\nprint(json.dumps(0))", 4),
        ("print(next(iter([1])))", 3),
        ("print(sum(range(10)))", 2),
        ("print([x for x in [1]])", 1),
        ("print(sum(x for x in [1]))", 0),
        ("for x in [1]: print(x)", -1),
    ]
    for src, expected_top in cases:
        r = judge_ladder(src)
        assert r.top_rung == expected_top, f"src={src!r} top={r.top_rung} expected={expected_top}"


def test_parse_failure_returns_minus_one():
    r = judge_ladder("def f(\n")
    assert not r.parses
    assert r.top_rung == -1
    assert not r.R0_loose


def test_helper_function_chain_is_not_recursion():
    # solve() -> helper() should NOT count as recursion (no cycle).
    src = (
        "def helper(x):\n"
        "    return x + 1\n"
        "def solve():\n"
        "    return helper(0)\n"
        "print(solve())\n"
    )
    r = judge_ladder(src)
    assert not r.has_recursion
    assert r.R0_loose


def test_mutual_recursion_is_recursion():
    src = (
        "def a(n):\n"
        "    return b(n-1) if n else 0\n"
        "def b(n):\n"
        "    return a(n-1) if n else 0\n"
        "print(a(5))\n"
    )
    r = judge_ladder(src)
    assert r.has_recursion
    assert not r.R0_loose


# ---- R5 library-recursion ----------------------------------------------

def test_json_dumps_violates_R5_only():
    src = "import json\nprint(json.dumps({'a':[1,2]}))"
    assert _rungs(src) == (True, True, True, True, True, False)


def test_pickle_dumps_violates_R5_only():
    src = "import pickle\nprint(pickle.dumps(0))"
    assert _rungs(src) == (True, True, True, True, True, False)


def test_copy_deepcopy_violates_R5_only():
    src = "import copy\nprint(copy.deepcopy([1,[2]]))"
    assert _rungs(src) == (True, True, True, True, True, False)


def test_pprint_pprint_violates_R5_only():
    src = "import pprint\npprint.pprint([1,[2,3]])"
    assert _rungs(src) == (True, True, True, True, True, False)


def test_ast_walk_violates_R5_via_for():
    # ast.walk(...) returns an iterator. Used in a for, R0 fails — and we
    # also flag library-recursion. Top rung still -1 (no R0).
    src = "import ast\nfor n in ast.walk(ast.parse('1')): pass"
    r = judge_ladder(src)
    assert r.has_library_recursion
    assert not r.R0_loose


def test_from_json_import_dumps_resolves_to_library_recursion():
    src = "from json import dumps\nprint(dumps({'a':1}))"
    assert _rungs(src) == (True, True, True, True, True, False)


def test_self_walking_repr_violates_R5():
    src = (
        "class T:\n"
        "  def __init__(self, v, l=None, r=None):\n"
        "    self.val=v; self.left=l; self.right=r\n"
        "  def __repr__(self):\n"
        "    return f'T({self.val},{self.left},{self.right})'\n"
        "print(T(0))\n"
    )
    r = judge_ladder(src)
    assert r.has_library_recursion
    assert not r.R5_strict_max_plus_libraryrecursion
    # The class __init__ does self.left=l (an Attribute write on self, not a
    # read of .left for traversal), so the R4 check is unaffected.


def test_bare_repr_flags_needs_review_not_R5():
    src = "print(repr({1: 2}))"
    r = judge_ladder(src)
    assert not r.has_library_recursion
    assert r.needs_manual_review
    # Still R5-compliant since not a definite hit.
    assert r.R5_strict_max_plus_libraryrecursion


def test_fstring_flags_needs_review_not_R5():
    src = "x = 1\nprint(f'value={x}')"
    r = judge_ladder(src)
    assert not r.has_library_recursion
    assert r.needs_manual_review
    assert r.R5_strict_max_plus_libraryrecursion


def test_plain_fstring_no_format_value_does_not_flag():
    src = "print('hello')"
    r = judge_ladder(src)
    assert not r.needs_manual_review
    assert r.R5_strict_max_plus_libraryrecursion


def test_json_encoder_chained_call_violates_R5():
    # json.JSONEncoder().encode(x): chained call where the inner is json.JSONEncoder.
    src = "import json\nprint(json.JSONEncoder().encode({'a':1}))"
    r = judge_ladder(src)
    assert r.has_library_recursion
    assert not r.R5_strict_max_plus_libraryrecursion


def test_non_tree_class_with_simple_repr_does_not_violate_R5():
    # A class with a __repr__ that does NOT walk a tree edge — should not
    # flag library-recursion.
    src = (
        "class Point:\n"
        "  def __init__(self, x, y):\n"
        "    self.x = x; self.y = y\n"
        "  def __repr__(self):\n"
        "    return 'Point'\n"
        "print(Point(1,2))\n"
    )
    r = judge_ladder(src)
    assert not r.has_library_recursion
    assert r.R5_strict_max_plus_libraryrecursion
