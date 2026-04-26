"""Unit tests for the AST checks. Run: python -m pytest src/test_ast_checks.py"""

from ast_checks import (
    check_no_recursion,
    check_no_classes,
    check_no_nested_functions,
    check_no_mutation,
    check_no_loops,
    check_no_helpers,
    check_stdlib_whitelist,
)


# --- no_loops ---

def test_no_loops_pass_comprehension():
    assert check_no_loops("def f(n): return sum(i*i for i in range(n))")

def test_no_loops_pass_no_iteration():
    assert check_no_loops("def f(a, b): return a + b")

def test_no_loops_pass_recursion():
    # recursion is fine here — separate constraint
    assert check_no_loops("def f(n): return 1 if n<=0 else f(n-1)")

def test_no_loops_fail_for():
    assert not check_no_loops("def f(n):\n    s=0\n    for i in range(n): s+=i\n    return s")

def test_no_loops_fail_while():
    assert not check_no_loops("def f(n):\n    s=0\n    while n>0:\n        s+=n; n-=1\n    return s")

def test_no_loops_fail_for_in_comp_with_inner_for_stmt():
    # for-statement inside a function body, even if there's also a comprehension
    assert not check_no_loops("def f(n):\n    [i for i in range(n)]\n    for i in range(n): pass\n    return 0")


# --- no_helpers ---

def test_no_helpers_pass_single_func():
    assert check_no_helpers("def main():\n    return 1")

def test_no_helpers_pass_with_entry():
    assert check_no_helpers("def solve(): return 1", entry_point="solve")

def test_no_helpers_pass_lambda_inside():
    # lambdas inside the body are not separate module-level defs
    assert check_no_helpers("def main():\n    f = lambda x: x+1\n    return f(3)")

def test_no_helpers_fail_two_module_funcs():
    assert not check_no_helpers("def helper(x): return x+1\ndef main(): return helper(2)")

def test_no_helpers_fail_wrong_entry():
    assert not check_no_helpers("def other(): return 1", entry_point="solve")

def test_no_helpers_pass_zero_funcs_no_entry():
    # stdin-style program with top-level code is allowed when no entry_point given
    assert check_no_helpers("import sys\nprint(sys.stdin.read())")


# --- no_recursion ---

def test_no_recursion_pass_iterative():
    assert check_no_recursion("def f(n):\n    s = 0\n    for i in range(n): s = s + i\n    return s")

def test_no_recursion_pass_calls_builtin():
    assert check_no_recursion("def f(xs): return sum(xs)")

def test_no_recursion_pass_calls_external_lib():
    assert check_no_recursion("import math\ndef f(x): return math.sqrt(x)")

def test_no_recursion_fail_self_call():
    assert not check_no_recursion("def f(n):\n    if n <= 0: return 0\n    return f(n-1) + 1")

def test_no_recursion_fail_helper_route():
    src = "def helper(n): return f(n-1)\ndef f(n):\n    if n<=0: return 0\n    return helper(n)"
    assert not check_no_recursion(src)

def test_no_recursion_fail_method_attr():
    # method-style attr call to a local name (rare, but the check should catch it)
    src = "def f(n):\n    if n<=0: return 0\n    return self.f(n-1)"
    assert not check_no_recursion(src)


# --- no_classes ---

def test_no_classes_pass_plain():
    assert check_no_classes("def f(): return 1")

def test_no_classes_pass_imports():
    assert check_no_classes("from collections import Counter\ndef f(): return Counter('abc')")

def test_no_classes_pass_dataclass_call():
    # using a class as a value isn't defining one
    assert check_no_classes("def f(): return list()")

def test_no_classes_fail_basic():
    assert not check_no_classes("class A: pass\ndef f(): return A()")

def test_no_classes_fail_inside_function():
    assert not check_no_classes("def f():\n    class Inner: pass\n    return Inner()")

def test_no_classes_fail_dataclass():
    assert not check_no_classes("from dataclasses import dataclass\n@dataclass\nclass A: x: int")


# --- no_nested_functions ---

def test_no_nested_pass_module_level_funcs():
    assert check_no_nested_functions("def a(): return 1\ndef b(): return 2")

def test_no_nested_pass_no_funcs():
    assert check_no_nested_functions("x = 1\ny = 2")

def test_no_nested_pass_class_with_methods():
    # methods are inside ClassDef, not FunctionDef — should pass
    assert check_no_nested_functions("class A:\n    def m(self): return 1")

def test_no_nested_fail_inner_def():
    assert not check_no_nested_functions("def a():\n    def b(): return 1\n    return b()")

def test_no_nested_fail_lambda():
    assert not check_no_nested_functions("def a(): return (lambda x: x+1)(3)")

def test_no_nested_fail_deeply_nested():
    src = "def a():\n    def b():\n        def c(): return 1\n        return c()\n    return b()"
    assert not check_no_nested_functions(src)


# --- no_mutation ---

def test_no_mutation_pass_comprehension():
    assert check_no_mutation("def f(n): return [i*2 for i in range(n)]")

def test_no_mutation_pass_generator_expr():
    assert check_no_mutation("def f(n): return sum(i for i in range(n))")

def test_no_mutation_pass_immutable_assign():
    assert check_no_mutation("def f():\n    x = 1\n    y = x + 1\n    return y")

def test_no_mutation_fail_append():
    assert not check_no_mutation("def f(n):\n    out=[]\n    for i in range(n): out.append(i)\n    return out")

def test_no_mutation_fail_subscript_assign():
    assert not check_no_mutation("def f(n):\n    a=[0]*n\n    a[0]=1\n    return a")

def test_no_mutation_fail_aug_subscript():
    assert not check_no_mutation("def f(d, k):\n    d[k] += 1\n    return d")


# --- stdlib_whitelist ---

def test_stdlib_pass_no_imports():
    assert check_stdlib_whitelist("def f(): return 1")

def test_stdlib_pass_allowed_imports():
    src = "import math\nfrom collections import Counter\nfrom itertools import chain\nimport re\ndef f(): return math.pi"
    assert check_stdlib_whitelist(src)

def test_stdlib_pass_dotted_allowed():
    assert check_stdlib_whitelist("import collections.abc\ndef f(): return 1")

def test_stdlib_fail_numpy():
    assert not check_stdlib_whitelist("import numpy as np\ndef f(): return np.zeros(3)")

def test_stdlib_fail_from_other():
    assert not check_stdlib_whitelist("from functools import reduce\ndef f(xs): return reduce(lambda a,b:a+b, xs)")

def test_stdlib_fail_typing():
    assert not check_stdlib_whitelist("from typing import List\ndef f(xs: List[int]) -> int: return sum(xs)")
