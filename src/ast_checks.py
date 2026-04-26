"""AST-based constraint checks. Each returns True iff the constraint is obeyed."""

from __future__ import annotations

import ast
from typing import Callable, Dict


_MUTATING_METHODS = frozenset({
    "append", "pop", "extend", "insert", "remove", "clear",
    "update", "setdefault", "popitem", "sort", "reverse",
    "__setitem__", "__delitem__", "add", "discard",
})

_STDLIB_WHITELIST = frozenset({"math", "collections", "itertools", "re"})


def _try_parse(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def check_no_recursion(source: str) -> bool:
    """True iff no locally-defined function calls another locally-defined function by name.

    Routing through helpers still counts as recursion (a calls b, b calls a is recursion).
    We over-approximate: any call from inside any local function to any local function
    name is treated as recursive. This is conservative but matches intent.
    """
    tree = _try_parse(source)
    if tree is None:
        return False

    local_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(func):
            if isinstance(inner, ast.Call):
                callee = inner.func
                if isinstance(callee, ast.Name) and callee.id in local_names:
                    return False
                if isinstance(callee, ast.Attribute) and callee.attr in local_names:
                    return False
    return True


def check_no_classes(source: str) -> bool:
    tree = _try_parse(source)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return False
    return True


def check_no_nested_functions(source: str) -> bool:
    tree = _try_parse(source)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is node:
                    continue
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return False
    return True


def check_no_mutation(source: str) -> bool:
    """No subscript/attribute Assign or AugAssign, no calls to known mutating methods."""
    tree = _try_parse(source)
    if tree is None:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    return False
                if isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Subscript):
                            return False
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Subscript):
                return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _MUTATING_METHODS:
                return False
    return True


def check_no_loops(source: str) -> bool:
    """No `for` or `while` loops. Comprehensions are allowed (they encode iteration declaratively)."""
    tree = _try_parse(source)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            return False
    return True


def check_no_helpers(source: str, entry_point: str | None = None) -> bool:
    """Only one function definition at module level (the entry point).

    For stdin-style problems with no entry_point, allow exactly one module-level FunctionDef.
    Lambdas and nested defs inside that one function are allowed (this is a "no helpers" not
    a "no closures" rule). Methods inside classes are NOT covered (we don't expect classes here).
    """
    tree = _try_parse(source)
    if tree is None:
        return False
    module_funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(module_funcs) > 1:
        return False
    if entry_point is not None and module_funcs:
        if module_funcs[0].name != entry_point:
            return False
    return True


def check_stdlib_whitelist(source: str) -> bool:
    tree = _try_parse(source)
    if tree is None:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _STDLIB_WHITELIST:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue  # `from . import x` — relative; skip
            top = node.module.split(".")[0]
            if top not in _STDLIB_WHITELIST:
                return False
    return True


def check_no_loops_no_recursion(source: str) -> bool:
    return check_no_loops(source) and check_no_recursion(source)


CHECKS: Dict[str, Callable[[str], bool]] = {
    "no_recursion": check_no_recursion,
    "no_classes": check_no_classes,
    "no_nested_functions": check_no_nested_functions,
    "no_mutation": check_no_mutation,
    "no_loops": check_no_loops,
    "no_helpers": check_no_helpers,
    "no_loops_no_recursion": check_no_loops_no_recursion,
    "stdlib_whitelist": check_stdlib_whitelist,
}


CONSTRAINT_INSTRUCTIONS: Dict[str, str] = {
    "no_recursion": (
        "Do not use recursion in your solution. No function may call itself, "
        "directly or indirectly through helper functions."
    ),
    "no_classes": "Do not define any classes in your solution. Do not use the `class` keyword.",
    "no_nested_functions": (
        "Do not define any nested functions. All functions must be defined at module level. "
        "Do not use lambdas or inner def statements."
    ),
    "no_mutation": (
        "Do not mutate any collection. Do not assign to subscripts (e.g., `a[i] = x`), "
        "and do not call mutating methods like `.append`, `.pop`, `.extend`, `.insert`, "
        "`.remove`, `.clear`, `.update`, `.sort`, `.reverse`. Use comprehensions and "
        "functional constructs instead."
    ),
    "no_loops": (
        "Do not use any `for` or `while` loops. List/set/dict/generator comprehensions are allowed."
    ),
    "no_helpers": (
        "Do not define helper functions. Your solution must use exactly one module-level "
        "function definition. Inline any helper logic."
    ),
    "no_loops_no_recursion": (
        "Do not use any `for` or `while` loops, AND do not use recursion. No function may call itself "
        "directly or indirectly. List/set/dict/generator comprehensions are allowed. "
        "You will need to express iteration via comprehensions, map/filter/reduce, or other "
        "non-iterative constructs."
    ),
    "stdlib_whitelist": (
        "You may only import from the following standard-library modules: "
        "`math`, `collections`, `itertools`, `re`. Do not import anything else."
    ),
}
