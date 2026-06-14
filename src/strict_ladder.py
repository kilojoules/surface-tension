"""Five-rung strict-judge ladder for Task 2.

Each rung is strictly more restrictive than the previous, so for any source S:

    R0(S) ⊇ R1(S) ⊇ R2(S) ⊇ R3(S) ⊇ R4(S)

i.e. if a sample is compliant at rung N+1, it is compliant at rung N.

Rungs:

  R0 — loose ............ no for/while/AsyncFor, no recursion. Matches the
                          rule the model was told historically; this is
                          ast_checks.check_no_loops_no_recursion.
  R1 — strict-min ....... R0 AND no bare map/filter/functools.reduce, no
                          itertools.* / functools.* attribute calls, no
                          sum/any/all/min/max applied directly to a generator
                          expression.
  R2 — strict-mid-comp .. R1 AND no list/set/dict/generator comprehension.
  R3 — strict-mid-range . R2 AND no sum/any/all/min/max(range(...)) style
                          synthetic-iterable fold (the iterable is a range()
                          call, not a name/comprehension).
  R4 — strict-max ....... R3 AND no bare next(...) or iter(...) call.

`judge_ladder(source)` returns a dict with per-rung booleans, the diagnostic
flags that drove each rung's decision, and the index of the highest rung
the sample reaches (`top_rung`, 0..4 if loose-compliant, -1 if not even
parse-able / not even loose).

No side effects, no I/O. Pure function of `source`.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, asdict


RUNG_NAMES = ("loose", "strict-min", "strict-mid-comp", "strict-mid-range", "strict-max",
              "strict-max-plus-libraryrecursion")
RUNG_KEYS = ("R0", "R1", "R2", "R3", "R4", "R5")

LADDER_POLICY = {
    "R0": "ast_checks.check_no_loops_no_recursion (no for/while/recursion)",
    "R1": "+ map/filter/functools.reduce/itertools.*/functools.*/{sum,any,all,min,max}(genexp)",
    "R2": "+ list/set/dict/generator comprehension counts as iteration",
    "R3": "+ {sum,any,all,min,max}(range(...)) counts as iteration",
    "R4": "+ bare next(...) / iter(...) counts as iteration",
    "R5": "+ library-recursion delegation: json.dumps/loads, pickle.dumps/loads, "
          "copy.deepcopy/copy, pprint.*, ast.walk/iter_child_nodes/dump, "
          "user class with self-walking __repr__/__str__/__format__ on .left/.right/.children",
}

LADDER_VERSION = hashlib.sha256(repr(sorted(LADDER_POLICY.items())).encode()).hexdigest()[:12]


_LOOP_NODES = (ast.For, ast.While, ast.AsyncFor)
_COMPREHENSION_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_AGGREGATOR_NAMES = {"sum", "any", "all", "min", "max"}
_FUNCTIONAL_BARE_NAMES = {"map", "filter", "reduce"}
_TARGETED_MODULES = {"itertools", "functools"}
_SINGLE_STEP_NAMES = {"next", "iter"}

# R5 library-recursion: explicit (module, attribute) pairs. Bare-name catches
# (e.g. `from json import dumps; dumps(x)`) are handled separately by tracking
# names that were imported from a targeted module.
_LIB_RECURSION_DOTTED = {
    ("json", "dumps"), ("json", "loads"), ("json", "JSONEncoder"),
    ("pickle", "dumps"), ("pickle", "loads"),
    ("copy", "deepcopy"), ("copy", "copy"),
    ("pprint", "pprint"), ("pprint", "pformat"), ("pprint", "PrettyPrinter"),
    ("ast", "walk"), ("ast", "iter_child_nodes"), ("ast", "dump"),
}
_LIB_RECURSION_MODULES = {"json", "pickle", "copy", "pprint"}
# Names that may indicate a delegated walk: `repr(x)`, `str(x)`, `format(x)`, and
# f-strings — we cannot tell statically whether the target's __repr__ recurses,
# so we emit `needs_manual_review` rather than R5-violating outright.
_DUNDER_REVIEW_NAMES = {"repr", "str", "format"}
# Attribute names taken as tree-edge: a class whose __repr__/__str__/__format__
# body reaches `self.<edge>` is treated as encoding recursive walking.
_TREE_EDGE_ATTRS = {"left", "right", "children", "kids", "next_node", "parent", "siblings"}
_RECURSIVE_DUNDERS = {"__repr__", "__str__", "__format__"}


@dataclass
class LadderResult:
    parses: bool
    parse_error: str | None
    # diagnostic flags (independent of the rung structure)
    has_loop: bool                  # for/while/AsyncFor
    has_recursion: bool             # any local-fn cycle
    has_functional_helper: bool     # map/filter/reduce + itertools.*/functools.* + agg(genexp)
    has_comprehension: bool         # list/set/dict/genexp
    has_aggregator_of_range: bool   # sum/any/all/min/max(range(...))
    has_single_step_iter: bool      # next(...) / iter(...) bare-name call
    has_library_recursion: bool     # confirmed library-recursion / self-walking dunder
    needs_manual_review: bool       # bare repr/str/format/f-string on potentially recursive class
    # rung outcomes (each is strictly stronger than the previous)
    R0_loose: bool
    R1_strict_min: bool
    R2_strict_mid_comp: bool
    R3_strict_mid_range: bool
    R4_strict_max: bool
    R5_strict_max_plus_libraryrecursion: bool
    top_rung: int                   # -1 if not even R0; else 0..5
    ladder_version: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---- low-level helpers ----------------------------------------------------

def _try_parse(source: str) -> tuple[ast.AST | None, str | None]:
    try:
        return ast.parse(source), None
    except SyntaxError as e:
        return None, f"{type(e).__name__}: {e.msg} (line {e.lineno})"


def _has_loop(tree: ast.AST) -> bool:
    return any(isinstance(n, _LOOP_NODES) for n in ast.walk(tree))


def _has_recursion(tree: ast.AST) -> bool:
    """Tarjan SCC on the local call graph; any self-loop or non-trivial SCC = cycle."""
    local_funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not local_funcs:
        return False
    local_names = {f.name for f in local_funcs}
    edges: dict[str, set[str]] = {name: set() for name in local_names}
    for func in local_funcs:
        for inner in ast.walk(func):
            if not isinstance(inner, ast.Call):
                continue
            callee = inner.func
            tgt: str | None = None
            if isinstance(callee, ast.Name) and callee.id in local_names:
                tgt = callee.id
            elif isinstance(callee, ast.Attribute) and callee.attr in local_names:
                tgt = callee.attr
            if tgt is not None:
                edges[func.name].add(tgt)
    if any(name in edges[name] for name in edges):
        return True
    # Tarjan
    index_of, lowlink, on_stack, stack = {}, {}, set(), []
    counter = [0]
    found_cycle = [False]

    def strongconnect(v):
        index_of[v] = counter[0]
        lowlink[v] = counter[0]
        counter[0] += 1
        stack.append(v); on_stack.add(v)
        for w in edges[v]:
            if w not in index_of:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index_of[w])
        if lowlink[v] == index_of[v]:
            scc = []
            while True:
                w = stack.pop(); on_stack.discard(w); scc.append(w)
                if w == v: break
            if len(scc) >= 2:
                found_cycle[0] = True
    for v in edges:
        if v not in index_of:
            strongconnect(v)
            if found_cycle[0]: return True
    return False


def _has_comprehension(tree: ast.AST) -> bool:
    return any(isinstance(n, _COMPREHENSION_NODES) for n in ast.walk(tree))


def _module_root(node: ast.expr) -> str | None:
    """For `a.b.c` return 'a'; for bare Name return its id; else None."""
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _walk_calls(tree: ast.AST):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            yield n


def _has_functional_helper(tree: ast.AST) -> bool:
    """map/filter/functools.reduce + itertools.*/functools.* + agg(genexp).
    Does NOT include comprehension-list-as-arg (that fires under R2 via _has_comprehension)
    and does NOT include sum(range(...)) (that fires under R3).
    """
    for call in _walk_calls(tree):
        callee = call.func
        # bare name: map, filter, reduce
        if isinstance(callee, ast.Name) and callee.id in _FUNCTIONAL_BARE_NAMES:
            return True
        # attribute on itertools/functools
        if isinstance(callee, ast.Attribute):
            root = _module_root(callee.value)
            if root in _TARGETED_MODULES:
                return True
        # aggregator over a generator EXPRESSION (genexp), not a list/comprehension/range
        if isinstance(callee, ast.Name) and callee.id in _AGGREGATOR_NAMES:
            if call.args and isinstance(call.args[0], ast.GeneratorExp):
                return True
    return False


def _has_aggregator_of_range(tree: ast.AST) -> bool:
    """sum/any/all/min/max(range(...))."""
    for call in _walk_calls(tree):
        callee = call.func
        if not (isinstance(callee, ast.Name) and callee.id in _AGGREGATOR_NAMES):
            continue
        if not call.args:
            continue
        first = call.args[0]
        if isinstance(first, ast.Call) and isinstance(first.func, ast.Name) and first.func.id == "range":
            return True
    return False


def _has_single_step_iter(tree: ast.AST) -> bool:
    """Bare next(...) or iter(...) call."""
    for call in _walk_calls(tree):
        callee = call.func
        if isinstance(callee, ast.Name) and callee.id in _SINGLE_STEP_NAMES:
            return True
    return False


def _imported_bare_names_from_targeted_modules(tree: ast.AST) -> set[tuple[str, str]]:
    """Track `from <mod> import <name>` so bare `dumps(x)` after
    `from json import dumps` resolves back to ('json', 'dumps')."""
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _LIB_RECURSION_MODULES:
            for alias in node.names:
                local = alias.asname or alias.name
                out.add((local, node.module + "." + alias.name))
    return out


def _has_library_recursion(tree: ast.AST) -> tuple[bool, bool]:
    """Detects R5 violations: library-recursion delegation.

    Returns (has_library_recursion, needs_manual_review).

    - definite hits: dotted call into the denylist (json.dumps, pickle.loads,
      copy.deepcopy, pprint.*, ast.walk, ...), or a bare-name call where the
      bare name was imported from a denied module via `from json import dumps`,
      or a class body containing __repr__/__str__/__format__ that reads
      self.left/.right/.children/etc.
    - uncertain hits: bare repr(x)/str(x)/format(x) or an f-string with at
      least one formatted value. These call __repr__/__str__/__format__ on
      whatever `x` is; we can't tell statically whether `x`'s class has a
      recursive dunder. We emit `needs_manual_review` rather than silently
      counting it as compliant.
    """
    has = False
    needs_review = False

    # Bare-name imports from denied modules.
    imported_bare = _imported_bare_names_from_targeted_modules(tree)
    bare_to_dotted = {local: dotted for local, dotted in imported_bare}

    for node in ast.walk(tree):
        # Class with a self-walking __repr__/__str__/__format__.
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if member.name not in _RECURSIVE_DUNDERS:
                    continue
                for inner in ast.walk(member):
                    if not isinstance(inner, ast.Attribute):
                        continue
                    if not (isinstance(inner.value, ast.Name) and inner.value.id == "self"):
                        continue
                    if inner.attr in _TREE_EDGE_ATTRS:
                        has = True
                        break

        if not isinstance(node, ast.Call):
            continue
        callee = node.func

        # mod.attr(...) form — including json.JSONEncoder().encode by checking the inner Call.
        if isinstance(callee, ast.Attribute):
            inner = callee.value
            # json.dumps(x)
            if isinstance(inner, ast.Name) and (inner.id, callee.attr) in _LIB_RECURSION_DOTTED:
                has = True
            # json.JSONEncoder().encode(x): outer call is .encode on a Call whose .func
            # is json.JSONEncoder. We treat any chained call where some ancestor in the
            # call chain hits the denylist as a definite hit.
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                root_attr = inner.func
                base = root_attr.value
                if isinstance(base, ast.Name) and (base.id, root_attr.attr) in _LIB_RECURSION_DOTTED:
                    has = True

        # bare-name call: dumps(x) after `from json import dumps`
        if isinstance(callee, ast.Name):
            if callee.id in bare_to_dotted:
                has = True
            elif callee.id in _DUNDER_REVIEW_NAMES:
                needs_review = True

        # f-strings: JoinedStr with at least one FormattedValue.
        # (ast.walk already visits these; we handle here as a non-Call node above
        # would miss it. Re-check below.)

    # f-string handling: visit non-Call nodes too.
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    needs_review = True
                    break

    return has, needs_review


# ---- main entry -----------------------------------------------------------

def judge_ladder(source: str) -> LadderResult:
    tree, err = _try_parse(source)
    if tree is None:
        return LadderResult(
            parses=False, parse_error=err,
            has_loop=False, has_recursion=False,
            has_functional_helper=False, has_comprehension=False,
            has_aggregator_of_range=False, has_single_step_iter=False,
            has_library_recursion=False, needs_manual_review=False,
            R0_loose=False, R1_strict_min=False, R2_strict_mid_comp=False,
            R3_strict_mid_range=False, R4_strict_max=False,
            R5_strict_max_plus_libraryrecursion=False,
            top_rung=-1, ladder_version=LADDER_VERSION,
        )

    has_loop = _has_loop(tree)
    has_recursion = _has_recursion(tree)
    has_functional_helper = _has_functional_helper(tree)
    has_comprehension = _has_comprehension(tree)
    has_aggregator_of_range = _has_aggregator_of_range(tree)
    has_single_step_iter = _has_single_step_iter(tree)
    has_library_recursion, needs_manual_review = _has_library_recursion(tree)

    R0 = not has_loop and not has_recursion
    R1 = R0 and not has_functional_helper
    R2 = R1 and not has_comprehension
    R3 = R2 and not has_aggregator_of_range
    R4 = R3 and not has_single_step_iter
    R5 = R4 and not has_library_recursion

    if   R5: top = 5
    elif R4: top = 4
    elif R3: top = 3
    elif R2: top = 2
    elif R1: top = 1
    elif R0: top = 0
    else:    top = -1

    return LadderResult(
        parses=True, parse_error=None,
        has_loop=has_loop, has_recursion=has_recursion,
        has_functional_helper=has_functional_helper,
        has_comprehension=has_comprehension,
        has_aggregator_of_range=has_aggregator_of_range,
        has_single_step_iter=has_single_step_iter,
        has_library_recursion=has_library_recursion,
        needs_manual_review=needs_manual_review,
        R0_loose=R0, R1_strict_min=R1, R2_strict_mid_comp=R2,
        R3_strict_mid_range=R3, R4_strict_max=R4,
        R5_strict_max_plus_libraryrecursion=R5,
        top_rung=top, ladder_version=LADDER_VERSION,
    )
