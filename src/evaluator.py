"""Sandboxed test evaluator.

Runs candidate code + test runner in a fresh Python subprocess in a tempdir,
with a timeout and process-group kill.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResult:
    passed: bool
    timed_out: bool
    error: Optional[str]


def _build_script(candidate_code: str, entry_point: str, test_runner: str) -> str:
    return (
        "import sys\n"
        "import math\n"  # common helper
        "_ns = {}\n"
        f"_candidate_src = {candidate_code!r}\n"
        f"_test_src = {test_runner!r}\n"
        "try:\n"
        "    exec(compile(_candidate_src, '<candidate>', 'exec'), _ns)\n"
        "except Exception as e:\n"
        "    print(f'CANDIDATE_EXEC_FAIL: {type(e).__name__}: {e}', file=sys.stderr)\n"
        "    sys.exit(2)\n"
        f"if {entry_point!r} not in _ns:\n"
        f"    print('ENTRY_POINT_MISSING: {entry_point}', file=sys.stderr)\n"
        "    sys.exit(3)\n"
        "try:\n"
        "    exec(compile(_test_src, '<tests>', 'exec'), _ns)\n"
        "except AssertionError as e:\n"
        "    print(f'TEST_ASSERT_FAIL: {e}', file=sys.stderr)\n"
        "    sys.exit(4)\n"
        "except Exception as e:\n"
        "    print(f'TEST_EXC: {type(e).__name__}: {e}', file=sys.stderr)\n"
        "    sys.exit(5)\n"
        "print('OK')\n"
    )


def evaluate(candidate_code: str, entry_point: str, test_runner: str, timeout_s: float = 10.0) -> EvalResult:
    """Run candidate + tests in a sandboxed subprocess. Returns EvalResult."""
    script = _build_script(candidate_code, entry_point, test_runner)

    with tempfile.TemporaryDirectory() as tmp:
        script_path = os.path.join(tmp, "_run.py")
        with open(script_path, "w") as f:
            f.write(script)

        try:
            proc = subprocess.run(
                [sys.executable, "-I", script_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as te:
            # On timeout, the run() context already terminated the process. Belt-and-suspenders.
            try:
                if te.__cause__ is not None and hasattr(te.__cause__, "pid"):
                    os.killpg(os.getpgid(te.__cause__.pid), signal.SIGKILL)
            except Exception:
                pass
            return EvalResult(False, True, f"timeout after {timeout_s}s")

    if proc.returncode == 0 and proc.stdout.strip().endswith("OK"):
        return EvalResult(True, False, None)
    err = (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else f"exit {proc.returncode}"
    return EvalResult(False, False, err[:300])


def _normalize_output(s: str) -> str:
    """Whitespace-normalize: strip trailing whitespace per line, drop trailing blank lines."""
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def evaluate_stdin(candidate_code: str, tests: list, timeout_s: float = 10.0) -> EvalResult:
    """Run candidate as a script. For each test, pipe `input`, capture stdout, compare to `output`.

    All tests must pass for the candidate to be considered correct. Exits early on first failure.
    """
    if not tests:
        return EvalResult(False, False, "no_tests")

    with tempfile.TemporaryDirectory() as tmp:
        script_path = os.path.join(tmp, "_run.py")
        with open(script_path, "w") as f:
            f.write(candidate_code)

        for i, t in enumerate(tests):
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", script_path],
                    input=t["input"],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=tmp,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                return EvalResult(False, True, f"timeout on test {i}")

            if proc.returncode != 0:
                err = (proc.stderr or "").strip().splitlines()
                msg = err[-1][:200] if err else f"exit {proc.returncode}"
                return EvalResult(False, False, f"runtime_err on test {i}: {msg}")

            got = _normalize_output(proc.stdout)
            want = _normalize_output(t["output"])
            if got != want:
                return EvalResult(False, False, f"wrong_output on test {i}: got {got[:80]!r} want {want[:80]!r}")

    return EvalResult(True, False, None)


if __name__ == "__main__":
    # Self-test (function mode)
    code = "def add(a, b):\n    return a + b\n"
    test = "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n"
    print("pass case:", evaluate(code, "add", test))

    bad = "def add(a, b):\n    return a - b\n"
    print("fail case:", evaluate(bad, "add", test))

    inf = "def add(a, b):\n    while True: pass\n"
    print("timeout case:", evaluate(inf, "add", test, timeout_s=2))

    missing = "def other(): pass\n"
    print("missing entry:", evaluate(missing, "add", test))

    # Self-test (stdin mode)
    stdin_code = "a, b = map(int, input().split())\nprint(a + b)\n"
    stdin_tests = [{"input": "1 2\n", "output": "3\n"}, {"input": "10 20\n", "output": "30\n"}]
    print("stdin pass:", evaluate_stdin(stdin_code, stdin_tests))

    stdin_bad = "a, b = map(int, input().split())\nprint(a - b)\n"
    print("stdin fail:", evaluate_stdin(stdin_bad, stdin_tests))

    stdin_inf = "import sys\nwhile True: pass\n"
    print("stdin timeout:", evaluate_stdin(stdin_inf, stdin_tests, timeout_s=2))

    stdin_crash = "raise RuntimeError('boom')\n"
    print("stdin runtime err:", evaluate_stdin(stdin_crash, stdin_tests))
