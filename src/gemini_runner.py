"""Run a single Gemini CLI prompt and return the raw response."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# Spec called for Gemini 2.0 Flash, but only 2.5-flash variants are available via the CLI as of 2026-04-25.
# Override via GEMINI_MODEL env var (e.g. gemini-2.5-flash-lite when 2.5-flash quota is exhausted).

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class GenResult:
    raw: str
    code: Optional[str]
    error: Optional[str]
    elapsed_s: float
    model: str


def extract_code(raw: str) -> Optional[str]:
    """Extract the largest Python code block from the response. If none, return raw if it parses."""
    blocks = _CODE_BLOCK_RE.findall(raw)
    if blocks:
        # Largest block is most likely the full solution
        return max(blocks, key=len).strip()
    # Fallback: see if the entire response parses as Python
    stripped = raw.strip()
    if stripped.startswith("def ") or stripped.startswith("import ") or stripped.startswith("from "):
        return stripped
    return None


_DEFAULT_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "90"))


def generate(prompt: str, model: str = MODEL, timeout_s: float = _DEFAULT_TIMEOUT) -> GenResult:
    """Run `gemini -m <model> -p <prompt>` and return the result."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["gemini", "-m", model, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "GEMINI_NONINTERACTIVE": "1"},
        )
    except subprocess.TimeoutExpired:
        return GenResult("", None, f"cli_timeout_{timeout_s}s", time.time() - t0, model)
    elapsed = time.time() - t0

    raw = (proc.stdout or "")
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()[-300:]
        return GenResult(raw, None, f"cli_exit_{proc.returncode}: {err}", elapsed, model)

    code = extract_code(raw)
    return GenResult(raw, code, None if code else "no_code_block_found", elapsed, model)


if __name__ == "__main__":
    # Smoke test
    r = generate("Write a Python function `add(a, b)` that returns the sum. "
                 "Return only the function inside a Python code block.")
    print(f"elapsed: {r.elapsed_s:.1f}s")
    print(f"model:   {r.model}")
    print(f"error:   {r.error}")
    print(f"code:    {r.code!r}")
    print(f"raw[:200]: {r.raw[:200]!r}")
