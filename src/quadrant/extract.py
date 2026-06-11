"""Code extraction for the quadrant experiment.

Brief says: "Pull the code block from each output. If multiple, take the LAST
fenced block. If none parses, mark `parses=False` and route to the failing
bucket — do not silently drop."

This diverges intentionally from `src/sweep_local.py:extract_code`, which takes
the LARGEST fenced block. The two should not be substituted for one another.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class Extraction:
    code: str | None     # None means "no fenced block AND raw doesn't look like code"
    parses: bool
    parse_error: str | None
    source: str          # "last_fenced", "unfenced_codey", "empty"


def extract(raw: str) -> Extraction:
    """Last-fenced-block first; fall back to the whole raw output ONLY if it
    starts with code-shaped tokens. Otherwise return code=None.

    `parses` is set by attempting ast.parse on whatever code we extracted.
    A None code returns parses=False with source="empty".
    """
    blocks = _FENCE_RE.findall(raw)
    if blocks:
        code = blocks[-1].strip()
        return _wrap(code, "last_fenced")

    stripped = raw.strip()
    if stripped.startswith(("def ", "import ", "from ", "class ", "@")):
        return _wrap(stripped, "unfenced_codey")

    return Extraction(code=None, parses=False, parse_error=None, source="empty")


def _wrap(code: str, source: str) -> Extraction:
    try:
        ast.parse(code)
        return Extraction(code=code, parses=True, parse_error=None, source=source)
    except SyntaxError as e:
        return Extraction(
            code=code, parses=False,
            parse_error=f"{type(e).__name__}: {e.msg} (line {e.lineno})",
            source=source,
        )
