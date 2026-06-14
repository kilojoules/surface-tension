"""Module-isolated helper for persisting raw model responses to disk.

The current eval pipelines (`src/sweep.py` and `src/sweep_local.py`) save raw
output only on the extraction-failure branch, so the prose rationale that the
model emits BEFORE a successfully-extracted code block is discarded. This
module provides a single helper that writes the raw response to a `__raw.txt`
sibling of the existing `__s<n>.py` source file. It is kept in its own module
(rather than added directly into `sweep.py`) so we can unit-test it with no
sweep.py edits applied yet — dry-running the patch before committing.

Naming convention matches `_save_source`: same directory, same `safe_id`,
same `constraint__condition__sN` suffix, but with `__raw.txt` instead of `.py`.

Once approved, the integration is two lines per sweep file (see
`paper/data/rationale_label/SAVE_RAW_PATCH.md`).
"""
from __future__ import annotations

import os
from typing import Protocol


class _RowLike(Protocol):
    """Structural type matching the `Row` dataclass in src/sweep.py and src/sweep_local.py."""
    problem_id: str
    constraint: str
    condition: str
    sample_idx: int


def raw_filename(row: _RowLike) -> str:
    """Mirror `_save_source` naming, swapping `.py` for `__raw.txt`."""
    safe_id = row.problem_id.replace("/", "_")
    return f"{safe_id}__{row.constraint}__{row.condition}__s{row.sample_idx}__raw.txt"


def save_raw(source_dir: str, row: _RowLike, raw: str) -> str:
    """Persist `raw` to `<source_dir>/<safe_id>__<constraint>__<condition>__sN__raw.txt`.

    Idempotent (overwrite). Returns the absolute path written.
    """
    os.makedirs(source_dir, exist_ok=True)
    path = os.path.join(source_dir, raw_filename(row))
    with open(path, "w") as f:
        f.write(raw)
    return path
