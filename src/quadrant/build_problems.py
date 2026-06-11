"""Build the quadrant problem cohort.

The user-selected cohort is the **deck-matched clean-17** — the same 17 LCB-medium
problems used as the held-out clean set in `paper/main.tex`. The earlier-cited
"136" was attempts (17 × 8), not distinct problems. Cohort decision: deck-matched.

Source manifest:
    data/problems_lcb_clean17.jsonl  (17 lines, schema below)

Target (Phase-1 reader is `quadrant/generate.py`):
    paper/data/quadrant/problems_lcb_clean17.jsonl   — required-fields only
    paper/data/quadrant/problems_lcb_clean17.meta.json  — provenance sidecar

Source schema:
    id, benchmark, mode, prompt, entry_point, stdin_tests, canonical, contest_date

We emit ONLY {problem_id, prompt} as required; audit fields (benchmark, mode,
contest_date, source_dataset, lcb_version) are included but `generate.py`
ignores unknown keys, so they're safe to add.

The constraint string ("no loops, no recursion") is NOT injected here. It's
applied in generate.py's system prompt at generation time. This file is a
guard against double-stating it.

What this script does NOT do (per spec):
  - no GPU, no torch
  - no test execution
  - no id invention (canonical lcb/<question_id> only)
  - no test-runner resolution check (that's a Phase-1 startup check on the
    GPU host)

Determinism: stable sort by problem_id; the script writes the same bytes
on every run for the same source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

# Keep in lockstep with quadrant/generate.py:CONSTRAINT_INSTRUCTION. If anyone
# edits one without the other this file's constraint-absence guard will
# silently fail. The guard's signature phrases catch any reasonable variant
# of the no-loops/no-recursion rule.
CONSTRAINT_PHRASES = (
    "Do not use any `for` or `while` loops",
    "no for or while loops",
    "do not use recursion",
)

EXPECTED_COUNT = 17   # deck-matched clean-17 cohort — see paper/main.tex


@dataclass
class BuildResult:
    out_path: str
    meta_path: str
    count: int
    out_sha256: str
    source_path: str
    source_sha256: str


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_loader(source_path: str) -> list[dict]:
    rows = []
    with open(source_path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _project_row(src: dict) -> dict:
    """Re-shape source row → target schema. Keeps the canonical LCB id as
    `problem_id` so the GPU-host test runner resolves correctly."""
    pid = src["id"]      # 'lcb/<question_id>' — canonical
    return {
        # required by generate.py
        "problem_id": pid,
        "prompt": src["prompt"],
        # audit-only — generate.py ignores unknown keys
        "source_id": pid,
        "benchmark": src.get("benchmark"),
        "mode": src.get("mode"),
        "entry_point": src.get("entry_point"),
        "contest_date": src.get("contest_date"),
        "source_dataset": "livecodebench",
        "lcb_version": "release_v5",      # from src/loaders_lcb.py:LCB_VERSION
    }


# --- assertions (any failure aborts) ---------------------------------------

def _assert_count(rows: list[dict], expected: int) -> None:
    if len(rows) != expected:
        raise AssertionError(
            f"cohort count mismatch: got {len(rows)} rows, expected {expected}. "
            "The deck-matched clean-17 manifest has exactly 17 problems "
            "(see paper/main.tex). If you intended a different cohort, change "
            "the source path AND --expected-count explicitly — never silently."
        )


def _assert_unique_problem_ids(rows: list[dict]) -> None:
    ids = [r["problem_id"] for r in rows]
    if len(set(ids)) != len(ids):
        from collections import Counter
        dupes = [pid for pid, n in Counter(ids).items() if n > 1]
        raise AssertionError(f"duplicate problem_id(s): {dupes}")


def _assert_prompts_nonempty(rows: list[dict]) -> None:
    bad = [r["problem_id"] for r in rows if not r["prompt"].strip()]
    if bad:
        raise AssertionError(f"empty/whitespace prompt for: {bad}")


def _assert_constraint_absent(rows: list[dict]) -> None:
    offending = []
    for r in rows:
        for phrase in CONSTRAINT_PHRASES:
            if phrase.lower() in r["prompt"].lower():
                offending.append((r["problem_id"], phrase))
                break
    if offending:
        raise AssertionError(
            "constraint string found in source prompt(s); refusing to emit a "
            "cohort where the experimental constraint is double-stated:\n  "
            + "\n  ".join(f"{pid}: {phrase!r}" for pid, phrase in offending)
        )


# --- main build -----------------------------------------------------------

def build(
    source_path: str,
    out_path: str,
    meta_path: str,
    *,
    expected_count: int = EXPECTED_COUNT,
    force: bool = False,
    loader: Callable[[str], list[dict]] = _default_loader,
) -> BuildResult:
    """Read source, project, assert, sort, and write. Idempotent: producing
    the same source path yields byte-identical output and meta files."""
    rows_src = loader(source_path)
    rows = [_project_row(s) for s in rows_src]
    # Sort BEFORE assertions so the duplicate check sees stable ordering.
    rows.sort(key=lambda r: r["problem_id"])

    _assert_count(rows, expected_count)
    _assert_unique_problem_ids(rows)
    _assert_prompts_nonempty(rows)
    _assert_constraint_absent(rows)

    out_bytes = ("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode()
    out_sha = _sha256_bytes(out_bytes)
    # Source hash is best-effort: production CLI passes a real file; offline
    # tests pass a synthetic path with a fake loader and skip the hash.
    source_sha = _sha256_file(source_path) if os.path.exists(source_path) else None

    # Determinism + clobber guard
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            existing = f.read()
        if existing == out_bytes:
            # identical re-run — refresh meta and return
            pass
        elif not force:
            existing_sha = _sha256_bytes(existing)
            raise SystemExit(
                f"refusing to overwrite {out_path}: existing SHA-256 "
                f"{existing_sha} != new {out_sha}. Pass --force to clobber. "
                "If a generation run is using the old cohort, swapping it "
                "mid-run silently changes what the rows correspond to."
            )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(out_bytes)

    meta = {
        "cohort_label": "clean-17 (deck-matched)",
        "cohort_rationale": (
            "Same 17 LCB-medium problems used as the held-out clean set in the "
            "Surface Tension deck (paper/main.tex). 'Earlier '136' in briefs "
            "was 17 problems × 8 attempts, not 136 distinct problems."
        ),
        "selection_method": "manifest_reuse",   # NOT 'derived'
        "source_path": os.path.relpath(source_path),
        "source_sha256": source_sha,
        "loader_module": "src/loaders_lcb.py",
        "lcb_version": "release_v5",
        "count": len(rows),
        "out_path": os.path.relpath(out_path),
        "out_sha256": out_sha,
        "schema_required": ["problem_id", "prompt"],
        "schema_audit": [
            "source_id", "benchmark", "mode", "entry_point", "contest_date",
            "source_dataset", "lcb_version",
        ],
        "constraint_check_phrases": list(CONSTRAINT_PHRASES),
        "constraint_check_status": "passed",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return BuildResult(
        out_path=out_path, meta_path=meta_path, count=len(rows),
        out_sha256=out_sha, source_path=source_path, source_sha256=source_sha,
    )


def _cli() -> int:
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=f"{ROOT}/data/problems_lcb_clean17.jsonl",
                    help="canonical manifest (default: data/problems_lcb_clean17.jsonl)")
    ap.add_argument("--out",    default=f"{ROOT}/paper/data/quadrant/problems_lcb_clean17.jsonl",
                    help="output JSONL path")
    ap.add_argument("--meta",   default=f"{ROOT}/paper/data/quadrant/problems_lcb_clean17.meta.json",
                    help="provenance sidecar path")
    ap.add_argument("--expected-count", type=int, default=EXPECTED_COUNT,
                    help="how many problems to expect (abort if mismatch)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if existing output has a different hash")
    args = ap.parse_args()

    if not os.path.exists(args.source):
        print(f"source manifest missing: {args.source}", file=sys.stderr)
        return 2

    result = build(
        args.source, args.out, args.meta,
        expected_count=args.expected_count, force=args.force,
    )
    print(f"[build_problems] count={result.count}")
    print(f"[build_problems] out  {result.out_path}  sha={result.out_sha256}")
    print(f"[build_problems] meta {result.meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
