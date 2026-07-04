"""Incremental judge runner with idempotent resume.

The Vast.ai box can vanish at any time (interruptible, balance hits zero, host
reboot, etc.). If we hold 16k judgments in memory and write at the end, one
disconnect = total loss.

This runner:
  - Reads inputs JSONL streaming (one self-report row per line).
  - Appends each judgment to the output JSONL as soon as it returns, with
    fsync, so a hard kill at most loses the one in-flight call.
  - Resumes idempotently: on restart, it reads the output JSONL, builds a key
    set of completed (model, problem_id, sample_idx), and skips those inputs.
  - Handles partial last lines from a prior crash (file truncated mid-write).
  - Sources its self-report text from input["self_report_text"].

Input row shape (minimum):
  {"model": str, "problem_id": str, "sample_idx": int, "self_report_text": str}

Output row shape (input fields preserved + judge_* fields appended):
  {... input fields ...,
   "judge_label": str, "judge_endorsement": bool,
   "judge_confidence": float, "judge_ambiguous": bool,
   "judge_span": str, "judge_parse_error": str|None,
   "rubric_version": str}

Logged progress goes to stderr so the JSONL output stream stays clean.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable

from .claim_judge import Backend, RUBRIC_VERSION, judge


KEY_FIELDS = ("model", "problem_id", "sample_idx")


def _row_key(row: dict) -> tuple:
    return tuple(row[k] for k in KEY_FIELDS)


def _read_jsonl_lenient(path: str) -> list[dict]:
    """Read a JSONL file tolerating a partial trailing line (from a crash
    mid-fsync). Complete lines are returned; the trailing partial is dropped
    silently (it will be re-judged on the next pass)."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Partial last line — drop it. The corresponding input will
                # be re-judged because its key isn't in the completed set.
                continue
    return rows


def already_completed_keys(out_path: str) -> set[tuple]:
    """Returns the set of (model, problem_id, sample_idx) keys already in the
    output JSONL. Tolerates a partial last line."""
    return {_row_key(r) for r in _read_jsonl_lenient(out_path)
            if all(k in r for k in KEY_FIELDS)}


def assert_resume_rubric_matches(out_path: str) -> None:
    """Refuse to resume onto judgments produced under a different rubric.

    Resume keys only on (model, problem_id, sample_idx), so without this guard
    a run started after a RUBRIC_VERSION bump would SKIP the stale-rubric rows
    (leaving them with old-semantics labels and no per-construct fields) while
    appending new-rubric rows — a silently mixed file that analyze then rejects
    wholesale with a misleading 'has the judge been run?' error. Fail early and
    clearly instead: the operator must re-judge into a fresh output file."""
    existing = {r.get("rubric_version") for r in _read_jsonl_lenient(out_path)
                if r.get("rubric_version") is not None}
    stale = existing - {RUBRIC_VERSION}
    if stale:
        raise SystemExit(
            f"FATAL: {out_path} contains judgments under rubric "
            f"{sorted(stale)} but the current rubric is {RUBRIC_VERSION!r}. "
            "Rubric versions have different label semantics and must not be "
            "pooled. Re-judge into a fresh --out file (do not resume)."
        )


def _stream_inputs(in_path: str) -> Iterable[dict]:
    with open(in_path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@dataclass
class RunStats:
    total_inputs: int
    already_done: int
    judged_this_run: int
    parse_errors: int
    ambiguous: int


def run(
    in_path: str,
    out_path: str,
    backend: Backend,
    *,
    progress_every: int = 50,
    log_stream=sys.stderr,
) -> RunStats:
    """Append judgments to `out_path` for every row in `in_path` whose key is
    not already present. Crashing mid-run and re-invoking with the same args
    resumes from where it stopped."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    assert_resume_rubric_matches(out_path)
    done = already_completed_keys(out_path)

    total = 0
    judged = 0
    parse_errors = 0
    ambiguous = 0

    for row in _stream_inputs(in_path):
        total += 1
        key = _row_key(row)
        if key in done:
            continue

        out = judge(row["self_report_text"], backend)
        record = dict(row)
        record["judge_label"] = out.self_compliance_claim
        # Also fill the canonical field analyze.py reads, so judgments.jsonl
        # is analyze-ready without a separate merge step (the judge_label /
        # self_compliance_claim split predates rubric v1).
        record["self_compliance_claim"] = out.self_compliance_claim
        # Per-construct claims (rubric v1) — construct-level deception axis.
        record["claims_for_while"] = out.claims_for_while
        record["claims_recursion"] = out.claims_recursion
        record["claims_functional"] = out.claims_functional
        record["claims_comprehension"] = out.claims_comprehension
        record["judge_endorsement"] = out.rule_endorsement
        record["judge_confidence"] = out.confidence
        record["judge_ambiguous"] = out.ambiguous
        record["judge_span"] = out.span
        record["judge_parse_error"] = out.parse_error
        record["rubric_version"] = out.rubric_version

        # Append + fsync immediately. If the box dies right now, we lose at
        # most this one judgment.
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

        done.add(key)
        judged += 1
        if out.parse_error:
            parse_errors += 1
        if out.ambiguous:
            ambiguous += 1

        if progress_every and judged % progress_every == 0:
            print(f"[judge_runner] judged={judged}  parse_errors={parse_errors}  "
                  f"ambiguous={ambiguous}", file=log_stream, flush=True)

    return RunStats(
        total_inputs=total,
        already_done=total - judged,
        judged_this_run=judged,
        parse_errors=parse_errors,
        ambiguous=ambiguous,
    )


# --- CLI -----------------------------------------------------------------

def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="JSONL of self-reports (must have model, problem_id, sample_idx, self_report_text)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="JSONL of judgments — appended to, idempotent on rerun")
    ap.add_argument("--fake", action="store_true",
                    help="Use offline FakeBackend (echoes no_claim). For end-to-end smoke only.")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    if args.fake:
        from .claim_judge import FakeBackend
        backend: Backend = FakeBackend(rules=[])
    else:
        from .vllm_backend import VLLMBackend
        backend = VLLMBackend()

    stats = run(args.in_path, args.out_path, backend,
                progress_every=args.progress_every)
    print(f"[judge_runner] done  total={stats.total_inputs}  "
          f"already_done={stats.already_done}  judged={stats.judged_this_run}  "
          f"parse_errors={stats.parse_errors}  ambiguous={stats.ambiguous}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
