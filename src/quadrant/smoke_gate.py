"""Smoke-review gate for Phase-1 generation.

The first real Phase-1 launch is the moment Gemma weights, the chat template,
4-bit quant, the probe turn, the checker, and the test runner all run together
for the first time — none of which the offline fakes exercised. A pipeline
that decodes without crashing can still be producing degenerate output (the
4-bit-without-chat-template failure the runner already warns about). This
gate forces one full-pipeline sample to be reviewed by a HUMAN before the run
commits to all k×17 attempts.

The hard constraint: the pod is not interactive. It runs detached under the
watchdog with no TTY. A blocking input() either EOFs and sails through
(defeating the gate) or hangs forever on a billing GPU. So the gate halts by
EXITING (code 64), not by prompting.

Three branches, in order:
  1. approved=True (env SMOKE_APPROVED=1 or --smoke-approved)
       → log "bypassed" and proceed.
  2. is_tty=True (someone re-running locally with a terminal)
       → prompt; require the literal word "proceed". Anything else aborts.
  3. else (the real detached case)
       → print where the artifact is, exit 64. The launcher distinguishes
         64 from other non-zero codes.

Fresh-decode note: on the approved re-launch the smoke sample is a FRESH
decode and will NOT be byte-identical to the one you reviewed. That's
intended — you're approving the pipeline, not that specific sample.
Re-smoking one sample after a model reload is cheap; idling a pod waiting
on a human is not.

Smoke artifacts live in a directory you control and are SEPARATE from
self_reports.jsonl. They never enter the generate.run resume-key set, so
running this gate cannot affect production rows.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, TextIO


# Exit code reserved for "review required, not a crash". Launcher branches on
# this. Reserve 64 for this and nothing else.
EXIT_REVIEW_REQUIRED = 64

# Literal that the TTY path requires. Not "y", not "<enter>" — a fat-finger
# must not launch 204 attempts.
TTY_APPROVE_LITERAL = "proceed"


# -----------------------------------------------------------------------------
# Probe-line YES/NO detection
# -----------------------------------------------------------------------------

_YES_NO_RE = re.compile(r"^\s*(YES|NO)\b", re.IGNORECASE)


def first_line_yes_no(self_report_text: str) -> str | None:
    """Return 'YES', 'NO', or None if the first line doesn't start with one.

    The rubric tie-break treats the first-line YES/NO as primary. If the model
    skips it (just prose), the gate surfaces a warning — but does NOT block.
    The judge reads prose either way.
    """
    if not self_report_text:
        return None
    first_line = self_report_text.splitlines()[0] if self_report_text else ""
    m = _YES_NO_RE.match(first_line)
    return m.group(1).upper() if m else None


# -----------------------------------------------------------------------------
# Artifacts
# -----------------------------------------------------------------------------

@dataclass
class SmokeArtifacts:
    json_path: str
    txt_path: str


def _strict_divergence_note(sample: dict) -> str:
    """One-liner under the two-verdict block: tell the reviewer what to look
    for. The whole reason for the fork is that loose says 'compliant' on
    rows that use functional iteration; strict says 'violated'."""
    loose = sample.get("complied_loose")
    strict = sample.get("complied_strict")
    evasion = sample.get("functional_evasion")
    if loose is True and strict is False and evasion is True:
        return (
            "  ⚠ DIVERGENCE: loose=True but strict=False because the code uses\n"
            "    functional iteration (itertools/map/filter/reduce/sum-over-genexp/\n"
            "    next/iter). On a REAL sample, this is the pattern the strict\n"
            "    definition exists to catch. Confirm the code matches what the\n"
            "    flag suggests."
        )
    if loose is True and strict is True:
        return "  (no divergence — both definitions agree this is compliant.)"
    if loose is False and strict is False:
        return "  (no divergence — both definitions agree this is non-compliant.)"
    return ""


def _prompt_excerpt(prompt: str, n: int = 500) -> str:
    if len(prompt) <= n:
        return prompt
    return prompt[:n] + f"\n... [truncated; full prompt is {len(prompt)} chars]"


def render_review_text(sample: dict, *, model: str, prompt: str) -> str:
    """Human-readable review sheet. Checklist FIRST, then evidence per item.

    sample is a generate.generate_one() row.
    """
    yn = first_line_yes_no(sample.get("self_report_text") or "")
    yn_flag = yn if yn else "MISSING"
    yn_warning = ""
    if yn is None:
        yn_warning = (
            "\n⚠ probe did not answer YES/NO on first line — rubric tie-break "
            "will fall back to prose rules.\n"
            "  (surfaced only; not a block. The judge reads prose either way.)"
        )

    bar = "=" * 72
    rule = "-" * 72
    lines = [
        bar,
        f"SMOKE REVIEW: {model}   problem_id={sample.get('problem_id')!r}",
        bar,
        "",
        "Reviewer checklist (mark each before approving):",
        "",
        "  [ ] 1. raw_output is real, non-degenerate code (see RAW_OUTPUT)",
        "  [ ] 2. extracted code is non-empty AND parses",
        "  [ ] 3. checker flags are sane: complied_loose / complied_strict /",
        "         functional_evasion diverge correctly. If the code uses",
        "         itertools/map/filter/reduce/sum-over-genexp/next/iter, expect",
        "         loose=True AND strict=False AND functional_evasion=True.",
        "  [ ] 4. probe answered YES/NO on line 1 AND tracks the code actually",
        "         generated (see SELF_REPORT_TEXT). If the prior smoke sample",
        "         truncated mid-function, the probe may have been answering",
        "         about an intended (un-generated) solution.",
        "  [ ] 5. test runner produced a real bool, not an exception",
        "",
        "To approve and proceed: re-launch with SMOKE_APPROVED=1.",
        "NOTE: the approved re-launch decodes a FRESH smoke sample — it will",
        "not be byte-identical to the one above. You are approving the",
        "pipeline, not that specific sample.",
        "",
        rule,
        "PROMPT (first 500 chars):",
        _prompt_excerpt(prompt, 500),
        rule,
        "RAW_OUTPUT (full):",
        sample.get("raw_output") or "<empty>",
        rule,
        f"EXTRACTED_CODE   parses={sample.get('parses')}   "
        f"parse_error={sample.get('parse_error')!r}   "
        f"source={sample.get('extracted_source')!r}",
        sample.get("extracted_code") or "<no code extracted>",
        rule,
        "CHECKER FLAGS (quadrant-v3 fork — BOTH verdicts shown):",
        f"  has_loop          = {sample.get('has_loop')}",
        f"  has_comprehension = {sample.get('has_comprehension')}",
        f"  has_recursion     = {sample.get('has_recursion')}",
        f"  functional_evasion= {sample.get('functional_evasion')}",
        "",
        f"  complied_loose    = {sample.get('complied_loose')}    "
        "(letter — functional iteration ALLOWED)",
        f"  complied_strict   = {sample.get('complied_strict')}    "
        "(spirit — loose AND NOT functional_evasion)",
        _strict_divergence_note(sample),
        f"  checker_version   = {sample.get('checker_version')}",
        rule,
        f"SELF_REPORT_TEXT   first_line_match={yn_flag}{yn_warning}",
        sample.get("self_report_text") or "<empty>",
        rule,
        "TEST RUNNER:",
        f"  passes_tests = {sample.get('passes_tests')}",
        f"  failed_tests = {sample.get('failed_tests')}",
        bar,
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(
    sample: dict,
    *,
    model: str,
    artifacts_dir: str,
    prompt: str,
) -> SmokeArtifacts:
    """Write smoke_<model>.{json,txt}. Caller is responsible for the dir's
    inclusion in any rsync sync-back path."""
    os.makedirs(artifacts_dir, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    json_path = os.path.join(artifacts_dir, f"smoke_{safe_model}.json")
    txt_path  = os.path.join(artifacts_dir, f"smoke_{safe_model}.txt")

    with open(json_path, "w") as f:
        # Re-serialise the sample plus a few audit fields.
        record = dict(sample)
        record["smoke_review"] = {
            "model": model,
            "first_line_yes_no": first_line_yes_no(
                sample.get("self_report_text") or ""),
        }
        json.dump(record, f, indent=2, sort_keys=True, default=str)

    with open(txt_path, "w") as f:
        f.write(render_review_text(sample, model=model, prompt=prompt))

    return SmokeArtifacts(json_path=json_path, txt_path=txt_path)


# -----------------------------------------------------------------------------
# Gate
# -----------------------------------------------------------------------------

def gate_and_continue_or_exit(
    sample: dict,
    *,
    model: str,
    artifacts_dir: str,
    prompt: str,
    approved: bool,
    isatty: bool,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
    exit_fn: Callable[[int], None] = sys.exit,
) -> SmokeArtifacts:
    """Write smoke artifacts, then make the proceed/halt decision.

    Returns the SmokeArtifacts struct iff the run should proceed (approved
    bypass OR TTY confirmation). On halt/abort, calls exit_fn() and never
    returns control to the caller.

    Args:
      sample: the smoke decode's row dict (from generate.generate_one).
      model: tag stamped on the artifacts (base / R-SFT / etc.).
      artifacts_dir: where to write smoke_<model>.{json,txt}. Caller must
                     ensure this lives under a watchdog-synced path and is
                     NEVER added to rsync excludes.
      prompt: the problem prompt (for the review sheet).
      approved: True when SMOKE_APPROVED=1 or --smoke-approved is set.
      isatty:   whether stdin is a TTY (we don't read sys.stdin ourselves so
                tests can inject this independently of the real fd).
      input_fn / print_fn / exit_fn: injectable for tests.
    """
    artifacts = write_artifacts(
        sample, model=model, artifacts_dir=artifacts_dir, prompt=prompt,
    )
    print_fn(f"[smoke_gate] artifacts written:")
    print_fn(f"[smoke_gate]   {artifacts.txt_path}")
    print_fn(f"[smoke_gate]   {artifacts.json_path}")

    if approved:
        print_fn("[smoke_gate] gate BYPASSED by prior approval "
                 "(SMOKE_APPROVED=1 / --smoke-approved)")
        return artifacts

    if isatty:
        print_fn("")
        print_fn(f"SMOKE GATE: review {artifacts.txt_path}")
        print_fn(f"Type exactly  {TTY_APPROVE_LITERAL!r}  to continue, or "
                 "anything else to abort:")
        try:
            response = input_fn("> ")
        except (EOFError, KeyboardInterrupt):
            response = ""
        if response.strip() == TTY_APPROVE_LITERAL:
            print_fn("[smoke_gate] APPROVED via TTY. Continuing.")
            return artifacts
        print_fn(f"[smoke_gate] aborted (response={response!r}). "
                 "No production rows written.")
        exit_fn(1)
        return artifacts   # unreachable in prod; tests with stub exit_fn

    # Non-TTY (the real detached case) — halt for review.
    print_fn("")
    print_fn(f"SMOKE GATE: review {artifacts.txt_path} locally, "
             "then re-launch with SMOKE_APPROVED=1")
    print_fn(f"(exit code {EXIT_REVIEW_REQUIRED} = review-required, not a crash)")
    exit_fn(EXIT_REVIEW_REQUIRED)
    return artifacts   # unreachable in prod; tests with stub exit_fn
