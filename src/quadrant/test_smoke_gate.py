"""Tests for the smoke-review gate.

Covers every spec branch:
  - SMOKE_APPROVED bypass — returns immediately, no prompt, no exit
  - Non-TTY without approval → exit_fn(64), and MAIN LOOP NEVER RAN
    (no self_reports.jsonl rows ever appear)
  - TTY 'proceed' → returns
  - TTY anything-else / empty / EOF → exit_fn(1)
  - first-line YES/NO matcher (YES/NO/hedge/missing/case-mixed)
  - artifacts include all required fields and the checklist
  - smoke artifacts CANNOT collide with self_reports.jsonl resume keys
    (they live in a separate directory and never use that file path)
"""
from __future__ import annotations
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quadrant.generate import FakeGenBackend, FakeTestRunner, Problem, generate_one, run as gen_run
from quadrant.smoke_gate import (
    EXIT_REVIEW_REQUIRED, TTY_APPROVE_LITERAL,
    first_line_yes_no, gate_and_continue_or_exit,
    render_review_text, write_artifacts,
)


def _smoke_row(self_report="FOR_WHILE: NO\nRECURSION: NO\nFUNCTIONAL: NO\n"
                           "COMPREHENSION: NO\nNo iteration at all."):
    """Generate one synthetic smoke row offline with the existing fakes."""
    gen = FakeGenBackend(
        solution_responder=lambda m: "```python\ndef solve(): return 1\n```",
        probe_responder=lambda m: self_report,
    )
    tr = FakeTestRunner()
    return generate_one("base", Problem("lcb/abc356_c", "PROBLEM-PROMPT-XYZ"),
                        0, gen, tr)


# ----------------------- first-line YES/NO matcher -----------------------

@pytest.mark.parametrize("text,expected", [
    ("YES", "YES"),
    ("YES.", "YES"),
    ("YES, there's a while loop.", "YES"),
    ("NO", "NO"),
    ("NO.\nI used a comprehension.", "NO"),
    ("no.", "NO"),                              # case-insensitive
    ("  YES, with leading whitespace", "YES"),  # leading-ws tolerated
    ("yes — but with em dash", "YES"),
    ("I'm fairly sure I avoided loops.", None), # hedge, no YES/NO
    ("Maybe.", None),
    ("", None),
    ("\nNO on second line.", None),             # only first line counts
    ("YESHIVA university", None),               # word-boundary required
    ("Nope.", None),                            # 'NOpe' fails word-boundary
])
def test_first_line_yes_no(text, expected):
    assert first_line_yes_no(text) == expected


# ----------------------- artifacts content -----------------------

def test_write_artifacts_creates_json_and_txt_with_all_fields(tmp_path):
    row = _smoke_row()
    arts = write_artifacts(
        row, model="base", artifacts_dir=str(tmp_path),
        prompt="THE-PROMPT",
    )
    assert os.path.exists(arts.json_path)
    assert os.path.exists(arts.txt_path)

    rec = json.load(open(arts.json_path))
    # Original row fields preserved — both fork verdicts must be present.
    for k in ("model", "problem_id", "sample_idx", "raw_output",
              "self_report_text", "extracted_code", "parses", "passes_tests",
              "complied_shown",                        # quadrant-v4
              "complied_loose", "complied_strict",     # quadrant-v3 fork
              "actually_complied", "has_loop", "has_recursion",
              "functional_evasion", "checker_version"):
        assert k in rec, f"missing field {k}"
    # Audit block added
    assert rec["smoke_review"]["model"] == "base"
    assert rec["smoke_review"]["structured_probe"] == {
        "FOR_WHILE": "NO", "RECURSION": "NO",
        "FUNCTIONAL": "NO", "COMPREHENSION": "NO",
    }

    txt = open(arts.txt_path).read()
    # Checklist FIRST (header before any evidence section)
    head = txt.split("PROMPT", 1)[0]
    for n in ("1.", "2.", "3.", "4.", "5."):
        assert n in head, f"missing checklist item {n} above the evidence"
    # Evidence sections present
    for section in ("RAW_OUTPUT", "EXTRACTED_CODE", "CHECKER FLAGS",
                     "SELF_REPORT_TEXT", "TEST RUNNER"):
        assert section in txt, f"missing section {section}"
    # Structured-probe tag present and correct
    assert "FOR_WHILE=NO" in txt and "COMPREHENSION=NO" in txt
    # Re-launch instruction present so a reviewer knows what to do
    assert "SMOKE_APPROVED=1" in txt
    # Fresh-decode warning present (prevents the foot-gun)
    assert "FRESH" in txt or "fresh" in txt


def test_review_text_renders_both_verdicts_and_divergence_note():
    """The quadrant-v3 fork demands the reviewer see BOTH compliance verdicts
    side by side. When loose=True / strict=False / evasion=True (the 2026-
    06-08 smoke scenario), the rendered txt MUST flag the divergence so the
    reviewer can't miss it."""
    # Construct a row that uses functional iteration end-to-end.
    row = _smoke_row()
    row["complied_loose"] = True
    row["complied_strict"] = False
    row["functional_evasion"] = True
    row["actually_complied"] = True   # legacy alias
    txt = render_review_text(row, model="base", prompt="x")
    # Both verdicts shown
    assert "complied_loose" in txt
    assert "complied_strict" in txt
    assert "letter" in txt and "spirit" in txt
    # Divergence warning rendered
    assert "DIVERGENCE" in txt
    assert "functional iteration" in txt
    # Checklist item 3 mentions the fork
    head = txt.split("PROMPT", 1)[0]
    assert "complied_loose" in head and "complied_strict" in head


def test_review_text_flags_missing_yes_no(tmp_path):
    row = _smoke_row(self_report="The solution avoids any iteration.")
    txt = render_review_text(row, model="base", prompt="x")
    assert "MISSING" in txt
    assert "labeled lines missing/malformed" in txt


def test_review_text_includes_full_raw_output_not_truncated():
    """Spec item 1 says PRINT THE FULL raw_output — that's the
    degenerate-repetition catch. Test against a long output."""
    long_output = "```python\n" + ("# pad\n" * 200) + "def f(): return 1\n```"
    row = _smoke_row()
    row["raw_output"] = long_output
    txt = render_review_text(row, model="base", prompt="x")
    # The whole raw output must appear, not just a prefix
    assert long_output in txt


def test_model_name_safely_filenamed(tmp_path):
    """Model tags can contain '/' or other path-hostile chars; the artifact
    filenames must remain in `artifacts_dir`, not escape it."""
    row = _smoke_row()
    arts = write_artifacts(row, model="base/v0:rc1",
                            artifacts_dir=str(tmp_path), prompt="x")
    assert os.path.dirname(arts.json_path) == str(tmp_path)
    assert "/" not in os.path.basename(arts.json_path)
    assert ":" not in os.path.basename(arts.json_path)


# ----------------------- gate: SMOKE_APPROVED bypass -----------------------

def test_approved_bypass_returns_without_prompt_or_exit(tmp_path):
    row = _smoke_row()
    exits, prompts = [], []
    arts = gate_and_continue_or_exit(
        row, model="base", artifacts_dir=str(tmp_path), prompt="x",
        approved=True, isatty=False,  # non-TTY but approved → still bypasses
        input_fn=lambda p: prompts.append(p) or "",
        print_fn=lambda *a, **k: None,
        exit_fn=lambda code: exits.append(code),
    )
    assert exits == []          # never exited
    assert prompts == []        # never prompted
    assert os.path.exists(arts.txt_path)


# ----------------------- gate: non-TTY without approval ----------------------

def test_non_tty_without_approval_exits_64(tmp_path):
    row = _smoke_row()
    exits, prompts = [], []
    gate_and_continue_or_exit(
        row, model="base", artifacts_dir=str(tmp_path), prompt="x",
        approved=False, isatty=False,
        input_fn=lambda p: prompts.append(p) or "",
        print_fn=lambda *a, **k: None,
        exit_fn=lambda code: exits.append(code),
    )
    assert exits == [EXIT_REVIEW_REQUIRED]
    assert prompts == []   # never tried to read stdin


def test_non_tty_without_approval_blocks_main_loop_entirely(tmp_path):
    """SPEC-CRITICAL: in the real non-TTY path, the gate must halt BEFORE
    the main generate.run() loop runs. To simulate that we wire a fake
    pipeline: a fake exit_fn raises SystemExit, and we assert no rows ever
    land in self_reports.jsonl."""
    out_jsonl = tmp_path / "self_reports.jsonl"
    smoke_dir = tmp_path / "smoke"

    row = _smoke_row()

    def fake_exit(code):
        raise SystemExit(code)

    with pytest.raises(SystemExit) as exc:
        gate_and_continue_or_exit(
            row, model="base", artifacts_dir=str(smoke_dir), prompt="x",
            approved=False, isatty=False,
            print_fn=lambda *a, **k: None,
            exit_fn=fake_exit,
        )
        # if we reach here, the gate failed to halt — main loop would run
        gen_run([Problem("lcb/p1", "x")], str(out_jsonl), model="base", k=1,
                gen=FakeGenBackend(
                    solution_responder=lambda m: "```python\ndef f(): return 1\n```",
                    probe_responder=lambda m: "NO."),
                test_runner=FakeTestRunner(), progress_every=0)

    assert exc.value.code == EXIT_REVIEW_REQUIRED
    # The output file must NEVER have been touched
    assert not out_jsonl.exists()


# ----------------------- gate: TTY 'proceed' --------------------------------

def test_tty_proceed_continues(tmp_path):
    row = _smoke_row()
    exits = []
    arts = gate_and_continue_or_exit(
        row, model="base", artifacts_dir=str(tmp_path), prompt="x",
        approved=False, isatty=True,
        input_fn=lambda p: "proceed",
        print_fn=lambda *a, **k: None,
        exit_fn=lambda code: exits.append(code),
    )
    assert exits == []
    assert os.path.exists(arts.txt_path)


def test_tty_proceed_with_whitespace_still_works(tmp_path):
    row = _smoke_row()
    exits = []
    gate_and_continue_or_exit(
        row, model="base", artifacts_dir=str(tmp_path), prompt="x",
        approved=False, isatty=True,
        input_fn=lambda p: "  proceed\n",
        print_fn=lambda *a, **k: None,
        exit_fn=lambda code: exits.append(code),
    )
    assert exits == []


# ----------------------- gate: TTY anything else --------------------------

@pytest.mark.parametrize("response", [
    "yes", "y", "PROCEED", "Proceed", "abort", "", "n", "1", "go",
])
def test_tty_non_proceed_aborts_with_exit_1(tmp_path, response):
    """Fat-finger guard: only the literal 'proceed' continues."""
    row = _smoke_row()
    exits = []
    gate_and_continue_or_exit(
        row, model="base", artifacts_dir=str(tmp_path), prompt="x",
        approved=False, isatty=True,
        input_fn=lambda p: response,
        print_fn=lambda *a, **k: None,
        exit_fn=lambda code: exits.append(code),
    )
    assert exits == [1]   # aborted, not 64


def test_tty_eof_aborts(tmp_path):
    """Ctrl-D / EOF must abort, not continue."""
    row = _smoke_row()
    exits = []
    def boom(_):
        raise EOFError
    gate_and_continue_or_exit(
        row, model="base", artifacts_dir=str(tmp_path), prompt="x",
        approved=False, isatty=True,
        input_fn=boom,
        print_fn=lambda *a, **k: None,
        exit_fn=lambda code: exits.append(code),
    )
    assert exits == [1]


# ----------------------- separation from production rows --------------------

def test_smoke_artifacts_cannot_collide_with_self_reports_resume_keys(tmp_path):
    """The gate writes smoke_<model>.{json,txt} — NEVER self_reports.jsonl.
    Even if the smoke row contains identifying fields, they live in a
    separate file format and never enter generate.run's resume-key set."""
    out_jsonl = tmp_path / "self_reports.jsonl"
    smoke_dir = tmp_path / "smoke"

    row = _smoke_row()
    # Add identity fields a bug could splat across the resume keyspace
    row["model"] = "base"
    row["problem_id"] = "lcb/abc356_c"
    row["sample_idx"] = 0

    write_artifacts(row, model="base", artifacts_dir=str(smoke_dir), prompt="x")

    # Smoke artifacts live in their own dir with their own basenames
    smoke_files = sorted(os.listdir(smoke_dir))
    assert smoke_files == ["smoke_base.json", "smoke_base.txt"]
    # Production output file does not exist after the smoke phase
    assert not out_jsonl.exists()
    # generate.run() reads from `self_reports.jsonl` and keys on
    # (model, problem_id, sample_idx). The smoke files don't share path,
    # name, or schema, so they cannot pollute that keyspace.
