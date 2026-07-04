"""Tests for the incremental judge runner.

Covers:
  - happy path: every input row produces one output row
  - idempotent resume: rerunning skips completed keys, judges only missing
  - partial trailing line in output (simulated crash mid-fsync) is dropped
    on resume and the underlying input is re-judged
  - per-line fsync: rows are visible on disk before run() returns
  - parse_error and ambiguous counters in RunStats
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadrant.claim_judge import FakeBackend
from quadrant.judge_runner import already_completed_keys, run


def _resp(claim="no_claim", endorsement=False, conf=0.9, ambiguous=False, span=""):
    return json.dumps({
        "self_compliance_claim": claim, "rule_endorsement": endorsement,
        "confidence": conf, "ambiguous": ambiguous, "span": span,
    })


def _write_inputs(path: str, rows: list[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


# ----------------------- happy path ----------------------

def test_run_produces_one_output_per_input():
    with tempfile.TemporaryDirectory() as d:
        in_p = f"{d}/in.jsonl"
        out_p = f"{d}/out.jsonl"
        _write_inputs(in_p, [
            {"model": "base", "problem_id": "p1", "sample_idx": 0,
             "self_report_text": "I avoided loops."},
            {"model": "base", "problem_id": "p1", "sample_idx": 1,
             "self_report_text": "I used a for loop."},
        ])
        be = FakeBackend(rules=[
            ("avoided loops", _resp("asserts_compliance", span="avoided loops")),
            ("for loop",      _resp("asserts_violation",  span="for loop")),
        ])
        stats = run(in_p, out_p, be, progress_every=0)
        assert stats.total_inputs == 2
        assert stats.judged_this_run == 2
        assert stats.already_done == 0
        rows = _read_jsonl(out_p)
        assert len(rows) == 2
        assert rows[0]["judge_label"] == "asserts_compliance"
        assert rows[1]["judge_label"] == "asserts_violation"
        # input fields preserved
        assert rows[0]["model"] == "base"
        assert rows[0]["problem_id"] == "p1"
        assert rows[0]["sample_idx"] == 0
        # rubric version stamped
        assert rows[0]["rubric_version"] == "v1"


# ----------------------- idempotent resume ----------------------

def test_resume_skips_completed_keys_and_judges_only_missing():
    with tempfile.TemporaryDirectory() as d:
        in_p = f"{d}/in.jsonl"
        out_p = f"{d}/out.jsonl"
        _write_inputs(in_p, [
            {"model": "base", "problem_id": "p1", "sample_idx": 0,
             "self_report_text": "A"},
            {"model": "base", "problem_id": "p1", "sample_idx": 1,
             "self_report_text": "B"},
            {"model": "base", "problem_id": "p2", "sample_idx": 0,
             "self_report_text": "C"},
        ])
        # Pre-populate output with the first two rows as if the prior run got
        # that far before being killed.
        with open(out_p, "w") as f:
            f.write(json.dumps({
                "model": "base", "problem_id": "p1", "sample_idx": 0,
                "self_report_text": "A",
                "judge_label": "no_claim", "judge_endorsement": False,
                "judge_confidence": 0.9, "judge_ambiguous": False,
                "judge_span": "", "judge_parse_error": None,
                "rubric_version": "v1",
            }) + "\n")
            f.write(json.dumps({
                "model": "base", "problem_id": "p1", "sample_idx": 1,
                "self_report_text": "B",
                "judge_label": "asserts_compliance", "judge_endorsement": False,
                "judge_confidence": 0.95, "judge_ambiguous": False,
                "judge_span": "x", "judge_parse_error": None,
                "rubric_version": "v1",
            }) + "\n")

        be = FakeBackend(rules=[("C", _resp("asserts_violation", span="loop"))])
        stats = run(in_p, out_p, be, progress_every=0)
        # Only the third row should have been judged.
        assert stats.total_inputs == 3
        assert stats.already_done == 2
        assert stats.judged_this_run == 1

        rows = _read_jsonl(out_p)
        assert len(rows) == 3
        # Existing rows preserved verbatim (no re-judging)
        assert rows[0]["judge_label"] == "no_claim"
        assert rows[1]["judge_label"] == "asserts_compliance"
        # New row appended
        assert rows[2]["judge_label"] == "asserts_violation"
        assert rows[2]["sample_idx"] == 0
        assert rows[2]["problem_id"] == "p2"


# ----------------------- partial line on prior crash ----------------------

def test_partial_trailing_line_is_dropped_and_input_re_judged():
    """Simulate a crash mid-fsync: the last line of the output JSONL is a
    truncated, unparseable fragment. On resume, the runner should drop the
    fragment, mark that input as not-done, and judge it cleanly."""
    with tempfile.TemporaryDirectory() as d:
        in_p = f"{d}/in.jsonl"
        out_p = f"{d}/out.jsonl"
        _write_inputs(in_p, [
            {"model": "base", "problem_id": "p1", "sample_idx": 0,
             "self_report_text": "A"},
            {"model": "base", "problem_id": "p1", "sample_idx": 1,
             "self_report_text": "B"},
        ])
        # Write a clean first row + a partial second row.
        with open(out_p, "w") as f:
            f.write(json.dumps({
                "model": "base", "problem_id": "p1", "sample_idx": 0,
                "self_report_text": "A",
                "judge_label": "no_claim", "judge_endorsement": False,
                "judge_confidence": 0.9, "judge_ambiguous": False,
                "judge_span": "", "judge_parse_error": None,
                "rubric_version": "v1",
            }) + "\n")
            f.write('{"model": "base", "problem_id": "p1", "sample_idx": 1, "self_report')  # truncated!

        # Keys already done should NOT include sample_idx=1
        done = already_completed_keys(out_p)
        assert ("base", "p1", 0) in done
        assert ("base", "p1", 1) not in done

        be = FakeBackend(rules=[("B", _resp("asserts_compliance", span="x"))])
        stats = run(in_p, out_p, be, progress_every=0)
        assert stats.judged_this_run == 1
        # Output now has the clean first row + the new clean second row, but
        # the partial fragment is still on disk — _read_jsonl_lenient ignores
        # it on subsequent runs, so the file remains "correct" for downstream
        # readers using the same lenient parse.
        # (We don't try to rewrite the file in-place; that would risk dataloss
        # if interrupted mid-rewrite. Append-only is the safer policy.)


# ----------------------- per-line fsync ----------------------

class _SpyBackend:
    """Backend that snapshots the output file AFTER each call returns. Lets us
    prove the previous row was written to disk before the next call started."""
    def __init__(self, out_path, snapshots):
        self.out_path = out_path
        self.snapshots = snapshots
        self.call = 0

    def chat(self, system: str, user: str) -> str:
        # snapshot output BEFORE we respond — captures state after previous fsync
        contents = open(self.out_path).read() if os.path.exists(self.out_path) else ""
        self.snapshots.append(contents)
        self.call += 1
        return _resp("no_claim", span="")


def test_each_judgment_fsyncd_before_next_call():
    """Snapshot the on-disk output between calls — by call N, the prior N-1
    judgments must be present on disk."""
    with tempfile.TemporaryDirectory() as d:
        in_p = f"{d}/in.jsonl"
        out_p = f"{d}/out.jsonl"
        _write_inputs(in_p, [
            {"model": "base", "problem_id": "p1", "sample_idx": i,
             "self_report_text": f"r{i}"} for i in range(3)
        ])
        snapshots: list[str] = []
        be = _SpyBackend(out_p, snapshots)
        run(in_p, out_p, be, progress_every=0)

        # Three calls, three snapshots
        assert len(snapshots) == 3
        # Before the first call: file doesn't exist or is empty
        assert snapshots[0] == ""
        # Before the second call: first row already on disk
        assert snapshots[1].count("\n") == 1
        # Before the third call: two rows on disk
        assert snapshots[2].count("\n") == 2


# ----------------------- counters ----------------------

def test_stats_counts_parse_errors_and_ambiguous():
    with tempfile.TemporaryDirectory() as d:
        in_p = f"{d}/in.jsonl"
        out_p = f"{d}/out.jsonl"
        _write_inputs(in_p, [
            {"model": "base", "problem_id": "p1", "sample_idx": 0,
             "self_report_text": "trigger_parse_error"},
            {"model": "base", "problem_id": "p1", "sample_idx": 1,
             "self_report_text": "trigger_ambiguous"},
            {"model": "base", "problem_id": "p1", "sample_idx": 2,
             "self_report_text": "clean"},
        ])
        be = FakeBackend(rules=[
            ("trigger_parse_error", "not json"),
            ("trigger_ambiguous", _resp("asserts_compliance", ambiguous=True)),
            ("clean",  _resp("no_claim")),
        ])
        stats = run(in_p, out_p, be, progress_every=0)
        assert stats.parse_errors == 1
        assert stats.ambiguous == 2     # parse_error row also gets ambiguous=True (safe default)
