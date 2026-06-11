"""Tests for build_problems.py.

Covers:
  - happy path: small fake loader → 17-row output with required schema
  - count assertion fires when source has wrong number of rows
  - duplicate problem_id assertion
  - empty-prompt assertion
  - constraint-absence guard fires when prompt contains the rule
  - determinism: re-running yields identical SHA-256
  - clobber guard: refuses to overwrite different-hash output without --force
  - schema: problem_id + prompt present; audit fields included
  - sort by problem_id is stable
"""
from __future__ import annotations
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quadrant.build_problems import (
    CONSTRAINT_PHRASES, EXPECTED_COUNT, build,
)


def _fake_loader(rows):
    return lambda path: rows


def _src(pid, prompt="problem prompt body"):
    """Builds a row matching the source schema in data/problems_lcb_clean17.jsonl."""
    return {
        "id": pid,
        "benchmark": "lcb_medium",
        "mode": "stdin",
        "prompt": prompt,
        "entry_point": None,
        "stdin_tests": [],
        "canonical": None,
        "contest_date": "2024-09-01",
    }


def _17(prefix="lcb/abc"):
    return [_src(f"{prefix}{i:03d}_c") for i in range(17)]


# ----------------------- happy path -----------------------

def test_happy_path_writes_files_with_required_schema():
    with tempfile.TemporaryDirectory() as d:
        out = f"{d}/out.jsonl"; meta = f"{d}/out.meta.json"
        res = build("fake-source", out, meta,
                    loader=_fake_loader(_17()))
        assert res.count == 17
        rows = [json.loads(l) for l in open(out)]
        assert len(rows) == 17
        # required fields
        assert all("problem_id" in r and "prompt" in r for r in rows)
        # audit fields
        for r in rows:
            assert "source_id" in r and "lcb_version" in r
        # canonical id preserved
        assert all(r["problem_id"].startswith("lcb/") for r in rows)
        # meta sidecar
        m = json.load(open(meta))
        assert m["count"] == 17
        assert m["out_sha256"] == res.out_sha256
        assert m["selection_method"] == "manifest_reuse"
        assert m["cohort_label"].startswith("clean-17")


def test_sort_by_problem_id_is_stable():
    rows = _17()
    # shuffle deliberately
    rows = list(reversed(rows))
    with tempfile.TemporaryDirectory() as d:
        out = f"{d}/out.jsonl"; meta = f"{d}/out.meta.json"
        build("fake", out, meta, loader=_fake_loader(rows))
        emitted_ids = [json.loads(l)["problem_id"] for l in open(out)]
        assert emitted_ids == sorted(emitted_ids)


# ----------------------- assertions ----------------------

def test_count_assertion_fires_on_wrong_size():
    rows = _17()[:10]
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(AssertionError, match="cohort count mismatch"):
            build("fake", f"{d}/o.jsonl", f"{d}/o.meta.json",
                  loader=_fake_loader(rows))


def test_count_assertion_message_mentions_silent_swap():
    """Make sure the error message warns about silent cohort swaps — that's
    the whole point of the gate."""
    rows = _17()[:10]
    with tempfile.TemporaryDirectory() as d:
        try:
            build("fake", f"{d}/o.jsonl", f"{d}/o.meta.json",
                  loader=_fake_loader(rows))
        except AssertionError as e:
            assert "silently" in str(e)
        else:
            pytest.fail("expected AssertionError")


def test_duplicate_problem_id_fails():
    rows = _17()
    rows[1] = _src("lcb/abc000_c")  # duplicate of rows[0]
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(AssertionError, match="duplicate problem_id"):
            build("fake", f"{d}/o.jsonl", f"{d}/o.meta.json",
                  loader=_fake_loader(rows))


def test_empty_prompt_fails():
    rows = _17()
    rows[3]["prompt"] = "   \n  "
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(AssertionError, match="empty/whitespace prompt"):
            build("fake", f"{d}/o.jsonl", f"{d}/o.meta.json",
                  loader=_fake_loader(rows))


def test_constraint_phrase_in_prompt_fails():
    rows = _17()
    rows[5]["prompt"] = (
        "Solve this. Hint: Do not use any `for` or `while` loops in your code."
    )
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(AssertionError, match="constraint string found"):
            build("fake", f"{d}/o.jsonl", f"{d}/o.meta.json",
                  loader=_fake_loader(rows))


def test_constraint_phrase_check_is_case_insensitive():
    rows = _17()
    rows[5]["prompt"] = "Solve. do not use recursion please."
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(AssertionError):
            build("fake", f"{d}/o.jsonl", f"{d}/o.meta.json",
                  loader=_fake_loader(rows))


def test_constraint_phrases_listed_in_meta():
    """Audit trail: the meta sidecar should record exactly which phrases were
    checked, so a reader can verify what 'constraint check passed' meant."""
    with tempfile.TemporaryDirectory() as d:
        out = f"{d}/o.jsonl"; meta = f"{d}/o.meta.json"
        build("fake", out, meta, loader=_fake_loader(_17()))
        m = json.load(open(meta))
        assert m["constraint_check_phrases"] == list(CONSTRAINT_PHRASES)
        assert m["constraint_check_status"] == "passed"


# ----------------------- determinism + clobber guard ----------------------

def test_determinism_re_run_produces_identical_hash():
    rows = _17()
    with tempfile.TemporaryDirectory() as d:
        out = f"{d}/o.jsonl"; meta = f"{d}/o.meta.json"
        r1 = build("fake", out, meta, loader=_fake_loader(rows))
        # second run: same source → same hash; should NOT raise
        r2 = build("fake", out, meta, loader=_fake_loader(rows))
        assert r1.out_sha256 == r2.out_sha256


def test_clobber_guard_refuses_different_hash_without_force(tmp_path):
    out = tmp_path / "o.jsonl"; meta = tmp_path / "o.meta.json"
    build("fake", str(out), str(meta), loader=_fake_loader(_17()))
    # now write a CORRECT-sized but different-prompts manifest → different hash
    rows2 = [_src(f"lcb/zzz{i:03d}_d", prompt=f"different prompt {i}")
             for i in range(17)]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        build("fake", str(out), str(meta), loader=_fake_loader(rows2))


def test_clobber_guard_allows_force(tmp_path):
    out = tmp_path / "o.jsonl"; meta = tmp_path / "o.meta.json"
    build("fake", str(out), str(meta), loader=_fake_loader(_17()))
    rows2 = [_src(f"lcb/zzz{i:03d}_d", prompt=f"different prompt {i}")
             for i in range(17)]
    res = build("fake", str(out), str(meta), loader=_fake_loader(rows2), force=True)
    # output reflects the new rows
    emitted = [json.loads(l)["problem_id"] for l in open(out)]
    assert emitted[0].startswith("lcb/zzz")


# ----------------------- generate.py reader contract -----------------------

def test_emitted_rows_load_via_generate_load_problems(tmp_path):
    """End-to-end: build outputs a file that quadrant/generate.py:load_problems
    consumes without error and surfaces the canonical ids."""
    from quadrant.generate import load_problems
    out = tmp_path / "o.jsonl"; meta = tmp_path / "o.meta.json"
    build("fake", str(out), str(meta), loader=_fake_loader(_17()))
    problems = load_problems(str(out))
    assert len(problems) == 17
    assert all(p.problem_id.startswith("lcb/") for p in problems)
    assert all(p.prompt for p in problems)
