"""Dry-run tests for `raw_response_save.save_raw`.

These verify that the helper persists raw to disk with the expected naming and
content BEFORE any edits land in `src/sweep.py` / `src/sweep_local.py`. If the
patch is later applied to `_save_source`'s neighbourhood, these tests
constitute the "prove it persists first" checkpoint requested in the brief.

Run from repo root:
    python -m pytest src/test_save_raw.py -q
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from raw_response_save import raw_filename, save_raw


@dataclass
class FakeRow:
    """Minimal stand-in for the `Row` dataclass in sweep_local.py/sweep.py."""
    problem_id: str
    constraint: str
    condition: str
    sample_idx: int


def test_filename_matches_save_source_convention():
    row = FakeRow(problem_id="lcb/abc356_c",
                  constraint="none", condition="unconstrained", sample_idx=3)
    assert raw_filename(row) == "lcb_abc356_c__none__unconstrained__s3__raw.txt"


def test_filename_swaps_slash_in_problem_id():
    row = FakeRow("tree/height", "none", "unconstrained", 0)
    assert raw_filename(row) == "tree_height__none__unconstrained__s0__raw.txt"


def test_writes_file_at_expected_path(tmp_path):
    row = FakeRow("lcb/abc356_c", "none", "unconstrained", 0)
    out = save_raw(str(tmp_path), row, "rationale prose\n```python\nprint(1)\n```\n")
    assert str(tmp_path / "lcb_abc356_c__none__unconstrained__s0__raw.txt") == out


def test_content_round_trip_preserves_prose_and_codeblock(tmp_path):
    raw = (
        "The constraint to avoid loops forces a functional style. "
        "I will use `pow(base, exp, mod)`.\n\n"
        "```python\n"
        "import sys\n"
        "print(pow(10, 3, 998244353))\n"
        "```\n"
    )
    row = FakeRow("lcb/abc357_d", "none", "unconstrained", 0)
    path = save_raw(str(tmp_path), row, raw)
    assert open(path).read() == raw


def test_overwrite_is_idempotent(tmp_path):
    row = FakeRow("lcb/abc356_c", "none", "unconstrained", 0)
    save_raw(str(tmp_path), row, "first")
    path = save_raw(str(tmp_path), row, "second")
    assert open(path).read() == "second"


def test_makes_parent_dir(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    row = FakeRow("lcb/abc356_c", "none", "unconstrained", 0)
    save_raw(str(nested), row, "x")
    assert (nested / "lcb_abc356_c__none__unconstrained__s0__raw.txt").exists()


def test_empty_raw_is_persisted_as_empty_file(tmp_path):
    row = FakeRow("lcb/abc356_c", "none", "unconstrained", 0)
    path = save_raw(str(tmp_path), row, "")
    assert open(path).read() == ""
