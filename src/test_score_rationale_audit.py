"""Tests for the κ-scoring tool. No compute; pure logic."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from score_rationale_audit import _cohen_kappa  # type: ignore


def test_kappa_perfect_agreement():
    assert _cohen_kappa(25, 0, 0, 25) == pytest.approx(1.0)


def test_kappa_zero_observed_agreement():
    # All disagree; κ should be very negative or zero
    k = _cohen_kappa(0, 25, 25, 0)
    assert k < 0.0


def test_kappa_chance_agreement_only():
    # Both raters say YES 50% of the time, independent → p_obs = p_exp → κ ≈ 0.
    k = _cohen_kappa(13, 12, 12, 13)  # near-independent 50/50
    assert abs(k) < 0.1


def test_kappa_high():
    # 90% raw agreement, balanced classes
    k = _cohen_kappa(22, 3, 2, 23)
    assert k > 0.7


def test_kappa_empty():
    assert _cohen_kappa(0, 0, 0, 0) == 0.0


def test_kappa_all_yes_both_raters():
    # If both raters always YES, base rate is 100% → p_exp = 1 → κ = 1 if p_obs = 1.
    assert _cohen_kappa(50, 0, 0, 0) == pytest.approx(1.0)


def test_endtoend_perfect(tmp_path):
    rid_a = tmp_path / "a.jsonl"
    rid_b = tmp_path / "b.jsonl"
    rows = []
    for i, v in enumerate(["YES"] * 10 + ["NO"] * 10):
        rows.append({"response_id": f"r{i}", "asserts_compliance": v, "justification": "x"})
    rid_a.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rid_b.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "k.md"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "score_rationale_audit.py"),
         str(rid_a), str(rid_b), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert "**Cohen's κ:** 1.000" in text
    assert "**Raw agreement:** 100.0%" in text


def test_endtoend_disagreement(tmp_path):
    rid_a = tmp_path / "a.jsonl"
    rid_b = tmp_path / "b.jsonl"
    a_rows = [{"response_id": f"r{i}", "asserts_compliance": v, "justification": f"agent says {v}"}
              for i, v in enumerate(["YES", "YES", "NO", "NO"])]
    b_rows = [{"response_id": f"r{i}", "asserts_compliance": v, "justification": f"owner says {v}"}
              for i, v in enumerate(["YES", "NO", "NO", "YES"])]
    rid_a.write_text("".join(json.dumps(r) + "\n" for r in a_rows))
    rid_b.write_text("".join(json.dumps(r) + "\n" for r in b_rows))
    out = tmp_path / "k.md"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "score_rationale_audit.py"),
         str(rid_a), str(rid_b), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert "r1" in text and "r3" in text  # disagreement IDs
    assert "agent says YES" in text
    assert "owner says NO" in text
