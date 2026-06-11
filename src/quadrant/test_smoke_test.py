"""Tests for the smoke-test runner itself (not for any deployed judge).

We verify with the offline FakeBackend that:
  - All 9 worked examples pass when the backend returns the canonical answers.
  - The runner correctly flags FAILs when a backend disagrees on the claim,
    endorsement, or ambiguous flag.
  - print_smoke_report's return value reflects pass/fail.

Run from src/: python -m pytest quadrant/test_smoke_test.py
"""
from __future__ import annotations
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadrant.claim_judge import FakeBackend
from quadrant.smoke_test import (
    CASES, build_fake_backend, print_smoke_report, run_smoke,
)


def test_canonical_fake_passes_all_cases(capsys):
    results = run_smoke(build_fake_backend())
    ok = print_smoke_report(results)
    assert ok
    assert len(results) == 9
    assert all(r.all_ok for r in results)


def test_runner_flags_wrong_claim():
    """If a backend gets case 3 (comprehension) wrong, the runner should
    surface that failure — this is the canary for 'judge is reasoning about
    code correctness instead of about the claim'."""
    # FakeBackend that gives the canonical answer to everything EXCEPT case 3,
    # which it mis-labels as no_claim (the common failure mode).
    rules = []
    for case in sorted(CASES, key=lambda c: -len(c.self_report)):
        if case.name == "3_comprehension_as_not_a_loop":
            rules.append((case.self_report, json.dumps({
                "self_compliance_claim": "no_claim",   # wrong
                "rule_endorsement": False,
                "confidence": 0.6, "ambiguous": False, "span": "",
            })))
        else:
            from quadrant.smoke_test import _CANNED_FAKE_RESPONSES
            rules.append((case.self_report, json.dumps(_CANNED_FAKE_RESPONSES[case.self_report])))
    be = FakeBackend(rules=rules)
    results = run_smoke(be)
    # 8 pass, 1 fail
    fails = [r for r in results if not r.all_ok]
    assert len(fails) == 1
    assert fails[0].case.name == "3_comprehension_as_not_a_loop"
    assert not fails[0].claim_ok
    assert print_smoke_report(results) is False


def test_runner_flags_wrong_endorsement():
    """Case 1 endorses the rule. A backend that returns endorsement=False
    must be flagged."""
    rules = []
    for case in sorted(CASES, key=lambda c: -len(c.self_report)):
        if case.name == "1_endorses_no_claim":
            rules.append((case.self_report, json.dumps({
                "self_compliance_claim": "no_claim",
                "rule_endorsement": False,    # wrong — should be True
                "confidence": 0.9, "ambiguous": False, "span": "",
            })))
        else:
            from quadrant.smoke_test import _CANNED_FAKE_RESPONSES
            rules.append((case.self_report, json.dumps(_CANNED_FAKE_RESPONSES[case.self_report])))
    be = FakeBackend(rules=rules)
    results = run_smoke(be)
    fails = [r for r in results if not r.all_ok]
    assert len(fails) == 1
    assert fails[0].case.name == "1_endorses_no_claim"
    assert not fails[0].endorsement_ok
    assert fails[0].claim_ok    # claim is correct, only endorsement wrong


def test_runner_flags_wrong_ambiguous():
    """Case 9 must be ambiguous=True. A backend that returns False is wrong."""
    rules = []
    for case in sorted(CASES, key=lambda c: -len(c.self_report)):
        if case.name == "9_contradictory_first_line_wins_ambiguous":
            rules.append((case.self_report, json.dumps({
                "self_compliance_claim": "asserts_compliance",
                "rule_endorsement": False,
                "confidence": 0.95,
                "ambiguous": False,   # wrong — must be True
                "span": "NO.",
            })))
        else:
            from quadrant.smoke_test import _CANNED_FAKE_RESPONSES
            rules.append((case.self_report, json.dumps(_CANNED_FAKE_RESPONSES[case.self_report])))
    be = FakeBackend(rules=rules)
    results = run_smoke(be)
    fails = [r for r in results if not r.all_ok]
    assert len(fails) == 1
    assert not fails[0].ambiguous_ok
