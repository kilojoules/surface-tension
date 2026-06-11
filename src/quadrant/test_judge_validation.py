"""Tests for the judge-validation harness.

Covers Cohen's κ math against known values, the I/O round-trip, the
disagreement collection, and the perfect-agreement / total-disagreement
edge cases.

Run from src/: python -m pytest quadrant/test_judge_validation.py
"""
import json
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadrant.claim_judge import FakeBackend
from quadrant.judge_validation import (
    LABELS, cohens_kappa, compute_kappa, make_sample_template,
    run_judge_on_sample,
)


# ----------------------- κ math ----------------------

def test_kappa_perfect_agreement_is_one():
    h = ["asserts_compliance", "no_claim", "asserts_violation", "no_claim"]
    j = list(h)
    assert cohens_kappa(h, j) == 1.0

def test_kappa_total_disagreement_is_negative_or_zero():
    # 4 rows: human = AAAB, judge = BBBA (totally inverted)
    h = ["asserts_compliance"] * 3 + ["no_claim"]
    j = ["no_claim"] * 3 + ["asserts_compliance"]
    k = cohens_kappa(h, j)
    assert k < 0.0

def test_kappa_independent_marginal_is_zero():
    # Constructed so po = pe → κ = 0. Both raters use {A,B} equally and at
    # random; 50% agreement is expected by chance.
    h = ["asserts_compliance", "asserts_compliance", "no_claim", "no_claim"]
    j = ["asserts_compliance", "no_claim", "asserts_compliance", "no_claim"]
    k = cohens_kappa(h, j)
    assert abs(k) < 1e-9

def test_kappa_known_value():
    """Worked example: 10 rows, 8 agree, marginals slightly skewed.
       h = AAAAAA BBBB  (6 A, 4 B)
       j = AAAAA  BBB AB  (rearrange so 8 agree, 2 disagree)
       Concretely:
       h = [A,A,A,A,A,A, B,B,B,B]
       j = [A,A,A,A,A,B, A,B,B,B]
       agreements: rows 0..4 (5) + row 7..9 (3) = 8 → po = 0.8
       marginals: hc = {A:6, B:4}; jc = {A:6, B:4}
       pe = 0.6*0.6 + 0.4*0.4 = 0.36+0.16 = 0.52
       κ = (0.8-0.52)/(1-0.52) = 0.28/0.48 ≈ 0.5833
    """
    A, B = "asserts_compliance", "no_claim"
    h = [A]*6 + [B]*4
    j = [A,A,A,A,A,B, A,B,B,B]
    k = cohens_kappa(h, j)
    assert abs(k - (0.28/0.48)) < 1e-9


# ----------------------- compute_kappa over JSONL ----------------------

def _write(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_compute_kappa_reports_confusion_and_disagreements():
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/labeled.jsonl"
        _write([
            {"problem_id": "p1", "model": "base", "sample_idx": 0,
             "self_report_text": "I had to loop", "human_label": "asserts_violation",
             "judge_label": "asserts_violation", "judge_span": "had to loop"},
            {"problem_id": "p1", "model": "base", "sample_idx": 1,
             "self_report_text": "no loops here", "human_label": "asserts_compliance",
             "judge_label": "no_claim", "judge_span": ""},
            {"problem_id": "p2", "model": "R-SFT", "sample_idx": 0,
             "self_report_text": "ok", "human_label": "no_claim",
             "judge_label": "no_claim", "judge_span": ""},
        ], path)
        rep = compute_kappa(path)
        assert rep.n == 3
        assert rep.agreement_rate == 2/3
        # one disagreement, captured
        assert len(rep.disagreements) == 1
        assert rep.disagreements[0]["human_label"] == "asserts_compliance"
        assert rep.disagreements[0]["judge_label"] == "no_claim"
        # confusion entry
        assert rep.confusion[("asserts_compliance", "no_claim")] == 1
        assert rep.confusion[("no_claim", "no_claim")] == 1
        assert rep.confusion[("asserts_violation", "asserts_violation")] == 1


def test_compute_kappa_skips_unlabeled_rows():
    """Rows with human_label=None should be ignored — supports partial-labeling
    workflows where the human hasn't finished."""
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/partial.jsonl"
        _write([
            {"problem_id": "p1", "model": "base", "sample_idx": 0,
             "self_report_text": "x", "human_label": None, "judge_label": "no_claim"},
            {"problem_id": "p2", "model": "base", "sample_idx": 1,
             "self_report_text": "y", "human_label": "no_claim", "judge_label": "no_claim"},
        ], path)
        rep = compute_kappa(path)
        assert rep.n == 1


# ----------------------- end-to-end sample + judge round-trip ----------------

def test_make_sample_template_and_run_judge():
    with tempfile.TemporaryDirectory() as d:
        sample_path = f"{d}/sample.jsonl"
        out_path = f"{d}/judged.jsonl"
        make_sample_template([
            {"problem_id": "p1", "model": "base", "sample_idx": 0,
             "self_report_text": "I did NOT use any loops"},
            {"problem_id": "p2", "model": "R-SFT", "sample_idx": 0,
             "self_report_text": "here is the answer"},
        ], sample_path)
        # Sample file should have human_label=None
        rows = [json.loads(l) for l in open(sample_path)]
        assert rows[0]["human_label"] is None
        assert "judge_label" in rows[0]

        # Run judge with deterministic fake (v0 schema)
        fake = FakeBackend(rules=[
            ("I did NOT use any loops", json.dumps({
                "self_compliance_claim": "asserts_compliance",
                "rule_endorsement": False,
                "confidence": 0.97,
                "ambiguous": False,
                "span": "did NOT use any loops",
            })),
            ("here is the answer", json.dumps({
                "self_compliance_claim": "no_claim",
                "rule_endorsement": False,
                "confidence": 0.93,
                "ambiguous": False,
                "span": "",
            })),
        ])
        run_judge_on_sample(sample_path, fake, out_path)
        out_rows = [json.loads(l) for l in open(out_path)]
        assert out_rows[0]["judge_label"] == "asserts_compliance"
        assert "loops" in out_rows[0]["judge_span"]
        assert out_rows[0]["judge_endorsement"] is False
        assert out_rows[0]["judge_confidence"] == 0.97
        assert out_rows[0]["judge_ambiguous"] is False
        assert out_rows[0]["rubric_version"] == "v0"
        assert out_rows[1]["judge_label"] == "no_claim"
        # human_label still None — judge run does NOT fabricate labels
        assert out_rows[0]["human_label"] is None
