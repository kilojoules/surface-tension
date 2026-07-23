"""Tests for build_dpo_pairs.form_pairs — both pair policies.

compliance-v1 must stay byte-identical to the policy that built DPO-r1/r2
(the historical pair sets were reconstructed from the round logs; see the
r1-log replay test, which runs when the author's vast_logs are present).
"""
import os
import re

import pytest

from build_dpo_pairs import form_pairs

R1_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vast_logs", "m5si116hq9hn6o", "dpo_pairs_r1.log")


def _pools(n_cp, n_cf, n_ch, n_vf):
    return ([f"P{i}" for i in range(n_cp)], [f"F{i}" for i in range(n_cf)],
            [f"C{i}" for i in range(n_ch)], [f"V{i}" for i in range(n_vf)])


def test_v1_enumeration_row_major_ranked():
    # r1 abc308_d shape: compliant=3 (pass=1), violating=5 (cheat=3) -> 6 pairs
    cp, cf, ch, vf = _pools(1, 2, 3, 2)
    pairs = form_pairs(cp, cf, ch, vf, policy="compliance-v1", max_pairs=6)
    assert pairs == [("P0", "C0"), ("P0", "C1"), ("P0", "C2"),
                     ("P0", "V0"), ("P0", "V1"), ("F0", "C0")]


def test_v1_requires_both_sides():
    cp, cf, ch, vf = _pools(3, 2, 0, 0)     # all-compliant problem
    assert form_pairs(cp, cf, ch, vf, policy="compliance-v1") == []
    cp, cf, ch, vf = _pools(0, 0, 2, 3)     # all-violating problem
    assert form_pairs(cp, cf, ch, vf, policy="compliance-v1") == []


def test_v1_chosen_may_fail_tests():
    # the measured failure mode: comp-fail chosen over a passing cheat
    cp, cf, ch, vf = _pools(0, 1, 1, 0)
    assert form_pairs(cp, cf, ch, vf, policy="compliance-v1") == [("F0", "C0")]


def test_v2_chosen_must_pass():
    # same pools: v2 refuses the failing chosen
    cp, cf, ch, vf = _pools(0, 1, 1, 0)
    assert form_pairs(cp, cf, ch, vf, policy="pass-v2") == []


def test_v2_unlocks_all_compliant_problems():
    # all-compliant problem, mixed pass/fail: v1 emits nothing, v2 emits
    # the new comp-pass > comp-fail pairs (gradient on passing)
    cp, cf, ch, vf = _pools(2, 3, 0, 0)
    assert form_pairs(cp, cf, ch, vf, policy="compliance-v1") == []
    v2 = form_pairs(cp, cf, ch, vf, policy="pass-v2", max_pairs=4)
    assert v2 == [("P0", "F0"), ("P0", "F1"), ("P0", "F2"), ("P1", "F0")]


def test_v2_rejected_ranking_cheats_first():
    cp, cf, ch, vf = _pools(1, 1, 1, 1)
    v2 = form_pairs(cp, cf, ch, vf, policy="pass-v2", max_pairs=3)
    assert v2 == [("P0", "C0"), ("P0", "V0"), ("P0", "F0")]


def test_max_pairs_cap_and_unknown_policy():
    cp, cf, ch, vf = _pools(4, 0, 4, 0)
    assert len(form_pairs(cp, cf, ch, vf, policy="compliance-v1", max_pairs=6)) == 6
    with pytest.raises(ValueError):
        form_pairs(cp, cf, ch, vf, policy="nope")


@pytest.mark.skipif(not os.path.exists(R1_LOG), reason="author-machine round log not present")
def test_v1_replays_the_r1_round_log():
    """Replay the real r1 build log's per-problem pool counts through
    form_pairs and require the per-problem emitted-pair counts and the 113
    total to match — pins compliance-v1 to the policy that actually trained
    DPO-r1."""
    total = 0
    for line in open(R1_LOG, errors="ignore"):
        m = re.search(r"compliant=(\d+)(?: \(pass=(\d+)\))? "
                      r"violating=(\d+)(?: \(cheat=(\d+)\))? → (\d+) pairs", line)
        if not m:
            continue
        comp, cpass, viol, cheat, expected = (int(m.group(1)), int(m.group(2) or 0),
                                              int(m.group(3)), int(m.group(4) or 0),
                                              int(m.group(5)))
        cp, cf, ch, vf = _pools(cpass, comp - cpass, cheat, viol - cheat)
        got = form_pairs(cp, cf, ch, vf, policy="compliance-v1", max_pairs=6)
        assert len(got) == expected, (line, len(got))
        total += len(got)
    assert total == 113
