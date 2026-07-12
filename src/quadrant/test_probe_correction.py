"""Tests for probe_correction: the fast problem-level path must equal the
reference implementation, the identity audit must catch lossy dedup, and the
binomial bounds must match closed forms."""
import numpy as np
import pytest

from quadrant.probe_correction import (
    audit_identity, clopper_pearson_upper, foldlegal_baserate_auroc,
    grouped_scores_from_problem_vecs, _max_layer_auroc, _problem_folds,
    oracle_baserate_auroc, perm_null_grouped_max, problem_level_lopo,
    reference_grouped_scores,
)
from quadrant.probe_mechinterp import per_layer_auroc


def _synth(P=8, k=6, L=4, D=7, seed=3, flip=0.25):
    """Problem-duplicated data: one vector per problem, k samples each,
    labels = problem tendency with per-sample noise (mixed problems exist)."""
    rng = np.random.RandomState(seed)
    prob_vecs = rng.randn(P, L, D)
    tend = (rng.rand(P) > 0.5).astype(int)
    # plant a weak signal in layer 1 so AUROC isn't pure noise
    prob_vecs[:, 1, 0] += 2.0 * tend
    X, y, groups = [], [], []
    for p in range(P):
        for _ in range(k):
            lab = tend[p] ^ int(rng.rand() < flip)
            X.append(prob_vecs[p])
            y.append(lab)
            groups.append(f"prob_{p:02d}")
    return (np.stack(X).astype(np.float64), np.array(y), groups,
            prob_vecs.astype(np.float64), sorted(set(groups)))


def test_clopper_pearson_closed_forms():
    # x=0: upper bound is 1 - alpha**(1/n)
    assert clopper_pearson_upper(0, 35, 0.05) == pytest.approx(1 - 0.05 ** (1 / 35))
    assert clopper_pearson_upper(0, 36, 0.05) == pytest.approx(1 - 0.05 ** (1 / 36))
    assert clopper_pearson_upper(35, 35) == 1.0
    # two-sided (alpha/2) bound is wider than one-sided
    assert clopper_pearson_upper(1, 95, 0.025) > clopper_pearson_upper(1, 95, 0.05)
    # more data -> tighter bound
    assert clopper_pearson_upper(0, 100) < clopper_pearson_upper(0, 10)


def test_fast_grouped_path_matches_reference():
    X, y, groups, prob_vecs, pids = _synth()
    folds = _problem_folds(groups, 5, 0)
    scores = grouped_scores_from_problem_vecs(prob_vecs, pids, y, groups, folds)
    fast = _max_layer_auroc(scores, y)
    ref, _, _ = per_layer_auroc(X, y, k=5, seed=0, groups=groups)
    assert fast == pytest.approx(float(np.nanmax(ref)), abs=1e-9)


def test_fast_grouped_scores_match_reference_scores():
    # the load-bearing validation: score-by-score agreement with the repo's
    # own fold/direction computation (AUROC-level comparison is confounded by
    # the reference's FP-noise tie-breaking on real data)
    X, y, groups, prob_vecs, pids = _synth(P=9, k=5, seed=13)
    folds = _problem_folds(groups, 5, 0)
    fast = grouped_scores_from_problem_vecs(prob_vecs, pids, y, groups, folds)
    ref = reference_grouped_scores(X, y, groups, k=5, seed=0)
    assert np.array_equal(np.isnan(fast), np.isnan(ref))
    m = ~np.isnan(fast)
    np.testing.assert_allclose(fast[m], ref[m], atol=1e-9)


def test_fast_grouped_path_matches_reference_all_layers():
    X, y, groups, prob_vecs, pids = _synth(P=10, k=4, seed=11)
    folds = _problem_folds(groups, 5, 0)
    scores = grouped_scores_from_problem_vecs(prob_vecs, pids, y, groups, folds)
    per_layer_fast = [
        _max_layer_auroc(scores[:, [l]], y) for l in range(scores.shape[1])]
    ref, _, _ = per_layer_auroc(X, y, k=5, seed=0, groups=groups)
    np.testing.assert_allclose(per_layer_fast, ref, atol=1e-9)


def test_audit_identity_detects_lossy_dedup():
    X, y, groups, prob_vecs, pids = _synth(P=3, k=3)
    acts = {}
    i = 0
    for p, pid in enumerate(pids):
        for s in range(3):
            acts[(pid, s)] = prob_vecs[p].astype(np.float32)
            i += 1
    audit = audit_identity(acts)
    assert audit["all_within_problem_identical"]
    assert audit["n_problems"] == 3 and audit["n_samples"] == 9
    assert audit["min_between_problem_l2"] > 0
    # perturb one sample -> audit must flag it
    acts[(pids[0], 1)] = acts[(pids[0], 1)] + 1e-3
    assert not audit_identity(acts)["all_within_problem_identical"]


def test_perm_null_structure_and_signal():
    X, y, groups, prob_vecs, pids = _synth(P=12, k=5, seed=7, flip=0.05)
    res = perm_null_grouped_max(prob_vecs, pids, y, groups, n_perms=200, seed=0)
    assert 0 < res["p_value"] <= 1
    assert res["n_perms_valid"] <= 200
    # planted strong problem-level signal: observed should beat the null median
    assert res["observed_grouped_max_auroc"] > res["null_p50"]


def test_perm_null_no_signal_is_insignificant():
    rng = np.random.RandomState(0)
    P, k = 10, 5
    prob_vecs = rng.randn(P, 3, 6)
    pids = [f"p{i:02d}" for i in range(P)]
    y, groups = [], []
    for i, pid in enumerate(pids):
        lab = i % 2
        for _ in range(k):
            y.append(lab)
            groups.append(pid)
    res = perm_null_grouped_max(prob_vecs, pids, np.array(y), groups,
                                n_perms=200, seed=0)
    assert res["p_value"] > 0.01   # labels carry no vector information


def test_oracle_baserate_auroc_reads_problem_identity():
    # labels perfectly determined by problem -> problem base rates are a
    # perfect predictor with zero activation information
    y = np.array([1] * 6 + [0] * 6)
    groups = ["a"] * 6 + ["b"] * 6
    assert oracle_baserate_auroc(y, groups) == 1.0


def test_problem_level_lopo_keys():
    X, y, groups, prob_vecs, pids = _synth(P=10, k=4, seed=5)
    res = problem_level_lopo(prob_vecs, pids, y, groups, n_perms=100)
    assert res["n_problems"] == 10
    assert 0 < res["p_value"] <= 1


def test_lopo_reports_both_tie_assignments():
    # one problem split exactly 50/50 -> both assignments must be reported
    rng = np.random.RandomState(2)
    P, k = 6, 4
    prob_vecs = rng.randn(P, 3, 5)
    pids = [f"p{i}" for i in range(P)]
    y, groups = [], []
    for i, pid in enumerate(pids):
        labs = [1, 0, 1, 0] if i == 0 else [i % 2] * k   # p0 is tied
        y += labs
        groups += [pid] * k
    res = problem_level_lopo(prob_vecs, pids, np.array(y), groups, n_perms=50)
    assert res["n_tied_problems"] == 1
    assert "tie_as_pos" in res and "tie_as_neg" in res
    assert res["tie_as_pos"]["n_pos_problems"] == res["tie_as_neg"]["n_pos_problems"] + 1


def test_foldlegal_baserate_below_or_equal_oracle():
    # the CV-legal memorizer never sees the scored sample's own label, so on
    # problem-clustered labels it should not beat the full-data oracle
    X, y, groups, prob_vecs, pids = _synth(P=10, k=6, seed=9, flip=0.2)
    fl = foldlegal_baserate_auroc(y, groups)
    orc = oracle_baserate_auroc(y, groups)
    assert 0.5 < fl <= orc + 1e-9
