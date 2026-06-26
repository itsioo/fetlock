import numpy as np

from fetlock.soundness import (
    accuracy,
    benjamini_hochberg,
    cohens_kappa,
    delong_test,
    expected_calibration_error,
    f1_score,
    holm_bonferroni,
    paired_bootstrap_auc_diff,
    partial_eta_squared,
    permutation_test,
    roc_auc,
    sensitivity,
    specificity,
)


def _trapezoid_auc(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    return float(np.trapezoid(tpr, fpr))


def test_auc_matches_trapezoid_integration():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 300)
    scores = rng.random(300) + 0.4 * labels
    assert abs(roc_auc(scores, labels) - _trapezoid_auc(scores, labels)) < 1e-6


def test_perfect_separation_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(scores, labels) == 1.0
    assert sensitivity(scores, labels) == 1.0
    assert specificity(scores, labels) == 1.0
    assert f1_score(scores, labels) == 1.0
    assert accuracy(scores, labels) == 1.0


def test_delong_auc_consistent_and_symmetric():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 200)
    a = rng.random(200) + 0.5 * labels
    b = rng.random(200) + 0.2 * labels
    result = delong_test(a, b, labels)
    assert abs(result.auc_a - roc_auc(a, labels)) < 1e-9
    flipped = delong_test(b, a, labels)
    assert abs(result.z + flipped.z) < 1e-9
    assert 0.0 <= result.p_value <= 1.0


def test_bootstrap_zero_for_identical_predictors():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, 150)
    scores = rng.random(150)
    result = paired_bootstrap_auc_diff(scores, scores, labels, n_boot=300, seed=4)
    assert abs(result.mean) < 1e-9


def test_holm_is_more_conservative_than_bh():
    pvals = np.array([0.001, 0.012, 0.03, 0.2])
    holm_adj, _ = holm_bonferroni(pvals)
    bh_adj, _ = benjamini_hochberg(pvals)
    assert np.all(holm_adj + 1e-12 >= bh_adj)


def test_partial_eta_squared_in_unit_interval():
    rng = np.random.default_rng(3)
    sex = np.repeat([0, 1], 40)
    method = np.tile([0, 1], 40)
    values = 0.5 * sex + 0.3 * method + 0.4 * sex * method + rng.normal(0, 0.1, 80)
    eta = partial_eta_squared(values, sex, method)
    assert 0.0 <= eta <= 1.0
    assert eta > 0.05


def test_kappa_endpoints():
    a = np.array([0, 1, 0, 1, 1])
    assert abs(cohens_kappa(a, a) - 1.0) < 1e-9
    rng = np.random.default_rng(5)
    big = rng.integers(0, 2, 2000)
    other = rng.integers(0, 2, 2000)
    assert abs(cohens_kappa(big, other)) < 0.1


def test_permutation_and_ece_ranges():
    rng = np.random.default_rng(6)
    a = rng.normal(1.0, 0.2, 50)
    b = rng.normal(0.0, 0.2, 50)
    assert permutation_test(a, b, n_perm=500, seed=1) < 0.05
    probs = rng.random(200)
    labels = (probs > 0.5).astype(np.int64)
    assert 0.0 <= expected_calibration_error(probs, labels) <= 1.0
