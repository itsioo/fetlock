import numpy as np

from fetlock.farrier.stages import ANCHOR_ROIS, ANCHOR_TARGET, run_concordance
from fetlock.soundness import fmri_concordance, pearson_r


def test_pearson_against_numpy_corrcoef():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64)
    y = 0.6 * x + 0.5 * rng.standard_normal(64)
    assert abs(pearson_r(x, y) - np.corrcoef(x, y)[0, 1]) < 1e-9


def test_panel_recovers_planted_targets(bench_config):
    cells = run_concordance(bench_config)
    by_roi = {cell.roi: cell for cell in cells}
    for roi, target in zip(ANCHOR_ROIS, ANCHOR_TARGET):
        assert abs(by_roi[roi].r - target) < 0.05


def test_unrelated_series_is_not_significant():
    rng = np.random.default_rng(1)
    wearable = {"feat": rng.standard_normal(50)}
    roi = {"region": rng.standard_normal(50)}
    cells = fmri_concordance(wearable, roi, [("feat", "region")], n_perm=1000, n_boot=500, seed=2)
    assert not cells[0].fdr_reject


def test_confidence_interval_brackets_estimate():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(60)
    y = 0.5 * x + 0.5 * rng.standard_normal(60)
    result = fmri_concordance({"f": x}, {"r": y}, [("f", "r")], n_perm=500, n_boot=800, seed=4)[0]
    assert result.ci_low <= result.r <= result.ci_high
