import numpy as np
import pytest

from fetlock.config import SplitSpec
from fetlock.paddock import stratified_subject_split, synthesize_cohort
from fetlock.paddock.stalls import build_window_table
from fetlock.paddock.windows import sliding_windows, standardize


def test_window_count_matches_hop():
    signal = np.zeros((6, 1000), dtype=np.float32)
    windows = sliding_windows(signal, length=200, hop=100)
    assert windows.shape == (9, 6, 200)


def test_window_rejects_short_signal():
    with pytest.raises(ValueError):
        sliding_windows(np.zeros((6, 50), dtype=np.float32), length=200, hop=100)


def test_standardize_is_zero_mean_unit_scale():
    rng = np.random.default_rng(0)
    windows = rng.normal(3.0, 5.0, size=(40, 6, 128)).astype(np.float32)
    out = standardize(windows)
    assert np.allclose(out.mean(axis=(0, 2)), 0.0, atol=1e-4)
    assert np.allclose(out.std(axis=(0, 2)), 1.0, atol=1e-3)


def test_split_has_no_subject_leakage():
    subjects = synthesize_cohort(120, 100, 4.0, seed=1)
    parts = stratified_subject_split(subjects, SplitSpec(), seed=1)
    ids = [{s.subject_id for s in group} for group in (parts.train, parts.val, parts.test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert sum(len(group) for group in ids) == 120


def test_split_keeps_both_sexes_in_train():
    subjects = synthesize_cohort(120, 100, 4.0, seed=2)
    parts = stratified_subject_split(subjects, SplitSpec(), seed=2)
    sexes = {s.sex for s in parts.train}
    assert sexes == {0, 1}


def test_channel_selection_drops_gyro():
    subjects = synthesize_cohort(20, 100, 4.0, seed=3)
    table, labels, sexes = build_window_table(subjects, 128, 64, (0, 1, 2))
    assert table.shape[1] == 3
    assert len(labels) == len(sexes) == table.shape[0]
