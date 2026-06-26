import numpy as np

from fetlock.paddock.augments import (
    CHANNEL_SHUFFLE,
    ROTATION,
    SCALING,
    TIME_WARP,
    augment_view,
    make_pretext_view,
    time_reverse,
)


def _window(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((6, 128)).astype(np.float32)


def test_every_augmentation_preserves_shape():
    window = _window(0)
    rng = np.random.default_rng(0)
    for kind in (ROTATION, SCALING, TIME_WARP, CHANNEL_SHUFFLE):
        assert augment_view(window, kind, rng).shape == window.shape


def test_rotation_preserves_triplet_norm():
    window = _window(1)
    rotated = augment_view(window, ROTATION, np.random.default_rng(1))
    original_norm = np.linalg.norm(window[0:3], axis=0)
    rotated_norm = np.linalg.norm(rotated[0:3], axis=0)
    assert np.allclose(original_norm, rotated_norm, atol=1e-4)


def test_time_reverse_is_involution():
    window = _window(2)
    assert np.array_equal(time_reverse(time_reverse(window)), window)


def test_channel_shuffle_is_a_permutation():
    window = _window(3)
    shuffled = augment_view(window, CHANNEL_SHUFFLE, np.random.default_rng(5))
    assert np.allclose(np.sort(window, axis=0), np.sort(shuffled, axis=0))


def test_pretext_view_labels_are_consistent():
    view = make_pretext_view(_window(4), np.random.default_rng(9))
    assert view.transform_label in range(4)
    assert view.order_label in (0, 1)
    if view.order_label == 1:
        assert np.array_equal(view.order_input, time_reverse(_window(4)))
