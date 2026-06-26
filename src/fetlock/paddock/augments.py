from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

ROTATION = 0
SCALING = 1
TIME_WARP = 2
CHANNEL_SHUFFLE = 3
TRANSFORM_NAMES = ("rotation", "scaling", "time_warp", "channel_shuffle")


@dataclass(frozen=True)
class PretextView:
    view_a: NDArray[np.float32]
    view_b: NDArray[np.float32]
    transform_label: int
    order_input: NDArray[np.float32]
    order_label: int


def _triplets(channels: int) -> List[range]:
    if channels % 3 == 0:
        return [range(g * 3, g * 3 + 3) for g in range(channels // 3)]
    return [range(channels)]


def _random_rotation(rng: np.random.Generator) -> NDArray[np.float64]:
    a, b, c = rng.uniform(-np.pi, np.pi, size=3)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float64)


def _rotate(window: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    out = window.copy()
    for group in _triplets(window.shape[0]):
        idx = list(group)
        if len(idx) == 3:
            out[idx] = (_random_rotation(rng) @ window[idx]).astype(np.float32)
        else:
            out[idx] = window[idx] * np.float32(rng.uniform(0.7, 1.3))
    return out


def _scale(window: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    factor = rng.uniform(0.7, 1.3, size=(window.shape[0], 1)).astype(np.float32)
    return (window * factor).astype(np.float32)


def _time_warp(window: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    length = window.shape[1]
    speed = np.clip(rng.normal(1.0, 0.2, size=length), 0.5, 1.5)
    warped_axis = np.cumsum(speed)
    warped_axis *= (length - 1) / warped_axis[-1]
    base = np.arange(length, dtype=np.float64)
    out = np.empty_like(window)
    for channel in range(window.shape[0]):
        out[channel] = np.interp(base, warped_axis, window[channel]).astype(np.float32)
    return out


def _channel_shuffle(window: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    out = window.copy()
    for group in _triplets(window.shape[0]):
        idx = np.array(list(group))
        out[idx] = window[idx[rng.permutation(len(idx))]]
    return out


def augment_view(
    window: NDArray[np.float32], kind: int, rng: np.random.Generator
) -> NDArray[np.float32]:
    if kind == ROTATION:
        return _rotate(window, rng)
    if kind == SCALING:
        return _scale(window, rng)
    if kind == TIME_WARP:
        return _time_warp(window, rng)
    if kind == CHANNEL_SHUFFLE:
        return _channel_shuffle(window, rng)
    raise ValueError(f"unknown augmentation kind {kind}")


def time_reverse(window: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.ascontiguousarray(window[:, ::-1])


def make_pretext_view(window: NDArray[np.float32], rng: np.random.Generator) -> PretextView:
    kinds = rng.choice(len(TRANSFORM_NAMES), size=2, replace=False)
    view_a = augment_view(window, int(kinds[0]), rng)
    view_b = augment_view(window, int(kinds[1]), rng)
    flipped = bool(rng.random() < 0.5)
    order_input = time_reverse(window) if flipped else window.copy()
    return PretextView(
        view_a=view_a,
        view_b=view_b,
        transform_label=int(kinds[0]),
        order_input=order_input,
        order_label=int(flipped),
    )
