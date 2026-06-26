import numpy as np
from numpy.typing import NDArray


def sliding_windows(signal: NDArray[np.float32], length: int, hop: int) -> NDArray[np.float32]:
    channels, total = signal.shape
    if total < length:
        raise ValueError(f"signal of {total} samples is shorter than window of {length}")
    starts = range(0, total - length + 1, hop)
    out = np.empty((len(starts), channels, length), dtype=np.float32)
    for row, start in enumerate(starts):
        out[row] = signal[:, start : start + length]
    return out


def standardize(windows: NDArray[np.float32]) -> NDArray[np.float32]:
    mean = windows.mean(axis=(0, 2), keepdims=True)
    std = windows.std(axis=(0, 2), keepdims=True) + 1e-6
    return ((windows - mean) / std).astype(np.float32)
