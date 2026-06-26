from typing import Dict, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from fetlock.paddock.augments import make_pretext_view
from fetlock.paddock.cohort import Subject
from fetlock.paddock.windows import sliding_windows, standardize


def build_window_table(
    subjects: List[Subject],
    length: int,
    hop: int,
    channel_index: Tuple[int, ...],
) -> Tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    chunks: List[NDArray[np.float32]] = []
    labels: List[int] = []
    sexes: List[int] = []
    selector = list(channel_index)
    for subject in subjects:
        windows = sliding_windows(subject.signal[selector], length, hop)
        chunks.append(windows)
        labels.extend([subject.label] * len(windows))
        sexes.extend([subject.sex] * len(windows))
    if not chunks:
        raise ValueError("no subjects produced any windows")
    table = standardize(np.concatenate(chunks, axis=0))
    return table, np.asarray(labels, dtype=np.int64), np.asarray(sexes, dtype=np.int64)


class SupervisedStalls(Dataset):
    def __init__(
        self,
        subjects: List[Subject],
        length: int,
        hop: int,
        channel_index: Tuple[int, ...],
    ) -> None:
        self.windows, self.labels, self.sexes = build_window_table(
            subjects, length, hop, channel_index
        )

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "signal": torch.from_numpy(self.windows[index]),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "sex": torch.tensor(int(self.sexes[index]), dtype=torch.long),
        }


class PretextStalls(Dataset):
    def __init__(
        self,
        subjects: List[Subject],
        length: int,
        hop: int,
        channel_index: Tuple[int, ...],
        seed: int,
    ) -> None:
        self.windows, _, _ = build_window_table(subjects, length, hop, channel_index)
        self.seed = seed

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed * 1_000_003 + index) & 0x7FFFFFFF)
        view = make_pretext_view(self.windows[index], rng)
        return {
            "view_a": torch.from_numpy(view.view_a),
            "view_b": torch.from_numpy(view.view_b),
            "transform_label": torch.tensor(view.transform_label, dtype=torch.long),
            "order_input": torch.from_numpy(view.order_input),
            "order_label": torch.tensor(view.order_label, dtype=torch.long),
        }
