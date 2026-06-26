import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

CHANNEL_ORDER = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")
SEX_MALE = 0
SEX_FEMALE = 1
SEX_UNKNOWN = -1


@dataclass(frozen=True)
class Subject:
    subject_id: str
    sex: int
    age_band: int
    severity: int
    label: int
    signal: NDArray[np.float32]


def _phase(
    t: NDArray[np.float64], cadence: float, drift: NDArray[np.float64]
) -> NDArray[np.float64]:
    return 2.0 * np.pi * cadence * t + drift


def _subject_signal(
    rng: np.random.Generator,
    sex: int,
    label: int,
    sample_rate: int,
    seconds: float,
) -> NDArray[np.float32]:
    length = round(sample_rate * seconds)
    t = np.arange(length, dtype=np.float64) / float(sample_rate)
    cadence = float(rng.normal(1.8, 0.12))
    stride_var = 0.02 + 0.06 * label
    drift = np.cumsum(rng.normal(0.0, stride_var, size=length)) * 0.04
    phase = _phase(t, cadence, drift)

    rom = 1.0 - 0.4 * label
    dorsi_gain = 1.0 - 0.5 * label - 0.22 * label * (sex == SEX_FEMALE)
    plantar_gain = 1.0 - 0.3 * label

    channels = np.empty((len(CHANNEL_ORDER), length), dtype=np.float64)
    channels[0] = rom * np.cos(phase)
    channels[1] = 0.8 * rom * np.sin(2.0 * phase)
    channels[2] = 0.6 * np.sin(phase + 0.3) + 0.2
    channels[3] = dorsi_gain * np.cos(phase)
    channels[4] = plantar_gain * np.sin(phase + 0.5)
    channels[5] = 0.5 * np.sin(3.0 * phase) + rng.normal(0.0, 0.05 + 0.12 * label, size=length)

    noise = rng.normal(0.0, 0.05, size=channels.shape)
    return (channels + noise).astype(np.float32)


def synthesize_cohort(
    n_subjects: int,
    sample_rate: int,
    seconds: float,
    seed: int,
    female_fraction: float = 0.6,
    oa_fraction: float = 0.5,
) -> List[Subject]:
    rng = np.random.default_rng(seed)
    subjects: List[Subject] = []
    for index in range(n_subjects):
        sex = SEX_FEMALE if rng.random() < female_fraction else SEX_MALE
        label = int(rng.random() < oa_fraction)
        age_band = int(rng.integers(0, 3))
        severity = int(rng.integers(1, 5)) if label else 0
        signal = _subject_signal(rng, sex, label, sample_rate, seconds)
        subjects.append(
            Subject(
                subject_id=f"S{index:05d}",
                sex=sex,
                age_band=age_band,
                severity=severity,
                label=label,
                signal=signal,
            )
        )
    return subjects


def load_cohort(root: Union[str, Path], channels: int) -> List[Subject]:
    root = Path(root)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no manifest.jsonl under {root}; point data_root at a prepared cohort"
        )
    subjects: List[Subject] = []
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            array = np.load(root / row["signal"])
            subjects.append(
                Subject(
                    subject_id=str(row["subject_id"]),
                    sex=int(row["sex"]),
                    age_band=int(row["age_band"]),
                    severity=int(row["severity"]),
                    label=int(row["label"]),
                    signal=array[:channels].astype(np.float32),
                )
            )
    return subjects
