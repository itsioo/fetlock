from pathlib import Path

import pytest

from fetlock.config import Config, load_config
from fetlock.paddock import StratifiedSplit, stratified_subject_split, synthesize_cohort

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bench_config() -> Config:
    return load_config(ROOT / "dockets" / "jobs" / "_bench.toml")


@pytest.fixture
def jobs_dir() -> Path:
    return ROOT / "dockets" / "jobs"


@pytest.fixture
def split() -> StratifiedSplit:
    subjects = synthesize_cohort(80, 100, 6.4, seed=7)
    return stratified_subject_split(
        subjects, load_config(ROOT / "dockets" / "jobs" / "_bench.toml").split, seed=7
    )
