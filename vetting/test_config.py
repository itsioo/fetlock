import pytest
from cattrs.errors import ClassValidationError

from fetlock.config import load_config


def test_every_job_loads(jobs_dir):
    for path in sorted(jobs_dir.glob("*.toml")):
        config = load_config(path)
        assert config.run.effective_batch > 0
        assert config.window.length > 0


def test_main_job_matches_reported_profile(jobs_dir):
    config = load_config(jobs_dir / "main.toml")
    assert config.run.world_size == 8
    assert config.run.effective_batch == 8 * 256 * 24
    assert config.optim.lr == 2e-7
    assert len(config.run.seeds) == 10


def test_dotted_override_is_applied(jobs_dir):
    config = load_config(jobs_dir / "main.toml", {"optim.lr": 1e-5, "schedule.epochs": 9})
    assert config.optim.lr == 1e-5
    assert config.schedule.epochs == 9


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("stage = 'finetune'\nnonsense_key = 3\n", encoding="utf-8")
    with pytest.raises(ClassValidationError):
        load_config(bad)


def test_gyro_only_uses_three_channels(jobs_dir):
    config = load_config(jobs_dir / "ablation_gyro_only.toml")
    assert config.ablation.channel_index == (3, 4, 5)
    assert config.encoder.in_channels == 3
